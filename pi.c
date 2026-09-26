#ifndef _WIN32
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#endif

#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include <gmp.h>
#include <omp.h>

#ifdef _WIN32
#include <direct.h>
#include <io.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>
#endif

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#define DEFAULT_DIGITS                 1000000UL
#define DEFAULT_MAX_RUNTIME_SECONDS    (5UL * 3600UL)
#define DEFAULT_CHUNK_TERMS            100000UL
#define MAX_CHECKPOINT_NODES           128U
#define GUARD_DIGITS                   20UL

#define DEFAULT_STATE_FILE             "state.json"
#define DEFAULT_PI_FILE                "pi_c_hyperspeed.txt"
#define DEFAULT_CHECKPOINT_DIR         "pi_checkpoint"

#define STATE_VERSION                  2U
#define MANIFEST_VERSION               1U
#define TUPLE_VERSION                  1U

static const unsigned char MANIFEST_MAGIC[8] = {
    'P', 'I', 'M', 'A', 'N', 'I', 'F', '2'
};
static const unsigned char MANIFEST_END_MAGIC[8] = {
    'P', 'I', 'M', 'A', 'N', 'E', 'N', 'D'
};
static const unsigned char TUPLE_MAGIC[8] = {
    'P', 'I', 'T', 'U', 'P', 'L', 'E', '2'
};
static const unsigned char TUPLE_END_MAGIC[8] = {
    'P', 'I', 'T', 'U', 'P', 'E', 'N', 'D'
};

mpz_t CONST_Q;

static volatile sig_atomic_t g_stop = 0;

typedef struct {
    const char *state_file;
    const char *pi_file;
    const char *checkpoint_dir;
    const char *checkpoint_manifest;
} Paths;

typedef struct {
    unsigned long max_runtime_seconds; /* 0 disables the time limit. */
    unsigned long chunk_terms;         /* 0 selects the bounded default. */
    unsigned long max_chunks_per_run;  /* 0 disables this test control. */
} RunConfig;

typedef struct {
    unsigned long completed_digits;
    unsigned long next_digits;
    uint64_t pi_bytes;
    uint64_t pi_checksum;
    unsigned long discarded_checkpoint_target;
    int checkpoint_active;
    unsigned long checkpoint_target_digits;
    unsigned long checkpoint_completed_iterations;
    unsigned long checkpoint_total_iterations;
} State;

/*
 * The manifest lists a balanced stack of immutable tuple files.  GMP values
 * are deliberately not kept in the manifest or rewritten at every boundary:
 * a new/merged range writes only its own tuple, then atomically replaces the
 * small manifest that names it.
 */
typedef struct {
    uint64_t id;
    unsigned long start;
    unsigned long end;
} CheckpointNode;

typedef struct {
    unsigned long target_digits;
    unsigned long total_iterations;
    unsigned long completed_iterations;
    uint64_t generation;
    size_t count;
    CheckpointNode nodes[MAX_CHECKPOINT_NODES];
} Checkpoint;

typedef struct {
    uint64_t bytes;
    uint64_t checksum;
} PiFileInfo;

typedef enum {
    CHECKPOINT_MISSING,
    CHECKPOINT_VALID,
    CHECKPOINT_INVALID
} CheckpointLoadResult;

typedef enum {
    RUN_DONE,
    RUN_PAUSED,
    RUN_ERROR
} RunResult;

static void on_signal(int sig) {
    (void)sig;
    g_stop = 1;
}

static const char *path_from_env(const char *name, const char *fallback) {
    const char *value = getenv(name);
    return value != NULL && value[0] != '\0' ? value : fallback;
}

static int parse_ulong(const char *text, int allow_zero, unsigned long *value) {
    char *end = NULL;
    unsigned long parsed;

    if (text == NULL || text[0] == '\0' || text[0] == '-') {
        return 0;
    }
    errno = 0;
    parsed = strtoul(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' ||
        (!allow_zero && parsed == 0)) {
        return 0;
    }
    *value = parsed;
    return 1;
}

static unsigned long env_ulong(const char *name, unsigned long fallback,
                               int allow_zero) {
    const char *value = getenv(name);
    unsigned long parsed;

    if (value == NULL || value[0] == '\0') {
        return fallback;
    }
    if (!parse_ulong(value, allow_zero, &parsed)) {
        fprintf(stderr, "[!] Ignoring invalid %s=%s\n", name, value);
        return fallback;
    }
    return parsed;
}

static int u64_to_ulong(uint64_t value, unsigned long *result) {
    if (value > (uint64_t)ULONG_MAX) {
        return 0;
    }
    *result = (unsigned long)value;
    return 1;
}

static int make_temp_path(char *buffer, size_t size, const char *path) {
    int written = snprintf(buffer, size, "%s.tmp", path);
    return written >= 0 && (size_t)written < size;
}

static int flush_and_close(FILE *file) {
    int ok = 1;

    if (fflush(file) != 0) {
        ok = 0;
    }
#ifdef _WIN32
    if (ok && _commit(_fileno(file)) != 0) {
        ok = 0;
    }
#else
    if (ok && fsync(fileno(file)) != 0) {
        ok = 0;
    }
#endif
    if (fclose(file) != 0) {
        ok = 0;
    }
    return ok;
}

/* A file fsync does not make its post-rename directory entry durable. */
static int sync_parent_directory(const char *path) {
#ifdef _WIN32
    (void)path;
    return 1; /* MOVEFILE_WRITE_THROUGH below supplies the Windows equivalent. */
#else
    char directory[PATH_MAX];
    const char *separator = strrchr(path, '/');
    size_t length;
    int descriptor;
    int ok;

    if (separator == NULL) {
        (void)snprintf(directory, sizeof(directory), ".");
    } else if (separator == path) {
        (void)snprintf(directory, sizeof(directory), "/");
    } else {
        length = (size_t)(separator - path);
        if (length >= sizeof(directory)) {
            return 0;
        }
        memcpy(directory, path, length);
        directory[length] = '\0';
    }
    descriptor = open(directory, O_RDONLY);
    if (descriptor < 0) {
        return 0;
    }
    ok = fsync(descriptor) == 0;
    (void)close(descriptor);
    return ok;
#endif
}

static int atomic_replace(const char *temporary_path, const char *final_path) {
#ifdef _WIN32
    return MoveFileExA(temporary_path, final_path,
                       MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
    return rename(temporary_path, final_path) == 0 &&
           sync_parent_directory(final_path);
#endif
}

static int finish_atomic_write(FILE *file, const char *temporary_path,
                               const char *final_path) {
    if (!flush_and_close(file)) {
        fprintf(stderr, "[!] Could not flush %s: %s\n", temporary_path,
                strerror(errno));
        (void)remove(temporary_path);
        return 0;
    }
    if (!atomic_replace(temporary_path, final_path)) {
        fprintf(stderr, "[!] Could not replace %s: %s\n", final_path,
                strerror(errno));
        (void)remove(temporary_path);
        return 0;
    }
    return 1;
}

static uint64_t fnv1a_update(uint64_t hash, const void *data, size_t length) {
    const unsigned char *bytes = (const unsigned char *)data;
    size_t i;

    for (i = 0; i < length; ++i) {
        hash ^= bytes[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static int write_json_string(FILE *file, const char *value) {
    const unsigned char *p = (const unsigned char *)value;

    if (fputc('"', file) == EOF) {
        return 0;
    }
    while (*p != '\0') {
        if (*p == '"' || *p == '\\') {
            if (fputc('\\', file) == EOF || fputc(*p, file) == EOF) {
                return 0;
            }
        } else if (*p < 0x20) {
            if (fprintf(file, "\\u%04x", (unsigned int)*p) < 0) {
                return 0;
            }
        } else if (fputc(*p, file) == EOF) {
            return 0;
        }
        ++p;
    }
    return fputc('"', file) != EOF;
}

static int json_get_u64(const char *json, const char *key, uint64_t *value) {
    char needle[128];
    const char *position;
    char *end = NULL;
    unsigned long long parsed;

    if (snprintf(needle, sizeof(needle), "\"%s\"", key) < 0) {
        return 0;
    }
    position = strstr(json, needle);
    if (position == NULL) {
        return 0;
    }
    position = strchr(position + strlen(needle), ':');
    if (position == NULL) {
        return 0;
    }
    ++position;
    while (isspace((unsigned char)*position)) {
        ++position;
    }
    if (*position == '-') {
        return 0;
    }

    errno = 0;
    parsed = strtoull(position, &end, 10);
    if (errno == ERANGE || end == position) {
        return 0;
    }
    while (isspace((unsigned char)*end)) {
        ++end;
    }
    if (*end != ',' && *end != '}' && *end != '\0') {
        return 0;
    }
    *value = (uint64_t)parsed;
    return 1;
}

static int json_get_hex64(const char *json, const char *key, uint64_t *value) {
    char needle[128];
    const char *position;
    char hex[17];
    char *end = NULL;
    unsigned long long parsed;
    size_t length = 0;

    if (snprintf(needle, sizeof(needle), "\"%s\"", key) < 0) {
        return 0;
    }
    position = strstr(json, needle);
    if (position == NULL) {
        return 0;
    }
    position = strchr(position + strlen(needle), ':');
    if (position == NULL) {
        return 0;
    }
    ++position;
    while (isspace((unsigned char)*position)) {
        ++position;
    }
    if (*position++ != '"') {
        return 0;
    }
    while (isxdigit((unsigned char)position[length]) && length < 16U) {
        hex[length] = position[length];
        ++length;
    }
    if (length != 16U || position[length] != '"') {
        return 0;
    }
    hex[length] = '\0';
    errno = 0;
    parsed = strtoull(hex, &end, 16);
    if (errno == ERANGE || end == hex || *end != '\0') {
        return 0;
    }
    *value = (uint64_t)parsed;
    return 1;
}

static void state_default(State *state) {
    memset(state, 0, sizeof(*state));
    state->next_digits = DEFAULT_DIGITS;
}

/*
 * State v1 stored "digits" as the next target.  Preserve that behavior while
 * upgrading existing repositories to explicit completed/next fields.
 */
static int read_state(const Paths *paths, State *state, int *was_legacy) {
    char buffer[8192];
    FILE *file;
    size_t bytes;
    uint64_t value;

    state_default(state);
    *was_legacy = 0;
    file = fopen(paths->state_file, "rb");
    if (file == NULL) {
        if (errno != ENOENT) {
            fprintf(stderr, "[!] Could not read %s: %s\n", paths->state_file,
                    strerror(errno));
        }
        return errno == ENOENT;
    }
    bytes = fread(buffer, 1, sizeof(buffer) - 1U, file);
    if (ferror(file) || (bytes == sizeof(buffer) - 1U && !feof(file))) {
        fprintf(stderr, "[!] %s is too large or unreadable.\n", paths->state_file);
        (void)fclose(file);
        return 0;
    }
    buffer[bytes] = '\0';
    (void)fclose(file);

    if (json_get_u64(buffer, "version", &value) && value > STATE_VERSION) {
        fprintf(stderr, "[!] Unsupported state version %" PRIu64 ".\n", value);
        return 0;
    }
    if (json_get_u64(buffer, "next_digits", &value)) {
        if (!u64_to_ulong(value, &state->next_digits) ||
            state->next_digits == 0) {
            return 0;
        }
    } else {
        if (!json_get_u64(buffer, "digits", &value) ||
            !u64_to_ulong(value, &state->next_digits) ||
            state->next_digits == 0) {
            return 0;
        }
        *was_legacy = 1;
    }
    if (json_get_u64(buffer, "completed_digits", &value) &&
        !u64_to_ulong(value, &state->completed_digits)) {
        return 0;
    }
    (void)json_get_u64(buffer, "pi_bytes", &state->pi_bytes);
    (void)json_get_hex64(buffer, "pi_checksum", &state->pi_checksum);
    if (json_get_u64(buffer, "discarded_checkpoint_target", &value) &&
        !u64_to_ulong(value, &state->discarded_checkpoint_target)) {
        return 0;
    }
    if (json_get_u64(buffer, "checkpoint_active", &value) && value != 0) {
        state->checkpoint_active = 1;
        if (!json_get_u64(buffer, "checkpoint_target_digits", &value) ||
            !u64_to_ulong(value, &state->checkpoint_target_digits) ||
            !json_get_u64(buffer, "checkpoint_completed_iterations", &value) ||
            !u64_to_ulong(value, &state->checkpoint_completed_iterations) ||
            !json_get_u64(buffer, "checkpoint_total_iterations", &value) ||
            !u64_to_ulong(value, &state->checkpoint_total_iterations)) {
            state->checkpoint_active = 0;
        }
    }
    return 1;
}

static int write_state(const Paths *paths, const State *state) {
    char temporary_path[PATH_MAX];
    FILE *file;
    int ok = 1;

    if (!make_temp_path(temporary_path, sizeof(temporary_path),
                        paths->state_file)) {
        fprintf(stderr, "[!] State path is too long.\n");
        return 0;
    }
    file = fopen(temporary_path, "wb");
    if (file == NULL) {
        fprintf(stderr, "[!] Could not write %s: %s\n", temporary_path,
                strerror(errno));
        return 0;
    }

    ok &= fprintf(file,
                  "{\n"
                  "  \"version\": %u,\n"
                  "  \"completed_digits\": %lu,\n"
                  "  \"next_digits\": %lu,\n"
                  "  \"pi_file\": ",
                  STATE_VERSION, state->completed_digits,
                  state->next_digits) >= 0;
    ok &= write_json_string(file, paths->pi_file);
    ok &= fprintf(file,
                  ",\n"
                  "  \"pi_bytes\": %" PRIu64 ",\n"
                  "  \"pi_checksum\": \"%016" PRIx64 "\",\n"
                  "  \"discarded_checkpoint_target\": %lu,\n"
                  "  \"checkpoint_dir\": ",
                  state->pi_bytes, state->pi_checksum,
                  state->discarded_checkpoint_target) >= 0;
    ok &= write_json_string(file, paths->checkpoint_dir);
    ok &= fprintf(file,
                  ",\n"
                  "  \"checkpoint_active\": %d,\n"
                  "  \"checkpoint_target_digits\": %lu,\n"
                  "  \"checkpoint_completed_iterations\": %lu,\n"
                  "  \"checkpoint_total_iterations\": %lu,\n"
                  "  \"updated\": %ld\n"
                  "}\n",
                  state->checkpoint_active ? 1 : 0,
                  state->checkpoint_target_digits,
                  state->checkpoint_completed_iterations,
                  state->checkpoint_total_iterations,
                  (long)time(NULL)) >= 0;

    if (!ok) {
        fprintf(stderr, "[!] Failed while serializing %s.\n", temporary_path);
        (void)fclose(file);
        (void)remove(temporary_path);
        return 0;
    }
    return finish_atomic_write(file, temporary_path, paths->state_file);
}

static int write_u64(FILE *file, uint64_t value) {
    unsigned char bytes[8];
    int i;

    for (i = 7; i >= 0; --i) {
        bytes[7 - i] = (unsigned char)(value >> (unsigned int)(i * 8));
    }
    return fwrite(bytes, 1, sizeof(bytes), file) == sizeof(bytes);
}

static int read_u64(FILE *file, uint64_t *value) {
    unsigned char bytes[8];
    uint64_t parsed = 0;
    size_t i;

    if (fread(bytes, 1, sizeof(bytes), file) != sizeof(bytes)) {
        return 0;
    }
    for (i = 0; i < sizeof(bytes); ++i) {
        parsed = (parsed << 8U) | bytes[i];
    }
    *value = parsed;
    return 1;
}

static int checkpoint_directory_exists(const char *path) {
#ifdef _WIN32
    struct _stat info;
    return _stat(path, &info) == 0 && (info.st_mode & _S_IFDIR) != 0;
#else
    struct stat info;
    return stat(path, &info) == 0 && S_ISDIR(info.st_mode);
#endif
}

static int ensure_checkpoint_directory(const Paths *paths) {
    if (checkpoint_directory_exists(paths->checkpoint_dir)) {
        return 1;
    }
#ifdef _WIN32
    if (_mkdir(paths->checkpoint_dir) == 0) {
        if (!sync_parent_directory(paths->checkpoint_dir)) {
            fprintf(stderr, "[!] Could not sync checkpoint directory %s: %s\n",
                    paths->checkpoint_dir, strerror(errno));
            return 0;
        }
        return 1;
    }
#else
    if (mkdir(paths->checkpoint_dir, 0777) == 0) {
        if (!sync_parent_directory(paths->checkpoint_dir)) {
            fprintf(stderr, "[!] Could not sync checkpoint directory %s: %s\n",
                    paths->checkpoint_dir, strerror(errno));
            return 0;
        }
        return 1;
    }
#endif
    if (errno == EEXIST && checkpoint_directory_exists(paths->checkpoint_dir)) {
        return 1;
    }
    fprintf(stderr, "[!] Could not create checkpoint directory %s: %s\n",
            paths->checkpoint_dir, strerror(errno));
    return 0;
}

static int manifest_path(const Paths *paths, char *buffer, size_t size) {
    int written;

    if (paths->checkpoint_manifest != NULL &&
        paths->checkpoint_manifest[0] != '\0') {
        written = snprintf(buffer, size, "%s", paths->checkpoint_manifest);
    } else {
        written = snprintf(buffer, size, "%s/manifest.bin",
                           paths->checkpoint_dir);
    }
    return written >= 0 && (size_t)written < size;
}

static int tuple_path(const Paths *paths, uint64_t id, char *buffer,
                      size_t size) {
    int written = snprintf(buffer, size, "%s/node-%020" PRIu64 ".bin",
                           paths->checkpoint_dir, id);
    return written >= 0 && (size_t)written < size;
}

static void checkpoint_init(Checkpoint *checkpoint) {
    memset(checkpoint, 0, sizeof(*checkpoint));
}

static int checkpoint_validate_layout(const Checkpoint *checkpoint) {
    size_t i;
    unsigned long previous_end = 0;

    if (checkpoint->target_digits == 0 || checkpoint->total_iterations == 0 ||
        checkpoint->completed_iterations == 0 ||
        checkpoint->completed_iterations > checkpoint->total_iterations ||
        checkpoint->count == 0 || checkpoint->count > MAX_CHECKPOINT_NODES) {
        return 0;
    }
    for (i = 0; i < checkpoint->count; ++i) {
        const CheckpointNode *node = &checkpoint->nodes[i];
        if (node->id == 0 || node->id > checkpoint->generation ||
            node->start != previous_end || node->end <= node->start ||
            node->end > checkpoint->total_iterations) {
            return 0;
        }
        previous_end = node->end;
    }
    return previous_end == checkpoint->completed_iterations;
}

static int tuple_write(const Paths *paths, uint64_t id, unsigned long start,
                       unsigned long end, mpz_t p, mpz_t q, mpz_t t) {
    char path[PATH_MAX];
    char temporary_path[PATH_MAX];
    FILE *file;
    int ok = 1;

    if (!ensure_checkpoint_directory(paths) ||
        !tuple_path(paths, id, path, sizeof(path)) ||
        !make_temp_path(temporary_path, sizeof(temporary_path), path)) {
        fprintf(stderr, "[!] Tuple checkpoint path is too long.\n");
        return 0;
    }
    file = fopen(temporary_path, "wb");
    if (file == NULL) {
        fprintf(stderr, "[!] Could not write %s: %s\n", temporary_path,
                strerror(errno));
        return 0;
    }

    ok &= fwrite(TUPLE_MAGIC, 1, sizeof(TUPLE_MAGIC), file) ==
          sizeof(TUPLE_MAGIC);
    ok &= write_u64(file, TUPLE_VERSION);
    ok &= write_u64(file, id);
    ok &= write_u64(file, start);
    ok &= write_u64(file, end);
    ok &= mpz_out_raw(file, p) != 0;
    ok &= mpz_out_raw(file, q) != 0;
    ok &= mpz_out_raw(file, t) != 0;
    ok &= fwrite(TUPLE_END_MAGIC, 1, sizeof(TUPLE_END_MAGIC), file) ==
          sizeof(TUPLE_END_MAGIC);

    if (!ok) {
        fprintf(stderr, "[!] Failed while serializing %s.\n", temporary_path);
        (void)fclose(file);
        (void)remove(temporary_path);
        return 0;
    }
    return finish_atomic_write(file, temporary_path, path);
}

static int tuple_read(const Paths *paths, const CheckpointNode *node,
                      mpz_t p, mpz_t q, mpz_t t) {
    char path[PATH_MAX];
    FILE *file;
    unsigned char magic[8];
    uint64_t value;
    unsigned long parsed_start;
    unsigned long parsed_end;
    int valid = 1;

    if (!tuple_path(paths, node->id, path, sizeof(path))) {
        return 0;
    }
    file = fopen(path, "rb");
    if (file == NULL) {
        return 0;
    }
    if (fread(magic, 1, sizeof(magic), file) != sizeof(magic) ||
        memcmp(magic, TUPLE_MAGIC, sizeof(magic)) != 0 ||
        !read_u64(file, &value) || value != TUPLE_VERSION ||
        !read_u64(file, &value) || value != node->id ||
        !read_u64(file, &value) || !u64_to_ulong(value, &parsed_start) ||
        !read_u64(file, &value) || !u64_to_ulong(value, &parsed_end) ||
        parsed_start != node->start || parsed_end != node->end ||
        mpz_inp_raw(p, file) == 0 || mpz_inp_raw(q, file) == 0 ||
        mpz_inp_raw(t, file) == 0 ||
        fread(magic, 1, sizeof(magic), file) != sizeof(magic) ||
        memcmp(magic, TUPLE_END_MAGIC, sizeof(magic)) != 0 ||
        fgetc(file) != EOF || ferror(file) || mpz_sgn(p) <= 0 ||
        mpz_sgn(q) <= 0 || mpz_sgn(t) == 0) {
        valid = 0;
    }
    (void)fclose(file);
    return valid;
}

static int tuple_validate(const Paths *paths, const CheckpointNode *node) {
    mpz_t p, q, t;
    int valid;

    mpz_inits(p, q, t, NULL);
    valid = tuple_read(paths, node, p, q, t);
    mpz_clears(p, q, t, NULL);
    return valid;
}

static int manifest_write(const Paths *paths, const Checkpoint *checkpoint) {
    char path[PATH_MAX];
    char temporary_path[PATH_MAX];
    FILE *file;
    size_t i;
    int ok = 1;

    if (!checkpoint_validate_layout(checkpoint) ||
        !ensure_checkpoint_directory(paths) ||
        !manifest_path(paths, path, sizeof(path)) ||
        !make_temp_path(temporary_path, sizeof(temporary_path), path)) {
        fprintf(stderr, "[!] Could not prepare checkpoint manifest.\n");
        return 0;
    }
    file = fopen(temporary_path, "wb");
    if (file == NULL) {
        fprintf(stderr, "[!] Could not write %s: %s\n", temporary_path,
                strerror(errno));
        return 0;
    }
    ok &= fwrite(MANIFEST_MAGIC, 1, sizeof(MANIFEST_MAGIC), file) ==
          sizeof(MANIFEST_MAGIC);
    ok &= write_u64(file, MANIFEST_VERSION);
    ok &= write_u64(file, checkpoint->target_digits);
    ok &= write_u64(file, checkpoint->total_iterations);
    ok &= write_u64(file, checkpoint->completed_iterations);
    ok &= write_u64(file, checkpoint->generation);
    ok &= write_u64(file, checkpoint->count);
    for (i = 0; ok && i < checkpoint->count; ++i) {
        const CheckpointNode *node = &checkpoint->nodes[i];
        ok &= write_u64(file, node->id);
        ok &= write_u64(file, node->start);
        ok &= write_u64(file, node->end);
    }
    ok &= fwrite(MANIFEST_END_MAGIC, 1, sizeof(MANIFEST_END_MAGIC), file) ==
          sizeof(MANIFEST_END_MAGIC);

    if (!ok) {
        fprintf(stderr, "[!] Failed while serializing %s.\n", temporary_path);
        (void)fclose(file);
        (void)remove(temporary_path);
        return 0;
    }
    return finish_atomic_write(file, temporary_path, path);
}

static CheckpointLoadResult checkpoint_load(const Paths *paths,
                                             Checkpoint *checkpoint) {
    char path[PATH_MAX];
    FILE *file;
    unsigned char magic[8];
    uint64_t value;
    uint64_t count;
    size_t i;
    int valid = 1;

    checkpoint_init(checkpoint);
    if (!manifest_path(paths, path, sizeof(path))) {
        return CHECKPOINT_INVALID;
    }
    file = fopen(path, "rb");
    if (file == NULL) {
        return errno == ENOENT ? CHECKPOINT_MISSING : CHECKPOINT_INVALID;
    }
    if (fread(magic, 1, sizeof(magic), file) != sizeof(magic) ||
        memcmp(magic, MANIFEST_MAGIC, sizeof(magic)) != 0 ||
        !read_u64(file, &value) || value != MANIFEST_VERSION ||
        !read_u64(file, &value) ||
        !u64_to_ulong(value, &checkpoint->target_digits) ||
        !read_u64(file, &value) ||
        !u64_to_ulong(value, &checkpoint->total_iterations) ||
        !read_u64(file, &value) ||
        !u64_to_ulong(value, &checkpoint->completed_iterations) ||
        !read_u64(file, &checkpoint->generation) ||
        !read_u64(file, &count) || count == 0 ||
        count > MAX_CHECKPOINT_NODES) {
        valid = 0;
    }
    if (valid) {
        checkpoint->count = (size_t)count;
        for (i = 0; i < checkpoint->count; ++i) {
            CheckpointNode *node = &checkpoint->nodes[i];
            if (!read_u64(file, &node->id) || !read_u64(file, &value) ||
                !u64_to_ulong(value, &node->start) ||
                !read_u64(file, &value) ||
                !u64_to_ulong(value, &node->end)) {
                valid = 0;
                break;
            }
        }
    }
    if (valid &&
        (fread(magic, 1, sizeof(magic), file) != sizeof(magic) ||
         memcmp(magic, MANIFEST_END_MAGIC, sizeof(magic)) != 0 ||
         fgetc(file) != EOF || ferror(file))) {
        valid = 0;
    }
    (void)fclose(file);

    if (valid) {
        valid = checkpoint_validate_layout(checkpoint);
    }
    /*
     * A manifest is usable only when every named immutable tuple validates.
     * A leftover manifest.bin.tmp is intentionally ignored.
     */
    for (i = 0; valid && i < checkpoint->count; ++i) {
        valid = tuple_validate(paths, &checkpoint->nodes[i]);
    }
    if (!valid) {
        fprintf(stderr,
                "[!] Ignoring corrupt/incomplete checkpoint manifest; "
                "restarting from the committed state target.\n");
        checkpoint_init(checkpoint);
        return CHECKPOINT_INVALID;
    }
    return CHECKPOINT_VALID;
}

static void state_set_checkpoint(State *state, const Checkpoint *checkpoint) {
    state->checkpoint_active = 1;
    state->checkpoint_target_digits = checkpoint->target_digits;
    state->checkpoint_completed_iterations = checkpoint->completed_iterations;
    state->checkpoint_total_iterations = checkpoint->total_iterations;
}

static void state_clear_checkpoint(State *state) {
    state->checkpoint_active = 0;
    state->checkpoint_target_digits = 0;
    state->checkpoint_completed_iterations = 0;
    state->checkpoint_total_iterations = 0;
}

/*
 * Tuple(s) are fsynced before manifest.bin, and manifest.bin before state.
 * Thus state.json can never advertise work whose inputs are absent.  If a
 * crash occurs after the manifest but before state, the self-describing
 * manifest is selected as the newer source of truth at startup.
 */
static int persist_checkpoint(const Paths *paths, State *state,
                              const Checkpoint *checkpoint) {
    State updated = *state;

    if (!manifest_write(paths, checkpoint)) {
        return 0;
    }
    updated.next_digits = checkpoint->target_digits;
    updated.discarded_checkpoint_target = 0;
    state_set_checkpoint(&updated, checkpoint);
    if (!write_state(paths, &updated)) {
        fprintf(stderr,
                "[!] Manifest is durable but state.json was not updated; "
                "the next run will still resume the manifest.\n");
        return 0;
    }
    *state = updated;
    return 1;
}

static void remove_tuple(const Paths *paths, uint64_t id) {
    char path[PATH_MAX];

    if (tuple_path(paths, id, path, sizeof(path)) &&
        remove(path) != 0 && errno != ENOENT) {
        fprintf(stderr, "[!] Could not remove obsolete %s: %s\n", path,
                strerror(errno));
    }
}

/* Call this only after state has durably made the listed checkpoint stale or
 * explicitly fenced it off.  The manifest is removed before its tuples so a
 * restart never observes a manifest that points at deleted data. */
static int discard_checkpoint_manifest_first(const Paths *paths,
                                             const Checkpoint *checkpoint) {
    char path[PATH_MAX];
    size_t i;

    /*
     * Fence the manifest before deleting tuple files.  Once unlink succeeds,
     * an explicit target override cannot be undone by a later no-argument
     * launch, even if this process is killed during cleanup.
     */
    if (!manifest_path(paths, path, sizeof(path)) ||
        (remove(path) != 0 && errno != ENOENT)) {
        fprintf(stderr, "[!] Could not remove checkpoint manifest %s: %s\n",
                paths->checkpoint_dir, strerror(errno));
        return 0;
    }
    for (i = 0; i < checkpoint->count; ++i) {
        remove_tuple(paths, checkpoint->nodes[i].id);
    }
    return 1;
}

static int checkpoint_append_chunk(const Paths *paths, State *state,
                                   Checkpoint *checkpoint, unsigned long start,
                                   unsigned long end, mpz_t p, mpz_t q,
                                   mpz_t t) {
    CheckpointNode *node;

    if (checkpoint->count >= MAX_CHECKPOINT_NODES ||
        start != checkpoint->completed_iterations || start >= end) {
        return 0;
    }
    ++checkpoint->generation;
    if (checkpoint->generation == 0 ||
        !tuple_write(paths, checkpoint->generation, start, end, p, q, t)) {
        return 0;
    }
    node = &checkpoint->nodes[checkpoint->count++];
    node->id = checkpoint->generation;
    node->start = start;
    node->end = end;
    checkpoint->completed_iterations = end;
    return persist_checkpoint(paths, state, checkpoint);
}

static int checkpoint_merge_nodes(const Paths *paths, State *state,
                                  Checkpoint *checkpoint, size_t index) {
    CheckpointNode left;
    CheckpointNode right;
    CheckpointNode *replacement;
    mpz_t p1, q1, t1, p2, q2, t2, temporary;
    size_t i;
    int ok;

    if (index + 1U >= checkpoint->count ||
        checkpoint->nodes[index].end != checkpoint->nodes[index + 1U].start) {
        return 0;
    }
    left = checkpoint->nodes[index];
    right = checkpoint->nodes[index + 1U];
    mpz_inits(p1, q1, t1, p2, q2, t2, temporary, NULL);
    ok = tuple_read(paths, &left, p1, q1, t1) &&
         tuple_read(paths, &right, p2, q2, t2);
    if (ok) {
        /* T = P_left * T_right + Q_right * T_left */
        mpz_mul(temporary, p1, t2);
        mpz_mul(t1, t1, q2);
        mpz_add(t1, t1, temporary);
        mpz_mul(p1, p1, p2);
        mpz_mul(q1, q1, q2);
        ++checkpoint->generation;
        ok = checkpoint->generation != 0 &&
             tuple_write(paths, checkpoint->generation, left.start, right.end,
                         p1, q1, t1);
    }
    mpz_clears(p1, q1, t1, p2, q2, t2, temporary, NULL);
    if (!ok) {
        return 0;
    }

    replacement = &checkpoint->nodes[index];
    replacement->id = checkpoint->generation;
    replacement->start = left.start;
    replacement->end = right.end;
    for (i = index + 1U; i + 1U < checkpoint->count; ++i) {
        checkpoint->nodes[i] = checkpoint->nodes[i + 1U];
    }
    --checkpoint->count;

    /*
     * The parent tuple is already durable.  Do not remove either child until
     * the manifest points at the parent; otherwise a crash would lose both.
     */
    if (!persist_checkpoint(paths, state, checkpoint)) {
        return 0;
    }
    remove_tuple(paths, left.id);
    remove_tuple(paths, right.id);
    return 1;
}

static int should_pause(double process_start, const RunConfig *config,
                        unsigned long chunks_this_run);

static int checkpoint_merge_equal_tail(const Paths *paths, State *state,
                                       Checkpoint *checkpoint,
                                       double process_start,
                                       const RunConfig *config,
                                       unsigned long chunks_this_run,
                                       int *paused) {
    *paused = 0;
    while (checkpoint->count >= 2U) {
        CheckpointNode *left = &checkpoint->nodes[checkpoint->count - 2U];
        CheckpointNode *right = &checkpoint->nodes[checkpoint->count - 1U];
        if (left->end != right->start ||
            left->end - left->start != right->end - right->start) {
            break;
        }
        /*
         * Merges are individually atomic, but may be expensive at large
         * precisions. Never begin the next one after this slice has expired.
         */
        if (should_pause(process_start, config, chunks_this_run)) {
            *paused = 1;
            break;
        }
        if (!checkpoint_merge_nodes(paths, state, checkpoint,
                                    checkpoint->count - 2U)) {
            return 0;
        }
    }
    return 1;
}

static int checkpoint_reduce_one(const Paths *paths, State *state,
                                 Checkpoint *checkpoint) {
    return checkpoint_merge_nodes(paths, state, checkpoint, 0);
}

static unsigned long iterations_for_digits(unsigned long digits) {
    return (unsigned long)((long double)digits / 14.181647462L) + 1UL;
}

static int total_iterations_for_target(unsigned long target_digits,
                                       unsigned long *total_iterations) {
    unsigned long working_digits;

    if (target_digits > ULONG_MAX - GUARD_DIGITS) {
        return 0;
    }
    working_digits = target_digits + GUARD_DIGITS;
    *total_iterations = iterations_for_digits(working_digits);
    return *total_iterations != 0;
}

static unsigned long choose_chunk_terms(unsigned long total_iterations,
                                        const RunConfig *config) {
    if (config->chunk_terms != 0) {
        return config->chunk_terms < total_iterations
                   ? config->chunk_terms
                   : total_iterations;
    }
    /*
     * Never let an automatically selected checkpoint boundary scale with the
     * target.  A record-size target would otherwise create a multi-million
     * term, non-interruptible binary-splitting call that can outlive a hosted
     * runner slice and make no durable progress.
     */
    return total_iterations < DEFAULT_CHUNK_TERMS
               ? total_iterations
               : DEFAULT_CHUNK_TERMS;
}

static int runtime_expired(double process_start, const RunConfig *config) {
    return config->max_runtime_seconds != 0 &&
           omp_get_wtime() - process_start >=
               (double)config->max_runtime_seconds;
}

static int should_pause(double process_start, const RunConfig *config,
                        unsigned long chunks_this_run) {
    return g_stop || runtime_expired(process_start, config) ||
           (config->max_chunks_per_run != 0 &&
            chunks_this_run >= config->max_chunks_per_run);
}

/* ---------- Parallel binary splitting ---------- */

static void bs(unsigned long a, unsigned long b, mpz_t p, mpz_t q, mpz_t t,
               int threads) {
    if (b - a == 1UL) {
        if (a == 0UL) {
            mpz_set_ui(p, 1UL);
            mpz_set_ui(q, 1UL);
        } else {
            mpz_set_ui(p, 2UL * a - 1UL);
            mpz_mul_ui(p, p, 6UL * a - 1UL);
            mpz_mul_ui(p, p, 6UL * a - 5UL);

            mpz_set_ui(q, a);
            mpz_mul_ui(q, q, a);
            mpz_mul_ui(q, q, a);
            mpz_mul(q, q, CONST_Q);
        }

        mpz_set_ui(t, 545140134UL);
        mpz_mul_ui(t, t, a);
        mpz_add_ui(t, t, 13591409UL);
        mpz_mul(t, t, p);
        if (a & 1UL) {
            mpz_neg(t, t);
        }
    } else {
        unsigned long middle = a + (b - a) / 2UL;
        mpz_t p2, q2, t2, temporary;

        mpz_inits(p2, q2, t2, NULL);
        if (threads > 1) {
#pragma omp task shared(p, q, t)
            bs(a, middle, p, q, t, threads / 2);
#pragma omp task shared(p2, q2, t2)
            bs(middle, b, p2, q2, t2, threads - threads / 2);
#pragma omp taskwait
        } else {
            bs(a, middle, p, q, t, 1);
            bs(middle, b, p2, q2, t2, 1);
        }
        mpz_init(temporary);
        mpz_mul(temporary, p, t2);
        mpz_mul(t, t, q2);
        mpz_add(t, t, temporary);
        mpz_mul(p, p, p2);
        mpz_mul(q, q, q2);
        mpz_clears(p2, q2, t2, temporary, NULL);
    }
}

static int calculate_pi(const Paths *paths, unsigned long target_digits,
                        const Checkpoint *checkpoint, mpz_t pi_scaled) {
    const CheckpointNode *root;
    mpz_t p, q, t, ten_pow, c_scaled, guard_scale;
    unsigned long working_digits;
    int ok = 0;

    if (checkpoint->count != 1U ||
        checkpoint->completed_iterations != checkpoint->total_iterations ||
        target_digits > ULONG_MAX - GUARD_DIGITS) {
        return 0;
    }
    working_digits = target_digits + GUARD_DIGITS;
    if (working_digits > ULONG_MAX / 2UL) {
        return 0;
    }
    root = &checkpoint->nodes[0];
    mpz_inits(p, q, t, ten_pow, c_scaled, guard_scale, NULL);
    if (!tuple_read(paths, root, p, q, t)) {
        fprintf(stderr, "[!] Final checkpoint tuple is unreadable.\n");
        goto done;
    }
    mpz_ui_pow_ui(ten_pow, 10UL, 2UL * working_digits);
    mpz_mul_ui(c_scaled, ten_pow, 10005UL);
    mpz_sqrt(c_scaled, c_scaled);
    mpz_mul_ui(c_scaled, c_scaled, 426880UL);
    mpz_mul(pi_scaled, c_scaled, q);
    mpz_tdiv_q(pi_scaled, pi_scaled, t);

    /* Discard fixed guard digits only after all high-precision math is done. */
    mpz_ui_pow_ui(guard_scale, 10UL, GUARD_DIGITS);
    mpz_tdiv_q(pi_scaled, pi_scaled, guard_scale);
    ok = 1;

done:
    mpz_clears(p, q, t, ten_pow, c_scaled, guard_scale, NULL);
    return ok;
}

static int write_zeros(FILE *file, unsigned long count) {
    char zeros[8192];
    unsigned long remaining = count;

    memset(zeros, '0', sizeof(zeros));
    while (remaining != 0) {
        size_t batch = remaining < sizeof(zeros) ? (size_t)remaining
                                                  : sizeof(zeros);
        if (fwrite(zeros, 1, batch, file) != batch) {
            return 0;
        }
        remaining -= (unsigned long)batch;
    }
    return 1;
}

static int validate_and_digest_pi_file(const char *path,
                                       unsigned long target_digits,
                                       PiFileInfo *info) {
    static const char known_prefix[] =
        "3.14159265358979323846264338327950288419716939937510";
    FILE *file = fopen(path, "rb");
    unsigned char buffer[65536];
    char prefix[sizeof(known_prefix) - 1U];
    size_t bytes_read;
    size_t prefix_length;
    uint64_t bytes = 0;
    uint64_t checksum = UINT64_C(14695981039346656037);

    if (file == NULL) {
        return 0;
    }
    prefix_length = (uint64_t)target_digits + 2U <
                            (uint64_t)(sizeof(known_prefix) - 1U)
                        ? (size_t)((uint64_t)target_digits + 2U)
                        : sizeof(known_prefix) - 1U;
    if (fread(prefix, 1, prefix_length, file) != prefix_length ||
        memcmp(prefix, known_prefix, prefix_length) != 0) {
        fprintf(stderr, "[!] Computed output does not match the known pi prefix.\n");
        (void)fclose(file);
        return 0;
    }
    rewind(file);
    while ((bytes_read = fread(buffer, 1, sizeof(buffer), file)) != 0) {
        bytes += bytes_read;
        checksum = fnv1a_update(checksum, buffer, bytes_read);
    }
    if (ferror(file)) {
        (void)fclose(file);
        return 0;
    }
    (void)fclose(file);
    if (bytes != (uint64_t)target_digits + 2U) {
        fprintf(stderr, "[!] Output size does not match the requested precision.\n");
        return 0;
    }
    info->bytes = bytes;
    info->checksum = checksum;
    return 1;
}

/*
 * mpz_out_str streams the fractional integer directly to disk.  This avoids
 * allocating another decimal string as large as the result just to save it.
 */
static int save_pi(const Paths *paths, mpz_t pi_scaled,
                   unsigned long target_digits, PiFileInfo *info) {
    char temporary_path[PATH_MAX];
    FILE *file;
    mpz_t scale, integer_part, fractional_part;
    size_t fractional_length;
    size_t written;
    int ok = 1;

    if (target_digits > ULONG_MAX - 2UL ||
        !make_temp_path(temporary_path, sizeof(temporary_path),
                        paths->pi_file)) {
        return 0;
    }
    file = fopen(temporary_path, "wb");
    if (file == NULL) {
        fprintf(stderr, "[!] Could not write %s: %s\n", temporary_path,
                strerror(errno));
        return 0;
    }
    mpz_inits(scale, integer_part, fractional_part, NULL);
    mpz_ui_pow_ui(scale, 10UL, target_digits);
    mpz_fdiv_q(integer_part, pi_scaled, scale);
    mpz_mod(fractional_part, pi_scaled, scale);
    fractional_length = mpz_sizeinbase(fractional_part, 10);
    if (mpz_cmp_ui(integer_part, 3UL) != 0 ||
        fractional_length > (size_t)target_digits) {
        ok = 0;
    }
    if (ok) {
        ok &= fwrite("3.", 1, 2, file) == 2;
        ok &= write_zeros(file, target_digits - (unsigned long)fractional_length);
        written = mpz_out_str(file, 10, fractional_part);
        ok &= written == fractional_length;
    }
    mpz_clears(scale, integer_part, fractional_part, NULL);
    if (!ok) {
        fprintf(stderr, "[!] Failed while streaming %s.\n", temporary_path);
        (void)fclose(file);
        (void)remove(temporary_path);
        return 0;
    }
    /*
     * Verify the temporary output before replacing the prior completed file;
     * a math regression must not destroy the last known-good decimal result.
     */
    if (!flush_and_close(file)) {
        (void)remove(temporary_path);
        return 0;
    }
    if (!validate_and_digest_pi_file(temporary_path, target_digits, info)) {
        (void)remove(temporary_path);
        return 0;
    }
    if (!atomic_replace(temporary_path, paths->pi_file)) {
        fprintf(stderr, "[!] Could not replace %s: %s\n", paths->pi_file,
                strerror(errno));
        (void)remove(temporary_path);
        return 0;
    }
    return 1;
}

static RunResult run_target(const Paths *paths, const RunConfig *config,
                            State *state, Checkpoint *checkpoint,
                            unsigned long target_digits,
                            double process_start,
                            unsigned long *chunks_this_run) {
    unsigned long total_iterations;
    unsigned long chunk_terms;
    int merge_paused;
    int num_threads = omp_get_max_threads();

    if (!total_iterations_for_target(target_digits, &total_iterations)) {
        fprintf(stderr, "[!] Target is too large for this build.\n");
        return RUN_ERROR;
    }
    if (checkpoint->count != 0 &&
        (checkpoint->target_digits != target_digits ||
         checkpoint->total_iterations != total_iterations)) {
        fprintf(stderr, "[~] Ignoring checkpoint for a different target.\n");
        checkpoint_init(checkpoint);
    }
    if (checkpoint->count == 0) {
        checkpoint_init(checkpoint);
        checkpoint->target_digits = target_digits;
        checkpoint->total_iterations = total_iterations;
    }

    chunk_terms = choose_chunk_terms(total_iterations, config);
    printf("[+] Target %lu digits: %lu/%lu terms complete, chunks of %lu "
           "terms (%d threads).\n",
           target_digits, checkpoint->completed_iterations, total_iterations,
           chunk_terms, num_threads);
    fflush(stdout);

    /* Finish any deferred small-stack work before accepting another chunk. */
    if (!checkpoint_merge_equal_tail(paths, state, checkpoint, process_start,
                                     config, *chunks_this_run,
                                     &merge_paused)) {
        return RUN_ERROR;
    }
    if (merge_paused) {
        printf("[~] Paused before deferred checkpoint reductions.\n");
        return RUN_PAUSED;
    }

    while (checkpoint->completed_iterations < total_iterations) {
        unsigned long start;
        unsigned long end;
        double chunk_start;
        mpz_t p, q, t;

        if (should_pause(process_start, config, *chunks_this_run)) {
            printf("[~] Pausing before the next chunk; durable work is retained.\n");
            return RUN_PAUSED;
        }
        start = checkpoint->completed_iterations;
        end = total_iterations - start > chunk_terms
                  ? start + chunk_terms
                  : total_iterations;
        printf("  [-] Calculating terms %lu..%lu\n", start, end);
        fflush(stdout);
        chunk_start = omp_get_wtime();

        mpz_inits(p, q, t, NULL);
#pragma omp parallel
        {
#pragma omp single
            bs(start, end, p, q, t, num_threads);
        }
        if (!checkpoint_append_chunk(paths, state, checkpoint, start, end,
                                     p, q, t)) {
            mpz_clears(p, q, t, NULL);
            return RUN_ERROR;
        }
        mpz_clears(p, q, t, NULL);
        ++*chunks_this_run;
        printf("  [OK] Checkpointed %lu/%lu terms in %.2fs.\n",
               checkpoint->completed_iterations, total_iterations,
               omp_get_wtime() - chunk_start);
        fflush(stdout);

        if (should_pause(process_start, config, *chunks_this_run)) {
            printf("[~] Paused at a durable chunk boundary.\n");
            return RUN_PAUSED;
        }
        if (!checkpoint_merge_equal_tail(paths, state, checkpoint,
                                         process_start, config,
                                         *chunks_this_run, &merge_paused)) {
            return RUN_ERROR;
        }
        if (merge_paused) {
            printf("[~] Paused before checkpoint reductions.\n");
            return RUN_PAUSED;
        }
    }

    /*
     * The final stack usually has only log2(chunk-count) nodes, but each
     * reduction is still journaled so a kill during final combination does
     * not re-run the finished chunks.
     */
    while (checkpoint->count > 1U) {
        if (should_pause(process_start, config, *chunks_this_run)) {
            printf("[~] Paused before final checkpoint reduction.\n");
            return RUN_PAUSED;
        }
        if (!checkpoint_reduce_one(paths, state, checkpoint)) {
            return RUN_ERROR;
        }
        printf("  [-] Reduced checkpoint stack to %llu ranges.\n",
               (unsigned long long)checkpoint->count);
        fflush(stdout);
    }

    if (should_pause(process_start, config, *chunks_this_run)) {
        printf("[~] Full math checkpoint is durable; output will be written "
               "on resume.\n");
        return RUN_PAUSED;
    }

    {
        State completed = *state;
        PiFileInfo pi_info;
        mpz_t pi_scaled;
        unsigned long next_digits;
        double math_start = omp_get_wtime();

        mpz_init(pi_scaled);
        if (!calculate_pi(paths, target_digits, checkpoint, pi_scaled)) {
            mpz_clear(pi_scaled);
            return RUN_ERROR;
        }
        printf("  [-] Final math with %lu guard digits: %.2fs\n",
               GUARD_DIGITS, omp_get_wtime() - math_start);
        fflush(stdout);
        if (!save_pi(paths, pi_scaled, target_digits, &pi_info)) {
            fprintf(stderr,
                    "[!] Output was not saved; keeping the full checkpoint "
                    "for retry.\n");
            mpz_clear(pi_scaled);
            return RUN_ERROR;
        }
        mpz_clear(pi_scaled);

        if (target_digits > ULONG_MAX / 2UL) {
            fprintf(stderr,
                    "[!] Cannot automatically double %lu digits on this "
                    "build; keeping it as the next target.\n",
                    target_digits);
            next_digits = target_digits;
        } else {
            next_digits = target_digits * 2UL;
        }
        completed.completed_digits = target_digits;
        completed.next_digits = next_digits;
        completed.pi_bytes = pi_info.bytes;
        completed.pi_checksum = pi_info.checksum;
        completed.discarded_checkpoint_target = 0;
        state_clear_checkpoint(&completed);

        /* Never advance the manifest until the decimal output is durable. */
        if (!write_state(paths, &completed)) {
            fprintf(stderr,
                    "[!] Output exists but state was not advanced; retaining "
                    "the complete checkpoint for retry.\n");
            return RUN_ERROR;
        }
        *state = completed;
        /*
         * Keep the complete root through process exit.  The workflow caches
         * it before uploading the result and committing state, so a transient
         * upload failure can regenerate the output without repeating the
         * calculation.  A later launch sees it as stale (the completed count
         * now includes this target) and removes it before starting the next
         * target.
         */
        printf("[OK] %lu digits saved; next target is %lu digits.\n",
               target_digits, next_digits);
        fflush(stdout);
    }
    return RUN_DONE;
}

static void print_usage(const char *program) {
    fprintf(stderr,
            "Usage: %s [digits]\n"
            "Environment controls:\n"
            "  PI_MAX_RUNTIME_SECONDS=seconds (0 disables the limit)\n"
            "  PI_CHUNK_TERMS=terms (checkpoint frequency)\n"
            "  PI_MAX_CHUNKS_PER_RUN=count (deterministic graceful pause)\n"
            "  PI_STATE_FILE, PI_OUTPUT_FILE, PI_CHECKPOINT_DIR, "
            "PI_CHECKPOINT_FILE\n",
            program);
}

int main(int argc, char **argv) {
    Paths paths;
    RunConfig config;
    State state;
    Checkpoint checkpoint;
    CheckpointLoadResult checkpoint_result;
    unsigned long target_digits;
    unsigned long cli_digits = 0;
    unsigned long chunks_this_run = 0;
    int cli_target = 0;
    int legacy_state = 0;
    double process_start;

    if (argc > 2) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }
    if (argc == 2) {
        if (!parse_ulong(argv[1], 0, &cli_digits)) {
            fprintf(stderr, "[!] Invalid digit target: %s\n", argv[1]);
            return EXIT_FAILURE;
        }
        cli_target = 1;
    }

    paths.state_file = path_from_env("PI_STATE_FILE", DEFAULT_STATE_FILE);
    paths.pi_file = path_from_env("PI_OUTPUT_FILE", DEFAULT_PI_FILE);
    paths.checkpoint_dir =
        path_from_env("PI_CHECKPOINT_DIR", DEFAULT_CHECKPOINT_DIR);
    paths.checkpoint_manifest = getenv("PI_CHECKPOINT_FILE");
    config.max_runtime_seconds = env_ulong(
        "PI_MAX_RUNTIME_SECONDS", DEFAULT_MAX_RUNTIME_SECONDS, 1);
    config.chunk_terms = env_ulong("PI_CHUNK_TERMS", 0, 0);
    config.max_chunks_per_run =
        env_ulong("PI_MAX_CHUNKS_PER_RUN", 0, 1);

    signal(SIGTERM, on_signal);
    signal(SIGINT, on_signal);

    if (!read_state(&paths, &state, &legacy_state)) {
        fprintf(stderr,
                "[!] State read failed; refusing to overwrite existing progress.\n");
        return EXIT_FAILURE;
    }
    if (legacy_state) {
        printf("[~] Loaded legacy state; it will be upgraded at checkpoint.\n");
    }

    checkpoint_result = checkpoint_load(&paths, &checkpoint);
    if (cli_target) {
        target_digits = cli_digits;
        if (checkpoint_result == CHECKPOINT_VALID &&
            checkpoint.target_digits != target_digits) {
            State overridden = state;

            printf("[~] Explicit target discards a checkpoint for %lu digits.\n",
                   checkpoint.target_digits);
            /*
             * Fence the old checkpoint in durable state before unlinking its
             * manifest.  If this process dies during cleanup, a later
             * no-argument invocation will honor the fence instead of
             * resurrecting the user-discarded target.
             */
            overridden.next_digits = target_digits;
            overridden.discarded_checkpoint_target = checkpoint.target_digits;
            state_clear_checkpoint(&overridden);
            if (!write_state(&paths, &overridden)) {
                return EXIT_FAILURE;
            }
            state = overridden;
            if (!discard_checkpoint_manifest_first(&paths, &checkpoint)) {
                fprintf(stderr,
                        "[!] The old checkpoint is fenced in state but could "
                        "not be cleaned up.\n");
                return EXIT_FAILURE;
            }
            checkpoint_init(&checkpoint);
        } else if (target_digits != state.next_digits) {
            state.next_digits = target_digits;
            state.discarded_checkpoint_target = 0;
            state_clear_checkpoint(&state);
            if (!write_state(&paths, &state)) {
                return EXIT_FAILURE;
            }
        }
    } else if (checkpoint_result == CHECKPOINT_VALID &&
               checkpoint.target_digits == state.discarded_checkpoint_target) {
        printf("[~] Ignoring a checkpoint explicitly discarded for %lu digits.\n",
               checkpoint.target_digits);
        (void)discard_checkpoint_manifest_first(&paths, &checkpoint);
        checkpoint_init(&checkpoint);
        target_digits = state.next_digits;
    } else if (checkpoint_result == CHECKPOINT_VALID &&
               checkpoint.target_digits > state.completed_digits &&
               total_iterations_for_target(checkpoint.target_digits,
                                           &target_digits) &&
               checkpoint.total_iterations == target_digits) {
        /*
         * A fully validated manifest takes precedence over an older state
         * file, including the crash window after manifest replacement.
         */
        target_digits = checkpoint.target_digits;
        if (state.next_digits != target_digits) {
            printf("[~] Resuming newer manifest target %lu (state requested "
                   "%lu).\n",
                   target_digits, state.next_digits);
            state.next_digits = target_digits;
        }
    } else {
        if (checkpoint_result == CHECKPOINT_VALID) {
            printf("[~] Ignoring stale checkpoint for %lu digits.\n",
                   checkpoint.target_digits);
            if (checkpoint.target_digits <= state.completed_digits) {
                (void)discard_checkpoint_manifest_first(&paths, &checkpoint);
            }
            checkpoint_init(&checkpoint);
        }
        target_digits = state.next_digits;
    }

    mpz_init_set_str(CONST_Q, "10939058860032000", 10);
    process_start = omp_get_wtime();
    {
        RunResult result = run_target(&paths, &config, &state, &checkpoint,
                                      target_digits, process_start,
                                      &chunks_this_run);
        mpz_clear(CONST_Q);
        if (result == RUN_ERROR) {
            return EXIT_FAILURE;
        }
        /*
         * A result is published and its state committed by the workflow as a
         * single-target transaction.  Do not checkpoint the next target in
         * this process, or an output-upload failure could leave a newer
         * cached checkpoint that causes the completed output to be skipped.
         */
        return EXIT_SUCCESS;
    }
}
