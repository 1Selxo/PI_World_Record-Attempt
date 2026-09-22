#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <gmp.h>
#include <time.h>
#include <unistd.h>
#include <signal.h>
#include <omp.h>

mpz_t CONST_Q;

// ---------- Checkpointing helpers ----------

#define MAX_RUNTIME_SECONDS (5 * 3600)   // stop at 5 hours
#define STATE_FILE          "state.json"
#define PI_FILE             "pi_c_hyperspeed.txt"

// Set to non-zero by SIGTERM handler so we exit between phases cleanly.
static volatile sig_atomic_t g_stop = 0;

static void on_signal(int sig) {
    (void)sig;
    g_stop = 1;
}

// Read starting digits from state.json. Returns default if missing/unreadable.
static unsigned long read_start_digits(unsigned long default_digits) {
    FILE *f = fopen(STATE_FILE, "r");
    if (!f) return default_digits;
    char buf[256] = {0};
    fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    char *p = strstr(buf, "\"digits\"");
    if (!p) return default_digits;
    p = strchr(p, ':');
    if (!p) return default_digits;
    p++;
    while (*p == ' ' || *p == '\t') p++;
    unsigned long v = strtoul(p, NULL, 10);
    return v ? v : default_digits;
}

// Atomically write state so a crash mid-write doesn't corrupt it.
static void write_state(unsigned long digits, const char *pi_file) {
    char tmpname[64];
    snprintf(tmpname, sizeof(tmpname), "%s.tmp", STATE_FILE);
    FILE *f = fopen(tmpname, "w");
    if (!f) return;
    fprintf(f,
        "{\n"
        "  \"digits\": %lu,\n"
        "  \"pi_file\": \"%s\",\n"
        "  \"updated\": %ld\n"
        "}\n",
        digits, pi_file, (long)time(NULL));
    fclose(f);
    rename(tmpname, STATE_FILE);
}

// Save pi text atomically (writes to .tmp, then renames).
static int save_pi(const char *filename, mpz_t pi_int) {
    char tmpname[128];
    snprintf(tmpname, sizeof(tmpname), "%s.tmp", filename);
    FILE *f = fopen(tmpname, "w");
    if (!f) return 0;
    char *str = mpz_get_str(NULL, 10, pi_int);
    fprintf(f, "3.%s", str + 1);
    fclose(f);

    void (*freefunc)(void *, size_t);
    mp_get_memory_functions(NULL, NULL, &freefunc);
    freefunc(str, strlen(str) + 1);

    return rename(tmpname, filename) == 0;
}

// ---------- Binary splitting (unchanged core) ----------

void bs_seq(unsigned long a, unsigned long b, int depth, mpz_t P, mpz_t Q, mpz_t T,
            mpz_t *P_pool, mpz_t *Q_pool, mpz_t *T_pool, mpz_t tmp) {
    if (b - a == 1) {
        if (a == 0) {
            mpz_set_ui(P, 1);
            mpz_set_ui(Q, 1);
        } else {
            mpz_set_ui(P, 2 * a - 1);
            mpz_mul_ui(P, P, 6 * a - 1);
            mpz_mul_ui(P, P, 6 * a - 5);

            mpz_set(Q, CONST_Q);
            mpz_mul_ui(Q, Q, a);
            mpz_mul_ui(Q, Q, a);
            mpz_mul_ui(Q, Q, a);
        }

        mpz_set_ui(T, 545140134);
        mpz_mul_ui(T, T, a);
        mpz_add_ui(T, T, 13591409);
        mpz_mul(T, T, P);

        if (a & 1) mpz_neg(T, T);
    } else {
        unsigned long m = (a + b) / 2;

        bs_seq(a, m, depth + 1, P_pool[depth], Q_pool[depth], T_pool[depth],
               P_pool, Q_pool, T_pool, tmp);

        bs_seq(m, b, depth + 1, P, Q, T,
               P_pool, Q_pool, T_pool, tmp);

        mpz_mul(tmp, P_pool[depth], T);
        mpz_mul(T, Q, T_pool[depth]);
        mpz_add(T, T, tmp);

        mpz_mul(P, P_pool[depth], P);
        mpz_mul(Q, Q_pool[depth], Q);
    }
}

void bs_parallel(unsigned long a, unsigned long b, mpz_t P, mpz_t Q, mpz_t T, int threads_left) {
    if (threads_left <= 1 || (b - a) < 1000) {
        mpz_t P_pool[32], Q_pool[32], T_pool[32], tmp;
        for (int i = 0; i < 32; i++) {
            mpz_inits(P_pool[i], Q_pool[i], T_pool[i], NULL);
        }
        mpz_init(tmp);

        bs_seq(a, b, 0, P, Q, T, P_pool, Q_pool, T_pool, tmp);

        for (int i = 0; i < 32; i++) {
            mpz_clears(P_pool[i], Q_pool[i], T_pool[i], NULL);
        }
        mpz_clear(tmp);
        return;
    }

    unsigned long m = (a + b) / 2;
    mpz_t Pam, Qam, Tam;
    mpz_inits(Pam, Qam, Tam, NULL);

    #pragma omp task shared(Pam, Qam, Tam)
    bs_parallel(a, m, Pam, Qam, Tam, threads_left / 2);

    #pragma omp task shared(P, Q, T)
    bs_parallel(m, b, P, Q, T, threads_left - (threads_left / 2));

    #pragma omp taskwait

    mpz_t tmp;
    mpz_init(tmp);

    mpz_mul(tmp, Pam, T);
    mpz_mul(T, Q, Tam);
    mpz_add(T, T, tmp);

    mpz_mul(P, Pam, P);
    mpz_mul(Q, Qam, Q);

    mpz_clears(Pam, Qam, Tam, tmp, NULL);
}

// ---------- Main ----------

int main(int argc, char **argv) {
    signal(SIGTERM, on_signal);
    signal(SIGINT,  on_signal);

    unsigned long digits = (argc > 1)
        ? strtoul(argv[1], NULL, 10)
        : read_start_digits(1000000);

    mpz_init_set_str(CONST_Q, "10939058860032000", 10);

    int num_threads = omp_get_max_threads();
    printf("Resuming from %lu digits (%d threads).\n", digits, num_threads);
    fflush(stdout);

    double process_start = omp_get_wtime();

    while (1) {
        if (g_stop) {
            printf("[!] Signal received. Saving state and exiting.\n");
            write_state(digits, PI_FILE);
            break;
        }

        double elapsed_total = omp_get_wtime() - process_start;
        if (elapsed_total >= MAX_RUNTIME_SECONDS) {
            printf("[!] Time budget reached (%.1fs). Saving state for resume.\n",
                   elapsed_total);
            write_state(digits, PI_FILE);
            break;
        }

        printf("\n[+] Calculating %lu digits...\n", digits);
        fflush(stdout);
        double start_time = omp_get_wtime();

        unsigned long iterations = digits / 14.181647462 + 1;

        mpz_t P, Q, T;
        mpz_inits(P, Q, T, NULL);

        #pragma omp parallel
        {
            #pragma omp single
            {
                bs_parallel(0, iterations, P, Q, T, num_threads);
            }
        }

        double end_bs = omp_get_wtime();
        printf("  [-] Binary splitting: %.4f s\n", end_bs - start_time);
        fflush(stdout);

        mpz_t ten_pow, C_scaled, pi_int;
        mpz_inits(ten_pow, C_scaled, pi_int, NULL);

        mpz_ui_pow_ui(ten_pow, 10, 2 * digits);
        mpz_mul_ui(C_scaled, ten_pow, 10005);
        mpz_sqrt(C_scaled, C_scaled);
        mpz_mul_ui(C_scaled, C_scaled, 426880);

        mpz_mul(pi_int, C_scaled, Q);
        mpz_tdiv_q(pi_int, pi_int, T);

        double end_calc = omp_get_wtime();
        printf("  [-] Final math: %.4f s\n", end_calc - end_bs);
        fflush(stdout);

        if (!save_pi(PI_FILE, pi_int)) {
            fprintf(stderr, "[!] Failed to save pi.\n");
        }

        unsigned long next_digits = digits * 2;

        // Write state AFTER every successful save so a kill mid-loop
        // still resumes from the correct point.
        write_state(next_digits, PI_FILE);

        double end_save = omp_get_wtime();
        printf("[OK] %lu digits saved in %.4f s (session total %.4f s)\n",
               digits, end_save - end_calc, end_save - process_start);
        fflush(stdout);

        mpz_clears(P, Q, T, ten_pow, C_scaled, pi_int, NULL);

        double iter_time = end_save - start_time;

        // If the NEXT iteration clearly won't fit in the remaining budget, stop now.
        if (omp_get_wtime() - process_start + iter_time * 1.2 >= MAX_RUNTIME_SECONDS) {
            printf("[!] Next iteration (~%.1fs) won't fit. Exiting for resume.\n", iter_time);
            break;
        }

        digits = next_digits;
    }

    mpz_clear(CONST_Q);
    return 0;
}
