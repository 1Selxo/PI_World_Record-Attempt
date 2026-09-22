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

static volatile sig_atomic_t g_stop = 0;

static void on_signal(int sig) {
    (void)sig;
    g_stop = 1;
}

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

static int save_pi(const char *filename, mpz_t pi_int) {
    char tmpname[128];
    snprintf(tmpname, sizeof(tmpname), "%s.tmp", filename);
    FILE *f = fopen(tmpname, "w");
    if (!f) return 0;
    
    // Convert to base 10 string
    char *str = mpz_get_str(NULL, 10, pi_int);
    fprintf(f, "3.%s", str + 1);
    fclose(f);

    void (*freefunc)(void *, size_t);
    mp_get_memory_functions(NULL, NULL, &freefunc);
    freefunc(str, strlen(str) + 1);

    return rename(tmpname, filename) == 0;
}

// Automatically sync to Github
static void sync_to_github(unsigned long digits) {
    printf("  [~] Syncing %lu digits to GitHub...\n", digits);
    char command[512];
    snprintf(command, sizeof(command), 
        "git add %s %s && git commit -m \"Auto-save: Calculated %lu digits\" && git push", 
        STATE_FILE, PI_FILE, digits);
    
    int ret = system(command);
    if (ret == 0) {
        printf("  [OK] Successfully pushed to GitHub.\n");
    } else {
        printf("  [!] Failed to push to GitHub (Check git credentials/network).\n");
    }
}

// ---------- Cleaned up, hyper-fast Binary Splitting ----------
// Replaces the extremely slow array pooling with proper Log(N) recursion

void bs(unsigned long a, unsigned long b, mpz_t P, mpz_t Q, mpz_t T, int threads) {
    if (b - a == 1) {
        if (a == 0) {
            mpz_set_ui(P, 1);
            mpz_set_ui(Q, 1);
        } else {
            mpz_set_ui(P, 2 * a - 1);
            mpz_mul_ui(P, P, 6 * a - 1);
            mpz_mul_ui(P, P, 6 * a - 5);

            mpz_set_ui(Q, a);
            mpz_mul_ui(Q, Q, a);
            mpz_mul_ui(Q, Q, a);
            mpz_mul(Q, Q, CONST_Q);
        }

        mpz_set_ui(T, 545140134);
        mpz_mul_ui(T, T, a);
        mpz_add_ui(T, T, 13591409);
        mpz_mul(T, T, P);

        if (a & 1) mpz_neg(T, T);
    } else {
        unsigned long m = (a + b) / 2;
        mpz_t P2, Q2, T2;
        mpz_inits(P2, Q2, T2, NULL);

        if (threads > 1) {
            #pragma omp task shared(P, Q, T)
            bs(a, m, P, Q, T, threads / 2);

            #pragma omp task shared(P2, Q2, T2)
            bs(m, b, P2, Q2, T2, threads - (threads / 2));

            #pragma omp taskwait
        } else {
            bs(a, m, P, Q, T, 1);
            bs(m, b, P2, Q2, T2, 1);
        }

        mpz_t tmp;
        mpz_init(tmp);

        // T = P1 * T2 + Q2 * T1
        mpz_mul(tmp, P, T2);
        mpz_mul(T, T, Q2);
        mpz_add(T, T, tmp);

        // P = P1 * P2
        mpz_mul(P, P, P2);
        // Q = Q1 * Q2
        mpz_mul(Q, Q, Q2);

        mpz_clears(P2, Q2, T2, tmp, NULL);
    }
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
            printf("[!] Time budget reached (%.1fs). Saving state for resume.\n", elapsed_total);
            write_state(digits, PI_FILE);
            break;
        }

        printf("\n[+] Calculating %lu digits...\n", digits);
        fflush(stdout);
        double start_time = omp_get_wtime();

        unsigned long iterations = digits / 14.181647462 + 1;

        mpz_t P, Q, T;
        mpz_inits(P, Q, T, NULL);

        // Trigger binary splitting
        #pragma omp parallel
        {
            #pragma omp single
            {
                bs(0, iterations, P, Q, T, num_threads);
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
        write_state(next_digits, PI_FILE);
        
        // PUSH TO GITHUB HERE
        sync_to_github(digits);

        double end_save = omp_get_wtime();
        printf("[OK] %lu digits saved in %.4f s (session total %.4f s)\n",
               digits, end_save - end_calc, end_save - process_start);
        fflush(stdout);

        mpz_clears(P, Q, T, ten_pow, C_scaled, pi_int, NULL);

        double iter_time = end_save - start_time;
        if (omp_get_wtime() - process_start + iter_time * 1.2 >= MAX_RUNTIME_SECONDS) {
            printf("[!] Next iteration (~%.1fs) won't fit. Exiting for resume.\n", iter_time);
            break;
        }

        digits = next_digits;
    }

    mpz_clear(CONST_Q);
    return 0;
}
