/* The session lock is reentrant for the thread that holds it: a caller that follows the omega_gpu_session.h
 * pattern (omega_gpu_session_lock(), then call API helpers that take the same lock internally) must not deadlock
 * itself. Host-only: codegen does not touch the device. A watchdog fails the test after 10 s instead of hanging. */
#include "omega_gpu_session.h"
#include "omega_gpu_elementwise_api.h"
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#define WATCHDOG_S 10

static atomic_int g_done;
static atomic_int g_other_in; /* the second thread is inside the lock */

static void sleep_ms(long ms) { struct timespec ts = {.tv_sec = 0, .tv_nsec = ms * 1000 * 1000}; nanosleep(&ts, NULL); }

static void *other(void *arg) {
    (void)arg;
    omega_gpu_session_lock();
    atomic_store(&g_other_in, 1);
    omega_gpu_session_unlock();
    return NULL;
}

static void *worker(void *arg) {
    (void)arg;
    OmegaBlackwellKernel k;
    omega_gpu_session_lock();
    int rc = omega_gpu_elementwise_codegen(OMEGA_GPU_EW_EX2, NULL, 0.0f, &k); /* takes the session lock inside */
    omega_gpu_session_unlock();
    if (rc != OMEGA_GPU_EW_OK) { printf("FAIL codegen under the held lock returned %d\n", rc); _exit(1); }
    free(k.code);

    /* three deep, and the lock must still be held until the outermost unlock */
    omega_gpu_session_lock();
    omega_gpu_session_lock();
    omega_gpu_session_lock();
    omega_gpu_session_unlock();
    omega_gpu_session_unlock();
    omega_gpu_session_unlock();

    /* after the outermost unlock another thread of the process can take it again (a leaked count would hang here) */
    omega_gpu_session_lock();
    omega_gpu_session_unlock();
    atomic_store(&g_done, 1);
    return NULL;
}

int main(void) {
    pthread_t t;
    if (pthread_create(&t, NULL, worker, NULL) != 0) { printf("FAIL pthread_create\n"); return 1; }
    for (int i = 0; i < WATCHDOG_S * 100 && !atomic_load(&g_done); i++) {
        sleep_ms(10);
    }
    if (!atomic_load(&g_done)) {
        printf("FAIL session lock taken twice by one thread did not return within %d s (self-deadlock)\n", WATCHDOG_S);
        fflush(stdout);
        _exit(1);
    }
    pthread_join(t, NULL);

    /* an inner unlock must not release the outer hold: a second thread stays out until the outermost unlock */
    pthread_t o;
    omega_gpu_session_lock();
    omega_gpu_session_lock();
    if (pthread_create(&o, NULL, other, NULL) != 0) { printf("FAIL pthread_create\n"); return 1; }
    sleep_ms(100);
    if (atomic_load(&g_other_in)) { printf("FAIL second thread entered while the lock was held\n"); return 1; }
    omega_gpu_session_unlock();
    sleep_ms(100);
    if (atomic_load(&g_other_in)) { printf("FAIL inner unlock released the outer hold\n"); return 1; }
    omega_gpu_session_unlock();
    pthread_join(o, NULL);
    if (!atomic_load(&g_other_in)) { printf("FAIL second thread never got the lock after the outermost unlock\n"); return 1; }

    printf("PASS gpu_session_lock_reentrant\n");
    return 0;
}
