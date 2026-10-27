/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Drake Stapleton (AIEN / aien-dev)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef OMEGA_NUMERIC_LIFECYCLE_H
#define OMEGA_NUMERIC_LIFECYCLE_H

#include <stdatomic.h>

/* One owner across the SIMT, reduction and DIV/SQRT launchers. Defined in
 * omega_numeric.c, which all three already link. No reset API: an uncertain
 * completion requires ending the process and investigating the device. This
 * does not assert that process exit itself resets or recovers the GPU. */
enum { OMEGA_NUMERIC_NATIVE_IDLE, OMEGA_NUMERIC_NATIVE_BUSY,
       OMEGA_NUMERIC_NATIVE_UNCERTAIN };
extern atomic_int omega_numeric_native_state;

static inline int omega_numeric_native_acquire(void) {
    int expected = OMEGA_NUMERIC_NATIVE_IDLE;
    return atomic_compare_exchange_strong(&omega_numeric_native_state, &expected,
                                          OMEGA_NUMERIC_NATIVE_BUSY) ? 0 : -1;
}
static inline int omega_numeric_native_uncertain(void) {
    return atomic_load(&omega_numeric_native_state) == OMEGA_NUMERIC_NATIVE_UNCERTAIN;
}
static inline void omega_numeric_native_poison(void) {
    atomic_store(&omega_numeric_native_state, OMEGA_NUMERIC_NATIVE_UNCERTAIN);
}
static inline void omega_numeric_native_release(void) {
    int expected = OMEGA_NUMERIC_NATIVE_BUSY;
    /* Never clear a poison latch, even if a cleanup caller arrives late. */
    (void)atomic_compare_exchange_strong(&omega_numeric_native_state, &expected,
                                         OMEGA_NUMERIC_NATIVE_IDLE);
}
#endif
