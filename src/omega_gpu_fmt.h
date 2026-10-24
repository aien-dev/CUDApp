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

/* omega_gpu_fmt.h -- bounded message formatting for the freestanding GPU code (native v1 P07 cut C session
 * part, aienos#286). The session and the tensor APIs format their error stages with it in both builds, so a
 * stage reads the same with or without libc. Conversions: %s %c %d %u %x %ld %lu %lld %llu %zu %%; anything
 * else is copied literally. Always NUL-terminates inside cap (cap > 0); truncates, never overflows. Returns
 * the number of characters written (without the NUL). */
#ifndef OMEGA_GPU_FMT_H
#define OMEGA_GPU_FMT_H
#include <stddef.h>
int omega_gpu_fmt(char *out, size_t cap, const char *fmt, ...) __attribute__((format(printf, 3, 4)));
#endif
