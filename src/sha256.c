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

/* SHA-256 is implemented in Omega: om/sha256/sha256.om, compiled by the seeded compiler and bound
 * through the omega-bridge native call ABI v1 (generated: src/omega_om_units.inc, regenerate with
 * om/units.sh). This file only keeps the sha256_* entry points: the Omega unit takes at most
 * OMSHA_PIECE bytes per call because ABI v1 arrays have a fixed length. No hashing logic lives here. */

#include "sha256.h"
#include <stdlib.h>
#include <string.h>

#include "omega_om_units.inc"

_Static_assert(sizeof(sha256_ctx) == 104, "om/sha256/sha256.om reads sha256_ctx as 104 bytes");

enum { OMSHA_INIT = 0, OMSHA_ABSORB = 1, OMSHA_FINAL = 2, OMSHA_ONESHOT = 3 };
#define OMSHA_PIECE 4096 /* the length of sha256.om's `b view uchar 4096` */

/* Lent as the unit's unused input views: as bytes (`b view uchar 4096`) and as aligned 8-byte cells
 * (`b8 view ulong 512`, read only by the one-shot path). */
static const uint64_t omsha_zero[OMSHA_PIECE / 8];

/* An Omega trap or a refused argument here is a broken invariant (n <= OMSHA_PIECE, valid op, sizes fixed by
 * the types above), never a recoverable condition, so it aborts like a failed assert. The unit returns a
 * digest packed into the first four scratch words, in byte order, which out receives. */
static void omsha_call(sha256_ctx *ctx, const uint8_t *block, const uint64_t *block8, size_t n, unsigned op,
                       uint8_t *out)
{
    uint64_t unused, sched[64]; /* message-schedule scratch; the Omega unit writes it before reading */
    if (omega_omsha(&unused, ctx, sizeof *ctx, block, OMSHA_PIECE, n, op, sched, 64, block8, OMSHA_PIECE / 8) != 0)
        abort();
    if (out)
        memcpy(out, sched, SHA256_DIGEST_SIZE);
}

void sha256_init(sha256_ctx *ctx)
{
    omsha_call(ctx, (const uint8_t *)omsha_zero, omsha_zero, 0, OMSHA_INIT, NULL);
}

void sha256_update(sha256_ctx *ctx, const uint8_t *data, size_t len)
{
    while (len >= OMSHA_PIECE) {
        omsha_call(ctx, data, omsha_zero, OMSHA_PIECE, OMSHA_ABSORB, NULL);
        data += OMSHA_PIECE;
        len -= OMSHA_PIECE;
    }
    if (len > 0) {
        uint8_t tail[OMSHA_PIECE]; /* only tail[0..len) is read */
        memcpy(tail, data, len);
        omsha_call(ctx, tail, omsha_zero, len, OMSHA_ABSORB, NULL);
    }
}

void sha256_final(sha256_ctx *ctx, uint8_t digest[SHA256_DIGEST_SIZE])
{
    omsha_call(ctx, (const uint8_t *)omsha_zero, omsha_zero, 0, OMSHA_FINAL, digest);
}

void sha256_hash(const uint8_t *data, size_t len, uint8_t digest[SHA256_DIGEST_SIZE])
{
    sha256_ctx ctx;
    if (len < OMSHA_PIECE) { /* one call; the unit reads the copy 8 bytes at a time */
        uint64_t piece[OMSHA_PIECE / 8];
        memset((uint8_t *)piece + (len & ~(size_t)63), 0, 64); /* the block holding the tail */
        if (len > 0)
            memcpy(piece, data, len);
        omsha_call(&ctx, (const uint8_t *)piece, piece, len, OMSHA_ONESHOT, digest);
        return;
    }
    sha256_init(&ctx);
    sha256_update(&ctx, data, len);
    sha256_final(&ctx, digest);
}
