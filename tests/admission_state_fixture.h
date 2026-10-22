/* SPDX-License-Identifier: MIT */
/* Deliberately forge private v1 persistence in an isolated FIXTURE directory to
 * model PID reuse/previous-boot records. This is not a public disk-format API.
 * Admission decisions still execute the actual product implementation. */
typedef struct {
    uint64_t generation, bytes, start;
    int64_t pid;
    uint32_t state, usage;
    char boot[40];
} FixtureRecord;
typedef struct {
    uint64_t magic, serial, budget, keep, checksum;
    FixtureRecord records[OMEGA_ADMISSION_SLOTS];
} FixtureDatabase;
static void forge_owner(OmegaGpuReservation token, int previous_boot) {
    char path[300]; snprintf(path, sizeof path, "%s/state", root);
    int fd = open(path, O_RDWR); CHECK(fd >= 0);
    FixtureDatabase *db = malloc(sizeof *db); CHECK(db != NULL);
    CHECK(pread(fd, db, sizeof *db, 0) == (ssize_t)sizeof *db);
    CHECK(db->magic == 0x43323441444d0001ull);
    CHECK(db->records[token.slot].generation == token.generation);
    if (previous_boot) db->records[token.slot].boot[0] = db->records[token.slot].boot[0] == '0' ? '1' : '0';
    else db->records[token.slot].start++; /* same live PID, different birth */
    db->checksum = 0;
    /* Independent byte-wise FNV-1a checksum, only for fixture integrity. */
    uint64_t hash = 0xcbf29ce484222325ull;
    unsigned char *p = (unsigned char *)db;
    for (size_t i = 0; i < sizeof *db; i++) hash = (hash ^ p[i]) * 0x100000001b3ull;
    db->checksum = hash;
    CHECK(pwrite(fd, db, sizeof *db, 0) == (ssize_t)sizeof *db);
    CHECK(!fdatasync(fd)); close(fd); free(db);
}
