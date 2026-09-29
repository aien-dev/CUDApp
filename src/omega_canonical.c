#include "omega_canonical.h"
#include "sha256.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

static void write_u16_be(uint8_t *p, uint16_t v) {
    p[0] = (uint8_t)((v >> 8) & 0xff);
    p[1] = (uint8_t)(v & 0xff);
}

static void write_u32_be(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)((v >> 24) & 0xff);
    p[1] = (uint8_t)((v >> 16) & 0xff);
    p[2] = (uint8_t)((v >> 8) & 0xff);
    p[3] = (uint8_t)(v & 0xff);
}

int omega_compare_semantic_id(const SemanticId *a, const SemanticId *b) {
    return memcmp(a->bytes, b->bytes, OMEGA_ID_BYTES);
}

void omega_hex_semantic_id(const SemanticId *id, char out_hex[65]) {
    static const char hex[] = "0123456789abcdef";
    for (int i = 0; i < OMEGA_ID_BYTES; ++i) {
        out_hex[i * 2]     = hex[(id->bytes[i] >> 4) & 0x0f];
        out_hex[i * 2 + 1] = hex[id->bytes[i] & 0x0f];
    }
    out_hex[64] = '\0';
}

int omega_parse_hex_semantic_id(const char *hex, SemanticId *out_id) {
    if (!hex || strlen(hex) != 64) return -1;
    for (int i = 0; i < 32; ++i) {
        char buf[3] = { hex[i * 2], hex[i * 2 + 1], 0 };
        char *end = NULL;
        long val = strtol(buf, &end, 16);
        if (end != buf + 2) return -1;
        out_id->bytes[i] = (uint8_t)val;
    }
    return 0;
}

uint8_t omega_canonical_object_version(uint8_t kind) {
    return kind == KIND_EFFECT ? OMEGA_EFFECT_VERSION : OMEGA_VERSION;
}

int omega_canonical_check_header(uint8_t version, uint8_t kind) {
    if (kind == KIND_EFFECT && version == OMEGA_VERSION) return OMEGA_CANON_ERR_EFFECT_V1;
    if (version != omega_canonical_object_version(kind)) return OMEGA_CANON_ERR_VERSION;
    return OMEGA_CANON_OK;
}

int omega_canonical_encode(const OmegaObject *obj, uint8_t *out_buf, size_t max_len, size_t *out_len) {
    if (!obj || !out_buf || !out_len) return -1;

    size_t pos = 0;

    /* 1. Header: magic "OMG0" (0x4F, 0x4D, 0x47, 0x30), object encoding
     *    version of the kind (0x02 effect, 0x01 otherwise), kind */
    if (pos + 6 > max_len) return -1;
    out_buf[pos++] = 0x4F;
    out_buf[pos++] = 0x4D;
    out_buf[pos++] = 0x47;
    out_buf[pos++] = 0x30;
    out_buf[pos++] = omega_canonical_object_version((uint8_t)obj->kind);
    out_buf[pos++] = (uint8_t)obj->kind;

    /* 2. Attributes (sorted by key ascending) */
    uint16_t attr_indices[OMEGA_MAX_ATTRIBUTES];
    uint16_t n_attrs = obj->attr_count;
    if (n_attrs > OMEGA_MAX_ATTRIBUTES) return -1;
    for (uint16_t i = 0; i < n_attrs; ++i) attr_indices[i] = i;

    /* Insertion sort for stable canonical attribute order */
    for (uint16_t i = 1; i < n_attrs; ++i) {
        uint16_t key_idx = attr_indices[i];
        int j = i - 1;
        while (j >= 0 && strcmp(obj->attributes[attr_indices[j]].key, obj->attributes[key_idx].key) > 0) {
            attr_indices[j + 1] = attr_indices[j];
            j--;
        }
        attr_indices[j + 1] = key_idx;
    }

    if (pos + 2 > max_len) return -1;
    write_u16_be(&out_buf[pos], n_attrs);
    pos += 2;

    for (uint16_t i = 0; i < n_attrs; ++i) {
        const OmegaAttribute *attr = &obj->attributes[attr_indices[i]];
        size_t klen = strlen(attr->key);
        if (klen > 255) return -1;
        if (pos + 1 + klen + 2 + attr->val_len > max_len) return -1;
        out_buf[pos++] = (uint8_t)klen;
        memcpy(&out_buf[pos], attr->key, klen);
        pos += klen;
        write_u16_be(&out_buf[pos], attr->val_len);
        pos += 2;
        if (attr->val_len > 0) {
            memcpy(&out_buf[pos], attr->value, attr->val_len);
            pos += attr->val_len;
        }
    }

    /* 3. Relations (sorted primarily by kind, secondarily by target_id) */
    uint16_t rel_indices[OMEGA_MAX_RELATIONS];
    uint16_t n_rels = obj->rel_count;
    if (n_rels > OMEGA_MAX_RELATIONS) return -1;
    for (uint16_t i = 0; i < n_rels; ++i) rel_indices[i] = i;

    for (uint16_t i = 1; i < n_rels; ++i) {
        uint16_t cur = rel_indices[i];
        int j = i - 1;
        while (j >= 0) {
            const OmegaRelation *ra = &obj->relations[rel_indices[j]];
            const OmegaRelation *rb = &obj->relations[cur];
            int cmp = (int)ra->kind - (int)rb->kind;
            if (cmp == 0) {
                cmp = memcmp(ra->target_id.bytes, rb->target_id.bytes, OMEGA_ID_BYTES);
            }
            if (cmp <= 0) break;
            rel_indices[j + 1] = rel_indices[j];
            j--;
        }
        rel_indices[j + 1] = cur;
    }

    if (pos + 2 > max_len) return -1;
    write_u16_be(&out_buf[pos], n_rels);
    pos += 2;

    for (uint16_t i = 0; i < n_rels; ++i) {
        const OmegaRelation *rel = &obj->relations[rel_indices[i]];
        if (pos + 2 + OMEGA_ID_BYTES > max_len) return -1;
        write_u16_be(&out_buf[pos], rel->kind);
        pos += 2;
        memcpy(&out_buf[pos], rel->target_id.bytes, OMEGA_ID_BYTES);
        pos += OMEGA_ID_BYTES;
    }

    /* 4. Constraints (sorted primarily by kind, secondarily by payload) */
    uint16_t const_indices[OMEGA_MAX_CONSTRAINTS];
    uint16_t n_consts = obj->const_count;
    if (n_consts > OMEGA_MAX_CONSTRAINTS) return -1;
    for (uint16_t i = 0; i < n_consts; ++i) const_indices[i] = i;

    for (uint16_t i = 1; i < n_consts; ++i) {
        uint16_t cur = const_indices[i];
        int j = i - 1;
        while (j >= 0) {
            const OmegaConstraint *ca = &obj->constraints[const_indices[j]];
            const OmegaConstraint *cb = &obj->constraints[cur];
            int cmp = (int)ca->kind - (int)cb->kind;
            if (cmp == 0) {
                uint16_t min_len = ca->payload_len < cb->payload_len ? ca->payload_len : cb->payload_len;
                cmp = memcmp(ca->payload, cb->payload, min_len);
                if (cmp == 0) {
                    cmp = (int)ca->payload_len - (int)cb->payload_len;
                }
            }
            if (cmp <= 0) break;
            const_indices[j + 1] = const_indices[j];
            j--;
        }
        const_indices[j + 1] = cur;
    }

    if (pos + 2 > max_len) return -1;
    write_u16_be(&out_buf[pos], n_consts);
    pos += 2;

    for (uint16_t i = 0; i < n_consts; ++i) {
        const OmegaConstraint *c = &obj->constraints[const_indices[i]];
        if (pos + 2 + 2 + c->payload_len > max_len) return -1;
        write_u16_be(&out_buf[pos], c->kind);
        pos += 2;
        write_u16_be(&out_buf[pos], c->payload_len);
        pos += 2;
        if (c->payload_len > 0) {
            memcpy(&out_buf[pos], c->payload, c->payload_len);
            pos += c->payload_len;
        }
    }

    /* 5. Payload */
    if (pos + 4 + obj->payload_len > max_len) return -1;
    write_u32_be(&out_buf[pos], obj->payload_len);
    pos += 4;
    if (obj->payload_len > 0) {
        memcpy(&out_buf[pos], obj->payload, obj->payload_len);
        pos += obj->payload_len;
    }

    *out_len = pos;
    return 0;
}

int omega_compute_semantic_id(OmegaObject *obj) {
    if (!obj) return -1;
    uint8_t canon_buf[4096];
    size_t canon_len = 0;
    if (omega_canonical_encode(obj, canon_buf, sizeof(canon_buf), &canon_len) != 0) {
        return -1;
    }
    sha256_hash(canon_buf, canon_len, obj->id.bytes);
    obj->has_id = true;
    return 0;
}
