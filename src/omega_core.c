#include "omega_core.h"
#include "omega_canonical.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

OmegaGraph* omega_graph_create(void) {
    OmegaGraph *g = (OmegaGraph*)calloc(1, sizeof(OmegaGraph));
    return g;
}

void omega_graph_destroy(OmegaGraph *g) {
    if (g) free(g);
}

OmegaObject* omega_graph_add_object(OmegaGraph *g, SemanticKind kind) {
    if (!g || g->object_count >= OMEGA_MAX_GRAPH_OBJECTS) return NULL;
    OmegaObject *obj = &g->objects[g->object_count++];
    memset(obj, 0, sizeof(OmegaObject));
    obj->kind = kind;
    return obj;
}

OmegaObject* omega_graph_find_object(OmegaGraph *g, const SemanticId *id) {
    if (!g || !id) return NULL;
    for (uint16_t i = 0; i < g->object_count; ++i) {
        if (g->objects[i].has_id && omega_compare_semantic_id(&g->objects[i].id, id) == 0) {
            return &g->objects[i];
        }
    }
    return NULL;
}

const OmegaObject* omega_graph_find_object_const(const OmegaGraph *g, const SemanticId *id) {
    if (!g || !id) return NULL;
    for (uint16_t i = 0; i < g->object_count; ++i) {
        if (g->objects[i].has_id && omega_compare_semantic_id(&g->objects[i].id, id) == 0) {
            return &g->objects[i];
        }
    }
    return NULL;
}

int omega_object_add_attribute(OmegaObject *obj, const char *key, const uint8_t *val, uint16_t val_len) {
    if (!obj || !key || obj->attr_count >= OMEGA_MAX_ATTRIBUTES) return -1;
    OmegaAttribute *attr = &obj->attributes[obj->attr_count++];
    strncpy(attr->key, key, OMEGA_MAX_KEY_LEN - 1);
    attr->key[OMEGA_MAX_KEY_LEN - 1] = '\0';
    if (val && val_len > 0) {
        uint16_t copy_len = val_len < OMEGA_MAX_VAL_LEN ? val_len : OMEGA_MAX_VAL_LEN;
        memcpy(attr->value, val, copy_len);
        attr->val_len = copy_len;
    } else {
        attr->val_len = 0;
    }
    return 0;
}

int omega_object_add_relation(OmegaObject *obj, RelationKind kind, const SemanticId *target) {
    if (!obj || !target || obj->rel_count >= OMEGA_MAX_RELATIONS) return -1;
    OmegaRelation *rel = &obj->relations[obj->rel_count++];
    rel->kind = (uint16_t)kind;
    memcpy(rel->target_id.bytes, target->bytes, OMEGA_ID_BYTES);
    return 0;
}

int omega_object_add_constraint(OmegaObject *obj, ConstraintKind kind, const uint8_t *payload, uint16_t payload_len) {
    if (!obj || obj->const_count >= OMEGA_MAX_CONSTRAINTS) return -1;
    OmegaConstraint *c = &obj->constraints[obj->const_count++];
    c->kind = (uint16_t)kind;
    c->payload_len = payload_len < 128 ? payload_len : 128;
    if (payload && c->payload_len > 0) {
        memcpy(c->payload, payload, c->payload_len);
    }
    return 0;
}

OmegaObject* omega_build_type_unit(OmegaGraph *g) {
    OmegaObject *obj = omega_graph_add_object(g, KIND_TYPE);
    if (!obj) return NULL;
    TypePayload tp;
    memset(&tp, 0, sizeof(tp));
    tp.tag = TYPE_UNIT;
    memcpy(obj->payload, &tp, sizeof(tp));
    obj->payload_len = sizeof(tp);
    omega_compute_semantic_id(obj);
    return obj;
}

OmegaObject* omega_build_type_bool(OmegaGraph *g) {
    OmegaObject *obj = omega_graph_add_object(g, KIND_TYPE);
    if (!obj) return NULL;
    TypePayload tp;
    memset(&tp, 0, sizeof(tp));
    tp.tag = TYPE_BOOL;
    memcpy(obj->payload, &tp, sizeof(tp));
    obj->payload_len = sizeof(tp);
    omega_compute_semantic_id(obj);
    return obj;
}

OmegaObject* omega_build_type_uint(OmegaGraph *g, uint16_t width) {
    OmegaObject *obj = omega_graph_add_object(g, KIND_TYPE);
    if (!obj) return NULL;
    TypePayload tp;
    memset(&tp, 0, sizeof(tp));
    tp.tag = TYPE_UNSIGNED_INT;
    tp.width = width;
    memcpy(obj->payload, &tp, sizeof(tp));
    obj->payload_len = sizeof(tp);
    omega_compute_semantic_id(obj);
    return obj;
}

OmegaObject* omega_build_type_signed_int(OmegaGraph *g, uint16_t width) {
    OmegaObject *obj = omega_graph_add_object(g, KIND_TYPE);
    if (!obj) return NULL;
    TypePayload tp;
    memset(&tp, 0, sizeof(tp));
    tp.tag = TYPE_SIGNED_INT;
    tp.width = width;
    memcpy(obj->payload, &tp, sizeof(tp));
    obj->payload_len = sizeof(tp);
    omega_compute_semantic_id(obj);
    return obj;
}

OmegaObject* omega_build_type_bitvector(OmegaGraph *g, uint16_t width) {
    OmegaObject *obj = omega_graph_add_object(g, KIND_TYPE);
    if (!obj) return NULL;
    TypePayload tp;
    memset(&tp, 0, sizeof(tp));
    tp.tag = TYPE_BITVECTOR;
    tp.width = width;
    memcpy(obj->payload, &tp, sizeof(tp));
    obj->payload_len = sizeof(tp);
    omega_compute_semantic_id(obj);
    return obj;
}

OmegaObject* omega_build_type_byte(OmegaGraph *g) {
    OmegaObject *obj = omega_graph_add_object(g, KIND_TYPE);
    if (!obj) return NULL;
    TypePayload tp;
    memset(&tp, 0, sizeof(tp));
    tp.tag = TYPE_BYTE;
    tp.width = 8;
    memcpy(obj->payload, &tp, sizeof(tp));
    obj->payload_len = sizeof(tp);
    omega_compute_semantic_id(obj);
    return obj;
}

OmegaObject* omega_build_type_sequence(OmegaGraph *g, const SemanticId *elem_type, uint32_t len) {
    OmegaObject *obj = omega_graph_add_object(g, KIND_TYPE);
    if (!obj) return NULL;
    TypePayload tp;
    memset(&tp, 0, sizeof(tp));
    tp.tag = TYPE_SEQUENCE;
    tp.length = len;
    if (elem_type) memcpy(tp.elem_type.bytes, elem_type->bytes, OMEGA_ID_BYTES);
    memcpy(obj->payload, &tp, sizeof(tp));
    obj->payload_len = sizeof(tp);
    omega_compute_semantic_id(obj);
    return obj;
}

OmegaObject* omega_build_type_cap_ref(OmegaGraph *g) {
    OmegaObject *obj = omega_graph_add_object(g, KIND_TYPE);
    if (!obj) return NULL;
    TypePayload tp;
    memset(&tp, 0, sizeof(tp));
    tp.tag = TYPE_CAPABILITY_REF;
    memcpy(obj->payload, &tp, sizeof(tp));
    obj->payload_len = sizeof(tp);
    omega_compute_semantic_id(obj);
    return obj;
}

OmegaObject* omega_build_val_bool(OmegaGraph *g, const SemanticId *bool_type_id, bool val) {
    OmegaObject *obj = omega_graph_add_object(g, KIND_VALUE);
    if (!obj) return NULL;
    ValuePayload vp;
    memset(&vp, 0, sizeof(vp));
    if (bool_type_id) memcpy(vp.type_id.bytes, bool_type_id->bytes, OMEGA_ID_BYTES);
    vp.byte_len = 1;
    vp.bytes[0] = val ? 1 : 0;
    memcpy(obj->payload, &vp, sizeof(vp));
    obj->payload_len = sizeof(vp);
    omega_compute_semantic_id(obj);
    return obj;
}

OmegaObject* omega_build_val_uint(OmegaGraph *g, const SemanticId *uint_type_id, uint16_t width, uint64_t val) {
    OmegaObject *obj = omega_graph_add_object(g, KIND_VALUE);
    if (!obj) return NULL;
    ValuePayload vp;
    memset(&vp, 0, sizeof(vp));
    if (uint_type_id) memcpy(vp.type_id.bytes, uint_type_id->bytes, OMEGA_ID_BYTES);
    uint16_t nbytes = (width + 7) / 8;
    if (nbytes == 0) nbytes = 1;
    if (nbytes > 64) nbytes = 64;
    vp.byte_len = nbytes;
    for (int i = 0; i < nbytes; ++i) {
        int shift = (nbytes - 1 - i) * 8;
        if (shift < 64) {
            vp.bytes[i] = (uint8_t)((val >> shift) & 0xff);
        } else {
            vp.bytes[i] = 0;
        }
    }
    memcpy(obj->payload, &vp, sizeof(vp));
    obj->payload_len = sizeof(vp);
    omega_compute_semantic_id(obj);
    return obj;
}

OmegaObject* omega_build_param(OmegaGraph *g, const SemanticId *type_id, uint16_t index) {
    OmegaObject *obj = omega_graph_add_object(g, KIND_VALUE);
    if (!obj) return NULL;
    uint8_t idx[2] = { (uint8_t)(index >> 8), (uint8_t)index };
    if (omega_object_add_attribute(obj, "omega.param", idx, sizeof idx) != 0) return NULL;
    ValuePayload vp;
    memset(&vp, 0, sizeof(vp));
    if (type_id) memcpy(vp.type_id.bytes, type_id->bytes, OMEGA_ID_BYTES);
    vp.byte_len = 0;
    memcpy(obj->payload, &vp, sizeof(vp));
    obj->payload_len = sizeof(vp);
    omega_compute_semantic_id(obj);
    return obj;
}

OmegaObject* omega_build_op_binary(OmegaGraph *g, OpCode op, OverflowPolicy ov, const SemanticId *type_id) {
    OmegaObject *obj = omega_graph_add_object(g, KIND_OPERATION);
    if (!obj) return NULL;
    OperationPayload opp;
    memset(&opp, 0, sizeof(opp));
    opp.opcode = op;
    opp.overflow = ov;
    if (type_id) {
        memcpy(opp.type_id.bytes, type_id->bytes, OMEGA_ID_BYTES);
        opp.arity = 2;
        memcpy(opp.input_types[0].bytes, type_id->bytes, OMEGA_ID_BYTES);
        memcpy(opp.input_types[1].bytes, type_id->bytes, OMEGA_ID_BYTES);
        memcpy(opp.output_type.bytes, type_id->bytes, OMEGA_ID_BYTES);
    }
    memcpy(obj->payload, &opp, sizeof(opp));
    obj->payload_len = sizeof(opp);
    omega_compute_semantic_id(obj);
    return obj;
}

OmegaObject* omega_build_apply(OmegaGraph *g, const SemanticId *op_id, const SemanticId *arg1, const SemanticId *arg2) {
    OmegaObject *obj = omega_graph_add_object(g, KIND_OPERATION);
    if (!obj) return NULL;
    ApplyPayload app;
    memset(&app, 0, sizeof(app));
    if (op_id) memcpy(app.op_id.bytes, op_id->bytes, OMEGA_ID_BYTES);
    app.operand_count = 2;
    if (arg1) memcpy(app.operands[0].bytes, arg1->bytes, OMEGA_ID_BYTES);
    if (arg2) memcpy(app.operands[1].bytes, arg2->bytes, OMEGA_ID_BYTES);
    memcpy(obj->payload, &app, sizeof(app));
    obj->payload_len = sizeof(app);
    omega_compute_semantic_id(obj);
    return obj;
}

/* ---- effect payload v2 (spec/effect-cap64-migration.md) ---- */

static void eff_put16(uint8_t *p, uint16_t v) { p[0] = (uint8_t)(v >> 8); p[1] = (uint8_t)v; }
static void eff_put32(uint8_t *p, uint32_t v) {
    for (int i = 0; i < 4; i++) p[i] = (uint8_t)(v >> (24 - 8 * i));
}
static void eff_put64(uint8_t *p, uint64_t v) {
    for (int i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (56 - 8 * i));
}
static uint16_t eff_get16(const uint8_t *p) { return (uint16_t)(((uint16_t)p[0] << 8) | p[1]); }
static uint32_t eff_get32(const uint8_t *p) {
    uint32_t v = 0;
    for (int i = 0; i < 4; i++) v = (v << 8) | p[i];
    return v;
}
static uint64_t eff_get64(const uint8_t *p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v = (v << 8) | p[i];
    return v;
}

enum {
    EFF_OFF_RESOURCE = 0x00, EFF_OFF_OPERATION = 0x02, EFF_OFF_SLOT = 0x04,
    EFF_OFF_GENERATION = 0x08, EFF_OFF_CAP_REF = 0x10, EFF_OFF_PARAM_LEN = 0x30,
    EFF_OFF_PARAMS = 0x32
};
_Static_assert(EFF_OFF_PARAMS + OMEGA_EFFECT_PARAM_MAX == OMEGA_EFFECT_PAYLOAD_LEN,
               "effect payload v2 is 178 bytes");
_Static_assert(OMEGA_EFFECT_PAYLOAD_LEN != OMEGA_EFFECT_LEGACY_V1_LEN, "v1 and v2 lengths differ");

int omega_effect_payload_encode(const EffectPayload *eff, uint8_t out[OMEGA_EFFECT_PAYLOAD_LEN]) {
    if (!eff || !out) return OMEGA_EFFECT_ERR_ARG;
    eff_put16(out + EFF_OFF_RESOURCE, eff->resource_class);
    eff_put16(out + EFF_OFF_OPERATION, eff->operation_code);
    eff_put32(out + EFF_OFF_SLOT, eff->capability_slot);
    eff_put64(out + EFF_OFF_GENERATION, eff->capability_generation);
    memcpy(out + EFF_OFF_CAP_REF, eff->capability_ref.bytes, OMEGA_ID_BYTES);
    eff_put16(out + EFF_OFF_PARAM_LEN, eff->param_len);
    memcpy(out + EFF_OFF_PARAMS, eff->param_bytes, OMEGA_EFFECT_PARAM_MAX);
    return OMEGA_EFFECT_OK;
}

int omega_effect_payload_decode(const uint8_t *bytes, size_t len, EffectPayload *out) {
    if (out) memset(out, 0, sizeof *out);
    if (!bytes || !out) return OMEGA_EFFECT_ERR_ARG;
    if (len == OMEGA_EFFECT_LEGACY_V1_LEN) return OMEGA_EFFECT_ERR_LEGACY_V1;
    if (len != OMEGA_EFFECT_PAYLOAD_LEN) return OMEGA_EFFECT_ERR_LENGTH;
    EffectPayload e;
    memset(&e, 0, sizeof e);
    e.resource_class = eff_get16(bytes + EFF_OFF_RESOURCE);
    e.operation_code = eff_get16(bytes + EFF_OFF_OPERATION);
    e.capability_slot = eff_get32(bytes + EFF_OFF_SLOT);
    e.capability_generation = eff_get64(bytes + EFF_OFF_GENERATION);
    memcpy(e.capability_ref.bytes, bytes + EFF_OFF_CAP_REF, OMEGA_ID_BYTES);
    e.param_len = eff_get16(bytes + EFF_OFF_PARAM_LEN);
    if (e.resource_class == 0) return OMEGA_EFFECT_ERR_RESOURCE;
    if (e.param_len > OMEGA_EFFECT_PARAM_MAX) return OMEGA_EFFECT_ERR_PARAM_LEN;
    for (size_t i = e.param_len; i < OMEGA_EFFECT_PARAM_MAX; i++)
        if (bytes[EFF_OFF_PARAMS + i] != 0) return OMEGA_EFFECT_ERR_PADDING;
    memcpy(e.param_bytes, bytes + EFF_OFF_PARAMS, e.param_len);
    *out = e;
    return OMEGA_EFFECT_OK;
}

int omega_effect_write(OmegaObject *obj, const EffectPayload *eff) {
    if (!obj || !eff || obj->kind != KIND_EFFECT) return OMEGA_EFFECT_ERR_ARG;
    int rc = omega_effect_payload_encode(eff, obj->payload);
    if (rc != OMEGA_EFFECT_OK) return rc;
    obj->payload_len = OMEGA_EFFECT_PAYLOAD_LEN;
    return OMEGA_EFFECT_OK;
}

int omega_effect_read(const OmegaObject *obj, EffectPayload *out) {
    if (out) memset(out, 0, sizeof *out);
    if (!obj || !out || obj->kind != KIND_EFFECT) return OMEGA_EFFECT_ERR_ARG;
    if (obj->payload_len > OMEGA_MAX_PAYLOAD_LEN) return OMEGA_EFFECT_ERR_LENGTH;
    return omega_effect_payload_decode(obj->payload, obj->payload_len, out);
}

const char *omega_effect_strerror(int code) {
    switch (code) {
    case OMEGA_EFFECT_OK: return "ok";
    case OMEGA_EFFECT_ERR_ARG: return "not an effect object";
    case OMEGA_EFFECT_ERR_LENGTH: return "effect payload length is not 178 bytes";
    case OMEGA_EFFECT_ERR_LEGACY_V1: return "legacy v1 effect payload (32-bit generation) refused";
    case OMEGA_EFFECT_ERR_RESOURCE: return "effect resource class is 0 (invalid resource)";
    case OMEGA_EFFECT_ERR_PARAM_LEN: return "effect param_len exceeds 128";
    case OMEGA_EFFECT_ERR_PADDING: return "effect parameter padding is not zero";
    default: return "unknown effect error";
    }
}

OmegaObject* omega_build_effect(OmegaGraph *g, uint16_t res_class, uint16_t op_code, uint32_t cap_slot, uint64_t cap_gen) {
    OmegaObject *obj = omega_graph_add_object(g, KIND_EFFECT);
    if (!obj) return NULL;
    EffectPayload eff;
    memset(&eff, 0, sizeof(eff));
    eff.resource_class = res_class;
    eff.operation_code = op_code;
    eff.capability_slot = cap_slot;
    eff.capability_generation = cap_gen;
    omega_effect_write(obj, &eff);
    omega_compute_semantic_id(obj);
    return obj;
}

int omega_eval_pure_binary_uint(OpCode op, OverflowPolicy ov, uint16_t width, uint64_t a, uint64_t b, uint64_t *out_res) {
    if (!out_res) return -1;
    uint64_t mask = (width >= 64) ? 0xFFFFFFFFFFFFFFFFULL : ((1ULL << width) - 1ULL);
    a &= mask;
    b &= mask;

    uint64_t res = 0;
    bool overflowed = false;

    switch (op) {
        case OP_ADD: {
            res = a + b;
            if (res < a || (res & ~mask) != 0) {
                overflowed = true;
            }
            break;
        }
        case OP_SUB: {
            if (a < b) {
                overflowed = true;
                res = (a + (mask + 1)) - b;
            } else {
                res = a - b;
            }
            break;
        }
        case OP_MUL: {
            __uint128_t full = (__uint128_t)a * (__uint128_t)b;
            if ((full >> width) != 0) {
                overflowed = true;
            }
            res = (uint64_t)(full & mask);
            break;
        }
        case OP_DIV: {
            if (b == 0) return -1; /* Division by zero failure */
            res = a / b;
            break;
        }
        case OP_EQUAL: {
            *out_res = (a == b) ? 1 : 0;
            return 0;
        }
        case OP_LESS_THAN: {
            *out_res = (a < b) ? 1 : 0;
            return 0;
        }
        case OP_AND: {
            res = a & b;
            break;
        }
        case OP_OR: {
            res = a | b;
            break;
        }
        default:
            return -1;
    }

    if (overflowed) {
        if (ov == OVERFLOW_FAIL_CLOSED) return -2;
        if (ov == OVERFLOW_SATURATE) res = mask;
        /* OVERFLOW_WRAP naturally applies mask */
    }

    *out_res = res & mask;
    return 0;
}
