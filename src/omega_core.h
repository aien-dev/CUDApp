#ifndef OMEGA_CORE_H
#define OMEGA_CORE_H

#include "omega_types.h"

OmegaGraph* omega_graph_create(void);
void omega_graph_destroy(OmegaGraph *g);
OmegaObject* omega_graph_add_object(OmegaGraph *g, SemanticKind kind);
OmegaObject* omega_graph_find_object(OmegaGraph *g, const SemanticId *id);
const OmegaObject* omega_graph_find_object_const(const OmegaGraph *g, const SemanticId *id);

/* Attribute / Relation / Constraint attachers */
int omega_object_add_attribute(OmegaObject *obj, const char *key, const uint8_t *val, uint16_t val_len);
int omega_object_add_relation(OmegaObject *obj, RelationKind kind, const SemanticId *target);
int omega_object_add_constraint(OmegaObject *obj, ConstraintKind kind, const uint8_t *payload, uint16_t payload_len);

/* Builders for canonical types */
OmegaObject* omega_build_type_unit(OmegaGraph *g);
OmegaObject* omega_build_type_bool(OmegaGraph *g);
OmegaObject* omega_build_type_uint(OmegaGraph *g, uint16_t width);
OmegaObject* omega_build_type_signed_int(OmegaGraph *g, uint16_t width);
OmegaObject* omega_build_type_bitvector(OmegaGraph *g, uint16_t width);
OmegaObject* omega_build_type_byte(OmegaGraph *g);
OmegaObject* omega_build_type_sequence(OmegaGraph *g, const SemanticId *elem_type, uint32_t len);
OmegaObject* omega_build_type_cap_ref(OmegaGraph *g);

/* Builders for canonical values */
OmegaObject* omega_build_val_bool(OmegaGraph *g, const SemanticId *bool_type_id, bool val);
OmegaObject* omega_build_val_uint(OmegaGraph *g, const SemanticId *uint_type_id, uint16_t width, uint64_t val);

/* Builders for operations and applications */
OmegaObject* omega_build_op_binary(OmegaGraph *g, OpCode op, OverflowPolicy ov, const SemanticId *type_id);
OmegaObject* omega_build_apply(OmegaGraph *g, const SemanticId *op_id, const SemanticId *arg1, const SemanticId *arg2);

/* Builder for effect */
OmegaObject* omega_build_effect(OmegaGraph *g, uint16_t res_class, uint16_t op_code, uint32_t cap_slot, uint32_t cap_gen);

/* Evaluation of pure operations */
int omega_eval_pure_binary_uint(OpCode op, OverflowPolicy ov, uint16_t width, uint64_t a, uint64_t b, uint64_t *out_res);

#endif /* OMEGA_CORE_H */
