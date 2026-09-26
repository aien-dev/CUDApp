/* =========================================================================
 * BOOTSTRAP REPRESENTATION - NOT PERMANENT SEMANTIC DEFINITION
 * This C header is a temporary bootstrap scaffolding for Milestone 4.
 * The canonical definition of OMEGA semantics is independent of host language ABIs.
 * ========================================================================= */

#ifndef OMEGA_TYPES_H
#define OMEGA_TYPES_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#define OMEGA_MAGIC 0x30474D4F /* "OMG0" in little-endian / network bytes: 0x4F,0x4D,0x47,0x30 */
#define OMEGA_VERSION 0x01
#define OMEGA_ID_BYTES 32
#define OMEGA_MAX_ATTRIBUTES 32
#define OMEGA_MAX_RELATIONS 64
#define OMEGA_MAX_CONSTRAINTS 32
#define OMEGA_MAX_KEY_LEN 64
#define OMEGA_MAX_VAL_LEN 512
#define OMEGA_MAX_PAYLOAD_LEN 1024
#define OMEGA_MAX_GRAPH_OBJECTS 256

typedef struct {
    uint8_t bytes[OMEGA_ID_BYTES];
} SemanticId;

typedef enum {
    KIND_INVALID      = 0x00,
    KIND_VALUE        = 0x01,
    KIND_TYPE         = 0x02,
    KIND_OPERATION    = 0x03,
    KIND_RELATION     = 0x04,
    KIND_CONSTRAINT   = 0x05,
    KIND_MEMORY       = 0x06,
    KIND_MACHINE      = 0x07,
    KIND_EFFECT       = 0x08,
    KIND_REALIZATION  = 0x09,
    KIND_EVIDENCE     = 0x0A,
    KIND_PROOF        = 0x0B
} SemanticKind;

typedef enum {
    TYPE_INVALID            = 0x00,
    TYPE_UNIT               = 0x01,
    TYPE_BOOL               = 0x02,
    TYPE_UNSIGNED_INT       = 0x03,
    TYPE_SIGNED_INT         = 0x04,
    TYPE_BITVECTOR          = 0x05,
    TYPE_BYTE               = 0x06,
    TYPE_SEQUENCE           = 0x07,
    TYPE_TUPLE              = 0x08,
    TYPE_ADDRESS            = 0x09,
    TYPE_RESOURCE           = 0x0A,
    TYPE_CAPABILITY_REF     = 0x0B,
    TYPE_EFFECT_INTENT_REF  = 0x0C,
    TYPE_EFFECT_RECEIPT_REF = 0x0D
} TypeTag;

typedef enum {
    OP_INVALID    = 0x00,
    OP_IDENTITY   = 0x01,
    OP_CONSTANT   = 0x02,
    OP_ADD        = 0x03,
    OP_SUB        = 0x04,
    OP_MUL        = 0x05,
    OP_DIV        = 0x06,
    OP_EQUAL      = 0x07,
    OP_LESS_THAN  = 0x08,
    OP_AND        = 0x09,
    OP_OR         = 0x0A,
    OP_NOT        = 0x0B,
    OP_SELECT     = 0x0C,
    OP_CONCAT     = 0x0D,
    OP_SLICE      = 0x0E,
    OP_COMPILE    = 0x0F
} OpCode;

typedef enum {
    OVERFLOW_DEFAULT     = 0x00,
    OVERFLOW_WRAP        = 0x01,
    OVERFLOW_SATURATE    = 0x02,
    OVERFLOW_FAIL_CLOSED = 0x03
} OverflowPolicy;

typedef enum {
    REL_INVALID       = 0x00,
    REL_EQUAL         = 0x01,
    REL_NOT_EQUAL     = 0x02,
    REL_LESS_THAN     = 0x03,
    REL_CONTAINS      = 0x04,
    REL_SUBSET_OF     = 0x05,
    REL_DEPENDS_ON    = 0x06,
    REL_DERIVED_FROM  = 0x07,
    REL_SATISFIES     = 0x08,
    REL_EQUIVALENT_TO = 0x09
} RelationKind;

typedef enum {
    CONST_INVALID       = 0x00,
    CONST_EQUALITY      = 0x01,
    CONST_INEQUALITY    = 0x02,
    CONST_RANGE         = 0x03,
    CONST_TYPE          = 0x04,
    CONST_LENGTH        = 0x05,
    CONST_CONTAINMENT   = 0x06,
    CONST_PRECONDITION  = 0x07,
    CONST_POSTCONDITION = 0x08,
    CONST_INVARIANT     = 0x09
} ConstraintKind;

typedef struct {
    char key[OMEGA_MAX_KEY_LEN];
    uint8_t value[OMEGA_MAX_VAL_LEN];
    uint16_t val_len;
} OmegaAttribute;

typedef struct {
    uint16_t kind; /* RelationKind */
    SemanticId target_id;
} OmegaRelation;

typedef struct {
    uint16_t kind; /* ConstraintKind */
    uint16_t payload_len;
    uint8_t payload[128];
} OmegaConstraint;

/* Kind-specific payloads */

typedef struct {
    TypeTag tag;
    uint16_t width;        /* for int/bitvector/address */
    uint32_t length;       /* for sequence */
    SemanticId elem_type;  /* for sequence/pointer */
} TypePayload;

typedef struct {
    SemanticId type_id;
    uint16_t byte_len;
    uint8_t bytes[64];
} ValuePayload;

typedef struct {
    OpCode opcode;
    OverflowPolicy overflow;
    SemanticId type_id;
    uint8_t arity;
    SemanticId input_types[4];
    SemanticId output_type;
} OperationPayload;

typedef struct {
    SemanticId op_id;
    uint8_t operand_count;
    SemanticId operands[4];
} ApplyPayload;

typedef struct {
    uint16_t resource_class;
    uint16_t operation_code;
    uint32_t capability_slot;
    uint32_t capability_generation;
    SemanticId capability_ref;
    uint16_t param_len;
    uint8_t param_bytes[128];
} EffectPayload;

typedef struct {
    SemanticId id;
    bool has_id;
    SemanticKind kind;
    
    uint16_t attr_count;
    OmegaAttribute attributes[OMEGA_MAX_ATTRIBUTES];

    uint16_t rel_count;
    OmegaRelation relations[OMEGA_MAX_RELATIONS];

    uint16_t const_count;
    OmegaConstraint constraints[OMEGA_MAX_CONSTRAINTS];

    uint32_t payload_len;
    uint8_t payload[OMEGA_MAX_PAYLOAD_LEN];
} OmegaObject;

typedef struct {
    uint16_t object_count;
    OmegaObject objects[OMEGA_MAX_GRAPH_OBJECTS];
} OmegaGraph;

#endif /* OMEGA_TYPES_H */
