#ifndef LLVMD_GEN_STRUCTS_H
#define LLVMD_GEN_STRUCTS_H

struct StructInitializer;

LLConstant* DtoConstStructInitializer(StructInitializer* si);

/**
 * Resolves the llvm type for a struct
 */
void DtoResolveStruct(StructDeclaration* sd);

/**
 * Provides the llvm declaration for a struct
 */
void DtoDeclareStruct(StructDeclaration* sd);

/**
 * Constructs the constant default initializer a struct
 */
void DtoConstInitStruct(StructDeclaration* sd);

/**
 * Provides the llvm definition for a struct
 */
void DtoDefineStruct(StructDeclaration* sd);

/**
 * Returns a boolean=true if the two structs are equal
 */
LLValue* DtoStructEquals(TOK op, DValue* lhs, DValue* rhs);

typedef LLSmallVector<unsigned, 3> DStructIndexVector;
LLValue* DtoIndexStruct(LLValue* ptr, StructDeclaration* sd, Type* t, unsigned os, DStructIndexVector& idxs);

struct DUnionField
{
    unsigned offset;
    size_t size;
    std::vector<const LLType*> types;
    LLConstant* init;
    size_t initsize;

    DUnionField() {
        offset = 0;
        size = 0;
        init = NULL;
        initsize = 0;
    }
};

struct DUnionIdx
{
    unsigned idx,idxos;
    LLConstant* c;

    DUnionIdx()
    : idx(0), c(0) {}
    DUnionIdx(unsigned _idx, unsigned _idxos, LLConstant* _c)
    : idx(_idx), idxos(_idxos), c(_c) {}
    bool operator<(const DUnionIdx& i) const {
        return (idx < i.idx) || (idx == i.idx && idxos < i.idxos);
    }
};

class DUnion
{
    std::vector<DUnionField> fields;
    bool ispacked;
public:
    DUnion();
    LLConstant* getConst(std::vector<DUnionIdx>& in);
};

#endif
