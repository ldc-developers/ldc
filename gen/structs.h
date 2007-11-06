#ifndef LLVMD_GEN_STRUCTS_H
#define LLVMD_GEN_STRUCTS_H

struct StructInitializer;
const llvm::Type* DtoStructType(Type* t);
llvm::Value* DtoStructZeroInit(llvm::Value* v);
llvm::Value* DtoStructCopy(llvm::Value* dst, llvm::Value* src);
llvm::Constant* DtoConstStructInitializer(StructInitializer* si);
llvm::Value* DtoIndexStruct(llvm::Value* ptr, StructDeclaration* sd, Type* t, unsigned os, std::vector<unsigned>& idxs);

struct DUnionField
{
    unsigned offset;
    size_t size;
    std::vector<const llvm::Type*> types;
    llvm::Constant* init;
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
    llvm::Constant* c;

    DUnionIdx()
    : idx(0), c(0) {}
    DUnionIdx(unsigned _idx, unsigned _idxos, llvm::Constant* _c)
    : idx(_idx), idxos(_idxos), c(_c) {}
    bool operator<(const DUnionIdx& i) const {
        return (idx < i.idx) || (idx == i.idx && idxos < i.idxos);
    }
};

class DUnion
{
    std::vector<DUnionField> fields;
public:
    DUnion();
    llvm::Constant* getConst(std::vector<DUnionIdx>& in);
};

#endif
