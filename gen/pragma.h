#ifndef PRAGMA_H
#define PRAGMA_H

#include <string>

struct PragmaDeclaration;
struct Dsymbol;
struct Scope;

enum Pragma
{
    LLVMnone,
    LLVMintrinsic,
    LLVMno_typeinfo,
    LLVMno_moduleinfo,
    LLVMalloca,
    LLVMva_start,
    LLVMva_copy,
    LLVMva_end,
    LLVMva_arg,
    LLVMinline_asm,
    LLVMinline_ir,
    LLVMfence,
    LLVMatomic_store,
    LLVMatomic_load,
    LLVMatomic_cmp_xchg,
    LLVMatomic_rmw,
    LLVMbitop_bt,
    LLVMbitop_btc,
    LLVMbitop_btr,
    LLVMbitop_bts
};

Pragma DtoGetPragma(Scope *sc, PragmaDeclaration *decl, std::string &arg1str);
void DtoCheckPragma(PragmaDeclaration *decl, Dsymbol *sym,
                    Pragma llvm_internal, const std::string &arg1str);

#endif // PRAGMA_H
