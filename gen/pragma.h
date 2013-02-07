//===-- gen/pragma.h - LDC-specific pragma handling -------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Code for handling the LDC-specific pragmas.
//
//===----------------------------------------------------------------------===//

#ifndef LDC_GEN_PRAGMA_H
#define LDC_GEN_PRAGMA_H

#include <string>

struct PragmaDeclaration;
struct Dsymbol;
struct Scope;

enum Pragma
{
    LLVMnone, // Not an LDC pragma.
    LLVMignore, // Pragma has already been processed in DtoGetPragma, ignore.
    LLVMintrinsic,
    LLVMglobal_crt_ctor,
    LLVMglobal_crt_dtor,
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

#endif // LDC_GEN_PRAGMA_H
