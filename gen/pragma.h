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

#pragma once

#include <string>

class PragmaDeclaration;
class FuncDeclaration;
class Dsymbol;
struct Scope;
class Expression;

// Remember to keep this enum in-sync with dpragma.d
enum LDCPragma {
  LLVMnone = 0,   // Not an LDC pragma.
  LLVMignore, // Pragma has already been processed in DtoGetPragma, ignore.
  LLVMintrinsic,
  LLVMglobal_crt_ctor,
  LLVMglobal_crt_dtor,
  LLVMno_typeinfo,
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
  LLVMbitop_bts,
  LLVMbitop_vld,
  LLVMbitop_vst,
  LLVMextern_weak,
  LLVMprofile_instr
};

LDCPragma DtoGetPragma(Scope *sc, PragmaDeclaration *decl, const char *&arg1str);
void DtoCheckPragma(PragmaDeclaration *decl, Dsymbol *sym, LDCPragma llvm_internal,
                    const char * const arg1str);
bool DtoCheckProfileInstrPragma(Expression *arg, bool &value);
bool DtoIsIntrinsic(FuncDeclaration *fd);
bool DtoIsMagicIntrinsic(FuncDeclaration *fd);
