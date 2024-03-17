//===-- gen/pragma.d - LDC-specific pragma handling ---------------*- D -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// D bindings for pragma.h/.cpp
//
//===----------------------------------------------------------------------===//

module gen.dpragma;

import dmd.attrib;
import dmd.dscope;
import dmd.dsymbol;
import dmd.expression;
import dmd.func;

extern (C++) enum LDCPragma : int {
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
  LLVMextern_weak
};

extern (C++) LDCPragma DtoGetPragma(Scope* sc, PragmaDeclaration decl, ref const(char)* arg1str);
extern (C++) void DtoCheckPragma(PragmaDeclaration decl, Dsymbol sym, LDCPragma llvm_internal, const char* arg1str);
extern (C++) bool DtoCheckProfileInstrPragma(Expression arg, ref bool value);
extern (C++) bool DtoIsIntrinsic(FuncDeclaration fd);
