//===-- gen/dynamiccompile.h - jit support ----------------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Dynamic compilation routines.
//
//===----------------------------------------------------------------------===//

#ifndef LDC_GEN_DYNAMICCOMPILE_H
#define LDC_GEN_DYNAMICCOMPILE_H

struct IRState;
struct IrFunction;
struct IrGlobal;

void generateBitcodeForDynamicCompile(IRState *irs);
void declareDynamicCompiledFunction(IRState *irs, IrFunction *func);
void defineDynamicCompiledFunction(IRState *irs, IrFunction *func);
void addDynamicCompiledVar(IRState *irs, IrGlobal *var);

#endif // LDC_GEN_DYNAMICCOMPILE_H
