//===-- gen/runtimecompile.h - jit support ----------------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Jit routines.
//
//===----------------------------------------------------------------------===//

#ifndef LDC_GEN_RUNTIMECOMPILE_H
#define LDC_GEN_RUNTIMECOMPILE_H

struct IRState;
struct IrFunction;
struct IrGlobal;

void generateBitcodeForRuntimeCompile(IRState *irs);
void declareRuntimeCompiledFunction(IRState *irs, IrFunction *func);
void defineRuntimeCompiledFunction(IRState *irs, IrFunction *func);
void addRuntimeCompiledVar(IRState *irs, IrGlobal *var);

#endif // LDC_GEN_RUNTIMECOMPILE_H
