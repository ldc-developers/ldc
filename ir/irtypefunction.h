//===-- ir/irtypefunction.h - IrType subclasses for callables ---*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Provides the IrType subclasses used to represent D function/delegate types.
//
//===----------------------------------------------------------------------===//

#ifndef __LDC_IR_IRTYPEFUNCTION_H__
#define __LDC_IR_IRTYPEFUNCTION_H__

#include "ir/irtype.h"

struct IrFuncTy;

///
class IrTypeFunction : public IrType {
public:
  ///
  static IrTypeFunction *get(Type *dt);

  ///
  IrTypeFunction *isFunction() override { return this; }

  ///
  IrFuncTy &getIrFuncTy() override { return irFty; }

protected:
  ///
  IrTypeFunction(Type *dt, llvm::Type *lt, IrFuncTy irFty);
  ///
  IrFuncTy irFty;
};

///
class IrTypeDelegate : public IrType {
public:
  ///
  static IrTypeDelegate *get(Type *dt);

  ///
  IrTypeDelegate *isDelegate() override { return this; }

  ///
  IrFuncTy &getIrFuncTy() override { return irFty; }

protected:
  ///
  IrTypeDelegate(Type *dt, LLType *lt, IrFuncTy irFty);
  ///
  IrFuncTy irFty;
};

#endif
