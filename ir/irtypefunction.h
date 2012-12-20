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
class IrTypeFunction : public IrType
{
public:
    ///
    static IrTypeFunction* get(Type* dt, Type* nestedContextOverride = 0);

    ///
    IrTypeFunction* isFunction()  { return this; }

    IrFuncTy* fty() { return irfty; }

protected:
    ///
    IrTypeFunction(Type* dt, llvm::Type* lt);

    ///
    IrFuncTy* irfty;
};

///
class IrTypeDelegate : public IrType
{
public:
    ///
    static IrTypeDelegate* get(Type* dt);

    ///
    IrTypeDelegate* isDelegate()    { return this; }

protected:
    ///
    IrTypeDelegate(Type* dt, llvm::Type* lt);
};

#endif
