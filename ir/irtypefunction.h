#ifndef __LDC_IR_IRTYPEFUNCTION_H__
#define __LDC_IR_IRTYPEFUNCTION_H__

#include "ir/irtype.h"

struct IrFuncTy;

///
class IrTypeFunction : public IrType
{
public:
    ///
    IrTypeFunction(Type* dt);

    ///
    IrTypeFunction* isFunction()  { return this; }

    ///
    llvm::Type* buildType();

    IrFuncTy* fty() { return irfty; }

protected:
    llvm::Type* func2llvm(Type* dt);
    ///
    IrFuncTy* irfty;
};

///
class IrTypeDelegate : public IrType
{
public:
    ///
    IrTypeDelegate(Type* dt);

    ///
    IrTypeDelegate* isDelegate()    { return this; }

    ///
    llvm::Type* buildType();
protected:
    llvm::Type* delegate2llvm(Type* dt);
};

#endif
