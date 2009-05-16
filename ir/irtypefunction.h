#ifndef __LDC_IR_IRTYPEFUNCTION_H__
#define __LDC_IR_IRTYPEFUNCTION_H__

#include "ir/irtype.h"

class IrFuncTy;

///
class IrTypeFunction : public IrType
{
public:
    ///
    IrTypeFunction(Type* dt);

    ///
    IrTypeFunction* isFunction()  { return this; }

    ///
    const llvm::Type* buildType();

    IrFuncTy* fty() { return irfty; }

protected:
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
    const llvm::Type* buildType();
};

#endif
