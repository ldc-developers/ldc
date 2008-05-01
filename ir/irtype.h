#ifndef LLVMDC_IR_IRTYPE_H
#define LLVMDC_IR_IRTYPE_H

#include "ir/ir.h"

namespace llvm {
    class PATypeHolder;
}

struct IrType
{
    llvm::PATypeHolder* type;
    llvm::PATypeHolder* vtblType;
};

#endif
