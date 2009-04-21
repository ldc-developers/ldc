#ifndef LDC_IR_IRDTYPE_H
#define LDC_IR_IRDTYPE_H

#include <set>

namespace llvm {
    class PATypeHolder;
}

struct IrDType
{
    IrDType();
    llvm::PATypeHolder* type;
};

#endif
