#ifndef LDC_IR_IRTYPE_H
#define LDC_IR_IRTYPE_H

#include <set>

namespace llvm {
    class PATypeHolder;
}

struct IrType
{
    static std::set<IrType*> list;
    static void resetAll();

    // overload all of these to make sure
    // the static list is up to date
    IrType();
    IrType(const IrType& s);
    ~IrType();

    void reset();

    llvm::PATypeHolder* type;
    llvm::PATypeHolder* vtblType;
};

#endif
