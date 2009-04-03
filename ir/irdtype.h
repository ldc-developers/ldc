#ifndef LDC_IR_IRDTYPE_H
#define LDC_IR_IRDTYPE_H

#include <set>

namespace llvm {
    class PATypeHolder;
}

struct IrDType
{
    static std::set<IrDType*> list;
    static void resetAll();

    // overload all of these to make sure
    // the static list is up to date
    IrDType();
    IrDType(const IrDType& s);
    ~IrDType();

    void reset();

    llvm::PATypeHolder* type;
};

#endif
