#ifndef LDC_IR_IRDSYMBOL_H
#define LDC_IR_IRDSYMBOL_H

#include <set>

struct IrModule;
struct IrFunction;
struct IrStruct;
struct IrGlobal;
struct IrLocal;
struct IrParameter;
struct IrField;
struct IrVar;
struct Dsymbol;

namespace llvm {
    class Value;
}

struct IrDsymbol
{
    static std::set<IrDsymbol*> list;
    static void resetAll();

    // overload all of these to make sure
    // the static list is up to date
    IrDsymbol();
    IrDsymbol(const IrDsymbol& s);
    ~IrDsymbol();

    void reset();

    Module* DModule;

    bool resolved;
    bool declared;
    bool initialized;
    bool defined;

    IrModule* irModule;

    IrStruct* irStruct;

    IrFunction* irFunc;

    IrGlobal* irGlobal;
    union {
        IrLocal* irLocal;
        IrParameter *irParam;
    };
    IrField* irField;
    IrVar* getIrVar();
    llvm::Value*& getIrValue();

    bool isSet();
};

#endif
