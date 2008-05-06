#ifndef LLVMDC_IR_IRSYMBOL_H
#define LLVMDC_IR_IRSYMBOL_H

#include <set>

struct IrModule;
struct IrFunction;
struct IrStruct;
struct IrGlobal;
struct IrLocal;
struct IrField;
struct IrVar;
struct Dsymbol;

namespace llvm {
    struct Value;
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
    IrLocal* irLocal;
    IrField* irField;
    IrVar* getIrVar();
    llvm::Value*& getIrValue();

    bool isSet();
};

#endif
