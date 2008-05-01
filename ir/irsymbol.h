#ifndef LLVMDC_IR_IRSYMBOL_H
#define LLVMDC_IR_IRSYMBOL_H

#include "ir/ir.h"

struct IrModule;
struct IrFunction;
struct IrStruct;
struct IrGlobal;
struct IrLocal;
struct IrField;
struct IrVar;

struct IrDsymbol
{
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
    IrVar* getIrVar()
    {
        assert(irGlobal || irLocal || irField);
        return irGlobal ? (IrVar*)irGlobal : irLocal ? (IrVar*)irLocal : (IrVar*)irField;
    }
    llvm::Value*& getIrValue() { return getIrVar()->value; }
};

#endif
