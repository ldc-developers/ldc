//===-- ir/irmetadata.h - Codegen state for D symbols ------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Represents the status of a D symbol on its way though the codegen process.
//
//===----------------------------------------------------------------------===//

#ifndef LDC_IR_IRMETAD_H
#define LDC_IR_IRMETAD_H

struct IrModule;
struct IrFunction;
struct IrAggr;
struct IrGlobal;
struct IrLocal;
struct IrParameter;
struct IrField;
struct IrVar;
class Dsymbol;
class AggregateDeclaration;
class FuncDeclaration;
class VarDeclaration;
class Module;

namespace llvm {
    class Value;
}

struct IrMetadata
{
    enum Type
    {
        NotSet,
        ModuleType,
        AggrType,
        FuncType,
        GlobalType,
        LocalType,
        ParamterType,
        FieldType
    };

    enum State
    {
        Initial,
        Resolved,
        Declared,
        Initialized,
        Defined
    };

    IrMetadata();

    static void resetAll();

    void reset();

    Type type() const { return m_type; }
    State state() const { return m_state; }

    bool isResolved() const { return m_state >= Resolved; }
    bool isDeclared() const { return m_state >= Declared; }
    bool isInitialized() const { return m_state >= Initialized; }
    bool isDefined() const { return m_state >= Defined; }

    void setResolved();
    void setDeclared();
    void setInitialized();
    void setDefined();
private:
    friend IrVar *getIrVar(VarDeclaration *);
    friend IrGlobal *getIrGlobal(VarDeclaration *, bool);
    friend IrLocal *getIrLocal(VarDeclaration *, bool);
    friend IrParameter *getIrParameter(VarDeclaration *, bool);
    friend IrField *getIrField(VarDeclaration *, bool);
    friend IrFunction *getIrFunc(FuncDeclaration *, bool);
    friend IrAggr *getIrAggr(AggregateDeclaration *, bool);
    friend IrModule* getIrModule(Module *);

    union {
        void*        irData;
        IrModule*    irModule;
        IrAggr*      irAggr;
        IrFunction*  irFunc;
        IrVar*       irVar;
        IrGlobal*    irGlobal;
        IrLocal*     irLocal;
        IrParameter* irParam;
        IrField*     irField;
    };
    Type m_type;
    State m_state;
};

IrMetadata *getIrMetadata(Dsymbol *sym);

IrVar *getIrVar(VarDeclaration *decl);
bool isIrVarCreated(VarDeclaration *decl);

IrGlobal *getIrGlobal(VarDeclaration *decl, bool create = false);
bool isIrGlobalCreated(VarDeclaration *decl);

IrLocal *getIrLocal(VarDeclaration *decl, bool create = false);
bool isIrLocalCreated(VarDeclaration *decl);

IrParameter *getIrParameter(VarDeclaration *decl, bool create = false);
bool isIrParameterCreated(VarDeclaration *decl);

IrField *getIrField(VarDeclaration *decl, bool create = false);
bool isIrFieldCreated(VarDeclaration *decl);

IrFunction *getIrFunc(FuncDeclaration *decl, bool create = false);
bool isIrFuncCreated(FuncDeclaration *decl);

IrAggr *getIrAggr(AggregateDeclaration *decl, bool create = false);
bool isIrAggrCreated(AggregateDeclaration *decl);

IrModule *getIrModule(Module *m);

#endif // LDC_IR_IRMETAD_H