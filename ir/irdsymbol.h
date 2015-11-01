//===-- ir/irdsymbol.h - Codegen state for D symbols ------------*- C++ -*-===//
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

#ifndef LDC_IR_IRDSYMBOL_H
#define LDC_IR_IRDSYMBOL_H

#include <vector>

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

struct IrDsymbol {
  enum Type {
    NotSet,
    ModuleType,
    AggrType,
    FuncType,
    GlobalType,
    LocalType,
    ParamterType,
    FieldType
  };

  enum State { Initial, Resolved, Declared, Initialized, Defined };

  static std::vector<IrDsymbol *> list;
  static void resetAll();

  // overload all of these to make sure
  // the static list is up to date
  IrDsymbol();
  IrDsymbol(const IrDsymbol &s);
  ~IrDsymbol();

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
  friend IrModule *getIrModule(Module *m);
  friend IrAggr *getIrAggr(AggregateDeclaration *decl, bool create);
  friend IrFunction *getIrFunc(FuncDeclaration *decl, bool create);
  friend IrVar *getIrVar(VarDeclaration *decl);
  friend IrGlobal *getIrGlobal(VarDeclaration *decl, bool create);
  friend IrLocal *getIrLocal(VarDeclaration *decl, bool create);
  friend IrParameter *getIrParameter(VarDeclaration *decl, bool create);
  friend IrField *getIrField(VarDeclaration *decl, bool create);

  union {
    void *irData = nullptr;
    IrModule *irModule;
    IrAggr *irAggr;
    IrFunction *irFunc;
    IrVar *irVar;
    IrGlobal *irGlobal;
    IrLocal *irLocal;
    IrParameter *irParam;
    IrField *irField;
  };
  Type m_type = Type::NotSet;
  State m_state = State::Initial;
};

#endif
