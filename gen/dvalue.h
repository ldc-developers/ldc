//===-- gen/dvalue.h - D value abstractions ---------------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// These classes are used for generating the IR. They encapsulate an LLVM value
// together with a D type and provide an uniform interface for the most common
// operations. When more specialize handling is necessary, they hold enough
// information to do-the-right-thing (TM).
//
//===----------------------------------------------------------------------===//

#ifndef LDC_GEN_DVALUE_H
#define LDC_GEN_DVALUE_H

#include "root.h"
#include <cassert>

class Type;
class Dsymbol;
class VarDeclaration;
class FuncDeclaration;

namespace llvm {
class Value;
class Type;
class Constant;
}

class DImValue;
class DConstValue;
class DNullValue;
class DVarValue;
class DFuncValue;
class DSliceValue;

// base class for d-values
class DValue {
public:
  Type *type;
  explicit DValue(Type *ty) : type(ty) {}
  virtual ~DValue() = default;

  Type *&getType() {
    assert(type);
    return type;
  }

  virtual llvm::Value *getLVal() {
    assert(0);
    return nullptr;
  }
  virtual llvm::Value *getRVal() {
    assert(0);
    return nullptr;
  }

  virtual bool isLVal() { return false; }

  /// Returns true iff the value can be accessed at the end of the entry basic
  /// block of the current function, in the sense that it is either not derived
  /// from an llvm::Instruction (but from a global, constant, etc.) or that
  /// instruction is part of the entry basic block.
  ///
  /// In other words, whatever value the result of getLVal()/getRVal() might be
  /// derived from then certainly dominates uses in all other basic blocks of
  /// the function.
  virtual bool definedInFuncEntryBB() = 0;

  virtual DImValue *isIm() { return nullptr; }
  virtual DConstValue *isConst() { return nullptr; }
  virtual DNullValue *isNull() { return nullptr; }
  virtual DVarValue *isVar() { return nullptr; }
  virtual DSliceValue *isSlice() { return nullptr; }
  virtual DFuncValue *isFunc() { return nullptr; }

protected:
  DValue() = default;
  DValue(const DValue &) {}
  DValue &operator=(const DValue &other) {
    type = other.type;
    return *this;
  }
};

// immediate d-value
class DImValue : public DValue {
public:
  DImValue(Type *t, llvm::Value *v) : DValue(t), val(v) {}

  llvm::Value *getRVal() override {
    assert(val);
    return val;
  }

  bool definedInFuncEntryBB() override;

  DImValue *isIm() override { return this; }

protected:
  llvm::Value *val;
};

// constant d-value
class DConstValue : public DValue {
public:
  DConstValue(Type *t, llvm::Constant *con) : DValue(t), c(con) {}

  llvm::Value *getRVal() override;

  bool definedInFuncEntryBB() override { return true; }

  DConstValue *isConst() override { return this; }

  llvm::Constant *c;
};

// null d-value
class DNullValue : public DConstValue {
public:
  DNullValue(Type *t, llvm::Constant *con) : DConstValue(t, con) {}
  DNullValue *isNull() override { return this; }
};

/// This is really a misnomer, DVarValue represents generic lvalues, which
/// might or might not come from variable declarations.
// TODO: Rename this, probably remove getLVal() from parent since this is the
// only lvalue. The isSpecialRefVar case should probably also be its own
// subclass.
class DVarValue : public DValue {
public:
  DVarValue(Type *t, llvm::Value *llvmValue, bool isSpecialRefVar = false);

  bool isLVal() override { return true; }
  llvm::Value *getLVal() override;
  llvm::Value *getRVal() override;

  /// Returns the underlying storage for special internal ref variables.
  /// Illegal to call on any other value.
  llvm::Value *getRefStorage();

  bool definedInFuncEntryBB() override;

  DVarValue *isVar() override { return this; }

protected:
  llvm::Value *const val;
  bool const isSpecialRefVar;
};

// slice d-value
class DSliceValue : public DValue {
public:
  DSliceValue(Type *t, llvm::Value *l, llvm::Value *p)
      : DValue(t), len(l), ptr(p) {}

  llvm::Value *getRVal() override;

  bool definedInFuncEntryBB() override;

  DSliceValue *isSlice() override { return this; }

  llvm::Value *len;
  llvm::Value *ptr;
};

// function d-value
class DFuncValue : public DValue {
public:
  DFuncValue(Type *t, FuncDeclaration *fd, llvm::Value *v,
             llvm::Value *vt = nullptr);
  DFuncValue(FuncDeclaration *fd, llvm::Value *v, llvm::Value *vt = nullptr);

  llvm::Value *getRVal() override;

  bool definedInFuncEntryBB() override;

  DFuncValue *isFunc() override { return this; }

  FuncDeclaration *func;
  llvm::Value *val;
  llvm::Value *vthis;
};

#endif // LDC_GEN_DVALUE_H
