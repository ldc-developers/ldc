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
class DLValue;
class DFuncValue;
class DSliceValue;

// base class for d-values
class DValue {
public:
  Type *const type;

  virtual ~DValue() = default;

  virtual llvm::Value *getLVal() {
    assert(0 && "DValue is not a LL lvalue!");
    return nullptr;
  }
  virtual llvm::Value *getRVal() { return val; }

  virtual bool isLVal() { return false; }

  /// Returns true iff the value can be accessed at the end of the entry basic
  /// block of the current function, in the sense that it is either not derived
  /// from an llvm::Instruction (but from a global, constant, etc.) or that
  /// instruction is part of the entry basic block.
  ///
  /// In other words, whatever value the result of getLVal()/getRVal() might be
  /// derived from then certainly dominates uses in all other basic blocks of
  /// the function.
  virtual bool definedInFuncEntryBB();

  virtual DImValue *isIm() { return nullptr; }
  virtual DConstValue *isConst() { return nullptr; }
  virtual DNullValue *isNull() { return nullptr; }
  virtual DLValue *isVar() { return nullptr; }
  virtual DSliceValue *isSlice() { return nullptr; }
  virtual DFuncValue *isFunc() { return nullptr; }

protected:
  llvm::Value *const val;

  DValue(Type *t, llvm::Value *v) : type(t), val(v) {
    assert(type);
    assert(val);
  }
};

// immediate d-value
class DImValue : public DValue {
public:
  DImValue(Type *t, llvm::Value *v);

  DImValue *isIm() override { return this; }
};

// constant d-value
class DConstValue : public DValue {
public:
  DConstValue(Type *t, llvm::Constant *con);

  bool definedInFuncEntryBB() override { return true; }

  DConstValue *isConst() override { return this; }
};

// null d-value
class DNullValue : public DConstValue {
public:
  DNullValue(Type *t, llvm::Constant *con) : DConstValue(t, con) {}

  DNullValue *isNull() override { return this; }
};

/// Represents a D value in memory via a low-level lvalue.
/// This doesn't imply that the D value is an lvalue too - e.g., we always
/// keep structs and static arrays in memory.
// TODO: Probably remove getLVal() from parent since this is the only lvalue.
// The isSpecialRefVar case should probably also be its own subclass.
class DLValue : public DValue {
public:
  DLValue(Type *t, llvm::Value *v, bool isSpecialRefVar = false);

  bool isLVal() override { return true; }

  llvm::Value *getLVal() override;
  llvm::Value *getRVal() override;

  /// Returns the underlying storage for special internal ref variables.
  /// Illegal to call on any other value.
  llvm::Value *getRefStorage();

  DLValue *isVar() override { return this; }

protected:
  const bool isSpecialRefVar;
};

// slice d-value
class DSliceValue : public DValue {
public:
  DSliceValue(Type *t, llvm::Value *length, llvm::Value *ptr);

  DSliceValue *isSlice() override { return this; }

  llvm::Value *getLength();
  llvm::Value *getPtr();
};

// function d-value
class DFuncValue : public DValue {
public:
  FuncDeclaration *func;
  llvm::Value *vthis;

  DFuncValue(Type *t, FuncDeclaration *fd, llvm::Value *v,
             llvm::Value *vt = nullptr);
  DFuncValue(FuncDeclaration *fd, llvm::Value *v, llvm::Value *vt = nullptr);

  bool definedInFuncEntryBB() override;

  DFuncValue *isFunc() override { return this; }
};

#endif // LDC_GEN_DVALUE_H
