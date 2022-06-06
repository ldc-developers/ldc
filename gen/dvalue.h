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

#pragma once

class Type;
class Dsymbol;
class VarDeclaration;
class FuncDeclaration;

namespace llvm {
class Value;
class Type;
class Constant;
}

class DValue;
class DRValue;
class DImValue;
class DConstValue;
class DNullValue;
class DLValue;
class DSpecialRefValue;
class DSliceValue;
class DFuncValue;

/// Represents an immutable pair of LLVM value and associated D type.
class DValue {
public:
  Type *const type;

  virtual ~DValue() = default;

  /// Returns true iff the value can be accessed at the end of the entry basic
  /// block of the current function, in the sense that it is either not derived
  /// from an llvm::Instruction (but from a global, constant, etc.) or that
  /// instruction is part of the entry basic block.
  ///
  /// In other words, whatever the value might be derived from then certainly
  /// dominates uses in all other basic blocks of the function.
  virtual bool definedInFuncEntryBB();

  virtual DRValue *getRVal() { return nullptr; }

  virtual DLValue *isLVal() { return nullptr; }
  virtual DSpecialRefValue *isSpecialRef() { return nullptr; }

  virtual DRValue *isRVal() { return nullptr; }
  virtual DImValue *isIm() { return nullptr; }
  virtual DConstValue *isConst() { return nullptr; }
  virtual DNullValue *isNull() { return nullptr; }
  virtual DSliceValue *isSlice() { return nullptr; }
  virtual DFuncValue *isFunc() { return nullptr; }

protected:
  llvm::Value *const val;

  DValue(Type *t, llvm::Value *v);

  friend llvm::Value *DtoRVal(DValue *v);
};

/// Represents a D rvalue via a low-level rvalue.
class DRValue : public DValue {
public:
  DRValue *getRVal() override { return this; }

  DRValue *isRVal() override { return this; }

protected:
  DRValue(Type *t, llvm::Value *v);
};

/// Represents an immediate D value (simple rvalue with no special properties
/// like being a compile-time constant) via a low-level rvalue.
/// Restricted to primitive types such as pointers (incl. class references),
/// integral and floating-point types.
class DImValue : public DRValue {
public:
  DImValue(Type *t, llvm::Value *v);

  DImValue *isIm() override { return this; }
};

/// Represents a D compile-time constant via a low-level constant.
class DConstValue : public DRValue {
public:
  DConstValue(Type *t, llvm::Constant *con);

  bool definedInFuncEntryBB() override { return true; }

  DConstValue *isConst() override { return this; }
};

/// Represents a D compile-time null constant.
class DNullValue : public DConstValue {
public:
  DNullValue(Type *t, llvm::Constant *con) : DConstValue(t, con) {}

  DNullValue *isNull() override { return this; }
};

/// Represents a D slice (dynamic array).
class DSliceValue : public DRValue {
  llvm::Value *const length = nullptr;
  llvm::Value *const ptr = nullptr;

  DSliceValue(Type *t, llvm::Value *pair, llvm::Value *length,
              llvm::Value *ptr);

public:
  DSliceValue(Type *t, llvm::Value *pair);
  DSliceValue(Type *t, llvm::Value *length, llvm::Value *ptr);

  DSliceValue *isSlice() override { return this; }

  llvm::Value *getLength();
  llvm::Value *getPtr();
};

/// Represents a D function value with optional this/context pointer.
class DFuncValue : public DRValue {
public:
  FuncDeclaration *func;
  llvm::Value *vthis;

  DFuncValue(Type *t, FuncDeclaration *fd, llvm::Value *v,
             llvm::Value *vt = nullptr);
  DFuncValue(FuncDeclaration *fd, llvm::Value *v, llvm::Value *vt = nullptr);

  bool definedInFuncEntryBB() override;

  DFuncValue *isFunc() override { return this; }
};

/// Represents a D value in memory via a low-level lvalue (pointer).
/// This doesn't imply that the D value is an lvalue too - e.g., we always
/// keep structs and static arrays in memory.
class DLValue : public DValue {
public:
  DLValue(Type *t, llvm::Value *v);

  DRValue *getRVal() override;
  virtual DLValue *getLVal() { return this; }

  DLValue *isLVal() override { return this; }

protected:
  DLValue(llvm::Value *v, Type *t) : DValue(t, v) {}

  friend llvm::Value *DtoLVal(DValue *v);
};

/// Represents special internal ref variables.
class DSpecialRefValue : public DLValue {
public:
  DSpecialRefValue(Type *t, llvm::Value *v);

  DRValue *getRVal() override;
  DLValue *getLVal() override;
  llvm::Value *getRefStorage() { return val; }

  DSpecialRefValue *isSpecialRef() override { return this; }
};

inline llvm::Value *DtoRVal(DValue *v) { return v->getRVal()->val; }
llvm::Value *DtoLVal(DValue *v);
