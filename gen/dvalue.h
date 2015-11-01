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
class DFieldValue;
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

  virtual DImValue *isIm() { return nullptr; }
  virtual DConstValue *isConst() { return nullptr; }
  virtual DNullValue *isNull() { return nullptr; }
  virtual DVarValue *isVar() { return nullptr; }
  virtual DFieldValue *isField() { return nullptr; }
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

  DImValue *isIm() override { return this; }

protected:
  llvm::Value *val;
};

// constant d-value
class DConstValue : public DValue {
public:
  DConstValue(Type *t, llvm::Constant *con) : DValue(t), c(con) {}

  llvm::Value *getRVal() override;

  DConstValue *isConst() override { return this; }

  llvm::Constant *c;
};

// null d-value
class DNullValue : public DConstValue {
public:
  DNullValue(Type *t, llvm::Constant *con) : DConstValue(t, con) {}
  DNullValue *isNull() override { return this; }
};

// variable d-value
class DVarValue : public DValue {
public:
  DVarValue(Type *t, VarDeclaration *vd, llvm::Value *llvmValue);
  DVarValue(Type *t, llvm::Value *llvmValue);

  bool isLVal() override { return true; }
  llvm::Value *getLVal() override;
  llvm::Value *getRVal() override;

  /// Returns the underlying storage for special internal ref variables.
  /// Illegal to call on any other value.
  virtual llvm::Value *getRefStorage();

  DVarValue *isVar() override { return this; }

  VarDeclaration *var;

protected:
  llvm::Value *val;
};

// field d-value
class DFieldValue : public DVarValue {
public:
  DFieldValue(Type *t, llvm::Value *llvmValue) : DVarValue(t, llvmValue) {}
  DFieldValue *isField() override { return this; }
};

// slice d-value
class DSliceValue : public DValue {
public:
  DSliceValue(Type *t, llvm::Value *l, llvm::Value *p)
      : DValue(t), len(l), ptr(p) {}

  llvm::Value *getRVal() override;

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

  DFuncValue *isFunc() override { return this; }

  FuncDeclaration *func;
  llvm::Value *val;
  llvm::Value *vthis;
};

#endif // LDC_GEN_DVALUE_H
