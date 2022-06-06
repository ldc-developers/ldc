//===-- gen/rttibuilder.h - TypeInfo generation helper ----------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// This class is used to build the global TypeInfo/ClassInfo/... constants
// required for the D runtime type information system.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constant.h"

class ClassDeclaration;
class Dsymbol;
class FuncDeclaration;
class Type;
namespace llvm {
class StructType;
class GlobalVariable;
}

class RTTIBuilder {
  /// The offset (in bytes) at which the previously pushed field ended.
  uint64_t prevFieldEnd = 0;

public:
  // 15 is enough for any D2 ClassInfo including 64 bit pointer alignment
  // padding
  llvm::SmallVector<llvm::Constant *, 15> inits;

  explicit RTTIBuilder(Type *baseType);

  void push(llvm::Constant *C);
  void push_null(Type *T);
  void push_null_vp();
  void push_null_void_array();
  void push_uint(unsigned u);
  void push_size(uint64_t s);
  void push_size_as_vp(uint64_t s);
  void push_string(const char *str);
  void push_typeinfo(Type *t);

  /// pushes the function pointer or a null void* if it cannot.
  void push_funcptr(FuncDeclaration *fd, Type *castto = nullptr);

  /// pushes the array slice given.
  void push_array(uint64_t dim, llvm::Constant *ptr);

  /// pushes void[] slice, dim is used directly, ptr is cast to void* .
  void push_void_array(uint64_t dim, llvm::Constant *ptr);

  /// pushes void[] slice with data.
  /// CI is the constant initializer the array should point to, the length
  /// and ptr are resolved automatically
  void push_void_array(llvm::Constant *CI, Type *valtype, Dsymbol *mangle_sym);

  /// pushes valtype[] slice with data.
  /// CI is the constant initializer that .ptr should point to
  /// dim is .length member directly
  /// valtype provides the D element type, .ptr is cast to valtype->pointerTo()
  /// mangle_sym provides the mangle prefix for the symbol generated.
  void push_array(llvm::Constant *CI, uint64_t dim, Type *valtype,
                  Dsymbol *mangle_sym);

  /// Creates the initializer constant and assigns it to the global.
  void finalize(llvm::GlobalVariable *gvar);

  /// Creates the initializer constant and assigns it to the global.
  llvm::Constant *get_constant(llvm::StructType *initType);
};
