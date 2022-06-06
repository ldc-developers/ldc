//===-- ir/irtypestruct.h - IrType for structs and unions -------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "ir/irtypeaggr.h"

class StructDeclaration;
class TypeStruct;

/// IrType for struct/union types.
class IrTypeStruct : public IrTypeAggr {
public:
  ///
  static IrTypeStruct *get(StructDeclaration *sd);

  ///
  IrTypeStruct *isStruct() override { return this; }

  ///
  static void resetDComputeTypes();
  
protected:
  ///
  explicit IrTypeStruct(StructDeclaration *sd);

  ///
  static std::vector<IrTypeStruct*> dcomputeTypes;

  /// StructDeclaration this type represents.
  StructDeclaration *sd = nullptr;

  /// DMD TypeStruct of this type.
  TypeStruct *ts = nullptr;
};
