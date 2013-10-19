//===-- ir/irtypestruct.h - IrType for structs and unions -------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#ifndef LDC_IR_IRTYPESTRUCT_H
#define LDC_IR_IRTYPESTRUCT_H

#include "ir/irtypeaggr.h"

class StructDeclaration;
class TypeStruct;

/// IrType for struct/union types.
class IrTypeStruct : public IrTypeAggr
{
public:
    ///
    static IrTypeStruct* get(StructDeclaration* sd);

    ///
    IrTypeStruct* isStruct()    { return this; }

protected:
    ///
    IrTypeStruct(StructDeclaration* sd);

    /// StructDeclaration this type represents.
    StructDeclaration* sd;

    /// DMD TypeStruct of this type.
    TypeStruct* ts;
};

#endif
