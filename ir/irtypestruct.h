//===-- ir/irtypestruct.h - IrType subclasses for aggregates ----*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Provides the IrType subclasses used to represent D struct types
// (see irtypeclass.h for the class version).
//
//===----------------------------------------------------------------------===//

#ifndef __LDC_IR_IRTYPESTRUCT_H__
#define __LDC_IR_IRTYPESTRUCT_H__

#include "ir/irtype.h"
#include "llvm/ADT/ArrayRef.h"
#include <utility>

//////////////////////////////////////////////////////////////////////////////

namespace llvm {
    class StructType;
}

struct AggregateDeclaration;
struct StructDeclaration;
struct TypeStruct;

//////////////////////////////////////////////////////////////////////////////

class IrTypeAggr : public IrType
{
public:
    ///
    IrTypeAggr* isAggr()            { return this; }

    ///
    typedef std::vector<VarDeclaration*>::iterator iterator;

    ///
    iterator def_begin()        { return default_fields.begin(); }

    ///
    iterator def_end()          { return default_fields.end(); }

    /// A pair of a member variable declaration and an associated initializer
    /// constant.
    typedef std::pair<VarDeclaration*, llvm::Constant*> VarInitConst;

    /// Creates an initializer constant for the struct type with the given
    /// fields set to the provided constants. The remaining space (not
    /// explicitly specified fields, padding) is default-initialized.
    ///
    /// The optional initializerType parmeter can be used to specify the exact
    /// LLVM type to use for the initializer. If non-null and non-opaque, the
    /// type must exactly match the generated constant. This parameter is used
    /// mainly for supporting legacy code.
    ///
    /// Note that in the general case (if e.g. unions are involved), the
    /// returned type is not necessarily the same as getLLType().
    llvm::Constant* createInitializerConstant(
        llvm::ArrayRef<VarInitConst> initializers,
        llvm::StructType* initializerType = 0);

protected:
    ///
    IrTypeAggr(AggregateDeclaration* ad);

    /// AggregateDeclaration this type represents.
    AggregateDeclaration* aggr;

    /// Sorted list of all default fields.
    /// A default field is a field that contributes to the default initializer
    /// and the default type, and thus it has it's own unique GEP index into
    /// the aggregate.
    /// For classes, field of any super classes are not included.
    std::vector<VarDeclaration*> default_fields;
};

//////////////////////////////////////////////////////////////////////////////

class IrTypeStruct : public IrTypeAggr
{
public:
    ///
    static IrTypeStruct* get(StructDeclaration* sd);

    ///
    IrTypeStruct* isStruct()    { return this; }

    ///
    llvm::Type* buildType();

protected:
    ///
    IrTypeStruct(StructDeclaration* sd);

    /// StructDeclaration this type represents.
    StructDeclaration* sd;

    /// DMD TypeStruct of this type.
    TypeStruct* ts;
};

//////////////////////////////////////////////////////////////////////////////

#endif
