#ifndef __LDC_IR_IRTYPESTRUCT_H__
#define __LDC_IR_IRTYPESTRUCT_H__

#include "ir/irtype.h"

//////////////////////////////////////////////////////////////////////////////

struct AggregateDeclaration;
struct StructDeclaration;
struct TypeStruct;

//////////////////////////////////////////////////////////////////////////////

class IrTypeAggr : public IrType
{
public:
    ///
    IrTypeAggr(AggregateDeclaration* ad);

    ///
    IrTypeAggr* isAggr()            { return this; }

protected:
    /// AggregateDeclaration this type represents.
    AggregateDeclaration* aggr;
};

//////////////////////////////////////////////////////////////////////////////

class IrTypeStruct : public IrTypeAggr
{
public:
    ///
    IrTypeStruct(StructDeclaration* sd);

    ///
    IrTypeStruct* isStruct()    { return this; }

    ///
    const llvm::Type* buildType();

protected:
    /// StructDeclaration this type represents.
    StructDeclaration* sd;

    /// DMD TypeStruct of this type.
    TypeStruct* ts;
};

//////////////////////////////////////////////////////////////////////////////

#endif
