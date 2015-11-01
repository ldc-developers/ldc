//===-- structs.cpp -------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "aggregate.h"
#include "declaration.h"
#include "init.h"
#include "module.h"
#include "mtype.h"
#include "gen/arrays.h"
#include "gen/dvalue.h"
#include "gen/functions.h"
#include "gen/irstate.h"
#include "gen/llvm.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/structs.h"
#include "gen/tollvm.h"
#include "ir/iraggr.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/ManagedStatic.h"
#include <algorithm>

//////////////////////////////////////////////////////////////////////////////////////////

void DtoResolveStruct(StructDeclaration* sd)
{
    DtoResolveStruct(sd, sd->loc);
}

void DtoResolveStruct(StructDeclaration* sd, Loc& callerLoc)
{
    // Make sure to resolve each struct type exactly once.
    if (sd->ir.isResolved()) return;
    sd->ir.setResolved();

    IF_LOG Logger::println("Resolving struct type: %s (%s)", sd->toChars(), sd->loc.toChars());
    LOG_SCOPE;

    // make sure type exists
    DtoType(sd->type);

    // if it's a forward declaration, all bets are off. The type should be enough
    if (sd->sizeok != SIZEOKdone)
    {
        error(callerLoc, "struct %s.%s unknown size", sd->getModule()->toChars(), sd->toChars());
        fatal();
    }

    // create the IrAggr
    getIrAggr(sd, true);

    // Set up our field metadata.
    for (auto vd : sd->fields)
    {
        IF_LOG {
            if (isIrFieldCreated(vd))
                Logger::println("struct field already exists");
        }
        getIrField(vd, true);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////   D STRUCT UTILITIES     ////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////

LLValue* DtoStructEquals(TOK op, DValue* lhs, DValue* rhs)
{
    Type* t = lhs->getType()->toBasetype();
    assert(t->ty == Tstruct);

    // set predicate
    llvm::ICmpInst::Predicate cmpop;
    if (op == TOKequal || op == TOKidentity)
        cmpop = llvm::ICmpInst::ICMP_EQ;
    else
        cmpop = llvm::ICmpInst::ICMP_NE;

    // empty struct? EQ always true, NE always false
    if (static_cast<TypeStruct*>(t)->sym->fields.dim == 0)
        return DtoConstBool(cmpop == llvm::ICmpInst::ICMP_EQ);

    // call memcmp
    size_t sz = getTypePaddedSize(DtoType(t));
    LLValue* val = DtoMemCmp(lhs->getRVal(), rhs->getRVal(), DtoConstSize_t(sz));
    return gIR->ir->CreateICmp(cmpop, val, LLConstantInt::get(val->getType(), 0, false));
}

//////////////////////////////////////////////////////////////////////////////////////////

/// Return the type returned by DtoUnpaddedStruct called on a value of the
/// specified type.
/// Union types will get expanded into a struct, with a type for each member.
LLType* DtoUnpaddedStructType(Type* dty) {
    assert(dty->ty == Tstruct);

    typedef llvm::DenseMap<Type*, llvm::StructType*> CacheT;
    static llvm::ManagedStatic<CacheT> cache;
    auto it = cache->find(dty);
    if (it != cache->end())
        return it->second;

    TypeStruct* sty = static_cast<TypeStruct*>(dty);
    VarDeclarations& fields = sty->sym->fields;

    std::vector<LLType*> types;
    types.reserve(fields.dim);

    for (unsigned i = 0; i < fields.dim; i++) {
        LLType* fty;
        if (fields[i]->type->ty == Tstruct) {
            // Nested structs are the only members that can contain padding
            fty = DtoUnpaddedStructType(fields[i]->type);
        } else {
            fty = DtoType(fields[i]->type);
        }
        types.push_back(fty);
    }
    LLStructType* Ty = LLStructType::get(gIR->context(), types);
    cache->insert(std::make_pair(dty, Ty));
    return Ty;
}

/// Return the struct value represented by v without the padding fields.
/// Unions will be expanded, with a value for each member.
/// Note: v must be a pointer to a struct, but the return value will be a
///       first-class struct value.
LLValue* DtoUnpaddedStruct(Type* dty, LLValue* v) {
    assert(dty->ty == Tstruct);
    TypeStruct* sty = static_cast<TypeStruct*>(dty);
    VarDeclarations& fields = sty->sym->fields;

    LLValue* newval = llvm::UndefValue::get(DtoUnpaddedStructType(dty));

    for (unsigned i = 0; i < fields.dim; i++) {
        LLValue* fieldptr = DtoIndexAggregate(v, sty->sym, fields[i]);
        LLValue* fieldval;
        if (fields[i]->type->ty == Tstruct) {
            // Nested structs are the only members that can contain padding
            fieldval = DtoUnpaddedStruct(fields[i]->type, fieldptr);
        } else {
            fieldval = DtoLoad(fieldptr);
        }
        newval = DtoInsertValue(newval, fieldval, i);
    }
    return newval;
}

/// Undo the transformation performed by DtoUnpaddedStruct, writing to lval.
void DtoPaddedStruct(Type* dty, LLValue* v, LLValue* lval) {
    assert(dty->ty == Tstruct);
    TypeStruct* sty = static_cast<TypeStruct*>(dty);
    VarDeclarations& fields = sty->sym->fields;

    for (unsigned i = 0; i < fields.dim; i++) {
        LLValue* fieldptr = DtoIndexAggregate(lval, sty->sym, fields[i]);
        LLValue* fieldval = DtoExtractValue(v, i);
        if (fields[i]->type->ty == Tstruct) {
            // Nested structs are the only members that can contain padding
            DtoPaddedStruct(fields[i]->type, fieldval, fieldptr);
        } else {
            DtoStore(fieldval, fieldptr);
        }
    }
}
