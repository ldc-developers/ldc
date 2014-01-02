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
#include "gen/utils.h"
#include "ir/iraggr.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/ManagedStatic.h"
#include <algorithm>

//////////////////////////////////////////////////////////////////////////////////////////

void DtoResolveStruct(StructDeclaration* sd)
{
    // Make sure to resolve each struct type exactly once.
    if (sd->ir.resolved) return;
    sd->ir.resolved = true;

    Logger::println("Resolving struct type: %s (%s)", sd->toChars(), sd->loc.toChars());
    LOG_SCOPE;

    // make sure type exists
    DtoType(sd->type);

    // if it's a forward declaration, all bets are off. The type should be enough
    if (sd->sizeok != SIZEOKdone)
        return;

    // create the IrAggr
    IrAggr* iraggr = new IrAggr(sd);
    sd->ir.irAggr = iraggr;

    // Set up our field metadata.
    for (ArrayIter<VarDeclaration> it(sd->fields); !it.done(); it.next())
    {
        VarDeclaration* vd = it.get();
        assert(!vd->ir.irField);
        (void)new IrField(vd);
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

    // call memcmp
    size_t sz = getTypePaddedSize(DtoType(t));
    LLValue* val = DtoMemCmp(lhs->getRVal(), rhs->getRVal(), DtoConstSize_t(sz));
    return gIR->ir->CreateICmp(cmpop, val, LLConstantInt::get(val->getType(), 0, false), "tmp");
}

//////////////////////////////////////////////////////////////////////////////////////////

LLValue* DtoIndexStruct(LLValue* src, StructDeclaration* sd, VarDeclaration* vd)
{
    Logger::println("indexing struct field %s:", vd->toPrettyChars());
    LOG_SCOPE;

    DtoResolveStruct(sd);

    // vd must be a field
    IrField* field = vd->ir.irField;
    assert(field);

    // get the start pointer
    LLType* st = getPtrToType(DtoType(sd->type));

    // cast to the formal struct type
    src = DtoBitCast(src, st);

    // gep to the index
    LLValue* val = DtoGEPi(src, 0, field->index);

    // do we need to offset further? (union area)
    if (field->unionOffset)
    {
        // cast to void*
        val = DtoBitCast(val, getVoidPtrType());
        // offset
        val = DtoGEPi1(val, field->unionOffset);
    }

    // cast it to the right type
    val = DtoBitCast(val, getPtrToType(i1ToI8(DtoType(vd->type))));

    if (Logger::enabled())
        Logger::cout() << "value: " << *val << '\n';

    return val;
}

//////////////////////////////////////////////////////////////////////////////////////////

/// Return the type returned by DtoUnpaddedStruct called on a value of the
/// specified type.
/// Union types will get expanded into a struct, with a type for each member.
LLType* DtoUnpaddedStructType(Type* dty) {
    assert(dty->ty == Tstruct);

    typedef llvm::DenseMap<Type*, llvm::StructType*> CacheT;
    static llvm::ManagedStatic<CacheT> cache;
    CacheT::iterator it = cache->find(dty);
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
        LLValue* fieldptr = DtoIndexStruct(v, sty->sym, fields[i]);
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
        LLValue* fieldptr = DtoIndexStruct(lval, sty->sym, fields[i]);
        LLValue* fieldval = DtoExtractValue(v, i);
        if (fields[i]->type->ty == Tstruct) {
            // Nested structs are the only members that can contain padding
            DtoPaddedStruct(fields[i]->type, fieldval, fieldptr);
        } else {
            DtoStore(fieldval, fieldptr);
        }
    }
}
