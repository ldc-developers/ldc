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
#include "ir/irstruct.h"
#include "ir/irtypestruct.h"
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
    if (sd->sizeok != 1)
        return;

    // create the IrStruct
    IrStruct* irstruct = new IrStruct(sd);
    sd->ir.irStruct = irstruct;

    // Set up our field metadata.
    for (ArrayIter<VarDeclaration> it(sd->fields); !it.done(); it.next())
    {
        VarDeclaration* vd = it.get();
        assert(!vd->ir.irField);
        (void)new IrField(vd);
    }

    // perform definition
    bool emitGlobalData = mustDefineSymbol(sd);
    if (emitGlobalData)
    {
        // emit the initZ symbol
        LLGlobalVariable* initZ = irstruct->getInitSymbol();

        // set initZ initializer
        initZ->setInitializer(irstruct->getDefaultInit());
    }

    // emit members
    if (sd->members)
    {
        for (ArrayIter<Dsymbol> it(sd->members); !it.done(); it.next())
        {
            it.get()->codegen(Type::sir);
        }
    }

    if (emitGlobalData)
    {
        // emit typeinfo
        DtoTypeInfoOf(sd->type);
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
    val = DtoBitCast(val, getPtrToType(DtoType(vd->type)));

    if (Logger::enabled())
        Logger::cout() << "value: " << *val << '\n';

    return val;
}

//////////////////////////////////////////////////////////////////////////////////////////

// helper function that adds zero bytes to a vector of constants
size_t add_zeros(std::vector<llvm::Value*>& values, size_t diff)
{
    size_t n = values.size();
    bool is64 = global.params.is64bit;
    while (diff)
    {
        if (is64 && diff % 8 == 0)
        {
            values.push_back(LLConstant::getNullValue(llvm::Type::getInt64Ty(gIR->context())));
            diff -= 8;
        }
        else if (diff % 4 == 0)
        {
            values.push_back(LLConstant::getNullValue(llvm::Type::getInt32Ty(gIR->context())));
            diff -= 4;
        }
        else if (diff % 2 == 0)
        {
            values.push_back(LLConstant::getNullValue(llvm::Type::getInt16Ty(gIR->context())));
            diff -= 2;
        }
        else
        {
            values.push_back(LLConstant::getNullValue(llvm::Type::getInt8Ty(gIR->context())));
            diff -= 1;
        }
    }
    return values.size() - n;
}

std::vector<llvm::Value*> DtoStructLiteralValues(const StructDeclaration* sd,
                                                 const std::vector<llvm::Value*>& inits,
                                                 bool isConst)
{
    // get arrays
    size_t nvars = sd->fields.dim;
    VarDeclaration** vars = (VarDeclaration**)sd->fields.data;

    assert(inits.size() == nvars);

    // first locate all explicit initializers
    std::vector<VarDeclaration*> explicitInits;
    for (size_t i=0; i < nvars; i++)
    {
        if (inits[i])
        {
            explicitInits.push_back(vars[i]);
        }
    }

    // vector of values to build aggregate from
    std::vector<llvm::Value*> values;

    // offset trackers
    size_t lastoffset = 0;
    size_t lastsize = 0;

    // index of next explicit init
    size_t exidx = 0;
    // number of explicit inits
    size_t nex = explicitInits.size();

    // for through each field and build up the struct, padding with zeros
    size_t i;
    for (i=0; i<nvars; i++)
    {
        VarDeclaration* var = vars[i];

        // get var info
        size_t os = var->offset;
        size_t sz = var->type->size();

        // get next explicit
        VarDeclaration* nextVar = NULL;
        size_t nextOs = 0;
        if (exidx < nex)
        {
            nextVar = explicitInits[exidx];
            nextOs = nextVar->offset;
        }
        // none, rest is defaults
        else
        {
            break;
        }

        // not explicit initializer, default initialize if there is room, otherwise skip
        if (!inits[i])
        {
            // default init if there is room
            // (past current offset) and (small enough to fit before next explicit)
            if ((os >= lastoffset + lastsize) && (os+sz <= nextOs))
            {
                // add any 0 padding needed before this field
                if (os > lastoffset + lastsize)
                {
                    //printf("1added %lu zeros\n", os - lastoffset - lastsize);
                    add_zeros(values, os - lastoffset - lastsize);
                }

                // get field default init
                IrField* f = var->ir.irField;
                assert(f);
                values.push_back(f->getDefaultInit());

                lastoffset = os;
                lastsize = sz;
                //printf("added default: %s : %lu (%lu)\n", var->toChars(), os, sz);
            }
            // skip
            continue;
        }

        assert(nextVar == var);

        // add any 0 padding needed before this field
        if (!isConst && os > lastoffset + lastsize)
        {
            //printf("added %lu zeros\n", os - lastoffset - lastsize);
            add_zeros(values, os - lastoffset - lastsize);
        }

        size_t repCount = 1;
        // compute repCount to fill each array dimension
        for (Type *varType = var->type;
             varType->ty == Tsarray;
             varType = varType->nextOf())
        {
            repCount *= static_cast<TypeSArray*>(varType)->dim->toUInteger();
        }

        // add the expression values
        std::fill_n(std::back_inserter(values), repCount, inits[i]);

        // update offsets
        lastoffset = os;
#if DMDV2
        // sometimes size of the initializer is less than size of the variable,
        // so make sure that lastsize is correct
        if (inits[i]->getType()->isSized())
            lastsize = ceil(gDataLayout->getTypeSizeInBits(inits[i]->getType()) / 8.0);
        else
#endif
        lastsize = sz;

        // go to next explicit init
        exidx++;

        //printf("added field: %s : %lu (%lu)\n", var->toChars(), os, sz);
    }

    // fill out rest with default initializers
    LLType* structtype = DtoType(sd->type);
    size_t structsize = getTypePaddedSize(structtype);

    // FIXME: this could probably share some code with the above
    if (structsize > lastoffset+lastsize)
    {
        for (/*continue from first loop*/; i < nvars; i++)
        {
            VarDeclaration* var = vars[i];

            // get var info
            size_t os = var->offset;
            size_t sz = var->type->size();

            // skip?
            if (os < lastoffset + lastsize)
                continue;

            // add any 0 padding needed before this field
            if (os > lastoffset + lastsize)
            {
                //printf("2added %lu zeros\n", os - lastoffset - lastsize);
                add_zeros(values, os - lastoffset - lastsize);
            }

            // get field default init
            IrField* f = var->ir.irField;
            assert(f);
            values.push_back(f->getDefaultInit());

            lastoffset = os;
            lastsize = sz;
            //printf("2added default: %s : %lu (%lu)\n", var->toChars(), os, sz);
        }
    }

    // add any 0 padding needed at the end of the literal
    if (structsize > lastoffset+lastsize)
    {
        //printf("3added %lu zeros\n", structsize - lastoffset - lastsize);
        add_zeros(values, structsize - lastoffset - lastsize);
    }

    return values;
}

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
    Array& fields = sty->sym->fields;

    std::vector<LLType*> types;

    for (unsigned i = 0; i < fields.dim; i++) {
        VarDeclaration* vd = static_cast<VarDeclaration*>(fields.data[i]);
        LLType* fty;
        if (vd->type->ty == Tstruct) {
            // Nested structs are the only members that can contain padding
            fty = DtoUnpaddedStructType(vd->type);
        } else {
            fty = DtoType(vd->type);
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
    Array& fields = sty->sym->fields;

    LLValue* newval = llvm::UndefValue::get(DtoUnpaddedStructType(dty));

    for (unsigned i = 0; i < fields.dim; i++) {
        VarDeclaration* vd = static_cast<VarDeclaration*>(fields.data[i]);
        LLValue* fieldptr = DtoIndexStruct(v, sty->sym, vd);
        LLValue* fieldval;
        if (vd->type->ty == Tstruct) {
            // Nested structs are the only members that can contain padding
            fieldval = DtoUnpaddedStruct(vd->type, fieldptr);
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
    Array& fields = sty->sym->fields;

    for (unsigned i = 0; i < fields.dim; i++) {
        VarDeclaration* vd = static_cast<VarDeclaration*>(fields.data[i]);
        LLValue* fieldptr = DtoIndexStruct(lval, sty->sym, vd);
        LLValue* fieldval = DtoExtractValue(v, i);
        if (vd->type->ty == Tstruct) {
            // Nested structs are the only members that can contain padding
            DtoPaddedStruct(vd->type, fieldval, fieldptr);
        } else {
            DtoStore(fieldval, fieldptr);
        }
    }
}
