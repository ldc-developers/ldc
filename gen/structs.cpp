#include <algorithm>

#include "gen/llvm.h"

#include "mtype.h"
#include "aggregate.h"
#include "init.h"
#include "declaration.h"

#include "gen/irstate.h"
#include "gen/tollvm.h"
#include "gen/llvmhelpers.h"
#include "gen/arrays.h"
#include "gen/logger.h"
#include "gen/structs.h"
#include "gen/dvalue.h"
#include "gen/functions.h"
#include "gen/utils.h"

#include "ir/irstruct.h"
#include "ir/irtypestruct.h"

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
    const LLType* st = getPtrToType(DtoType(sd->type));

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

void DtoResolveStruct(StructDeclaration* sd)
{
    // don't do anything if already been here
    if (sd->ir.resolved) return;
    // make sure above works :P
    sd->ir.resolved = true;

    // log what we're doing
    Logger::println("Resolving struct type: %s (%s)", sd->toChars(), sd->locToChars());
    LOG_SCOPE;

    // make sure type exists
    DtoType(sd->type);

    // create the IrStruct
    IrStruct* irstruct = new IrStruct(sd);
    sd->ir.irStruct = irstruct;

    // emit the initZ symbol
    LLGlobalVariable* initZ = irstruct->getInitSymbol();

    // perform definition
    if (mustDefineSymbol(sd))
    {
        // set initZ initializer
        initZ->setInitializer(irstruct->getDefaultInit());
    }

    // emit members
    if (sd->members)
    {
        ArrayIter<Dsymbol> it(*sd->members);
        while (!it.done())
        {
            Dsymbol* member = it.get();
            if (member)
                member->codegen(Type::sir);
            it.next();
        }
    }

    // emit typeinfo
    DtoTypeInfoOf(sd->type);
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
