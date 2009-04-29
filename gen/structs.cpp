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

void DtoResolveStruct(StructDeclaration* sd)
{
    // don't do anything if already been here
    if (sd->ir.resolved) return;
    // make sure above works :P
    sd->ir.resolved = true;

    // log what we're doing
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

    // perform definition
    bool needs_def = mustDefineSymbol(sd);
    if (needs_def)
    {
        // emit the initZ symbol
        LLGlobalVariable* initZ = irstruct->getInitSymbol();

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

    if (needs_def)
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

// helper function that adds zero bytes to a vector of constants
size_t add_zeros(std::vector<llvm::Value*>& values, size_t diff)
{
    size_t n = values.size();
    bool is64 = global.params.is64bit;
    while (diff)
    {
        if (is64 && diff % 8 == 0)
        {
            values.push_back(llvm::Constant::getNullValue(llvm::Type::Int64Ty));
            diff -= 8;
        }
        else if (diff % 4 == 0)
        {
            values.push_back(llvm::Constant::getNullValue(llvm::Type::Int32Ty));
            diff -= 4;
        }
        else if (diff % 2 == 0)
        {
            values.push_back(llvm::Constant::getNullValue(llvm::Type::Int16Ty));
            diff -= 2;
        }
        else
        {
            values.push_back(llvm::Constant::getNullValue(llvm::Type::Int8Ty));
            diff -= 1;
        }
    }
    return values.size() - n;
}

std::vector<llvm::Value*> DtoStructLiteralValues(const StructDeclaration* sd, const std::vector<llvm::Value*>& inits)
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
        if (os > lastoffset + lastsize)
        {
            //printf("added %lu zeros\n", os - lastoffset - lastsize);
            add_zeros(values, os - lastoffset - lastsize);
        }

        // add the expression value
        values.push_back(inits[i]);

        // update offsets
        lastoffset = os;
        lastsize = sz;

        // go to next explicit init
        exidx++;

        //printf("added field: %s : %lu (%lu)\n", var->toChars(), os, sz);
    }

    // fill out rest with default initializers
    const LLType* structtype = DtoType(sd->type);
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