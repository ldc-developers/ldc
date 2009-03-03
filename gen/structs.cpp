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

#include "ir/irstruct.h"

//////////////////////////////////////////////////////////////////////////////////////////

// pair of var and its init
typedef std::pair<VarDeclaration*,Initializer*> VarInitPair;

// comparison func for qsort
static int varinit_offset_cmp_func(const void* p1, const void* p2)
{
    VarDeclaration* v1 = ((VarInitPair*)p1)->first;
    VarDeclaration* v2 = ((VarInitPair*)p2)->first;
    if (v1->offset < v2->offset)
        return -1;
    else if (v1->offset > v2->offset)
        return 1;
    else
        return 0;
}

/*
this uses a simple algorithm to build the correct constant

(1) first sort the explicit initializers by offset... well, DMD doesn't :)

(2) if there is NO space before the next explicit initializeer, goto (9)
(3) find the next default initializer that fits before it, if NOT found goto (7)
(4) insert zero padding up to the next default initializer
(5) insert the next default initializer
(6) goto (2)

(7) insert zero padding up to the next explicit initializer

(9) insert the next explicit initializer
(10) goto (2)

(11) done

(next can be the end too)

*/

// return the next default initializer to use or null
static VarDeclaration* nextDefault(IrStruct* irstruct, size_t& idx, size_t pos, size_t offset)
{
    IrStruct::VarDeclVector& defaults = irstruct->defVars;
    size_t ndefaults = defaults.size();

    // for each valid index
    while(idx < ndefaults)
    {
        VarDeclaration* v = defaults[idx];

        // skip defaults before pos
        if (v->offset < pos)
        {
            idx++;
            continue;
        }

        // this var default fits
        if (v->offset >= pos && v->offset + v->type->size() <= offset)
            return v;

        // not usable
        break;
    }

    // not usable
    return NULL;
}

LLConstant* DtoConstStructInitializer(StructInitializer* si)
{
    Logger::println("DtoConstStructInitializer: %s", si->toChars());
    LOG_SCOPE;

    // get TypeStruct
    assert(si->ad);
    TypeStruct* ts = (TypeStruct*)si->ad->type;

    // force constant initialization of the symbol
    DtoForceConstInitDsymbol(si->ad);

    // get formal type
    const llvm::StructType* structtype = isaStruct(ts->ir.type->get());

#if 0
    // log it
    if (Logger::enabled())
        Logger::cout() << "llvm struct type: " << *structtype << '\n';
#endif

    // sanity check
    assert(si->value.dim > 0);
    assert(si->value.dim == si->vars.dim);

    // vector of final initializer constants
    std::vector<LLConstant*> inits;

    // get the ir struct
    IrStruct* irstruct = si->ad->ir.irStruct;

    // get default fields
    IrStruct::VarDeclVector& defaults = irstruct->defVars;
    size_t ndefaults = defaults.size();

    // make sure si->vars is sorted by offset
    std::vector<VarInitPair> vars;
    size_t nvars = si->vars.dim;
    vars.resize(nvars);

    // fill pair vector
    for (size_t i = 0; i < nvars; i++)
    {
        VarDeclaration* var = (VarDeclaration*)si->vars.data[i];
        Initializer* ini = (Initializer*)si->value.data[i];
        assert(var);
        assert(ini);
        vars[i] = std::make_pair(var, ini);
    }
    // sort it
    qsort(&vars[0], nvars, sizeof(VarInitPair), &varinit_offset_cmp_func);

    // check integrity
    // and do error checking, since the frontend does not verify static struct initializers
    size_t lastoffset = 0;
    size_t lastsize = 0;
    bool overlap = false;
    for (size_t i=0; i < nvars; i++)
    {
        // next explicit init var
        VarDeclaration* var = vars[i].first;
        Logger::println("var = %s : +%u", var->toChars(), var->offset);

        // I would have thought this to be a frontend check
        for (size_t j=i+1; j<nvars; j++)
        {
            if (j == i)
                continue;
            VarDeclaration* var2 = vars[j].first;
            if (var2->offset >= var->offset && var2->offset < var->offset + var->type->size())
            {
                fprintf(stdmsg, "Error: %s: initializer '%s' overlaps with '%s'\n", si->loc.toChars(), var->toChars(), var2->toChars());
                overlap = true;
            }
        }

        // update offsets
        lastoffset = var->offset;
        lastsize = var->type->size();
    }

    // error handling, report all overlaps before aborting
    if (overlap)
    {
        error("%s: overlapping union initializers", si->loc.toChars());
    }

    // go through each explicit initalizer, falling back to defaults or zeros when necessary
    lastoffset = 0;
    lastsize = 0;

    size_t j=0; // defaults

    for (size_t i=0; i < nvars; i++)
    {
        // get var and init
        VarDeclaration* var = vars[i].first;
        Initializer* ini = vars[i].second;

        size_t offset = var->offset;
        size_t size = var->type->size();

        // if there is space before the next explicit initializer
Lpadding:
        size_t pos = lastoffset+lastsize;
        if (offset > pos)
        {
            // find the the next default initializer that fits in this space
            VarDeclaration* nextdef = nextDefault(irstruct, j, lastoffset+lastsize, offset);

            // found
            if (nextdef)
            {
                // need zeros before the default
                if (nextdef->offset > pos)
                {
                    Logger::println("inserting %lu byte padding at %lu", nextdef->offset - pos, pos);
                    addZeros(inits, pos, nextdef->offset);
                }

                // do the default
                Logger::println("adding default field: %s : +%u", nextdef->toChars(), nextdef->offset);
                if (!nextdef->ir.irField->constInit)
                    nextdef->ir.irField->constInit = DtoConstInitializer(nextdef->loc, nextdef->type, nextdef->init);
                LLConstant* c = nextdef->ir.irField->constInit;
                inits.push_back(c);

                // update offsets
                lastoffset = nextdef->offset;
                lastsize = nextdef->type->size();

                // check if more defaults would fit
                goto Lpadding;
            }
            // not found, pad with zeros
            else
            {
                Logger::println("inserting %lu byte padding at %lu", offset - pos, pos);
                addZeros(inits, pos, offset);
                // offsets are updated by the explicit initializer
            }
        }

        // insert next explicit
        Logger::println("adding explicit field: %s : +%lu", var->toChars(), offset);
        LOG_SCOPE;
        LLConstant* c = DtoConstInitializer(var->loc, var->type, ini);
        inits.push_back(c);

        lastoffset = offset;
        lastsize = size;
    }

    // there might still be padding after the last one, make sure that is defaulted/zeroed as well
    size_t structsize = getTypePaddedSize(structtype);

    // if there is space before the next explicit initializer
    // FIXME: this should be handled in the loop above as well
Lpadding2:
    size_t pos = lastoffset+lastsize;
    if (structsize > pos)
    {
        // find the the next default initializer that fits in this space
        VarDeclaration* nextdef = nextDefault(irstruct, j, lastoffset+lastsize, structsize);

        // found
        if (nextdef)
        {
            // need zeros before the default
            if (nextdef->offset > pos)
            {
                Logger::println("inserting %lu byte padding at %lu", nextdef->offset - pos, pos);
                addZeros(inits, pos, nextdef->offset);
            }

            // do the default
            Logger::println("adding default field: %s : +%u", nextdef->toChars(), nextdef->offset);
            if (!nextdef->ir.irField->constInit)
                nextdef->ir.irField->constInit = DtoConstInitializer(nextdef->loc, nextdef->type, nextdef->init);
            LLConstant* c = nextdef->ir.irField->constInit;
            inits.push_back(c);

            // update offsets
            lastoffset = nextdef->offset;
            lastsize = nextdef->type->size();

            // check if more defaults would fit
            goto Lpadding2;
        }
        // not found, pad with zeros
        else
        {
            Logger::println("inserting %lu byte padding at %lu", structsize - pos, pos);
            addZeros(inits, pos, structsize);
            lastoffset = pos;
            lastsize = structsize - pos;
        }
    }

    assert(lastoffset+lastsize == structsize);

    // make the constant struct
    LLConstant* c = LLConstantStruct::get(inits, si->ad->ir.irStruct->packed);
    if (Logger::enabled())
    {
        Logger::cout() << "constant struct initializer: " << *c << '\n';
    }
    assert(getTypePaddedSize(c->getType()) == structsize);
    return c;
}

//////////////////////////////////////////////////////////////////////////////////////////

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
                    addZeros(values, lastoffset + lastsize, os);
                }

                // get field default init
                IrField* f = var->ir.irField;
                assert(f);
                if (!f->constInit)
                    f->constInit = DtoConstInitializer(var->loc, var->type, var->init);

                values.push_back(f->constInit);

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
            addZeros(values, lastoffset + lastsize, os);
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
                addZeros(values, lastoffset + lastsize, os);
            }

            // get field default init
            IrField* f = var->ir.irField;
            assert(f);
            if (!f->constInit)
                f->constInit = DtoConstInitializer(var->loc, var->type, var->init);

            values.push_back(f->constInit);

            lastoffset = os;
            lastsize = sz;
            //printf("2added default: %s : %lu (%lu)\n", var->toChars(), os, sz);
        }
    }

    // add any 0 padding needed at the end of the literal
    if (structsize > lastoffset+lastsize)
    {
        //printf("3added %lu zeros\n", structsize - lastoffset - lastsize);
        addZeros(values, lastoffset + lastsize, structsize);
    }

    return values;
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

void DtoResolveStruct(StructDeclaration* sd)
{
    // don't do anything if already been here
    if (sd->ir.resolved) return;
    // make sure above works :P
    sd->ir.resolved = true;

    // log what we're doing
    Logger::println("Resolving struct type: %s (%s)", sd->toChars(), sd->locToChars());
    LOG_SCOPE;

    // get the DMD TypeStruct
    TypeStruct* ts = (TypeStruct*)sd->type;

    // create the IrStruct
    IrStruct* irstruct = new IrStruct(sd);
    sd->ir.irStruct = irstruct;

    // create the type
    ts->ir.type = new LLPATypeHolder(llvm::OpaqueType::get());

    // handle forward declaration structs (opaques)
    // didn't even know D had those ...
    if (sd->sizeok != 1)
    {
        // nothing more to do
        return;
    }

    // make this struct current
    gIR->structs.push_back(irstruct);

    // get some info
    bool ispacked = (ts->alignsize() == 1);

    // set irstruct info
    irstruct->packed = ispacked;

    // methods, fields
    Array* arr = sd->members;
    for (int k=0; k < arr->dim; k++) {
        Dsymbol* s = (Dsymbol*)arr->data[k];
        s->toObjFile(0);
    }

    const LLType* ST = irstruct->build();

#if 0
    std::cout << sd->kind() << ' ' << sd->toPrettyChars() << " type: " << *ST << '\n';

    // add fields
    for (int k=0; k < fields->dim; k++)
    {
        VarDeclaration* v = (VarDeclaration*)fields->data[k];
        printf("  field: %s %s\n", v->type->toChars(), v->toChars());
        printf("    index: %u offset: %u\n", v->ir.irField->index, v->ir.irField->unionOffset);
    }

    unsigned llvmSize = (unsigned)getTypePaddedSize(ST);
    unsigned dmdSize = (unsigned)sd->type->size();
    printf("  llvm size: %u     dmd size: %u\n", llvmSize, dmdSize);
    assert(llvmSize == dmdSize);

#endif

    /*for (int k=0; k < sd->members->dim; k++) {
        Dsymbol* dsym = (Dsymbol*)(sd->members->data[k]);
        dsym->toObjFile();
    }*/

    Logger::println("doing struct fields");

    // refine abstract types for stuff like: struct S{S* next;}
    llvm::cast<llvm::OpaqueType>(ts->ir.type->get())->refineAbstractTypeTo(ST);
    ST = ts->ir.type->get();

    // name type
    if (sd->parent->isModule()) {
        gIR->module->addTypeName(sd->mangle(),ST);
    }

    gIR->structs.pop_back();

    gIR->declareList.push_back(sd);
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoDeclareStruct(StructDeclaration* sd)
{
    if (sd->ir.declared) return;
    sd->ir.declared = true;

    Logger::println("DtoDeclareStruct(%s): %s", sd->toChars(), sd->loc.toChars());
    LOG_SCOPE;

    TypeStruct* ts = (TypeStruct*)sd->type->toBasetype();

    std::string initname("_D");
    initname.append(sd->mangle());
    initname.append("6__initZ");

    llvm::GlobalValue::LinkageTypes _linkage = DtoExternalLinkage(sd);
    llvm::GlobalVariable* initvar = new llvm::GlobalVariable(sd->ir.irStruct->initOpaque.get(), true, _linkage, NULL, initname, gIR->module);
    sd->ir.irStruct->init = initvar;

    gIR->constInitList.push_back(sd);
    if (mustDefineSymbol(sd))
        gIR->defineList.push_back(sd);
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoConstInitStruct(StructDeclaration* sd)
{
    if (sd->ir.initialized) return;
    sd->ir.initialized = true;

    Logger::println("DtoConstInitStruct(%s): %s", sd->toChars(), sd->loc.toChars());
    LOG_SCOPE;

    IrStruct* irstruct = sd->ir.irStruct;
    gIR->structs.push_back(irstruct);

    const llvm::StructType* structtype = isaStruct(sd->type->ir.type->get());

    // always generate the constant initalizer
    assert(!irstruct->constInit);
    if (sd->zeroInit)
    {
        Logger::println("Zero initialized");
        irstruct->constInit = llvm::ConstantAggregateZero::get(structtype);
    }
    else
    {
        Logger::println("Not zero initialized");

        LLConstant* c = irstruct->buildDefaultConstInit();
        irstruct->constInit = c;
    }

    // refine __initZ global type to the one of the initializer
    llvm::cast<llvm::OpaqueType>(irstruct->initOpaque.get())->refineAbstractTypeTo(irstruct->constInit->getType());

    gIR->structs.pop_back();

    // emit typeinfo
    if (sd->getModule() == gIR->dmodule && sd->llvmInternal != LLVMno_typeinfo)
        DtoTypeInfoOf(sd->type, false);
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoDefineStruct(StructDeclaration* sd)
{
    if (sd->ir.defined) return;
    sd->ir.defined = true;

    Logger::println("DtoDefineStruct(%s): %s", sd->toChars(), sd->loc.toChars());
    LOG_SCOPE;

    assert(sd->type->ty == Tstruct);
    TypeStruct* ts = (TypeStruct*)sd->type;
    sd->ir.irStruct->init->setInitializer(sd->ir.irStruct->constInit);

    sd->ir.DModule = gIR->dmodule;
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
