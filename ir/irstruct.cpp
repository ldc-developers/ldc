#include "gen/llvm.h"

#include "mtype.h"
#include "aggregate.h"
#include "declaration.h"
#include "init.h"

#include "ir/irstruct.h"
#include "gen/irstate.h"
#include "gen/tollvm.h"
#include "gen/logger.h"
#include "gen/llvmhelpers.h"

IrInterface::IrInterface(BaseClass* b)
:   vtblInitTy(llvm::OpaqueType::get())
{
    base = b;
    decl = b->base;
    vtblInit = NULL;
    vtbl = NULL;
    infoTy = NULL;
    infoInit = NULL;
    info = NULL;

    index = 0;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

IrStruct::IrStruct(AggregateDeclaration* aggr)
:   initOpaque(llvm::OpaqueType::get()),
    classInfoOpaque(llvm::OpaqueType::get()),
    vtblTy(llvm::OpaqueType::get()),
    vtblInitTy(llvm::OpaqueType::get())
{
    aggrdecl = aggr;
    defaultFound = false;
    anon = NULL;
    index = 0;

    type = aggr->type;
    defined = false;
    constinited = false;

    interfaceInfos = NULL;
    vtbl = NULL;
    constVtbl = NULL;

    init = NULL;
    constInit = NULL;

    classInfo = NULL;
    constClassInfo = NULL;
    classInfoDeclared = false;
    classInfoDefined = false;

    packed = false;

    dwarfComposite = NULL;
}

IrStruct::~IrStruct()
{
}

//////////////////////////////////////////

void IrStruct::pushAnon(bool isunion)
{
    anon = new Anon(isunion, anon);
}

//////////////////////////////////////////

void IrStruct::popAnon()
{
    assert(anon);

    const LLType* BT;

    // get the anon type
    if (anon->isunion)
    {
        // get biggest type in block
        const LLType* biggest = getBiggestType(&anon->types[0], anon->types.size());
        std::vector<const LLType*> vec(1, biggest);
        BT = LLStructType::get(vec, aggrdecl->ir.irStruct->packed);
    }
    else
    {
        // build a struct from the types
        BT = LLStructType::get(anon->types, aggrdecl->ir.irStruct->packed);
    }

    // pop anon
    Anon* tmp = anon;
    anon = anon->parent;
    delete tmp;

    // is there a parent anon?
    if (anon)
    {
        // make sure type gets pushed in the anon, not the main
        anon->types.push_back(BT);
        // index is only manipulated at the top level, anons use raw offsets
    }
    // no parent anon, finally add to aggrdecl
    else
    {
        types.push_back(BT);
        // only advance to next position if main is not a union
        if (!aggrdecl->isUnionDeclaration())
        {
            index++;
        }
    }
}

//////////////////////////////////////////

void IrStruct::addVar(VarDeclaration * var)
{
    TypeVector* tvec = &types;
    if (anon)
    {
        // make sure type gets pushed in the anon, not the main
        tvec = &anon->types;

        // set but don't advance index
        var->ir.irField->index = index;

        // set offset in bytes from start of anon block
        var->ir.irField->unionOffset = var->offset - var->offset2;
    }
    else if (aggrdecl->isUnionDeclaration())
    {
        // set but don't advance index
        var->ir.irField->index = index;
    }
    else
    {
        // set and advance index
        var->ir.irField->index = index++;
    }

    // add type
    tvec->push_back(DtoType(var->type));

    // add var
    varDecls.push_back(var);
}

//////////////////////////////////////////

const LLType* IrStruct::build()
{
    // if types is empty, add a byte
    if (types.empty())
    {
        types.push_back(LLType::Int8Ty);
    }

    // union type
    if (aggrdecl->isUnionDeclaration())
    {
        const LLType* biggest = getBiggestType(&types[0], types.size());
        std::vector<const LLType*> vec(1, biggest);
        return LLStructType::get(vec, aggrdecl->ir.irStruct->packed);
    }
    // struct/class type
    else
    {
        return LLStructType::get(types, aggrdecl->ir.irStruct->packed);
    }
}

void addZeros(std::vector<const llvm::Type*>& inits, size_t pos, size_t offset)
{
    assert(offset > pos);
    size_t diff = offset - pos;

    size_t sz;

    do
    {
        if (pos%8 == 0 && diff >= 8)
            sz = 8;
        else if (pos%4 == 0 && diff >= 4)
            sz = 4;
        else if (pos%2 == 0 && diff >= 2)
            sz = 2;
        else // if (pos % 1 == 0)
            sz = 1;
        inits.push_back(LLIntegerType::get(sz*8));
        pos += sz;
        diff -= sz;
    } while (pos < offset);

    assert(pos == offset);
}

void addZeros(std::vector<llvm::Constant*>& inits, size_t pos, size_t offset)
{
    assert(offset > pos);
    size_t diff = offset - pos;

    size_t sz;

    do
    {
        if (pos%8 == 0 && diff >= 8)
            sz = 8;
        else if (pos%4 == 0 && diff >= 4)
            sz = 4;
        else if (pos%2 == 0 && diff >= 2)
            sz = 2;
        else // if (pos % 1 == 0)
            sz = 1;
        inits.push_back(LLConstant::getNullValue(LLIntegerType::get(sz*8)));
        pos += sz;
        diff -= sz;
    } while (pos < offset);

    assert(pos == offset);
}

// FIXME: body is exact copy of above
void addZeros(std::vector<llvm::Value*>& inits, size_t pos, size_t offset)
{
    assert(offset > pos);
    size_t diff = offset - pos;

    size_t sz;

    do
    {
        if (pos%8 == 0 && diff >= 8)
            sz = 8;
        else if (pos%4 == 0 && diff >= 4)
            sz = 4;
        else if (pos%2 == 0 && diff >= 2)
            sz = 2;
        else // if (pos % 1 == 0)
            sz = 1;
        inits.push_back(LLConstant::getNullValue(LLIntegerType::get(sz*8)));
        pos += sz;
        diff -= sz;
    } while (pos < offset);

    assert(pos == offset);
}

void IrStruct::buildDefaultConstInit(std::vector<llvm::Constant*>& inits)
{
    assert(!defaultFound);
    defaultFound = true;

    const llvm::StructType* structtype = isaStruct(aggrdecl->type->ir.type->get());
    Logger::cout() << "struct type: " << *structtype << '\n';

    size_t lastoffset = 0;
    size_t lastsize = 0;

    {
        Logger::println("Find the default fields");
        LOG_SCOPE;

        // go through all vars and find the ones that contribute to the default
        size_t nvars = varDecls.size();
        for (size_t i=0; i<nvars; i++)
        {
            VarDeclaration* var = varDecls[i];

            Logger::println("field %s %s = %s : +%u", var->type->toChars(), var->toChars(), var->init ? var->init->toChars() : var->type->defaultInit(var->loc)->toChars(), var->offset);

            // only add vars that don't overlap
            size_t offset = var->offset;
            size_t size = var->type->size();
            if (offset >= lastoffset+lastsize)
            {
                Logger::println("  added");
                lastoffset = offset;
                lastsize = size;
                defVars.push_back(var);
            }
        }
    }

    {
        Logger::println("Build the default initializer");
        LOG_SCOPE;

        lastoffset = 0;
        lastsize = 0;

        // go through the default vars and build the default constant initializer
        // adding zeros along the way to live up to alignment expectations
        size_t nvars = defVars.size();
        for (size_t i=0; i<nvars; i++)
        {
            VarDeclaration* var = defVars[i];

            Logger::println("field %s %s = %s : +%u", var->type->toChars(), var->toChars(), var->init ? var->init->toChars() : var->type->defaultInit(var->loc)->toChars(), var->offset);

            // get offset and size
            size_t offset = var->offset;
            size_t size = var->type->size();

            // is there space in between last last offset and this one?
            // if so, fill it with zeros
            if (offset > lastoffset+lastsize)
            {
                size_t pos = lastoffset + lastsize;
                addZeros(inits, pos, offset);
            }

            // add the field
            // lazily default initialize
            if (!var->ir.irField->constInit)
                var->ir.irField->constInit = DtoConstInitializer(var->loc, var->type, var->init);
            inits.push_back(var->ir.irField->constInit);

            lastoffset = offset;
            lastsize = var->type->size();
        }

        // there might still be padding after the last one, make sure that is zeroed as well
        // is there space in between last last offset and this one?
        size_t structsize = getABITypeSize(structtype);

        if (structsize > lastoffset+lastsize)
        {
            size_t pos = lastoffset + lastsize;
            addZeros(inits, pos, structsize);
        }
    }
}

LLConstant* IrStruct::buildDefaultConstInit()
{
    // doesn't work for classes, they add stuff before and maybe after data fields
    assert(!aggrdecl->isClassDeclaration());

    // initializer llvm constant list
    std::vector<LLConstant*> inits;

    // just start with an empty list
    buildDefaultConstInit(inits);

    // build the constant
    // note that the type matches the initializer, not the aggregate in cases with unions
    LLConstant* c = LLConstantStruct::get(inits, aggrdecl->ir.irStruct->packed);
    Logger::cout() << "llvm constant: " << *c << '\n';
//     assert(0);
    return c;
}
