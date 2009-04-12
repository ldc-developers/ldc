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
#include "gen/classes.h"
#include "gen/structs.h"
#include "gen/functions.h"
#include "gen/runtime.h"
#include "gen/dvalue.h"
#include "gen/nested.h"

#include "ir/irstruct.h"

//////////////////////////////////////////////////////////////////////////////////////////

// adds the base interfaces of b and the given iri to IrStruct's interfaceMap
static void add_base_interfaces(IrStruct* to, IrInterface* iri, BaseClass* b)
{
    for (unsigned j = 0; j < b->baseInterfaces_dim; j++)
    {
        BaseClass *bc = &b->baseInterfaces[j];
        // add to map
        if (to->interfaceMap.find(bc->base) == to->interfaceMap.end())
        {
            to->interfaceMap.insert(std::make_pair(bc->base, iri));
        }
        // add base interfaces
        add_base_interfaces(to, iri, bc);
    }
}

// adds interface b to target, if newinstance != 0, then target must provide all
// functions required to implement b (it reimplements b)
static void add_interface(ClassDeclaration* target, BaseClass* b, int newinstance)
{
    Logger::println("adding interface: %s", b->base->toChars());
    LOG_SCOPE;

    InterfaceDeclaration* inter = b->base->isInterfaceDeclaration();
    DtoResolveClass(inter);

    assert(inter);
    IrStruct* irstruct = target->ir.irStruct;
    assert(irstruct);

    // add interface to map/list
    // if it's already inserted in the map, it's because another interface has it as baseclass
    // but if it appears here, it's because we're reimplementing it, so we overwrite the IrInterface entry
    IrInterface* iri;
    bool overwrite = false;
    if (irstruct->interfaceMap.find(inter) != irstruct->interfaceMap.end())
    {
        overwrite = true;
    }

    iri = new IrInterface(b);
    // add to map
    if (overwrite)
        irstruct->interfaceMap[b->base] = iri;
    else
        irstruct->interfaceMap.insert(std::make_pair(b->base, iri));

    // add to ordered list
    irstruct->interfaceVec.push_back(iri);

    // add to classinfo interfaces
    if (newinstance)
        irstruct->classInfoInterfaces.push_back(iri);

    // recursively assign this iri to all base interfaces
    add_base_interfaces(irstruct, iri, b);

    // build the interface vtable
    b->fillVtbl(target, &iri->vtblDecls, newinstance);

    // add the vtable type
    assert(inter->type->ir.type);
    irstruct->types.push_back( inter->type->ir.type->get() );
    // set and increment index
    iri->index = irstruct->index++;
}

//////////////////////////////////////////////////////////////////////////////////////////

static void add_class_data(ClassDeclaration* target, ClassDeclaration* cd)
{
    Logger::println("Adding data from class: %s", cd->toChars());
    LOG_SCOPE;

    // recurse into baseClasses
    if (cd->baseClass)
    {
        add_class_data(target, cd->baseClass);
        //offset = baseClass->structsize;
    }

    // add members
    Array* arr = cd->members;
    for (int k=0; k < arr->dim; k++) {
        Dsymbol* s = (Dsymbol*)arr->data[k];
        s->codegen(Type::sir);
    }

    // add interfaces
    if (cd->vtblInterfaces)
    {
        Logger::println("num vtbl interfaces: %u", cd->vtblInterfaces->dim);
        for (int i = 0; i < cd->vtblInterfaces->dim; i++)
        {
            BaseClass *b = (BaseClass *)cd->vtblInterfaces->data[i];
            assert(b);
            // create new instances only for explicitly derived interfaces
            add_interface(target, b, (cd == target));
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

static void DtoDeclareInterface(InterfaceDeclaration* cd);
static void DtoConstInitInterface(InterfaceDeclaration* cd);
static void DtoDefineInterface(InterfaceDeclaration* cd);

static void DtoResolveInterface(InterfaceDeclaration* cd)
{
    if (cd->ir.resolved) return;
    cd->ir.resolved = true;

    Logger::println("DtoResolveInterface(%s): %s", cd->toPrettyChars(), cd->loc.toChars());
    LOG_SCOPE;

    // get the TypeClass
    assert(cd->type->ty == Tclass);
    TypeClass* ts = (TypeClass*)cd->type;

    // create the IrStruct, we need somewhere to store the classInfo
    assert(!cd->ir.irStruct);
    IrStruct* irstruct = new IrStruct(cd);
    cd->ir.irStruct = irstruct;

    // create the type
    const LLType* t = LLArrayType::get(getVoidPtrType(), cd->vtbl.dim);
    assert(!ts->ir.type);
    ts->ir.type = new LLPATypeHolder(getPtrToType(t));

    // ... and ClassInfo
    std::string varname("_D");
    varname.append(cd->mangle());
    varname.append("11__InterfaceZ");

    // create global
    irstruct->classInfo = new llvm::GlobalVariable(irstruct->classInfoOpaque.get(), false, DtoLinkage(cd), NULL, varname, gIR->module);

    // handle base interfaces
    if (cd->baseclasses.dim)
    {
        Logger::println("num baseclasses: %u", cd->baseclasses.dim);
        LOG_SCOPE;

        for (int i=0; i<cd->baseclasses.dim; i++)
        {
            BaseClass* bc = (BaseClass*)cd->baseclasses.data[i];
            Logger::println("baseclass %d: %s", i, bc->base->toChars());

            InterfaceDeclaration* id = bc->base->isInterfaceDeclaration();
            assert(id);

            DtoResolveInterface(id);

            // add to interfaceInfos
            IrInterface* iri = new IrInterface(bc);
            irstruct->interfaceVec.push_back(iri);
            irstruct->classInfoInterfaces.push_back(iri);
        }
    }

    // request declaration
    DtoDeclareInterface(cd);

    // handle members
    // like "nested" interfaces
    Array* arr = cd->members;
    for (int k=0; k < arr->dim; k++) {
        Dsymbol* s = (Dsymbol*)arr->data[k];
        s->codegen(Type::sir);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

// FIXME: this needs to be cleaned up

void DtoResolveClass(ClassDeclaration* cd)
{
    if (InterfaceDeclaration* id = cd->isInterfaceDeclaration())
    {
        DtoResolveInterface(id);
        return;
    }

    // make sure the base classes are processed first
    if (cd->baseClass)
        cd->baseClass->codegen(Type::sir);

    if (cd->ir.resolved) return;
    cd->ir.resolved = true;

    Logger::println("DtoResolveClass(%s): %s", cd->toPrettyChars(), cd->loc.toChars());
    LOG_SCOPE;

    // get the TypeClass
    assert(cd->type->ty == Tclass);
    TypeClass* ts = (TypeClass*)cd->type;

    // create the IrStruct
    assert(!cd->ir.irStruct);
    IrStruct* irstruct = new IrStruct(cd);
    cd->ir.irStruct = irstruct;

    // create the type
    ts->ir.type = new LLPATypeHolder(llvm::OpaqueType::get());

    // if it just a forward declaration?
    if (cd->sizeok != 1)
    {
        // just name the type
        gIR->module->addTypeName(cd->mangle(), ts->ir.type->get());
        return;
    }

    // create the symbols we're going to need
    // avoids chicken egg problems
    llvm::GlobalValue::LinkageTypes _linkage = DtoLinkage(cd);

    // there is always a vtbl symbol
    std::string varname("_D");
    varname.append(cd->mangle());
    varname.append("6__vtblZ");
    irstruct->vtbl = new llvm::GlobalVariable(irstruct->vtblInitTy.get(), true, _linkage, NULL, varname, gIR->module);

    // ... and initZ
    varname = "_D";
    varname.append(cd->mangle());
    varname.append("6__initZ");
    irstruct->init = new llvm::GlobalVariable(irstruct->initOpaque.get(), true, _linkage, NULL, varname, gIR->module);

    // ... and ClassInfo
    varname = "_D";
    varname.append(cd->mangle());
    varname.append("7__ClassZ");

    // create global
    irstruct->classInfo = new llvm::GlobalVariable(irstruct->classInfoOpaque.get(), false, _linkage, NULL, varname, gIR->module);

    // push state
    gIR->structs.push_back(irstruct);

    // add vtable
    irstruct->types.push_back(getPtrToType(irstruct->vtblTy.get()));
    irstruct->index++;

    // add monitor
    irstruct->types.push_back(getVoidPtrType());
    irstruct->index++;

    // add class data fields and interface vtables recursively
    add_class_data(cd, cd);

    // check if errors occured while building interface vtables
    if (global.errors)
        fatal();

    // create type
    assert(irstruct->index == irstruct->types.size());
    const LLType* structtype = irstruct->build();

    // refine abstract types for stuff like: class C {C next;}
    llvm::PATypeHolder* spa = ts->ir.type;
    llvm::cast<llvm::OpaqueType>(spa->get())->refineAbstractTypeTo(structtype);
    structtype = isaStruct(spa->get());
    assert(structtype);

    // name the type
    gIR->module->addTypeName(cd->mangle(), ts->ir.type->get());

    // refine vtable type

    // void*[vtbl.dim]
    llvm::cast<llvm::OpaqueType>(irstruct->vtblTy.get())->refineAbstractTypeTo(LLArrayType::get(getVoidPtrType(), cd->vtbl.dim));

    // log
//     if (Logger::enabled())
//         Logger::cout() << "final class type: " << *ts->ir.type->get() << '\n';

    // pop state
    gIR->structs.pop_back();

    // queue declare
    DtoDeclareClass(cd);
}

//////////////////////////////////////////////////////////////////////////////////////////

static void DtoDeclareInterface(InterfaceDeclaration* cd)
{
    DtoResolveInterface(cd);

    if (cd->ir.declared) return;
    cd->ir.declared = true;

    Logger::println("DtoDeclareInterface(%s): %s", cd->toPrettyChars(), cd->locToChars());
    LOG_SCOPE;

    assert(cd->ir.irStruct);
    IrStruct* irstruct = cd->ir.irStruct;

    // get interface info type
    const llvm::StructType* infoTy = DtoInterfaceInfoType();

    // interface info array
    if (!irstruct->interfaceVec.empty()) {
        // symbol name
        std::string nam = "_D";
        nam.append(cd->mangle());
        nam.append("16__interfaceInfosZ");

        llvm::GlobalValue::LinkageTypes linkage = DtoLinkage(cd);

        // resolve array type
        const llvm::ArrayType* arrTy = llvm::ArrayType::get(infoTy, irstruct->interfaceVec.size());
        // declare global
        irstruct->interfaceInfos = new llvm::GlobalVariable(arrTy, true, linkage, NULL, nam, gIR->module);

        // do each interface info
        unsigned idx = 0;
        size_t n = irstruct->interfaceVec.size();
        for (size_t i=0; i < n; i++)
        {
            IrInterface* iri = irstruct->interfaceVec[i];
            ClassDeclaration* id = iri->decl;

            // always create interfaceinfos
            LLConstant* idxs[2] = {DtoConstUint(0), DtoConstUint(idx)};
            iri->info = llvm::ConstantExpr::getGetElementPtr(irstruct->interfaceInfos, idxs, 2);
            idx++;
        }
    }

    // declare the classinfo
    DtoDeclareClassInfo(cd);

    // request const init
    DtoConstInitInterface(cd);

    // emit typeinfo and request definition
    if (mustDefineSymbol(cd))
    {
        DtoDefineInterface(cd);
        DtoTypeInfoOf(cd->type, false);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

// FIXME: this needs to be cleaned up

void DtoDeclareClass(ClassDeclaration* cd)
{
    if (InterfaceDeclaration* id = cd->isInterfaceDeclaration())
    {
        DtoDeclareInterface(id);
        return;
    }

    DtoResolveClass(cd);

    if (cd->ir.declared) return;
    cd->ir.declared = true;

    Logger::println("DtoDeclareClass(%s): %s", cd->toPrettyChars(), cd->locToChars());
    LOG_SCOPE;

    //printf("declare class: %s\n", cd->toPrettyChars());

    assert(cd->type->ty == Tclass);
    TypeClass* ts = (TypeClass*)cd->type;

    assert(cd->ir.irStruct);
    IrStruct* irstruct = cd->ir.irStruct;

    gIR->structs.push_back(irstruct);

    bool needs_definition = false;
    if (mustDefineSymbol(cd)) {
        needs_definition = true;
    }

    llvm::GlobalValue::LinkageTypes _linkage = DtoLinkage(cd);

    // get interface info type
    const llvm::StructType* infoTy = DtoInterfaceInfoType();

    // interface info array
    if (!irstruct->interfaceVec.empty()) {
        // symbol name
        std::string nam = "_D";
        nam.append(cd->mangle());
        nam.append("16__interfaceInfosZ");
        // resolve array type
        const llvm::ArrayType* arrTy = llvm::ArrayType::get(infoTy, irstruct->interfaceVec.size());
        // declare global
        irstruct->interfaceInfos = new llvm::GlobalVariable(arrTy, true, _linkage, NULL, nam, gIR->module);
    }

    // DMD gives abstract classes a full ClassInfo, so we do it as well

    // interface vtables
    unsigned idx = 0;
    for (IrStruct::InterfaceVectorIter i=irstruct->interfaceVec.begin(); i!=irstruct->interfaceVec.end(); ++i)
    {
        IrInterface* iri = *i;
        ClassDeclaration* id = iri->decl;

        std::string nam("_D");
        nam.append(cd->mangle());
        nam.append("11__interface");
        nam.append(id->mangle());
        nam.append("6__vtblZ");

        iri->vtbl = new llvm::GlobalVariable(iri->vtblInitTy.get(), true, _linkage, 0, nam, gIR->module);

        // always set the interface info as it's need as the first vtbl entry
        LLConstant* idxs[2] = {DtoConstUint(0), DtoConstUint(idx)};
        iri->info = llvm::ConstantExpr::getGetElementPtr(irstruct->interfaceInfos, idxs, 2);
        idx++;
    }

    gIR->structs.pop_back();

    // request const init
    DtoConstInitClass(cd);

    // define ? (set initializers)
    if (needs_definition)
        DtoDefineClass(cd);

    // classinfo
    DtoDeclareClassInfo(cd);

    // do typeinfo ?
    if (needs_definition)
        DtoTypeInfoOf(cd->type, false);
}

//////////////////////////////////////////////////////////////////////////////

// adds data fields and interface vtables to the constant initializer of class cd
static size_t init_class_initializer(std::vector<LLConstant*>& inits, ClassDeclaration* target, ClassDeclaration* cd, size_t offsetbegin)
{
    // first do baseclasses
    if (cd->baseClass)
    {
        offsetbegin = init_class_initializer(inits, target, cd->baseClass, offsetbegin);
    }

    Logger::println("adding data of %s to %s starting at %lu", cd->toChars(), target->toChars(), offsetbegin);
    LOG_SCOPE;

    // add default fields
    VarDeclaration** fields = (VarDeclaration**)cd->fields.data;
    size_t nfields = cd->fields.dim;

    std::vector<VarDeclaration*> defVars;
    defVars.reserve(nfields);

    size_t lastoffset = offsetbegin;
    size_t lastsize = 0;

    // find fields that contribute to default
    for (size_t i=0; i<nfields; i++)
    {
        VarDeclaration* var = fields[i];
        // only add vars that don't overlap
        size_t offset = var->offset;
        size_t size = var->type->size();
        if (offset >= lastoffset+lastsize)
        {
            Logger::println("  added %s", var->toChars());
            lastoffset = offset;
            lastsize = size;
            defVars.push_back(var);
        }
        else
        {
            Logger::println("  skipped %s at offset %u, current pos is %lu", var->toChars(), var->offset, lastoffset+lastsize);
        }
    }

    // reset offsets, we're going from beginning again
    lastoffset = offsetbegin;
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
        // and build its constant initializer lazily
        if (!var->ir.irField->constInit)
            var->ir.irField->constInit = DtoConstInitializer(var->loc, var->type, var->init);
        inits.push_back(var->ir.irField->constInit);

        lastoffset = offset;
        lastsize = var->type->size();
    }

    // if it's a class, and it implements interfaces, add the vtables - as found in the target class!
    IrStruct* irstruct = target->ir.irStruct;

    size_t nvtbls = cd->vtblInterfaces->dim;
    for(size_t i=0; i<nvtbls; i++)
    {
        BaseClass* bc = (BaseClass*)cd->vtblInterfaces->data[i];
        IrStruct::InterfaceMap::iterator iter = irstruct->interfaceMap.find(bc->base);
        assert(iter != irstruct->interfaceMap.end());

        IrInterface* iri = iter->second;
        if (iri->vtbl)
            inits.push_back(iri->vtbl);
        else // abstract impl
            inits.push_back(getNullPtr(getVoidPtrType()));

        lastoffset += lastsize;
        lastsize = PTRSIZE;
    }

    // return next offset
    return lastoffset + lastsize;
}

//////////////////////////////////////////////////////////////////////////////

// build the vtable initializer for class cd
static void init_class_vtbl_initializer(ClassDeclaration* cd)
{
    cd->codegen(Type::sir);

    // generate vtable initializer
    std::vector<LLConstant*> sinits(cd->vtbl.dim, NULL);

    IrStruct* irstruct = cd->ir.irStruct;

    assert(cd->vtbl.dim > 1);

    DtoDeclareClassInfo(cd);

    // first entry always classinfo
    assert(irstruct->classInfo);
    sinits[0] = DtoBitCast(irstruct->classInfo, DtoType(ClassDeclaration::classinfo->type));

    // add virtual functions
    for (int k=1; k < cd->vtbl.dim; k++)
    {
        Dsymbol* dsym = (Dsymbol*)cd->vtbl.data[k];
        assert(dsym);

//         Logger::println("vtbl[%d] = %s", k, dsym->toChars());

        FuncDeclaration* fd = dsym->isFuncDeclaration();
        assert(fd);

        // if function is abstract,
        // or class is abstract, and func has no body,
        // emit a null vtbl entry
        if (fd->isAbstract() || (cd->isAbstract() && !fd->fbody))
        {
            sinits[k] = getNullPtr(getVoidPtrType());
        }
        else
        {
            fd->codegen(Type::sir);
            Logger::println("F = %s", fd->toPrettyChars());
            assert(fd->ir.irFunc);
            assert(fd->ir.irFunc->func);
            sinits[k] = fd->ir.irFunc->func;
        }

//         if (Logger::enabled())
//             Logger::cout() << "vtbl[" << k << "] = " << *sinits[k] << std::endl;
    }

    irstruct->constVtbl = LLConstantStruct::get(sinits);

    // refine type
    llvm::cast<llvm::OpaqueType>(irstruct->vtblInitTy.get())->refineAbstractTypeTo(irstruct->constVtbl->getType());

//     if (Logger::enabled())
//         Logger::cout() << "vtbl initializer: " << *irstruct->constVtbl << std::endl;
}

//////////////////////////////////////////////////////////////////////////////

static void init_class_interface_vtbl_initializers(ClassDeclaration* cd)
{
    IrStruct* irstruct = cd->ir.irStruct;

    // don't do anything if list is empty
    if (irstruct->interfaceVec.empty())
        return;

    std::vector<LLConstant*> inits;
    std::vector<LLConstant*> infoInits(3);

    // go through each interface
    size_t ninter = irstruct->interfaceVec.size();
    for (size_t i=0; i<ninter; i++)
    {
        IrInterface* iri = irstruct->interfaceVec[i];
        Logger::println("interface %s", iri->decl->toChars());

        // build vtable intializer for this interface implementation
        Array& arr = iri->vtblDecls;
        size_t narr = arr.dim;

        if (narr > 0)
        {
            inits.resize(narr, NULL);

            // first is always the interface info
            assert(iri->info);
            inits[0] = iri->info;

            // build vtable
            for (size_t j=1; j < narr; j++)
            {
                Dsymbol* dsym = (Dsymbol*)arr.data[j];
                if (!dsym)
                {
                    inits[j] = getNullPtr(getVoidPtrType());
                    continue;
                }

                //Logger::println("ivtbl[%d] = %s", j, dsym->toChars());

                // must all be functions
                FuncDeclaration* fd = dsym->isFuncDeclaration();
                assert(fd);

                if (fd->isAbstract())
                    inits[j] = getNullPtr(getVoidPtrType());
                else
                {
                    fd->codegen(Type::sir);

                    assert(fd->ir.irFunc->func);
                    inits[j] = fd->ir.irFunc->func;
                }

                //if (Logger::enabled())
                //    Logger::cout() << "ivtbl[" << j << "] = " << *inits[j] << std::endl;
            }

            // build the constant
            iri->vtblInit = LLConstantStruct::get(inits);
        }

        // build the interface info for ClassInfo
        // generate interface info initializer

        iri->decl->codegen(Type::sir);

        // classinfo
        IrStruct* iris = iri->decl->ir.irStruct;
        assert(iris);
        assert(iris->classInfo);
        infoInits[0] = DtoBitCast(iris->classInfo, DtoType(ClassDeclaration::classinfo->type));

        // vtbl
        LLConstant* c;
        if (iri->vtbl)
            c = llvm::ConstantExpr::getBitCast(iri->vtbl, getPtrToType(getVoidPtrType()));
        else
            c = getNullPtr(getPtrToType(getVoidPtrType()));
        infoInits[1] = DtoConstSlice(DtoConstSize_t(narr), c);

        // offset
        size_t ioff;
        if (iri->index == 0)
            ioff = 0;
        else
            ioff = gTargetData->getStructLayout(isaStruct(cd->type->ir.type->get()))->getElementOffset(iri->index);

        Logger::println("DMD interface offset: %d, LLVM interface offset: %lu", iri->base->offset, ioff);
        assert(iri->base->offset == ioff);
        infoInits[2] = DtoConstUint(ioff);

        // create interface info initializer constant
        iri->infoInit = llvm::cast<llvm::ConstantStruct>(llvm::ConstantStruct::get(infoInits));
    }
}

//////////////////////////////////////////////////////////////////////////////

static void DtoConstInitInterface(InterfaceDeclaration* cd)
{
    DtoDeclareInterface(cd);

    if (cd->ir.initialized) return;
    cd->ir.initialized = true;

    Logger::println("DtoConstInitClass(%s): %s", cd->toPrettyChars(), cd->loc.toChars());
    LOG_SCOPE;

    init_class_interface_vtbl_initializers(cd);
}

//////////////////////////////////////////////////////////////////////////////

void DtoConstInitClass(ClassDeclaration* cd)
{
    if (InterfaceDeclaration* it = cd->isInterfaceDeclaration())
    {
        DtoConstInitInterface(it);
        return;
    }

    DtoDeclareClass(cd);

    if (cd->ir.initialized) return;
    cd->ir.initialized = true;

    Logger::println("DtoConstInitClass(%s): %s", cd->toPrettyChars(), cd->loc.toChars());
    LOG_SCOPE;

    // get IrStruct
    IrStruct* irstruct = cd->ir.irStruct;
    gIR->structs.push_back(irstruct);

    // get types
    TypeClass* tc = (TypeClass*)cd->type;
    const llvm::StructType* structtype = isaStruct(tc->ir.type->get());
    assert(structtype);
    const llvm::ArrayType* vtbltype = isaArray(irstruct->vtblTy.get());
    assert(vtbltype);

    // build initializer list
    std::vector<LLConstant*> inits;
    inits.reserve(irstruct->varDecls.size());

    // vtable is always first
    assert(irstruct->vtbl != 0);
    inits.push_back(irstruct->vtbl);

    // then comes monitor
    inits.push_back(LLConstant::getNullValue(getVoidPtrType()));

    // recursively do data and interface vtables
    init_class_initializer(inits, cd, cd, 2 * PTRSIZE);

    // build vtable initializer
    init_class_vtbl_initializer(cd);

    // build interface vtables
    init_class_interface_vtbl_initializers(cd);

    // build constant from inits
    assert(!irstruct->constInit);
    irstruct->constInit = LLConstantStruct::get(inits); // classes are never packed

    // refine __initZ global type to the one of the initializer
    llvm::cast<llvm::OpaqueType>(irstruct->initOpaque.get())->refineAbstractTypeTo(irstruct->constInit->getType());

    // build initializers for static member variables
    size_t n = irstruct->staticVars.size();
    for (size_t i = 0; i < n; ++i)
    {
        DtoConstInitGlobal(irstruct->staticVars[i]);
    }
    // This is all we use it for. Clear the memory!
    irstruct->staticVars.clear();

//     if (Logger::enabled())
//     {
//         Logger::cout() << "class " << cd->toChars() << std::endl;
//         Logger::cout() << "type " << *cd->type->ir.type->get() << std::endl;
//         Logger::cout() << "initializer " << *irstruct->constInit << std::endl;
//     }

    gIR->structs.pop_back();
}

//////////////////////////////////////////////////////////////////////////////////////////

static void DefineInterfaceInfos(IrStruct* irstruct)
{
    // always do interface info array when possible
    std::vector<LLConstant*> infoInits;

    size_t n = irstruct->interfaceVec.size();
    infoInits.reserve(n);

    for (size_t i=0; i < n; i++)
    {
        IrInterface* iri = irstruct->interfaceVec[i];
        assert(iri->infoInit);
        infoInits.push_back(iri->infoInit);
    }

    // set initializer
    if (!infoInits.empty())
    {
        const LLArrayType* arrty = LLArrayType::get(infoInits[0]->getType(), infoInits.size());
        LLConstant* arrInit = llvm::ConstantArray::get(arrty, infoInits);
        irstruct->interfaceInfos->setInitializer(arrInit);
    }
    else
    {
        assert(irstruct->interfaceInfos == NULL);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

static void DtoDefineInterface(InterfaceDeclaration* cd)
{
    DtoConstInitInterface(cd);

    if (cd->ir.defined) return;
    cd->ir.defined = true;

    Logger::println("DtoDefineClass(%s): %s", cd->toPrettyChars(), cd->loc.toChars());
    LOG_SCOPE;

    // defined interface infos
    DefineInterfaceInfos(cd->ir.irStruct);

    // define the classinfo
    if (mustDefineSymbol(cd))
    {
        DtoDefineClassInfo(cd);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

// FIXME: clean this up

void DtoDefineClass(ClassDeclaration* cd)
{
    if (InterfaceDeclaration* id = cd->isInterfaceDeclaration())
    {
        DtoDefineInterface(id);
        return;
    }

    DtoConstInitClass(cd);

    if (cd->ir.defined) return;
    cd->ir.defined = true;

    Logger::println("DtoDefineClass(%s): %s", cd->toPrettyChars(), cd->loc.toChars());
    LOG_SCOPE;

    // get the struct (class) type
    assert(cd->type->ty == Tclass);
    TypeClass* ts = (TypeClass*)cd->type;

    IrStruct* irstruct = cd->ir.irStruct;

    assert(mustDefineSymbol(cd));

    // sanity check
    assert(irstruct->init);
    assert(irstruct->constInit);
    assert(irstruct->vtbl);
    assert(irstruct->constVtbl);

//     if (Logger::enabled())
//     {
//         Logger::cout() << "initZ: " << *irstruct->init << std::endl;
//         Logger::cout() << "cinitZ: " << *irstruct->constInit << std::endl;
//         Logger::cout() << "vtblZ: " << *irstruct->vtbl << std::endl;
//         Logger::cout() << "cvtblZ: " << *irstruct->constVtbl << std::endl;
//     }

    // set initializers
    irstruct->init->setInitializer(irstruct->constInit);
    irstruct->vtbl->setInitializer(irstruct->constVtbl);

    // initialize interface vtables
    size_t n = irstruct->interfaceVec.size();
    for (size_t i=0; i<n; i++)
    {
        IrInterface* iri = irstruct->interfaceVec[i];
        Logger::println("interface %s", iri->base->base->toChars());
        assert(iri->vtblInit);

        // refine the init type
        llvm::cast<llvm::OpaqueType>(iri->vtblInitTy.get())->refineAbstractTypeTo(iri->vtblInit->getType());

        // apply initializer
        assert(iri->vtbl);
        iri->vtbl->setInitializer(iri->vtblInit);
    }

    DefineInterfaceInfos(irstruct);

    // generate classinfo
    DtoDefineClassInfo(cd);
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* DtoNewClass(Loc loc, TypeClass* tc, NewExp* newexp)
{
    // resolve type
    tc->sym->codegen(Type::sir);

    // allocate
    LLValue* mem;
    if (newexp->onstack)
    {
        mem = DtoAlloca(DtoType(tc)->getContainedType(0), ".newclass_alloca");
    }
    // custom allocator
    else if (newexp->allocator)
    {
        newexp->allocator->codegen(Type::sir);
        DFuncValue dfn(newexp->allocator, newexp->allocator->ir.irFunc->func);
        DValue* res = DtoCallFunction(newexp->loc, NULL, &dfn, newexp->newargs);
        mem = DtoBitCast(res->getRVal(), DtoType(tc), ".newclass_custom");
    }
    // default allocator
    else
    {
        llvm::Function* fn = LLVM_D_GetRuntimeFunction(gIR->module, "_d_allocclass");
        LLConstant* ci = DtoBitCast(tc->sym->ir.irStruct->classInfo, DtoType(ClassDeclaration::classinfo->type));
        mem = gIR->CreateCallOrInvoke(fn, ci, ".newclass_gc_alloc").getInstruction();
        mem = DtoBitCast(mem, DtoType(tc), ".newclass_gc");
    }

    // init
    DtoInitClass(tc, mem);

    // init inner-class outer reference
    if (newexp->thisexp)
    {
        Logger::println("Resolving outer class");
        LOG_SCOPE;
        DValue* thisval = newexp->thisexp->toElem(gIR);
        size_t idx = tc->sym->vthis->ir.irField->index;
        LLValue* src = thisval->getRVal();
        LLValue* dst = DtoGEPi(mem,0,idx,"tmp");
        if (Logger::enabled())
            Logger::cout() << "dst: " << *dst << "\nsrc: " << *src << '\n';
        DtoStore(src, dst);
    }
    // set the context for nested classes
    else if (tc->sym->isNested() && tc->sym->vthis)
    {
        Logger::println("Resolving nested context");
        LOG_SCOPE;

        // get context
        LLValue* nest = DtoNestedContext(loc, tc->sym);

        // store into right location
        size_t idx = tc->sym->vthis->ir.irField->index;
        LLValue* gep = DtoGEPi(mem,0,idx,"tmp");
        DtoStore(DtoBitCast(nest, gep->getType()->getContainedType(0)), gep);
    }

    // call constructor
    if (newexp->member)
    {
        Logger::println("Calling constructor");
        assert(newexp->arguments != NULL);
        newexp->member->codegen(Type::sir);
        DFuncValue dfn(newexp->member, newexp->member->ir.irFunc->func, mem);
        return DtoCallFunction(newexp->loc, tc, &dfn, newexp->arguments);
    }

    // return default constructed class
    return new DImValue(tc, mem);
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoInitClass(TypeClass* tc, LLValue* dst)
{
    tc->sym->codegen(Type::sir);

    size_t presz = 2*getTypePaddedSize(DtoSize_t());
    uint64_t n = getTypePaddedSize(tc->ir.type->get()) - presz;

    // set vtable field seperately, this might give better optimization
    assert(tc->sym->ir.irStruct->vtbl);
    LLValue* tmp = DtoGEPi(dst,0,0,"vtbl");
    LLValue* val = DtoBitCast(tc->sym->ir.irStruct->vtbl, tmp->getType()->getContainedType(0));
    DtoStore(val, tmp);

    // monitor always defaults to zero
    tmp = DtoGEPi(dst,0,1,"monitor");
    val = llvm::Constant::getNullValue(tmp->getType()->getContainedType(0));
    DtoStore(val, tmp);

    // done?
    if (n == 0)
        return;

    // copy the rest from the static initializer
    assert(tc->sym->ir.irStruct->init);

    LLValue* dstarr = DtoGEPi(dst,0,2,"tmp");
    LLValue* srcarr = DtoGEPi(tc->sym->ir.irStruct->init,0,2,"tmp");

    DtoMemCpy(dstarr, srcarr, DtoConstSize_t(n));
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoFinalizeClass(LLValue* inst)
{
    // get runtime function
    llvm::Function* fn = LLVM_D_GetRuntimeFunction(gIR->module, "_d_callfinalizer");
    // build args
    LLSmallVector<LLValue*,1> arg;
    arg.push_back(DtoBitCast(inst, fn->getFunctionType()->getParamType(0), ".tmp"));
    // call
    gIR->CreateCallOrInvoke(fn, arg.begin(), arg.end(), "");
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* DtoCastClass(DValue* val, Type* _to)
{
    Logger::println("DtoCastClass(%s, %s)", val->getType()->toChars(), _to->toChars());
    LOG_SCOPE;

    Type* to = _to->toBasetype();

    // class -> pointer
    if (to->ty == Tpointer) {
        Logger::println("to pointer");
        const LLType* tolltype = DtoType(_to);
        LLValue* rval = DtoBitCast(val->getRVal(), tolltype);
        return new DImValue(_to, rval);
    }
    // class -> bool
    else if (to->ty == Tbool) {
        Logger::println("to bool");
        LLValue* llval = val->getRVal();
        LLValue* zero = LLConstant::getNullValue(llval->getType());
        return new DImValue(_to, gIR->ir->CreateICmpNE(llval, zero, "tmp"));
    }

    // must be class/interface
    assert(to->ty == Tclass);
    TypeClass* tc = (TypeClass*)to;

    // from type
    Type* from = val->getType()->toBasetype();
    TypeClass* fc = (TypeClass*)from;

    // x -> interface
    if (InterfaceDeclaration* it = tc->sym->isInterfaceDeclaration()) {
        Logger::println("to interface");
        // interface -> interface
        if (fc->sym->isInterfaceDeclaration()) {
            Logger::println("from interface");
            return DtoDynamicCastInterface(val, _to);
        }
        // class -> interface - static cast
        else if (it->isBaseOf(fc->sym,NULL)) {
            Logger::println("static down cast");
            // get the from class
            ClassDeclaration* cd = fc->sym->isClassDeclaration();
            DtoResolveClass(cd); // add this
            IrStruct* irstruct = cd->ir.irStruct;
            // find interface impl
            IrStruct::InterfaceMapIter iriter = irstruct->interfaceMap.find(it);
            assert(iriter != irstruct->interfaceMap.end());
            IrInterface* iri = iriter->second;
            // offset pointer
            LLValue* v = val->getRVal();
            LLValue* orig = v;
            v = DtoGEPi(v, 0, iri->index);
            const LLType* ifType = DtoType(_to);
            if (Logger::enabled())
            {
                Logger::cout() << "V = " << *v << std::endl;
                Logger::cout() << "T = " << *ifType << std::endl;
            }
            v = DtoBitCast(v, ifType);
            // Check whether the original value was null, and return null if so.
            // Sure we could have jumped over the code above in this case, but
            // it's just a GEP and (maybe) a pointer-to-pointer BitCast, so it
            // should be pretty cheap and perfectly safe even if the original was null.
            LLValue* isNull = gIR->ir->CreateICmpEQ(orig, LLConstant::getNullValue(orig->getType()), ".nullcheck");
            v = gIR->ir->CreateSelect(isNull, LLConstant::getNullValue(ifType), v, ".interface");
            // return r-value
            return new DImValue(_to, v);
        }
        // class -> interface
        else {
            Logger::println("from object");
            return DtoDynamicCastObject(val, _to);
        }
    }
    // x -> class
    else {
        Logger::println("to class");
        int poffset;
        // interface -> class
        if (fc->sym->isInterfaceDeclaration()) {
            Logger::println("interface cast");
            return DtoDynamicCastInterface(val, _to);
        }
        // class -> class - static down cast
        else if (tc->sym->isBaseOf(fc->sym,NULL)) {
            Logger::println("static down cast");
            const LLType* tolltype = DtoType(_to);
            LLValue* rval = DtoBitCast(val->getRVal(), tolltype);
            return new DImValue(_to, rval);
        }
        // class -> class - dynamic up cast
        else {
            Logger::println("dynamic up cast");
            return DtoDynamicCastObject(val, _to);
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* DtoDynamicCastObject(DValue* val, Type* _to)
{
    // call:
    // Object _d_dynamic_cast(Object o, ClassInfo c)

    ClassDeclaration::object->codegen(Type::sir);
    ClassDeclaration::classinfo->codegen(Type::sir);

    llvm::Function* func = LLVM_D_GetRuntimeFunction(gIR->module, "_d_dynamic_cast");
    const llvm::FunctionType* funcTy = func->getFunctionType();

    std::vector<LLValue*> args;

    // Object o
    LLValue* obj = val->getRVal();
    obj = DtoBitCast(obj, funcTy->getParamType(0));
    assert(funcTy->getParamType(0) == obj->getType());

    // ClassInfo c
    TypeClass* to = (TypeClass*)_to->toBasetype();
    to->sym->codegen(Type::sir);
    assert(to->sym->ir.irStruct->classInfo);
    LLValue* cinfo = to->sym->ir.irStruct->classInfo;
    // unfortunately this is needed as the implementation of object differs somehow from the declaration
    // this could happen in user code as well :/
    cinfo = DtoBitCast(cinfo, funcTy->getParamType(1));
    assert(funcTy->getParamType(1) == cinfo->getType());

    // call it
    LLValue* ret = gIR->CreateCallOrInvoke2(func, obj, cinfo, "tmp").getInstruction();

    // cast return value
    ret = DtoBitCast(ret, DtoType(_to));

    return new DImValue(_to, ret);
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* DtoCastInterfaceToObject(DValue* val, Type* to)
{
    // call:
    // Object _d_toObject(void* p)

    llvm::Function* func = LLVM_D_GetRuntimeFunction(gIR->module, "_d_toObject");
    const llvm::FunctionType* funcTy = func->getFunctionType();

    // void* p
    LLValue* tmp = val->getRVal();
    tmp = DtoBitCast(tmp, funcTy->getParamType(0));

    // call it
    LLValue* ret = gIR->CreateCallOrInvoke(func, tmp, "tmp").getInstruction();

    // cast return value
    if (to != NULL)
        ret = DtoBitCast(ret, DtoType(to));
    else
        to = ClassDeclaration::object->type;

    return new DImValue(to, ret);
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* DtoDynamicCastInterface(DValue* val, Type* _to)
{
    // call:
    // Object _d_interface_cast(void* p, ClassInfo c)

    ClassDeclaration::object->codegen(Type::sir);
    ClassDeclaration::classinfo->codegen(Type::sir);

    llvm::Function* func = LLVM_D_GetRuntimeFunction(gIR->module, "_d_interface_cast");
    const llvm::FunctionType* funcTy = func->getFunctionType();

    std::vector<LLValue*> args;

    // void* p
    LLValue* ptr = val->getRVal();
    ptr = DtoBitCast(ptr, funcTy->getParamType(0));

    // ClassInfo c
    TypeClass* to = (TypeClass*)_to->toBasetype();
    to->sym->codegen(Type::sir);
    assert(to->sym->ir.irStruct->classInfo);
    LLValue* cinfo = to->sym->ir.irStruct->classInfo;
    // unfortunately this is needed as the implementation of object differs somehow from the declaration
    // this could happen in user code as well :/
    cinfo = DtoBitCast(cinfo, funcTy->getParamType(1));

    // call it
    LLValue* ret = gIR->CreateCallOrInvoke2(func, ptr, cinfo, "tmp").getInstruction();

    // cast return value
    ret = DtoBitCast(ret, DtoType(_to));

    return new DImValue(_to, ret);
}

//////////////////////////////////////////////////////////////////////////////////////////

LLValue* DtoIndexClass(LLValue* src, ClassDeclaration* cd, VarDeclaration* vd)
{
    Logger::println("indexing class field %s:", vd->toPrettyChars());
    LOG_SCOPE;

    if (Logger::enabled())
        Logger::cout() << "src: " << *src << '\n';

    // make sure class is resolved
    DtoResolveClass(cd);

    // vd must be a field
    IrField* field = vd->ir.irField;
    assert(field);

    // get the start pointer
    const LLType* st = DtoType(cd->type);
    // cast to the struct type
    src = DtoBitCast(src, st);

    // gep to the index
    if (Logger::enabled())
    {
        Logger::cout() << "src2: " << *src << '\n';
        Logger::cout() << "index: " << field->index << '\n';
        Logger::cout() << "srctype: " << *src->getType() << '\n';
    }
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

LLValue* DtoVirtualFunctionPointer(DValue* inst, FuncDeclaration* fdecl)
{
    // sanity checks
    assert(fdecl->isVirtual());
    assert(!fdecl->isFinal());
    assert(fdecl->vtblIndex > 0); // 0 is always ClassInfo/Interface*
    assert(inst->getType()->toBasetype()->ty == Tclass);

    // get instance
    LLValue* vthis = inst->getRVal();
    if (Logger::enabled())
        Logger::cout() << "vthis: " << *vthis << '\n';

    LLValue* funcval = vthis;
    // get the vtbl for objects
    if (!fdecl->isMember()->isInterfaceDeclaration())
        funcval = DtoGEPi(funcval, 0, 0, "tmp");
    // load vtbl ptr
    funcval = DtoLoad(funcval);
    // index vtbl
    funcval = DtoGEPi(funcval, 0, fdecl->vtblIndex, fdecl->toChars());
    // load funcptr
    funcval = DtoAlignedLoad(funcval);

    if (Logger::enabled())
        Logger::cout() << "funcval: " << *funcval << '\n';

    // cast to final funcptr type
    funcval = DtoBitCast(funcval, getPtrToType(DtoType(fdecl->type)));
    if (Logger::enabled())
        Logger::cout() << "funcval casted: " << *funcval << '\n';

    return funcval;
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoDeclareClassInfo(ClassDeclaration* cd)
{
    IrStruct* irstruct = cd->ir.irStruct;

    if (irstruct->classInfoDeclared) return;
    irstruct->classInfoDeclared = true;

    Logger::println("DtoDeclareClassInfo(%s)", cd->toChars());
    LOG_SCOPE;

    // resovle ClassInfo
    ClassDeclaration* cinfo = ClassDeclaration::classinfo;
    DtoResolveClass(cinfo);
}

//////////////////////////////////////////////////////////////////////////////////////////

#if GENERATE_OFFTI

// build a single element for the OffsetInfo[] of ClassInfo
static LLConstant* build_offti_entry(ClassDeclaration* cd, VarDeclaration* vd)
{
    std::vector<LLConstant*> inits(2);

    // size_t offset;
    //
    assert(vd->ir.irField);
    // grab the offset from llvm and the formal class type
    size_t offset = gTargetData->getStructLayout(isaStruct(cd->type->ir.type->get()))->getElementOffset(vd->ir.irField->index);
    // offset nested struct/union fields
    offset += vd->ir.irField->unionOffset;

    // assert that it matches DMD
    Logger::println("offsets: %lu vs %u", offset, vd->offset);
    assert(offset == vd->offset);

    inits[0] = DtoConstSize_t(offset);

    // TypeInfo ti;
    inits[1] = DtoTypeInfoOf(vd->type, true);

    // done
    return llvm::ConstantStruct::get(inits);
}

static LLConstant* build_offti_array(ClassDeclaration* cd, const LLType* arrayT)
{
    IrStruct* irstruct = cd->ir.irStruct;

    size_t nvars = irstruct->varDecls.size();
    std::vector<LLConstant*> arrayInits(nvars);

    for (size_t i=0; i<nvars; i++)
    {
        arrayInits[i] = build_offti_entry(cd, irstruct->varDecls[i]);
    }

    LLConstant* size = DtoConstSize_t(nvars);
    LLConstant* ptr;

    if (nvars == 0)
        return LLConstant::getNullValue( arrayT );

    // array type
    const llvm::ArrayType* arrTy = llvm::ArrayType::get(arrayInits[0]->getType(), nvars);
    LLConstant* arrInit = llvm::ConstantArray::get(arrTy, arrayInits);

    // mangle
    std::string name(cd->type->vtinfo->toChars());
    name.append("__OffsetTypeInfos");

    // create symbol
    llvm::GlobalVariable* gvar = new llvm::GlobalVariable(arrTy,true,DtoInternalLinkage(cd),arrInit,name,gIR->module);
    ptr = DtoBitCast(gvar, getPtrToType(arrTy->getElementType()));

    return DtoConstSlice(size, ptr);
}

#endif // GENERATE_OFFTI

static LLConstant* build_class_dtor(ClassDeclaration* cd)
{
    FuncDeclaration* dtor = cd->dtor;

    // if no destructor emit a null
    if (!dtor)
        return getNullPtr(getVoidPtrType());

    dtor->codegen(Type::sir);
    return llvm::ConstantExpr::getBitCast(dtor->ir.irFunc->func, getPtrToType(LLType::Int8Ty));
}

static unsigned build_classinfo_flags(ClassDeclaration* cd)
{
    // adapted from original dmd code
    unsigned flags = 0;
    //flags |= isCOMclass(); // IUnknown
    bool hasOffTi = false;
    if (cd->ctor) flags |= 8;
    for (ClassDeclaration *cd2 = cd; cd2; cd2 = cd2->baseClass)
    {
    if (cd2->members)
    {
        for (size_t i = 0; i < cd2->members->dim; i++)
        {
        Dsymbol *sm = (Dsymbol *)cd2->members->data[i];
        if (sm->isVarDeclaration() && !sm->isVarDeclaration()->isDataseg()) // is this enough?
            hasOffTi = true;
        //printf("sm = %s %s\n", sm->kind(), sm->toChars());
        if (sm->hasPointers())
            goto L2;
        }
    }
    }
    flags |= 2;         // no pointers
L2:
    if (hasOffTi)
        flags |= 4;
    return flags;
}

void DtoDefineClassInfo(ClassDeclaration* cd)
{
//     The layout is:
//        {
//         void **vptr;
//         monitor_t monitor;
//         byte[] initializer;     // static initialization data
//         char[] name;        // class name
//         void *[] vtbl;
//         Interface[] interfaces;
//         ClassInfo *base;        // base class
//         void *destructor;
//         void *invariant;        // class invariant
//         uint flags;
//         void *deallocator;
//         OffsetTypeInfo[] offTi;
//         void *defaultConstructor;
//        }

    IrStruct* ir = cd->ir.irStruct;

    if (ir->classInfoDefined) return;
    ir->classInfoDefined = true;

    Logger::println("DtoDefineClassInfo(%s)", cd->toChars());
    LOG_SCOPE;

    assert(cd->type->ty == Tclass);
    assert(ir->classInfo);

    TypeClass* cdty = (TypeClass*)cd->type;
    if (!cd->isInterfaceDeclaration())
    {
        assert(ir->init);
        assert(ir->constInit);
        assert(ir->vtbl);
        assert(ir->constVtbl);
    }

    // holds the list of initializers for llvm
    std::vector<LLConstant*> inits;

    ClassDeclaration* cinfo = ClassDeclaration::classinfo;
    cinfo->codegen(Type::sir);

    LLConstant* c;

    const LLType* voidPtr = getVoidPtrType();
    const LLType* voidPtrPtr = getPtrToType(voidPtr);

    // own vtable
    c = cinfo->ir.irStruct->vtbl;
    assert(c);
    inits.push_back(c);

    // monitor
    c = LLConstant::getNullValue(voidPtr);
    inits.push_back(c);

    // byte[] init
    if (cd->isInterfaceDeclaration())
        c = DtoConstSlice(DtoConstSize_t(0), LLConstant::getNullValue(voidPtr));
    else
    {
        c = DtoBitCast(ir->init, voidPtr);
        //Logger::cout() << *ir->constInit->getType() << std::endl;
        size_t initsz = getTypePaddedSize(ir->init->getType()->getContainedType(0));
        c = DtoConstSlice(DtoConstSize_t(initsz), c);
    }
    inits.push_back(c);

    // class name
    // from dmd
    char *name = cd->ident->toChars();
    size_t namelen = strlen(name);
    if (!(namelen > 9 && memcmp(name, "TypeInfo_", 9) == 0))
    {
        name = cd->toPrettyChars();
        namelen = strlen(name);
    }
    c = DtoConstString(name);
    inits.push_back(c);

    // vtbl array
    if (cd->isInterfaceDeclaration())
        c = DtoConstSlice(DtoConstSize_t(0), LLConstant::getNullValue(getPtrToType(voidPtr)));
    else {
        c = DtoBitCast(ir->vtbl, voidPtrPtr);
        c = DtoConstSlice(DtoConstSize_t(cd->vtbl.dim), c);
    }
    inits.push_back(c);

    // interfaces array
    VarDeclaration* intersVar = (VarDeclaration*)cinfo->fields.data[3];
    const LLType* intersTy = DtoType(intersVar->type);
    if (!ir->interfaceInfos)
        c = LLConstant::getNullValue(intersTy);
    else {
        const LLType* t = intersTy->getContainedType(1); // .ptr
        // cast to Interface*
        c = DtoBitCast(ir->interfaceInfos, t);
        size_t isz = ir->interfaceVec.size();
        size_t iisz = ir->classInfoInterfaces.size();
        assert(iisz <= isz);
        // offset - we only want the 'iisz' last ones
        LLConstant* idx = DtoConstUint(isz - iisz);
        c = llvm::ConstantExpr::getGetElementPtr(c, &idx, 1);
        // make array
        c = DtoConstSlice(DtoConstSize_t(iisz), c);
    }
    inits.push_back(c);

    // base classinfo
    // interfaces never get a base , just the interfaces[]
    if (cd->baseClass && !cd->isInterfaceDeclaration()) {
        DtoDeclareClassInfo(cd->baseClass);
        c = cd->baseClass->ir.irStruct->classInfo;
        assert(c);
        inits.push_back(c);
    }
    else {
        // null
        c = LLConstant::getNullValue(DtoType(cinfo->type));
        inits.push_back(c);
    }

    // destructor
    if (cd->isInterfaceDeclaration())
        c = LLConstant::getNullValue(voidPtr);
    else
        c = build_class_dtor(cd);
    inits.push_back(c);

    // invariant
    VarDeclaration* invVar = (VarDeclaration*)cinfo->fields.data[6];
    const LLType* invTy = DtoType(invVar->type);
    if (cd->inv)
    {
        cd->inv->codegen(Type::sir);
        c = cd->inv->ir.irFunc->func;
        c = DtoBitCast(c, invTy);
    }
    else
        c = LLConstant::getNullValue(invTy);
    inits.push_back(c);

    // uint flags
    if (cd->isInterfaceDeclaration())
        c = DtoConstUint(0);
    else {
        unsigned flags = build_classinfo_flags(cd);
        c = DtoConstUint(flags);
    }
    inits.push_back(c);

    // deallocator
    if (cd->aggDelete)
    {
        cd->aggDelete->codegen(Type::sir);
        c = cd->aggDelete->ir.irFunc->func;
        c = DtoBitCast(c, voidPtr);
    }
    else
        c = LLConstant::getNullValue(voidPtr);
    inits.push_back(c);

    // offset typeinfo
    VarDeclaration* offTiVar = (VarDeclaration*)cinfo->fields.data[9];
    const LLType* offTiTy = DtoType(offTiVar->type);

#if GENERATE_OFFTI

    if (cd->isInterfaceDeclaration())
        c = LLConstant::getNullValue(offTiTy);
    else
        c = build_offti_array(cd, offTiTy);

#else // GENERATE_OFFTI

    c = LLConstant::getNullValue(offTiTy);

#endif // GENERATE_OFFTI

    inits.push_back(c);

    // default constructor
    if (cd->defaultCtor)
    {
        cd->defaultCtor->codegen(Type::sir);
        c = isaConstant(cd->defaultCtor->ir.irFunc->func);
        c = DtoBitCast(c, voidPtr);
    }
    else
        c = LLConstant::getNullValue(voidPtr);
    inits.push_back(c);

#if DMDV2

    // xgetMembers
    VarDeclaration* xgetVar = (VarDeclaration*)cinfo->fields.data[11];
    const LLType* xgetTy = DtoType(xgetVar->type);

    // FIXME: fill it out!
    inits.push_back( LLConstant::getNullValue(xgetTy) );
#endif

    /*size_t n = inits.size();
    for (size_t i=0; i<n; ++i)
    {
        Logger::cout() << "inits[" << i << "]: " << *inits[i] << '\n';
    }*/

    // build the initializer
    LLConstant* finalinit = llvm::ConstantStruct::get(inits);
    //Logger::cout() << "built the classinfo initializer:\n" << *finalinit <<'\n';
    ir->constClassInfo = finalinit;

    // refine the type
    llvm::cast<llvm::OpaqueType>(ir->classInfoOpaque.get())->refineAbstractTypeTo(finalinit->getType());

    // apply initializer
    ir->classInfo->setInitializer(finalinit);
}
