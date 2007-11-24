#include <sstream>
#include "gen/llvm.h"

#include "mtype.h"
#include "aggregate.h"
#include "init.h"
#include "declaration.h"

#include "gen/irstate.h"
#include "gen/tollvm.h"
#include "gen/arrays.h"
#include "gen/logger.h"
#include "gen/classes.h"
#include "gen/functions.h"
#include "gen/runtime.h"
#include "gen/dvalue.h"

//////////////////////////////////////////////////////////////////////////////////////////

static void LLVM_AddBaseClassData(BaseClasses* bcs)
{
    // add base class data members first
    for (int j=0; j<bcs->dim; j++)
    {
        BaseClass* bc = (BaseClass*)(bcs->data[j]);
        assert(bc);
        if (bc->base->isInterfaceDeclaration())
            continue; // interfaces only has methods

        LLVM_AddBaseClassData(&bc->base->baseclasses);

        Logger::println("Adding base class members of %s", bc->base->toChars());
        LOG_SCOPE;

        for (int k=0; k < bc->base->members->dim; k++) {
            Dsymbol* dsym = (Dsymbol*)(bc->base->members->data[k]);
            if (dsym->isVarDeclaration())
            {
                dsym->toObjFile();
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoResolveClass(ClassDeclaration* cd)
{
    if (cd->llvmResolved) return;
    cd->llvmResolved = true;

    // first resolve the base class
    if (cd->baseClass) {
        DtoResolveClass(cd->baseClass);
    }

    // resolve interfaces
    if (cd->vtblInterfaces) {
        for (int i=0; i < cd->vtblInterfaces->dim; i++) {
            BaseClass *b = (BaseClass *)cd->vtblInterfaces->data[i];
            ClassDeclaration *id = b->base;
            DtoResolveClass(id);
            // Fill in vtbl[]
            b->fillVtbl(cd, &b->vtbl, 1);
        }
    }

    Logger::println("DtoResolveClass(%s)", cd->toPrettyChars());
    LOG_SCOPE;

    assert(cd->type->ty == Tclass);
    TypeClass* ts = (TypeClass*)cd->type;

    assert(!cd->llvmIRStruct);
    IRStruct* irstruct = new IRStruct(ts);
    cd->llvmIRStruct = irstruct;

    gIR->structs.push_back(irstruct);
    gIR->classes.push_back(cd);

    // add vtable
    ts->llvmVtblType = new llvm::PATypeHolder(llvm::OpaqueType::get());
    const llvm::Type* vtabty = llvm::PointerType::get(ts->llvmVtblType->get());

    std::vector<const llvm::Type*> fieldtypes;
    fieldtypes.push_back(vtabty);

    // add interface vtables
    if (cd->vtblInterfaces)
    for (size_t i = 0; i < cd->vtblInterfaces->dim; i++)
    {
        BaseClass *b = (BaseClass *)cd->vtblInterfaces->data[i];
        ClassDeclaration *id = b->base;
        assert(id->type->ty == Tclass);
        TypeClass* itc = (TypeClass*)id->type;
        const llvm::Type* ivtblTy = llvm::PointerType::get(itc->llvmVtblType->get());
        fieldtypes.push_back(ivtblTy);

        // add this interface to the map
        IRInterface* iri = new IRInterface(b, isaStruct(itc->llvmVtblType->get()));
        irstruct->interfaces.insert(std::make_pair(id, iri));
    }

    // base classes first
    LLVM_AddBaseClassData(&cd->baseclasses);

    // then add own members
    for (int k=0; k < cd->members->dim; k++) {
        Dsymbol* dsym = (Dsymbol*)(cd->members->data[k]);
        dsym->toObjFile();
    }

    // add field types
    for (IRStruct::OffsetMap::iterator i=irstruct->offsets.begin(); i!=irstruct->offsets.end(); ++i) {
        fieldtypes.push_back(i->second.type);
    }

    const llvm::StructType* structtype = llvm::StructType::get(fieldtypes);
    // refine abstract types for stuff like: class C {C next;}
    assert(irstruct->recty != 0);

    llvm::PATypeHolder& spa = irstruct->recty;
    llvm::cast<llvm::OpaqueType>(spa.get())->refineAbstractTypeTo(structtype);
    structtype = isaStruct(spa.get());

    if (!ts->llvmType)
        ts->llvmType = new llvm::PATypeHolder(structtype);
    else
        *ts->llvmType = structtype;

    if (cd->parent->isModule()) {
        gIR->module->addTypeName(cd->mangle(), ts->llvmType->get());
    }
    else {
        assert(0 && "class parent is not a module");
    }

    // build interface info type
    std::vector<const llvm::Type*> infoTypes;
    // ClassInfo classinfo
    ClassDeclaration* cinfod = ClassDeclaration::classinfo;
    DtoResolveClass(cinfod);
    infoTypes.push_back(llvm::PointerType::get(cinfod->type->llvmType->get()));
    // void*[] vtbl
    std::vector<const llvm::Type*> infoVtbltypes;
    infoVtbltypes.push_back(DtoSize_t());
    const llvm::Type* byteptrptrty = llvm::PointerType::get(llvm::PointerType::get(llvm::Type::Int8Ty));
    infoVtbltypes.push_back(byteptrptrty);
    infoTypes.push_back(llvm::StructType::get(infoVtbltypes));
    // int offset
    infoTypes.push_back(llvm::Type::Int32Ty);
    // create type
    const llvm::StructType* infoTy = llvm::StructType::get(infoTypes);

    // create vtable type
    llvm::GlobalVariable* svtblVar = 0;
    std::vector<const llvm::Type*> sinits_ty;

    for (int k=0; k < cd->vtbl.dim; k++)
    {
        Dsymbol* dsym = (Dsymbol*)cd->vtbl.data[k];
        assert(dsym);
        //Logger::cout() << "vtblsym: " << dsym->toChars() << '\n';

        if (FuncDeclaration* fd = dsym->isFuncDeclaration()) {
            DtoResolveFunction(fd);
            assert(fd->type->ty == Tfunction);
            TypeFunction* tf = (TypeFunction*)fd->type;
            const llvm::Type* fpty = llvm::PointerType::get(tf->llvmType->get());
            sinits_ty.push_back(fpty);
        }
        else if (ClassDeclaration* cd2 = dsym->isClassDeclaration()) {
            Logger::println("*** ClassDeclaration in vtable: %s", cd2->toChars());
            const llvm::Type* cinfoty;
            if (cd->isInterfaceDeclaration()) {
                cinfoty = infoTy;
            }
            else if (cd != cinfod) {
                DtoResolveClass(cinfod);
                cinfoty = cinfod->type->llvmType->get();
            }
            else {
                // this is the ClassInfo class, the type is this type
                cinfoty = ts->llvmType->get();
            }
            const llvm::Type* cty = llvm::PointerType::get(cinfoty);
            sinits_ty.push_back(cty);
        }
        else
        assert(0);
    }

    assert(!sinits_ty.empty());
    const llvm::StructType* svtbl_ty = llvm::StructType::get(sinits_ty);

    std::string styname(cd->mangle());
    styname.append("__vtblType");
    gIR->module->addTypeName(styname, svtbl_ty);

    // refine for final vtable type
    llvm::cast<llvm::OpaqueType>(ts->llvmVtblType->get())->refineAbstractTypeTo(svtbl_ty);

    gIR->classes.pop_back();
    gIR->structs.pop_back();

    gIR->declareList.push_back(cd);
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoDeclareClass(ClassDeclaration* cd)
{
    if (cd->llvmDeclared) return;
    cd->llvmDeclared = true;

    Logger::println("DtoDeclareClass(%s)", cd->toPrettyChars());
    LOG_SCOPE;

    assert(cd->type->ty == Tclass);
    TypeClass* ts = (TypeClass*)cd->type;

    assert(cd->llvmIRStruct);
    IRStruct* irstruct = cd->llvmIRStruct;

    gIR->structs.push_back(irstruct);
    gIR->classes.push_back(cd);

    bool needs_definition = false;
    if (cd->parent->isModule()) {
        needs_definition = (cd->getModule() == gIR->dmodule);
    }

    // interface vtables are emitted by the class implementing them
    // also interfaces have no static initializer
    if (!cd->isInterfaceDeclaration()) {
        // vtable
        std::string varname("_D");
        varname.append(cd->mangle());
        varname.append("6__vtblZ");

        llvm::GlobalValue::LinkageTypes _linkage = llvm::GlobalValue::ExternalLinkage;

        const llvm::StructType* svtbl_ty = isaStruct(ts->llvmVtblType->get());
        cd->llvmVtbl = new llvm::GlobalVariable(svtbl_ty, true, _linkage, 0, varname, gIR->module);

        // build interface info type
        std::vector<const llvm::Type*> types;
        // ClassInfo classinfo
        ClassDeclaration* cd2 = ClassDeclaration::classinfo;
        DtoResolveClass(cd2);
        types.push_back(llvm::PointerType::get(cd2->type->llvmType->get()));
        // void*[] vtbl
        std::vector<const llvm::Type*> vtbltypes;
        vtbltypes.push_back(DtoSize_t());
        const llvm::Type* byteptrptrty = llvm::PointerType::get(llvm::PointerType::get(llvm::Type::Int8Ty));
        vtbltypes.push_back(byteptrptrty);
        types.push_back(llvm::StructType::get(vtbltypes));
        // int offset
        types.push_back(llvm::Type::Int32Ty);
        // create type
        const llvm::StructType* infoTy = llvm::StructType::get(types);

        // interface info array
        if (needs_definition && cd->vtblInterfaces->dim > 0) {
            // symbol name
            std::string nam = "_D";
            nam.append(cd->mangle());
            nam.append("16__interfaceInfosZ");
            // resolve array type
            const llvm::ArrayType* arrTy = llvm::ArrayType::get(infoTy, cd->vtblInterfaces->dim);
            // declare global
            irstruct->interfaceInfosTy = arrTy;
            irstruct->interfaceInfos = new llvm::GlobalVariable(arrTy, true, llvm::GlobalValue::InternalLinkage, 0, nam, gIR->module);
        }

        // interface vtables
        unsigned idx = 0;
        for (IRStruct::InterfaceIter i=irstruct->interfaces.begin(); i!=irstruct->interfaces.end(); ++i)
        {
            ClassDeclaration* id = i->first;
            IRInterface* iri = i->second;

            std::string nam("_D");
            nam.append(cd->mangle());
            nam.append("11__interface");
            nam.append(id->mangle());
            nam.append("6__vtblZ");

            assert(iri->vtblTy);
            iri->vtbl = new llvm::GlobalVariable(iri->vtblTy, true, _linkage, 0, nam, gIR->module);
            iri->infoTy = infoTy;
            llvm::Constant* idxs[2] = {DtoConstUint(0), DtoConstUint(idx)};
            iri->info = llvm::ConstantExpr::getGetElementPtr(irstruct->interfaceInfos, idxs, 2);
            idx++;
        }

        // init
        std::string initname("_D");
        initname.append(cd->mangle());
        initname.append("6__initZ");

        llvm::GlobalVariable* initvar = new llvm::GlobalVariable(ts->llvmType->get(), true, _linkage, NULL, initname, gIR->module);
        ts->llvmInit = initvar;
    }

    gIR->classes.pop_back();
    gIR->structs.pop_back();

    gIR->constInitList.push_back(cd);
    if (needs_definition)
        gIR->defineList.push_back(cd);

    // classinfo
    DtoDeclareClassInfo(cd);

    // typeinfo
    if (cd->parent->isModule() && cd->getModule() == gIR->dmodule)
        cd->type->getTypeInfo(NULL);
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoConstInitClass(ClassDeclaration* cd)
{
    if (cd->llvmInitialized) return;
    cd->llvmInitialized = true;

    if (cd->isInterfaceDeclaration())
        return; // nothing to do

    Logger::println("DtoConstInitClass(%s)", cd->toPrettyChars());
    LOG_SCOPE;

    IRStruct* irstruct = cd->llvmIRStruct;
    gIR->structs.push_back(irstruct);
    gIR->classes.push_back(cd);

    // make sure each offset knows its default initializer
    for (IRStruct::OffsetMap::iterator i=irstruct->offsets.begin(); i!=irstruct->offsets.end(); ++i)
    {
        IRStruct::Offset* so = &i->second;
        llvm::Constant* finit = DtoConstFieldInitializer(so->var->type, so->var->init);
        so->init = finit;
        so->var->llvmConstInit = finit;
    }

    // fill out fieldtypes/inits
    std::vector<llvm::Constant*> fieldinits;

    // first field is always the vtable
    assert(cd->llvmVtbl != 0);
    fieldinits.push_back(cd->llvmVtbl);

    // next comes interface vtables
    for (IRStruct::InterfaceIter i=irstruct->interfaces.begin(); i!=irstruct->interfaces.end(); ++i)
    {
        IRInterface* iri = i->second;
        assert(iri->vtbl);
        fieldinits.push_back(iri->vtbl);
    }

    // rest
    for (IRStruct::OffsetMap::iterator i=irstruct->offsets.begin(); i!=irstruct->offsets.end(); ++i) {
        Logger::println("adding fieldinit for: %s", i->second.var->toChars());
        fieldinits.push_back(i->second.init);
    }

    // get the struct (class) type
    assert(cd->type->ty == Tclass);
    TypeClass* ts = (TypeClass*)cd->type;
    const llvm::StructType* structtype = isaStruct(ts->llvmType->get());

    // generate initializer
#if 0
    Logger::cout() << cd->toPrettyChars() << " | " << *structtype << '\n';

    for(size_t i=0; i<structtype->getNumElements(); ++i) {
        Logger::cout() << "s#" << i << " = " << *structtype->getElementType(i) << '\n';
    }

    for(size_t i=0; i<fieldinits.size(); ++i) {
        Logger::cout() << "i#" << i << " = " << *fieldinits[i]->getType() << '\n';
    }
#endif

    llvm::Constant* _init = llvm::ConstantStruct::get(structtype, fieldinits);
    assert(_init);
    cd->llvmInitZ = _init;

    // generate vtable initializer
    std::vector<llvm::Constant*> sinits;

    for (int k=0; k < cd->vtbl.dim; k++)
    {
        Dsymbol* dsym = (Dsymbol*)cd->vtbl.data[k];
        assert(dsym);
        //Logger::cout() << "vtblsym: " << dsym->toChars() << '\n';

        if (FuncDeclaration* fd = dsym->isFuncDeclaration()) {
            DtoForceDeclareDsymbol(fd);
            assert(fd->llvmValue);
            llvm::Constant* c = llvm::cast<llvm::Constant>(fd->llvmValue);
            sinits.push_back(c);
        }
        else if (ClassDeclaration* cd2 = dsym->isClassDeclaration()) {
            assert(cd->llvmClass);
            llvm::Constant* c = cd->llvmClass;
            sinits.push_back(c);
        }
        else
        assert(0);
    }

    const llvm::StructType* svtbl_ty = isaStruct(ts->llvmVtblType->get());

#if 0
    for (size_t i=0; i< sinits.size(); ++i)
    {
        Logger::cout() << "field[" << i << "] = " << *svtbl_ty->getElementType(i) << '\n';
        Logger::cout() << "init [" << i << "] = " << *sinits[i]->getType() << '\n';
        assert(svtbl_ty->getElementType(i) == sinits[i]->getType());
    }
#endif

    llvm::Constant* cvtblInit = llvm::ConstantStruct::get(svtbl_ty, sinits);
    cd->llvmConstVtbl = llvm::cast<llvm::ConstantStruct>(cvtblInit);

    // create interface vtable const initalizers
    int idx = 1;
    int idxScale = (global.params.is64bit) ? 8 : 4;
    for (IRStruct::InterfaceIter i=irstruct->interfaces.begin(); i!=irstruct->interfaces.end(); ++i)
    {
        ClassDeclaration* id = i->first;
        assert(id->type->ty == Tclass);
        TypeClass* its = (TypeClass*)id->type;

        IRInterface* iri = i->second;
        BaseClass* b = iri->base;

        const llvm::StructType* ivtbl_ty = isaStruct(its->llvmVtblType->get());

        // generate interface info initializer
        std::vector<llvm::Constant*> infoInits;
        // classinfo
        assert(id->llvmClass);
        llvm::Constant* c = id->llvmClass;
        infoInits.push_back(c);
        // vtbl
        const llvm::Type* byteptrptrty = llvm::PointerType::get(llvm::PointerType::get(llvm::Type::Int8Ty));
        c = llvm::ConstantExpr::getBitCast(iri->vtbl, byteptrptrty);
        c = DtoConstSlice(DtoConstSize_t(b->vtbl.dim), c);
        infoInits.push_back(c);
        // offset
        infoInits.push_back(DtoConstInt(idx*idxScale));
        // create interface info initializer constant
        iri->infoInit = llvm::cast<llvm::ConstantStruct>(llvm::ConstantStruct::get(iri->infoTy, infoInits));

        // generate vtable initializer
        std::vector<llvm::Constant*> iinits;

        // add interface info
        iinits.push_back(iri->info);

        for (int k=1; k < b->vtbl.dim; k++)
        {
            Logger::println("interface vtbl const init nr. %d", k);
            Dsymbol* dsym = (Dsymbol*)b->vtbl.data[k];
            FuncDeclaration* fd = dsym->isFuncDeclaration();
            assert(fd);
            DtoForceDeclareDsymbol(fd);
            assert(fd->llvmValue);
            llvm::Constant* c = llvm::cast<llvm::Constant>(fd->llvmValue);
            // we have to bitcast, as the type created in ResolveClass expects a different this type
            c = llvm::ConstantExpr::getBitCast(c, iri->vtblTy->getContainedType(k));
            iinits.push_back(c);
        }

#if 1
        for (size_t x=0; x< iinits.size(); ++x)
        {
            Logger::cout() << "field[" << x << "] = " << *ivtbl_ty->getElementType(x) << "\n\n";
            Logger::cout() << "init [" << x << "] = " << *iinits[x] << "\n\n";
            assert(ivtbl_ty->getElementType(x) == iinits[x]->getType());
        }
#endif

        llvm::Constant* civtblInit = llvm::ConstantStruct::get(ivtbl_ty, iinits);
        iri->vtblInit = llvm::cast<llvm::ConstantStruct>(civtblInit);

        idx++;
    }

    gIR->classes.pop_back();
    gIR->structs.pop_back();
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoDefineClass(ClassDeclaration* cd)
{
    if (cd->llvmDefined) return;
    cd->llvmDefined = true;

    Logger::println("DtoDefineClass(%s)", cd->toPrettyChars());
    LOG_SCOPE;

    // get the struct (class) type
    assert(cd->type->ty == Tclass);
    TypeClass* ts = (TypeClass*)cd->type;

    bool def = false;
    if (cd->parent->isModule() && cd->getModule() == gIR->dmodule) {
        // interfaces don't have initializers
        if (!cd->isInterfaceDeclaration()) {
            ts->llvmInit->setInitializer(cd->llvmInitZ);
            cd->llvmVtbl->setInitializer(cd->llvmConstVtbl);

            // initialize interface vtables
            IRStruct* irstruct = cd->llvmIRStruct;
            std::vector<llvm::Constant*> infoInits;
            for (IRStruct::InterfaceIter i=irstruct->interfaces.begin(); i!=irstruct->interfaces.end(); ++i)
            {
                IRInterface* iri = i->second;
                iri->vtbl->setInitializer(iri->vtblInit);
                infoInits.push_back(iri->infoInit);
            }
            // initialize interface info array
            if (!infoInits.empty()) {
                llvm::Constant* arrInit = llvm::ConstantArray::get(irstruct->interfaceInfosTy, infoInits);
                irstruct->interfaceInfos->setInitializer(arrInit);
            }
        }
        def = true;
    }

    // generate classinfo
    if (def) DtoDefineClassInfo(cd);
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoCallClassDtors(TypeClass* tc, llvm::Value* instance)
{
    Array* arr = &tc->sym->dtors;
    for (size_t i=0; i<arr->dim; i++)
    {
        FuncDeclaration* fd = (FuncDeclaration*)arr->data[i];
        assert(fd->llvmValue);
        new llvm::CallInst(fd->llvmValue, instance, "", gIR->scopebb());
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoInitClass(TypeClass* tc, llvm::Value* dst)
{
    assert(gIR);

    assert(tc->llvmType);
    uint64_t size_t_size = gTargetData->getTypeSize(DtoSize_t());
    uint64_t n = gTargetData->getTypeSize(tc->llvmType->get()) - size_t_size;

    // set vtable field
    llvm::Value* vtblvar = DtoGEPi(dst,0,0,"tmp",gIR->scopebb());
    assert(tc->sym->llvmVtbl);
    new llvm::StoreInst(tc->sym->llvmVtbl, vtblvar, gIR->scopebb());

    // copy the static initializer
    if (n > 0) {
        assert(tc->llvmInit);
        assert(dst->getType() == tc->llvmInit->getType());

        llvm::Type* arrty = llvm::PointerType::get(llvm::Type::Int8Ty);

        llvm::Value* dstarr = new llvm::BitCastInst(dst,arrty,"tmp",gIR->scopebb());
        dstarr = DtoGEPi(dstarr,size_t_size,"tmp",gIR->scopebb());

        llvm::Value* srcarr = new llvm::BitCastInst(tc->llvmInit,arrty,"tmp",gIR->scopebb());
        srcarr = DtoGEPi(srcarr,size_t_size,"tmp",gIR->scopebb());

        llvm::Function* fn = LLVM_DeclareMemCpy32();
        std::vector<llvm::Value*> llargs;
        llargs.resize(4);
        llargs[0] = dstarr;
        llargs[1] = srcarr;
        llargs[2] = llvm::ConstantInt::get(llvm::Type::Int32Ty, n, false);
        llargs[3] = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0, false);

        new llvm::CallInst(fn, llargs.begin(), llargs.end(), "", gIR->scopebb());
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* DtoCastObjectToInterface(DValue* val, Type* _to)
{
    // call:
    // Object _d_dynamic_cast(Object o, ClassInfo c)

    llvm::Function* func = LLVM_D_GetRuntimeFunction(gIR->module, "_d_dynamic_cast");
    const llvm::FunctionType* funcTy = func->getFunctionType();

    std::vector<llvm::Value*> args;

    // Object o
    llvm::Value* tmp = val->getRVal();
    tmp = DtoBitCast(tmp, funcTy->getParamType(0));
    args.push_back(tmp);

    // ClassInfo c
    TypeClass* to = (TypeClass*)DtoDType(_to);
    DtoForceDeclareDsymbol(to->sym);
    assert(to->sym->llvmClass);
    args.push_back(to->sym->llvmClass);

    // call it
    llvm::Value* ret = gIR->ir->CreateCall(func, args.begin(), args.end(), "tmp");
    ret = DtoBitCast(ret, DtoType(_to));
    return new DImValue(_to, ret);
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* DtoCastInterfaceToObject(DValue* val)
{
    // call:
    // Object _d_toObject(void* p)

    llvm::Function* func = LLVM_D_GetRuntimeFunction(gIR->module, "_d_toObject");
    const llvm::FunctionType* funcTy = func->getFunctionType();

    // void* p
    llvm::Value* tmp = val->getRVal();
    tmp = DtoBitCast(tmp, funcTy->getParamType(0));

    // call it
    llvm::Value* ret = gIR->ir->CreateCall(func, tmp, "tmp");
    return new DImValue(ClassDeclaration::object->type, ret);
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoDeclareClassInfo(ClassDeclaration* cd)
{
    if (cd->llvmClassDeclared) return;
    cd->llvmClassDeclared = true;

    Logger::println("DtoDeclareClassInfo(%s)", cd->toChars());
    LOG_SCOPE;

    ClassDeclaration* cinfo = ClassDeclaration::classinfo;
    DtoResolveClass(cinfo);

    std::string gname("_D");
    gname.append(cd->mangle());
    if (!cd->isInterfaceDeclaration())
        gname.append("7__ClassZ");
    else
        gname.append("11__InterfaceZ");

    const llvm::Type* st = cinfo->type->llvmType->get();

    cd->llvmClass = new llvm::GlobalVariable(st, true, llvm::GlobalValue::ExternalLinkage, NULL, gname, gIR->module);
}

static llvm::Constant* build_offti_entry(VarDeclaration* vd)
{
    std::vector<const llvm::Type*> types;
    std::vector<llvm::Constant*> inits;

    types.push_back(DtoSize_t());

    size_t offset = vd->offset; // TODO might not be the true offset
    // dmd only accounts for the vtable, not classinfo or monitor
    if (global.params.is64bit)
        offset += 8;
    else
        offset += 4;
    inits.push_back(DtoConstSize_t(offset));

    vd->type->getTypeInfo(NULL);
    assert(vd->type->vtinfo);
    DtoForceDeclareDsymbol(vd->type->vtinfo);
    llvm::Constant* c = isaConstant(vd->type->vtinfo->llvmValue);

    const llvm::Type* tiTy = llvm::PointerType::get(Type::typeinfo->type->llvmType->get());
    Logger::cout() << "tiTy = " << *tiTy << '\n';

    types.push_back(tiTy);
    inits.push_back(llvm::ConstantExpr::getBitCast(c, tiTy));

    const llvm::StructType* sTy = llvm::StructType::get(types);
    return llvm::ConstantStruct::get(sTy, inits);
}

static llvm::Constant* build_offti_array(ClassDeclaration* cd, llvm::Constant* init)
{
    const llvm::StructType* initTy = isaStruct(init->getType());
    assert(initTy);

    std::vector<llvm::Constant*> arrayInits;
    for (ClassDeclaration *cd2 = cd; cd2; cd2 = cd2->baseClass)
    {
    if (cd2->members)
    {
        for (size_t i = 0; i < cd2->members->dim; i++)
        {
        Dsymbol *sm = (Dsymbol *)cd2->members->data[i];
        if (VarDeclaration* vd = sm->isVarDeclaration()) // is this enough?
        {
            llvm::Constant* c = build_offti_entry(vd);
            assert(c);
            arrayInits.push_back(c);
        }
        }
    }
    }

    size_t ninits = arrayInits.size();
    llvm::Constant* size = DtoConstSize_t(ninits);
    llvm::Constant* ptr;

    if (ninits > 0) {
        // OffsetTypeInfo type
        std::vector<const llvm::Type*> elemtypes;
        elemtypes.push_back(DtoSize_t());
        const llvm::Type* tiTy = llvm::PointerType::get(Type::typeinfo->type->llvmType->get());
        elemtypes.push_back(tiTy);
        const llvm::StructType* sTy = llvm::StructType::get(elemtypes);

        // array type
        const llvm::ArrayType* arrTy = llvm::ArrayType::get(sTy, ninits);
        llvm::Constant* arrInit = llvm::ConstantArray::get(arrTy, arrayInits);

        std::string name(cd->type->vtinfo->toChars());
        name.append("__OffsetTypeInfos");
        llvm::GlobalVariable* gvar = new llvm::GlobalVariable(arrTy,true,llvm::GlobalValue::InternalLinkage,arrInit,name,gIR->module);
        ptr = llvm::ConstantExpr::getBitCast(gvar, llvm::PointerType::get(sTy));
    }
    else {
        ptr = llvm::ConstantPointerNull::get(isaPointer(initTy->getElementType(1)));
    }

    return DtoConstSlice(size, ptr);
}

static llvm::Constant* build_class_dtor(ClassDeclaration* cd)
{
    // construct the function
    std::vector<const llvm::Type*> paramTypes;
    paramTypes.push_back(llvm::PointerType::get(cd->type->llvmType->get()));

    const llvm::FunctionType* fnTy = llvm::FunctionType::get(llvm::Type::VoidTy, paramTypes, false);

    if (cd->dtors.dim == 0) {
        return llvm::ConstantPointerNull::get(llvm::PointerType::get(llvm::Type::Int8Ty));
    }
    else if (cd->dtors.dim == 1) {
        DtorDeclaration *d = (DtorDeclaration *)cd->dtors.data[0];
        DtoForceDeclareDsymbol(d);
        assert(d->llvmValue);
        return llvm::ConstantExpr::getBitCast(isaConstant(d->llvmValue), llvm::PointerType::get(llvm::Type::Int8Ty));
    }

    std::string gname("_D");
    gname.append(cd->mangle());
    gname.append("12__destructorMFZv");

    llvm::Function* func = new llvm::Function(fnTy, llvm::GlobalValue::InternalLinkage, gname, gIR->module);
    llvm::Value* thisptr = func->arg_begin();
    thisptr->setName("this");

    llvm::BasicBlock* bb = new llvm::BasicBlock("entry", func);
    LLVMBuilder builder(bb);

    for (size_t i = 0; i < cd->dtors.dim; i++)
    {
        DtorDeclaration *d = (DtorDeclaration *)cd->dtors.data[i];
        DtoForceDeclareDsymbol(d);
        assert(d->llvmValue);
        builder.CreateCall(d->llvmValue, thisptr);
    }
    builder.CreateRetVoid();

    return llvm::ConstantExpr::getBitCast(func, llvm::PointerType::get(llvm::Type::Int8Ty));
}

static uint build_classinfo_flags(ClassDeclaration* cd)
{
    // adapted from original dmd code
    uint flags = 0;
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
        if (sm->isVarDeclaration()) // is this enough?
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

    if (cd->llvmClassDefined) return;
    cd->llvmClassDefined = true;

    Logger::println("DtoDefineClassInfo(%s)", cd->toChars());
    LOG_SCOPE;

    assert(cd->type->ty == Tclass);
    assert(cd->llvmClass);

    TypeClass* cdty = (TypeClass*)cd->type;
    if (!cd->isInterfaceDeclaration()) {
        assert(cd->llvmInitZ);
        assert(cd->llvmVtbl);
        assert(cd->llvmConstVtbl);
        assert(cdty->llvmInit);
    }

    // holds the list of initializers for llvm
    std::vector<llvm::Constant*> inits;

    ClassDeclaration* cinfo = ClassDeclaration::classinfo;
    DtoForceConstInitDsymbol(cinfo);
    assert(cinfo->llvmInitZ);

    llvm::Constant* c;

    // own vtable
    c = cinfo->llvmInitZ->getOperand(0);
    assert(c);
    inits.push_back(c);

    // monitor
    // TODO no monitors yet

    // byte[] init
    const llvm::Type* byteptrty = llvm::PointerType::get(llvm::Type::Int8Ty);
    if (cd->isInterfaceDeclaration()) {
        c = cinfo->llvmInitZ->getOperand(1);
    }
    else {
        c = llvm::ConstantExpr::getBitCast(cdty->llvmInit, byteptrty);
        assert(!cd->llvmInitZ->getType()->isAbstract());
        size_t initsz = gTargetData->getTypeSize(cd->llvmInitZ->getType());
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
    if (cd->isInterfaceDeclaration()) {
        c = cinfo->llvmInitZ->getOperand(3);
    }
    else {
        const llvm::Type* byteptrptrty = llvm::PointerType::get(byteptrty);
        assert(!cd->llvmVtbl->getType()->isAbstract());
        c = llvm::ConstantExpr::getBitCast(cd->llvmVtbl, byteptrptrty);
        assert(!cd->llvmConstVtbl->getType()->isAbstract());
        size_t vtblsz = cd->llvmConstVtbl->getType()->getNumElements();
        c = DtoConstSlice(DtoConstSize_t(vtblsz), c);
    }
    inits.push_back(c);

    // interfaces array
    IRStruct* irstruct = cd->llvmIRStruct;
    if (cd->isInterfaceDeclaration() || !irstruct->interfaceInfos) {
        c = cinfo->llvmInitZ->getOperand(4);
    }
    else {
        const llvm::Type* t = cinfo->llvmInitZ->getOperand(4)->getType()->getContainedType(1);
        c = llvm::ConstantExpr::getBitCast(irstruct->interfaceInfos, t);
        size_t iisz = irstruct->interfaceInfosTy->getNumElements();
        c = DtoConstSlice(DtoConstSize_t(iisz), c);
    }
    inits.push_back(c);

    // base classinfo
    if (cd->baseClass && !cd->isInterfaceDeclaration()) {
        DtoDeclareClassInfo(cd->baseClass);
        c = cd->baseClass->llvmClass;
        assert(c);
        inits.push_back(c);
    }
    else {
        // null
        c = cinfo->llvmInitZ->getOperand(5);
        inits.push_back(c);
    }

    // destructor
    if (cd->isInterfaceDeclaration()) {
        c = cinfo->llvmInitZ->getOperand(6);
    }
    else {
        c = build_class_dtor(cd);
    }
    inits.push_back(c);

    // invariant
    // TODO
    c = cinfo->llvmInitZ->getOperand(7);
    inits.push_back(c);

    // uint flags
    if (cd->isInterfaceDeclaration()) {
        c = cinfo->llvmInitZ->getOperand(8);
    }
    else {
        uint flags = build_classinfo_flags(cd);
        c = DtoConstUint(flags);
    }
    inits.push_back(c);

    // allocator
    // TODO
    c = cinfo->llvmInitZ->getOperand(9);
    inits.push_back(c);

    // offset typeinfo
    if (cd->isInterfaceDeclaration()) {
        c = cinfo->llvmInitZ->getOperand(10);
    }
    else {
        c = build_offti_array(cd, cinfo->llvmInitZ->getOperand(10));
    }
    inits.push_back(c);

    // default constructor
    if (cd->defaultCtor && !cd->isInterfaceDeclaration()) {
        DtoForceDeclareDsymbol(cd->defaultCtor);
        c = isaConstant(cd->defaultCtor->llvmValue);
        //const llvm::Type* toTy = cinfo->llvmInitZ->getOperand(11)->getType();
        c = llvm::ConstantExpr::getBitCast(c, llvm::PointerType::get(llvm::Type::Int8Ty)); // toTy);
    }
    else {
        c = cinfo->llvmInitZ->getOperand(11);
    }
    inits.push_back(c);

    /*size_t n = inits.size();
    for (size_t i=0; i<n; ++i)
    {
        Logger::cout() << "inits[" << i << "]: " << *inits[i] << '\n';
    }*/

    // build the initializer
    const llvm::StructType* st = isaStruct(cinfo->llvmInitZ->getType());
    llvm::Constant* finalinit = llvm::ConstantStruct::get(st, inits);
    //Logger::cout() << "built the classinfo initializer:\n" << *finalinit <<'\n';

    cd->llvmClassZ = finalinit;
    cd->llvmClass->setInitializer(finalinit);
}
