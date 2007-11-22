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

//////////////////////////////////////////////////////////////////////////////////////////

static void LLVM_AddBaseClassData(BaseClasses* bcs)
{
    // add base class data members first
    for (int j=0; j<bcs->dim; j++)
    {
        BaseClass* bc = (BaseClass*)(bcs->data[j]);
        assert(bc);
        Logger::println("Adding base class members of %s", bc->base->toChars());
        LOG_SCOPE;

        LLVM_AddBaseClassData(&bc->base->baseclasses);
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
    // resolve typeinfo
    //DtoResolveClass(ClassDeclaration::typeinfo);
    // resolve classinfo
    //DtoResolveClass(ClassDeclaration::classinfo);

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

    bool needs_definition = false;
    if (cd->parent->isModule()) {
        gIR->module->addTypeName(cd->mangle(), ts->llvmType->get());
        needs_definition = (cd->getModule() == gIR->dmodule);
    }
    else {
        assert(0 && "class parent is not a module");
    }

    // generate vtable
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
        else if (ClassDeclaration* cd = dsym->isClassDeclaration()) {
            //Logger::println("*** ClassDeclaration in vtable: %s", cd->toChars());
            const llvm::Type* cinfoty;
            if (cd != ClassDeclaration::classinfo) {
                cd = ClassDeclaration::classinfo;
                DtoResolveClass(cd);
                cinfoty = cd->type->llvmType->get();
            }
            else {
                cinfoty = ts->llvmType->get();
            }
            const llvm::Type* cty = llvm::PointerType::get(cd->type->llvmType->get());
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

    // vtable
    std::string varname("_D");
    varname.append(cd->mangle());
    varname.append("6__vtblZ");

    std::string styname(cd->mangle());
    styname.append("__vtblTy");

    llvm::GlobalValue::LinkageTypes _linkage = llvm::GlobalValue::ExternalLinkage;

    const llvm::StructType* svtbl_ty = isaStruct(ts->llvmVtblType->get());
    cd->llvmVtbl = new llvm::GlobalVariable(svtbl_ty, true, _linkage, 0, varname, gIR->module);

    // init
    std::string initname("_D");
    initname.append(cd->mangle());
    initname.append("6__initZ");

    llvm::GlobalVariable* initvar = new llvm::GlobalVariable(ts->llvmType->get(), true, _linkage, NULL, initname, gIR->module);
    ts->llvmInit = initvar;

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
    /*Logger::cout() << cd->toPrettyChars() << " | " << *structtype << '\n';

    for(size_t i=0; i<structtype->getNumElements(); ++i) {
        Logger::cout() << "s#" << i << " = " << *structtype->getElementType(i) << '\n';
    }

    for(size_t i=0; i<fieldinits.size(); ++i) {
        Logger::cout() << "i#" << i << " = " << *fieldinits[i]->getType() << '\n';
    }*/

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

    /*for (size_t i=0; i< sinits.size(); ++i)
    {
        Logger::cout() << "field[" << i << "] = " << *svtbl_ty->getElementType(i) << '\n';
        Logger::cout() << "init [" << i << "] = " << *sinits[i]->getType() << '\n';
        assert(svtbl_ty->getElementType(i) == sinits[i]->getType());
    }*/

    llvm::Constant* cvtblInit = llvm::ConstantStruct::get(svtbl_ty, sinits);
    cd->llvmConstVtbl = llvm::cast<llvm::ConstantStruct>(cvtblInit);

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
        ts->llvmInit->setInitializer(cd->llvmInitZ);
        cd->llvmVtbl->setInitializer(cd->llvmConstVtbl);
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
    gname.append("7__ClassZ");

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
        if (VarDeclaration* vd = sm->isVarDeclaration())
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
    assert(cd->llvmInitZ);
    assert(cd->llvmVtbl);
    assert(cd->llvmConstVtbl);

    TypeClass* cdty = (TypeClass*)cd->type;
    assert(cdty->llvmInit);

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
    c = llvm::ConstantExpr::getBitCast(cdty->llvmInit, byteptrty);
    assert(!cd->llvmInitZ->getType()->isAbstract());
    size_t initsz = gTargetData->getTypeSize(cd->llvmInitZ->getType());
    c = DtoConstSlice(DtoConstSize_t(initsz), c);
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
    const llvm::Type* byteptrptrty = llvm::PointerType::get(byteptrty);
    assert(!cd->llvmVtbl->getType()->isAbstract());
    c = llvm::ConstantExpr::getBitCast(cd->llvmVtbl, byteptrptrty);
    assert(!cd->llvmConstVtbl->getType()->isAbstract());
    size_t vtblsz = gTargetData->getTypeSize(cd->llvmConstVtbl->getType());
    c = DtoConstSlice(DtoConstSize_t(vtblsz), c);
    inits.push_back(c);

    // interfaces array
    // TODO
    c = cinfo->llvmInitZ->getOperand(4);
    inits.push_back(c);

    // base classinfo
    if (cd->baseClass) {
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
    // TODO
    c = cinfo->llvmInitZ->getOperand(6);
    inits.push_back(c);

    // invariant
    // TODO
    c = cinfo->llvmInitZ->getOperand(7);
    inits.push_back(c);

    // uint flags, adapted from original dmd code
    uint flags = 0;
    //flags |= 4; // has offTi
    //flags |= isCOMclass(); // IUnknown
    if (cd->ctor) flags |= 8;
    for (ClassDeclaration *cd2 = cd; cd2; cd2 = cd2->baseClass)
    {
    if (cd2->members)
    {
        for (size_t i = 0; i < cd2->members->dim; i++)
        {
        Dsymbol *sm = (Dsymbol *)cd2->members->data[i];
        //printf("sm = %s %s\n", sm->kind(), sm->toChars());
        if (sm->hasPointers())
            goto L2;
        }
    }
    }
    flags |= 2;         // no pointers
L2:
    c = DtoConstUint(flags);
    inits.push_back(c);

    // allocator
    // TODO
    c = cinfo->llvmInitZ->getOperand(9);
    inits.push_back(c);

    // offset typeinfo
    c = build_offti_array(cd, cinfo->llvmInitZ->getOperand(10));
    inits.push_back(c);

    // default constructor
    if (cd->defaultCtor) {
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
