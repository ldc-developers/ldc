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

        bc->base->toObjFile();

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

void DtoDeclareClass(ClassDeclaration* cd)
{
    if (cd->llvmTouched) return;
    cd->llvmTouched = true;

    Logger::println("DtoDeclareClass(%s)\n", cd->toPrettyChars());
    LOG_SCOPE;

    assert(cd->type->ty == Tclass);
    TypeClass* ts = (TypeClass*)cd->type;

    assert(!cd->llvmIRStruct);
    IRStruct* irstruct = new IRStruct(ts);
    cd->llvmIRStruct = irstruct;

    gIR->structs.push_back(irstruct);
    gIR->classes.push_back(cd);

    // add vtable
    llvm::PATypeHolder pa = llvm::OpaqueType::get();
    const llvm::Type* vtabty = llvm::PointerType::get(pa);

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
    {
    llvm::PATypeHolder& spa = irstruct->recty;
    llvm::cast<llvm::OpaqueType>(spa.get())->refineAbstractTypeTo(structtype);
    structtype = isaStruct(spa.get());
    }

    // create the type
    ts->llvmType = new llvm::PATypeHolder(structtype);

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
    std::vector<llvm::Constant*> sinits;
    std::vector<const llvm::Type*> sinits_ty;
    sinits.reserve(cd->vtbl.dim);
    sinits_ty.reserve(cd->vtbl.dim);

    for (int k=0; k < cd->vtbl.dim; k++)
    {
        Dsymbol* dsym = (Dsymbol*)cd->vtbl.data[k];
        assert(dsym);
        //Logger::cout() << "vtblsym: " << dsym->toChars() << '\n';

        if (FuncDeclaration* fd = dsym->isFuncDeclaration()) {
            fd->toObjFile();
            assert(fd->llvmValue);
            llvm::Constant* c = llvm::cast<llvm::Constant>(fd->llvmValue);
            sinits.push_back(c);
            sinits_ty.push_back(c->getType());
        }
        else if (ClassDeclaration* cd = dsym->isClassDeclaration()) {
            const llvm::Type* cty = llvm::PointerType::get(llvm::Type::Int8Ty);
            llvm::Constant* c = llvm::Constant::getNullValue(cty);
            sinits.push_back(c);
            sinits_ty.push_back(cty);
        }
        else
        assert(0);
    }

    const llvm::StructType* svtbl_ty = 0;
    if (!sinits.empty())
    {
        llvm::GlobalValue::LinkageTypes _linkage = llvm::GlobalValue::ExternalLinkage;

        std::string varname("_D");
        varname.append(cd->mangle());
        varname.append("6__vtblZ");

        std::string styname(cd->mangle());
        styname.append("__vtblTy");

        svtbl_ty = llvm::StructType::get(sinits_ty);
        gIR->module->addTypeName(styname, svtbl_ty);
        svtblVar = new llvm::GlobalVariable(svtbl_ty, true, _linkage, 0, varname, gIR->module);

        cd->llvmConstVtbl = llvm::cast<llvm::ConstantStruct>(llvm::ConstantStruct::get(svtbl_ty, sinits));
        if (needs_definition)
            svtblVar->setInitializer(cd->llvmConstVtbl);
        cd->llvmVtbl = svtblVar;
    }

    // refine for final vtable type
    llvm::cast<llvm::OpaqueType>(pa.get())->refineAbstractTypeTo(svtbl_ty);

    std::string initname("_D");
    initname.append(cd->mangle());
    initname.append("6__initZ");
    llvm::GlobalValue::LinkageTypes _linkage = llvm::GlobalValue::ExternalLinkage;
    llvm::GlobalVariable* initvar = new llvm::GlobalVariable(ts->llvmType->get(), true, _linkage, NULL, initname, gIR->module);
    ts->llvmInit = initvar;

    gIR->classes.pop_back();
    gIR->structs.pop_back();

    gIR->constInitQueue.push_back(cd);
    if (needs_definition)
    gIR->defineQueue.push_back(cd);
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoConstInitClass(ClassDeclaration* cd)
{
    IRStruct* irstruct = cd->llvmIRStruct;
    if (irstruct->constinited) return;
    irstruct->constinited = true;

    Logger::println("DtoConstInitClass(%s)\n", cd->toPrettyChars());
    LOG_SCOPE;

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
        fieldinits.push_back(i->second.init);
    }

    // get the struct (class) type
    assert(cd->type->ty == Tclass);
    TypeClass* ts = (TypeClass*)cd->type;
    const llvm::StructType* structtype = isaStruct(ts->llvmType->get());

    // generate initializer
    Logger::cout() << cd->toPrettyChars() << " | " << *structtype << '\n';
    Logger::println("%u %u fields", structtype->getNumElements(), fieldinits.size());
    llvm::Constant* _init = llvm::ConstantStruct::get(structtype, fieldinits);
    assert(_init);
    cd->llvmInitZ = _init;

    gIR->classes.pop_back();
    gIR->structs.pop_back();
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoDefineClass(ClassDeclaration* cd)
{
    IRStruct* irstruct = cd->llvmIRStruct;
    if (irstruct->defined) return;
    irstruct->defined = true;

    Logger::println("DtoDefineClass(%s)\n", cd->toPrettyChars());
    LOG_SCOPE;

    // get the struct (class) type
    assert(cd->type->ty == Tclass);
    TypeClass* ts = (TypeClass*)cd->type;

    bool def = false;
    if (cd->parent->isModule() && cd->getModule() == gIR->dmodule) {
        ts->llvmInit->setInitializer(cd->llvmInitZ);
        def = true;
    }

    // generate classinfo
    DtoDeclareClassInfo(cd);
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
    if (cd->llvmClass)
        return;

    Logger::println("CLASS INFO DECLARATION: %s", cd->toChars());
    LOG_SCOPE;

    ClassDeclaration* cinfo = ClassDeclaration::classinfo;
    cinfo->toObjFile();

    const llvm::Type* st = cinfo->type->llvmType->get();

    std::string gname("_D");
    gname.append(cd->mangle());
    gname.append("7__ClassZ");

    cd->llvmClass = new llvm::GlobalVariable(st, true, llvm::GlobalValue::ExternalLinkage, NULL, gname, gIR->module);
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

    if (cd->llvmClassZ)
        return;

    Logger::println("CLASS INFO DEFINITION: %s", cd->toChars());
    LOG_SCOPE;
    assert(cd->llvmClass);

    // holds the list of initializers for llvm
    std::vector<llvm::Constant*> inits;

    ClassDeclaration* cinfo = ClassDeclaration::classinfo;
    DtoConstInitClass(cinfo);
    assert(cinfo->llvmInitZ);

    llvm::Constant* c;

    // own vtable
    c = cinfo->llvmInitZ->getOperand(0);
    assert(c);
    inits.push_back(c);

    // monitor
    // TODO no monitors yet

    // initializer
    c = cinfo->llvmInitZ->getOperand(1);
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
    c = cinfo->llvmInitZ->getOperand(3);
    inits.push_back(c);

    // interfaces array
    c = cinfo->llvmInitZ->getOperand(4);
    inits.push_back(c);

    // base classinfo
    c = cinfo->llvmInitZ->getOperand(5);
    inits.push_back(c);

    // destructor
    c = cinfo->llvmInitZ->getOperand(6);
    inits.push_back(c);

    // invariant
    c = cinfo->llvmInitZ->getOperand(7);
    inits.push_back(c);

    // flags
    c = cinfo->llvmInitZ->getOperand(8);
    inits.push_back(c);

    // allocator
    c = cinfo->llvmInitZ->getOperand(9);
    inits.push_back(c);

    // offset typeinfo
    c = cinfo->llvmInitZ->getOperand(10);
    inits.push_back(c);

    // default constructor
    c = cinfo->llvmInitZ->getOperand(11);
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
