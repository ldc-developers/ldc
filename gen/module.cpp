#include "gen/llvm.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Module.h"
#include "llvm/LinkAllPasses.h"

#include "mars.h"
#include "module.h"
#include "mtype.h"
#include "declaration.h"
#include "statement.h"
#include "enum.h"
#include "aggregate.h"
#include "init.h"
#include "attrib.h"
#include "id.h"
#include "import.h"
#include "template.h"
#include "scope.h"

#include "gen/abi.h"
#include "gen/arrays.h"
#include "gen/classes.h"
#include "gen/functions.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/programs.h"
#include "gen/rttibuilder.h"
#include "gen/runtime.h"
#include "gen/structs.h"
#include "gen/todebug.h"
#include "gen/tollvm.h"

#include "ir/irvar.h"
#include "ir/irmodule.h"
#include "ir/irtype.h"

#if DMDV2
#define NEW_MODULEINFO_LAYOUT 1
#endif


static llvm::Function* build_module_function(const std::string &name, const std::list<FuncDeclaration*> &funcs,
                                             const std::list<VarDeclaration*> &gates = std::list<VarDeclaration*>())
{
    if (gates.empty()) {
        if (funcs.empty())
            return NULL;

        if (funcs.size() == 1)
            return funcs.front()->ir.irFunc->func;
    }

    std::vector<LLType*> argsTy;
    LLFunctionType* fnTy = LLFunctionType::get(LLType::getVoidTy(gIR->context()),argsTy,false);
    assert(gIR->module->getFunction(name) == NULL);
    llvm::Function* fn = llvm::Function::Create(fnTy, llvm::GlobalValue::InternalLinkage, name, gIR->module);
    fn->setCallingConv(DtoCallingConv(0, LINKd));

    llvm::BasicBlock* bb = llvm::BasicBlock::Create(gIR->context(), "entry", fn);
    IRBuilder<> builder(bb);

    // debug info
    DtoDwarfSubProgramInternal(name.c_str(), name.c_str());

    // Call ctor's
    typedef std::list<FuncDeclaration*>::const_iterator FuncIterator;
    for (FuncIterator itr = funcs.begin(), end = funcs.end(); itr != end; ++itr) {
        llvm::Function* f = (*itr)->ir.irFunc->func;
        llvm::CallInst* call = builder.CreateCall(f,"");
        call->setCallingConv(DtoCallingConv(0, LINKd));
    }

    // Increment vgate's
    typedef std::list<VarDeclaration*>::const_iterator GatesIterator;
    for (GatesIterator itr = gates.begin(), end = gates.end(); itr != end; ++itr) {
        assert((*itr)->ir.irGlobal);
        llvm::Value* val = (*itr)->ir.irGlobal->value;
        llvm::Value* rval = builder.CreateLoad(val, "vgate");
        llvm::Value* res = builder.CreateAdd(rval, DtoConstUint(1), "vgate");
        builder.CreateStore(res, val);
    }

    builder.CreateRetVoid();
    return fn;
}

// build module ctor

llvm::Function* build_module_ctor()
{
    std::string name("_D");
    name.append(gIR->dmodule->mangle());
    name.append("6__ctorZ");
#if DMDV2
    return build_module_function(name, gIR->ctors, gIR->gates);
#else
    return build_module_function(name, gIR->ctors);
#endif
}

// build module dtor

static llvm::Function* build_module_dtor()
{
    std::string name("_D");
    name.append(gIR->dmodule->mangle());
    name.append("6__dtorZ");
    return build_module_function(name, gIR->dtors);
}

// build module unittest

static llvm::Function* build_module_unittest()
{
    std::string name("_D");
    name.append(gIR->dmodule->mangle());
    name.append("10__unittestZ");
    return build_module_function(name, gIR->unitTests);
}

#if DMDV2

// build module shared ctor

llvm::Function* build_module_shared_ctor()
{
    std::string name("_D");
    name.append(gIR->dmodule->mangle());
    name.append("13__shared_ctorZ");
    return build_module_function(name, gIR->sharedCtors, gIR->sharedGates);
}

// build module shared dtor

static llvm::Function* build_module_shared_dtor()
{
    std::string name("_D");
    name.append(gIR->dmodule->mangle());
    name.append("13__shared_dtorZ");
    return build_module_function(name, gIR->sharedDtors);
}

#endif

// build ModuleReference and register function, to register the module info in the global linked list
static LLFunction* build_module_reference_and_ctor(LLConstant* moduleinfo)
{
    // build ctor type
    LLFunctionType* fty = LLFunctionType::get(LLType::getVoidTy(gIR->context()), std::vector<LLType*>(), false);

    // build ctor name
    std::string fname = "_D";
    fname += gIR->dmodule->mangle();
    fname += "16__moduleinfoCtorZ";

    // build a function that registers the moduleinfo in the global moduleinfo linked list
    LLFunction* ctor = LLFunction::Create(fty, LLGlobalValue::InternalLinkage, fname, gIR->module);

    // provide the default initializer
    LLStructType* modulerefTy = DtoModuleReferenceType();
    std::vector<LLConstant*> mrefvalues;
    mrefvalues.push_back(LLConstant::getNullValue(modulerefTy->getContainedType(0)));
    mrefvalues.push_back(llvm::ConstantExpr::getBitCast(moduleinfo, modulerefTy->getContainedType(1)));
    LLConstant* thismrefinit = LLConstantStruct::get(modulerefTy, mrefvalues);

    // create the ModuleReference node for this module
    std::string thismrefname = "_D";
    thismrefname += gIR->dmodule->mangle();
    thismrefname += "11__moduleRefZ";
    LLGlobalVariable* thismref = new LLGlobalVariable(*gIR->module, modulerefTy, false, LLGlobalValue::InternalLinkage, thismrefinit, thismrefname);

    // make sure _Dmodule_ref is declared
    LLConstant* mref = gIR->module->getNamedGlobal("_Dmodule_ref");
    LLType *modulerefPtrTy = getPtrToType(modulerefTy);
    if (!mref)
        mref = new LLGlobalVariable(*gIR->module, modulerefPtrTy, false, LLGlobalValue::ExternalLinkage, NULL, "_Dmodule_ref");
    mref = DtoBitCast(mref, getPtrToType(modulerefPtrTy));

    // make the function insert this moduleinfo as the beginning of the _Dmodule_ref linked list
    llvm::BasicBlock* bb = llvm::BasicBlock::Create(gIR->context(), "moduleinfoCtorEntry", ctor);
    IRBuilder<> builder(bb);

    // debug info
    llvm::DISubprogram subprog = DtoDwarfSubProgramInternal(fname.c_str(), fname.c_str());

    // get current beginning
    LLValue* curbeg = builder.CreateLoad(mref, "current");

    // put current beginning as the next of this one
    LLValue* gep = builder.CreateStructGEP(thismref, 0, "next");
    builder.CreateStore(curbeg, gep);

    // replace beginning
    builder.CreateStore(thismref, mref);

    // return
    builder.CreateRetVoid();

    return ctor;
}

llvm::Module* Module::genLLVMModule(llvm::LLVMContext& context, Ir* sir)
{
    bool logenabled = Logger::enabled();
    if (llvmForceLogging && !logenabled)
    {
        Logger::enable();
    }

    Logger::println("Generating module: %s\n", (md ? md->toChars() : toChars()));
    LOG_SCOPE;

    if (global.params.verbose_cg)
        printf("codegen: %s (%s)\n", toPrettyChars(), srcfile->toChars());

    assert(!global.errors);

    // name the module
    llvm::StringRef mname(toChars());
    if (md != 0)
        mname = md->toChars();

    // create a new ir state
    // TODO look at making the instance static and moving most functionality into IrModule where it belongs
    IRState ir(new llvm::Module(mname, context));
    gIR = &ir;
    ir.dmodule = this;

    // reset all IR data stored in Dsymbols
    IrDsymbol::resetAll();

    sir->setState(&ir);

    // set target triple
    ir.module->setTargetTriple(global.params.targetTriple);

    // set final data layout
    ir.module->setDataLayout(global.params.dataLayout);
    if (Logger::enabled())
        Logger::cout() << "Final data layout: " << global.params.dataLayout << '\n';

    // allocate the target abi
    gABI = TargetABI::getTarget();

    // debug info
    DtoDwarfCompileUnit(this);

    // handle invalid 'objectø module
    if (!ClassDeclaration::object) {
        error("is missing 'class Object'");
        fatal();
    }
    if (!ClassDeclaration::classinfo) {
        error("is missing 'class ClassInfo'");
        fatal();
    }

    LLVM_D_InitRuntime();

    // process module members
    for (unsigned k=0; k < members->dim; k++) {
        Dsymbol* dsym = (Dsymbol*)(members->data[k]);
        assert(dsym);
        dsym->codegen(sir);
    }

    // emit function bodies
    sir->emitFunctionBodies();

    // for singleobj-compilation, fully emit all seen template instances
    if (global.params.singleObj)
    {
        while (!ir.seenTemplateInstances.empty())
        {
            IRState::TemplateInstanceSet::iterator it, end = ir.seenTemplateInstances.end();
            for (it = ir.seenTemplateInstances.begin(); it != end; ++it)
                (*it)->codegen(sir);
            ir.seenTemplateInstances.clear();

            // emit any newly added function bodies
            sir->emitFunctionBodies();
        }
    }

    // finilize debug info
    DtoDwarfModuleEnd();

    // generate ModuleInfo
    genmoduleinfo();

    // verify the llvm
    if (!global.params.noVerify) {
        std::string verifyErr;
        Logger::println("Verifying module...");
        LOG_SCOPE;
        if (llvm::verifyModule(*ir.module,llvm::ReturnStatusAction,&verifyErr))
        {
            error("%s", verifyErr.c_str());
            fatal();
        }
        else {
            Logger::println("Verification passed!");
        }
    }

    gIR = NULL;

    if (llvmForceLogging && !logenabled)
    {
        Logger::disable();
    }

    sir->setState(NULL);

    return ir.module;
}

llvm::GlobalVariable* Module::moduleInfoSymbol()
{
    // create name
    std::string MIname("_D");
    MIname.append(mangle());
    MIname.append("8__ModuleZ");

    if (gIR->dmodule != this) {
        LLType* moduleinfoTy = DtoType(moduleinfo->type);
        LLGlobalVariable *var = gIR->module->getGlobalVariable(MIname);
        if (!var)
            var = new llvm::GlobalVariable(*gIR->module, moduleinfoTy, false, llvm::GlobalValue::ExternalLinkage, NULL, MIname);
        return var;
    }

    if (moduleInfoVar)
        return moduleInfoVar;

    // declare global
    // flags will be modified at runtime so can't make it constant
    moduleInfoVar = new llvm::GlobalVariable(*gIR->module, moduleInfoType, false, llvm::GlobalValue::ExternalLinkage, NULL, MIname);

    return moduleInfoVar;
}

// Put out instance of ModuleInfo for this Module
void Module::genmoduleinfo()
{
    // resolve ModuleInfo
    if (!moduleinfo)
    {
        error("object.d is missing the ModuleInfo class");
        fatal();
    }
    // check for patch
    else
    {
#if DMDV2
        unsigned sizeof_ModuleInfo = 16 * PTRSIZE;
#else
        unsigned sizeof_ModuleInfo = 14 * PTRSIZE;
#endif
        if (sizeof_ModuleInfo != moduleinfo->structsize)
        {
            error("object.d ModuleInfo class is incorrect");
            fatal();
        }
    }

    // use the RTTIBuilder
    RTTIBuilder b(moduleinfo);

    // some types
    LLType* moduleinfoTy = moduleinfo->type->irtype->getType();
    LLType* classinfoTy = ClassDeclaration::classinfo->type->irtype->getType();

    // importedModules[]
    std::vector<LLConstant*> importInits;
    LLConstant* importedModules = 0;
    llvm::ArrayType* importedModulesTy = 0;
    for (size_t i = 0; i < aimports.dim; i++)
    {
        Module *m = (Module *)aimports.data[i];
        if (!m->needModuleInfo() || m == this)
            continue;

        // declare the imported module info
        std::string m_name("_D");
        m_name.append(m->mangle());
        m_name.append("8__ModuleZ");
        llvm::GlobalVariable* m_gvar = gIR->module->getGlobalVariable(m_name);
        if (!m_gvar) m_gvar = new llvm::GlobalVariable(*gIR->module, moduleinfoTy, false, llvm::GlobalValue::ExternalLinkage, NULL, m_name);
        importInits.push_back(m_gvar);
    }
    // has import array?
    if (!importInits.empty())
    {
        importedModulesTy = llvm::ArrayType::get(getPtrToType(moduleinfoTy), importInits.size());
        importedModules = LLConstantArray::get(importedModulesTy, importInits);
    }

    // localClasses[]
    LLConstant* localClasses = 0;
    llvm::ArrayType* localClassesTy = 0;
    ClassDeclarations aclasses;
    //printf("members->dim = %d\n", members->dim);
    for (size_t i = 0; i < members->dim; i++)
    {
        Dsymbol *member;

        member = (Dsymbol *)members->data[i];
        //printf("\tmember '%s'\n", member->toChars());
        member->addLocalClass(&aclasses);
    }
    // fill inits
    std::vector<LLConstant*> classInits;
    for (size_t i = 0; i < aclasses.dim; i++)
    {
        ClassDeclaration* cd = (ClassDeclaration*)aclasses.data[i];
        cd->codegen(Type::sir);

        if (cd->isInterfaceDeclaration())
        {
            Logger::println("skipping interface '%s' in moduleinfo", cd->toPrettyChars());
            continue;
        }
        else if (cd->sizeok != 1)
        {
            Logger::println("skipping opaque class declaration '%s' in moduleinfo", cd->toPrettyChars());
            continue;
        }
        Logger::println("class: %s", cd->toPrettyChars());
        LLConstant *c = DtoBitCast(cd->ir.irStruct->getClassInfoSymbol(), getPtrToType(classinfoTy));
        classInits.push_back(c);
    }
    // has class array?
    if (!classInits.empty())
    {
        localClassesTy = llvm::ArrayType::get(getPtrToType(classinfoTy), classInits.size());
        localClasses = LLConstantArray::get(localClassesTy, classInits);
    }

#if NEW_MODULEINFO_LAYOUT

    // These must match the values in druntime/src/object_.d
    #define MIstandalone      4
    #define MItlsctor         8
    #define MItlsdtor         0x10
    #define MIctor            0x20
    #define MIdtor            0x40
    #define MIxgetMembers     0x80
    #define MIictor           0x100
    #define MIunitTest        0x200
    #define MIimportedModules 0x400
    #define MIlocalClasses    0x800
    #define MInew             0x80000000   // it's the "new" layout

    llvm::Function* fsharedctor = build_module_shared_ctor();
    llvm::Function* fshareddtor = build_module_shared_dtor();
    llvm::Function* funittest = build_module_unittest();
    llvm::Function* fctor = build_module_ctor();
    llvm::Function* fdtor = build_module_dtor();

    unsigned flags = MInew;
    if (fctor)
        flags |= MItlsctor;
    if (fdtor)
        flags |= MItlsdtor;
    if (fsharedctor)
        flags |= MIctor;
    if (fshareddtor)
        flags |= MIdtor;
#if 0
    if (fgetmembers)
        flags |= MIxgetMembers;
    if (fictor)
        flags |= MIictor;
#endif
    if (funittest)
        flags |= MIunitTest;
    if (importedModules)
        flags |= MIimportedModules;
    if (localClasses)
        flags |= MIlocalClasses;

    if (!needmoduleinfo)
        flags |= MIstandalone;

    b.push_uint(flags); // flags
    b.push_uint(0);     // index

    if (fctor)
        b.push(fctor);
    if (fdtor)
        b.push(fdtor);
    if (fsharedctor)
        b.push(fsharedctor);
    if (fshareddtor)
        b.push(fshareddtor);
#if 0
    if (fgetmembers)
        b.push(fgetmembers);
    if (fictor)
        b.push(fictor);
#endif
    if (funittest)
        b.push(funittest);
    if (importedModules) {
        b.push_size(importInits.size());
        b.push(importedModules);
    }
    if (localClasses) {
        b.push_size(classInits.size());
        b.push(localClasses);
    }

    // Put out module name as a 0-terminated string, to save bytes
    b.push(DtoConstStringPtr(toPrettyChars()));

#else
    //     The layout is:
    //         char[]          name;
    //         ModuleInfo[]    importedModules;
    //         ClassInfo[]     localClasses;
    //         uint            flags;
    //
    //         void function() ctor;
    //         void function() dtor;
    //         void function() unitTest;
    //
    //         void* xgetMembers;
    //         void function() ictor;
    //
    //         version(D_Version2) {
    //             void *sharedctor;
    //             void *shareddtor;
    //             uint index;
    //             void*[1] reserved;
    //         }

    LLConstant *c = 0;

    // name
    b.push_string(toPrettyChars());

    // importedModules
    if (importedModules)
    {
        std::string m_name("_D");
        m_name.append(mangle());
        m_name.append("9__importsZ");
        llvm::GlobalVariable* m_gvar = gIR->module->getGlobalVariable(m_name);
        if (!m_gvar) m_gvar = new llvm::GlobalVariable(*gIR->module, importedModulesTy, true, llvm::GlobalValue::InternalLinkage, importedModules, m_name);
        c = llvm::ConstantExpr::getBitCast(m_gvar, getPtrToType(importedModulesTy->getElementType()));
        c = DtoConstSlice(DtoConstSize_t(importInits.size()), c);
    }
    else
    {
        c = DtoConstSlice(DtoConstSize_t(0), getNullValue(getPtrToType(moduleinfoTy)));
    }
    b.push(c);

    // localClasses
    if (localClasses)
    {
        std::string m_name("_D");
        m_name.append(mangle());
        m_name.append("9__classesZ");
        assert(gIR->module->getGlobalVariable(m_name) == NULL);
        llvm::GlobalVariable* m_gvar = new llvm::GlobalVariable(*gIR->module, localClassesTy, true, llvm::GlobalValue::InternalLinkage, localClasses, m_name);
        c = DtoGEPi(m_gvar, 0, 0);
        c = DtoConstSlice(DtoConstSize_t(classInits.size()), c);
    }
    else
    {
        c = DtoConstSlice( DtoConstSize_t(0), getNullValue(getPtrToType(getPtrToType(classinfoTy))) );
    }
    b.push(c);

    // flags (4 means MIstandalone)
    unsigned mi_flags = needmoduleinfo ? 0 : 4;
    b.push_uint(mi_flags);

    // function pointer type for next three fields
    LLType* fnptrTy = getPtrToType(LLFunctionType::get(LLType::getVoidTy(gIR->context()), std::vector<LLType*>(), false));

    // ctor
#if DMDV2
    llvm::Function* fctor = build_module_shared_ctor();
#else
    llvm::Function* fctor = build_module_ctor();
#endif
    c = fctor ? fctor : getNullValue(fnptrTy);
    b.push(c);

    // dtor
#if DMDV2
    llvm::Function* fdtor = build_module_shared_dtor();
#else
    llvm::Function* fdtor = build_module_dtor();
#endif
    c = fdtor ? fdtor : getNullValue(fnptrTy);
    b.push(c);

    // unitTest
    llvm::Function* unittest = build_module_unittest();
    c = unittest ? unittest : getNullValue(fnptrTy);
    b.push(c);

    // xgetMembers
    c = getNullValue(getVoidPtrType());
    b.push(c);

    // ictor
    c = getNullValue(fnptrTy);
    b.push(c);

#if DMDV2

    // tls ctor
    fctor = build_module_ctor();
    c = fctor ? fctor : getNullValue(fnptrTy);
    b.push(c);

    // tls dtor
    fdtor = build_module_dtor();
    c = fdtor ? fdtor : getNullValue(fnptrTy);
    b.push(c);

    // index + reserved void*[1]
    LLType* AT = llvm::ArrayType::get(getVoidPtrType(), 2);
    c = getNullValue(AT);
    b.push(c);

#endif

#endif

    /*Logger::println("MODULE INFO INITIALIZERS");
    for (size_t i=0; i<initVec.size(); ++i)
    {
        Logger::cout() << *initVec[i] << '\n';
        if (initVec[i]->getType() != moduleinfoTy->getElementType(i))
            assert(0);
    }*/

    // create and set initializer
    b.finalize(moduleInfoType, moduleInfoSymbol());

    // build the modulereference and ctor for registering it
    LLFunction* mictor = build_module_reference_and_ctor(moduleInfoSymbol());

    // register this ctor in the magic llvm.global_ctors appending array
    LLFunctionType* magicfty = LLFunctionType::get(LLType::getVoidTy(gIR->context()), std::vector<LLType*>(), false);
    std::vector<LLType*> magictypes;
    magictypes.push_back(LLType::getInt32Ty(gIR->context()));
    magictypes.push_back(getPtrToType(magicfty));
    LLStructType* magicsty = LLStructType::get(gIR->context(), magictypes);

    // make the constant element
    std::vector<LLConstant*> magicconstants;
    magicconstants.push_back(DtoConstUint(65535));
    magicconstants.push_back(mictor);
    LLConstant* magicinit = LLConstantStruct::get(magicsty, magicconstants);

    // declare the appending array
    llvm::ArrayType* appendArrTy = llvm::ArrayType::get(magicsty, 1);
    std::vector<LLConstant*> appendInits(1, magicinit);
    LLConstant* appendInit = LLConstantArray::get(appendArrTy, appendInits);
    std::string appendName("llvm.global_ctors");
    new llvm::GlobalVariable(*gIR->module, appendArrTy, true, llvm::GlobalValue::AppendingLinkage, appendInit, appendName);
}
