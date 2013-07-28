//===-- module.cpp --------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "aggregate.h"
#include "attrib.h"
#include "declaration.h"
#include "enum.h"
#include "id.h"
#include "import.h"
#include "init.h"
#include "mars.h"
#include "module.h"
#include "mtype.h"
#include "scope.h"
#include "statement.h"
#include "target.h"
#include "template.h"
#include "gen/abi.h"
#include "gen/arrays.h"
#include "gen/classes.h"
#include "gen/functions.h"
#include "gen/irstate.h"
#include "gen/llvm.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/optimizer.h"
#include "gen/programs.h"
#include "gen/rttibuilder.h"
#include "gen/runtime.h"
#include "gen/structs.h"
#include "gen/tollvm.h"
#include "ir/irdsymbol.h"
#include "ir/irmodule.h"
#include "ir/irtype.h"
#include "ir/irvar.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/LinkAllPasses.h"
#if LDC_LLVM_VER >= 303
#include "llvm/IR/Module.h"
#include "llvm/IR/DataLayout.h"
#else
#include "llvm/Module.h"
#if LDC_LLVM_VER == 302
#include "llvm/DataLayout.h"
#else
#include "llvm/Target/TargetData.h"
#endif
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

    std::string const symbolName = gABI->mangleForLLVM(name, LINKd);
    assert(gIR->module->getFunction(symbolName) == NULL);
    llvm::Function* fn = llvm::Function::Create(fnTy,
        llvm::GlobalValue::InternalLinkage, symbolName, gIR->module);
    fn->setCallingConv(gABI->callingConv(LINKd));

    llvm::BasicBlock* bb = llvm::BasicBlock::Create(gIR->context(), "entry", fn);
    IRBuilder<> builder(bb);

    // debug info
    gIR->DBuilder.EmitSubProgramInternal(name.c_str(), symbolName.c_str());

    // Call ctor's
    typedef std::list<FuncDeclaration*>::const_iterator FuncIterator;
    for (FuncIterator itr = funcs.begin(), end = funcs.end(); itr != end; ++itr) {
        llvm::Function* f = (*itr)->ir.irFunc->func;
        llvm::CallInst* call = builder.CreateCall(f,"");
        call->setCallingConv(gABI->callingConv(LINKd));
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
    return build_module_function(name, gIR->ctors, gIR->gates);
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
    LLConstant* mrefvalues[] = {
        LLConstant::getNullValue(modulerefTy->getContainedType(0)),
        llvm::ConstantExpr::getBitCast(moduleinfo, modulerefTy->getContainedType(1))
    };
    LLConstant* thismrefinit = LLConstantStruct::get(modulerefTy, llvm::ArrayRef<LLConstant*>(mrefvalues));

    // create the ModuleReference node for this module
    std::string thismrefname = "_D";
    thismrefname += gIR->dmodule->mangle();
    thismrefname += "11__moduleRefZ";
    LLGlobalVariable* thismref = getOrCreateGlobal(Loc(), *gIR->module,
        modulerefTy, false, LLGlobalValue::InternalLinkage, thismrefinit,
        thismrefname);
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
    gIR->DBuilder.EmitSubProgramInternal(fname.c_str(), fname.c_str());

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

    Logger::println("Generating module: %s", (md ? md->toChars() : toChars()));
    LOG_SCOPE;

    if (global.params.verbose_cg)
        printf("codegen: %s (%s)\n", toPrettyChars(), srcfile->toChars());

    assert(!global.errors);

    // name the module
#if 1
    // Temporary workaround for http://llvm.org/bugs/show_bug.cgi?id=11479 –
    // just use the source file name, as it is unlikely to collide with a
    // symbol name used somewhere in the module.
    llvm::StringRef mname(srcfile->toChars());
#else
    llvm::StringRef mname(toChars());
    if (md != 0)
        mname = md->toChars();
#endif

    // create a new ir state
    // TODO look at making the instance static and moving most functionality into IrModule where it belongs
    IRState ir(new llvm::Module(mname, context));
    gIR = &ir;
    ir.dmodule = this;

    // reset all IR data stored in Dsymbols
    IrDsymbol::resetAll();

    sir->setState(&ir);

    // set target triple
    ir.module->setTargetTriple(global.params.targetTriple.str());

    // set final data layout
    ir.module->setDataLayout(gDataLayout->getStringRepresentation());
    if (Logger::enabled())
        Logger::cout() << "Final data layout: " << ir.module->getDataLayout() << '\n';

    // allocate the target abi
    gABI = TargetABI::getTarget();

    // debug info
    gIR->DBuilder.EmitCompileUnit(this);

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
        Dsymbol* dsym = static_cast<Dsymbol*>(members->data[k]);
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

    // finalize debug info
    gIR->DBuilder.EmitModuleEnd();

    // generate ModuleInfo
    genmoduleinfo();

    // verify the llvm
    verifyModule(*ir.module);

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
    MIname.append("12__ModuleInfoZ");

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
    moduleInfoVar = getOrCreateGlobal(loc, *gIR->module, moduleInfoType,
        false, llvm::GlobalValue::ExternalLinkage, NULL, MIname);

    return moduleInfoVar;
}

// Put out instance of ModuleInfo for this Module
void Module::genmoduleinfo()
{
    // resolve ModuleInfo
    if (!moduleinfo)
    {
        error("object.d is missing the ModuleInfo struct");
        fatal();
    }
    // check for patch
    else
    {
        unsigned sizeof_ModuleInfo = 16 * Target::ptrsize;
        if (sizeof_ModuleInfo != moduleinfo->structsize)
        {
            error("object.d ModuleInfo class is incorrect");
            fatal();
        }
    }

    // use the RTTIBuilder
    RTTIBuilder b(moduleinfo);

    // some types
    LLType* moduleinfoTy = moduleinfo->type->irtype->getLLType();
    LLType* classinfoTy = ClassDeclaration::classinfo->type->irtype->getLLType();

    // importedModules[]
    std::vector<LLConstant*> importInits;
    LLConstant* importedModules = 0;
    llvm::ArrayType* importedModulesTy = 0;
    for (size_t i = 0; i < aimports.dim; i++)
    {
        Module *m = static_cast<Module *>(aimports.data[i]);
        if (!m->needModuleInfo() || m == this)
            continue;

        // declare the imported module info
        std::string m_name("_D");
        m_name.append(m->mangle());
        m_name.append("12__ModuleInfoZ");
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

        member = static_cast<Dsymbol *>(members->data[i]);
        //printf("\tmember '%s'\n", member->toChars());
        member->addLocalClass(&aclasses);
    }
    // fill inits
    std::vector<LLConstant*> classInits;
    for (size_t i = 0; i < aclasses.dim; i++)
    {
        ClassDeclaration* cd = static_cast<ClassDeclaration*>(aclasses.data[i]);
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
        LLConstant *c = DtoBitCast(cd->ir.irAggr->getClassInfoSymbol(), classinfoTy);
        classInits.push_back(c);
    }
    // has class array?
    if (!classInits.empty())
    {
        localClassesTy = llvm::ArrayType::get(classinfoTy, classInits.size());
        localClasses = LLConstantArray::get(localClassesTy, classInits);
    }

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

    // Put out module name as a 0-terminated string.
    const char *name = toPrettyChars();
    const size_t len = strlen(name) + 1;
    llvm::IntegerType *it = llvm::IntegerType::getInt8Ty(gIR->context());
    llvm::ArrayType *at = llvm::ArrayType::get(it, len);
    b.push(toConstantArray(it, at, name, len, false));

    // create and set initializer
    b.finalize(moduleInfoType, moduleInfoSymbol());

    // build the modulereference and ctor for registering it
    LLFunction* mictor = build_module_reference_and_ctor(moduleInfoSymbol());

    AppendFunctionToLLVMGlobalCtorsDtors(mictor, 65535, true);
}
