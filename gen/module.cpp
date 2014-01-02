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
#include "llvm/Support/CommandLine.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/LinkAllPasses.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
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

static llvm::cl::opt<bool> preservePaths("op",
    llvm::cl::desc("Do not strip paths from source file"),
    llvm::cl::ZeroOrMore);

static llvm::cl::opt<bool> fqnNames("oq",
    llvm::cl::desc("Write object files with fully qualified names"),
    llvm::cl::ZeroOrMore);

static void check_and_add_output_file(Module* NewMod, const std::string& str)
{
    typedef std::map<std::string, Module*> map_t;
    static map_t files;

    map_t::iterator i = files.find(str);
    if (i != files.end()) {
        Module* ThisMod = i->second;
        error(Loc(), "Output file '%s' for module '%s' collides with previous module '%s'. See the -oq option",
            str.c_str(), NewMod->toPrettyChars(), ThisMod->toPrettyChars());
        fatal();
    }
    files.insert(std::make_pair(str, NewMod));
}

void Module::buildTargetFiles(bool singleObj)
{
    if (objfile &&
       (!doDocComment || docfile) &&
       (!doHdrGen || hdrfile))
        return;

    if (!objfile) {
        if (global.params.output_o)
            objfile = Module::buildFilePath(global.params.objname, global.params.objdir,
                global.params.targetTriple.isOSWindows() ? global.obj_ext_alt : global.obj_ext);
        else if (global.params.output_bc)
            objfile = Module::buildFilePath(global.params.objname, global.params.objdir, global.bc_ext);
        else if (global.params.output_ll)
            objfile = Module::buildFilePath(global.params.objname, global.params.objdir, global.ll_ext);
        else if (global.params.output_s)
            objfile = Module::buildFilePath(global.params.objname, global.params.objdir, global.s_ext);
    }
    if (doDocComment && !docfile)
        docfile = Module::buildFilePath(global.params.docname, global.params.docdir, global.doc_ext);
    if (doHdrGen && !hdrfile)
        hdrfile = Module::buildFilePath(global.params.hdrname, global.params.hdrdir, global.hdr_ext);

    // safety check: never allow obj, doc or hdr file to have the source file's name
    if (Port::stricmp(FileName::name(objfile->name->str), FileName::name((char*)this->arg)) == 0) {
        error("Output object files with the same name as the source file are forbidden");
        fatal();
    }
    if (docfile && Port::stricmp(FileName::name(docfile->name->str), FileName::name((char*)this->arg)) == 0) {
        error("Output doc files with the same name as the source file are forbidden");
        fatal();
    }
    if (hdrfile && Port::stricmp(FileName::name(hdrfile->name->str), FileName::name((char*)this->arg)) == 0) {
        error("Output header files with the same name as the source file are forbidden");
        fatal();
    }

    // LDC
    // another safety check to make sure we don't overwrite previous output files
    if (!singleObj)
        check_and_add_output_file(this, objfile->name->str);
    if (docfile)
        check_and_add_output_file(this, docfile->name->str);
    if (hdrfile)
        check_and_add_output_file(this, hdrfile->name->str);
}

File* Module::buildFilePath(const char* forcename, const char* path, const char* ext)
{
    const char *argobj;
    if (forcename) {
        argobj = forcename;
    } else {
        if (preservePaths)
            argobj = (char*)this->arg;
        else
            argobj = FileName::name((char*)this->arg);

        if (fqnNames) {
            char *name = md ? md->toChars() : toChars();
            argobj = FileName::replaceName((char*)argobj, name);

            // add ext, otherwise forceExt will make nested.module into nested.bc
            size_t len = strlen(argobj);
            size_t extlen = strlen(ext);
            char* s = (char *)alloca(len + 1 + extlen + 1);
            memcpy(s, argobj, len);
            s[len] = '.';
            memcpy(s + len + 1, ext, extlen + 1);
            s[len+1+extlen] = 0;
            argobj = s;
        }
    }

    if (!FileName::absolute(argobj))
        argobj = FileName::combine(path, argobj);

    FileName::ensurePathExists(FileName::path(argobj));

    // always append the extension! otherwise hard to make output switches consistent
    //    if (forcename)
    //	return new File(argobj);
    //    else
        // allow for .o and .obj on windows
#if _WIN32
    if (ext == global.params.objdir && FileName::ext(argobj)
        && Port::stricmp(FileName::ext(argobj), global.obj_ext_alt) == 0)
    return new File((char*)argobj);
#endif
    return new File(FileName::forceExt(argobj, ext));
}

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

/// Builds the body for the ldc.dso_ctor and ldc.dso_dtor functions.
///
/// Pseudocode:
/// if (dsoInitialized == executeWhenInitialized) {
///     dsoInitiaized = !executeWhenInitialized;
///     auto record = CompilerDSOData(1, dsoSlot, minfoBeg, minfoEnd);
///     _d_dso_registry(&record);
/// }
static void build_dso_ctor_dtor_body(
    llvm::Function* targetFunc,
    llvm::GlobalVariable* dsoInitiaized,
    llvm::GlobalVariable* dsoSlot,
    llvm::GlobalVariable* minfoBeg,
    llvm::GlobalVariable* minfoEnd,
    bool executeWhenInitialized
) {
    llvm::Function* const dsoRegistry = LLVM_D_GetRuntimeFunction(
        gIR->module, "_d_dso_registry");
    llvm::Type* const recordPtrTy = dsoRegistry->getFunctionType()->getContainedType(1);
    llvm::Type* const recordTy = recordPtrTy->getContainedType(0);

    llvm::BasicBlock* const entryBB =
        llvm::BasicBlock::Create(gIR->context(), "", targetFunc);
    llvm::BasicBlock* const initBB =
        llvm::BasicBlock::Create(gIR->context(), "init", targetFunc);
    llvm::BasicBlock* const endBB =
        llvm::BasicBlock::Create(gIR->context(), "end", targetFunc);

    {
        IRBuilder<> b(entryBB);
        llvm::Value* initialized = b.CreateLoad(dsoInitiaized);
        if (executeWhenInitialized)
            b.CreateCondBr(initialized, initBB, endBB);
        else
            b.CreateCondBr(initialized, endBB, initBB);
    }
    {
        IRBuilder<> b(initBB);
        b.CreateStore(b.getInt1(!executeWhenInitialized), dsoInitiaized);

        llvm::Value* record = b.CreateAlloca(recordTy);
        b.CreateStore(DtoConstSize_t(1), b.CreateStructGEP(record, 0)); // version
        b.CreateStore(dsoSlot, b.CreateStructGEP(record, 1)); // slot
        b.CreateStore(minfoBeg, b.CreateStructGEP(record, 2));
        b.CreateStore(minfoEnd, b.CreateStructGEP(record, 3));

        b.CreateCall(dsoRegistry, record);
        b.CreateBr(endBB);
    }
    {
        IRBuilder<> b(endBB);
        b.CreateRetVoid();
    }
}

static void build_dso_registry_calls(llvm::Constant* thisModuleInfo)
{
    // Build the ModuleInfo reference and bracketing symbols.
    llvm::Type* const moduleInfoPtrTy =
        getPtrToType(DtoType(Module::moduleinfo->type));

    // Order is important here: We must create the symbols in the
    // bracketing sections right before/after the ModuleInfo reference
    // so that they end up in the correct order in the object file.
    llvm::GlobalVariable* minfoBeg = new llvm::GlobalVariable(
        *gIR->module,
        moduleInfoPtrTy,
        false, // FIXME: mRelocModel != llvm::Reloc::PIC_
        llvm::GlobalValue::LinkOnceODRLinkage,
        getNullPtr(moduleInfoPtrTy),
        "_minfo_beg"
    );
    minfoBeg->setSection(".minfo_beg");
    minfoBeg->setVisibility(llvm::GlobalValue::HiddenVisibility);

    std::string thismrefname = "_D";
    thismrefname += gIR->dmodule->mangle();
    thismrefname += "11__moduleRefZ";
    llvm::GlobalVariable* thismref = new llvm::GlobalVariable(
        *gIR->module,
        moduleInfoPtrTy,
        false, // FIXME: mRelocModel != llvm::Reloc::PIC_
        llvm::GlobalValue::LinkOnceODRLinkage,
        DtoBitCast(thisModuleInfo, moduleInfoPtrTy),
        thismrefname
    );
    thismref->setSection(".minfo");
    gIR->usedArray.push_back(thismref);

    llvm::GlobalVariable* minfoEnd = new llvm::GlobalVariable(
        *gIR->module,
        moduleInfoPtrTy,
        false, // FIXME: mRelocModel != llvm::Reloc::PIC_
        llvm::GlobalValue::LinkOnceODRLinkage,
        getNullPtr(moduleInfoPtrTy),
        "_minfo_end"
    );
    minfoEnd->setSection(".minfo_end");
    minfoEnd->setVisibility(llvm::GlobalValue::HiddenVisibility);

    // Build the ctor to invoke _d_dso_registry.
    llvm::GlobalVariable* dsoSlot = new llvm::GlobalVariable(
        *gIR->module,
        getVoidPtrType(),
        false,
        llvm::GlobalValue::LinkOnceODRLinkage,
        getNullPtr(getVoidPtrType()),
        "ldc.dso_slot"
    );
    dsoSlot->setVisibility(llvm::GlobalValue::HiddenVisibility);
    llvm::GlobalVariable* dsoInitiaized = new llvm::GlobalVariable(
        *gIR->module,
        llvm::Type::getInt1Ty(gIR->context()),
        false,
        llvm::GlobalValue::LinkOnceODRLinkage,
        llvm::ConstantInt::getFalse(gIR->context()),
        "ldc.dso_initialized"
    );
    dsoInitiaized->setVisibility(llvm::GlobalValue::HiddenVisibility);

    llvm::Function* dsoCtor = llvm::Function::Create(
        llvm::FunctionType::get(llvm::Type::getVoidTy(gIR->context()), false),
        llvm::GlobalValue::LinkOnceODRLinkage,
        "ldc.dso_ctor",
        gIR->module
    );
    dsoCtor->setVisibility(llvm::GlobalValue::HiddenVisibility);
    build_dso_ctor_dtor_body(dsoCtor, dsoInitiaized, dsoSlot, minfoBeg, minfoEnd, false);
    llvm::appendToGlobalCtors(*gIR->module, dsoCtor, 65535);

    llvm::Function* dsoDtor = llvm::Function::Create(
        llvm::FunctionType::get(llvm::Type::getVoidTy(gIR->context()), false),
        llvm::GlobalValue::LinkOnceODRLinkage,
        "ldc.dso_dtor",
        gIR->module
    );
    dsoDtor->setVisibility(llvm::GlobalValue::HiddenVisibility);
    build_dso_ctor_dtor_body(dsoDtor, dsoInitiaized, dsoSlot, minfoBeg, minfoEnd, true);
    llvm::appendToGlobalDtors(*gIR->module, dsoDtor, 65535);
}

static void build_llvm_used_array(IRState* p)
{
    if (p->usedArray.empty()) return;

    std::vector<llvm::Constant*> usedVoidPtrs;
    usedVoidPtrs.reserve(p->usedArray.size());

    for (std::vector<llvm::Constant*>::iterator it = p->usedArray.begin(),
        end = p->usedArray.end(); it != end; ++it)
    {
        usedVoidPtrs.push_back(DtoBitCast(*it, getVoidPtrType()));
    }

    llvm::ArrayType *arrayType = llvm::ArrayType::get(
        getVoidPtrType(), usedVoidPtrs.size());
    llvm::GlobalVariable* llvmUsed = new llvm::GlobalVariable(
        *p->module,
        arrayType,
        false,
        llvm::GlobalValue::AppendingLinkage,
        llvm::ConstantArray::get(arrayType, usedVoidPtrs),
        "llvm.used"
    );
    llvmUsed->setSection("llvm.metadata");
}

llvm::Module* Module::genLLVMModule(llvm::LLVMContext& context)
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

    LLVM_D_InitRuntime();

    // process module members
    for (unsigned k=0; k < members->dim; k++) {
        Dsymbol* dsym = static_cast<Dsymbol*>(members->data[k]);
        assert(dsym);
        dsym->codegen(&ir);
    }

    // for singleobj-compilation, fully emit all seen template instances
    if (global.params.singleObj)
    {
        while (!ir.seenTemplateInstances.empty())
        {
            IRState::TemplateInstanceSet::iterator it, end = ir.seenTemplateInstances.end();
            for (it = ir.seenTemplateInstances.begin(); it != end; ++it)
                (*it)->codegen(&ir);
            ir.seenTemplateInstances.clear();
        }
    }

    // finalize debug info
    gIR->DBuilder.EmitModuleEnd();

    // generate ModuleInfo
    genmoduleinfo();

    build_llvm_used_array(&ir);

#if LDC_LLVM_VER >= 303
    // Add the linker options metadata flag.
    ir.module->addModuleFlag(llvm::Module::AppendUnique, "Linker Options",
                             llvm::MDNode::get(ir.context(), ir.LinkerMetadataArgs));
#endif

    // verify the llvm
    verifyModule(*ir.module);

    gIR = NULL;

    if (llvmForceLogging && !logenabled)
    {
        Logger::disable();
    }

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

    if (!moduleInfoVar) {
        // declare global
        // flags will be modified at runtime so can't make it constant
        LLType *moduleInfoType = llvm::StructType::create(llvm::getGlobalContext());
        moduleInfoVar = getOrCreateGlobal(loc, *gIR->module, moduleInfoType,
            false, llvm::GlobalValue::ExternalLinkage, NULL, MIname);
    }

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
        // The base struct should consist only of _flags/_index.
        if (moduleinfo->structsize != 4 + 4)
        {
            error("object.d ModuleInfo class is incorrect");
            fatal();
        }
    }

    // use the RTTIBuilder
    RTTIBuilder b(moduleinfo);

    // some types
    LLType* moduleinfoTy = moduleinfo->type->irtype->getLLType();
    LLType* classinfoTy = Type::typeinfoclass->type->irtype->getLLType();

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
        ClassDeclaration* cd = aclasses[i];
        DtoResolveClass(cd);

        if (cd->isInterfaceDeclaration())
        {
            Logger::println("skipping interface '%s' in moduleinfo", cd->toPrettyChars());
            continue;
        }
        else if (cd->sizeok != SIZEOKdone)
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
    LLGlobalVariable *moduleInfoSym = moduleInfoSymbol();
    b.finalize(moduleInfoSym->getType()->getPointerElementType(), moduleInfoSym);
    moduleInfoSym->setLinkage(llvm::GlobalValue::ExternalLinkage);

    if (global.params.isLinux) {
        build_dso_registry_calls(moduleInfoSym);
    } else {
        // build the modulereference and ctor for registering it
        LLFunction* mictor = build_module_reference_and_ctor(moduleInfoSym);
        AppendFunctionToLLVMGlobalCtorsDtors(mictor, 65535, true);
    }
}
