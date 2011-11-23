
// Copyright (c) 1999-2004 by Digital Mars
// All Rights Reserved
// written by Walter Bright
// www.digitalmars.com
// License for redistribution is by either the Artistic License
// in artistic.txt, or the GNU General Public License in gnu.txt.
// See the included readme.txt for details.

#include <cstddef>
#include <fstream>

#include "gen/llvm.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/LinkAllPasses.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/CodeGen/MachineCodeEmitter.h"
#include "llvm/LLVMContext.h"

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
#include "gen/cl_options.h"
#include "gen/functions.h"
#include "gen/irstate.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/optimizer.h"
#include "gen/programs.h"
#include "gen/rttibuilder.h"
#include "gen/runtime.h"
#include "gen/structs.h"
#include "gen/todebug.h"
#include "gen/tollvm.h"

#include "ir/irvar.h"
#include "ir/irmodule.h"
#include "ir/irtype.h"

//////////////////////////////////////////////////////////////////////////////////////////

static llvm::cl::opt<bool> noVerify("noverify",
    llvm::cl::desc("Do not run the validation pass before writing bitcode"),
    llvm::cl::ZeroOrMore);

//////////////////////////////////////////////////////////////////////////////////////////

// fwd decl
void emit_file(llvm::TargetMachine &Target, llvm::Module& m, llvm::raw_fd_ostream& Out,
               llvm::TargetMachine::CodeGenFileType fileType);

//////////////////////////////////////////////////////////////////////////////////////////

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

    #ifndef DISABLE_DEBUG_INFO
    // debug info
    if (global.params.symdebug)
        DtoDwarfCompileUnit(this);
    #endif

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
    if (opts::singleObj)
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

    // finilize debugging
    #ifndef DISABLE_DEBUG_INFO
    if (global.params.symdebug)
        DtoDwarfModuleEnd();
    #endif

    // generate ModuleInfo
    genmoduleinfo();

    // verify the llvm
    if (!noVerify) {
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

void writeModule(llvm::Module* m, std::string filename)
{
    // run optimizer
    bool reverify = ldc_optimize_module(m);

    // verify the llvm
    if (!noVerify && reverify) {
        std::string verifyErr;
        Logger::println("Verifying module... again...");
        LOG_SCOPE;
        if (llvm::verifyModule(*m,llvm::ReturnStatusAction,&verifyErr))
        {
            error("%s", verifyErr.c_str());
            fatal();
        }
        else {
            Logger::println("Verification passed!");
        }
    }

    // eventually do our own path stuff, dmd's is a bit strange.
    typedef llvm::sys::Path LLPath;

    // write LLVM bitcode
    if (global.params.output_bc) {
        LLPath bcpath = LLPath(filename);
        bcpath.eraseSuffix();
        bcpath.appendSuffix(std::string(global.bc_ext));
        Logger::println("Writing LLVM bitcode to: %s\n", bcpath.c_str());
        std::string errinfo;
        llvm::raw_fd_ostream bos(bcpath.c_str(), errinfo, llvm::raw_fd_ostream::F_Binary);
        if (bos.has_error())
        {
            error("cannot write LLVM bitcode file '%s': %s", bcpath.c_str(), errinfo.c_str());
            fatal();
        }
        llvm::WriteBitcodeToFile(m, bos);
    }

    // write LLVM IR
    if (global.params.output_ll) {
        LLPath llpath = LLPath(filename);
        llpath.eraseSuffix();
        llpath.appendSuffix(std::string(global.ll_ext));
        Logger::println("Writing LLVM asm to: %s\n", llpath.c_str());
        std::string errinfo;
        llvm::raw_fd_ostream aos(llpath.c_str(), errinfo);
        if (aos.has_error())
        {
            error("cannot write LLVM asm file '%s': %s", llpath.c_str(), errinfo.c_str());
            fatal();
        }
        m->print(aos, NULL);
    }

    // write native assembly
    if (global.params.output_s) {
        LLPath spath = LLPath(filename);
        spath.eraseSuffix();
        spath.appendSuffix(std::string(global.s_ext));
        Logger::println("Writing native asm to: %s\n", spath.c_str());
        std::string err;
        {
            llvm::raw_fd_ostream out(spath.c_str(), err);
            if (err.empty())
            {
                emit_file(*gTargetMachine, *m, out, llvm::TargetMachine::CGFT_AssemblyFile);
            }
            else
            {
                error("cannot write native asm: %s", err.c_str());
                fatal();
            }
        }
    }

    if (global.params.output_o) {
        LLPath objpath = LLPath(filename);
        Logger::println("Writing object file to: %s\n", objpath.c_str());
        std::string err;
        {
            llvm::raw_fd_ostream out(objpath.c_str(), err);
            if (err.empty())
            {
                emit_file(*gTargetMachine, *m, out, llvm::TargetMachine::CGFT_ObjectFile);
            }
            else
            {
                error("cannot write object file: %s", err.c_str());
                fatal();
            }
        }
    }
}

/* ================================================================== */

// based on llc code, University of Illinois Open Source License
void emit_file(llvm::TargetMachine &Target, llvm::Module& m, llvm::raw_fd_ostream& out,
               llvm::TargetMachine::CodeGenFileType fileType)
{
    using namespace llvm;

    // Build up all of the passes that we want to do to the module.
    FunctionPassManager Passes(&m);

    if (const TargetData *TD = Target.getTargetData())
        Passes.add(new TargetData(*TD));
    else
        Passes.add(new TargetData(&m));

    // Last argument is enum CodeGenOpt::Level OptLevel
    // debug info doesn't work properly with OptLevel != None!
    CodeGenOpt::Level LastArg = CodeGenOpt::Default;
    if (global.params.symdebug || !optimize())
        LastArg = CodeGenOpt::None;
    else if (optLevel() >= 3)
        LastArg = CodeGenOpt::Aggressive;

    llvm::formatted_raw_ostream fout(out);
    if (Target.addPassesToEmitFile(Passes, fout, fileType, LastArg))
        assert(0 && "no support for asm output");

    Passes.doInitialization();

    // Run our queue of passes all at once now, efficiently.
    for (llvm::Module::iterator I = m.begin(), E = m.end(); I != E; ++I)
        if (!I->isDeclaration())
            Passes.run(*I);

    Passes.doFinalization();

    // release module from module provider so we can delete it ourselves
    //std::string Err;
    //llvm::Module* rmod = Provider.releaseModule(&Err);
    //assert(rmod);
}

/* ================================================================== */

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
    #ifndef DISABLE_DEBUG_INFO
    if(global.params.symdebug)
        DtoDwarfSubProgramInternal(name.c_str(), name.c_str());
    #endif

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
    #ifndef DISABLE_DEBUG_INFO
    llvm::DISubprogram subprog;
    if(global.params.symdebug)
        subprog = DtoDwarfSubProgramInternal(fname.c_str(), fname.c_str());
    #endif

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
//     The layout is:
//         {
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

    // name
    b.push_string(toPrettyChars());

    // importedModules[]
    std::vector<LLConstant*> importInits;
    LLConstant* c = 0;
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
        llvm::ArrayType* importArrTy = llvm::ArrayType::get(getPtrToType(moduleinfoTy), importInits.size());
        c = LLConstantArray::get(importArrTy, importInits);
        std::string m_name("_D");
        m_name.append(mangle());
        m_name.append("9__importsZ");
        llvm::GlobalVariable* m_gvar = gIR->module->getGlobalVariable(m_name);
        if (!m_gvar) m_gvar = new llvm::GlobalVariable(*gIR->module, importArrTy, true, llvm::GlobalValue::InternalLinkage, c, m_name);
        c = llvm::ConstantExpr::getBitCast(m_gvar, getPtrToType(importArrTy->getElementType()));
        c = DtoConstSlice(DtoConstSize_t(importInits.size()), c);
    }
    else
    {
        c = DtoConstSlice( DtoConstSize_t(0), getNullValue(getPtrToType(moduleinfoTy)) );
    }
    b.push(c);

    // localClasses[]
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
        c = DtoBitCast(cd->ir.irStruct->getClassInfoSymbol(), getPtrToType(classinfoTy));
        classInits.push_back(c);
    }
    // has class array?
    if (!classInits.empty())
    {
        llvm::ArrayType* classArrTy = llvm::ArrayType::get(getPtrToType(classinfoTy), classInits.size());
        c = LLConstantArray::get(classArrTy, classInits);
        std::string m_name("_D");
        m_name.append(mangle());
        m_name.append("9__classesZ");
        assert(gIR->module->getGlobalVariable(m_name) == NULL);
        llvm::GlobalVariable* m_gvar = new llvm::GlobalVariable(*gIR->module, classArrTy, true, llvm::GlobalValue::InternalLinkage, c, m_name);
        c = DtoGEPi(m_gvar, 0, 0);
        c = DtoConstSlice(DtoConstSize_t(classInits.size()), c);
    }
    else
        c = DtoConstSlice( DtoConstSize_t(0), getNullValue(getPtrToType(getPtrToType(classinfoTy))) );
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
