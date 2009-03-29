
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
#include "llvm/ModuleProvider.h"
#include "llvm/PassManager.h"
#include "llvm/LinkAllPasses.h"
#include "llvm/System/Program.h"
#include "llvm/System/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetMachine.h"

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

#include "gen/irstate.h"
#include "gen/logger.h"
#include "gen/tollvm.h"
#include "gen/llvmhelpers.h"
#include "gen/arrays.h"
#include "gen/structs.h"
#include "gen/classes.h"
#include "gen/functions.h"
#include "gen/todebug.h"
#include "gen/runtime.h"
#include "gen/abi.h"
#include "gen/cl_options.h"
#include "gen/optimizer.h"

#include "ir/irvar.h"
#include "ir/irmodule.h"

//////////////////////////////////////////////////////////////////////////////////////////

static llvm::cl::opt<bool> noVerify("noverify",
    llvm::cl::desc("Do not run the validation pass before writing bitcode"),
    llvm::cl::ZeroOrMore);

//////////////////////////////////////////////////////////////////////////////////////////

// fwd decl
void write_asm_to_file(llvm::TargetMachine &Target, llvm::Module& m, llvm::raw_fd_ostream& Out);
void assemble(const llvm::sys::Path& asmpath, const llvm::sys::Path& objpath);

//////////////////////////////////////////////////////////////////////////////////////////

llvm::Module* Module::genLLVMModule(Ir* sir)
{
    bool logenabled = Logger::enabled();
    if (llvmForceLogging && !logenabled)
    {
        Logger::enable();
    }

    Logger::println("Generating module: %s\n", (md ? md->toChars() : toChars()));
    LOG_SCOPE;

    //printf("codegen: %s\n", srcfile->toChars());

    assert(!global.errors);

    // name the module
    std::string mname(toChars());
    if (md != 0)
        mname = md->toChars();

    // create a new ir state
    // TODO look at making the instance static and moving most functionality into IrModule where it belongs
    IRState ir(new llvm::Module(mname));
    gIR = &ir;
    ir.dmodule = this;

    // reset all IR data stored in Dsymbols and Types
    IrDsymbol::resetAll();
    IrType::resetAll();

    sir->setState(&ir);

    // module ir state
    // might already exist via import, just overwrite since
    // the global created for the filename must belong to the right llvm module
    // FIXME: but shouldn't this always get reset between modules? like other IrSymbols
    this->ir.irModule = new IrModule(this, srcfile->toChars());

    // set target triple
    ir.module->setTargetTriple(global.params.targetTriple);

    // set final data layout
    ir.module->setDataLayout(global.params.dataLayout);
    if (Logger::enabled())
        Logger::cout() << "Final data layout: " << global.params.dataLayout << '\n';

    // allocate the target abi
    gABI = TargetABI::getTarget();

    // debug info
    if (global.params.symdebug) {
        RegisterDwarfSymbols(ir.module);
        DtoDwarfCompileUnit(this);
    }

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
    for (int k=0; k < members->dim; k++) {
        Dsymbol* dsym = (Dsymbol*)(members->data[k]);
        assert(dsym);
        dsym->codegen(sir);
    }

    // emit function bodies
    sir->emitFunctionBodies();

    // generate ModuleInfo
    genmoduleinfo();

    // emit usedArray
    if (!ir.usedArray.empty())
    {
        const LLArrayType* usedTy = LLArrayType::get(getVoidPtrType(), ir.usedArray.size());
        LLConstant* usedInit = LLConstantArray::get(usedTy, ir.usedArray);
        LLGlobalVariable* usedArray = new LLGlobalVariable(usedTy, true, LLGlobalValue::AppendingLinkage, usedInit, "llvm.used", ir.module);
        usedArray->setSection("llvm.metadata");
    }

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
            //error("%s", verifyErr.c_str());
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
        std::ofstream bos(bcpath.c_str(), std::ios::binary);
        llvm::WriteBitcodeToFile(m, bos);
    }

    // write LLVM IR
    if (global.params.output_ll) {
        LLPath llpath = LLPath(filename);
        llpath.eraseSuffix();
        llpath.appendSuffix(std::string(global.ll_ext));
        Logger::println("Writing LLVM asm to: %s\n", llpath.c_str());
        std::ofstream aos(llpath.c_str());
        m->print(aos, NULL);
    }

    // write native assembly
    if (global.params.output_s || global.params.output_o) {
        LLPath spath = LLPath(filename);
        spath.eraseSuffix();
        spath.appendSuffix(std::string(global.s_ext));
        if (!global.params.output_s) {
            spath.createTemporaryFileOnDisk();
        }
        Logger::println("Writing native asm to: %s\n", spath.c_str());
        std::string err;
        {
            llvm::raw_fd_ostream out(spath.c_str(), false, err);
            write_asm_to_file(*gTargetMachine, *m, out);
        }

        // call gcc to convert assembly to object file
        if (global.params.output_o) {
            LLPath objpath = LLPath(filename);
            assemble(spath, objpath);
        }

        if (!global.params.output_s) {
            spath.eraseFromDisk();
        }
    }
}

/* ================================================================== */

// based on llc code, University of Illinois Open Source License
void write_asm_to_file(llvm::TargetMachine &Target, llvm::Module& m, llvm::raw_fd_ostream& out)
{
    using namespace llvm;

    // Build up all of the passes that we want to do to the module.
    ExistingModuleProvider Provider(&m);
    FunctionPassManager Passes(&Provider);

    Passes.add(new TargetData(*Target.getTargetData()));

    // Ask the target to add backend passes as necessary.
    MachineCodeEmitter *MCE = 0;

//TODO: May want to switch it on for -O0?
    bool Fast = false;
    FileModel::Model mod = Target.addPassesToEmitFile(Passes, out, TargetMachine::AssemblyFile, Fast);
    assert(mod == FileModel::AsmFile);

    bool err = Target.addPassesToEmitFileFinish(Passes, MCE, Fast);
    assert(!err);

    Passes.doInitialization();

    // Run our queue of passes all at once now, efficiently.
    for (llvm::Module::iterator I = m.begin(), E = m.end(); I != E; ++I)
        if (!I->isDeclaration())
            Passes.run(*I);

    Passes.doFinalization();

    // release module from module provider so we can delete it ourselves
    std::string Err;
    llvm::Module* rmod = Provider.releaseModule(&Err);
    assert(rmod);
}

/* ================================================================== */

// uses gcc to make an obj out of an assembly file
// based on llvm-ld code, University of Illinois Open Source License
void assemble(const llvm::sys::Path& asmpath, const llvm::sys::Path& objpath)
{
    using namespace llvm;

    const char *cc;
#if !_WIN32
    cc = getenv("CC");
    if (!cc)
#endif
	cc = "gcc";

    sys::Path gcc = llvm::sys::Program::FindProgramByName(cc);
    if (gcc.empty())
    {
        error("failed to locate gcc");
        fatal();
    }

    // Run GCC to assemble and link the program into native code.
    //
    // Note:
    //  We can't just assemble and link the file with the system assembler
    //  and linker because we don't know where to put the _start symbol.
    //  GCC mysteriously knows how to do it.
    std::vector<std::string> args;
    args.push_back(gcc.toString());
    args.push_back("-fno-strict-aliasing");
    args.push_back("-O3");
    args.push_back("-c");
    args.push_back("-xassembler");
    args.push_back(asmpath.toString());
    args.push_back("-o");
    args.push_back(objpath.toString());

    //FIXME: only use this if needed?
    args.push_back("-fpic");

    //FIXME: enforce 64 bit
    if (global.params.is64bit)
        args.push_back("-m64");
    else
        // Assume 32-bit?
        args.push_back("-m32");

    // Now that "args" owns all the std::strings for the arguments, call the c_str
    // method to get the underlying string array.  We do this game so that the
    // std::string array is guaranteed to outlive the const char* array.
    std::vector<const char *> Args;
    for (unsigned i = 0, e = args.size(); i != e; ++i)
        Args.push_back(args[i].c_str());
    Args.push_back(0);

    if (Logger::enabled()) {
        Logger::println("Assembling with: ");
        std::vector<const char*>::const_iterator I = Args.begin(), E = Args.end();
        std::ostream& logstr = *Logger::cout().stream();
        for (; I != E; ++I)
            if (*I)
                logstr << "'" << *I << "'" << " ";
        logstr << "\n" << std::flush;
    }

    // Run the compiler to assembly the program.
    std::string ErrMsg;
    int R = sys::Program::ExecuteAndWait(
        gcc, &Args[0], 0, 0, 0, 0, &ErrMsg);
    if (R)
    {
        error("Failed to invoke gcc. %s", ErrMsg.c_str());
        fatal();
    }
}


/* ================================================================== */

// the following code generates functions and needs to output
// debug info. these macros are useful for that
#define DBG_TYPE    ( getPtrToType(llvm::StructType::get(NULL,NULL)) )
#define DBG_CAST(X) ( llvm::ConstantExpr::getBitCast(X, DBG_TYPE) )

// build module ctor

llvm::Function* build_module_ctor()
{
    if (gIR->ctors.empty())
        return NULL;

    size_t n = gIR->ctors.size();
    if (n == 1)
        return gIR->ctors[0]->ir.irFunc->func;

    std::string name("_D");
    name.append(gIR->dmodule->mangle());
    name.append("6__ctorZ");

    std::vector<const LLType*> argsTy;
    const llvm::FunctionType* fnTy = llvm::FunctionType::get(LLType::VoidTy,argsTy,false);
    assert(gIR->module->getFunction(name) == NULL);
    llvm::Function* fn = llvm::Function::Create(fnTy, llvm::GlobalValue::InternalLinkage, name, gIR->module);
    fn->setCallingConv(DtoCallingConv(0, LINKd));

    llvm::BasicBlock* bb = llvm::BasicBlock::Create("entry", fn);
    IRBuilder<> builder(bb);

    // debug info
    LLGlobalVariable* subprog;
    if(global.params.symdebug) {
        subprog = DtoDwarfSubProgramInternal(name.c_str(), name.c_str()).getGV();
        builder.CreateCall(gIR->module->getFunction("llvm.dbg.func.start"), DBG_CAST(subprog));
    }

    for (size_t i=0; i<n; i++) {
        llvm::Function* f = gIR->ctors[i]->ir.irFunc->func;
        llvm::CallInst* call = builder.CreateCall(f,"");
        call->setCallingConv(DtoCallingConv(0, LINKd));
    }

    // debug info end
    if(global.params.symdebug)
        builder.CreateCall(gIR->module->getFunction("llvm.dbg.region.end"), DBG_CAST(subprog));

    builder.CreateRetVoid();
    return fn;
}

// build module dtor

static llvm::Function* build_module_dtor()
{
    if (gIR->dtors.empty())
        return NULL;

    size_t n = gIR->dtors.size();
    if (n == 1)
        return gIR->dtors[0]->ir.irFunc->func;

    std::string name("_D");
    name.append(gIR->dmodule->mangle());
    name.append("6__dtorZ");

    std::vector<const LLType*> argsTy;
    const llvm::FunctionType* fnTy = llvm::FunctionType::get(LLType::VoidTy,argsTy,false);
    assert(gIR->module->getFunction(name) == NULL);
    llvm::Function* fn = llvm::Function::Create(fnTy, llvm::GlobalValue::InternalLinkage, name, gIR->module);
    fn->setCallingConv(DtoCallingConv(0, LINKd));

    llvm::BasicBlock* bb = llvm::BasicBlock::Create("entry", fn);
    IRBuilder<> builder(bb);

    // debug info
    LLGlobalVariable* subprog;
    if(global.params.symdebug) {
        subprog = DtoDwarfSubProgramInternal(name.c_str(), name.c_str()).getGV();
        builder.CreateCall(gIR->module->getFunction("llvm.dbg.func.start"), DBG_CAST(subprog));
    }

    for (size_t i=0; i<n; i++) {
        llvm::Function* f = gIR->dtors[i]->ir.irFunc->func;
        llvm::CallInst* call = builder.CreateCall(f,"");
        call->setCallingConv(DtoCallingConv(0, LINKd));
    }

    // debug info end
    if(global.params.symdebug)
        builder.CreateCall(gIR->module->getFunction("llvm.dbg.region.end"), DBG_CAST(subprog));

    builder.CreateRetVoid();
    return fn;
}

// build module unittest

static llvm::Function* build_module_unittest()
{
    if (gIR->unitTests.empty())
        return NULL;

    size_t n = gIR->unitTests.size();
    if (n == 1)
        return gIR->unitTests[0]->ir.irFunc->func;

    std::string name("_D");
    name.append(gIR->dmodule->mangle());
    name.append("10__unittestZ");

    std::vector<const LLType*> argsTy;
    const llvm::FunctionType* fnTy = llvm::FunctionType::get(LLType::VoidTy,argsTy,false);
    assert(gIR->module->getFunction(name) == NULL);
    llvm::Function* fn = llvm::Function::Create(fnTy, llvm::GlobalValue::InternalLinkage, name, gIR->module);
    fn->setCallingConv(DtoCallingConv(0, LINKd));

    llvm::BasicBlock* bb = llvm::BasicBlock::Create("entry", fn);
    IRBuilder<> builder(bb);

    // debug info
    LLGlobalVariable* subprog;
    if(global.params.symdebug) {
        subprog = DtoDwarfSubProgramInternal(name.c_str(), name.c_str()).getGV();
        builder.CreateCall(gIR->module->getFunction("llvm.dbg.func.start"), DBG_CAST(subprog));
    }

    for (size_t i=0; i<n; i++) {
        llvm::Function* f = gIR->unitTests[i]->ir.irFunc->func;
        llvm::CallInst* call = builder.CreateCall(f,"");
        call->setCallingConv(DtoCallingConv(0, LINKd));
    }

    // debug info end
    if(global.params.symdebug)
        builder.CreateCall(gIR->module->getFunction("llvm.dbg.region.end"), DBG_CAST(subprog));

    builder.CreateRetVoid();
    return fn;
}

// build ModuleReference and register function, to register the module info in the global linked list
static LLFunction* build_module_reference_and_ctor(LLConstant* moduleinfo)
{
    // build ctor type
    const LLFunctionType* fty = LLFunctionType::get(LLType::VoidTy, std::vector<const LLType*>(), false);

    // build ctor name
    std::string fname = "_D";
    fname += gIR->dmodule->mangle();
    fname += "16__moduleinfoCtorZ";

    // build a function that registers the moduleinfo in the global moduleinfo linked list
    LLFunction* ctor = LLFunction::Create(fty, LLGlobalValue::InternalLinkage, fname, gIR->module);

    // provide the default initializer
    const LLStructType* modulerefTy = DtoModuleReferenceType();
    std::vector<LLConstant*> mrefvalues;
    mrefvalues.push_back(LLConstant::getNullValue(modulerefTy->getContainedType(0)));
    mrefvalues.push_back(llvm::ConstantExpr::getBitCast(moduleinfo, modulerefTy->getContainedType(1)));
    LLConstant* thismrefinit = LLConstantStruct::get(modulerefTy, mrefvalues);

    // create the ModuleReference node for this module
    std::string thismrefname = "_D";
    thismrefname += gIR->dmodule->mangle();
    thismrefname += "11__moduleRefZ";
    LLGlobalVariable* thismref = new LLGlobalVariable(modulerefTy, false, LLGlobalValue::InternalLinkage, thismrefinit, thismrefname, gIR->module);

    // make sure _Dmodule_ref is declared
    LLGlobalVariable* mref = gIR->module->getNamedGlobal("_Dmodule_ref");
    if (!mref)
        mref = new LLGlobalVariable(getPtrToType(modulerefTy), false, LLGlobalValue::ExternalLinkage, NULL, "_Dmodule_ref", gIR->module);

    // make the function insert this moduleinfo as the beginning of the _Dmodule_ref linked list
    llvm::BasicBlock* bb = llvm::BasicBlock::Create("moduleinfoCtorEntry", ctor);
    IRBuilder<> builder(bb);

    // debug info
    LLGlobalVariable* subprog;
    if(global.params.symdebug) {
        subprog = DtoDwarfSubProgramInternal(fname.c_str(), fname.c_str()).getGV();
        builder.CreateCall(gIR->module->getFunction("llvm.dbg.func.start"), DBG_CAST(subprog));
    }

    // get current beginning
    LLValue* curbeg = builder.CreateLoad(mref, "current");

    // put current beginning as the next of this one
    LLValue* gep = builder.CreateStructGEP(thismref, 0, "next");
    builder.CreateStore(curbeg, gep);

    // replace beginning
    builder.CreateStore(thismref, mref);

    // debug info end
    if(global.params.symdebug)
        builder.CreateCall(gIR->module->getFunction("llvm.dbg.region.end"), DBG_CAST(subprog));

    // return
    builder.CreateRetVoid();

    return ctor;
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
//         }

    // resolve ModuleInfo
    if (!moduleinfo)
    {
        error("object.d is missing the ModuleInfo class");
        fatal();
    }

    moduleinfo->codegen(Type::sir);

    // check for patch
    if (moduleinfo->fields.dim != 9)
    {
        error("object.d ModuleInfo class is incorrect");
        fatal();
    }

    // moduleinfo llvm struct type
    const llvm::StructType* moduleinfoTy = isaStruct(moduleinfo->type->ir.type->get());
    // classinfo llvm struct type
    const llvm::StructType* classinfoTy = isaStruct(ClassDeclaration::classinfo->type->ir.type->get());

    // initializer vector
    std::vector<LLConstant*> initVec;
    LLConstant* c = 0;

    // vtable
    c = moduleinfo->ir.irStruct->vtbl;
    initVec.push_back(c);

    // monitor
    c = getNullPtr(getPtrToType(LLType::Int8Ty));
    initVec.push_back(c);

    // name
    char *name = toPrettyChars();
    c = DtoConstString(name);
    initVec.push_back(c);

    // importedModules[]
    int aimports_dim = aimports.dim;
    std::vector<LLConstant*> importInits;
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
        if (!m_gvar) m_gvar = new llvm::GlobalVariable(moduleinfoTy, false, llvm::GlobalValue::ExternalLinkage, NULL, m_name, gIR->module);
        importInits.push_back(m_gvar);
    }
    // has import array?
    if (!importInits.empty())
    {
        const llvm::ArrayType* importArrTy = llvm::ArrayType::get(getPtrToType(moduleinfoTy), importInits.size());
        c = llvm::ConstantArray::get(importArrTy, importInits);
        std::string m_name("_D");
        m_name.append(mangle());
        m_name.append("9__importsZ");
        llvm::GlobalVariable* m_gvar = gIR->module->getGlobalVariable(m_name);
        if (!m_gvar) m_gvar = new llvm::GlobalVariable(importArrTy, true, llvm::GlobalValue::InternalLinkage, c, m_name, gIR->module);
        c = llvm::ConstantExpr::getBitCast(m_gvar, getPtrToType(importArrTy->getElementType()));
        c = DtoConstSlice(DtoConstSize_t(importInits.size()), c);
    }
    else
        c = DtoConstSlice( DtoConstSize_t(0), getNullValue(getPtrToType(moduleinfoTy)) );
    initVec.push_back(c);

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
        assert(cd->ir.irStruct->classInfo);
        c = DtoBitCast(cd->ir.irStruct->classInfo, getPtrToType(classinfoTy));
        classInits.push_back(c);
    }
    // has class array?
    if (!classInits.empty())
    {
        const llvm::ArrayType* classArrTy = llvm::ArrayType::get(getPtrToType(classinfoTy), classInits.size());
        c = llvm::ConstantArray::get(classArrTy, classInits);
        std::string m_name("_D");
        m_name.append(mangle());
        m_name.append("9__classesZ");
        assert(gIR->module->getGlobalVariable(m_name) == NULL);
        llvm::GlobalVariable* m_gvar = new llvm::GlobalVariable(classArrTy, true, llvm::GlobalValue::InternalLinkage, c, m_name, gIR->module);
        c = DtoBitCast(m_gvar, getPtrToType(classinfoTy));
        c = DtoConstSlice(DtoConstSize_t(classInits.size()), c);
    }
    else
        c = DtoConstSlice( DtoConstSize_t(0), getNullValue(getPtrToType(classinfoTy)) );
    initVec.push_back(c);

    // flags
    c = DtoConstUint(0);
    if (!needmoduleinfo)
        c = DtoConstUint(4);        // flags (4 means MIstandalone)
    initVec.push_back(c);

    // function pointer type for next three fields
    const LLType* fnptrTy = getPtrToType(LLFunctionType::get(LLType::VoidTy, std::vector<const LLType*>(), false));

    // ctor
    llvm::Function* fctor = build_module_ctor();
    c = fctor ? fctor : getNullValue(fnptrTy);
    initVec.push_back(c);

    // dtor
    llvm::Function* fdtor = build_module_dtor();
    c = fdtor ? fdtor : getNullValue(fnptrTy);
    initVec.push_back(c);

    // unitTest
    llvm::Function* unittest = build_module_unittest();
    c = unittest ? unittest : getNullValue(fnptrTy);
    initVec.push_back(c);

    // xgetMembers
    c = getNullValue(getVoidPtrType());
    initVec.push_back(c);

    // ictor
    c = getNullValue(fnptrTy);
    initVec.push_back(c);

    /*Logger::println("MODULE INFO INITIALIZERS");
    for (size_t i=0; i<initVec.size(); ++i)
    {
        Logger::cout() << *initVec[i] << '\n';
        if (initVec[i]->getType() != moduleinfoTy->getElementType(i))
            assert(0);
    }*/

    // create initializer
    LLConstant* constMI = llvm::ConstantStruct::get(initVec);

    // create name
    std::string MIname("_D");
    MIname.append(mangle());
    MIname.append("8__ModuleZ");

    // declare global
    // flags will be modified at runtime so can't make it constant

    // it makes no sense that the our own module info already exists!
    assert(!gIR->module->getGlobalVariable(MIname));
    llvm::GlobalVariable* gvar = new llvm::GlobalVariable(constMI->getType(), false, llvm::GlobalValue::ExternalLinkage, constMI, MIname, gIR->module);

    // build the modulereference and ctor for registering it
    LLFunction* mictor = build_module_reference_and_ctor(gvar);

    // register this ctor in the magic llvm.global_ctors appending array
    const LLFunctionType* magicfty = LLFunctionType::get(LLType::VoidTy, std::vector<const LLType*>(), false);
    std::vector<const LLType*> magictypes;
    magictypes.push_back(LLType::Int32Ty);
    magictypes.push_back(getPtrToType(magicfty));
    const LLStructType* magicsty = LLStructType::get(magictypes);

    // make the constant element
    std::vector<LLConstant*> magicconstants;
    magicconstants.push_back(DtoConstUint(65535));
    magicconstants.push_back(mictor);
    LLConstant* magicinit = LLConstantStruct::get(magicsty, magicconstants);

    // declare the appending array
    const llvm::ArrayType* appendArrTy = llvm::ArrayType::get(magicsty, 1);
    std::vector<LLConstant*> appendInits(1, magicinit);
    LLConstant* appendInit = llvm::ConstantArray::get(appendArrTy, appendInits);
    std::string appendName("llvm.global_ctors");
    llvm::GlobalVariable* appendVar = new llvm::GlobalVariable(appendArrTy, true, llvm::GlobalValue::AppendingLinkage, appendInit, appendName, gIR->module);
}
