
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
#include "llvm/System/Program.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/CodeGen/MachineCodeEmitter.h"

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
void write_asm_to_file(llvm::TargetMachine &Target, llvm::Module& m, llvm::raw_fd_ostream& Out);
void assemble(const llvm::sys::Path& asmpath, const llvm::sys::Path& objpath);

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
    for (int k=0; k < members->dim; k++) {
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
            llvm::raw_fd_ostream out(spath.c_str(), err);
            if (err.empty())
            {
                write_asm_to_file(*gTargetMachine, *m, out);
            }
            else
            {
                error("cannot write native asm: %s", err.c_str());
                fatal();
            }
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
    if (Target.addPassesToEmitFile(Passes, fout, TargetMachine::CGFT_AssemblyFile, LastArg))
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

// uses gcc to make an obj out of an assembly file
// based on llvm-ld code, University of Illinois Open Source License
void assemble(const llvm::sys::Path& asmpath, const llvm::sys::Path& objpath)
{
    using namespace llvm;

    sys::Path gcc = getGcc();

    // Run GCC to assemble and link the program into native code.
    //
    // Note:
    //  We can't just assemble and link the file with the system assembler
    //  and linker because we don't know where to put the _start symbol.
    //  GCC mysteriously knows how to do it.
    std::vector<std::string> args;
    args.push_back(gcc.str());
    args.push_back("-fno-strict-aliasing");
    args.push_back("-O3");
    args.push_back("-c");
    args.push_back("-xassembler");
    args.push_back(asmpath.str());
    args.push_back("-o");
    args.push_back(objpath.str());

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
        Stream logstr = Logger::cout();
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

static llvm::Function* build_module_function(const std::string &name, const std::list<FuncDeclaration*> &funcs)
{
    if (funcs.empty())
        return NULL;

    size_t n = funcs.size();
    if (n == 1)
        return funcs.front()->ir.irFunc->func;

    std::vector<const LLType*> argsTy;
    const llvm::FunctionType* fnTy = llvm::FunctionType::get(LLType::getVoidTy(gIR->context()),argsTy,false);
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

    typedef std::list<FuncDeclaration*>::const_iterator Iterator;
    for (Iterator itr = funcs.begin(), end = funcs.end(); itr != end; ++itr) {
        llvm::Function* f = (*itr)->ir.irFunc->func;
        llvm::CallInst* call = builder.CreateCall(f,"");
        call->setCallingConv(DtoCallingConv(0, LINKd));
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
    return build_module_function(name, gIR->ctors);
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
    return build_module_function(name, gIR->sharedCtors);
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
    const LLFunctionType* fty = LLFunctionType::get(LLType::getVoidTy(gIR->context()), std::vector<const LLType*>(), false);

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
    LLGlobalVariable* thismref = new LLGlobalVariable(*gIR->module, modulerefTy, false, LLGlobalValue::InternalLinkage, thismrefinit, thismrefname);

    // make sure _Dmodule_ref is declared
    LLGlobalVariable* mref = gIR->module->getNamedGlobal("_Dmodule_ref");
    if (!mref)
        mref = new LLGlobalVariable(*gIR->module, getPtrToType(modulerefTy), false, LLGlobalValue::ExternalLinkage, NULL, "_Dmodule_ref");

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
    const LLType* moduleinfoTy = moduleinfo->type->irtype->getPA();
    const LLType* classinfoTy = ClassDeclaration::classinfo->type->irtype->getPA();

    // name
    b.push_string(toPrettyChars());

    // importedModules[]
    int aimports_dim = aimports.dim;
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
        const llvm::ArrayType* importArrTy = llvm::ArrayType::get(getPtrToType(moduleinfoTy), importInits.size());
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
        const llvm::ArrayType* classArrTy = llvm::ArrayType::get(getPtrToType(classinfoTy), classInits.size());
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
    const LLType* fnptrTy = getPtrToType(LLFunctionType::get(LLType::getVoidTy(gIR->context()), std::vector<const LLType*>(), false));

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
    const LLType* AT = llvm::ArrayType::get(getVoidPtrType(), 2);
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

    // create initializer
    LLConstant* constMI = b.get_constant();

    // create name
    std::string MIname("_D");
    MIname.append(mangle());
    MIname.append("8__ModuleZ");

    // declare global
    // flags will be modified at runtime so can't make it constant

    // it makes no sense that the our own module info already exists!
    assert(!gIR->module->getGlobalVariable(MIname));
    llvm::GlobalVariable* gvar = new llvm::GlobalVariable(*gIR->module, constMI->getType(), false, llvm::GlobalValue::ExternalLinkage, constMI, MIname);

    // build the modulereference and ctor for registering it
    LLFunction* mictor = build_module_reference_and_ctor(gvar);

    // register this ctor in the magic llvm.global_ctors appending array
    const LLFunctionType* magicfty = LLFunctionType::get(LLType::getVoidTy(gIR->context()), std::vector<const LLType*>(), false);
    std::vector<const LLType*> magictypes;
    magictypes.push_back(LLType::getInt32Ty(gIR->context()));
    magictypes.push_back(getPtrToType(magicfty));
    const LLStructType* magicsty = LLStructType::get(gIR->context(), magictypes);

    // make the constant element
    std::vector<LLConstant*> magicconstants;
    magicconstants.push_back(DtoConstUint(65535));
    magicconstants.push_back(mictor);
    LLConstant* magicinit = LLConstantStruct::get(magicsty, magicconstants);

    // declare the appending array
    const llvm::ArrayType* appendArrTy = llvm::ArrayType::get(magicsty, 1);
    std::vector<LLConstant*> appendInits(1, magicinit);
    LLConstant* appendInit = LLConstantArray::get(appendArrTy, appendInits);
    std::string appendName("llvm.global_ctors");
    llvm::GlobalVariable* appendVar = new llvm::GlobalVariable(*gIR->module, appendArrTy, true, llvm::GlobalValue::AppendingLinkage, appendInit, appendName);
}
