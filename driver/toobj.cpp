//===-- toobj.cpp ---------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "driver/toobj.h"
#include "driver/tool.h"
#include "gen/irstate.h"
#include "gen/logger.h"
#include "gen/optimizer.h"
#include "gen/programs.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/PassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/Program.h"
#include "llvm/Target/TargetMachine.h"
#if LDC_LLVM_VER >= 303
#include "llvm/IR/Module.h"
#else
#include "llvm/Module.h"
#endif
#include <cstddef>
#include <fstream>

// based on llc code, University of Illinois Open Source License
static void codegenModule(llvm::TargetMachine &Target, llvm::Module& m,
    llvm::raw_fd_ostream& out, llvm::TargetMachine::CodeGenFileType fileType)
{
    using namespace llvm;

    // Build up all of the passes that we want to do to the module.
    FunctionPassManager Passes(&m);

#if LDC_LLVM_VER >= 302
    if (const DataLayout *DL = Target.getDataLayout())
        Passes.add(new DataLayout(*DL));
    else
        Passes.add(new DataLayout(&m));
#else
    if (const TargetData *TD = Target.getTargetData())
        Passes.add(new TargetData(*TD));
    else
        Passes.add(new TargetData(&m));
#endif

#if LDC_LLVM_VER >= 303
    Target.addAnalysisPasses(Passes);
#endif

    llvm::formatted_raw_ostream fout(out);
    if (Target.addPassesToEmitFile(Passes, fout, fileType, codeGenOptLevel()))
        llvm_unreachable("no support for asm output");

    Passes.doInitialization();

    // Run our queue of passes all at once now, efficiently.
    for (llvm::Module::iterator I = m.begin(), E = m.end(); I != E; ++I)
        if (!I->isDeclaration())
            Passes.run(*I);

    Passes.doFinalization();
}

static void assemble(const llvm::sys::Path& asmpath, const llvm::sys::Path& objpath)
{
    std::vector<std::string> args;
    args.push_back("-O3");
    args.push_back("-c");
    args.push_back("-xassembler");
    args.push_back(asmpath.str());
    args.push_back("-o");
    args.push_back(objpath.str());

    if (global.params.is64bit)
        args.push_back("-m64");
    else
        args.push_back("-m32");

    // Run the compiler to assembly the program.
    llvm::sys::Path gcc(getGcc());
    int R = executeToolAndWait(gcc, args, global.params.verbose);
    if (R)
    {
        error("Error while invoking external assembler.");
        fatal();
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

void writeModule(llvm::Module* m, std::string filename)
{
    // run optimizer
    ldc_optimize_module(m);

    // We don't use the integrated assembler with MinGW as it does not support
    // emitting DW2 exception handling tables.
    bool const assembleExternally = global.params.output_o &&
        global.params.targetTriple.getOS() == llvm::Triple::MinGW32;

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
    if (global.params.output_s || assembleExternally) {
        LLPath spath = LLPath(filename);
        spath.eraseSuffix();
        spath.appendSuffix(std::string(global.s_ext));
        if (!global.params.output_s) spath.createTemporaryFileOnDisk();

        Logger::println("Writing native asm to: %s\n", spath.c_str());
        std::string err;
        {
            llvm::raw_fd_ostream out(spath.c_str(), err);
            if (err.empty())
            {
                codegenModule(*gTargetMachine, *m, out, llvm::TargetMachine::CGFT_AssemblyFile);
            }
            else
            {
                error("cannot write native asm: %s", err.c_str());
                fatal();
            }
        }

        if (assembleExternally)
        {
            LLPath objpath(filename);
            assemble(spath, objpath);
        }

        if (!global.params.output_s) spath.eraseFromDisk();
    }

    if (global.params.output_o && !assembleExternally) {
        LLPath objpath = LLPath(filename);
        Logger::println("Writing object file to: %s\n", objpath.c_str());
        std::string err;
        {
            llvm::raw_fd_ostream out(objpath.c_str(), err, llvm::raw_fd_ostream::F_Binary);
            if (err.empty())
            {
                codegenModule(*gTargetMachine, *m, out, llvm::TargetMachine::CGFT_ObjectFile);
            }
            else
            {
                error("cannot write object file: %s", err.c_str());
                fatal();
            }
        }
    }
}
