
// Copyright (c) 1999-2004 by Digital Mars
// All Rights Reserved
// written by Walter Bright
// www.digitalmars.com
// License for redistribution is by either the Artistic License
// in artistic.txt, or the GNU General Public License in gnu.txt.
// See the included readme.txt for details.

#include <cstddef>
#include <fstream>

#include "llvm/Analysis/Verifier.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Target/TargetMachine.h"

#include "gen/irstate.h"
#include "gen/logger.h"
#include "gen/optimizer.h"


// fwd decl
void emit_file(llvm::TargetMachine &Target, llvm::Module& m, llvm::raw_fd_ostream& Out,
               llvm::TargetMachine::CodeGenFileType fileType);

//////////////////////////////////////////////////////////////////////////////////////////

void writeModule(llvm::Module* m, std::string filename)
{
    // run optimizer
    bool reverify = ldc_optimize_module(m);

    // verify the llvm
    if (!global.params.noVerify && reverify) {
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
            llvm::raw_fd_ostream out(objpath.c_str(), err, llvm::raw_fd_ostream::F_Binary);
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
