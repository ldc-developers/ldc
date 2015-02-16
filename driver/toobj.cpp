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
#if LDC_LLVM_VER >= 305
#include "llvm/IR/Verifier.h"
#else
#include "llvm/Analysis/Verifier.h"
#endif
#include "llvm/Bitcode/ReaderWriter.h"
#if LDC_LLVM_VER >= 307
#include "llvm/IR/LegacyPassManager.h"
#elif
#include "llvm/PassManager.h"
#endif
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/Program.h"
#if LDC_LLVM_VER < 304
#include "llvm/Support/PathV1.h"
#endif
#include "llvm/Target/TargetMachine.h"
#if LDC_LLVM_VER >= 307
#include "llvm/Analysis/TargetTransformInfo.h"
#endif
#if LDC_LLVM_VER >= 306
#include "llvm/Target/TargetSubtargetInfo.h"
#endif
#if LDC_LLVM_VER >= 303
#include "llvm/IR/Module.h"
#else
#include "llvm/Module.h"
#endif
#include <cstddef>
#include <fstream>

#if LDC_LLVM_VER < 304
namespace llvm {
namespace sys {
namespace fs {
enum OpenFlags {
  F_Excl = llvm::raw_fd_ostream::F_Excl,
  F_Append = llvm::raw_fd_ostream::F_Append,
  F_Binary = llvm::raw_fd_ostream::F_Binary
};

}
}
}
#endif

// based on llc code, University of Illinois Open Source License
static void codegenModule(llvm::TargetMachine &Target, llvm::Module& m,
    llvm::raw_fd_ostream& out, llvm::TargetMachine::CodeGenFileType fileType)
{
    using namespace llvm;

    // Create a PassManager to hold and optimize the collection of passes we are
    // about to build.
#if LDC_LLVM_VER >= 307
    legacy::
#endif
    PassManager Passes;

#if LDC_LLVM_VER >= 306
    Passes.add(new DataLayoutPass());
#elif LDC_LLVM_VER == 305
    if (const DataLayout *DL = Target.getDataLayout())
        Passes.add(new DataLayoutPass(*DL));
    else
        Passes.add(new DataLayoutPass(&m));
#elif LDC_LLVM_VER >= 302
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

#if LDC_LLVM_VER >= 307
    // Add internal analysis passes from the target machine.
    Passes.add(createTargetTransformInfoWrapperPass(Target.getTargetIRAnalysis()));
#elif LDC_LLVM_VER >= 303
    Target.addAnalysisPasses(Passes);
#endif

    llvm::formatted_raw_ostream fout(out);
    if (Target.addPassesToEmitFile(Passes, fout, fileType, codeGenOptLevel()))
        llvm_unreachable("no support for asm output");

    Passes.run(m);
}

static void assemble(const std::string &asmpath, const std::string &objpath)
{
    std::vector<std::string> args;
    args.push_back("-O3");
    args.push_back("-c");
    args.push_back("-xassembler");
    args.push_back(asmpath);
    args.push_back("-o");
    args.push_back(objpath);

    if (global.params.is64bit)
        args.push_back("-m64");
    else
        args.push_back("-m32");

    // Run the compiler to assembly the program.
    std::string gcc(getGcc());
    int R = executeToolAndWait(gcc, args, global.params.verbose);
    if (R)
    {
        error(Loc(), "Error while invoking external assembler.");
        fatal();
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

void writeModule(llvm::Module* m, std::string filename)
{
    // run optimizer
    ldc_optimize_module(m);

#if LDC_LLVM_VER >= 305
    // There is no integrated assembler on AIX because XCOFF is not supported.
    // Starting with LLVM 3.5 the integrated assembler can be used with MinGW.
    bool const assembleExternally = global.params.output_o &&
        global.params.targetTriple.getOS() == llvm::Triple::AIX;
#else
    // (We require LLVM 3.5 with AIX.)
    // We don't use the integrated assembler with MinGW as it does not support
    // emitting DW2 exception handling tables.
    bool const assembleExternally = global.params.output_o &&
        global.params.targetTriple.getOS() == llvm::Triple::MinGW32;
#endif

    // eventually do our own path stuff, dmd's is a bit strange.
    typedef llvm::SmallString<128> LLPath;

    // write LLVM bitcode
    if (global.params.output_bc) {
        LLPath bcpath = LLPath(filename);
        llvm::sys::path::replace_extension(bcpath, global.bc_ext);
        Logger::println("Writing LLVM bitcode to: %s\n", bcpath.c_str());
#if LDC_LLVM_VER >= 306
        std::error_code errinfo;
#else
        std::string errinfo;
#endif
        llvm::raw_fd_ostream bos(bcpath.c_str(), errinfo, 
#if LDC_LLVM_VER >= 305
                llvm::sys::fs::F_None
#else
                llvm::sys::fs::F_Binary
#endif
                );
        if (bos.has_error())
        {
            error(Loc(), "cannot write LLVM bitcode file '%s': %s", bcpath.c_str(),
#if LDC_LLVM_VER >= 306
            errinfo
#else
            errinfo.c_str()
#endif
            );
            fatal();
        }
        llvm::WriteBitcodeToFile(m, bos);
    }

    // write LLVM IR
    if (global.params.output_ll) {
        LLPath llpath = LLPath(filename);
        llvm::sys::path::replace_extension(llpath, global.ll_ext);
        Logger::println("Writing LLVM asm to: %s\n", llpath.c_str());
#if LDC_LLVM_VER >= 306
        std::error_code errinfo;
#else
        std::string errinfo;
#endif
        llvm::raw_fd_ostream aos(llpath.c_str(), errinfo,
#if LDC_LLVM_VER >= 305
            llvm::sys::fs::F_None
#else
            llvm::sys::fs::F_Binary
#endif
            );
        if (aos.has_error())
        {
            error(Loc(), "cannot write LLVM asm file '%s': %s", llpath.c_str(),
#if LDC_LLVM_VER >= 306
            errinfo
#else
            errinfo.c_str()
#endif
            );
            fatal();
        }
        m->print(aos, NULL);
    }

    // write native assembly
    if (global.params.output_s || assembleExternally) {
#if LDC_LLVM_VER >= 304
        LLPath spath = LLPath(filename);
        llvm::sys::path::replace_extension(spath, global.s_ext);
        if (!global.params.output_s)
            llvm::sys::fs::createUniqueFile("ldc-%%%%%%%.s", spath);
#else
        // Pre-3.4 versions don't have a createUniqueFile overload that does
        // not open the file.
        llvm::sys::Path spath(filename);
        spath.eraseSuffix();
        spath.appendSuffix(std::string(global.s_ext));
        if (!global.params.output_s)
            spath.createTemporaryFileOnDisk();
#endif

        Logger::println("Writing native asm to: %s\n", spath.c_str());
#if LDC_LLVM_VER >= 306
        std::error_code errinfo;
#else
        std::string errinfo;
#endif
        {
            llvm::raw_fd_ostream out(spath.c_str(), errinfo,
#if LDC_LLVM_VER >= 305
                llvm::sys::fs::F_None
#else
                llvm::sys::fs::F_Binary
#endif
                );
#if LDC_LLVM_VER >= 306
            if (!errinfo)
#else
            if (errinfo.empty())
#endif
            {
                codegenModule(*gTargetMachine, *m, out, llvm::TargetMachine::CGFT_AssemblyFile);
            }
            else
            {
                error(Loc(), "cannot write native asm: %s",
#if LDC_LLVM_VER >= 306
                errinfo
#else
                errinfo.c_str()
#endif
                );
                fatal();
            }
        }

        if (assembleExternally)
        {
            LLPath objpath(filename);
            assemble(spath.str(), objpath.str());
        }

        if (!global.params.output_s)
        {
#if LDC_LLVM_VER < 305
            bool existed;
            llvm::sys::fs::remove(spath.str(), existed);
#else
            llvm::sys::fs::remove(spath.str());
#endif
        }
    }

    if (global.params.output_o && !assembleExternally) {
        LLPath objpath = LLPath(filename);
        Logger::println("Writing object file to: %s\n", objpath.c_str());
#if LDC_LLVM_VER >= 306
        std::error_code errinfo;
#else
        std::string errinfo;
#endif
        {
            llvm::raw_fd_ostream out(objpath.c_str(), errinfo, 
#if LDC_LLVM_VER >= 305
                llvm::sys::fs::F_None
#else
                llvm::sys::fs::F_Binary
#endif
                );
#if LDC_LLVM_VER >= 306
            if (!errinfo)
#else
            if (errinfo.empty())
#endif
            {
                codegenModule(*gTargetMachine, *m, out, llvm::TargetMachine::CGFT_ObjectFile);
            }
            else
            {
                error(Loc(), "cannot write object file: %s",
#if LDC_LLVM_VER >= 306
                errinfo
#else
                errinfo.c_str()
#endif
                );
                fatal();
            }
        }
    }
}
