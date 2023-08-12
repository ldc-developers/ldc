//===-- toobj.cpp ---------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "driver/toobj.h"

#include "dmd/errors.h"
#include "driver/cl_options.h"
#include "driver/cache.h"
#include "driver/targetmachine.h"
#include "driver/timetrace.h"
#include "driver/tool.h"
#include "gen/irstate.h"
#include "gen/logger.h"
#include "gen/optimizer.h"
#include "gen/passes/Passes.h"
#include "llvm/IR/AssemblyAnnotationWriter.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Analysis/ModuleSummaryAnalysis.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/IR/Module.h"
#ifdef LDC_LLVM_SUPPORTED_TARGET_SPIRV
#if LDC_LLVM_VER < 1600
#include "LLVMSPIRVLib/LLVMSPIRVLib.h"
#endif
#endif
#include <cstddef>
#include <fstream>

using CodeGenFileType = llvm::CodeGenFileType;

static llvm::cl::opt<bool>
    NoIntegratedAssembler("no-integrated-as", llvm::cl::ZeroOrMore,
                          llvm::cl::Hidden,
                          llvm::cl::desc("Disable integrated assembler"));

namespace {

// The dllimport relocation pass on Windows is *not* an optimization pass.
// We run it separately right after the optimization passes, in order to
// finalize the IR - e.g., for -output-{bc,ll}, which are dumped before
// (potentially) running the codegen passes below in codegenModule().
void runDLLImportRelocationPass(llvm::TargetMachine &Target, llvm::Module &m) {
  llvm::legacy::PassManager pm;
  pm.add(createDLLImportRelocationPass());
  pm.run(m);
}

// based on llc code, University of Illinois Open Source License
void codegenModule(llvm::TargetMachine &Target, llvm::Module &m,
                   const char *filename,
                   CodeGenFileType fileType) {
  using namespace llvm;

  const ComputeBackend::Type cb = getComputeTargetType(&m);

  if (cb == ComputeBackend::SPIRV) {
#ifdef LDC_LLVM_SUPPORTED_TARGET_SPIRV
#if LDC_LLVM_VER < 1600
    IF_LOG Logger::println("running createSPIRVWriterPass()");
    std::ofstream out(filename, std::ofstream::binary);
    llvm::createSPIRVWriterPass(out)->runOnModule(m);
    IF_LOG Logger::println("Success.");
    return;
#endif
#else
    error(Loc(), "Trying to target SPIRV, but LDC is not built to do so!");
    return;
#endif
  }

  std::error_code errinfo;
  llvm::ToolOutputFile out(filename, errinfo, llvm::sys::fs::OF_None);
  if (errinfo) {
    error(Loc(), "cannot write file '%s': %s", filename,
          errinfo.message().c_str());
    fatal();
  }

  // The DataLayout is already set at the module (in module.cpp,
  // method Module::genLLVMModule())
  // FIXME: Introduce new command line switch default-data-layout to
  // override the module data layout

  // Create a PassManager to hold and optimize the collection of passes we are
  // about to build.
  legacy::PassManager Passes;

  // Add internal analysis passes from the target machine.
  Passes.add(
      createTargetTransformInfoWrapperPass(Target.getTargetIRAnalysis()));

  // Add an appropriate TargetLibraryInfo pass for the module's triple.
  auto tlii = createTLII(m);
  Passes.add(new llvm::TargetLibraryInfoWrapperPass(*tlii));

  if (Target.addPassesToEmitFile(
          Passes,
          out.os(), // Output file
          nullptr,  // DWO output file
          // Always generate assembly for ptx as it is an assembly format
          // The PTX backend fails if we pass anything else.
          (cb == ComputeBackend::NVPTX) ? CGFT_AssemblyFile : fileType,
          codeGenOptLevel())) {
    llvm_unreachable("no support for asm output");
  }

  Passes.run(m);

  // Terminate upon errors during the LLVM passes.
  if (global.errors || global.warnings) {
    Logger::println("Aborting because of errors/warnings during LLVM passes");
    fatal();
  }

  out.keep();
}

}

static void assemble(const std::string &asmpath, const std::string &objpath) {
  std::vector<std::string> args;
  args.push_back("-O3");
  args.push_back("-c");
  args.push_back("-xassembler");
  args.push_back(asmpath);
  args.push_back("-o");
  args.push_back(objpath);

  appendTargetArgsForGcc(args);

  // Run the compiler to assembly the program.
  int R = executeToolAndWait(Loc(), getGcc(), args, global.params.verbose);
  if (R) {
    error(Loc(), "Error while invoking external assembler.");
    fatal();
  }
}

////////////////////////////////////////////////////////////////////////////////

namespace {
using namespace llvm;

class AssemblyAnnotator : public AssemblyAnnotationWriter {
// Find the MDNode which corresponds to the DISubprogram data that described F.
  static DISubprogram *FindSubprogram(const Function *F,
                                      DebugInfoFinder &Finder)
  {
    for (DISubprogram *Subprogram : Finder.subprograms())
      if (Subprogram->describes(F))
        return Subprogram;
    return nullptr;
  }

  static llvm::StringRef GetDisplayName(const Function *F) {
    llvm::DebugInfoFinder Finder;
    Finder.processModule(*F->getParent());
    if (DISubprogram *N = FindSubprogram(F, Finder)) {
      return N->getName();
    }
    return "";
  }

  const llvm::DataLayout &DL;

public:
  AssemblyAnnotator(const llvm::DataLayout &dl) : DL(dl) {}

  void emitFunctionAnnot(const Function *F,
                         formatted_raw_ostream &os) override {
    os << "; [#uses = " << F->getNumUses() << ']';

    // show demangled name
    llvm::StringRef funcName = GetDisplayName(F);
    if (!funcName.empty()) {
      os << " [display name = " << funcName << ']';
    }
    os << '\n';
  }

  void printInfoComment(const Value &val, formatted_raw_ostream &os) override {
    bool padding = false;
    if (!val.getType()->isVoidTy()) {
      os.PadToColumn(50);
      padding = true;
      os << "; [#uses = " << val.getNumUses();
      if (isa<GetElementPtrInst>(&val) || isa<PHINode>(&val)) {
        // Only print type for instructions where it is not obvious
        // from being repeated in its parameters. Might need to be
        // extended, but GEPs/PHIs are the most common ones.
        os << ", type = " << *val.getType();
      } else if (auto ai = dyn_cast<AllocaInst>(&val)) {
        os << ", size/byte = "
           << DL.getTypeAllocSize(ai->getAllocatedType());
      }
      os << ']';
    }

    const Instruction *instr = dyn_cast<Instruction>(&val);
    if (!instr) {
      return;
    }

    if (const DebugLoc &debugLoc = instr->getDebugLoc())
    {
      if (!padding) {
        os.PadToColumn(50);
        padding = true;
        os << ';';
      }
      os << " [debug line = ";
      debugLoc.print(os);
      os << ']';
    }
    if (const DbgDeclareInst *DDI = dyn_cast<DbgDeclareInst>(instr)) {
      DILocalVariable *Var(DDI->getVariable());
      if (!padding) {
        os.PadToColumn(50);
        os << ";";
      }
      os << " [debug variable = " << Var->getName() << ']';
    } else if (const DbgValueInst *DVI = dyn_cast<DbgValueInst>(instr)) {
      DILocalVariable *Var(DVI->getVariable());
      if (!padding) {
        os.PadToColumn(50);
        os << ";";
      }
      os << " [debug variable = " << Var->getName() << ']';
    } else if (const CallInst *callinstr = dyn_cast<CallInst>(instr)) {
      const Function *F = callinstr->getCalledFunction();
      if (!F) {
        return;
      }

      StringRef funcName = GetDisplayName(F);
      if (!funcName.empty()) {
        if (!padding) {
          os.PadToColumn(50);
          os << ";";
        }
        os << " [display name = " << funcName << ']';
      }
    } else if (const InvokeInst *invokeinstr = dyn_cast<InvokeInst>(instr)) {
      const Function *F = invokeinstr->getCalledFunction();
      if (!F) {
        return;
      }

      StringRef funcName = GetDisplayName(F);
      if (!funcName.empty()) {
        if (!padding) {
          os.PadToColumn(50);
          os << ";";
        }
        os << " [display name = " << funcName << ']';
      }
    }
  }
};

void writeObjectFile(llvm::Module *m, const char *filename) {
  IF_LOG Logger::println("Writing object file to: %s", filename);
  codegenModule(*gTargetMachine, *m, filename,
                CGFT_ObjectFile);
}

bool shouldAssembleExternally() {
  // There is no integrated assembler on AIX because XCOFF is not supported.
  // Starting with LLVM 3.5 the integrated assembler can be used with MinGW.
  return global.params.output_o &&
         (NoIntegratedAssembler ||
          global.params.targetTriple->getOS() == llvm::Triple::AIX);
}

bool shouldOutputObjectFile() {
  return global.params.output_o && !shouldAssembleExternally();
}
} // end of anonymous namespace

std::string replaceExtensionWith(const DArray<const char> &ext,
                                 const char *filename) {
  const auto outputFlags = {global.params.output_o, global.params.output_bc,
                            global.params.output_ll, global.params.output_s,
                            global.params.output_mlir};
  const auto numOutputFiles =
      std::count_if(outputFlags.begin(), outputFlags.end(),
                    [](OUTPUTFLAG flag) { return flag != 0; });

  if (numOutputFiles == 1)
    return filename;
  llvm::SmallString<128> buffer(filename);
  llvm::sys::path::replace_extension(buffer,
                                     llvm::StringRef(ext.ptr, ext.length));
  return {buffer.data(), buffer.size()};
}

void writeModule(llvm::Module *m, const char *filename) {
  const bool doLTO = opts::isUsingLTO();
  const bool outputObj = shouldOutputObjectFile();
  const bool assembleExternally = shouldAssembleExternally();

  // Use cached object code if possible.
  // TODO: combine LDC's cache and LTO (the advantage is skipping the IR
  // optimization).
  const bool useIR2ObjCache = !opts::cacheDir.empty() && outputObj && !doLTO;
  llvm::SmallString<32> moduleHash;
  if (useIR2ObjCache) {
    ::TimeTraceScope timeScope("Check object cache", filename);
    llvm::SmallString<128> cacheDir(opts::cacheDir.c_str());
    llvm::sys::fs::make_absolute(cacheDir);
    opts::cacheDir = cacheDir.c_str();

    IF_LOG Logger::println("Use IR-to-Object cache in %s",
                           opts::cacheDir.c_str());
    LOG_SCOPE

    cache::calculateModuleHash(m, moduleHash);
    std::string cacheFile = cache::cacheLookup(moduleHash);
    if (!cacheFile.empty()) {
      cache::recoverObjectFile(moduleHash, filename);
      return;
    }
  }

  // run LLVM optimization passes
  {
    ::TimeTraceScope timeScope("Optimize", filename);
    ldc_optimize_module(m);
  }

  if (global.params.dllimport != DLLImport::none) {
    ::TimeTraceScope timeScope("dllimport relocation", filename);
    runDLLImportRelocationPass(*gTargetMachine, *m);
  }

  // Check if there are any errors before writing files.
  // Note: LLVM passes can add new warnings/errors (warnings become errors with
  // `-w`) such that we reach here with errors that did not trigger earlier
  // termination of the compiler.
  if (global.errors) {
    Logger::println("Aborting because of errors");
    fatal();
  }

  // Everything beyond this point is writing file(s) to disk.
  ::TimeTraceScope timeScope("Write file(s)", filename);

  // make sure the output directory exists
  const auto directory = llvm::sys::path::parent_path(filename);
  if (!directory.empty()) {
    if (auto ec = llvm::sys::fs::create_directories(directory)) {
      error(Loc(), "failed to create output directory: %s\n%s",
            directory.data(), ec.message().c_str());
      fatal();
    }
  }

  // write LLVM bitcode
  const bool emitBitcodeAsObjectFile =
      doLTO && outputObj && !global.params.output_bc;
  if (global.params.output_bc || emitBitcodeAsObjectFile) {
    std::string bcpath = emitBitcodeAsObjectFile
                             ? filename
                             : replaceExtensionWith(bc_ext, filename);
    Logger::println("Writing LLVM bitcode to: %s\n", bcpath.c_str());
    std::error_code errinfo;
    llvm::ToolOutputFile bos(bcpath.c_str(), errinfo, llvm::sys::fs::OF_None);
    if (bos.os().has_error()) {
      error(Loc(), "cannot write LLVM bitcode file '%s': %s", bcpath.c_str(),
            errinfo.message().c_str());
      fatal();
    }

    auto &M = *m;

    if (opts::isUsingThinLTO()) {
      Logger::println("Creating module summary for ThinLTO");

      llvm::ProfileSummaryInfo PSI(*m);

      // When the function freq info callback is set to nullptr, LLVM will
      // calculate it automatically for us.
      auto moduleSummaryIndex = buildModuleSummaryIndex(
          *m, /* function freq callback */ nullptr, &PSI);

      llvm::WriteBitcodeToFile(M, bos.os(), true, &moduleSummaryIndex,
                               /* generate ThinLTO hash */ true);
    } else {
      llvm::WriteBitcodeToFile(M, bos.os());
    }

    // Terminate upon errors during the LLVM passes.
    if (global.errors || global.warnings) {
      Logger::println(
          "Aborting because of errors/warnings during bitcode LLVM passes");
      fatal();
    }

    bos.keep();
  }

  // write LLVM IR
  if (global.params.output_ll) {
    const auto llpath = replaceExtensionWith(ll_ext, filename);
    Logger::println("Writing LLVM IR to: %s\n", llpath.c_str());
    std::error_code errinfo;
    llvm::ToolOutputFile aos(llpath.c_str(), errinfo, llvm::sys::fs::OF_None);
    if (aos.os().has_error()) {
      error(Loc(), "cannot write LLVM IR file '%s': %s", llpath.c_str(),
            errinfo.message().c_str());
      fatal();
    }
    AssemblyAnnotator annotator(m->getDataLayout());
    m->print(aos.os(), &annotator);

    // Terminate upon errors during the LLVM passes.
    if (global.errors || global.warnings) {
      Logger::println("Aborting because of errors/warnings during LLVM passes");
      fatal();
    }

    aos.keep();
  }

  const bool writeObj = outputObj && !emitBitcodeAsObjectFile;
  // write native assembly
  if (global.params.output_s || assembleExternally) {
    std::string spath;
    if (!global.params.output_s) {
      llvm::SmallString<16> buffer;
      llvm::sys::fs::createUniqueFile("ldc-%%%%%%%.s", buffer);
      spath = {buffer.data(), buffer.size()};
    } else {
      spath = replaceExtensionWith(s_ext, filename);
    }

    Logger::println("Writing asm to: %s\n", spath.c_str());
    if (writeObj) {
      // Clone module if we have both output-o and output-s flags
      // to avoid running 'addPassesToEmitFile' passes twice on same module
      auto clonedModule = llvm::CloneModule(*m);
      codegenModule(*gTargetMachine, *clonedModule, spath.c_str(),
                    CGFT_AssemblyFile);
    } else {
      codegenModule(*gTargetMachine, *m, spath.c_str(),
                    CGFT_AssemblyFile);
    }

    if (assembleExternally) {
      assemble(spath, filename);
    }

    if (!global.params.output_s) {
      llvm::sys::fs::remove(spath);
    }
  }

  if (writeObj) {
    writeObjectFile(m, filename);
    if (useIR2ObjCache) {
      cache::cacheObjectFile(filename, moduleHash);
    }
  }
}
