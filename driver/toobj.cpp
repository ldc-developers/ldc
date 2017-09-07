//===-- toobj.cpp ---------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "driver/toobj.h"

#include "driver/cl_options.h"
#include "driver/cache.h"
#include "driver/targetmachine.h"
#include "driver/tool.h"
#include "gen/irstate.h"
#include "gen/logger.h"
#include "gen/optimizer.h"
#include "llvm/IR/AssemblyAnnotationWriter.h"
#include "llvm/IR/Verifier.h"
#if LDC_LLVM_VER >= 309
#include "llvm/Analysis/ModuleSummaryAnalysis.h"
#endif
#if LDC_LLVM_VER >= 400
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#else
#include "llvm/Bitcode/ReaderWriter.h"
#endif
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Path.h"
#ifdef LDC_LLVM_SUPPORTED_TARGET_SPIRV
#include "llvm/Support/SPIRV.h"
#endif
#include "llvm/Target/TargetMachine.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include "llvm/IR/Module.h"
#include <cstddef>
#include <fstream>

static llvm::cl::opt<bool>
    NoIntegratedAssembler("no-integrated-as", llvm::cl::ZeroOrMore,
                          llvm::cl::Hidden,
                          llvm::cl::desc("Disable integrated assembler"));

// based on llc code, University of Illinois Open Source License
static void codegenModule(llvm::TargetMachine &Target, llvm::Module &m,
                          llvm::raw_fd_ostream &out,
                          llvm::TargetMachine::CodeGenFileType fileType) {
  using namespace llvm;

// Create a PassManager to hold and optimize the collection of passes we are
// about to build.
  legacy::PassManager Passes;
  ComputeBackend::Type cb = getComputeTargetType(&m);

  if (cb == ComputeBackend::SPIRV) {
#ifdef LDC_LLVM_SUPPORTED_TARGET_SPIRV
    IF_LOG Logger::println("running createSPIRVWriterPass()");
    llvm::createSPIRVWriterPass(out)->runOnModule(m);
    IF_LOG Logger::println("Success.");
#else
    error(Loc(), "Trying to target SPIRV, but LDC is not built to do so!");
#endif

    return;
  }

  // The DataLayout is already set at the module (in module.cpp,
  // method Module::genLLVMModule())
  // FIXME: Introduce new command line switch default-data-layout to
  // override the module data layout

  // Add internal analysis passes from the target machine.
  Passes.add(
      createTargetTransformInfoWrapperPass(Target.getTargetIRAnalysis()));

  if (Target.addPassesToEmitFile(Passes,
                                 out,
        // Always generate assembly for ptx as it is an assembly format
        // The PTX backend fails if we pass anything else.
        (cb == ComputeBackend::NVPTX) ? llvm::TargetMachine::CGFT_AssemblyFile
                                      : fileType,
          codeGenOptLevel())) {
    llvm_unreachable("no support for asm output");
  }

  Passes.run(m);
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
  int R = executeToolAndWait(getGcc(), args, global.params.verbose);
  if (R) {
    error(Loc(), "Error while invoking external assembler.");
    fatal();
  }
}

////////////////////////////////////////////////////////////////////////////////

namespace {
using namespace llvm;
static void printDebugLoc(const DebugLoc &debugLoc, formatted_raw_ostream &os) {
  os << debugLoc.getLine() << ":" << debugLoc.getCol();
  if (DILocation *IDL = debugLoc.getInlinedAt()) {
    os << "@";
    printDebugLoc(IDL, os);
  }
}

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
    if (DISubprogram *N = FindSubprogram(F, Finder))
    {
#if LDC_LLVM_VER >= 500
      return N->getName();
#else
      return N->getDisplayName();
#endif
    }
    return "";
  }

public:
  void emitFunctionAnnot(const Function *F,
                         formatted_raw_ostream &os) LLVM_OVERRIDE {
    os << "; [#uses = " << F->getNumUses() << ']';

    // show demangled name
    llvm::StringRef funcName = GetDisplayName(F);
    if (!funcName.empty()) {
      os << " [display name = " << funcName << ']';
    }
    os << '\n';
  }

  void printInfoComment(const Value &val,
                        formatted_raw_ostream &os) LLVM_OVERRIDE {
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
      } else if (isa<AllocaInst>(&val)) {
        os << ", size/byte = "
           << gDataLayout->getTypeAllocSize(val.getType()->getContainedType(0));
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
      printDebugLoc(debugLoc, os);
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
  std::error_code errinfo;
  {
    llvm::raw_fd_ostream out(filename, errinfo, llvm::sys::fs::F_None);
    if (!errinfo)
    {
      codegenModule(*gTargetMachine, *m, out,
                    llvm::TargetMachine::CGFT_ObjectFile);
    } else {
      error(Loc(), "cannot write object file '%s': %s", filename,
            errinfo.message().c_str());
      fatal();
    }
  }
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

bool shouldDoLTO(llvm::Module *m) {
#if LDC_LLVM_VER < 309
  return false;
#else
#if LDC_LLVM_VER == 309
  // LLVM 3.9 bug: can't do ThinLTO with modules that have module-scope inline
  // assembly blocks (duplicate definitions upon importing from such a module).
  // https://llvm.org/bugs/show_bug.cgi?id=30610
  if (opts::isUsingThinLTO() && !m->getModuleInlineAsm().empty())
    return false;
#endif
  return opts::isUsingLTO();
#endif
}
} // end of anonymous namespace

void writeModule(llvm::Module *m, const char *filename) {
  const bool doLTO = shouldDoLTO(m);
  const bool outputObj = shouldOutputObjectFile();
  const bool assembleExternally = shouldAssembleExternally();

  // Use cached object code if possible.
  // TODO: combine LDC's cache and LTO (the advantage is skipping the IR
  // optimization).
  const bool useIR2ObjCache = !opts::cacheDir.empty() && outputObj && !doLTO;
  llvm::SmallString<32> moduleHash;
  if (useIR2ObjCache) {
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

  // run optimizer
  ldc_optimize_module(m);

  // make sure the output directory exists
  const auto directory = llvm::sys::path::parent_path(filename);
  if (!directory.empty()) {
    if (auto ec = llvm::sys::fs::create_directories(directory)) {
      error(Loc(), "failed to create output directory: %s\n%s",
            directory.data(), ec.message().c_str());
      fatal();
    }
  }

  const auto outputFlags = {global.params.output_o, global.params.output_bc,
                            global.params.output_ll, global.params.output_s};
  const auto numOutputFiles =
      std::count_if(outputFlags.begin(), outputFlags.end(),
                    [](OUTPUTFLAG flag) { return flag != 0; });

  const auto replaceExtensionWith = [=](const char *ext) -> std::string {
    if (numOutputFiles == 1)
      return filename;
    llvm::SmallString<128> buffer(filename);
    llvm::sys::path::replace_extension(buffer, ext);
    return buffer.str();
  };

  // write LLVM bitcode
  if (global.params.output_bc || (doLTO && outputObj)) {
    std::string bcpath =
        (doLTO && outputObj) ? filename : replaceExtensionWith(global.bc_ext);
    Logger::println("Writing LLVM bitcode to: %s\n", bcpath.c_str());
    std::error_code errinfo;
    llvm::raw_fd_ostream bos(bcpath.c_str(), errinfo, llvm::sys::fs::F_None);
    if (bos.has_error()) {
      error(Loc(), "cannot write LLVM bitcode file '%s': %s", bcpath.c_str(),
            errinfo.message().c_str());
      fatal();
    }
    if (opts::isUsingThinLTO()) {
#if LDC_LLVM_VER >= 309
      Logger::println("Creating module summary for ThinLTO");
#if LDC_LLVM_VER == 309
      // When the function freq info callback is set to nullptr, LLVM will
      // calculate it automatically for us.
      llvm::ModuleSummaryIndexBuilder indexBuilder(
          m, /* function freq callback */ nullptr);
      auto &moduleSummaryIndex = indexBuilder.getIndex();
#else
      llvm::ProfileSummaryInfo PSI(*m);

      // When the function freq info callback is set to nullptr, LLVM will
      // calculate it automatically for us.
      auto moduleSummaryIndex = buildModuleSummaryIndex(
          *m, /* function freq callback */ nullptr, &PSI);
#endif

      llvm::WriteBitcodeToFile(m, bos, true, &moduleSummaryIndex,
                               /* generate ThinLTO hash */ true);
#endif
    } else {
      llvm::WriteBitcodeToFile(m, bos);
    }
  }

  // write LLVM IR
  if (global.params.output_ll) {
    const auto llpath = replaceExtensionWith(global.ll_ext);
    Logger::println("Writing LLVM IR to: %s\n", llpath.c_str());
    std::error_code errinfo;
    llvm::raw_fd_ostream aos(llpath.c_str(), errinfo, llvm::sys::fs::F_None);
    if (aos.has_error()) {
      error(Loc(), "cannot write LLVM IR file '%s': %s", llpath.c_str(),
            errinfo.message().c_str());
      fatal();
    }
    AssemblyAnnotator annotator;
    m->print(aos, &annotator);
  }

  // write native assembly
  if (global.params.output_s || assembleExternally) {
    std::string spath;
    if (!global.params.output_s) {
      llvm::SmallString<16> buffer;
      llvm::sys::fs::createUniqueFile("ldc-%%%%%%%.s", buffer);
      spath = buffer.str();
    } else {
      spath = replaceExtensionWith(global.s_ext);
    }

    Logger::println("Writing asm to: %s\n", spath.c_str());
    std::error_code errinfo;
    {
      llvm::raw_fd_ostream out(spath.c_str(), errinfo, llvm::sys::fs::F_None);
      if (!errinfo)
      {
        codegenModule(*gTargetMachine, *m, out,
                      llvm::TargetMachine::CGFT_AssemblyFile);
      } else {
        error(Loc(), "cannot write asm: %s", errinfo.message().c_str());
        fatal();
      }
    }

    if (assembleExternally) {
      assemble(spath, filename);
    }

    if (!global.params.output_s) {
      llvm::sys::fs::remove(spath);
    }
  }

  if (outputObj && !doLTO) {
    writeObjectFile(m, filename);
    if (useIR2ObjCache) {
      cache::cacheObjectFile(filename, moduleHash);
    }
  }
}
