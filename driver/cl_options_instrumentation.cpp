//===-- driver/cl_options_instrumentation.cpp ------------------*-  C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Defines the LDC command line options related to instrumentation, such as PGO
// -finstrument-functions, etc.
// Options for the instrumentation for the sanitizers is not part of this.
//
//===----------------------------------------------------------------------===//

#include "driver/cl_options_instrumentation.h"

#include "errors.h"
#include "globals.h"
#include "llvm/ADT/Triple.h"

namespace {
namespace cl = llvm::cl;

#if LDC_LLVM_VER >= 309
/// Option for generating IR-based PGO instrumentation (LLVM pass)
cl::opt<std::string> IRPGOInstrGenFile(
    "fprofile-generate", cl::value_desc("filename"),
    cl::desc("Generate instrumented code to collect a runtime "
             "profile into default.profraw (overriden by "
             "'=<filename>' or LLVM_PROFILE_FILE env var)"),
    cl::ZeroOrMore, cl::ValueOptional);

/// Option for generating IR-based PGO instrumentation (LLVM pass)
cl::opt<std::string> IRPGOInstrUseFile(
    "fprofile-use", cl::ZeroOrMore, cl::value_desc("filename"),
    cl::desc("Use instrumentation data for profile-guided optimization"),
    cl::ValueRequired);
#endif

/// Option for generating frontend-based PGO instrumentation
cl::opt<std::string> ASTPGOInstrGenFile(
    "fprofile-instr-generate", cl::value_desc("filename"),
    cl::desc("Generate instrumented code to collect a runtime "
             "profile into default.profraw (overriden by "
             "'=<filename>' or LLVM_PROFILE_FILE env var)"),
    cl::ZeroOrMore, cl::ValueOptional);

/// Option for generating frontend-based PGO instrumentation
cl::opt<std::string> ASTPGOInstrUseFile(
    "fprofile-instr-use", cl::ZeroOrMore, cl::value_desc("filename"),
    cl::desc("Use instrumentation data for profile-guided optimization"),
    cl::ValueRequired);

} // anonymous namespace

namespace opts {

PGOKind pgoMode = PGO_None;

cl::opt<bool>
    instrumentFunctions("finstrument-functions", cl::ZeroOrMore,
                        cl::desc("Instrument function entry and exit with "
                                 "GCC-compatible profiling calls"));

// DMD-style profiling (`dmd -profile`)
static cl::opt<bool> dmdFunctionTrace(
    "fdmd-trace-functions", cl::ZeroOrMore,
    cl::desc("DMD-style runtime performance profiling of generated code"));

void initializeInstrumentationOptionsFromCmdline(const llvm::Triple &triple) {
  if (ASTPGOInstrGenFile.getNumOccurrences() > 0) {
    pgoMode = PGO_ASTBasedInstr;
    if (ASTPGOInstrGenFile.empty()) {
#if LDC_LLVM_VER >= 309
      // profile-rt provides a default filename by itself
      global.params.datafileInstrProf = nullptr;
#else
      global.params.datafileInstrProf = "default.profraw";
#endif
    } else {
      initFromPathString(global.params.datafileInstrProf, ASTPGOInstrGenFile);
    }
  } else if (!ASTPGOInstrUseFile.empty()) {
    pgoMode = PGO_ASTBasedUse;
    initFromPathString(global.params.datafileInstrProf, ASTPGOInstrUseFile);
  }
#if LDC_LLVM_VER >= 309
  else if (IRPGOInstrGenFile.getNumOccurrences() > 0) {
    pgoMode = PGO_IRBasedInstr;
    if (IRPGOInstrGenFile.empty()) {
      global.params.datafileInstrProf = "default_%m.profraw";
    } else {
      initFromPathString(global.params.datafileInstrProf, IRPGOInstrGenFile);
    }
  } else if (!IRPGOInstrUseFile.empty()) {
    pgoMode = PGO_IRBasedUse;
    initFromPathString(global.params.datafileInstrProf, IRPGOInstrUseFile);
  }
#endif

  // There is a bug in (our use of?) LLVM where codegen errors with
  // PGO_IRBasedInstr for Windows targets. So disable IRBased PGO on Windows for
  // now.
  if ((pgoMode == PGO_IRBasedInstr) && triple.isOSWindows()) {
    error(Loc(),
          "'-fprofile-generate' is not yet supported for Windows targets.");
  }

  if (dmdFunctionTrace)
    global.params.trace = true;
}

} // namespace opts
