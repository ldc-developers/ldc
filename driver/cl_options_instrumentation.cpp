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

#include "dmd/errors.h"
#include "dmd/globals.h"
#include "gen/to_string.h"
#include "llvm/ADT/Triple.h"

namespace {
namespace cl = llvm::cl;

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

cl::opt<int> fXRayInstructionThreshold(
    "fxray-instruction-threshold", cl::value_desc("value"),
    cl::desc("Sets the minimum function size to instrument with XRay"),
    cl::init(200), cl::ZeroOrMore, cl::ValueRequired);

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

cl::opt<bool> fXRayInstrument(
    "fxray-instrument", cl::ZeroOrMore,
    cl::desc("Generate XRay instrumentation sleds on function entry and exit"));

llvm::StringRef getXRayInstructionThresholdString() {
  // The instruction threshold is constant during one compiler invoke, so we
  // can cache the int->string conversion result.
  static std::string thresholdString =
      ldc::to_string(fXRayInstructionThreshold);
  return thresholdString;
}

cl::opt<CFProtectionType> fCFProtection(
    "fcf-protection",
    cl::desc("Instrument control-flow architecture protection"), cl::ZeroOrMore,
    cl::ValueOptional,
    cl::values(clEnumValN(CFProtectionType::None, "none", ""),
               clEnumValN(CFProtectionType::Branch, "branch", ""),
               clEnumValN(CFProtectionType::Return, "return", ""),
               clEnumValN(CFProtectionType::Full, "full", ""),
               clEnumValN(CFProtectionType::Full, "",
                          "") // default to "full" if no argument specified
               ),
    cl::init(CFProtectionType::None));

void initializeInstrumentationOptionsFromCmdline(const llvm::Triple &triple) {
  if (ASTPGOInstrGenFile.getNumOccurrences() > 0) {
    pgoMode = PGO_ASTBasedInstr;
    if (ASTPGOInstrGenFile.empty()) {
      // profile-rt provides a default filename by itself
      global.params.datafileInstrProf = nullptr;
    } else {
      global.params.datafileInstrProf = fromPathString(ASTPGOInstrGenFile).ptr;
    }
  } else if (!ASTPGOInstrUseFile.empty()) {
    pgoMode = PGO_ASTBasedUse;
    global.params.datafileInstrProf = fromPathString(ASTPGOInstrUseFile).ptr;
  } else if (IRPGOInstrGenFile.getNumOccurrences() > 0) {
    pgoMode = PGO_IRBasedInstr;
    if (IRPGOInstrGenFile.empty()) {
      global.params.datafileInstrProf = "default_%m.profraw";
    } else {
      global.params.datafileInstrProf = fromPathString(IRPGOInstrGenFile).ptr;
    }
  } else if (!IRPGOInstrUseFile.empty()) {
    pgoMode = PGO_IRBasedUse;
    global.params.datafileInstrProf = fromPathString(IRPGOInstrUseFile).ptr;
  }

  if (dmdFunctionTrace)
    global.params.trace = true;

  // fcf-protection is only valid for X86
  if (fCFProtection != CFProtectionType::None &&
      !(triple.getArch() == llvm::Triple::x86 ||
        triple.getArch() == llvm::Triple::x86_64)) {
    error(Loc(), "option '--fcf-protection' cannot be specified on this target "
                 "architecture");
  }
}

} // namespace opts
