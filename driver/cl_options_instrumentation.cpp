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

#include "globals.h"

namespace opts {

#if LDC_WITH_PGO
cl::opt<std::string>
    genfileInstrProf("fprofile-instr-generate", cl::value_desc("filename"),
                     cl::desc("Generate instrumented code to collect a runtime "
                              "profile into default.profraw (overriden by "
                              "'=<filename>' or LLVM_PROFILE_FILE env var)"),
                     cl::ZeroOrMore, cl::ValueOptional);

cl::opt<std::string> usefileInstrProf(
    "fprofile-instr-use", cl::ZeroOrMore, cl::value_desc("filename"),
    cl::desc("Use instrumentation data for profile-guided optimization"),
    cl::ValueRequired);
#endif

cl::opt<bool>
    instrumentFunctions("finstrument-functions", cl::ZeroOrMore,
                        cl::desc("Instrument function entry and exit with "
                                 "GCC-compatible profiling calls"));

void initializeInstrumentationOptionsFromCmdline() {
#if LDC_WITH_PGO
  if (genfileInstrProf.getNumOccurrences() > 0) {
    global.params.genInstrProf = true;
    if (genfileInstrProf.empty()) {
#if LDC_LLVM_VER >= 309
      // profile-rt provides a default filename by itself
      global.params.datafileInstrProf = nullptr;
#else
      global.params.datafileInstrProf = "default.profraw";
#endif
    } else {
      initFromPathString(global.params.datafileInstrProf, genfileInstrProf);
    }
  } else {
    global.params.genInstrProf = false;
    // If we don't have to generate instrumentation, we could be given a
    // profdata file:
    initFromPathString(global.params.datafileInstrProf, usefileInstrProf);
  }
#endif
}

} // namespace opts
