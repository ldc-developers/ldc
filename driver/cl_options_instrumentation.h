//===-- driver/cl_options_instrumentation.h --------------------*-  C++ -*-===//
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

#ifndef LDC_DRIVER_CL_OPTIONS_INSTRUMENTATION_H
#define LDC_DRIVER_CL_OPTIONS_INSTRUMENTATION_H

#include "gen/cl_helpers.h"

namespace opts {
namespace cl = llvm::cl;

// PGO options
extern cl::opt<std::string> genfileInstrProf;
extern cl::opt<std::string> usefileInstrProf;

extern cl::opt<bool> instrumentFunctions;

/// This initializes the instrumentation options, and checks the validity of the
/// commandline flags. It should be called only once.
void initializeInstrumentationOptionsFromCmdline();

} // namespace opts
#endif // LDC_DRIVER_CL_OPTIONS_INSTRUMENTATION_H
