//===-- tool.cpp ----------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "driver/tool.h"
#include "mars.h"
#include "llvm/Support/Program.h"

int executeToolAndWait(const std::string &tool,
                       std::vector<std::string> const &args, bool verbose) {
  // Construct real argument list; first entry is the tool itself.
#if LDC_LLVM_VER >= 700
  std::vector<llvm::StringRef> realargs;
  realargs.reserve(args.size() + 1);
#else
  std::vector<const char *> realargs;
  realargs.reserve(args.size() + 2);
#endif
  realargs.push_back(tool.c_str());
  for (const auto &arg : args) {
    realargs.push_back(arg.c_str());
  }

  // Print command line if requested
  if (verbose) {
    // Print it
    for (const auto &arg : realargs) {
      fprintf(global.stdmsg, "%s ",
#if LDC_LLVM_VER >= 700
              arg.str().c_str()
#else
              arg
#endif
      );
    }
    fprintf(global.stdmsg, "\n");
    fflush(global.stdmsg);
  }

#if LDC_LLVM_VER >= 700
  auto &argv = realargs;
  auto envVars = llvm::None;
#else
  realargs.push_back(nullptr); // terminate with null
  auto argv = &realargs[0];
  auto envVars = nullptr;
#endif

  // Execute tool.
  std::string errstr;
  if (int status = llvm::sys::ExecuteAndWait(tool, argv, envVars,
#if LDC_LLVM_VER >= 600
                                             {},
#else
                                             nullptr,
#endif
                                             0, 0, &errstr)) {
    error(Loc(), "%s failed with status: %d", tool.c_str(), status);
    if (!errstr.empty()) {
      error(Loc(), "message: %s", errstr.c_str());
    }
    return status;
  }
  return 0;
}
