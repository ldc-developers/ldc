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
  // Construct real argument list.
  // First entry is the tool itself, last entry must be NULL.
  std::vector<const char *> realargs;
  realargs.reserve(args.size() + 2);
  realargs.push_back(tool.c_str());
  for (const auto &arg : args) {
    realargs.push_back(arg.c_str());
  }
  realargs.push_back(nullptr);

  // Print command line if requested
  if (verbose) {
    // Print it
    for (size_t i = 0; i < realargs.size() - 1; i++) {
      fprintf(global.stdmsg, "%s ", realargs[i]);
    }
    fprintf(global.stdmsg, "\n");
    fflush(global.stdmsg);
  }

  // Execute tool.
  std::string errstr;
  if (int status = llvm::sys::ExecuteAndWait(tool, &realargs[0], nullptr,
                                             nullptr, 0, 0, &errstr)) {
    error(Loc(), "%s failed with status: %d", tool.c_str(), status);
    if (!errstr.empty()) {
      error(Loc(), "message: %s", errstr.c_str());
    }
    return status;
  }
  return 0;
}
