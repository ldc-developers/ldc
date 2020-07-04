//===-- options.cpp -------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the Boost Software License. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "options.h"

#include "callback_ostream.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"

bool parseOptions(Slice<Slice<const char>> args,
                  void (*errs)(void *, const char *, size_t),
                  void *errsContext) {
  llvm::SmallVector<std::string, 32 - 1> tempStrs;
  tempStrs.reserve(args.len);
  llvm::SmallVector<const char *, 32> tempOpts;
  tempOpts.reserve(args.len + 1);
  tempOpts.push_back("jit"); // dummy app name
  for (size_t i = 0; i < args.len; ++i) {
    auto &arg = args.data[i];
    tempStrs.push_back(std::string(arg.data, arg.len));
    tempOpts.push_back(tempStrs.back().c_str());
  }

  auto callback = [&](const char *str, size_t len) {
    if (errs != nullptr) {
      errs(errsContext, str, len);
    }
  };
  CallbackOstream os(callback);

  // There is no Option::setDefault() before llvm 60
  llvm::cl::ResetAllOptionOccurrences();
  for (auto &i : llvm::cl::getRegisteredOptions()) {
    i.second->setDefault();
  }
  auto res = llvm::cl::ParseCommandLineOptions(
      static_cast<int>(tempOpts.size()), tempOpts.data(), "", &os);
  os.flush();
  return res;
}
