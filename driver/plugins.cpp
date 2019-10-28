//===-- driver/plugins.cpp -------------------------------------*-  C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Implements functionality related to plugins (`-plugin=...`).
//
//===----------------------------------------------------------------------===//

#include "driver/plugins.h"

#if LDC_ENABLE_PLUGINS

#include "dmd/errors.h"
#include "dmd/globals.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DynamicLibrary.h"

namespace {
namespace cl = llvm::cl;

cl::list<std::string>
    pluginFiles("plugin", cl::CommaSeparated, cl::desc("Plugins to load."),
                cl::value_desc("dynamic_library.so,lib2.so"));

} // anonymous namespace

/// Loads all plugins. The static constructor of each plugin should take care of
/// the plugins registering themself with the rest of LDC/LLVM.
void loadAllPlugins() {
  for (auto &filename : pluginFiles) {
    std::string errorString;
    if (llvm::sys::DynamicLibrary::LoadLibraryPermanently(filename.c_str(),
                                                          &errorString)) {
      error(Loc(), "Error loading plugin '%s': %s", filename.c_str(),
            errorString.c_str());
    }
  }
}

#else // #if LDC_ENABLE_PLUGINS

void loadAllPlugins() {}

#endif // LDC_ENABLE_PLUGINS
