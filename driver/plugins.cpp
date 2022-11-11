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
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DynamicLibrary.h"

#if LDC_LLVM_VER >= 1400
#include "llvm/ADT/SmallVector.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/Error.h"

#include "driver/cl_options.h"
#endif

namespace {
namespace cl = llvm::cl;

cl::list<std::string> pluginFiles("plugin", cl::CommaSeparated,
                                  cl::desc("Pass plugins to load."),
                                  cl::value_desc("dynamic_library.so,lib2.so"));

} // anonymous namespace

#if LDC_LLVM_VER >= 1400

namespace {
llvm::SmallVector<llvm::PassPlugin, 1> plugins;
}
/// Loads all plugins for the new pass manager. These plugins will need to be
/// added When building the optimization pipeline.
void loadAllPluginsNewPM() {
  for (auto &filename : pluginFiles) {
    auto plugin = llvm::PassPlugin::Load(filename);
    if (!plugin) {
      error(Loc(), "Error loading plugin '%s': %s", filename.c_str(),
            llvm::toString(plugin.takeError()).c_str());
      continue;
    }
    plugins.emplace_back(plugin.get());
  }
}
void registerAllPluginsWithPassBuilder(llvm::PassBuilder &PB) {
  for (auto &plugin : plugins) {
    plugin.registerPassBuilderCallbacks(PB);
  }
}

#endif // LDC_LLVM_VER >= 1400

/// Loads all plugins for the legacy pass manaager. The static constructor of
/// each plugin should take care of the plugins registering themself with the
/// rest of LDC/LLVM.
void loadAllPluginsLegacyPM() {
  for (auto &filename : pluginFiles) {
    std::string errorString;
    if (llvm::sys::DynamicLibrary::LoadLibraryPermanently(filename.c_str(),
                                                          &errorString)) {
      error(Loc(), "Error loading plugin '%s': %s", filename.c_str(),
            errorString.c_str());
    }
  }
}

#if LDC_LLVM_VER >= 1400
void loadAllPlugins() {
  if (opts::isUsingLegacyPassManager())
    loadAllPluginsLegacyPM();
  else
    loadAllPluginsNewPM();
}
#else
void loadAllPlugins() { loadAllPluginsLegacyPM(); }
void registerAllPluginsWithPassBuilder(llvm::PassBuilder &) {}
#endif

#else // #if LDC_ENABLE_PLUGINS

void loadAllPlugins() {}
void registerAllPluginsWithPassBuilder(llvm::PassBuilder &) {}

#endif // LDC_ENABLE_PLUGINS
