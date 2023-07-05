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
// Note: plugins can be LLVM-plugins (to be registered with the pass manager)
// or dlang-plugins for semantic analysis.
//
//===----------------------------------------------------------------------===//

#include "driver/plugins.h"

#if LDC_ENABLE_PLUGINS

#include "dmd/errors.h"
#include "dmd/globals.h"
#include "dmd/module.h"
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

struct SemaPlugin {
  llvm::sys::DynamicLibrary library;
  void (*runSemanticAnalysis)(Module *);

  SemaPlugin(const llvm::sys::DynamicLibrary &library,
             void (*runSemanticAnalysis)(Module *))
      : library(library), runSemanticAnalysis(runSemanticAnalysis) {}
};

llvm::SmallVector<SemaPlugin, 1> sema_plugins;

} // anonymous namespace

// Tries to load plugin as SemanticAnalysis. Returns true on 'success', i.e. no
// further attempts needed.
bool loadSemanticAnalysisPlugin(const std::string &filename) {
  std::string errorString;
  auto library = llvm::sys::DynamicLibrary::getPermanentLibrary(
      filename.c_str(), &errorString);
  if (!library.isValid()) {
    error(Loc(), "Error loading plugin '%s': %s", filename.c_str(),
          errorString.c_str());
    return true; // No success, but no need to try loading again as LLVM plugin.
  }

  // SemanticAnalysis plugins need to export the `runSemanticAnalysis` function.
  void *runSemanticAnalysisFnPtr =
      library.getAddressOfSymbol("runSemanticAnalysis");

  // If the symbol isn't found, this is probably an LLVM plugin.
  if (!runSemanticAnalysisFnPtr)
    return false;

  sema_plugins.emplace_back(
      library, reinterpret_cast<void (*)(Module *)>(runSemanticAnalysisFnPtr));
  return true;
}

/// Loads plugin for the legacy pass manager. The static constructor of
/// the plugin should take care of the plugins registering themself with the
/// rest of LDC/LLVM.
void loadLLVMPluginLegacyPM(const std::string &filename) {
  std::string errorString;
  if (llvm::sys::DynamicLibrary::LoadLibraryPermanently(filename.c_str(),
                                                        &errorString)) {
    error(Loc(), "Error loading plugin '%s': %s", filename.c_str(),
          errorString.c_str());
  }
}

#if LDC_LLVM_VER >= 1400

namespace {
llvm::SmallVector<llvm::PassPlugin, 1> llvm_plugins;

/// Loads plugin for the new pass manager. The plugin will need to be
/// added explicitly when building the optimization pipeline.
void loadLLVMPluginNewPM(const std::string &filename) {

  auto plugin = llvm::PassPlugin::Load(filename);
  if (!plugin) {
    error(Loc(), "Error loading plugin '%s': %s", filename.c_str(),
          llvm::toString(plugin.takeError()).c_str());
    return;
  }
  llvm_plugins.emplace_back(plugin.get());
}

} // anonymous namespace

#endif // LDC_LLVM_VER >= 1400

void loadLLVMPlugin(const std::string &filename) {
#if LDC_LLVM_VER >= 1400
  if (opts::isUsingLegacyPassManager())
    loadLLVMPluginLegacyPM(filename);
  else
    loadLLVMPluginNewPM(filename);
#else
  loadLLVMPluginLegacyPM(filename);
#endif
}

void loadAllPlugins() {
  for (auto &filename : pluginFiles) {
    // First attempt to load plugin as SemanticAnalysis plugin. If unsuccesfull,
    // load as LLVM plugin.
    auto success = loadSemanticAnalysisPlugin(filename);
    if (!success)
      loadLLVMPlugin(filename);
  }
}

void registerAllPluginsWithPassBuilder(llvm::PassBuilder &PB) {
#if LDC_LLVM_VER >= 1400
  for (auto &plugin : llvm_plugins) {
    plugin.registerPassBuilderCallbacks(PB);
  }
#endif
}

void runAllSemanticAnalysisPlugins(Module *m) {
  for (auto &plugin : sema_plugins) {
    assert(plugin.runSemanticAnalysis);
    plugin.runSemanticAnalysis(m);
  }
}

#else // #if LDC_ENABLE_PLUGINS

class Module;

void loadAllPlugins() {}
void registerAllPluginsWithPassBuilder(llvm::PassBuilder &) {}
void runAllSemanticAnalysisPlugins(Module *m) {}

#endif // LDC_ENABLE_PLUGINS
