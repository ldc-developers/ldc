//===-- compile.cpp -------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the Boost Software License. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Jit runtime - shared library part.
// Defines jit runtime entry point and main compilation routines.
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <type_traits>

#include "callback_ostream.h"
#include "context.h"
#include "jit_context.h"
#include "optimizer.h"
#include "utils.h"

#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/IR/Mangler.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"

namespace {

#pragma pack(push, 1)

struct RtCompileFuncList {
  const char *name;
  void **func;
};

struct RtCompileSymList {
  const char *name;
  void *sym;
};

struct RtCompileVarList {
  const char *name;
  const void *init;
};

struct RtCompileModuleList {
  int32_t version;
  RtCompileModuleList *next;
  const char *irData;
  int32_t irDataSize;
  const RtCompileFuncList *funcList;
  int32_t funcListSize;
  const RtCompileSymList *symList;
  int32_t symListSize;
  const RtCompileVarList *varList;
  int32_t varListSize;
};

#pragma pack(pop)

template <typename T>
auto toArray(T *ptr, size_t size)
    -> llvm::ArrayRef<typename std::remove_cv<T>::type> {
  return llvm::ArrayRef<typename std::remove_cv<T>::type>(ptr, size);
}

template <typename F>
void enumModules(const RtCompileModuleList *modlist_head,
                 const Context &context, F &&fun) {
  auto current = modlist_head;
  while (current != nullptr) {
    interruptPoint(context, "check version");
    if (current->version != ApiVersion) {
      fatal(context, "Module was built with different jit api version");
    }
    fun(*current);
    current = current->next;
  }
}

struct JitModuleInfo final {
private:
  struct Func {
    llvm::StringRef name;
    void **thunkVar;
  };
  std::vector<Func> funcs;

public:
  JitModuleInfo(const Context &context,
                const RtCompileModuleList *modlist_head) {
    enumModules(modlist_head, context, [&](const RtCompileModuleList &current) {
      for (auto &&fun : toArray(current.funcList, static_cast<std::size_t>(
                                                      current.funcListSize))) {
        funcs.push_back({fun.name, fun.func});
      }
    });
  }

  const std::vector<Func> &functions() const { return funcs; }
};

std::string decorate(const std::string &name,
                     const llvm::DataLayout &datalayout) {
  llvm::SmallVector<char, 64> ret;
  llvm::Mangler::getNameWithPrefix(ret, name, datalayout);
  assert(!ret.empty());
  return std::string(ret.data(), ret.size());
}

JITContext &getJit() {
  static JITContext jit;
  return jit;
}

void setRtCompileVars(const Context &context, llvm::Module &module,
                      llvm::ArrayRef<RtCompileVarList> vals) {
  for (auto &&val : vals) {
    setRtCompileVar(context, module, val.name, val.init);
  }
}

void *resolveSymbol(llvm::JITSymbol &symbol) {
  auto addr = symbol.getAddress();
  if (!addr) {
    consumeError(addr.takeError());
    return nullptr;
  } else {
    return reinterpret_cast<void *>(addr.get());
  }
}

void dumpModule(const Context &context, const llvm::Module &module,
                DumpStage stage) {
  if (nullptr != context.dumpHandler) {
    auto callback = [&](const char *str, size_t len) {
      context.dumpHandler(context.dumpHandlerData, stage, str, len);
    };

    CallbackOstream os(callback);
    module.print(os, nullptr, false, true);
  }
}

void setFunctionsTarget(llvm::Module &module, const llvm::TargetMachine &TM) {
  // Set function target cpu to host if it wasn't set explicitly
  for (auto &&func : module.functions()) {
    if (!func.hasFnAttribute("target-cpu")) {
      func.addFnAttr("target-cpu", TM.getTargetCPU());
    }

    if (!func.hasFnAttribute("target-features")) {
      auto featStr = TM.getTargetFeatureString();
      if (!featStr.empty()) {
        func.addFnAttr("target-features", featStr);
      }
    }
  }
}

struct JitFinaliser final {
  JITContext &jit;
  bool finalized = false;
  explicit JitFinaliser(JITContext &j) : jit(j) {}
  ~JitFinaliser() {
    if (!finalized) {
      jit.reset();
    }
  }

  void finalze() { finalized = true; }
};

void rtCompileProcessImplSoInternal(const RtCompileModuleList *modlist_head,
                                    const Context &context) {
  if (nullptr == modlist_head) {
    // No jit modules to compile
    return;
  }
  interruptPoint(context, "Init");
  JITContext &myJit = getJit();

  JitModuleInfo moduleInfo(context, modlist_head);
  std::unique_ptr<llvm::Module> finalModule;
  auto &symMap = myJit.getSymMap();
  symMap.clear();
  auto &layout = myJit.getDataLayout();
  OptimizerSettings settings;
  settings.optLevel = context.optLevel;
  settings.sizeLevel = context.sizeLevel;
  enumModules(modlist_head, context, [&](const RtCompileModuleList &current) {
    interruptPoint(context, "load IR");
    auto buff = llvm::MemoryBuffer::getMemBuffer(
        llvm::StringRef(current.irData,
                        static_cast<std::size_t>(current.irDataSize)),
        "", false);
    interruptPoint(context, "parse IR");
    auto mod = llvm::parseBitcodeFile(*buff, myJit.getContext());
    if (!mod) {
      fatal(context, "Unable to parse IR");
    } else {
      llvm::Module &module = **mod;
      const auto name = module.getName();
      interruptPoint(context, "Verify module", name.data());
      verifyModule(context, module);

      dumpModule(context, module, DumpStage::OriginalModule);
      setFunctionsTarget(module, myJit.getTargetMachine());

      module.setDataLayout(myJit.getTargetMachine().createDataLayout());

      interruptPoint(context, "setRtCompileVars", name.data());
      setRtCompileVars(context, module,
                       toArray(current.varList,
                               static_cast<std::size_t>(current.varListSize)));

      if (nullptr == finalModule) {
        finalModule = std::move(*mod);
      } else {
        if (llvm::Linker::linkModules(*finalModule, std::move(*mod))) {
          fatal(context, "Can't merge module");
        }
      }

      for (auto &&sym : toArray(current.symList, static_cast<std::size_t>(
                                                     current.symListSize))) {
        symMap.insert(std::make_pair(decorate(sym.name, layout), sym.sym));
      }
    }
  });

  assert(nullptr != finalModule);
  dumpModule(context, *finalModule, DumpStage::MergedModule);
  interruptPoint(context, "Optimize final module");
  optimizeModule(context, myJit.getTargetMachine(), settings, *finalModule);

  interruptPoint(context, "Verify final module");
  verifyModule(context, *finalModule);

  dumpModule(context, *finalModule, DumpStage::OptimizedModule);

  interruptPoint(context, "Codegen final module");
  if (nullptr != context.dumpHandler) {
    auto callback = [&](const char *str, size_t len) {
      context.dumpHandler(context.dumpHandlerData, DumpStage::FinalAsm, str,
                          len);
    };

    CallbackOstream os(callback);
    if (myJit.addModule(std::move(finalModule), &os)) {
      fatal(context, "Can't codegen module");
    }
  } else {
    if (myJit.addModule(std::move(finalModule), nullptr)) {
      fatal(context, "Can't codegen module");
    }
  }

  JitFinaliser jitFinalizer(myJit);
  interruptPoint(context, "Resolve functions");
  for (auto &&fun : moduleInfo.functions()) {
    auto decorated = decorate(fun.name, layout);
    auto symbol = myJit.findSymbol(decorated);
    auto addr = resolveSymbol(symbol);
    if (nullptr == addr) {
      std::string desc = std::string("Symbol not found in jitted code: \"") +
                         fun.name.data() + "\" (\"" + decorated + "\")";
      fatal(context, desc);
    } else {
      *fun.thunkVar = addr;
    }

    if (nullptr != context.interruptPointHandler) {
      std::stringstream ss;
      ss << fun.name.data() << " to " << addr;
      auto str = ss.str();
      interruptPoint(context, "Resolved", str.c_str());
    }
  }
  jitFinalizer.finalze();
}

} // anon namespace

extern "C" {

#ifdef _WIN32
__declspec(dllexport)
#else
__attribute__ ((visibility ("default")))
#endif
void JIT_API_ENTRYPOINT(const void *modlist_head, const Context *context,
                        size_t contextSize) {
  assert(nullptr != context);
  assert(sizeof(*context) == contextSize);
  rtCompileProcessImplSoInternal(
      static_cast<const RtCompileModuleList *>(modlist_head), *context);
}
}
