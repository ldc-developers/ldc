
#include <cassert>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>

#include "callback_ostream.h"
#include "context.h"
#include "optimizer.h"
#include "utils.h"

#include "llvm/Support/ManagedStatic.h"

#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/LambdaResolver.h"
#include "llvm/ExecutionEngine/RuntimeDyld.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"

#if LDC_LLVM_VER >= 500
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#else
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#endif

#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"

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
  RtCompileModuleList *next;
  const char *irData;
  int irDataSize;
  RtCompileFuncList *funcList;
  int funcListSize;
  RtCompileSymList *symList;
  int symListSize;
  RtCompileVarList *varList;
  int varListSize;
};

#pragma pack(pop)

llvm::SmallVector<std::string, 4> getHostAttrs() {
  llvm::SmallVector<std::string, 4> features;
  llvm::StringMap<bool> hostFeatures;
  if (llvm::sys::getHostCPUFeatures(hostFeatures)) {
    for (auto &&f : hostFeatures) {
      features.push_back(((f.second ? "+" : "-") + f.first()).str());
    }
  }
  return features;
}

using SymMap = std::map<std::string, void *>;

struct llvm_init_obj {
  llvm_init_obj() {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
  }
};

std::string decorate(const std::string &name) {
#if defined(__APPLE__)
  return "_" + name;
#elif defined(_WIN32) && defined(_M_IX86)
  assert(!name.empty());
  if (0x1 == name[0]) {
    return name.substr(1);
  }
  return "_" + name;
#else
  return name;
#endif
}

auto getSymbolInProcess(const std::string &name)
    -> decltype(llvm::RTDyldMemoryManager::getSymbolAddressInProcess(name)) {
  assert(!name.empty());
#if defined(_WIN32)
  if ('_' == name[0]) {
    return llvm::RTDyldMemoryManager::getSymbolAddressInProcess(name.substr(1));
  }
  return llvm::RTDyldMemoryManager::getSymbolAddressInProcess(name);
#else
  return llvm::RTDyldMemoryManager::getSymbolAddressInProcess(name);
#endif
}

class MyJIT {
private:
  llvm_init_obj initObj;
  llvm::llvm_shutdown_obj shutdownObj;
  std::unique_ptr<llvm::TargetMachine> targetmachine;
  const llvm::DataLayout dataLayout;
#if LDC_LLVM_VER >= 500
  using ObjectLayerT = llvm::orc::RTDyldObjectLinkingLayer;
  using CompileLayerT =
      llvm::orc::IRCompileLayer<ObjectLayerT, llvm::orc::SimpleCompiler>;
  using ModuleHandleT = std::vector<CompileLayerT::ModuleHandleT>;
#else
  using ObjectLayerT = llvm::orc::ObjectLinkingLayer<>;
  using CompileLayerT = llvm::orc::IRCompileLayer<ObjectLayerT>;
  using ModuleHandleT = CompileLayerT::ModuleSetHandleT;
#endif
  ObjectLayerT objectLayer;
  CompileLayerT compileLayer;
  llvm::LLVMContext context;
  bool compiled = false;
  ModuleHandleT moduleHandle;

public:
  MyJIT()
      : targetmachine(
            llvm::EngineBuilder()
                .setRelocationModel(llvm::Reloc::Static)
                .selectTarget(llvm::Triple(llvm::sys::getProcessTriple()),
                              llvm::StringRef(), llvm::sys::getHostCPUName(),
                              getHostAttrs())),
        dataLayout(targetmachine->createDataLayout()),
#if LDC_LLVM_VER >= 500
        objectLayer(
            []() { return std::make_shared<llvm::SectionMemoryManager>(); }),
#endif
        compileLayer(objectLayer, llvm::orc::SimpleCompiler(*targetmachine)) {
    llvm::sys::DynamicLibrary::LoadLibraryPermanently(nullptr);
  }

  llvm::TargetMachine &getTargetMachine() { return *targetmachine; }

  void addModules(std::vector<std::unique_ptr<llvm::Module>> &&modules,
                  const SymMap &symMap) {
    reset();
    // Build our symbol resolver:
    // Lambda 1: Look back into the JIT itself to find symbols that are part of
    //           the same "logical dylib".
    // Lambda 2: Search for external symbols in the host process.
    auto Resolver = llvm::orc::createLambdaResolver(
        [&](const std::string &name) {
          if (auto Sym = compileLayer.findSymbol(name, false)) {
            return Sym;
          }
          return llvm::JITSymbol(nullptr);
        },
        [&](const std::string &name) {
          auto it = symMap.find(name);
          if (symMap.end() != it) {
            return llvm::JITSymbol(
                reinterpret_cast<llvm::JITTargetAddress>(it->second),
                llvm::JITSymbolFlags::Exported);
          }
          if (auto SymAddr = getSymbolInProcess(name)) {
            return llvm::JITSymbol(SymAddr, llvm::JITSymbolFlags::Exported);
          }
          return llvm::JITSymbol(nullptr);
        });

    // Add the set to the JIT with the resolver we created above
#if LDC_LLVM_VER >= 500
    for (auto &&module : modules) {
      auto handle = compileLayer.addModule(std::move(module), Resolver);
      assert(handle);
      moduleHandle.push_back(handle.get());
    }
    modules.clear();
#else
    moduleHandle = compileLayer.addModuleSet(
        std::move(modules), llvm::make_unique<llvm::SectionMemoryManager>(),
        std::move(Resolver));
#endif
    compiled = true;
  }

  llvm::JITSymbol findSymbol(const std::string &name) {
    return compileLayer.findSymbol(name, false);
  }

  llvm::LLVMContext &getContext() { return context; }

  void removeModule(const ModuleHandleT &H) {
#if LDC_LLVM_VER >= 500
    for (auto &&handle : H) {
      cantFail(compileLayer.removeModule(handle));
    }
#else
    compileLayer.removeModuleSet(H);
#endif
  }

  void reset() {
    if (compiled) {
      removeModule(moduleHandle);
      moduleHandle = {};
      compiled = false;
    }
  }
};

void setRtCompileVars(const Context &context, llvm::Module &module,
                      llvm::ArrayRef<RtCompileVarList> vals) {
  for (auto &&val : vals) {
    setRtCompileVar(context, module, val.name, val.init);
  }
}

template <typename T> llvm::ArrayRef<T> toArray(T *ptr, size_t size) {
  return llvm::ArrayRef<T>(ptr, size);
}

void *resolveSymbol(llvm::JITSymbol &symbol) {
  auto addr = symbol.getAddress();
#if LDC_LLVM_VER >= 500
  if (!addr) {
    consumeError(addr.takeError());
    return nullptr;
  } else {
    return reinterpret_cast<void *>(addr.get());
  }
#else
  return reinterpret_cast<void *>(addr);
#endif
}

struct JitFinaliser final {
  MyJIT &jit;
  bool finalized = false;
  explicit JitFinaliser(MyJIT &j) : jit(j) {}
  ~JitFinaliser() {
    if (!finalized) {
      jit.reset();
    }
  }

  void finalze() { finalized = true; }
};

MyJIT &getJit() {
  static MyJIT jit;
  return jit;
}

void rtCompileProcessImplSoInternal(const RtCompileModuleList *modlist_head,
                                    const Context &context) {
  interruptPoint(context, "Init");
  MyJIT &myJit = getJit();
  auto current = modlist_head;

  std::vector<std::pair<std::string, void **>> functions;
  std::vector<std::unique_ptr<llvm::Module>> ms;
  SymMap symMap;
  OptimizerSettings settings;
  settings.optLevel = context.optLevel;
  settings.sizeLevel = context.sizeLevel;
  while (nullptr != current) {
    interruptPoint(context, "load IR");
    auto buff = llvm::MemoryBuffer::getMemBuffer(
        llvm::StringRef(current->irData, current->irDataSize), "", false);
    interruptPoint(context, "parse IR");
    auto mod = llvm::parseBitcodeFile(*buff, myJit.getContext());
    if (!mod) {
      fatal(context, "Unable to parse IR");
    } else {
      llvm::Module &module = **mod;
      const auto name = module.getName();
      interruptPoint(context, "Verify module", name.data());
      ::verifyModule(context, module);
      module.setDataLayout(myJit.getTargetMachine().createDataLayout());

      interruptPoint(context, "setRtCompileVars", name.data());
      setRtCompileVars(context, module,
                       toArray(current->varList, current->varListSize));

      interruptPoint(context, "Optimize module", name.data());
      optimizeModule(context, myJit.getTargetMachine(), settings, module);

      interruptPoint(context, "Verify module", name.data());
      ::verifyModule(context, module);
      if (nullptr != context.dumpHandler) {
        auto callback = [&](const char *str, size_t len) {
          context.dumpHandler(context.dumpHandlerData, str, len);
        };

        CallbackOstream os(callback);
        module.print(os, nullptr, false, true);
      }
      ms.push_back(std::move(*mod));

      for (auto &&fun : toArray(current->funcList, current->funcListSize)) {
        functions.push_back(std::make_pair(fun.name, fun.func));
      }

      for (auto &&sym : toArray(current->symList, current->symListSize)) {
        symMap.insert(std::make_pair(decorate(sym.name), sym.sym));
      }
    }
    current = current->next;
  }

  interruptPoint(context, "Add modules");
  myJit.addModules(std::move(ms), symMap);

  JitFinaliser jitFinalizer(myJit);
  interruptPoint(context, "Resolve functions");
  for (auto &&fun : functions) {
    auto decorated = decorate(fun.first);
    auto symbol = myJit.findSymbol(decorated);
    auto addr = resolveSymbol(symbol);
    if (nullptr == addr) {
      std::string desc = std::string("Symbol not found in jitted code: \"") +
                         fun.first + "\" (\"" + decorated + "\")";
      fatal(context, desc);
    } else {
      *fun.second = addr;
    }

    if (nullptr != context.interruptPointHandler) {
      std::stringstream ss;
      ss << fun.first << " to " << addr;
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
#endif
    void rtCompileProcessImplSo(const void *modlist_head,
                                const Context *context, size_t contextSize) {
  assert(nullptr != context);
  assert(sizeof(*context) == contextSize);
  rtCompileProcessImplSoInternal(
      static_cast<const RtCompileModuleList *>(modlist_head), *context);
}
}
