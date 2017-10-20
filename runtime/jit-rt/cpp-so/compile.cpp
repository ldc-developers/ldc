
#include <cassert>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>

#include "callback_ostream.h"
#include "context.h"
#include "optimizer.h"
#include "utils.h"

#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/LambdaResolver.h"
#include "llvm/ExecutionEngine/RuntimeDyld.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/Cloning.h"

#if LDC_LLVM_VER >= 500
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#else
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#endif

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
  using ModuleHandleT = CompileLayerT::ModuleHandleT;
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

  bool addModule(std::unique_ptr<llvm::Module> module, const SymMap &symMap) {
    assert(nullptr != module);
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
    auto result = compileLayer.addModule(std::move(module), Resolver);
    if (!result) {
      return true;
    }
    moduleHandle = result.get();
#else
    std::vector<std::unique_ptr<llvm::Module>> modules;
    modules.emplace_back(std::move(module));
    moduleHandle = compileLayer.addModuleSet(
        std::move(modules), llvm::make_unique<llvm::SectionMemoryManager>(),
        std::move(Resolver));
#endif
    compiled = true;
    return false;
  }

  llvm::JITSymbol findSymbol(const std::string &name) {
    return compileLayer.findSymbol(name, false);
  }

  llvm::LLVMContext &getContext() { return context; }

  void reset() {
    if (compiled) {
      removeModule(moduleHandle);
      moduleHandle = {};
      compiled = false;
    }
  }

private:
  void removeModule(const ModuleHandleT &handle) {
#if LDC_LLVM_VER >= 500
    cantFail(compileLayer.removeModule(handle));
#else
    compileLayer.removeModuleSet(handle);
#endif
  }
};

MyJIT &getJit() {
  static MyJIT jit;
  return jit;
}

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

void dumpModuleAsm(const Context &context, const llvm::Module &module,
                   llvm::TargetMachine &TM) {
  if (nullptr != context.dumpHandler) {
    auto callback = [&](const char *str, size_t len) {
      context.dumpHandler(context.dumpHandlerData, DumpStage::FinalAsm, str,
                          len);
    };

    // TODO: I am not sure if passes added by addPassesToEmitFile can modify
    // module, so clone source module to be sure, also, it allow preserve
    // constness
    auto newModule = llvm::CloneModule(&module);

    llvm::legacy::PassManager PM;

    llvm::SmallVector<char, 0> asmBufferSV;
    llvm::raw_svector_ostream asmStream(asmBufferSV);

    if (TM.addPassesToEmitFile(PM, asmStream,
                               llvm::TargetMachine::CGFT_AssemblyFile)) {
      fatal(context, "Target does not support asm emission.");
    }
    PM.run(*newModule);

    callback(asmBufferSV.data(), asmBufferSV.size());
  }
}

void setFunctionsTarget(llvm::Module &module, llvm::TargetMachine &TM) {
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

void rtCompileProcessImplSoInternal(const RtCompileModuleList *modlist_head,
                                    const Context &context) {
  interruptPoint(context, "Init");
  MyJIT &myJit = getJit();
  auto current = modlist_head;

  std::vector<std::pair<std::string, void **>> functions;
  std::unique_ptr<llvm::Module> finalModule;
  SymMap symMap;
  OptimizerSettings settings;
  settings.optLevel = context.optLevel;
  settings.sizeLevel = context.sizeLevel;
  while (nullptr != current) {
    interruptPoint(context, "load IR");
    auto buff = llvm::MemoryBuffer::getMemBuffer(
        llvm::StringRef(current->irData,
                        static_cast<std::size_t>(current->irDataSize)),
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
                       toArray(current->varList,
                               static_cast<std::size_t>(current->varListSize)));

      if (nullptr == finalModule) {
        finalModule = std::move(*mod);
      } else {
        if (llvm::Linker::linkModules(*finalModule, std::move(*mod))) {
          fatal(context, "Can't merge module");
        }
      }

      for (auto &&fun :
           toArray(current->funcList,
                   static_cast<std::size_t>(current->funcListSize))) {
        functions.push_back(std::make_pair(fun.name, fun.func));
      }

      for (auto &&sym : toArray(current->symList, static_cast<std::size_t>(
                                                      current->symListSize))) {
        symMap.insert(std::make_pair(decorate(sym.name), sym.sym));
      }
    }
    current = current->next;
  }

  assert(nullptr != finalModule);
  dumpModule(context, *finalModule, DumpStage::MergedModule);
  interruptPoint(context, "Optimize final module");
  optimizeModule(context, myJit.getTargetMachine(), settings, *finalModule);

  interruptPoint(context, "Verify final module");
  verifyModule(context, *finalModule);

  dumpModule(context, *finalModule, DumpStage::OptimizedModule);
  dumpModuleAsm(context, *finalModule, myJit.getTargetMachine());

  interruptPoint(context, "Codegen final module");
  if (myJit.addModule(std::move(finalModule), symMap)) {
    fatal(context, "Can't codegen module");
  }

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
