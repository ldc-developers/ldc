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
#include "disassembler.h"
#include "optimizer.h"
#include "utils.h"

#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/LambdaResolver.h"
#include "llvm/ExecutionEngine/Orc/ObjectTransformLayer.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
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

#if LDC_LLVM_VER >= 700
#include "llvm/ExecutionEngine/Orc/Legacy.h"
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
    llvm::InitializeNativeTargetDisassembler();
    llvm::InitializeNativeTargetAsmPrinter();
  }
};

std::string decorate(const std::string &name) {
#if __APPLE__
  return "_" + name;
#elif _WIN32
  assert(!name.empty());
  if (name[0] == 0x1)
    return name.substr(1);
#if _M_IX86
  return "_" + name;
#else
  return name;
#endif
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

struct ModuleListener {
  llvm::TargetMachine &targetmachine;
  llvm::raw_ostream *stream = nullptr;

  ModuleListener(llvm::TargetMachine &tm) : targetmachine(tm) {}

  template <typename T> auto operator()(T &&object) -> T {
    if (nullptr != stream) {
      disassemble(targetmachine, *object->getBinary(), *stream);
    }
    return std::move(object);
  }
};

class MyJIT {
private:
  llvm_init_obj initObj;
  llvm::llvm_shutdown_obj shutdownObj;
  std::unique_ptr<llvm::TargetMachine> targetmachine;
  const llvm::DataLayout dataLayout;
  using ObjectLayerT = llvm::orc::RTDyldObjectLinkingLayer;
  using ListenerLayerT =
      llvm::orc::ObjectTransformLayer<ObjectLayerT, ModuleListener>;
  using CompileLayerT =
      llvm::orc::IRCompileLayer<ListenerLayerT, llvm::orc::SimpleCompiler>;
#if LDC_LLVM_VER >= 700
  using ModuleHandleT = llvm::orc::VModuleKey;
  llvm::orc::SymbolStringPool stringPool;
  llvm::orc::ExecutionSession execSession;
  std::shared_ptr<llvm::orc::SymbolResolver> resolver;
#else
  using ModuleHandleT = CompileLayerT::ModuleHandleT;
#endif
  ObjectLayerT objectLayer;
  ListenerLayerT listenerlayer;
  CompileLayerT compileLayer;
  llvm::LLVMContext context;
  bool compiled = false;
  ModuleHandleT moduleHandle;
  SymMap symMap;

  struct ListenerCleaner final {
    MyJIT &owner;
    ListenerCleaner(MyJIT &o, llvm::raw_ostream *stream) : owner(o) {
      owner.listenerlayer.getTransform().stream = stream;
    }
    ~ListenerCleaner() { owner.listenerlayer.getTransform().stream = nullptr; }
  };

public:
  MyJIT()
      : targetmachine(llvm::EngineBuilder().selectTarget(
            llvm::Triple(llvm::sys::getProcessTriple()), llvm::StringRef(),
            llvm::sys::getHostCPUName(), getHostAttrs())),
        dataLayout(targetmachine->createDataLayout()),
#if LDC_LLVM_VER >= 700
        execSession(stringPool), resolver(createResolver()),
        objectLayer(execSession,
                    [this](llvm::orc::VModuleKey) {
                      return llvm::orc::RTDyldObjectLinkingLayer::Resources{
                          std::make_shared<llvm::SectionMemoryManager>(),
                          resolver};
                    }),
#else
        objectLayer(
            []() { return std::make_shared<llvm::SectionMemoryManager>(); }),
#endif
        listenerlayer(objectLayer, ModuleListener(*targetmachine)),
        compileLayer(listenerlayer, llvm::orc::SimpleCompiler(*targetmachine)) {
    llvm::sys::DynamicLibrary::LoadLibraryPermanently(nullptr);
  }

  llvm::TargetMachine &getTargetMachine() { return *targetmachine; }

  bool addModule(std::unique_ptr<llvm::Module> module,
                 llvm::raw_ostream *asmListener) {
    assert(nullptr != module);
    reset();

    ListenerCleaner cleaner(*this, asmListener);
    // Add the set to the JIT with the resolver we created above
#if LDC_LLVM_VER >= 700
    auto handle = execSession.allocateVModule();
    auto result = compileLayer.addModule(handle, std::move(module));
    if (result) {
      execSession.releaseVModule(handle);
      return true;
    }
    moduleHandle = handle;
#else
    auto result = compileLayer.addModule(std::move(module), createResolver());
    if (!result) {
      return true;
    }
    moduleHandle = result.get();
#endif
    compiled = true;
    return false;
  }

  llvm::JITSymbol findSymbol(const std::string &name) {
    return compileLayer.findSymbol(name, false);
  }

  llvm::LLVMContext &getContext() { return context; }

  SymMap &getSymMap() { return symMap; }

  void reset() {
    if (compiled) {
      removeModule(moduleHandle);
      moduleHandle = {};
      compiled = false;
    }
  }

private:
  void removeModule(const ModuleHandleT &handle) {
    cantFail(compileLayer.removeModule(handle));
#if LDC_LLVM_VER >= 700
    execSession.releaseVModule(handle);
#endif
  }

#if LDC_LLVM_VER >= 700
  std::shared_ptr<llvm::orc::SymbolResolver> createResolver() {
    return llvm::orc::createLegacyLookupResolver(
        [this](const std::string &name) -> llvm::JITSymbol {
          if (auto Sym = compileLayer.findSymbol(name, false)) {
            return Sym;
          } else if (auto Err = Sym.takeError()) {
            return std::move(Err);
          }
          auto it = symMap.find(name);
          if (symMap.end() != it) {
            return llvm::JITSymbol(
                reinterpret_cast<llvm::JITTargetAddress>(it->second),
                llvm::JITSymbolFlags::Exported);
          }
          if (auto SymAddr = getSymbolInProcess(name)) {
            return llvm::JITSymbol(SymAddr, llvm::JITSymbolFlags::Exported);
          }
          return nullptr;
        },
        [](llvm::Error Err) {
          llvm::cantFail(std::move(Err), "lookupFlags failed");
        });
  }
#else
  std::shared_ptr<llvm::JITSymbolResolver> createResolver() {
    // Build our symbol resolver:
    // Lambda 1: Look back into the JIT itself to find symbols that are part of
    //           the same "logical dylib".
    // Lambda 2: Search for external symbols in the host process.
    return llvm::orc::createLambdaResolver(
        [this](const std::string &name) {
          if (auto Sym = compileLayer.findSymbol(name, false)) {
            return Sym;
          }
          return llvm::JITSymbol(nullptr);
        },
        [this](const std::string &name) {
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
  }
#endif
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

template <typename T>
auto toArray(T *ptr, size_t size)
    -> llvm::ArrayRef<typename std::remove_cv<T>::type> {
  return llvm::ArrayRef<typename std::remove_cv<T>::type>(ptr, size);
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

void rtCompileProcessImplSoInternal(const RtCompileModuleList *modlist_head,
                                    const Context &context) {
  if (nullptr == modlist_head) {
    // No jit modules to compile
    return;
  }
  interruptPoint(context, "Init");
  MyJIT &myJit = getJit();

  std::vector<std::pair<std::string, void **>> functions;
  std::unique_ptr<llvm::Module> finalModule;
  auto &symMap = myJit.getSymMap();
  symMap.clear();
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

      for (auto &&fun : toArray(current.funcList, static_cast<std::size_t>(
                                                      current.funcListSize))) {
        functions.push_back(std::make_pair(fun.name, fun.func));
      }

      for (auto &&sym : toArray(current.symList, static_cast<std::size_t>(
                                                     current.symListSize))) {
        symMap.insert(std::make_pair(decorate(sym.name), sym.sym));
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
