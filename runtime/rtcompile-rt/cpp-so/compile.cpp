
#include <cassert>
#include <stdexcept>
#include <map>

#include "optimizer.h"
#include "context.h"
#include "utils.h"
#include "callback_ostream.h"

#include "llvm/Support/ManagedStatic.h"

#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/RuntimeDyld.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/LambdaResolver.h"
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"

#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/Host.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Bitcode/BitcodeReader.h"

namespace {

#pragma pack(push,1)

struct RtCompileFuncList
{
  const char* name;
  void** func;
};

struct RtCompileSymList
{
  const char* name;
  void* sym;
};

struct RtCompileVarList
{
  const char* name;
  const void* init;
};

struct RtComileModuleList
{
  RtComileModuleList* next;
  const char* irData;
  int irDataSize;
  RtCompileFuncList* funcList;
  int funcListSize;
  RtCompileSymList* symList;
  int symListSize;
  RtCompileVarList* varList;
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

using SymMap = std::map<std::string, void*>;

struct llvm_init_obj {
  llvm_init_obj() {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
  }
};

class MyJIT {
private:
  llvm_init_obj initObj;
  llvm::llvm_shutdown_obj shutdownObj;
  std::unique_ptr<llvm::TargetMachine> targetmachine;
  const llvm::DataLayout dataLayout;
  using ObjectLayerT = llvm::orc::ObjectLinkingLayer<>;
  ObjectLayerT objectLayer;
  using CompileLayerT = llvm::orc::IRCompileLayer<ObjectLayerT>;
  CompileLayerT compileLayer;
  llvm::LLVMContext context;
  typedef CompileLayerT::ModuleSetHandleT ModuleHandleT;
  bool compiled = false;
  ModuleHandleT moduleHandle;

public:

  MyJIT():
    targetmachine(llvm::EngineBuilder()
                  .setRelocationModel(llvm::Reloc::Static)
                  .selectTarget(llvm::Triple(llvm::sys::getProcessTriple()),
                                llvm::StringRef(),
                                llvm::sys::getHostCPUName(),
                                getHostAttrs())),
    dataLayout(targetmachine->createDataLayout()),
    compileLayer(objectLayer, llvm::orc::SimpleCompiler(*targetmachine))
  {
    llvm::sys::DynamicLibrary::LoadLibraryPermanently(nullptr);
  }

  llvm::TargetMachine& getTargetMachine() { return *targetmachine; }

  void addModules(std::vector<std::unique_ptr<llvm::Module>> &&modules,
                          const SymMap& symMap) {
    reset();
    // Build our symbol resolver:
    // Lambda 1: Look back into the JIT itself to find symbols that are part of
    //           the same "logical dylib".
    // Lambda 2: Search for external symbols in the host process.
    auto Resolver = llvm::orc::createLambdaResolver(
                      [&](const std::string& name) {
      if (auto Sym = compileLayer.findSymbol(name, false)) {
        return Sym;
      }
      return llvm::JITSymbol(nullptr);
    },
    [&](const std::string& name) {
      auto it = symMap.find(name);
      if (symMap.end() != it) {
        return llvm::JITSymbol(reinterpret_cast<llvm::JITTargetAddress>(it->second),
                               llvm::JITSymbolFlags::Exported);
      }
      if (auto SymAddr = llvm::RTDyldMemoryManager::getSymbolAddressInProcess(name)) {
        return llvm::JITSymbol(SymAddr, llvm::JITSymbolFlags::Exported);
      }
      return llvm::JITSymbol(nullptr);
    });

    // Add the set to the JIT with the resolver we created above and a newly
    // created SectionMemoryManager.
    moduleHandle = compileLayer.addModuleSet(std::move(modules),
                                             llvm::make_unique<llvm::SectionMemoryManager>(),
                                             std::move(Resolver));
    compiled = true;
  }

  llvm::JITSymbol findSymbol(const std::string &name) {
    return compileLayer.findSymbol(name, false);
  }

  llvm::LLVMContext& getContext() { return context; }

//  void removeModule(ModuleHandle H) {
//    CompileLayer.removeModuleSet(H);
//  }

  void reset() {
    if (compiled) {
      compileLayer.removeModuleSet(moduleHandle);
      compiled = false;
    }
  }

};

void setRtCompileVars(const Context &context,
                      llvm::Module& module,
                      llvm::ArrayRef<RtCompileVarList> vals) {
  for (auto&& val: vals) {
    setRtCompileVar(context, module, val.name, val.init);
  }
}

template<typename T>
llvm::ArrayRef<T> toArray(T* ptr, size_t size) {
  return llvm::ArrayRef<T>(ptr, size);
}

struct JitFinaliser final {
  MyJIT& jit;
  bool finalized = false;
  explicit JitFinaliser(MyJIT& j):
    jit(j) {}
  ~JitFinaliser() {
    if (!finalized) {
      jit.reset();
    }
  }

  void finalze() { finalized = true; }
};

MyJIT& getJit()
{
  static MyJIT jit;
  return jit;
}

void rtCompileProcessImplSoInternal(const RtComileModuleList* modlist_head, const Context& context) {
  interruptPoint(context, "Init");
  MyJIT& myJit = getJit();
  auto current = modlist_head;

  std::vector<std::pair<std::string, void**> > functions;
  std::vector<std::unique_ptr<llvm::Module>> ms;
  SymMap symMap;
  OptimizerSettings settings;
  settings.optLevel = context.optLevel;
  settings.sizeLeve = context.sizeLeve;
  while (nullptr != current) {
    interruptPoint(context, "load IR");
    auto buff = llvm::MemoryBuffer::getMemBuffer(llvm::StringRef(current->irData, current->irDataSize), "", false);
    interruptPoint(context, "parse IR");
    auto mod = llvm::parseBitcodeFile(*buff, myJit.getContext());
    if (!mod) {
      fatal(context, "Unable to parse IR");
    }
    else {
      llvm::Module& module = **mod;
      const auto name = module.getName();
      interruptPoint(context,"Verify module", name.data());
      ::verifyModule(context, module);
      module.setDataLayout(myJit.getTargetMachine().createDataLayout());

      interruptPoint(context, "setRtCompileVars", name.data());
      setRtCompileVars(context,
                       module,
                       toArray(current->varList, current->varListSize));

      interruptPoint(context, "Optimize module", name.data());
      optimizeModule(context, myJit.getTargetMachine(), settings, module);

      interruptPoint(context, "Verify module", name.data());
      ::verifyModule(context, module);
      if (nullptr != context.dumpHandler) {
        auto callback =[&](const char* str, size_t len) {
          context.dumpHandler(context.dumpHandlerData, str, len);
        };

        CallbackOstream os(callback);
        module.print(os, nullptr, false, true);
      }
      ms.push_back(std::move(*mod));

      for (auto&& fun: toArray(current->funcList, current->funcListSize)) {
        functions.push_back(std::make_pair(fun.name, fun.func));
      }

      for (auto&& sym: toArray(current->symList, current->symListSize)) {
        symMap.insert(std::make_pair(sym.name, sym.sym));
      }
    }
    current = current->next;
  }

  interruptPoint(context, "Add modules");
  myJit.addModules(std::move(ms), symMap);
  JitFinaliser jitFinalizer(myJit);
  interruptPoint(context, "Resolve functions");
  for (auto&& fun: functions) {
    auto symbol = myJit.findSymbol(fun.first);
    const auto addr = symbol.getAddress();
    if (0 == addr) {
      std::string desc = std::string("Symbol not found in jitted code: ") + fun.first;
      fatal(context, desc);
    }
    else {
      *fun.second = reinterpret_cast<void*>(addr);
    }
  }
  jitFinalizer.finalze();
}

} // anon namespace

extern "C" {

#ifdef _WIN32
__declspec(dllexport)
#endif
void rtCompileProcessImplSo(const void* modlist_head,
                            const Context* context,
                            size_t contextSize) {
  assert(nullptr != context);
  assert(sizeof(*context) == contextSize);
  try {
    rtCompileProcessImplSoInternal(
          static_cast<const RtComileModuleList*>(modlist_head),
          *context);
  }
  catch (const std::exception& e) {
    std::string desc = std::string("Exception was thrown: ") + e.what();
    fatal(*context, desc);
  }
}

}
