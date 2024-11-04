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
#include <unordered_map>

#include "bind.h"
#include "callback_ostream.h"
#include "context.h"
#include "jit_context.h"
#include "optimizer.h"
#include "options.h"
#include "utils.h"

#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Mangler.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"

namespace {

#pragma pack(push, 1)

struct RtCompileFuncList {
  const char *name;
  void **func;
  void *originalFunc;
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
auto toArray(T *ptr,
             size_t size) -> llvm::ArrayRef<typename std::remove_cv<T>::type> {
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

std::string decorate(llvm::StringRef name, const llvm::DataLayout &datalayout) {
  assert(!name.empty());
  llvm::SmallVector<char, 64> ret;
  llvm::Mangler::getNameWithPrefix(ret, name, datalayout);
  assert(!ret.empty());
  return std::string(ret.data(), ret.size());
}

struct JitModuleInfo final {
private:
  struct Func final {
    llvm::StringRef name;
    void **thunkVar;
    void *originalFunc;
  };
  std::vector<Func> funcs;
  mutable std::unordered_map<const void *, const Func *> funcsMap;

  struct BindHandle final {
    std::string name;
    void *handle = nullptr;
  };
  std::vector<BindHandle> bindHandles;

public:
  JitModuleInfo(const Context &context,
                const RtCompileModuleList *modlist_head) {
    enumModules(modlist_head, context, [&](const RtCompileModuleList &current) {
      for (auto &&fun : toArray(current.funcList, static_cast<std::size_t>(
                                                      current.funcListSize))) {
        funcs.push_back({fun.name, fun.func, fun.originalFunc});
      }
    });
  }

  const std::vector<Func> &functions() const { return funcs; }

  const std::unordered_map<const void *, const Func *> &functionsMap() const {
    if (funcsMap.empty() && !funcs.empty()) {
      for (auto &&fun : funcs) {
        funcsMap.insert({fun.originalFunc, &fun});
      }
    }
    return funcsMap;
  }

  const Func *getFunc(const void *ptr) const {
    assert(ptr != nullptr);
    auto &funcMap = functionsMap();
    auto it = funcMap.find(ptr);
    if (funcMap.end() != it) {
      return it->second;
    }
    return nullptr;
  }

  const std::vector<BindHandle> &getBindHandles() const { return bindHandles; }

  void addBindHandle(llvm::StringRef name, void *handle) {
    assert(!name.empty());
    assert(handle != nullptr);
    BindHandle h;
    h.name = name.str();
    h.handle = handle;
    bindHandles.emplace_back(std::move(h));
  }
};

void *resolveSymbol(llvm::Expected<llvm::orc::ExecutorSymbolDef> &symbol) {
  if (!symbol) {
    consumeError(symbol.takeError());
    return nullptr;
  } else {
    return symbol->getAddress().toPtr<void *>();
  }
}

static inline llvm::Function *
getIrFunc(const void *ptr, JitModuleInfo &moduleInfo, llvm::Module &module) {
  assert(ptr != nullptr);
  auto funcDesc = moduleInfo.getFunc(ptr);
  if (funcDesc == nullptr) {
    return nullptr;
  }
  return module.getFunction(funcDesc->name);
}

void generateBind(const Context &context, DynamicCompilerContext &jitContext,
                  JitModuleInfo &moduleInfo, llvm::Module &module) {
  std::unordered_map<const void *, llvm::Function *> bindFuncs;
  bindFuncs.reserve(jitContext.getBindInstances().size() * 2);

  auto genBind = [&](void *bindPtr, void *originalFunc, void *exampleFunc,
                     const llvm::ArrayRef<ParamSlice> &params) {
    assert(bindPtr != nullptr);
    assert(bindFuncs.end() == bindFuncs.find(bindPtr));
    auto funcToInline = getIrFunc(originalFunc, moduleInfo, module);
    if (funcToInline == nullptr) {
      fatal(context, "Bind: function body not available");
    }
    auto exampleIrFunc = getIrFunc(exampleFunc, moduleInfo, module);
    assert(exampleIrFunc != nullptr);
    auto errhandler = [&](const std::string &str) { fatal(context, str); };
    auto overrideHandler = [&](llvm::Type &type, const void *data,
                               size_t size) -> llvm::Constant * {
      if (type.isPointerTy()) {
        auto getBindFunc = [&]() {
          auto handle = *static_cast<void *const *>(data);
          return handle != nullptr && jitContext.hasBindFunction(handle)
                     ? handle
                     : nullptr;
        };

        llvm::Function *maybeFunction = getIrFunc(
            *reinterpret_cast<void *const *>(data), moduleInfo, module);
        if (size == sizeof(void *) && maybeFunction) {
          return llvm::ConstantExpr::getBitCast(maybeFunction, &type);
        }
        if (auto handle = getBindFunc()) {
          auto it = bindFuncs.find(handle);
          assert(bindFuncs.end() != it);
          auto bindIrFunc = it->second;
          auto funcPtrType = bindIrFunc->getType();
          auto globalVar1 = new llvm::GlobalVariable(
              module, funcPtrType, true, llvm::GlobalValue::PrivateLinkage,
              bindIrFunc, ".jit_bind_handle");
          return llvm::ConstantExpr::getBitCast(globalVar1, &type);
        }
      }
      return nullptr;
    };
    auto func = bindParamsToFunc(module, *funcToInline, *exampleIrFunc, params,
                                 errhandler, BindOverride(overrideHandler));
    moduleInfo.addBindHandle(func->getName(), bindPtr);
    bindFuncs.insert({bindPtr, func});
  };
  for (auto &&bind : jitContext.getBindInstances()) {
    auto bindPtr = bind.first;
    auto &bindDesc = bind.second;
    assert(bindDesc.originalFunc != nullptr);
    genBind(bindPtr, bindDesc.originalFunc, bindDesc.exampleFunc,
            bindDesc.params);
  }
}

void applyBind(const Context &context, DynamicCompilerContext &jitContext,
               const JitModuleInfo &moduleInfo) {
  auto &layout = jitContext.getDataLayout();
  for (auto &elem : moduleInfo.getBindHandles()) {
    auto decorated = decorate(elem.name, layout);
    auto symbol = jitContext.lookup(decorated);
    auto addr = resolveSymbol(symbol);
    if (nullptr == addr) {
      std::string desc = std::string("Symbol not found in jitted code: \"") +
                         elem.name + "\" (\"" + decorated + "\")";
      fatal(context, desc);
    } else {
      auto handle = static_cast<void **>(elem.handle);
      *handle = addr;
    }
  }
}

DynamicCompilerContext &getJit(DynamicCompilerContext *context) {
  if (context != nullptr) {
    return *context;
  }
  static DynamicCompilerContext jit(/*mainContext*/ true);
  return jit;
}

void setRtCompileVars(const Context &context, llvm::Module &module,
                      llvm::ArrayRef<RtCompileVarList> vals) {
  for (auto &&val : vals) {
    setRtCompileVar(context, module, val.name, val.init);
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

void setFunctionsTarget(llvm::Module &module, const llvm::TargetMachine *TM) {
  // Set function target cpu to host if it wasn't set explicitly
  for (auto &&func : module.functions()) {
    if (!func.hasFnAttribute("target-cpu")) {
      func.addFnAttr("target-cpu", TM->getTargetCPU());
    }

    if (!func.hasFnAttribute("target-features")) {
      auto featStr = TM->getTargetFeatureString();
      if (!featStr.empty()) {
        func.addFnAttr("target-features", featStr);
      }
    }
  }
}

struct JitFinaliser final {
  DynamicCompilerContext &jit;
  bool finalized = false;
  explicit JitFinaliser(DynamicCompilerContext &j) : jit(j) {}
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
  DynamicCompilerContext &myJit = getJit(context.compilerContext);

  JitModuleInfo moduleInfo(context, modlist_head);
  llvm::orc::ThreadSafeModule finalModule;
  myJit.clearSymMap();
  auto &layout = myJit.getDataLayout();
  OptimizerSettings settings;
  settings.optLevel = context.optLevel;
  settings.sizeLevel = context.sizeLevel;
  auto TM = myJit.getTargetMachine();
  enumModules(modlist_head, context, [&](const RtCompileModuleList &current) {
    interruptPoint(context, "load IR");
    auto buff = llvm::MemoryBuffer::getMemBuffer(
        llvm::StringRef(current.irData,
                        static_cast<std::size_t>(current.irDataSize)),
        "", false);
    interruptPoint(context, "parse IR");
    auto mod = llvm::parseBitcodeFile(*buff, *myJit.getContext());
    if (!mod) {
      fatal(context, "Unable to parse IR: " + llvm::toString(mod.takeError()));
    } else {
      llvm::Module &module = **mod;
      const auto name = module.getName();
      interruptPoint(context, "Verify module", name.data());
      verifyModule(context, module);

      dumpModule(context, module, DumpStage::OriginalModule);
      setFunctionsTarget(module, TM);

      module.setDataLayout(TM->createDataLayout());

      interruptPoint(context, "setRtCompileVars", name.data());
      setRtCompileVars(context, module,
                       toArray(current.varList,
                               static_cast<std::size_t>(current.varListSize)));

      if (!finalModule) {
        finalModule = llvm::orc::ThreadSafeModule(std::move(*mod),
                                                  myJit.getThreadSafeContext());
      } else {
        finalModule.withModuleDo([&](llvm::Module &M) {
          if (llvm::Linker::linkModules(M, std::move(*mod))) {
            fatal(context, "Can't merge module");
          }
        });
      }

      for (auto &&sym : toArray(current.symList, static_cast<std::size_t>(
                                                     current.symListSize))) {
        myJit.addSymbol(decorate(sym.name, layout), sym.sym);
      }
    }
  });

  assert(!!finalModule);

  finalModule.withModuleDo([&](llvm::Module &M) {
    interruptPoint(context, "Generate bind functions");
    generateBind(context, myJit, moduleInfo, M);
    dumpModule(context, M, DumpStage::MergedModule);
    interruptPoint(context, "Optimize final module");
    optimizeModule(settings, &M, TM);

    interruptPoint(context, "Verify final module");
    verifyModule(context, M);

    dumpModule(context, M, DumpStage::OptimizedModule);
  });

  interruptPoint(context, "Codegen final module");
  auto callback = [&](const char *str, size_t len) {
    context.dumpHandler(context.dumpHandlerData, DumpStage::FinalAsm, str, len);
  };
  CallbackOstream os{callback};
  std::unique_ptr<DynamicCompilerContext::ListenerCleaner> listener{};
  if (nullptr != context.dumpHandler) {
    listener = myJit.addScopedListener(&os);
  }
  if (auto err = myJit.addModule(std::move(finalModule))) {
    fatal(context, "Can't codegen module: " + llvm::toString(std::move(err)));
  }

  JitFinaliser jitFinalizer(myJit);
  /*if (myJit.isMainContext())*/ {
    interruptPoint(context, "Resolve functions");
    for (auto &&fun : moduleInfo.functions()) {
      if (fun.thunkVar == nullptr) {
        continue;
      }
      auto decorated = decorate(fun.name, layout);
      auto symbol = myJit.lookup(decorated);
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
  }
  interruptPoint(context, "Update bind handles");
  applyBind(context, myJit, moduleInfo);
  jitFinalizer.finalze();
}

} // anon namespace

extern "C" {
EXTERNAL void JIT_API_ENTRYPOINT(const void *modlist_head,
                                 const Context *context, size_t contextSize) {
  assert(nullptr != context);
  assert(sizeof(*context) == contextSize);
  rtCompileProcessImplSoInternal(
      static_cast<const RtCompileModuleList *>(modlist_head), *context);
}

EXTERNAL void JIT_REG_BIND_PAYLOAD(class DynamicCompilerContext *context,
                                   void *handle, void *originalFunc,
                                   void *exampleFunc, const ParamSlice *params,
                                   size_t paramsSize) {
  assert(handle != nullptr);
  assert(originalFunc != nullptr);
  assert(exampleFunc != nullptr);
  DynamicCompilerContext &myJit = getJit(context);
  myJit.registerBind(handle, originalFunc, exampleFunc,
                     toArray(params, paramsSize));
}

EXTERNAL void JIT_UNREG_BIND_PAYLOAD(class DynamicCompilerContext *context,
                                     void *handle) {
  assert(handle != nullptr);
  DynamicCompilerContext &myJit = getJit(context);
  myJit.unregisterBind(handle);
}

EXTERNAL DynamicCompilerContext *JIT_CREATE_COMPILER_CONTEXT() {
  return new DynamicCompilerContext(false);
}

EXTERNAL void JIT_DESTROY_COMPILER_CONTEXT(DynamicCompilerContext *context) {
  assert(context != nullptr);
  delete context;
}

EXTERNAL bool JIT_SET_OPTS(const Slice<Slice<const char>> *args,
                           void (*errs)(void *, const char *, size_t),
                           void *errsContext) {
  assert(args != nullptr);
  return parseOptions(*args, errs, errsContext);
}
}
