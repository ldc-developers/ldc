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
#include <memory>
#include <sstream>
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
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"

namespace {

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

void *resolveSymbol(llvm::Expected<llvm::orc::ExecutorAddr> &symbol) {
  if (!symbol) {
    consumeError(symbol.takeError());
    return nullptr;
  }
  return symbol->toPtr<void *>();
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
      // due to ABI rewrites, function pointers can be an integer literal on
      // aarch64 we will do a quick probe here to check if it's a function
      // pointer
      bool maybeIntPtr =
          (jitContext.getTargetTriple().isAArch64() && type.isIntegerTy(64)) ||
          (jitContext.getTargetTriple().isOSWindows() && type.isIntegerTy(32));
      if (type.isPointerTy() || maybeIntPtr) {
        auto getBindFunc = [&]() {
          auto handle = *static_cast<void *const *>(data);
          return handle != nullptr && jitContext.hasBindFunction(handle)
                     ? handle
                     : nullptr;
        };

        auto *maybeFunctionPtr = *reinterpret_cast<void *const *>(data);
        llvm::Function *maybeFunction =
            maybeFunctionPtr ? getIrFunc(maybeFunctionPtr, moduleInfo, module)
                             : nullptr;
        if (size == sizeof(void *) && maybeFunction) {
          return maybeIntPtr
                     ? llvm::ConstantExpr::getPtrToInt(maybeFunction, &type)
                     : llvm::ConstantExpr::getBitCast(maybeFunction, &type);
        }
        if (auto handle = getBindFunc()) {
          auto it = bindFuncs.find(handle);
          if (bindFuncs.end() == it && maybeIntPtr) {
            // maybe it's really not a pointer
            return nullptr;
          }
          assert(bindFuncs.end() != it);
          auto bindIrFunc = it->second;
          auto funcPtrType = bindIrFunc->getType();
          auto globalVar1 = new llvm::GlobalVariable(
              module, funcPtrType, true, llvm::GlobalValue::PrivateLinkage,
              bindIrFunc, ".jit_bind_handle");
          return maybeIntPtr
                     ? llvm::ConstantExpr::getPtrToInt(globalVar1, &type)
                     : llvm::ConstantExpr::getBitCast(globalVar1, &type);
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
  for (auto &elem : moduleInfo.getBindHandles()) {
    auto symbol = jitContext.lookup(elem.name);
    auto addr = resolveSymbol(symbol);
    if (nullptr == addr) {
      std::string desc = std::string("Symbol not found in jitted code: \"") +
                         elem.name + "\" (\"" + jitContext.mangle(elem.name) +
                         "\")";
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
  static std::unique_ptr<DynamicCompilerContext> jit =
      DynamicCompilerContext::Create(/*mainContext*/ true);
  return *jit;
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

#ifndef LDC_JITRT_USE_JITLINK
static void insertABIHacks(const Context &context,
                           DynamicCompilerContext &jitContext,
                           llvm::Module &module) {
  bool isAArch64 = jitContext.getTargetTriple().isAArch64();
  if (isAArch64) {
    // insert DW.ref._d_eh_personality stub
    auto targetSymbol = module.getFunction("_d_eh_personality");
    if (!targetSymbol) {
      return;
    }
    constexpr const char *thunkName = "_d_eh_personality__thunk";
    auto *pointerType = llvm::PointerType::getUnqual(module.getContext());
    auto *thunkVariable = new llvm::GlobalVariable(
        pointerType, true, llvm::GlobalValue::ExternalLinkage,
        llvm::ConstantExpr::getBitCast(targetSymbol, pointerType), thunkName);
    module.insertGlobalVariable(thunkVariable);
    auto targetSymbolAddr = jitContext.lookup(thunkName);
    assert(targetSymbolAddr);
    jitContext.addSymbol("DW.ref._d_eh_personality",
                         targetSymbolAddr->toPtr<void *>());
  }
}
#endif

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

      module.setDataLayout(myJit.getDataLayout());

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

      llvm::orc::SymbolMap symMap{
          static_cast<unsigned int>(current.symListSize)};
      for (int i = 0; i < current.symListSize; ++i) {
        const auto &sym = current.symList[i];
        symMap[myJit.mangleAndIntern(sym.name)] = {
            llvm::orc::ExecutorAddr::fromPtr(sym.sym),
            llvm::JITSymbolFlags::Exported};
      }
      myJit.addSymbols(symMap);
    }
  });

  assert(!!finalModule);

  finalModule.withModuleDo([&](llvm::Module &M) {
    interruptPoint(context, "Generate bind functions");
    generateBind(context, myJit, moduleInfo, M);
#ifndef LDC_JITRT_USE_JITLINK
    insertABIHacks(context, myJit, M);
#endif
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
      auto symbol = myJit.lookup(fun.name);
      auto addr = resolveSymbol(symbol);
      if (nullptr == addr) {
        std::string desc = std::string("Symbol not found in jitted code: \"") +
                           fun.name.data() + "\" (\"" + myJit.mangle(fun.name) +
                           "\")";
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
  return DynamicCompilerContext::Create(/*mainContext*/ false).release();
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
