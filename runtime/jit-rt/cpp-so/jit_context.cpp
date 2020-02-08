//===-- jit_context.cpp ---------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the Boost Software License. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Jit runtime - shared library part.
// Defines jit context which stores evrything required for compilation.
//
//===----------------------------------------------------------------------===//

#include "jit_context.h"

#include <cassert>

#include "llvm/ADT/StringExtras.h"
#include "llvm/ExecutionEngine/RuntimeDyld.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"

namespace {

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

struct StaticInitHelper {
  StaticInitHelper() {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetDisassembler();
    llvm::InitializeNativeTargetAsmPrinter();
  }
};

StaticInitHelper &staticInit() {
  // Initialization may not be thread safe
  // Wrap it into static dummy object initialization
  static StaticInitHelper obj;
  return obj;
}

std::unique_ptr<llvm::TargetMachine> createTargetMachine() {
  staticInit();

  std::string triple(llvm::sys::getProcessTriple());
  std::string error;
  auto target = llvm::TargetRegistry::lookupTarget(triple, error);
  assert(target != nullptr);
  std::unique_ptr<llvm::TargetMachine> ret(target->createTargetMachine(
      triple, llvm::sys::getHostCPUName(), llvm::join(getHostAttrs(), ","), {},
      llvm::Optional<llvm::Reloc::Model>{},
#if LDC_LLVM_VER == 500
      llvm::CodeModel::JITDefault
#else
      llvm::Optional<llvm::CodeModel::Model>{}, llvm::CodeGenOpt::Default,
      /*jit*/ true
#endif
      ));
  assert(ret != nullptr);
  return ret;
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

} // anon namespace

DynamicCompilerContext::ListenerCleaner::ListenerCleaner(
    DynamicCompilerContext &o, llvm::raw_ostream *stream)
    : owner(o) {
  owner.listener_stream = stream;
}

DynamicCompilerContext::ListenerCleaner::~ListenerCleaner() {
  owner.listener_stream = nullptr;
}

DynamicCompilerContext::DynamicCompilerContext(bool isMainContext)
    : targetmachine(createTargetMachine()),
      dataLayout(targetmachine->createDataLayout()),
      stringPool(std::make_shared<llvm::orc::SymbolStringPool>()),
      execSession(stringPool),
#if LDC_LLVM_VER >= 900
      objectLayer(execSession,
                  []() {
                    return std::make_unique<llvm::SectionMemoryManager>();
                  }),
      listenerlayer(execSession, objectLayer, ModuleListener(*targetmachine,
                                                &listener_stream)),
      compileLayer(execSession, listenerlayer, llvm::orc::SimpleCompiler(*targetmachine)),
      context(std::make_unique<llvm::LLVMContext>()),
#else
      resolver(createResolver()),
      objectLayer(execSession,
                  [this](llvm::orc::VModuleKey) {
                    return ObjectLayerT::Resources{
                        std::make_shared<llvm::SectionMemoryManager>(),
                        resolver};
                  }),
      listenerlayer(objectLayer, ModuleListener(*targetmachine,
                                                &listener_stream)),
      compileLayer(listenerlayer, llvm::orc::SimpleCompiler(*targetmachine)),
#endif
      mainContext(isMainContext) {
  llvm::sys::DynamicLibrary::LoadLibraryPermanently(nullptr);
#if LDC_LLVM_VER >= 900
  auto generator = [this](llvm::orc::JITDylib &Parent, const llvm::orc::SymbolNameSet &Names)->llvm::Expected<llvm::orc::SymbolNameSet> {
    llvm::orc::SymbolNameSet Added;
    llvm::orc::SymbolMap NewSymbols;

    for (auto &Name : Names) {
      if ((*Name).empty())
        continue;

      auto it = symMap.find(*Name);
      if (symMap.end() != it) {
        auto SymAddr = llvm::pointerToJITTargetAddress(it->second);
        Added.insert(Name);
        NewSymbols[Name] = llvm::JITEvaluatedSymbol(SymAddr, llvm::JITSymbolFlags::Exported);
      }
      else if (auto SymAddr = getSymbolInProcess(*Name)) {
        Added.insert(Name);
        NewSymbols[Name] = llvm::JITEvaluatedSymbol(SymAddr, llvm::JITSymbolFlags::Exported);
      }
    }
    if (!NewSymbols.empty())
      cantFail(Parent.define(absoluteSymbols(std::move(NewSymbols))));

    return Added;
  };
  execSession.getMainJITDylib().setGenerator(generator);
#endif
}

DynamicCompilerContext::~DynamicCompilerContext() {}

llvm::Error
DynamicCompilerContext::addModule(std::unique_ptr<llvm::Module> module,
                                  llvm::raw_ostream *asmListener) {
  assert(nullptr != module);
  reset();

  ListenerCleaner cleaner(*this, asmListener);
  // Add the set to the JIT with the resolver we created above
  auto handle = execSession.allocateVModule();
#if LDC_LLVM_VER >= 900
  auto &dylib = execSession.getMainJITDylib();
//  llvm::orc::SymbolMap symbols;
//  for (auto &s : symMap) {
//    symbols[stringPool->intern(s.first)] = llvm::JITEvaluatedSymbol(llvm::pointerToJITTargetAddress(s.second), llvm::JITSymbolFlags::Exported);
//    std::cout << "aaaaaaaaaaa " << s.first << std::endl;
//  }
//  if (auto err = dylib.define(absoluteSymbols(std::move(symbols), handle))) {
//    return err;
//  }
  auto result = compileLayer.add(dylib, llvm::orc::ThreadSafeModule(std::move(module), context), handle);
#else
  auto result = compileLayer.addModule(handle, std::move(module));
#endif
  if (result) {
    execSession.releaseVModule(handle);
    return result;
  }
#if LDC_LLVM_VER < 900
  if (auto err = compileLayer.emitAndFinalize(handle)) {
    execSession.releaseVModule(handle);
    return err;
  }
#endif
  moduleHandle = handle;
  compiled = true;
  return llvm::Error::success();
}

llvm::JITSymbol DynamicCompilerContext::findSymbol(const std::string &name) {
#if LDC_LLVM_VER >= 900
  llvm::orc::JITDylib *libs[] = {
    &execSession.getMainJITDylib()
  };
  auto ret = execSession.lookup(libs, name);
  if (!ret) {
    return nullptr;
  }
  return ret.get();
#else
  return compileLayer.findSymbol(name, false);
#endif
}

llvm::LLVMContext &DynamicCompilerContext::getContext()
{
#if LDC_LLVM_VER >= 900
  return *context.getContext();
#else
  return context;
#endif
}

void DynamicCompilerContext::clearSymMap() { symMap.clear(); }

void DynamicCompilerContext::addSymbol(std::string &&name, void *value) {
  symMap.emplace(std::make_pair(std::move(name), value));
}

void DynamicCompilerContext::reset() {
  if (compiled) {
    removeModule(moduleHandle);
    moduleHandle = {};
    compiled = false;
  }
}

void DynamicCompilerContext::registerBind(
    void *handle, void *originalFunc, void *exampleFunc,
    const llvm::ArrayRef<ParamSlice> &params) {
  assert(bindInstances.count(handle) == 0);
  BindDesc::ParamsVec vec(params.begin(), params.end());
  bindInstances.insert({handle, {originalFunc, exampleFunc, std::move(vec)}});
}

void DynamicCompilerContext::unregisterBind(void *handle) {
  assert(bindInstances.count(handle) == 1);
  bindInstances.erase(handle);
}

bool DynamicCompilerContext::hasBindFunction(const void *handle) const {
  assert(handle != nullptr);
  auto it = bindInstances.find(const_cast<void *>(handle));
  return it != bindInstances.end();
}

bool DynamicCompilerContext::isMainContext() const { return mainContext; }

void DynamicCompilerContext::removeModule(const ModuleHandleT &handle) {
#if LDC_LLVM_VER < 900
  cantFail(compileLayer.removeModule(handle));
#endif
  execSession.releaseVModule(handle);
}

#if LDC_LLVM_VER < 900
std::shared_ptr<llvm::orc::SymbolResolver>
DynamicCompilerContext::createResolver() {
  return llvm::orc::createLegacyLookupResolver(
      execSession,
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
#endif
