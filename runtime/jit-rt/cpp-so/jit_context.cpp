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
      llvm::Optional<llvm::CodeModel::Model>{}, llvm::CodeGenOpt::Default,
      /*jit*/ true));
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
  owner.listenerlayer.getTransform().stream = stream;
}

DynamicCompilerContext::ListenerCleaner::~ListenerCleaner() {
  owner.listenerlayer.getTransform().stream = nullptr;
}

DynamicCompilerContext::DynamicCompilerContext(bool isMainContext)
    : targetmachine(createTargetMachine()),
      dataLayout(targetmachine->createDataLayout()),
      stringPool(std::make_shared<llvm::orc::SymbolStringPool>()),
      execSession(stringPool), resolver(createResolver()),
      objectLayer(execSession,
                  [this](llvm::orc::VModuleKey) {
                    return ObjectLayerT::Resources{
                        std::make_shared<llvm::SectionMemoryManager>(),
                        resolver};
                  }),
      listenerlayer(objectLayer, ModuleListener(*targetmachine)),
      compileLayer(listenerlayer, llvm::orc::SimpleCompiler(*targetmachine)),
      mainContext(isMainContext) {
  llvm::sys::DynamicLibrary::LoadLibraryPermanently(nullptr);
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
  auto result = compileLayer.addModule(handle, std::move(module));
  if (result) {
    execSession.releaseVModule(handle);
    return result;
  }
  if (auto err = compileLayer.emitAndFinalize(handle)) {
    execSession.releaseVModule(handle);
    return err;
  }
  moduleHandle = handle;
  compiled = true;
  return llvm::Error::success();
}

llvm::JITSymbol DynamicCompilerContext::findSymbol(const std::string &name) {
  return compileLayer.findSymbol(name, false);
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
  cantFail(compileLayer.removeModule(handle));
  execSession.releaseVModule(handle);
}

std::shared_ptr<llvm::orc::SymbolResolver>
DynamicCompilerContext::createResolver() {
  return llvm::orc::createLegacyLookupResolver(
      execSession,
      [this](llvm::StringRef name_) -> llvm::JITSymbol {
        const std::string name = name_.str();
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
