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

std::unique_ptr<llvm::TargetMachine> createTargetMachine() {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetDisassembler();
  llvm::InitializeNativeTargetAsmPrinter();

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

JITContext::ListenerCleaner::ListenerCleaner(JITContext &o,
                                             llvm::raw_ostream *stream)
    : owner(o) {
  owner.listenerlayer.getTransform().stream = stream;
}

JITContext::ListenerCleaner::~ListenerCleaner() {
  owner.listenerlayer.getTransform().stream = nullptr;
}

JITContext::JITContext()
    : targetmachine(createTargetMachine()),
      dataLayout(targetmachine->createDataLayout()),
#if LDC_LLVM_VER >= 700
      stringPool(std::make_shared<llvm::orc::SymbolStringPool>()),
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

JITContext::~JITContext() {}

bool JITContext::addModule(std::unique_ptr<llvm::Module> module,
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

llvm::JITSymbol JITContext::findSymbol(const std::string &name) {
  return compileLayer.findSymbol(name, false);
}

void JITContext::clearSymMap() { symMap.clear(); }

void JITContext::addSymbol(std::string &&name, void *value) {
  symMap.emplace(std::make_pair(std::move(name), value));
}

void JITContext::reset() {
  if (compiled) {
    removeModule(moduleHandle);
    moduleHandle = {};
    compiled = false;
  }
}

void JITContext::removeModule(const ModuleHandleT &handle) {
  cantFail(compileLayer.removeModule(handle));
#if LDC_LLVM_VER >= 700
  execSession.releaseVModule(handle);
#endif
}

#if LDC_LLVM_VER >= 700
std::shared_ptr<llvm::orc::SymbolResolver> JITContext::createResolver() {
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
#else
std::shared_ptr<llvm::JITSymbolResolver> JITContext::createResolver() {
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
