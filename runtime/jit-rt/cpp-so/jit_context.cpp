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
#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/IR/LLVMContext.h>

#include "llvm/ADT/StringExtras.h"
#include "llvm/ExecutionEngine/RuntimeDyld.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/MC/TargetRegistry.h"
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
    llvm::InitializeNativeTargetAsmParser();
    llvm::InitializeNativeTargetAsmPrinter();
  }
};

StaticInitHelper &staticInit() {
  // Initialization may not be thread safe
  // Wrap it into static dummy object initialization
  static StaticInitHelper obj;
  return obj;
}

llvm::orc::JITTargetMachineBuilder createTargetMachine() {
  staticInit();

  auto autoJTMB = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (autoJTMB) {
    return *autoJTMB;
  }
  std::string triple(llvm::sys::getProcessTriple());
  llvm::orc::JITTargetMachineBuilder JTMB{llvm::Triple{triple}};
  JTMB.setFeatures(llvm::join(getHostAttrs(), ","));
  return JTMB;
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

class LDCSymbolDefinitionGenerator : public llvm::orc::DefinitionGenerator {
private:
  DynamicCompilerContext *owner;

public:
  explicit LDCSymbolDefinitionGenerator(DynamicCompilerContext *o) : owner(o) {}

  llvm::Error
  tryToGenerate(llvm::orc::LookupState &LS, llvm::orc::LookupKind K,
                llvm::orc::JITDylib &JD,
                llvm::orc::JITDylibLookupFlags JDLookupFlags,
                const llvm::orc::SymbolLookupSet &Symbols) override {
    llvm::orc::SymbolMap NewSymbols{};
    for (auto &symbol : Symbols) {
      auto name = symbol.first;
      auto result = owner->findSymbol((*name).str());
      if (!result) {
        continue;
      }
      NewSymbols[symbol.first] = *result;
    }

    return JD.define(llvm::orc::absoluteSymbols(NewSymbols));
  }
};

DynamicCompilerContext::DynamicCompilerContext(bool isMainContext)
    : jtmb(createTargetMachine()),
      targetmachine(cantFail(jtmb.createTargetMachine())),
      dataLayout(cantFail(jtmb.getDefaultDataLayoutForTarget())),
      execSession(std::make_unique<llvm::orc::ExecutionSession>(
          cantFail(llvm::orc::SelfExecutorProcessControl::Create()))),
      objectLayer(*execSession,
#ifdef LDC_JITRT_USE_JITLINK
                  cantFail(llvm::jitlink::InProcessMemoryManager::Create())
#else
          []() { return std::make_unique<llvm::SectionMemoryManager>(); }
#endif
                      ),
      listener_stream(nullptr),
      listenerlayer(*execSession, objectLayer,
                    [&](std::unique_ptr<llvm::MemoryBuffer> object)
                        -> llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>> {
                      if (nullptr != listener_stream) {
                        auto objFile = llvm::cantFail(
                            llvm::object::ObjectFile::createObjectFile(
                                object->getMemBufferRef()));
                        disassemble(*targetmachine, *objFile, *listener_stream);
                      }
                      return object;
                    }),
      compileLayer(*execSession, listenerlayer,
                   std::make_unique<llvm::orc::ConcurrentIRCompiler>(jtmb)),
      context(std::make_unique<llvm::LLVMContext>()),
      moduleHandle(execSession->createBareJITDylib("ldc_jit_module")),
      mainContext(isMainContext) {
#ifdef LDC_JITRT_USE_JITLINK
  objectLayer.addPlugin(std::make_unique<llvm::orc::EHFrameRegistrationPlugin>(
      *execSession,
      std::make_unique<llvm::jitlink::InProcessEHFrameRegistrar>()));
#endif
  moduleHandle.addGenerator(
      std::make_unique<LDCSymbolDefinitionGenerator>(this));
  moduleHandle.addGenerator(
      cantFail(llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
          dataLayout.getGlobalPrefix())));
}

DynamicCompilerContext::~DynamicCompilerContext() {
  reset();
  removeModule(moduleHandle);
  cantFail(execSession->endSession());
}

llvm::Error
DynamicCompilerContext::addModule(llvm::orc::ThreadSafeModule module) {
  assert(!!module);
  reset();

  // Add the set to the JIT with the resolver we created above
  auto result = compileLayer.add(moduleHandle, std::move(module));
  if (result) {
    cantFail(execSession->removeJITDylib(moduleHandle));
    return result;
  }
  compiled = true;
  return llvm::Error::success();
}

llvm::Optional<llvm::orc::ExecutorSymbolDef>
DynamicCompilerContext::findSymbol(const std::string &name) {
  auto it = symMap.find(name);
  if (symMap.end() != it) {
    return llvm::orc::ExecutorSymbolDef{
        llvm::orc::ExecutorAddr::fromPtr(it->second),
        llvm::JITSymbolFlags::Exported};
  }
  return {};
}

llvm::Expected<llvm::orc::ExecutorSymbolDef>
DynamicCompilerContext::lookup(const std::string &name) {
  return execSession->lookup({&moduleHandle}, name);
}

void DynamicCompilerContext::clearSymMap() { symMap.clear(); }

void DynamicCompilerContext::addSymbol(std::string &&name, void *value) {
  symMap.emplace(std::make_pair(std::move(name), value));
}

void DynamicCompilerContext::reset() {
  if (compiled) {
    cantFail(moduleHandle.clear());
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

void DynamicCompilerContext::removeModule(ModuleHandleT &handle) {
  cantFail(execSession->removeJITDylib(handle));
}
