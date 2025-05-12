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
#include "llvm/ExecutionEngine/JITLink/EHFrameSupport.h"
#if LDC_LLVM_VER >= 2000 && defined(LDC_JITRT_USE_JITLINK)
#include "llvm/ExecutionEngine/Orc/EHFrameRegistrationPlugin.h"
#endif
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/ObjectTransformLayer.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"

namespace {

static llvm::SmallVector<std::string, 4> getHostAttrs() {
  llvm::SmallVector<std::string, 4> features;
  llvm::StringMap<bool> hostFeatures;
#if LDC_LLVM_VER >= 1901
  hostFeatures = llvm::sys::getHostCPUFeatures();
#else
  if (llvm::sys::getHostCPUFeatures(hostFeatures))
#endif
  {
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

static llvm::orc::JITTargetMachineBuilder createTargetMachine() {
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

DynamicCompilerContext::DynamicCompilerContext(
    llvm::orc::LLJITBuilderState state, llvm::Error error,
    std::unique_ptr<llvm::TargetMachine> targetmachine, bool isMainContext)
    : llvm::orc::LLJIT(state, error),
      context(std::make_unique<llvm::LLVMContext>()),
      // we need this targetmachine here because LLJIT ctor will free the
      // targetmachine field in LLJITBuilderState before we can even use it
      targetmachine(std::move(targetmachine)), listenerstream(nullptr),
      compiled(false), mainContext(isMainContext) {
  assert(!error);
  // setup the assembly code listener
  // we assume LLJIT's own ObjTransformLayer is empty (at least this is the case
  // for LLVM 12~20)
  this->ObjTransformLayer->setTransform(
      [&](std::unique_ptr<llvm::MemoryBuffer> object)
          -> llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>> {
        if (nullptr != listenerstream) {
          auto objFile =
              llvm::cantFail(llvm::object::ObjectFile::createObjectFile(
                  object->getMemBufferRef()));
          disassemble(*this->targetmachine, *objFile, *listenerstream);
        }
        return object;
      });
}

static llvm::orc::LLJITBuilder buildLLJITforLDC() {
  llvm::orc::LLJITBuilder builder{};
  builder.setJITTargetMachineBuilder(createTargetMachine())
      .setLinkProcessSymbolsByDefault(true)
  // we override the object linking layer if we are using LLVM JITLink.
  // For RuntimeDyld, we use LLJIT's default setup process
  // (which includes a lot of platform-related workarounds we need)
  // on LLVM 20+, LLJIT will auto-configure eh-frame plugin and
  // we avoid configuring the eh-frame plugin ourselves to avoid double registration
#if defined(LDC_JITRT_USE_JITLINK) && LDC_LLVM_VER < 2000
      .setObjectLinkingLayerCreator([&](llvm::orc::ExecutionSession &ES,
                                        const llvm::Triple &TT) {
        auto linker = std::make_unique<llvm::orc::ObjectLinkingLayer>(
            ES, cantFail(llvm::jitlink::InProcessMemoryManager::Create()));
        // explicitly register EH frame support (for exception handling)
        linker->addPlugin(
            std::make_unique<llvm::orc::EHFrameRegistrationPlugin>(
                ES,
                std::make_unique<llvm::jitlink::InProcessEHFrameRegistrar>()));
        return linker;
      })
#endif
      ;
  cantFail(builder.prepareForConstruction());
  return builder;
}

std::unique_ptr<DynamicCompilerContext>
DynamicCompilerContext::Create(bool isMainContext) {
  auto builder = buildLLJITforLDC();
  auto TM = cantFail(builder.JTMB->createTargetMachine());
  // std::make_unique is unusable here because it does not work when the
  // target class constructor is private
  return std::unique_ptr<DynamicCompilerContext>{
      new DynamicCompilerContext(std::move(builder), llvm::Error::success(),
                                 std::move(TM), isMainContext)};
}

llvm::Error
DynamicCompilerContext::addModule(llvm::orc::ThreadSafeModule module) {
  assert(!!module);
  reset();

  auto error = this->addIRModule(*this->Main, std::move(module));
  if (error) {
    return error;
  }
  compiled = true;
  return llvm::Error::success();
}

void DynamicCompilerContext::reset() {
  if (compiled) {
    // note that we don't remove the JD because that will destroy the lookup
    // order and link order we have setup
    cantFail(this->Main->clear());
    compiled = false;
  }
}

DynamicCompilerContext::~DynamicCompilerContext() { reset(); }


void DynamicCompilerContext::addSymbols(llvm::orc::SymbolMap &symbols) {
  // we define the static symbol in the process symbols JD to avoid symbol
  // conflicts in the Main JD (pre-fabricated LLJIT instance will still check
  // ProcessSymbols if the symbol it needs does not exist in the Main JD)
  cantFail(ProcessSymbols->define(llvm::orc::absoluteSymbols(symbols)));
}

void DynamicCompilerContext::addSymbol(std::string &&name, void *value) {
  llvm::orc::SymbolMap symbols{1};
  symbols[mangleAndIntern(name)] = {llvm::orc::ExecutorAddr::fromPtr(value),
                                    llvm::JITSymbolFlags::Exported};
  cantFail(ProcessSymbols->define(llvm::orc::absoluteSymbols(symbols)));
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
