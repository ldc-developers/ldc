//===-- jit_context.h - jit support -----------------------------*- C++ -*-===//
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

#ifndef JIT_CONTEXT_H
#define JIT_CONTEXT_H

#include <map>
#include <memory>
#include <utility>

#include <llvm/ExecutionEngine/JITSymbol.h>
#include <llvm/ExecutionEngine/Orc/CompileUtils.h>
#include <llvm/ExecutionEngine/Orc/IRCompileLayer.h>
#include <llvm/ExecutionEngine/Orc/LambdaResolver.h>
#include <llvm/ExecutionEngine/Orc/ObjectTransformLayer.h>
#include <llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/Support/ManagedStatic.h>

#if LDC_LLVM_VER >= 700
#include <llvm/ExecutionEngine/Orc/Legacy.h>
#endif

#include "disassembler.h"

namespace llvm {
class raw_ostream;
class TargetMachine;
} // namespace llvm

using SymMap = std::map<std::string, void *>;

class JITContext final {
private:
  struct ModuleListener {
    llvm::TargetMachine &targetmachine;
    llvm::raw_ostream *stream = nullptr;

    ModuleListener(llvm::TargetMachine &tm) : targetmachine(tm) {}

    template <typename T> auto operator()(T &&object) -> T {
      if (nullptr != stream) {
#if LDC_LLVM_VER >= 700
        auto objFile =
            llvm::cantFail(llvm::object::ObjectFile::createObjectFile(
                object->getMemBufferRef()));
        disassemble(targetmachine, *objFile, *stream);
#else
        disassemble(targetmachine, *object->getBinary(), *stream);
#endif
      }
      return std::move(object);
    }
  };
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
  std::shared_ptr<llvm::orc::SymbolStringPool> stringPool;
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
    JITContext &owner;
    ListenerCleaner(JITContext &o, llvm::raw_ostream *stream) : owner(o) {
      owner.listenerlayer.getTransform().stream = stream;
    }
    ~ListenerCleaner() { owner.listenerlayer.getTransform().stream = nullptr; }
  };

public:
  JITContext();
  ~JITContext();

  llvm::TargetMachine &getTargetMachine() { return *targetmachine; }
  const llvm::DataLayout &getDataLayout() const { return dataLayout; }

  bool addModule(std::unique_ptr<llvm::Module> module,
                 llvm::raw_ostream *asmListener);

  llvm::JITSymbol findSymbol(const std::string &name);

  llvm::LLVMContext &getContext() { return context; }

  SymMap &getSymMap() { return symMap; }

  void reset();

private:
  void removeModule(const ModuleHandleT &handle);

#if LDC_LLVM_VER >= 700
  std::shared_ptr<llvm::orc::SymbolResolver> createResolver();
#else
  std::shared_ptr<llvm::JITSymbolResolver> createResolver();
#endif
};

#endif // JIT_CONTEXT_H
