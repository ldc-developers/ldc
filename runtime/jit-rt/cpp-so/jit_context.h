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

#pragma once

#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
#include <map>
#include <memory>
#include <utility>

#include "llvm/ADT/MapVector.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/ObjectTransformLayer.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/ManagedStatic.h"


#if LDC_LLVM_VER < 1700
#include "llvm/ADT/Optional.h"
#else
#include <optional>
namespace llvm {
template <typename T> using Optional = std::optional<T>;
}
#endif

#include "context.h"
#include "disassembler.h"

namespace llvm {
class raw_ostream;
class TargetMachine;
} // namespace llvm

using SymMap = std::map<std::string, void *>;
class LDCSymbolDefinitionGenerator;

class DynamicCompilerContext final {
  friend class LDCSymbolDefinitionGenerator;
private:
  llvm::orc::JITTargetMachineBuilder jtmb;
  std::unique_ptr<llvm::TargetMachine> targetmachine;
  const llvm::DataLayout dataLayout;
  using ObjectLayerT = llvm::orc::RTDyldObjectLinkingLayer;
  using ListenerLayerT =
      llvm::orc::ObjectTransformLayer;
  using CompileLayerT =
      llvm::orc::IRCompileLayer;
  using ModuleHandleT = llvm::orc::JITDylib;
  std::unique_ptr<llvm::orc::ExecutionSession> execSession;
  ObjectLayerT objectLayer;
  llvm::raw_ostream *listener_stream;
  ListenerLayerT listenerlayer;
  CompileLayerT compileLayer;
  llvm::orc::ThreadSafeContext context;
  bool compiled = false;
  ModuleHandleT& moduleHandle;
  SymMap symMap;

  struct BindDesc final {
    void *originalFunc;
    void *exampleFunc;
    using ParamsVec = llvm::SmallVector<ParamSlice, 5>;
    ParamsVec params;
  };
  llvm::MapVector<void *, BindDesc> bindInstances;
  const bool mainContext = false;

  llvm::Optional<llvm::orc::ExecutorSymbolDef> findSymbol(const std::string &name);
public:
  struct ListenerCleaner final {
    DynamicCompilerContext &owner;
    ListenerCleaner(DynamicCompilerContext &o, llvm::raw_ostream *stream);
    ~ListenerCleaner();
  };
  explicit DynamicCompilerContext(bool isMainContext);
  ~DynamicCompilerContext();

  llvm::TargetMachine *getTargetMachine() { return targetmachine.get(); }
  const llvm::DataLayout &getDataLayout() const { return dataLayout; }

  llvm::Error addModule(llvm::orc::ThreadSafeModule module);

  llvm::Expected<llvm::orc::ExecutorSymbolDef> lookup(const std::string &name);

  llvm::LLVMContext *getContext() { return context.getContext(); }

  llvm::orc::ThreadSafeContext getThreadSafeContext() const { return context; }

  void clearSymMap();

  void addSymbol(std::string &&name, void *value);

  void reset();

  void registerBind(void *handle, void *originalFunc, void *exampleFunc,
                    const llvm::ArrayRef<ParamSlice> &params);

  void unregisterBind(void *handle);

  bool hasBindFunction(const void *handle) const;

  const llvm::MapVector<void *, BindDesc> &getBindInstances() const {
    return bindInstances;
  }

  bool isMainContext() const;

  std::unique_ptr<ListenerCleaner> addScopedListener(llvm::raw_ostream *stream) {
    return std::make_unique<ListenerCleaner>(*this, stream);
  }

private:
  void removeModule(ModuleHandleT &handle);

  std::shared_ptr<llvm::JITSymbolResolver> createResolver();
};
