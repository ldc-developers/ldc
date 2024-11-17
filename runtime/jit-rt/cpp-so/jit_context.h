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

#include "llvm/ADT/MapVector.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/IR/LLVMContext.h"

#ifdef LDC_JITRT_USE_JITLINK
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#else
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#endif

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

class DynamicCompilerContext final : public llvm::orc::LLJIT {
private:
#ifdef LDC_JITRT_USE_JITLINK
  using ObjectLayerT = llvm::orc::ObjectLinkingLayer;
#else
  using ObjectLayerT = llvm::orc::RTDyldObjectLinkingLayer;
#endif
  using CompileLayerT = llvm::orc::IRCompileLayer;
  llvm::orc::ThreadSafeContext context;
  std::unique_ptr<llvm::TargetMachine> targetmachine;
  llvm::raw_ostream *listenerstream;
  struct BindDesc final {
    void *originalFunc;
    void *exampleFunc;
    using ParamsVec = llvm::SmallVector<ParamSlice, 5>;
    ParamsVec params;
  };
  llvm::MapVector<void *, BindDesc> bindInstances;
  bool compiled;
  bool mainContext;

  // internal constructor
  DynamicCompilerContext(llvm::orc::LLJITBuilderState S, llvm::Error Err,
                         std::unique_ptr<llvm::TargetMachine> TM,
                         bool isMainContext);

public:
  struct ListenerCleaner final {
    DynamicCompilerContext &owner;
    ListenerCleaner(DynamicCompilerContext &o, llvm::raw_ostream *stream)
        : owner(o) {
      owner.listenerstream = stream;
    }
    ~ListenerCleaner() { owner.listenerstream = nullptr; }
  };
  static std::unique_ptr<DynamicCompilerContext> Create(bool isMainContext);
  ~DynamicCompilerContext();
  llvm::TargetMachine *getTargetMachine() const { return targetmachine.get(); }
  const llvm::DataLayout &getDataLayout() const { return DL; }
  llvm::Error addModule(llvm::orc::ThreadSafeModule module);
  void reset();
  void registerBind(void *handle, void *originalFunc, void *exampleFunc,
                    const llvm::ArrayRef<ParamSlice> &params);

  void unregisterBind(void *handle);

  bool hasBindFunction(const void *handle) const;

  const llvm::MapVector<void *, BindDesc> &getBindInstances() const {
    return bindInstances;
  }

  void addSymbol(std::string &&name, void *value);
  void addSymbols(llvm::orc::SymbolMap &symbols);
  void clearSymMap() { cantFail(ProcessSymbols->clear()); }
  llvm::orc::ThreadSafeContext getThreadSafeContext() const { return context; }
  llvm::LLVMContext *getContext() { return context.getContext(); }

  bool isMainContext() const { return mainContext; }

  std::unique_ptr<ListenerCleaner>
  addScopedListener(llvm::raw_ostream *stream) {
    return std::make_unique<ListenerCleaner>(*this, stream);
  }
};
