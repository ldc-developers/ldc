#ifndef COMPILE_LAYER_HPP
#define COMPILE_LAYER_HPP

#include <llvm/ExecutionEngine/Orc/CompileUtils.h>
#include <llvm/ExecutionEngine/Orc/IRCompileLayer.h>

template<typename BaseLayerT>
class CompilerLayer {
public:

  using Compiler = llvm::orc::SimpleCompiler;
  using ModuleHandleT = typename BaseLayerT::ObjHandleT;

  CompilerLayer(BaseLayerT &BaseLayer, llvm::TargetMachine& TM):
    baseLayer(BaseLayer),
    compiler(BaseLayer, Compiler(TM)) {}

  llvm::Expected<ModuleHandleT> addModule(std::shared_ptr<llvm::Module> M,
                                          std::shared_ptr<llvm::JITSymbolResolver> Resolver) {
    return compiler.addModule(std::move(M), std::move(Resolver));
  }

  llvm::Expected<ModuleHandleT> addObject(
      std::unique_ptr<llvm::MemoryBuffer> ObjBuffer,
      std::shared_ptr<llvm::JITSymbolResolver> Resolver) {
    auto Obj =
        llvm::object::ObjectFile::createObjectFile(ObjBuffer->getMemBufferRef());
    if (!Obj) {
      return Obj.takeError();
    }
    using CompileResult = llvm::object::OwningBinary<llvm::object::ObjectFile>;
    return baseLayer.addObject(
          std::make_shared<CompileResult>(std::move(*Obj), std::move(ObjBuffer)),
          std::move(Resolver));
  }

  llvm::Error removeModule(ModuleHandleT H) {
    return baseLayer.removeObject(H);
  }

  llvm::JITSymbol findSymbol(const std::string &Name,
                             bool ExportedSymbolsOnly) {
    return baseLayer.findSymbol(Name, ExportedSymbolsOnly);
  }
private:
  BaseLayerT &baseLayer;
  llvm::orc::IRCompileLayer<BaseLayerT, Compiler> compiler;
};

#endif // COMPILE_LAYER_HPP
