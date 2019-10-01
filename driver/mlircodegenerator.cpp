//===-- mlircodegenerator.cpp ---------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "driver/mlircodegenerator.h"

#include "dmd/compiler.h"
#include "dmd/errors.h"
#include "dmd/id.h"
#include "dmd/module.h"
#include "dmd/scope.h"
#include "driver/cl_options.h"
#include "driver/cl_options_instrumentation.h"
#include "driver/linker.h"
#include "driver/toobj.h"
#include "gen/dynamiccompile.h"
#include "gen/logger.h"
#include "gen/modules.h"
#include "gen/runtime.h"

//MLIR just support LLVM >= 1000
using std::make_unique;

//MLIR doesn't have any portability to implement createAndSetDiagnosticsOutputFile
//or

/*namespace {

void emitLinkerOptions(IRState &irs, llvm::Module &M, llvm::LLVMContext
 * &ctx) {
  auto *linkerOptionsMD = M.getOrInsertNamedMetadata("llvm.linker.options");

  // Add the new operands in front of the existing ones, such that linker
  // options of .bc files passed on the cmdline are put _after_ the compiled .d
  // file.

  // Temporarily store metadata nodes that are already present
  llvm::SmallVector<llvm::MDNode *, 5> oldMDNodes;
  for (auto *MD : linkerOptionsMD->operands())
    oldMDNodes.push_back(MD);

  // Clear the list and add the new metadata nodes.
  linkerOptionsMD->clearOperands();
  for (auto *MD : irs.LinkerMetadataArgs)
    linkerOptionsMD->addOperand(MD);

  // Re-add metadata nodes that were already present
  for (auto *MD : oldMDNodes)
    linkerOptionsMD->addOperand(MD);
}

 void emitLLVMUsedArray(IRState &irs) {
  if (irs.usedArray.empty()) {
    return;
  }

  auto *i8PtrType = llvm::Type::getInt8PtrTy(irs.context());

  // Convert all elements to i8* (the expected type for llvm.used)
  for (auto &elem : irs.usedArray) {
    elem = llvm::ConstantExpr::getBitCast(elem, i8PtrType);
  }

  auto *arrayType = llvm::ArrayType::get(i8PtrType, irs.usedArray.size());
  auto *llvmUsed = new llvm::GlobalVariable(
      irs.module, arrayType, false, llvm::GlobalValue::AppendingLinkage,
      llvm::ConstantArray::get(arrayType, irs.usedArray), "llvm.used");
  llvmUsed->setSection("llvm.metadata");
}
}*/

namespace ldc {
MLIRCodeGenerator::MLIRCodeGenerator(mlir::MLIRContext &context, bool singleObj)
    : context_(context), moduleCount_(0), singleObj_(singleObj), MLir_
    (nullptr) {
  // Set the context to discard value names when not generating textual IR.
  if (!global.params.output_mlll) {
    context_.setDiscardValueNames(true);
  }
}

MLIRCodeGenerator::~MLIRCodeGenerator() {
  /*if (singleObj_) { TODO: Make this possible in MLIR
    // For singleObj builds, the first object file name is the one for the first
    // source file (e.g., `b.o` for `ldc2 a.o b.d c.d`).
    const char *filename = global.params.objfiles[0];

    // If there are bitcode files passed on the cmdline, add them after all
    // other source files have been added to the (singleobj) module.
    insertBitcodeFiles(mlir_->module, mlir_->context(), global.params
    .bitcodeFiles);

    writeAndFreeLLModule(filename);
  }*/
}

void MLIRCodeGenerator::prepareLLModule(Module *m) {
  ++moduleCount_;

  if (singleObj_ && mlir_) {
    return;
  }

  assert(!mlir_);

  // See http://llvm.org/bugs/show_bug.cgi?id=11479 – just use the source file
  // name, as it should not collide with a symbol name used somewhere in the
  // module.
  ir_ = new IRState(m->srcfile.toChars(), context_);
  ir_->module.setTargetTriple(global.params.targetTriple->str());
  ir_->module.setDataLayout(*gDataLayout);

  // TODO: Make ldc::DIBuilder per-Module to be able to emit several CUs for
  // single-object compilations?
  ir_->DBuilder.EmitCompileUnit(m);

  IrDsymbol::resetAll();
}

void CodeGenerator::finishLLModule(Module *m) {
  if (singleObj_) {
    return;
  }

  // Add bitcode files passed on the cmdline to
  // the first module only, to avoid duplications.
  if (moduleCount_ == 1) {
    insertBitcodeFiles(ir_->module, ir_->context(), global.params.bitcodeFiles);
  }

  writeAndFreeLLModule(m->objfile.toChars());
}

void CodeGenerator::writeAndFreeLLModule(const char *filename) {
  ir_->objc.finalize();

  // Issue #1829: make sure all replaced global variables are replaced
  // everywhere.
  ir_->replaceGlobals();

  ir_->DBuilder.Finalize();
  generateBitcodeForDynamicCompile(ir_);

  emitLLVMUsedArray(*ir_);
  emitLinkerOptions(*ir_, ir_->module, ir_->context());

  // Emit ldc version as llvm.ident metadata.
  llvm::NamedMDNode *IdentMetadata =
      ir_->module.getOrInsertNamedMetadata("llvm.ident");
  std::string Version("ldc version ");
  Version.append(global.ldc_version.ptr, global.ldc_version.length);
  llvm::Metadata *IdentNode[] = {llvm::MDString::get(ir_->context(), Version)};
  IdentMetadata->addOperand(llvm::MDNode::get(ir_->context(), IdentNode));

  std::unique_ptr<llvm::ToolOutputFile> diagnosticsOutputFile =
      createAndSetDiagnosticsOutputFile(*ir_, context_, filename);

  writeModule(&ir_->module, filename);

  if (diagnosticsOutputFile)
    diagnosticsOutputFile->keep();

  delete ir_;
  ir_ = nullptr;
}

namespace {
/// Emits a declaration for the given symbol, which is assumed to be of type
/// i8*, and defines a second globally visible i8* that contains the address
/// of the first symbol.
void emitSymbolAddrGlobal(llvm::Module &lm, const char *symbolName,
                          const char *addrName) {
  llvm::Type *voidPtr =
      llvm::PointerType::get(llvm::Type::getInt8Ty(lm.getContext()), 0);
  auto targetSymbol = new llvm::GlobalVariable(
      lm, voidPtr, false, llvm::GlobalValue::ExternalWeakLinkage, nullptr,
      symbolName);
  new llvm::GlobalVariable(
      lm, voidPtr, false, llvm::GlobalValue::ExternalLinkage,
      llvm::ConstantExpr::getBitCast(targetSymbol, voidPtr), addrName);
}
}

void CodeGenerator::emit(Module *m) {
  bool const loggerWasEnabled = Logger::enabled();
  if (m->llvmForceLogging && !loggerWasEnabled) {
    Logger::enable();
  }

  IF_LOG Logger::println("CodeGenerator::emit(%s)", m->toPrettyChars());
  LOG_SCOPE;

  if (global.params.verbose_cg) {
    printf("codegen: %s (%s)\n", m->toPrettyChars(), m->srcfile.toChars());
  }

  if (global.errors) {
    Logger::println("Aborting because of errors");
    fatal();
  }

  prepareLLModule(m);

  codegenModule(ir_, m);
  if (m == rootHasMain) {
    codegenModule(ir_, entrypoint);

    if (global.params.targetTriple->getEnvironment() == llvm::Triple::Android) {
      // On Android, bracket TLS data with the symbols _tlsstart and _tlsend, as
      // done with dmd
      auto startSymbol = new llvm::GlobalVariable(
          ir_->module, llvm::Type::getInt32Ty(ir_->module.getContext()), false,
          llvm::GlobalValue::ExternalLinkage,
          llvm::ConstantInt::get(ir_->module.getContext(), APInt(32, 0)),
          "_tlsstart", &*(ir_->module.global_begin()));
      startSymbol->setSection(".tdata");

      auto endSymbol = new llvm::GlobalVariable(
          ir_->module, llvm::Type::getInt32Ty(ir_->module.getContext()), false,
          llvm::GlobalValue::ExternalLinkage,
          llvm::ConstantInt::get(ir_->module.getContext(), APInt(32, 0)),
          "_tlsend");
      endSymbol->setSection(".tcommon");
    } else if (global.params.targetTriple->isOSLinux()) {
      // On Linux, strongly define the excecutabe BSS bracketing symbols in
      // the main module for druntime use (see rt.sections_elf_shared).
      emitSymbolAddrGlobal(ir_->module, "__bss_start", "_d_execBssBegAddr");
      emitSymbolAddrGlobal(ir_->module, "_end", "_d_execBssEndAddr");
    }
  }

  finishLLModule(m);

  if (m->llvmForceLogging && !loggerWasEnabled) {
    Logger::disable();
  }
}
}
