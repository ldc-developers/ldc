//===-- codegenerator.cpp -------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "driver/codegenerator.h"

#include "dmd/compiler.h"
#include "dmd/errors.h"
#include "dmd/globals.h"
#include "dmd/id.h"
#include "dmd/module.h"
#include "dmd/scope.h"
#include "driver/cl_options.h"
#include "driver/cl_options_instrumentation.h"
#include "driver/cl_options_sanitizers.h"
#include "driver/linker.h"
#include "driver/toobj.h"
#include "gen/dynamiccompile.h"
#include "gen/logger.h"
#include "gen/modules.h"
#include "gen/runtime.h"
#include "ir/irdsymbol.h"
#if LDC_LLVM_VER >= 1400
#include "llvm/IR/DiagnosticInfo.h"
#endif
#include "llvm/IR/LLVMRemarkStreamer.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/YAMLTraits.h"
#if LDC_MLIR_ENABLED
#if LDC_LLVM_VER >= 1200
#include "mlir/IR/BuiltinOps.h"
#else
#include "mlir/IR/Module.h"
#endif
#include "mlir/IR/MLIRContext.h"
#endif

namespace {

std::unique_ptr<llvm::ToolOutputFile>
createAndSetDiagnosticsOutputFile(IRState &irs, llvm::LLVMContext &ctx,
                                  llvm::StringRef filename) {
  std::unique_ptr<llvm::ToolOutputFile> diagnosticsOutputFile;

  // Set LLVM Diagnostics outputfile if requested
  if (opts::saveOptimizationRecord.getNumOccurrences() > 0) {
    llvm::SmallString<128> diagnosticsFilename;
    if (!opts::saveOptimizationRecord.empty()) {
      diagnosticsFilename = opts::saveOptimizationRecord.getValue();
    } else {
      diagnosticsFilename = filename;
      llvm::sys::path::replace_extension(diagnosticsFilename, "opt.yaml");
    }

    // If there is instrumentation data available, also output function hotness
    const bool withHotness = opts::isUsingPGOProfile();

    auto remarksFileOrError = llvm::setupLLVMOptimizationRemarks(
        ctx, diagnosticsFilename, "", "", withHotness);
    if (llvm::Error e = remarksFileOrError.takeError()) {
      irs.dmodule->error("Could not create file %s: %s",
                         diagnosticsFilename.c_str(),
                         llvm::toString(std::move(e)).c_str());
      fatal();
    }
    diagnosticsOutputFile = std::move(*remarksFileOrError);
  }

  return diagnosticsOutputFile;
}

void addLinkerMetadata(llvm::Module &M, const char *name,
                       llvm::ArrayRef<llvm::MDNode *> newOperands) {
  if (newOperands.empty())
    return;

  llvm::NamedMDNode *node = M.getOrInsertNamedMetadata(name);

  // Add the new operands in front of the existing ones, such that linker
  // options of .bc files passed on the cmdline are put _after_ the compiled .d
  // file.

  // Temporarily store metadata nodes that are already present
  llvm::SmallVector<llvm::MDNode *, 5> oldMDNodes;
  for (auto *MD : node->operands())
    oldMDNodes.push_back(MD);

  // Clear the list and add the new metadata nodes.
  node->clearOperands();
  for (auto *MD : newOperands)
    node->addOperand(MD);

  // Re-add metadata nodes that were already present
  for (auto *MD : oldMDNodes)
    node->addOperand(MD);
}

/// Add the "llvm.{linker.options,dependent-libraries}" metadata.
/// If the metadata is already present, merge it with the new data.
void emitLinkerOptions(IRState &irs) {
  llvm::Module &M = irs.module;
  addLinkerMetadata(M, "llvm.linker.options", irs.linkerOptions);
  addLinkerMetadata(M, "llvm.dependent-libraries", irs.linkerDependentLibs);
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

bool inlineAsmDiagnostic(IRState *irs, const llvm::SMDiagnostic &d,
                         unsigned locCookie) {
  if (!locCookie) {
    d.print(nullptr, llvm::errs());
    return true;
  }

  // replace the `<inline asm>` dummy filename by the LOC of the actual D
  // expression/statement (`myfile.d(123)`)
  const Loc &loc = irs->getInlineAsmSrcLoc(locCookie);
  const char *filename = loc.toChars(/*showColumns*/ false);

  // keep on using llvm::SMDiagnostic::print() for nice, colorful output
  llvm::SMDiagnostic d2(*d.getSourceMgr(), d.getLoc(), filename, d.getLineNo(),
                        d.getColumnNo(), d.getKind(), d.getMessage(),
                        d.getLineContents(), d.getRanges(), d.getFixIts());
  d2.print(nullptr, llvm::errs());
  return true;
}

#if LDC_LLVM_VER < 1300
void inlineAsmDiagnosticHandler(const llvm::SMDiagnostic &d, void *context,
                                unsigned locCookie) {
  if (d.getKind() == llvm::SourceMgr::DK_Error) {
    ++global.errors;
  } else if (global.params.warnings == DIAGNOSTICerror &&
             d.getKind() == llvm::SourceMgr::DK_Warning) {
    ++global.warnings;
  }

  inlineAsmDiagnostic(static_cast<IRState *>(context), d, locCookie);
}
#else
struct InlineAsmDiagnosticHandler : public llvm::DiagnosticHandler {
  IRState *irs;
  InlineAsmDiagnosticHandler(IRState *irs) : irs(irs) {}

    // return false to defer to LLVMContext::diagnose()
  bool handleDiagnostics(const llvm::DiagnosticInfo &DI) override {
    if (DI.getKind() == llvm::SourceMgr::DK_Error ||
        DI.getSeverity() == llvm::DS_Error) {
      ++global.errors;
    } else if (global.params.warnings == DIAGNOSTICerror &&
               (DI.getKind() == llvm::SourceMgr::DK_Warning ||
                DI.getSeverity() == llvm::DS_Warning)) {
      ++global.warnings;
    }

    if (DI.getKind() != llvm::DK_SrcMgr)
        return false;

    const auto &DISM = llvm::cast<llvm::DiagnosticInfoSrcMgr>(DI);

    return inlineAsmDiagnostic(irs, DISM.getSMDiag(), DISM.getLocCookie());
  }
};
#endif

} // anonymous namespace

namespace ldc {
CodeGenerator::CodeGenerator(llvm::LLVMContext &context,
#if LDC_MLIR_ENABLED
                             mlir::MLIRContext &mlirContext,
#endif
                             bool singleObj)
    : context_(context),
#if LDC_MLIR_ENABLED
      mlirContext_(mlirContext),
#endif
      moduleCount_(0), singleObj_(singleObj), ir_(nullptr) {
  // Set the context to discard value names when not generating textual IR and
  // when ASan or MSan are not enabled.
  if (!global.params.output_ll && !opts::fNoDiscardValueNames &&
      !opts::isSanitizerEnabled(opts::AddressSanitizer |
                                opts::MemorySanitizer)) {
    context_.setDiscardValueNames(true);
  }
}

CodeGenerator::~CodeGenerator() {
  if (singleObj_ && moduleCount_ > 0) {
    // For singleObj builds, the first object file name is the one for the first
    // source file (e.g., `b.o` for `ldc2 a.o b.d c.d`).
    const char *filename = global.params.objfiles[0];

    // If there are bitcode files passed on the cmdline, add them after all
    // other source files have been added to the (singleobj) module.
    insertBitcodeFiles(ir_->module, ir_->context(), global.params.bitcodeFiles);

    writeAndFreeLLModule(filename);
  }
}

void CodeGenerator::prepareLLModule(Module *m) {
  ++moduleCount_;

  if (singleObj_ && ir_) {
    return;
  }

  assert(!ir_);

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

  ir_->DBuilder.Finalize();
  generateBitcodeForDynamicCompile(ir_);

  emitLLVMUsedArray(*ir_);
  emitLinkerOptions(*ir_);

  // Issue #1829: make sure all replaced global variables are replaced
  // everywhere.
  ir_->replaceGlobals();

  // Emit ldc version as llvm.ident metadata.
  llvm::NamedMDNode *IdentMetadata =
      ir_->module.getOrInsertNamedMetadata("llvm.ident");
  std::string Version("ldc version ");
  Version.append(global.ldc_version.ptr, global.ldc_version.length);
  llvm::Metadata *IdentNode[] = {llvm::MDString::get(ir_->context(), Version)};
  IdentMetadata->addOperand(llvm::MDNode::get(ir_->context(), IdentNode));

#if LDC_LLVM_VER < 1300
  context_.setInlineAsmDiagnosticHandler(inlineAsmDiagnosticHandler, ir_);
#else
  context_.setDiagnosticHandler(
          std::make_unique<InlineAsmDiagnosticHandler>(ir_));
#endif

  std::unique_ptr<llvm::ToolOutputFile> diagnosticsOutputFile =
      createAndSetDiagnosticsOutputFile(*ir_, context_, filename);

  writeModule(&ir_->module, filename);

  if (diagnosticsOutputFile)
    diagnosticsOutputFile->keep();

  delete ir_;
  ir_ = nullptr;
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

  finishLLModule(m);

  if (m->llvmForceLogging && !loggerWasEnabled) {
    Logger::disable();
  }
}

#if LDC_MLIR_ENABLED
void CodeGenerator::emitMLIR(Module *m) {
  bool const loggerWasEnabled = Logger::enabled();
  if (m->llvmForceLogging && !loggerWasEnabled) {
    Logger::enable();
  }

  IF_LOG Logger::println("CodeGenerator::emitMLIR(%s)", m->toPrettyChars());
  LOG_SCOPE;

  if (global.params.verbose_cg) {
    printf("codegen: %s (%s)\n", m->toPrettyChars(), m->srcfile.toChars());
  }

  if (global.errors) {
    Logger::println("Aborting because of errors");
    fatal();
  }

  mlir::OwningModuleRef module;
  /*module = mlirGen(mlirContext, m, irs);
  if(!module){
    IF_LOG Logger::println("Error generating MLIR:'%s'", llpath.c_str());
    fatal();
  }*/

  writeMLIRModule(&module, m->objfile.toChars());

  if (m->llvmForceLogging && !loggerWasEnabled) {
    Logger::disable();
  }
}

void CodeGenerator::writeMLIRModule(mlir::OwningModuleRef *module,
                                    const char *filename) {
  // Write MLIR
  if (global.params.output_mlir) {
    const auto llpath = replaceExtensionWith(mlir_ext, filename);
    Logger::println("Writting MLIR to %s\n", llpath.c_str());
    std::error_code errinfo;
    llvm::ToolOutputFile aos(llpath, errinfo, llvm::sys::fs::OF_None);

    if (aos.os().has_error()) {
      error(Loc(), "Cannot write MLIR file '%s': %s", llpath.c_str(),
            errinfo.message().c_str());
      fatal();
    }

    // module->print(aos);

    // Terminate upon errors during the LLVM passes.
    if (global.errors || global.warnings) {
      Logger::println(
          "Aborting because of errors/warnings during bitcode LLVM passes");
      fatal();
    }

    aos.keep();
  }
}

#endif
}
