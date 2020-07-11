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
#include "driver/linker.h"
#include "driver/toobj.h"
#include "gen/dynamiccompile.h"
#include "gen/logger.h"
#include "gen/modules.h"
#include "gen/runtime.h"
#if LDC_LLVM_VER >= 900
#include "llvm/IR/RemarkStreamer.h"
#endif
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/YAMLTraits.h"

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

#if LDC_LLVM_VER >= 900
    auto remarksFileOrError = llvm::setupOptimizationRemarks(
        ctx, diagnosticsFilename, "", "", withHotness);
    if (llvm::Error e = remarksFileOrError.takeError()) {
      irs.dmodule->error("Could not create file %s: %s",
                         diagnosticsFilename.c_str(),
                         llvm::toString(std::move(e)).c_str());
      fatal();
    }
    diagnosticsOutputFile = std::move(*remarksFileOrError);
#else
    std::error_code EC;
    diagnosticsOutputFile = llvm::make_unique<llvm::ToolOutputFile>(
        diagnosticsFilename, EC, llvm::sys::fs::F_None);
    if (EC) {
      irs.dmodule->error("Could not create file %s: %s",
                         diagnosticsFilename.c_str(), EC.message().c_str());
      fatal();
    }

    ctx.setDiagnosticsOutputFile(
        llvm::make_unique<llvm::yaml::Output>(diagnosticsOutputFile->os()));

    if (withHotness) {
      ctx.setDiagnosticsHotnessRequested(true);
    }
#endif // LDC_LLVM_VER < 900
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

void inlineAsmDiagnosticHandler(const llvm::SMDiagnostic &d, void *context,
                                unsigned locCookie) {
  if (d.getKind() == llvm::SourceMgr::DK_Error)
    ++global.errors;

  if (!locCookie) {
    d.print(nullptr, llvm::errs());
    return;
  }

  // replace the `<inline asm>` dummy filename by the LOC of the actual D
  // expression/statement (`myfile.d(123)`)
  const Loc &loc =
      static_cast<IRState *>(context)->getInlineAsmSrcLoc(locCookie);
  const char *filename = loc.toChars(/*showColumns*/ false);

  // keep on using llvm::SMDiagnostic::print() for nice, colorful output
  llvm::SMDiagnostic d2(*d.getSourceMgr(), d.getLoc(), filename, d.getLineNo(),
                        d.getColumnNo(), d.getKind(), d.getMessage(),
                        d.getLineContents(), d.getRanges(), d.getFixIts());
  d2.print(nullptr, llvm::errs());
}

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
  // Set the context to discard value names when not generating textual IR.
  if (!global.params.output_ll) {
    context_.setDiscardValueNames(true);
  }
}

CodeGenerator::~CodeGenerator() {
  if (singleObj_) {
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

  context_.setInlineAsmDiagnosticHandler(inlineAsmDiagnosticHandler, ir_);

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
    const auto llpath = replaceExtensionWith(global.mlir_ext, filename);
    Logger::println("Writting MLIR to %s\n", llpath.c_str());
    std::error_code errinfo;
    llvm::raw_fd_ostream aos(llpath, errinfo, llvm::sys::fs::F_None);

    if (aos.has_error()) {
      error(Loc(), "Cannot write MLIR file '%s': %s", llpath.c_str(),
            errinfo.message().c_str());
      fatal();
    }

    // module->print(aos);
  }
}

#endif
}
