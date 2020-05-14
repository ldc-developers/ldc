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

#if LDC_LLVM_VER < 600
namespace llvm {
using ToolOutputFile = tool_output_file;
}
#endif

namespace {

std::unique_ptr<llvm::ToolOutputFile>
createAndSetDiagnosticsOutputFile(IRState &irs, llvm::LLVMContext &ctx,
                                  llvm::StringRef filename) {
  std::unique_ptr<llvm::ToolOutputFile> diagnosticsOutputFile;

#if LDC_LLVM_VER >= 400
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
#if LDC_LLVM_VER >= 500
      ctx.setDiagnosticsHotnessRequested(true);
#else
      ctx.setDiagnosticHotnessRequested(true);
#endif
    }
#endif // LDC_LLVM_VER < 900
  }
#endif

  return diagnosticsOutputFile;
}

#if LDC_LLVM_VER < 500
/// Add the Linker Options module flag.
/// If the flag is already present, merge it with the new data.
void emitLinkerOptions(IRState &irs) {
  llvm::Module &M = irs.module;
  llvm::LLVMContext &ctx = irs.context();
  if (!M.getModuleFlag("Linker Options")) {
    M.addModuleFlag(llvm::Module::AppendUnique, "Linker Options",
                    llvm::MDNode::get(ctx, irs.linkerOptions));
  } else {
    // Merge the Linker Options with the pre-existing one
    // (this can happen when passing a .bc file on the commandline)

    auto *moduleFlags = M.getModuleFlagsMetadata();
    for (unsigned i = 0, e = moduleFlags->getNumOperands(); i < e; ++i) {
      auto *flag = moduleFlags->getOperand(i);
      if (flag->getNumOperands() < 3)
        continue;
      auto optionsMDString =
          llvm::dyn_cast_or_null<llvm::MDString>(flag->getOperand(1));
      if (!optionsMDString || optionsMDString->getString() != "Linker Options")
        continue;

      // If we reach here, we found the Linker Options flag.

      // Add the old Linker Options to our linkerOptions list.
      auto *oldLinkerOptions = llvm::cast<llvm::MDNode>(flag->getOperand(2));
      for (const auto &Option : oldLinkerOptions->operands()) {
        irs.linkerOptions.push_back(Option);
      }

      // Replace Linker Options with a newly created list.
      llvm::Metadata *Ops[3] = {
          llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
              llvm::Type::getInt32Ty(ctx), llvm::Module::AppendUnique)),
          llvm::MDString::get(ctx, "Linker Options"),
          llvm::MDNode::get(ctx, irs.linkerOptions)};
      moduleFlags->setOperand(i, llvm::MDNode::get(ctx, Ops));

      break;
    }
  }
}
#else
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
#endif

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

void setGlobalArrayAddressSpace(const char *Array, llvm::Module &M,
                                unsigned addressSpace) {
  if (addressSpace == 0) {
    return;
  }

  llvm::SmallVector<llvm::Constant *, 16> CurrentCtors;
  llvm::IRBuilder<> IRB(M.getContext());
  llvm::FunctionType *FnTy = llvm::FunctionType::get(IRB.getVoidTy(), false);
  llvm::StructType *EltTy = llvm::StructType::get(
      IRB.getInt32Ty(), llvm::PointerType::get(FnTy, addressSpace),
      IRB.getInt8PtrTy());

  if (llvm::GlobalVariable *GVCtor = M.getNamedGlobal(Array)) {
    if (llvm::Constant *Init = GVCtor->getInitializer()) {
      unsigned n = Init->getNumOperands();
      CurrentCtors.reserve(n + 1);
      for (unsigned i = 0; i != n; ++i)
        CurrentCtors.push_back(llvm::cast<llvm::Constant>(Init->getOperand(i)));
    }
    GVCtor->eraseFromParent();
  }
  
  if (CurrentCtors.size() <= 0) {
    return;
  }

  llvm::ArrayType *AT = llvm::ArrayType::get(EltTy, CurrentCtors.size());
  llvm::Constant *NewInit = llvm::ConstantArray::get(AT, CurrentCtors);

  (void)new llvm::GlobalVariable(M, NewInit->getType(), false,
                                 llvm::GlobalValue::AppendingLinkage, NewInit,
                                 Array);
}

} // anonymous namespace

namespace ldc {
CodeGenerator::CodeGenerator(llvm::LLVMContext &context, bool singleObj)
    : context_(context), moduleCount_(0), singleObj_(singleObj), ir_(nullptr) {
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

#if LDC_LLVM_VER >= 800
  // Set the proper address space for global arrays
  setGlobalArrayAddressSpace(
      "llvm.global_ctors", ir_->module,
      ir_->module.getDataLayout().getProgramAddressSpace());
  setGlobalArrayAddressSpace(
      "llvm.global_dtors", ir_->module,
      ir_->module.getDataLayout().getProgramAddressSpace());
#endif

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
}
