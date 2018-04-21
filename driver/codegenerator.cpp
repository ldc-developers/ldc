//===-- codegenerator.cpp -------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "driver/codegenerator.h"

#include "id.h"
#include "mars.h"
#include "module.h"
#include "scope.h"
#include "driver/cl_options.h"
#include "driver/cl_options_instrumentation.h"
#include "driver/linker.h"
#include "driver/toobj.h"
#include "gen/logger.h"
#include "gen/modules.h"
#include "gen/runtime.h"
#include "gen/dynamiccompile.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/YAMLTraits.h"

/// The module with the frontend-generated C main() definition.
extern Module *entrypoint; // defined in dmd/mars.d

/// The module that contains the actual D main() (_Dmain) definition.
extern Module *rootHasMain; // defined in dmd/mars.d

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

    // If there is instrumentation data available, also output function hotness
    if (opts::isUsingPGOProfile()) {
#if LDC_LLVM_VER >= 500
      ctx.setDiagnosticsHotnessRequested(true);
#else
      ctx.setDiagnosticHotnessRequested(true);
#endif
    }
  }
#endif

  return diagnosticsOutputFile;
}

} // anonymous namespace

namespace {

#if LDC_LLVM_VER < 500
/// Add the Linker Options module flag.
/// If the flag is already present, merge it with the new data.
void emitLinkerOptions(IRState &irs, llvm::Module &M, llvm::LLVMContext &ctx) {
  if (!M.getModuleFlag("Linker Options")) {
    M.addModuleFlag(llvm::Module::AppendUnique, "Linker Options",
                    llvm::MDNode::get(ctx, irs.LinkerMetadataArgs));
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

      // Add the old Linker Options to our LinkerMetadataArgs list.
      auto *oldLinkerOptions = llvm::cast<llvm::MDNode>(flag->getOperand(2));
      for (const auto &Option : oldLinkerOptions->operands()) {
        irs.LinkerMetadataArgs.push_back(Option);
      }

      // Replace Linker Options with a newly created list.
      llvm::Metadata *Ops[3] = {
          llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
              llvm::Type::getInt32Ty(ctx), llvm::Module::AppendUnique)),
          llvm::MDString::get(ctx, "Linker Options"),
          llvm::MDNode::get(ctx, irs.LinkerMetadataArgs)};
      moduleFlags->setOperand(i, llvm::MDNode::get(ctx, Ops));

      break;
    }
  }
}
#else
/// Add the "llvm.linker.options" metadata.
/// If the metadata is already present, merge it with the new data.
void emitLinkerOptions(IRState &irs, llvm::Module &M, llvm::LLVMContext &ctx) {
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

}

namespace ldc {
CodeGenerator::CodeGenerator(llvm::LLVMContext &context, bool singleObj)
    : context_(context), moduleCount_(0), singleObj_(singleObj), ir_(nullptr) {
#if LDC_LLVM_VER >= 309
  // Set the context to discard value names when not generating textual IR.
  if (!global.params.output_ll) {
    context_.setDiscardValueNames(true);
  }
#endif
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
  ir_ = new IRState(m->srcfile->toChars(), context_);
  ir_->module.setTargetTriple(global.params.targetTriple->str());
#if LDC_LLVM_VER >= 308
  ir_->module.setDataLayout(*gDataLayout);
#else
  ir_->module.setDataLayout(gDataLayout->getStringRepresentation());
#endif

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

  writeAndFreeLLModule(m->objfile->name->str);
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
  Version.append(global.ldc_version);
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
    printf("codegen: %s (%s)\n", m->toPrettyChars(), m->srcfile->toChars());
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
