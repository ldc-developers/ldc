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
#include "parse.h"
#include "scope.h"
#include "driver/toobj.h"
#include "gen/logger.h"
#include "gen/runtime.h"

void codegenModule(IRState *irs, Module *m, bool emitFullModuleInfo);

namespace {
Module *g_entrypointModule = nullptr;
Module *g_dMainModule = nullptr;
}

/// Callback to generate a C main() function, invoked by the frontend.
void genCmain(Scope *sc) {
  if (g_entrypointModule) {
    return;
  }

  /* The D code to be generated is provided as D source code in the form of a
   * string.
   * Note that Solaris, for unknown reasons, requires both a main() and an
   * _main()
   */
  static utf8_t code[] = "extern(C) {\n\
    int _d_run_main(int argc, char **argv, void* mainFunc);\n\
    int _Dmain(char[][] args);\n\
    int main(int argc, char **argv) { return _d_run_main(argc, argv, &_Dmain); }\n\
    version (Solaris) int _main(int argc, char** argv) { return main(argc, argv); }\n\
    }\n\
    pragma(LDC_no_moduleinfo);\n";

  Identifier *id = Id::entrypoint;
  auto m = new Module("__entrypoint.d", id, 0, 0);

  Parser p(m, code, sizeof(code) / sizeof(code[0]), 0);
  p.scanloc = Loc();
  p.nextToken();
  m->members = p.parseModule();
  assert(p.token.value == TOKeof);

  char v = global.params.verbose;
  global.params.verbose = 0;
  m->importedFrom = m;
  m->importAll(nullptr);
  m->semantic();
  m->semantic2();
  m->semantic3();
  global.params.verbose = v;

  g_entrypointModule = m;
  g_dMainModule = sc->module;
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

#if LDC_LLVM_VER < 500
/// Add the Linker Options module flag.
/// If the flag is already present, merge it with the new data.
void emitLinkerOptions(IRState &irs, llvm::Module &M, llvm::LLVMContext &ctx) {
#if LDC_LLVM_VER == 305
  M.addModuleFlag(
      llvm::Module::AppendUnique, "Linker Options",
      llvm::MDNode::get(ctx, irs.LinkerMetadataArgs));
#else
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
#endif
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

} // anonymous namespace

namespace ldc {
CodeGenerator::CodeGenerator(llvm::LLVMContext &context, bool singleObj)
    : context_(context), moduleCount_(0), singleObj_(singleObj), ir_(nullptr),
      firstModuleObjfileName_(nullptr) {
  if (!ClassDeclaration::object) {
    error(Loc(), "declaration for class Object not found; druntime not "
                 "configured properly");
    fatal();
  }
}

CodeGenerator::~CodeGenerator() {
  if (singleObj_) {
    const char *oname;
    const char *filename;
    if ((oname = global.params.exefile) || (oname = global.params.objname)) {
      filename = FileName::forceExt(
          oname, global.params.targetTriple.isOSWindows() ? global.obj_ext_alt
                                                          : global.obj_ext);
      if (global.params.objdir) {
        filename =
            FileName::combine(global.params.objdir, FileName::name(filename));
      }
    } else {
      filename = firstModuleObjfileName_;
    }

    writeAndFreeLLModule(filename);
  }
}

void CodeGenerator::prepareLLModule(Module *m) {
  if (!firstModuleObjfileName_) {
    firstModuleObjfileName_ = m->objfile->name->str;
  }
  ++moduleCount_;

  if (singleObj_ && ir_) {
    return;
  }

  assert(!ir_);

  // See http://llvm.org/bugs/show_bug.cgi?id=11479 – just use the source file
  // name, as it should not collide with a symbol name used somewhere in the
  // module.
  ir_ = new IRState(m->srcfile->toChars(), context_);
  ir_->module.setTargetTriple(global.params.targetTriple.str());
#if LDC_LLVM_VER >= 308
  ir_->module.setDataLayout(*gDataLayout);
#else
  ir_->module.setDataLayout(gDataLayout->getStringRepresentation());
#endif

  // TODO: Make ldc::DIBuilder per-Module to be able to emit several CUs for
  // singleObj compilations?
  ir_->DBuilder.EmitCompileUnit(m);

  IrDsymbol::resetAll();
}

void CodeGenerator::finishLLModule(Module *m) {
  if (singleObj_) {
    return;
  }

  m->deleteObjFile();
  writeAndFreeLLModule(m->objfile->name->str);
}

void CodeGenerator::writeAndFreeLLModule(const char *filename) {
  ir_->DBuilder.Finalize();

  emitLinkerOptions(*ir_, ir_->module, ir_->context());

  // Emit ldc version as llvm.ident metadata.
  llvm::NamedMDNode *IdentMetadata =
      ir_->module.getOrInsertNamedMetadata("llvm.ident");
  std::string Version("ldc version ");
  Version.append(global.ldc_version);
#if LDC_LLVM_VER >= 306
  llvm::Metadata *IdentNode[] =
#else
  llvm::Value *IdentNode[] =
#endif
      {llvm::MDString::get(ir_->context(), Version)};
  IdentMetadata->addOperand(llvm::MDNode::get(ir_->context(), IdentNode));

  writeModule(&ir_->module, filename);
  global.params.objfiles->push(const_cast<char *>(filename));
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
    printf("codegen: %s (%s)\n", m->toPrettyChars(), m->srcfile->toChars());
  }

  if (global.errors) {
    Logger::println("Aborting because of errors");
    fatal();
  }

  prepareLLModule(m);

  // If we are compiling to a single object file then only the first module
  // needs to generate a call to _d_dso_registry(). All other modules only add
  // a module reference.
  // FIXME Find better name.
  const bool emitFullModuleInfo =
      !singleObj_ || (singleObj_ && moduleCount_ == 1);
  codegenModule(ir_, m, emitFullModuleInfo);
  if (m == g_dMainModule) {
    codegenModule(ir_, g_entrypointModule, emitFullModuleInfo);

    // On Linux, strongly define the excecutabe BSS bracketing symbols in
    // the main module for druntime use (see rt.sections_linux).
    if (global.params.targetTriple.isOSLinux()) {
      emitSymbolAddrGlobal(ir_->module, "__bss_start", "_d_execBssBegAddr");
      emitSymbolAddrGlobal(ir_->module, "_end", "_d_execBssEndAddr");
    }

    // On Android, bracket TLS data with the symbols _tlsstart and _tlsend, as
    // done with dmd
    if (global.params.targetTriple.getEnvironment() == llvm::Triple::Android) {
      auto startSymbol = new llvm::GlobalVariable(
          ir_->module, llvm::Type::getInt32Ty(ir_->module.getContext()), false,
          llvm::GlobalValue::ExternalLinkage,
          llvm::ConstantInt::get(ir_->module.getContext(), APInt(32,0)),
          "_tlsstart", &*(ir_->module.global_begin()));
      startSymbol->setSection(".tdata");

      auto endSymbol = new llvm::GlobalVariable(
          ir_->module, llvm::Type::getInt32Ty(ir_->module.getContext()), false,
          llvm::GlobalValue::ExternalLinkage,
          llvm::ConstantInt::get(ir_->module.getContext(), APInt(32,0)),
          "_tlsend");
      endSymbol->setSection(".tcommon");
    }
  }

  finishLLModule(m);

  if (m->llvmForceLogging && !loggerWasEnabled) {
    Logger::disable();
  }
}
}
