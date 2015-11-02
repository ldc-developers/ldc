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
        pragma(LDC_no_moduleinfo);\n\
        ";

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
}

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

  // Add the linker options metadata flag.
  ir_->module.addModuleFlag(
      llvm::Module::AppendUnique, "Linker Options",
      llvm::MDNode::get(ir_->context(), ir_->LinkerMetadataArgs));

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
    if (global.params.isLinux) {
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
