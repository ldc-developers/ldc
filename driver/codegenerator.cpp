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
#include "driver/toobj.h"
#include "gen/logger.h"
#include "gen/runtime.h"

void codegenModule(IRState *irs, Module *m, bool emitFullModuleInfo);

/// The module with the frontend-generated C main() definition.
extern Module *g_entrypointModule;

/// The module that contains the actual D main() (_Dmain) definition.
extern Module *g_dMainModule;

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
          oname, global.params.targetTriple->isOSWindows() ? global.obj_ext_alt
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
  ir_->module.setTargetTriple(global.params.targetTriple->str());
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

    // On Android, bracket TLS data with the symbols _tlsstart and _tlsend, as
    // done with dmd
    if (global.params.targetTriple->getEnvironment() == llvm::Triple::Android) {
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
