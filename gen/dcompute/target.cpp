//===-- gen/dcompute/target.cpp -------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#if LDC_LLVM_SUPPORTED_TARGET_SPIRV || LDC_LLVM_SUPPORTED_TARGET_NVPTX

#include "dmd/dsymbol.h"
#include "dmd/errors.h"
#include "dmd/expression.h"
#include "dmd/module.h"
#include "dmd/nspace.h"
#include "dmd/scope.h"
#include "dmd/template.h"
#include "driver/linker.h"
#include "driver/toobj.h"
#include "driver/cl_options.h"
#include "gen/dcompute/druntime.h"
#include "gen/dcompute/target.h"
#include "gen/llvmhelpers.h"
#include "gen/pragma.h"
#include "gen/runtime.h"
#include <string>

class DcomputeTypeResetVisitor : public Visitor {

  void visit(Dsymbol *sym) override {
    // Needed to keep from base class throwing assertion
  }

  void visit(ScopeDsymbol *scope) override {
    if (!isError(scope) && scope->members) {
      for (auto sym : *scope->members)
        sym->accept(this);
    }
  }

  void visit(ExpStatement *stmt) override {
    if (auto e = stmt->exp) {
      e->accept(this);
    }
  }

  void visit(Expression *e) override {
    // Needed to keep from base class throwing assertion
  }
  
  void visit(DeclarationExp *e) override {
    if (auto vd = e->declaration->isVarDeclaration()) {
      vd->accept(this);
    }
  }

  void visit(VarDeclaration* vd) {
    Type *type = vd->type;

    bool is_dcompute_type = false;
    if(type->ty == Tpointer) {
      if(type->nextOf()->ty == Tstruct) {
        is_dcompute_type = isFromLDC_DCompute(static_cast<TypeStruct *>(type->nextOf())->sym);
      }
    } else if(type->ty == Tstruct) {
      is_dcompute_type = isFromLDC_DCompute(static_cast<TypeStruct *>(type)->sym);
    }

    if(is_dcompute_type && type->ctype) {
      delete type->ctype;
      type->ctype = nullptr;
    }
  }
  
  void visit(FuncDeclaration *decl) override {
    if(decl->parameters) {
      for(auto parameter : *decl->parameters) {
        parameter->accept(this);
      }
    }

    if(decl->fbody) {
      decl->fbody->accept(this);
    }
  }

  void visit(CompoundStatement *stmt) override {
    for (auto s : *stmt->statements) {
      if (s) {
        s->accept(this);
      }
    }
  }

  void visit(AttribDeclaration *decl) override {
    Dsymbols *d = decl->include(nullptr);

    if (d) {
      for (auto s : *d) {
        s->accept(this);
      }
    }
  }

  void visit(Nspace *ns) override {
    if (!isError(ns) && ns->members) {
      for (auto sym : *ns->members)
        sym->accept(this);
    }
  }
};

void Declaration_reset(Dsymbol *decl, IRState *irs) {
  DcomputeTypeResetVisitor v;
  decl->accept(&v);
}

void DComputeTarget::doCodeGen(Module *m) {
  // Reset any generated type info for dcompute types.
  // The ll types get generated when the host code gets
  // gen'd which means the address space info is not
  // properly set.
  for (unsigned k = 0; k < m->members->length; k++) {
    Dsymbol *dsym = (*m->members)[k];
    assert(dsym);
    Declaration_reset(dsym, _ir);
  }

  // process module members
  for (unsigned k = 0; k < m->members->length; k++) {
    Dsymbol *dsym = (*m->members)[k];
    assert(dsym);
    Declaration_codegen(dsym, _ir);
  }

  if (global.errors)
    fatal();
}

void DComputeTarget::emit(Module *m) {
  // Reset the global ABI to the target's ABI. Necessary because we have
  // multiple ABI we are trying to target. Also reset gIR. These are both
  // reused. MAJOR HACK.
  gABI = abi;
  gIR = _ir;
  gTargetMachine = targetMachine;
  doCodeGen(m);
}

void DComputeTarget::writeModule() {
  addMetadata();

  std::string filename;
  llvm::raw_string_ostream os(filename);
  os << opts::dcomputeFilePrefix << '_' << short_name << tversion << '_'
     << (global.params.is64bit ? 64 : 32) << '.' << binSuffix;

  const char *path =
      FileName::combine(global.params.objdir.ptr, os.str().c_str());

  ::writeModule(&_ir->module, path);

  delete _ir;
  _ir = nullptr;
}

#endif
