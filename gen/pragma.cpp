//===-- pragma.cpp --------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "pragma.h"
#include "attrib.h"
#include "declaration.h"
#include "expression.h"
#include "id.h"
#include "module.h"
#include "scope.h"
#include "template.h"
#include "gen/inlineir.h"
#include "gen/llvmhelpers.h"
#include "llvm/Support/CommandLine.h"

static bool parseStringExp(Expression *e, const char *&res) {
  e = e->optimize(WANTvalue);
  if (e->op != TOKstring) {
    return false;
  }
  auto s = static_cast<StringExp *>(e);
  res = s->toStringz();
  return true;
}

static bool parseIntExp(Expression *e, dinteger_t &res) {
  IntegerExp *i = nullptr;

  e = e->optimize(WANTvalue);
  if (e->op == TOKint64 && (i = static_cast<IntegerExp *>(e))) {
    res = i->getInteger();
    return true;
  }
  return false;
}

static bool parseBoolExp(Expression *e, bool &res) {
  e = e->optimize(WANTvalue);
  if (e->op == TOKint64 && e->type->equals(Type::tbool)) {
    IntegerExp *i = static_cast<IntegerExp *>(e);
    res = i->isBool(true);
    return true;
  }
  return false;
}

LDCPragma DtoGetPragma(Scope *sc, PragmaDeclaration *decl,
                       const char *&arg1str) {
  Identifier *ident = decl->ident;
  Expressions *args = decl->args;
  Expression *expr =
      (args && args->dim > 0) ? expressionSemantic((*args)[0], sc) : nullptr;

  // pragma(LDC_intrinsic, "string") { funcdecl(s) }
  if (ident == Id::LDC_intrinsic) {
    if (!args || args->dim != 1 || !parseStringExp(expr, arg1str)) {
      error(Loc(), "requires exactly 1 string literal parameter");
      fatal();
    }

    // Recognize LDC-specific pragmas.
    struct LdcIntrinsic {
      std::string name;
      LDCPragma pragma;
    };
    static LdcIntrinsic ldcIntrinsic[] = {
        {"bitop.bt", LLVMbitop_bt},   {"bitop.btc", LLVMbitop_btc},
        {"bitop.btr", LLVMbitop_btr}, {"bitop.bts", LLVMbitop_bts},
        {"bitop.vld", LLVMbitop_vld}, {"bitop.vst", LLVMbitop_vst},
    };

    static std::string prefix = "ldc.";
    size_t arg1str_length = strlen(arg1str);
    if (arg1str_length > prefix.length() &&
        std::equal(prefix.begin(), prefix.end(), arg1str)) {
      // Got ldc prefix, binary search through ldcIntrinsic.
      std::string name(arg1str + prefix.length());
      size_t i = 0, j = sizeof(ldcIntrinsic) / sizeof(ldcIntrinsic[0]);
      do {
        size_t k = (i + j) / 2;
        int cmp = name.compare(ldcIntrinsic[k].name);
        if (!cmp) {
          return ldcIntrinsic[k].pragma;
        }
        if (cmp < 0) {
          j = k;
        } else {
          i = k + 1;
        }
      } while (i != j);
    }

    return LLVMintrinsic;
  }

  // pragma(LDC_global_crt_ctor [, priority]) { funcdecl(s) }
  if (ident == Id::LDC_global_crt_ctor || ident == Id::LDC_global_crt_dtor) {
    dinteger_t priority;
    if (args) {
      if (args->dim != 1 || !parseIntExp(expr, priority)) {
        error(Loc(), "requires at most 1 integer literal parameter");
        fatal();
      }
      if (priority > 65535) {
        error(Loc(), "priority may not be greater than 65535");
        priority = 65535;
      }
    } else {
      priority = 65535;
    }
    char buf[8];
    sprintf(buf, "%llu", static_cast<unsigned long long>(priority));
    arg1str = strdup(buf);
    return ident == Id::LDC_global_crt_ctor ? LLVMglobal_crt_ctor
                                            : LLVMglobal_crt_dtor;
  }

  // pragma(LDC_no_typeinfo) { typedecl(s) }
  if (ident == Id::LDC_no_typeinfo) {
    if (args && args->dim > 0) {
      error(Loc(), "takes no parameters");
      fatal();
    }
    return LLVMno_typeinfo;
  }

  // pragma(LDC_no_moduleinfo) ;
  if (ident == Id::LDC_no_moduleinfo) {
    if (args && args->dim > 0) {
      error(Loc(), "takes no parameters");
      fatal();
    }
    sc->_module->noModuleInfo = true;
    return LLVMignore;
  }

  // pragma(LDC_alloca) { funcdecl(s) }
  if (ident == Id::LDC_alloca) {
    if (args && args->dim > 0) {
      error(Loc(), "takes no parameters");
      fatal();
    }
    return LLVMalloca;
  }

  // pragma(LDC_va_start) { templdecl(s) }
  if (ident == Id::LDC_va_start) {
    if (args && args->dim > 0) {
      error(Loc(), "takes no parameters");
      fatal();
    }
    return LLVMva_start;
  }

  // pragma(LDC_va_copy) { funcdecl(s) }
  if (ident == Id::LDC_va_copy) {
    if (args && args->dim > 0) {
      error(Loc(), "takes no parameters");
      fatal();
    }
    return LLVMva_copy;
  }

  // pragma(LDC_va_end) { funcdecl(s) }
  if (ident == Id::LDC_va_end) {
    if (args && args->dim > 0) {
      error(Loc(), "takes no parameters");
      fatal();
    }
    return LLVMva_end;
  }

  // pragma(LDC_va_arg) { templdecl(s) }
  if (ident == Id::LDC_va_arg) {
    if (args && args->dim > 0) {
      error(Loc(), "takes no parameters");
      fatal();
    }
    return LLVMva_arg;
  }

  // pragma(LDC_fence) { funcdecl(s) }
  if (ident == Id::LDC_fence) {
    if (args && args->dim > 0) {
      error(Loc(), "takes no parameters");
      fatal();
    }
    return LLVMfence;
  }

  // pragma(LDC_atomic_load) { templdecl(s) }
  if (ident == Id::LDC_atomic_load) {
    if (args && args->dim > 0) {
      error(Loc(), "takes no parameters");
      fatal();
    }
    return LLVMatomic_load;
  }

  // pragma(LDC_atomic_store) { templdecl(s) }
  if (ident == Id::LDC_atomic_store) {
    if (args && args->dim > 0) {
      error(Loc(), "takes no parameters");
      fatal();
    }
    return LLVMatomic_store;
  }

  // pragma(LDC_atomic_cmp_xchg) { templdecl(s) }
  if (ident == Id::LDC_atomic_cmp_xchg) {
    if (args && args->dim > 0) {
      error(Loc(), "takes no parameters");
      fatal();
    }
    return LLVMatomic_cmp_xchg;
  }

  // pragma(LDC_atomic_rmw, "string") { templdecl(s) }
  if (ident == Id::LDC_atomic_rmw) {
    if (!args || args->dim != 1 || !parseStringExp(expr, arg1str)) {
      error(Loc(), "requires exactly 1 string literal parameter");
      fatal();
    }
    return LLVMatomic_rmw;
  }

  // pragma(LDC_verbose);
  if (ident == Id::LDC_verbose) {
    if (args && args->dim > 0) {
      error(Loc(), "takes no parameters");
      fatal();
    }
    sc->_module->llvmForceLogging = true;
    return LLVMignore;
  }

  // pragma(LDC_inline_asm) { templdecl(s) }
  if (ident == Id::LDC_inline_asm) {
    if (args && args->dim > 0) {
      error(Loc(), "takes no parameters");
      fatal();
    }
    return LLVMinline_asm;
  }

  // pragma(LDC_inline_ir) { templdecl(s) }
  if (ident == Id::LDC_inline_ir) {
    if (args && args->dim > 0) {
      error(Loc(), "takes no parameters");
      fatal();
    }
    return LLVMinline_ir;
  }

  // pragma(LDC_extern_weak) { vardecl(s) }
  if (ident == Id::LDC_extern_weak) {
    if (args && args->dim > 0) {
      error(Loc(), "takes no parameters");
      fatal();
    }
    return LLVMextern_weak;
  }

  // pragma(LDC_profile_instr, [true | false])
  if (ident == Id::LDC_profile_instr) {
    // checking of this pragma is done in DtoCheckProfileInstrPragma()
    return LLVMprofile_instr;
  }

  return LLVMnone;
}

void DtoCheckPragma(PragmaDeclaration *decl, Dsymbol *s,
                    LDCPragma llvm_internal, const char *const arg1str) {
  if (llvm_internal == LLVMnone || llvm_internal == LLVMignore ||
      llvm_internal == LLVMprofile_instr) {
    return;
  }

  if (s->llvmInternal) {
    error(Loc(),
          "multiple LDC specific pragmas not allowed not affect the same "
          "declaration (`%s` at '%s')",
          s->toChars(), s->loc.toChars());
    fatal();
  }

  Identifier *ident = decl->ident;

  switch (llvm_internal) {
  case LLVMintrinsic:
    if (FuncDeclaration *fd = s->isFuncDeclaration()) {
      fd->llvmInternal = llvm_internal;
      fd->intrinsicName = strdup(arg1str);
      fd->mangleOverride = strdup(fd->intrinsicName);
    } else if (TemplateDeclaration *td = s->isTemplateDeclaration()) {
      td->llvmInternal = llvm_internal;
      td->intrinsicName = strdup(arg1str);
    } else {
      error(s->loc, "the `%s` pragma is only allowed on function or template "
                    "declarations",
            ident->toChars());
      fatal();
    }
    break;

  case LLVMglobal_crt_ctor:
  case LLVMglobal_crt_dtor:
    if (FuncDeclaration *fd = s->isFuncDeclaration()) {
      assert(fd->type->ty == Tfunction);
      TypeFunction *type = static_cast<TypeFunction *>(fd->type);
      Type *retType = type->next;
      if (retType->ty != Tvoid || type->parameters->dim > 0 ||
          (fd->isMember() && !fd->isStatic())) {
        error(s->loc,
              "the `%s` pragma is only allowed on `void` functions which take "
              "no arguments",
              ident->toChars());
        fd->llvmInternal = LLVMnone;
        break;
      }

      fd->llvmInternal = llvm_internal;
      fd->priority = std::atoi(arg1str);
    } else {
      error(s->loc, "the `%s` pragma is only allowed on function declarations",
            ident->toChars());
      s->llvmInternal = LLVMnone;
    }
    break;

  case LLVMatomic_rmw:
    if (TemplateDeclaration *td = s->isTemplateDeclaration()) {
      td->llvmInternal = llvm_internal;
      td->intrinsicName = strdup(arg1str);
    } else {
      error(s->loc, "the `%s` pragma is only allowed on template declarations",
            ident->toChars());
      fatal();
    }
    break;

  case LLVMva_start:
  case LLVMva_arg:
  case LLVMatomic_load:
  case LLVMatomic_store:
  case LLVMatomic_cmp_xchg:
    if (TemplateDeclaration *td = s->isTemplateDeclaration()) {
      if (td->parameters->dim != 1) {
        error(
            s->loc,
            "the `%s` pragma template must have exactly one template parameter",
            ident->toChars());
        fatal();
      } else if (!td->onemember) {
        error(s->loc, "the `%s` pragma template must have exactly one member",
              ident->toChars());
        fatal();
      } else if (td->overnext || td->overroot) {
        error(s->loc, "the `%s` pragma template must not be overloaded",
              ident->toChars());
        fatal();
      }
      td->llvmInternal = llvm_internal;
    } else {
      error(s->loc, "the `%s` pragma is only allowed on template declarations",
            ident->toChars());
      fatal();
    }
    break;

  case LLVMva_copy:
  case LLVMva_end:
  case LLVMfence:
  case LLVMbitop_bt:
  case LLVMbitop_btc:
  case LLVMbitop_btr:
  case LLVMbitop_bts:
  case LLVMbitop_vld:
  case LLVMbitop_vst:
    if (FuncDeclaration *fd = s->isFuncDeclaration()) {
      fd->llvmInternal = llvm_internal;
    } else {
      error(s->loc, "the `%s` pragma is only allowed on function declarations",
            ident->toChars());
      fatal();
    }
    break;

  case LLVMno_typeinfo:
    s->llvmInternal = llvm_internal;
    break;

  case LLVMalloca:
    if (FuncDeclaration *fd = s->isFuncDeclaration()) {
      fd->llvmInternal = llvm_internal;
    } else {
      error(s->loc,
            "the `%s` pragma must only be used on function declarations "
            "of type `void* function(uint nbytes)`",
            ident->toChars());
      fatal();
    }
    break;

  case LLVMinline_asm:
    if (TemplateDeclaration *td = s->isTemplateDeclaration()) {
      if (td->parameters->dim > 1) {
        error(s->loc, "the `%s` pragma template must have exactly zero or one "
                      "template parameters",
              ident->toChars());
        fatal();
      } else if (!td->onemember) {
        error(s->loc, "the `%s` pragma template must have exactly one member",
              ident->toChars());
        fatal();
      }
      td->llvmInternal = llvm_internal;
    } else {
      error(s->loc, "the `%s` pragma is only allowed on template declarations",
            ident->toChars());
      fatal();
    }
    break;

  case LLVMinline_ir:
    DtoCheckInlineIRPragma(ident, s);
    {
      TemplateDeclaration *td = s->isTemplateDeclaration();
      assert(td != nullptr);
      td->llvmInternal = llvm_internal;
    }
    break;

  case LLVMextern_weak:
    if (VarDeclaration *vd = s->isVarDeclaration()) {
      if (!vd->isDataseg() || !(vd->storage_class & STCextern)) {
        error(s->loc, "`%s` requires storage class `extern`", ident->toChars());
        fatal();
      }

      // It seems like the interaction between weak symbols and thread-local
      // storage is not well-defined (the address of an undefined weak TLS
      // symbol is non-zero on the ELF static TLS model on Linux x86_64).
      // Thus, just disallow this altogether.
      if (vd->isThreadlocal()) {
        error(s->loc, "`%s` cannot be applied to thread-local variable `%s`",
              ident->toChars(), vd->toPrettyChars());
        fatal();
      }
      vd->llvmInternal = llvm_internal;
    } else {
      // Currently, things like "pragma(LDC_extern_weak) extern int foo;"
      // fail because 'extern' creates an intermediate
      // StorageClassDeclaration. This might eventually be fixed by making
      // extern_weak a proper storage class.
      error(s->loc, "the `%s` pragma can only be specified directly on "
                    "variable declarations for now",
            ident->toChars());
      fatal();
    }
    break;

  default:
    warning(s->loc,
            "the LDC specific pragma `%s` is not yet implemented, ignoring",
            ident->toChars());
  }
}

bool DtoIsIntrinsic(FuncDeclaration *fd) {
  switch (fd->llvmInternal) {
  case LLVMintrinsic:
  case LLVMalloca:
  case LLVMfence:
  case LLVMatomic_store:
  case LLVMatomic_load:
  case LLVMatomic_cmp_xchg:
  case LLVMatomic_rmw:
  case LLVMbitop_bt:
  case LLVMbitop_btc:
  case LLVMbitop_btr:
  case LLVMbitop_bts:
  case LLVMbitop_vld:
  case LLVMbitop_vst:
    return true;

  default:
    return DtoIsVaIntrinsic(fd);
  }
}

bool DtoIsVaIntrinsic(FuncDeclaration *fd) {
  return (fd->llvmInternal == LLVMva_start || fd->llvmInternal == LLVMva_copy ||
          fd->llvmInternal == LLVMva_end);
}

// pragma(LDC_profile_instr, [true | false])
// Return false if an error occurred.
bool DtoCheckProfileInstrPragma(Expression *arg, bool &value) {
  return parseBoolExp(arg, value);
}
