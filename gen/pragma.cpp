//===-- pragma.cpp --------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/pragma.h"

#include "dmd/attrib.h"
#include "dmd/declaration.h"
#include "dmd/errors.h"
#include "dmd/expression.h"
#include "dmd/id.h"
#include "dmd/identifier.h"
#include "dmd/module.h"
#include "dmd/scope.h"
#include "dmd/template.h"
#include "gen/inlineir.h"
#include "gen/llvmhelpers.h"
#include "llvm/Support/CommandLine.h"

namespace {
bool parseStringExp(Expression *e, const char *&res) {
  e = e->optimize(WANTvalue);
  if (e->op != TOKstring) {
    return false;
  }
  auto se = static_cast<StringExp *>(e);
  auto size = (se->len + 1) * se->sz;
  auto s = static_cast<char *>(mem.xmalloc(size));
  se->writeTo(s, true);
  res = s;
  return true;
}

bool parseIntExp(Expression *e, dinteger_t &res) {
  e = e->optimize(WANTvalue);
  if (auto i = e->isIntegerExp()) {
    res = i->getInteger();
    return true;
  }
  return false;
}

bool parseBoolExp(Expression *e, bool &res) {
  e = e->optimize(WANTvalue);
  if (auto i = e->isIntegerExp()) {
    if (e->type->equals(Type::tbool)) {
      res = i->isBool(true);
      return true;
    }
  }
  return false;
}

// Applies an action to matching symbols, recursively descending into nested
// AttribDeclarations, and returns the number of applications.
template <typename T>
int applyPragma(Dsymbol *s, std::function<T *(Dsymbol *)> predicate,
                std::function<void(T *)> action) {
  if (T *matchingDecl = predicate(s)) {
    if (matchingDecl->llvmInternal != LLVMnone) {
      s->error("multiple LDC specific pragmas are not allowed");
      fatal();
    }
    action(matchingDecl);
    return 1;
  }

  if (auto ad = s->isAttribDeclaration()) {
    if (ad->decl) {
      int count = 0;
      for (auto child : *ad->decl) {
        count += applyPragma(child, predicate, action);
      }
      return count;
    }
  }

  return 0;
}

int applyFunctionPragma(Dsymbol *s,
                        std::function<void(FuncDeclaration *)> action) {
  return applyPragma<FuncDeclaration>(
      s, [](Dsymbol *s) { return s->isFuncDeclaration(); }, action);
}

int applyTemplatePragma(Dsymbol *s,
                        std::function<void(TemplateDeclaration *)> action) {
  return applyPragma<TemplateDeclaration>(
      s, [](Dsymbol *s) { return s->isTemplateDeclaration(); }, action);
}

int applyVariablePragma(Dsymbol *s,
                        std::function<void(VarDeclaration *)> action) {
  return applyPragma<VarDeclaration>(
      s, [](Dsymbol *s) { return s->isVarDeclaration(); }, action);
}
} // anonymous namespace

LDCPragma DtoGetPragma(Scope *sc, PragmaDeclaration *decl,
                       const char *&arg1str) {
  Identifier *ident = decl->ident;
  Expressions *args = decl->args;
  Expression *expr =
      (args && args->length > 0) ? expressionSemantic((*args)[0], sc) : nullptr;

  // pragma(LDC_intrinsic, "string") { funcdecl(s) }
  if (ident == Id::LDC_intrinsic) {
    if (!args || args->length != 1 || !parseStringExp(expr, arg1str)) {
      decl->error("requires exactly 1 string literal parameter");
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

  // pragma(LDC_global_crt_{c,d}tor [, priority]) { funcdecl(s) }
  // pragma(crt_{con,de}structor [, priority]) { funcdecl(s) }
  if (ident == Id::LDC_global_crt_ctor || ident == Id::LDC_global_crt_dtor ||
      ident == Id::crt_constructor || ident == Id::crt_destructor) {
    dinteger_t priority;
    if (args) {
      if (args->length != 1 || !parseIntExp(expr, priority)) {
        decl->error("requires at most 1 integer literal parameter");
        fatal();
      }
      if (priority > 65535) {
        decl->error("priority may not be greater than 65535");
        priority = 65535;
      }
    } else {
      priority = 65535;
    }
    char buf[8];
    sprintf(buf, "%llu", static_cast<unsigned long long>(priority));
    arg1str = strdup(buf);
    return ident == Id::LDC_global_crt_ctor || ident == Id::crt_constructor
               ? LLVMglobal_crt_ctor
               : LLVMglobal_crt_dtor;
  }

  // pragma(LDC_no_typeinfo) { typedecl(s) }
  if (ident == Id::LDC_no_typeinfo) {
    if (args && args->length > 0) {
      decl->error("takes no parameters");
      fatal();
    }
    return LLVMno_typeinfo;
  }

  // pragma(LDC_no_moduleinfo) ;
  if (ident == Id::LDC_no_moduleinfo) {
    if (args && args->length > 0) {
      decl->error("takes no parameters");
      fatal();
    }
    sc->_module->noModuleInfo = true;
    return LLVMignore;
  }

  // pragma(LDC_alloca) { funcdecl(s) }
  if (ident == Id::LDC_alloca) {
    if (args && args->length > 0) {
      decl->error("takes no parameters");
      fatal();
    }
    return LLVMalloca;
  }

  // pragma(LDC_va_start) { templdecl(s) }
  if (ident == Id::LDC_va_start) {
    if (args && args->length > 0) {
      decl->error("takes no parameters");
      fatal();
    }
    return LLVMva_start;
  }

  // pragma(LDC_va_copy) { funcdecl(s) }
  if (ident == Id::LDC_va_copy) {
    if (args && args->length > 0) {
      decl->error("takes no parameters");
      fatal();
    }
    return LLVMva_copy;
  }

  // pragma(LDC_va_end) { funcdecl(s) }
  if (ident == Id::LDC_va_end) {
    if (args && args->length > 0) {
      decl->error("takes no parameters");
      fatal();
    }
    return LLVMva_end;
  }

  // pragma(LDC_va_arg) { templdecl(s) }
  if (ident == Id::LDC_va_arg) {
    if (args && args->length > 0) {
      decl->error("takes no parameters");
      fatal();
    }
    return LLVMva_arg;
  }

  // pragma(LDC_fence) { funcdecl(s) }
  if (ident == Id::LDC_fence) {
    if (args && args->length > 0) {
      decl->error("takes no parameters");
      fatal();
    }
    return LLVMfence;
  }

  // pragma(LDC_atomic_load) { templdecl(s) }
  if (ident == Id::LDC_atomic_load) {
    if (args && args->length > 0) {
      decl->error("takes no parameters");
      fatal();
    }
    return LLVMatomic_load;
  }

  // pragma(LDC_atomic_store) { templdecl(s) }
  if (ident == Id::LDC_atomic_store) {
    if (args && args->length > 0) {
      decl->error("takes no parameters");
      fatal();
    }
    return LLVMatomic_store;
  }

  // pragma(LDC_atomic_cmp_xchg) { templdecl(s) }
  if (ident == Id::LDC_atomic_cmp_xchg) {
    if (args && args->length > 0) {
      decl->error("takes no parameters");
      fatal();
    }
    return LLVMatomic_cmp_xchg;
  }

  // pragma(LDC_atomic_rmw, "string") { templdecl(s) }
  if (ident == Id::LDC_atomic_rmw) {
    if (!args || args->length != 1 || !parseStringExp(expr, arg1str)) {
      decl->error("requires exactly 1 string literal parameter");
      fatal();
    }
    return LLVMatomic_rmw;
  }

  // pragma(LDC_verbose);
  if (ident == Id::LDC_verbose) {
    if (args && args->length > 0) {
      decl->error("takes no parameters");
      fatal();
    }
    sc->_module->llvmForceLogging = true;
    return LLVMignore;
  }

  // pragma(LDC_inline_asm) { templdecl(s) }
  if (ident == Id::LDC_inline_asm) {
    if (args && args->length > 0) {
      decl->error("takes no parameters");
      fatal();
    }
    return LLVMinline_asm;
  }

  // pragma(LDC_inline_ir) { templdecl(s) }
  if (ident == Id::LDC_inline_ir) {
    if (args && args->length > 0) {
      decl->error("takes no parameters");
      fatal();
    }
    return LLVMinline_ir;
  }

  // pragma(LDC_extern_weak) { vardecl(s) }
  if (ident == Id::LDC_extern_weak) {
    if (args && args->length > 0) {
      decl->error("takes no parameters");
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

  Identifier *ident = decl->ident;

  switch (llvm_internal) {
  case LLVMintrinsic: {
    const char *mangle = strdup(arg1str);
    int count = applyFunctionPragma(s, [=](FuncDeclaration *fd) {
      fd->llvmInternal = llvm_internal;
      fd->intrinsicName = mangle;
      fd->mangleOverride = {strlen(mangle), mangle};
    });
    count += applyTemplatePragma(s, [=](TemplateDeclaration *td) {
      td->llvmInternal = llvm_internal;
      td->intrinsicName = mangle;
    });
    if (count != 1) {
      error(s->loc,
            "the `%s` pragma doesn't affect exactly 1 function/template "
            "declaration",
            ident->toChars());
      fatal();
    }
    break;
  }

  case LLVMglobal_crt_ctor:
  case LLVMglobal_crt_dtor: {
    const unsigned char flag = llvm_internal == LLVMglobal_crt_ctor ? 1 : 2;
    const int count = applyFunctionPragma(s, [=](FuncDeclaration *fd) {
      assert(fd->type->ty == Tfunction);
      TypeFunction *type = static_cast<TypeFunction *>(fd->type);
      Type *retType = type->next;
      if (retType->ty != Tvoid || type->parameterList.length() > 0 ||
          (fd->isMember() && !fd->isStatic())) {
        error(s->loc,
              "the `%s` pragma is only allowed on `void` functions which take "
              "no arguments",
              ident->toChars());
        fd->isCrtCtorDtor &= ~flag;
      } else {
        fd->isCrtCtorDtor |= flag;
        fd->priority = std::atoi(arg1str);
      }
    });
    if (count != 1) {
      error(s->loc,
            "the `%s` pragma doesn't affect exactly 1 function declaration",
            ident->toChars());
      fatal();
    }
    break;
  }

  case LLVMatomic_rmw: {
    const int count = applyTemplatePragma(s, [=](TemplateDeclaration *td) {
      td->llvmInternal = llvm_internal;
      td->intrinsicName = strdup(arg1str);
    });
    if (count != 1) {
      error(s->loc,
            "the `%s` pragma doesn't affect exactly 1 template declaration",
            ident->toChars());
      fatal();
    }
    break;
  }

  case LLVMva_start:
  case LLVMva_arg:
  case LLVMatomic_load:
  case LLVMatomic_store:
  case LLVMatomic_cmp_xchg: {
    const int count = applyTemplatePragma(s, [=](TemplateDeclaration *td) {
      if (td->parameters->length != 1) {
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
    });
    if (count != 1) {
      error(s->loc,
            "the `%s` pragma doesn't affect exactly 1 template declaration",
            ident->toChars());
      fatal();
    }
    break;
  }

  case LLVMalloca:
  case LLVMva_copy:
  case LLVMva_end:
  case LLVMfence:
  case LLVMbitop_bt:
  case LLVMbitop_btc:
  case LLVMbitop_btr:
  case LLVMbitop_bts:
  case LLVMbitop_vld:
  case LLVMbitop_vst: {
    const int count = applyFunctionPragma(s, [=](FuncDeclaration *fd) {
      fd->llvmInternal = llvm_internal;
    });
    if (count != 1) {
      error(s->loc,
            "the `%s` pragma doesn't affect exactly 1 function declaration",
            ident->toChars());
      fatal();
    }
    break;
  }

  case LLVMno_typeinfo:
    applyPragma<Dsymbol>(s, [](Dsymbol *s) { return s; },
                         [=](Dsymbol *s) { s->llvmInternal = llvm_internal; });
    break;

  case LLVMinline_asm: {
    const int count = applyTemplatePragma(s, [=](TemplateDeclaration *td) {
      if (td->parameters->length > 1) {
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
    });
    if (count != 1) {
      error(s->loc,
            "the `%s` pragma doesn't affect exactly 1 template declaration",
            ident->toChars());
      fatal();
    }
    break;
  }

  case LLVMinline_ir: {
    DtoCheckInlineIRPragma(ident, s);
    const int count = applyTemplatePragma(s, [=](TemplateDeclaration *td) {
      td->llvmInternal = llvm_internal;
    });
    if (count != 1) {
      error(s->loc,
            "the `%s` pragma doesn't affect exactly 1 template declaration",
            ident->toChars());
      fatal();
    }
    break;
  }

  case LLVMextern_weak: {
    int count = applyVariablePragma(s, [=](VarDeclaration *vd) {
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
    });
    count += applyFunctionPragma(s, [=](FuncDeclaration *fd) {
      if (fd->fbody) {
        error(s->loc, "`%s` cannot be applied to function definitions", ident->toChars());
        fatal();
      }
      fd->llvmInternal = llvm_internal;
    });

    if (count == 0) {
      error(s->loc,
            "the `%s` pragma doesn't affect any variable or function "
            "declarations",
            ident->toChars());
      fatal();
    }
    break;
  }

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
