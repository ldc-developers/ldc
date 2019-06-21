#include "gen/inlineir.h"

#include "dmd/declaration.h"
#include "dmd/errors.h"
#include "dmd/expression.h"
#include "dmd/identifier.h"
#include "dmd/mtype.h"
#include "dmd/template.h"
#include "gen/attributes.h"
#include "gen/irstate.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/tollvm.h"
#include "gen/to_string.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/SourceMgr.h"

namespace {

/// Sets LLVMContext::setDiscardValueNames(false) upon construction and restores
/// the previous value upon destruction.
struct TempDisableDiscardValueNames {
  llvm::LLVMContext &ctx;
  bool previousValue;

  TempDisableDiscardValueNames(llvm::LLVMContext &context)
      : ctx(context), previousValue(context.shouldDiscardValueNames()) {
    ctx.setDiscardValueNames(false);
  }

  ~TempDisableDiscardValueNames() { ctx.setDiscardValueNames(previousValue); }
};

/// Adds the idol's function attributes to the wannabe
/// Note: don't add function _parameter_ attributes
void copyFnAttributes(llvm::Function *wannabe, llvm::Function *idol) {
  auto attrSet = idol->getAttributes();
  auto fnAttrSet = attrSet.getFnAttributes();
  wannabe->addAttributes(LLAttributeSet::FunctionIndex, fnAttrSet);
}

std::string exprToString(StringExp *strexp) {
  assert(strexp != nullptr);
  assert(strexp->sz == 1);
  return std::string(strexp->toPtr(), strexp->numberOfCodeUnits());
}
} // anonymous namespace

void DtoCheckInlineIRPragma(Identifier *ident, Dsymbol *s) {
  assert(ident != nullptr);
  assert(s != nullptr);
  if (TemplateDeclaration *td = s->isTemplateDeclaration()) {
    Dsymbol *member = td->onemember;
    if (!member) {
      error(s->loc, "the `%s` pragma template must have exactly one member",
            ident->toChars());
      fatal();
    }
    FuncDeclaration *fun = member->isFuncDeclaration();
    if (!fun) {
      error(
          s->loc,
          "the `%s` pragma template's member must be a function declaration",
          ident->toChars());
      fatal();
    }
    // The magic inlineIR template is one of
    // pragma(LDC_inline_ir)
    //   R inlineIR(string code, R, P...)(P);
    // pragma(LDC_inline_ir)
    //   R inlineIREx(string prefix, string code, string suffix, R, P...)(P);

    TemplateParameters &params = *td->parameters;
    bool valid_params =
        (params.dim == 3 || params.dim == 5) &&
        params[params.dim - 2]->isTemplateTypeParameter() &&
        params[params.dim - 1]->isTemplateTupleParameter();

    if (valid_params) {
      for (d_size_t i = 0; i < (params.dim - 2); ++i) {
        TemplateValueParameter *p0 = params[i]->isTemplateValueParameter();
        valid_params = valid_params && p0 && p0->valType == Type::tstring;
      }
    }

    if (!valid_params) {
      error(s->loc,
            "the `%s` pragma template must have three "
            "(string, type and type tuple) or "
            "five (string, string, string, type and type tuple) parameters",
            ident->toChars());
      fatal();
    }
  } else {
    error(s->loc, "the `%s` pragma is only allowed on template declarations",
          ident->toChars());
    fatal();
  }
}

DValue *DtoInlineIRExpr(Loc &loc, FuncDeclaration *fdecl,
                        Expressions *arguments, LLValue *sretPointer) {
  IF_LOG Logger::println("DtoInlineIRExpr @ %s", loc.toChars());
  LOG_SCOPE;

  // LLVM can't read textual IR with a Context that discards named Values, so
  // temporarily disable value name discarding.
  TempDisableDiscardValueNames tempDisable(gIR->context());

  // Generate a random new function name. Because the inlineIR function is
  // always inlined, this name does not escape the current compiled module; not
  // even at -O0.
  static size_t namecounter = 0;
  std::string mangled_name = "inline.ir." + ldc::to_string(namecounter++);
  TemplateInstance *tinst = fdecl->parent->isTemplateInstance();
  assert(tinst);

  // 1. Define the inline function (define a new function for each call)
  {
    // The magic inlineIR template is one of
    // pragma(LDC_inline_ir)
    //   R inlineIR(string code, R, P...)(P);
    // pragma(LDC_inline_ir)
    //   R inlineIREx(string prefix, string code, string suffix, R, P...)(P);
    Objects &objs = tinst->tdtypes;
    assert(objs.dim == 3 || objs.dim == 5);
    const bool isExtended = (objs.dim == 5);

    std::string prefix;
    std::string code;
    std::string suffix;
    if (isExtended) {
      Expression *a0 = isExpression(objs[0]);
      assert(a0);
      StringExp *prefexp = a0->toStringExp();
      Expression *a1 = isExpression(objs[1]);
      assert(a1);
      StringExp *strexp = a1->toStringExp();
      Expression *a2 = isExpression(objs[2]);
      assert(a2);
      StringExp *suffexp = a2->toStringExp();
      prefix = exprToString(prefexp);
      code = exprToString(strexp);
      suffix = exprToString(suffexp);
    } else {
      Expression *a0 = isExpression(objs[0]);
      assert(a0);
      StringExp *strexp = a0->toStringExp();
      code = exprToString(strexp);
    }

    Type *ret = isType(objs[isExtended ? 3 : 1]);
    assert(ret);

    Tuple *args = isTuple(objs[isExtended ? 4 : 2]);
    assert(args);
    Objects &arg_types = args->objects;

    std::string str;
    llvm::raw_string_ostream stream(str);
    if (!prefix.empty()) {
      stream << prefix << "\n";
    }
    stream << "define " << *DtoType(ret) << " @" << mangled_name << "(";

    for (size_t i = 0;;) {
      Type *ty = isType(arg_types[i]);
      // assert(ty);
      if (!ty) {
        error(tinst->loc,
              "All parameters of a template defined with pragma "
              "`LDC_inline_ir`, except for the first one or the first three"
              ", should be types");
        fatal();
      }
      stream << *DtoType(ty);

      i++;
      if (i >= arg_types.dim) {
        break;
      }

      stream << ", ";
    }

    if (ret->ty == Tvoid) {
      code.append("\nret void");
    }

    stream << ")\n{\n" << code << "\n}";
    if (!suffix.empty()) {
      stream << "\n" << suffix;
    }

    llvm::SMDiagnostic err;

    std::unique_ptr<llvm::Module> m =
        llvm::parseAssemblyString(stream.str().c_str(), err, gIR->context());

    std::string errstr = err.getMessage();
    if (!errstr.empty()) {
      error(tinst->loc,
            "can't parse inline LLVM IR:\n`%s`\n%s\n%s\nThe input string "
            "was:\n`%s`",
            err.getLineContents().str().c_str(),
            (std::string(err.getColumnNo(), ' ') + '^').c_str(), errstr.c_str(),
            stream.str().c_str());
    }

    m->setDataLayout(gIR->module.getDataLayout());

    llvm::Linker(gIR->module).linkInModule(std::move(m));
  }

  // 2. Call the function that was just defined and return the returnvalue
  {
    llvm::Function *fun = gIR->module.getFunction(mangled_name);

    // Apply some parent function attributes to the inlineIR function too. This
    // is needed e.g. when the parent function has "unsafe-fp-math"="true"
    // applied.
    {
      assert(!gIR->funcGenStates.empty() && "Inline ir outside function");
      auto enclosingFunc = gIR->topfunc();
      assert(enclosingFunc);
      copyFnAttributes(fun, enclosingFunc);
    }

    fun->setLinkage(llvm::GlobalValue::PrivateLinkage);
    fun->removeFnAttr(llvm::Attribute::NoInline);
    fun->addFnAttr(llvm::Attribute::AlwaysInline);
    fun->setCallingConv(llvm::CallingConv::C);

    // Build the runtime arguments
    llvm::SmallVector<llvm::Value *, 8> args;
    args.reserve(arguments->dim);
    for (auto arg : *arguments) {
      args.push_back(DtoRVal(arg));
    }

    llvm::Value *rv = gIR->ir->CreateCall(fun, args);
    Type *type = fdecl->type->nextOf();

    if (sretPointer) {
      DtoStore(rv, DtoBitCast(sretPointer, getPtrToType(rv->getType())));
      return new DLValue(type, sretPointer);
    }

    // dump struct and static array return values to memory
    if (DtoIsInMemoryOnly(type->toBasetype())) {
      LLValue *lval = DtoAllocaDump(rv, type, ".__ir_ret");
      return new DLValue(type, lval);
    }

    // return call as im value
    return new DImValue(type, rv);
  }
}
