#include "gen/inlineir.h"

#include "ddmd/expression.h"
#include "ddmd/mtype.h"
#include "gen/attributes.h"
#include "gen/llvmhelpers.h"
#include "declaration.h"
#include "template.h"
#include "gen/irstate.h"
#include "gen/logger.h"
#include "gen/tollvm.h"
#include "gen/to_string.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/Linker/Linker.h"

namespace {

/// Sets LLVMContext::setDiscardValueNames(false) upon construction and restores
/// the previous value upon destruction.
struct TempDisableDiscardValueNames {
#if LDC_LLVM_VER >= 309
  llvm::LLVMContext &ctx;
  bool previousValue;

  TempDisableDiscardValueNames(llvm::LLVMContext &context)
      : ctx(context), previousValue(context.shouldDiscardValueNames()) {
    ctx.setDiscardValueNames(false);
  }

  ~TempDisableDiscardValueNames() { ctx.setDiscardValueNames(previousValue); }
#else
  TempDisableDiscardValueNames(llvm::LLVMContext &context) {}
#endif
};

/// Adds the idol's function attributes to the wannabe
/// Note: don't add function _parameter_ attributes
void copyFnAttributes(llvm::Function *wannabe, llvm::Function *idol) {
  auto attrSet = idol->getAttributes();
  auto fnAttrSet = attrSet.getFnAttributes();
  wannabe->addAttributes(LLAttributeSet::FunctionIndex, fnAttrSet);
}
} // anonymous namespace

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

    Objects &objs = tinst->tdtypes;
    assert(objs.dim == 3);

    Expression *a0 = isExpression(objs[0]);
    assert(a0);
    StringExp *strexp = a0->toStringExp();
    assert(strexp);
    assert(strexp->sz == 1);
    std::string code(strexp->toPtr(), strexp->numberOfCodeUnits());

    Type *ret = isType(objs[1]);
    assert(ret);

    Tuple *a2 = isTuple(objs[2]);
    assert(a2);
    Objects &arg_types = a2->objects;

    std::string str;
    llvm::raw_string_ostream stream(str);
    stream << "define " << *DtoType(ret) << " @" << mangled_name << "(";

    for (size_t i = 0;;) {
      Type *ty = isType(arg_types[i]);
      // assert(ty);
      if (!ty) {
        error(tinst->loc, "All parameters of a template defined with pragma "
                          "LDC_inline_ir, except for the first one, should be "
                          "types");
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

    llvm::SMDiagnostic err;

    std::unique_ptr<llvm::Module> m =
        llvm::parseAssemblyString(stream.str().c_str(), err, gIR->context());

    std::string errstr = err.getMessage();
    if (!errstr.empty()) {
      error(
          tinst->loc,
          "can't parse inline LLVM IR:\n%s\n%s\n%s\nThe input string was: \n%s",
          err.getLineContents().str().c_str(),
          (std::string(err.getColumnNo(), ' ') + '^').c_str(), errstr.c_str(),
          stream.str().c_str());
    }

    m->setDataLayout(gIR->module.getDataLayout());

#if LDC_LLVM_VER >= 308
    llvm::Linker(gIR->module).linkInModule(std::move(m));
#else
    llvm::Linker(&gIR->module).linkInModule(m.get());
#endif
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

    // work around missing tuple support for users of the return value
    if (type->toBasetype()->ty == Tstruct) {
      // make a copy
      llvm::Value *mem = DtoAlloca(type, ".__ir_tuple_ret");
      DtoStore(rv, DtoBitCast(mem, getPtrToType(rv->getType())));
      return new DLValue(type, mem);
    }

    // return call as im value
    return new DImValue(type, rv);
  }
}
