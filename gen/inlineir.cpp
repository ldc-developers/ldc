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
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/Linker/Linker.h"

namespace {

/// Adds the idol's function attributes to the wannabe
void copyFnAttributes(llvm::Function *wannabe, llvm::Function *idol) {
  auto attrSet = idol->getAttributes();
  auto fnAttrSet = attrSet.getFnAttributes();
  wannabe->addAttributes(llvm::AttributeSet::FunctionIndex, fnAttrSet);
}
} // anonymous namespace

DValue *DtoInlineIRExpr(Loc &loc, FuncDeclaration *fdecl,
                        Expressions *arguments) {
  IF_LOG Logger::println("DtoInlineIRExpr @ %s", loc.toChars());
  LOG_SCOPE;

  // Generate a random new function name. Because the inlineIR function is
  // always inlined, this name does not escape the current compiled module; not
  // even at -O0.
  static size_t namecounter = 0;
  std::string mangled_name = "inline.ir." + std::to_string(namecounter++);
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

#if LDC_LLVM_VER >= 306
    std::unique_ptr<llvm::Module> m =
        llvm::parseAssemblyString(stream.str().c_str(), err, gIR->context());
#else
    llvm::Module *m = llvm::ParseAssemblyString(stream.str().c_str(), NULL, err,
                                                gIR->context());
#endif

    std::string errstr = err.getMessage();
    if (errstr != "") {
      error(
          tinst->loc,
          "can't parse inline LLVM IR:\n%s\n%s\n%s\nThe input string was: \n%s",
          err.getLineContents().str().c_str(),
          (std::string(err.getColumnNo(), ' ') + '^').c_str(), errstr.c_str(),
          stream.str().c_str());
    }

#if LDC_LLVM_VER >= 308
    llvm::Linker(gIR->module).linkInModule(std::move(m));
#elif LDC_LLVM_VER >= 306
    llvm::Linker(&gIR->module).linkInModule(m.get());
#else
    std::string errstr2 = "";
    llvm::Linker(&gIR->module).linkInModule(m, &errstr2);
    if (errstr2 != "")
      error(tinst->loc, "Error when linking in llvm inline ir: %s",
            errstr2.c_str());
#endif
  }

  // 2. Call the function that was just defined and return the returnvalue
  {
    llvm::Function *fun = gIR->module.getFunction(mangled_name);

    // Apply some parent function attributes to the inlineIR function too. This
    // is needed e.g. when the parent function has "unsafe-fp-math"="true"
    // applied.
    {
      assert(!gIR->functions.empty() && "Inline ir outside function");
      auto enclosingFunc = gIR->topfunc();
      assert(enclosingFunc);
      copyFnAttributes(fun, enclosingFunc);
    }

    fun->setLinkage(llvm::GlobalValue::PrivateLinkage);
    fun->addFnAttr(llvm::Attribute::AlwaysInline);
    fun->setCallingConv(llvm::CallingConv::C);

    // Build the runtime arguments
    size_t n = arguments->dim;
    llvm::SmallVector<llvm::Value *, 8> args;
    args.reserve(n);
    for (size_t i = 0; i < n; i++) {
      args.push_back(toElem((*arguments)[i])->getRVal());
    }

    llvm::Value *rv = gIR->ir->CreateCall(fun, args);

    // work around missing tuple support for users of the return value
    Type *type = fdecl->type->nextOf()->toBasetype();
    if (type->ty == Tstruct) {
      // make a copy
      llvm::Value *mem = DtoAlloca(type, ".__ir_tuple_ret");

      TypeStruct *ts = static_cast<TypeStruct *>(type);
      size_t n = ts->sym->fields.dim;
      for (size_t i = 0; i < n; i++) {
        llvm::Value *v = gIR->ir->CreateExtractValue(rv, i, "");
        llvm::Value *gep = DtoGEPi(mem, 0, i);
        DtoStore(v, gep);
      }

      return new DVarValue(fdecl->type->nextOf(), mem);
    }

    // return call as im value
    return new DImValue(fdecl->type->nextOf(), rv);
  }
}
