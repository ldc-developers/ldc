#include "gen/inlineir.h"
#include "gen/llvmhelpers.h"

#include "declaration.h"
#include "template.h"
#include "gen/irstate.h"
#include "gen/tollvm.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/Linker/Linker.h"

llvm::Function *DtoInlineIRFunction(FuncDeclaration *fdecl) {
  const char *mangled_name = mangleExact(fdecl);
  TemplateInstance *tinst = fdecl->parent->isTemplateInstance();
  assert(tinst);

  Objects &objs = tinst->tdtypes;
  assert(objs.dim == 3);

  Expression *a0 = isExpression(objs[0]);
  assert(a0);
  StringExp *strexp = a0->toStringExp();
  assert(strexp);
  assert(strexp->sz == 1);
  std::string code(static_cast<char *>(strexp->string), strexp->len);

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
                        "llvm_inline_ir, except for the first one, should be "
                        "types");
      fatal();
    }
    stream << *DtoType(ty);

    i++;
    if (i >= arg_types.dim)
      break;

    stream << ", ";
  }

  if (ret->ty == Tvoid)
    code.append("\nret void");

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
  if (errstr != "")
    error(tinst->loc,
          "can't parse inline LLVM IR:\n%s\n%s\n%s\nThe input string was: \n%s",
          err.getLineContents().str().c_str(),
          (std::string(err.getColumnNo(), ' ') + '^').c_str(), errstr.c_str(),
          stream.str().c_str());

#if LDC_LLVM_VER >= 306
  llvm::Linker(&gIR->module).linkInModule(m.get());
#else
  std::string errstr2 = "";
  llvm::Linker(&gIR->module).linkInModule(m, &errstr2);
  if (errstr2 != "")
    error(tinst->loc, "Error when linking in llvm inline ir: %s",
          errstr2.c_str());
#endif

  LLFunction *fun = gIR->module.getFunction(mangled_name);
  fun->setLinkage(llvm::GlobalValue::LinkOnceODRLinkage);
  SET_COMDAT(fun, gIR->module);
  fun->addFnAttr(LLAttribute::AlwaysInline);
  return fun;
}
