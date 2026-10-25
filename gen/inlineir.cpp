#include "gen/inlineir.h"

#include "dmd/declaration.h"
#include "dmd/dsymbol.h"
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
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/SourceMgr.h"
#if LDC_MLIR_ENABLED
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/TypeFromLLVM.h"
#endif // LDC_MLIR_ENABLED
#include <memory>

using namespace dmd;

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
  auto fnAttrSet = attrSet.getFnAttrs();
  wannabe->addFnAttrs(llvm::AttrBuilder(getGlobalContext(), fnAttrSet));
}

llvm::StringRef exprToString(StringExp *strexp) {
  assert(strexp);
  auto str = strexp->peekString();
  return {str.ptr, str.length};
}

/// The template arguments of an instantiated inline IR/MLIR template.
struct InlineIRArgs {
  llvm::StringRef prefix, code, suffix;
  Type *ret;
  llvm::SmallVector<Type *, 8> paramTypes;
};

/// Checks that the pragma is applied to a template declaration of the expected
/// shape. The magic template is one of
///   pragma(LDC_inline_ir)
///     R inlineIR(string code, R, P...)(P);
///   pragma(LDC_inline_ir)
///     R inlineIREx(string prefix, string code, string suffix, R, P...)(P);
/// `LDC_inline_mlir` only accepts the first (3-parameter) form.
void checkInlinePragmaTemplate(Identifier *ident, Dsymbol *s,
                               bool allowPrefixSuffix) {
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

    TemplateParameters &params = *td->parameters;
    bool valid_params =
        (params.length == 3 || (allowPrefixSuffix && params.length == 5)) &&
        params[params.length - 2]->isTemplateTypeParameter() &&
        params[params.length - 1]->isTemplateTupleParameter();

    if (valid_params) {
      for (d_size_t i = 0; i < (params.length - 2); ++i) {
        TemplateValueParameter *p0 = params[i]->isTemplateValueParameter();
        valid_params = valid_params && p0 && p0->valType == Type::tstring;
      }
    }

    if (!valid_params) {
      if (allowPrefixSuffix) {
        error(s->loc,
              "the `%s` pragma template must have three "
              "(string, type and type tuple) or "
              "five (string, string, string, type and type tuple) parameters",
              ident->toChars());
      } else {
        error(s->loc,
              "the `%s` pragma template must have three "
              "(string, type and type tuple) parameters",
              ident->toChars());
      }
      fatal();
    }
  } else {
    error(s->loc, "the `%s` pragma is only allowed on template declarations",
          ident->toChars());
    fatal();
  }
}

InlineIRArgs extractInlineIRArgs(TemplateInstance *tinst,
                                 const char *pragmaName) {
  Objects &objs = tinst->tdtypes;
  assert(objs.length == 3 || objs.length == 5);
  const bool isExtended = (objs.length == 5);

  InlineIRArgs result;
  if (isExtended) {
    Expression *a0 = isExpression(objs[0]);
    assert(a0);
    StringExp *prefexp = toStringExp(a0);
    Expression *a1 = isExpression(objs[1]);
    assert(a1);
    StringExp *strexp = toStringExp(a1);
    Expression *a2 = isExpression(objs[2]);
    assert(a2);
    StringExp *suffexp = toStringExp(a2);
    result.prefix = exprToString(prefexp);
    result.code = exprToString(strexp);
    result.suffix = exprToString(suffexp);
  } else {
    Expression *a0 = isExpression(objs[0]);
    assert(a0);
    StringExp *strexp = toStringExp(a0);
    result.code = exprToString(strexp);
  }

  result.ret = isType(objs[isExtended ? 3 : 1]);
  assert(result.ret);

  Tuple *args = isTuple(objs[isExtended ? 4 : 2]);
  assert(args);
  for (auto arg : args->objects) {
    Type *ty = isType(arg);
    if (!ty) {
      error(tinst->loc,
            "All parameters of a template defined with pragma "
            "`%s`, except for the first %s, should be types",
            pragmaName, isExtended ? "three" : "one");
      fatal();
    }
    result.paramTypes.push_back(ty);
  }

  return result;
}

std::unique_ptr<llvm::Module>
parseInlineLLVMIR(const InlineIRArgs &inlineIRArgs, llvm::StringRef fnName,
                  Loc loc) {
  // LLVM can't read textual IR with a Context that discards named Values, so
  // temporarily disable value name discarding.
  TempDisableDiscardValueNames tempDisable(gIR->context());

  std::string str;
  llvm::raw_string_ostream stream(str);
  if (!inlineIRArgs.prefix.empty()) {
    stream << inlineIRArgs.prefix << "\n";
  }
  stream << "define " << *DtoType(inlineIRArgs.ret) << " @" << fnName << "(";

  for (size_t i = 0; i < inlineIRArgs.paramTypes.size(); ++i) {
    if (i != 0)
      stream << ", ";
    stream << *DtoType(inlineIRArgs.paramTypes[i]);
  }

  stream << ")\n{\n" << inlineIRArgs.code;
  if (inlineIRArgs.ret->ty == TY::Tvoid) {
    stream << "\nret void";
  }
  stream << "\n}";
  if (!inlineIRArgs.suffix.empty()) {
    stream << "\n" << inlineIRArgs.suffix;
  }

  llvm::SMDiagnostic err;

  std::unique_ptr<llvm::Module> m =
      llvm::parseAssemblyString(stream.str().c_str(), err, gIR->context());

  std::string errstr(err.getMessage());
  if (!errstr.empty()) {
    error(loc,
          "can't parse inline LLVM IR:\n`%s`\n%s\n%s\nThe input string "
          "was:\n`%s`",
          err.getLineContents().str().c_str(),
          (std::string(err.getColumnNo(), ' ') + '^').c_str(), errstr.c_str(),
          stream.str().c_str());
    fatal();
  }

  return m;
}

#if LDC_MLIR_ENABLED
std::unique_ptr<llvm::Module> parseInlineMLIR(const InlineIRArgs &inlineIRArgs,
                                              llvm::StringRef fnName, Loc loc) {
  mlir::DialectRegistry registry;
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  registry.insert<mlir::func::FuncDialect, mlir::arith::ArithDialect,
                  mlir::cf::ControlFlowDialect, mlir::scf::SCFDialect,
                  mlir::LLVM::LLVMDialect>();

  mlir::MLIRContext context(registry);
  context.loadDialect<mlir::func::FuncDialect, mlir::arith::ArithDialect,
                      mlir::cf::ControlFlowDialect, mlir::scf::SCFDialect,
                      mlir::LLVM::LLVMDialect>();
  mlir::LLVM::TypeFromLLVMIRTranslator typeTranslator(context);

  std::string str;
  llvm::raw_string_ostream stream(str);
  stream << "func.func @" << fnName << "(";
  for (size_t i = 0; i < inlineIRArgs.paramTypes.size(); ++i) {
    if (i != 0)
      stream << ", ";
    mlir::Type mlirTy =
        typeTranslator.translateType(DtoType(inlineIRArgs.paramTypes[i]));
    stream << "%arg" << i << ": " << mlirTy;
  }
  stream << ")";

  const bool isVoid = inlineIRArgs.ret->ty == TY::Tvoid;
  if (!isVoid) {
    stream << " -> " << typeTranslator.translateType(DtoType(inlineIRArgs.ret));
  }

  stream << "\n{\n" << inlineIRArgs.code;
  if (isVoid) {
    stream << "\n  return";
  }
  stream << "\n}";

  IF_LOG Logger::println("Constructed inline MLIR function:\n%s",
                         stream.str().c_str());

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(stream.str(), &context);
  if (!module) {
    error(loc, "failed to parse inline MLIR module");
    fatal();
  }
  if (mlir::failed(module->verify())) {
    error(loc, "inline MLIR module verification failed");
    fatal();
  }

  mlir::PassManager pm(&context);
#if LLVM_VERSION_MAJOR >= 21
  pm.addPass(mlir::createSCFToControlFlowPass());
#else
  pm.addPass(mlir::createConvertSCFToCFPass());
#endif
  pm.addPass(mlir::createArithToLLVMConversionPass());
  pm.addPass(mlir::createConvertControlFlowToLLVMPass());
  pm.addPass(mlir::createConvertFuncToLLVMPass());
  // Clean up any remaining conversion casts before translating to LLVM IR.
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());

  if (mlir::failed(pm.run(*module))) {
    error(loc, "MLIR lowering pass pipeline failed!");
    fatal();
  }

  IF_LOG {
    Logger::println("Lowered inline MLIR to the llvm dialect:");
    module->dump();
  }

  std::unique_ptr<llvm::Module> m =
      mlir::translateModuleToLLVMIR(*module, gIR->context());
  if (!m) {
    error(loc, "Failed to translate MLIR IR to LLVM IR.");
    fatal();
  }

  return m;
}
#endif // LDC_MLIR_ENABLED

/// Links the module containing the inline function `fnName` into the current
/// module, calls that function with the runtime arguments and returns the
/// result.
DValue *linkAndCallInlineFunction(std::unique_ptr<llvm::Module> m,
                                  llvm::StringRef fnName,
                                  FuncDeclaration *fdecl,
                                  Expressions *arguments,
                                  LLValue *sretPointer) {
  m->setDataLayout(gIR->module.getDataLayout());
  llvm::Linker(gIR->module).linkInModule(std::move(m));

  llvm::Function *fun = gIR->module.getFunction(fnName);
  assert(fun);

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
  args.reserve(arguments->length);
  for (auto arg : *arguments) {
    args.push_back(DtoRVal(arg));
  }

  llvm::Value *rv = gIR->ir->CreateCall(fun, args);
  Type *type = fdecl->type->nextOf();

  if (sretPointer) {
    DtoStore(rv, sretPointer);
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

} // anonymous namespace

void DtoCheckInlineIRPragma(Identifier *ident, Dsymbol *s) {
  checkInlinePragmaTemplate(ident, s, /*allowPrefixSuffix=*/true);
}

void DtoCheckInlineMLIRPragma(Identifier *ident, Dsymbol *s) {
  checkInlinePragmaTemplate(ident, s, /*allowPrefixSuffix=*/false);
}

DValue *DtoInlineIRExpr(Loc loc, FuncDeclaration *fdecl, Expressions *arguments,
                        LLValue *sretPointer) {
  IF_LOG Logger::println("DtoInlineIRExpr @ %s", loc.toChars());
  LOG_SCOPE;

  // Generate a random new function name. Because the inlineIR function is
  // always inlined, this name does not escape the current compiled module; not
  // even at -O0.
  static size_t namecounter = 0;
  std::string mangled_name = "inline.ir." + ldc::to_string(namecounter++);
  TemplateInstance *tinst = fdecl->parent->isTemplateInstance();
  assert(tinst);

  const InlineIRArgs inlineIRArgs = extractInlineIRArgs(tinst, "LDC_inline_ir");
  std::unique_ptr<llvm::Module> m =
      parseInlineLLVMIR(inlineIRArgs, mangled_name, tinst->loc);
  return linkAndCallInlineFunction(std::move(m), mangled_name, fdecl, arguments,
                                   sretPointer);
}

DValue *DtoInlineMLIR(Loc loc, FuncDeclaration *fdecl, Expressions *arguments,
                      LLValue *sretPointer) {
#if !LDC_MLIR_ENABLED
  error(loc, "Inline MLIR is not supported by this LDC build (built without "
             "MLIR support)");
  fatal();
#else
  IF_LOG Logger::println("DtoInlineMLIR @ %s", loc.toChars());
  LOG_SCOPE;

  // Generate a random new function name. Because the inlineMLIR function is
  // always inlined, this name does not escape the current compiled module; not
  // even at -O0.
  static size_t namecounter = 0;
  std::string mangled_name = "inline.mlir." + ldc::to_string(namecounter++);
  TemplateInstance *tinst = fdecl->parent->isTemplateInstance();
  assert(tinst);

  const InlineIRArgs inlineIRArgs =
      extractInlineIRArgs(tinst, "LDC_inline_mlir");
  std::unique_ptr<llvm::Module> m =
      parseInlineMLIR(inlineIRArgs, mangled_name, loc);
  return linkAndCallInlineFunction(std::move(m), mangled_name, fdecl, arguments,
                                   sretPointer);
#endif // LDC_MLIR_ENABLED
}
