//===-- gen/inlinemlir.cpp - Inline MLIR implementation-----------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.

#if LDC_MLIR_ENABLED

#include "gen/inlinemlir.h"
#include "gen/dvalue.h"
#include "gen/irstate.h"
#include "gen/llvm.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/to_string.h"
#include "gen/tollvm.h"
#include "dmd/arraytypes.h"
#include "dmd/declaration.h"
#include "dmd/identifier.h"
#include "dmd/template.h"
#include "dmd/errors.h"
#include "expression.h"
#include "mtype.h"
#include <cassert>
#include <memory>
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/raw_ostream.h"

// Dialects
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include <mlir/IR/Types.h>
#include "mlir/Support/LLVM.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Target/LLVMIR/TypeFromLLVM.h"

using namespace dmd;

namespace {
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
} // anonymous namespace

void DtoCheckInlineMLIRPragma(Identifier *ident, Dsymbol *s) {
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
        ident->toChars()
      );
      fatal();
    }

    // The magic inlineMLIR template is one of
    // pragma(LDC_inline_mlir)
    //  R inlineMLIR(string code, R, P...)(P);
    TemplateParameters &params = *td->parameters;
    bool valid_params = (params.length == 3) &&
                        params[params.length - 2]->isTemplateTypeParameter() &&
                        params[params.length - 1]->isTemplateTupleParameter();

    if (valid_params) {
      for(d_size_t i = 0;i < (params.length - 2); i++) {
        TemplateValueParameter *p0 = params[i]->isTemplateValueParameter();
        valid_params = valid_params && p0 && p0->valType == Type::tstring;
      }
    }

    if (!valid_params) {
      error(s->loc,
        "the `%s` pragma template must have three "
        "(string, type and type tuple) parameters",
        ident->toChars());
      fatal();
    }
  } else {
    error(s->loc,
      "the `%s` pragma is only allowed on template declarations",
      ident->toChars());
    fatal();
  }
}

DValue *DtoInlineMLIR(Loc loc, FuncDeclaration *fdecl, Expressions *arguments, llvm::Value *sretPointer) {
#if !LDC_MLIR_ENABLED
  error(loc, "Inline MLIR is not supported by this LDC build (built without MLIR support)");
  fatal();
#else
  {
    IF_LOG Logger::println("DtoInlineMLIR @ %s", loc.toChars());
    LOG_SCOPE;
  }

  {
    IF_LOG Logger::println(" runtime arguments: %s", arguments ? arguments->toChars(): "");
    LOG_SCOPE;
  }

  // Generate a random new function name. Because the inlineMLIR function is
  // always inlined, this name does not escape the current compiled module; not
  // even at -O0.
  static size_t namecounter = 0;
  std::string mangled_name = "inline.mlir." + ldc::to_string(namecounter++);
  TemplateInstance *tinst = fdecl->parent->isTemplateInstance();
  assert(tinst);

  // 1. Define the inline function (define a new function for each call);
  {
      // The magic inlineMLIR template:
      // pragma(LDC_inline_mlir)
      // R inlineMLIR(string code, R, P...)(P);
      Objects &objects = tinst->tdtypes;

      assert(objects.length == 3);

      Expression *a0 = isExpression(objects[0]);
      assert(a0);

      llvm::StringRef code;

      StringExp *strexp = toStringExp(a0);

      code = exprToString(strexp);

      {
          int length = static_cast<int>(code.size());

          IF_LOG Logger::println("Extracted MLIR code: \n%.*s", length, code.data());
          LOG_SCOPE;
      }

      Type *ret = isType(objects[1]);
      assert(ret);

      Tuple *args = isTuple(objects[2]);
      assert(args);

      Objects &arg_types = args->objects;

      mlir::DialectRegistry registry;
      mlir::registerAllToLLVMIRTranslations(registry);
      registry.insert<mlir::func::FuncDialect,
                      mlir::arith::ArithDialect,
                      mlir::cf::ControlFlowDialect,
                      mlir::scf::SCFDialect,
                      mlir::LLVM::LLVMDialect>();

      mlir::MLIRContext context(registry);
      mlir::LLVM::TypeFromLLVMIRTranslator typeTranslator(context);

      context.loadDialect<mlir::func::FuncDialect,
                          mlir::arith::ArithDialect,
                          mlir::cf::ControlFlowDialect,
                          mlir::scf::SCFDialect,
                          mlir::LLVM::LLVMDialect>();

      std::string str;
      llvm::raw_string_ostream stream(str);

      // Declare mlir function using `func` dialect
      stream << "func.func " << "@" << mangled_name << "(";

      for(size_t i = 0; i < arg_types.length; i ++) {
        Type *ty = isType(arg_types[i]);

        if (!ty) {
          error(tinst->loc, "All parameters of a template defined with pragma "
                            "`LDC_inline_mlir`, except for the first one, should be types");
          fatal();
        }

        if (i != 0) {
          stream << ", ";
        }

        mlir::Type mlir_ty = typeTranslator.translateType(DtoType(ty));

        stream << "%arg" << i << ": " << mlir_ty;
      }

      stream << ")";

      if (ret->ty != TY::Tvoid) {
        mlir::Type mlir_return_ty = typeTranslator.translateType(DtoType(ret));
        stream << " -> " << mlir_return_ty;
      }

      stream << "\n{\n" << code;

      if (ret->ty == TY::Tvoid) {
        stream << "\n\treturn";
      }

      stream << "\n}";

      {
        IF_LOG Logger::println("Constructed function that later be inlined: \n%s", stream.str().c_str());
        LOG_SCOPE
      }

      mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(stream.str().c_str(), &context);

      if (!module) {
        error(loc, "failed to parse inline MLIR module");
        fatal();
      }

      if (failed(module->verify())) {
        error(loc, "inline MLIR module verification failed");
        fatal();
      }

      {
        IF_LOG Logger::println("Successfully parsed and verified inline MLIR module");
        LOG_SCOPE;
      }

      mlir::PassManager pm(&context);

      pm.addPass(mlir::createConvertSCFToCFPass());
      pm.addPass(mlir::createArithToLLVMConversionPass());
      pm.addPass(mlir::createConvertControlFlowToLLVMPass());
      pm.addPass(mlir::createConvertFuncToLLVMPass());

      // Clean up any remaining unnecessary conversion casts before lowering to LLVM IR
      pm.addPass(mlir::createReconcileUnrealizedCastsPass());

      if (mlir::failed(pm.run(*module))) {
        error(loc, "MLIR lowering pass pipeline failed!");
        fatal();
      }

      {
        IF_LOG Logger::println("Lowered MLIR IR to LLVM IR dialect: ");
        IF_LOG module->dump();
        LOG_SCOPE;
      }

      std::unique_ptr<llvm::Module> llvmModule =
      mlir::translateModuleToLLVMIR(*module, gIR->context());

      if (!llvmModule) {
        error(loc, "Failed to translate MLIR IR to LLVM IR.");
        fatal();
      }

      {
        IF_LOG Logger::println("Final Lowered LLVM IR: ");
        IF_LOG llvmModule->print(llvm::outs(), nullptr);
        LOG_SCOPE;
      }

      llvmModule->setDataLayout(gIR->module.getDataLayout());

      llvm::Linker(gIR->module).linkInModule(std::move(llvmModule));
    }
    // 2. Call the function that was just defined and return the returnValue
    {
      llvm::Function *fun = gIR->module.getFunction(mangled_name);

      // Apply some parent function attributes to the inlineMLIR function too. This
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
      llvm::SmallVector<llvm::Value*, 8> args;
      args.reserve(arguments->length);

      for(auto arg: *arguments) {
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
#endif
}
#endif // MLIR_ENABLED
