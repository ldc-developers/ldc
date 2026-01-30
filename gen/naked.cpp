//===-- naked.cpp ---------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "dmd/declaration.h"
#include "dmd/errors.h"
#include "dmd/expression.h"
#include "dmd/identifier.h"
#include "dmd/mangle.h"
#include "dmd/statement.h"
#include "dmd/template.h"
#include "gen/dvalue.h"
#include "gen/funcgenstate.h"
#include "gen/functions.h"
#include "gen/irstate.h"
#include "gen/llvm.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/mangling.h"
#include "gen/tollvm.h"
#include "ir/irfunction.h"
#include "llvm/IR/InlineAsm.h"
#include <cassert>
#include <sstream>

using namespace dmd;

////////////////////////////////////////////////////////////////////////////////
// FIXME: Integrate these functions
void AsmStatement_toNakedIR(InlineAsmStatement *stmt, IRState *irs);

////////////////////////////////////////////////////////////////////////////////

class ToNakedIRVisitor : public Visitor {
  IRState *irs;

public:
  explicit ToNakedIRVisitor(IRState *irs) : irs(irs) {}

  //////////////////////////////////////////////////////////////////////////

  // Import all functions from class Visitor
  using Visitor::visit;

  //////////////////////////////////////////////////////////////////////////

  void visit(Statement *stmt) override {
    error(stmt->loc, "Statement not allowed in naked function");
  }

  //////////////////////////////////////////////////////////////////////////

  void visit(InlineAsmStatement *stmt) override {
    AsmStatement_toNakedIR(stmt, irs);
  }

  //////////////////////////////////////////////////////////////////////////

  void visit(CompoundStatement *stmt) override {
    IF_LOG Logger::println("CompoundStatement::toNakedIR(): %s",
                           stmt->loc.toChars());
    LOG_SCOPE;

    if (stmt->statements) {
      for (auto s : *stmt->statements) {
        if (s) {
          s->accept(this);
        }
      }
    }
  }

  //////////////////////////////////////////////////////////////////////////

  void visit(ExpStatement *stmt) override {
    IF_LOG Logger::println("ExpStatement::toNakedIR(): %s",
                           stmt->loc.toChars());
    LOG_SCOPE;

    // This happens only if there is a ; at the end:
    // asm { naked; ... };
    // Is this a legal AST?
    if (!stmt->exp) {
      return;
    }

    // only expstmt supported in declarations
    if (!stmt->exp || stmt->exp->op != EXP::declaration) {
      visit(static_cast<Statement *>(stmt));
      return;
    }

    DeclarationExp *d = static_cast<DeclarationExp *>(stmt->exp);
    VarDeclaration *vd = d->declaration->isVarDeclaration();
    FuncDeclaration *fd = d->declaration->isFuncDeclaration();
    EnumDeclaration *ed = d->declaration->isEnumDeclaration();

    // and only static variable/function declaration
    // no locals or nested stuffies!
    if (!vd && !fd && !ed) {
      visit(static_cast<Statement *>(stmt));
      return;
    }
    if (vd && !(vd->storage_class & (STCstatic | STCmanifest))) {
      error(vd->loc, "non-static variable `%s` not allowed in naked function",
            vd->toChars());
      return;
    }
    if (fd && !fd->isStatic()) {
      error(fd->loc,
            "non-static nested function `%s` not allowed in naked function",
            fd->toChars());
      return;
    }
    // enum decls should always be safe

    // make sure the symbols gets processed
    // TODO: codegen() here is likely incorrect
    Declaration_codegen(d->declaration, irs);
  }

  //////////////////////////////////////////////////////////////////////////

  void visit(LabelStatement *stmt) override {
    IF_LOG Logger::println("LabelStatement::toNakedIR(): %s",
                           stmt->loc.toChars());
    LOG_SCOPE;

    // Use printLabelName to match how label references are generated in asm-x86.h.
    // This ensures label definitions match the quoted format used in jump instructions.
    printLabelName(irs->nakedAsm, mangleExact(irs->func()->decl),
                   stmt->ident->toChars());
    irs->nakedAsm << ":\n";

    if (stmt->statement) {
      stmt->statement->accept(this);
    }
  }
};

////////////////////////////////////////////////////////////////////////////////

void DtoDefineNakedFunction(FuncDeclaration *fd) {
  IF_LOG Logger::println("DtoDefineNakedFunction(%s)", mangleExact(fd));
  LOG_SCOPE;

  // Get the proper IR mangle name (includes Windows calling convention decoration)
  TypeFunction *tf = fd->type->isTypeFunction();
  const std::string irMangle = getIRMangledName(fd, tf ? tf->linkage : LINK::d);

  // Get or create the LLVM function first, before visiting the body.
  // The visitor may call Declaration_codegen which needs an IR insert point.
  llvm::Module &module = gIR->module;
  llvm::Function *func = module.getFunction(irMangle);

  if (!func) {
    // Create function type using the existing infrastructure
    llvm::FunctionType *funcType = DtoFunctionType(fd);

    // Create the function with ExternalLinkage initially.
    // setLinkage() below will set the correct linkage.
    func = llvm::Function::Create(funcType, llvm::GlobalValue::ExternalLinkage,
                                  irMangle, &module);
  } else if (!func->empty()) {
    // Function already has a body - this can happen if the function was
    // already defined (e.g., template instantiation in another module).
    // Don't add another body.
    return;
  } else if (func->hasFnAttribute(llvm::Attribute::Naked)) {
    // Function already has naked attribute - it was already processed
    return;
  }

  // Set linkage and visibility using the standard infrastructure.
  // This correctly handles:
  // - Lambdas (internal linkage, no dllexport)
  // - Templates (weak_odr linkage with COMDAT)
  // - Exported functions (dllexport on Windows)
  // - Regular functions (external linkage)
  setLinkage(DtoLinkage(fd), func);
  setVisibility(fd, func);

  // Set naked attribute - this tells LLVM not to generate prologue/epilogue
  func->addFnAttr(llvm::Attribute::Naked);

  // Prevent optimizations that might clone or modify the function.
  // The inline asm contains labels that would conflict if duplicated.
  func->addFnAttr(llvm::Attribute::OptimizeNone);
  func->addFnAttr(llvm::Attribute::NoInline);

  // Set other common attributes
  func->addFnAttr(llvm::Attribute::NoUnwind);

  // Create entry basic block and set insert point before visiting body.
  // The visitor's ExpStatement::visit may call Declaration_codegen for
  // static symbols, which may need an active IR insert point.
  llvm::BasicBlock *entryBB =
      llvm::BasicBlock::Create(gIR->context(), "entry", func);

  // Save current insert point and switch to new function.
  // Use gIR->setInsertPoint() instead of gIR->ir->SetInsertPoint() because
  // the latter goes through IRBuilderHelper::operator->() which asserts that
  // there's a valid insert block. At module scope, there may not be one yet.
  // gIR->setInsertPoint() accesses the builder directly and also returns an
  // RAII guard that restores the previous state when it goes out of scope.
  const auto savedInsertPoint = gIR->setInsertPoint(entryBB);

  // Clear the nakedAsm stream and collect the function body
  std::ostringstream &asmstr = gIR->nakedAsm;
  asmstr.str("");

  // Use the visitor to collect asm statements into nakedAsm
  ToNakedIRVisitor visitor(gIR);
  fd->fbody->accept(&visitor);

  if (global.errors) {
    fatal();
  }

  // Get the collected asm string and escape $ characters for LLVM inline asm.
  // In LLVM inline asm, $N refers to operand N, so literal $ must be escaped as $$.
  std::string asmBody;
  {
    std::string raw = asmstr.str();
    asmBody.reserve(raw.size() * 2);  // Worst case: all $ characters
    for (char c : raw) {
      if (c == '$') {
        asmBody += "$$";
      } else {
        asmBody += c;
      }
    }
  }
  asmstr.str("");  // Clear for potential reuse

  // Create inline asm - the entire function body is a single asm block
  // No constraints needed since naked functions handle everything in asm
  llvm::FunctionType *asmFuncType =
      llvm::FunctionType::get(llvm::Type::getVoidTy(gIR->context()), false);

  llvm::InlineAsm *inlineAsm = llvm::InlineAsm::get(
      asmFuncType,
      asmBody,
      "",      // No constraints
      true,    // Has side effects
      false,   // Not align stack
      llvm::InlineAsm::AD_ATT  // AT&T syntax
  );

  gIR->ir->CreateCall(inlineAsm);

  // Naked functions don't return normally through LLVM IR
  gIR->ir->CreateUnreachable();

  // The savedInsertPoint RAII guard automatically restores the insert point
  // when it goes out of scope.
}

////////////////////////////////////////////////////////////////////////////////

void emitABIReturnAsmStmt(IRAsmBlock *asmblock, Loc loc,
                          FuncDeclaration *fdecl) {
  IF_LOG Logger::println("emitABIReturnAsmStmt(%s)", mangleExact(fdecl));
  LOG_SCOPE;

  auto as = new IRAsmStmt;

  LLType *llretTy = DtoType(fdecl->type->nextOf());
  asmblock->retty = llretTy;
  asmblock->retn = 1;

  // FIXME: This should probably be handled by the TargetABI somehow.
  //        It should be able to do this for a greater variety of types.

  const auto &triple = *global.params.targetTriple;
  Type *const rt = fdecl->type->nextOf()->toBasetype();

  // x86
  if (triple.getArch() == llvm::Triple::x86) {
    if (rt->isIntegral() || rt->ty == TY::Tpointer || rt->ty == TY::Tclass ||
        rt->ty == TY::Taarray) {
      if (size(rt) == 8) {
        as->out.c = "=A,";
      } else {
        as->out.c = "={ax},";
      }
    } else if (rt->isFloating()) {
      if (rt->isComplex()) {
        if (fdecl->_linkage() == LINK::d) {
          // extern(D) always returns on the FPU stack
          as->out.c = "={st},={st(1)},";
          asmblock->retn = 2;
        } else if (rt->ty == TY::Tcomplex32) {
          // non-extern(D) cfloat is returned as i64
          as->out.c = "=A,";
          asmblock->retty = LLType::getInt64Ty(gIR->context());
        } else {
          // non-extern(D) cdouble and creal are returned via sret
          // don't add anything!
          asmblock->retty = LLType::getVoidTy(gIR->context());
          asmblock->retn = 0;
          return;
        }
      } else {
        as->out.c = "={st},";
      }
    } else if (rt->ty == TY::Tarray || rt->ty == TY::Tdelegate) {
      as->out.c = "={ax},={dx},";
      asmblock->retn = 2;
#if 0
      // this is to show how to allocate a temporary for the return value
      // in case the appropriate multi register constraint isn't supported.
      // this way abi return from inline asm can still be emulated.
      // note that "$<<out0>>" etc in the asm will translate to the correct
      // numbered output when the asm block in finalized

      // generate asm
      as->out.c = "=*m,=*m,";
      LLValue* tmp = DtoRawAlloca(llretTy, 0, ".tmp_asm_ret");
      as->out.push_back( tmp );
      as->out.push_back( DtoGEP(tmp, 0, 1) );
      as->code = "movd %eax, $<<out0>>" "\n\t" "mov %edx, $<<out1>>";

      // fix asmblock
      asmblock->retn = 0;
      asmblock->retemu = true;
      asmblock->asmBlock->abiret = tmp;

      // add "ret" stmt at the end of the block
      asmblock->s.push_back(as);

      // done, we don't want anything pushed in the front of the block
      return;
#endif
    } else {
      error(loc, "unimplemented return type `%s` for implicit abi return",
            rt->toChars());
      fatal();
    }
  }

  // x86_64
  else if (triple.getArch() == llvm::Triple::x86_64) {
    if (rt->isIntegral() || rt->ty == TY::Tpointer || rt->ty == TY::Tclass ||
        rt->ty == TY::Taarray) {
      as->out.c = "={ax},";
    } else if (rt->isFloating()) {
      const bool isWin64 = triple.isOSWindows();

      if (rt == Type::tcomplex80 && !isWin64) {
        // On x87 stack, re=st, im=st(1)
        as->out.c = "={st},={st(1)},";
        asmblock->retn = 2;
      } else if ((rt == Type::tfloat80 || rt == Type::timaginary80) &&
                 !triple.isWindowsMSVCEnvironment()) {
        // On x87 stack
        as->out.c = "={st},";
      } else if (rt == Type::tcomplex32) {
        if (isWin64) {
          // cfloat on Win64 -> %rax
          as->out.c = "={ax},";
          asmblock->retty = LLType::getInt64Ty(gIR->context());
        } else {
          // cfloat on Posix -> %xmm0 (extract two floats)
          as->out.c = "={xmm0},";
          asmblock->retty = LLType::getDoubleTy(gIR->context());
        }
      } else if (rt->isComplex()) {
        if (isWin64) {
          // Win64: cdouble and creal are returned via sret
          // don't add anything!
          asmblock->retty = LLType::getVoidTy(gIR->context());
          asmblock->retn = 0;
          return;
        } else {
          // cdouble on Posix -> re=%xmm0, im=%xmm1
          as->out.c = "={xmm0},={xmm1},";
          asmblock->retn = 2;
        }
      } else {
        // Plain float/double/ifloat/idouble
        as->out.c = "={xmm0},";
      }
    } else if (rt->ty == TY::Tarray || rt->ty == TY::Tdelegate) {
      as->out.c = "={ax},={dx},";
      asmblock->retn = 2;
    } else {
      error(loc, "unimplemented return type `%s` for implicit abi return",
            rt->toChars());
      fatal();
    }
  }

  // unsupported
  else {
    error(loc,
          "this target (%s) does not implement inline asm falling off the end "
          "of the function",
          triple.str().c_str());
    fatal();
  }

  // return values always go in the front
  asmblock->s.push_front(as);
}

////////////////////////////////////////////////////////////////////////////////

// sort of kinda related to naked ...

DValue *DtoInlineAsmExpr(Loc loc, FuncDeclaration *fd,
                         Expressions *arguments, LLValue *sretPointer) {
  assert(fd->toParent()->isTemplateInstance() && "invalid inline __asm expr");
  assert(arguments->length >= 2 && "invalid __asm call");

  // get code param
  Expression *e = (*arguments)[0];
  IF_LOG Logger::println("code exp: %s", e->toChars());
  StringExp *se = static_cast<StringExp *>(e);
  if (e->op != EXP::string_ || se->sz != 1) {
    error(e->loc, "`__asm` code argument is not a `char[]` string literal");
    fatal();
  }
  const DString codeStr = se->peekString();
  const llvm::StringRef code = {codeStr.ptr, codeStr.length};

  // get constraints param
  e = (*arguments)[1];
  IF_LOG Logger::println("constraint exp: %s", e->toChars());
  se = static_cast<StringExp *>(e);
  if (e->op != EXP::string_ || se->sz != 1) {
    error(e->loc,
          "`__asm` constraints argument is not a `char[]` string literal");
    fatal();
  }
  const DString constraintsStr = se->peekString();
  const llvm::StringRef constraints = {constraintsStr.ptr,
                                       constraintsStr.length};

  auto constraintInfo = llvm::InlineAsm::ParseConstraints(constraints);
  // build runtime arguments
  const size_t n = arguments->length - 2;
  LLSmallVector<LLValue *, 8> operands;
  LLSmallVector<LLType *, 8> indirectTypes;
  operands.reserve(n);

  Type *returnType = fd->type->nextOf();
  const size_t cisize = constraintInfo.size();
  const size_t minRequired = n + (returnType->ty == TY::Tvoid ? 0 : 1);
  if (cisize < minRequired) {
    error(se->loc,
          "insufficient number of constraints (%zu) for number of additional "
          "arguments %s(%zu)",
          cisize, returnType->ty == TY::Tvoid ? "" : "and return type ",
          minRequired);
    fatal();
  }

  size_t i = 0;
  for (; i < n; i++) {
    Expression *ee = (*arguments)[2 + i];
    operands.push_back(DtoRVal(ee));
    if (constraintInfo[i].isIndirect) {
      if (TypePointer *pt = ee->type->isTypePointer())
        indirectTypes.push_back(DtoType(pt->nextOf()));
      else
        indirectTypes.push_back(DtoType(ee->type));
    }
  }

  LLType *irReturnType = DtoType(returnType->toBasetype());

  for (; i < cisize; i++) {
    if (!constraintInfo[i].isIndirect)
      continue;
    if (constraintInfo[i].Type == llvm::InlineAsm::ConstraintPrefix::isOutput) {
      indirectTypes.push_back(DtoType(returnType));
    } else {
      error(loc, "indirect constraint %d doesn't correspond to an argument or output", (unsigned)i);
      fatal();
    }
  }

  LLValue *rv =
      DtoInlineAsmExpr(loc, code, constraints, operands, indirectTypes, irReturnType);

  // work around missing tuple support for users of the return value
  if (sretPointer || returnType->ty == TY::Tstruct) {
    auto lvalue = sretPointer;
    if (!lvalue)
      lvalue = DtoAlloca(returnType, ".__asm_tuple_ret");
    DtoStore(rv, lvalue);
    return new DLValue(returnType, lvalue);
  }

  // return call as im value
  return new DImValue(returnType, rv);
}

llvm::CallInst *DtoInlineAsmExpr(Loc loc, llvm::StringRef code,
                                 llvm::StringRef constraints,
                                 llvm::ArrayRef<llvm::Value *> operands,
                                 llvm::ArrayRef<llvm::Type *> indirectTypes,
                                 llvm::Type *returnType) {
  IF_LOG Logger::println("DtoInlineAsmExpr @ %s", loc.toChars());
  LOG_SCOPE;

  LLSmallVector<LLType *, 8> operandTypes;
  operandTypes.reserve(operands.size());
  for (auto *o : operands)
    operandTypes.push_back(o->getType());

  // build asm function type
  llvm::FunctionType *FT =
      llvm::FunctionType::get(returnType, operandTypes, false);

  if (auto err = llvm::InlineAsm::verify(FT, constraints)) {
    error(loc, "inline asm constraints are invalid");
    llvm::errs() << err;
    fatal();
  }

  // build asm call
  bool sideeffect = true;
  llvm::InlineAsm *ia = llvm::InlineAsm::get(FT, code, constraints, sideeffect);

  auto call = gIR->createInlineAsmCall(loc, ia, operands, indirectTypes);

  return call;
}
