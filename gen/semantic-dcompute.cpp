//===-- semantic-dcompute.cpp ---------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Validation for @compute code:
//      enforce: @nogc, nothrow, all function calls are to modules that are also
//          @compute. The enforcemnt of nothrow is simpler because all functions
//          are assumed to not throw. We only need to check for ThrowStatement.
//          We dont use dmd's nothrow detection because it still allows errors.
//
//      ban: classes, interfaces, asm, typeid, global variables, synhronized,
//          associative arrays, pragma(lib,...)
//
//===----------------------------------------------------------------------===//

#include "gen/recursivevisitor.h"
#include "gen/uda.h"
#include "gen/dcomputetarget.h"
#include "gen/logger.h"
#include "ddmd/declaration.h"
#include "ddmd/module.h"
#include "ddmd/identifier.h"
#include "id.h"

struct DComputeSemanticAnalyser : public StoppableVisitor {

  void visit(InterfaceDeclaration *decl) override {
    decl->error("interfaces and classes not allowed in @compute code");
    stop = true;
  }
  void visit(ClassDeclaration *decl) override {
    decl->error("interfaces and classes not allowed in @compute code");
    stop = true;
  }
  void visit(VarDeclaration *decl) override {
    // Don't print multiple errors for 'synchronized'. see visit(CallExp*)
    if (decl->isDataseg() && strncmp(decl->toChars(), "__critsec", 9)) {
      decl->error("global variables not allowed in @compute code");
      stop = true;
      return;
    }

    if (decl->type->ty == Taarray) {
      decl->error("associative arrays not allowed in @compute code");
      stop = true;
    }
    // includes interfaces
    else if (decl->type->ty == Tclass)
    {
      decl->error("interfaces and classes not allowed in @compute code");
    }
  }
  void visit(PragmaDeclaration *decl) override {
    if (decl->ident == Id::lib) {
      decl->error(
          "linking additional libraries not supported in @compute code");
      stop = true;
    }
  }

  // Nogc enforcement.
  // No need to check AssocArrayLiteral because AA's are banned anyway
  void visit(ArrayLiteralExp *e) override {
    if (e->type->ty != Tarray || !e->elements || !e->elements->dim)
      return;
    e->error("array literal in @compute code not allowed");
    stop = true;
  }
  void visit(NewExp *e) override {
    e->error("cannot use 'new' in @compute code");
    stop = true;
  }

  void visit(DeleteExp *e) override {
    e->error("cannot use 'delete' in @compute code");
    stop = true;
  }
  // No need to check IndexExp because AA's are banned anyway
  void visit(AssignExp *e) override {
    if (e->e1->op == TOKarraylength) {
      e->error("setting 'length' in @compute code not allowed");
      stop = true;
    }
  }

  void visit(CatAssignExp *e) override {
    e->error("cannot use operator ~= in @compute code");
    stop = true;
  }
  void visit(CatExp *e) override {
    e->error("cannot use operator ~ in @compute code");
    stop = true;
  }
  // Ban typeid(T)
  void visit(TypeidExp *e) override {
    e->error("typeinfo not available in @compute code");
    stop = true;
  }

  void visit(StringExp *e) override {
    e->error("string literals not allowed in @compue code");
    stop = true;
  }
  void visit(CompoundAsmStatement *e) override {
    e->error("asm not allowed in @compute code");
    stop = true;
  }
  void visit(AsmStatement *e) override {
    e->error("asm not allowed in @compute code");
    stop = true;
  }

  // Enforce nothrow. Disallow 'catch' as it is dead code.
  // try...finally is allowed to facilitate scope(exit)
  void visit(TryCatchStatement *e) override {
    e->error("no exceptions in @compute code");
    stop = true;
  }
  void visit(ThrowStatement *e) override {
    e->error("no exceptions in @compute code");
    stop = true;
  }
  void visit(SwitchStatement *e) override {
    if (!e->condition->type->isintegral()) {
      e->error("cannot switch on strings in @compute code");
      stop = true;
    }
  }

  void visit(IfStatement *stmt) override {
    // Code inside an if(__dcompute_reflect(0,0)) { ...} is explicitly
    // for the host and is therefore allowed to call non @compute functions.
    // Thus, the if-statement body's code should not be checked for
    // @compute semantics and the recursive visitor should stop here.
    if (stmt->condition->op == TOKcall) {
      auto ce = (CallExp *)stmt->condition;
      if (ce->f && ce->f->ident &&
          !strcmp(ce->f->ident->string, "__dcompute_reflect"))
      {
        auto arg1 = (DComputeTarget::ID)(*ce->arguments)[0]->toInteger();
        if (arg1 == DComputeTarget::Host)
          stop = true;
      }
    }
  }
  void visit(CallExp *e) override {
    // SynchronizedStatement is lowered to
    //    Critsec __critsec105; // 105 == line number
    //    _d_criticalenter(& __critsec105); <--
    //    ...                                 |
    //    _d_criticalexit( & __critsec105);   |
    // So we intercept it with the CallExp ----

    if (!strncmp(e->toChars(), "_d_criticalenter", 16)) {
      e->error("cannot use 'synchronized' in @compute code");
      stop = true;
      return;
    }

    if (!strncmp(e->toChars(), "_d_criticalexit", 15)) {
      stop = true;
      return;
    }
    Module *m = e->f->getModule();
    if (m == nullptr || hasComputeAttr(m) == DComputeCompileFor::hostOnly) {
      e->error("can only call functions from other @compute modules in "
               "@compute code");
      stop = true;
    }
  }

  // Override the default assert(0) behavior of Visitor:
  void visit(Statement *) override {}   // do nothing
  void visit(Expression *) override {}  // do nothing
  void visit(Declaration *) override {} // do nothing
  void visit(Initializer *) override {} // do nothing
  void visit(Dsymbol *) override {}     // do nothing
};

void dcomputeSemanticAnalysis(Module *m) {
  DComputeSemanticAnalyser v;
  RecursiveWalker r(&v);
  for (unsigned k = 0; k < m->members->dim; k++) {
    Dsymbol *dsym = (*m->members)[k];
    assert(dsym);
    IF_LOG Logger::println("dcomputeSema: %s: %s", m->toPrettyChars(),
                           dsym->toPrettyChars());
    LOG_SCOPE
    dsym->accept(&r);
  }
}
