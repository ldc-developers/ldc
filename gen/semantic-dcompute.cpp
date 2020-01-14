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
//   enforce: @nogc, nothrow, all function calls are to modules that are also
//   @compute. The enforcemnt of nothrow is simpler because all functions
//   are assumed to not throw. We only need to check for ThrowStatement.
//   We dont use dmd's nothrow detection because it still allows errors.
//
//   ban: classes, interfaces, asm, typeid, global variables, synhronized,
//        associative arrays, pragma(lib,...)
//
//===----------------------------------------------------------------------===//

#include "dmd/declaration.h"
#include "dmd/expression.h"
#include "dmd/id.h"
#include "dmd/identifier.h"
#include "dmd/module.h"
#include "dmd/template.h"
#include "gen/dcompute/target.h"
#include "gen/logger.h"
#include "gen/recursivevisitor.h"
#include "gen/uda.h"

struct DComputeSemanticAnalyser : public StoppableVisitor {
  FuncDeclaration *currentFunction;
  // In @compute code only calls to other functions in `@compute` code are
  // allowed.
  // However, a @kernel function taking a template alias function parameter is
  // allowed, but while the alias appears in the symbol table of the module of
  // the
  // template declaration, it's module of origin is the module at the point of
  // instansiation so we need to check for that.
  bool isNonComputeCallExpVaild(CallExp *ce) {
    FuncDeclaration *f = ce->f;
    if (f->ident == Id::dcReflect)
      return true;
    if (currentFunction == nullptr)
      return false;
    TemplateInstance *inst = currentFunction->isInstantiated();
    if (!inst)
      return false;

    Objects *tiargs = inst->tiargs;
    size_t i = 0, len = tiargs->length;
    IF_LOG Logger::println("checking against: %s (%p) (dyncast=%d)",
                           f->toPrettyChars(), (void *)f, f->dyncast());
    LOG_SCOPE
    for (; i < len; i++) {
      RootObject *o = (*tiargs)[i];
      if (o->dyncast() != DYNCAST_EXPRESSION)
        continue;
      Expression *e = (Expression *)o;
      if (e->op != TOKfunction)
        continue;
      if (f->equals((((FuncExp *)e)->fd))) {
        IF_LOG Logger::println("match");
        return true;
      }
    }
    return false;
  }

  using StoppableVisitor::visit;

  void visit(InterfaceDeclaration *decl) override {
    decl->error("interfaces and classes not allowed in `@compute` code");
    stop = true;
  }

  void visit(ClassDeclaration *decl) override {
    decl->error("interfaces and classes not allowed in `@compute` code");
    stop = true;
  }

  void visit(VarDeclaration *decl) override {
    // Don't print multiple errors for 'synchronized'. see visit(CallExp*)
    if (decl->isDataseg()) {
      if (strncmp(decl->toChars(), "__critsec", 9) &&
        strncmp(decl->toChars(), "typeid", 6)) {
        decl->error("global variables not allowed in `@compute` code");
      }
      // Ignore typeid: it is ignored by codegen.
      stop = true;
      return;
    }

    if (decl->type->ty == Taarray) {
      decl->error("associative arrays not allowed in `@compute` code");
      stop = true;
    }
    // includes interfaces
    else if (decl->type->ty == Tclass) {
      decl->error("interfaces and classes not allowed in `@compute` code");
    }
  }
  void visit(PragmaDeclaration *decl) override {
    if (decl->ident == Id::lib) {
      decl->error(
          "linking additional libraries not supported in `@compute` code");
      stop = true;
    }
  }

  // Nogc enforcement.
  // No need to check AssocArrayLiteral because AA's are banned anyway
  void visit(ArrayLiteralExp *e) override {
    if (e->type->ty != Tarray || !e->elements || !e->elements->length)
      return;
    e->error("array literal in `@compute` code not allowed");
    stop = true;
  }
  void visit(NewExp *e) override {
    e->error("cannot use `new` in `@compute` code");
    stop = true;
  }

  void visit(DeleteExp *e) override {
    e->error("cannot use `delete` in `@compute` code");
    stop = true;
  }
  // No need to check IndexExp because AA's are banned anyway
  void visit(AssignExp *e) override {
    if (e->e1->op == TOKarraylength) {
      e->error("setting `length` in `@compute` code not allowed");
      stop = true;
    }
  }

  void visit(CatAssignExp *e) override {
    e->error("cannot use operator `~=` in `@compute` code");
    stop = true;
  }
  void visit(CatExp *e) override {
    e->error("cannot use operator `~` in `@compute` code");
    stop = true;
  }
  // Ban typeid(T)
  void visit(TypeidExp *e) override {
    e->error("typeinfo not available in `@compute` code");
    stop = true;
  }

  void visit(StringExp *e) override {
    e->error("string literals not allowed in `@compute` code");
    stop = true;
  }
  void visit(CompoundAsmStatement *e) override {
    e->error("asm not allowed in `@compute` code");
    stop = true;
  }
  void visit(AsmStatement *e) override {
    e->error("asm not allowed in `@compute` code");
    stop = true;
  }

  // Enforce nothrow. Disallow 'catch' as it is dead code.
  // try...finally is allowed to facilitate scope(exit)
  void visit(TryCatchStatement *e) override {
    e->error("no exceptions in `@compute` code");
    stop = true;
  }
  void visit(ThrowStatement *e) override {
    e->error("no exceptions in `@compute` code");
    stop = true;
  }
  void visit(SwitchStatement *e) override {
    if (auto ce = e->condition->isCallExp()) {
      if (ce->f->ident == Id::__switch) {
        e->error("cannot `switch` on strings in `@compute` code");
        stop = true;
      }
    }
  }

  void visit(IfStatement *stmt) override {
    // Don't descend into ctfe only code
    if (stmt->condition->op == TOKvar) {
      auto ve = (VarExp *)stmt->condition;
      if (ve->var->ident == Id::ctfe) {
        if (stmt->elsebody)
          visit(stmt->elsebody);
        stop = true;
      }
    } else if (stmt->condition->op == TOKnot) {
      auto ne = (NotExp *) stmt->condition;
      if (ne->e1->op == TOKvar) {
        auto ve = (VarExp * )ne->e1;
        if (ve->var->ident == Id::ctfe) {
          visit(stmt->ifbody);
        stop = true;
        }
      }
    }
    // Code inside an if(__dcompute_reflect(0,0)) { ...} is explicitly
    // for the host and is therefore allowed to call non @compute functions.
    // Thus, the if-statement body's code should not be checked for
    // @compute semantics and the recursive visitor should stop here.
    if (auto ce = stmt->condition->isCallExp()) {
      if (ce->f && ce->f->ident == Id::dcReflect) {
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

    if (e->f->ident == Id::criticalenter) {
      e->error("cannot use `synchronized` in `@compute` code");
      stop = true;
      return;
    }

    if (e->f->ident == Id::criticalexit) {
      stop = true;
      return;
    }
      
    Module *m = e->f->getModule();
    if ((m == nullptr || (hasComputeAttr(m) == DComputeCompileFor::hostOnly)) &&
        !isNonComputeCallExpVaild(e)) {
      e->error("can only call functions from other `@compute` modules in "
               "`@compute` code");
      stop = true;
    }
  }

  void visit(FuncDeclaration *fd) override {
    if (hasKernelAttr(fd) && fd->vthis) {
      fd->error("`@kernel` functions must not require `this`");
      stop = true;
      return;
    }

    IF_LOG Logger::println("current function = %s", fd->toChars());
    currentFunction = fd;
  }

  void visit(TemplateDeclaration*) override {
    // Don't try to analyse uninstansiated templates.
    stop = true;
  }

  void visit(TemplateInstance *ti) override {
    // object.RTInfo(Impl) template instantiations are skipped during codegen,
    // as they contain unsupported global variables.
    if (ti->tempdecl == Type::rtinfo || ti->tempdecl == Type::rtinfoImpl) {
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
  for (unsigned k = 0; k < m->members->length; k++) {
    Dsymbol *dsym = (*m->members)[k];
    assert(dsym);
    IF_LOG Logger::println("dcomputeSema: %s: %s", m->toPrettyChars(),
                           dsym->toPrettyChars());
    LOG_SCOPE
    v.currentFunction = nullptr;

    dsym->accept(&r);
  }
}
