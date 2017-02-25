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
#include "ddmd/template.h"
#include "id.h"

// In @compute code only calls to other function in @compute core are allowed.
// However, a @kernel function taking a template alias function parameter is
// allowed, but while the alias appears in the symbol table of the module of the
// template declaration, it's module of origin is the module at the point of
// instansiation so we need to check for that.
FuncDeclaration *currentKernel;

bool isNonComputeCallExpVaild(CallExp *ce) {
  if (currentKernel == nullptr)
    return false;

  TemplateInstance* inst;
  if (!(inst =currentKernel->isInstantiated()))
    return false;

  FuncDeclaration* f = ce->f;
  Objects *tiargs = inst->tiargs;
  size_t i = 0,len = tiargs->dim;
  IF_LOG Logger::println("checking against: %s (%p) (dyncast=%d)",
                  f->toPrettyChars(),(void*)f, f->dyncast());
  LOG_SCOPE
  for (; i < len; i++) {
    RootObject *o = (*tiargs)[i];
    int d = o->dyncast();
    if (d != DYNCAST_EXPRESSION)
      continue;
    Expression *e = (Expression*)o;
    if (e->op != TOKfunction)
      continue;
    if (f->equals((((FuncExp*)e)->fd))) {
      IF_LOG Logger::println("match");
      return true;
    }
  }
  return false;
}

struct DComputeSemanticAnalyser : public StoppableVisitor {
  // Keep track of the outermost (i.e. not nested) function


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
          !strcmp(ce->f->ident->toChars(), "__dcompute_reflect"))
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
    if (!e->f)
      return;
    Module *m = e->f->getModule();
    if (m == nullptr || ((hasComputeAttr(m) == DComputeCompileFor::hostOnly)
        && !isNonComputeCallExpVaild(e))) {
      e->error("can only call functions from other @compute modules in "
               "@compute code");
      stop = true;
    }
  }
    
  void visit(FuncDeclaration * fd) override {
    if (hasKernelAttr(fd)) {
      if (fd->vthis) {
        fd->error("@kernel functions msut not require 'this'");
        stop = true;
        return;
      }
        
      Logger::println("current kernel = %s",fd->toChars());
      currentKernel = fd;
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
    currentKernel = nullptr;
    dsym->accept(&r);
  }
}
