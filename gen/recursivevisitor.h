//===-- gen/recursivevisitor.h - Code Coverage Analysis ---------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
//
// The RecursiveWalker is based on apply.c from DMD, with the following license:
//
// Compiler implementation of the D programming language
// Copyright (c) 1999-2014 by Digital Mars
// All Rights Reserved
// written by Walter Bright
// http://www.digitalmars.com
// Distributed under the Boost Software License, Version 1.0.
// http://www.boost.org/LICENSE_1_0.txt
// https://github.com/D-Programming-Language/dmd/blob/master/src/apply.c
//
// See the LICENSE file for more details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a RecursiveVisitor with default implementations that
// recursively visits all statements unless overridden.
// It also contains a RecursiveWalker, that recursively walks the tree
// (depth-first) and calls the visitor for each node before recursing deeper
// down the tree.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "dmd/attrib.h"
#include "dmd/declaration.h"
#include "dmd/errors.h"
#include "dmd/init.h"
#include "dmd/statement.h"
#include "dmd/visitor.h"

class RecursiveVisitor : public Visitor {
public:
  template <class T> void recurse(T *s) {
    if (s)
      s->accept(this);
  }

  template <class T> void recurse(Array<T> *stmts) {
    if (stmts) {
      for (auto s : *stmts) {
        recurse(s);
      }
    }
  }

  using Visitor::visit;

  void visit(CompoundStatement *stmt) override {
    for (auto s : *stmt->statements) {
      recurse(s);
    }
  }

  void visit(ReturnStatement *stmt) override { recurse(stmt->exp); }

  void visit(ExpStatement *stmt) override { recurse(stmt->exp); }

  void visit(IfStatement *stmt) override {
    recurse(stmt->condition);
    recurse(stmt->ifbody);
    recurse(stmt->elsebody);
  }

  void visit(ScopeStatement *stmt) override { recurse(stmt->statement); }

  void visit(WhileStatement *stmt) override {
    recurse(stmt->condition);
    recurse(stmt->_body);
  }

  void visit(DoStatement *stmt) override {
    recurse(stmt->_body);
    recurse(stmt->condition);
  }

  void visit(ForStatement *stmt) override {
    recurse(stmt->_init);
    recurse(stmt->condition);
    recurse(stmt->_body);
    recurse(stmt->increment);
  }

  void visit(ForeachStatement *stmt) override {
    recurse(stmt->aggr);
    recurse(stmt->_body);
  }

  void visit(ForeachRangeStatement *stmt) override {
    recurse(stmt->lwr);
    recurse(stmt->upr);
    recurse(stmt->_body);
  }

  void visit(ScopeGuardStatement *stmt) override {
    error(stmt->loc,
          "Internal Compiler Error: ScopeGuardStatement should have been "
          "lowered by frontend.");
    fatal();
  }

  void visit(ThrowStatement *stmt) override {
    recurse(stmt);
    recurse(stmt->exp);
  }

  void visit(TryFinallyStatement *stmt) override {
    recurse(stmt->_body);
    recurse(stmt->finalbody);
  }

  void visit(TryCatchStatement *stmt) override {
    recurse(stmt->_body);
    for (auto c : *stmt->catches) {
      recurse(c->handler);
    }
  }

  void visit(SwitchStatement *stmt) override {
    recurse(stmt->condition);
    recurse(stmt->_body);
    for (auto cs : *stmt->cases) {
      recurse(cs);
    }
  }

  void visit(CaseStatement *stmt) override { recurse(stmt->statement); }

  void visit(DefaultStatement *stmt) override { recurse(stmt->statement); }

  void visit(UnrolledLoopStatement *stmt) override {
    recurse(stmt->statements);
  }

  void visit(LabelStatement *stmt) override { recurse(stmt->statement); }

  void visit(SynchronizedStatement *stmt) override {
    recurse(stmt->exp);
    recurse(stmt->_body);
  }

  void visit(WithStatement *stmt) override {
    recurse(stmt->wthis);
    recurse(stmt->exp);
    recurse(stmt->_body);
  }

  void visit(DebugStatement *stmt) override {
    recurse(stmt);
    recurse(stmt->statement);
  }

  void visit(NewExp *e) override {
    recurse(e->thisexp);
    recurse(e->argprefix);
    recurse(e->arguments);
  }

  void visit(NewAnonClassExp *e) override {
    recurse(e->thisexp);
    recurse(e->arguments);
  }

  void visit(UnaExp *e) override { recurse(e->e1); }

  void visit(BinExp *e) override {
    recurse(e->e1);
    recurse(e->e2);
  }

  void visit(AssertExp *e) override {
    recurse(e->e1);
    recurse(e->msg);
  }

  void visit(CallExp *e) override {
    recurse(e->e1);
    recurse(e->arguments);
  }

  void visit(ArrayExp *e) override {
    recurse(e->e1);
    recurse(e->arguments);
  }

  void visit(SliceExp *e) override {
    recurse(e->e1);
    recurse(e->lwr);
    recurse(e->upr);
  }

  void visit(ArrayLiteralExp *e) override { recurse(e->elements); }

  void visit(AssocArrayLiteralExp *e) override {
    recurse(e->keys);
    recurse(e->values);
  }

  void visit(StructLiteralExp *e) override {
    // use stageflags to prevent infinite recursion
    if (e->stageflags & 0x2000)
      return;
    int old = e->stageflags;
    e->stageflags |= 0x2000;
    recurse(e->elements);
    e->stageflags = old;
  }

  void visit(TupleExp *e) override {
    recurse(e->e0);
    recurse(e->exps);
  }

  void visit(CondExp *e) override {
    recurse(e->econd);
    recurse(e->e1);
    recurse(e->e2);
  }

  void visit(DeclarationExp *e) override { recurse(e->declaration); }

  void visit(VarDeclaration *decl) override {
    recurse(decl->_init);
    recurse(decl->edtor);
  }

  void visit(ExpInitializer *init) override { recurse(init->exp); }

  // Override the default assert(0) behavior of Visitor:
  void visit(Statement *) override {}   // do nothing
  void visit(Expression *) override {}  // do nothing
  void visit(Declaration *) override {} // do nothing
  void visit(Initializer *) override {} // do nothing
  void visit(Dsymbol *) override {}     // do nothing
};

///////////////////////////////////////////////////////////////////////////////

/// A recursive AST walker, that walks both Statements and Expressions
/// The recursion stops at a depth where the visitor sets stop to true.
/// If `continueAfterStop` is true, the visitor's stop is reset to false and
/// traversal continues at the next node in the hierarchy at the same level as
/// where stop was set.
class RecursiveWalker : public Visitor {
public:
  StoppableVisitor *v;
  bool continueAfterStop;

  RecursiveWalker(StoppableVisitor *visitor, bool _continueAfterStop = true)
      : v(visitor), continueAfterStop(_continueAfterStop) {}

  template <class T> bool recurse(T *stmt) {
    if (stmt) {
      stmt->accept(this);
    }
    return v->stop;
  }

  template <class T> bool recurse(Array<T> *stmts) {
    if (stmts) {
      for (auto s : *stmts) {
        if (recurse(s))
          break;
      }
    }
    return v->stop;
  }

  template <class T> bool call_visitor(T *obj) {
    obj->accept(v);
    if (v->stop && continueAfterStop) {
      // Reset stop to false, so that traversion continues at neighboring node
      // in the tree.
      v->stop = false;
      return true;
    }
    return v->stop;
  }

  using Visitor::visit;

  void visit(AttribDeclaration *ad) override {
    call_visitor(ad) || recurse(ad->decl);
  }

  void visit(FuncDeclaration *fd) override {
    call_visitor(fd) || recurse(fd->fbody);
  }

  void visit(CompoundStatement *stmt) override {
    call_visitor(stmt) || recurse(stmt->statements);
  }

  void visit(ReturnStatement *stmt) override {
    call_visitor(stmt) || recurse(stmt->exp);
  }

  void visit(ExpStatement *stmt) override {
    call_visitor(stmt) || recurse(stmt->exp);
  }

  void visit(IfStatement *stmt) override {
    call_visitor(stmt) || recurse(stmt->condition) || recurse(stmt->ifbody) ||
        recurse(stmt->elsebody);
  }

  void visit(ScopeStatement *stmt) override {
    call_visitor(stmt) || recurse(stmt->statement);
  }

  void visit(WhileStatement *stmt) override {
    call_visitor(stmt) || recurse(stmt->condition) || recurse(stmt->_body);
  }

  void visit(DoStatement *stmt) override {
    call_visitor(stmt) || recurse(stmt->_body) || recurse(stmt->condition);
  }

  void visit(ForStatement *stmt) override {
    call_visitor(stmt) || recurse(stmt->_init) || recurse(stmt->condition) ||
        recurse(stmt->_body) || recurse(stmt->increment);
  }

  void visit(ForeachStatement *stmt) override {
    call_visitor(stmt) || recurse(stmt->aggr) || recurse(stmt->_body);
  }

  void visit(ForeachRangeStatement *stmt) override {
    call_visitor(stmt) || recurse(stmt->lwr) || recurse(stmt->upr) ||
        recurse(stmt->_body);
  }

  void visit(ScopeGuardStatement *stmt) override {
    call_visitor(stmt) || recurse(stmt->statement);
  }

  void visit(ThrowStatement *stmt) override {
    call_visitor(stmt) || recurse(stmt->exp);
  }

  void visit(TryFinallyStatement *stmt) override {
    call_visitor(stmt) || recurse(stmt->_body) || recurse(stmt->finalbody);
  }

  void visit(TryCatchStatement *stmt) override {
    if (call_visitor(stmt) || recurse(stmt->_body))
      return;

    for (auto c : *stmt->catches) {
      if (recurse(c->handler))
        return;
    }
  }

  void visit(SwitchStatement *stmt) override {
    call_visitor(stmt) || recurse(stmt->condition) || recurse(stmt->_body);
  }

  void visit(CaseStatement *stmt) override {
    call_visitor(stmt) || recurse(stmt->statement);
  }

  void visit(DefaultStatement *stmt) override {
    call_visitor(stmt) || recurse(stmt->statement);
  }

  void visit(UnrolledLoopStatement *stmt) override {
    call_visitor(stmt) || recurse(stmt->statements);
  }

  void visit(LabelStatement *stmt) override {
    call_visitor(stmt) || recurse(stmt->statement);
  }

  void visit(SynchronizedStatement *stmt) override {
    call_visitor(stmt) || recurse(stmt->exp) || recurse(stmt->_body);
  }

  void visit(WithStatement *stmt) override {
    call_visitor(stmt) || recurse(stmt->wthis) || recurse(stmt->exp) ||
        recurse(stmt->_body);
  }

  void visit(PragmaStatement *stmt) override {
    call_visitor(stmt) || recurse(stmt->_body);
  }
  void visit(DebugStatement *stmt) override {
    call_visitor(stmt) || recurse(stmt->statement);
  }

  void visit(NewExp *e) override {
    call_visitor(e) || recurse(e->thisexp) || recurse(e->argprefix) ||
        recurse(e->arguments);
  }

  void visit(NewAnonClassExp *e) override {
    call_visitor(e) || recurse(e->thisexp) || recurse(e->arguments);
  }

  void visit(UnaExp *e) override { call_visitor(e) || recurse(e->e1); }

  void visit(BinExp *e) override {
    call_visitor(e) || recurse(e->e1) || recurse(e->e2);
  }

  void visit(AssertExp *e) override {
    call_visitor(e) || recurse(e->e1) || recurse(e->msg);
  }

  void visit(CallExp *e) override {
    call_visitor(e) || recurse(e->e1) || recurse(e->arguments);
  }

  void visit(ArrayExp *e) override {
    call_visitor(e) || recurse(e->e1) || recurse(e->arguments);
  }

  void visit(SliceExp *e) override {
    call_visitor(e) || recurse(e->e1) || recurse(e->lwr) || recurse(e->upr);
  }

  void visit(ArrayLiteralExp *e) override {
    call_visitor(e) || recurse(e->elements);
  }

  void visit(AssocArrayLiteralExp *e) override {
    call_visitor(e) || recurse(e->keys) || recurse(e->values);
  }

  void visit(StructLiteralExp *e) override {
    // use stageflags to prevent infinite recursion
    if (e->stageflags & 0x2000)
      return;
    int old = e->stageflags;
    e->stageflags |= 0x2000;
    call_visitor(e) || recurse(e->elements);
    e->stageflags = old;
  }

  void visit(TupleExp *e) override {
    call_visitor(e) || recurse(e->e0) || recurse(e->exps);
  }

  void visit(CondExp *e) override {
    call_visitor(e) || recurse(e->econd) || recurse(e->e1) || recurse(e->e2);
  }

  void visit(DeclarationExp *e) override {
    call_visitor(e) || recurse(e->declaration);
  }

  void visit(VarDeclaration *decl) override {
    call_visitor(decl) || recurse(decl->_init) || recurse(decl->edtor);
  }

  void visit(ExpInitializer *init) override {
    call_visitor(init) || recurse(init->exp);
  }

  void visit(ScopeDsymbol *s) override {
    call_visitor(s) || recurse(s->members);
  }

  // Override the default assert(0) behavior of Visitor:
  void visit(Statement *stmt) override { call_visitor(stmt); }
  void visit(Expression *exp) override { call_visitor(exp); }
  void visit(Declaration *decl) override { call_visitor(decl); }
  void visit(Initializer *init) override { call_visitor(init); }
  void visit(Dsymbol *init) override {}
};
