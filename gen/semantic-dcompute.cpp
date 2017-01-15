//===-- dcomputevailditychecker.cpp ---------------------------------------===//
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
//          are assumed to not throw. We only need to check for ThrowStatement
//
//      ban: classes, interfaces, asm, typeid, global variables
//
//===----------------------------------------------------------------------===//


#include "gen/recursivevisitor.h"

struct DComputeSemantic : public StoppableVisitor {
    // InterfaceDeclaration
    // ClassDeclaration
    // VarDeclaration : isDataSeg (global variable)
    // PragmaDeclaration : pragma(lib, "...")
    // nogc
    // nothrow
    // no typeid
    // no string switches
    // no synchronized
    
    void visit(InterfaceDeclaration *decl) {
        decl->error("interfaces and classes not allowed in @compute code");
        stop = true;
    }
    void visit(ClassDeclaration *decl) {
        decl->error("interfaces and classes not allowed in @compute code");
        stop = true;
    }
    void visit(VarDeclaration *decl) {
        if (decl->isDataseg())
        {
            decl->error("global variables not allowed in @compute code");
            stop = true;
        }
        if (decl->type->ty == Taarray)
        {
            decl->error("associative arrays not allowed in @compute code");
            stop = true;
        }
    }
    void visit(PragmaDeclaration *decl) {
        if (decl->ident == Id::lib)
        {
            decl->error("linking additional libraries not supported in @compute code");
            stop = true;
        }
    }
    
    // Nogc enforcement.
    // No need to check AssocArrayLiteral because AA's are benned anyway
    void visit(ArrayLiteralExp* e) override
    {
        if (e->type.ty != Tarray || !e->elements || !e->elements->dim)
            return;
        e->error("array literal in @compute code not allowed");
        stop = true;
    }
    void visit(NewExp* e) override
    {
        e->error("cannot use 'new' in @compute code");
        stop = true;
    }
    
    void visit(DeleteExp* e) override
    {
        e->error("cannot use 'delete' in @compute code");
        stop = true;
    }
    // No need to check IndexExp because AA's are banned anyway
    void visit(AssignExp* e) override
    {
        if (e->e1->op == TOKarraylength)
        {
            e->error("setting 'length' in @compute code not allowed");
            stop = true;
        }
    }
    
    void visit(CatAssignExp* e) override
    {
        e->error("cannot use operator ~= in @compute code");
        stop = true;
    }
    void visit(CatExp* e) override
    {
        e->error("cannot use operator ~ in @compute code");
        stop = true;
    }
    // Ban typeid(T)
    void visit(TypeIdExpression *e) override
    {
        e->error("typeinfo not available in @compute code");
        stop = true;
    }
    void visit(SynchronizedStatement *e)
    {
        e->error("cannot use 'synchronized' in @compute code");
        stop = true;
    }
    
    void visit(CallExp *e)
    {
        if (!hasComputeAttr(e->f->getModule())) {
            e->error("can only call functions from other @compute modules in @compute code");
            stop = true;
        }
    }
    void visit(StringLiteralExp *e)
    {
        e->error("string literals not allowed in @compue code");
        stop = true;
    }
    void visit(CompoundAsmStatement *e)
    {
        e->error("asm not allowed in @compute code");
        stop = true;
    }
    void visit(AsmStatement *e)
    {
        e->error("asm not allowed in @compute code");
        stop = true;
    }
    // nothrow
    
    void visit(TryCatchStatement *e)
    {
        e->error("no exceptions in @compute code");
        stop = true;
    }
    void visit(ThrowStatement *e)
    {
        e->error("no exceptions in @compute code");
        stop = true;
    }
    void visit(SwitchStatement *e)
    {
        if (!s->condition->type->isintegral())
        {
            e->error("cannot switch on strings in @compute code");
            stop = true;
        }
    }
}

void dcomputeSemanticAnalysis(Module * m)
{
    DComputeSemantic v;
    RecursiveWalker r(&v);
    for (unsigned k = 0; k < m->members->dim; k++) {
        Dsymbol *dsym = (*m->members)[k];
        assert(dsym);
        dsym->accept(&r);
    }
}
