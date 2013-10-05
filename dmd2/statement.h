
// Compiler implementation of the D programming language
// Copyright (c) 1999-2012 by Digital Mars
// All Rights Reserved
// written by Walter Bright
// http://www.digitalmars.com
// License for redistribution is by either the Artistic License
// in artistic.txt, or the GNU General Public License in gnu.txt.
// See the included readme.txt for details.

#ifndef DMD_STATEMENT_H
#define DMD_STATEMENT_H

#ifdef __DMC__
#pragma once
#endif /* __DMC__ */

#include "root.h"

#include "arraytypes.h"
#include "dsymbol.h"
#include "lexer.h"

struct OutBuffer;
struct Scope;
struct Expression;
struct LabelDsymbol;
struct Identifier;
struct IfStatement;
struct ExpStatement;
struct DefaultStatement;
struct VarDeclaration;
struct Condition;
struct Module;
struct Token;
struct InlineCostState;
struct InlineDoState;
struct InlineScanState;
struct ReturnStatement;
struct CompoundStatement;
struct Parameter;
struct StaticAssert;
struct AsmStatement;
#if IN_LLVM
struct AsmBlockStatement;
#endif
struct GotoStatement;
struct ScopeStatement;
struct TryCatchStatement;
struct TryFinallyStatement;
struct CaseStatement;
struct DefaultStatement;
struct LabelStatement;
struct HdrGenState;
struct InterState;
#if IN_LLVM
struct CaseStatement;
struct LabelStatement;
struct SynchronizedStatement;
#endif

enum TOK;

#if IN_LLVM
namespace llvm
{
    class Value;
    class BasicBlock;
    class ConstantInt;
}
#endif

typedef bool (*sapply_fp_t)(Statement *, void *);

// Back end
struct IRState;
struct Blockx;

#ifdef IN_GCC
union tree_node; typedef union tree_node block;
union tree_node; typedef union tree_node elem;
#elif IN_LLVM
class DValue;
typedef DValue elem;
#else
struct block;
struct elem;
#endif
struct code;

/* How a statement exits; this is returned by blockExit()
 */
enum BE
{
    BEnone =     0,
    BEfallthru = 1,
    BEthrow =    2,
    BEreturn =   4,
    BEgoto =     8,
    BEhalt =     0x10,
    BEbreak =    0x20,
    BEcontinue = 0x40,
    BEany = (BEfallthru | BEthrow | BEreturn | BEgoto | BEhalt),
};

struct Statement : Object
{
    Loc loc;
    virtual ~Statement() {}

    Statement(Loc loc);
    virtual Statement *syntaxCopy();

    void print();
    char *toChars();

    void error(const char *format, ...);
    void warning(const char *format, ...);
    void deprecation(const char *format, ...);
    virtual void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    virtual ScopeStatement *isScopeStatement() { return NULL; }
    virtual Statement *semantic(Scope *sc);
    Statement *semanticScope(Scope *sc, Statement *sbreak, Statement *scontinue);
    Statement *semanticNoScope(Scope *sc);
    virtual Statement *getRelatedLabeled() { return this; }
    virtual bool hasBreak();
    virtual bool hasContinue();
    bool usesEH();
    virtual bool usesEHimpl();
    virtual int blockExit(bool mustNotThrow);
    bool comeFrom();
    virtual bool comeFromImpl();
    bool hasCode();
    virtual bool hasCodeImpl();
    virtual Statement *scopeCode(Scope *sc, Statement **sentry, Statement **sexit, Statement **sfinally);
    virtual Statements *flatten(Scope *sc);
    virtual Expression *interpret(InterState *istate);
    virtual bool apply(sapply_fp_t fp, void *param);
    virtual Statement *last();

    virtual int inlineCost(InlineCostState *ics);
    virtual Expression *doInline(InlineDoState *ids);
    virtual Statement *doInlineStatement(InlineDoState *ids);
    virtual Statement *inlineScan(InlineScanState *iss);

    // Back end
    virtual void toIR(IRState *irs);

    // Avoid dynamic_cast
    virtual ExpStatement *isExpStatement() { return NULL; }
    virtual CompoundStatement *isCompoundStatement() { return NULL; }
    virtual ReturnStatement *isReturnStatement() { return NULL; }
    virtual IfStatement *isIfStatement() { return NULL; }
    virtual CaseStatement *isCaseStatement() { return NULL; }
    virtual DefaultStatement *isDefaultStatement() { return NULL; }
    virtual LabelStatement *isLabelStatement() { return NULL; }

#if IN_LLVM
    virtual AsmBlockStatement *isAsmBlockStatement() { return NULL; }
    virtual AsmBlockStatement* endsWithAsm();
#endif
};

struct PeelStatement : Statement
{
    Statement *s;

    PeelStatement(Statement *s);
    Statement *semantic(Scope *sc);
    bool apply(sapply_fp_t fp, void *param);
};

struct ExpStatement : Statement
{
    Expression *exp;

    ExpStatement(Loc loc, Expression *exp);
    ExpStatement(Loc loc, Dsymbol *s);
    Statement *syntaxCopy();
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    Statement *semantic(Scope *sc);
    Expression *interpret(InterState *istate);
    int blockExit(bool mustNotThrow);
    bool hasCodeImpl();
    Statement *scopeCode(Scope *sc, Statement **sentry, Statement **sexit, Statement **sfinally);

    int inlineCost(InlineCostState *ics);
    Expression *doInline(InlineDoState *ids);
    Statement *doInlineStatement(InlineDoState *ids);
    Statement *inlineScan(InlineScanState *iss);

    void toIR(IRState *irs);

    ExpStatement *isExpStatement() { return this; }
};

struct DtorExpStatement : ExpStatement
{
    /* Wraps an expression that is the destruction of 'var'
     */

    VarDeclaration *var;

    DtorExpStatement(Loc loc, Expression *exp, VarDeclaration *v);
    Statement *syntaxCopy();
    void toIR(IRState *irs);
};

struct CompileStatement : Statement
{
    Expression *exp;

    CompileStatement(Loc loc, Expression *exp);
    Statement *syntaxCopy();
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    Statements *flatten(Scope *sc);
    Statement *semantic(Scope *sc);
    int blockExit(bool mustNotThrow);
};

struct CompoundStatement : Statement
{
    Statements *statements;

    CompoundStatement(Loc loc, Statements *s);
    CompoundStatement(Loc loc, Statement *s1);
    CompoundStatement(Loc loc, Statement *s1, Statement *s2);
    Statement *syntaxCopy();
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    Statement *semantic(Scope *sc);
    int blockExit(bool mustNotThrow);
    bool hasCodeImpl();
    Statements *flatten(Scope *sc);
    ReturnStatement *isReturnStatement();
    Expression *interpret(InterState *istate);
    bool apply(sapply_fp_t fp, void *param);
    Statement *last();

    int inlineCost(InlineCostState *ics);
    Expression *doInline(InlineDoState *ids);
    Statement *doInlineStatement(InlineDoState *ids);
    Statement *inlineScan(InlineScanState *iss);

    void toIR(IRState *irs);

    CompoundStatement *isCompoundStatement() { return this; }

#if IN_LLVM
    virtual AsmBlockStatement* endsWithAsm();
#endif
};

struct CompoundDeclarationStatement : CompoundStatement
{
    CompoundDeclarationStatement(Loc loc, Statements *s);
    Statement *syntaxCopy();
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
};

/* The purpose of this is so that continue will go to the next
 * of the statements, and break will go to the end of the statements.
 */
struct UnrolledLoopStatement : Statement
{
    Statements *statements;

    UnrolledLoopStatement(Loc loc, Statements *statements);
    Statement *syntaxCopy();
    Statement *semantic(Scope *sc);
    bool hasBreak();
    bool hasContinue();
    int blockExit(bool mustNotThrow);
    Expression *interpret(InterState *istate);
    bool apply(sapply_fp_t fp, void *param);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);

    int inlineCost(InlineCostState *ics);
    Expression *doInline(InlineDoState *ids);
    Statement *doInlineStatement(InlineDoState *ids);
    Statement *inlineScan(InlineScanState *iss);

    void toIR(IRState *irs);
};

struct ScopeStatement : Statement
{
    Statement *statement;

    ScopeStatement(Loc loc, Statement *s);
    Statement *syntaxCopy();
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    ScopeStatement *isScopeStatement() { return this; }
    Statement *semantic(Scope *sc);
    bool hasBreak();
    bool hasContinue();
    int blockExit(bool mustNotThrow);
    bool hasCodeImpl();
    Expression *interpret(InterState *istate);
    bool apply(sapply_fp_t fp, void *param);

    int inlineCost(InlineCostState *ics);
    Expression *doInline(InlineDoState *ids);
    Statement *doInlineStatement(InlineDoState *ids);
    Statement *inlineScan(InlineScanState *iss);

    void toIR(IRState *irs);
};

struct WhileStatement : Statement
{
    Expression *condition;
    Statement *body;

    WhileStatement(Loc loc, Expression *c, Statement *b);
    Statement *syntaxCopy();
    Statement *semantic(Scope *sc);
    bool hasBreak();
    bool hasContinue();
    int blockExit(bool mustNotThrow);
    Expression *interpret(InterState *istate);
    bool apply(sapply_fp_t fp, void *param);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);

    Statement *inlineScan(InlineScanState *iss);

    void toIR(IRState *irs);
};

struct DoStatement : Statement
{
    Statement *body;
    Expression *condition;

    DoStatement(Loc loc, Statement *b, Expression *c);
    Statement *syntaxCopy();
    Statement *semantic(Scope *sc);
    bool hasBreak();
    bool hasContinue();
    int blockExit(bool mustNotThrow);
    Expression *interpret(InterState *istate);
    bool apply(sapply_fp_t fp, void *param);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);

    Statement *inlineScan(InlineScanState *iss);

    void toIR(IRState *irs);
};

struct ForStatement : Statement
{
    Statement *init;
    Expression *condition;
    Expression *increment;
    Statement *body;
    int nest;

    // When wrapped in try/finally clauses, this points to the outermost one,
    // which may have an associated label. Internal break/continue statements
    // treat that label as referring to this loop.
    Statement *relatedLabeled;

    ForStatement(Loc loc, Statement *init, Expression *condition, Expression *increment, Statement *body);
    Statement *syntaxCopy();
    Statement *semanticInit(Scope *sc, Statements *ainit, size_t i);
    Statement *semantic(Scope *sc);
    Statement *scopeCode(Scope *sc, Statement **sentry, Statement **sexit, Statement **sfinally);
    Statement *getRelatedLabeled() { return relatedLabeled ? relatedLabeled : this; }
    bool hasBreak();
    bool hasContinue();
    int blockExit(bool mustNotThrow);
    Expression *interpret(InterState *istate);
    bool apply(sapply_fp_t fp, void *param);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);

    int inlineCost(InlineCostState *ics);
    Statement *inlineScan(InlineScanState *iss);
    Statement *doInlineStatement(InlineDoState *ids);

    void toIR(IRState *irs);
};

struct ForeachStatement : Statement
{
    enum TOK op;                // TOKforeach or TOKforeach_reverse
    Parameters *arguments;      // array of Parameter*'s
    Expression *aggr;
    Statement *body;

    VarDeclaration *key;
    VarDeclaration *value;

    FuncDeclaration *func;      // function we're lexically in

    Statements *cases;          // put breaks, continues, gotos and returns here
    CompoundStatements *gotos;  // forward referenced goto's go here

    ForeachStatement(Loc loc, enum TOK op, Parameters *arguments, Expression *aggr, Statement *body);
    Statement *syntaxCopy();
    Statement *semantic(Scope *sc);
    bool checkForArgTypes();
    int inferAggregate(Scope *sc, Dsymbol *&sapply);
    int inferApplyArgTypes(Scope *sc, Dsymbol *&sapply);
    bool hasBreak();
    bool hasContinue();
    int blockExit(bool mustNotThrow);
    Expression *interpret(InterState *istate);
    bool apply(sapply_fp_t fp, void *param);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);

    Statement *inlineScan(InlineScanState *iss);

    void toIR(IRState *irs);
};

#if DMDV2
struct ForeachRangeStatement : Statement
{
    enum TOK op;                // TOKforeach or TOKforeach_reverse
    Parameter *arg;             // loop index variable
    Expression *lwr;
    Expression *upr;
    Statement *body;

    VarDeclaration *key;

    ForeachRangeStatement(Loc loc, enum TOK op, Parameter *arg,
        Expression *lwr, Expression *upr, Statement *body);
    Statement *syntaxCopy();
    Statement *semantic(Scope *sc);
    bool hasBreak();
    bool hasContinue();
    int blockExit(bool mustNotThrow);
    Expression *interpret(InterState *istate);
    bool apply(sapply_fp_t fp, void *param);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);

    Statement *inlineScan(InlineScanState *iss);

    void toIR(IRState *irs);
};
#endif

struct IfStatement : Statement
{
    Parameter *arg;
    Expression *condition;
    Statement *ifbody;
    Statement *elsebody;

    VarDeclaration *match;      // for MatchExpression results

    IfStatement(Loc loc, Parameter *arg, Expression *condition, Statement *ifbody, Statement *elsebody);
    Statement *syntaxCopy();
    Statement *semantic(Scope *sc);
    Expression *interpret(InterState *istate);
    bool apply(sapply_fp_t fp, void *param);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    int blockExit(bool mustNotThrow);
    IfStatement *isIfStatement() { return this; }

    int inlineCost(InlineCostState *ics);
    Expression *doInline(InlineDoState *ids);
    Statement *doInlineStatement(InlineDoState *ids);
    Statement *inlineScan(InlineScanState *iss);

    void toIR(IRState *irs);
};

struct ConditionalStatement : Statement
{
    Condition *condition;
    Statement *ifbody;
    Statement *elsebody;

    ConditionalStatement(Loc loc, Condition *condition, Statement *ifbody, Statement *elsebody);
    Statement *syntaxCopy();
    Statement *semantic(Scope *sc);
    Statements *flatten(Scope *sc);
    int blockExit(bool mustNotThrow);
    bool apply(sapply_fp_t fp, void *param);

    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
};

struct PragmaStatement : Statement
{
    Identifier *ident;
    Expressions *args;          // array of Expression's
    Statement *body;

    PragmaStatement(Loc loc, Identifier *ident, Expressions *args, Statement *body);
    Statement *syntaxCopy();
    Statement *semantic(Scope *sc);
    int blockExit(bool mustNotThrow);
    bool apply(sapply_fp_t fp, void *param);

    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);

    void toIR(IRState *irs);
};

struct StaticAssertStatement : Statement
{
    StaticAssert *sa;

    StaticAssertStatement(StaticAssert *sa);
    Statement *syntaxCopy();
    Statement *semantic(Scope *sc);
    int blockExit(bool mustNotThrow);

    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
};

struct SwitchStatement : Statement
{
    Expression *condition;
    Statement *body;
    bool isFinal;

    DefaultStatement *sdefault;
#if !IN_LLVM
    TryFinallyStatement *tf;
#endif
    GotoCaseStatements gotoCases;  // array of unresolved GotoCaseStatement's
    CaseStatements *cases;         // array of CaseStatement's
    int hasNoDefault;           // !=0 if no default statement
    int hasVars;                // !=0 if has variable case values

#if IN_LLVM
    Statement *enclosingScopeExit;
#endif

    SwitchStatement(Loc loc, Expression *c, Statement *b, bool isFinal);
    Statement *syntaxCopy();
    Statement *semantic(Scope *sc);
    bool hasBreak();
    int blockExit(bool mustNotThrow);
    Expression *interpret(InterState *istate);
    bool apply(sapply_fp_t fp, void *param);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);

    Statement *inlineScan(InlineScanState *iss);

    void toIR(IRState *irs);
};

struct CaseStatement : Statement
{
    Expression *exp;
    Statement *statement;

    int index;          // which case it is (since we sort this)
    block *cblock;      // back end: label for the block

#if IN_LLVM
    Statement *enclosingScopeExit;
#endif

    CaseStatement(Loc loc, Expression *exp, Statement *s);
    Statement *syntaxCopy();
    Statement *semantic(Scope *sc);
    int compare(Object *obj);
    int blockExit(bool mustNotThrow);
    bool comeFromImpl();
    Expression *interpret(InterState *istate);
    bool apply(sapply_fp_t fp, void *param);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    CaseStatement *isCaseStatement() { return this; }

    Statement *inlineScan(InlineScanState *iss);

    void toIR(IRState *irs);

#if IN_LLVM
    llvm::BasicBlock* bodyBB;
    llvm::Value* llvmIdx;
#endif
};

#if DMDV2

struct CaseRangeStatement : Statement
{
    Expression *first;
    Expression *last;
    Statement *statement;

    CaseRangeStatement(Loc loc, Expression *first, Expression *last, Statement *s);
    Statement *syntaxCopy();
    Statement *semantic(Scope *sc);
    bool apply(sapply_fp_t fp, void *param);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
};

#endif

struct DefaultStatement : Statement
{
    Statement *statement;
#ifdef IN_GCC
    block *cblock;      // back end: label for the block
#endif

#if IN_LLVM
    Statement *enclosingScopeExit;
#endif

    DefaultStatement(Loc loc, Statement *s);
    Statement *syntaxCopy();
    Statement *semantic(Scope *sc);
    int blockExit(bool mustNotThrow);
    bool comeFromImpl();
    Expression *interpret(InterState *istate);
    bool apply(sapply_fp_t fp, void *param);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    DefaultStatement *isDefaultStatement() { return this; }

    Statement *inlineScan(InlineScanState *iss);

    void toIR(IRState *irs);

#if IN_LLVM
    llvm::BasicBlock* bodyBB;
#endif
};

struct GotoDefaultStatement : Statement
{
    SwitchStatement *sw;

    GotoDefaultStatement(Loc loc);
    Statement *syntaxCopy();
    Statement *semantic(Scope *sc);
    Expression *interpret(InterState *istate);
    int blockExit(bool mustNotThrow);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);

    void toIR(IRState *irs);
};

struct GotoCaseStatement : Statement
{
    Expression *exp;            // NULL, or which case to goto
    CaseStatement *cs;          // case statement it resolves to
    SwitchStatement *sw;

    GotoCaseStatement(Loc loc, Expression *exp);
    Statement *syntaxCopy();
    Statement *semantic(Scope *sc);
    Expression *interpret(InterState *istate);
    int blockExit(bool mustNotThrow);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);

    void toIR(IRState *irs);
};

struct SwitchErrorStatement : Statement
{
    SwitchErrorStatement(Loc loc);
    int blockExit(bool mustNotThrow);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);

    void toIR(IRState *irs);
};

struct ReturnStatement : Statement
{
    Expression *exp;
    bool implicit0;             // this is an implicit "return 0;"

    ReturnStatement(Loc loc, Expression *exp);
    Statement *syntaxCopy();
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    Statement *semantic(Scope *sc);
    int blockExit(bool mustNotThrow);
    Expression *interpret(InterState *istate);

    int inlineCost(InlineCostState *ics);
    Expression *doInline(InlineDoState *ids);
    Statement *doInlineStatement(InlineDoState *ids);
    Statement *inlineScan(InlineScanState *iss);

    void toIR(IRState *irs);

    ReturnStatement *isReturnStatement() { return this; }
};

struct BreakStatement : Statement
{
    Identifier *ident;

    BreakStatement(Loc loc, Identifier *ident);
    Statement *syntaxCopy();
    Statement *semantic(Scope *sc);
    Expression *interpret(InterState *istate);
    int blockExit(bool mustNotThrow);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);

    void toIR(IRState *irs);

#if IN_LLVM
    // LDC: only set if ident is set: label statement to jump to
    LabelStatement *target;
#endif
};

struct ContinueStatement : Statement
{
    Identifier *ident;

    ContinueStatement(Loc loc, Identifier *ident);
    Statement *syntaxCopy();
    Statement *semantic(Scope *sc);
    Expression *interpret(InterState *istate);
    int blockExit(bool mustNotThrow);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);

    void toIR(IRState *irs);

#if IN_LLVM
    // LDC: only set if ident is set: label statement to jump to
    LabelStatement *target;
#endif
};

struct SynchronizedStatement : Statement
{
    Expression *exp;
    Statement *body;

    SynchronizedStatement(Loc loc, Expression *exp, Statement *body);
    Statement *syntaxCopy();
    Statement *semantic(Scope *sc);
    bool hasBreak();
    bool hasContinue();
    bool usesEHimpl();
    int blockExit(bool mustNotThrow);
    bool apply(sapply_fp_t fp, void *param);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);

    Statement *inlineScan(InlineScanState *iss);

// Back end
    elem *esync;
    SynchronizedStatement(Loc loc, elem *esync, Statement *body);
    void toIR(IRState *irs);
#if IN_LLVM
    llvm::Value* llsync;
#endif
};

struct WithStatement : Statement
{
    Expression *exp;
    Statement *body;
    VarDeclaration *wthis;

    WithStatement(Loc loc, Expression *exp, Statement *body);
    Statement *syntaxCopy();
    Statement *semantic(Scope *sc);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    int blockExit(bool mustNotThrow);
    Expression *interpret(InterState *istate);
    bool apply(sapply_fp_t fp, void *param);

    Statement *inlineScan(InlineScanState *iss);

    void toIR(IRState *irs);
};

struct TryCatchStatement : Statement
{
    Statement *body;
    Catches *catches;

    TryCatchStatement(Loc loc, Statement *body, Catches *catches);
    Statement *syntaxCopy();
    Statement *semantic(Scope *sc);
    bool hasBreak();
    bool usesEHimpl();
    int blockExit(bool mustNotThrow);
    Expression *interpret(InterState *istate);
    bool apply(sapply_fp_t fp, void *param);

    Statement *inlineScan(InlineScanState *iss);

    void toIR(IRState *irs);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
};

struct Catch : Object
{
    Loc loc;
    Type *type;
    Identifier *ident;
    VarDeclaration *var;
    Statement *handler;
    bool internalCatch;         // was generated by the compiler,
                                // wasn't present in source code

    Catch(Loc loc, Type *t, Identifier *id, Statement *handler);
    Catch *syntaxCopy();
    void semantic(Scope *sc);
    int blockExit(bool mustNotThrow);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
};

struct TryFinallyStatement : Statement
{
    Statement *body;
    Statement *finalbody;

    TryFinallyStatement(Loc loc, Statement *body, Statement *finalbody);
    Statement *syntaxCopy();
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    Statement *semantic(Scope *sc);
    bool hasBreak();
    bool hasContinue();
    bool usesEHimpl();
    int blockExit(bool mustNotThrow);
    Expression *interpret(InterState *istate);
    bool apply(sapply_fp_t fp, void *param);

    Statement *inlineScan(InlineScanState *iss);

    void toIR(IRState *irs);
};

struct OnScopeStatement : Statement
{
    TOK tok;
    Statement *statement;

    OnScopeStatement(Loc loc, TOK tok, Statement *statement);
    Statement *syntaxCopy();
    int blockExit(bool mustNotThrow);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    Statement *semantic(Scope *sc);
    bool usesEHimpl();
    Statement *scopeCode(Scope *sc, Statement **sentry, Statement **sexit, Statement **sfinally);
    Expression *interpret(InterState *istate);
    bool apply(sapply_fp_t fp, void *param);

    void toIR(IRState *irs);
};

struct ThrowStatement : Statement
{
    Expression *exp;
    bool internalThrow;         // was generated by the compiler,
                                // wasn't present in source code

    ThrowStatement(Loc loc, Expression *exp);
    Statement *syntaxCopy();
    Statement *semantic(Scope *sc);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    int blockExit(bool mustNotThrow);
    Expression *interpret(InterState *istate);

    Statement *inlineScan(InlineScanState *iss);

    void toIR(IRState *irs);
};

struct DebugStatement : Statement
{
    Statement *statement;

    DebugStatement(Loc loc, Statement *statement);
    Statement *syntaxCopy();
    Statement *semantic(Scope *sc);
    Statements *flatten(Scope *sc);
    bool apply(sapply_fp_t fp, void *param);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
};

struct GotoStatement : Statement
{
    Identifier *ident;
    LabelDsymbol *label;
#if !IN_LLVM
    TryFinallyStatement *tf;
#else
    TryFinallyStatement *enclosingFinally;
    Statement* enclosingScopeExit;
#endif

    GotoStatement(Loc loc, Identifier *ident);
    Statement *syntaxCopy();
    Statement *semantic(Scope *sc);
    int blockExit(bool mustNotThrow);
    Expression *interpret(InterState *istate);

    void toIR(IRState *irs);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
};

struct LabelStatement : Statement
{
    Identifier *ident;
    Statement *statement;
#if !IN_LLVM
    TryFinallyStatement *tf;
#else
    TryFinallyStatement *enclosingFinally;
    Statement* enclosingScopeExit;
#endif
    block *lblock;              // back end

    Blocks *fwdrefs;            // forward references to this LabelStatement

    LabelStatement(Loc loc, Identifier *ident, Statement *statement);
    Statement *syntaxCopy();
    Statement *semantic(Scope *sc);
    Statements *flatten(Scope *sc);
    int blockExit(bool mustNotThrow);
    bool comeFromImpl();
    Expression *interpret(InterState *istate);
    bool apply(sapply_fp_t fp, void *param);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);

    Statement *inlineScan(InlineScanState *iss);
    LabelStatement *isLabelStatement() { return this; }

    void toIR(IRState *irs);
};

struct LabelDsymbol : Dsymbol
{
    LabelStatement *statement;

    LabelDsymbol(Identifier *ident);
    LabelDsymbol *isLabel();
};

struct AsmStatement : Statement
{
    Token *tokens;
    code *asmcode;
    unsigned asmalign;          // alignment of this statement
    unsigned regs;              // mask of registers modified (must match regm_t in back end)
    unsigned char refparam;     // !=0 if function parameter is referenced
    unsigned char naked;        // !=0 if function is to be naked

    AsmStatement(Loc loc, Token *tokens);
    Statement *syntaxCopy();
    Statement *semantic(Scope *sc);
    int blockExit(bool mustNotThrow);
    bool comeFromImpl();
    Expression *interpret(InterState *istate);

    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);

    //int inlineCost(InlineCostState *ics);
    //Expression *doInline(InlineDoState *ids);
    //Statement *inlineScan(InlineScanState *iss);

    void toIR(IRState *irs);

 #if IN_LLVM
    // non-zero if this is a branch, contains the target labels identifier
    Identifier* isBranchToLabel;
#endif
};

struct ImportStatement : Statement
{
    Dsymbols *imports;          // Array of Import's

    ImportStatement(Loc loc, Dsymbols *imports);
    Statement *syntaxCopy();
    Statement *semantic(Scope *sc);
    int blockExit(bool mustNotThrow);
    bool hasCodeImpl();
    Expression *interpret(InterState *istate);

    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);

    int inlineCost(InlineCostState *ics);
    Expression *doInline(InlineDoState *ids);
    Statement *doInlineStatement(InlineDoState *ids);

    void toIR(IRState *irs);
};

#if IN_LLVM
struct AsmBlockStatement : CompoundStatement
{
    TryFinallyStatement* enclosingFinally;
    Statement* enclosingScopeExit;

    AsmBlockStatement(Loc loc, Statements *s);
    Statements *flatten(Scope *sc);
    Statement *syntaxCopy();
    Statement *semantic(Scope *sc);

    CompoundStatement *isCompoundStatement() { return NULL; }
    AsmBlockStatement *isAsmBlockStatement() { return this; }

    void toIR(IRState *irs);
    AsmBlockStatement* endsWithAsm();

    llvm::Value* abiret;
};
#endif

#endif /* DMD_STATEMENT_H */
