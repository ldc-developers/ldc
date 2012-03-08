
// Compiler implementation of the D programming language
// Copyright (c) 1999-2011 by Digital Mars
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
struct AsmBlockStatement;
struct GotoStatement;
struct ScopeStatement;
struct TryCatchStatement;
struct TryFinallyStatement;
struct CaseStatement;
struct DefaultStatement;
struct LabelStatement;
struct HdrGenState;
struct InterState;
struct CaseStatement;
struct LabelStatement;
struct VolatileStatement;
struct SynchronizedStatement;

enum TOK;
#if IN_LLVM
namespace llvm
{
    class Value;
    class BasicBlock;
    class ConstantInt;
}
#endif

// Back end
struct IRState;
struct Blockx;
#if IN_LLVM
struct DValue;
typedef DValue elem;
#endif

#if IN_GCC
union tree_node; typedef union tree_node block;
//union tree_node; typedef union tree_node elem;
#else
struct block;
//struct elem;
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

    Statement(Loc loc);
    virtual Statement *syntaxCopy();

    void print();
    char *toChars();

    void error(const char *format, ...) IS_PRINTF(2);
    void warning(const char *format, ...) IS_PRINTF(2);
    virtual void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    virtual TryCatchStatement *isTryCatchStatement() { return NULL; }
    virtual GotoStatement *isGotoStatement() { return NULL; }
    virtual AsmStatement *isAsmStatement() { return NULL; }
    virtual AsmBlockStatement *isAsmBlockStatement() { return NULL; }
    int incontract;
    virtual ScopeStatement *isScopeStatement() { return NULL; }
    virtual Statement *semantic(Scope *sc);
    Statement *semanticScope(Scope *sc, Statement *sbreak, Statement *scontinue);
    Statement *semanticNoScope(Scope *sc);
    virtual int hasBreak();
    virtual int hasContinue();
    virtual int usesEH();
    virtual int blockExit(bool mustNotThrow);
    virtual int comeFrom();
    virtual int isEmpty();
    virtual void scopeCode(Scope *sc, Statement **sentry, Statement **sexit, Statement **sfinally);
    virtual Statements *flatten(Scope *sc);
    virtual Expression *interpret(InterState *istate);

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
    virtual CaseStatement* isCaseStatement() { return NULL; }
    virtual DefaultStatement *isDefaultStatement() { return NULL; }
    virtual LabelStatement* isLabelStatement() { return NULL; }

#if IN_LLVM
    virtual void toNakedIR(IRState *irs);
    virtual AsmBlockStatement* endsWithAsm();
#endif
};

struct PeelStatement : Statement
{
    Statement *s;

    PeelStatement(Statement *s);
    Statement *semantic(Scope *sc);
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
    int isEmpty();
    void scopeCode(Scope *sc, Statement **sentry, Statement **sexit, Statement **sfinally);

    int inlineCost(InlineCostState *ics);
    Expression *doInline(InlineDoState *ids);
    Statement *doInlineStatement(InlineDoState *ids);
    Statement *inlineScan(InlineScanState *iss);

    void toIR(IRState *irs);

#if IN_LLVM
    void toNakedIR(IRState *irs);
#endif

    ExpStatement *isExpStatement() { return this; }
};

struct CompileStatement : Statement
{
    Expression *exp;

    CompileStatement(Loc loc, Expression *exp);
    Statement *syntaxCopy();
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    Statements *flatten(Scope *sc);
    Statement *semantic(Scope *sc);
};

struct CompoundStatement : Statement
{
    Statements *statements;

    CompoundStatement(Loc loc, Statements *s);
    CompoundStatement(Loc loc, Statement *s1);
    CompoundStatement(Loc loc, Statement *s1, Statement *s2);
    virtual Statement *syntaxCopy();
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    virtual Statement *semantic(Scope *sc);
    int usesEH();
    int blockExit(bool mustNotThrow);
    int comeFrom();
    int isEmpty();
    virtual Statements *flatten(Scope *sc);
    ReturnStatement *isReturnStatement();
    Expression *interpret(InterState *istate);

    int inlineCost(InlineCostState *ics);
    Expression *doInline(InlineDoState *ids);
    Statement *doInlineStatement(InlineDoState *ids);
    Statement *inlineScan(InlineScanState *iss);

    virtual void toIR(IRState *irs);

    // LDC
    virtual void toNakedIR(IRState *irs);
    virtual AsmBlockStatement* endsWithAsm();

    virtual CompoundStatement *isCompoundStatement() { return this; }
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
    int hasBreak();
    int hasContinue();
    int usesEH();
    int blockExit(bool mustNotThrow);
    int comeFrom();
    Expression *interpret(InterState *istate);
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
    int hasBreak();
    int hasContinue();
    int usesEH();
    int blockExit(bool mustNotThrow);
    int comeFrom();
    int isEmpty();
    Expression *interpret(InterState *istate);

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
    int hasBreak();
    int hasContinue();
    int usesEH();
    int blockExit(bool mustNotThrow);
    int comeFrom();
    Expression *interpret(InterState *istate);
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
    int hasBreak();
    int hasContinue();
    int usesEH();
    int blockExit(bool mustNotThrow);
    int comeFrom();
    Expression *interpret(InterState *istate);
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

    ForStatement(Loc loc, Statement *init, Expression *condition, Expression *increment, Statement *body);
    Statement *syntaxCopy();
    Statement *semantic(Scope *sc);
    void scopeCode(Scope *sc, Statement **sentry, Statement **sexit, Statement **sfinally);
    int hasBreak();
    int hasContinue();
    int usesEH();
    int blockExit(bool mustNotThrow);
    int comeFrom();
    Expression *interpret(InterState *istate);
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
    int hasBreak();
    int hasContinue();
    int usesEH();
    int blockExit(bool mustNotThrow);
    int comeFrom();
    Expression *interpret(InterState *istate);
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
    int hasBreak();
    int hasContinue();
    int usesEH();
    int blockExit(bool mustNotThrow);
    int comeFrom();
    Expression *interpret(InterState *istate);
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
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    int usesEH();
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
    int usesEH();
    int blockExit(bool mustNotThrow);

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
    int usesEH();
    int blockExit(bool mustNotThrow);

    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
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

    DefaultStatement *sdefault;

    GotoCaseStatements gotoCases;  // array of unresolved GotoCaseStatement's
    CaseStatements *cases;         // array of CaseStatement's
    int hasNoDefault;           // !=0 if no default statement

    // LDC
    Statement *enclosingScopeExit;

    SwitchStatement(Loc loc, Expression *c, Statement *b);
    Statement *syntaxCopy();
    Statement *semantic(Scope *sc);
    int hasBreak();
    int usesEH();
    int blockExit(bool mustNotThrow);
    Expression *interpret(InterState *istate);
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

    // LDC
    Statement *enclosingScopeExit;

    CaseStatement(Loc loc, Expression *exp, Statement *s);
    Statement *syntaxCopy();
    Statement *semantic(Scope *sc);
    int compare(Object *obj);
    int usesEH();
    int blockExit(bool mustNotThrow);
    int comeFrom();
    Expression *interpret(InterState *istate);
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
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
};

#endif

struct DefaultStatement : Statement
{
    Statement *statement;
#if IN_GCC
    block *cblock;      // back end: label for the block
#endif

    // LDC
    Statement *enclosingScopeExit;

    DefaultStatement(Loc loc, Statement *s);
    Statement *syntaxCopy();
    Statement *semantic(Scope *sc);
    int usesEH();
    int blockExit(bool mustNotThrow);
    int comeFrom();
    Expression *interpret(InterState *istate);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    DefaultStatement *isDefaultStatement() { return this; }

    Statement *inlineScan(InlineScanState *iss);

    void toIR(IRState *irs);

    // LDC
    llvm::BasicBlock* bodyBB;
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

    // LDC: only set if ident is set: label statement to jump to
    LabelStatement *target;
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

    // LDC: only set if ident is set: label statement to jump to
    LabelStatement *target;
};

struct SynchronizedStatement : Statement
{
    Expression *exp;
    Statement *body;

    SynchronizedStatement(Loc loc, Expression *exp, Statement *body);
    Statement *syntaxCopy();
    Statement *semantic(Scope *sc);
    int hasBreak();
    int hasContinue();
    int usesEH();
    int blockExit(bool mustNotThrow);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);

    Statement *inlineScan(InlineScanState *iss);

// Back end
    elem *esync;
    SynchronizedStatement(Loc loc, elem *esync, Statement *body);
    void toIR(IRState *irs);
    llvm::Value* llsync;
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
    int usesEH();
    int blockExit(bool mustNotThrow);
    Expression *interpret(InterState *istate);

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
    int hasBreak();
    int usesEH();
    int blockExit(bool mustNotThrow);
    Expression *interpret(InterState *istate);

    Statement *inlineScan(InlineScanState *iss);

    void toIR(IRState *irs);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    TryCatchStatement *isTryCatchStatement() { return this; }
};

struct Catch : Object
{
    Loc loc;
    Type *type;
    Identifier *ident;
    VarDeclaration *var;
    Statement *handler;

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
    int hasBreak();
    int hasContinue();
    int usesEH();
    int blockExit(bool mustNotThrow);
    Expression *interpret(InterState *istate);

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
    int usesEH();
    void scopeCode(Scope *sc, Statement **sentry, Statement **sexit, Statement **sfinally);
    Expression *interpret(InterState *istate);

    void toIR(IRState *irs);
};

struct ThrowStatement : Statement
{
    Expression *exp;

    ThrowStatement(Loc loc, Expression *exp);
    Statement *syntaxCopy();
    Statement *semantic(Scope *sc);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    int blockExit(bool mustNotThrow);
    Expression *interpret(InterState *istate);

    Statement *inlineScan(InlineScanState *iss);

    void toIR(IRState *irs);
};

struct VolatileStatement : Statement
{
    Statement *statement;

    VolatileStatement(Loc loc, Statement *statement);
    Statement *syntaxCopy();
    Statement *semantic(Scope *sc);
    Statements *flatten(Scope *sc);
    int blockExit(bool mustNotThrow);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);

    Statement *inlineScan(InlineScanState *iss);

    void toIR(IRState *irs);
};

struct GotoStatement : Statement
{
    Identifier *ident;
    LabelDsymbol *label;
    TryFinallyStatement *enclosingFinally;
    Statement* enclosingScopeExit;

    GotoStatement(Loc loc, Identifier *ident);
    Statement *syntaxCopy();
    Statement *semantic(Scope *sc);
    int blockExit(bool mustNotThrow);
    Expression *interpret(InterState *istate);

    void toIR(IRState *irs);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    GotoStatement *isGotoStatement() { return this; }
};

struct LabelStatement : Statement
{
    Identifier *ident;
    Statement *statement;
    TryFinallyStatement *enclosingFinally;
    Statement* enclosingScopeExit;
    block *lblock;              // back end

    Blocks *fwdrefs;            // forward references to this LabelStatement

    LabelStatement(Loc loc, Identifier *ident, Statement *statement);
    Statement *syntaxCopy();
    Statement *semantic(Scope *sc);
    Statements *flatten(Scope *sc);
    int usesEH();
    int blockExit(bool mustNotThrow);
    int comeFrom();
    Expression *interpret(InterState *istate);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);

    Statement *inlineScan(InlineScanState *iss);
    LabelStatement *isLabelStatement() { return this; }

    void toIR(IRState *irs);

    // LDC
    bool asmLabel;       // for labels inside inline assembler
    void toNakedIR(IRState *irs);
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
    int comeFrom();
    Expression *interpret(InterState *istate);

    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    virtual AsmStatement *isAsmStatement() { return this; }

    void toIR(IRState *irs);

    // LDC
    // non-zero if this is a branch, contains the target labels identifier
    Identifier* isBranchToLabel;

    void toNakedIR(IRState *irs);
};

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
    void toNakedIR(IRState *irs);
    AsmBlockStatement* endsWithAsm();

    llvm::Value* abiret;
};

#endif /* DMD_STATEMENT_H */
