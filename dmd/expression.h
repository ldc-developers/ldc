
// Compiler implementation of the D programming language
// Copyright (c) 1999-2012 by Digital Mars
// All Rights Reserved
// written by Walter Bright
// http://www.digitalmars.com
// License for redistribution is by either the Artistic License
// in artistic.txt, or the GNU General Public License in gnu.txt.
// See the included readme.txt for details.

#ifndef DMD_EXPRESSION_H
#define DMD_EXPRESSION_H

#include "mars.h"
#include "identifier.h"
#include "lexer.h"
#include "arraytypes.h"

struct Type;
struct Scope;
struct TupleDeclaration;
struct VarDeclaration;
struct FuncDeclaration;
struct FuncLiteralDeclaration;
struct Declaration;
struct CtorDeclaration;
struct NewDeclaration;
struct Dsymbol;
struct Import;
struct Module;
struct ScopeDsymbol;
struct InlineCostState;
struct InlineDoState;
struct InlineScanState;
struct Expression;
struct Declaration;
struct AggregateDeclaration;
struct StructDeclaration;
struct TemplateInstance;
struct TemplateDeclaration;
struct ClassDeclaration;
struct HdrGenState;
struct BinExp;
struct AssignExp;
struct InterState;
struct OverloadSet;
struct Initializer;
struct StringExp;

enum TOK;

#if IN_DMD
// Back end
struct IRState;
struct dt_t;
struct elem;
struct Symbol;          // back end symbol
#endif

#ifdef IN_GCC
union tree_node; typedef union tree_node elem;
#endif

#if IN_LLVM
struct IRState;
struct DValue;
namespace llvm {
    class Constant;
    class ConstantInt;
    class StructType;
}
#endif

void initPrecedence();

typedef int (*apply_fp_t)(Expression *, void *);

Expression *resolveProperties(Scope *sc, Expression *e);
void accessCheck(Loc loc, Scope *sc, Expression *e, Declaration *d);
Dsymbol *search_function(AggregateDeclaration *ad, Identifier *funcid);
void inferApplyArgTypes(enum TOK op, Parameters *arguments, Expression *aggr, Module* from);
void argExpTypesToCBuffer(OutBuffer *buf, Expressions *arguments, HdrGenState *hgs);
void argsToCBuffer(OutBuffer *buf, Expressions *arguments, HdrGenState *hgs);
void expandTuples(Expressions *exps);
FuncDeclaration *hasThis(Scope *sc);
Expression *fromConstInitializer(int result, Expression *e);
int arrayExpressionCanThrow(Expressions *exps);
TemplateDeclaration *getFuncTemplateDecl(Dsymbol *s);

/* Interpreter: what form of return value expression is required?
 */
enum CtfeGoal
{   ctfeNeedRvalue,   // Must return an Rvalue
    ctfeNeedLvalue,   // Must return an Lvalue
    ctfeNeedAnyValue, // Can return either an Rvalue or an Lvalue
    ctfeNeedLvalueRef,// Must return a reference to an Lvalue (for ref types)
    ctfeNeedNothing   // The return value is not required
};

struct IntRange
{   uinteger_t imin;
    uinteger_t imax;
};

struct Expression : Object
{
    Loc loc;                    // file location
    enum TOK op;                // handy to minimize use of dynamic_cast
    Type *type;                 // !=NULL means that semantic() has been run
    int size;                   // # of bytes in Expression so we can copy() it

    Expression(Loc loc, enum TOK op, int size);
    Expression *copy();
    virtual Expression *syntaxCopy();
    virtual int apply(apply_fp_t fp, void *param);
    virtual Expression *semantic(Scope *sc);
    Expression *trySemantic(Scope *sc);

    int dyncast() { return DYNCAST_EXPRESSION; }        // kludge for template.isExpression()

    void print();
    char *toChars();
    virtual void dump(int indent);
    void error(const char *format, ...) IS_PRINTF(2);
    void warning(const char *format, ...) IS_PRINTF(2);
    virtual int rvalue();

    static Expression *combine(Expression *e1, Expression *e2);
    static Expressions *arraySyntaxCopy(Expressions *exps);

    virtual dinteger_t toInteger();
    virtual uinteger_t toUInteger();
    virtual real_t toReal();
    virtual real_t toImaginary();
    virtual complex_t toComplex();
    virtual StringExp *toString();
    virtual void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    virtual void toMangleBuffer(OutBuffer *buf);
    virtual int isLvalue();
    virtual Expression *toLvalue(Scope *sc, Expression *e);
    virtual Expression *modifiableLvalue(Scope *sc, Expression *e);
    virtual Expression *implicitCastTo(Scope *sc, Type *t);
    virtual MATCH implicitConvTo(Type *t);
    virtual Expression *castTo(Scope *sc, Type *t);
    virtual void checkEscape();
    virtual void checkEscapeRef();
    void checkScalar();
    void checkNoBool();
    Expression *checkIntegral();
    Expression *checkArithmetic();
    void checkDeprecated(Scope *sc, Dsymbol *s);
    virtual Expression *checkToBoolean();
    Expression *checkToPointer();
    Expression *addressOf(Scope *sc);
    Expression *deref();
    Expression *integralPromotions(Scope *sc);

    Expression *toDelegate(Scope *sc, Type *t);

    virtual Expression *optimize(int result);
    #define WANTflags   1
    #define WANTvalue   2
    // A compile-time result is required. Give an error if not possible
    #define WANTinterpret 4
    // Same as WANTvalue, but also expand variables as far as possible
    #define WANTexpand  8

    // Entry point for CTFE.
    // A compile-time result is required. Give an error if not possible
    Expression *ctfeInterpret();

    // Implementation of CTFE for this expression
    virtual Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);

    virtual int isConst();
    virtual int isBool(int result);
    virtual int isBit();
    bool hasSideEffect();
    virtual int checkSideEffect(int flag);
    virtual int canThrow();

    virtual int inlineCost3(InlineCostState *ics);
    virtual Expression *doInline(InlineDoState *ids);
    virtual Expression *inlineScan(InlineScanState *iss);
    Expression *inlineCopy(Scope *sc);

    // For operator overloading
    virtual int isCommutative();
    virtual Identifier *opId();
    virtual Identifier *opId_r();

    // For array ops
    virtual void buildArrayIdent(OutBuffer *buf, Expressions *arguments);
    virtual Expression *buildArrayLoop(Parameters *fparams);
    int isArrayOperand();

#if IN_DMD
    // Back end
    virtual elem *toElem(IRState *irs);
    virtual dt_t **toDt(dt_t **pdt);
#endif

#if IN_LLVM
    virtual DValue* toElem(IRState* irs);
    DValue *toElemDtor(IRState *irs);
    virtual llvm::Constant *toConstElem(IRState *irs);
    virtual void cacheLvalue(IRState* irs);

    llvm::Value* cachedLvalue;

    virtual AssignExp* isAssignExp() { return NULL; }
#endif
};

struct IntegerExp : Expression
{
    dinteger_t value;

    IntegerExp(Loc loc, dinteger_t value, Type *type);
    IntegerExp(dinteger_t value);
    int equals(Object *o);
    Expression *semantic(Scope *sc);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);
    char *toChars();
    void dump(int indent);
    dinteger_t toInteger();
    real_t toReal();
    real_t toImaginary();
    complex_t toComplex();
    int isConst();
    int isBool(int result);
    MATCH implicitConvTo(Type *t);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    void toMangleBuffer(OutBuffer *buf);
    Expression *toLvalue(Scope *sc, Expression *e);
#if IN_DMD
    elem *toElem(IRState *irs);
    dt_t **toDt(dt_t **pdt);
#elif IN_LLVM
    DValue* toElem(IRState* irs);
    llvm::Constant *toConstElem(IRState *irs);
#endif
};

struct ErrorExp : IntegerExp
{
    ErrorExp();

    Expression *implicitCastTo(Scope *sc, Type *t);
    Expression *castTo(Scope *sc, Type *t);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
};

struct RealExp : Expression
{
    real_t value;

    RealExp(Loc loc, real_t value, Type *type);
    int equals(Object *o);
    Expression *semantic(Scope *sc);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);
    char *toChars();
    dinteger_t toInteger();
    uinteger_t toUInteger();
    real_t toReal();
    real_t toImaginary();
    complex_t toComplex();
    Expression *castTo(Scope *sc, Type *t);
    int isConst();
    int isBool(int result);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    void toMangleBuffer(OutBuffer *buf);
#if IN_DMD
    elem *toElem(IRState *irs);
    dt_t **toDt(dt_t **pdt);
#elif IN_LLVM
    DValue* toElem(IRState* irs);
    llvm::Constant *toConstElem(IRState *irs);
#endif
};

struct ComplexExp : Expression
{
    complex_t value;

    ComplexExp(Loc loc, complex_t value, Type *type);
    int equals(Object *o);
    Expression *semantic(Scope *sc);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);
    char *toChars();
    dinteger_t toInteger();
    uinteger_t toUInteger();
    real_t toReal();
    real_t toImaginary();
    complex_t toComplex();
    Expression *castTo(Scope *sc, Type *t);
    int isConst();
    int isBool(int result);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    void toMangleBuffer(OutBuffer *buf);
    OutBuffer hexp;
#if IN_DMD
    elem *toElem(IRState *irs);
    dt_t **toDt(dt_t **pdt);
#elif IN_LLVM
    DValue* toElem(IRState* irs);
    llvm::Constant *toConstElem(IRState *irs);
#endif
};

struct IdentifierExp : Expression
{
    Identifier *ident;
    Declaration *var;

    IdentifierExp(Loc loc, Identifier *ident);
    IdentifierExp(Loc loc, Declaration *var);
    Expression *semantic(Scope *sc);
    char *toChars();
    void dump(int indent);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    int isLvalue();
    Expression *toLvalue(Scope *sc, Expression *e);
};

struct DollarExp : IdentifierExp
{
    DollarExp(Loc loc);
};

struct DsymbolExp : Expression
{
    Dsymbol *s;
    int hasOverloads;

    DsymbolExp(Loc loc, Dsymbol *s);
    Expression *semantic(Scope *sc);
    char *toChars();
    void dump(int indent);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    int isLvalue();
    Expression *toLvalue(Scope *sc, Expression *e);
};

struct ThisExp : Expression
{
    Declaration *var;

    ThisExp(Loc loc);
    Expression *semantic(Scope *sc);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);
    int isBool(int result);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    int isLvalue();
    Expression *toLvalue(Scope *sc, Expression *e);

    int inlineCost3(InlineCostState *ics);
    Expression *doInline(InlineDoState *ids);
    //Expression *inlineScan(InlineScanState *iss);

#if IN_DMD
    elem *toElem(IRState *irs);
#endif

#if IN_LLVM
    DValue* toElem(IRState* irs);
#endif
};

struct SuperExp : ThisExp
{
    SuperExp(Loc loc);
    Expression *semantic(Scope *sc);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);

    Expression *doInline(InlineDoState *ids);
    //Expression *inlineScan(InlineScanState *iss);
};

struct NullExp : Expression
{
    unsigned char committed;    // !=0 if type is committed

    NullExp(Loc loc, Type *t = NULL);
    Expression *semantic(Scope *sc);
    int isBool(int result);
    int isConst();
    StringExp *toString();
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    void toMangleBuffer(OutBuffer *buf);
    MATCH implicitConvTo(Type *t);
    Expression *castTo(Scope *sc, Type *t);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);
#if IN_DMD
    elem *toElem(IRState *irs);
    dt_t **toDt(dt_t **pdt);
#elif IN_LLVM
    DValue* toElem(IRState* irs);
    llvm::Constant *toConstElem(IRState *irs);
#endif
};

struct StringExp : Expression
{
    void *string;       // char, wchar, or dchar data
    size_t len;         // number of chars, wchars, or dchars
    unsigned char sz;   // 1: char, 2: wchar, 4: dchar
    unsigned char committed;    // !=0 if type is committed
    unsigned char postfix;      // 'c', 'w', 'd'
    bool ownedByCtfe;   // true = created in CTFE

    StringExp(Loc loc, char *s);
    StringExp(Loc loc, void *s, size_t len);
    StringExp(Loc loc, void *s, size_t len, unsigned char postfix);
    //Expression *syntaxCopy();
    int equals(Object *o);
    char *toChars();
    Expression *semantic(Scope *sc);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);
    size_t length();
    StringExp *toString();
    StringExp *toUTF8(Scope *sc);
    MATCH implicitConvTo(Type *t);
    Expression *castTo(Scope *sc, Type *t);
    int compare(Object *obj);
    int isBool(int result);
    int isLvalue();
    unsigned charAt(size_t i);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    void toMangleBuffer(OutBuffer *buf);
#if IN_DMD
    elem *toElem(IRState *irs);
    dt_t **toDt(dt_t **pdt);
#elif IN_LLVM
    DValue* toElem(IRState* irs);
    llvm::Constant *toConstElem(IRState *irs);
#endif
};

// Tuple

struct TupleExp : Expression
{
    Expressions *exps;

    TupleExp(Loc loc, Expressions *exps);
    TupleExp(Loc loc, TupleDeclaration *tup);
    Expression *syntaxCopy();
    int apply(apply_fp_t fp, void *param);
    int equals(Object *o);
    Expression *semantic(Scope *sc);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    void checkEscape();
    int checkSideEffect(int flag);
    Expression *optimize(int result);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);
    Expression *castTo(Scope *sc, Type *t);
#if IN_DMD
    elem *toElem(IRState *irs);
#endif

    Expression *doInline(InlineDoState *ids);
    Expression *inlineScan(InlineScanState *iss);

#if IN_LLVM
    DValue* toElem(IRState* irs);
#endif
};

struct ArrayLiteralExp : Expression
{
    Expressions *elements;
    bool ownedByCtfe;   // true = created in CTFE

    ArrayLiteralExp(Loc loc, Expressions *elements);
    ArrayLiteralExp(Loc loc, Expression *e);

    Expression *syntaxCopy();
    int apply(apply_fp_t fp, void *param);
    Expression *semantic(Scope *sc);
    int isBool(int result);
#if IN_DMD
    elem *toElem(IRState *irs);
#endif
    int checkSideEffect(int flag);
    StringExp *toString();
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    void toMangleBuffer(OutBuffer *buf);
    Expression *optimize(int result);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);
    MATCH implicitConvTo(Type *t);
    Expression *castTo(Scope *sc, Type *t);
#if IN_DMD
    dt_t **toDt(dt_t **pdt);
#endif

    Expression *doInline(InlineDoState *ids);
    Expression *inlineScan(InlineScanState *iss);

#if IN_LLVM
    DValue* toElem(IRState* irs);
    llvm::Constant *toConstElem(IRState *irs);
#endif
};

struct AssocArrayLiteralExp : Expression
{
    Expressions *keys;
    Expressions *values;
    bool ownedByCtfe;   // true = created in CTFE

    AssocArrayLiteralExp(Loc loc, Expressions *keys, Expressions *values);

    Expression *syntaxCopy();
    int apply(apply_fp_t fp, void *param);
    Expression *semantic(Scope *sc);
    int isBool(int result);
#if IN_DMD
    elem *toElem(IRState *irs);
#endif
    int checkSideEffect(int flag);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    void toMangleBuffer(OutBuffer *buf);
    Expression *optimize(int result);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);
    MATCH implicitConvTo(Type *t);
    Expression *castTo(Scope *sc, Type *t);

    Expression *doInline(InlineDoState *ids);
    Expression *inlineScan(InlineScanState *iss);

#if IN_LLVM
    DValue* toElem(IRState* irs);
    llvm::Constant *toConstElem(IRState *irs);
#endif
};

struct StructLiteralExp : Expression
{
    StructDeclaration *sd;      // which aggregate this is for
    Expressions *elements;      // parallels sd->fields[] with
                                // NULL entries for fields to skip
    Type *stype;                // final type of result (can be different from sd's type)

#if IN_DMD
    Symbol *sym;                // back end symbol to initialize with literal
#endif
    size_t soffset;             // offset from start of s
    int fillHoles;              // fill alignment 'holes' with zero
    bool ownedByCtfe;           // true = created in CTFE

    StructLiteralExp(Loc loc, StructDeclaration *sd, Expressions *elements, Type *stype = NULL);

    Expression *syntaxCopy();
    int apply(apply_fp_t fp, void *param);
    Expression *semantic(Scope *sc);
    Expression *getField(Type *type, unsigned offset);
    int getFieldIndex(Type *type, unsigned offset);
#if IN_DMD
    elem *toElem(IRState *irs);
#endif
    int checkSideEffect(int flag);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    void toMangleBuffer(OutBuffer *buf);
    Expression *optimize(int result);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);
#if IN_DMD
    dt_t **toDt(dt_t **pdt);
#endif
    Expression *toLvalue(Scope *sc, Expression *e);

    int inlineCost3(InlineCostState *ics);
    Expression *doInline(InlineDoState *ids);
    Expression *inlineScan(InlineScanState *iss);

#if IN_LLVM
    DValue* toElem(IRState* irs);
    llvm::Constant *toConstElem(IRState *irs);
    llvm::StructType *constType;
#endif
};

Expression *typeDotIdExp(Loc loc, Type *type, Identifier *ident);
#if IN_DMD
#endif

#if IN_LLVM
    DValue* toElem(IRState* irs);
#endif

struct TypeExp : Expression
{
    TypeExp(Loc loc, Type *type);
    Expression *syntaxCopy();
    Expression *semantic(Scope *sc);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    Expression *optimize(int result);
#if IN_DMD
    elem *toElem(IRState *irs);
#endif

#if IN_LLVM
    DValue* toElem(IRState* irs);
#endif
};

struct ScopeExp : Expression
{
    ScopeDsymbol *sds;

    ScopeExp(Loc loc, ScopeDsymbol *sds);
    Expression *syntaxCopy();
    Expression *semantic(Scope *sc);
#if IN_DMD
    elem *toElem(IRState *irs);
#endif
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);

#if IN_LLVM
    DValue* toElem(IRState* irs);
#endif
};

struct TemplateExp : Expression
{
    TemplateDeclaration *td;

    TemplateExp(Loc loc, TemplateDeclaration *td);
    int rvalue();
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
};

struct NewExp : Expression
{
    /* thisexp.new(newargs) newtype(arguments)
     */
    Expression *thisexp;        // if !NULL, 'this' for class being allocated
    Expressions *newargs;       // Array of Expression's to call new operator
    Type *newtype;
    Expressions *arguments;     // Array of Expression's

    CtorDeclaration *member;    // constructor function
    NewDeclaration *allocator;  // allocator function
    int onstack;                // allocate on stack

    NewExp(Loc loc, Expression *thisexp, Expressions *newargs,
        Type *newtype, Expressions *arguments);
    Expression *syntaxCopy();
    int apply(apply_fp_t fp, void *param);
    Expression *semantic(Scope *sc);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);
    Expression *optimize(int result);
#if IN_DMD
    elem *toElem(IRState *irs);
#endif
    int checkSideEffect(int flag);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);

    //int inlineCost3(InlineCostState *ics);
    Expression *doInline(InlineDoState *ids);
    //Expression *inlineScan(InlineScanState *iss);

#if IN_LLVM
    DValue* toElem(IRState* irs);
#endif
};

struct NewAnonClassExp : Expression
{
    /* thisexp.new(newargs) class baseclasses { } (arguments)
     */
    Expression *thisexp;        // if !NULL, 'this' for class being allocated
    Expressions *newargs;       // Array of Expression's to call new operator
    ClassDeclaration *cd;       // class being instantiated
    Expressions *arguments;     // Array of Expression's to call class constructor

    NewAnonClassExp(Loc loc, Expression *thisexp, Expressions *newargs,
        ClassDeclaration *cd, Expressions *arguments);
    Expression *syntaxCopy();
    int apply(apply_fp_t fp, void *param);
    Expression *semantic(Scope *sc);
    int checkSideEffect(int flag);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
};

#if DMDV2
struct SymbolExp : Expression
{
    Declaration *var;
    int hasOverloads;

    SymbolExp(Loc loc, enum TOK op, int size, Declaration *var, int hasOverloads);

    elem *toElem(IRState *irs);
};
#endif

// Offset from symbol

struct SymOffExp : Expression
{
    Declaration *var;
    unsigned offset;
    Module* m;  // starting point for overload resolution

    SymOffExp(Loc loc, Declaration *var, unsigned offset);
    Expression *semantic(Scope *sc);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);
    void checkEscape();
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    int isConst();
    int isBool(int result);
    Expression *doInline(InlineDoState *ids);
    MATCH implicitConvTo(Type *t);
    Expression *castTo(Scope *sc, Type *t);

#if IN_DMD
    elem *toElem(IRState *irs);
    dt_t **toDt(dt_t **pdt);
#endif

#if IN_LLVM
    DValue* toElem(IRState* irs);
#endif
};

// Variable

struct VarExp : Expression
{
    Declaration *var;

    VarExp(Loc loc, Declaration *var);
    int equals(Object *o);
    Expression *semantic(Scope *sc);
    Expression *optimize(int result);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);
    void dump(int indent);
    char *toChars();
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    void checkEscape();
    void checkEscapeRef();
    int isLvalue();
    Expression *toLvalue(Scope *sc, Expression *e);
    Expression *modifiableLvalue(Scope *sc, Expression *e);
#if IN_DMD
    elem *toElem(IRState *irs);
    dt_t **toDt(dt_t **pdt);
#endif

    Expression *doInline(InlineDoState *ids);
    //Expression *inlineScan(InlineScanState *iss);

#if IN_LLVM
    DValue* toElem(IRState* irs);
    llvm::Constant *toConstElem(IRState *irs);
    void cacheLvalue(IRState* irs);
#endif
};

#if DMDV2
// Overload Set

struct OverExp : Expression
{
    OverloadSet *vars;

    OverExp(OverloadSet *s);
    int isLvalue();
    Expression *toLvalue(Scope *sc, Expression *e);
};
#endif

// Function/Delegate literal

struct FuncExp : Expression
{
    FuncLiteralDeclaration *fd;

    FuncExp(Loc loc, FuncLiteralDeclaration *fd);
    Expression *syntaxCopy();
    Expression *semantic(Scope *sc);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);
    char *toChars();
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
#if IN_DMD
    elem *toElem(IRState *irs);
    dt_t **toDt(dt_t **pdt);
#endif

    int inlineCost3(InlineCostState *ics);
    //Expression *doInline(InlineDoState *ids);
    //Expression *inlineScan(InlineScanState *iss);

#if IN_LLVM
    DValue* toElem(IRState* irs);
    llvm::Constant *toConstElem(IRState *irs);
#endif
};

// Declaration of a symbol

struct DeclarationExp : Expression
{
    Dsymbol *declaration;

    DeclarationExp(Loc loc, Dsymbol *declaration);
    Expression *syntaxCopy();
    Expression *semantic(Scope *sc);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);
    int checkSideEffect(int flag);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
#if IN_DMD
    elem *toElem(IRState *irs);
#endif

    int inlineCost3(InlineCostState *ics);
    Expression *doInline(InlineDoState *ids);
    Expression *inlineScan(InlineScanState *iss);

#if IN_LLVM
    DValue* toElem(IRState* irs);
#endif
};

struct TypeidExp : Expression
{
    Type *typeidType;

    TypeidExp(Loc loc, Type *typeidType);
    Expression *syntaxCopy();
    Expression *semantic(Scope *sc);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
};

#if DMDV2
struct TraitsExp : Expression
{
    Identifier *ident;
    Objects *args;

    TraitsExp(Loc loc, Identifier *ident, Objects *args);
    Expression *syntaxCopy();
    Expression *semantic(Scope *sc);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
};
#endif

struct HaltExp : Expression
{
    HaltExp(Loc loc);
    Expression *semantic(Scope *sc);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    int checkSideEffect(int flag);

#if IN_DMD
    elem *toElem(IRState *irs);
#endif

#if IN_LLVM
    DValue* toElem(IRState* irs);
#endif
};

struct IsExp : Expression
{
    /* is(targ id tok tspec)
     * is(targ id == tok2)
     */
    Type *targ;
    Identifier *id;     // can be NULL
    enum TOK tok;       // ':' or '=='
    Type *tspec;        // can be NULL
    enum TOK tok2;      // 'struct', 'union', 'typedef', etc.

    IsExp(Loc loc, Type *targ, Identifier *id, enum TOK tok, Type *tspec, enum TOK tok2);
    Expression *syntaxCopy();
    Expression *semantic(Scope *sc);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
};

/****************************************************************/

struct UnaExp : Expression
{
    Expression *e1;

    UnaExp(Loc loc, enum TOK op, int size, Expression *e1);
    Expression *syntaxCopy();
    int apply(apply_fp_t fp, void *param);
    Expression *semantic(Scope *sc);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    Expression *optimize(int result);
    void dump(int indent);
    Expression *interpretCommon(InterState *istate, CtfeGoal goal, Expression *(*fp)(Type *, Expression *));

    Expression *doInline(InlineDoState *ids);
    Expression *inlineScan(InlineScanState *iss);

    Expression *op_overload(Scope *sc); // doesn't need to be virtual
};

struct BinExp : Expression
{
    Expression *e1;
    Expression *e2;

    BinExp(Loc loc, enum TOK op, int size, Expression *e1, Expression *e2);
    Expression *syntaxCopy();
    int apply(apply_fp_t fp, void *param);
    Expression *semantic(Scope *sc);
    Expression *semanticp(Scope *sc);
    Expression *commonSemanticAssign(Scope *sc);
    Expression *commonSemanticAssignIntegral(Scope *sc);
    int checkSideEffect(int flag);
    void checkComplexMulAssign();
    void checkComplexAddAssign();
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    Expression *scaleFactor(Scope *sc);
    Expression *typeCombine(Scope *sc);
    Expression *optimize(int result);
    int isunsigned();
    void incompatibleTypes();
    void dump(int indent);
    Expression *interpretCommon(InterState *istate, CtfeGoal goal,
        Expression *(*fp)(Type *, Expression *, Expression *));
    Expression *interpretCommon2(InterState *istate, CtfeGoal goal,
        Expression *(*fp)(Loc, TOK, Type *, Expression *, Expression *));
    Expression *interpretAssignCommon(InterState *istate, CtfeGoal goal,
        Expression *(*fp)(Type *, Expression *, Expression *), int post = 0);
    Expression *interpretFourPointerRelation(InterState *istate, CtfeGoal goal);
    Expression *arrayOp(Scope *sc);

    Expression *doInline(InlineDoState *ids);
    Expression *inlineScan(InlineScanState *iss);

    Expression *op_overload(Scope *sc);

#if IN_DMD
    elem *toElemBin(IRState *irs, int op);
#endif
};

struct BinAssignExp : BinExp
{
    BinAssignExp(Loc loc, enum TOK op, int size, Expression *e1, Expression *e2)
        : BinExp(loc, op, size, e1, e2)
    {
    }

    Expression *semantic(Scope *sc);
    int isLvalue();
};

/****************************************************************/

struct CompileExp : UnaExp
{
    CompileExp(Loc loc, Expression *e);
    Expression *semantic(Scope *sc);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
};

struct FileExp : UnaExp
{
    FileExp(Loc loc, Expression *e);
    Expression *semantic(Scope *sc);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
};

struct AssertExp : UnaExp
{
    Expression *msg;

    AssertExp(Loc loc, Expression *e, Expression *msg = NULL);
    Expression *syntaxCopy();
    int apply(apply_fp_t fp, void *param);
    Expression *semantic(Scope *sc);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);
    int checkSideEffect(int flag);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);

    Expression *doInline(InlineDoState *ids);
    Expression *inlineScan(InlineScanState *iss);

#if IN_DMD
    elem *toElem(IRState *irs);
#endif

#if IN_LLVM
    DValue* toElem(IRState* irs);
#endif
};

struct DotIdExp : UnaExp
{
    Identifier *ident;

    DotIdExp(Loc loc, Expression *e, Identifier *ident);
    Expression *semantic(Scope *sc);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    void dump(int i);
};

struct DotTemplateExp : UnaExp
{
    TemplateDeclaration *td;

    DotTemplateExp(Loc loc, Expression *e, TemplateDeclaration *td);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
};

struct DotVarExp : UnaExp
{
    Declaration *var;

    DotVarExp(Loc loc, Expression *e, Declaration *var);
    Expression *semantic(Scope *sc);
    int isLvalue();
    Expression *toLvalue(Scope *sc, Expression *e);
    Expression *modifiableLvalue(Scope *sc, Expression *e);
    Expression *optimize(int result);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    void dump(int indent);
#if IN_DMD
    elem *toElem(IRState *irs);
#endif

#if IN_LLVM
    DValue* toElem(IRState* irs);
    void cacheLvalue(IRState* irs);
#endif
};

struct DotTemplateInstanceExp : UnaExp
{
    TemplateInstance *ti;

    DotTemplateInstanceExp(Loc loc, Expression *e, TemplateInstance *ti);
    Expression *syntaxCopy();
    Expression *semantic(Scope *sc);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    void dump(int indent);
};

struct DelegateExp : UnaExp
{
    FuncDeclaration *func;
    Module* m;  // starting point for overload resolution
    int hasOverloads;

    DelegateExp(Loc loc, Expression *e, FuncDeclaration *func);
    Expression *semantic(Scope *sc);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);
    MATCH implicitConvTo(Type *t);
    Expression *castTo(Scope *sc, Type *t);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    void dump(int indent);

    int inlineCost3(InlineCostState *ics);
#if IN_DMD
    elem *toElem(IRState *irs);
#endif

#if IN_LLVM
    DValue* toElem(IRState* irs);
#endif
};

struct DotTypeExp : UnaExp
{
    Dsymbol *sym;               // symbol that represents a type

    DotTypeExp(Loc loc, Expression *e, Dsymbol *sym);
    Expression *semantic(Scope *sc);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
#if IN_DMD
    elem *toElem(IRState *irs);
#endif

#if IN_LLVM
    DValue* toElem(IRState* irs);
#endif
};

struct CallExp : UnaExp
{
    Expressions *arguments;     // function arguments

    CallExp(Loc loc, Expression *e, Expressions *exps);
    CallExp(Loc loc, Expression *e);
    CallExp(Loc loc, Expression *e, Expression *earg1);
    CallExp(Loc loc, Expression *e, Expression *earg1, Expression *earg2);

    Expression *syntaxCopy();
    int apply(apply_fp_t fp, void *param);
    Expression *semantic(Scope *sc);
    Expression *optimize(int result);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);
    int checkSideEffect(int flag);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    void dump(int indent);
#if IN_DMD
    elem *toElem(IRState *irs);
#endif
    int isLvalue();
    Expression *toLvalue(Scope *sc, Expression *e);
    Expression *modifiableLvalue(Scope *sc, Expression *e);

    int inlineCost3(InlineCostState *ics);
    Expression *doInline(InlineDoState *ids);
    Expression *inlineScan(InlineScanState *iss);

#if IN_LLVM
    void cacheLvalue(IRState* p);
    DValue* toElem(IRState* irs);
#endif
};

struct AddrExp : UnaExp
{
    Module* m;  // starting point for overload resolution

    AddrExp(Loc loc, Expression *e);
    Expression *semantic(Scope *sc);
    void checkEscape();
#if IN_DMD
    elem *toElem(IRState *irs);
#endif
    MATCH implicitConvTo(Type *t);
    Expression *castTo(Scope *sc, Type *t);
    Expression *optimize(int result);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);

#if IN_LLVM
    DValue* toElem(IRState* irs);
    llvm::Constant *toConstElem(IRState *irs);
#endif
};

struct PtrExp : UnaExp
{
    PtrExp(Loc loc, Expression *e);
    PtrExp(Loc loc, Expression *e, Type *t);
    Expression *semantic(Scope *sc);
    int isLvalue();
    void checkEscapeRef();
    Expression *toLvalue(Scope *sc, Expression *e);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
#if IN_DMD
    elem *toElem(IRState *irs);
#endif
    Expression *optimize(int result);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);

#if IN_LLVM
    DValue* toElem(IRState* irs);
    void cacheLvalue(IRState* irs);
#endif
};

struct NegExp : UnaExp
{
    NegExp(Loc loc, Expression *e);
    Expression *semantic(Scope *sc);
    Expression *optimize(int result);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);
    void buildArrayIdent(OutBuffer *buf, Expressions *arguments);
    Expression *buildArrayLoop(Parameters *fparams);

    // For operator overloading
    Identifier *opId();

#if IN_DMD
    elem *toElem(IRState *irs);
#endif

#if IN_LLVM
    DValue* toElem(IRState* irs);
#endif
};

struct UAddExp : UnaExp
{
    UAddExp(Loc loc, Expression *e);
    Expression *semantic(Scope *sc);

    // For operator overloading
    Identifier *opId();
};

struct ComExp : UnaExp
{
    ComExp(Loc loc, Expression *e);
    Expression *semantic(Scope *sc);
    Expression *optimize(int result);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);
    void buildArrayIdent(OutBuffer *buf, Expressions *arguments);
    Expression *buildArrayLoop(Parameters *fparams);

    // For operator overloading
    Identifier *opId();

#if IN_DMD
    elem *toElem(IRState *irs);
#endif

#if IN_LLVM
    DValue* toElem(IRState* irs);
#endif
};

struct NotExp : UnaExp
{
    NotExp(Loc loc, Expression *e);
    Expression *semantic(Scope *sc);
    Expression *optimize(int result);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);
    int isBit();
#if IN_DMD
    elem *toElem(IRState *irs);
#endif

#if IN_LLVM
    DValue* toElem(IRState* irs);
#endif
};

struct BoolExp : UnaExp
{
    BoolExp(Loc loc, Expression *e, Type *type);
    Expression *semantic(Scope *sc);
    Expression *optimize(int result);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);
    int isBit();
#if IN_DMD
    elem *toElem(IRState *irs);
#endif

#if IN_LLVM
    DValue* toElem(IRState* irs);
#endif
};

struct DeleteExp : UnaExp
{
    DeleteExp(Loc loc, Expression *e);
    Expression *semantic(Scope *sc);
    Expression *checkToBoolean();
    int checkSideEffect(int flag);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
#if IN_DMD
    elem *toElem(IRState *irs);
#endif

#if IN_LLVM
    DValue* toElem(IRState* irs);
#endif
};

struct CastExp : UnaExp
{
    // Possible to cast to one type while painting to another type
    Type *to;                   // type to cast to

    CastExp(Loc loc, Expression *e, Type *t);
    Expression *syntaxCopy();
    Expression *semantic(Scope *sc);
    Expression *optimize(int result);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);
    int checkSideEffect(int flag);
    void checkEscape();
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    void buildArrayIdent(OutBuffer *buf, Expressions *arguments);
    Expression *buildArrayLoop(Parameters *fparams);
#if IN_DMD
    elem *toElem(IRState *irs);
#endif

    // For operator overloading
    Identifier *opId();

#if IN_LLVM
    DValue* toElem(IRState* irs);
    llvm::Constant *toConstElem(IRState *irs);
#endif
};


struct SliceExp : UnaExp
{
    Expression *upr;            // NULL if implicit 0
    Expression *lwr;            // NULL if implicit [length - 1]
    VarDeclaration *lengthVar;

    SliceExp(Loc loc, Expression *e1, Expression *lwr, Expression *upr);
    Expression *syntaxCopy();
    int apply(apply_fp_t fp, void *param);
    Expression *semantic(Scope *sc);
    void checkEscape();
    void checkEscapeRef();
    int isLvalue();
    Expression *toLvalue(Scope *sc, Expression *e);
    Expression *modifiableLvalue(Scope *sc, Expression *e);
    int isBool(int result);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    Expression *optimize(int result);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);
    void dump(int indent);
#if IN_DMD
    elem *toElem(IRState *irs);
#endif
    void buildArrayIdent(OutBuffer *buf, Expressions *arguments);
    Expression *buildArrayLoop(Parameters *fparams);
    int canThrow();

    Expression *doInline(InlineDoState *ids);
    Expression *inlineScan(InlineScanState *iss);

#if IN_LLVM
    DValue* toElem(IRState* irs);
    llvm::Constant *toConstElem(IRState *irs);
#endif
};

struct ArrayLengthExp : UnaExp
{
    ArrayLengthExp(Loc loc, Expression *e1);
    Expression *semantic(Scope *sc);
    Expression *optimize(int result);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
#if IN_DMD
    elem *toElem(IRState *irs);
#endif

#if IN_LLVM
    DValue* toElem(IRState* irs);
#endif
};

// e1[a0,a1,a2,a3,...]

struct ArrayExp : UnaExp
{
    Expressions *arguments;             // Array of Expression's

    ArrayExp(Loc loc, Expression *e1, Expressions *arguments);
    Expression *syntaxCopy();
    int apply(apply_fp_t fp, void *param);
    Expression *semantic(Scope *sc);
    int isLvalue();
    Expression *toLvalue(Scope *sc, Expression *e);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);

    // For operator overloading
    Identifier *opId();

    Expression *doInline(InlineDoState *ids);
    Expression *inlineScan(InlineScanState *iss);
};

/****************************************************************/

struct DotExp : BinExp
{
    DotExp(Loc loc, Expression *e1, Expression *e2);
    Expression *semantic(Scope *sc);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
};

struct CommaExp : BinExp
{
    CommaExp(Loc loc, Expression *e1, Expression *e2);
    Expression *semantic(Scope *sc);
    void checkEscape();
    void checkEscapeRef();
    int isLvalue();
    Expression *toLvalue(Scope *sc, Expression *e);
    Expression *modifiableLvalue(Scope *sc, Expression *e);
    int isBool(int result);
    int checkSideEffect(int flag);
    Expression *optimize(int result);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);
#if IN_DMD
    elem *toElem(IRState *irs);
#endif

#if IN_LLVM
    void cacheLvalue(IRState* p);
    DValue* toElem(IRState* irs);
#endif
};

struct IndexExp : BinExp
{
    VarDeclaration *lengthVar;
    int modifiable;

    IndexExp(Loc loc, Expression *e1, Expression *e2);
    Expression *semantic(Scope *sc);
    int isLvalue();
    Expression *toLvalue(Scope *sc, Expression *e);
    Expression *modifiableLvalue(Scope *sc, Expression *e);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    Expression *optimize(int result);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);
    Expression *doInline(InlineDoState *ids);

#if IN_DMD
    elem *toElem(IRState *irs);
#elif IN_LLVM
    DValue* toElem(IRState* irs);
    llvm::Constant *toConstElem(IRState *irs);
    void cacheLvalue(IRState* irs);
#endif
};

/* For both i++ and i--
 */
struct PostExp : BinExp
{
    PostExp(enum TOK op, Loc loc, Expression *e);
    Expression *semantic(Scope *sc);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    Identifier *opId();    // For operator overloading
#if IN_DMD
    elem *toElem(IRState *irs);
#endif

#if IN_LLVM
    DValue* toElem(IRState* irs);
#endif
};

struct AssignExp : BinExp
{   int ismemset;       // !=0 if setting the contents of an array

    AssignExp(Loc loc, Expression *e1, Expression *e2);
    Expression *semantic(Scope *sc);
    Expression *checkToBoolean();
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);
    Identifier *opId();    // For operator overloading
    void buildArrayIdent(OutBuffer *buf, Expressions *arguments);
    Expression *buildArrayLoop(Parameters *fparams);
#if IN_DMD
    elem *toElem(IRState *irs);
#endif

#if IN_LLVM
    DValue* toElem(IRState* irs);
#endif

    AssignExp* isAssignExp() { return this; }
};

struct ConstructExp : AssignExp
{
    ConstructExp(Loc loc, Expression *e1, Expression *e2);
};

#if IN_DMD
#define ASSIGNEXP_TOELEM    elem *toElem(IRState *irs);
#elif IN_LLVM
#define ASSIGNEXP_TOELEM    DValue* toElem(IRState *irs);
#else
#define ASSIGNEXP_TOELEM
#endif

#define ASSIGNEXP(op)   \
struct op##AssignExp : BinAssignExp                             \
{                                                               \
    op##AssignExp(Loc loc, Expression *e1, Expression *e2);     \
    S(Expression *semantic(Scope *sc);)                          \
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);                  \
    X(void buildArrayIdent(OutBuffer *buf, Expressions *arguments);) \
    X(Expression *buildArrayLoop(Parameters *fparams);)         \
                                                                \
    Identifier *opId();    /* For operator overloading */       \
                                                                \
    ASSIGNEXP_TOELEM                                            \
};

#define X(a) a
#define S(a) a
ASSIGNEXP(Add)
ASSIGNEXP(Min)
ASSIGNEXP(Mul)
ASSIGNEXP(Div)
ASSIGNEXP(Mod)
ASSIGNEXP(And)
ASSIGNEXP(Or)
ASSIGNEXP(Xor)
#undef S

#if DMDV2
#define S(a) a
ASSIGNEXP(Pow)
#undef S
#endif

#undef X

#define X(a)
#define S(a)

ASSIGNEXP(Shl)
ASSIGNEXP(Shr)
ASSIGNEXP(Ushr)
#undef S

#define S(a) a
ASSIGNEXP(Cat)

#undef S
#undef X
#undef ASSIGNEXP
#undef ASSIGNEXP_TOELEM

struct AddExp : BinExp
{
    AddExp(Loc loc, Expression *e1, Expression *e2);
    Expression *semantic(Scope *sc);
    Expression *optimize(int result);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);
    void buildArrayIdent(OutBuffer *buf, Expressions *arguments);
    Expression *buildArrayLoop(Parameters *fparams);

    // For operator overloading
    int isCommutative();
    Identifier *opId();
    Identifier *opId_r();

#if IN_DMD
    elem *toElem(IRState *irs);
#endif

#if IN_LLVM
    llvm::Constant* toConstElem(IRState* p);
    DValue* toElem(IRState* irs);
#endif
};

struct MinExp : BinExp
{
    MinExp(Loc loc, Expression *e1, Expression *e2);
    Expression *semantic(Scope *sc);
    Expression *optimize(int result);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);
    void buildArrayIdent(OutBuffer *buf, Expressions *arguments);
    Expression *buildArrayLoop(Parameters *fparams);

    // For operator overloading
    Identifier *opId();
    Identifier *opId_r();

#if IN_DMD
    elem *toElem(IRState *irs);
#endif

#if IN_LLVM
    llvm::Constant* toConstElem(IRState* p);
    DValue* toElem(IRState* irs);
#endif
};

struct CatExp : BinExp
{
    CatExp(Loc loc, Expression *e1, Expression *e2);
    Expression *semantic(Scope *sc);
    Expression *optimize(int result);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);

    // For operator overloading
    Identifier *opId();
    Identifier *opId_r();

#if IN_DMD
    elem *toElem(IRState *irs);
#endif

#if IN_LLVM
    DValue* toElem(IRState* irs);
#endif
};

struct MulExp : BinExp
{
    MulExp(Loc loc, Expression *e1, Expression *e2);
    Expression *semantic(Scope *sc);
    Expression *optimize(int result);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);
    void buildArrayIdent(OutBuffer *buf, Expressions *arguments);
    Expression *buildArrayLoop(Parameters *fparams);

    // For operator overloading
    int isCommutative();
    Identifier *opId();
    Identifier *opId_r();

#if IN_DMD
    elem *toElem(IRState *irs);
#endif

#if IN_LLVM
    DValue* toElem(IRState* irs);
#endif
};

struct DivExp : BinExp
{
    DivExp(Loc loc, Expression *e1, Expression *e2);
    Expression *semantic(Scope *sc);
    Expression *optimize(int result);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);
    void buildArrayIdent(OutBuffer *buf, Expressions *arguments);
    Expression *buildArrayLoop(Parameters *fparams);

    // For operator overloading
    Identifier *opId();
    Identifier *opId_r();

#if IN_DMD
    elem *toElem(IRState *irs);
#endif

#if IN_LLVM
    DValue* toElem(IRState* irs);
#endif
};

struct ModExp : BinExp
{
    ModExp(Loc loc, Expression *e1, Expression *e2);
    Expression *semantic(Scope *sc);
    Expression *optimize(int result);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);
    void buildArrayIdent(OutBuffer *buf, Expressions *arguments);
    Expression *buildArrayLoop(Parameters *fparams);

    // For operator overloading
    Identifier *opId();
    Identifier *opId_r();

#if IN_DMD
    elem *toElem(IRState *irs);
#endif

#if IN_LLVM
    DValue* toElem(IRState* irs);
#endif
};

#if DMDV2
struct PowExp : BinExp
{
    PowExp(Loc loc, Expression *e1, Expression *e2);
    Expression *semantic(Scope *sc);

    // For operator overloading
    Identifier *opId();
    Identifier *opId_r();
};
#endif

struct ShlExp : BinExp
{
    ShlExp(Loc loc, Expression *e1, Expression *e2);
    Expression *semantic(Scope *sc);
    Expression *optimize(int result);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);

    // For operator overloading
    Identifier *opId();
    Identifier *opId_r();

#if IN_DMD
    elem *toElem(IRState *irs);
#endif

#if IN_LLVM
    DValue* toElem(IRState* irs);
#endif
};

struct ShrExp : BinExp
{
    ShrExp(Loc loc, Expression *e1, Expression *e2);
    Expression *semantic(Scope *sc);
    Expression *optimize(int result);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);

    // For operator overloading
    Identifier *opId();
    Identifier *opId_r();

#if IN_DMD
    elem *toElem(IRState *irs);
#endif

#if IN_LLVM
    DValue* toElem(IRState* irs);
#endif
};

struct UshrExp : BinExp
{
    UshrExp(Loc loc, Expression *e1, Expression *e2);
    Expression *semantic(Scope *sc);
    Expression *optimize(int result);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);

    // For operator overloading
    Identifier *opId();
    Identifier *opId_r();

#if IN_DMD
    elem *toElem(IRState *irs);
#endif

#if IN_LLVM
    DValue* toElem(IRState* irs);
#endif
};

struct AndExp : BinExp
{
    AndExp(Loc loc, Expression *e1, Expression *e2);
    Expression *semantic(Scope *sc);
    Expression *optimize(int result);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);
    void buildArrayIdent(OutBuffer *buf, Expressions *arguments);
    Expression *buildArrayLoop(Parameters *fparams);

    // For operator overloading
    int isCommutative();
    Identifier *opId();
    Identifier *opId_r();

#if IN_DMD
    elem *toElem(IRState *irs);
#endif

#if IN_LLVM
    DValue* toElem(IRState* irs);
#endif
};

struct OrExp : BinExp
{
    OrExp(Loc loc, Expression *e1, Expression *e2);
    Expression *semantic(Scope *sc);
    Expression *optimize(int result);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);
    void buildArrayIdent(OutBuffer *buf, Expressions *arguments);
    Expression *buildArrayLoop(Parameters *fparams);

    // For operator overloading
    int isCommutative();
    Identifier *opId();
    Identifier *opId_r();

#if IN_DMD
    elem *toElem(IRState *irs);
#endif

#if IN_LLVM
    DValue* toElem(IRState* irs);
#endif
};

struct XorExp : BinExp
{
    XorExp(Loc loc, Expression *e1, Expression *e2);
    Expression *semantic(Scope *sc);
    Expression *optimize(int result);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);
    void buildArrayIdent(OutBuffer *buf, Expressions *arguments);
    Expression *buildArrayLoop(Parameters *fparams);

    // For operator overloading
    int isCommutative();
    Identifier *opId();
    Identifier *opId_r();

#if IN_DMD
    elem *toElem(IRState *irs);
#endif

#if IN_LLVM
    DValue* toElem(IRState* irs);
#endif
};

struct OrOrExp : BinExp
{
    OrOrExp(Loc loc, Expression *e1, Expression *e2);
    Expression *semantic(Scope *sc);
    Expression *checkToBoolean();
    int isBit();
    Expression *optimize(int result);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);
    int checkSideEffect(int flag);
#if IN_DMD
    elem *toElem(IRState *irs);
#endif

#if IN_LLVM
    DValue* toElem(IRState* irs);
#endif
};

struct AndAndExp : BinExp
{
    AndAndExp(Loc loc, Expression *e1, Expression *e2);
    Expression *semantic(Scope *sc);
    Expression *checkToBoolean();
    int isBit();
    Expression *optimize(int result);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);
    int checkSideEffect(int flag);
#if IN_DMD
    elem *toElem(IRState *irs);
#endif

#if IN_LLVM
    DValue* toElem(IRState* irs);
#endif
};

struct CmpExp : BinExp
{
    CmpExp(enum TOK op, Loc loc, Expression *e1, Expression *e2);
    Expression *semantic(Scope *sc);
    Expression *optimize(int result);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);
    int isBit();

    // For operator overloading
    int isCommutative();
    Identifier *opId();

#if IN_DMD
    elem *toElem(IRState *irs);
#endif

#if IN_LLVM
    DValue* toElem(IRState* irs);
#endif
};

struct InExp : BinExp
{
    InExp(Loc loc, Expression *e1, Expression *e2);
    Expression *semantic(Scope *sc);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);
    int isBit();

    // For operator overloading
    Identifier *opId();
    Identifier *opId_r();

#if IN_DMD
    elem *toElem(IRState *irs);
#endif

#if IN_LLVM
    DValue* toElem(IRState* irs);
#endif
};

struct RemoveExp : BinExp
{
    RemoveExp(Loc loc, Expression *e1, Expression *e2);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
#if IN_DMD
    elem *toElem(IRState *irs);
#endif

#if IN_LLVM
    DValue* toElem(IRState* irs);
#endif
};

// == and !=

struct EqualExp : BinExp
{
    EqualExp(enum TOK op, Loc loc, Expression *e1, Expression *e2);
    Expression *semantic(Scope *sc);
    Expression *optimize(int result);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);
    int isBit();

    // For operator overloading
    int isCommutative();
    Identifier *opId();

#if IN_DMD
    elem *toElem(IRState *irs);
#endif

#if IN_LLVM
    DValue* toElem(IRState* irs);
#endif
};

// === and !===

struct IdentityExp : BinExp
{
    IdentityExp(enum TOK op, Loc loc, Expression *e1, Expression *e2);
    Expression *semantic(Scope *sc);
    int isBit();
    Expression *optimize(int result);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);
#if IN_DMD
    elem *toElem(IRState *irs);
#endif

#if IN_LLVM
    DValue* toElem(IRState* irs);
#endif
};

/****************************************************************/

struct CondExp : BinExp
{
    Expression *econd;

    CondExp(Loc loc, Expression *econd, Expression *e1, Expression *e2);
    Expression *syntaxCopy();
    int apply(apply_fp_t fp, void *param);
    Expression *semantic(Scope *sc);
    Expression *optimize(int result);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);
    void checkEscape();
    void checkEscapeRef();
    int isLvalue();
    Expression *toLvalue(Scope *sc, Expression *e);
    Expression *modifiableLvalue(Scope *sc, Expression *e);
    Expression *checkToBoolean();
    int checkSideEffect(int flag);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    MATCH implicitConvTo(Type *t);
    Expression *castTo(Scope *sc, Type *t);

    Expression *doInline(InlineDoState *ids);
    Expression *inlineScan(InlineScanState *iss);

#if IN_DMD
    elem *toElem(IRState *irs);
#endif

#if IN_LLVM
    DValue* toElem(IRState* irs);
#endif
};

#if DMDV2
/****************************************************************/

struct DefaultInitExp : Expression
{
    enum TOK subop;             // which of the derived classes this is

    DefaultInitExp(Loc loc, enum TOK subop, int size);
    virtual Expression *resolve(Loc loc, Scope *sc) = 0;
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
};

struct FileInitExp : DefaultInitExp
{
    FileInitExp(Loc loc);
    Expression *semantic(Scope *sc);
    Expression *resolve(Loc loc, Scope *sc);
};

struct LineInitExp : DefaultInitExp
{
    LineInitExp(Loc loc);
    Expression *semantic(Scope *sc);
    Expression *resolve(Loc loc, Scope *sc);
};
#endif

/****************************************************************/

#if IN_LLVM

// this stuff is strictly LDC

// Special expression to represent a LLVM GetElementPtr instruction.
struct GEPExp : UnaExp
{
    unsigned index;
    Identifier* ident;

    GEPExp(Loc loc, Expression* e, Identifier* id, unsigned idx);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    Expression *toLvalue(Scope *sc, Expression *e);

    DValue* toElem(IRState* irs);
    llvm::Constant *toConstElem(IRState *irs);
};

#endif

/****************************************************************/

/* Special values used by the interpreter
 */
#define EXP_CANT_INTERPRET      ((Expression *)1)
#define EXP_CONTINUE_INTERPRET  ((Expression *)2)
#define EXP_BREAK_INTERPRET     ((Expression *)3)
#define EXP_GOTO_INTERPRET      ((Expression *)4)
#define EXP_VOID_INTERPRET      ((Expression *)5)

Expression *expType(Type *type, Expression *e);

Expression *Neg(Type *type, Expression *e1);
Expression *Com(Type *type, Expression *e1);
Expression *Not(Type *type, Expression *e1);
Expression *Bool(Type *type, Expression *e1);
Expression *Cast(Type *type, Type *to, Expression *e1);
Expression *ArrayLength(Type *type, Expression *e1);
Expression *Ptr(Type *type, Expression *e1);

Expression *Add(Type *type, Expression *e1, Expression *e2);
Expression *Min(Type *type, Expression *e1, Expression *e2);
Expression *Mul(Type *type, Expression *e1, Expression *e2);
Expression *Div(Type *type, Expression *e1, Expression *e2);
Expression *Mod(Type *type, Expression *e1, Expression *e2);
Expression *Shl(Type *type, Expression *e1, Expression *e2);
Expression *Shr(Type *type, Expression *e1, Expression *e2);
Expression *Ushr(Type *type, Expression *e1, Expression *e2);
Expression *And(Type *type, Expression *e1, Expression *e2);
Expression *Or(Type *type, Expression *e1, Expression *e2);
Expression *Xor(Type *type, Expression *e1, Expression *e2);
Expression *Index(Type *type, Expression *e1, Expression *e2);
Expression *Cat(Type *type, Expression *e1, Expression *e2);

Expression *Equal(enum TOK op, Type *type, Expression *e1, Expression *e2);
Expression *Cmp(enum TOK op, Type *type, Expression *e1, Expression *e2);
Expression *Identity(enum TOK op, Type *type, Expression *e1, Expression *e2);

Expression *Slice(Type *type, Expression *e1, Expression *lwr, Expression *upr);

// Const-folding functions used by CTFE

void sliceAssignArrayLiteralFromString(ArrayLiteralExp *existingAE, StringExp *newval, int firstIndex);
void sliceAssignStringFromArrayLiteral(StringExp *existingSE, ArrayLiteralExp *newae, int firstIndex);
void sliceAssignStringFromString(StringExp *existingSE, StringExp *newstr, int firstIndex);

int sliceCmpStringWithString(StringExp *se1, StringExp *se2, size_t lo1, size_t lo2, size_t len);
int sliceCmpStringWithArray(StringExp *se1, ArrayLiteralExp *ae2, size_t lo1, size_t lo2, size_t len);


#endif /* DMD_EXPRESSION_H */
