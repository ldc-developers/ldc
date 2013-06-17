
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
#include "intrange.h"

struct Type;
struct TypeVector;
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
struct InterState;
#if IN_DMD
struct Symbol;          // back end symbol
#endif
struct OverloadSet;
struct Initializer;
struct StringExp;
#if IN_LLVM
struct AssignExp;
struct SymbolDeclaration;
#endif

enum TOK;

#if IN_DMD
// Back end
struct IRState;
struct dt_t;
#endif

#ifdef IN_GCC
union tree_node; typedef union tree_node elem;
#endif
#if IN_DMD
struct elem;
#endif

#if IN_LLVM
struct IRState;
class DValue;
namespace llvm {
    class Constant;
    class ConstantInt;
    class GlobalVariable;
    class StructType;
}
#endif

void initPrecedence();

typedef int (*apply_fp_t)(Expression *, void *);

Expression *resolveProperties(Scope *sc, Expression *e);
void accessCheck(Loc loc, Scope *sc, Expression *e, Declaration *d);
Expression *build_overload(Loc loc, Scope *sc, Expression *ethis, Expression *earg, Dsymbol *d);
Dsymbol *search_function(ScopeDsymbol *ad, Identifier *funcid);
void argExpTypesToCBuffer(OutBuffer *buf, Expressions *arguments, HdrGenState *hgs);
void argsToCBuffer(OutBuffer *buf, Expressions *arguments, HdrGenState *hgs);
void expandTuples(Expressions *exps);
TupleDeclaration *isAliasThisTuple(Expression *e);
int expandAliasThisTuples(Expressions *exps, size_t starti = 0);
FuncDeclaration *hasThis(Scope *sc);
Expression *fromConstInitializer(int result, Expression *e);
int arrayExpressionCanThrow(Expressions *exps, bool mustNotThrow);
TemplateDeclaration *getFuncTemplateDecl(Dsymbol *s);
void valueNoDtor(Expression *e);
int modifyFieldVar(Loc loc, Scope *sc, VarDeclaration *var, Expression *e1);
#if DMDV2
Expression *resolveAliasThis(Scope *sc, Expression *e);
Expression *callCpCtor(Loc loc, Scope *sc, Expression *e, int noscope);
bool checkPostblit(Loc loc, Type *t);
#endif
struct ArrayExp *resolveOpDollar(Scope *sc, struct ArrayExp *ae);
struct SliceExp *resolveOpDollar(Scope *sc, struct SliceExp *se);
Expressions *arrayExpressionSemantic(Expressions *exps, Scope *sc);

/* Interpreter: what form of return value expression is required?
 */
enum CtfeGoal
{   ctfeNeedRvalue,   // Must return an Rvalue
    ctfeNeedLvalue,   // Must return an Lvalue
    ctfeNeedAnyValue, // Can return either an Rvalue or an Lvalue
    ctfeNeedLvalueRef,// Must return a reference to an Lvalue (for ref types)
    ctfeNeedNothing   // The return value is not required
};

#define WANTflags   1
#define WANTvalue   2
// A compile-time result is required. Give an error if not possible
#define WANTinterpret 4
// Same as WANTvalue, but also expand variables as far as possible
#define WANTexpand  8

struct Expression : Object
{
    Loc loc;                    // file location
    enum TOK op;                // handy to minimize use of dynamic_cast
    Type *type;                 // !=NULL means that semantic() has been run
    unsigned char size;         // # of bytes in Expression so we can copy() it
    unsigned char parens;       // if this is a parenthesized expression

    Expression(Loc loc, enum TOK op, int size);
    static void init();
    Expression *copy();
    virtual Expression *syntaxCopy();
    virtual int apply(apply_fp_t fp, void *param);
    virtual Expression *semantic(Scope *sc);
    Expression *trySemantic(Scope *sc);
    Expression *ctfeSemantic(Scope *sc);

    int dyncast() { return DYNCAST_EXPRESSION; }        // kludge for template.isExpression()

    void print();
    char *toChars();
    virtual void dump(int indent);
    void error(const char *format, ...);
    void warning(const char *format, ...);
    void deprecation(const char *format, ...);
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
    virtual IntRange getIntRange();
    virtual Expression *castTo(Scope *sc, Type *t);
    virtual Expression *inferType(Type *t, int flag = 0, Scope *sc = NULL, TemplateParameters *tparams = NULL);
    virtual void checkEscape();
    virtual void checkEscapeRef();
    virtual Expression *resolveLoc(Loc loc, Scope *sc);
    void checkScalar();
    void checkNoBool();
    Expression *checkIntegral();
    Expression *checkArithmetic();
    void checkDeprecated(Scope *sc, Dsymbol *s);
    void checkPurity(Scope *sc, FuncDeclaration *f);
    void checkPurity(Scope *sc, VarDeclaration *v, Expression *e1);
    void checkSafety(Scope *sc, FuncDeclaration *f);
    virtual int checkModifiable(Scope *sc, int flag = 0);
    virtual Expression *checkToBoolean(Scope *sc);
    virtual Expression *addDtorHook(Scope *sc);
    Expression *checkToPointer();
    Expression *addressOf(Scope *sc);
    Expression *deref();
    Expression *integralPromotions(Scope *sc);
    Expression *isTemp();

    Expression *toDelegate(Scope *sc, Type *t);

    virtual Expression *optimize(int result, bool keepLvalue = false);

    // Entry point for CTFE.
    // A compile-time result is required. Give an error if not possible
    Expression *ctfeInterpret();

    // Implementation of CTFE for this expression
    virtual Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);

    virtual int isConst();
    virtual int isBool(int result);
    virtual int isBit();
    bool hasSideEffect();
    void discardValue();
    void useValue();
    int canThrow(bool mustNotThrow);

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
    elem *toElemDtor(IRState *irs);
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
    IntRange getIntRange();
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
    Expression *toLvalue(Scope *sc, Expression *e);
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

    DsymbolExp(Loc loc, Dsymbol *s, int hasOverloads = 0);
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
    Expression *modifiableLvalue(Scope *sc, Expression *e);

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
    int equals(Object *o);
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
    Expression *semantic(Scope *sc);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);
    size_t length();
    StringExp *toString();
    StringExp *toUTF8(Scope *sc);
    Expression *implicitCastTo(Scope *sc, Type *t);
    MATCH implicitConvTo(Type *t);
    Expression *castTo(Scope *sc, Type *t);
    int compare(Object *obj);
    int isBool(int result);
    int isLvalue();
    Expression *toLvalue(Scope *sc, Expression *e);
    Expression *modifiableLvalue(Scope *sc, Expression *e);
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
    Expression *e0;     // side-effect part
    /* Tuple-field access may need to take out its side effect part.
     * For example:
     *      foo().tupleof
     * is rewritten as:
     *      (ref __tup = foo(); tuple(__tup.field0, __tup.field1, ...))
     * The declaration of temporary variable __tup will be stored in TupleExp::e0.
     */
    Expressions *exps;

    TupleExp(Loc loc, Expression *e0, Expressions *exps);
    TupleExp(Loc loc, Expressions *exps);
    TupleExp(Loc loc, TupleDeclaration *tup);
    Expression *syntaxCopy();
    int apply(apply_fp_t fp, void *param);
    int equals(Object *o);
    Expression *semantic(Scope *sc);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    void checkEscape();
    Expression *optimize(int result, bool keepLvalue = false);
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
#if IN_LLVM
    DValue* toElem(IRState* irs);
#else
    elem *toElem(IRState *irs);
#endif
    StringExp *toString();
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    void toMangleBuffer(OutBuffer *buf);
    Expression *optimize(int result, bool keepLvalue = false);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);
    MATCH implicitConvTo(Type *t);
    Expression *castTo(Scope *sc, Type *t);
    Expression *inferType(Type *t, int flag = 0, Scope *sc = NULL, TemplateParameters *tparams = NULL);
#if IN_LLVM
    llvm::Constant *toConstElem(IRState *irs);
#else
    dt_t **toDt(dt_t **pdt);
#endif

    Expression *doInline(InlineDoState *ids);
    Expression *inlineScan(InlineScanState *iss);
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
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    void toMangleBuffer(OutBuffer *buf);
    Expression *optimize(int result, bool keepLvalue = false);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);
    MATCH implicitConvTo(Type *t);
    Expression *castTo(Scope *sc, Type *t);
    Expression *inferType(Type *t, int flag = 0, Scope *sc = NULL, TemplateParameters *tparams = NULL);

    Expression *doInline(InlineDoState *ids);
    Expression *inlineScan(InlineScanState *iss);

#if IN_LLVM
    DValue* toElem(IRState* irs);
#endif
};

// scrubReturnValue is running
#define stageScrub          0x1
// hasNonConstPointers is running
#define stageSearchPointers 0x2
// optimize is running
#define stageOptimize       0x4
// apply is running
#define stageApply          0x8
//inlineScan is running
#define stageInlineScan     0x10

struct StructLiteralExp : Expression
{
    StructDeclaration *sd;      // which aggregate this is for
    Expressions *elements;      // parallels sd->fields[] with
                                // NULL entries for fields to skip
    Type *stype;                // final type of result (can be different from sd's type)

#if IN_DMD
    Symbol *sinit;              // if this is a defaultInitLiteral, this symbol contains the default initializer
    Symbol *sym;                // back end symbol to initialize with literal
#endif
    size_t soffset;             // offset from start of s
    int fillHoles;              // fill alignment 'holes' with zero
    bool ownedByCtfe;           // true = created in CTFE
    int ctorinit;

    StructLiteralExp *origin;   // pointer to the origin instance of the expression.
                                // once a new expression is created, origin is set to 'this'.
                                // anytime when an expression copy is created, 'origin' pointer is set to
                                // 'origin' pointer value of the original expression.

    StructLiteralExp *inlinecopy; // those fields need to prevent a infinite recursion when one field of struct initialized with 'this' pointer.
    int stageflags;               // anytime when recursive function is calling, 'stageflags' marks with bit flag of
                                  // current stage and unmarks before return from this function.
                                  // 'inlinecopy' uses similar 'stageflags' and from multiple evaluation 'doInline'
                                  // (with infinite recursion) of this expression.

    StructLiteralExp(Loc loc, StructDeclaration *sd, Expressions *elements, Type *stype = NULL);
    int equals(Object *o);
    Expression *syntaxCopy();
    int apply(apply_fp_t fp, void *param);
    Expression *semantic(Scope *sc);
    Expression *getField(Type *type, unsigned offset);
    int getFieldIndex(Type *type, unsigned offset);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    void toMangleBuffer(OutBuffer *buf);
    Expression *optimize(int result, bool keepLvalue = false);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);
#if IN_LLVM
    DValue* toElem(IRState* irs);
    // With the introduction of pointers returned from CTFE, struct literals can
    // now contain pointers to themselves. While in toElem, contains a pointer
    // to the memory used to build the literal for resolving such references.
    llvm::Value* inProgressMemory;

    llvm::Constant *toConstElem(IRState *irs);
    // A global variable for taking the address of this struct literal constant,
    // if it already exists. Used to resolve self-references.
    llvm::GlobalVariable *globalVar;

    /// Set if this is really the result of a struct .init access and should be
    /// resolved codegen'd as an access to the given SymbolDeclaration.
    // LDC_FIXME: Figure out whether this, i.e. imitating the DMD behavior, is
    // really the best way to fix the nested struct constant folding issue.
    SymbolDeclaration *sinit;
#else
    dt_t **toDt(dt_t **pdt);
    Symbol *toSymbol();
#endif
    MATCH implicitConvTo(Type *t);
    Expression *castTo(Scope *sc, Type *t);

    int inlineCost3(InlineCostState *ics);
    Expression *doInline(InlineDoState *ids);
    Expression *inlineScan(InlineScanState *iss);
};

struct DotIdExp;
DotIdExp *typeDotIdExp(Loc loc, Type *type, Identifier *ident);

struct TypeExp : Expression
{
    TypeExp(Loc loc, Type *type);
    Expression *syntaxCopy();
    Expression *semantic(Scope *sc);
    int rvalue();
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    Expression *optimize(int result, bool keepLvalue = false);
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
    FuncDeclaration *fd;

    TemplateExp(Loc loc, TemplateDeclaration *td, FuncDeclaration *fd = NULL);
    int rvalue();
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    int isLvalue();
    Expression *toLvalue(Scope *sc, Expression *e);
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
    Expression *optimize(int result, bool keepLvalue = false);
    MATCH implicitConvTo(Type *t);
#if IN_DMD
    elem *toElem(IRState *irs);
#endif
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
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
};

#if DMDV2
struct SymbolExp : Expression
{
    Declaration *var;
    int hasOverloads;

    SymbolExp(Loc loc, enum TOK op, int size, Declaration *var, int hasOverloads);

#if IN_DMD
    elem *toElem(IRState *irs);
#endif

#if IN_LLVM
    DValue* toElem(IRState* irs);
#endif
};
#endif

// Offset from symbol

struct SymOffExp : SymbolExp
{
    unsigned offset;

    SymOffExp(Loc loc, Declaration *var, unsigned offset, int hasOverloads = 0);
    Expression *semantic(Scope *sc);
    Expression *optimize(int result, bool keepLvalue = false);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);
    void checkEscape();
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    int isConst();
    int isBool(int result);
    Expression *doInline(InlineDoState *ids);
    MATCH implicitConvTo(Type *t);
    Expression *castTo(Scope *sc, Type *t);

#if IN_LLVM
    DValue* toElem(IRState* irs);
    llvm::Constant* toConstElem(IRState* irs);
#else
    dt_t **toDt(dt_t **pdt);
#endif
};

// Variable

struct VarExp : SymbolExp
{
    VarExp(Loc loc, Declaration *var, int hasOverloads = 0);
    int equals(Object *o);
    Expression *semantic(Scope *sc);
    Expression *optimize(int result, bool keepLvalue = false);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);
    void dump(int indent);
    char *toChars();
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    void checkEscape();
    void checkEscapeRef();
    int checkModifiable(Scope *sc, int flag);
    int isLvalue();
    Expression *toLvalue(Scope *sc, Expression *e);
    Expression *modifiableLvalue(Scope *sc, Expression *e);
#if IN_DMD
    dt_t **toDt(dt_t **pdt);
#endif

    int inlineCost3(InlineCostState *ics);
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

    OverExp(Loc loc, OverloadSet *s);
    int isLvalue();
    Expression *toLvalue(Scope *sc, Expression *e);
};
#endif

// Function/Delegate literal

struct FuncExp : Expression
{
    FuncLiteralDeclaration *fd;
    TemplateDeclaration *td;
    enum TOK tok;

    FuncExp(Loc loc, FuncLiteralDeclaration *fd, TemplateDeclaration *td = NULL);
    Expression *syntaxCopy();
    Expression *semantic(Scope *sc);
    Expression *semantic(Scope *sc, Expressions *arguments);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);
    Expression *implicitCastTo(Scope *sc, Type *t);
    MATCH implicitConvTo(Type *t);
    Expression *castTo(Scope *sc, Type *t);
    Expression *inferType(Type *t, int flag = 0, Scope *sc = NULL, TemplateParameters *tparams = NULL);
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
    Object *obj;

    TypeidExp(Loc loc, Object *obj);
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
    TemplateParameters *parameters;

    IsExp(Loc loc, Type *targ, Identifier *id, enum TOK tok, Type *tspec,
        enum TOK tok2, TemplateParameters *parameters);
    Expression *syntaxCopy();
    Expression *semantic(Scope *sc);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
};

/****************************************************************/

struct UnaExp : Expression
{
    Expression *e1;
    Type *att1; // Save alias this type to detect recursion

    UnaExp(Loc loc, enum TOK op, int size, Expression *e1);
    Expression *syntaxCopy();
    int apply(apply_fp_t fp, void *param);
    Expression *semantic(Scope *sc);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    Expression *optimize(int result, bool keepLvalue = false);
    void dump(int indent);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);
    Expression *resolveLoc(Loc loc, Scope *sc);

    Expression *doInline(InlineDoState *ids);
    Expression *inlineScan(InlineScanState *iss);

    virtual Expression *op_overload(Scope *sc);
};

struct BinExp : Expression
{
    Expression *e1;
    Expression *e2;

    Type *att1; // Save alias this type to detect recursion
    Type *att2; // Save alias this type to detect recursion

    BinExp(Loc loc, enum TOK op, int size, Expression *e1, Expression *e2);
    Expression *syntaxCopy();
    int apply(apply_fp_t fp, void *param);
    Expression *semantic(Scope *sc);
    Expression *semanticp(Scope *sc);
    Expression *checkComplexOpAssign(Scope *sc);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    Expression *scaleFactor(Scope *sc);
    Expression *typeCombine(Scope *sc);
    Expression *optimize(int result, bool keepLvalue = false);
    int isunsigned();
    Expression *incompatibleTypes();
    void dump(int indent);

    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);
    Expression *interpretCommon(InterState *istate, CtfeGoal goal,
        Expression *(*fp)(Type *, Expression *, Expression *));
    Expression *interpretCompareCommon(InterState *istate, CtfeGoal goal,
        int (*fp)(Loc, TOK, Expression *, Expression *));
    Expression *interpretAssignCommon(InterState *istate, CtfeGoal goal,
        Expression *(*fp)(Type *, Expression *, Expression *), int post = 0);
    Expression *interpretFourPointerRelation(InterState *istate, CtfeGoal goal);
    virtual Expression *arrayOp(Scope *sc);

    Expression *doInline(InlineDoState *ids);
    Expression *inlineScan(InlineScanState *iss);

    Expression *op_overload(Scope *sc);
    Expression *compare_overload(Scope *sc, Identifier *id);
    Expression *reorderSettingAAElem(Scope *sc);

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
    Expression *arrayOp(Scope *sc);

    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);

    Expression *op_overload(Scope *sc);

    int isLvalue();
    Expression *toLvalue(Scope *sc, Expression *ex);
    Expression *modifiableLvalue(Scope *sc, Expression *e);
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
    Expression *semanticX(Scope *sc);
    Expression *semanticY(Scope *sc, int flag);
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
    int hasOverloads;

    DotVarExp(Loc loc, Expression *e, Declaration *var, int hasOverloads = 0);
    Expression *semantic(Scope *sc);
    int checkModifiable(Scope *sc, int flag);
    int isLvalue();
    Expression *toLvalue(Scope *sc, Expression *e);
    Expression *modifiableLvalue(Scope *sc, Expression *e);
    Expression *optimize(int result, bool keepLvalue = false);
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

    DotTemplateInstanceExp(Loc loc, Expression *e, Identifier *name, Objects *tiargs);
    Expression *syntaxCopy();
    TemplateDeclaration *getTempdecl(Scope *sc);
    Expression *semantic(Scope *sc);
    Expression *semanticY(Scope *sc, int flag);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    void dump(int indent);
};

struct DelegateExp : UnaExp
{
    FuncDeclaration *func;
    int hasOverloads;

    DelegateExp(Loc loc, Expression *e, FuncDeclaration *func, int hasOverloads = 0);
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
    FuncDeclaration *f;         // symbol to call

    CallExp(Loc loc, Expression *e, Expressions *exps);
    CallExp(Loc loc, Expression *e);
    CallExp(Loc loc, Expression *e, Expression *earg1);
    CallExp(Loc loc, Expression *e, Expression *earg1, Expression *earg2);

    Expression *syntaxCopy();
    int apply(apply_fp_t fp, void *param);
    Expression *semantic(Scope *sc);
    Expression *optimize(int result, bool keepLvalue = false);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    void dump(int indent);
#if IN_DMD
    elem *toElem(IRState *irs);
#endif
    int isLvalue();
    Expression *toLvalue(Scope *sc, Expression *e);
    Expression *addDtorHook(Scope *sc);
    MATCH implicitConvTo(Type *t);

    int inlineCost3(InlineCostState *ics);
    Expression *doInline(InlineDoState *ids);
    Expression *inlineScan(InlineScanState *iss);

#if IN_LLVM
    DValue* toElem(IRState* irs);
    void cacheLvalue(IRState* p);
#endif
};

struct AddrExp : UnaExp
{
    AddrExp(Loc loc, Expression *e);
    Expression *semantic(Scope *sc);
    void checkEscape();
#if IN_DMD
    elem *toElem(IRState *irs);
#endif
    MATCH implicitConvTo(Type *t);
    Expression *castTo(Scope *sc, Type *t);
    Expression *optimize(int result, bool keepLvalue = false);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);
#if IN_LLVM
    DValue* toElem(IRState* irs);
    llvm::Constant *toConstElem(IRState *irs);
#else
    dt_t **toDt(dt_t **pdt);
#endif
};

struct PtrExp : UnaExp
{
    PtrExp(Loc loc, Expression *e);
    PtrExp(Loc loc, Expression *e, Type *t);
    Expression *semantic(Scope *sc);
    void checkEscapeRef();
    int checkModifiable(Scope *sc, int flag);
    int isLvalue();
    Expression *toLvalue(Scope *sc, Expression *e);
    Expression *modifiableLvalue(Scope *sc, Expression *e);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
#if IN_DMD
    elem *toElem(IRState *irs);
#endif
    Expression *optimize(int result, bool keepLvalue = false);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);

    // For operator overloading
    Identifier *opId();

#if IN_LLVM
    DValue* toElem(IRState* irs);
    void cacheLvalue(IRState* irs);
#endif
};

struct NegExp : UnaExp
{
    NegExp(Loc loc, Expression *e);
    Expression *semantic(Scope *sc);
    Expression *optimize(int result, bool keepLvalue = false);
    void buildArrayIdent(OutBuffer *buf, Expressions *arguments);
    Expression *buildArrayLoop(Parameters *fparams);
    IntRange getIntRange();

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
    Expression *optimize(int result, bool keepLvalue = false);
    void buildArrayIdent(OutBuffer *buf, Expressions *arguments);
    Expression *buildArrayLoop(Parameters *fparams);
    IntRange getIntRange();

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
    Expression *optimize(int result, bool keepLvalue = false);
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
    Expression *optimize(int result, bool keepLvalue = false);
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
    Expression *checkToBoolean(Scope *sc);
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
    unsigned mod;               // MODxxxxx

    CastExp(Loc loc, Expression *e, Type *t);
    CastExp(Loc loc, Expression *e, unsigned mod);
    Expression *syntaxCopy();
    Expression *semantic(Scope *sc);
    MATCH implicitConvTo(Type *t);
    IntRange getIntRange();
    Expression *optimize(int result, bool keepLvalue = false);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);
    void checkEscape();
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    void buildArrayIdent(OutBuffer *buf, Expressions *arguments);
    Expression *buildArrayLoop(Parameters *fparams);
#if IN_DMD
    elem *toElem(IRState *irs);
#endif

    // For operator overloading
    Identifier *opId();
    Expression *op_overload(Scope *sc);

#if IN_LLVM
    DValue* toElem(IRState* irs);
    llvm::Constant *toConstElem(IRState *irs);
#else
    dt_t **toDt(dt_t **pdt);
#endif
};

struct VectorExp : UnaExp
{
    TypeVector *to;             // the target vector type before semantic()
    unsigned dim;               // number of elements in the vector

    VectorExp(Loc loc, Expression *e, Type *t);
    Expression *syntaxCopy();
    Expression *semantic(Scope *sc);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
#if IN_DMD
    elem *toElem(IRState *irs);
    dt_t **toDt(dt_t **pdt);
#endif
#if IN_LLVM
    DValue* toElem(IRState* irs);
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
    int checkModifiable(Scope *sc, int flag);
    int isLvalue();
    Expression *toLvalue(Scope *sc, Expression *e);
    Expression *modifiableLvalue(Scope *sc, Expression *e);
    int isBool(int result);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    Type *toStaticArrayType();
    MATCH implicitConvTo(Type *t);
    Expression *castTo(Scope *sc, Type *t);
    Expression *optimize(int result, bool keepLvalue = false);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);
    void dump(int indent);
#if IN_DMD
    elem *toElem(IRState *irs);
#endif
    void buildArrayIdent(OutBuffer *buf, Expressions *arguments);
    Expression *buildArrayLoop(Parameters *fparams);

    Expression *doInline(InlineDoState *ids);
    Expression *inlineScan(InlineScanState *iss);

#if IN_LLVM
    DValue* toElem(IRState* irs);
#endif
};

struct ArrayLengthExp : UnaExp
{
    ArrayLengthExp(Loc loc, Expression *e1);
    Expression *semantic(Scope *sc);
    Expression *optimize(int result, bool keepLvalue = false);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
#if IN_DMD
    elem *toElem(IRState *irs);
#endif

    static Expression *rewriteOpAssign(BinExp *exp);

#if IN_LLVM
    DValue* toElem(IRState* irs);
#endif
};

// e1[a0,a1,a2,a3,...]

struct ArrayExp : UnaExp
{
    Expressions *arguments;             // Array of Expression's
    size_t currentDimension;            // for opDollar
    VarDeclaration *lengthVar;

    ArrayExp(Loc loc, Expression *e1, Expressions *arguments);
    Expression *syntaxCopy();
    int apply(apply_fp_t fp, void *param);
    Expression *semantic(Scope *sc);
    int isLvalue();
    Expression *toLvalue(Scope *sc, Expression *e);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);

    // For operator overloading
    Identifier *opId();
    Expression *op_overload(Scope *sc);

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
    int checkModifiable(Scope *sc, int flag);
    IntRange getIntRange();
    int isLvalue();
    Expression *toLvalue(Scope *sc, Expression *e);
    Expression *modifiableLvalue(Scope *sc, Expression *e);
    int isBool(int result);
    MATCH implicitConvTo(Type *t);
    Expression *addDtorHook(Scope *sc);
    Expression *castTo(Scope *sc, Type *t);
    Expression *optimize(int result, bool keepLvalue = false);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);
#if IN_DMD
    elem *toElem(IRState *irs);
#endif

#if IN_LLVM
    DValue* toElem(IRState* irs);
    void cacheLvalue(IRState* irs);
#endif
};

struct IndexExp : BinExp
{
    VarDeclaration *lengthVar;
    int modifiable;

    IndexExp(Loc loc, Expression *e1, Expression *e2);
    Expression *syntaxCopy();
    Expression *semantic(Scope *sc);
    int checkModifiable(Scope *sc, int flag);
    int isLvalue();
    Expression *toLvalue(Scope *sc, Expression *e);
    Expression *modifiableLvalue(Scope *sc, Expression *e);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    Expression *optimize(int result, bool keepLvalue = false);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);
    Expression *doInline(InlineDoState *ids);

#if IN_DMD
    elem *toElem(IRState *irs);
#elif IN_LLVM
    DValue* toElem(IRState* irs);
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

/* For both ++i and --i
 */
struct PreExp : UnaExp
{
    PreExp(enum TOK op, Loc loc, Expression *e);
    Expression *semantic(Scope *sc);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
};

struct AssignExp : BinExp
{   int ismemset;       // !=0 if setting the contents of an array

    AssignExp(Loc loc, Expression *e1, Expression *e2);
    Expression *semantic(Scope *sc);
    Expression *checkToBoolean(Scope *sc);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);
    Identifier *opId();    // For operator overloading
    void buildArrayIdent(OutBuffer *buf, Expressions *arguments);
    Expression *buildArrayLoop(Parameters *fparams);
#if IN_DMD
    elem *toElem(IRState *irs);
#endif

#if IN_LLVM
    DValue* toElem(IRState* irs);
    virtual AssignExp* isAssignExp() { return this; }
#endif
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
    X(void buildArrayIdent(OutBuffer *buf, Expressions *arguments);) \
    X(Expression *buildArrayLoop(Parameters *fparams);)         \
                                                                \
    Identifier *opId();    /* For operator overloading */       \
                                                                \
    ASSIGNEXP_TOELEM                                            \
};

#define X(a) a
#define S(a)
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

#undef S
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
    Expression *optimize(int result, bool keepLvalue = false);
    void buildArrayIdent(OutBuffer *buf, Expressions *arguments);
    Expression *buildArrayLoop(Parameters *fparams);
    IntRange getIntRange();

    // For operator overloading
    int isCommutative();
    Identifier *opId();
    Identifier *opId_r();

#if IN_DMD
    elem *toElem(IRState *irs);
#endif

#if IN_LLVM
    llvm::Constant *toConstElem(IRState* p);
    DValue* toElem(IRState* irs);
#endif
};

struct MinExp : BinExp
{
    MinExp(Loc loc, Expression *e1, Expression *e2);
    Expression *semantic(Scope *sc);
    Expression *optimize(int result, bool keepLvalue = false);
    void buildArrayIdent(OutBuffer *buf, Expressions *arguments);
    Expression *buildArrayLoop(Parameters *fparams);
    IntRange getIntRange();

    // For operator overloading
    Identifier *opId();
    Identifier *opId_r();

#if IN_DMD
    elem *toElem(IRState *irs);
#endif

#if IN_LLVM
    llvm::Constant *toConstElem(IRState* p);
    DValue* toElem(IRState* irs);
#endif
};

struct CatExp : BinExp
{
    CatExp(Loc loc, Expression *e1, Expression *e2);
    Expression *semantic(Scope *sc);
    Expression *optimize(int result, bool keepLvalue = false);
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
    Expression *optimize(int result, bool keepLvalue = false);
    void buildArrayIdent(OutBuffer *buf, Expressions *arguments);
    Expression *buildArrayLoop(Parameters *fparams);
    IntRange getIntRange();

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
    Expression *optimize(int result, bool keepLvalue = false);
    void buildArrayIdent(OutBuffer *buf, Expressions *arguments);
    Expression *buildArrayLoop(Parameters *fparams);
    IntRange getIntRange();

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
    Expression *optimize(int result, bool keepLvalue = false);
    void buildArrayIdent(OutBuffer *buf, Expressions *arguments);
    Expression *buildArrayLoop(Parameters *fparams);
    IntRange getIntRange();

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
    Expression *optimize(int result, bool keepLvalue = false);
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
#endif

struct ShlExp : BinExp
{
    ShlExp(Loc loc, Expression *e1, Expression *e2);
    Expression *semantic(Scope *sc);
    Expression *optimize(int result, bool keepLvalue = false);
    IntRange getIntRange();

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
    Expression *optimize(int result, bool keepLvalue = false);
    IntRange getIntRange();

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
    Expression *optimize(int result, bool keepLvalue = false);
    IntRange getIntRange();

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
    Expression *optimize(int result, bool keepLvalue = false);
    void buildArrayIdent(OutBuffer *buf, Expressions *arguments);
    Expression *buildArrayLoop(Parameters *fparams);
    IntRange getIntRange();

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
    Expression *optimize(int result, bool keepLvalue = false);
    void buildArrayIdent(OutBuffer *buf, Expressions *arguments);
    Expression *buildArrayLoop(Parameters *fparams);
    MATCH implicitConvTo(Type *t);
    IntRange getIntRange();

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
    Expression *optimize(int result, bool keepLvalue = false);
    void buildArrayIdent(OutBuffer *buf, Expressions *arguments);
    Expression *buildArrayLoop(Parameters *fparams);
    MATCH implicitConvTo(Type *t);
    IntRange getIntRange();

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
    Expression *checkToBoolean(Scope *sc);
    int isBit();
    Expression *optimize(int result, bool keepLvalue = false);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);
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
    Expression *checkToBoolean(Scope *sc);
    int isBit();
    Expression *optimize(int result, bool keepLvalue = false);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);
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
    Expression *optimize(int result, bool keepLvalue = false);
    int isBit();

    // For operator overloading
    int isCommutative();
    Identifier *opId();
    Expression *op_overload(Scope *sc);

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
    Expression *optimize(int result, bool keepLvalue = false);
    int isBit();

    // For operator overloading
    int isCommutative();
    Identifier *opId();
    Expression *op_overload(Scope *sc);

#if IN_DMD
    elem *toElem(IRState *irs);
#endif

#if IN_LLVM
    DValue* toElem(IRState* irs);
#endif
};

// is and !is

struct IdentityExp : BinExp
{
    IdentityExp(enum TOK op, Loc loc, Expression *e1, Expression *e2);
    Expression *semantic(Scope *sc);
    int isBit();
    Expression *optimize(int result, bool keepLvalue = false);
#if IN_LLVM
    DValue* toElem(IRState* irs);
#else
    elem *toElem(IRState *irs);
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
    Expression *optimize(int result, bool keepLvalue = false);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);
    void checkEscape();
    void checkEscapeRef();
    int checkModifiable(Scope *sc, int flag);
    int isLvalue();
    Expression *toLvalue(Scope *sc, Expression *e);
    Expression *modifiableLvalue(Scope *sc, Expression *e);
    Expression *checkToBoolean(Scope *sc);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    MATCH implicitConvTo(Type *t);
    Expression *castTo(Scope *sc, Type *t);
    Expression *inferType(Type *t, int flag = 0, Scope *sc = NULL, TemplateParameters *tparams = NULL);

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
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
};

struct FileInitExp : DefaultInitExp
{
    FileInitExp(Loc loc);
    Expression *semantic(Scope *sc);
    Expression *resolveLoc(Loc loc, Scope *sc);
};

struct LineInitExp : DefaultInitExp
{
    LineInitExp(Loc loc);
    Expression *semantic(Scope *sc);
    Expression *resolveLoc(Loc loc, Scope *sc);
};

struct ModuleInitExp : DefaultInitExp
{
    ModuleInitExp(Loc loc);
    Expression *semantic(Scope *sc);
    Expression *resolveLoc(Loc loc, Scope *sc);
};

struct FuncInitExp : DefaultInitExp
{
    FuncInitExp(Loc loc);
    Expression *semantic(Scope *sc);
    Expression *resolveLoc(Loc loc, Scope *sc);
};

struct PrettyFuncInitExp : DefaultInitExp
{
    PrettyFuncInitExp(Loc loc);
    Expression *semantic(Scope *sc);
    Expression *resolveLoc(Loc loc, Scope *sc);
};

#endif

/****************************************************************/

#if IN_LLVM

// this stuff is strictly LDC

struct GEPExp : UnaExp
{
    unsigned index;
    Identifier* ident;

    GEPExp(Loc loc, Expression* e, Identifier* id, unsigned idx);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    Expression *toLvalue(Scope *sc, Expression *e);

    DValue* toElem(IRState* irs);
};

#endif

/****************************************************************/

/* Special values used by the interpreter
 */
extern Expression *EXP_CANT_INTERPRET;
extern Expression *EXP_CONTINUE_INTERPRET;
extern Expression *EXP_BREAK_INTERPRET;
extern Expression *EXP_GOTO_INTERPRET;
extern Expression *EXP_VOID_INTERPRET;

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
Expression *Pow(Type *type, Expression *e1, Expression *e2);
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

void sliceAssignArrayLiteralFromString(ArrayLiteralExp *existingAE, StringExp *newval, size_t firstIndex);
void sliceAssignStringFromArrayLiteral(StringExp *existingSE, ArrayLiteralExp *newae, size_t firstIndex);
void sliceAssignStringFromString(StringExp *existingSE, StringExp *newstr, size_t firstIndex);

int sliceCmpStringWithString(StringExp *se1, StringExp *se2, size_t lo1, size_t lo2, size_t len);
int sliceCmpStringWithArray(StringExp *se1, ArrayLiteralExp *ae2, size_t lo1, size_t lo2, size_t len);


#endif /* DMD_EXPRESSION_H */
