
/* Compiler implementation of the D programming language
 * Copyright (c) 1999-2014 by Digital Mars
 * All Rights Reserved
 * written by Walter Bright
 * http://www.digitalmars.com
 * Distributed under the Boost Software License, Version 1.0.
 * http://www.boost.org/LICENSE_1_0.txt
 * https://github.com/D-Programming-Language/dmd/blob/master/src/expression.h
 */

#ifndef DMD_EXPRESSION_H
#define DMD_EXPRESSION_H

#include "mars.h"
#include "identifier.h"
#include "arraytypes.h"
#include "intrange.h"
#include "visitor.h"
#include "tokens.h"

class Type;
class TypeVector;
struct Scope;
class TupleDeclaration;
class VarDeclaration;
class FuncDeclaration;
class FuncLiteralDeclaration;
class Declaration;
class CtorDeclaration;
class NewDeclaration;
class Dsymbol;
class Import;
class Module;
class ScopeDsymbol;
class Expression;
class Declaration;
class AggregateDeclaration;
class StructDeclaration;
class TemplateInstance;
class TemplateDeclaration;
class ClassDeclaration;
class BinExp;
#if IN_DMD
struct Symbol;          // back end symbol
#endif
class OverloadSet;
class Initializer;
class StringExp;
class ArrayExp;
class SliceExp;
#if IN_LLVM
class AssignExp;
class SymbolDeclaration;
#endif

struct UnionExp;

#if IN_LLVM
struct IRState;
namespace llvm {
    class GlobalVariable;
    class Value;
}
#endif

void initPrecedence();

Expression *resolveProperties(Scope *sc, Expression *e);
Expression *resolvePropertiesOnly(Scope *sc, Expression *e1);
bool checkAccess(Loc loc, Scope *sc, Expression *e, Declaration *d);
Expression *build_overload(Loc loc, Scope *sc, Expression *ethis, Expression *earg, Dsymbol *d);
Dsymbol *search_function(ScopeDsymbol *ad, Identifier *funcid);
void expandTuples(Expressions *exps);
TupleDeclaration *isAliasThisTuple(Expression *e);
int expandAliasThisTuples(Expressions *exps, size_t starti = 0);
FuncDeclaration *hasThis(Scope *sc);
Expression *fromConstInitializer(int result, Expression *e);
bool arrayExpressionSemantic(Expressions *exps, Scope *sc);
TemplateDeclaration *getFuncTemplateDecl(Dsymbol *s);
Expression *valueNoDtor(Expression *e);
int modifyFieldVar(Loc loc, Scope *sc, VarDeclaration *var, Expression *e1);
Expression *resolveAliasThis(Scope *sc, Expression *e, bool gag = false);
Expression *callCpCtor(Scope *sc, Expression *e);
Expression *resolveOpDollar(Scope *sc, ArrayExp *ae, Expression **pe0);
Expression *resolveOpDollar(Scope *sc, ArrayExp *ae, IntervalExp *ie, Expression **pe0);
Expression *integralPromotions(Expression *e, Scope *sc);
void discardValue(Expression *e);
bool isTrivialExp(Expression *e);

int isConst(Expression *e);
Expression *toDelegate(Expression *e, Type* t, Scope *sc);
AggregateDeclaration *isAggregate(Type *t);
IntRange getIntRange(Expression *e);
bool checkNonAssignmentArrayOp(Expression *e, bool suggestion = false);
bool isUnaArrayOp(TOK op);
bool isBinArrayOp(TOK op);
bool isBinAssignArrayOp(TOK op);
bool isArrayOpOperand(Expression *e);
Expression *arrayOp(BinExp *e, Scope *sc);
Expression *arrayOp(BinAssignExp *e, Scope *sc);
bool hasSideEffect(Expression *e);
bool canThrow(Expression *e, FuncDeclaration *func, bool mustNotThrow);
Expression *Expression_optimize(Expression *e, int result, bool keepLvalue);
MATCH implicitConvTo(Expression *e, Type *t);
Expression *implicitCastTo(Expression *e, Scope *sc, Type *t);
Expression *castTo(Expression *e, Scope *sc, Type *t);
Expression *ctfeInterpret(Expression *);
Expression *inlineCopy(Expression *e, Scope *sc);
Expression *op_overload(Expression *e, Scope *sc);
Type *toStaticArrayType(SliceExp *e);
Expression *scaleFactor(BinExp *be, Scope *sc);
Expression *typeCombine(BinExp *be, Scope *sc);
Expression *inferType(Expression *e, Type *t, int flag = 0);
Expression *semanticTraits(TraitsExp *e, Scope *sc);
Type *getIndirection(Type *t);

Expression *checkGC(Scope *sc, Expression *e);

bool checkEscape(Scope *sc, Expression *e, bool gag);
bool checkEscapeRef(Scope *sc, Expression *e, bool gag);

/* Run CTFE on the expression, but allow the expression to be a TypeExp
 * or a tuple containing a TypeExp. (This is required by pragma(msg)).
 */
Expression *ctfeInterpretForPragmaMsg(Expression *e);

enum OwnedBy
{
    OWNEDcode,      // normal code expression in AST
    OWNEDctfe,      // value expression for CTFE
    OWNEDcache,     // constant value cached for CTFE
};

#define WANTvalue   0   // default
#define WANTexpand  1   // expand const/immutable variables if possible

class Expression : public RootObject
{
public:
    Loc loc;                    // file location
    Type *type;                 // !=NULL means that semantic() has been run
    TOK op;                     // to minimize use of dynamic_cast
    unsigned char size;         // # of bytes in Expression so we can copy() it
    unsigned char parens;       // if this is a parenthesized expression

    Expression(Loc loc, TOK op, int size);
    static void init();
    Expression *copy();
    virtual Expression *syntaxCopy();
    virtual Expression *semantic(Scope *sc);
    Expression *trySemantic(Scope *sc);

    // kludge for template.isExpression()
    int dyncast() { return DYNCAST_EXPRESSION; }

    void print();
    char *toChars();
    void error(const char *format, ...);
    void warning(const char *format, ...);
    void deprecation(const char *format, ...);

    // creates a single expression which is effectively (e1, e2)
    // this new expression does not necessarily need to have valid D source code representation,
    // for example, it may include declaration expressions
    static Expression *combine(Expression *e1, Expression *e2);
    static Expression *extractLast(Expression *e, Expression **pe0);
    static Expressions *arraySyntaxCopy(Expressions *exps);

    virtual dinteger_t toInteger();
    virtual uinteger_t toUInteger();
    virtual real_t toReal();
    virtual real_t toImaginary();
    virtual complex_t toComplex();
    virtual StringExp *toStringExp();
    virtual bool isLvalue();
    virtual Expression *toLvalue(Scope *sc, Expression *e);
    virtual Expression *modifiableLvalue(Scope *sc, Expression *e);
    Expression *implicitCastTo(Scope *sc, Type *t)
    {
        return ::implicitCastTo(this, sc, t);
    }
    MATCH implicitConvTo(Type *t)
    {
        return ::implicitConvTo(this, t);
    }
    Expression *castTo(Scope *sc, Type *t)
    {
        return ::castTo(this, sc, t);
    }
    virtual Expression *resolveLoc(Loc loc, Scope *sc);
    virtual bool checkValue();
    bool checkScalar();
    bool checkNoBool();
    bool checkIntegral();
    bool checkArithmetic();
    void checkDeprecated(Scope *sc, Dsymbol *s);
    bool checkPurity(Scope *sc, FuncDeclaration *f);
    bool checkPurity(Scope *sc, VarDeclaration *v);
    bool checkSafety(Scope *sc, FuncDeclaration *f);
    bool checkNogc(Scope *sc, FuncDeclaration *f);
    bool checkPostblit(Scope *sc, Type *t);
    bool checkRightThis(Scope *sc);
    bool checkReadModifyWrite(TOK rmwOp, Expression *ex = NULL);
    virtual int checkModifiable(Scope *sc, int flag = 0);
    virtual Expression *toBoolean(Scope *sc);
    virtual Expression *addDtorHook(Scope *sc);
    Expression *addressOf();
    Expression *deref();

    Expression *optimize(int result, bool keepLvalue = false)
    {
        return Expression_optimize(this, result, keepLvalue);
    }

    // Entry point for CTFE.
    // A compile-time result is required. Give an error if not possible
    Expression *ctfeInterpret()
    {
        return ::ctfeInterpret(this);
    }

    int isConst() { return ::isConst(this); }
    virtual bool isBool(bool result);
    Expression *op_overload(Scope *sc)
    {
        return ::op_overload(this, sc);
    }

    virtual void accept(Visitor *v) { v->visit(this); }
#if IN_LLVM
    llvm::Value* cachedLvalue;
#endif
};

class IntegerExp : public Expression
{
public:
    dinteger_t value;

    IntegerExp(Loc loc, dinteger_t value, Type *type);
    IntegerExp(dinteger_t value);
    bool equals(RootObject *o);
    Expression *semantic(Scope *sc);
    dinteger_t toInteger();
    real_t toReal();
    real_t toImaginary();
    complex_t toComplex();
    bool isBool(bool result);
    Expression *toLvalue(Scope *sc, Expression *e);
    void accept(Visitor *v) { v->visit(this); }
    dinteger_t getInteger() { return value; }
    void setInteger(dinteger_t value);

private:
    void normalize();
};

class ErrorExp : public Expression
{
public:
    ErrorExp();

    Expression *toLvalue(Scope *sc, Expression *e);
    void accept(Visitor *v) { v->visit(this); }

    static ErrorExp *errorexp; // handy shared value
};

class RealExp : public Expression
{
public:
    real_t value;

    RealExp(Loc loc, real_t value, Type *type);
    bool equals(RootObject *o);
    Expression *semantic(Scope *sc);
    dinteger_t toInteger();
    uinteger_t toUInteger();
    real_t toReal();
    real_t toImaginary();
    complex_t toComplex();
    bool isBool(bool result);
    void accept(Visitor *v) { v->visit(this); }
};

class ComplexExp : public Expression
{
public:
    complex_t value;

    ComplexExp(Loc loc, complex_t value, Type *type);
    bool equals(RootObject *o);
    Expression *semantic(Scope *sc);
    dinteger_t toInteger();
    uinteger_t toUInteger();
    real_t toReal();
    real_t toImaginary();
    complex_t toComplex();
    bool isBool(bool result);
    void accept(Visitor *v) { v->visit(this); }
};

class IdentifierExp : public Expression
{
public:
    Identifier *ident;
    Declaration *var;

    IdentifierExp(Loc loc, Identifier *ident);
    static IdentifierExp *create(Loc loc, Identifier *ident);
    Expression *semantic(Scope *sc);
    bool isLvalue();
    Expression *toLvalue(Scope *sc, Expression *e);
    void accept(Visitor *v) { v->visit(this); }
};

class DollarExp : public IdentifierExp
{
public:
    DollarExp(Loc loc);
    void accept(Visitor *v) { v->visit(this); }
};

class DsymbolExp : public Expression
{
public:
    Dsymbol *s;
    bool hasOverloads;

    DsymbolExp(Loc loc, Dsymbol *s, bool hasOverloads = false);
    Expression *semantic(Scope *sc);
    bool isLvalue();
    Expression *toLvalue(Scope *sc, Expression *e);
    void accept(Visitor *v) { v->visit(this); }
};

class ThisExp : public Expression
{
public:
    Declaration *var;

    ThisExp(Loc loc);
    Expression *semantic(Scope *sc);
    bool isBool(bool result);
    bool isLvalue();
    Expression *toLvalue(Scope *sc, Expression *e);

    void accept(Visitor *v) { v->visit(this); }
};

class SuperExp : public ThisExp
{
public:
    SuperExp(Loc loc);
    Expression *semantic(Scope *sc);

    void accept(Visitor *v) { v->visit(this); }
};

class NullExp : public Expression
{
public:
    unsigned char committed;    // !=0 if type is committed

    NullExp(Loc loc, Type *t = NULL);
    bool equals(RootObject *o);
    Expression *semantic(Scope *sc);
    bool isBool(bool result);
    StringExp *toStringExp();
    void accept(Visitor *v) { v->visit(this); }
};

class StringExp : public Expression
{
public:
    void *string;       // char, wchar, or dchar data
    size_t len;         // number of chars, wchars, or dchars
    unsigned char sz;   // 1: char, 2: wchar, 4: dchar
    unsigned char committed;    // !=0 if type is committed
    utf8_t postfix;      // 'c', 'w', 'd'
    OwnedBy ownedByCtfe;

    StringExp(Loc loc, char *s);
    StringExp(Loc loc, void *s, size_t len);
    StringExp(Loc loc, void *s, size_t len, utf8_t postfix);
    static StringExp *create(Loc loc, char *s);
    bool equals(RootObject *o);
    Expression *semantic(Scope *sc);
    size_t length(int encSize = 4);
    StringExp *toStringExp();
    StringExp *toUTF8(Scope *sc);
    int compare(RootObject *obj);
    bool isBool(bool result);
    bool isLvalue();
    Expression *toLvalue(Scope *sc, Expression *e);
    Expression *modifiableLvalue(Scope *sc, Expression *e);
    unsigned charAt(uinteger_t i);
    void accept(Visitor *v) { v->visit(this); }
};

// Tuple

class TupleExp : public Expression
{
public:
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
    bool equals(RootObject *o);
    Expression *semantic(Scope *sc);

    void accept(Visitor *v) { v->visit(this); }
};

class ArrayLiteralExp : public Expression
{
public:
    Expressions *elements;
    OwnedBy ownedByCtfe;

    ArrayLiteralExp(Loc loc, Expressions *elements);
    ArrayLiteralExp(Loc loc, Expression *e);

    Expression *syntaxCopy();
    bool equals(RootObject *o);
    Expression *semantic(Scope *sc);
    bool isBool(bool result);
    StringExp *toStringExp();

    void accept(Visitor *v) { v->visit(this); }
};

class AssocArrayLiteralExp : public Expression
{
public:
    Expressions *keys;
    Expressions *values;
    OwnedBy ownedByCtfe;

    AssocArrayLiteralExp(Loc loc, Expressions *keys, Expressions *values);
    bool equals(RootObject *o);
    Expression *syntaxCopy();
    Expression *semantic(Scope *sc);
    bool isBool(bool result);

    void accept(Visitor *v) { v->visit(this); }
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
// toCBuffer is running
#define stageToCBuffer      0x20

class StructLiteralExp : public Expression
{
public:
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
    OwnedBy ownedByCtfe;

    // pointer to the origin instance of the expression.
    // once a new expression is created, origin is set to 'this'.
    // anytime when an expression copy is created, 'origin' pointer is set to
    // 'origin' pointer value of the original expression.
    StructLiteralExp *origin;

    // those fields need to prevent a infinite recursion when one field of struct initialized with 'this' pointer.
    StructLiteralExp *inlinecopy;

    // anytime when recursive function is calling, 'stageflags' marks with bit flag of
    // current stage and unmarks before return from this function.
    // 'inlinecopy' uses similar 'stageflags' and from multiple evaluation 'doInline'
    // (with infinite recursion) of this expression.
    int stageflags;

    StructLiteralExp(Loc loc, StructDeclaration *sd, Expressions *elements, Type *stype = NULL);
    static StructLiteralExp *create(Loc loc, StructDeclaration *sd, void *elements, Type *stype = NULL);
    bool equals(RootObject *o);
    Expression *syntaxCopy();
    Expression *semantic(Scope *sc);
    Expression *getField(Type *type, unsigned offset);
    int getFieldIndex(Type *type, unsigned offset);
    Expression *addDtorHook(Scope *sc);
#if IN_LLVM
    // With the introduction of pointers returned from CTFE, struct literals can
    // now contain pointers to themselves. While in toElem, contains a pointer
    // to the memory used to build the literal for resolving such references.
    llvm::Value* inProgressMemory;

    // A global variable for taking the address of this struct literal constant,
    // if it already exists. Used to resolve self-references.
    llvm::GlobalVariable *globalVar;

    /// Set if this is really the result of a struct .init access and should be
    /// resolved codegen'd as an access to the given SymbolDeclaration.
    // LDC_FIXME: Figure out whether this, i.e. imitating the DMD behavior, is
    // really the best way to fix the nested struct constant folding issue.
    SymbolDeclaration *sinit;
#endif

    void accept(Visitor *v) { v->visit(this); }
};

class DotIdExp;
DotIdExp *typeDotIdExp(Loc loc, Type *type, Identifier *ident);

class TypeExp : public Expression
{
public:
    TypeExp(Loc loc, Type *type);
    Expression *syntaxCopy();
    Expression *semantic(Scope *sc);
    bool checkValue();
    void accept(Visitor *v) { v->visit(this); }
};

class ScopeExp : public Expression
{
public:
    ScopeDsymbol *sds;

    ScopeExp(Loc loc, ScopeDsymbol *sds);
    Expression *syntaxCopy();
    Expression *semantic(Scope *sc);
    void accept(Visitor *v) { v->visit(this); }
};

class TemplateExp : public Expression
{
public:
    TemplateDeclaration *td;
    FuncDeclaration *fd;

    TemplateExp(Loc loc, TemplateDeclaration *td, FuncDeclaration *fd = NULL);
    bool isLvalue();
    Expression *toLvalue(Scope *sc, Expression *e);
    bool checkValue();
    void accept(Visitor *v) { v->visit(this); }
};

class NewExp : public Expression
{
public:
    /* thisexp.new(newargs) newtype(arguments)
     */
    Expression *thisexp;        // if !NULL, 'this' for class being allocated
    Expressions *newargs;       // Array of Expression's to call new operator
    Type *newtype;
    Expressions *arguments;     // Array of Expression's

    Expression *argprefix;      // expression to be evaluated just before arguments[]

    CtorDeclaration *member;    // constructor function
    NewDeclaration *allocator;  // allocator function
    int onstack;                // allocate on stack

    NewExp(Loc loc, Expression *thisexp, Expressions *newargs,
        Type *newtype, Expressions *arguments);
    Expression *syntaxCopy();
    Expression *semantic(Scope *sc);

    void accept(Visitor *v) { v->visit(this); }
};

class NewAnonClassExp : public Expression
{
public:
    /* thisexp.new(newargs) class baseclasses { } (arguments)
     */
    Expression *thisexp;        // if !NULL, 'this' for class being allocated
    Expressions *newargs;       // Array of Expression's to call new operator
    ClassDeclaration *cd;       // class being instantiated
    Expressions *arguments;     // Array of Expression's to call class constructor

    NewAnonClassExp(Loc loc, Expression *thisexp, Expressions *newargs,
        ClassDeclaration *cd, Expressions *arguments);
    Expression *syntaxCopy();
    Expression *semantic(Scope *sc);
    void accept(Visitor *v) { v->visit(this); }
};

class SymbolExp : public Expression
{
public:
    Declaration *var;
    bool hasOverloads;

    SymbolExp(Loc loc, TOK op, int size, Declaration *var, bool hasOverloads);

    void accept(Visitor *v) { v->visit(this); }
};

// Offset from symbol

class SymOffExp : public SymbolExp
{
public:
    dinteger_t offset;

    SymOffExp(Loc loc, Declaration *var, dinteger_t offset, bool hasOverloads = false);
    Expression *semantic(Scope *sc);
    bool isBool(bool result);

    void accept(Visitor *v) { v->visit(this); }
};

// Variable

class VarExp : public SymbolExp
{
public:
    VarExp(Loc loc, Declaration *var, bool hasOverloads = false);
    static VarExp *create(Loc loc, Declaration *var, bool hasOverloads = false);
    bool equals(RootObject *o);
    Expression *semantic(Scope *sc);
    int checkModifiable(Scope *sc, int flag);
    bool checkReadModifyWrite();
    bool isLvalue();
    Expression *toLvalue(Scope *sc, Expression *e);
    Expression *modifiableLvalue(Scope *sc, Expression *e);

    void accept(Visitor *v) { v->visit(this); }
};

// Overload Set

class OverExp : public Expression
{
public:
    OverloadSet *vars;

    OverExp(Loc loc, OverloadSet *s);
    bool isLvalue();
    Expression *toLvalue(Scope *sc, Expression *e);
    void accept(Visitor *v) { v->visit(this); }
};

// Function/Delegate literal

class FuncExp : public Expression
{
public:
    FuncLiteralDeclaration *fd;
    TemplateDeclaration *td;
    TOK tok;

    FuncExp(Loc loc, Dsymbol *s);
    void genIdent(Scope *sc);
    Expression *syntaxCopy();
    Expression *semantic(Scope *sc);
    Expression *semantic(Scope *sc, Expressions *arguments);
    MATCH matchType(Type *to, Scope *sc, FuncExp **pfe, int flag = 0);
    char *toChars();
    bool checkValue();

    void accept(Visitor *v) { v->visit(this); }
};

// Declaration of a symbol

// D grammar allows declarations only as statements. However in AST representation
// it can be part of any expression. This is used, for example, during internal
// syntax re-writes to inject hidden symbols.
class DeclarationExp : public Expression
{
public:
    Dsymbol *declaration;

    DeclarationExp(Loc loc, Dsymbol *declaration);
    Expression *syntaxCopy();
    Expression *semantic(Scope *sc);

    void accept(Visitor *v) { v->visit(this); }
};

class TypeidExp : public Expression
{
public:
    RootObject *obj;

    TypeidExp(Loc loc, RootObject *obj);
    Expression *syntaxCopy();
    Expression *semantic(Scope *sc);
    void accept(Visitor *v) { v->visit(this); }
};

class TraitsExp : public Expression
{
public:
    Identifier *ident;
    Objects *args;

    TraitsExp(Loc loc, Identifier *ident, Objects *args);
    Expression *syntaxCopy();
    Expression *semantic(Scope *sc);
    void accept(Visitor *v) { v->visit(this); }
};

class HaltExp : public Expression
{
public:
    HaltExp(Loc loc);
    Expression *semantic(Scope *sc);

    void accept(Visitor *v) { v->visit(this); }
};

class IsExp : public Expression
{
public:
    /* is(targ id tok tspec)
     * is(targ id == tok2)
     */
    Type *targ;
    Identifier *id;     // can be NULL
    TOK tok;       // ':' or '=='
    Type *tspec;        // can be NULL
    TOK tok2;      // 'struct', 'union', 'typedef', etc.
    TemplateParameters *parameters;

    IsExp(Loc loc, Type *targ, Identifier *id, TOK tok, Type *tspec,
        TOK tok2, TemplateParameters *parameters);
    Expression *syntaxCopy();
    Expression *semantic(Scope *sc);
    void accept(Visitor *v) { v->visit(this); }
};

/****************************************************************/

class UnaExp : public Expression
{
public:
    Expression *e1;
    Type *att1; // Save alias this type to detect recursion

    UnaExp(Loc loc, TOK op, int size, Expression *e1);
    Expression *syntaxCopy();
    Expression *semantic(Scope *sc) = 0;
    Expression *unaSemantic(Scope *sc);
    Expression *resolveLoc(Loc loc, Scope *sc);

    void accept(Visitor *v) { v->visit(this); }
};

typedef UnionExp (*fp_t)(Type *, Expression *, Expression *);
typedef int (*fp2_t)(Loc loc, TOK, Expression *, Expression *);

class BinExp : public Expression
{
public:
    Expression *e1;
    Expression *e2;

    Type *att1; // Save alias this type to detect recursion
    Type *att2; // Save alias this type to detect recursion

    BinExp(Loc loc, TOK op, int size, Expression *e1, Expression *e2);
    Expression *syntaxCopy();
    Expression *semantic(Scope *sc) = 0;
    Expression *binSemantic(Scope *sc);
    Expression *binSemanticProp(Scope *sc);
    Expression *incompatibleTypes();
    Expression *checkOpAssignTypes(Scope *sc);
    bool checkIntegralBin();
    bool checkArithmeticBin();

    Expression *reorderSettingAAElem(Scope *sc);

    void accept(Visitor *v) { v->visit(this); }
};

class BinAssignExp : public BinExp
{
public:
    BinAssignExp(Loc loc, TOK op, int size, Expression *e1, Expression *e2)
        : BinExp(loc, op, size, e1, e2)
    {
    }

    Expression *semantic(Scope *sc);

    bool isLvalue();
    Expression *toLvalue(Scope *sc, Expression *ex);
    Expression *modifiableLvalue(Scope *sc, Expression *e);
    void accept(Visitor *v) { v->visit(this); }
};

/****************************************************************/

class CompileExp : public UnaExp
{
public:
    CompileExp(Loc loc, Expression *e);
    Expression *semantic(Scope *sc);
    void accept(Visitor *v) { v->visit(this); }
};

class FileExp : public UnaExp
{
public:
    FileExp(Loc loc, Expression *e);
    Expression *semantic(Scope *sc);
    void accept(Visitor *v) { v->visit(this); }
};

class AssertExp : public UnaExp
{
public:
    Expression *msg;

    AssertExp(Loc loc, Expression *e, Expression *msg = NULL);
    Expression *syntaxCopy();
    Expression *semantic(Scope *sc);

    void accept(Visitor *v) { v->visit(this); }
};

class DotIdExp : public UnaExp
{
public:
    Identifier *ident;

    DotIdExp(Loc loc, Expression *e, Identifier *ident);
    static DotIdExp *create(Loc loc, Expression *e, Identifier *ident);
    Expression *semantic(Scope *sc);
    Expression *semanticX(Scope *sc);
    Expression *semanticY(Scope *sc, int flag);
    void accept(Visitor *v) { v->visit(this); }
};

class DotTemplateExp : public UnaExp
{
public:
    TemplateDeclaration *td;

    DotTemplateExp(Loc loc, Expression *e, TemplateDeclaration *td);
    Expression *semantic(Scope *sc);
    void accept(Visitor *v) { v->visit(this); }
};

class DotVarExp : public UnaExp
{
public:
    Declaration *var;
    bool hasOverloads;

    DotVarExp(Loc loc, Expression *e, Declaration *var, bool hasOverloads = false);
    Expression *semantic(Scope *sc);
    int checkModifiable(Scope *sc, int flag);
    bool checkReadModifyWrite();
    bool isLvalue();
    Expression *toLvalue(Scope *sc, Expression *e);
    Expression *modifiableLvalue(Scope *sc, Expression *e);
    void accept(Visitor *v) { v->visit(this); }
};

class DotTemplateInstanceExp : public UnaExp
{
public:
    TemplateInstance *ti;

    DotTemplateInstanceExp(Loc loc, Expression *e, Identifier *name, Objects *tiargs);
    DotTemplateInstanceExp(Loc loc, Expression *e, TemplateInstance *ti);
    Expression *syntaxCopy();
    bool findTempDecl(Scope *sc);
    Expression *semantic(Scope *sc);
    Expression *semanticY(Scope *sc, int flag);
    void accept(Visitor *v) { v->visit(this); }
};

class DelegateExp : public UnaExp
{
public:
    FuncDeclaration *func;
    bool hasOverloads;

    DelegateExp(Loc loc, Expression *e, FuncDeclaration *func, bool hasOverloads = false);
    Expression *semantic(Scope *sc);

    void accept(Visitor *v) { v->visit(this); }
};

class DotTypeExp : public UnaExp
{
public:
    Dsymbol *sym;               // symbol that represents a type

    DotTypeExp(Loc loc, Expression *e, Dsymbol *sym);
    Expression *semantic(Scope *sc);
    void accept(Visitor *v) { v->visit(this); }
};

class CallExp : public UnaExp
{
public:
    Expressions *arguments;     // function arguments
    FuncDeclaration *f;         // symbol to call
    bool directcall;            // true if a virtual call is devirtualized

    CallExp(Loc loc, Expression *e, Expressions *exps);
    CallExp(Loc loc, Expression *e);
    CallExp(Loc loc, Expression *e, Expression *earg1);
    CallExp(Loc loc, Expression *e, Expression *earg1, Expression *earg2);

    static CallExp *create(Loc loc, Expression *e, Expressions *exps);
    static CallExp *create(Loc loc, Expression *e);
    static CallExp *create(Loc loc, Expression *e, Expression *earg1);

    Expression *syntaxCopy();
    Expression *semantic(Scope *sc);
    bool isLvalue();
    Expression *toLvalue(Scope *sc, Expression *e);
    Expression *addDtorHook(Scope *sc);

    void accept(Visitor *v) { v->visit(this); }
};

class AddrExp : public UnaExp
{
public:
    AddrExp(Loc loc, Expression *e);
    Expression *semantic(Scope *sc);

    void accept(Visitor *v) { v->visit(this); }
};

class PtrExp : public UnaExp
{
public:
    PtrExp(Loc loc, Expression *e);
    PtrExp(Loc loc, Expression *e, Type *t);
    Expression *semantic(Scope *sc);
    int checkModifiable(Scope *sc, int flag);
    bool isLvalue();
    Expression *toLvalue(Scope *sc, Expression *e);
    Expression *modifiableLvalue(Scope *sc, Expression *e);

    void accept(Visitor *v) { v->visit(this); }
};

class NegExp : public UnaExp
{
public:
    NegExp(Loc loc, Expression *e);
    Expression *semantic(Scope *sc);

    void accept(Visitor *v) { v->visit(this); }
};

class UAddExp : public UnaExp
{
public:
    UAddExp(Loc loc, Expression *e);
    Expression *semantic(Scope *sc);

    void accept(Visitor *v) { v->visit(this); }
};

class ComExp : public UnaExp
{
public:
    ComExp(Loc loc, Expression *e);
    Expression *semantic(Scope *sc);

    void accept(Visitor *v) { v->visit(this); }
};

class NotExp : public UnaExp
{
public:
    NotExp(Loc loc, Expression *e);
    Expression *semantic(Scope *sc);
    void accept(Visitor *v) { v->visit(this); }
};

class BoolExp : public UnaExp
{
public:
    BoolExp(Loc loc, Expression *e, Type *type);
    Expression *semantic(Scope *sc);
    void accept(Visitor *v) { v->visit(this); }
};

class DeleteExp : public UnaExp
{
public:
    DeleteExp(Loc loc, Expression *e);
    Expression *semantic(Scope *sc);
    Expression *toBoolean(Scope *sc);
    void accept(Visitor *v) { v->visit(this); }
};

class CastExp : public UnaExp
{
public:
    // Possible to cast to one type while painting to another type
    Type *to;                   // type to cast to
    unsigned char mod;          // MODxxxxx

    CastExp(Loc loc, Expression *e, Type *t);
    CastExp(Loc loc, Expression *e, unsigned char mod);
    Expression *syntaxCopy();
    Expression *semantic(Scope *sc);

    void accept(Visitor *v) { v->visit(this); }
};

class VectorExp : public UnaExp
{
public:
    TypeVector *to;             // the target vector type before semantic()
    unsigned dim;               // number of elements in the vector

    VectorExp(Loc loc, Expression *e, Type *t);
    Expression *syntaxCopy();
    Expression *semantic(Scope *sc);
    void accept(Visitor *v) { v->visit(this); }
};

class SliceExp : public UnaExp
{
public:
    Expression *upr;            // NULL if implicit 0
    Expression *lwr;            // NULL if implicit [length - 1]
    VarDeclaration *lengthVar;
    bool upperIsInBounds;       // true if upr <= e1.length
    bool lowerIsLessThanUpper;  // true if lwr <= upr

    SliceExp(Loc loc, Expression *e1, IntervalExp *ie);
    SliceExp(Loc loc, Expression *e1, Expression *lwr, Expression *upr);
    Expression *syntaxCopy();
    Expression *semantic(Scope *sc);
    int checkModifiable(Scope *sc, int flag);
    bool isLvalue();
    Expression *toLvalue(Scope *sc, Expression *e);
    Expression *modifiableLvalue(Scope *sc, Expression *e);
    bool isBool(bool result);

    void accept(Visitor *v) { v->visit(this); }
};

class ArrayLengthExp : public UnaExp
{
public:
    ArrayLengthExp(Loc loc, Expression *e1);
    Expression *semantic(Scope *sc);

    static Expression *rewriteOpAssign(BinExp *exp);
    void accept(Visitor *v) { v->visit(this); }
};

class IntervalExp : public Expression
{
public:
    Expression *lwr;
    Expression *upr;

    IntervalExp(Loc loc, Expression *lwr, Expression *upr);
    Expression *syntaxCopy();
    Expression *semantic(Scope *sc);
    void accept(Visitor *v) { v->visit(this); }
};

class DelegatePtrExp : public UnaExp
{
public:
    DelegatePtrExp(Loc loc, Expression *e1);
    Expression *semantic(Scope *sc);
    bool isLvalue();
    Expression *toLvalue(Scope *sc, Expression *e);
    void accept(Visitor *v) { v->visit(this); }
};

class DelegateFuncptrExp : public UnaExp
{
public:
    DelegateFuncptrExp(Loc loc, Expression *e1);
    Expression *semantic(Scope *sc);
    bool isLvalue();
    Expression *toLvalue(Scope *sc, Expression *e);
    void accept(Visitor *v) { v->visit(this); }
};

// e1[a0,a1,a2,a3,...]

class ArrayExp : public UnaExp
{
public:
    Expressions *arguments;             // Array of Expression's
    size_t currentDimension;            // for opDollar
    VarDeclaration *lengthVar;

    ArrayExp(Loc loc, Expression *e1, Expression *index = NULL);
    ArrayExp(Loc loc, Expression *e1, Expressions *args);
    Expression *syntaxCopy();
    Expression *semantic(Scope *sc);
    bool isLvalue();
    Expression *toLvalue(Scope *sc, Expression *e);

    void accept(Visitor *v) { v->visit(this); }
};

/****************************************************************/

class DotExp : public BinExp
{
public:
    DotExp(Loc loc, Expression *e1, Expression *e2);
    Expression *semantic(Scope *sc);
    void accept(Visitor *v) { v->visit(this); }
};

class CommaExp : public BinExp
{
public:
    CommaExp(Loc loc, Expression *e1, Expression *e2);
    Expression *semantic(Scope *sc);
    int checkModifiable(Scope *sc, int flag);
    bool isLvalue();
    Expression *toLvalue(Scope *sc, Expression *e);
    Expression *modifiableLvalue(Scope *sc, Expression *e);
    bool isBool(bool result);
    Expression *addDtorHook(Scope *sc);
    void accept(Visitor *v) { v->visit(this); }
};

class IndexExp : public BinExp
{
public:
    VarDeclaration *lengthVar;
    bool modifiable;
    bool indexIsInBounds;       // true if 0 <= e2 && e2 <= e1.length - 1

    IndexExp(Loc loc, Expression *e1, Expression *e2);
    Expression *syntaxCopy();
    Expression *semantic(Scope *sc);
    int checkModifiable(Scope *sc, int flag);
    bool isLvalue();
    Expression *toLvalue(Scope *sc, Expression *e);
    Expression *modifiableLvalue(Scope *sc, Expression *e);

    Expression *markSettingAAElem();

    void accept(Visitor *v) { v->visit(this); }
};

/* For both i++ and i--
 */
class PostExp : public BinExp
{
public:
    PostExp(TOK op, Loc loc, Expression *e);
    Expression *semantic(Scope *sc);
    void accept(Visitor *v) { v->visit(this); }
};

/* For both ++i and --i
 */
class PreExp : public UnaExp
{
public:
    PreExp(TOK op, Loc loc, Expression *e);
    Expression *semantic(Scope *sc);
    void accept(Visitor *v) { v->visit(this); }
};

class AssignExp : public BinExp
{
public:
    // &1 != 0 if setting the contents of an array
    // &2 != 0 if setting the content of ref variable
    int ismemset;

    AssignExp(Loc loc, Expression *e1, Expression *e2);
    Expression *semantic(Scope *sc);
    bool isLvalue();
    Expression *toLvalue(Scope *sc, Expression *ex);
    Expression *toBoolean(Scope *sc);

    void accept(Visitor *v) { v->visit(this); }
};

class ConstructExp : public AssignExp
{
public:
    ConstructExp(Loc loc, Expression *e1, Expression *e2);
    void accept(Visitor *v) { v->visit(this); }
};

class BlitExp : public AssignExp
{
public:
    BlitExp(Loc loc, Expression *e1, Expression *e2);
    void accept(Visitor *v) { v->visit(this); }
};

class AddAssignExp : public BinAssignExp
{
public:
    AddAssignExp(Loc loc, Expression *e1, Expression *e2);
    void accept(Visitor *v) { v->visit(this); }
};

class MinAssignExp : public BinAssignExp
{
public:
    MinAssignExp(Loc loc, Expression *e1, Expression *e2);
    void accept(Visitor *v) { v->visit(this); }
};

class MulAssignExp : public BinAssignExp
{
public:
    MulAssignExp(Loc loc, Expression *e1, Expression *e2);
    void accept(Visitor *v) { v->visit(this); }
};

class DivAssignExp : public BinAssignExp
{
public:
    DivAssignExp(Loc loc, Expression *e1, Expression *e2);
    void accept(Visitor *v) { v->visit(this); }
};

class ModAssignExp : public BinAssignExp
{
public:
    ModAssignExp(Loc loc, Expression *e1, Expression *e2);
    void accept(Visitor *v) { v->visit(this); }
};

class AndAssignExp : public BinAssignExp
{
public:
    AndAssignExp(Loc loc, Expression *e1, Expression *e2);
    void accept(Visitor *v) { v->visit(this); }
};

class OrAssignExp : public BinAssignExp
{
public:
    OrAssignExp(Loc loc, Expression *e1, Expression *e2);
    void accept(Visitor *v) { v->visit(this); }
};

class XorAssignExp : public BinAssignExp
{
public:
    XorAssignExp(Loc loc, Expression *e1, Expression *e2);
    void accept(Visitor *v) { v->visit(this); }
};

class PowAssignExp : public BinAssignExp
{
public:
    PowAssignExp(Loc loc, Expression *e1, Expression *e2);
    Expression *semantic(Scope *sc);
    void accept(Visitor *v) { v->visit(this); }
};

class ShlAssignExp : public BinAssignExp
{
public:
    ShlAssignExp(Loc loc, Expression *e1, Expression *e2);
    void accept(Visitor *v) { v->visit(this); }
};

class ShrAssignExp : public BinAssignExp
{
public:
    ShrAssignExp(Loc loc, Expression *e1, Expression *e2);
    void accept(Visitor *v) { v->visit(this); }
};

class UshrAssignExp : public BinAssignExp
{
public:
    UshrAssignExp(Loc loc, Expression *e1, Expression *e2);
    void accept(Visitor *v) { v->visit(this); }
};

class CatAssignExp : public BinAssignExp
{
public:
    CatAssignExp(Loc loc, Expression *e1, Expression *e2);
    Expression *semantic(Scope *sc);
    void accept(Visitor *v) { v->visit(this); }
};

class AddExp : public BinExp
{
public:
    AddExp(Loc loc, Expression *e1, Expression *e2);
    Expression *semantic(Scope *sc);

    void accept(Visitor *v) { v->visit(this); }
};

class MinExp : public BinExp
{
public:
    MinExp(Loc loc, Expression *e1, Expression *e2);
    Expression *semantic(Scope *sc);

    void accept(Visitor *v) { v->visit(this); }
};

class CatExp : public BinExp
{
public:
    CatExp(Loc loc, Expression *e1, Expression *e2);
    Expression *semantic(Scope *sc);

    void accept(Visitor *v) { v->visit(this); }
};

class MulExp : public BinExp
{
public:
    MulExp(Loc loc, Expression *e1, Expression *e2);
    Expression *semantic(Scope *sc);

    void accept(Visitor *v) { v->visit(this); }
};

class DivExp : public BinExp
{
public:
    DivExp(Loc loc, Expression *e1, Expression *e2);
    Expression *semantic(Scope *sc);

    void accept(Visitor *v) { v->visit(this); }
};

class ModExp : public BinExp
{
public:
    ModExp(Loc loc, Expression *e1, Expression *e2);
    Expression *semantic(Scope *sc);

    void accept(Visitor *v) { v->visit(this); }
};

class PowExp : public BinExp
{
public:
    PowExp(Loc loc, Expression *e1, Expression *e2);
    Expression *semantic(Scope *sc);

    void accept(Visitor *v) { v->visit(this); }
};

class ShlExp : public BinExp
{
public:
    ShlExp(Loc loc, Expression *e1, Expression *e2);
    Expression *semantic(Scope *sc);

    void accept(Visitor *v) { v->visit(this); }
};

class ShrExp : public BinExp
{
public:
    ShrExp(Loc loc, Expression *e1, Expression *e2);
    Expression *semantic(Scope *sc);

    void accept(Visitor *v) { v->visit(this); }
};

class UshrExp : public BinExp
{
public:
    UshrExp(Loc loc, Expression *e1, Expression *e2);
    Expression *semantic(Scope *sc);

    void accept(Visitor *v) { v->visit(this); }
};

class AndExp : public BinExp
{
public:
    AndExp(Loc loc, Expression *e1, Expression *e2);
    Expression *semantic(Scope *sc);

    void accept(Visitor *v) { v->visit(this); }
};

class OrExp : public BinExp
{
public:
    OrExp(Loc loc, Expression *e1, Expression *e2);
    Expression *semantic(Scope *sc);

    void accept(Visitor *v) { v->visit(this); }
};

class XorExp : public BinExp
{
public:
    XorExp(Loc loc, Expression *e1, Expression *e2);
    Expression *semantic(Scope *sc);

    void accept(Visitor *v) { v->visit(this); }
};

class OrOrExp : public BinExp
{
public:
    OrOrExp(Loc loc, Expression *e1, Expression *e2);
    Expression *semantic(Scope *sc);
    Expression *toBoolean(Scope *sc);
    void accept(Visitor *v) { v->visit(this); }
};

class AndAndExp : public BinExp
{
public:
    AndAndExp(Loc loc, Expression *e1, Expression *e2);
    Expression *semantic(Scope *sc);
    Expression *toBoolean(Scope *sc);
    void accept(Visitor *v) { v->visit(this); }
};

class CmpExp : public BinExp
{
public:
    CmpExp(TOK op, Loc loc, Expression *e1, Expression *e2);
    Expression *semantic(Scope *sc);

    void accept(Visitor *v) { v->visit(this); }
};

class InExp : public BinExp
{
public:
    InExp(Loc loc, Expression *e1, Expression *e2);
    Expression *semantic(Scope *sc);

    void accept(Visitor *v) { v->visit(this); }
};

class RemoveExp : public BinExp
{
public:
    RemoveExp(Loc loc, Expression *e1, Expression *e2);
    Expression *semantic(Scope *sc);
    void accept(Visitor *v) { v->visit(this); }
};

// == and !=

class EqualExp : public BinExp
{
public:
    EqualExp(TOK op, Loc loc, Expression *e1, Expression *e2);
    Expression *semantic(Scope *sc);

    void accept(Visitor *v) { v->visit(this); }
};

// is and !is

class IdentityExp : public BinExp
{
public:
    IdentityExp(TOK op, Loc loc, Expression *e1, Expression *e2);
    Expression *semantic(Scope *sc);
    void accept(Visitor *v) { v->visit(this); }
};

/****************************************************************/

class CondExp : public BinExp
{
public:
    Expression *econd;

    CondExp(Loc loc, Expression *econd, Expression *e1, Expression *e2);
    Expression *syntaxCopy();
    Expression *semantic(Scope *sc);
    int checkModifiable(Scope *sc, int flag);
    bool isLvalue();
    Expression *toLvalue(Scope *sc, Expression *e);
    Expression *modifiableLvalue(Scope *sc, Expression *e);
    Expression *toBoolean(Scope *sc);
    void hookDtors(Scope *sc);

    void accept(Visitor *v) { v->visit(this); }
};

/****************************************************************/

class DefaultInitExp : public Expression
{
public:
    TOK subop;             // which of the derived classes this is

    DefaultInitExp(Loc loc, TOK subop, int size);
    void accept(Visitor *v) { v->visit(this); }
};

class FileInitExp : public DefaultInitExp
{
public:
    FileInitExp(Loc loc);
    Expression *semantic(Scope *sc);
    Expression *resolveLoc(Loc loc, Scope *sc);
    void accept(Visitor *v) { v->visit(this); }
};

class LineInitExp : public DefaultInitExp
{
public:
    LineInitExp(Loc loc);
    Expression *semantic(Scope *sc);
    Expression *resolveLoc(Loc loc, Scope *sc);
    void accept(Visitor *v) { v->visit(this); }
};

class ModuleInitExp : public DefaultInitExp
{
public:
    ModuleInitExp(Loc loc);
    Expression *semantic(Scope *sc);
    Expression *resolveLoc(Loc loc, Scope *sc);
    void accept(Visitor *v) { v->visit(this); }
};

class FuncInitExp : public DefaultInitExp
{
public:
    FuncInitExp(Loc loc);
    Expression *semantic(Scope *sc);
    Expression *resolveLoc(Loc loc, Scope *sc);
    void accept(Visitor *v) { v->visit(this); }
};

class PrettyFuncInitExp : public DefaultInitExp
{
public:
    PrettyFuncInitExp(Loc loc);
    Expression *semantic(Scope *sc);
    Expression *resolveLoc(Loc loc, Scope *sc);
    void accept(Visitor *v) { v->visit(this); }
};

/****************************************************************/

/* A type meant as a union of all the Expression types,
 * to serve essentially as a Variant that will sit on the stack
 * during CTFE to reduce memory consumption.
 */
struct UnionExp
{
    UnionExp() { }  // yes, default constructor does nothing

    UnionExp(Expression *e)
    {
        memcpy(this, (void *)e, e->size);
    }

    /* Extract pointer to Expression
     */
    Expression *exp() { return (Expression *)&u; }

    /* Convert to an allocated Expression
     */
    Expression *copy();

private:
    union
    {
        char exp       [sizeof(Expression)];
        char integerexp[sizeof(IntegerExp)];
        char errorexp  [sizeof(ErrorExp)];
        char realexp   [sizeof(RealExp)];
        char complexexp[sizeof(ComplexExp)];
        char symoffexp [sizeof(SymOffExp)];
        char stringexp [sizeof(StringExp)];
        char arrayliteralexp [sizeof(ArrayLiteralExp)];
        char assocarrayliteralexp [sizeof(AssocArrayLiteralExp)];
        char structliteralexp [sizeof(StructLiteralExp)];
        char nullexp   [sizeof(NullExp)];
        char dotvarexp [sizeof(DotVarExp)];
        char addrexp   [sizeof(AddrExp)];
        char indexexp  [sizeof(IndexExp)];
        char sliceexp  [sizeof(SliceExp)];

        // Ensure that the union is suitably aligned.
        longdouble for_alignment_only;
    } u;
};

/****************************************************************/

/* Special values used by the interpreter
 */

Expression *expType(Type *type, Expression *e);

UnionExp Neg(Type *type, Expression *e1);
UnionExp Com(Type *type, Expression *e1);
UnionExp Not(Type *type, Expression *e1);
UnionExp Bool(Type *type, Expression *e1);
UnionExp Cast(Type *type, Type *to, Expression *e1);
UnionExp ArrayLength(Type *type, Expression *e1);
UnionExp Ptr(Type *type, Expression *e1);

UnionExp Add(Type *type, Expression *e1, Expression *e2);
UnionExp Min(Type *type, Expression *e1, Expression *e2);
UnionExp Mul(Type *type, Expression *e1, Expression *e2);
UnionExp Div(Type *type, Expression *e1, Expression *e2);
UnionExp Mod(Type *type, Expression *e1, Expression *e2);
UnionExp Pow(Type *type, Expression *e1, Expression *e2);
UnionExp Shl(Type *type, Expression *e1, Expression *e2);
UnionExp Shr(Type *type, Expression *e1, Expression *e2);
UnionExp Ushr(Type *type, Expression *e1, Expression *e2);
UnionExp And(Type *type, Expression *e1, Expression *e2);
UnionExp Or(Type *type, Expression *e1, Expression *e2);
UnionExp Xor(Type *type, Expression *e1, Expression *e2);
UnionExp Index(Type *type, Expression *e1, Expression *e2);
UnionExp Cat(Type *type, Expression *e1, Expression *e2);

UnionExp Equal(TOK op, Type *type, Expression *e1, Expression *e2);
UnionExp Cmp(TOK op, Type *type, Expression *e1, Expression *e2);
UnionExp Identity(TOK op, Type *type, Expression *e1, Expression *e2);

UnionExp Slice(Type *type, Expression *e1, Expression *lwr, Expression *upr);

// Const-folding functions used by CTFE

void sliceAssignArrayLiteralFromString(ArrayLiteralExp *existingAE, StringExp *newval, size_t firstIndex);
void sliceAssignStringFromArrayLiteral(StringExp *existingSE, ArrayLiteralExp *newae, size_t firstIndex);
void sliceAssignStringFromString(StringExp *existingSE, StringExp *newstr, size_t firstIndex);

int sliceCmpStringWithString(StringExp *se1, StringExp *se2, size_t lo1, size_t lo2, size_t len);
int sliceCmpStringWithArray(StringExp *se1, ArrayLiteralExp *ae2, size_t lo1, size_t lo2, size_t len);


#endif /* DMD_EXPRESSION_H */
