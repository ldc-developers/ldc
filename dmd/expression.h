
/* Compiler implementation of the D programming language
 * Copyright (C) 1999-2020 by The D Language Foundation, All Rights Reserved
 * written by Walter Bright
 * http://www.digitalmars.com
 * Distributed under the Boost Software License, Version 1.0.
 * http://www.boost.org/LICENSE_1_0.txt
 * https://github.com/dlang/dmd/blob/master/src/dmd/expression.h
 */

#pragma once

#include "ast_node.h"
#include "complex_t.h"
#include "globals.h"
#include "arraytypes.h"
#include "visitor.h"
#include "tokens.h"

#include "root/dcompat.h"

class Type;
class TypeVector;
struct Scope;
class TupleDeclaration;
class VarDeclaration;
class FuncDeclaration;
class FuncLiteralDeclaration;
class CtorDeclaration;
class NewDeclaration;
class Dsymbol;
class ScopeDsymbol;
class Expression;
class Declaration;
class StructDeclaration;
class TemplateInstance;
class TemplateDeclaration;
class ClassDeclaration;
class OverloadSet;
class StringExp;
struct UnionExp;
#ifdef IN_GCC
typedef union tree_node Symbol;
#elif !IN_LLVM
struct Symbol;          // back end symbol
#endif

#if IN_LLVM
namespace llvm {
    class Value;
}

// in expressionsem.d
Expression *expressionSemantic(Expression *e, Scope *sc);
// in typesem.d
Expression *defaultInit(Type *mt, const Loc &loc);
#endif

void expandTuples(Expressions *exps);
bool isTrivialExp(Expression *e);
bool hasSideEffect(Expression *e);
bool canThrow(Expression *e, FuncDeclaration *func, bool mustNotThrow);

typedef unsigned char OwnedBy;
enum
{
    OWNEDcode,      // normal code expression in AST
    OWNEDctfe,      // value expression for CTFE
    OWNEDcache      // constant value cached for CTFE
};

#if IN_LLVM
#define WANTvalue  0    // default
#define WANTexpand 1    // expand const/immutable variables if possible
#endif

class Expression : public ASTNode
{
public:
    TOK op;                     // to minimize use of dynamic_cast
    unsigned char size;         // # of bytes in Expression so we can copy() it
    unsigned char parens;       // if this is a parenthesized expression
    Type *type;                 // !=NULL means that semantic() has been run
    Loc loc;                    // file location

    static void _init();
    Expression *copy();
    virtual Expression *syntaxCopy();

    // kludge for template.isExpression()
    DYNCAST dyncast() const { return DYNCAST_EXPRESSION; }

    const char *toChars() const;
    void error(const char *format, ...) const;
    void warning(const char *format, ...) const;
    void deprecation(const char *format, ...) const;

    virtual dinteger_t toInteger();
    virtual uinteger_t toUInteger();
    virtual real_t toReal();
    virtual real_t toImaginary();
    virtual complex_t toComplex();
    virtual StringExp *toStringExp();
    virtual TupleExp *toTupleExp();
    virtual bool isLvalue();
    virtual Expression *toLvalue(Scope *sc, Expression *e);
    virtual Expression *modifiableLvalue(Scope *sc, Expression *e);
    Expression *implicitCastTo(Scope *sc, Type *t);
    MATCH implicitConvTo(Type *t);
    Expression *castTo(Scope *sc, Type *t);
    virtual Expression *resolveLoc(const Loc &loc, Scope *sc);
    virtual bool checkType();
    virtual bool checkValue();
    bool checkDeprecated(Scope *sc, Dsymbol *s);
    virtual int checkModifiable(Scope *sc, int flag = 0);
    virtual Expression *toBoolean(Scope *sc);
    virtual Expression *addDtorHook(Scope *sc);
    Expression *addressOf();
    Expression *deref();

    Expression *optimize(int result, bool keepLvalue = false);

    // Entry point for CTFE.
    // A compile-time result is required. Give an error if not possible
    Expression *ctfeInterpret();
    int isConst();
    virtual bool isBool(bool result);

    virtual bool hasCode()
    {
        return true;
    }

    IntegerExp* isIntegerExp();
    ErrorExp* isErrorExp();
    VoidInitExp* isVoidInitExp();
    RealExp* isRealExp();
    ComplexExp* isComplexExp();
    IdentifierExp* isIdentifierExp();
    DollarExp* isDollarExp();
    DsymbolExp* isDsymbolExp();
    ThisExp* isThisExp();
    SuperExp* isSuperExp();
    NullExp* isNullExp();
    StringExp* isStringExp();
    TupleExp* isTupleExp();
    ArrayLiteralExp* isArrayLiteralExp();
    AssocArrayLiteralExp* isAssocArrayLiteralExp();
    StructLiteralExp* isStructLiteralExp();
    TypeExp* isTypeExp();
    ScopeExp* isScopeExp();
    TemplateExp* isTemplateExp();
    NewExp* isNewExp();
    NewAnonClassExp* isNewAnonClassExp();
    SymOffExp* isSymOffExp();
    VarExp* isVarExp();
    OverExp* isOverExp();
    FuncExp* isFuncExp();
    DeclarationExp* isDeclarationExp();
    TypeidExp* isTypeidExp();
    TraitsExp* isTraitsExp();
    HaltExp* isHaltExp();
    IsExp* isExp();
    CompileExp* isCompileExp();
    ImportExp* isImportExp();
    AssertExp* isAssertExp();
    DotIdExp* isDotIdExp();
    DotTemplateExp* isDotTemplateExp();
    DotVarExp* isDotVarExp();
    DotTemplateInstanceExp* isDotTemplateInstanceExp();
    DelegateExp* isDelegateExp();
    DotTypeExp* isDotTypeExp();
    CallExp* isCallExp();
    AddrExp* isAddrExp();
    PtrExp* isPtrExp();
    NegExp* isNegExp();
    UAddExp* isUAddExp();
    ComExp* isComExp();
    NotExp* isNotExp();
    DeleteExp* isDeleteExp();
    CastExp* isCastExp();
    VectorExp* isVectorExp();
    VectorArrayExp* isVectorArrayExp();
    SliceExp* isSliceExp();
    ArrayLengthExp* isArrayLengthExp();
    ArrayExp* isArrayExp();
    DotExp* isDotExp();
    CommaExp* isCommaExp();
    IntervalExp* isIntervalExp();
    DelegatePtrExp* isDelegatePtrExp();
    DelegateFuncptrExp* isDelegateFuncptrExp();
    IndexExp* isIndexExp();
    PostExp* isPostExp();
    PreExp* isPreExp();
    AssignExp* isAssignExp();
    ConstructExp* isConstructExp();
    BlitExp* isBlitExp();
    AddAssignExp* isAddAssignExp();
    MinAssignExp* isMinAssignExp();
    MulAssignExp* isMulAssignExp();
    DivAssignExp* isDivAssignExp();
    ModAssignExp* isModAssignExp();
    AndAssignExp* isAndAssignExp();
    OrAssignExp* isOrAssignExp();
    XorAssignExp* isXorAssignExp();
    PowAssignExp* isPowAssignExp();
    ShlAssignExp* isShlAssignExp();
    ShrAssignExp* isShrAssignExp();
    UshrAssignExp* isUshrAssignExp();
    CatAssignExp* isCatAssignExp();
    AddExp* isAddExp();
    MinExp* isMinExp();
    CatExp* isCatExp();
    MulExp* isMulExp();
    DivExp* isDivExp();
    ModExp* isModExp();
    PowExp* isPowExp();
    ShlExp* isShlExp();
    ShrExp* isShrExp();
    UshrExp* isUshrExp();
    AndExp* isAndExp();
    OrExp* isOrExp();
    XorExp* isXorExp();
    LogicalExp* isLogicalExp();
    InExp* isInExp();
    RemoveExp* isRemoveExp();
    EqualExp* isEqualExp();
    IdentityExp* isIdentityExp();
    CondExp* isCondExp();
    DefaultInitExp* isDefaultInitExp();
    FileInitExp* isFileInitExp();
    LineInitExp* isLineInitExp();
    ModuleInitExp* isModuleInitExp();
    FuncInitExp* isFuncInitExp();
    PrettyFuncInitExp* isPrettyFuncInitExp();

    void accept(Visitor *v) { v->visit(this); }
};

class IntegerExp : public Expression
{
public:
    dinteger_t value;

    static IntegerExp *create(Loc loc, dinteger_t value, Type *type);
    static void emplace(UnionExp *pue, Loc loc, dinteger_t value, Type *type);
    bool equals(const RootObject *o) const;
    dinteger_t toInteger();
    real_t toReal();
    real_t toImaginary();
    complex_t toComplex();
    bool isBool(bool result);
    Expression *toLvalue(Scope *sc, Expression *e);
    void accept(Visitor *v) { v->visit(this); }
    dinteger_t getInteger() { return value; }
    void setInteger(dinteger_t value);
    template<int v>
    static IntegerExp literal();
};

class ErrorExp : public Expression
{
public:
    Expression *toLvalue(Scope *sc, Expression *e);
    void accept(Visitor *v) { v->visit(this); }

    static ErrorExp *errorexp; // handy shared value
};

class RealExp : public Expression
{
public:
    real_t value;

    static RealExp *create(Loc loc, real_t value, Type *type);
    static void emplace(UnionExp *pue, Loc loc, real_t value, Type *type);
    bool equals(const RootObject *o) const;
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

    static ComplexExp *create(Loc loc, complex_t value, Type *type);
    static void emplace(UnionExp *pue, Loc loc, complex_t value, Type *type);
    bool equals(const RootObject *o) const;
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

    static IdentifierExp *create(Loc loc, Identifier *ident);
    bool isLvalue();
    Expression *toLvalue(Scope *sc, Expression *e);
    void accept(Visitor *v) { v->visit(this); }
};

class DollarExp : public IdentifierExp
{
public:
    void accept(Visitor *v) { v->visit(this); }
};

class DsymbolExp : public Expression
{
public:
    Dsymbol *s;
    bool hasOverloads;

    Expression *syntaxCopy();
    bool isLvalue();
    Expression *toLvalue(Scope *sc, Expression *e);
    void accept(Visitor *v) { v->visit(this); }
};

class ThisExp : public Expression
{
public:
    VarDeclaration *var;

    Expression *syntaxCopy();
    bool isBool(bool result);
    bool isLvalue();
    Expression *toLvalue(Scope *sc, Expression *e);

    void accept(Visitor *v) { v->visit(this); }
};

class SuperExp : public ThisExp
{
public:
    void accept(Visitor *v) { v->visit(this); }
};

class NullExp : public Expression
{
public:
    unsigned char committed;    // !=0 if type is committed

    bool equals(const RootObject *o) const;
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

    static StringExp *create(Loc loc, char *s);
    static StringExp *create(Loc loc, void *s, size_t len);
    static void emplace(UnionExp *pue, Loc loc, char *s);
    static void emplace(UnionExp *pue, Loc loc, void *s, size_t len);
    bool equals(const RootObject *o) const;
    StringExp *toStringExp();
    StringExp *toUTF8(Scope *sc);
    bool isBool(bool result);
    bool isLvalue();
    Expression *toLvalue(Scope *sc, Expression *e);
    Expression *modifiableLvalue(Scope *sc, Expression *e);
    unsigned charAt(uinteger_t i) const;
    void accept(Visitor *v) { v->visit(this); }
#if IN_LLVM
    // The D version returns a slice.
    DString peekString() const
    {
        assert(sz == 1);
        return {len, static_cast<const char *>(string)};
    }
    // ditto
    DArray<const unsigned char> peekData() const
    {
        return {len * sz, static_cast<const unsigned char *>(string)};
    }
#endif
    size_t numberOfCodeUnits(int tynto = 0) const;
    void writeTo(void* dest, bool zero, int tyto = 0) const;
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

    static TupleExp *create(Loc loc, Expressions *exps);
    TupleExp *toTupleExp();
    Expression *syntaxCopy();
    bool equals(const RootObject *o) const;

    void accept(Visitor *v) { v->visit(this); }
};

class ArrayLiteralExp : public Expression
{
public:
    Expression *basis;
    Expressions *elements;
    OwnedBy ownedByCtfe;

    static ArrayLiteralExp *create(Loc loc, Expressions *elements);
    static void emplace(UnionExp *pue, Loc loc, Expressions *elements);
    Expression *syntaxCopy();
    bool equals(const RootObject *o) const;
    Expression *getElement(d_size_t i); // use opIndex instead
    Expression *opIndex(d_size_t i);
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

    bool equals(const RootObject *o) const;
    Expression *syntaxCopy();
    bool isBool(bool result);

    void accept(Visitor *v) { v->visit(this); }
};

class StructLiteralExp : public Expression
{
public:
    StructDeclaration *sd;      // which aggregate this is for
    Expressions *elements;      // parallels sd->fields[] with NULL entries for fields to skip
    Type *stype;                // final type of result (can be different from sd's type)

#if IN_LLVM
    // With the introduction of pointers returned from CTFE, struct literals can
    // now contain pointers to themselves. While in toElem, contains a pointer
    // to the memory used to build the literal for resolving such references.
    llvm::Value *inProgressMemory;
#else
    Symbol *sym;        // back end symbol to initialize with literal
#endif

    /** pointer to the origin instance of the expression.
     * once a new expression is created, origin is set to 'this'.
     * anytime when an expression copy is created, 'origin' pointer is set to
     * 'origin' pointer value of the original expression.
     */
    StructLiteralExp *origin;

    // those fields need to prevent a infinite recursion when one field of struct initialized with 'this' pointer.
    StructLiteralExp *inlinecopy;

    /** anytime when recursive function is calling, 'stageflags' marks with bit flag of
     * current stage and unmarks before return from this function.
     * 'inlinecopy' uses similar 'stageflags' and from multiple evaluation 'doInline'
     * (with infinite recursion) of this expression.
     */
    int stageflags;

    bool useStaticInit;         // if this is true, use the StructDeclaration's init symbol
    bool isOriginal;            // used when moving instances to indicate `this is this.origin`
    OwnedBy ownedByCtfe;

    static StructLiteralExp *create(Loc loc, StructDeclaration *sd, void *elements, Type *stype = NULL);
    bool equals(const RootObject *o) const;
    Expression *syntaxCopy();
    Expression *getField(Type *type, unsigned offset);
    int getFieldIndex(Type *type, unsigned offset);
    Expression *addDtorHook(Scope *sc);

    void accept(Visitor *v) { v->visit(this); }
};

class TypeExp : public Expression
{
public:
    Expression *syntaxCopy();
    bool checkType();
    bool checkValue();
    void accept(Visitor *v) { v->visit(this); }
};

class ScopeExp : public Expression
{
public:
    ScopeDsymbol *sds;

    Expression *syntaxCopy();
    bool checkType();
    bool checkValue();
    void accept(Visitor *v) { v->visit(this); }
};

class TemplateExp : public Expression
{
public:
    TemplateDeclaration *td;
    FuncDeclaration *fd;

    bool isLvalue();
    Expression *toLvalue(Scope *sc, Expression *e);
    bool checkType();
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
    bool onstack;               // allocate on stack
    bool thrownew;              // this NewExp is the expression of a ThrowStatement

    static NewExp *create(Loc loc, Expression *thisexp, Expressions *newargs, Type *newtype, Expressions *arguments);
    Expression *syntaxCopy();

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

    Expression *syntaxCopy();
    void accept(Visitor *v) { v->visit(this); }
};

class SymbolExp : public Expression
{
public:
    Declaration *var;
    bool hasOverloads;
    Dsymbol *originalScope;

    void accept(Visitor *v) { v->visit(this); }
};

// Offset from symbol

class SymOffExp : public SymbolExp
{
public:
    dinteger_t offset;

    bool isBool(bool result);

    void accept(Visitor *v) { v->visit(this); }
};

// Variable

class VarExp : public SymbolExp
{
public:
    bool delegateWasExtracted;
    static VarExp *create(Loc loc, Declaration *var, bool hasOverloads = true);
    bool equals(const RootObject *o) const;
    int checkModifiable(Scope *sc, int flag);
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

    bool equals(const RootObject *o) const;
    Expression *syntaxCopy();
    const char *toChars() const;
    bool checkType();
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

    Expression *syntaxCopy();

    bool hasCode();

    void accept(Visitor *v) { v->visit(this); }
};

class TypeidExp : public Expression
{
public:
    RootObject *obj;

    Expression *syntaxCopy();
    void accept(Visitor *v) { v->visit(this); }
};

class TraitsExp : public Expression
{
public:
    Identifier *ident;
    Objects *args;

    Expression *syntaxCopy();
    void accept(Visitor *v) { v->visit(this); }
};

class HaltExp : public Expression
{
public:
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
    Type *tspec;        // can be NULL
    TemplateParameters *parameters;
    TOK tok;       // ':' or '=='
    TOK tok2;      // 'struct', 'union', etc.

    Expression *syntaxCopy();
    void accept(Visitor *v) { v->visit(this); }
};

/****************************************************************/

class UnaExp : public Expression
{
public:
    Expression *e1;
    Type *att1; // Save alias this type to detect recursion

    Expression *syntaxCopy();
    Expression *incompatibleTypes();
    Expression *resolveLoc(const Loc &loc, Scope *sc);

    void accept(Visitor *v) { v->visit(this); }
};

class BinExp : public Expression
{
public:
    Expression *e1;
    Expression *e2;

    Type *att1; // Save alias this type to detect recursion
    Type *att2; // Save alias this type to detect recursion

    Expression *syntaxCopy();
    Expression *incompatibleTypes();

    Expression *reorderSettingAAElem(Scope *sc);

    void accept(Visitor *v) { v->visit(this); }
};

class BinAssignExp : public BinExp
{
public:
    bool isLvalue();
    Expression *toLvalue(Scope *sc, Expression *ex);
    Expression *modifiableLvalue(Scope *sc, Expression *e);
    void accept(Visitor *v) { v->visit(this); }
};

/****************************************************************/

class CompileExp : public UnaExp
{
public:
    void accept(Visitor *v) { v->visit(this); }
};

class ImportExp : public UnaExp
{
public:
    void accept(Visitor *v) { v->visit(this); }
};

class AssertExp : public UnaExp
{
public:
    Expression *msg;

    Expression *syntaxCopy();

    void accept(Visitor *v) { v->visit(this); }
};

class DotIdExp : public UnaExp
{
public:
    Identifier *ident;
    bool noderef;       // true if the result of the expression will never be dereferenced
    bool wantsym;       // do not replace Symbol with its initializer during semantic()

    static DotIdExp *create(Loc loc, Expression *e, Identifier *ident);
    void accept(Visitor *v) { v->visit(this); }
};

class DotTemplateExp : public UnaExp
{
public:
    TemplateDeclaration *td;

    void accept(Visitor *v) { v->visit(this); }
};

class DotVarExp : public UnaExp
{
public:
    Declaration *var;
    bool hasOverloads;

    int checkModifiable(Scope *sc, int flag);
    bool isLvalue();
    Expression *toLvalue(Scope *sc, Expression *e);
    Expression *modifiableLvalue(Scope *sc, Expression *e);
    void accept(Visitor *v) { v->visit(this); }
};

class DotTemplateInstanceExp : public UnaExp
{
public:
    TemplateInstance *ti;

    Expression *syntaxCopy();
    bool findTempDecl(Scope *sc);
    void accept(Visitor *v) { v->visit(this); }
};

class DelegateExp : public UnaExp
{
public:
    FuncDeclaration *func;
    bool hasOverloads;
    VarDeclaration *vthis2;  // container for multi-context


    void accept(Visitor *v) { v->visit(this); }
};

class DotTypeExp : public UnaExp
{
public:
    Dsymbol *sym;               // symbol that represents a type

    void accept(Visitor *v) { v->visit(this); }
};

class CallExp : public UnaExp
{
public:
    Expressions *arguments;     // function arguments
    FuncDeclaration *f;         // symbol to call
    bool directcall;            // true if a virtual call is devirtualized
    VarDeclaration *vthis2;     // container for multi-context

    static CallExp *create(Loc loc, Expression *e, Expressions *exps);
    static CallExp *create(Loc loc, Expression *e);
    static CallExp *create(Loc loc, Expression *e, Expression *earg1);
    static CallExp *create(Loc loc, FuncDeclaration *fd, Expression *earg1);

    Expression *syntaxCopy();
    bool isLvalue();
    Expression *toLvalue(Scope *sc, Expression *e);
    Expression *addDtorHook(Scope *sc);

    void accept(Visitor *v) { v->visit(this); }
};

class AddrExp : public UnaExp
{
public:
    void accept(Visitor *v) { v->visit(this); }
};

class PtrExp : public UnaExp
{
public:
    int checkModifiable(Scope *sc, int flag);
    bool isLvalue();
    Expression *toLvalue(Scope *sc, Expression *e);
    Expression *modifiableLvalue(Scope *sc, Expression *e);

    void accept(Visitor *v) { v->visit(this); }
};

class NegExp : public UnaExp
{
public:
    void accept(Visitor *v) { v->visit(this); }
};

class UAddExp : public UnaExp
{
public:
    void accept(Visitor *v) { v->visit(this); }
};

class ComExp : public UnaExp
{
public:
    void accept(Visitor *v) { v->visit(this); }
};

class NotExp : public UnaExp
{
public:
    void accept(Visitor *v) { v->visit(this); }
};

class DeleteExp : public UnaExp
{
public:
    bool isRAII;
    Expression *toBoolean(Scope *sc);
    void accept(Visitor *v) { v->visit(this); }
};

class CastExp : public UnaExp
{
public:
    // Possible to cast to one type while painting to another type
    Type *to;                   // type to cast to
    unsigned char mod;          // MODxxxxx

    Expression *syntaxCopy();

    void accept(Visitor *v) { v->visit(this); }
};

class VectorExp : public UnaExp
{
public:
    TypeVector *to;             // the target vector type before semantic()
    unsigned dim;               // number of elements in the vector
    OwnedBy ownedByCtfe;

    static VectorExp *create(Loc loc, Expression *e, Type *t);
    static void emplace(UnionExp *pue, Loc loc, Expression *e, Type *t);
    Expression *syntaxCopy();
    void accept(Visitor *v) { v->visit(this); }
};

class VectorArrayExp : public UnaExp
{
public:
    bool isLvalue();
    Expression *toLvalue(Scope *sc, Expression *e);
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
    bool arrayop;               // an array operation, rather than a slice

    Expression *syntaxCopy();
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
    void accept(Visitor *v) { v->visit(this); }
};

class IntervalExp : public Expression
{
public:
    Expression *lwr;
    Expression *upr;

    Expression *syntaxCopy();
    void accept(Visitor *v) { v->visit(this); }
};

class DelegatePtrExp : public UnaExp
{
public:
    bool isLvalue();
    Expression *toLvalue(Scope *sc, Expression *e);
    Expression *modifiableLvalue(Scope *sc, Expression *e);
    void accept(Visitor *v) { v->visit(this); }
};

class DelegateFuncptrExp : public UnaExp
{
public:
    bool isLvalue();
    Expression *toLvalue(Scope *sc, Expression *e);
    Expression *modifiableLvalue(Scope *sc, Expression *e);
    void accept(Visitor *v) { v->visit(this); }
};

// e1[a0,a1,a2,a3,...]

class ArrayExp : public UnaExp
{
public:
    Expressions *arguments;             // Array of Expression's
    size_t currentDimension;            // for opDollar
    VarDeclaration *lengthVar;

    Expression *syntaxCopy();
    bool isLvalue();
    Expression *toLvalue(Scope *sc, Expression *e);

    void accept(Visitor *v) { v->visit(this); }
};

/****************************************************************/

class DotExp : public BinExp
{
public:
    void accept(Visitor *v) { v->visit(this); }
};

class CommaExp : public BinExp
{
public:
    bool isGenerated;
    bool allowCommaExp;
    int checkModifiable(Scope *sc, int flag);
    bool isLvalue();
    Expression *toLvalue(Scope *sc, Expression *e);
    Expression *modifiableLvalue(Scope *sc, Expression *e);
    bool isBool(bool result);
    Expression *toBoolean(Scope *sc);
    Expression *addDtorHook(Scope *sc);
    void accept(Visitor *v) { v->visit(this); }
};

class IndexExp : public BinExp
{
public:
    VarDeclaration *lengthVar;
    bool modifiable;
    bool indexIsInBounds;       // true if 0 <= e2 && e2 <= e1.length - 1

    Expression *syntaxCopy();
    int checkModifiable(Scope *sc, int flag);
    bool isLvalue();
    Expression *toLvalue(Scope *sc, Expression *e);
    Expression *modifiableLvalue(Scope *sc, Expression *e);

    void accept(Visitor *v) { v->visit(this); }
};

/* For both i++ and i--
 */
class PostExp : public BinExp
{
public:
    void accept(Visitor *v) { v->visit(this); }
};

/* For both ++i and --i
 */
class PreExp : public UnaExp
{
public:
    void accept(Visitor *v) { v->visit(this); }
};

enum MemorySet
{
    blockAssign     = 1,    // setting the contents of an array
    referenceInit   = 2     // setting the reference of STCref variable
};

class AssignExp : public BinExp
{
public:
    int memset;         // combination of MemorySet flags

    bool isLvalue();
    Expression *toLvalue(Scope *sc, Expression *ex);
    Expression *toBoolean(Scope *sc);

    void accept(Visitor *v) { v->visit(this); }
};

class ConstructExp : public AssignExp
{
public:
    void accept(Visitor *v) { v->visit(this); }
};

class BlitExp : public AssignExp
{
public:
    void accept(Visitor *v) { v->visit(this); }
};

class AddAssignExp : public BinAssignExp
{
public:
    void accept(Visitor *v) { v->visit(this); }
};

class MinAssignExp : public BinAssignExp
{
public:
    void accept(Visitor *v) { v->visit(this); }
};

class MulAssignExp : public BinAssignExp
{
public:
    void accept(Visitor *v) { v->visit(this); }
};

class DivAssignExp : public BinAssignExp
{
public:
    void accept(Visitor *v) { v->visit(this); }
};

class ModAssignExp : public BinAssignExp
{
public:
    void accept(Visitor *v) { v->visit(this); }
};

class AndAssignExp : public BinAssignExp
{
public:
    void accept(Visitor *v) { v->visit(this); }
};

class OrAssignExp : public BinAssignExp
{
public:
    void accept(Visitor *v) { v->visit(this); }
};

class XorAssignExp : public BinAssignExp
{
public:
    void accept(Visitor *v) { v->visit(this); }
};

class PowAssignExp : public BinAssignExp
{
public:
    void accept(Visitor *v) { v->visit(this); }
};

class ShlAssignExp : public BinAssignExp
{
public:
    void accept(Visitor *v) { v->visit(this); }
};

class ShrAssignExp : public BinAssignExp
{
public:
    void accept(Visitor *v) { v->visit(this); }
};

class UshrAssignExp : public BinAssignExp
{
public:
    void accept(Visitor *v) { v->visit(this); }
};

class CatAssignExp : public BinAssignExp
{
public:
    void accept(Visitor *v) { v->visit(this); }
};

class AddExp : public BinExp
{
public:
    void accept(Visitor *v) { v->visit(this); }
};

class MinExp : public BinExp
{
public:
    void accept(Visitor *v) { v->visit(this); }
};

class CatExp : public BinExp
{
public:
    void accept(Visitor *v) { v->visit(this); }
};

class MulExp : public BinExp
{
public:
    void accept(Visitor *v) { v->visit(this); }
};

class DivExp : public BinExp
{
public:
    void accept(Visitor *v) { v->visit(this); }
};

class ModExp : public BinExp
{
public:
    void accept(Visitor *v) { v->visit(this); }
};

class PowExp : public BinExp
{
public:
    void accept(Visitor *v) { v->visit(this); }
};

class ShlExp : public BinExp
{
public:
    void accept(Visitor *v) { v->visit(this); }
};

class ShrExp : public BinExp
{
public:
    void accept(Visitor *v) { v->visit(this); }
};

class UshrExp : public BinExp
{
public:
    void accept(Visitor *v) { v->visit(this); }
};

class AndExp : public BinExp
{
public:
    void accept(Visitor *v) { v->visit(this); }
};

class OrExp : public BinExp
{
public:
    void accept(Visitor *v) { v->visit(this); }
};

class XorExp : public BinExp
{
public:
    void accept(Visitor *v) { v->visit(this); }
};

class LogicalExp : public BinExp
{
public:
    Expression *toBoolean(Scope *sc);
    void accept(Visitor *v) { v->visit(this); }
};

class CmpExp : public BinExp
{
public:
    void accept(Visitor *v) { v->visit(this); }
};

class InExp : public BinExp
{
public:
    void accept(Visitor *v) { v->visit(this); }
};

class RemoveExp : public BinExp
{
public:
    void accept(Visitor *v) { v->visit(this); }
};

// == and !=

class EqualExp : public BinExp
{
public:
    void accept(Visitor *v) { v->visit(this); }
};

// is and !is

class IdentityExp : public BinExp
{
public:
    void accept(Visitor *v) { v->visit(this); }
};

/****************************************************************/

class CondExp : public BinExp
{
public:
    Expression *econd;

    Expression *syntaxCopy();
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

    void accept(Visitor *v) { v->visit(this); }
};

class FileInitExp : public DefaultInitExp
{
public:
    Expression *resolveLoc(const Loc &loc, Scope *sc);
    void accept(Visitor *v) { v->visit(this); }
};

class LineInitExp : public DefaultInitExp
{
public:
    Expression *resolveLoc(const Loc &loc, Scope *sc);
    void accept(Visitor *v) { v->visit(this); }
};

class ModuleInitExp : public DefaultInitExp
{
public:
    Expression *resolveLoc(const Loc &loc, Scope *sc);
    void accept(Visitor *v) { v->visit(this); }
};

class FuncInitExp : public DefaultInitExp
{
public:
    Expression *resolveLoc(const Loc &loc, Scope *sc);
    void accept(Visitor *v) { v->visit(this); }
};

class PrettyFuncInitExp : public DefaultInitExp
{
public:
    Expression *resolveLoc(const Loc &loc, Scope *sc);
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
    // Ensure that the union is suitably aligned.
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((aligned(8)))
#elif defined(_MSC_VER)
    __declspec(align(8))
#elif defined(__DMC__)
    #pragma pack(8)
#endif
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
        char vectorexp [sizeof(VectorExp)];
    } u;
#if defined(__DMC__)
    #pragma pack()
#endif
};

/****************************************************************/

class ObjcClassReferenceExp : public Expression
{
public:
    ClassDeclaration* classDeclaration;

    void accept(Visitor *v) { v->visit(this); }
};
