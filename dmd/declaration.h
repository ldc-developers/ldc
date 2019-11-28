
/* Compiler implementation of the D programming language
 * Copyright (C) 1999-2019 by The D Language Foundation, All Rights Reserved
 * written by Walter Bright
 * http://www.digitalmars.com
 * Distributed under the Boost Software License, Version 1.0.
 * http://www.boost.org/LICENSE_1_0.txt
 * https://github.com/dlang/dmd/blob/master/src/dmd/declaration.h
 */

#pragma once

#include "dsymbol.h"
#include "mtype.h"
#include "tokens.h"

class Expression;
class Statement;
class LabelDsymbol;
class Initializer;
class ForeachStatement;
struct Ensure
{
    Identifier *id;
    Statement *ensure;
};
class FuncDeclaration;
class StructDeclaration;
struct ObjcSelector;
struct IntRange;

#define STCundefined    0LL
#define STCstatic       1LL
#define STCextern       2LL
#define STCconst        4LL
#define STCfinal        8LL
#define STCabstract     0x10LL
#define STCparameter    0x20LL
#define STCfield        0x40LL
#define STCoverride     0x80LL
#define STCauto         0x100LL
#define STCsynchronized 0x200LL
#define STCdeprecated   0x400LL
#define STCin           0x800LL         // in parameter
#define STCout          0x1000LL        // out parameter
#define STClazy         0x2000LL        // lazy parameter
#define STCforeach      0x4000LL        // variable for foreach loop
#define STCvariadic     0x10000LL       // the 'variadic' parameter in: T foo(T a, U b, V variadic...)
#define STCctorinit     0x20000LL       // can only be set inside constructor
#define STCtemplateparameter  0x40000LL // template parameter
#define STCscope        0x80000LL
#define STCimmutable    0x100000LL
#define STCref          0x200000LL
#define STCinit         0x400000LL      // has explicit initializer
#define STCmanifest     0x800000LL      // manifest constant
#define STCnodtor       0x1000000LL     // don't run destructor
#define STCnothrow      0x2000000LL     // never throws exceptions
#define STCpure         0x4000000LL     // pure function
#define STCtls          0x8000000LL     // thread local
#define STCalias        0x10000000LL    // alias parameter
#define STCshared       0x20000000LL    // accessible from multiple threads
// accessible from multiple threads
// but not typed as "shared"
#define STCgshared      0x40000000LL
#define STCwild         0x80000000LL    // for "wild" type constructor
#define STC_TYPECTOR    (STCconst | STCimmutable | STCshared | STCwild)
#define STC_FUNCATTR    (STCref | STCnothrow | STCnogc | STCpure | STCproperty | STCsafe | STCtrusted | STCsystem)

#define STCproperty      0x100000000LL
#define STCsafe          0x200000000LL
#define STCtrusted       0x400000000LL
#define STCsystem        0x800000000LL
#define STCctfe          0x1000000000LL  // can be used in CTFE, even if it is static
#define STCdisable       0x2000000000LL  // for functions that are not callable
#define STCresult        0x4000000000LL  // for result variables passed to out contracts
#define STCnodefaultctor 0x8000000000LL  // must be set inside constructor
#define STCtemp          0x10000000000LL // temporary variable
#define STCrvalue        0x20000000000LL // force rvalue for variables
#define STCnogc          0x40000000000LL // @nogc
#define STCvolatile      0x80000000000LL // destined for volatile in the back end
#define STCreturn        0x100000000000LL // 'return ref' or 'return scope' for function parameters
#define STCautoref       0x200000000000LL // Mark for the already deduced 'auto ref' parameter
#define STCinference     0x400000000000LL // do attribute inference
#define STCexptemp       0x800000000000LL // temporary variable that has lifetime restricted to an expression
#define STCmaybescope    0x1000000000000LL // parameter might be 'scope'
#define STCscopeinferred 0x2000000000000LL // 'scope' has been inferred and should not be part of mangling
#define STCfuture        0x4000000000000LL // introducing new base class function
#define STClocal         0x8000000000000LL // do not forward (see dmd.dsymbol.ForwardingScopeDsymbol).
#define STCreturninferred 0x10000000000000LL   // 'return' has been inferred and should not be part of mangling

void ObjectNotFound(Identifier *id);

/**************************************************************/

class Declaration : public Dsymbol
{
public:
    Type *type;
    Type *originalType;         // before semantic analysis
    StorageClass storage_class;
    Prot protection;
    LINK linkage;
    int inuse;                  // used to detect cycles
    DString mangleOverride;     // overridden symbol with pragma(mangle, "...")

    const char *kind() const;
    d_uns64 size(const Loc &loc);

    Dsymbol *search(const Loc &loc, Identifier *ident, int flags = SearchLocalsOnly);

    bool isStatic() const { return (storage_class & STCstatic) != 0; }
    virtual bool isDelete();
    virtual bool isDataseg();
    virtual bool isThreadlocal();
    virtual bool isCodeseg() const;
    bool isCtorinit() const     { return (storage_class & STCctorinit) != 0; }
    bool isFinal() const        { return (storage_class & STCfinal) != 0; }
    virtual bool isAbstract()   { return (storage_class & STCabstract) != 0; }
    bool isConst() const        { return (storage_class & STCconst) != 0; }
    bool isImmutable() const    { return (storage_class & STCimmutable) != 0; }
    bool isWild() const         { return (storage_class & STCwild) != 0; }
    bool isAuto() const         { return (storage_class & STCauto) != 0; }
    bool isScope() const        { return (storage_class & STCscope) != 0; }
    bool isSynchronized() const { return (storage_class & STCsynchronized) != 0; }
    bool isParameter() const    { return (storage_class & STCparameter) != 0; }
    bool isDeprecated() const   { return (storage_class & STCdeprecated) != 0; }
    bool isOverride() const     { return (storage_class & STCoverride) != 0; }
    bool isResult() const       { return (storage_class & STCresult) != 0; }
    bool isField() const        { return (storage_class & STCfield) != 0; }

    bool isIn()  const  { return (storage_class & STCin) != 0; }
    bool isOut() const  { return (storage_class & STCout) != 0; }
    bool isRef() const  { return (storage_class & STCref) != 0; }

    bool isFuture() const { return (storage_class & STCfuture) != 0; }

    Prot prot();

    Declaration *isDeclaration() { return this; }
    void accept(Visitor *v) { v->visit(this); }
};

/**************************************************************/

class TupleDeclaration : public Declaration
{
public:
    Objects *objects;
    bool isexp;                 // true: expression tuple

    TypeTuple *tupletype;       // !=NULL if this is a type tuple

    Dsymbol *syntaxCopy(Dsymbol *);
    const char *kind() const;
    Type *getType();
    Dsymbol *toAlias2();
    bool needThis();

    TupleDeclaration *isTupleDeclaration() { return this; }
    void accept(Visitor *v) { v->visit(this); }
};

/**************************************************************/

class AliasDeclaration : public Declaration
{
public:
    Dsymbol *aliassym;
    Dsymbol *overnext;          // next in overload list
    Dsymbol *_import;           // !=NULL if unresolved internal alias for selective import
    bool wasTemplateParameter; /// indicates wether the alias was created to make a template parameter visible in the scope, i.e as a member.

    static AliasDeclaration *create(Loc loc, Identifier *id, Type *type);
    Dsymbol *syntaxCopy(Dsymbol *);
    bool overloadInsert(Dsymbol *s);
    const char *kind() const;
    Type *getType();
    Dsymbol *toAlias();
    Dsymbol *toAlias2();
    bool isOverloadable() const;

    AliasDeclaration *isAliasDeclaration() { return this; }
    void accept(Visitor *v) { v->visit(this); }
};

/**************************************************************/

class OverDeclaration : public Declaration
{
public:
    Dsymbol *overnext;          // next in overload list
    Dsymbol *aliassym;
    bool hasOverloads;

    const char *kind() const;
    bool equals(const RootObject *o) const;
    bool overloadInsert(Dsymbol *s);

    Dsymbol *toAlias();
    Dsymbol *isUnique();
    bool isOverloadable() const;

    OverDeclaration *isOverDeclaration() { return this; }
    void accept(Visitor *v) { v->visit(this); }
};

/**************************************************************/

class VarDeclaration : public Declaration
{
public:
    Initializer *_init;
    unsigned offset;
    unsigned sequenceNumber;     // order the variables are declared
    FuncDeclarations nestedrefs; // referenced by these lexically nested functions
    structalign_t alignment;
    bool isargptr;              // if parameter that _argptr points to
    bool ctorinit;              // it has been initialized in a ctor
    bool iscatchvar;            // this is the exception object variable in catch() clause
    bool onstack;               // it is a class that was allocated on the stack
#if IN_LLVM
    bool onstackWithDtor;       // it is a class that was allocated on the stack and needs destruction
#endif
    bool mynew;                 // it is a class new'd with custom operator new
    int canassign;              // it can be assigned to
    bool overlapped;            // if it is a field and has overlapping
    bool overlapUnsafe;         // if it is an overlapping field and the overlaps are unsafe
    bool doNotInferScope;       // do not infer 'scope' for this variable
    bool doNotInferReturn;      // do not infer 'return' for this variable
    unsigned char isdataseg;    // private data for isDataseg
    Dsymbol *aliassym;          // if redone as alias to another symbol
    VarDeclaration *lastVar;    // Linked list of variables for goto-skips-init detection
    unsigned endlinnum;         // line number of end of scope that this var lives in

    // When interpreting, these point to the value (NULL if value not determinable)
    // The index of this variable on the CTFE stack, ~0u if not allocated
    unsigned ctfeAdrOnStack;
    Expression *edtor;          // if !=NULL, does the destruction of the variable
    IntRange *range;            // if !NULL, the variable is known to be within the range

    VarDeclarations *maybes;    // STCmaybescope variables that are assigned to this STCmaybescope variable

private:
    bool _isAnonymous;

public:
    static VarDeclaration *create(const Loc &loc, Type *t, Identifier *id, Initializer *init, StorageClass storage_class = STCundefined);
    Dsymbol *syntaxCopy(Dsymbol *);
    void setFieldOffset(AggregateDeclaration *ad, unsigned *poffset, bool isunion);
    const char *kind() const;
    AggregateDeclaration *isThis();
    bool needThis();
    bool isAnonymous();
    bool isExport() const;
    bool isImportedSymbol() const;
    bool isDataseg();
    bool isThreadlocal();
    bool isCTFE();
    bool isOverlappedWith(VarDeclaration *v);
    bool hasPointers();
    bool canTakeAddressOf();
    bool needsScopeDtor();
    bool enclosesLifetimeOf(VarDeclaration *v) const;
    void checkCtorConstInit();
    Dsymbol *toAlias();
    // Eliminate need for dynamic_cast
    VarDeclaration *isVarDeclaration() { return (VarDeclaration *)this; }
    void accept(Visitor *v) { v->visit(this); }
};

/**************************************************************/

// This is a shell around a back end symbol

class SymbolDeclaration : public Declaration
{
public:
    StructDeclaration *dsym;

    // Eliminate need for dynamic_cast
    SymbolDeclaration *isSymbolDeclaration() { return (SymbolDeclaration *)this; }
    void accept(Visitor *v) { v->visit(this); }
};

class TypeInfoDeclaration : public VarDeclaration
{
public:
    Type *tinfo;

    static TypeInfoDeclaration *create(Type *tinfo);
    Dsymbol *syntaxCopy(Dsymbol *);
    const char *toChars() const;

    TypeInfoDeclaration *isTypeInfoDeclaration() { return this; }
    void accept(Visitor *v) { v->visit(this); }
};

class TypeInfoStructDeclaration : public TypeInfoDeclaration
{
public:
    static TypeInfoStructDeclaration *create(Type *tinfo);

    void accept(Visitor *v) { v->visit(this); }
};

class TypeInfoClassDeclaration : public TypeInfoDeclaration
{
public:
    static TypeInfoClassDeclaration *create(Type *tinfo);

    void accept(Visitor *v) { v->visit(this); }
};

class TypeInfoInterfaceDeclaration : public TypeInfoDeclaration
{
public:
    static TypeInfoInterfaceDeclaration *create(Type *tinfo);

    void accept(Visitor *v) { v->visit(this); }
};

class TypeInfoPointerDeclaration : public TypeInfoDeclaration
{
public:
    static TypeInfoPointerDeclaration *create(Type *tinfo);

    void accept(Visitor *v) { v->visit(this); }
};

class TypeInfoArrayDeclaration : public TypeInfoDeclaration
{
public:
    static TypeInfoArrayDeclaration *create(Type *tinfo);

    void accept(Visitor *v) { v->visit(this); }
};

class TypeInfoStaticArrayDeclaration : public TypeInfoDeclaration
{
public:
    static TypeInfoStaticArrayDeclaration *create(Type *tinfo);

    void accept(Visitor *v) { v->visit(this); }
};

class TypeInfoAssociativeArrayDeclaration : public TypeInfoDeclaration
{
public:
    static TypeInfoAssociativeArrayDeclaration *create(Type *tinfo);

    void accept(Visitor *v) { v->visit(this); }
};

class TypeInfoEnumDeclaration : public TypeInfoDeclaration
{
public:
    static TypeInfoEnumDeclaration *create(Type *tinfo);

    void accept(Visitor *v) { v->visit(this); }
};

class TypeInfoFunctionDeclaration : public TypeInfoDeclaration
{
public:
    static TypeInfoFunctionDeclaration *create(Type *tinfo);

    void accept(Visitor *v) { v->visit(this); }
};

class TypeInfoDelegateDeclaration : public TypeInfoDeclaration
{
public:
    static TypeInfoDelegateDeclaration *create(Type *tinfo);

    void accept(Visitor *v) { v->visit(this); }
};

class TypeInfoTupleDeclaration : public TypeInfoDeclaration
{
public:
    static TypeInfoTupleDeclaration *create(Type *tinfo);

    void accept(Visitor *v) { v->visit(this); }
};

class TypeInfoConstDeclaration : public TypeInfoDeclaration
{
public:
    static TypeInfoConstDeclaration *create(Type *tinfo);

    void accept(Visitor *v) { v->visit(this); }
};

class TypeInfoInvariantDeclaration : public TypeInfoDeclaration
{
public:
    static TypeInfoInvariantDeclaration *create(Type *tinfo);

    void accept(Visitor *v) { v->visit(this); }
};

class TypeInfoSharedDeclaration : public TypeInfoDeclaration
{
public:
    static TypeInfoSharedDeclaration *create(Type *tinfo);

    void accept(Visitor *v) { v->visit(this); }
};

class TypeInfoWildDeclaration : public TypeInfoDeclaration
{
public:
    static TypeInfoWildDeclaration *create(Type *tinfo);

    void accept(Visitor *v) { v->visit(this); }
};

class TypeInfoVectorDeclaration : public TypeInfoDeclaration
{
public:
    static TypeInfoVectorDeclaration *create(Type *tinfo);

    void accept(Visitor *v) { v->visit(this); }
};

/**************************************************************/

class ThisDeclaration : public VarDeclaration
{
public:
    Dsymbol *syntaxCopy(Dsymbol *);
    ThisDeclaration *isThisDeclaration() { return this; }
    void accept(Visitor *v) { v->visit(this); }
};

enum ILS
{
    ILSuninitialized,   // not computed yet
    ILSno,              // cannot inline
    ILSyes              // can inline
};

/**************************************************************/

enum BUILTIN
{
    BUILTINunknown = -1,        // not known if this is a builtin
    BUILTINno,                  // this is not a builtin
    BUILTINyes                  // this is a builtin
};

Expression *eval_builtin(Loc loc, FuncDeclaration *fd, Expressions *arguments);
BUILTIN isBuiltin(FuncDeclaration *fd);
void builtin_init();

class FuncDeclaration : public Declaration
{
public:
    struct HiddenParameters
    {
        VarDeclaration *this_;
        bool isThis2;
        VarDeclaration *selector;
    };

    Statements *frequires;              // in contracts
    Ensures *fensures;                  // out contracts
    Statement *frequire;                // lowered in contract
    Statement *fensure;                 // lowered out contract
    Statement *fbody;

    FuncDeclarations foverrides;        // functions this function overrides
    FuncDeclaration *fdrequire;         // function that does the in contract
    FuncDeclaration *fdensure;          // function that does the out contract

    Expressions *fdrequireParams;       // argument list for __require
    Expressions *fdensureParams;        // argument list for __ensure

    const char *mangleString;           // mangled symbol created from mangleExact()

#if IN_LLVM
    const char *intrinsicName;
    uint32_t priority;

    // true if overridden with the pragma(LDC_allow_inline); statement
    bool allowInlining;

    // true if set with the pragma(LDC_never_inline); statement
    bool neverInline;

    // Whether to emit instrumentation code if -fprofile-instr-generate is specified,
    // the value is set with pragma(LDC_profile_instr, true|false)
    bool emitInstrumentation;
#endif

    VarDeclaration *vresult;            // result variable for out contracts
    LabelDsymbol *returnLabel;          // where the return goes

    // used to prevent symbols in different
    // scopes from having the same name
    DsymbolTable *localsymtab;
    VarDeclaration *vthis;              // 'this' parameter (member and nested)
    bool isThis2;                       // has a dual-context 'this' parameter
    VarDeclaration *v_arguments;        // '_arguments' parameter
    ObjcSelector *selector;             // Objective-C method selector (member function only)
    VarDeclaration *selectorParameter;  // Objective-C implicit selector parameter

    VarDeclaration *v_argptr;           // '_argptr' variable
    VarDeclarations *parameters;        // Array of VarDeclaration's for parameters
    DsymbolTable *labtab;               // statement label symbol table
    Dsymbol *overnext;                  // next in overload list
    FuncDeclaration *overnext0;         // next in overload list (only used during IFTI)
    Loc endloc;                         // location of closing curly bracket
    int vtblIndex;                      // for member functions, index into vtbl[]
    bool naked;                         // true if naked
    bool generated;                     // true if function was generated by the compiler rather than
                                        // supplied by the user
    unsigned char isCrtCtorDtor;        // has attribute pragma(crt_constructor(1)/crt_destructor(2))
                                        // not set before the glue layer
    ILS inlineStatusStmt;
    ILS inlineStatusExp;
    PINLINE inlining;

    int inlineNest;                     // !=0 if nested inline
    bool isArrayOp;                     // true if array operation
    bool eh_none;                       /// true if no exception unwinding is needed

    // true if errors in semantic3 this function's frame ptr
    bool semantic3Errors;
    ForeachStatement *fes;              // if foreach body, this is the foreach
    BaseClass* interfaceVirtual;        // if virtual, but only appears in interface vtbl[]
    bool introducing;                   // true if 'introducing' function
    // if !=NULL, then this is the type
    // of the 'introducing' function
    // this one is overriding
    Type *tintro;
    bool inferRetType;                  // true if return type is to be inferred
    StorageClass storage_class2;        // storage class for template onemember's

    // Things that should really go into Scope

    // 1 if there's a return exp; statement
    // 2 if there's a throw statement
    // 4 if there's an assert(0)
    // 8 if there's inline asm
    // 16 if there are multiple return statements
    int hasReturnExp;

    // Support for NRVO (named return value optimization)
    bool nrvo_can;                      // true means we can do it
    VarDeclaration *nrvo_var;           // variable to replace with shidden
    Symbol *shidden;                    // hidden pointer passed to function

    ReturnStatements *returns;

    GotoStatements *gotos;              // Gotos with forward references

    // set if this is a known, builtin function we can evaluate at compile time
    BUILTIN builtin;

    // set if someone took the address of this function
    int tookAddressOf;
    bool requiresClosure;               // this function needs a closure

    // local variables in this function which are referenced by nested functions
    VarDeclarations closureVars;

    /** Outer variables which are referenced by this nested function
     * (the inverse of closureVars)
     */
    VarDeclarations outerVars;

    // Sibling nested functions which called this one
    FuncDeclarations siblingCallers;

    FuncDeclarations *inlinedNestedCallees;

    unsigned flags;                     // FUNCFLAGxxxxx

    static FuncDeclaration *create(const Loc &loc, const Loc &endloc, Identifier *id, StorageClass storage_class, Type *type);
    Dsymbol *syntaxCopy(Dsymbol *);
    bool functionSemantic();
    bool functionSemantic3();
    bool equals(const RootObject *o) const;

    int overrides(FuncDeclaration *fd);
    int findVtblIndex(Dsymbols *vtbl, int dim, bool fix17349 = true);
    BaseClass *overrideInterface();
    bool overloadInsert(Dsymbol *s);
    bool inUnittest();
    MATCH leastAsSpecialized(FuncDeclaration *g);
    LabelDsymbol *searchLabel(Identifier *ident);
    int getLevel(FuncDeclaration *fd, int intypeof); // lexical nesting level difference
    int getLevelAndCheck(const Loc &loc, Scope *sc, FuncDeclaration *fd);
    const char *toPrettyChars(bool QualifyTypes = false);
    const char *toFullSignature();  // for diagnostics, e.g. 'int foo(int x, int y) pure'
    bool isMain() const;
    bool isCMain() const;
    bool isWinMain() const;
    bool isDllMain() const;
    bool isExport() const;
    bool isImportedSymbol() const;
    bool isCodeseg() const;
    bool isOverloadable() const;
    bool isAbstract();
    PURE isPure();
    PURE isPureBypassingInference();
    bool isSafe();
    bool isSafeBypassingInference();
    bool isTrusted();

    bool isNogc();
    bool isNogcBypassingInference();

    virtual bool isNested() const;
    AggregateDeclaration *isThis();
    bool needThis();
    bool isVirtualMethod();
    virtual bool isVirtual() const;
    bool isFinalFunc() const;
    virtual bool addPreInvariant();
    virtual bool addPostInvariant();
    const char *kind() const;
    bool isUnique();
    bool needsClosure();
    bool hasNestedFrameRefs();
    ParameterList getParameterList();

    static FuncDeclaration *genCfunc(Parameters *args, Type *treturn, const char *name, StorageClass stc=0);
    static FuncDeclaration *genCfunc(Parameters *args, Type *treturn, Identifier *id, StorageClass stc=0);

    FuncDeclaration *isFuncDeclaration() { return this; }

    virtual FuncDeclaration *toAliasFunc() { return this; }
    void accept(Visitor *v) { v->visit(this); }
};

class FuncAliasDeclaration : public FuncDeclaration
{
public:
    FuncDeclaration *funcalias;
    bool hasOverloads;

    FuncAliasDeclaration *isFuncAliasDeclaration() { return this; }
    const char *kind() const;

    FuncDeclaration *toAliasFunc();
    void accept(Visitor *v) { v->visit(this); }
};

class FuncLiteralDeclaration : public FuncDeclaration
{
public:
    TOK tok;                       // TOKfunction or TOKdelegate
    Type *treq;                         // target of return type inference

    // backend
    bool deferToObj;

    Dsymbol *syntaxCopy(Dsymbol *);
    bool isNested() const;
    AggregateDeclaration *isThis();
    bool isVirtual() const;
    bool addPreInvariant();
    bool addPostInvariant();

    void modifyReturns(Scope *sc, Type *tret);

    FuncLiteralDeclaration *isFuncLiteralDeclaration() { return this; }
    const char *kind() const;
    const char *toPrettyChars(bool QualifyTypes = false);
    void accept(Visitor *v) { v->visit(this); }
};

class CtorDeclaration : public FuncDeclaration
{
public:
    bool isCpCtor;
    Dsymbol *syntaxCopy(Dsymbol *);
    const char *kind() const;
    const char *toChars() const;
    bool isVirtual() const;
    bool addPreInvariant();
    bool addPostInvariant();

    CtorDeclaration *isCtorDeclaration() { return this; }
    void accept(Visitor *v) { v->visit(this); }
};

class PostBlitDeclaration : public FuncDeclaration
{
public:
    Dsymbol *syntaxCopy(Dsymbol *);
    bool isVirtual() const;
    bool addPreInvariant();
    bool addPostInvariant();
    bool overloadInsert(Dsymbol *s);

    PostBlitDeclaration *isPostBlitDeclaration() { return this; }
    void accept(Visitor *v) { v->visit(this); }
};

class DtorDeclaration : public FuncDeclaration
{
public:
    Dsymbol *syntaxCopy(Dsymbol *);
    const char *kind() const;
    const char *toChars() const;
    bool isVirtual() const;
    bool addPreInvariant();
    bool addPostInvariant();
    bool overloadInsert(Dsymbol *s);

    DtorDeclaration *isDtorDeclaration() { return this; }
    void accept(Visitor *v) { v->visit(this); }
};

class StaticCtorDeclaration : public FuncDeclaration
{
public:
    Dsymbol *syntaxCopy(Dsymbol *);
    AggregateDeclaration *isThis();
    bool isVirtual() const;
    bool addPreInvariant();
    bool addPostInvariant();
    bool hasStaticCtorOrDtor();

    StaticCtorDeclaration *isStaticCtorDeclaration() { return this; }
    void accept(Visitor *v) { v->visit(this); }
};

class SharedStaticCtorDeclaration : public StaticCtorDeclaration
{
public:
    Dsymbol *syntaxCopy(Dsymbol *);

    SharedStaticCtorDeclaration *isSharedStaticCtorDeclaration() { return this; }
    void accept(Visitor *v) { v->visit(this); }
};

class StaticDtorDeclaration : public FuncDeclaration
{
public:
    VarDeclaration *vgate;      // 'gate' variable

    Dsymbol *syntaxCopy(Dsymbol *);
    AggregateDeclaration *isThis();
    bool isVirtual() const;
    bool hasStaticCtorOrDtor();
    bool addPreInvariant();
    bool addPostInvariant();

    StaticDtorDeclaration *isStaticDtorDeclaration() { return this; }
    void accept(Visitor *v) { v->visit(this); }
};

class SharedStaticDtorDeclaration : public StaticDtorDeclaration
{
public:
    Dsymbol *syntaxCopy(Dsymbol *);

    SharedStaticDtorDeclaration *isSharedStaticDtorDeclaration() { return this; }
    void accept(Visitor *v) { v->visit(this); }
};

class InvariantDeclaration : public FuncDeclaration
{
public:
    Dsymbol *syntaxCopy(Dsymbol *);
    bool isVirtual() const;
    bool addPreInvariant();
    bool addPostInvariant();

    InvariantDeclaration *isInvariantDeclaration() { return this; }
    void accept(Visitor *v) { v->visit(this); }
};

class UnitTestDeclaration : public FuncDeclaration
{
public:
    char *codedoc; /** For documented unittest. */

    // toObjFile() these nested functions after this one
    FuncDeclarations deferredNested;

    Dsymbol *syntaxCopy(Dsymbol *);
    AggregateDeclaration *isThis();
    bool isVirtual() const;
    bool addPreInvariant();
    bool addPostInvariant();

    UnitTestDeclaration *isUnitTestDeclaration() { return this; }
    void accept(Visitor *v) { v->visit(this); }
};

class NewDeclaration : public FuncDeclaration
{
public:
    Parameters *parameters;
    VarArg varargs;

    Dsymbol *syntaxCopy(Dsymbol *);
    const char *kind() const;
    bool isVirtual() const;
    bool addPreInvariant();
    bool addPostInvariant();

    NewDeclaration *isNewDeclaration() { return this; }
    void accept(Visitor *v) { v->visit(this); }
};


class DeleteDeclaration : public FuncDeclaration
{
public:
    Parameters *parameters;

    Dsymbol *syntaxCopy(Dsymbol *);
    const char *kind() const;
    bool isDelete();
    bool isVirtual() const;
    bool addPreInvariant();
    bool addPostInvariant();

    DeleteDeclaration *isDeleteDeclaration() { return this; }
    void accept(Visitor *v) { v->visit(this); }
};
