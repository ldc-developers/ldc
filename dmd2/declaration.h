
// Compiler implementation of the D programming language
// Copyright (c) 1999-2011 by Digital Mars
// All Rights Reserved
// written by Walter Bright
// http://www.digitalmars.com
// License for redistribution is by either the Artistic License
// in artistic.txt, or the GNU General Public License in gnu.txt.
// See the included readme.txt for details.

#ifndef DMD_DECLARATION_H
#define DMD_DECLARATION_H

#ifdef __DMC__
#pragma once
#endif /* __DMC__ */

#if IN_LLVM
#include <set>
#include <map>
#include <string>
#if LDC_LLVM_VER >= 305
#include "llvm/IR/DebugInfo.h"
#elif LDC_LLVM_VER >= 302
#include "llvm/DebugInfo.h"
#else
#include "llvm/Analysis/DebugInfo.h"
#endif
#endif

#include "dsymbol.h"
#include "lexer.h"
#include "mtype.h"

struct Expression;
struct Statement;
struct LabelDsymbol;
#if IN_LLVM
struct LabelStatement;
#endif
struct Initializer;
struct Module;
struct InlineScanState;
struct ForeachStatement;
struct FuncDeclaration;
struct ExpInitializer;
struct StructDeclaration;
struct TupleType;
struct InterState;
struct IRState;
#if IN_LLVM
struct AnonDeclaration;
#endif

enum PROT;
enum LINK;
enum TOK;
enum MATCH;
enum PURE;

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
#define STCcomdat       0x8000LL        // should go into COMDAT record
#define STCvariadic     0x10000LL       // variadic function argument
#define STCctorinit     0x20000LL       // can only be set inside constructor
#define STCtemplateparameter  0x40000LL // template parameter
#define STCscope        0x80000LL       // template parameter
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
#define STCgshared      0x40000000LL    // accessible from multiple threads
                                        // but not typed as "shared"
#define STCwild         0x80000000LL    // for "wild" type constructor
#define STC_TYPECTOR    (STCconst | STCimmutable | STCshared | STCwild)
#define STC_FUNCATTR    (STCref | STCnothrow | STCpure | STCproperty | STCsafe | STCtrusted | STCsystem)

#define STCproperty     0x100000000LL
#define STCsafe         0x200000000LL
#define STCtrusted      0x400000000LL
#define STCsystem       0x800000000LL
#define STCctfe         0x1000000000LL  // can be used in CTFE, even if it is static
#define STCdisable      0x2000000000LL  // for functions that are not callable
#define STCresult       0x4000000000LL  // for result variables passed to out contracts
#define STCnodefaultctor 0x8000000000LL  // must be set inside constructor
#define STCtemp         0x10000000000LL  // temporary variable introduced by inlining
                                         // and used only in backend process, so it's rvalue

#define STCStorageClass (STCauto | STCscope | STCstatic | STCextern | STCconst | STCfinal | \
        STCabstract | STCsynchronized | STCdeprecated | STCoverride | STClazy | STCalias | \
        STCout | STCin | \
        STCmanifest | STCimmutable | STCshared | STCnothrow | STCpure | STCref | STCtls | \
        STCgshared | STCproperty | STCsafe | STCtrusted | STCsystem | STCdisable)

struct Match
{
    int count;                  // number of matches found
    MATCH last;                 // match level of lastf
    FuncDeclaration *lastf;     // last matching function we found
    FuncDeclaration *nextf;     // current matching function
    FuncDeclaration *anyf;      // pick a func, any func, to use for error recovery
};

void overloadResolveX(Match *m, FuncDeclaration *f,
        Type *tthis, Expressions *arguments);
int overloadApply(FuncDeclaration *fstart,
        int (*fp)(void *, FuncDeclaration *),
        void *param);

void ObjectNotFound(Identifier *id);

enum Semantic
{
    SemanticStart,      // semantic has not been run
    SemanticIn,         // semantic() is in progress
    SemanticDone,       // semantic() has been run
    Semantic2Done,      // semantic2() has been run
};

/**************************************************************/

struct Declaration : Dsymbol
{
    Type *type;
    Type *originalType;         // before semantic analysis
    StorageClass storage_class;
    enum PROT protection;
    enum LINK linkage;
    int inuse;                  // used to detect cycles
    const char *mangleOverride;      // overridden symbol with pragma(mangle, "...")
    enum Semantic sem;

    Declaration(Identifier *id);
    void semantic(Scope *sc);
    const char *kind();
    unsigned size(Loc loc);
    int checkModify(Loc loc, Scope *sc, Type *t, Expression *e1, int flag);

    Dsymbol *search(Loc loc, Identifier *ident, int flags);

    void emitComment(Scope *sc);
    void toJson(JsonOut *json);
    virtual void jsonProperties(JsonOut *json);
    void toDocBuffer(OutBuffer *buf, Scope *sc);

    const char *mangle(bool isv = false);
    bool isStatic() { return (storage_class & STCstatic) != 0; }
    virtual int isDelete();
    virtual int isDataseg();
    virtual int isThreadlocal();
    virtual int isCodeseg();
    bool isCtorinit()     { return (storage_class & STCctorinit) != 0; }
    bool isFinal()        { return (storage_class & STCfinal) != 0; }
    bool isAbstract()     { return (storage_class & STCabstract) != 0; }
    bool isConst()        { return (storage_class & STCconst) != 0; }
    bool isImmutable()    { return (storage_class & STCimmutable) != 0; }
    bool isWild()         { return (storage_class & STCwild) != 0; }
    bool isAuto()         { return (storage_class & STCauto) != 0; }
    bool isScope()        { return (storage_class & STCscope) != 0; }
    bool isSynchronized() { return (storage_class & STCsynchronized) != 0; }
    bool isParameter()    { return (storage_class & STCparameter) != 0; }
    bool isDeprecated()   { return (storage_class & STCdeprecated) != 0; }
    bool isOverride()     { return (storage_class & STCoverride) != 0; }
    bool isResult()       { return (storage_class & STCresult) != 0; }
    bool isField()        { return (storage_class & STCfield) != 0; }

    bool isIn()    { return (storage_class & STCin) != 0; }
    bool isOut()   { return (storage_class & STCout) != 0; }
    bool isRef()   { return (storage_class & STCref) != 0; }

    enum PROT prot();

    Declaration *isDeclaration() { return this; }
};

/**************************************************************/

struct TupleDeclaration : Declaration
{
    Objects *objects;
    int isexp;                  // 1: expression tuple

    TypeTuple *tupletype;       // !=NULL if this is a type tuple

    TupleDeclaration(Loc loc, Identifier *ident, Objects *objects);
    Dsymbol *syntaxCopy(Dsymbol *);
    const char *kind();
    Type *getType();
    int needThis();

    TupleDeclaration *isTupleDeclaration() { return this; }

#if IN_LLVM
    void semantic3(Scope *sc);
    /// Codegen traversal
    void codegen(IRState* ir);
#endif
};

/**************************************************************/

struct TypedefDeclaration : Declaration
{
    Type *basetype;
    Initializer *init;

    TypedefDeclaration(Loc loc, Identifier *ident, Type *basetype, Initializer *init);
    Dsymbol *syntaxCopy(Dsymbol *);
    void semantic(Scope *sc);
    void semantic2(Scope *sc);
    const char *mangle(bool isv = false);
    const char *kind();
    Type *getType();
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    void toJson(JsonOut *json);
    Type *htype;
    Type *hbasetype;

    void toDocBuffer(OutBuffer *buf, Scope *sc);

#if IN_DMD
    void toObjFile(int multiobj);                       // compile to .obj file
    void toDebug();
    int cvMember(unsigned char *p);
#endif

    TypedefDeclaration *isTypedefDeclaration() { return this; }

#if IN_DMD
    Symbol *sinit;
    Symbol *toInitializer();
#endif

#if IN_LLVM
    /// Codegen traversal
    void codegen(IRState* ir);
#endif
};

/**************************************************************/

struct AliasDeclaration : Declaration
{
    Dsymbol *aliassym;
    Dsymbol *overnext;          // next in overload list
    Dsymbol *import;            // !=NULL if unresolved internal alias for selective import
    int inSemantic;

    AliasDeclaration(Loc loc, Identifier *ident, Type *type);
    AliasDeclaration(Loc loc, Identifier *ident, Dsymbol *s);
    Dsymbol *syntaxCopy(Dsymbol *);
    void semantic(Scope *sc);
    int overloadInsert(Dsymbol *s);
    const char *kind();
    Type *getType();
    Dsymbol *toAlias();
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    Type *htype;
    Dsymbol *haliassym;

    void toDocBuffer(OutBuffer *buf, Scope *sc);

    AliasDeclaration *isAliasDeclaration() { return this; }
};

/**************************************************************/

struct VarDeclaration : Declaration
{
    Initializer *init;
    unsigned offset;
    bool noscope;                // no auto semantics
#if DMDV2
    FuncDeclarations nestedrefs; // referenced by these lexically nested functions
    bool isargptr;              // if parameter that _argptr points to
#else
    int nestedref;              // referenced by a lexically nested function
#endif
    structalign_t alignment;
    bool ctorinit;              // it has been initialized in a ctor
    short onstack;              // 1: it has been allocated on the stack
                                // 2: on stack, run destructor anyway
    int canassign;              // it can be assigned to
    Dsymbol *aliassym;          // if redone as alias to another symbol

    // When interpreting, these point to the value (NULL if value not determinable)
    // The index of this variable on the CTFE stack, -1 if not allocated
    int ctfeAdrOnStack;
    // The various functions are used only to detect compiler CTFE bugs
    Expression *getValue();
    bool hasValue();
    void setValueNull();
    void setValueWithoutChecking(Expression *newval);
    void setValue(Expression *newval);

#if DMDV2
    VarDeclaration *rundtor;    // if !NULL, rundtor is tested at runtime to see
                                // if the destructor should be run. Used to prevent
                                // dtor calls on postblitted vars
    Expression *edtor;          // if !=NULL, does the destruction of the variable
#endif

    VarDeclaration(Loc loc, Type *t, Identifier *id, Initializer *init);
    Dsymbol *syntaxCopy(Dsymbol *);
    void semantic(Scope *sc);
    void setFieldOffset(AggregateDeclaration *ad, unsigned *poffset, bool isunion);
    void semantic2(Scope *sc);
    const char *kind();
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    void toJson(JsonOut *json);
    Type *htype;
    Initializer *hinit;
    AggregateDeclaration *isThis();
    int needThis();
    int isExport();
    int isImportedSymbol();
    int isDataseg();
    int isThreadlocal();
    int isCTFE();
    int hasPointers();
#if DMDV2
    int canTakeAddressOf();
    int needsAutoDtor();
#endif
    Expression *callScopeDtor(Scope *sc);
    ExpInitializer *getExpInitializer();
    Expression *getConstInitializer(bool needFullType = true);
    void checkCtorConstInit();
    void checkNestedReference(Scope *sc, Loc loc);
    Dsymbol *toAlias();
#if IN_DMD
    void toObjFile(int multiobj);                       // compile to .obj file
    Symbol *toSymbol();
    int cvMember(unsigned char *p);
#endif
    const char *mangle(bool isv = false);
    // Eliminate need for dynamic_cast
    VarDeclaration *isVarDeclaration() { return (VarDeclaration *)this; }

#if IN_LLVM
    /// Codegen traversal
    void codegen(IRState* ir);

    /// Index into parent aggregate.
    /// Set during type generation.
    unsigned aggrIndex;

    /// Variables that wouldn't have gotten semantic3'ed if we weren't inlining set this flag.
    bool availableExternally;
    /// Override added to set above flag.
    void semantic3(Scope *sc);

    /// This var is used by a naked function.
    bool nakedUse;

    // debug description
    llvm::DIVariable debugVariable;
    llvm::DISubprogram debugFunc;
#endif
};

/**************************************************************/

// This is a shell around a back end symbol

struct SymbolDeclaration : Declaration
{
    StructDeclaration *dsym;

    SymbolDeclaration(Loc loc, StructDeclaration *dsym);

#if IN_DMD
    Symbol *toSymbol();
#endif

    // Eliminate need for dynamic_cast
    SymbolDeclaration *isSymbolDeclaration() { return (SymbolDeclaration *)this; }
};

struct ClassInfoDeclaration : VarDeclaration
{
    ClassDeclaration *cd;

    ClassInfoDeclaration(ClassDeclaration *cd);
    Dsymbol *syntaxCopy(Dsymbol *);
    void semantic(Scope *sc);

    void emitComment(Scope *sc);
    void toJson(JsonOut *json);

#if IN_DMD
    Symbol *toSymbol();
#endif

    ClassInfoDeclaration* isClassInfoDeclaration() { return this; }
};

struct ModuleInfoDeclaration : VarDeclaration
{
    Module *mod;

    ModuleInfoDeclaration(Module *mod);
    Dsymbol *syntaxCopy(Dsymbol *);
    void semantic(Scope *sc);

    void emitComment(Scope *sc);
    void toJson(JsonOut *json);

#if IN_DMD
    Symbol *toSymbol();
#endif
};

struct TypeInfoDeclaration : VarDeclaration
{
    Type *tinfo;

    TypeInfoDeclaration(Type *tinfo, int internal);
    Dsymbol *syntaxCopy(Dsymbol *);
    void semantic(Scope *sc);

    void emitComment(Scope *sc);
    void toJson(JsonOut *json);

#if IN_DMD
    void toObjFile(int multiobj);                       // compile to .obj file
    Symbol *toSymbol();
    virtual void toDt(dt_t **pdt);
#endif

    virtual TypeInfoDeclaration* isTypeInfoDeclaration() { return this; }

#if IN_LLVM
    /// Codegen traversal
    void codegen(IRState* ir);
    virtual void llvmDefine();
#endif
};

struct TypeInfoStructDeclaration : TypeInfoDeclaration
{
    TypeInfoStructDeclaration(Type *tinfo);

#if IN_DMD
    void toDt(dt_t **pdt);
#endif

#if IN_LLVM
    void llvmDefine();
#endif
};

struct TypeInfoClassDeclaration : TypeInfoDeclaration
{
    TypeInfoClassDeclaration(Type *tinfo);

#if IN_DMD
    Symbol *toSymbol();
    void toDt(dt_t **pdt);
#endif

#if IN_LLVM
    // TypeInfoClassDeclaration instances are different; they describe
    // __ClassZ/__InterfaceZ symbols instead of a TypeInfo_….init one. DMD also
    // generates them for SomeInterface.classinfo access, so we can't just
    // distinguish between them using tinfo and thus need to override codegen().
    void codegen(IRState* ir);
    void llvmDefine();
#endif
};

struct TypeInfoInterfaceDeclaration : TypeInfoDeclaration
{
    TypeInfoInterfaceDeclaration(Type *tinfo);

#if IN_DMD
    void toDt(dt_t **pdt);
#endif

#if IN_LLVM
    void llvmDefine();
#endif
};

struct TypeInfoTypedefDeclaration : TypeInfoDeclaration
{
    TypeInfoTypedefDeclaration(Type *tinfo);

#if IN_DMD
    void toDt(dt_t **pdt);
#endif

#if IN_LLVM
    void llvmDefine();
#endif
};

struct TypeInfoPointerDeclaration : TypeInfoDeclaration
{
    TypeInfoPointerDeclaration(Type *tinfo);

#if IN_DMD
    void toDt(dt_t **pdt);
#endif

#if IN_LLVM
    void llvmDefine();
#endif
};

struct TypeInfoArrayDeclaration : TypeInfoDeclaration
{
    TypeInfoArrayDeclaration(Type *tinfo);

#if IN_DMD
    void toDt(dt_t **pdt);
#endif

#if IN_LLVM
    void llvmDefine();
#endif
};

struct TypeInfoStaticArrayDeclaration : TypeInfoDeclaration
{
    TypeInfoStaticArrayDeclaration(Type *tinfo);

#if IN_DMD
    void toDt(dt_t **pdt);
#endif

#if IN_LLVM
    void llvmDefine();
#endif
};

struct TypeInfoAssociativeArrayDeclaration : TypeInfoDeclaration
{
    TypeInfoAssociativeArrayDeclaration(Type *tinfo);

#if IN_DMD
    void toDt(dt_t **pdt);
#endif

#if IN_LLVM
    void llvmDefine();
#endif
};

struct TypeInfoEnumDeclaration : TypeInfoDeclaration
{
    TypeInfoEnumDeclaration(Type *tinfo);

#if IN_DMD
    void toDt(dt_t **pdt);
#endif

#if IN_LLVM
    void llvmDefine();
#endif
};

struct TypeInfoFunctionDeclaration : TypeInfoDeclaration
{
    TypeInfoFunctionDeclaration(Type *tinfo);

#if IN_DMD
    void toDt(dt_t **pdt);
#endif

#if IN_LLVM
    void llvmDefine();
#endif
};

struct TypeInfoDelegateDeclaration : TypeInfoDeclaration
{
    TypeInfoDelegateDeclaration(Type *tinfo);

#if IN_DMD
    void toDt(dt_t **pdt);
#endif

#if IN_LLVM
    void llvmDefine();
#endif
};

struct TypeInfoTupleDeclaration : TypeInfoDeclaration
{
    TypeInfoTupleDeclaration(Type *tinfo);

#if IN_DMD
    void toDt(dt_t **pdt);
#endif

#if IN_LLVM
    void llvmDefine();
#endif
};

#if DMDV2
struct TypeInfoConstDeclaration : TypeInfoDeclaration
{
    TypeInfoConstDeclaration(Type *tinfo);

#if IN_DMD
    void toDt(dt_t **pdt);
#endif

#if IN_LLVM
    void llvmDefine();
#endif
};

struct TypeInfoInvariantDeclaration : TypeInfoDeclaration
{
    TypeInfoInvariantDeclaration(Type *tinfo);

#if IN_DMD
    void toDt(dt_t **pdt);
#endif

#if IN_LLVM
    void llvmDefine();
#endif
};

struct TypeInfoSharedDeclaration : TypeInfoDeclaration
{
    TypeInfoSharedDeclaration(Type *tinfo);

#if IN_DMD
    void toDt(dt_t **pdt);
#endif

#if IN_LLVM
    void llvmDefine();
#endif
};

struct TypeInfoWildDeclaration : TypeInfoDeclaration
{
    TypeInfoWildDeclaration(Type *tinfo);

#if IN_DMD
    void toDt(dt_t **pdt);
#endif

#if IN_LLVM
    void llvmDefine();
#endif
};

struct TypeInfoVectorDeclaration : TypeInfoDeclaration
{
    TypeInfoVectorDeclaration(Type *tinfo);

#if IN_DMD
    void toDt(dt_t **pdt);
#endif

#if IN_LLVM
    void llvmDefine();
#endif
};
#endif

/**************************************************************/

struct ThisDeclaration : VarDeclaration
{
    ThisDeclaration(Loc loc, Type *t);
    Dsymbol *syntaxCopy(Dsymbol *);
    ThisDeclaration *isThisDeclaration() { return this; }
};

enum ILS
{
    ILSuninitialized,   // not computed yet
    ILSno,              // cannot inline
    ILSyes,             // can inline
};

/**************************************************************/
#if DMDV2

enum BUILTIN
{
    BUILTINunknown = -1,        // not known if this is a builtin
    BUILTINnot,                 // this is not a builtin
    BUILTINsin,                 // std.math.sin
    BUILTINcos,                 // std.math.cos
    BUILTINtan,                 // std.math.tan
    BUILTINsqrt,                // std.math.sqrt
    BUILTINfabs,                // std.math.fabs
    BUILTINatan2,               // std.math.atan2
    BUILTINrndtol,              // std.math.rndtol
    BUILTINexpm1,               // std.math.expm1
    BUILTINexp2,                // std.math.exp2
    BUILTINyl2x,                // std.math.yl2x
    BUILTINyl2xp1,              // std.math.yl2xp1
    BUILTINbsr,                 // core.bitop.bsr
    BUILTINbsf,                 // core.bitop.bsf
    BUILTINbswap,               // core.bitop.bswap
#ifdef IN_GCC
    BUILTINgcc,                 // GCC builtin
#endif
};

Expression *eval_builtin(Loc loc, enum BUILTIN builtin, Expressions *arguments);

#else
enum BUILTIN { };
#endif

struct FuncDeclaration : Declaration
{
    Types *fthrows;                     // Array of Type's of exceptions (not used)
    Statement *frequire;
    Statement *fensure;
    Statement *fbody;

    FuncDeclarations foverrides;        // functions this function overrides
    FuncDeclaration *fdrequire;         // function that does the in contract
    FuncDeclaration *fdensure;          // function that does the out contract

#if IN_LLVM
    // Argument lists for the __require/__ensure calls. NULL if not a virtual
    // function with contracts.
    Expressions *fdrequireParams;
    Expressions *fdensureParams;
#endif

    Identifier *outId;                  // identifier for out statement
    VarDeclaration *vresult;            // variable corresponding to outId
    LabelDsymbol *returnLabel;          // where the return goes
    Scope *scout;                       // out contract scope for vresult->semantic

    DsymbolTable *localsymtab;          // used to prevent symbols in different
                                        // scopes from having the same name
    VarDeclaration *vthis;              // 'this' parameter (member and nested)
    VarDeclaration *v_arguments;        // '_arguments' parameter
#ifdef IN_GCC
    VarDeclaration *v_arguments_var;    // '_arguments' variable
    VarDeclaration *v_argptr;           // '_argptr' variable
#endif
    VarDeclaration *v_argsave;          // save area for args passed in registers for variadic functions
    VarDeclarations *parameters;        // Array of VarDeclaration's for parameters
    DsymbolTable *labtab;               // statement label symbol table
    Declaration *overnext;              // next in overload list
    Loc endloc;                         // location of closing curly bracket
    int vtblIndex;                      // for member functions, index into vtbl[]
    bool naked;                         // !=0 if naked
    ILS inlineStatusStmt;
    ILS inlineStatusExp;
    int inlineNest;                     // !=0 if nested inline
#if IN_LLVM
    char isArrayOp;                     // 1 if compiler-generated array op, 2 if druntime-provided
#else
    bool isArrayOp;                     // !=0 if array operation
#endif
    FuncDeclaration *dArrayOp;          // D version of array op for ctfe
    enum PASS semanticRun;
    int semantic3Errors;                // !=0 if errors in semantic3
                                        // this function's frame ptr
    ForeachStatement *fes;              // if foreach body, this is the foreach
    bool introducing;                   // !=0 if 'introducing' function
    Type *tintro;                       // if !=NULL, then this is the type
                                        // of the 'introducing' function
                                        // this one is overriding
    int inferRetType;                   // !=0 if return type is to be inferred
    StorageClass storage_class2;        // storage class for template onemember's

    // Things that should really go into Scope
    int hasReturnExp;                   // 1 if there's a return exp; statement
                                        // 2 if there's a throw statement
                                        // 4 if there's an assert(0)
                                        // 8 if there's inline asm

    // Support for NRVO (named return value optimization)
    bool nrvo_can;                      // !=0 means we can do it
    VarDeclaration *nrvo_var;           // variable to replace with shidden
#if IN_DMD
    Symbol *shidden;                    // hidden pointer passed to function
#endif

    ReturnStatements *returns;

#if DMDV2
    enum BUILTIN builtin;               // set if this is a known, builtin
                                        // function we can evaluate at compile
                                        // time

    int tookAddressOf;                  // set if someone took the address of
                                        // this function
    bool requiresClosure;               // this function needs a closure
    VarDeclarations closureVars;        // local variables in this function
                                        // which are referenced by nested
                                        // functions
    FuncDeclarations siblingCallers;    // Sibling nested functions which
                                        // called this one
#if IN_DMD
    FuncDeclarations deferred;          // toObjFile() these functions after this one
#endif

    unsigned flags;
    #define FUNCFLAGpurityInprocess 1   // working on determining purity
    #define FUNCFLAGsafetyInprocess 2   // working on determining safety
    #define FUNCFLAGnothrowInprocess 4  // working on determining nothrow
#else
    int nestedFrameRef;                 // !=0 if nested variables referenced
#endif

    FuncDeclaration(Loc loc, Loc endloc, Identifier *id, StorageClass storage_class, Type *type);
    Dsymbol *syntaxCopy(Dsymbol *);
    void semantic(Scope *sc);
    void semantic2(Scope *sc);
    void semantic3(Scope *sc);
    bool functionSemantic();
    bool functionSemantic3();
    // called from semantic3
    VarDeclaration *declareThis(Scope *sc, AggregateDeclaration *ad);
    int equals(Object *o);

    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    void bodyToCBuffer(OutBuffer *buf, HdrGenState *hgs);
    void toJson(JsonOut *json);
    int overrides(FuncDeclaration *fd);
    int findVtblIndex(Dsymbols *vtbl, int dim);
    int overloadInsert(Dsymbol *s);
    FuncDeclaration *overloadExactMatch(Type *t);
    FuncDeclaration *overloadResolve(Loc loc, Type *tthis, Expressions *arguments, int flags = 0);
    MATCH leastAsSpecialized(FuncDeclaration *g);
    LabelDsymbol *searchLabel(Identifier *ident);
    AggregateDeclaration *isThis();
    AggregateDeclaration *isMember2();
    int getLevel(Loc loc, Scope *sc, FuncDeclaration *fd); // lexical nesting level difference
    void appendExp(Expression *e);
    void appendState(Statement *s);
    const char *mangle(bool isv = false);
    const char *toPrettyChars();
    const char *toFullSignature();  // for diagnostics, e.g. 'int foo(int x, int y) pure'
    int isMain();
    int isWinMain();
    int isDllMain();
    enum BUILTIN isBuiltin();
    int isExport();
    int isImportedSymbol();
    int isAbstract();
    int isCodeseg();
    int isOverloadable();
    int hasOverloads();
    enum PURE isPure();
    enum PURE isPureBypassingInference();
    bool setImpure();
    int isSafe();
    bool isSafeBypassingInference();
    int isTrusted();
    bool setUnsafe();
    bool isolateReturn();
    bool parametersIntersect(Type *t);
    virtual int isNested();
    int needThis();
    int isVirtualMethod();
    virtual int isVirtual();
    virtual int isFinal();
    virtual int addPreInvariant();
    virtual int addPostInvariant();
    Expression *interpret(InterState *istate, Expressions *arguments, Expression *thisexp = NULL);
    void inlineScan();
    int canInline(int hasthis, int hdrscan, int statementsToo);
    Expression *expandInline(InlineScanState *iss, Expression *ethis, Expressions *arguments, Statement **ps);
    const char *kind();
    void toDocBuffer(OutBuffer *buf, Scope *sc);
    FuncDeclaration *isUnique();
    void checkNestedReference(Scope *sc, Loc loc);
    int needsClosure();
    int hasNestedFrameRefs();
    void buildResultVar();
    Statement *mergeFrequire(Statement *, Expressions *params = 0);
    Statement *mergeFensure(Statement *, Expressions *params = 0);
    Parameters *getParameters(int *pvarargs);

// LDC: give argument types to runtime functions
    static FuncDeclaration *genCfunc(Parameters *args, Type *treturn, const char *name);
    static FuncDeclaration *genCfunc(Parameters *args, Type *treturn, Identifier *id);

#if IN_DMD
    Symbol *toSymbol();
    Symbol *toThunkSymbol(int offset);  // thunk version
    void toObjFile(int multiobj);                       // compile to .obj file
    int cvMember(unsigned char *p);
    void buildClosure(IRState *irs); // Should this be inside or outside the #if IN_DMD?
#endif
    FuncDeclaration *isFuncDeclaration() { return this; }

    virtual FuncDeclaration *toAliasFunc() { return this; }

#if IN_LLVM
    IrFuncTy irFty;

    /// Codegen traversal
    void codegen(IRState* ir);

    std::string intrinsicName;
    uint32_t priority;

    bool isIntrinsic();
    bool isVaIntrinsic();

    // we keep our own table of label statements as LabelDsymbolS
    // don't always carry their corresponding statement along ...
    typedef std::map<const char*, LabelStatement*> LabelMap;
    LabelMap labmap;

    // Functions that wouldn't have gotten semantic3'ed if we weren't inlining set this flag.
    bool availableExternally;

    // true if overridden with the pragma(LDC_allow_inline); stmt
    bool allowInlining;

    // true if set with the pragma(LDC_never_inline); stmt
    bool neverInline;

    // true if has inline assembler
    bool inlineAsm;
#endif
};

#if DMDV2
FuncDeclaration *resolveFuncCall(Loc loc, Scope *sc, Dsymbol *s,
        Objects *tiargs,
        Type *tthis,
        Expressions *arguments,
        int flags = 0);
#endif

struct FuncAliasDeclaration : FuncDeclaration
{
    FuncDeclaration *funcalias;
    int hasOverloads;

    FuncAliasDeclaration(FuncDeclaration *funcalias, int hasOverloads = 1);

    FuncAliasDeclaration *isFuncAliasDeclaration() { return this; }
    const char *kind();
#if IN_DMD
    Symbol *toSymbol();
#endif
    const char *mangle(bool isv = false) { return toAliasFunc()->mangle(isv); }

    FuncDeclaration *toAliasFunc();
};

struct FuncLiteralDeclaration : FuncDeclaration
{
    enum TOK tok;                       // TOKfunction or TOKdelegate
    Type *treq;                         // target of return type inference

    FuncLiteralDeclaration(Loc loc, Loc endloc, Type *type, enum TOK tok,
        ForeachStatement *fes);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    Dsymbol *syntaxCopy(Dsymbol *);
    int isNested();
    int isVirtual();

    FuncLiteralDeclaration *isFuncLiteralDeclaration() { return this; }
    const char *kind();

#if IN_LLVM
    // If this is only used as alias parameter to a template instantiation,
    // keep track of which one, as the function will only be codegen'ed in the
    // module the template instance is pushed to, which is not always the same
    // as this->module because of the importedFrom check in
    // TemplateInstance::semantic and the fact that importedFrom is only set
    // once for the first module.
    TemplateInstance *owningTemplate;
#endif
};

struct CtorDeclaration : FuncDeclaration
{
    CtorDeclaration(Loc loc, Loc endloc, StorageClass stc, Type *type);
    Dsymbol *syntaxCopy(Dsymbol *);
    void semantic(Scope *sc);
    const char *kind();
    char *toChars();
    int isVirtual();
    int addPreInvariant();
    int addPostInvariant();

    CtorDeclaration *isCtorDeclaration() { return this; }
};

#if DMDV2
struct PostBlitDeclaration : FuncDeclaration
{
    PostBlitDeclaration(Loc loc, Loc endloc, StorageClass stc, Identifier *id);
    Dsymbol *syntaxCopy(Dsymbol *);
    void semantic(Scope *sc);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    void toJson(JsonOut *json);
    int isVirtual();
    int addPreInvariant();
    int addPostInvariant();
    int overloadInsert(Dsymbol *s);
    void emitComment(Scope *sc);

    PostBlitDeclaration *isPostBlitDeclaration() { return this; }
};
#endif

struct DtorDeclaration : FuncDeclaration
{
    DtorDeclaration(Loc loc, Loc endloc);
    DtorDeclaration(Loc loc, Loc endloc, StorageClass stc, Identifier *id);
    Dsymbol *syntaxCopy(Dsymbol *);
    void semantic(Scope *sc);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    const char *kind();
    char *toChars();
    int isVirtual();
    int addPreInvariant();
    int addPostInvariant();
    int overloadInsert(Dsymbol *s);
    void emitComment(Scope *sc);

    DtorDeclaration *isDtorDeclaration() { return this; }
};

struct StaticCtorDeclaration : FuncDeclaration
{
    StaticCtorDeclaration(Loc loc, Loc endloc);
    StaticCtorDeclaration(Loc loc, Loc endloc, const char *name);
    Dsymbol *syntaxCopy(Dsymbol *);
    void semantic(Scope *sc);
    AggregateDeclaration *isThis();
    int isVirtual();
    int addPreInvariant();
    int addPostInvariant();
    bool hasStaticCtorOrDtor();
    void emitComment(Scope *sc);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);

    StaticCtorDeclaration *isStaticCtorDeclaration() { return this; }
};

#if DMDV2
struct SharedStaticCtorDeclaration : StaticCtorDeclaration
{
    SharedStaticCtorDeclaration(Loc loc, Loc endloc);
    Dsymbol *syntaxCopy(Dsymbol *);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);

    SharedStaticCtorDeclaration *isSharedStaticCtorDeclaration() { return this; }
};
#endif

struct StaticDtorDeclaration : FuncDeclaration
{   VarDeclaration *vgate;      // 'gate' variable

    StaticDtorDeclaration(Loc loc, Loc endloc);
    StaticDtorDeclaration(Loc loc, Loc endloc, const char *name);
    Dsymbol *syntaxCopy(Dsymbol *);
    void semantic(Scope *sc);
    AggregateDeclaration *isThis();
    int isVirtual();
    bool hasStaticCtorOrDtor();
    int addPreInvariant();
    int addPostInvariant();
    void emitComment(Scope *sc);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);

    StaticDtorDeclaration *isStaticDtorDeclaration() { return this; }
};

#if DMDV2
struct SharedStaticDtorDeclaration : StaticDtorDeclaration
{
    SharedStaticDtorDeclaration(Loc loc, Loc endloc);
    Dsymbol *syntaxCopy(Dsymbol *);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);

    SharedStaticDtorDeclaration *isSharedStaticDtorDeclaration() { return this; }
};
#endif

struct InvariantDeclaration : FuncDeclaration
{
    InvariantDeclaration(Loc loc, Loc endloc, StorageClass stc, Identifier *id = NULL);
    Dsymbol *syntaxCopy(Dsymbol *);
    void semantic(Scope *sc);
    int isVirtual();
    int addPreInvariant();
    int addPostInvariant();
    void emitComment(Scope *sc);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);

    InvariantDeclaration *isInvariantDeclaration() { return this; }
};

struct UnitTestDeclaration : FuncDeclaration
{
    char *codedoc; /** For documented unittest. */
    UnitTestDeclaration(Loc loc, Loc endloc, char *codedoc);
    Dsymbol *syntaxCopy(Dsymbol *);
    void semantic(Scope *sc);
    AggregateDeclaration *isThis();
    int isVirtual();
    int addPreInvariant();
    int addPostInvariant();
    void emitComment(Scope *sc);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);

    UnitTestDeclaration *isUnitTestDeclaration() { return this; }
};

struct NewDeclaration : FuncDeclaration
{   Parameters *arguments;
    int varargs;

    NewDeclaration(Loc loc, Loc endloc, Parameters *arguments, int varargs);
    Dsymbol *syntaxCopy(Dsymbol *);
    void semantic(Scope *sc);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    const char *kind();
    int isVirtual();
    int addPreInvariant();
    int addPostInvariant();

    NewDeclaration *isNewDeclaration() { return this; }
};


struct DeleteDeclaration : FuncDeclaration
{   Parameters *arguments;

    DeleteDeclaration(Loc loc, Loc endloc, Parameters *arguments);
    Dsymbol *syntaxCopy(Dsymbol *);
    void semantic(Scope *sc);
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    const char *kind();
    int isDelete();
    int isVirtual();
    int addPreInvariant();
    int addPostInvariant();
    DeleteDeclaration *isDeleteDeclaration() { return this; }
};

#endif /* DMD_DECLARATION_H */
