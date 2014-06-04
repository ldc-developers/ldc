
// Compiler implementation of the D programming language
// Copyright (c) 1999-2013 by Digital Mars
// All Rights Reserved
// written by Walter Bright
// http://www.digitalmars.com
// License for redistribution is by either the Artistic License
// in artistic.txt, or the GNU General Public License in gnu.txt.
// See the included readme.txt for details.

#ifndef DMD_MTYPE_H
#define DMD_MTYPE_H

#ifdef __DMC__
#pragma once
#endif /* __DMC__ */

#include "root.h"
#include "stringtable.h"

#include "arraytypes.h"
#include "expression.h"

#if IN_LLVM
#include "../ir/irfuncty.h"
namespace llvm { class Type; }
class IrType;
#endif

struct Scope;
struct Identifier;
struct Expression;
struct StructDeclaration;
struct ClassDeclaration;
struct VarDeclaration;
struct EnumDeclaration;
struct TypedefDeclaration;
struct TypeInfoDeclaration;
struct Dsymbol;
struct TemplateInstance;
struct CppMangleState;
struct TemplateDeclaration;
struct JsonOut;
enum LINK;

struct TypeBasic;
struct HdrGenState;
struct Parameter;

// Back end
#ifdef IN_GCC
union tree_node; typedef union tree_node TYPE;
typedef TYPE type;
#elif IN_LLVM
#else
typedef struct TYPE type;
#endif

#if IN_DMD
struct Symbol;
#endif

struct TypeTuple;

enum ENUMTY
{
    Tarray,             // slice array, aka T[]
    Tsarray,            // static array, aka T[dimension]
    Taarray,            // associative array, aka T[type]
    Tpointer,
    Treference,
    Tfunction,
    Tident,
    Tclass,
    Tstruct,
    Tenum,

    Ttypedef,
    Tdelegate,
    Tnone,
    Tvoid,
    Tint8,
    Tuns8,
    Tint16,
    Tuns16,
    Tint32,
    Tuns32,

    Tint64,
    Tuns64,
    Tfloat32,
    Tfloat64,
    Tfloat80,
    Timaginary32,
    Timaginary64,
    Timaginary80,
    Tcomplex32,
    Tcomplex64,

    Tcomplex80,
    Tbool,
    Tchar,
    Twchar,
    Tdchar,
    Terror,
    Tinstance,
    Ttypeof,
    Ttuple,
    Tslice,

    Treturn,
    Tnull,
    Tvector,
    Tint128,
    Tuns128,
    TMAX
};
typedef unsigned char TY;       // ENUMTY

#define Tascii Tchar

extern int Tsize_t;
extern int Tptrdiff_t;


/* pick this order of numbers so switch statements work better
 */
#define MODconst     1  // type is const
#define MODimmutable 4  // type is immutable
#define MODshared    2  // type is shared
#define MODwild      8  // type is wild
#define MODmutable   0x10       // type is mutable (only used in wildcard matching)

struct Type : Object
{
    TY ty;
    unsigned char mod;  // modifiers MODxxxx
    char *deco;

    /* These are cached values that are lazily evaluated by constOf(), invariantOf(), etc.
     * They should not be referenced by anybody but mtype.c.
     * They can be NULL if not lazily evaluated yet.
     * Note that there is no "shared immutable", because that is just immutable
     * Naked == no MOD bits
     */

    Type *cto;          // MODconst ? naked version of this type : const version
    Type *ito;          // MODimmutable ? naked version of this type : immutable version
    Type *sto;          // MODshared ? naked version of this type : shared mutable version
    Type *scto;         // MODshared|MODconst ? naked version of this type : shared const version
    Type *wto;          // MODwild ? naked version of this type : wild version
    Type *swto;         // MODshared|MODwild ? naked version of this type : shared wild version

    Type *pto;          // merged pointer to this type
    Type *rto;          // reference to this type
    Type *arrayof;      // array of this type
    TypeInfoDeclaration *vtinfo;        // TypeInfo object for this Type

#if IN_DMD
    type *ctype;        // for back end
#endif

    static Type *tvoid;
    static Type *tint8;
    static Type *tuns8;
    static Type *tint16;
    static Type *tuns16;
    static Type *tint32;
    static Type *tuns32;
    static Type *tint64;
    static Type *tuns64;
    static Type *tint128;
    static Type *tuns128;
    static Type *tfloat32;
    static Type *tfloat64;
    static Type *tfloat80;

    static Type *timaginary32;
    static Type *timaginary64;
    static Type *timaginary80;

    static Type *tcomplex32;
    static Type *tcomplex64;
    static Type *tcomplex80;

    static Type *tbool;
    static Type *tchar;
    static Type *twchar;
    static Type *tdchar;

    // Some special types
    static Type *tshiftcnt;
    static Type *tboolean;
    static Type *tvoidptr;              // void*
    static Type *tstring;               // immutable(char)[]
    static Type *tvalist;               // va_list alias
    static Type *terror;                // for error recovery
    static Type *tnull;                 // for null type

    static Type *tsize_t;               // matches size_t alias
    static Type *tptrdiff_t;            // matches ptrdiff_t alias
    static Type *thash_t;               // matches hash_t alias
    static Type *tindex;                // array/ptr index

    static ClassDeclaration *typeinfo;
    static ClassDeclaration *typeinfoclass;
    static ClassDeclaration *typeinfointerface;
    static ClassDeclaration *typeinfostruct;
    static ClassDeclaration *typeinfotypedef;
    static ClassDeclaration *typeinfopointer;
    static ClassDeclaration *typeinfoarray;
    static ClassDeclaration *typeinfostaticarray;
    static ClassDeclaration *typeinfoassociativearray;
    static ClassDeclaration *typeinfovector;
    static ClassDeclaration *typeinfoenum;
    static ClassDeclaration *typeinfofunction;
    static ClassDeclaration *typeinfodelegate;
    static ClassDeclaration *typeinfotypelist;
    static ClassDeclaration *typeinfoconst;
    static ClassDeclaration *typeinfoinvariant;
    static ClassDeclaration *typeinfoshared;
    static ClassDeclaration *typeinfowild;

    static TemplateDeclaration *associativearray;
    static TemplateDeclaration *rtinfo;

    static Type *basic[TMAX];
    static unsigned char mangleChar[TMAX];
    static unsigned short sizeTy[TMAX];
    static StringTable stringtable;

    // These tables are for implicit conversion of binary ops;
    // the indices are the type of operand one, followed by operand two.
    static unsigned char impcnvResult[TMAX][TMAX];
    static unsigned char impcnvType1[TMAX][TMAX];
    static unsigned char impcnvType2[TMAX][TMAX];

    // If !=0, give warning on implicit conversion
    static unsigned char impcnvWarn[TMAX][TMAX];

    Type(TY ty);
    virtual const char *kind();
    virtual Type *syntaxCopy();
    int equals(Object *o);
    int dyncast() { return DYNCAST_TYPE; } // kludge for template.isType()
    int covariant(Type *t, StorageClass *pstc = NULL);
    char *toChars();
    static char needThisPrefix();
    static void init();

    #define SIZE_INVALID (~(d_uns64)0)
    d_uns64 size();
    virtual d_uns64 size(Loc loc);
    virtual unsigned alignsize();
    virtual Type *semantic(Loc loc, Scope *sc);
    Type *trySemantic(Loc loc, Scope *sc);
    virtual void toDecoBuffer(OutBuffer *buf, int flag = 0);
    Type *merge();
    Type *merge2();
    virtual void toCBuffer(OutBuffer *buf, Identifier *ident, HdrGenState *hgs);
    virtual void toCBuffer2(OutBuffer *buf, HdrGenState *hgs, int mod);
    void toCBuffer3(OutBuffer *buf, HdrGenState *hgs, int mod);
    void modToBuffer(OutBuffer *buf);
    char *modToChars();
    virtual void toJson(JsonOut *json);
#if CPP_MANGLE
    virtual void toCppMangle(OutBuffer *buf, CppMangleState *cms);
#endif
    virtual int isintegral();
    virtual int isfloating();   // real, imaginary, or complex
    virtual int isreal();
    virtual int isimaginary();
    virtual int iscomplex();
    virtual int isscalar();
    virtual int isunsigned();
    virtual int isscope();
    virtual int isString();
    virtual int isAssignable();
    virtual int checkBoolean(); // if can be converted to boolean value
    virtual void checkDeprecated(Loc loc, Scope *sc);
    int isConst()       { return mod & MODconst; }
    int isImmutable()   { return mod & MODimmutable; }
    int isMutable()     { return !(mod & (MODconst | MODimmutable | MODwild)); }
    int isShared()      { return mod & MODshared; }
    int isSharedConst() { return mod == (MODshared | MODconst); }
    int isWild()        { return mod & MODwild; }
    int isSharedWild()  { return mod == (MODshared | MODwild); }
    int isNaked()       { return mod == 0; }
    Type *nullAttributes();
    Type *constOf();
    Type *invariantOf();
    Type *mutableOf();
    Type *sharedOf();
    Type *sharedConstOf();
    Type *unSharedOf();
    Type *wildOf();
    Type *sharedWildOf();
    void fixTo(Type *t);
    void check();
    Type *addSTC(StorageClass stc);
    Type *castMod(unsigned mod);
    Type *addMod(unsigned mod);
    virtual Type *addStorageClass(StorageClass stc);
    Type *pointerTo();
    Type *referenceTo();
    Type *arrayOf();
    Type *aliasthisOf();
    int checkAliasThisRec();
    virtual Type *makeConst();
    virtual Type *makeInvariant();
    virtual Type *makeShared();
    virtual Type *makeSharedConst();
    virtual Type *makeWild();
    virtual Type *makeSharedWild();
    virtual Type *makeMutable();
    virtual Dsymbol *toDsymbol(Scope *sc);
    virtual Type *toBasetype();
    virtual int isBaseOf(Type *t, int *poffset);
    virtual MATCH implicitConvTo(Type *to);
    virtual MATCH constConv(Type *to);
    virtual unsigned wildConvTo(Type *tprm);
    Type *substWildTo(unsigned mod);
    virtual Type *toHeadMutable();
    virtual ClassDeclaration *isClassHandle();
    virtual Expression *getProperty(Loc loc, Identifier *ident, int flag);
    virtual Expression *dotExp(Scope *sc, Expression *e, Identifier *ident, int flag);
    virtual structalign_t alignment();
    Expression *noMember(Scope *sc, Expression *e, Identifier *ident, int flag);
    virtual Expression *defaultInit(Loc loc = Loc());
    virtual Expression *defaultInitLiteral(Loc loc);
    virtual Expression *voidInitLiteral(VarDeclaration *var);
    virtual int isZeroInit(Loc loc = Loc());                // if initializer is 0
#if IN_DMD
    virtual dt_t **toDt(dt_t **pdt);
#endif
    Identifier *getTypeInfoIdent(int internal);
    virtual MATCH deduceType(Scope *sc, Type *tparam, TemplateParameters *parameters, Objects *dedtypes, unsigned *wildmatch = NULL);
    virtual void resolve(Loc loc, Scope *sc, Expression **pe, Type **pt, Dsymbol **ps);
    Expression *getInternalTypeInfo(Scope *sc);
    Expression *getTypeInfo(Scope *sc);
    virtual TypeInfoDeclaration *getTypeInfoDeclaration();
    virtual int builtinTypeInfo();
    virtual Type *reliesOnTident(TemplateParameters *tparams = NULL);
    virtual int hasWild();
    virtual Expression *toExpression();
    virtual int hasPointers();
    virtual TypeTuple *toArgTypes();
    virtual Type *nextOf();
    uinteger_t sizemask();
    virtual int needsDestruction();
    virtual bool needsNested();


    static void error(Loc loc, const char *format, ...) IS_PRINTF(2);
    static void warning(Loc loc, const char *format, ...) IS_PRINTF(2);

#if IN_DMD
    // For backend
    virtual unsigned totym();
    virtual type *toCtype();
    virtual type *toCParamtype();
    virtual Symbol *toSymbol();
#endif

    // For eliminating dynamic_cast
    virtual TypeBasic *isTypeBasic();

#if IN_LLVM
    IrType* irtype;
#endif
};

struct TypeError : Type
{
    TypeError();
    Type *syntaxCopy();

    void toCBuffer(OutBuffer *buf, Identifier *ident, HdrGenState *hgs);

    d_uns64 size(Loc loc);
    Expression *getProperty(Loc loc, Identifier *ident, int flag);
    Expression *dotExp(Scope *sc, Expression *e, Identifier *ident, int flag);
    Expression *defaultInit(Loc loc);
    Expression *defaultInitLiteral(Loc loc);
    TypeTuple *toArgTypes();
};

struct TypeNext : Type
{
    Type *next;

    TypeNext(TY ty, Type *next);
    void toDecoBuffer(OutBuffer *buf, int flag);
    void checkDeprecated(Loc loc, Scope *sc);
    Type *reliesOnTident(TemplateParameters *tparams = NULL);
    int hasWild();
    Type *nextOf();
    Type *makeConst();
    Type *makeInvariant();
    Type *makeShared();
    Type *makeSharedConst();
    Type *makeWild();
    Type *makeSharedWild();
    Type *makeMutable();
    MATCH constConv(Type *to);
    unsigned wildConvTo(Type *tprm);
    void transitive();
};

struct TypeBasic : Type
{
    const char *dstring;
    unsigned flags;

    TypeBasic(TY ty);
    const char *kind();
    Type *syntaxCopy();
    d_uns64 size(Loc loc);
    unsigned alignsize();
#if IN_LLVM
    signed long alignment();
#endif
    Expression *getProperty(Loc loc, Identifier *ident, int flag);
    Expression *dotExp(Scope *sc, Expression *e, Identifier *ident, int flag);
    char *toChars();
    void toCBuffer2(OutBuffer *buf, HdrGenState *hgs, int mod);
#if CPP_MANGLE
    void toCppMangle(OutBuffer *buf, CppMangleState *cms);
#endif
    int isintegral();
    int isfloating();
    int isreal();
    int isimaginary();
    int iscomplex();
    int isscalar();
    int isunsigned();
    MATCH implicitConvTo(Type *to);
    Expression *defaultInit(Loc loc);
    int isZeroInit(Loc loc);
    int builtinTypeInfo();
    TypeTuple *toArgTypes();

    // For eliminating dynamic_cast
    TypeBasic *isTypeBasic();
};

struct TypeVector : Type
{
    Type *basetype;

    TypeVector(Loc loc, Type *basetype);
    const char *kind();
    Type *syntaxCopy();
    Type *semantic(Loc loc, Scope *sc);
    d_uns64 size(Loc loc);
    unsigned alignsize();
    Expression *getProperty(Loc loc, Identifier *ident, int flag);
    Expression *dotExp(Scope *sc, Expression *e, Identifier *ident, int flag);
    char *toChars();
    void toCBuffer2(OutBuffer *buf, HdrGenState *hgs, int mod);
    void toDecoBuffer(OutBuffer *buf, int flag);
    void toJson(JsonOut *json);
    MATCH deduceType(Scope *sc, Type *tparam, TemplateParameters *parameters, Objects *dedtypes, unsigned *wildmatch = NULL);
#if CPP_MANGLE
    void toCppMangle(OutBuffer *buf, CppMangleState *cms);
#endif
    int isintegral();
    int isfloating();
    int isscalar();
    int isunsigned();
    int checkBoolean();
    MATCH implicitConvTo(Type *to);
    Expression *defaultInit(Loc loc);
#if IN_LLVM
    Expression *defaultInitLiteral(Loc loc);
#endif
    TypeBasic *elementType();
    int isZeroInit(Loc loc);
    TypeInfoDeclaration *getTypeInfoDeclaration();
    TypeTuple *toArgTypes();

#if IN_DMD
    type *toCtype();
#endif
};

struct TypeArray : TypeNext
{
    TypeArray(TY ty, Type *next);
    Expression *dotExp(Scope *sc, Expression *e, Identifier *ident, int flag);
};

// Static array, one with a fixed dimension
struct TypeSArray : TypeArray
{
    Expression *dim;

    TypeSArray(Type *t, Expression *dim);
    const char *kind();
    Type *syntaxCopy();
    d_uns64 size(Loc loc);
    unsigned alignsize();
    Type *semantic(Loc loc, Scope *sc);
    void resolve(Loc loc, Scope *sc, Expression **pe, Type **pt, Dsymbol **ps);
    void toDecoBuffer(OutBuffer *buf, int flag);
    void toCBuffer2(OutBuffer *buf, HdrGenState *hgs, int mod);
    void toJson(JsonOut *json);
    Expression *dotExp(Scope *sc, Expression *e, Identifier *ident, int flag);
    int isString();
    int isZeroInit(Loc loc);
    structalign_t alignment();
    MATCH constConv(Type *to);
    MATCH implicitConvTo(Type *to);
    Expression *defaultInit(Loc loc);
    Expression *defaultInitLiteral(Loc loc);
    Expression *voidInitLiteral(VarDeclaration *var);
#if IN_DMD
    dt_t **toDt(dt_t **pdt);
    dt_t **toDtElem(dt_t **pdt, Expression *e);
#endif
    MATCH deduceType(Scope *sc, Type *tparam, TemplateParameters *parameters, Objects *dedtypes, unsigned *wildmatch = NULL);
    TypeInfoDeclaration *getTypeInfoDeclaration();
    Expression *toExpression();
    int hasPointers();
    int needsDestruction();
    bool needsNested();
    TypeTuple *toArgTypes();
#if CPP_MANGLE
    void toCppMangle(OutBuffer *buf, CppMangleState *cms);
#endif

#if IN_DMD
    type *toCtype();
    type *toCParamtype();
#endif
};

// Dynamic array, no dimension
struct TypeDArray : TypeArray
{
    TypeDArray(Type *t);
    const char *kind();
    Type *syntaxCopy();
    d_uns64 size(Loc loc);
    unsigned alignsize();
    Type *semantic(Loc loc, Scope *sc);
    void resolve(Loc loc, Scope *sc, Expression **pe, Type **pt, Dsymbol **ps);
    void toDecoBuffer(OutBuffer *buf, int flag);
    void toCBuffer2(OutBuffer *buf, HdrGenState *hgs, int mod);
    void toJson(JsonOut *json);
    Expression *dotExp(Scope *sc, Expression *e, Identifier *ident, int flag);
    int isString();
    int isZeroInit(Loc loc);
    int checkBoolean();
    MATCH implicitConvTo(Type *to);
    Expression *defaultInit(Loc loc);
    int builtinTypeInfo();
    MATCH deduceType(Scope *sc, Type *tparam, TemplateParameters *parameters, Objects *dedtypes, unsigned *wildmatch = NULL);
    TypeInfoDeclaration *getTypeInfoDeclaration();
    int hasPointers();
    TypeTuple *toArgTypes();
#if CPP_MANGLE
    void toCppMangle(OutBuffer *buf, CppMangleState *cms);
#endif

#if IN_DMD
    type *toCtype();
#endif
};

struct TypeAArray : TypeArray
{
    Type *index;                // key type
    Loc loc;
    Scope *sc;

    StructDeclaration *impl;    // implementation

    TypeAArray(Type *t, Type *index);
    const char *kind();
    Type *syntaxCopy();
    d_uns64 size(Loc loc);
    Type *semantic(Loc loc, Scope *sc);
    StructDeclaration *getImpl();
    void resolve(Loc loc, Scope *sc, Expression **pe, Type **pt, Dsymbol **ps);
    void toDecoBuffer(OutBuffer *buf, int flag);
    void toCBuffer2(OutBuffer *buf, HdrGenState *hgs, int mod);
    void toJson(JsonOut *json);
    Expression *dotExp(Scope *sc, Expression *e, Identifier *ident, int flag);
    Expression *defaultInit(Loc loc);
    MATCH deduceType(Scope *sc, Type *tparam, TemplateParameters *parameters, Objects *dedtypes, unsigned *wildmatch = NULL);
    int isZeroInit(Loc loc);
    int checkBoolean();
    TypeInfoDeclaration *getTypeInfoDeclaration();
    Type *reliesOnTident(TemplateParameters *tparams);
    Expression *toExpression();
    int hasPointers();
    TypeTuple *toArgTypes();
    MATCH implicitConvTo(Type *to);
    MATCH constConv(Type *to);
#if CPP_MANGLE
    void toCppMangle(OutBuffer *buf, CppMangleState *cms);
#endif

#if IN_DMD
    // Back end
    Symbol *aaGetSymbol(const char *func, int flags);

    type *toCtype();
#endif
};

struct TypePointer : TypeNext
{
    TypePointer(Type *t);
    const char *kind();
    Type *syntaxCopy();
    Type *semantic(Loc loc, Scope *sc);
    d_uns64 size(Loc loc);
    void toCBuffer2(OutBuffer *buf, HdrGenState *hgs, int mod);
    void toJson(JsonOut *json);
    MATCH implicitConvTo(Type *to);
    MATCH constConv(Type *to);
    int isscalar();
    Expression *defaultInit(Loc loc);
    int isZeroInit(Loc loc);
    TypeInfoDeclaration *getTypeInfoDeclaration();
    int hasPointers();
    TypeTuple *toArgTypes();
#if CPP_MANGLE
    void toCppMangle(OutBuffer *buf, CppMangleState *cms);
#endif

#if IN_DMD
    type *toCtype();
#endif
};

struct TypeReference : TypeNext
{
    TypeReference(Type *t);
    const char *kind();
    Type *syntaxCopy();
    Type *semantic(Loc loc, Scope *sc);
    d_uns64 size(Loc loc);
    void toCBuffer2(OutBuffer *buf, HdrGenState *hgs, int mod);
    void toJson(JsonOut *json);
    Expression *dotExp(Scope *sc, Expression *e, Identifier *ident, int flag);
    Expression *defaultInit(Loc loc);
    int isZeroInit(Loc loc);
#if CPP_MANGLE
    void toCppMangle(OutBuffer *buf, CppMangleState *cms);
#endif
};

enum RET
{
    RETregs     = 1,    // returned in registers
    RETstack    = 2,    // returned on stack
};

enum TRUST
{
    TRUSTdefault = 0,
    TRUSTsystem = 1,    // @system (same as TRUSTdefault)
    TRUSTtrusted = 2,   // @trusted
    TRUSTsafe = 3,      // @safe
};

enum PURE
{
    PUREimpure = 0,     // not pure at all
    PUREweak = 1,       // no mutable globals are read or written
    PUREconst = 2,      // parameters are values or const
    PUREstrong = 3,     // parameters are values or immutable
    PUREfwdref = 4,     // it's pure, but not known which level yet
};

struct TypeFunction : TypeNext
{
    // .next is the return type

    Parameters *parameters;     // function parameters
    int varargs;        // 1: T t, ...) style for variable number of arguments
                        // 2: T t ...) style for variable number of arguments
    bool isnothrow;     // true: nothrow
    bool isproperty;    // can be called without parentheses
    bool isref;         // true: returns a reference
    enum LINK linkage;  // calling convention
    enum TRUST trust;   // level of trust
    enum PURE purity;   // PURExxxx
    bool iswild;        // is inout function
    Expressions *fargs; // function arguments

    int inuse;

    TypeFunction(Parameters *parameters, Type *treturn, int varargs, enum LINK linkage, StorageClass stc = 0);
    const char *kind();
    TypeFunction *copy();
    Type *syntaxCopy();
    Type *semantic(Loc loc, Scope *sc);
    void purityLevel();
    void toDecoBuffer(OutBuffer *buf, int flag);
    void toCBuffer(OutBuffer *buf, Identifier *ident, HdrGenState *hgs);
    void toCBufferWithAttributes(OutBuffer *buf, Identifier *ident, HdrGenState* hgs, TypeFunction *attrs, TemplateDeclaration *td);
    void toCBuffer2(OutBuffer *buf, HdrGenState *hgs, int mod);
    void toJson(JsonOut *json);
    void attributesToCBuffer(OutBuffer *buf, int mod);
    MATCH deduceType(Scope *sc, Type *tparam, TemplateParameters *parameters, Objects *dedtypes, unsigned *wildmatch = NULL);
    TypeInfoDeclaration *getTypeInfoDeclaration();
    Type *reliesOnTident(TemplateParameters *tparams = NULL);
    bool hasLazyParameters();
#if CPP_MANGLE
    void toCppMangle(OutBuffer *buf, CppMangleState *cms);
#endif
    bool parameterEscapes(Parameter *p);
    Type *addStorageClass(StorageClass stc);

    MATCH callMatch(Type *tthis, Expressions *toargs, int flag = 0);
#if IN_DMD
    type *toCtype();
#endif

    enum RET retStyle();

#if IN_DMD
    unsigned totym();
#endif

    Expression *defaultInit(Loc loc);

#if IN_LLVM
    IrFuncTy irFty;
#endif
};

struct TypeDelegate : TypeNext
{
    // .next is a TypeFunction

    TypeDelegate(Type *t);
    const char *kind();
    Type *syntaxCopy();
    Type *semantic(Loc loc, Scope *sc);
    d_uns64 size(Loc loc);
    unsigned alignsize();
    MATCH implicitConvTo(Type *to);
    void toCBuffer2(OutBuffer *buf, HdrGenState *hgs, int mod);
    void toJson(JsonOut *json);
    Expression *defaultInit(Loc loc);
    int isZeroInit(Loc loc);
    int checkBoolean();
    TypeInfoDeclaration *getTypeInfoDeclaration();
    Expression *dotExp(Scope *sc, Expression *e, Identifier *ident, int flag);
    int hasPointers();
    TypeTuple *toArgTypes();
#if CPP_MANGLE
    void toCppMangle(OutBuffer *buf, CppMangleState *cms);
#endif

#if IN_DMD
    type *toCtype();
#endif

#if IN_LLVM
    IrFuncTy irFty;
#endif
};

struct TypeQualified : Type
{
    Loc loc;
    Objects idents;         // array of Identifier and TypeInstance,
                            // representing ident.ident!tiargs.ident. ... etc.

    TypeQualified(TY ty, Loc loc);
    void syntaxCopyHelper(TypeQualified *t);
    void addIdent(Identifier *ident);
    void addInst(TemplateInstance *inst);
    void toCBuffer2Helper(OutBuffer *buf, HdrGenState *hgs);
    void toJson(JsonOut *json);
    d_uns64 size(Loc loc);
    void resolveHelper(Loc loc, Scope *sc, Dsymbol *s, Dsymbol *scopesym,
        Expression **pe, Type **pt, Dsymbol **ps);
};

struct TypeIdentifier : TypeQualified
{
    Identifier *ident;
    Dsymbol *originalSymbol; // The symbol representing this identifier, before alias resolution

    TypeIdentifier(Loc loc, Identifier *ident);
    const char *kind();
    Type *syntaxCopy();
    //char *toChars();
    void toDecoBuffer(OutBuffer *buf, int flag);
    void toCBuffer2(OutBuffer *buf, HdrGenState *hgs, int mod);
    void toJson(JsonOut *json);
    void resolve(Loc loc, Scope *sc, Expression **pe, Type **pt, Dsymbol **ps);
    Dsymbol *toDsymbol(Scope *sc);
    Type *semantic(Loc loc, Scope *sc);
    MATCH deduceType(Scope *sc, Type *tparam, TemplateParameters *parameters, Objects *dedtypes, unsigned *wildmatch = NULL);
    Type *reliesOnTident(TemplateParameters *tparams = NULL);
    Expression *toExpression();
};

/* Similar to TypeIdentifier, but with a TemplateInstance as the root
 */
struct TypeInstance : TypeQualified
{
    TemplateInstance *tempinst;

    TypeInstance(Loc loc, TemplateInstance *tempinst);
    const char *kind();
    Type *syntaxCopy();
    //char *toChars();
    //void toDecoBuffer(OutBuffer *buf, int flag);
    void toCBuffer2(OutBuffer *buf, HdrGenState *hgs, int mod);
    void toJson(JsonOut *json);
    void resolve(Loc loc, Scope *sc, Expression **pe, Type **pt, Dsymbol **ps);
    Type *semantic(Loc loc, Scope *sc);
    Dsymbol *toDsymbol(Scope *sc);
    Type *reliesOnTident(TemplateParameters *tparams = NULL);
    MATCH deduceType(Scope *sc, Type *tparam, TemplateParameters *parameters, Objects *dedtypes, unsigned *wildmatch = NULL);
    Expression *toExpression();
};

struct TypeTypeof : TypeQualified
{
    Expression *exp;
    int inuse;

    TypeTypeof(Loc loc, Expression *exp);
    const char *kind();
    Type *syntaxCopy();
    Dsymbol *toDsymbol(Scope *sc);
    void toCBuffer2(OutBuffer *buf, HdrGenState *hgs, int mod);
    void toJson(JsonOut *json);
    void resolve(Loc loc, Scope *sc, Expression **pe, Type **pt, Dsymbol **ps);
    Type *semantic(Loc loc, Scope *sc);
    d_uns64 size(Loc loc);
};

struct TypeReturn : TypeQualified
{
    TypeReturn(Loc loc);
    const char *kind();
    Type *syntaxCopy();
    Dsymbol *toDsymbol(Scope *sc);
    void resolve(Loc loc, Scope *sc, Expression **pe, Type **pt, Dsymbol **ps);
    Type *semantic(Loc loc, Scope *sc);
    void toCBuffer2(OutBuffer *buf, HdrGenState *hgs, int mod);
    void toJson(JsonOut *json);
};

// Whether alias this dependency is recursive or not.
enum AliasThisRec
{
    RECno = 0,      // no alias this recursion
    RECyes = 1,     // alias this has recursive dependency
    RECfwdref = 2,  // not yet known

    RECtracing = 0x4, // mark in progress of implicitConvTo/wildConvTo
};

struct TypeStruct : Type
{
    StructDeclaration *sym;
    enum AliasThisRec att;

    TypeStruct(StructDeclaration *sym);
    const char *kind();
    d_uns64 size(Loc loc);
    unsigned alignsize();
    char *toChars();
    Type *syntaxCopy();
    Type *semantic(Loc loc, Scope *sc);
    Dsymbol *toDsymbol(Scope *sc);
    void toDecoBuffer(OutBuffer *buf, int flag);
    void toCBuffer2(OutBuffer *buf, HdrGenState *hgs, int mod);
    void toJson(JsonOut *json);
    Expression *dotExp(Scope *sc, Expression *e, Identifier *ident, int flag);
    structalign_t alignment();
    Expression *defaultInit(Loc loc);
    Expression *defaultInitLiteral(Loc loc);
    Expression *voidInitLiteral(VarDeclaration *var);
    int isZeroInit(Loc loc);
    int isAssignable();
    int checkBoolean();
    int needsDestruction();
    bool needsNested();
#if IN_DMD
    dt_t **toDt(dt_t **pdt);
#endif
    MATCH deduceType(Scope *sc, Type *tparam, TemplateParameters *parameters, Objects *dedtypes, unsigned *wildmatch = NULL);
    TypeInfoDeclaration *getTypeInfoDeclaration();
    int hasPointers();
    TypeTuple *toArgTypes();
    MATCH implicitConvTo(Type *to);
    MATCH constConv(Type *to);
    unsigned wildConvTo(Type *tprm);
    Type *toHeadMutable();
#if CPP_MANGLE
    void toCppMangle(OutBuffer *buf, CppMangleState *cms);
#endif

#if IN_DMD
    type *toCtype();
#elif IN_LLVM
    // LDC
    // cache the hasUnalignedFields check
    // 0 = not checked, 1 = aligned, 2 = unaligned
    int unaligned;
#endif
};

struct TypeEnum : Type
{
    EnumDeclaration *sym;

    TypeEnum(EnumDeclaration *sym);
    const char *kind();
    Type *syntaxCopy();
    d_uns64 size(Loc loc);
    unsigned alignsize();
    char *toChars();
    Type *semantic(Loc loc, Scope *sc);
    Dsymbol *toDsymbol(Scope *sc);
    void toDecoBuffer(OutBuffer *buf, int flag);
    void toCBuffer2(OutBuffer *buf, HdrGenState *hgs, int mod);
    void toJson(JsonOut *json);
    Expression *dotExp(Scope *sc, Expression *e, Identifier *ident, int flag);
    Expression *getProperty(Loc loc, Identifier *ident, int flag);
    int isintegral();
    int isfloating();
    int isreal();
    int isimaginary();
    int iscomplex();
    int isscalar();
    int isunsigned();
    int checkBoolean();
    int isAssignable();
    int needsDestruction();
    bool needsNested();
    MATCH implicitConvTo(Type *to);
    MATCH constConv(Type *to);
    Type *toBasetype();
    Expression *defaultInit(Loc loc);
    int isZeroInit(Loc loc);
    MATCH deduceType(Scope *sc, Type *tparam, TemplateParameters *parameters, Objects *dedtypes, unsigned *wildmatch = NULL);
    TypeInfoDeclaration *getTypeInfoDeclaration();
    int hasPointers();
    TypeTuple *toArgTypes();
#if CPP_MANGLE
    void toCppMangle(OutBuffer *buf, CppMangleState *cms);
#endif

#if IN_DMD
    type *toCtype();
#endif
};

struct TypeTypedef : Type
{
    TypedefDeclaration *sym;

    TypeTypedef(TypedefDeclaration *sym);
    const char *kind();
    Type *syntaxCopy();
    d_uns64 size(Loc loc);
    unsigned alignsize();
    char *toChars();
    Type *semantic(Loc loc, Scope *sc);
    Dsymbol *toDsymbol(Scope *sc);
    void toDecoBuffer(OutBuffer *buf, int flag);
    void toCBuffer2(OutBuffer *buf, HdrGenState *hgs, int mod);
    void toJson(JsonOut *json);
    Expression *dotExp(Scope *sc, Expression *e, Identifier *ident, int flag);
    structalign_t alignment();
    Expression *getProperty(Loc loc, Identifier *ident, int flag);
    int isintegral();
    int isfloating();
    int isreal();
    int isimaginary();
    int iscomplex();
    int isscalar();
    int isunsigned();
    int checkBoolean();
    int isAssignable();
    int needsDestruction();
    bool needsNested();
    Type *toBasetype();
    MATCH implicitConvTo(Type *to);
    MATCH constConv(Type *to);
    Type *toHeadMutable();
    Expression *defaultInit(Loc loc);
    Expression *defaultInitLiteral(Loc loc);
    int isZeroInit(Loc loc);
#if IN_DMD
    dt_t **toDt(dt_t **pdt);
#endif
    MATCH deduceType(Scope *sc, Type *tparam, TemplateParameters *parameters, Objects *dedtypes, unsigned *wildmatch = NULL);
    TypeInfoDeclaration *getTypeInfoDeclaration();
    int hasPointers();
    TypeTuple *toArgTypes();
    int hasWild();
#if CPP_MANGLE
    void toCppMangle(OutBuffer *buf, CppMangleState *cms);
#endif

#if IN_DMD
    type *toCtype();
    type *toCParamtype();
#endif
};

struct TypeClass : Type
{
    ClassDeclaration *sym;
    enum AliasThisRec att;

    TypeClass(ClassDeclaration *sym);
    const char *kind();
    d_uns64 size(Loc loc);
    char *toChars();
    Type *syntaxCopy();
    Type *semantic(Loc loc, Scope *sc);
    Dsymbol *toDsymbol(Scope *sc);
    void toDecoBuffer(OutBuffer *buf, int flag);
    void toCBuffer2(OutBuffer *buf, HdrGenState *hgs, int mod);
    void toJson(JsonOut *json);
    Expression *dotExp(Scope *sc, Expression *e, Identifier *ident, int flag);
    ClassDeclaration *isClassHandle();
    int isBaseOf(Type *t, int *poffset);
    MATCH implicitConvTo(Type *to);
    MATCH constConv(Type *to);
    unsigned wildConvTo(Type *tprm);
    Type *toHeadMutable();
    Expression *defaultInit(Loc loc);
    int isZeroInit(Loc loc);
    MATCH deduceType(Scope *sc, Type *tparam, TemplateParameters *parameters, Objects *dedtypes, unsigned *wildmatch = NULL);
    int isscope();
    int checkBoolean();
    TypeInfoDeclaration *getTypeInfoDeclaration();
    int hasPointers();
    TypeTuple *toArgTypes();
    int builtinTypeInfo();
#if CPP_MANGLE
    void toCppMangle(OutBuffer *buf, CppMangleState *cms);
#endif

#if IN_DMD
    type *toCtype();

    Symbol *toSymbol();
#endif
};

struct TypeTuple : Type
{
    Parameters *arguments;      // types making up the tuple

    TypeTuple(Parameters *arguments);
    TypeTuple(Expressions *exps);
    TypeTuple();
    TypeTuple(Type *t1);
    TypeTuple(Type *t1, Type *t2);
    const char *kind();
    Type *syntaxCopy();
    Type *semantic(Loc loc, Scope *sc);
    int equals(Object *o);
    Type *reliesOnTident(TemplateParameters *tparams = NULL);
    void toCBuffer2(OutBuffer *buf, HdrGenState *hgs, int mod);
    void toDecoBuffer(OutBuffer *buf, int flag);
    void toJson(JsonOut *json);
    Expression *getProperty(Loc loc, Identifier *ident, int flag);
    Expression *defaultInit(Loc loc);
    TypeInfoDeclaration *getTypeInfoDeclaration();
};

struct TypeSlice : TypeNext
{
    Expression *lwr;
    Expression *upr;

    TypeSlice(Type *next, Expression *lwr, Expression *upr);
    const char *kind();
    Type *syntaxCopy();
    Type *semantic(Loc loc, Scope *sc);
    void resolve(Loc loc, Scope *sc, Expression **pe, Type **pt, Dsymbol **ps);
    void toCBuffer2(OutBuffer *buf, HdrGenState *hgs, int mod);
    void toJson(JsonOut *json);
};

struct TypeNull : Type
{
    TypeNull();
    const char *kind();

    Type *syntaxCopy();
    void toDecoBuffer(OutBuffer *buf, int flag);
    MATCH implicitConvTo(Type *to);
    int checkBoolean();

    void toCBuffer(OutBuffer *buf, Identifier *ident, HdrGenState *hgs);
    void toJson(JsonOut *json);

    d_uns64 size(Loc loc);
    Expression *defaultInit(Loc loc);
};

/**************************************************************/

//enum InOut { None, In, Out, InOut, Lazy };

struct Parameter : Object
{
    //enum InOut inout;
    StorageClass storageClass;
    Type *type;
    Identifier *ident;
    Expression *defaultArg;

    Parameter(StorageClass storageClass, Type *type, Identifier *ident, Expression *defaultArg);
    Parameter *syntaxCopy();
    Type *isLazyArray();
    void toDecoBuffer(OutBuffer *buf);
    int dyncast() { return DYNCAST_PARAMETER; } // kludge for template.isType()
    static Parameters *arraySyntaxCopy(Parameters *args);
    static char *argsTypesToChars(Parameters *args, int varargs);
#if CPP_MANGLE
    static void argsCppMangle(OutBuffer *buf, CppMangleState *cms, Parameters *arguments, int varargs);
#endif
    static void argsToCBuffer(OutBuffer *buf, HdrGenState *hgs, Parameters *arguments, int varargs);
    static void argsToDecoBuffer(OutBuffer *buf, Parameters *arguments);
    static int isTPL(Parameters *arguments);
    static size_t dim(Parameters *arguments);
    static Parameter *getNth(Parameters *arguments, size_t nth, size_t *pn = NULL);

    typedef int (*ForeachDg)(void *ctx, size_t paramidx, Parameter *param);
    static int foreach(Parameters *args, ForeachDg dg, void *ctx, size_t *pn=NULL);
};

extern int Tsize_t;
extern int Tptrdiff_t;

int arrayTypeCompatible(Loc loc, Type *t1, Type *t2);
int arrayTypeCompatibleWithoutCasting(Loc loc, Type *t1, Type *t2);
void MODtoBuffer(OutBuffer *buf, unsigned char mod);
char *MODtoChars(unsigned char mod);
int MODimplicitConv(unsigned char modfrom, unsigned char modto);
int MODmethodConv(unsigned char modfrom, unsigned char modto);
int MODmerge(unsigned char mod1, unsigned char mod2);
void identifierToDocBuffer(Identifier* ident, OutBuffer *buf, HdrGenState *hgs);

#endif /* DMD_MTYPE_H */
