
// Compiler implementation of the D programming language
// Copyright (c) 1999-2007 by Digital Mars
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
// llvm
#include "../ir/irdtype.h"
#include "../ir/irfuncty.h"
namespace llvm { class Type; }
struct Ir;

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
enum LINK;

struct TypeBasic;
struct HdrGenState;
struct Argument;

// Back end
#if IN_GCC
union tree_node; typedef union tree_node TYPE;
typedef TYPE type;
#endif

#if IN_DMD
typedef struct TYPE type;
struct Symbol;
#endif

enum TY
{
    Tarray,		// dynamic array
    Tsarray,		// static array
    Taarray,		// associative array
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

    Tbit,
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
    TMAX
};

#define Tascii Tchar

extern int Tsize_t;
extern int Tptrdiff_t;


struct Type : Object
{
    TY ty;
    unsigned char mod;	// modifiers MODxxxx
	/* pick this order of numbers so switch statements work better
	 */
	#define MODconst     1	// type is const
	#define MODinvariant 4	// type is invariant
	#define MODshared    2	// type is shared
    char *deco;
    Type *pto;		// merged pointer to this type
    Type *rto;		// reference to this type
    Type *arrayof;	// array of this type
    TypeInfoDeclaration *vtinfo;	// TypeInfo object for this Type

#if IN_DMD
    type *ctype;	// for back end
#endif

    #define tvoid	basic[Tvoid]
    #define tint8	basic[Tint8]
    #define tuns8	basic[Tuns8]
    #define tint16	basic[Tint16]
    #define tuns16	basic[Tuns16]
    #define tint32	basic[Tint32]
    #define tuns32	basic[Tuns32]
    #define tint64	basic[Tint64]
    #define tuns64	basic[Tuns64]
    #define tfloat32	basic[Tfloat32]
    #define tfloat64	basic[Tfloat64]
    #define tfloat80	basic[Tfloat80]

    #define timaginary32 basic[Timaginary32]
    #define timaginary64 basic[Timaginary64]
    #define timaginary80 basic[Timaginary80]

    #define tcomplex32	basic[Tcomplex32]
    #define tcomplex64	basic[Tcomplex64]
    #define tcomplex80	basic[Tcomplex80]

    #define tbit	basic[Tbit]
    #define tbool	basic[Tbool]
    #define tchar	basic[Tchar]
    #define twchar	basic[Twchar]
    #define tdchar	basic[Tdchar]

    // Some special types
    #define tshiftcnt	tint32		// right side of shift expression
//    #define tboolean	tint32		// result of boolean expression
    #define tboolean	tbool		// result of boolean expression
    #define tindex	tint32		// array/ptr index
    static Type *tvoidptr;		// void*
    #define terror	basic[Terror]	// for error recovery

    #define tsize_t	basic[Tsize_t]		// matches size_t alias
    #define tptrdiff_t	basic[Tptrdiff_t]	// matches ptrdiff_t alias
    #define thash_t	tsize_t			// matches hash_t alias

    static ClassDeclaration *typeinfo;
    static ClassDeclaration *typeinfoclass;
    static ClassDeclaration *typeinfointerface;
    static ClassDeclaration *typeinfostruct;
    static ClassDeclaration *typeinfotypedef;
    static ClassDeclaration *typeinfopointer;
    static ClassDeclaration *typeinfoarray;
    static ClassDeclaration *typeinfostaticarray;
    static ClassDeclaration *typeinfoassociativearray;
    static ClassDeclaration *typeinfoenum;
    static ClassDeclaration *typeinfofunction;
    static ClassDeclaration *typeinfodelegate;
    static ClassDeclaration *typeinfotypelist;

    static Type *basic[TMAX];
    static unsigned char mangleChar[TMAX];
    static StringTable stringtable;

    // These tables are for implicit conversion of binary ops;
    // the indices are the type of operand one, followed by operand two.
    static unsigned char impcnvResult[TMAX][TMAX];
    static unsigned char impcnvType1[TMAX][TMAX];
    static unsigned char impcnvType2[TMAX][TMAX];

    // If !=0, give warning on implicit conversion
    static unsigned char impcnvWarn[TMAX][TMAX];

    Type(TY ty, Type *next);
    virtual Type *syntaxCopy();
    int equals(Object *o);
    int dyncast() { return DYNCAST_TYPE; } // kludge for template.isType()
    int covariant(Type *t);
    char *toChars();
    static char needThisPrefix();
#if IN_LLVM
    static void init(Ir*);
#else
    static void init();
#endif
    d_uns64 size();
    virtual d_uns64 size(Loc loc);
    virtual unsigned alignsize();
    virtual Type *semantic(Loc loc, Scope *sc);
    // append the mangleof or a string uniquely identifying this type to buf
    virtual void toDecoBuffer(OutBuffer *buf, bool mangle);
    Type *merge();
    virtual void toCBuffer(OutBuffer *buf, Identifier *ident, HdrGenState *hgs);
    virtual void toCBuffer2(OutBuffer *buf, HdrGenState *hgs, int mod);
    void toCBuffer3(OutBuffer *buf, HdrGenState *hgs, int mod);
    virtual int isbit();
    virtual int isintegral();
    virtual int isfloating();	// real, imaginary, or complex
    virtual int isreal();
    virtual int isimaginary();
    virtual int iscomplex();
    virtual int isscalar();
    virtual int isunsigned();
    virtual int isauto();
    virtual int isString();
    virtual int checkBoolean();	// if can be converted to boolean value
    void checkDeprecated(Loc loc, Scope *sc);
    Type *pointerTo();
    Type *referenceTo();
    Type *arrayOf();
    virtual Dsymbol *toDsymbol(Scope *sc);
    virtual Type *toBasetype();
    virtual int isBaseOf(Type *t, int *poffset);
    virtual MATCH implicitConvTo(Type *to);
    virtual ClassDeclaration *isClassHandle();
    virtual Expression *getProperty(Loc loc, Identifier *ident);
    virtual Expression *dotExp(Scope *sc, Expression *e, Identifier *ident);
    virtual unsigned memalign(unsigned salign);
    virtual Expression *defaultInit(Loc loc = 0);
    virtual int isZeroInit();		// if initializer is 0
#if IN_DMD
    virtual dt_t **toDt(dt_t **pdt);
#endif
    Identifier *getTypeInfoIdent(int internal);
    virtual MATCH deduceType(Scope *sc, Type *tparam, TemplateParameters *parameters, Objects *dedtypes);
    virtual void resolve(Loc loc, Scope *sc, Expression **pe, Type **pt, Dsymbol **ps);
    Expression *getInternalTypeInfo(Scope *sc);
    Expression *getTypeInfo(Scope *sc);
    virtual TypeInfoDeclaration *getTypeInfoDeclaration();
    virtual int builtinTypeInfo();
    virtual Type *reliesOnTident();
    virtual Expression *toExpression();
    virtual int hasPointers();
    Type *next;
    Type *nextOf() { return next; }

    static void error(Loc loc, const char *format, ...);
    static void warning(Loc loc, const char *format, ...);

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
    // LDC
    IrDType ir;
    static Ir* sir;

    IrType* irtype;
#endif
};

struct TypeBasic : Type
{
    const char *dstring;
    const char *cstring;
    unsigned flags;

    TypeBasic(TY ty);
    Type *syntaxCopy();
    d_uns64 size(Loc loc);
    unsigned alignsize();
    Expression *getProperty(Loc loc, Identifier *ident);
    Expression *dotExp(Scope *sc, Expression *e, Identifier *ident);
    char *toChars();
    void toCBuffer2(OutBuffer *buf, HdrGenState *hgs, int mod);
    int isintegral();
    int isbit();
    int isfloating();
    int isreal();
    int isimaginary();
    int iscomplex();
    int isscalar();
    int isunsigned();
    MATCH implicitConvTo(Type *to);
    Expression *defaultInit(Loc loc);
    int isZeroInit();
    int builtinTypeInfo();

    // For eliminating dynamic_cast
    TypeBasic *isTypeBasic();
};

struct TypeArray : Type
{
    TypeArray(TY ty, Type *next);
    Expression *dotExp(Scope *sc, Expression *e, Identifier *ident);
};

// Static array, one with a fixed dimension
struct TypeSArray : TypeArray
{
    Expression *dim;

    TypeSArray(Type *t, Expression *dim);
    Type *syntaxCopy();
    d_uns64 size(Loc loc);
    unsigned alignsize();
    Type *semantic(Loc loc, Scope *sc);
    void resolve(Loc loc, Scope *sc, Expression **pe, Type **pt, Dsymbol **ps);
    void toDecoBuffer(OutBuffer *buf, bool mangle);
    void toCBuffer2(OutBuffer *buf, HdrGenState *hgs, int mod);
    Expression *dotExp(Scope *sc, Expression *e, Identifier *ident);
    int isString();
    int isZeroInit();
    unsigned memalign(unsigned salign);
    MATCH implicitConvTo(Type *to);
    Expression *defaultInit(Loc loc);
#if IN_DMD
    dt_t **toDt(dt_t **pdt);
    dt_t **toDtElem(dt_t **pdt, Expression *e);
#endif
    MATCH deduceType(Scope *sc, Type *tparam, TemplateParameters *parameters, Objects *dedtypes);
    TypeInfoDeclaration *getTypeInfoDeclaration();
    Expression *toExpression();
    int hasPointers();

#if IN_DMD
    type *toCtype();
    type *toCParamtype();
#endif
};

// Dynamic array, no dimension
struct TypeDArray : TypeArray
{
    TypeDArray(Type *t);
    Type *syntaxCopy();
    d_uns64 size(Loc loc);
    unsigned alignsize();
    Type *semantic(Loc loc, Scope *sc);
    void toDecoBuffer(OutBuffer *buf, bool mangle);
    void toCBuffer2(OutBuffer *buf, HdrGenState *hgs, int mod);
    Expression *dotExp(Scope *sc, Expression *e, Identifier *ident);
    int isString();
    int isZeroInit();
    int checkBoolean();
    MATCH implicitConvTo(Type *to);
    Expression *defaultInit(Loc loc);
    int builtinTypeInfo();
    TypeInfoDeclaration *getTypeInfoDeclaration();
    int hasPointers();

#if IN_DMD
    type *toCtype();
#endif
};

struct TypeAArray : TypeArray
{
    Type *index;		// key type for type checking
    Type *key;			// actual key type

    TypeAArray(Type *t, Type *index);
    Type *syntaxCopy();
    d_uns64 size(Loc loc);
    Type *semantic(Loc loc, Scope *sc);
    void resolve(Loc loc, Scope *sc, Expression **pe, Type **pt, Dsymbol **ps);
    void toDecoBuffer(OutBuffer *buf, bool mangle);
    void toCBuffer2(OutBuffer *buf, HdrGenState *hgs, int mod);
    Expression *dotExp(Scope *sc, Expression *e, Identifier *ident);
    Expression *defaultInit(Loc loc);
    MATCH deduceType(Scope *sc, Type *tparam, TemplateParameters *parameters, Objects *dedtypes);
    int isZeroInit();
    int checkBoolean();
    TypeInfoDeclaration *getTypeInfoDeclaration();
    int hasPointers();

#if IN_DMD
    // Back end
    Symbol *aaGetSymbol(const char *func, int flags);

    type *toCtype();
#endif
};

struct TypePointer : Type
{
    TypePointer(Type *t);
    Type *syntaxCopy();
    Type *semantic(Loc loc, Scope *sc);
    d_uns64 size(Loc loc);
    void toCBuffer2(OutBuffer *buf, HdrGenState *hgs, int mod);
    MATCH implicitConvTo(Type *to);
    int isscalar();
    // LDC: pointers are unsigned
    int isunsigned() { return TRUE; };
    Expression *defaultInit(Loc loc);
    int isZeroInit();
    TypeInfoDeclaration *getTypeInfoDeclaration();
    int hasPointers();

#if IN_DMD
    type *toCtype();
#endif
};

struct TypeReference : Type
{
    TypeReference(Type *t);
    Type *syntaxCopy();
    d_uns64 size(Loc loc);
    void toCBuffer2(OutBuffer *buf, HdrGenState *hgs, int mod);
    Expression *dotExp(Scope *sc, Expression *e, Identifier *ident);
    Expression *defaultInit(Loc loc);
    int isZeroInit();
};

enum RET
{
    RETregs	= 1,	// returned in registers
    RETstack	= 2,	// returned on stack
};

struct TypeFunction : Type
{
    Arguments *parameters;	// function parameters
    int varargs;	// 1: T t, ...) style for variable number of arguments
			// 2: T t ...) style for variable number of arguments
    enum LINK linkage;	// calling convention

    int inuse;

    TypeFunction(Arguments *parameters, Type *treturn, int varargs, enum LINK linkage);
    Type *syntaxCopy();
    Type *semantic(Loc loc, Scope *sc);
    void toDecoBuffer(OutBuffer *buf, bool mangle);
    void toCBuffer(OutBuffer *buf, Identifier *ident, HdrGenState *hgs);
    void toCBuffer2(OutBuffer *buf, HdrGenState *hgs, int mod);
    MATCH deduceType(Scope *sc, Type *tparam, TemplateParameters *parameters, Objects *dedtypes);
    TypeInfoDeclaration *getTypeInfoDeclaration();
    Type *reliesOnTident();

    int callMatch(Expressions *toargs);
#if IN_DMD
    type *toCtype();
#endif

    enum RET retStyle();

#if IN_DMD
    unsigned totym();
#elif IN_LLVM
    // LDC
    IrFuncTy fty;

    FuncDeclaration* funcdecl;
#endif
};

struct TypeDelegate : Type
{
    TypeDelegate(Type *t);
    Type *syntaxCopy();
    Type *semantic(Loc loc, Scope *sc);
    d_uns64 size(Loc loc);
    unsigned alignsize(); // added in LDC
    void toCBuffer2(OutBuffer *buf, HdrGenState *hgs, int mod);
    Expression *defaultInit(Loc loc);
    int isZeroInit();
    int checkBoolean();
    TypeInfoDeclaration *getTypeInfoDeclaration();
    Expression *dotExp(Scope *sc, Expression *e, Identifier *ident);
    int hasPointers();

#if IN_DMD
    type *toCtype();
#endif
};

struct TypeQualified : Type
{
    Loc loc;
    Array idents;	// array of Identifier's representing ident.ident.ident etc.

    TypeQualified(TY ty, Loc loc);
    void syntaxCopyHelper(TypeQualified *t);
    void addIdent(Identifier *ident);
    void toCBuffer2Helper(OutBuffer *buf, HdrGenState *hgs);
    d_uns64 size(Loc loc);
    void resolveHelper(Loc loc, Scope *sc, Dsymbol *s, Dsymbol *scopesym,
	Expression **pe, Type **pt, Dsymbol **ps);
};

struct TypeIdentifier : TypeQualified
{
    Identifier *ident;

    TypeIdentifier(Loc loc, Identifier *ident);
    Type *syntaxCopy();
    //char *toChars();
    void toDecoBuffer(OutBuffer *buf, bool mangle);
    void toCBuffer2(OutBuffer *buf, HdrGenState *hgs, int mod);
    void resolve(Loc loc, Scope *sc, Expression **pe, Type **pt, Dsymbol **ps);
    Dsymbol *toDsymbol(Scope *sc);
    Type *semantic(Loc loc, Scope *sc);
    MATCH deduceType(Scope *sc, Type *tparam, TemplateParameters *parameters, Objects *dedtypes);
    Type *reliesOnTident();
    Expression *toExpression();
};

/* Similar to TypeIdentifier, but with a TemplateInstance as the root
 */
struct TypeInstance : TypeQualified
{
    TemplateInstance *tempinst;

    TypeInstance(Loc loc, TemplateInstance *tempinst);
    Type *syntaxCopy();
    //char *toChars();
    //void toDecoBuffer(OutBuffer *buf, bool mangle);
    void toCBuffer2(OutBuffer *buf, HdrGenState *hgs, int mod);
    void resolve(Loc loc, Scope *sc, Expression **pe, Type **pt, Dsymbol **ps);
    Type *semantic(Loc loc, Scope *sc);
    Dsymbol *toDsymbol(Scope *sc);
    MATCH deduceType(Scope *sc, Type *tparam, TemplateParameters *parameters, Objects *dedtypes);
};

struct TypeTypeof : TypeQualified
{
    Expression *exp;

    TypeTypeof(Loc loc, Expression *exp);
    Type *syntaxCopy();
    Dsymbol *toDsymbol(Scope *sc);
    void toCBuffer2(OutBuffer *buf, HdrGenState *hgs, int mod);
    Type *semantic(Loc loc, Scope *sc);
    d_uns64 size(Loc loc);
};

struct TypeStruct : Type
{
    StructDeclaration *sym;

    TypeStruct(StructDeclaration *sym);
    d_uns64 size(Loc loc);
    unsigned alignsize();
    char *toChars();
    Type *syntaxCopy();
    Type *semantic(Loc loc, Scope *sc);
    Dsymbol *toDsymbol(Scope *sc);
    void toDecoBuffer(OutBuffer *buf, bool mangle);
    void toCBuffer2(OutBuffer *buf, HdrGenState *hgs, int mod);
    Expression *dotExp(Scope *sc, Expression *e, Identifier *ident);
    unsigned memalign(unsigned salign);
    Expression *defaultInit(Loc loc);
    int isZeroInit();
    int checkBoolean();
#if IN_DMD
    dt_t **toDt(dt_t **pdt);
#endif
    MATCH deduceType(Scope *sc, Type *tparam, TemplateParameters *parameters, Objects *dedtypes);
    TypeInfoDeclaration *getTypeInfoDeclaration();
    int hasPointers();

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
    d_uns64 size(Loc loc);
    unsigned alignsize();
    char *toChars();
    Type *syntaxCopy();
    Type *semantic(Loc loc, Scope *sc);
    Dsymbol *toDsymbol(Scope *sc);
    void toDecoBuffer(OutBuffer *buf, bool mangle);
    void toCBuffer2(OutBuffer *buf, HdrGenState *hgs, int mod);
    Expression *dotExp(Scope *sc, Expression *e, Identifier *ident);
    Expression *getProperty(Loc loc, Identifier *ident);
    int isintegral();
    int isfloating();
    int isscalar();
    int isunsigned();
    MATCH implicitConvTo(Type *to);
    Type *toBasetype();
    Expression *defaultInit(Loc loc);
    int isZeroInit();
    MATCH deduceType(Scope *sc, Type *tparam, TemplateParameters *parameters, Objects *dedtypes);
    TypeInfoDeclaration *getTypeInfoDeclaration();
    int hasPointers();

#if IN_DMD
    type *toCtype();
#endif
};

struct TypeTypedef : Type
{
    TypedefDeclaration *sym;

    TypeTypedef(TypedefDeclaration *sym);
    Type *syntaxCopy();
    d_uns64 size(Loc loc);
    unsigned alignsize();
    char *toChars();
    Type *semantic(Loc loc, Scope *sc);
    Dsymbol *toDsymbol(Scope *sc);
    void toDecoBuffer(OutBuffer *buf, bool mangle);
    void toCBuffer2(OutBuffer *buf, HdrGenState *hgs, int mod);
    Expression *dotExp(Scope *sc, Expression *e, Identifier *ident);
    Expression *getProperty(Loc loc, Identifier *ident);
    int isbit();
    int isintegral();
    int isfloating();
    int isreal();
    int isimaginary();
    int iscomplex();
    int isscalar();
    int isunsigned();
    int checkBoolean();
    Type *toBasetype();
    MATCH implicitConvTo(Type *to);
    Expression *defaultInit(Loc loc);
    int isZeroInit();
#if IN_DMD
    dt_t **toDt(dt_t **pdt);
#endif
    MATCH deduceType(Scope *sc, Type *tparam, TemplateParameters *parameters, Objects *dedtypes);
    TypeInfoDeclaration *getTypeInfoDeclaration();
    int hasPointers();

#if IN_DMD
    type *toCtype();
    type *toCParamtype();
#endif
};

struct TypeClass : Type
{
    ClassDeclaration *sym;

    TypeClass(ClassDeclaration *sym);
    d_uns64 size(Loc loc);
    char *toChars();
    Type *syntaxCopy();
    Type *semantic(Loc loc, Scope *sc);
    Dsymbol *toDsymbol(Scope *sc);
    void toDecoBuffer(OutBuffer *buf, bool mangle);
    void toCBuffer2(OutBuffer *buf, HdrGenState *hgs, int mod);
    Expression *dotExp(Scope *sc, Expression *e, Identifier *ident);
    ClassDeclaration *isClassHandle();
    int isBaseOf(Type *t, int *poffset);
    MATCH implicitConvTo(Type *to);
    Expression *defaultInit(Loc loc);
    int isZeroInit();
    MATCH deduceType(Scope *sc, Type *tparam, TemplateParameters *parameters, Objects *dedtypes);
    int isauto();
    int checkBoolean();
    TypeInfoDeclaration *getTypeInfoDeclaration();
    int hasPointers();

#if IN_DMD
    type *toCtype();

    Symbol *toSymbol();
#endif
};

struct TypeTuple : Type
{
    Arguments *arguments;	// types making up the tuple

    TypeTuple(Arguments *arguments);
    TypeTuple(Expressions *exps);
    Type *syntaxCopy();
    Type *semantic(Loc loc, Scope *sc);
    int equals(Object *o);
    Type *reliesOnTident();
    void toCBuffer2(OutBuffer *buf, HdrGenState *hgs, int mod);
    void toDecoBuffer(OutBuffer *buf, bool mangle);
    Expression *getProperty(Loc loc, Identifier *ident);
    TypeInfoDeclaration *getTypeInfoDeclaration();
};

struct TypeSlice : Type
{
    Expression *lwr;
    Expression *upr;

    TypeSlice(Type *next, Expression *lwr, Expression *upr);
    Type *syntaxCopy();
    Type *semantic(Loc loc, Scope *sc);
    void resolve(Loc loc, Scope *sc, Expression **pe, Type **pt, Dsymbol **ps);
    void toCBuffer2(OutBuffer *buf, HdrGenState *hgs, int mod);
};

/**************************************************************/

//enum InOut { None, In, Out, InOut, Lazy };

struct Argument : Object
{
    //enum InOut inout;
    unsigned storageClass;
    Type *type;
    Identifier *ident;
    Expression *defaultArg;

    Argument(unsigned storageClass, Type *type, Identifier *ident, Expression *defaultArg);
    Argument *syntaxCopy();
    Type *isLazyArray();
    void toDecoBuffer(OutBuffer *buf, bool mangle);
    static Arguments *arraySyntaxCopy(Arguments *args);
    static char *argsTypesToChars(Arguments *args, int varargs);
    static void argsToCBuffer(OutBuffer *buf, HdrGenState *hgs, Arguments *arguments, int varargs);
    static void argsToDecoBuffer(OutBuffer *buf, Arguments *arguments, bool mangle);
    static size_t dim(Arguments *arguments);
    static Argument *getNth(Arguments *arguments, size_t nth, size_t *pn = NULL);
};

extern int PTRSIZE;
extern int REALSIZE;
extern int REALPAD;
extern int Tsize_t;
extern int Tptrdiff_t;

#endif /* DMD_MTYPE_H */
