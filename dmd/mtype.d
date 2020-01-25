/**
 * Compiler implementation of the
 * $(LINK2 http://www.dlang.org, D programming language).
 *
 * Copyright:   Copyright (C) 1999-2019 by The D Language Foundation, All Rights Reserved
 * Authors:     $(LINK2 http://www.digitalmars.com, Walter Bright)
 * License:     $(LINK2 http://www.boost.org/LICENSE_1_0.txt, Boost License 1.0)
 * Source:      $(LINK2 https://github.com/dlang/dmd/blob/master/src/dmd/mtype.d, _mtype.d)
 * Documentation:  https://dlang.org/phobos/dmd_mtype.html
 * Coverage:    https://codecov.io/gh/dlang/dmd/src/master/src/dmd/mtype.d
 */

module dmd.mtype;

import core.checkedint;
import core.stdc.stdarg;
import core.stdc.stdio;
import core.stdc.stdlib;
import core.stdc.string;

import dmd.aggregate;
import dmd.arraytypes;
import dmd.attrib;
import dmd.ast_node;
import dmd.gluelayer;
import dmd.dclass;
import dmd.declaration;
import dmd.denum;
import dmd.dmangle;
import dmd.dscope;
import dmd.dstruct;
import dmd.dsymbol;
import dmd.dsymbolsem;
import dmd.dtemplate;
import dmd.errors;
import dmd.expression;
import dmd.expressionsem;
import dmd.func;
import dmd.globals;
import dmd.hdrgen;
import dmd.id;
import dmd.identifier;
import dmd.init;
import dmd.opover;
import dmd.root.ctfloat;
import dmd.root.outbuffer;
import dmd.root.rmem;
import dmd.root.rootobject;
import dmd.root.stringtable;
import dmd.target;
import dmd.tokens;
import dmd.typesem;
import dmd.visitor;

version (IN_LLVM) import gen.llvmhelpers;

enum LOGDOTEXP = 0;         // log ::dotExp()
enum LOGDEFAULTINIT = 0;    // log ::defaultInit()

enum SIZE_INVALID = (~cast(d_uns64)0);   // error return from size() functions


/***************************
 * Return !=0 if modfrom can be implicitly converted to modto
 */
bool MODimplicitConv(MOD modfrom, MOD modto) pure nothrow @nogc @safe
{
    if (modfrom == modto)
        return true;

    //printf("MODimplicitConv(from = %x, to = %x)\n", modfrom, modto);
    auto X(T, U)(T m, U n)
    {
        return ((m << 4) | n);
    }

    switch (X(modfrom & ~MODFlags.shared_, modto & ~MODFlags.shared_))
    {
    case X(0, MODFlags.const_):
    case X(MODFlags.wild, MODFlags.const_):
    case X(MODFlags.wild, MODFlags.wildconst):
    case X(MODFlags.wildconst, MODFlags.const_):
        return (modfrom & MODFlags.shared_) == (modto & MODFlags.shared_);

    case X(MODFlags.immutable_, MODFlags.const_):
    case X(MODFlags.immutable_, MODFlags.wildconst):
        return true;
    default:
        return false;
    }
}

/***************************
 * Return MATCH.exact or MATCH.constant if a method of type '() modfrom' can call a method of type '() modto'.
 */
MATCH MODmethodConv(MOD modfrom, MOD modto) pure nothrow @nogc @safe
{
    if (modfrom == modto)
        return MATCH.exact;
    if (MODimplicitConv(modfrom, modto))
        return MATCH.constant;

    auto X(T, U)(T m, U n)
    {
        return ((m << 4) | n);
    }

    switch (X(modfrom, modto))
    {
    case X(0, MODFlags.wild):
    case X(MODFlags.immutable_, MODFlags.wild):
    case X(MODFlags.const_, MODFlags.wild):
    case X(MODFlags.wildconst, MODFlags.wild):
    case X(MODFlags.shared_, MODFlags.shared_ | MODFlags.wild):
    case X(MODFlags.shared_ | MODFlags.immutable_, MODFlags.shared_ | MODFlags.wild):
    case X(MODFlags.shared_ | MODFlags.const_, MODFlags.shared_ | MODFlags.wild):
    case X(MODFlags.shared_ | MODFlags.wildconst, MODFlags.shared_ | MODFlags.wild):
        return MATCH.constant;

    default:
        return MATCH.nomatch;
    }
}

/***************************
 * Merge mod bits to form common mod.
 */
MOD MODmerge(MOD mod1, MOD mod2) pure nothrow @nogc @safe
{
    if (mod1 == mod2)
        return mod1;

    //printf("MODmerge(1 = %x, 2 = %x)\n", mod1, mod2);
    MOD result = 0;
    if ((mod1 | mod2) & MODFlags.shared_)
    {
        // If either type is shared, the result will be shared
        result |= MODFlags.shared_;
        mod1 &= ~MODFlags.shared_;
        mod2 &= ~MODFlags.shared_;
    }
    if (mod1 == 0 || mod1 == MODFlags.mutable || mod1 == MODFlags.const_ || mod2 == 0 || mod2 == MODFlags.mutable || mod2 == MODFlags.const_)
    {
        // If either type is mutable or const, the result will be const.
        result |= MODFlags.const_;
    }
    else
    {
        // MODFlags.immutable_ vs MODFlags.wild
        // MODFlags.immutable_ vs MODFlags.wildconst
        //      MODFlags.wild vs MODFlags.wildconst
        assert(mod1 & MODFlags.wild || mod2 & MODFlags.wild);
        result |= MODFlags.wildconst;
    }
    return result;
}

/*********************************
 * Store modifier name into buf.
 */
void MODtoBuffer(OutBuffer* buf, MOD mod) nothrow
{
    buf.writestring(MODtoString(mod));
}

/*********************************
 * Returns:
 *   a human readable representation of `mod`,
 *   which is the token `mod` corresponds to
 */
const(char)* MODtoChars(MOD mod) nothrow pure
{
    /// Works because we return a literal
    return MODtoString(mod).ptr;
}

/// Ditto
string MODtoString(MOD mod) nothrow pure
{
    final switch (mod)
    {
    case 0:
        return "";

    case MODFlags.immutable_:
        return "immutable";

    case MODFlags.shared_:
        return "shared";

    case MODFlags.shared_ | MODFlags.const_:
        return "shared const";

    case MODFlags.const_:
        return "const";

    case MODFlags.shared_ | MODFlags.wild:
        return "shared inout";

    case MODFlags.wild:
        return "inout";

    case MODFlags.shared_ | MODFlags.wildconst:
        return "shared inout const";

    case MODFlags.wildconst:
        return "inout const";
    }
}


/************************************
 * Convert MODxxxx to STCxxx
 */
StorageClass ModToStc(uint mod) pure nothrow @nogc @safe
{
    StorageClass stc = 0;
    if (mod & MODFlags.immutable_)
        stc |= STC.immutable_;
    if (mod & MODFlags.const_)
        stc |= STC.const_;
    if (mod & MODFlags.wild)
        stc |= STC.wild;
    if (mod & MODFlags.shared_)
        stc |= STC.shared_;
    return stc;
}

private enum TFlags
{
    integral     = 1,
    floating     = 2,
    unsigned     = 4,
    real_        = 8,
    imaginary    = 0x10,
    complex      = 0x20,
    char_        = 0x40,
}

enum ENUMTY : int
{
    Tarray,     // slice array, aka T[]
    Tsarray,    // static array, aka T[dimension]
    Taarray,    // associative array, aka T[type]
    Tpointer,
    Treference,
    Tfunction,
    Tident,
    Tclass,
    Tstruct,
    Tenum,

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
    TTraits,
    Tmixin,
    TMAX,
}

alias Tarray = ENUMTY.Tarray;
alias Tsarray = ENUMTY.Tsarray;
alias Taarray = ENUMTY.Taarray;
alias Tpointer = ENUMTY.Tpointer;
alias Treference = ENUMTY.Treference;
alias Tfunction = ENUMTY.Tfunction;
alias Tident = ENUMTY.Tident;
alias Tclass = ENUMTY.Tclass;
alias Tstruct = ENUMTY.Tstruct;
alias Tenum = ENUMTY.Tenum;
alias Tdelegate = ENUMTY.Tdelegate;
alias Tnone = ENUMTY.Tnone;
alias Tvoid = ENUMTY.Tvoid;
alias Tint8 = ENUMTY.Tint8;
alias Tuns8 = ENUMTY.Tuns8;
alias Tint16 = ENUMTY.Tint16;
alias Tuns16 = ENUMTY.Tuns16;
alias Tint32 = ENUMTY.Tint32;
alias Tuns32 = ENUMTY.Tuns32;
alias Tint64 = ENUMTY.Tint64;
alias Tuns64 = ENUMTY.Tuns64;
alias Tfloat32 = ENUMTY.Tfloat32;
alias Tfloat64 = ENUMTY.Tfloat64;
alias Tfloat80 = ENUMTY.Tfloat80;
alias Timaginary32 = ENUMTY.Timaginary32;
alias Timaginary64 = ENUMTY.Timaginary64;
alias Timaginary80 = ENUMTY.Timaginary80;
alias Tcomplex32 = ENUMTY.Tcomplex32;
alias Tcomplex64 = ENUMTY.Tcomplex64;
alias Tcomplex80 = ENUMTY.Tcomplex80;
alias Tbool = ENUMTY.Tbool;
alias Tchar = ENUMTY.Tchar;
alias Twchar = ENUMTY.Twchar;
alias Tdchar = ENUMTY.Tdchar;
alias Terror = ENUMTY.Terror;
alias Tinstance = ENUMTY.Tinstance;
alias Ttypeof = ENUMTY.Ttypeof;
alias Ttuple = ENUMTY.Ttuple;
alias Tslice = ENUMTY.Tslice;
alias Treturn = ENUMTY.Treturn;
alias Tnull = ENUMTY.Tnull;
alias Tvector = ENUMTY.Tvector;
alias Tint128 = ENUMTY.Tint128;
alias Tuns128 = ENUMTY.Tuns128;
alias Ttraits = ENUMTY.TTraits;
alias Tmixin = ENUMTY.Tmixin;
alias TMAX = ENUMTY.TMAX;

alias TY = ubyte;

enum MODFlags : int
{
    const_       = 1,    // type is const
    immutable_   = 4,    // type is immutable
    shared_      = 2,    // type is shared
    wild         = 8,    // type is wild
    wildconst    = (MODFlags.wild | MODFlags.const_), // type is wild const
    mutable      = 0x10, // type is mutable (only used in wildcard matching)
}

alias MOD = ubyte;

/****************
 * dotExp() bit flags
 */
enum DotExpFlag
{
    gag     = 1,    // don't report "not a property" error and just return null
    noDeref = 2,    // the use of the expression will not attempt a dereference
}

/***************
 * Variadic argument lists
 * https://dlang.org/spec/function.html#variadic
 */
enum VarArg
{
    none     = 0,  /// fixed number of arguments
    variadic = 1,  /// (T t, ...)  can be C-style (core.stdc.stdarg) or D-style (core.vararg)
    typesafe = 2,  /// (T t ...) typesafe https://dlang.org/spec/function.html#typesafe_variadic_functions
                   ///   or https://dlang.org/spec/function.html#typesafe_variadic_functions
}


/***********************************************************
 */
extern (C++) abstract class Type : ASTNode
{
    TY ty;
    MOD mod; // modifiers MODxxxx
    char* deco;

    /* These are cached values that are lazily evaluated by constOf(), immutableOf(), etc.
     * They should not be referenced by anybody but mtype.c.
     * They can be NULL if not lazily evaluated yet.
     * Note that there is no "shared immutable", because that is just immutable
     * Naked == no MOD bits
     */
    Type cto;       // MODFlags.const_                 ? naked version of this type : const version
    Type ito;       // MODFlags.immutable_             ? naked version of this type : immutable version
    Type sto;       // MODFlags.shared_                ? naked version of this type : shared mutable version
    Type scto;      // MODFlags.shared_ | MODFlags.const_     ? naked version of this type : shared const version
    Type wto;       // MODFlags.wild                  ? naked version of this type : wild version
    Type wcto;      // MODFlags.wildconst             ? naked version of this type : wild const version
    Type swto;      // MODFlags.shared_ | MODFlags.wild      ? naked version of this type : shared wild version
    Type swcto;     // MODFlags.shared_ | MODFlags.wildconst ? naked version of this type : shared wild const version

    Type pto;       // merged pointer to this type
    Type rto;       // reference to this type
    Type arrayof;   // array of this type

    TypeInfoDeclaration vtinfo;     // TypeInfo object for this Type

    type* ctype;                    // for back end

    extern (C++) __gshared Type tvoid;
    extern (C++) __gshared Type tint8;
    extern (C++) __gshared Type tuns8;
    extern (C++) __gshared Type tint16;
    extern (C++) __gshared Type tuns16;
    extern (C++) __gshared Type tint32;
    extern (C++) __gshared Type tuns32;
    extern (C++) __gshared Type tint64;
    extern (C++) __gshared Type tuns64;
    extern (C++) __gshared Type tint128;
    extern (C++) __gshared Type tuns128;
    extern (C++) __gshared Type tfloat32;
    extern (C++) __gshared Type tfloat64;
    extern (C++) __gshared Type tfloat80;
    extern (C++) __gshared Type timaginary32;
    extern (C++) __gshared Type timaginary64;
    extern (C++) __gshared Type timaginary80;
    extern (C++) __gshared Type tcomplex32;
    extern (C++) __gshared Type tcomplex64;
    extern (C++) __gshared Type tcomplex80;
    extern (C++) __gshared Type tbool;
    extern (C++) __gshared Type tchar;
    extern (C++) __gshared Type twchar;
    extern (C++) __gshared Type tdchar;

    // Some special types
    extern (C++) __gshared Type tshiftcnt;
    extern (C++) __gshared Type tvoidptr;    // void*
    extern (C++) __gshared Type tstring;     // immutable(char)[]
    extern (C++) __gshared Type twstring;    // immutable(wchar)[]
    extern (C++) __gshared Type tdstring;    // immutable(dchar)[]
    extern (C++) __gshared Type tvalist;     // va_list alias
    extern (C++) __gshared Type terror;      // for error recovery
    extern (C++) __gshared Type tnull;       // for null type

    extern (C++) __gshared Type tsize_t;     // matches size_t alias
    extern (C++) __gshared Type tptrdiff_t;  // matches ptrdiff_t alias
    extern (C++) __gshared Type thash_t;     // matches hash_t alias

    extern (C++) __gshared ClassDeclaration dtypeinfo;
    extern (C++) __gshared ClassDeclaration typeinfoclass;
    extern (C++) __gshared ClassDeclaration typeinfointerface;
    extern (C++) __gshared ClassDeclaration typeinfostruct;
    extern (C++) __gshared ClassDeclaration typeinfopointer;
    extern (C++) __gshared ClassDeclaration typeinfoarray;
    extern (C++) __gshared ClassDeclaration typeinfostaticarray;
    extern (C++) __gshared ClassDeclaration typeinfoassociativearray;
    extern (C++) __gshared ClassDeclaration typeinfovector;
    extern (C++) __gshared ClassDeclaration typeinfoenum;
    extern (C++) __gshared ClassDeclaration typeinfofunction;
    extern (C++) __gshared ClassDeclaration typeinfodelegate;
    extern (C++) __gshared ClassDeclaration typeinfotypelist;
    extern (C++) __gshared ClassDeclaration typeinfoconst;
    extern (C++) __gshared ClassDeclaration typeinfoinvariant;
    extern (C++) __gshared ClassDeclaration typeinfoshared;
    extern (C++) __gshared ClassDeclaration typeinfowild;

    extern (C++) __gshared TemplateDeclaration rtinfo;
version (IN_LLVM)
{
    extern (C++) __gshared TemplateDeclaration rtinfoImpl;
}

    extern (C++) __gshared Type[TMAX] basic;

    extern (D) __gshared StringTable!Type stringtable;
    extern (D) private __gshared ubyte[TMAX] sizeTy = ()
        {
            ubyte[TMAX] sizeTy = __traits(classInstanceSize, TypeBasic);
            sizeTy[Tsarray] = __traits(classInstanceSize, TypeSArray);
            sizeTy[Tarray] = __traits(classInstanceSize, TypeDArray);
            sizeTy[Taarray] = __traits(classInstanceSize, TypeAArray);
            sizeTy[Tpointer] = __traits(classInstanceSize, TypePointer);
            sizeTy[Treference] = __traits(classInstanceSize, TypeReference);
            sizeTy[Tfunction] = __traits(classInstanceSize, TypeFunction);
            sizeTy[Tdelegate] = __traits(classInstanceSize, TypeDelegate);
            sizeTy[Tident] = __traits(classInstanceSize, TypeIdentifier);
            sizeTy[Tinstance] = __traits(classInstanceSize, TypeInstance);
            sizeTy[Ttypeof] = __traits(classInstanceSize, TypeTypeof);
            sizeTy[Tenum] = __traits(classInstanceSize, TypeEnum);
            sizeTy[Tstruct] = __traits(classInstanceSize, TypeStruct);
            sizeTy[Tclass] = __traits(classInstanceSize, TypeClass);
            sizeTy[Ttuple] = __traits(classInstanceSize, TypeTuple);
            sizeTy[Tslice] = __traits(classInstanceSize, TypeSlice);
            sizeTy[Treturn] = __traits(classInstanceSize, TypeReturn);
            sizeTy[Terror] = __traits(classInstanceSize, TypeError);
            sizeTy[Tnull] = __traits(classInstanceSize, TypeNull);
            sizeTy[Tvector] = __traits(classInstanceSize, TypeVector);
            sizeTy[Ttraits] = __traits(classInstanceSize, TypeTraits);
            sizeTy[Tmixin] = __traits(classInstanceSize, TypeMixin);
            return sizeTy;
        }();

    final extern (D) this(TY ty)
    {
        this.ty = ty;
    }

    const(char)* kind() const nothrow pure @nogc @safe
    {
        assert(false); // should be overridden
    }

    final Type copy() nothrow const
    {
        Type t = cast(Type)mem.xmalloc(sizeTy[ty]);
        memcpy(cast(void*)t, cast(void*)this, sizeTy[ty]);
        return t;
    }

    Type syntaxCopy()
    {
        fprintf(stderr, "this = %s, ty = %d\n", toChars(), ty);
        assert(0);
    }

    override bool equals(const RootObject o) const
    {
        Type t = cast(Type)o;
        //printf("Type::equals(%s, %s)\n", toChars(), t.toChars());
        // deco strings are unique
        // and semantic() has been run
        if (this == o || ((t && deco == t.deco) && deco !is null))
        {
            //printf("deco = '%s', t.deco = '%s'\n", deco, t.deco);
            return true;
        }
        //if (deco && t && t.deco) printf("deco = '%s', t.deco = '%s'\n", deco, t.deco);
        return false;
    }

    final bool equivalent(Type t)
    {
        return immutableOf().equals(t.immutableOf());
    }

    // kludge for template.isType()
    override final DYNCAST dyncast() const
    {
        return DYNCAST.type;
    }

    /*******************************
     * Covariant means that 'this' can substitute for 't',
     * i.e. a pure function is a match for an impure type.
     * Params:
     *      t = type 'this' is covariant with
     *      pstc = if not null, store STCxxxx which would make it covariant
     *      fix17349 = enable fix https://issues.dlang.org/show_bug.cgi?id=17349
     * Returns:
     *      0       types are distinct
     *      1       this is covariant with t
     *      2       arguments match as far as overloading goes,
     *              but types are not covariant
     *      3       cannot determine covariance because of forward references
     *      *pstc   STCxxxx which would make it covariant
     */
    final int covariant(Type t, StorageClass* pstc = null, bool fix17349 = true)
    {
        version (none)
        {
            printf("Type::covariant(t = %s) %s\n", t.toChars(), toChars());
            printf("deco = %p, %p\n", deco, t.deco);
            //    printf("ty = %d\n", next.ty);
            printf("mod = %x, %x\n", mod, t.mod);
        }
        if (pstc)
            *pstc = 0;
        StorageClass stc = 0;

        bool notcovariant = false;

        if (equals(t))
            return 1; // covariant

        TypeFunction t1 = this.isTypeFunction();
        TypeFunction t2 = t.isTypeFunction();

        if (!t1 || !t2)
            goto Ldistinct;

        if (t1.parameterList.varargs != t2.parameterList.varargs)
            goto Ldistinct;

        if (t1.parameterList.parameters && t2.parameterList.parameters)
        {
            size_t dim = t1.parameterList.length;
            if (dim != t2.parameterList.length)
                goto Ldistinct;

            for (size_t i = 0; i < dim; i++)
            {
                Parameter fparam1 = t1.parameterList[i];
                Parameter fparam2 = t2.parameterList[i];

                if (!fparam1.type.equals(fparam2.type))
                {
                    if (!fix17349)
                        goto Ldistinct;
                    Type tp1 = fparam1.type;
                    Type tp2 = fparam2.type;
                    if (tp1.ty == tp2.ty)
                    {
                        if (auto tc1 = tp1.isTypeClass())
                        {
                            if (tc1.sym == (cast(TypeClass)tp2).sym && MODimplicitConv(tp2.mod, tp1.mod))
                                goto Lcov;
                        }
                        else if (auto ts1 = tp1.isTypeStruct())
                        {
                            if (ts1.sym == (cast(TypeStruct)tp2).sym && MODimplicitConv(tp2.mod, tp1.mod))
                                goto Lcov;
                        }
                        else if (tp1.ty == Tpointer)
                        {
                            if (tp2.implicitConvTo(tp1))
                                goto Lcov;
                        }
                        else if (tp1.ty == Tarray)
                        {
                            if (tp2.implicitConvTo(tp1))
                                goto Lcov;
                        }
                        else if (tp1.ty == Tdelegate)
                        {
                            if (tp1.implicitConvTo(tp2))
                                goto Lcov;
                        }
                    }
                    goto Ldistinct;
                }
            Lcov:
                notcovariant |= !fparam1.isCovariant(t1.isref, fparam2);
            }
        }
        else if (t1.parameterList.parameters != t2.parameterList.parameters)
        {
            if (t1.parameterList.length || t2.parameterList.length)
                goto Ldistinct;
        }

        // The argument lists match
        if (notcovariant)
            goto Lnotcovariant;
        if (t1.linkage != t2.linkage)
            goto Lnotcovariant;

        {
            // Return types
            Type t1n = t1.next;
            Type t2n = t2.next;

            if (!t1n || !t2n) // happens with return type inference
                goto Lnotcovariant;

            if (t1n.equals(t2n))
                goto Lcovariant;
            if (t1n.ty == Tclass && t2n.ty == Tclass)
            {
                /* If same class type, but t2n is const, then it's
                 * covariant. Do this test first because it can work on
                 * forward references.
                 */
                if ((cast(TypeClass)t1n).sym == (cast(TypeClass)t2n).sym && MODimplicitConv(t1n.mod, t2n.mod))
                    goto Lcovariant;

                // If t1n is forward referenced:
                ClassDeclaration cd = (cast(TypeClass)t1n).sym;
                if (cd.semanticRun < PASS.semanticdone && !cd.isBaseInfoComplete())
                    cd.dsymbolSemantic(null);
                if (!cd.isBaseInfoComplete())
                {
                    return 3; // forward references
                }
            }
            if (t1n.ty == Tstruct && t2n.ty == Tstruct)
            {
                if ((cast(TypeStruct)t1n).sym == (cast(TypeStruct)t2n).sym && MODimplicitConv(t1n.mod, t2n.mod))
                    goto Lcovariant;
            }
            else if (t1n.ty == t2n.ty && t1n.implicitConvTo(t2n))
                goto Lcovariant;
            else if (t1n.ty == Tnull)
            {
                // NULL is covariant with any pointer type, but not with any
                // dynamic arrays, associative arrays or delegates.
                // https://issues.dlang.org/show_bug.cgi?id=8589
                // https://issues.dlang.org/show_bug.cgi?id=19618
                Type t2bn = t2n.toBasetype();
                if (t2bn.ty == Tnull || t2bn.ty == Tpointer || t2bn.ty == Tclass)
                    goto Lcovariant;
            }
        }
        goto Lnotcovariant;

    Lcovariant:
        if (t1.isref != t2.isref)
            goto Lnotcovariant;

        if (!t1.isref && (t1.isscope || t2.isscope))
        {
            StorageClass stc1 = t1.isscope ? STC.scope_ : 0;
            StorageClass stc2 = t2.isscope ? STC.scope_ : 0;
            if (t1.isreturn)
            {
                stc1 |= STC.return_;
                if (!t1.isscope)
                    stc1 |= STC.ref_;
            }
            if (t2.isreturn)
            {
                stc2 |= STC.return_;
                if (!t2.isscope)
                    stc2 |= STC.ref_;
            }
            if (!Parameter.isCovariantScope(t1.isref, stc1, stc2))
                goto Lnotcovariant;
        }

        // We can subtract 'return ref' from 'this', but cannot add it
        else if (t1.isreturn && !t2.isreturn)
            goto Lnotcovariant;

        /* Can convert mutable to const
         */
        if (!MODimplicitConv(t2.mod, t1.mod))
        {
            version (none)
            {
                //stop attribute inference with const
                // If adding 'const' will make it covariant
                if (MODimplicitConv(t2.mod, MODmerge(t1.mod, MODFlags.const_)))
                    stc |= STC.const_;
                else
                    goto Lnotcovariant;
            }
            else
            {
                goto Ldistinct;
            }
        }

        /* Can convert pure to impure, nothrow to throw, and nogc to gc
         */
        if (!t1.purity && t2.purity)
            stc |= STC.pure_;

        if (!t1.isnothrow && t2.isnothrow)
            stc |= STC.nothrow_;

        if (!t1.isnogc && t2.isnogc)
            stc |= STC.nogc;

        /* Can convert safe/trusted to system
         */
        if (t1.trust <= TRUST.system && t2.trust >= TRUST.trusted)
        {
            // Should we infer trusted or safe? Go with safe.
            stc |= STC.safe;
        }

        if (stc)
        {
            if (pstc)
                *pstc = stc;
            goto Lnotcovariant;
        }

        //printf("\tcovaraint: 1\n");
        return 1;

    Ldistinct:
        //printf("\tcovaraint: 0\n");
        return 0;

    Lnotcovariant:
        //printf("\tcovaraint: 2\n");
        return 2;
    }

    /********************************
     * For pretty-printing a type.
     */
    final override const(char)* toChars() const
    {
        OutBuffer buf;
        buf.reserve(16);
        HdrGenState hgs;
        hgs.fullQual = (ty == Tclass && !mod);

        .toCBuffer(this, &buf, null, &hgs);
        return buf.extractChars();
    }

    /// ditto
    final char* toPrettyChars(bool QualifyTypes = false)
    {
        OutBuffer buf;
        buf.reserve(16);
        HdrGenState hgs;
        hgs.fullQual = QualifyTypes;

        .toCBuffer(this, &buf, null, &hgs);
        return buf.extractChars();
    }

    static void _init()
    {
        stringtable._init(14000);

        // Set basic types
        __gshared TY* basetab =
        [
            Tvoid,
            Tint8,
            Tuns8,
            Tint16,
            Tuns16,
            Tint32,
            Tuns32,
            Tint64,
            Tuns64,
            Tint128,
            Tuns128,
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
            Terror
        ];

        for (size_t i = 0; basetab[i] != Terror; i++)
        {
            Type t = new TypeBasic(basetab[i]);
            t = t.merge();
            basic[basetab[i]] = t;
        }
        basic[Terror] = new TypeError();

        tvoid = basic[Tvoid];
        tint8 = basic[Tint8];
        tuns8 = basic[Tuns8];
        tint16 = basic[Tint16];
        tuns16 = basic[Tuns16];
        tint32 = basic[Tint32];
        tuns32 = basic[Tuns32];
        tint64 = basic[Tint64];
        tuns64 = basic[Tuns64];
        tint128 = basic[Tint128];
        tuns128 = basic[Tuns128];
        tfloat32 = basic[Tfloat32];
        tfloat64 = basic[Tfloat64];
        tfloat80 = basic[Tfloat80];

        timaginary32 = basic[Timaginary32];
        timaginary64 = basic[Timaginary64];
        timaginary80 = basic[Timaginary80];

        tcomplex32 = basic[Tcomplex32];
        tcomplex64 = basic[Tcomplex64];
        tcomplex80 = basic[Tcomplex80];

        tbool = basic[Tbool];
        tchar = basic[Tchar];
        twchar = basic[Twchar];
        tdchar = basic[Tdchar];

        tshiftcnt = tint32;
        terror = basic[Terror];
        tnull = basic[Tnull];
        tnull = new TypeNull();
        tnull.deco = tnull.merge().deco;

        tvoidptr = tvoid.pointerTo();
        tstring = tchar.immutableOf().arrayOf();
        twstring = twchar.immutableOf().arrayOf();
        tdstring = tdchar.immutableOf().arrayOf();
        tvalist = target.va_listType();

        const isLP64 = global.params.isLP64;

        tsize_t    = basic[isLP64 ? Tuns64 : Tuns32];
        tptrdiff_t = basic[isLP64 ? Tint64 : Tint32];
        thash_t = tsize_t;
    }

    /**
     * Deinitializes the global state of the compiler.
     *
     * This can be used to restore the state set by `_init` to its original
     * state.
     */
    static void deinitialize()
    {
        stringtable = stringtable.init;
    }

    final d_uns64 size()
    {
        return size(Loc.initial);
    }

    d_uns64 size(const ref Loc loc)
    {
        error(loc, "no size for type `%s`", toChars());
        return SIZE_INVALID;
    }

    uint alignsize()
    {
        return cast(uint)size(Loc.initial);
    }

    final Type trySemantic(const ref Loc loc, Scope* sc)
    {
        //printf("+trySemantic(%s) %d\n", toChars(), global.errors);

        // Needed to display any deprecations that were gagged
        auto tcopy = this.syntaxCopy();

        const errors = global.startGagging();
        Type t = typeSemantic(this, loc, sc);
        if (global.endGagging(errors) || t.ty == Terror) // if any errors happened
        {
            t = null;
        }
        else
        {
            // If `typeSemantic` succeeded, there may have been deprecations that
            // were gagged due the the `startGagging` above.  Run again to display
            // those deprecations.  https://issues.dlang.org/show_bug.cgi?id=19107
            if (global.gaggedWarnings > 0)
                typeSemantic(tcopy, loc, sc);
        }
        //printf("-trySemantic(%s) %d\n", toChars(), global.errors);
        return t;
    }

    /*************************************
     * This version does a merge even if the deco is already computed.
     * Necessary for types that have a deco, but are not merged.
     */
    final Type merge2()
    {
        //printf("merge2(%s)\n", toChars());
        Type t = this;
        assert(t);
        if (!t.deco)
            return t.merge();

        auto sv = stringtable.lookup(t.deco, strlen(t.deco));
        if (sv && sv.value)
        {
            t = sv.value;
            assert(t.deco);
        }
        else
            assert(0);
        return t;
    }

    /*********************************
     * Store this type's modifier name into buf.
     */
    final void modToBuffer(OutBuffer* buf) nothrow const
    {
        if (mod)
        {
            buf.writeByte(' ');
            MODtoBuffer(buf, mod);
        }
    }

    /*********************************
     * Return this type's modifier name.
     */
    final char* modToChars() nothrow const
    {
        OutBuffer buf;
        buf.reserve(16);
        modToBuffer(&buf);
        return buf.extractChars();
    }

    bool isintegral()
    {
        return false;
    }

    // real, imaginary, or complex
    bool isfloating()
    {
        return false;
    }

    bool isreal()
    {
        return false;
    }

    bool isimaginary()
    {
        return false;
    }

    bool iscomplex()
    {
        return false;
    }

    bool isscalar()
    {
        return false;
    }

    bool isunsigned()
    {
        return false;
    }

    bool ischar()
    {
        return false;
    }

    bool isscope()
    {
        return false;
    }

    bool isString()
    {
        return false;
    }

    /**************************
     * When T is mutable,
     * Given:
     *      T a, b;
     * Can we bitwise assign:
     *      a = b;
     * ?
     */
    bool isAssignable()
    {
        return true;
    }

    /**************************
     * Returns true if T can be converted to boolean value.
     */
    bool isBoolean()
    {
        return isscalar();
    }

    /*********************************
     * Check type to see if it is based on a deprecated symbol.
     */
    void checkDeprecated(const ref Loc loc, Scope* sc)
    {
        if (Dsymbol s = toDsymbol(sc))
        {
            s.checkDeprecated(loc, sc);
        }
    }

    final bool isConst() const nothrow pure @nogc @safe
    {
        return (mod & MODFlags.const_) != 0;
    }

    final bool isImmutable() const nothrow pure @nogc @safe
    {
        return (mod & MODFlags.immutable_) != 0;
    }

    final bool isMutable() const nothrow pure @nogc @safe
    {
        return (mod & (MODFlags.const_ | MODFlags.immutable_ | MODFlags.wild)) == 0;
    }

    final bool isShared() const nothrow pure @nogc @safe
    {
        return (mod & MODFlags.shared_) != 0;
    }

    final bool isSharedConst() const nothrow pure @nogc @safe
    {
        return (mod & (MODFlags.shared_ | MODFlags.const_)) == (MODFlags.shared_ | MODFlags.const_);
    }

    final bool isWild() const nothrow pure @nogc @safe
    {
        return (mod & MODFlags.wild) != 0;
    }

    final bool isWildConst() const nothrow pure @nogc @safe
    {
        return (mod & MODFlags.wildconst) == MODFlags.wildconst;
    }

    final bool isSharedWild() const nothrow pure @nogc @safe
    {
        return (mod & (MODFlags.shared_ | MODFlags.wild)) == (MODFlags.shared_ | MODFlags.wild);
    }

    final bool isNaked() const nothrow pure @nogc @safe
    {
        return mod == 0;
    }

    /********************************
     * Return a copy of this type with all attributes null-initialized.
     * Useful for creating a type with different modifiers.
     */
    final Type nullAttributes() nothrow const
    {
        uint sz = sizeTy[ty];
        Type t = cast(Type)mem.xmalloc(sz);
        memcpy(cast(void*)t, cast(void*)this, sz);
        // t.mod = NULL;  // leave mod unchanged
        t.deco = null;
        t.arrayof = null;
        t.pto = null;
        t.rto = null;
        t.cto = null;
        t.ito = null;
        t.sto = null;
        t.scto = null;
        t.wto = null;
        t.wcto = null;
        t.swto = null;
        t.swcto = null;
        t.vtinfo = null;
        t.ctype = null;
        if (t.ty == Tstruct)
            (cast(TypeStruct)t).att = AliasThisRec.fwdref;
        if (t.ty == Tclass)
            (cast(TypeClass)t).att = AliasThisRec.fwdref;
        return t;
    }

    /********************************
     * Convert to 'const'.
     */
    final Type constOf()
    {
        //printf("Type::constOf() %p %s\n", this, toChars());
        if (mod == MODFlags.const_)
            return this;
        if (cto)
        {
            assert(cto.mod == MODFlags.const_);
            return cto;
        }
        Type t = makeConst();
        t = t.merge();
        t.fixTo(this);
        //printf("-Type::constOf() %p %s\n", t, t.toChars());
        return t;
    }

    /********************************
     * Convert to 'immutable'.
     */
    final Type immutableOf()
    {
        //printf("Type::immutableOf() %p %s\n", this, toChars());
        if (isImmutable())
            return this;
        if (ito)
        {
            assert(ito.isImmutable());
            return ito;
        }
        Type t = makeImmutable();
        t = t.merge();
        t.fixTo(this);
        //printf("\t%p\n", t);
        return t;
    }

    /********************************
     * Make type mutable.
     */
    final Type mutableOf()
    {
        //printf("Type::mutableOf() %p, %s\n", this, toChars());
        Type t = this;
        if (isImmutable())
        {
            t = ito; // immutable => naked
            assert(!t || (t.isMutable() && !t.isShared()));
        }
        else if (isConst())
        {
            if (isShared())
            {
                if (isWild())
                    t = swcto; // shared wild const -> shared
                else
                    t = sto; // shared const => shared
            }
            else
            {
                if (isWild())
                    t = wcto; // wild const -> naked
                else
                    t = cto; // const => naked
            }
            assert(!t || t.isMutable());
        }
        else if (isWild())
        {
            if (isShared())
                t = sto; // shared wild => shared
            else
                t = wto; // wild => naked
            assert(!t || t.isMutable());
        }
        if (!t)
        {
            t = makeMutable();
            t = t.merge();
            t.fixTo(this);
        }
        else
            t = t.merge();
        assert(t.isMutable());
        return t;
    }

    final Type sharedOf()
    {
        //printf("Type::sharedOf() %p, %s\n", this, toChars());
        if (mod == MODFlags.shared_)
            return this;
        if (sto)
        {
            assert(sto.mod == MODFlags.shared_);
            return sto;
        }
        Type t = makeShared();
        t = t.merge();
        t.fixTo(this);
        //printf("\t%p\n", t);
        return t;
    }

    final Type sharedConstOf()
    {
        //printf("Type::sharedConstOf() %p, %s\n", this, toChars());
        if (mod == (MODFlags.shared_ | MODFlags.const_))
            return this;
        if (scto)
        {
            assert(scto.mod == (MODFlags.shared_ | MODFlags.const_));
            return scto;
        }
        Type t = makeSharedConst();
        t = t.merge();
        t.fixTo(this);
        //printf("\t%p\n", t);
        return t;
    }

    /********************************
     * Make type unshared.
     *      0            => 0
     *      const        => const
     *      immutable    => immutable
     *      shared       => 0
     *      shared const => const
     *      wild         => wild
     *      wild const   => wild const
     *      shared wild  => wild
     *      shared wild const => wild const
     */
    final Type unSharedOf()
    {
        //printf("Type::unSharedOf() %p, %s\n", this, toChars());
        Type t = this;

        if (isShared())
        {
            if (isWild())
            {
                if (isConst())
                    t = wcto; // shared wild const => wild const
                else
                    t = wto; // shared wild => wild
            }
            else
            {
                if (isConst())
                    t = cto; // shared const => const
                else
                    t = sto; // shared => naked
            }
            assert(!t || !t.isShared());
        }

        if (!t)
        {
            t = this.nullAttributes();
            t.mod = mod & ~MODFlags.shared_;
            t.ctype = ctype;
            t = t.merge();
            t.fixTo(this);
        }
        else
            t = t.merge();
        assert(!t.isShared());
        return t;
    }

    /********************************
     * Convert to 'wild'.
     */
    final Type wildOf()
    {
        //printf("Type::wildOf() %p %s\n", this, toChars());
        if (mod == MODFlags.wild)
            return this;
        if (wto)
        {
            assert(wto.mod == MODFlags.wild);
            return wto;
        }
        Type t = makeWild();
        t = t.merge();
        t.fixTo(this);
        //printf("\t%p %s\n", t, t.toChars());
        return t;
    }

    final Type wildConstOf()
    {
        //printf("Type::wildConstOf() %p %s\n", this, toChars());
        if (mod == MODFlags.wildconst)
            return this;
        if (wcto)
        {
            assert(wcto.mod == MODFlags.wildconst);
            return wcto;
        }
        Type t = makeWildConst();
        t = t.merge();
        t.fixTo(this);
        //printf("\t%p %s\n", t, t.toChars());
        return t;
    }

    final Type sharedWildOf()
    {
        //printf("Type::sharedWildOf() %p, %s\n", this, toChars());
        if (mod == (MODFlags.shared_ | MODFlags.wild))
            return this;
        if (swto)
        {
            assert(swto.mod == (MODFlags.shared_ | MODFlags.wild));
            return swto;
        }
        Type t = makeSharedWild();
        t = t.merge();
        t.fixTo(this);
        //printf("\t%p %s\n", t, t.toChars());
        return t;
    }

    final Type sharedWildConstOf()
    {
        //printf("Type::sharedWildConstOf() %p, %s\n", this, toChars());
        if (mod == (MODFlags.shared_ | MODFlags.wildconst))
            return this;
        if (swcto)
        {
            assert(swcto.mod == (MODFlags.shared_ | MODFlags.wildconst));
            return swcto;
        }
        Type t = makeSharedWildConst();
        t = t.merge();
        t.fixTo(this);
        //printf("\t%p %s\n", t, t.toChars());
        return t;
    }

    /**********************************
     * For our new type 'this', which is type-constructed from t,
     * fill in the cto, ito, sto, scto, wto shortcuts.
     */
    final void fixTo(Type t)
    {
        // If fixing this: immutable(T*) by t: immutable(T)*,
        // cache t to this.xto won't break transitivity.
        Type mto = null;
        Type tn = nextOf();
        if (!tn || ty != Tsarray && tn.mod == t.nextOf().mod)
        {
            switch (t.mod)
            {
            case 0:
                mto = t;
                break;

            case MODFlags.const_:
                cto = t;
                break;

            case MODFlags.wild:
                wto = t;
                break;

            case MODFlags.wildconst:
                wcto = t;
                break;

            case MODFlags.shared_:
                sto = t;
                break;

            case MODFlags.shared_ | MODFlags.const_:
                scto = t;
                break;

            case MODFlags.shared_ | MODFlags.wild:
                swto = t;
                break;

            case MODFlags.shared_ | MODFlags.wildconst:
                swcto = t;
                break;

            case MODFlags.immutable_:
                ito = t;
                break;

            default:
                break;
            }
        }
        assert(mod != t.mod);

        auto X(T, U)(T m, U n)
        {
            return ((m << 4) | n);
        }

        switch (mod)
        {
        case 0:
            break;

        case MODFlags.const_:
            cto = mto;
            t.cto = this;
            break;

        case MODFlags.wild:
            wto = mto;
            t.wto = this;
            break;

        case MODFlags.wildconst:
            wcto = mto;
            t.wcto = this;
            break;

        case MODFlags.shared_:
            sto = mto;
            t.sto = this;
            break;

        case MODFlags.shared_ | MODFlags.const_:
            scto = mto;
            t.scto = this;
            break;

        case MODFlags.shared_ | MODFlags.wild:
            swto = mto;
            t.swto = this;
            break;

        case MODFlags.shared_ | MODFlags.wildconst:
            swcto = mto;
            t.swcto = this;
            break;

        case MODFlags.immutable_:
            t.ito = this;
            if (t.cto)
                t.cto.ito = this;
            if (t.sto)
                t.sto.ito = this;
            if (t.scto)
                t.scto.ito = this;
            if (t.wto)
                t.wto.ito = this;
            if (t.wcto)
                t.wcto.ito = this;
            if (t.swto)
                t.swto.ito = this;
            if (t.swcto)
                t.swcto.ito = this;
            break;

        default:
            assert(0);
        }

        check();
        t.check();
        //printf("fixTo: %s, %s\n", toChars(), t.toChars());
    }

    /***************************
     * Look for bugs in constructing types.
     */
    final void check()
    {
        switch (mod)
        {
        case 0:
            if (cto)
                assert(cto.mod == MODFlags.const_);
            if (ito)
                assert(ito.mod == MODFlags.immutable_);
            if (sto)
                assert(sto.mod == MODFlags.shared_);
            if (scto)
                assert(scto.mod == (MODFlags.shared_ | MODFlags.const_));
            if (wto)
                assert(wto.mod == MODFlags.wild);
            if (wcto)
                assert(wcto.mod == MODFlags.wildconst);
            if (swto)
                assert(swto.mod == (MODFlags.shared_ | MODFlags.wild));
            if (swcto)
                assert(swcto.mod == (MODFlags.shared_ | MODFlags.wildconst));
            break;

        case MODFlags.const_:
            if (cto)
                assert(cto.mod == 0);
            if (ito)
                assert(ito.mod == MODFlags.immutable_);
            if (sto)
                assert(sto.mod == MODFlags.shared_);
            if (scto)
                assert(scto.mod == (MODFlags.shared_ | MODFlags.const_));
            if (wto)
                assert(wto.mod == MODFlags.wild);
            if (wcto)
                assert(wcto.mod == MODFlags.wildconst);
            if (swto)
                assert(swto.mod == (MODFlags.shared_ | MODFlags.wild));
            if (swcto)
                assert(swcto.mod == (MODFlags.shared_ | MODFlags.wildconst));
            break;

        case MODFlags.wild:
            if (cto)
                assert(cto.mod == MODFlags.const_);
            if (ito)
                assert(ito.mod == MODFlags.immutable_);
            if (sto)
                assert(sto.mod == MODFlags.shared_);
            if (scto)
                assert(scto.mod == (MODFlags.shared_ | MODFlags.const_));
            if (wto)
                assert(wto.mod == 0);
            if (wcto)
                assert(wcto.mod == MODFlags.wildconst);
            if (swto)
                assert(swto.mod == (MODFlags.shared_ | MODFlags.wild));
            if (swcto)
                assert(swcto.mod == (MODFlags.shared_ | MODFlags.wildconst));
            break;

        case MODFlags.wildconst:
            assert(!cto || cto.mod == MODFlags.const_);
            assert(!ito || ito.mod == MODFlags.immutable_);
            assert(!sto || sto.mod == MODFlags.shared_);
            assert(!scto || scto.mod == (MODFlags.shared_ | MODFlags.const_));
            assert(!wto || wto.mod == MODFlags.wild);
            assert(!wcto || wcto.mod == 0);
            assert(!swto || swto.mod == (MODFlags.shared_ | MODFlags.wild));
            assert(!swcto || swcto.mod == (MODFlags.shared_ | MODFlags.wildconst));
            break;

        case MODFlags.shared_:
            if (cto)
                assert(cto.mod == MODFlags.const_);
            if (ito)
                assert(ito.mod == MODFlags.immutable_);
            if (sto)
                assert(sto.mod == 0);
            if (scto)
                assert(scto.mod == (MODFlags.shared_ | MODFlags.const_));
            if (wto)
                assert(wto.mod == MODFlags.wild);
            if (wcto)
                assert(wcto.mod == MODFlags.wildconst);
            if (swto)
                assert(swto.mod == (MODFlags.shared_ | MODFlags.wild));
            if (swcto)
                assert(swcto.mod == (MODFlags.shared_ | MODFlags.wildconst));
            break;

        case MODFlags.shared_ | MODFlags.const_:
            if (cto)
                assert(cto.mod == MODFlags.const_);
            if (ito)
                assert(ito.mod == MODFlags.immutable_);
            if (sto)
                assert(sto.mod == MODFlags.shared_);
            if (scto)
                assert(scto.mod == 0);
            if (wto)
                assert(wto.mod == MODFlags.wild);
            if (wcto)
                assert(wcto.mod == MODFlags.wildconst);
            if (swto)
                assert(swto.mod == (MODFlags.shared_ | MODFlags.wild));
            if (swcto)
                assert(swcto.mod == (MODFlags.shared_ | MODFlags.wildconst));
            break;

        case MODFlags.shared_ | MODFlags.wild:
            if (cto)
                assert(cto.mod == MODFlags.const_);
            if (ito)
                assert(ito.mod == MODFlags.immutable_);
            if (sto)
                assert(sto.mod == MODFlags.shared_);
            if (scto)
                assert(scto.mod == (MODFlags.shared_ | MODFlags.const_));
            if (wto)
                assert(wto.mod == MODFlags.wild);
            if (wcto)
                assert(wcto.mod == MODFlags.wildconst);
            if (swto)
                assert(swto.mod == 0);
            if (swcto)
                assert(swcto.mod == (MODFlags.shared_ | MODFlags.wildconst));
            break;

        case MODFlags.shared_ | MODFlags.wildconst:
            assert(!cto || cto.mod == MODFlags.const_);
            assert(!ito || ito.mod == MODFlags.immutable_);
            assert(!sto || sto.mod == MODFlags.shared_);
            assert(!scto || scto.mod == (MODFlags.shared_ | MODFlags.const_));
            assert(!wto || wto.mod == MODFlags.wild);
            assert(!wcto || wcto.mod == MODFlags.wildconst);
            assert(!swto || swto.mod == (MODFlags.shared_ | MODFlags.wild));
            assert(!swcto || swcto.mod == 0);
            break;

        case MODFlags.immutable_:
            if (cto)
                assert(cto.mod == MODFlags.const_);
            if (ito)
                assert(ito.mod == 0);
            if (sto)
                assert(sto.mod == MODFlags.shared_);
            if (scto)
                assert(scto.mod == (MODFlags.shared_ | MODFlags.const_));
            if (wto)
                assert(wto.mod == MODFlags.wild);
            if (wcto)
                assert(wcto.mod == MODFlags.wildconst);
            if (swto)
                assert(swto.mod == (MODFlags.shared_ | MODFlags.wild));
            if (swcto)
                assert(swcto.mod == (MODFlags.shared_ | MODFlags.wildconst));
            break;

        default:
            assert(0);
        }

        Type tn = nextOf();
        if (tn && ty != Tfunction && tn.ty != Tfunction && ty != Tenum)
        {
            // Verify transitivity
            switch (mod)
            {
            case 0:
            case MODFlags.const_:
            case MODFlags.wild:
            case MODFlags.wildconst:
            case MODFlags.shared_:
            case MODFlags.shared_ | MODFlags.const_:
            case MODFlags.shared_ | MODFlags.wild:
            case MODFlags.shared_ | MODFlags.wildconst:
            case MODFlags.immutable_:
                assert(tn.mod == MODFlags.immutable_ || (tn.mod & mod) == mod);
                break;

            default:
                assert(0);
            }
            tn.check();
        }
    }

    /*************************************
     * Apply STCxxxx bits to existing type.
     * Use *before* semantic analysis is run.
     */
    final Type addSTC(StorageClass stc)
    {
        Type t = this;
        if (t.isImmutable())
        {
        }
        else if (stc & STC.immutable_)
        {
            t = t.makeImmutable();
        }
        else
        {
            if ((stc & STC.shared_) && !t.isShared())
            {
                if (t.isWild())
                {
                    if (t.isConst())
                        t = t.makeSharedWildConst();
                    else
                        t = t.makeSharedWild();
                }
                else
                {
                    if (t.isConst())
                        t = t.makeSharedConst();
                    else
                        t = t.makeShared();
                }
            }
            if ((stc & STC.const_) && !t.isConst())
            {
                if (t.isShared())
                {
                    if (t.isWild())
                        t = t.makeSharedWildConst();
                    else
                        t = t.makeSharedConst();
                }
                else
                {
                    if (t.isWild())
                        t = t.makeWildConst();
                    else
                        t = t.makeConst();
                }
            }
            if ((stc & STC.wild) && !t.isWild())
            {
                if (t.isShared())
                {
                    if (t.isConst())
                        t = t.makeSharedWildConst();
                    else
                        t = t.makeSharedWild();
                }
                else
                {
                    if (t.isConst())
                        t = t.makeWildConst();
                    else
                        t = t.makeWild();
                }
            }
        }
        return t;
    }

    /************************************
     * Apply MODxxxx bits to existing type.
     */
    final Type castMod(MOD mod)
    {
        Type t;
        switch (mod)
        {
        case 0:
            t = unSharedOf().mutableOf();
            break;

        case MODFlags.const_:
            t = unSharedOf().constOf();
            break;

        case MODFlags.wild:
            t = unSharedOf().wildOf();
            break;

        case MODFlags.wildconst:
            t = unSharedOf().wildConstOf();
            break;

        case MODFlags.shared_:
            t = mutableOf().sharedOf();
            break;

        case MODFlags.shared_ | MODFlags.const_:
            t = sharedConstOf();
            break;

        case MODFlags.shared_ | MODFlags.wild:
            t = sharedWildOf();
            break;

        case MODFlags.shared_ | MODFlags.wildconst:
            t = sharedWildConstOf();
            break;

        case MODFlags.immutable_:
            t = immutableOf();
            break;

        default:
            assert(0);
        }
        return t;
    }

    /************************************
     * Add MODxxxx bits to existing type.
     * We're adding, not replacing, so adding const to
     * a shared type => "shared const"
     */
    final Type addMod(MOD mod)
    {
        /* Add anything to immutable, and it remains immutable
         */
        Type t = this;
        if (!t.isImmutable())
        {
            //printf("addMod(%x) %s\n", mod, toChars());
            switch (mod)
            {
            case 0:
                break;

            case MODFlags.const_:
                if (isShared())
                {
                    if (isWild())
                        t = sharedWildConstOf();
                    else
                        t = sharedConstOf();
                }
                else
                {
                    if (isWild())
                        t = wildConstOf();
                    else
                        t = constOf();
                }
                break;

            case MODFlags.wild:
                if (isShared())
                {
                    if (isConst())
                        t = sharedWildConstOf();
                    else
                        t = sharedWildOf();
                }
                else
                {
                    if (isConst())
                        t = wildConstOf();
                    else
                        t = wildOf();
                }
                break;

            case MODFlags.wildconst:
                if (isShared())
                    t = sharedWildConstOf();
                else
                    t = wildConstOf();
                break;

            case MODFlags.shared_:
                if (isWild())
                {
                    if (isConst())
                        t = sharedWildConstOf();
                    else
                        t = sharedWildOf();
                }
                else
                {
                    if (isConst())
                        t = sharedConstOf();
                    else
                        t = sharedOf();
                }
                break;

            case MODFlags.shared_ | MODFlags.const_:
                if (isWild())
                    t = sharedWildConstOf();
                else
                    t = sharedConstOf();
                break;

            case MODFlags.shared_ | MODFlags.wild:
                if (isConst())
                    t = sharedWildConstOf();
                else
                    t = sharedWildOf();
                break;

            case MODFlags.shared_ | MODFlags.wildconst:
                t = sharedWildConstOf();
                break;

            case MODFlags.immutable_:
                t = immutableOf();
                break;

            default:
                assert(0);
            }
        }
        return t;
    }

    /************************************
     * Add storage class modifiers to type.
     */
    Type addStorageClass(StorageClass stc)
    {
        /* Just translate to MOD bits and let addMod() do the work
         */
        MOD mod = 0;
        if (stc & STC.immutable_)
            mod = MODFlags.immutable_;
        else
        {
            if (stc & (STC.const_ | STC.in_))
                mod |= MODFlags.const_;
            if (stc & STC.wild)
                mod |= MODFlags.wild;
            if (stc & STC.shared_)
                mod |= MODFlags.shared_;
        }
        return addMod(mod);
    }

    final Type pointerTo()
    {
        if (ty == Terror)
            return this;
        if (!pto)
        {
            Type t = new TypePointer(this);
            if (ty == Tfunction)
            {
                t.deco = t.merge().deco;
                pto = t;
            }
            else
                pto = t.merge();
        }
        return pto;
    }

    final Type referenceTo()
    {
        if (ty == Terror)
            return this;
        if (!rto)
        {
            Type t = new TypeReference(this);
            rto = t.merge();
        }
        return rto;
    }

    final Type arrayOf()
    {
        if (ty == Terror)
            return this;
        if (!arrayof)
        {
            Type t = new TypeDArray(this);
            arrayof = t.merge();
        }
        return arrayof;
    }

    // Make corresponding static array type without semantic
    final Type sarrayOf(dinteger_t dim)
    {
        assert(deco);
        Type t = new TypeSArray(this, new IntegerExp(Loc.initial, dim, Type.tsize_t));
        // according to TypeSArray::semantic()
        t = t.addMod(mod);
        t = t.merge();
        return t;
    }

    final Type aliasthisOf()
    {
        auto ad = isAggregate(this);
        if (!ad || !ad.aliasthis)
            return null;

        auto s = ad.aliasthis.sym;
        if (s.isAliasDeclaration())
            s = s.toAlias();

        if (s.isTupleDeclaration())
            return null;

        if (auto vd = s.isVarDeclaration())
        {
            auto t = vd.type;
            if (vd.needThis())
                t = t.addMod(this.mod);
            return t;
        }
        if (auto fd = s.isFuncDeclaration())
        {
            fd = resolveFuncCall(Loc.initial, null, fd, null, this, null, FuncResolveFlag.quiet);
            if (!fd || fd.errors || !fd.functionSemantic())
                return Type.terror;

            auto t = fd.type.nextOf();
            if (!t) // issue 14185
                return Type.terror;
            t = t.substWildTo(mod == 0 ? MODFlags.mutable : mod);
            return t;
        }
        if (auto d = s.isDeclaration())
        {
            assert(d.type);
            return d.type;
        }
        if (auto ed = s.isEnumDeclaration())
        {
            return ed.type;
        }
        if (auto td = s.isTemplateDeclaration())
        {
            assert(td._scope);
            auto fd = resolveFuncCall(Loc.initial, null, td, null, this, null, FuncResolveFlag.quiet);
            if (!fd || fd.errors || !fd.functionSemantic())
                return Type.terror;

            auto t = fd.type.nextOf();
            if (!t)
                return Type.terror;
            t = t.substWildTo(mod == 0 ? MODFlags.mutable : mod);
            return t;
        }

        //printf("%s\n", s.kind());
        return null;
    }

    extern (D) final bool checkAliasThisRec()
    {
        Type tb = toBasetype();
        AliasThisRec* pflag;
        if (tb.ty == Tstruct)
            pflag = &(cast(TypeStruct)tb).att;
        else if (tb.ty == Tclass)
            pflag = &(cast(TypeClass)tb).att;
        else
            return false;

        AliasThisRec flag = cast(AliasThisRec)(*pflag & AliasThisRec.typeMask);
        if (flag == AliasThisRec.fwdref)
        {
            Type att = aliasthisOf();
            flag = att && att.implicitConvTo(this) ? AliasThisRec.yes : AliasThisRec.no;
        }
        *pflag = cast(AliasThisRec)(flag | (*pflag & ~AliasThisRec.typeMask));
        return flag == AliasThisRec.yes;
    }

    Type makeConst()
    {
        //printf("Type::makeConst() %p, %s\n", this, toChars());
        if (cto)
            return cto;
        Type t = this.nullAttributes();
        t.mod = MODFlags.const_;
        //printf("-Type::makeConst() %p, %s\n", t, toChars());
        return t;
    }

    Type makeImmutable()
    {
        if (ito)
            return ito;
        Type t = this.nullAttributes();
        t.mod = MODFlags.immutable_;
        return t;
    }

    Type makeShared()
    {
        if (sto)
            return sto;
        Type t = this.nullAttributes();
        t.mod = MODFlags.shared_;
        return t;
    }

    Type makeSharedConst()
    {
        if (scto)
            return scto;
        Type t = this.nullAttributes();
        t.mod = MODFlags.shared_ | MODFlags.const_;
        return t;
    }

    Type makeWild()
    {
        if (wto)
            return wto;
        Type t = this.nullAttributes();
        t.mod = MODFlags.wild;
        return t;
    }

    Type makeWildConst()
    {
        if (wcto)
            return wcto;
        Type t = this.nullAttributes();
        t.mod = MODFlags.wildconst;
        return t;
    }

    Type makeSharedWild()
    {
        if (swto)
            return swto;
        Type t = this.nullAttributes();
        t.mod = MODFlags.shared_ | MODFlags.wild;
        return t;
    }

    Type makeSharedWildConst()
    {
        if (swcto)
            return swcto;
        Type t = this.nullAttributes();
        t.mod = MODFlags.shared_ | MODFlags.wildconst;
        return t;
    }

    Type makeMutable()
    {
        Type t = this.nullAttributes();
        t.mod = mod & MODFlags.shared_;
        return t;
    }

    Dsymbol toDsymbol(Scope* sc)
    {
        return null;
    }

    /*******************************
     * If this is a shell around another type,
     * get that other type.
     */
    Type toBasetype()
    {
        return this;
    }

    bool isBaseOf(Type t, int* poffset)
    {
        return 0; // assume not
    }

    /********************************
     * Determine if 'this' can be implicitly converted
     * to type 'to'.
     * Returns:
     *      MATCH.nomatch, MATCH.convert, MATCH.constant, MATCH.exact
     */
    MATCH implicitConvTo(Type to)
    {
        //printf("Type::implicitConvTo(this=%p, to=%p)\n", this, to);
        //printf("from: %s\n", toChars());
        //printf("to  : %s\n", to.toChars());
        if (this.equals(to))
            return MATCH.exact;
        return MATCH.nomatch;
    }

    /*******************************
     * Determine if converting 'this' to 'to' is an identity operation,
     * a conversion to const operation, or the types aren't the same.
     * Returns:
     *      MATCH.exact      'this' == 'to'
     *      MATCH.constant      'to' is const
     *      MATCH.nomatch    conversion to mutable or invariant
     */
    MATCH constConv(Type to)
    {
        //printf("Type::constConv(this = %s, to = %s)\n", toChars(), to.toChars());
        if (equals(to))
            return MATCH.exact;
        if (ty == to.ty && MODimplicitConv(mod, to.mod))
            return MATCH.constant;
        return MATCH.nomatch;
    }

    /***************************************
     * Compute MOD bits matching `this` argument type to wild parameter type.
     * Params:
     *  t = corresponding parameter type
     *  isRef = parameter is `ref` or `out`
     * Returns:
     *  MOD bits
     */
    MOD deduceWild(Type t, bool isRef)
    {
        //printf("Type::deduceWild this = '%s', tprm = '%s'\n", toChars(), tprm.toChars());
        if (t.isWild())
        {
            if (isImmutable())
                return MODFlags.immutable_;
            else if (isWildConst())
            {
                if (t.isWildConst())
                    return MODFlags.wild;
                else
                    return MODFlags.wildconst;
            }
            else if (isWild())
                return MODFlags.wild;
            else if (isConst())
                return MODFlags.const_;
            else if (isMutable())
                return MODFlags.mutable;
            else
                assert(0);
        }
        return 0;
    }

    Type substWildTo(uint mod)
    {
        //printf("+Type::substWildTo this = %s, mod = x%x\n", toChars(), mod);
        Type t;

        if (Type tn = nextOf())
        {
            // substitution has no effect on function pointer type.
            if (ty == Tpointer && tn.ty == Tfunction)
            {
                t = this;
                goto L1;
            }

            t = tn.substWildTo(mod);
            if (t == tn)
                t = this;
            else
            {
                if (ty == Tpointer)
                    t = t.pointerTo();
                else if (ty == Tarray)
                    t = t.arrayOf();
                else if (ty == Tsarray)
                    t = new TypeSArray(t, (cast(TypeSArray)this).dim.syntaxCopy());
                else if (ty == Taarray)
                {
                    t = new TypeAArray(t, (cast(TypeAArray)this).index.syntaxCopy());
                    (cast(TypeAArray)t).sc = (cast(TypeAArray)this).sc; // duplicate scope
                }
                else if (ty == Tdelegate)
                {
                    t = new TypeDelegate(t);
                }
                else
                    assert(0);

                t = t.merge();
            }
        }
        else
            t = this;

    L1:
        if (isWild())
        {
            if (mod == MODFlags.immutable_)
            {
                t = t.immutableOf();
            }
            else if (mod == MODFlags.wildconst)
            {
                t = t.wildConstOf();
            }
            else if (mod == MODFlags.wild)
            {
                if (isWildConst())
                    t = t.wildConstOf();
                else
                    t = t.wildOf();
            }
            else if (mod == MODFlags.const_)
            {
                t = t.constOf();
            }
            else
            {
                if (isWildConst())
                    t = t.constOf();
                else
                    t = t.mutableOf();
            }
        }
        if (isConst())
            t = t.addMod(MODFlags.const_);
        if (isShared())
            t = t.addMod(MODFlags.shared_);

        //printf("-Type::substWildTo t = %s\n", t.toChars());
        return t;
    }

    final Type unqualify(uint m)
    {
        Type t = mutableOf().unSharedOf();

        Type tn = ty == Tenum ? null : nextOf();
        if (tn && tn.ty != Tfunction)
        {
            Type utn = tn.unqualify(m);
            if (utn != tn)
            {
                if (ty == Tpointer)
                    t = utn.pointerTo();
                else if (ty == Tarray)
                    t = utn.arrayOf();
                else if (ty == Tsarray)
                    t = new TypeSArray(utn, (cast(TypeSArray)this).dim);
                else if (ty == Taarray)
                {
                    t = new TypeAArray(utn, (cast(TypeAArray)this).index);
                    (cast(TypeAArray)t).sc = (cast(TypeAArray)this).sc; // duplicate scope
                }
                else
                    assert(0);

                t = t.merge();
            }
        }
        t = t.addMod(mod & ~m);
        return t;
    }

    /**************************
     * Return type with the top level of it being mutable.
     */
    inout(Type) toHeadMutable() inout
    {
        if (!mod)
            return this;
        Type unqualThis = cast(Type) this;
        // `mutableOf` needs a mutable `this` only for caching
        return cast(inout(Type)) unqualThis.mutableOf();
    }

    inout(ClassDeclaration) isClassHandle() inout
    {
        return null;
    }

    /************************************
     * Return alignment to use for this type.
     */
    structalign_t alignment()
    {
        return STRUCTALIGN_DEFAULT;
    }

    /***************************************
     * Use when we prefer the default initializer to be a literal,
     * rather than a global immutable variable.
     */
    Expression defaultInitLiteral(const ref Loc loc)
    {
        static if (LOGDEFAULTINIT)
        {
            printf("Type::defaultInitLiteral() '%s'\n", toChars());
        }
        return defaultInit(this, loc);
    }

    // if initializer is 0
    bool isZeroInit(const ref Loc loc)
    {
        return false; // assume not
    }

    final Identifier getTypeInfoIdent()
    {
        // _init_10TypeInfo_%s
        OutBuffer buf;
        buf.reserve(32);
        mangleToBuffer(this, &buf);

        const slice = buf[];

        // Allocate buffer on stack, fail over to using malloc()
        char[128] namebuf;

        // Hash long symbol names
        char* name;
        int length;
        if (IN_LLVM && global.params.hashThreshold && (slice.length > global.params.hashThreshold))
        {
            import std.digest.md;
            auto md5hash = md5Of(slice);
            auto hashedname = toHexString(md5hash);
            static assert(hashedname.length < namebuf.length-30);
            name = namebuf.ptr;
            length = sprintf(name, "_D%lluTypeInfo_%.*s6__initZ",
                cast(ulong)9 + hashedname.length, hashedname.length, hashedname.ptr);
        }
        else
        {
        // else path is DDMD original:

        const namelen = 19 + size_t.sizeof * 3 + slice.length + 1;
        name = namelen <= namebuf.length ? namebuf.ptr : cast(char*)Mem.check(malloc(namelen));
        assert(name);

        length = sprintf(name, "_D%lluTypeInfo_%.*s6__initZ",
                cast(ulong)(9 + slice.length), cast(int)slice.length, slice.ptr);
        //printf("%p %s, deco = %s, name = %s\n", this, toChars(), deco, name);
        assert(0 < length && length < namelen); // don't overflow the buffer

        }

        auto id = Identifier.idPool(name, length);

        if (name != namebuf.ptr)
            free(name);
        return id;
    }

    /***************************************
     * Return !=0 if the type or any of its subtypes is wild.
     */
    int hasWild() const
    {
        return mod & MODFlags.wild;
    }

    /***************************************
     * Return !=0 if type has pointers that need to
     * be scanned by the GC during a collection cycle.
     */
    bool hasPointers()
    {
        //printf("Type::hasPointers() %s, %d\n", toChars(), ty);
        return false;
    }

    /*************************************
     * Detect if type has pointer fields that are initialized to void.
     * Local stack variables with such void fields can remain uninitialized,
     * leading to pointer bugs.
     * Returns:
     *  true if so
     */
    bool hasVoidInitPointers()
    {
        return false;
    }

    /*************************************
     * If this is a type of something, return that something.
     */
    Type nextOf()
    {
        return null;
    }

    /*************************************
     * If this is a type of static array, return its base element type.
     */
    final Type baseElemOf()
    {
        Type t = toBasetype();
        TypeSArray tsa;
        while ((tsa = t.isTypeSArray()) !is null)
            t = tsa.next.toBasetype();
        return t;
    }

    /*******************************************
     * Compute number of elements for a (possibly multidimensional) static array,
     * or 1 for other types.
     * Params:
     *  loc = for error message
     * Returns:
     *  number of elements, uint.max on overflow
     */
    final uint numberOfElems(const ref Loc loc)
    {
        //printf("Type::numberOfElems()\n");
        uinteger_t n = 1;
        Type tb = this;
        while ((tb = tb.toBasetype()).ty == Tsarray)
        {
            bool overflow = false;
            n = mulu(n, (cast(TypeSArray)tb).dim.toUInteger(), overflow);
            if (overflow || n >= uint.max)
            {
                error(loc, "static array `%s` size overflowed to %llu", toChars(), cast(ulong)n);
                return uint.max;
            }
            tb = (cast(TypeSArray)tb).next;
        }
        return cast(uint)n;
    }

    /****************************************
     * Return the mask that an integral type will
     * fit into.
     */
    final uinteger_t sizemask()
    {
        uinteger_t m;
        switch (toBasetype().ty)
        {
        case Tbool:
            m = 1;
            break;
        case Tchar:
        case Tint8:
        case Tuns8:
            m = 0xFF;
            break;
        case Twchar:
        case Tint16:
        case Tuns16:
            m = 0xFFFFU;
            break;
        case Tdchar:
        case Tint32:
        case Tuns32:
            m = 0xFFFFFFFFU;
            break;
        case Tint64:
        case Tuns64:
            m = 0xFFFFFFFFFFFFFFFFUL;
            break;
        default:
            assert(0);
        }
        return m;
    }

    /********************************
     * true if when type goes out of scope, it needs a destructor applied.
     * Only applies to value types, not ref types.
     */
    bool needsDestruction()
    {
        return false;
    }

    /*********************************
     *
     */
    bool needsNested()
    {
        return false;
    }

    /*************************************
     * https://issues.dlang.org/show_bug.cgi?id=14488
     * Check if the inner most base type is complex or imaginary.
     * Should only give alerts when set to emit transitional messages.
     * Params:
     *  loc = The source location.
     *  sc = scope of the type
     */
    extern (D) final bool checkComplexTransition(const ref Loc loc, Scope* sc)
    {
        if (sc.isDeprecated())
            return false;

        Type t = baseElemOf();
        while (t.ty == Tpointer || t.ty == Tarray)
            t = t.nextOf().baseElemOf();

        // Basetype is an opaque enum, nothing to check.
        if (t.ty == Tenum && !(cast(TypeEnum)t).sym.memtype)
            return false;

        if (t.isimaginary() || t.iscomplex())
        {
            Type rt;
            switch (t.ty)
            {
            case Tcomplex32:
            case Timaginary32:
                rt = Type.tfloat32;
                break;

            case Tcomplex64:
            case Timaginary64:
                rt = Type.tfloat64;
                break;

            case Tcomplex80:
            case Timaginary80:
                rt = Type.tfloat80;
                break;

            default:
                assert(0);
            }
            if (t.iscomplex())
            {
                deprecation(loc, "use of complex type `%s` is deprecated, use `std.complex.Complex!(%s)` instead",
                    toChars(), rt.toChars());
                return true;
            }
            else
            {
                deprecation(loc, "use of imaginary type `%s` is deprecated, use `%s` instead",
                    toChars(), rt.toChars());
                return true;
            }
        }
        return false;
    }

    // For eliminating dynamic_cast
    TypeBasic isTypeBasic()
    {
        return null;
    }

    final pure inout nothrow @nogc
    {
        inout(TypeError)      isTypeError()      { return ty == Terror     ? cast(typeof(return))this : null; }
        inout(TypeVector)     isTypeVector()     { return ty == Tvector    ? cast(typeof(return))this : null; }
        inout(TypeSArray)     isTypeSArray()     { return ty == Tsarray    ? cast(typeof(return))this : null; }
        inout(TypeDArray)     isTypeDArray()     { return ty == Tarray     ? cast(typeof(return))this : null; }
        inout(TypeAArray)     isTypeAArray()     { return ty == Taarray    ? cast(typeof(return))this : null; }
        inout(TypePointer)    isTypePointer()    { return ty == Tpointer   ? cast(typeof(return))this : null; }
        inout(TypeReference)  isTypeReference()  { return ty == Treference ? cast(typeof(return))this : null; }
        inout(TypeFunction)   isTypeFunction()   { return ty == Tfunction  ? cast(typeof(return))this : null; }
        inout(TypeDelegate)   isTypeDelegate()   { return ty == Tdelegate  ? cast(typeof(return))this : null; }
        inout(TypeIdentifier) isTypeIdentifier() { return ty == Tident     ? cast(typeof(return))this : null; }
        inout(TypeInstance)   isTypeInstance()   { return ty == Tinstance  ? cast(typeof(return))this : null; }
        inout(TypeTypeof)     isTypeTypeof()     { return ty == Ttypeof    ? cast(typeof(return))this : null; }
        inout(TypeReturn)     isTypeReturn()     { return ty == Treturn    ? cast(typeof(return))this : null; }
        inout(TypeStruct)     isTypeStruct()     { return ty == Tstruct    ? cast(typeof(return))this : null; }
        inout(TypeEnum)       isTypeEnum()       { return ty == Tenum      ? cast(typeof(return))this : null; }
        inout(TypeClass)      isTypeClass()      { return ty == Tclass     ? cast(typeof(return))this : null; }
        inout(TypeTuple)      isTypeTuple()      { return ty == Ttuple     ? cast(typeof(return))this : null; }
        inout(TypeSlice)      isTypeSlice()      { return ty == Tslice     ? cast(typeof(return))this : null; }
        inout(TypeNull)       isTypeNull()       { return ty == Tnull      ? cast(typeof(return))this : null; }
        inout(TypeMixin)      isTypeMixin()      { return ty == Tmixin     ? cast(typeof(return))this : null; }
        inout(TypeTraits)     isTypeTraits()     { return ty == Ttraits    ? cast(typeof(return))this : null; }
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }

    final TypeFunction toTypeFunction()
    {
        if (ty != Tfunction)
            assert(0);
        return cast(TypeFunction)this;
    }
}

/***********************************************************
 */
extern (C++) final class TypeError : Type
{
    extern (D) this()
    {
        super(Terror);
    }

    override Type syntaxCopy()
    {
        // No semantic analysis done, no need to copy
        return this;
    }

    override d_uns64 size(const ref Loc loc)
    {
        return SIZE_INVALID;
    }

    override Expression defaultInitLiteral(const ref Loc loc)
    {
        return new ErrorExp();
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) abstract class TypeNext : Type
{
    Type next;

    final extern (D) this(TY ty, Type next)
    {
        super(ty);
        this.next = next;
    }

    override final void checkDeprecated(const ref Loc loc, Scope* sc)
    {
        Type.checkDeprecated(loc, sc);
        if (next) // next can be NULL if TypeFunction and auto return type
            next.checkDeprecated(loc, sc);
    }

    override final int hasWild() const
    {
        if (ty == Tfunction)
            return 0;
        if (ty == Tdelegate)
            return Type.hasWild();
        return mod & MODFlags.wild || (next && next.hasWild());
    }

    /*******************************
     * For TypeFunction, nextOf() can return NULL if the function return
     * type is meant to be inferred, and semantic() hasn't yet ben run
     * on the function. After semantic(), it must no longer be NULL.
     */
    override final Type nextOf()
    {
        return next;
    }

    override final Type makeConst()
    {
        //printf("TypeNext::makeConst() %p, %s\n", this, toChars());
        if (cto)
        {
            assert(cto.mod == MODFlags.const_);
            return cto;
        }
        TypeNext t = cast(TypeNext)Type.makeConst();
        if (ty != Tfunction && next.ty != Tfunction && !next.isImmutable())
        {
            if (next.isShared())
            {
                if (next.isWild())
                    t.next = next.sharedWildConstOf();
                else
                    t.next = next.sharedConstOf();
            }
            else
            {
                if (next.isWild())
                    t.next = next.wildConstOf();
                else
                    t.next = next.constOf();
            }
        }
        //printf("TypeNext::makeConst() returns %p, %s\n", t, t.toChars());
        return t;
    }

    override final Type makeImmutable()
    {
        //printf("TypeNext::makeImmutable() %s\n", toChars());
        if (ito)
        {
            assert(ito.isImmutable());
            return ito;
        }
        TypeNext t = cast(TypeNext)Type.makeImmutable();
        if (ty != Tfunction && next.ty != Tfunction && !next.isImmutable())
        {
            t.next = next.immutableOf();
        }
        return t;
    }

    override final Type makeShared()
    {
        //printf("TypeNext::makeShared() %s\n", toChars());
        if (sto)
        {
            assert(sto.mod == MODFlags.shared_);
            return sto;
        }
        TypeNext t = cast(TypeNext)Type.makeShared();
        if (ty != Tfunction && next.ty != Tfunction && !next.isImmutable())
        {
            if (next.isWild())
            {
                if (next.isConst())
                    t.next = next.sharedWildConstOf();
                else
                    t.next = next.sharedWildOf();
            }
            else
            {
                if (next.isConst())
                    t.next = next.sharedConstOf();
                else
                    t.next = next.sharedOf();
            }
        }
        //printf("TypeNext::makeShared() returns %p, %s\n", t, t.toChars());
        return t;
    }

    override final Type makeSharedConst()
    {
        //printf("TypeNext::makeSharedConst() %s\n", toChars());
        if (scto)
        {
            assert(scto.mod == (MODFlags.shared_ | MODFlags.const_));
            return scto;
        }
        TypeNext t = cast(TypeNext)Type.makeSharedConst();
        if (ty != Tfunction && next.ty != Tfunction && !next.isImmutable())
        {
            if (next.isWild())
                t.next = next.sharedWildConstOf();
            else
                t.next = next.sharedConstOf();
        }
        //printf("TypeNext::makeSharedConst() returns %p, %s\n", t, t.toChars());
        return t;
    }

    override final Type makeWild()
    {
        //printf("TypeNext::makeWild() %s\n", toChars());
        if (wto)
        {
            assert(wto.mod == MODFlags.wild);
            return wto;
        }
        TypeNext t = cast(TypeNext)Type.makeWild();
        if (ty != Tfunction && next.ty != Tfunction && !next.isImmutable())
        {
            if (next.isShared())
            {
                if (next.isConst())
                    t.next = next.sharedWildConstOf();
                else
                    t.next = next.sharedWildOf();
            }
            else
            {
                if (next.isConst())
                    t.next = next.wildConstOf();
                else
                    t.next = next.wildOf();
            }
        }
        //printf("TypeNext::makeWild() returns %p, %s\n", t, t.toChars());
        return t;
    }

    override final Type makeWildConst()
    {
        //printf("TypeNext::makeWildConst() %s\n", toChars());
        if (wcto)
        {
            assert(wcto.mod == MODFlags.wildconst);
            return wcto;
        }
        TypeNext t = cast(TypeNext)Type.makeWildConst();
        if (ty != Tfunction && next.ty != Tfunction && !next.isImmutable())
        {
            if (next.isShared())
                t.next = next.sharedWildConstOf();
            else
                t.next = next.wildConstOf();
        }
        //printf("TypeNext::makeWildConst() returns %p, %s\n", t, t.toChars());
        return t;
    }

    override final Type makeSharedWild()
    {
        //printf("TypeNext::makeSharedWild() %s\n", toChars());
        if (swto)
        {
            assert(swto.isSharedWild());
            return swto;
        }
        TypeNext t = cast(TypeNext)Type.makeSharedWild();
        if (ty != Tfunction && next.ty != Tfunction && !next.isImmutable())
        {
            if (next.isConst())
                t.next = next.sharedWildConstOf();
            else
                t.next = next.sharedWildOf();
        }
        //printf("TypeNext::makeSharedWild() returns %p, %s\n", t, t.toChars());
        return t;
    }

    override final Type makeSharedWildConst()
    {
        //printf("TypeNext::makeSharedWildConst() %s\n", toChars());
        if (swcto)
        {
            assert(swcto.mod == (MODFlags.shared_ | MODFlags.wildconst));
            return swcto;
        }
        TypeNext t = cast(TypeNext)Type.makeSharedWildConst();
        if (ty != Tfunction && next.ty != Tfunction && !next.isImmutable())
        {
            t.next = next.sharedWildConstOf();
        }
        //printf("TypeNext::makeSharedWildConst() returns %p, %s\n", t, t.toChars());
        return t;
    }

    override final Type makeMutable()
    {
        //printf("TypeNext::makeMutable() %p, %s\n", this, toChars());
        TypeNext t = cast(TypeNext)Type.makeMutable();
        if (ty == Tsarray)
        {
            t.next = next.mutableOf();
        }
        //printf("TypeNext::makeMutable() returns %p, %s\n", t, t.toChars());
        return t;
    }

    override MATCH constConv(Type to)
    {
        //printf("TypeNext::constConv from = %s, to = %s\n", toChars(), to.toChars());
        if (equals(to))
            return MATCH.exact;

        if (!(ty == to.ty && MODimplicitConv(mod, to.mod)))
            return MATCH.nomatch;

        Type tn = to.nextOf();
        if (!(tn && next.ty == tn.ty))
            return MATCH.nomatch;

        MATCH m;
        if (to.isConst()) // whole tail const conversion
        {
            // Recursive shared level check
            m = next.constConv(tn);
            if (m == MATCH.exact)
                m = MATCH.constant;
        }
        else
        {
            //printf("\tnext => %s, to.next => %s\n", next.toChars(), tn.toChars());
            m = next.equals(tn) ? MATCH.constant : MATCH.nomatch;
        }
        return m;
    }

    override final MOD deduceWild(Type t, bool isRef)
    {
        if (ty == Tfunction)
            return 0;

        ubyte wm;

        Type tn = t.nextOf();
        if (!isRef && (ty == Tarray || ty == Tpointer) && tn)
        {
            wm = next.deduceWild(tn, true);
            if (!wm)
                wm = Type.deduceWild(t, true);
        }
        else
        {
            wm = Type.deduceWild(t, isRef);
            if (!wm && tn)
                wm = next.deduceWild(tn, true);
        }

        return wm;
    }

    final void transitive()
    {
        /* Invoke transitivity of type attributes
         */
        next = next.addMod(mod);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class TypeBasic : Type
{
    const(char)* dstring;
    uint flags;

    extern (D) this(TY ty)
    {
        super(ty);
        const(char)* d;
        uint flags = 0;
        switch (ty)
        {
        case Tvoid:
            d = Token.toChars(TOK.void_);
            break;

        case Tint8:
            d = Token.toChars(TOK.int8);
            flags |= TFlags.integral;
            break;

        case Tuns8:
            d = Token.toChars(TOK.uns8);
            flags |= TFlags.integral | TFlags.unsigned;
            break;

        case Tint16:
            d = Token.toChars(TOK.int16);
            flags |= TFlags.integral;
            break;

        case Tuns16:
            d = Token.toChars(TOK.uns16);
            flags |= TFlags.integral | TFlags.unsigned;
            break;

        case Tint32:
            d = Token.toChars(TOK.int32);
            flags |= TFlags.integral;
            break;

        case Tuns32:
            d = Token.toChars(TOK.uns32);
            flags |= TFlags.integral | TFlags.unsigned;
            break;

        case Tfloat32:
            d = Token.toChars(TOK.float32);
            flags |= TFlags.floating | TFlags.real_;
            break;

        case Tint64:
            d = Token.toChars(TOK.int64);
            flags |= TFlags.integral;
            break;

        case Tuns64:
            d = Token.toChars(TOK.uns64);
            flags |= TFlags.integral | TFlags.unsigned;
            break;

        case Tint128:
            d = Token.toChars(TOK.int128);
            flags |= TFlags.integral;
            break;

        case Tuns128:
            d = Token.toChars(TOK.uns128);
            flags |= TFlags.integral | TFlags.unsigned;
            break;

        case Tfloat64:
            d = Token.toChars(TOK.float64);
            flags |= TFlags.floating | TFlags.real_;
            break;

        case Tfloat80:
            d = Token.toChars(TOK.float80);
            flags |= TFlags.floating | TFlags.real_;
            break;

        case Timaginary32:
            d = Token.toChars(TOK.imaginary32);
            flags |= TFlags.floating | TFlags.imaginary;
            break;

        case Timaginary64:
            d = Token.toChars(TOK.imaginary64);
            flags |= TFlags.floating | TFlags.imaginary;
            break;

        case Timaginary80:
            d = Token.toChars(TOK.imaginary80);
            flags |= TFlags.floating | TFlags.imaginary;
            break;

        case Tcomplex32:
            d = Token.toChars(TOK.complex32);
            flags |= TFlags.floating | TFlags.complex;
            break;

        case Tcomplex64:
            d = Token.toChars(TOK.complex64);
            flags |= TFlags.floating | TFlags.complex;
            break;

        case Tcomplex80:
            d = Token.toChars(TOK.complex80);
            flags |= TFlags.floating | TFlags.complex;
            break;

        case Tbool:
            d = "bool";
            flags |= TFlags.integral | TFlags.unsigned;
            break;

        case Tchar:
            d = Token.toChars(TOK.char_);
            flags |= TFlags.integral | TFlags.unsigned | TFlags.char_;
            break;

        case Twchar:
            d = Token.toChars(TOK.wchar_);
            flags |= TFlags.integral | TFlags.unsigned | TFlags.char_;
            break;

        case Tdchar:
            d = Token.toChars(TOK.dchar_);
            flags |= TFlags.integral | TFlags.unsigned | TFlags.char_;
            break;

        default:
            assert(0);
        }
        this.dstring = d;
        this.flags = flags;
        merge(this);
    }

    override const(char)* kind() const
    {
        return dstring;
    }

    override Type syntaxCopy()
    {
        // No semantic analysis done on basic types, no need to copy
        return this;
    }

    override d_uns64 size(const ref Loc loc) const
    {
        uint size;
        //printf("TypeBasic::size()\n");
        switch (ty)
        {
        case Tint8:
        case Tuns8:
            size = 1;
            break;

        case Tint16:
        case Tuns16:
            size = 2;
            break;

        case Tint32:
        case Tuns32:
        case Tfloat32:
        case Timaginary32:
            size = 4;
            break;

        case Tint64:
        case Tuns64:
        case Tfloat64:
        case Timaginary64:
            size = 8;
            break;

        case Tfloat80:
        case Timaginary80:
            size = target.realsize;
            break;

        case Tcomplex32:
            size = 8;
            break;

        case Tcomplex64:
        case Tint128:
        case Tuns128:
            size = 16;
            break;

        case Tcomplex80:
            size = target.realsize * 2;
            break;

        case Tvoid:
            //size = Type::size();      // error message
            size = 1;
            break;

        case Tbool:
            size = 1;
            break;

        case Tchar:
            size = 1;
            break;

        case Twchar:
            size = 2;
            break;

        case Tdchar:
            size = 4;
            break;

        default:
            assert(0);
        }
        //printf("TypeBasic::size() = %d\n", size);
        return size;
    }

    override uint alignsize()
    {
        return target.alignsize(this);
    }

    override bool isintegral()
    {
        //printf("TypeBasic::isintegral('%s') x%x\n", toChars(), flags);
        return (flags & TFlags.integral) != 0;
    }

    override bool isfloating() const
    {
        return (flags & TFlags.floating) != 0;
    }

    override bool isreal() const
    {
        return (flags & TFlags.real_) != 0;
    }

    override bool isimaginary() const
    {
        return (flags & TFlags.imaginary) != 0;
    }

    override bool iscomplex() const
    {
        return (flags & TFlags.complex) != 0;
    }

    override bool isscalar() const
    {
        return (flags & (TFlags.integral | TFlags.floating)) != 0;
    }

    override bool isunsigned() const
    {
        return (flags & TFlags.unsigned) != 0;
    }

    override bool ischar() const
    {
        return (flags & TFlags.char_) != 0;
    }

    override MATCH implicitConvTo(Type to)
    {
        //printf("TypeBasic::implicitConvTo(%s) from %s\n", to.toChars(), toChars());
        if (this == to)
            return MATCH.exact;

        if (ty == to.ty)
        {
            if (mod == to.mod)
                return MATCH.exact;
            else if (MODimplicitConv(mod, to.mod))
                return MATCH.constant;
            else if (!((mod ^ to.mod) & MODFlags.shared_)) // for wild matching
                return MATCH.constant;
            else
                return MATCH.convert;
        }

        if (ty == Tvoid || to.ty == Tvoid)
            return MATCH.nomatch;
        if (to.ty == Tbool)
            return MATCH.nomatch;

        TypeBasic tob;
        if (to.ty == Tvector && to.deco)
        {
            TypeVector tv = cast(TypeVector)to;
            tob = tv.elementType();
        }
        else if (auto te = to.isTypeEnum())
        {
            EnumDeclaration ed = te.sym;
            if (ed.isSpecial())
            {
                /* Special enums that allow implicit conversions to them
                 * with a MATCH.convert
                 */
                tob = to.toBasetype().isTypeBasic();
            }
            else
                return MATCH.nomatch;
        }
        else
            tob = to.isTypeBasic();
        if (!tob)
            return MATCH.nomatch;

        if (flags & TFlags.integral)
        {
            // Disallow implicit conversion of integers to imaginary or complex
            if (tob.flags & (TFlags.imaginary | TFlags.complex))
                return MATCH.nomatch;

            // If converting from integral to integral
            if (tob.flags & TFlags.integral)
            {
                d_uns64 sz = size(Loc.initial);
                d_uns64 tosz = tob.size(Loc.initial);

                /* Can't convert to smaller size
                 */
                if (sz > tosz)
                    return MATCH.nomatch;
                /* Can't change sign if same size
                 */
                //if (sz == tosz && (flags ^ tob.flags) & TFlags.unsigned)
                //    return MATCH.nomatch;
            }
        }
        else if (flags & TFlags.floating)
        {
            // Disallow implicit conversion of floating point to integer
            if (tob.flags & TFlags.integral)
                return MATCH.nomatch;

            assert(tob.flags & TFlags.floating || to.ty == Tvector);

            // Disallow implicit conversion from complex to non-complex
            if (flags & TFlags.complex && !(tob.flags & TFlags.complex))
                return MATCH.nomatch;

            // Disallow implicit conversion of real or imaginary to complex
            if (flags & (TFlags.real_ | TFlags.imaginary) && tob.flags & TFlags.complex)
                return MATCH.nomatch;

            // Disallow implicit conversion to-from real and imaginary
            if ((flags & (TFlags.real_ | TFlags.imaginary)) != (tob.flags & (TFlags.real_ | TFlags.imaginary)))
                return MATCH.nomatch;
        }
        return MATCH.convert;
    }

    override bool isZeroInit(const ref Loc loc) const
    {
        switch (ty)
        {
        case Tchar:
        case Twchar:
        case Tdchar:
        case Timaginary32:
        case Timaginary64:
        case Timaginary80:
        case Tfloat32:
        case Tfloat64:
        case Tfloat80:
        case Tcomplex32:
        case Tcomplex64:
        case Tcomplex80:
            return false; // no
        default:
            return true; // yes
        }
    }

    // For eliminating dynamic_cast
    override TypeBasic isTypeBasic()
    {
        return this;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * The basetype must be one of:
 *   byte[16],ubyte[16],short[8],ushort[8],int[4],uint[4],long[2],ulong[2],float[4],double[2]
 * For AVX:
 *   byte[32],ubyte[32],short[16],ushort[16],int[8],uint[8],long[4],ulong[4],float[8],double[4]
 */
extern (C++) final class TypeVector : Type
{
    Type basetype;

    extern (D) this(Type basetype)
    {
        super(Tvector);
        this.basetype = basetype;
    }

    static TypeVector create(Type basetype)
    {
        return new TypeVector(basetype);
    }

    override const(char)* kind() const
    {
        return "vector";
    }

    override Type syntaxCopy()
    {
        return new TypeVector(basetype.syntaxCopy());
    }

    override d_uns64 size(const ref Loc loc)
    {
        return basetype.size();
    }

    override uint alignsize()
    {
        return cast(uint)basetype.size();
    }

    override bool isintegral()
    {
        //printf("TypeVector::isintegral('%s') x%x\n", toChars(), flags);
        return basetype.nextOf().isintegral();
    }

    override bool isfloating()
    {
        return basetype.nextOf().isfloating();
    }

    override bool isscalar()
    {
        return basetype.nextOf().isscalar();
    }

    override bool isunsigned()
    {
        return basetype.nextOf().isunsigned();
    }

    override bool isBoolean() const
    {
        return false;
    }

    override MATCH implicitConvTo(Type to)
    {
        //printf("TypeVector::implicitConvTo(%s) from %s\n", to.toChars(), toChars());
        if (this == to)
            return MATCH.exact;
        if (ty == to.ty)
            return MATCH.convert;
        return MATCH.nomatch;
    }

    override Expression defaultInitLiteral(const ref Loc loc)
    {
        //printf("TypeVector::defaultInitLiteral()\n");
        assert(basetype.ty == Tsarray);
        Expression e = basetype.defaultInitLiteral(loc);
        auto ve = new VectorExp(loc, e, this);
        ve.type = this;
        ve.dim = cast(int)(basetype.size(loc) / elementType().size(loc));
        return ve;
    }

    TypeBasic elementType()
    {
        assert(basetype.ty == Tsarray);
        TypeSArray t = cast(TypeSArray)basetype;
        TypeBasic tb = t.nextOf().isTypeBasic();
        assert(tb);
        return tb;
    }

    override bool isZeroInit(const ref Loc loc)
    {
        return basetype.isZeroInit(loc);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) abstract class TypeArray : TypeNext
{
    final extern (D) this(TY ty, Type next)
    {
        super(ty, next);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * Static array, one with a fixed dimension
 */
extern (C++) final class TypeSArray : TypeArray
{
    Expression dim;

    extern (D) this(Type t, Expression dim)
    {
        super(Tsarray, t);
        //printf("TypeSArray(%s)\n", dim.toChars());
        this.dim = dim;
    }

    override const(char)* kind() const
    {
        return "sarray";
    }

    override Type syntaxCopy()
    {
        Type t = next.syntaxCopy();
        Expression e = dim.syntaxCopy();
        t = new TypeSArray(t, e);
        t.mod = mod;
        return t;
    }

    override d_uns64 size(const ref Loc loc)
    {
        //printf("TypeSArray::size()\n");
        const n = numberOfElems(loc);
        const elemsize = baseElemOf().size(loc);
        bool overflow = false;
        const sz = mulu(n, elemsize, overflow);
        if (overflow || sz >= uint.max)
        {
            if (elemsize != SIZE_INVALID && n != uint.max)
                error(loc, "static array `%s` size overflowed to %lld", toChars(), cast(long)sz);
            return SIZE_INVALID;
        }
        return sz;
    }

    override uint alignsize()
    {
        return next.alignsize();
    }

    override bool isString()
    {
        TY nty = next.toBasetype().ty;
        return nty == Tchar || nty == Twchar || nty == Tdchar;
    }

    override bool isZeroInit(const ref Loc loc)
    {
        return next.isZeroInit(loc);
    }

    override structalign_t alignment()
    {
        return next.alignment();
    }

    override MATCH constConv(Type to)
    {
        if (auto tsa = to.isTypeSArray())
        {
            if (!dim.equals(tsa.dim))
                return MATCH.nomatch;
        }
        return TypeNext.constConv(to);
    }

    override MATCH implicitConvTo(Type to)
    {
        //printf("TypeSArray::implicitConvTo(to = %s) this = %s\n", to.toChars(), toChars());
        if (auto ta = to.isTypeDArray())
        {
            if (!MODimplicitConv(next.mod, ta.next.mod))
                return MATCH.nomatch;

            /* Allow conversion to void[]
             */
            if (ta.next.ty == Tvoid)
            {
                return MATCH.convert;
            }

            MATCH m = next.constConv(ta.next);
            if (m > MATCH.nomatch)
            {
                return MATCH.convert;
            }
            return MATCH.nomatch;
        }
        if (auto tsa = to.isTypeSArray())
        {
            if (this == to)
                return MATCH.exact;

            if (dim.equals(tsa.dim))
            {
                /* Since static arrays are value types, allow
                 * conversions from const elements to non-const
                 * ones, just like we allow conversion from const int
                 * to int.
                 */
                MATCH m = next.implicitConvTo(tsa.next);
                if (m >= MATCH.constant)
                {
                    if (mod != to.mod)
                        m = MATCH.constant;
                    return m;
                }
            }
        }
        return MATCH.nomatch;
    }

    override Expression defaultInitLiteral(const ref Loc loc)
    {
        static if (LOGDEFAULTINIT)
        {
            printf("TypeSArray::defaultInitLiteral() '%s'\n", toChars());
        }
        size_t d = cast(size_t)dim.toInteger();
        Expression elementinit;
        if (next.ty == Tvoid)
            elementinit = tuns8.defaultInitLiteral(loc);
        else
            elementinit = next.defaultInitLiteral(loc);
        auto elements = new Expressions(d);
        foreach (ref e; *elements)
            e = null;
        auto ae = new ArrayLiteralExp(Loc.initial, this, elementinit, elements);
        return ae;
    }

    override bool hasPointers()
    {
        /* Don't want to do this, because:
         *    struct S { T* array[0]; }
         * may be a variable length struct.
         */
        //if (dim.toInteger() == 0)
        //    return false;

        if (next.ty == Tvoid)
        {
            // Arrays of void contain arbitrary data, which may include pointers
            return true;
        }
        else
            return next.hasPointers();
    }

    override bool needsDestruction()
    {
        return next.needsDestruction();
    }

    /*********************************
     *
     */
    override bool needsNested()
    {
        return next.needsNested();
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * Dynamic array, no dimension
 */
extern (C++) final class TypeDArray : TypeArray
{
    extern (D) this(Type t)
    {
        super(Tarray, t);
        //printf("TypeDArray(t = %p)\n", t);
    }

    override const(char)* kind() const
    {
        return "darray";
    }

    override Type syntaxCopy()
    {
        Type t = next.syntaxCopy();
        if (t == next)
            t = this;
        else
        {
            t = new TypeDArray(t);
            t.mod = mod;
        }
        return t;
    }

    override d_uns64 size(const ref Loc loc) const
    {
        //printf("TypeDArray::size()\n");
        return target.ptrsize * 2;
    }

    override uint alignsize() const
    {
        // A DArray consists of two ptr-sized values, so align it on pointer size
        // boundary
        return target.ptrsize;
    }

    override bool isString()
    {
        TY nty = next.toBasetype().ty;
        return nty == Tchar || nty == Twchar || nty == Tdchar;
    }

    override bool isZeroInit(const ref Loc loc) const
    {
        return true;
    }

    override bool isBoolean() const
    {
        return true;
    }

    override MATCH implicitConvTo(Type to)
    {
        //printf("TypeDArray::implicitConvTo(to = %s) this = %s\n", to.toChars(), toChars());
        if (equals(to))
            return MATCH.exact;

        if (auto ta = to.isTypeDArray())
        {
            if (!MODimplicitConv(next.mod, ta.next.mod))
                return MATCH.nomatch; // not const-compatible

            /* Allow conversion to void[]
             */
            if (next.ty != Tvoid && ta.next.ty == Tvoid)
            {
                return MATCH.convert;
            }

            MATCH m = next.constConv(ta.next);
            if (m > MATCH.nomatch)
            {
                if (m == MATCH.exact && mod != to.mod)
                    m = MATCH.constant;
                return m;
            }
        }
        return Type.implicitConvTo(to);
    }

    override bool hasPointers() const
    {
        return true;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class TypeAArray : TypeArray
{
    Type index;     // key type
    Loc loc;
    Scope* sc;

    extern (D) this(Type t, Type index)
    {
        super(Taarray, t);
        this.index = index;
    }

    static TypeAArray create(Type t, Type index)
    {
        return new TypeAArray(t, index);
    }

    override const(char)* kind() const
    {
        return "aarray";
    }

    override Type syntaxCopy()
    {
        Type t = next.syntaxCopy();
        Type ti = index.syntaxCopy();
        if (t == next && ti == index)
            t = this;
        else
        {
            t = new TypeAArray(t, ti);
            t.mod = mod;
        }
        return t;
    }

    override d_uns64 size(const ref Loc loc) const
    {
        return target.ptrsize;
    }

    override bool isZeroInit(const ref Loc loc) const
    {
        return true;
    }

    override bool isBoolean() const
    {
        return true;
    }

    override bool hasPointers() const
    {
        return true;
    }

    override MATCH implicitConvTo(Type to)
    {
        //printf("TypeAArray::implicitConvTo(to = %s) this = %s\n", to.toChars(), toChars());
        if (equals(to))
            return MATCH.exact;

        if (auto ta = to.isTypeAArray())
        {
            if (!MODimplicitConv(next.mod, ta.next.mod))
                return MATCH.nomatch; // not const-compatible

            if (!MODimplicitConv(index.mod, ta.index.mod))
                return MATCH.nomatch; // not const-compatible

            MATCH m = next.constConv(ta.next);
            MATCH mi = index.constConv(ta.index);
            if (m > MATCH.nomatch && mi > MATCH.nomatch)
            {
                return MODimplicitConv(mod, to.mod) ? MATCH.constant : MATCH.nomatch;
            }
        }
        return Type.implicitConvTo(to);
    }

    override MATCH constConv(Type to)
    {
        if (auto taa = to.isTypeAArray())
        {
            MATCH mindex = index.constConv(taa.index);
            MATCH mkey = next.constConv(taa.next);
            // Pick the worst match
            return mkey < mindex ? mkey : mindex;
        }
        return Type.constConv(to);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class TypePointer : TypeNext
{
    extern (D) this(Type t)
    {
        super(Tpointer, t);
    }

    static TypePointer create(Type t)
    {
        return new TypePointer(t);
    }

    override const(char)* kind() const
    {
        return "pointer";
    }

    override Type syntaxCopy()
    {
        Type t = next.syntaxCopy();
        if (t == next)
            t = this;
        else
        {
            t = new TypePointer(t);
            t.mod = mod;
        }
        return t;
    }

    override d_uns64 size(const ref Loc loc) const
    {
        return target.ptrsize;
    }

    override MATCH implicitConvTo(Type to)
    {
        //printf("TypePointer::implicitConvTo(to = %s) %s\n", to.toChars(), toChars());
        if (equals(to))
            return MATCH.exact;

        if (next.ty == Tfunction)
        {
            if (auto tp = to.isTypePointer())
            {
                if (tp.next.ty == Tfunction)
                {
                    if (next.equals(tp.next))
                        return MATCH.constant;

                    if (next.covariant(tp.next) == 1)
                    {
                        Type tret = this.next.nextOf();
                        Type toret = tp.next.nextOf();
                        if (tret.ty == Tclass && toret.ty == Tclass)
                        {
                            /* https://issues.dlang.org/show_bug.cgi?id=10219
                             * Check covariant interface return with offset tweaking.
                             * interface I {}
                             * class C : Object, I {}
                             * I function() dg = function C() {}    // should be error
                             */
                            int offset = 0;
                            if (toret.isBaseOf(tret, &offset) && offset != 0)
                                return MATCH.nomatch;
                        }
                        return MATCH.convert;
                    }
                }
                else if (tp.next.ty == Tvoid)
                {
                    // Allow conversions to void*
                    return MATCH.convert;
                }
            }
            return MATCH.nomatch;
        }
        else if (auto tp = to.isTypePointer())
        {
            assert(tp.next);

            if (!MODimplicitConv(next.mod, tp.next.mod))
                return MATCH.nomatch; // not const-compatible

            /* Alloc conversion to void*
             */
            if (next.ty != Tvoid && tp.next.ty == Tvoid)
            {
                return MATCH.convert;
            }

            MATCH m = next.constConv(tp.next);
            if (m > MATCH.nomatch)
            {
                if (m == MATCH.exact && mod != to.mod)
                    m = MATCH.constant;
                return m;
            }
        }
        return MATCH.nomatch;
    }

    override MATCH constConv(Type to)
    {
        if (next.ty == Tfunction)
        {
            if (to.nextOf() && next.equals((cast(TypeNext)to).next))
                return Type.constConv(to);
            else
                return MATCH.nomatch;
        }
        return TypeNext.constConv(to);
    }

    override bool isscalar() const
    {
        return true;
    }

    override bool isZeroInit(const ref Loc loc) const
    {
        return true;
    }

    override bool hasPointers() const
    {
        return true;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class TypeReference : TypeNext
{
    extern (D) this(Type t)
    {
        super(Treference, t);
        // BUG: what about references to static arrays?
    }

    override const(char)* kind() const
    {
        return "reference";
    }

    override Type syntaxCopy()
    {
        Type t = next.syntaxCopy();
        if (t == next)
            t = this;
        else
        {
            t = new TypeReference(t);
            t.mod = mod;
        }
        return t;
    }

    override d_uns64 size(const ref Loc loc) const
    {
        return target.ptrsize;
    }

    override bool isZeroInit(const ref Loc loc) const
    {
        return true;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

enum RET : int
{
    regs         = 1,    // returned in registers
    stack        = 2,    // returned on stack
}

enum TRUST : int
{
    default_   = 0,
    system     = 1,    // @system (same as TRUST.default)
    trusted    = 2,    // @trusted
    safe       = 3,    // @safe
}

enum TRUSTformat : int
{
    TRUSTformatDefault,     // do not emit @system when trust == TRUST.default_
    TRUSTformatSystem,      // emit @system when trust == TRUST.default_
}

alias TRUSTformatDefault = TRUSTformat.TRUSTformatDefault;
alias TRUSTformatSystem = TRUSTformat.TRUSTformatSystem;

enum PURE : int
{
    impure      = 0,    // not pure at all
    fwdref      = 1,    // it's pure, but not known which level yet
    weak        = 2,    // no mutable globals are read or written
    const_      = 3,    // parameters are values or const
    strong      = 4,    // parameters are values or immutable
}

/***********************************************************
 */
extern (C++) final class TypeFunction : TypeNext
{
    // .next is the return type

    ParameterList parameterList;   // function parameters

    bool isnothrow;             // true: nothrow
    bool isnogc;                // true: is @nogc
    bool isproperty;            // can be called without parentheses
    bool isref;                 // true: returns a reference
    bool isreturn;              // true: 'this' is returned by ref
    bool isscope;               // true: 'this' is scope
    bool isreturninferred;      // true: 'this' is return from inference
    bool isscopeinferred;       // true: 'this' is scope from inference
    LINK linkage;               // calling convention
    TRUST trust;                // level of trust
    PURE purity = PURE.impure;
    ubyte iswild;               // bit0: inout on params, bit1: inout on qualifier
    Expressions* fargs;         // function arguments
    int inuse;
    bool incomplete;            // return type or default arguments removed

    extern (D) this(ParameterList pl, Type treturn, LINK linkage, StorageClass stc = 0)
    {
        super(Tfunction, treturn);
        //if (!treturn) *(char*)0=0;
        //    assert(treturn);
        assert(VarArg.none <= pl.varargs && pl.varargs <= VarArg.typesafe);
        this.parameterList = pl;
        this.linkage = linkage;

        if (stc & STC.pure_)
            this.purity = PURE.fwdref;
        if (stc & STC.nothrow_)
            this.isnothrow = true;
        if (stc & STC.nogc)
            this.isnogc = true;
        if (stc & STC.property)
            this.isproperty = true;

        if (stc & STC.ref_)
            this.isref = true;
        if (stc & STC.return_)
            this.isreturn = true;
        if (stc & STC.returninferred)
            this.isreturninferred = true;
        if (stc & STC.scope_)
            this.isscope = true;
        if (stc & STC.scopeinferred)
            this.isscopeinferred = true;

        this.trust = TRUST.default_;
        if (stc & STC.safe)
            this.trust = TRUST.safe;
        if (stc & STC.system)
            this.trust = TRUST.system;
        if (stc & STC.trusted)
            this.trust = TRUST.trusted;
    }

    static TypeFunction create(Parameters* parameters, Type treturn, VarArg varargs, LINK linkage, StorageClass stc = 0)
    {
        return new TypeFunction(ParameterList(parameters, varargs), treturn, linkage, stc);
    }

    override const(char)* kind() const
    {
        return "function";
    }

    override Type syntaxCopy()
    {
        Type treturn = next ? next.syntaxCopy() : null;
        Parameters* params = Parameter.arraySyntaxCopy(parameterList.parameters);
        auto t = new TypeFunction(ParameterList(params, parameterList.varargs), treturn, linkage);
        t.mod = mod;
        t.isnothrow = isnothrow;
        t.isnogc = isnogc;
        t.purity = purity;
        t.isproperty = isproperty;
        t.isref = isref;
        t.isreturn = isreturn;
        t.isscope = isscope;
        t.isreturninferred = isreturninferred;
        t.isscopeinferred = isscopeinferred;
        t.iswild = iswild;
        t.trust = trust;
        t.fargs = fargs;
        return t;
    }

    /********************************************
     * Set 'purity' field of 'this'.
     * Do this lazily, as the parameter types might be forward referenced.
     */
    void purityLevel()
    {
        TypeFunction tf = this;
        if (tf.purity != PURE.fwdref)
            return;

        /* Determine purity level based on mutability of t
         * and whether it is a 'ref' type or not.
         */
        static PURE purityOfType(bool isref, Type t)
        {
            if (isref)
            {
                if (t.mod & MODFlags.immutable_)
                    return PURE.strong;
                if (t.mod & (MODFlags.const_ | MODFlags.wild))
                    return PURE.const_;
                return PURE.weak;
            }

            t = t.baseElemOf();

            if (!t.hasPointers() || t.mod & MODFlags.immutable_)
                return PURE.strong;

            /* Accept immutable(T)[] and immutable(T)* as being strongly pure
             */
            if (t.ty == Tarray || t.ty == Tpointer)
            {
                Type tn = t.nextOf().toBasetype();
                if (tn.mod & MODFlags.immutable_)
                    return PURE.strong;
                if (tn.mod & (MODFlags.const_ | MODFlags.wild))
                    return PURE.const_;
            }

            /* The rest of this is too strict; fix later.
             * For example, the only pointer members of a struct may be immutable,
             * which would maintain strong purity.
             * (Just like for dynamic arrays and pointers above.)
             */
            if (t.mod & (MODFlags.const_ | MODFlags.wild))
                return PURE.const_;

            /* Should catch delegates and function pointers, and fold in their purity
             */
            return PURE.weak;
        }

        purity = PURE.strong; // assume strong until something weakens it

        /* Evaluate what kind of purity based on the modifiers for the parameters
         */
        const dim = tf.parameterList.length;
    Lloop: foreach (i; 0 .. dim)
        {
            Parameter fparam = tf.parameterList[i];
            Type t = fparam.type;
            if (!t)
                continue;

            if (fparam.storageClass & (STC.lazy_ | STC.out_))
            {
                purity = PURE.weak;
                break;
            }
            switch (purityOfType((fparam.storageClass & STC.ref_) != 0, t))
            {
                case PURE.weak:
                    purity = PURE.weak;
                    break Lloop; // since PURE.weak, no need to check further

                case PURE.const_:
                    purity = PURE.const_;
                    continue;

                case PURE.strong:
                    continue;

                default:
                    assert(0);
            }
        }

        if (purity > PURE.weak && tf.nextOf())
        {
            /* Adjust purity based on mutability of return type.
             * https://issues.dlang.org/show_bug.cgi?id=15862
             */
            const purity2 = purityOfType(tf.isref, tf.nextOf());
            if (purity2 < purity)
                purity = purity2;
        }
        tf.purity = purity;
    }

    /********************************************
     * Return true if there are lazy parameters.
     */
    bool hasLazyParameters()
    {
        size_t dim = parameterList.length;
        for (size_t i = 0; i < dim; i++)
        {
            Parameter fparam = parameterList[i];
            if (fparam.storageClass & STC.lazy_)
                return true;
        }
        return false;
    }

    /*******************************
     * Check for `extern (D) U func(T t, ...)` variadic function type,
     * which has `_arguments[]` added as the first argument.
     * Returns:
     *  true if D-style variadic
     */
    bool isDstyleVariadic() const pure nothrow
    {
        return linkage == LINK.d && parameterList.varargs == VarArg.variadic;
    }

    /***************************
     * Examine function signature for parameter p and see if
     * the value of p can 'escape' the scope of the function.
     * This is useful to minimize the needed annotations for the parameters.
     * Params:
     *  tthis = type of `this` parameter, null if none
     *  p = parameter to this function
     * Returns:
     *  true if escapes via assignment to global or through a parameter
     */
    bool parameterEscapes(Type tthis, Parameter p)
    {
        /* Scope parameters do not escape.
         * Allow 'lazy' to imply 'scope' -
         * lazy parameters can be passed along
         * as lazy parameters to the next function, but that isn't
         * escaping.
         */
        if (parameterStorageClass(tthis, p) & (STC.scope_ | STC.lazy_))
            return false;
        return true;
    }

    /************************************
     * Take the specified storage class for p,
     * and use the function signature to infer whether
     * STC.scope_ and STC.return_ should be OR'd in.
     * (This will not affect the name mangling.)
     * Params:
     *  tthis = type of `this` parameter, null if none
     *  p = parameter to this function
     * Returns:
     *  storage class with STC.scope_ or STC.return_ OR'd in
     */
    StorageClass parameterStorageClass(Type tthis, Parameter p)
    {
        //printf("parameterStorageClass(p: %s)\n", p.toChars());
        auto stc = p.storageClass;
        if (!global.params.vsafe)
            return stc;

        if (stc & (STC.scope_ | STC.return_ | STC.lazy_) || purity == PURE.impure)
            return stc;

        /* If haven't inferred the return type yet, can't infer storage classes
         */
        if (!nextOf())
            return stc;

        purityLevel();

        // See if p can escape via any of the other parameters
        if (purity == PURE.weak)
        {
            // Check escaping through parameters
            const dim = parameterList.length;
            foreach (const i; 0 .. dim)
            {
                Parameter fparam = parameterList[i];
                if (fparam == p)
                    continue;
                Type t = fparam.type;
                if (!t)
                    continue;
                t = t.baseElemOf();
                if (t.isMutable() && t.hasPointers())
                {
                    if (fparam.storageClass & (STC.ref_ | STC.out_))
                    {
                    }
                    else if (t.ty == Tarray || t.ty == Tpointer)
                    {
                        Type tn = t.nextOf().toBasetype();
                        if (!(tn.isMutable() && tn.hasPointers()))
                            continue;
                    }
                    return stc;
                }
            }

            // Check escaping through `this`
            if (tthis && tthis.isMutable())
            {
                auto tb = tthis.toBasetype();
                AggregateDeclaration ad;
                if (auto tc = tb.isTypeClass())
                    ad = tc.sym;
                else if (auto ts = tb.isTypeStruct())
                    ad = ts.sym;
                else
                    assert(0);
                foreach (VarDeclaration v; ad.fields)
                {
                    if (v.hasPointers())
                        return stc;
                }
            }
        }

        stc |= STC.scope_;

        /* Inferring STC.return_ here has false positives
         * for pure functions, producing spurious error messages
         * about escaping references.
         * Give up on it for now.
         */
        version (none)
        {
            Type tret = nextOf().toBasetype();
            if (isref || tret.hasPointers())
            {
                /* The result has references, so p could be escaping
                 * that way.
                 */
                stc |= STC.return_;
            }
        }

        return stc;
    }

    override Type addStorageClass(StorageClass stc)
    {
        //printf("addStorageClass(%llx) %d\n", stc, (stc & STC.scope_) != 0);
        TypeFunction t = Type.addStorageClass(stc).toTypeFunction();
        if ((stc & STC.pure_ && !t.purity) ||
            (stc & STC.nothrow_ && !t.isnothrow) ||
            (stc & STC.nogc && !t.isnogc) ||
            (stc & STC.scope_ && !t.isscope) ||
            (stc & STC.safe && t.trust < TRUST.trusted))
        {
            // Klunky to change these
            auto tf = new TypeFunction(t.parameterList, t.next, t.linkage, 0);
            tf.mod = t.mod;
            tf.fargs = fargs;
            tf.purity = t.purity;
            tf.isnothrow = t.isnothrow;
            tf.isnogc = t.isnogc;
            tf.isproperty = t.isproperty;
            tf.isref = t.isref;
            tf.isreturn = t.isreturn;
            tf.isscope = t.isscope;
            tf.isreturninferred = t.isreturninferred;
            tf.isscopeinferred = t.isscopeinferred;
            tf.trust = t.trust;
            tf.iswild = t.iswild;

            if (stc & STC.pure_)
                tf.purity = PURE.fwdref;
            if (stc & STC.nothrow_)
                tf.isnothrow = true;
            if (stc & STC.nogc)
                tf.isnogc = true;
            if (stc & STC.safe)
                tf.trust = TRUST.safe;
            if (stc & STC.scope_)
            {
                tf.isscope = true;
                if (stc & STC.scopeinferred)
                    tf.isscopeinferred = true;
            }

            tf.deco = tf.merge().deco;
            t = tf;
        }
        return t;
    }

    override Type substWildTo(uint)
    {
        if (!iswild && !(mod & MODFlags.wild))
            return this;

        // Substitude inout qualifier of function type to mutable or immutable
        // would break type system. Instead substitude inout to the most weak
        // qualifer - const.
        uint m = MODFlags.const_;

        assert(next);
        Type tret = next.substWildTo(m);
        Parameters* params = parameterList.parameters;
        if (mod & MODFlags.wild)
            params = parameterList.parameters.copy();
        for (size_t i = 0; i < params.dim; i++)
        {
            Parameter p = (*params)[i];
            Type t = p.type.substWildTo(m);
            if (t == p.type)
                continue;
            if (params == parameterList.parameters)
                params = parameterList.parameters.copy();
            (*params)[i] = new Parameter(p.storageClass, t, null, null, null);
        }
        if (next == tret && params == parameterList.parameters)
            return this;

        // Similar to TypeFunction::syntaxCopy;
        auto t = new TypeFunction(ParameterList(params, parameterList.varargs), tret, linkage);
        t.mod = ((mod & MODFlags.wild) ? (mod & ~MODFlags.wild) | MODFlags.const_ : mod);
        t.isnothrow = isnothrow;
        t.isnogc = isnogc;
        t.purity = purity;
        t.isproperty = isproperty;
        t.isref = isref;
        t.isreturn = isreturn;
        t.isscope = isscope;
        t.isreturninferred = isreturninferred;
        t.isscopeinferred = isscopeinferred;
        t.iswild = 0;
        t.trust = trust;
        t.fargs = fargs;
        return t.merge();
    }

    // arguments get specially formatted
    private const(char)* getParamError(Expression arg, Parameter par)
    {
        if (global.gag && !global.params.showGaggedErrors)
            return null;
        // show qualification when toChars() is the same but types are different
        auto at = arg.type.toChars();
        bool qual = !arg.type.equals(par.type) && strcmp(at, par.type.toChars()) == 0;
        if (qual)
            at = arg.type.toPrettyChars(true);
        OutBuffer buf;
        // only mention rvalue if it's relevant
        const rv = !arg.isLvalue() && par.storageClass & (STC.ref_ | STC.out_);
        buf.printf("cannot pass %sargument `%s` of type `%s` to parameter `%s`",
            rv ? "rvalue ".ptr : "".ptr, arg.toChars(), at,
            parameterToChars(par, this, qual));
        return buf.extractChars();
    }

    private extern(D) const(char)* getMatchError(A...)(const(char)* format, A args)
    {
        if (global.gag && !global.params.showGaggedErrors)
            return null;
        OutBuffer buf;
        buf.printf(format, args);
        return buf.extractChars();
    }

    /********************************
     * 'args' are being matched to function 'this'
     * Determine match level.
     * Params:
     *      tthis = type of `this` pointer, null if not member function
     *      args = array of function arguments
     *      flag = 1: performing a partial ordering match
     *      pMessage = address to store error message, or null
     *      sc = context
     * Returns:
     *      MATCHxxxx
     */
    extern (D) MATCH callMatch(Type tthis, Expression[] args, int flag = 0, const(char)** pMessage = null, Scope* sc = null)
    {
        //printf("TypeFunction::callMatch() %s\n", toChars());
        MATCH match = MATCH.exact; // assume exact match
        ubyte wildmatch = 0;

        if (tthis)
        {
            Type t = tthis;
            if (t.toBasetype().ty == Tpointer)
                t = t.toBasetype().nextOf(); // change struct* to struct
            if (t.mod != mod)
            {
                if (MODimplicitConv(t.mod, mod))
                    match = MATCH.constant;
                else if ((mod & MODFlags.wild) && MODimplicitConv(t.mod, (mod & ~MODFlags.wild) | MODFlags.const_))
                {
                    match = MATCH.constant;
                }
                else
                    return MATCH.nomatch;
            }
            if (isWild())
            {
                if (t.isWild())
                    wildmatch |= MODFlags.wild;
                else if (t.isConst())
                    wildmatch |= MODFlags.const_;
                else if (t.isImmutable())
                    wildmatch |= MODFlags.immutable_;
                else
                    wildmatch |= MODFlags.mutable;
            }
        }

        size_t nparams = parameterList.length;
        size_t nargs = args.length;
        if (nargs > nparams)
        {
            if (parameterList.varargs == VarArg.none)
            {
                // suppress early exit if an error message is wanted,
                // so we can check any matching args are valid
                if (!pMessage)
                    goto Nomatch;
            }
            // too many args; no match
            match = MATCH.convert; // match ... with a "conversion" match level
        }

        for (size_t u = 0; u < nargs; u++)
        {
            if (u >= nparams)
                break;
            Parameter p = parameterList[u];
            Expression arg = args[u];
            assert(arg);
            Type tprm = p.type;
            Type targ = arg.type;

            if (!(p.storageClass & STC.lazy_ && tprm.ty == Tvoid && targ.ty != Tvoid))
            {
                bool isRef = (p.storageClass & (STC.ref_ | STC.out_)) != 0;
                wildmatch |= targ.deduceWild(tprm, isRef);
            }
        }
        if (wildmatch)
        {
            /* Calculate wild matching modifier
             */
            if (wildmatch & MODFlags.const_ || wildmatch & (wildmatch - 1))
                wildmatch = MODFlags.const_;
            else if (wildmatch & MODFlags.immutable_)
                wildmatch = MODFlags.immutable_;
            else if (wildmatch & MODFlags.wild)
                wildmatch = MODFlags.wild;
            else
            {
                assert(wildmatch & MODFlags.mutable);
                wildmatch = MODFlags.mutable;
            }
        }

        for (size_t u = 0; u < nparams; u++)
        {
            MATCH m;

            Parameter p = parameterList[u];
            assert(p);
            if (u >= nargs)
            {
                if (p.defaultArg)
                    continue;
                // try typesafe variadics
                goto L1;
            }
            {
                Expression arg = args[u];
                assert(arg);
                //printf("arg: %s, type: %s\n", arg.toChars(), arg.type.toChars());

                Type targ = arg.type;
                Type tprm = wildmatch ? p.type.substWildTo(wildmatch) : p.type;

                if (p.storageClass & STC.lazy_ && tprm.ty == Tvoid && targ.ty != Tvoid)
                    m = MATCH.convert;
                else
                {
                    //printf("%s of type %s implicitConvTo %s\n", arg.toChars(), targ.toChars(), tprm.toChars());
                    if (flag)
                    {
                        // for partial ordering, value is an irrelevant mockup, just look at the type
                        m = targ.implicitConvTo(tprm);
                    }
                    else
                    {
                        const isRef = (p.storageClass & (STC.ref_ | STC.out_)) != 0;

                        StructDeclaration argStruct, prmStruct;

                        // first look for a copy constructor
                        if (arg.isLvalue() && !isRef && targ.ty == Tstruct && tprm.ty == Tstruct)
                        {
                            // if the argument and the parameter are of the same unqualified struct type
                            argStruct = (cast(TypeStruct)targ).sym;
                            prmStruct = (cast(TypeStruct)tprm).sym;
                        }

                        // check if the copy constructor may be called to copy the argument
                        if (argStruct && argStruct == prmStruct && argStruct.hasCopyCtor)
                        {
                            /* this is done by seeing if a call to the copy constructor can be made:
                             *
                             * typeof(tprm) __copytmp;
                             * copytmp.__copyCtor(arg);
                             */
                            auto tmp = new VarDeclaration(arg.loc, tprm, Identifier.generateId("__copytmp"), null);
                            tmp.storage_class = STC.rvalue | STC.temp | STC.ctfe;
                            tmp.dsymbolSemantic(sc);
                            Expression ve = new VarExp(arg.loc, tmp);
                            Expression e = new DotIdExp(arg.loc, ve, Id.ctor);
                            e = new CallExp(arg.loc, e, arg);
                            //printf("e = %s\n", e.toChars());
                            if(.trySemantic(e, sc))
                                m = MATCH.exact;
                            else
                            {
                                m = MATCH.nomatch;
                                if (pMessage)
                                {
                                    OutBuffer buf;
                                    buf.printf("`struct %s` does not define a copy constructor for `%s` to `%s` copies",
                                           argStruct.toChars(), targ.toChars(), tprm.toChars());
                                    *pMessage = buf.extractChars();
                                }
                                goto Nomatch;
                            }
                        }
                        else
                            m = arg.implicitConvTo(tprm);
                    }
                    //printf("match %d\n", m);
                }

                // Non-lvalues do not match ref or out parameters
                if (p.storageClass & (STC.ref_ | STC.out_))
                {
                    // https://issues.dlang.org/show_bug.cgi?id=13783
                    // Don't use toBasetype() to handle enum types.
                    Type ta = targ;
                    Type tp = tprm;
                    //printf("fparam[%d] ta = %s, tp = %s\n", u, ta.toChars(), tp.toChars());

                    if (m && !arg.isLvalue())
                    {
                        if (p.storageClass & STC.out_)
                        {
                            if (pMessage) *pMessage = getParamError(arg, p);
                            goto Nomatch;
                        }

                        if (arg.op == TOK.string_ && tp.ty == Tsarray)
                        {
                            if (ta.ty != Tsarray)
                            {
                                Type tn = tp.nextOf().castMod(ta.nextOf().mod);
                                dinteger_t dim = (cast(StringExp)arg).len;
                                ta = tn.sarrayOf(dim);
                            }
                        }
                        else if (arg.op == TOK.slice && tp.ty == Tsarray)
                        {
                            // Allow conversion from T[lwr .. upr] to ref T[upr-lwr]
                            if (ta.ty != Tsarray)
                            {
                                Type tn = ta.nextOf();
                                dinteger_t dim = (cast(TypeSArray)tp).dim.toUInteger();
                                ta = tn.sarrayOf(dim);
                            }
                        }
                        else if (!global.params.rvalueRefParam ||
                                 p.storageClass & STC.out_ ||
                                 !arg.type.isCopyable())  // can't copy to temp for ref parameter
                        {
                            if (pMessage) *pMessage = getParamError(arg, p);
                            goto Nomatch;
                        }
                        else
                        {
                            /* in functionParameters() we'll convert this
                             * rvalue into a temporary
                             */
                            m = MATCH.convert;
                        }
                    }

                    /* Find most derived alias this type being matched.
                     * https://issues.dlang.org/show_bug.cgi?id=15674
                     * Allow on both ref and out parameters.
                     */
                    while (1)
                    {
                        Type tab = ta.toBasetype();
                        Type tat = tab.aliasthisOf();
                        if (!tat || !tat.implicitConvTo(tprm))
                            break;
                        if (tat == tab)
                            break;
                        ta = tat;
                    }

                    /* A ref variable should work like a head-const reference.
                     * e.g. disallows:
                     *  ref T      <- an lvalue of const(T) argument
                     *  ref T[dim] <- an lvalue of const(T[dim]) argument
                     */
                    if (!ta.constConv(tp))
                    {
                        if (pMessage) *pMessage = getParamError(arg, p);
                        goto Nomatch;
                    }
                }
            }

            /* prefer matching the element type rather than the array
             * type when more arguments are present with T[]...
             */
            if (parameterList.varargs == VarArg.typesafe && u + 1 == nparams && nargs > nparams)
                goto L1;

            //printf("\tm = %d\n", m);
            if (m == MATCH.nomatch) // if no match
            {
            L1:
                if (parameterList.varargs == VarArg.typesafe && u + 1 == nparams) // if last varargs param
                {
                    Type tb = p.type.toBasetype();
                    TypeSArray tsa;
                    dinteger_t sz;

                    switch (tb.ty)
                    {
                    case Tsarray:
                        tsa = cast(TypeSArray)tb;
                        sz = tsa.dim.toInteger();
                        if (sz != nargs - u)
                        {
                            if (pMessage)
                                // Windows (Vista) OutBuffer.vprintf issue? 2nd argument always zero
                                //*pMessage = getMatchError("expected %d variadic argument(s), not %d", sz, nargs - u);
                            if (!global.gag || global.params.showGaggedErrors)
                            {
                                OutBuffer buf;
                                buf.printf("expected %d variadic argument(s)", sz);
                                buf.printf(", not %d", nargs - u);
                                *pMessage = buf.extractChars();
                            }
                            goto Nomatch;
                        }
                        goto case Tarray;
                    case Tarray:
                        {
                            TypeArray ta = cast(TypeArray)tb;
                            foreach (arg; args[u .. nargs])
                            {
                                assert(arg);

                                /* If lazy array of delegates,
                                 * convert arg(s) to delegate(s)
                                 */
                                Type tret = p.isLazyArray();
                                if (tret)
                                {
                                    if (ta.next.equals(arg.type))
                                        m = MATCH.exact;
                                    else if (tret.toBasetype().ty == Tvoid)
                                        m = MATCH.convert;
                                    else
                                    {
                                        m = arg.implicitConvTo(tret);
                                        if (m == MATCH.nomatch)
                                            m = arg.implicitConvTo(ta.next);
                                    }
                                }
                                else
                                    m = arg.implicitConvTo(ta.next);

                                if (m == MATCH.nomatch)
                                {
                                    if (pMessage) *pMessage = getParamError(arg, p);
                                    goto Nomatch;
                                }
                                if (m < match)
                                    match = m;
                            }
                            goto Ldone;
                        }
                    case Tclass:
                        // Should see if there's a constructor match?
                        // Or just leave it ambiguous?
                        goto Ldone;

                    default:
                        break;
                    }
                }
                if (pMessage && u < nargs)
                    *pMessage = getParamError(args[u], p);
                else if (pMessage)
                    *pMessage = getMatchError("missing argument for parameter #%d: `%s`",
                        u + 1, parameterToChars(p, this, false));
                goto Nomatch;
            }
            if (m < match)
                match = m; // pick worst match
        }

    Ldone:
        if (pMessage && !parameterList.varargs && nargs > nparams)
        {
            // all parameters had a match, but there are surplus args
            *pMessage = getMatchError("expected %d argument(s), not %d", nparams, nargs);
            goto Nomatch;
        }
        //printf("match = %d\n", match);
        return match;

    Nomatch:
        //printf("no match\n");
        return MATCH.nomatch;
    }

    extern (D) bool checkRetType(const ref Loc loc)
    {
        Type tb = next.toBasetype();
        if (tb.ty == Tfunction)
        {
            error(loc, "functions cannot return a function");
            next = Type.terror;
        }
        if (tb.ty == Ttuple)
        {
            error(loc, "functions cannot return a tuple");
            next = Type.terror;
        }
        if (!isref && (tb.ty == Tstruct || tb.ty == Tsarray))
        {
            if (auto ts = tb.baseElemOf().isTypeStruct())
            {
                if (!ts.sym.members)
                {
                    error(loc, "functions cannot return opaque type `%s` by value", tb.toChars());
                    next = Type.terror;
                }
            }
        }
        if (tb.ty == Terror)
            return true;
        return false;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class TypeDelegate : TypeNext
{
    // .next is a TypeFunction

    extern (D) this(Type t)
    {
        super(Tfunction, t);
        ty = Tdelegate;
    }

    static TypeDelegate create(Type t)
    {
        return new TypeDelegate(t);
    }

    override const(char)* kind() const
    {
        return "delegate";
    }

    override Type syntaxCopy()
    {
        Type t = next.syntaxCopy();
        if (t == next)
            t = this;
        else
        {
            t = new TypeDelegate(t);
            t.mod = mod;
        }
        return t;
    }

    override Type addStorageClass(StorageClass stc)
    {
        TypeDelegate t = cast(TypeDelegate)Type.addStorageClass(stc);
        if (!global.params.vsafe)
            return t;

        /* The rest is meant to add 'scope' to a delegate declaration if it is of the form:
         *  alias dg_t = void* delegate();
         *  scope dg_t dg = ...;
         */
        if(stc & STC.scope_)
        {
            auto n = t.next.addStorageClass(STC.scope_ | STC.scopeinferred);
            if (n != t.next)
            {
                t.next = n;
                t.deco = t.merge().deco; // mangling supposed to not be changed due to STC.scope_inferrred
            }
        }
        return t;
    }

    override d_uns64 size(const ref Loc loc) const
    {
        return target.ptrsize * 2;
    }

    override uint alignsize() const
    {
        return target.ptrsize;
    }

    override MATCH implicitConvTo(Type to)
    {
        //printf("TypeDelegate.implicitConvTo(this=%p, to=%p)\n", this, to);
        //printf("from: %s\n", toChars());
        //printf("to  : %s\n", to.toChars());
        if (this == to)
            return MATCH.exact;

        version (all)
        {
            // not allowing covariant conversions because it interferes with overriding
            if (to.ty == Tdelegate && this.nextOf().covariant(to.nextOf()) == 1)
            {
                Type tret = this.next.nextOf();
                Type toret = (cast(TypeDelegate)to).next.nextOf();
                if (tret.ty == Tclass && toret.ty == Tclass)
                {
                    /* https://issues.dlang.org/show_bug.cgi?id=10219
                     * Check covariant interface return with offset tweaking.
                     * interface I {}
                     * class C : Object, I {}
                     * I delegate() dg = delegate C() {}    // should be error
                     */
                    int offset = 0;
                    if (toret.isBaseOf(tret, &offset) && offset != 0)
                        return MATCH.nomatch;
                }
                return MATCH.convert;
            }
        }

        return MATCH.nomatch;
    }

    override bool isZeroInit(const ref Loc loc) const
    {
        return true;
    }

    override bool isBoolean() const
    {
        return true;
    }

    override bool hasPointers() const
    {
        return true;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/**
 * This is a shell containing a TraitsExp that can be
 * either resolved to a type or to a symbol.
 *
 * The point is to allow AliasDeclarationY to use `__traits()`, see issue 7804.
 */
extern (C++) final class TypeTraits : Type
{
    Loc loc;
    /// The expression to resolve as type or symbol.
    TraitsExp exp;
    /// After `typeSemantic` the symbol when `exp` doesn't represent a type.
    Dsymbol sym;

    final extern (D) this(const ref Loc loc, TraitsExp exp)
    {
        super(Ttraits);
        this.loc = loc;
        this.exp = exp;
    }

    override Type syntaxCopy()
    {
        TraitsExp te = cast(TraitsExp) exp.syntaxCopy();
        TypeTraits tt = new TypeTraits(loc, te);
        tt.mod = mod;
        return tt;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }

    override d_uns64 size(const ref Loc loc)
    {
        return SIZE_INVALID;
    }
}

/******
 * Implements mixin types.
 *
 * Semantic analysis will convert it to a real type.
 */
extern (C++) final class TypeMixin : Type
{
    Loc loc;
    Expressions* exps;

    extern (D) this(const ref Loc loc, Expressions* exps)
    {
        super(Tmixin);
        this.loc = loc;
        this.exps = exps;
    }

    override const(char)* kind() const
    {
        return "mixin";
    }

    override Type syntaxCopy()
    {
        return new TypeMixin(loc, Expression.arraySyntaxCopy(exps));
    }

   override Dsymbol toDsymbol(Scope* sc)
    {
        Type t;
        Expression e;
        Dsymbol s;
        resolve(this, loc, sc, &e, &t, &s);
        if (t)
            s = t.toDsymbol(sc);
        else if (e)
            s = getDsymbol(e);

        return s;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) abstract class TypeQualified : Type
{
    Loc loc;

    // array of Identifier and TypeInstance,
    // representing ident.ident!tiargs.ident. ... etc.
    Objects idents;

    final extern (D) this(TY ty, Loc loc)
    {
        super(ty);
        this.loc = loc;
    }

    final void syntaxCopyHelper(TypeQualified t)
    {
        //printf("TypeQualified::syntaxCopyHelper(%s) %s\n", t.toChars(), toChars());
        idents.setDim(t.idents.dim);
        for (size_t i = 0; i < idents.dim; i++)
        {
            RootObject id = t.idents[i];
            if (id.dyncast() == DYNCAST.dsymbol)
            {
                TemplateInstance ti = cast(TemplateInstance)id;
                ti = cast(TemplateInstance)ti.syntaxCopy(null);
                id = ti;
            }
            else if (id.dyncast() == DYNCAST.expression)
            {
                Expression e = cast(Expression)id;
                e = e.syntaxCopy();
                id = e;
            }
            else if (id.dyncast() == DYNCAST.type)
            {
                Type tx = cast(Type)id;
                tx = tx.syntaxCopy();
                id = tx;
            }
            idents[i] = id;
        }
    }

    final void addIdent(Identifier ident)
    {
        idents.push(ident);
    }

    final void addInst(TemplateInstance inst)
    {
        idents.push(inst);
    }

    final void addIndex(RootObject e)
    {
        idents.push(e);
    }

    override d_uns64 size(const ref Loc loc)
    {
        error(this.loc, "size of type `%s` is not known", toChars());
        return SIZE_INVALID;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class TypeIdentifier : TypeQualified
{
    Identifier ident;

    // The symbol representing this identifier, before alias resolution
    Dsymbol originalSymbol;

    extern (D) this(const ref Loc loc, Identifier ident)
    {
        super(Tident, loc);
        this.ident = ident;
    }

    override const(char)* kind() const
    {
        return "identifier";
    }

    override Type syntaxCopy()
    {
        auto t = new TypeIdentifier(loc, ident);
        t.syntaxCopyHelper(this);
        t.mod = mod;
        return t;
    }

    /*****************************************
     * See if type resolves to a symbol, if so,
     * return that symbol.
     */
    override Dsymbol toDsymbol(Scope* sc)
    {
        //printf("TypeIdentifier::toDsymbol('%s')\n", toChars());
        if (!sc)
            return null;

        Type t;
        Expression e;
        Dsymbol s;
        resolve(this, loc, sc, &e, &t, &s);
        if (t && t.ty != Tident)
            s = t.toDsymbol(sc);
        if (e)
            s = getDsymbol(e);

        return s;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * Similar to TypeIdentifier, but with a TemplateInstance as the root
 */
extern (C++) final class TypeInstance : TypeQualified
{
    TemplateInstance tempinst;

    extern (D) this(const ref Loc loc, TemplateInstance tempinst)
    {
        super(Tinstance, loc);
        this.tempinst = tempinst;
    }

    override const(char)* kind() const
    {
        return "instance";
    }

    override Type syntaxCopy()
    {
        //printf("TypeInstance::syntaxCopy() %s, %d\n", toChars(), idents.dim);
        auto t = new TypeInstance(loc, cast(TemplateInstance)tempinst.syntaxCopy(null));
        t.syntaxCopyHelper(this);
        t.mod = mod;
        return t;
    }

    override Dsymbol toDsymbol(Scope* sc)
    {
        Type t;
        Expression e;
        Dsymbol s;
        //printf("TypeInstance::semantic(%s)\n", toChars());
        resolve(this, loc, sc, &e, &t, &s);
        if (t && t.ty != Tinstance)
            s = t.toDsymbol(sc);
        return s;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class TypeTypeof : TypeQualified
{
    Expression exp;
    int inuse;

    extern (D) this(const ref Loc loc, Expression exp)
    {
        super(Ttypeof, loc);
        this.exp = exp;
    }

    override const(char)* kind() const
    {
        return "typeof";
    }

    override Type syntaxCopy()
    {
        //printf("TypeTypeof::syntaxCopy() %s\n", toChars());
        auto t = new TypeTypeof(loc, exp.syntaxCopy());
        t.syntaxCopyHelper(this);
        t.mod = mod;
        return t;
    }

    override Dsymbol toDsymbol(Scope* sc)
    {
        //printf("TypeTypeof::toDsymbol('%s')\n", toChars());
        Expression e;
        Type t;
        Dsymbol s;
        resolve(this, loc, sc, &e, &t, &s);
        return s;
    }

    override d_uns64 size(const ref Loc loc)
    {
        if (exp.type)
            return exp.type.size(loc);
        else
            return TypeQualified.size(loc);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class TypeReturn : TypeQualified
{
    extern (D) this(const ref Loc loc)
    {
        super(Treturn, loc);
    }

    override const(char)* kind() const
    {
        return "return";
    }

    override Type syntaxCopy()
    {
        auto t = new TypeReturn(loc);
        t.syntaxCopyHelper(this);
        t.mod = mod;
        return t;
    }

    override Dsymbol toDsymbol(Scope* sc)
    {
        Expression e;
        Type t;
        Dsymbol s;
        resolve(this, loc, sc, &e, &t, &s);
        return s;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

// Whether alias this dependency is recursive or not.
enum AliasThisRec : int
{
    no           = 0,    // no alias this recursion
    yes          = 1,    // alias this has recursive dependency
    fwdref       = 2,    // not yet known
    typeMask     = 3,    // mask to read no/yes/fwdref
    tracing      = 0x4,  // mark in progress of implicitConvTo/deduceWild
    tracingDT    = 0x8,  // mark in progress of deduceType
}

/***********************************************************
 */
extern (C++) final class TypeStruct : Type
{
    StructDeclaration sym;
    AliasThisRec att = AliasThisRec.fwdref;
    CPPMANGLE cppmangle = CPPMANGLE.def;

    extern (D) this(StructDeclaration sym)
    {
        super(Tstruct);
        this.sym = sym;
    }

    static TypeStruct create(StructDeclaration sym)
    {
        return new TypeStruct(sym);
    }

    override const(char)* kind() const
    {
        return "struct";
    }

    override d_uns64 size(const ref Loc loc)
    {
        return sym.size(loc);
    }

    override uint alignsize()
    {
        sym.size(Loc.initial); // give error for forward references
        return sym.alignsize;
    }

    override Type syntaxCopy()
    {
        return this;
    }

    override Dsymbol toDsymbol(Scope* sc)
    {
        return sym;
    }

    override structalign_t alignment()
    {
        if (sym.alignment == 0)
            sym.size(sym.loc);
        return sym.alignment;
    }

    /***************************************
     * Use when we prefer the default initializer to be a literal,
     * rather than a global immutable variable.
     */
    override Expression defaultInitLiteral(const ref Loc loc)
    {
        static if (LOGDEFAULTINIT)
        {
            printf("TypeStruct::defaultInitLiteral() '%s'\n", toChars());
        }
        sym.size(loc);
        if (sym.sizeok != Sizeok.done)
            return new ErrorExp();

        auto structelems = new Expressions(sym.nonHiddenFields());
        uint offset = 0;
        foreach (j; 0 .. structelems.dim)
        {
            VarDeclaration vd = sym.fields[j];
            Expression e;
            if (vd.inuse)
            {
                error(loc, "circular reference to `%s`", vd.toPrettyChars());
                return new ErrorExp();
            }
            if (vd.offset < offset || vd.type.size() == 0)
                e = null;
            else if (vd._init)
            {
                if (vd._init.isVoidInitializer())
                    e = null;
                else
                    e = vd.getConstInitializer(false);
            }
            else
                e = vd.type.defaultInitLiteral(loc);
            if (e && e.op == TOK.error)
                return e;
            if (e)
                offset = vd.offset + cast(uint)vd.type.size();
            (*structelems)[j] = e;
        }
        auto structinit = new StructLiteralExp(loc, sym, structelems);

        /* Copy from the initializer symbol for larger symbols,
         * otherwise the literals expressed as code get excessively large.
         */
        if (size(loc) > target.ptrsize * 4 && !needsNested())
            structinit.useStaticInit = true;

        structinit.type = this;
        return structinit;
    }

    override bool isZeroInit(const ref Loc loc) const
    {
        return sym.zeroInit;
    }

    override bool isAssignable()
    {
        bool assignable = true;
        uint offset = ~0; // dead-store initialize to prevent spurious warning

        sym.determineSize(sym.loc);

        /* If any of the fields are const or immutable,
         * then one cannot assign this struct.
         */
        for (size_t i = 0; i < sym.fields.dim; i++)
        {
            VarDeclaration v = sym.fields[i];
            //printf("%s [%d] v = (%s) %s, v.offset = %d, v.parent = %s\n", sym.toChars(), i, v.kind(), v.toChars(), v.offset, v.parent.kind());
            if (i == 0)
            {
            }
            else if (v.offset == offset)
            {
                /* If any fields of anonymous union are assignable,
                 * then regard union as assignable.
                 * This is to support unsafe things like Rebindable templates.
                 */
                if (assignable)
                    continue;
            }
            else
            {
                if (!assignable)
                    return false;
            }
            assignable = v.type.isMutable() && v.type.isAssignable();
            offset = v.offset;
            //printf(" -> assignable = %d\n", assignable);
        }

        return assignable;
    }

    override bool isBoolean() const
    {
        return false;
    }

    override bool needsDestruction() const
    {
        return sym.dtor !is null;
    }

    override bool needsNested()
    {
        if (sym.isNested())
            return true;

        for (size_t i = 0; i < sym.fields.dim; i++)
        {
            VarDeclaration v = sym.fields[i];
            if (!v.isDataseg() && v.type.needsNested())
                return true;
        }
        return false;
    }

    override bool hasPointers()
    {
        // Probably should cache this information in sym rather than recompute
        StructDeclaration s = sym;

        if (sym.members && !sym.determineFields() && sym.type != Type.terror)
            error(sym.loc, "no size because of forward references");

        foreach (VarDeclaration v; s.fields)
        {
            if (v.storage_class & STC.ref_ || v.hasPointers())
                return true;
        }
        return false;
    }

    override bool hasVoidInitPointers()
    {
        // Probably should cache this information in sym rather than recompute
        StructDeclaration s = sym;

        sym.size(Loc.initial); // give error for forward references
        foreach (VarDeclaration v; s.fields)
        {
            if (v._init && v._init.isVoidInitializer() && v.type.hasPointers())
                return true;
            if (!v._init && v.type.hasVoidInitPointers())
                return true;
        }
        return false;
    }

    extern (D) MATCH implicitConvToWithoutAliasThis(Type to)
    {
        MATCH m;

        if (ty == to.ty && sym == (cast(TypeStruct)to).sym)
        {
            m = MATCH.exact; // exact match
            if (mod != to.mod)
            {
                m = MATCH.constant;
                if (MODimplicitConv(mod, to.mod))
                {
                }
                else
                {
                    /* Check all the fields. If they can all be converted,
                     * allow the conversion.
                     */
                    uint offset = ~0; // dead-store to prevent spurious warning
                    for (size_t i = 0; i < sym.fields.dim; i++)
                    {
                        VarDeclaration v = sym.fields[i];
                        if (i == 0)
                        {
                        }
                        else if (v.offset == offset)
                        {
                            if (m > MATCH.nomatch)
                                continue;
                        }
                        else
                        {
                            if (m <= MATCH.nomatch)
                                return m;
                        }

                        // 'from' type
                        Type tvf = v.type.addMod(mod);

                        // 'to' type
                        Type tv = v.type.addMod(to.mod);

                        // field match
                        MATCH mf = tvf.implicitConvTo(tv);
                        //printf("\t%s => %s, match = %d\n", v.type.toChars(), tv.toChars(), mf);

                        if (mf <= MATCH.nomatch)
                            return mf;
                        if (mf < m) // if field match is worse
                            m = mf;
                        offset = v.offset;
                    }
                }
            }
        }
        return m;
    }

    extern (D) MATCH implicitConvToThroughAliasThis(Type to)
    {
        MATCH m;
        if (!(ty == to.ty && sym == (cast(TypeStruct)to).sym) && sym.aliasthis && !(att & AliasThisRec.tracing))
        {
            if (auto ato = aliasthisOf())
            {
                att = cast(AliasThisRec)(att | AliasThisRec.tracing);
                m = ato.implicitConvTo(to);
                att = cast(AliasThisRec)(att & ~AliasThisRec.tracing);
            }
            else
                m = MATCH.nomatch; // no match
        }
        return m;
    }

    override MATCH implicitConvTo(Type to)
    {
        //printf("TypeStruct::implicitConvTo(%s => %s)\n", toChars(), to.toChars());
        MATCH m = implicitConvToWithoutAliasThis(to);
        return m ? m : implicitConvToThroughAliasThis(to);
    }

    override MATCH constConv(Type to)
    {
        if (equals(to))
            return MATCH.exact;
        if (ty == to.ty && sym == (cast(TypeStruct)to).sym && MODimplicitConv(mod, to.mod))
            return MATCH.constant;
        return MATCH.nomatch;
    }

    override MOD deduceWild(Type t, bool isRef)
    {
        if (ty == t.ty && sym == (cast(TypeStruct)t).sym)
            return Type.deduceWild(t, isRef);

        ubyte wm = 0;

        if (t.hasWild() && sym.aliasthis && !(att & AliasThisRec.tracing))
        {
            if (auto ato = aliasthisOf())
            {
                att = cast(AliasThisRec)(att | AliasThisRec.tracing);
                wm = ato.deduceWild(t, isRef);
                att = cast(AliasThisRec)(att & ~AliasThisRec.tracing);
            }
        }

        return wm;
    }

    override inout(Type) toHeadMutable() inout
    {
        return this;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class TypeEnum : Type
{
    EnumDeclaration sym;

    extern (D) this(EnumDeclaration sym)
    {
        super(Tenum);
        this.sym = sym;
    }

    override const(char)* kind() const
    {
        return "enum";
    }

    override Type syntaxCopy()
    {
        return this;
    }

    override d_uns64 size(const ref Loc loc)
    {
        return sym.getMemtype(loc).size(loc);
    }

    Type memType(const ref Loc loc = Loc.initial)
    {
        return sym.getMemtype(loc);
    }
    override uint alignsize()
    {
        Type t = memType();
        if (t.ty == Terror)
            return 4;
        return t.alignsize();
    }

    override Dsymbol toDsymbol(Scope* sc)
    {
        return sym;
    }

    override bool isintegral()
    {
        return memType().isintegral();
    }

    override bool isfloating()
    {
        return memType().isfloating();
    }

    override bool isreal()
    {
        return memType().isreal();
    }

    override bool isimaginary()
    {
        return memType().isimaginary();
    }

    override bool iscomplex()
    {
        return memType().iscomplex();
    }

    override bool isscalar()
    {
        return memType().isscalar();
    }

    override bool isunsigned()
    {
        return memType().isunsigned();
    }

    override bool ischar()
    {
        return memType().ischar();
    }

    override bool isBoolean()
    {
        return memType().isBoolean();
    }

    override bool isString()
    {
        return memType().isString();
    }

    override bool isAssignable()
    {
        return memType().isAssignable();
    }

    override bool needsDestruction()
    {
        return memType().needsDestruction();
    }

    override bool needsNested()
    {
        return memType().needsNested();
    }

    override MATCH implicitConvTo(Type to)
    {
        MATCH m;
        //printf("TypeEnum::implicitConvTo() %s to %s\n", toChars(), to.toChars());
        if (ty == to.ty && sym == (cast(TypeEnum)to).sym)
            m = (mod == to.mod) ? MATCH.exact : MATCH.constant;
        else if (sym.getMemtype(Loc.initial).implicitConvTo(to))
            m = MATCH.convert; // match with conversions
        else
            m = MATCH.nomatch; // no match
        return m;
    }

    override MATCH constConv(Type to)
    {
        if (equals(to))
            return MATCH.exact;
        if (ty == to.ty && sym == (cast(TypeEnum)to).sym && MODimplicitConv(mod, to.mod))
            return MATCH.constant;
        return MATCH.nomatch;
    }

    override Type toBasetype()
    {
        if (!sym.members && !sym.memtype)
            return this;
        auto tb = sym.getMemtype(Loc.initial).toBasetype();
        return tb.castMod(mod);         // retain modifier bits from 'this'
    }

    override bool isZeroInit(const ref Loc loc)
    {
        return sym.getDefaultValue(loc).isBool(false);
    }

    override bool hasPointers()
    {
        return memType().hasPointers();
    }

    override bool hasVoidInitPointers()
    {
        return memType().hasVoidInitPointers();
    }

    override Type nextOf()
    {
        return memType().nextOf();
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class TypeClass : Type
{
    ClassDeclaration sym;
    AliasThisRec att = AliasThisRec.fwdref;
    CPPMANGLE cppmangle = CPPMANGLE.def;

    extern (D) this(ClassDeclaration sym)
    {
        super(Tclass);
        this.sym = sym;
    }

    override const(char)* kind() const
    {
        return "class";
    }

    override d_uns64 size(const ref Loc loc) const
    {
        return target.ptrsize;
    }

    override Type syntaxCopy()
    {
        return this;
    }

    override Dsymbol toDsymbol(Scope* sc)
    {
        return sym;
    }

    override inout(ClassDeclaration) isClassHandle() inout
    {
        return sym;
    }

    override bool isBaseOf(Type t, int* poffset)
    {
        if (t && t.ty == Tclass)
        {
            ClassDeclaration cd = (cast(TypeClass)t).sym;
            if (sym.isBaseOf(cd, poffset))
                return true;
        }
        return false;
    }

    extern (D) MATCH implicitConvToWithoutAliasThis(Type to)
    {
        MATCH m = constConv(to);
        if (m > MATCH.nomatch)
            return m;

        ClassDeclaration cdto = to.isClassHandle();
        if (cdto)
        {
            //printf("TypeClass::implicitConvTo(to = '%s') %s, isbase = %d %d\n", to.toChars(), toChars(), cdto.isBaseInfoComplete(), sym.isBaseInfoComplete());
            if (cdto.semanticRun < PASS.semanticdone && !cdto.isBaseInfoComplete())
                cdto.dsymbolSemantic(null);
            if (sym.semanticRun < PASS.semanticdone && !sym.isBaseInfoComplete())
                sym.dsymbolSemantic(null);
            if (cdto.isBaseOf(sym, null) && MODimplicitConv(mod, to.mod))
            {
                //printf("'to' is base\n");
                return MATCH.convert;
            }
        }
        return MATCH.nomatch;
    }

    extern (D) MATCH implicitConvToThroughAliasThis(Type to)
    {
        MATCH m;
        if (sym.aliasthis && !(att & AliasThisRec.tracing))
        {
            if (auto ato = aliasthisOf())
            {
                att = cast(AliasThisRec)(att | AliasThisRec.tracing);
                m = ato.implicitConvTo(to);
                att = cast(AliasThisRec)(att & ~AliasThisRec.tracing);
            }
        }
        return m;
    }

    override MATCH implicitConvTo(Type to)
    {
        //printf("TypeClass::implicitConvTo(to = '%s') %s\n", to.toChars(), toChars());
        MATCH m = implicitConvToWithoutAliasThis(to);
        return m ? m : implicitConvToThroughAliasThis(to);
    }

    override MATCH constConv(Type to)
    {
        if (equals(to))
            return MATCH.exact;
        if (ty == to.ty && sym == (cast(TypeClass)to).sym && MODimplicitConv(mod, to.mod))
            return MATCH.constant;

        /* Conversion derived to const(base)
         */
        int offset = 0;
        if (to.isBaseOf(this, &offset) && offset == 0 && MODimplicitConv(mod, to.mod))
        {
            // Disallow:
            //  derived to base
            //  inout(derived) to inout(base)
            if (!to.isMutable() && !to.isWild())
                return MATCH.convert;
        }

        return MATCH.nomatch;
    }

    override MOD deduceWild(Type t, bool isRef)
    {
        ClassDeclaration cd = t.isClassHandle();
        if (cd && (sym == cd || cd.isBaseOf(sym, null)))
            return Type.deduceWild(t, isRef);

        ubyte wm = 0;

        if (t.hasWild() && sym.aliasthis && !(att & AliasThisRec.tracing))
        {
            if (auto ato = aliasthisOf())
            {
                att = cast(AliasThisRec)(att | AliasThisRec.tracing);
                wm = ato.deduceWild(t, isRef);
                att = cast(AliasThisRec)(att & ~AliasThisRec.tracing);
            }
        }

        return wm;
    }

    override inout(Type) toHeadMutable() inout
    {
        return this;
    }

    override bool isZeroInit(const ref Loc loc) const
    {
        return true;
    }

    override bool isscope() const
    {
        return sym.stack;
    }

    override bool isBoolean() const
    {
        return true;
    }

    override bool hasPointers() const
    {
        return true;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class TypeTuple : Type
{
    Parameters* arguments;  // types making up the tuple

    extern (D) this(Parameters* arguments)
    {
        super(Ttuple);
        //printf("TypeTuple(this = %p)\n", this);
        this.arguments = arguments;
        //printf("TypeTuple() %p, %s\n", this, toChars());
        debug
        {
            if (arguments)
            {
                for (size_t i = 0; i < arguments.dim; i++)
                {
                    Parameter arg = (*arguments)[i];
                    assert(arg && arg.type);
                }
            }
        }
    }

    /****************
     * Form TypeTuple from the types of the expressions.
     * Assume exps[] is already tuple expanded.
     */
    extern (D) this(Expressions* exps)
    {
        super(Ttuple);
        auto arguments = new Parameters();
        if (exps)
        {
            arguments.setDim(exps.dim);
            for (size_t i = 0; i < exps.dim; i++)
            {
                Expression e = (*exps)[i];
                if (e.type.ty == Ttuple)
                    e.error("cannot form tuple of tuples");
                auto arg = new Parameter(STC.undefined_, e.type, null, null, null);
                (*arguments)[i] = arg;
            }
        }
        this.arguments = arguments;
        //printf("TypeTuple() %p, %s\n", this, toChars());
    }

    static TypeTuple create(Parameters* arguments)
    {
        return new TypeTuple(arguments);
    }

    /*******************************************
     * Type tuple with 0, 1 or 2 types in it.
     */
    extern (D) this()
    {
        super(Ttuple);
        arguments = new Parameters();
    }

    extern (D) this(Type t1)
    {
        super(Ttuple);
        arguments = new Parameters();
        arguments.push(new Parameter(0, t1, null, null, null));
    }

    extern (D) this(Type t1, Type t2)
    {
        super(Ttuple);
        arguments = new Parameters();
        arguments.push(new Parameter(0, t1, null, null, null));
        arguments.push(new Parameter(0, t2, null, null, null));
    }

    static TypeTuple create()
    {
        return new TypeTuple();
    }

    static TypeTuple create(Type t1)
    {
        return new TypeTuple(t1);
    }

    static TypeTuple create(Type t1, Type t2)
    {
        return new TypeTuple(t1, t2);
    }

    override const(char)* kind() const
    {
        return "tuple";
    }

    override Type syntaxCopy()
    {
        Parameters* args = Parameter.arraySyntaxCopy(arguments);
        Type t = new TypeTuple(args);
        t.mod = mod;
        return t;
    }

    override bool equals(const RootObject o) const
    {
        Type t = cast(Type)o;
        //printf("TypeTuple::equals(%s, %s)\n", toChars(), t.toChars());
        if (this == t)
            return true;
        if (auto tt = t.isTypeTuple())
        {
            if (arguments.dim == tt.arguments.dim)
            {
                for (size_t i = 0; i < tt.arguments.dim; i++)
                {
                    const Parameter arg1 = (*arguments)[i];
                    Parameter arg2 = (*tt.arguments)[i];
                    if (!arg1.type.equals(arg2.type))
                        return false;
                }
                return true;
            }
        }
        return false;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * This is so we can slice a TypeTuple
 */
extern (C++) final class TypeSlice : TypeNext
{
    Expression lwr;
    Expression upr;

    extern (D) this(Type next, Expression lwr, Expression upr)
    {
        super(Tslice, next);
        //printf("TypeSlice[%s .. %s]\n", lwr.toChars(), upr.toChars());
        this.lwr = lwr;
        this.upr = upr;
    }

    override const(char)* kind() const
    {
        return "slice";
    }

    override Type syntaxCopy()
    {
        Type t = new TypeSlice(next.syntaxCopy(), lwr.syntaxCopy(), upr.syntaxCopy());
        t.mod = mod;
        return t;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class TypeNull : Type
{
    extern (D) this()
    {
        //printf("TypeNull %p\n", this);
        super(Tnull);
    }

    override const(char)* kind() const
    {
        return "null";
    }

    override Type syntaxCopy()
    {
        // No semantic analysis done, no need to copy
        return this;
    }

    override MATCH implicitConvTo(Type to)
    {
        //printf("TypeNull::implicitConvTo(this=%p, to=%p)\n", this, to);
        //printf("from: %s\n", toChars());
        //printf("to  : %s\n", to.toChars());
        MATCH m = Type.implicitConvTo(to);
        if (m != MATCH.nomatch)
            return m;

        // NULL implicitly converts to any pointer type or dynamic array
        //if (type.ty == Tpointer && type.nextOf().ty == Tvoid)
        {
            Type tb = to.toBasetype();
            if (tb.ty == Tnull || tb.ty == Tpointer || tb.ty == Tarray || tb.ty == Taarray || tb.ty == Tclass || tb.ty == Tdelegate)
                return MATCH.constant;
        }

        return MATCH.nomatch;
    }

    override bool hasPointers()
    {
        /* Although null isn't dereferencable, treat it as a pointer type for
         * attribute inference, generic code, etc.
         */
        return true;
    }

    override bool isBoolean() const
    {
        return true;
    }

    override d_uns64 size(const ref Loc loc) const
    {
        return tvoidptr.size(loc);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * Encapsulate Parameters* so .length and [i] can be used on it.
 * https://dlang.org/spec/function.html#ParameterList
 */
extern (C++) struct ParameterList
{
    Parameters* parameters;
    VarArg varargs = VarArg.none;

    size_t length()
    {
        return Parameter.dim(parameters);
    }

    Parameter opIndex(size_t i)
    {
        return Parameter.getNth(parameters, i);
    }

    alias parameters this;
}


/***********************************************************
 */
extern (C++) final class Parameter : ASTNode
{
    import dmd.attrib : UserAttributeDeclaration;

    StorageClass storageClass;
    Type type;
    Identifier ident;
    Expression defaultArg;
    UserAttributeDeclaration userAttribDecl; // user defined attributes

    extern (D) this(StorageClass storageClass, Type type, Identifier ident, Expression defaultArg, UserAttributeDeclaration userAttribDecl)
    {
        this.type = type;
        this.ident = ident;
        this.storageClass = storageClass;
        this.defaultArg = defaultArg;
        this.userAttribDecl = userAttribDecl;
    }

    static Parameter create(StorageClass storageClass, Type type, Identifier ident, Expression defaultArg, UserAttributeDeclaration userAttribDecl)
    {
        return new Parameter(storageClass, type, ident, defaultArg, userAttribDecl);
    }

    Parameter syntaxCopy()
    {
        return new Parameter(storageClass, type ? type.syntaxCopy() : null, ident, defaultArg ? defaultArg.syntaxCopy() : null, userAttribDecl ? cast(UserAttributeDeclaration) userAttribDecl.syntaxCopy(null) : null);
    }

    /****************************************************
     * Determine if parameter is a lazy array of delegates.
     * If so, return the return type of those delegates.
     * If not, return NULL.
     *
     * Returns T if the type is one of the following forms:
     *      T delegate()[]
     *      T delegate()[dim]
     */
    Type isLazyArray()
    {
        Type tb = type.toBasetype();
        if (tb.ty == Tsarray || tb.ty == Tarray)
        {
            Type tel = (cast(TypeArray)tb).next.toBasetype();
            if (auto td = tel.isTypeDelegate())
            {
                TypeFunction tf = td.next.toTypeFunction();
                if (tf.parameterList.varargs == VarArg.none && tf.parameterList.length == 0)
                {
                    return tf.next; // return type of delegate
                }
            }
        }
        return null;
    }

    // kludge for template.isType()
    override DYNCAST dyncast() const
    {
        return DYNCAST.parameter;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }

    extern (D) static Parameters* arraySyntaxCopy(Parameters* parameters)
    {
        Parameters* params = null;
        if (parameters)
        {
            params = new Parameters(parameters.dim);
            for (size_t i = 0; i < params.dim; i++)
                (*params)[i] = (*parameters)[i].syntaxCopy();
        }
        return params;
    }

    /***************************************
     * Determine number of arguments, folding in tuples.
     */
    static size_t dim(Parameters* parameters)
    {
        size_t nargs = 0;

        int dimDg(size_t n, Parameter p)
        {
            ++nargs;
            return 0;
        }

        _foreach(parameters, &dimDg);
        return nargs;
    }

    /***************************************
     * Get nth Parameter, folding in tuples.
     * Returns:
     *      Parameter*      nth Parameter
     *      NULL            not found, *pn gets incremented by the number
     *                      of Parameters
     */
    static Parameter getNth(Parameters* parameters, size_t nth, size_t* pn = null)
    {
        Parameter param;

        int getNthParamDg(size_t n, Parameter p)
        {
            if (n == nth)
            {
                param = p;
                return 1;
            }
            return 0;
        }

        int res = _foreach(parameters, &getNthParamDg);
        return res ? param : null;
    }

    alias ForeachDg = extern (D) int delegate(size_t paramidx, Parameter param);

    /***************************************
     * Expands tuples in args in depth first order. Calls
     * dg(void *ctx, size_t argidx, Parameter *arg) for each Parameter.
     * If dg returns !=0, stops and returns that value else returns 0.
     * Use this function to avoid the O(N + N^2/2) complexity of
     * calculating dim and calling N times getNth.
     */
    extern (D) static int _foreach(Parameters* parameters, scope ForeachDg dg, size_t* pn = null)
    {
        assert(dg);
        if (!parameters)
            return 0;

        size_t n = pn ? *pn : 0; // take over index
        int result = 0;
        foreach (i; 0 .. parameters.dim)
        {
            Parameter p = (*parameters)[i];
            Type t = p.type.toBasetype();

            if (auto tu = t.isTypeTuple())
            {
                result = _foreach(tu.arguments, dg, &n);
            }
            else
                result = dg(n++, p);

            if (result)
                break;
        }

        if (pn)
            *pn = n; // update index
        return result;
    }

    override const(char)* toChars() const
    {
        return ident ? ident.toChars() : "__anonymous_param";
    }

    /*********************************
     * Compute covariance of parameters `this` and `p`
     * as determined by the storage classes of both.
     * Params:
     *  returnByRef = true if the function returns by ref
     *  p = Parameter to compare with
     * Returns:
     *  true = `this` can be used in place of `p`
     *  false = nope
     */
    bool isCovariant(bool returnByRef, const Parameter p) const pure nothrow @nogc @safe
    {
        enum stc = STC.ref_ | STC.in_ | STC.out_ | STC.lazy_;
        if ((this.storageClass & stc) != (p.storageClass & stc))
            return false;
        return isCovariantScope(returnByRef, this.storageClass, p.storageClass);
    }

    extern (D) private static bool isCovariantScope(bool returnByRef, StorageClass from, StorageClass to) pure nothrow @nogc @safe
    {
        if (from == to)
            return true;

        /* Shrinking the representation is necessary because StorageClass is so wide
         * Params:
         *   returnByRef = true if the function returns by ref
         *   stc = storage class of parameter
         */
        static uint buildSR(bool returnByRef, StorageClass stc) pure nothrow @nogc @safe
        {
            uint result;
            final switch (stc & (STC.ref_ | STC.scope_ | STC.return_))
            {
                case 0:                    result = SR.None;        break;
                case STC.ref_:               result = SR.Ref;         break;
                case STC.scope_:             result = SR.Scope;       break;
                case STC.return_ | STC.ref_:   result = SR.ReturnRef;   break;
                case STC.return_ | STC.scope_: result = SR.ReturnScope; break;
                case STC.ref_    | STC.scope_: result = SR.RefScope;    break;
                case STC.return_ | STC.ref_ | STC.scope_:
                    result = returnByRef ? SR.ReturnRef_Scope : SR.Ref_ReturnScope;
                    break;
            }
            return result;
        }

        /* result is true if the 'from' can be used as a 'to'
         */

        if ((from ^ to) & STC.ref_)               // differing in 'ref' means no covariance
            return false;

        return covariant[buildSR(returnByRef, from)][buildSR(returnByRef, to)];
    }

    /* Classification of 'scope-return-ref' possibilities
     */
    private enum SR
    {
        None,
        Scope,
        ReturnScope,
        Ref,
        ReturnRef,
        RefScope,
        ReturnRef_Scope,
        Ref_ReturnScope,
    }

    extern (D) private static bool[SR.max + 1][SR.max + 1] covariantInit() pure nothrow @nogc @safe
    {
        /* Initialize covariant[][] with this:

             From\To           n   rs  s
             None              X
             ReturnScope       X   X
             Scope             X   X   X

             From\To           r   rr  rs  rr-s r-rs
             Ref               X   X
             ReturnRef             X
             RefScope          X   X   X   X    X
             ReturnRef-Scope       X       X
             Ref-ReturnScope   X   X            X
        */
        bool[SR.max + 1][SR.max + 1] covariant;

        foreach (i; 0 .. SR.max + 1)
        {
            covariant[i][i] = true;
            covariant[SR.RefScope][i] = true;
        }
        covariant[SR.ReturnScope][SR.None]        = true;
        covariant[SR.Scope      ][SR.None]        = true;
        covariant[SR.Scope      ][SR.ReturnScope] = true;

        covariant[SR.Ref            ][SR.ReturnRef] = true;
        covariant[SR.ReturnRef_Scope][SR.ReturnRef] = true;
        covariant[SR.Ref_ReturnScope][SR.Ref      ] = true;
        covariant[SR.Ref_ReturnScope][SR.ReturnRef] = true;

        return covariant;
    }

    extern (D) private static immutable bool[SR.max + 1][SR.max + 1] covariant = covariantInit();
}

/*************************************************************
 * For printing two types with qualification when necessary.
 * Params:
 *    t1 = The first type to receive the type name for
 *    t2 = The second type to receive the type name for
 * Returns:
 *    The fully-qualified names of both types if the two type names are not the same,
 *    or the unqualified names of both types if the two type names are the same.
 */
const(char*)[2] toAutoQualChars(Type t1, Type t2)
{
    auto s1 = t1.toChars();
    auto s2 = t2.toChars();
    // show qualification only if it's different
    if (!t1.equals(t2) && strcmp(s1, s2) == 0)
    {
        s1 = t1.toPrettyChars(true);
        s2 = t2.toPrettyChars(true);
    }
    return [s1, s2];
}


/**
 * For each active modifier (MODFlags.const_, MODFlags.immutable_, etc) call `fp` with a
 * void* for the work param and a string representation of the attribute.
 */
void modifiersApply(const TypeFunction tf, void delegate(string) dg)
{
    immutable ubyte[4] modsArr = [MODFlags.const_, MODFlags.immutable_, MODFlags.wild, MODFlags.shared_];

    foreach (modsarr; modsArr)
    {
        if (tf.mod & modsarr)
        {
            dg(MODtoString(modsarr));
        }
    }
}

/**
 * For each active attribute (ref/const/nogc/etc) call `fp` with a void* for the
 * work param and a string representation of the attribute.
 */
void attributesApply(const TypeFunction tf, void delegate(string) dg, TRUSTformat trustFormat = TRUSTformatDefault)
{
    if (tf.purity)
        dg("pure");
    if (tf.isnothrow)
        dg("nothrow");
    if (tf.isnogc)
        dg("@nogc");
    if (tf.isproperty)
        dg("@property");
    if (tf.isref)
        dg("ref");
    if (tf.isreturn && !tf.isreturninferred)
        dg("return");
    if (tf.isscope && !tf.isscopeinferred)
        dg("scope");

    TRUST trustAttrib = tf.trust;

    if (trustAttrib == TRUST.default_)
    {
        if (trustFormat == TRUSTformatSystem)
            trustAttrib = TRUST.system;
        else
            return; // avoid calling with an empty string
    }

    dg(trustToString(trustAttrib));
}

/**
 * If the type is a class or struct, returns the symbol for it,
 * else null.
 */
extern (C++) AggregateDeclaration isAggregate(Type t)
{
    t = t.toBasetype();
    if (t.ty == Tclass)
        return (cast(TypeClass)t).sym;
    if (t.ty == Tstruct)
        return (cast(TypeStruct)t).sym;
    return null;
}

/***************************************************
 * Determine if type t can be indexed or sliced given that it is not an
 * aggregate with operator overloads.
 * Params:
 *      t = type to check
 * Returns:
 *      true if an expression of type t can be e1 in an array expression
 */
bool isIndexableNonAggregate(Type t)
{
    t = t.toBasetype();
    return (t.ty == Tpointer || t.ty == Tsarray || t.ty == Tarray || t.ty == Taarray ||
            t.ty == Ttuple || t.ty == Tvector);
}

/***************************************************
 * Determine if type t is copyable.
 * Params:
 *      t = type to check
 * Returns:
 *      true if we can copy it
 */
bool isCopyable(const Type t) pure nothrow @nogc
{
    //printf("isCopyable() %s\n", t.toChars());
    if (auto ts = t.isTypeStruct())
    {
        if (ts.sym.postblit &&
            ts.sym.postblit.storage_class & STC.disable)
            return false;
    }
    return true;
}
