/**
 * Compiler implementation of the
 * $(LINK2 http://www.dlang.org, D programming language).
 *
 * Copyright:   Copyright (C) 1999-2018 by The D Language Foundation, All Rights Reserved
 * Authors:     $(LINK2 http://www.digitalmars.com, Walter Bright)
 * License:     $(LINK2 http://www.boost.org/LICENSE_1_0.txt, Boost License 1.0)
 * Source:      $(LINK2 https://github.com/dlang/dmd/blob/master/src/dmd/target.d, _target.d)
 * Documentation:  https://dlang.org/phobos/dmd_target.html
 * Coverage:    https://codecov.io/gh/dlang/dmd/src/master/src/dmd/target.d
 */

module dmd.target;

import dmd.argtypes;
import dmd.cppmangle;
import dmd.cppmanglewin;
import dmd.dclass;
import dmd.declaration;
import dmd.dmodule;
import dmd.dsymbol;
import dmd.expression;
import dmd.globals;
import dmd.identifier;
import dmd.mtype;
import dmd.typesem;
import dmd.tokens : TOK;
import dmd.root.ctfloat;
import dmd.root.outbuffer;
version(IN_LLVM) import gen.llvmhelpers;

/**
 * Describes a back-end target. At present it is incomplete, but in the future
 * it should grow to contain most or all target machine and target O/S specific
 * information.
 *
 * In many cases, calls to sizeof() can't be used directly for getting data type
 * sizes since cross compiling is supported and would end up using the host
 * sizes rather than the target sizes.
 */
struct Target
{
    extern (C++) __gshared
    {
        // D ABI
        int ptrsize;              /// size of a pointer in bytes
        int realsize;             /// size a real consumes in memory
        int realpad;              /// padding added to the CPU real size to bring it up to realsize
        int realalignsize;        /// alignment for reals
        int classinfosize;        /// size of `ClassInfo`
        ulong maxStaticDataSize;  /// maximum size of static data

        // C ABI
        int c_longsize;           /// size of a C `long` or `unsigned long` type
        int c_long_doublesize;    /// size of a C `long double`

        // C++ ABI
        bool reverseCppOverloads; /// set if overloaded functions are grouped and in reverse order (such as in dmc and cl)
        bool cppExceptions;       /// set if catching C++ exceptions is supported
        char int64Mangle;         /// mangling character for C++ int64_t
        char uint64Mangle;        /// mangling character for C++ uint64_t
    }

  version(IN_LLVM)
  {
    extern (C++):

    struct FPTypeProperties
    {
        real_t max, min_normal, nan, snan, infinity, epsilon;
        d_int64 dig, mant_dig, max_exp, min_exp, max_10_exp, min_10_exp;

        static FPTypeProperties fromDHostCompiler(T)()
        {
            FPTypeProperties p;

            p.max = T.max;
            p.min_normal = T.min_normal;
            p.nan = T.nan;
            p.snan = T.init;
            p.infinity = T.infinity;
            p.epsilon = T.epsilon;

            p.dig = T.dig;
            p.mant_dig = T.mant_dig;
            p.max_exp = T.max_exp;
            p.min_exp = T.min_exp;
            p.max_10_exp = T.max_10_exp;
            p.min_10_exp = T.min_10_exp;

            return p;
        }
    }

    static __gshared FPTypeProperties FloatProperties = FPTypeProperties.fromDHostCompiler!float();
    static __gshared FPTypeProperties DoubleProperties = FPTypeProperties.fromDHostCompiler!double();
    static __gshared FPTypeProperties RealProperties = FPTypeProperties.fromDHostCompiler!real_t();

    // implemented in gen/target.cpp:
    static void _init();
    // Type sizes and support.
    static uint alignsize(Type type);
    static uint fieldalign(Type type);
    static uint critsecsize();
    static Type va_listType();  // get type of va_list
    static int isVectorTypeSupported(int sz, Type type);
    static bool isVectorOpSupported(Type type, TOK op, Type t2 = null);
    // CTFE support for cross-compilation.
    static Expression paintAsType(Expression e, Type type);
    // ABI and backend.
    static void loadModule(Module m);

    static const(char)* toCppMangle(Dsymbol s)
    {
        if (isTargetWindowsMSVC())
            return toCppMangleMSVC(s);
        else
            return toCppMangleItanium(s);
    }

    static const(char)* cppTypeInfoMangle(ClassDeclaration cd)
    {
        if (isTargetWindowsMSVC())
            return cppTypeInfoMangleMSVC(cd);
        else
            return cppTypeInfoMangleItanium(cd);
    }
  }
  else // !IN_LLVM
  {
    /**
     * Values representing all properties for floating point types
     */
    extern (C++) struct FPTypeProperties(T)
    {
        static __gshared
        {
            real_t max = T.max;                 /// largest representable value that's not infinity
            real_t min_normal = T.min_normal;   /// smallest representable normalized value that's not 0
            real_t nan = T.nan;                 /// NaN value
            real_t snan = T.init;               /// signalling NaN value
            real_t infinity = T.infinity;       /// infinity value
            real_t epsilon = T.epsilon;         /// smallest increment to the value 1

            d_int64 dig = T.dig;                /// number of decimal digits of precision
            d_int64 mant_dig = T.mant_dig;      /// number of bits in mantissa
            d_int64 max_exp = T.max_exp;        /// maximum int value such that 2$(SUPERSCRIPT `max_exp-1`) is representable
            d_int64 min_exp = T.min_exp;        /// minimum int value such that 2$(SUPERSCRIPT `min_exp-1`) is representable as a normalized value
            d_int64 max_10_exp = T.max_10_exp;  /// maximum int value such that 10$(SUPERSCRIPT `max_10_exp` is representable)
            d_int64 min_10_exp = T.min_10_exp;  /// minimum int value such that 10$(SUPERSCRIPT `min_10_exp`) is representable as a normalized value
        }
    }

    ///
    alias FloatProperties = FPTypeProperties!float;
    ///
    alias DoubleProperties = FPTypeProperties!double;
    ///
    alias RealProperties = FPTypeProperties!real_t;

    /**
     * Initialize the Target
     */
    extern (C++) static void _init()
    {
        // These have default values for 32 bit code, they get
        // adjusted for 64 bit code.
        ptrsize = 4;
        classinfosize = 0x4C; // 76

        /* gcc uses int.max for 32 bit compilations, and long.max for 64 bit ones.
         * Set to int.max for both, because the rest of the compiler cannot handle
         * 2^64-1 without some pervasive rework. The trouble is that much of the
         * front and back end uses 32 bit ints for sizes and offsets. Since C++
         * silently truncates 64 bit ints to 32, finding all these dependencies will be a problem.
         */
        maxStaticDataSize = int.max;

        if (global.params.isLP64)
        {
            ptrsize = 8;
            classinfosize = 0x98; // 152
        }
        if (global.params.isLinux || global.params.isFreeBSD || global.params.isOpenBSD || global.params.isDragonFlyBSD || global.params.isSolaris)
        {
            realsize = 12;
            realpad = 2;
            realalignsize = 4;
            c_longsize = 4;
        }
        else if (global.params.isOSX)
        {
            realsize = 16;
            realpad = 6;
            realalignsize = 16;
            c_longsize = 4;
        }
        else if (global.params.isWindows)
        {
            realsize = 10;
            realpad = 0;
            realalignsize = 2;
            reverseCppOverloads = true;
            c_longsize = 4;
            if (ptrsize == 4)
            {
                /* Optlink cannot deal with individual data chunks
                 * larger than 16Mb
                 */
                maxStaticDataSize = 0x100_0000;  // 16Mb
            }
        }
        else
            assert(0);
        if (global.params.is64bit)
        {
            if (global.params.isLinux || global.params.isFreeBSD || global.params.isDragonFlyBSD || global.params.isSolaris)
            {
                realsize = 16;
                realpad = 6;
                realalignsize = 16;
                c_longsize = 8;
            }
            else if (global.params.isOSX)
            {
                c_longsize = 8;
            }
        }
        c_long_doublesize = realsize;
        if (global.params.is64bit && global.params.isWindows)
            c_long_doublesize = 8;

        cppExceptions = global.params.isLinux || global.params.isFreeBSD ||
            global.params.isDragonFlyBSD || global.params.isOSX;

        int64Mangle  = global.params.isOSX ? 'x' : 'l';
        uint64Mangle = global.params.isOSX ? 'y' : 'm';
    }

    /**
     * Requested target memory alignment size of the given type.
     * Params:
     *      type = type to inspect
     * Returns:
     *      alignment in bytes
     */
    extern (C++) static uint alignsize(Type type)
    {
        assert(type.isTypeBasic());
        switch (type.ty)
        {
        case Tfloat80:
        case Timaginary80:
        case Tcomplex80:
            return Target.realalignsize;
        case Tcomplex32:
            if (global.params.isLinux || global.params.isOSX || global.params.isFreeBSD || global.params.isOpenBSD ||
                global.params.isDragonFlyBSD || global.params.isSolaris)
                return 4;
            break;
        case Tint64:
        case Tuns64:
        case Tfloat64:
        case Timaginary64:
        case Tcomplex64:
            if (global.params.isLinux || global.params.isOSX || global.params.isFreeBSD || global.params.isOpenBSD ||
                global.params.isDragonFlyBSD || global.params.isSolaris)
                return global.params.is64bit ? 8 : 4;
            break;
        default:
            break;
        }
        return cast(uint)type.size(Loc.initial);
    }

    /**
     * Requested target field alignment size of the given type.
     * Params:
     *      type = type to inspect
     * Returns:
     *      alignment in bytes
     */
    extern (C++) static uint fieldalign(Type type)
    {
        const size = type.alignsize();

        if ((global.params.is64bit || global.params.isOSX) && (size == 16 || size == 32))
            return size;

        return (8 < size) ? 8 : size;
    }

    /**
     * Size of the target OS critical section.
     * Returns:
     *      size in bytes
     */
    extern (C++) static uint critsecsize()
    {
        if (global.params.isWindows)
        {
            // sizeof(CRITICAL_SECTION) for Windows.
            return global.params.isLP64 ? 40 : 24;
        }
        else if (global.params.isLinux)
        {
            // sizeof(pthread_mutex_t) for Linux.
            if (global.params.is64bit)
                return global.params.isLP64 ? 40 : 32;
            else
                return global.params.isLP64 ? 40 : 24;
        }
        else if (global.params.isFreeBSD)
        {
            // sizeof(pthread_mutex_t) for FreeBSD.
            return global.params.isLP64 ? 8 : 4;
        }
        else if (global.params.isOpenBSD)
        {
            // sizeof(pthread_mutex_t) for OpenBSD.
            return global.params.isLP64 ? 8 : 4;
        }
        else if (global.params.isDragonFlyBSD)
        {
            // sizeof(pthread_mutex_t) for DragonFlyBSD.
            return global.params.isLP64 ? 8 : 4;
        }
        else if (global.params.isOSX)
        {
            // sizeof(pthread_mutex_t) for OSX.
            return global.params.isLP64 ? 64 : 44;
        }
        else if (global.params.isSolaris)
        {
            // sizeof(pthread_mutex_t) for Solaris.
            return 24;
        }
        assert(0);
    }

    /**
     * Type for the `va_list` type for the target.
     * NOTE: For Posix/x86_64 this returns the type which will really
     * be used for passing an argument of type va_list.
     * Returns:
     *      `Type` that represents `va_list`.
     */
    extern (C++) static Type va_listType()
    {
        if (global.params.isWindows)
        {
            return Type.tchar.pointerTo();
        }
        else if (global.params.isLinux || global.params.isFreeBSD || global.params.isOpenBSD || global.params.isDragonFlyBSD ||
            global.params.isSolaris || global.params.isOSX)
        {
            if (global.params.is64bit)
            {
                return (new TypeIdentifier(Loc.initial, Identifier.idPool("__va_list_tag"))).pointerTo();
            }
            else
            {
                return Type.tchar.pointerTo();
            }
        }
        else
        {
            assert(0);
        }
    }

    /**
     * Checks whether the target supports a vector type.
     * Params:
     *      sz   = vector type size in bytes
     *      type = vector element type
     * Returns:
     *      0   vector type is supported,
     *      1   vector type is not supported on the target at all
     *      2   vector element type is not supported
     *      3   vector size is not supported
     */
    extern (C++) static int isVectorTypeSupported(int sz, Type type)
    {
        if (!global.params.is64bit && !global.params.isOSX)
            return 1; // not supported
        switch (type.ty)
        {
        case Tvoid:
        case Tint8:
        case Tuns8:
        case Tint16:
        case Tuns16:
        case Tint32:
        case Tuns32:
        case Tfloat32:
        case Tint64:
        case Tuns64:
        case Tfloat64:
            break;
        default:
            return 2; // wrong base type
        }
        if (sz != 16 && !(global.params.cpu >= CPU.avx && sz == 32))
            return 3; // wrong size
        return 0;
    }

    /**
     * Checks whether the target supports the given operation for vectors.
     * Params:
     *      type = target type of operation
     *      op   = the unary or binary op being done on the `type`
     *      t2   = type of second operand if `op` is a binary operation
     * Returns:
     *      true if the operation is supported or type is not a vector
     */
    extern (C++) static bool isVectorOpSupported(Type type, TOK op, Type t2 = null)
    {
        import dmd.tokens;

        if (type.ty != Tvector)
            return true; // not a vector op
        auto tvec = cast(TypeVector) type;

        bool supported;
        switch (op)
        {
        case TOK.negate, TOK.uadd:
            supported = tvec.isscalar();
            break;

        case TOK.lessThan, TOK.greaterThan, TOK.lessOrEqual, TOK.greaterOrEqual, TOK.equal, TOK.notEqual, TOK.identity, TOK.notIdentity:
            supported = false;
            break;

        case TOK.unord, TOK.lg, TOK.leg, TOK.ule, TOK.ul, TOK.uge, TOK.ug, TOK.ue:
            supported = false;
            break;

        case TOK.leftShift, TOK.leftShiftAssign, TOK.rightShift, TOK.rightShiftAssign, TOK.unsignedRightShift, TOK.unsignedRightShiftAssign:
            supported = false;
            break;

        case TOK.add, TOK.addAssign, TOK.min, TOK.minAssign:
            supported = tvec.isscalar();
            break;

        case TOK.mul, TOK.mulAssign:
            // only floats and short[8]/ushort[8] (PMULLW)
            if (tvec.isfloating() || tvec.elementType().size(Loc.initial) == 2 ||
                // int[4]/uint[4] with SSE4.1 (PMULLD)
                global.params.cpu >= CPU.sse4_1 && tvec.elementType().size(Loc.initial) == 4)
                supported = true;
            else
                supported = false;
            break;

        case TOK.div, TOK.divAssign:
            supported = tvec.isfloating();
            break;

        case TOK.mod, TOK.modAssign:
            supported = false;
            break;

        case TOK.and, TOK.andAssign, TOK.or, TOK.orAssign, TOK.xor, TOK.xorAssign:
            supported = tvec.isintegral();
            break;

        case TOK.not:
            supported = false;
            break;

        case TOK.tilde:
            supported = tvec.isintegral();
            break;

        case TOK.pow, TOK.powAssign:
            supported = false;
            break;

        default:
            // import std.stdio : stderr, writeln;
            // stderr.writeln(op);
            assert(0, "unhandled op " ~ Token.toString(op));
        }
        return supported;
    }

    /**
     * Encode the given expression, which is assumed to be an rvalue literal
     * as another type for use in CTFE.
     * This corresponds roughly to the idiom `*cast(T*)&e`.
     * Params:
     *      e    = literal constant expression
     *      type = target type of the result
     * Returns:
     *      resulting `Expression` re-evaluated as `type`
     */
    extern (C++) static Expression paintAsType(Expression e, Type type)
    {
        // We support up to 512-bit values.
        ubyte[64] buffer;
        assert(e.type.size() == type.size());
        // Write the expression into the buffer.
        switch (e.type.ty)
        {
        case Tint32:
        case Tuns32:
        case Tint64:
        case Tuns64:
            encodeInteger(e, buffer.ptr);
            break;
        case Tfloat32:
        case Tfloat64:
            encodeReal(e, buffer.ptr);
            break;
        default:
            assert(0);
        }
        // Interpret the buffer as a new type.
        switch (type.ty)
        {
        case Tint32:
        case Tuns32:
        case Tint64:
        case Tuns64:
            return decodeInteger(e.loc, type, buffer.ptr);
        case Tfloat32:
        case Tfloat64:
            return decodeReal(e.loc, type, buffer.ptr);
        default:
            assert(0);
        }
    }

    /**
     * Perform any post parsing analysis on the given module.
     * Certain compiler backends (ie: GDC) have special placeholder
     * modules whose source are empty, but code gets injected
     * immediately after loading.
     * Params:
     *      m = module to inspect
     */
    extern (C++) static void loadModule(Module m)
    {
    }

    /**
     * Mangle the given symbol for C++ ABI.
     * Params:
     *      s = declaration with C++ linkage
     * Returns:
     *      string mangling of symbol
     */
    extern (C++) static const(char)* toCppMangle(Dsymbol s)
    {
        static if (TARGET.Linux || TARGET.OSX || TARGET.FreeBSD || TARGET.OpenBSD || TARGET.DragonFlyBSD || TARGET.Solaris)
            return toCppMangleItanium(s);
        else static if (TARGET.Windows)
            return toCppMangleMSVC(s);
        else
            static assert(0, "fix this");
    }

    /**
     * Get RTTI mangling of the given class declaration for C++ ABI.
     * Params:
     *      cd = class with C++ linkage
     * Returns:
     *      string mangling of C++ typeinfo
     */
    extern (C++) static const(char)* cppTypeInfoMangle(ClassDeclaration cd)
    {
        static if (TARGET.Linux || TARGET.OSX || TARGET.FreeBSD || TARGET.OpenBSD || TARGET.Solaris || TARGET.DragonFlyBSD)
            return cppTypeInfoMangleItanium(cd);
        else static if (TARGET.Windows)
            return cppTypeInfoMangleMSVC(cd);
        else
            static assert(0, "fix this");
    }
  } // !IN_LLVM

    /**
     * Gets vendor-specific type mangling for C++ ABI.
     * Params:
     *      t = type to inspect
     * Returns:
     *      string if type is mangled specially on target
     *      null if unhandled
     */
    extern (C++) static const(char)* cppTypeMangle(Type t)
    {
        return null;
    }

    /**
     * Get the type that will really be used for passing the given argument
     * to an `extern(C++)` function.
     * Params:
     *      p = parameter to be passed.
     * Returns:
     *      `Type` to use for parameter `p`.
     */
    extern (C++) static Type cppParameterType(Parameter p)
    {
        Type t = p.type.merge2();
        if (p.storageClass & (STC.out_ | STC.ref_))
            t = t.referenceTo();
        else if (p.storageClass & STC.lazy_)
        {
            // Mangle as delegate
            Type td = new TypeFunction(null, t, 0, LINK.d);
            td = new TypeDelegate(td);
            t = merge(t);
        }
        return t;
    }

    /**
     * Default system linkage for the target.
     * Returns:
     *      `LINK` to use for `extern(System)`
     */
    extern (C++) static LINK systemLinkage()
    {
        return global.params.isWindows ? LINK.windows : LINK.c;
    }

    /**
     * Describes how an argument type is passed to a function on target.
     * Params:
     *      t = type to break down
     * Returns:
     *      tuple of types if type is passed in one or more registers
     *      empty tuple if type is always passed on the stack
     */
    extern (C++) static TypeTuple toArgTypes(Type t)
    {
        return .toArgTypes(t);
    }
}

version(IN_LLVM) {} else {

/******************************
 * Private helpers for Target::paintAsType.
 */
// Write the integer value of 'e' into a unsigned byte buffer.
private void encodeInteger(Expression e, ubyte* buffer)
{
    dinteger_t value = e.toInteger();
    int size = cast(int)e.type.size();
    for (int p = 0; p < size; p++)
    {
        int offset = p; // Would be (size - 1) - p; on BigEndian
        buffer[offset] = ((value >> (p * 8)) & 0xFF);
    }
}

// Write the bytes encoded in 'buffer' into an integer and returns
// the value as a new IntegerExp.
private Expression decodeInteger(const ref Loc loc, Type type, ubyte* buffer)
{
    dinteger_t value = 0;
    int size = cast(int)type.size();
    for (int p = 0; p < size; p++)
    {
        int offset = p; // Would be (size - 1) - p; on BigEndian
        value |= (cast(dinteger_t)buffer[offset] << (p * 8));
    }
    return new IntegerExp(loc, value, type);
}

// Write the real_t value of 'e' into a unsigned byte buffer.
private void encodeReal(Expression e, ubyte* buffer)
{
    switch (e.type.ty)
    {
    case Tfloat32:
        {
            float* p = cast(float*)buffer;
            *p = cast(float)e.toReal();
            break;
        }
    case Tfloat64:
        {
            double* p = cast(double*)buffer;
            *p = cast(double)e.toReal();
            break;
        }
    default:
        assert(0);
    }
}

// Write the bytes encoded in 'buffer' into a real_t and returns
// the value as a new RealExp.
private Expression decodeReal(const ref Loc loc, Type type, ubyte* buffer)
{
    real_t value;
    switch (type.ty)
    {
    case Tfloat32:
        {
            float* p = cast(float*)buffer;
            value = real_t(*p);
            break;
        }
    case Tfloat64:
        {
            double* p = cast(double*)buffer;
            value = real_t(*p);
            break;
        }
    default:
        assert(0);
    }
    return new RealExp(loc, value, type);
}

} // !IN_LLVM
