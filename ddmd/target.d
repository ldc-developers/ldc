// Compiler implementation of the D programming language
// Copyright (c) 1999-2015 by Digital Mars
// All Rights Reserved
// written by Walter Bright
// http://www.digitalmars.com
// Distributed under the Boost Software License, Version 1.0.
// http://www.boost.org/LICENSE_1_0.txt

module ddmd.target;

import core.stdc.string;
import ddmd.dmodule;
import ddmd.expression;
import ddmd.globals;
import ddmd.identifier;
import ddmd.mtype;
import ddmd.root.ctfloat;
import ddmd.root.outbuffer;

version(IN_LLVM)
{

extern(C++) struct Target
{
    static __gshared int ptrsize;
    static __gshared int realsize;             // size a real consumes in memory
    static __gshared int realpad;              // 'padding' added to the CPU real size to bring it up to realsize
    static __gshared int realalignsize;        // alignment for reals
    static __gshared bool reverseCppOverloads; // with dmc and cl, overloaded functions are grouped and in reverse order
    static __gshared bool cppExceptions;       // set if catching C++ exceptions is supported
    static __gshared int c_longsize;           // size of a C 'long' or 'unsigned long' type
    static __gshared int c_long_doublesize;    // size of a C 'long double'
    static __gshared int classinfosize;        // size of 'ClassInfo'

    extern(D) template FPTypeProperties(T)
    {
        static real_t max() { return real_t(T.max); }
        static real_t min_normal() { return real_t(T.min_normal); }
        static real_t nan() { return real_t(T.nan); }
        static real_t snan() { return real_t(T.init); }
        static real_t infinity() { return real_t(T.infinity); }
        static real_t epsilon() { return real_t(T.epsilon); }

        enum : long
        {
            dig = T.dig,
            mant_dig = T.mant_dig,
            max_exp = T.max_exp,
            min_exp = T.min_exp,
            max_10_exp = T.max_10_exp,
            min_10_exp = T.min_10_exp
        }
    }

    alias FloatProperties = FPTypeProperties!float;
    alias DoubleProperties = FPTypeProperties!double;

    static struct RealProperties
    {
        static __gshared
        {
            real_t max = void;
            real_t min_normal = void;
            real_t nan = void;
            real_t snan = void;
            real_t infinity = void;
            real_t epsilon = void;

            long dig;
            long mant_dig;
            long max_exp;
            long min_exp;
            long max_10_exp;
            long min_10_exp;
        }
    }

    static void _init();
    // Type sizes and support.
    static uint alignsize(Type type);
    static uint fieldalign(Type type);
    static uint critsecsize();
    static Type va_listType();  // get type of va_list
    static int checkVectorType(int sz, Type type);
    // CTFE support for cross-compilation.
    static Expression paintAsType(Expression e, Type type);
    // ABI and backend.
    static void loadModule(Module m);
    static void prefixName(OutBuffer *buf, LINK linkage);
}

}
else
{

/***********************************************************
 */
struct Target
{
    extern (C++) static __gshared int ptrsize;
    extern (C++) static __gshared int realsize;             // size a real consumes in memory
    extern (C++) static __gshared int realpad;              // 'padding' added to the CPU real size to bring it up to realsize
    extern (C++) static __gshared int realalignsize;        // alignment for reals
    extern (C++) static __gshared bool reverseCppOverloads; // with dmc and cl, overloaded functions are grouped and in reverse order
    extern (C++) static __gshared bool cppExceptions;       // set if catching C++ exceptions is supported
    extern (C++) static __gshared int c_longsize;           // size of a C 'long' or 'unsigned long' type
    extern (C++) static __gshared int c_long_doublesize;    // size of a C 'long double'
    extern (C++) static __gshared int classinfosize;        // size of 'ClassInfo'

    template FPTypeProperties(T)
    {
        enum : real_t
        {
            max = T.max,
            min_normal = T.min_normal,
            nan = T.nan,
            snan = T.init,
            infinity = T.infinity,
            epsilon = T.epsilon
        }

        enum : long
        {
            dig = T.dig,
            mant_dig = T.mant_dig,
            max_exp = T.max_exp,
            min_exp = T.min_exp,
            max_10_exp = T.max_10_exp,
            min_10_exp = T.min_10_exp
        }
    }

    alias FloatProperties = FPTypeProperties!float;
    alias DoubleProperties = FPTypeProperties!double;
    alias RealProperties = FPTypeProperties!real;

    extern (C++) static void _init()
    {
        // These have default values for 32 bit code, they get
        // adjusted for 64 bit code.
        ptrsize = 4;
        classinfosize = 0x4C; // 76
        if (global.params.isLP64)
        {
            ptrsize = 8;
            classinfosize = 0x98; // 152
        }
        if (global.params.isLinux || global.params.isFreeBSD || global.params.isOpenBSD || global.params.isSolaris)
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
        }
        else
            assert(0);
        if (global.params.is64bit)
        {
            if (global.params.isLinux || global.params.isFreeBSD || global.params.isSolaris)
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

        cppExceptions = global.params.dwarfeh || global.params.isLinux || global.params.isFreeBSD ||
            (global.params.isOSX && global.params.is64bit);
    }

    /******************************
     * Return memory alignment size of type.
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
            if (global.params.isLinux || global.params.isOSX || global.params.isFreeBSD || global.params.isOpenBSD || global.params.isSolaris)
                return 4;
            break;
        case Tint64:
        case Tuns64:
        case Tfloat64:
        case Timaginary64:
        case Tcomplex64:
            if (global.params.isLinux || global.params.isOSX || global.params.isFreeBSD || global.params.isOpenBSD || global.params.isSolaris)
                return global.params.is64bit ? 8 : 4;
            break;
        default:
            break;
        }
        return cast(uint)type.size(Loc());
    }

    /******************************
     * Return field alignment size of type.
     */
    extern (C++) static uint fieldalign(Type type)
    {
        return type.alignsize();
    }

    /***********************************
     * Return size of OS critical section.
     * NOTE: can't use the sizeof() calls directly since cross compiling is
     * supported and would end up using the host sizes rather than the target
     * sizes.
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

    /***********************************
     * Returns a Type for the va_list type of the target.
     * NOTE: For Posix/x86_64 this returns the type which will really
     * be used for passing an argument of type va_list.
     */
    extern (C++) static Type va_listType()
    {
        if (global.params.isWindows)
        {
            return Type.tchar.pointerTo();
        }
        else if (global.params.isLinux || global.params.isFreeBSD || global.params.isOpenBSD || global.params.isSolaris || global.params.isOSX)
        {
            if (global.params.is64bit)
            {
                return (new TypeIdentifier(Loc(), Identifier.idPool("__va_list_tag"))).pointerTo();
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

    /*
     * Return true if the given type is supported for this target
     */
    extern (C++) static int checkVectorType(int sz, Type type)
    {
        if (!global.params.is64bit && !global.params.isOSX)
            return 1; // not supported
        if (sz != 16 && sz != 32)
            return 2; // wrong size
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
            return 3; // wrong base type
        }
        return 0;
    }

    /******************************
     * Encode the given expression, which is assumed to be an rvalue literal
     * as another type for use in CTFE.
     * This corresponds roughly to the idiom *(Type *)&e.
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

    /******************************
     * For the given module, perform any post parsing analysis.
     * Certain compiler backends (ie: GDC) have special placeholder
     * modules whose source are empty, but code gets injected
     * immediately after loading.
     */
    extern (C++) static void loadModule(Module m)
    {
    }

    /******************************
     * For the given symbol written to the OutBuffer, apply any
     * target-specific prefixes based on the given linkage.
     */
    extern (C++) static void prefixName(OutBuffer* buf, LINK linkage)
    {
        switch (linkage)
        {
        case LINKcpp:
            if (global.params.isOSX)
                buf.prependbyte('_');
            break;
        default:
            break;
        }
    }
}

/******************************
 * Private helpers for Target::paintAsType.
 */
// Write the integer value of 'e' into a unsigned byte buffer.
extern (C++) static void encodeInteger(Expression e, ubyte* buffer)
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
extern (C++) static Expression decodeInteger(Loc loc, Type type, ubyte* buffer)
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
extern (C++) static void encodeReal(Expression e, ubyte* buffer)
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
extern (C++) static Expression decodeReal(Loc loc, Type type, ubyte* buffer)
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