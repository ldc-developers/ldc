/**
 * Compiler implementation of the
 * $(LINK2 http://www.dlang.org, D programming language).
 *
 * Copyright:   Copyright (c) 1999-2017 by Digital Mars, All Rights Reserved
 * Authors:     $(LINK2 http://www.digitalmars.com, Walter Bright)
 * License:     $(LINK2 http://www.boost.org/LICENSE_1_0.txt, Boost License 1.0)
 * Source:      $(DMDSRC _target.d)
 */

module ddmd.target;

import ddmd.dmodule;
import ddmd.expression;
import ddmd.globals;
import ddmd.identifier;
import ddmd.mtype;
import ddmd.root.ctfloat;
import ddmd.root.outbuffer;

/***********************************************************
 */
struct Target
{
    extern (C++) static __gshared int ptrsize;
    extern (C++) static __gshared int realsize;             // size a real consumes in memory
    extern (C++) static __gshared int realpad;              // 'padding' added to the CPU real size to bring it up to realsize
    extern (C++) static __gshared int realalignsize;        // alignment for reals
    extern (C++) static __gshared bool realislongdouble;    // distinguish between C 'long double' and '__float128'
    extern (C++) static __gshared bool reverseCppOverloads; // with dmc and cl, overloaded functions are grouped and in reverse order
    extern (C++) static __gshared bool cppExceptions;       // set if catching C++ exceptions is supported
    extern (C++) static __gshared int c_longsize;           // size of a C 'long' or 'unsigned long' type
    extern (C++) static __gshared int c_long_doublesize;    // size of a C 'long double'
    extern (C++) static __gshared int classinfosize;        // size of 'ClassInfo'
    extern (C++) static __gshared ulong maxStaticDataSize;  // maximum size of static data

    extern (C++) struct FPTypeProperties(T)
    {
        static __gshared
        {
            real_t max = T.max;
            real_t min_normal = T.min_normal;
            real_t nan = T.nan;
            real_t snan = T.init;
            real_t infinity = T.infinity;
            real_t epsilon = T.epsilon;

            d_int64 dig = T.dig;
            d_int64 mant_dig = T.mant_dig;
            d_int64 max_exp = T.max_exp;
            d_int64 min_exp = T.min_exp;
            d_int64 max_10_exp = T.max_10_exp;
            d_int64 min_10_exp = T.min_10_exp;
        }
    }

    alias FloatProperties = FPTypeProperties!float;
    alias DoubleProperties = FPTypeProperties!double;
  version(IN_LLVM) {
    // host real_t may be double => make sure not to alias target's DoubleProperties
    extern (C++) static __gshared FPTypeProperties!real_t RealProperties;
  } else {
    alias RealProperties = FPTypeProperties!real_t;
  }

  version(IN_LLVM)
  {
    // implemented in gen/target.cpp:
    extern (C++):

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
  else // !IN_LLVM
  {
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
        realislongdouble = true;
        c_long_doublesize = realsize;
        if (global.params.is64bit && global.params.isWindows)
            c_long_doublesize = 8;

        cppExceptions = global.params.isLinux || global.params.isFreeBSD ||
            global.params.isOSX;
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
        const size = type.alignsize();

        if ((global.params.is64bit || global.params.isOSX) && (size == 16 || size == 32))
            return size;

        return (8 < size) ? 8 : size;
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

    /**
     * Checks whether the target supports a vector type with total size `sz`
     * (in bytes) and element type `type`.
     *
     * Returns: 0 if the type is supported, or else: 1 if vector types are not
     *     supported on the target at all, 2 if the given size isn't, or 3 if
     *     the element type isn't.
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
  } // !IN_LLVM
}

version(IN_LLVM) {} else {

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
