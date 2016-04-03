// Compiler implementation of the D programming language
// Copyright (c) 1999-2015 by Digital Mars
// All Rights Reserved
// written by Walter Bright
// http://www.digitalmars.com
// Distributed under the Boost Software License, Version 1.0.
// http://www.boost.org/LICENSE_1_0.txt

module ddmd.root.rmem;

import core.stdc.string;

version (GC)
{
    import core.memory : GC;

    extern (C++) struct Mem
    {
        static char* xstrdup(const(char)* p) nothrow
        {
            return p[0 .. strlen(p) + 1].dup.ptr;
        }

        static void xfree(void* p) nothrow
        {
        }

        static void* xmalloc(size_t n) nothrow
        {
            return GC.malloc(n);
        }

        static void* xcalloc(size_t size, size_t n) nothrow
        {
            return GC.calloc(size * n);
        }

        static void* xrealloc(void* p, size_t size) nothrow
        {
            return GC.realloc(p, size);
        }
    }

    extern (C++) __gshared Mem mem;
}
else
{
    import core.stdc.stdlib;
    import core.stdc.stdio;

    extern (C++) struct Mem
    {
        static char* xstrdup(const(char)* s) nothrow
        {
            if (s)
            {
                auto p = .strdup(s);
                if (p)
                    return p;
                error();
            }
            return null;
        }

        static void xfree(void* p) nothrow
        {
            if (p)
                .free(p);
        }

        static void* xmalloc(size_t size) nothrow
        {
            if (!size)
                return null;

            auto p = .malloc(size);
            if (!p)
                error();
            return p;
        }

        static void* xcalloc(size_t size, size_t n) nothrow
        {
            if (!size || !n)
                return null;

            auto p = .calloc(size, n);
            if (!p)
                error();
            return p;
        }

        static void* xrealloc(void* p, size_t size) nothrow
        {
            if (!size)
            {
                if (p)
                    .free(p);
                return null;
            }

            if (!p)
            {
                p = .malloc(size);
                if (!p)
                    error();
                return p;
            }

            p = .realloc(p, size);
            if (!p)
                error();
            return p;
        }

        static void error() nothrow
        {
            printf("Error: out of memory\n");
            exit(EXIT_FAILURE);
        }
    }

    extern (C++) __gshared Mem mem;

    enum CHUNK_SIZE = (256 * 4096 - 64);

    __gshared size_t heapleft = 0;
    __gshared void* heapp;

    extern (C) void* allocmemory(size_t m_size) nothrow
    {
        // 16 byte alignment is better (and sometimes needed) for doubles
        m_size = (m_size + 15) & ~15;

        // The layout of the code is selected so the most common case is straight through
        if (m_size <= heapleft)
        {
        L1:
            heapleft -= m_size;
            auto p = heapp;
            heapp = cast(void*)(cast(char*)heapp + m_size);
            return p;
        }

        if (m_size > CHUNK_SIZE)
        {
            auto p = malloc(m_size);
            if (p)
            {
                return p;
            }
            printf("Error: out of memory\n");
            exit(EXIT_FAILURE);
        }

        heapleft = CHUNK_SIZE;
        heapp = malloc(CHUNK_SIZE);
        if (!heapp)
        {
            printf("Error: out of memory\n");
            exit(EXIT_FAILURE);
        }
        goto L1;
    }

    version(DigitalMars)
    {
        extern (C) void* _d_allocmemory(size_t m_size) nothrow
        {
            return allocmemory(m_size);
        }

        extern (C) Object _d_newclass(const ClassInfo ci) nothrow
        {
            auto p = allocmemory(ci.init.length);
            p[0 .. ci.init.length] = cast(void[])ci.init[];
            return cast(Object)p;
        }

        extern (C) void* _d_newitemT(TypeInfo ti) nothrow
        {
            auto p = allocmemory(ti.tsize);
            (cast(ubyte*)p)[0 .. ti.init.length] = 0;
            return p;
        }

        extern (C) void* _d_newitemiT(TypeInfo ti) nothrow
        {
            auto p = allocmemory(ti.tsize);
            p[0 .. ti.init.length] = ti.init[];
            return p;
        }
    }
}
