// Compiler implementation of the D programming language
// Copyright (c) 1999-2015 by Digital Mars
// All Rights Reserved
// written by Walter Bright
// http://www.digitalmars.com
// Distributed under the Boost Software License, Version 1.0.
// http://www.boost.org/LICENSE_1_0.txt

module ddmd.root.port;

import core.stdc.ctype;
import core.stdc.string;

version(CRuntime_DigitalMars) __gshared extern (C) extern const(char)* __locale_decpoint;

extern (C++) struct Port
{
    static int memicmp(const char* s1, const char* s2, size_t n)
    {
        int result = 0;

        for (int i = 0; i < n; i++)
        {
            char c1 = s1[i];
            char c2 = s2[i];

            result = c1 - c2;
            if (result)
            {
                result = toupper(c1) - toupper(c2);
                if (result)
                    break;
            }
        }
        return result;
    }

    static char* strupr(char* s)
    {
        char* t = s;

        while (*s)
        {
            *s = cast(char)toupper(*s);
            s++;
        }

        return t;
    }

    // Little endian
    static void writelongLE(uint value, void* buffer)
    {
        auto p = cast(ubyte*)buffer;
        p[3] = cast(ubyte)(value >> 24);
        p[2] = cast(ubyte)(value >> 16);
        p[1] = cast(ubyte)(value >> 8);
        p[0] = cast(ubyte)(value);
    }

    // Little endian
    static uint readlongLE(void* buffer)
    {
        auto p = cast(ubyte*)buffer;
        return (((((p[3] << 8) | p[2]) << 8) | p[1]) << 8) | p[0];
    }

    // Big endian
    static void writelongBE(uint value, void* buffer)
    {
        auto p = cast(ubyte*)buffer;
        p[0] = cast(ubyte)(value >> 24);
        p[1] = cast(ubyte)(value >> 16);
        p[2] = cast(ubyte)(value >> 8);
        p[3] = cast(ubyte)(value);
    }

    // Big endian
    static uint readlongBE(void* buffer)
    {
        auto p = cast(ubyte*)buffer;
        return (((((p[0] << 8) | p[1]) << 8) | p[2]) << 8) | p[3];
    }

    // Little endian
    static uint readwordLE(void* buffer)
    {
        auto p = cast(ubyte*)buffer;
        return (p[1] << 8) | p[0];
    }

    // Big endian
    static uint readwordBE(void* buffer)
    {
        auto p = cast(ubyte*)buffer;
        return (p[0] << 8) | p[1];
    }

    version(IN_LLVM)
    {   // LDC_FIXME: look at the old port.c how to implement this with system calls for some OSses
        static int stricmp(const(char)* s1, const(char)* s2)
        {
            int result = 0;

            for (;;)
            {   char c1 = *s1;
                char c2 = *s2;

                result = c1 - c2;
                if (result)
                {
                    result = toupper(c1) - toupper(c2);
                    if (result)
                        break;
                }
                if (!c1)
                    break;
                s1++;
                s2++;
            }
            return result;
        }
    }
}
