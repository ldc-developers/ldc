/**
 * Compiler implementation of the D programming language
 * http://dlang.org
 *
 * Copyright: Copyright (C) 1999-2019 by The D Language Foundation, All Rights Reserved
 * Authors:   Walter Bright, http://www.digitalmars.com
 * License:   $(LINK2 http://www.boost.org/LICENSE_1_0.txt, Boost License 1.0)
 * Source:    $(LINK2 https://github.com/dlang/dmd/blob/master/src/dmd/root/outbuffer.d, root/_outbuffer.d)
 * Documentation: https://dlang.org/phobos/dmd_root_outbuffer.html
 * Coverage:    https://codecov.io/gh/dlang/dmd/src/master/src/dmd/root/outbuffer.d
 */

module dmd.root.outbuffer;

import core.stdc.stdarg;
import core.stdc.stdio;
import core.stdc.string;
import dmd.root.rmem;
import dmd.root.rootobject;

struct OutBuffer
{
    ubyte* data;
    size_t offset;
    size_t size;
    int level;
    bool doindent;
    private bool notlinehead;

    extern (C++) ~this() pure nothrow
    {
        mem.xfree(data);
    }

    extern (C++) char* extractData() pure nothrow @nogc @safe
    {
        char* p;
        p = cast(char*)data;
        data = null;
        offset = 0;
        size = 0;
        return p;
    }

    extern (C++) void destroy() pure nothrow @trusted
    {
        mem.xfree(extractData());
    }

    extern (C++) void reserve(size_t nbytes) pure nothrow
    {
        //printf("OutBuffer::reserve: size = %d, offset = %d, nbytes = %d\n", size, offset, nbytes);
        if (size - offset < nbytes)
        {
            /* Increase by factor of 1.5; round up to 16 bytes.
             * The odd formulation is so it will map onto single x86 LEA instruction.
             */
            size = (((offset + nbytes) * 3 + 30) / 2) & ~15;
            data = cast(ubyte*)mem.xrealloc(data, size);
        }
    }

    extern (C++) void setsize(size_t size) pure nothrow @nogc @safe
    {
        offset = size;
    }

    extern (C++) void reset() pure nothrow @nogc @safe
    {
        offset = 0;
    }

    private void indent() pure nothrow
    {
        if (level)
        {
            reserve(level);
            data[offset .. offset + level] = '\t';
            offset += level;
        }
        notlinehead = true;
    }

    extern (C++) void write(const(void)* data, size_t nbytes) pure nothrow
    {
        if (doindent && !notlinehead)
            indent();
        reserve(nbytes);
        memcpy(this.data + offset, data, nbytes);
        offset += nbytes;
    }

    extern (C++) void writestring(const(char)* string) pure nothrow
    {
        write(string, strlen(string));
    }

    void writestring(const(char)[] s) pure nothrow
    {
        write(s.ptr, s.length);
    }

    void writestring(string s) pure nothrow
    {
        write(s.ptr, s.length);
    }

    extern (C++) void prependstring(const(char)* string) pure nothrow
    {
        size_t len = strlen(string);
        reserve(len);
        memmove(data + len, data, offset);
        memcpy(data, string, len);
        offset += len;
    }

    // write newline
    extern (C++) void writenl() pure nothrow
    {
        version (Windows)
        {
            writeword(0x0A0D); // newline is CR,LF on Microsoft OS's
        }
        else
        {
            writeByte('\n');
        }
        if (doindent)
            notlinehead = false;
    }

    extern (C++) void writeByte(uint b) pure nothrow
    {
        if (doindent && !notlinehead && b != '\n')
            indent();
        reserve(1);
        this.data[offset] = cast(ubyte)b;
        offset++;
    }

    extern (C++) void writeUTF8(uint b) pure nothrow
    {
        reserve(6);
        if (b <= 0x7F)
        {
            this.data[offset] = cast(ubyte)b;
            offset++;
        }
        else if (b <= 0x7FF)
        {
            this.data[offset + 0] = cast(ubyte)((b >> 6) | 0xC0);
            this.data[offset + 1] = cast(ubyte)((b & 0x3F) | 0x80);
            offset += 2;
        }
        else if (b <= 0xFFFF)
        {
            this.data[offset + 0] = cast(ubyte)((b >> 12) | 0xE0);
            this.data[offset + 1] = cast(ubyte)(((b >> 6) & 0x3F) | 0x80);
            this.data[offset + 2] = cast(ubyte)((b & 0x3F) | 0x80);
            offset += 3;
        }
        else if (b <= 0x1FFFFF)
        {
            this.data[offset + 0] = cast(ubyte)((b >> 18) | 0xF0);
            this.data[offset + 1] = cast(ubyte)(((b >> 12) & 0x3F) | 0x80);
            this.data[offset + 2] = cast(ubyte)(((b >> 6) & 0x3F) | 0x80);
            this.data[offset + 3] = cast(ubyte)((b & 0x3F) | 0x80);
            offset += 4;
        }
        else
            assert(0);
    }

    extern (C++) void prependbyte(uint b) pure nothrow
    {
        reserve(1);
        memmove(data + 1, data, offset);
        data[0] = cast(ubyte)b;
        offset++;
    }

    extern (C++) void writewchar(uint w) pure nothrow
    {
        version (Windows)
        {
            writeword(w);
        }
        else
        {
            write4(w);
        }
    }

    extern (C++) void writeword(uint w) pure nothrow
    {
        version (Windows)
        {
            uint newline = 0x0A0D;
        }
        else
        {
            uint newline = '\n';
        }
        if (doindent && !notlinehead && w != newline)
            indent();

        reserve(2);
        *cast(ushort*)(this.data + offset) = cast(ushort)w;
        offset += 2;
    }

    extern (C++) void writeUTF16(uint w) pure nothrow
    {
        reserve(4);
        if (w <= 0xFFFF)
        {
            *cast(ushort*)(this.data + offset) = cast(ushort)w;
            offset += 2;
        }
        else if (w <= 0x10FFFF)
        {
            *cast(ushort*)(this.data + offset) = cast(ushort)((w >> 10) + 0xD7C0);
            *cast(ushort*)(this.data + offset + 2) = cast(ushort)((w & 0x3FF) | 0xDC00);
            offset += 4;
        }
        else
            assert(0);
    }

    extern (C++) void write4(uint w) pure nothrow
    {
        version (Windows)
        {
            bool notnewline = w != 0x000A000D;
        }
        else
        {
            bool notnewline = true;
        }
        if (doindent && !notlinehead && notnewline)
            indent();
        reserve(4);
        *cast(uint*)(this.data + offset) = w;
        offset += 4;
    }

    extern (C++) void write(const OutBuffer* buf) pure nothrow
    {
        if (buf)
        {
            reserve(buf.offset);
            memcpy(data + offset, buf.data, buf.offset);
            offset += buf.offset;
        }
    }

    extern (C++) void write(RootObject obj) /*nothrow*/
    {
        if (obj)
        {
            writestring(obj.toChars());
        }
    }

    extern (C++) void fill0(size_t nbytes) pure nothrow
    {
        reserve(nbytes);
        memset(data + offset, 0, nbytes);
        offset += nbytes;
    }

    extern (C++) void vprintf(const(char)* format, va_list args) nothrow
    {
        int count;
        if (doindent)
            write(null, 0); // perform indent
        uint psize = 128;
        for (;;)
        {
            reserve(psize);
            version (Windows)
            {
                count = _vsnprintf(cast(char*)data + offset, psize, format, args);
                if (count != -1)
                    break;
                psize *= 2;
            }
            else version (Posix)
            {
                va_list va;
                va_copy(va, args);
                /*
                 The functions vprintf(), vfprintf(), vsprintf(), vsnprintf()
                 are equivalent to the functions printf(), fprintf(), sprintf(),
                 snprintf(), respectively, except that they are called with a
                 va_list instead of a variable number of arguments. These
                 functions do not call the va_end macro. Consequently, the value
                 of ap is undefined after the call. The application should call
                 va_end(ap) itself afterwards.
                 */
                count = vsnprintf(cast(char*)data + offset, psize, format, va);
                va_end(va);
                if (count == -1)
                    psize *= 2;
                else if (count >= psize)
                    psize = count + 1;
                else
                    break;
            }
            else
            {
                assert(0);
            }
        }
        offset += count;
    }

    extern (C++) void printf(const(char)* format, ...) nothrow
    {
        va_list ap;
        va_start(ap, format);
        vprintf(format, ap);
        va_end(ap);
    }

    /**************************************
     * Convert `u` to a string and append it to the buffer.
     * Params:
     *  u = integral value to append
     */
    extern (C++) void print(ulong u) pure nothrow
    {
        //import core.internal.string;  // not available
        UnsignedStringBuf buf = void;
        writestring(unsignedToTempString(u, buf));
    }

    extern (C++) void bracket(char left, char right) pure nothrow
    {
        reserve(2);
        memmove(data + 1, data, offset);
        data[0] = left;
        data[offset + 1] = right;
        offset += 2;
    }

    /******************
     * Insert left at i, and right at j.
     * Return index just past right.
     */
    extern (C++) size_t bracket(size_t i, const(char)* left, size_t j, const(char)* right) pure nothrow
    {
        size_t leftlen = strlen(left);
        size_t rightlen = strlen(right);
        reserve(leftlen + rightlen);
        insert(i, left, leftlen);
        insert(j + leftlen, right, rightlen);
        return j + leftlen + rightlen;
    }

    extern (C++) void spread(size_t offset, size_t nbytes) pure nothrow
    {
        reserve(nbytes);
        memmove(data + offset + nbytes, data + offset, this.offset - offset);
        this.offset += nbytes;
    }

    /****************************************
     * Returns: offset + nbytes
     */
    extern (C++) size_t insert(size_t offset, const(void)* p, size_t nbytes) pure nothrow
    {
        spread(offset, nbytes);
        memmove(data + offset, p, nbytes);
        return offset + nbytes;
    }

    size_t insert(size_t offset, const(char)[] s) pure nothrow
    {
        return insert(offset, s.ptr, s.length);
    }

    extern (C++) void remove(size_t offset, size_t nbytes) pure nothrow @nogc
    {
        memmove(data + offset, data + offset + nbytes, this.offset - (offset + nbytes));
        this.offset -= nbytes;
    }

    extern (D) const(char)[] peekSlice() pure nothrow @nogc
    {
        return this[];
    }

    extern (D) const(char)[] opSlice() pure nothrow @nogc
    {
        return (cast(const char*)data)[0 .. offset];
    }

    /***********************************
     * Extract the data as a slice and take ownership of it.
     */
    extern (D) char[] extractSlice() pure nothrow @nogc
    {
        auto length = offset;
        auto p = extractData();
        return p[0 .. length];
    }

    // Append terminating null if necessary and get view of internal buffer
    extern (C++) char* peekChars() pure nothrow
    {
        if (!offset || data[offset - 1] != '\0')
        {
            writeByte(0);
            offset--; // allow appending more
        }
        return cast(char*)data;
    }

    // Append terminating null if necessary and take ownership of data
    extern (C++) char* extractChars() pure nothrow
    {
        if (!offset || data[offset - 1] != '\0')
            writeByte(0);
        return extractData();
    }
}

/****** copied from core.internal.string *************/

private:

alias UnsignedStringBuf = char[20];

char[] unsignedToTempString(ulong value, char[] buf, uint radix = 10) @safe pure nothrow @nogc
{
    size_t i = buf.length;
    do
    {
        if (value < radix)
        {
            ubyte x = cast(ubyte)value;
            buf[--i] = cast(char)((x < 10) ? x + '0' : x - 10 + 'a');
            break;
        }
        else
        {
            ubyte x = cast(ubyte)(value % radix);
            value = value / radix;
            buf[--i] = cast(char)((x < 10) ? x + '0' : x - 10 + 'a');
        }
    } while (value);
    return buf[i .. $];
}

/************* unit tests **************************************************/

unittest
{
    OutBuffer buf;
    buf.printf("betty");
    buf.insert(1, "xx".ptr, 2);
    buf.insert(3, "yy");
    buf.remove(4, 1);
    buf.bracket('(', ')');
    const char[] s = buf.peekSlice();
    assert(s == "(bxxyetty)");
    buf.destroy();
}

unittest
{
    OutBuffer buf;
    buf.writestring("abc".ptr);
    buf.prependstring("def");
    buf.prependbyte('x');
    OutBuffer buf2;
    buf2.writestring("mmm");
    buf.write(&buf2);
    char[] s = buf.extractSlice();
    assert(s == "xdefabcmmm");
}

unittest
{
    OutBuffer buf;
    buf.writeByte('a');
    char[] s = buf.extractSlice();
    assert(s == "a");

    buf.writeByte('b');
    char[] t = buf.extractSlice();
    assert(t == "b");
}

unittest
{
    OutBuffer buf;
    char* p = buf.peekChars();
    assert(*p == 0);

    buf.writeByte('s');
    char* q = buf.peekChars();
    assert(strcmp(q, "s") == 0);
}

unittest
{
    char[10] buf;
    char[] s = unsignedToTempString(278, buf[], 10);
    assert(s == "278");

    s = unsignedToTempString(1, buf[], 10);
    assert(s == "1");

    s = unsignedToTempString(8, buf[], 2);
    assert(s == "1000");

    s = unsignedToTempString(29, buf[], 16);
    assert(s == "1d");
}
