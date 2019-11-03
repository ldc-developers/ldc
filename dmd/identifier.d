/**
 * Compiler implementation of the
 * $(LINK2 http://www.dlang.org, D programming language).
 *
 * Copyright:   Copyright (C) 1999-2019 by The D Language Foundation, All Rights Reserved
 * Authors:     $(LINK2 http://www.digitalmars.com, Walter Bright)
 * License:     $(LINK2 http://www.boost.org/LICENSE_1_0.txt, Boost License 1.0)
 * Source:      $(LINK2 https://github.com/dlang/dmd/blob/master/src/dmd/identifier.d, _identifier.d)
 * Documentation:  https://dlang.org/phobos/dmd_identifier.html
 * Coverage:    https://codecov.io/gh/dlang/dmd/src/master/src/dmd/identifier.d
 */

module dmd.identifier;

import core.stdc.ctype;
import core.stdc.stdio;
import core.stdc.string;
import dmd.globals;
import dmd.id;
import dmd.root.outbuffer;
import dmd.root.rootobject;
import dmd.root.stringtable;
import dmd.tokens;
import dmd.utf;
import dmd.utils;


/***********************************************************
 */
extern (C++) final class Identifier : RootObject
{
private:
    const int value;
    const char[] name;

public:
    /**
       Construct an identifier from a D slice

       Note: Since `name` needs to be `\0` terminated for `toChars`,
       no slice overload is provided yet.

       Params:
         name = the identifier name
                There must be `'\0'` at `name[length]`.
         length = the length of `name`, excluding the terminating `'\0'`
         value = Identifier value (e.g. `Id.unitTest`) or `TOK.identifier`
     */
    extern (D) this(const(char)* name, size_t length, int value) nothrow
    {
        //printf("Identifier('%s', %d)\n", name, value);
        this.name = name[0 .. length];
        this.value = value;
    }

    extern (D) this(const(char)[] name, int value) nothrow
    {
        //printf("Identifier('%.*s', %d)\n", cast(int)name.length, name.ptr, value);
        this.name = name;
        this.value = value;
    }

    extern (D) this(const(char)* name) nothrow
    {
        //printf("Identifier('%s', %d)\n", name, value);
        this(name[0 .. strlen(name)], TOK.identifier);
    }

    /// Sentinel for an anonymous identifier.
    static Identifier anonymous() nothrow
    {
        __gshared Identifier anonymous;

        if (anonymous)
            return anonymous;

        return anonymous = new Identifier("__anonymous", TOK.identifier);
    }

    static Identifier create(const(char)* name) nothrow
    {
        return new Identifier(name);
    }

nothrow:
    override const(char)* toChars() const pure
    {
        return name.ptr;
    }

    extern (D) override const(char)[] toString() const pure
    {
        return name;
    }

    int getValue() const pure
    {
        return value;
    }

    const(char)* toHChars2() const
    {
        const(char)* p = null;
        if (this == Id.ctor)
            p = "this";
        else if (this == Id.dtor)
            p = "~this";
        else if (this == Id.unitTest)
            p = "unittest";
        else if (this == Id.dollar)
            p = "$";
        else if (this == Id.withSym)
            p = "with";
        else if (this == Id.result)
            p = "result";
        else if (this == Id.returnLabel)
            p = "return";
        else
        {
            p = toChars();
            if (*p == '_')
            {
                if (strncmp(p, "_staticCtor", 11) == 0)
                    p = "static this";
                else if (strncmp(p, "_staticDtor", 11) == 0)
                    p = "static ~this";
                else if (strncmp(p, "__invariant", 11) == 0)
                    p = "invariant";
            }
        }
        return p;
    }

    override DYNCAST dyncast() const
    {
        return DYNCAST.identifier;
    }

    private extern (D) __gshared StringTable stringtable;

    static Identifier generateId(const(char)* prefix)
    {
        __gshared size_t i;
        return generateId(prefix, ++i);
    }

    static Identifier generateId(const(char)* prefix, size_t i)
    {
        OutBuffer buf;
        buf.writestring(prefix);
        buf.print(i);
        return idPool(buf[]);
    }

    /***************************************
     * Generate deterministic named identifier based on a source location,
     * such that the name is consistent across multiple compilations.
     * A new unique name is generated. If the prefix+location is already in
     * the stringtable, an extra suffix is added (starting the count at "_1").
     *
     * Params:
     *      prefix      = first part of the identifier name.
     *      loc         = source location to use in the identifier name.
     * Returns:
     *      Identifier (inside Identifier.idPool) with deterministic name based
     *      on the source location.
     */
    extern (D) static Identifier generateIdWithLoc(string prefix, const ref Loc loc)
    {
        // generate `<prefix>_L<line>_C<col>`
        OutBuffer idBuf;
        idBuf.writestring(prefix);
        idBuf.writestring("_L");
        idBuf.print(loc.linnum);
        idBuf.writestring("_C");
        idBuf.print(loc.charnum);

        /**
         * Make sure the identifiers are unique per filename, i.e., per module/mixin
         * (`path/to/foo.d` and `path/to/foo.d-mixin-<line>`). See issues
         * https://issues.dlang.org/show_bug.cgi?id=16995
         * https://issues.dlang.org/show_bug.cgi?id=18097
         * https://issues.dlang.org/show_bug.cgi?id=18111
         * https://issues.dlang.org/show_bug.cgi?id=18880
         * https://issues.dlang.org/show_bug.cgi?id=18868
         * https://issues.dlang.org/show_bug.cgi?id=19058
         */
        static struct Key { Loc loc; string prefix; }
        __gshared uint[Key] counters;

        static if (__traits(compiles, counters.update(Key.init, () => 0u, (ref uint a) => 0u)))
        {
            // 2.082+
            counters.update(Key(loc, prefix),
                () => 1u,          // insertion
                (ref uint counter) // update
                {
                    idBuf.writestring("_");
                    idBuf.print(counter);
                    return counter + 1;
                }
            );
        }
        else
        {
            const key = Key(loc, prefix);
            if (auto pCounter = key in counters)
            {
                idBuf.writestring("_");
                idBuf.print((*pCounter)++);
            }
            else
                counters[key] = 1;
        }

        return idPool(idBuf[]);
    }

    /********************************************
     * Create an identifier in the string table.
     */
    static Identifier idPool(const(char)* s, uint len)
    {
        return idPool(s[0 .. len]);
    }

    extern (D) static Identifier idPool(const(char)[] s)
    {
        StringValue* sv = stringtable.update(s);
        Identifier id = cast(Identifier)sv.ptrvalue;
        if (!id)
        {
            id = new Identifier(sv.toString(), TOK.identifier);
            sv.ptrvalue = cast(char*)id;
        }
        return id;
    }

    extern (D) static Identifier idPool(const(char)* s, size_t len, int value)
    {
        return idPool(s[0 .. len], value);
    }

    extern (D) static Identifier idPool(const(char)[] s, int value)
    {
        auto sv = stringtable.insert(s, null);
        assert(sv);
        auto id = new Identifier(sv.toString(), value);
        sv.ptrvalue = cast(char*)id;
        return id;
    }

    /**********************************
     * Determine if string is a valid Identifier.
     * Params:
     *      str = string to check
     * Returns:
     *      false for invalid
     */
    static bool isValidIdentifier(const(char)* str)
    {
        return str && isValidIdentifier(str.toDString);
    }

    /**********************************
     * ditto
     */
    extern (D) static bool isValidIdentifier(const(char)[] str)
    {
        if (str.length == 0 ||
            (str[0] >= '0' && str[0] <= '9')) // beware of isdigit() on signed chars
        {
            return false;
        }

        size_t idx = 0;
        while (idx < str.length)
        {
            dchar dc;
            const q = utf_decodeChar(str.ptr, str.length, idx, dc);
            if (q ||
                !((dc >= 0x80 && isUniAlpha(dc)) || isalnum(dc) || dc == '_'))
            {
                return false;
            }
        }
        return true;
    }

    extern (D) static Identifier lookup(const(char)* s, size_t len)
    {
        return lookup(s[0 .. len]);
    }

    extern (D) static Identifier lookup(const(char)[] s)
    {
        auto sv = stringtable.lookup(s);
        if (!sv)
            return null;
        return cast(Identifier)sv.ptrvalue;
    }

    extern (D) static void initTable()
    {
        stringtable._init(28_000);
    }
}
