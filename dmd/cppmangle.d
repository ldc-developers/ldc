/**
 * Compiler implementation of the $(LINK2 http://www.dlang.org, D programming language)
 *
 * Do mangling for C++ linkage.
 *
 * Copyright: Copyright (C) 1999-2018 by The D Language Foundation, All Rights Reserved
 * Authors: Walter Bright, http://www.digitalmars.com
 * License:   $(LINK2 http://www.boost.org/LICENSE_1_0.txt, Boost License 1.0)
 * Source:    $(LINK2 https://github.com/dlang/dmd/blob/master/src/dmd/cppmangle.d, _cppmangle.d)
 * Documentation:  https://dlang.org/phobos/dmd_cppmangle.html
 * Coverage:    https://codecov.io/gh/dlang/dmd/src/master/src/dmd/cppmangle.d
 *
 * References:
 *  Follows Itanium C++ ABI 1.86 section 5.1
 *  http://refspecs.linux-foundation.org/cxxabi-1.86.html#mangling
 *  which is where the grammar comments come from.
 *
 * Bugs:
 *  https://issues.dlang.org/query.cgi
 *  enter `C++, mangling` as the keywords.
 */

module dmd.cppmangle;

import core.stdc.string;
import core.stdc.stdio;

import dmd.arraytypes;
import dmd.declaration;
import dmd.dsymbol;
import dmd.dtemplate;
import dmd.errors;
import dmd.expression;
import dmd.func;
import dmd.globals;
import dmd.id;
import dmd.identifier;
import dmd.mtype;
import dmd.root.outbuffer;
import dmd.root.rootobject;
import dmd.target;
import dmd.tokens;
import dmd.typesem;
import dmd.visitor;

extern (C++):

const(char)* toCppMangleItanium(Dsymbol s)
{
    //printf("toCppMangleItanium(%s)\n", s.toChars());
    OutBuffer buf;
    scope CppMangleVisitor v = new CppMangleVisitor(&buf, s.loc);
    v.mangleOf(s);
    return buf.extractString();
}

const(char)* cppTypeInfoMangleItanium(Dsymbol s)
{
    //printf("cppTypeInfoMangle(%s)\n", s.toChars());
    OutBuffer buf;
    buf.writestring("_ZTI");    // "TI" means typeinfo structure
    scope CppMangleVisitor v = new CppMangleVisitor(&buf, s.loc);
    v.cpp_mangle_name(s, false);
    return buf.extractString();
}

private final class CppMangleVisitor : Visitor
{
    alias visit = Visitor.visit;
    Objects components;         // array of components available for substitution
    OutBuffer* buf;             // append the mangling to buf[]
    Loc loc;                    // location for use in error messages

  final:
    bool substitute(RootObject p)
    {
        //printf("substitute %s\n", p ? p.toChars() : null);
        auto i = find(p);
        if (i >= 0)
        {
            //printf("\tmatch\n");
            /* Sequence is S_, S0_, .., S9_, SA_, ..., SZ_, S10_, ...
             */
            buf.writeByte('S');
            if (i)
            {
                // Write <seq-id> to buf
                void write_seq_id(size_t i)
                {
                    if (i >= 36)
                    {
                        write_seq_id(i / 36);
                        i %= 36;
                    }
                    i += (i < 10) ? '0' : 'A' - 10;
                    buf.writeByte(cast(char)i);
                }

                write_seq_id(i - 1);
            }
            buf.writeByte('_');
            return true;
        }
        return false;
    }

    /******
     * See if `p` exists in components[]
     * Returns:
     *  index if found, -1 if not
     */
    int find(RootObject p)
    {
        //printf("find %p %d %s\n", p, p.dyncast(), p ? p.toChars() : null);
        foreach (i, component; components)
        {
            if (p == component)
                return cast(int)i;
        }
        return -1;
    }

    /*********************
     * Append p to components[]
     */
    void append(RootObject p)
    {
        //printf("append %p %d %s\n", p, p.dyncast(), p ? p.toChars() : "null");
        components.push(p);
    }

    /************************
     * Determine if symbol is indeed the global ::std namespace.
     * Params:
     *  s = symbol to check
     * Returns:
     *  true if it is ::std
     */
    static bool isStd(Dsymbol s)
    {
        return (s &&
                s.ident == Id.std &&    // the right name
                s.isNspace() &&         // g++ disallows global "std" for other than a namespace
                !getQualifier(s));      // at global level
    }

    /******************************
     * Write the mangled representation of the template arguments.
     * Params:
     *  ti = the template instance
     */
    void template_args(TemplateInstance ti)
    {
        /* <template-args> ::= I <template-arg>+ E
         */
        if (!ti)                // could happen if std::basic_string is not a template
            return;
        buf.writeByte('I');
        foreach (i, o; *ti.tiargs)
        {
            TemplateDeclaration td = ti.tempdecl.isTemplateDeclaration();
            assert(td);
            TemplateParameter tp = (*td.parameters)[i];

            /*
             * <template-arg> ::= <type>               # type or template
             *                ::= X <expression> E     # expression
             *                ::= <expr-primary>       # simple expressions
             *                ::= I <template-arg>* E  # argument pack
             */
            if (TemplateTupleParameter tt = tp.isTemplateTupleParameter())
            {
                buf.writeByte('I');     // argument pack

                // mangle the rest of the arguments as types
                foreach (j; i .. (*ti.tiargs).dim)
                {
                    Type t = isType((*ti.tiargs)[j]);
                    assert(t);
                    t.accept(this);
                }

                buf.writeByte('E');
                break;
            }

            if (tp.isTemplateTypeParameter())
            {
                Type t = isType(o);
                assert(t);
                t.accept(this);
            }
            else if (TemplateValueParameter tv = tp.isTemplateValueParameter())
            {
                // <expr-primary> ::= L <type> <value number> E  # integer literal
                if (tv.valType.isintegral())
                {
                    Expression e = isExpression(o);
                    assert(e);
                    buf.writeByte('L');
                    tv.valType.accept(this);
                    auto val = e.toUInteger();
                    if (!tv.valType.isunsigned() && cast(sinteger_t)val < 0)
                    {
                        val = -val;
                        buf.writeByte('n');
                    }
                    buf.print(val);
                    buf.writeByte('E');
                }
                else
                {
                    ti.error("Internal Compiler Error: C++ `%s` template value parameter is not supported", tv.valType.toChars());
                    fatal();
                }
            }
            else if (tp.isTemplateAliasParameter())
            {
                Dsymbol d = isDsymbol(o);
                Expression e = isExpression(o);
                if (d && d.isFuncDeclaration())
                {
                    bool is_nested = d.toParent() &&
                        !d.toParent().isModule() &&
                        (cast(TypeFunction)d.isFuncDeclaration().type).linkage == LINK.cpp;
                    if (is_nested)
                        buf.writeByte('X');
                    buf.writeByte('L');
                    mangle_function(d.isFuncDeclaration());
                    buf.writeByte('E');
                    if (is_nested)
                        buf.writeByte('E');
                }
                else if (e && e.op == TOK.variable && (cast(VarExp)e).var.isVarDeclaration())
                {
                    VarDeclaration vd = (cast(VarExp)e).var.isVarDeclaration();
                    buf.writeByte('L');
                    mangle_variable(vd, true);
                    buf.writeByte('E');
                }
                else if (d && d.isTemplateDeclaration() && d.isTemplateDeclaration().onemember)
                {
                    if (!substitute(d))
                    {
                        cpp_mangle_name(d, false);
                    }
                }
                else
                {
                    ti.error("Internal Compiler Error: C++ `%s` template alias parameter is not supported", o.toChars());
                    fatal();
                }
            }
            else if (tp.isTemplateThisParameter())
            {
                ti.error("Internal Compiler Error: C++ `%s` template this parameter is not supported", o.toChars());
                fatal();
            }
            else
            {
                assert(0);
            }
        }
        buf.writeByte('E');
    }


    void source_name(Dsymbol s)
    {
        //printf("source_name(%s)\n", s.toChars());
        if (TemplateInstance ti = s.isTemplateInstance())
        {
            if (!substitute(ti.tempdecl))
            {
                append(ti.tempdecl);
                const name = ti.tempdecl.toAlias().ident.toString();
                buf.print(name.length);
                buf.writestring(name);
            }
            template_args(ti);
        }
        else
        {
            const name = s.ident.toString();
            buf.print(name.length);
            buf.writestring(name);
        }
    }

    /********
     * See if s is actually an instance of a template
     * Params:
     *  s = symbol
     * Returns:
     *  if s is instance of a template, return the instance, otherwise return s
     */
    Dsymbol getInstance(Dsymbol s)
    {
        Dsymbol p = s.toParent();
        if (p)
        {
            if (TemplateInstance ti = p.isTemplateInstance())
                return ti;
        }
        return s;
    }

    /********
     * Get qualifier for `s`, meaning the symbol
     * that s is in the symbol table of.
     * The module does not count as a qualifier, because C++
     * does not have modules.
     * Params:
     *  s = symbol that may have a qualifier
     *      s is rewritten to be TemplateInstance if s is one
     * Returns:
     *  qualifier, null if none
     */
    static Dsymbol getQualifier(Dsymbol s)
    {
        Dsymbol p = s.toParent();
        return (p && !p.isModule()) ? p : null;
    }

    // Detect type char
    static bool isChar(RootObject o)
    {
        Type t = isType(o);
        return (t && t.equals(Type.tchar));
    }

    // Detect type ::std::char_traits<char>
    static bool isChar_traits_char(RootObject o)
    {
        return isIdent_char(Id.char_traits, o);
    }

    // Detect type ::std::allocator<char>
    static bool isAllocator_char(RootObject o)
    {
        return isIdent_char(Id.allocator, o);
    }

    // Detect type ::std::ident<char>
    static bool isIdent_char(Identifier ident, RootObject o)
    {
        Type t = isType(o);
        if (!t || t.ty != Tstruct)
            return false;
        Dsymbol s = (cast(TypeStruct)t).toDsymbol(null);
        if (s.ident != ident)
            return false;
        Dsymbol p = s.toParent();
        if (!p)
            return false;
        TemplateInstance ti = p.isTemplateInstance();
        if (!ti)
            return false;
        Dsymbol q = getQualifier(ti);
        return isStd(q) && ti.tiargs.dim == 1 && isChar((*ti.tiargs)[0]);
    }

    /***
     * Detect template args <char, ::std::char_traits<char>>
     * and write st if found.
     * Returns:
     *  true if found
     */
    extern (D) bool char_std_char_traits_char(TemplateInstance ti, string st)
    {
        if (ti.tiargs.dim == 2 &&
            isChar((*ti.tiargs)[0]) &&
            isChar_traits_char((*ti.tiargs)[1]))
        {
            buf.writestring(st.ptr);
            return true;
        }
        return false;
    }


    void prefix_name(Dsymbol s)
    {
        //printf("prefix_name(%s)\n", s.toChars());
        if (!substitute(s))
        {
            auto si = getInstance(s);
            Dsymbol p = getQualifier(si);
            if (p)
            {
                if (isStd(p))
                {
                    TemplateInstance ti = si.isTemplateInstance();
                    if (ti)
                    {
                        if (s.ident == Id.allocator)
                        {
                            buf.writestring("Sa");
                            template_args(ti);
                            append(ti);
                            return;
                        }
                        if (s.ident == Id.basic_string)
                        {
                            // ::std::basic_string<char, ::std::char_traits<char>, ::std::allocator<char>>
                            if (ti.tiargs.dim == 3 &&
                                isChar((*ti.tiargs)[0]) &&
                                isChar_traits_char((*ti.tiargs)[1]) &&
                                isAllocator_char((*ti.tiargs)[2]))

                            {
                                buf.writestring("Ss");
                                return;
                            }
                            buf.writestring("Sb");      // ::std::basic_string
                            template_args(ti);
                            append(ti);
                            return;
                        }

                        // ::std::basic_istream<char, ::std::char_traits<char>>
                        if (s.ident == Id.basic_istream &&
                            char_std_char_traits_char(ti, "Si"))
                            return;

                        // ::std::basic_ostream<char, ::std::char_traits<char>>
                        if (s.ident == Id.basic_ostream &&
                            char_std_char_traits_char(ti, "So"))
                            return;

                        // ::std::basic_iostream<char, ::std::char_traits<char>>
                        if (s.ident == Id.basic_iostream &&
                            char_std_char_traits_char(ti, "Sd"))
                            return;
                    }
                    buf.writestring("St");
                }
                else
                    prefix_name(p);
            }
            source_name(si);
            if (!isStd(si))
                /* Do this after the source_name() call to keep components[]
                 * in the right order.
                 * https://issues.dlang.org/show_bug.cgi?id=17947
                 */
                append(si);
        }
    }


    void cpp_mangle_name(Dsymbol s, bool qualified)
    {
        //printf("cpp_mangle_name(%s, %d)\n", s.toChars(), qualified);
        Dsymbol p = s.toParent();
        Dsymbol se = s;
        bool write_prefix = true;
        if (p && p.isTemplateInstance())
        {
            se = p;
            if (find(p.isTemplateInstance().tempdecl) >= 0)
                write_prefix = false;
            p = p.toParent();
        }
        if (p && !p.isModule())
        {
            /* The N..E is not required if:
             * 1. the parent is 'std'
             * 2. 'std' is the initial qualifier
             * 3. there is no CV-qualifier or a ref-qualifier for a member function
             * ABI 5.1.8
             */
            if (isStd(p) && !qualified)
            {
                TemplateInstance ti = se.isTemplateInstance();
                if (s.ident == Id.allocator)
                {
                    buf.writestring("Sa"); // "Sa" is short for ::std::allocator
                    template_args(ti);
                }
                else if (s.ident == Id.basic_string)
                {
                    // ::std::basic_string<char, ::std::char_traits<char>, ::std::allocator<char>>
                    if (ti.tiargs.dim == 3 &&
                        isChar((*ti.tiargs)[0]) &&
                        isChar_traits_char((*ti.tiargs)[1]) &&
                        isAllocator_char((*ti.tiargs)[2]))

                    {
                        buf.writestring("Ss");
                        return;
                    }
                    buf.writestring("Sb");      // ::std::basic_string
                    template_args(ti);
                }
                else
                {
                    // ::std::basic_istream<char, ::std::char_traits<char>>
                    if (s.ident == Id.basic_istream)
                    {
                        if (char_std_char_traits_char(ti, "Si"))
                            return;
                    }
                    else if (s.ident == Id.basic_ostream)
                    {
                        if (char_std_char_traits_char(ti, "So"))
                            return;
                    }
                    else if (s.ident == Id.basic_iostream)
                    {
                        if (char_std_char_traits_char(ti, "Sd"))
                            return;
                    }
                    buf.writestring("St");
                    source_name(se);
                }
            }
            else
            {
                buf.writeByte('N');
                if (write_prefix)
                    prefix_name(p);
                source_name(se);
                buf.writeByte('E');
            }
        }
        else
            source_name(se);
        append(s);
    }

    void CV_qualifiers(Type t)
    {
        // CV-qualifiers are 'r': restrict, 'V': volatile, 'K': const
        if (t.isConst())
            buf.writeByte('K');
    }

    void mangle_variable(VarDeclaration d, bool is_temp_arg_ref)
    {
        // fake mangling for fields to fix https://issues.dlang.org/show_bug.cgi?id=16525
        if (!(d.storage_class & (STC.extern_ | STC.field | STC.gshared)))
        {
            d.error("Internal Compiler Error: C++ static non-`__gshared` non-`extern` variables not supported");
            fatal();
        }
        Dsymbol p = d.toParent();
        if (p && !p.isModule()) //for example: char Namespace1::beta[6] should be mangled as "_ZN10Namespace14betaE"
        {
            buf.writestring("_ZN");
            prefix_name(p);
            source_name(d);
            buf.writeByte('E');
        }
        else //char beta[6] should mangle as "beta"
        {
            if (!is_temp_arg_ref)
            {
                buf.writestring(d.ident.toChars());
            }
            else
            {
                buf.writestring("_Z");
                source_name(d);
            }
        }
    }

    void mangle_function(FuncDeclaration d)
    {
        //printf("mangle_function(%s)\n", d.toChars());
        /*
         * <mangled-name> ::= _Z <encoding>
         * <encoding> ::= <function name> <bare-function-type>
         *            ::= <data name>
         *            ::= <special-name>
         */
        TypeFunction tf = cast(TypeFunction)d.type;
        buf.writestring("_Z");

        if (TemplateDeclaration ftd = getFuncTemplateDecl(d))
        {
            /* It's an instance of a function template
             */
            TemplateInstance ti = d.parent.isTemplateInstance();
            assert(ti);
            source_name(ti);
            headOfType(tf.nextOf());  // mangle return type
        }
        else
        {
            Dsymbol p = d.toParent();
            if (p && !p.isModule() && tf.linkage == LINK.cpp)
            {
                /* <nested-name> ::= N [<CV-qualifiers>] <prefix> <unqualified-name> E
                 *               ::= N [<CV-qualifiers>] <template-prefix> <template-args> E
                 */
                buf.writeByte('N');
                CV_qualifiers(d.type);

                /* <prefix> ::= <prefix> <unqualified-name>
                 *          ::= <template-prefix> <template-args>
                 *          ::= <template-param>
                 *          ::= # empty
                 *          ::= <substitution>
                 *          ::= <prefix> <data-member-prefix>
                 */
                prefix_name(p);
                //printf("p: %s\n", buf.peekString());

                if (d.isDtorDeclaration())
                {
                    buf.writestring("D1");
                }
                else
                {
                    source_name(d);
                }
                buf.writeByte('E');
            }
            else
            {
                source_name(d);
            }
        }

        if (tf.linkage == LINK.cpp) //Template args accept extern "C" symbols with special mangling
        {
            assert(tf.ty == Tfunction);
            mangleFunctionParameters(tf.parameters, tf.varargs);
        }
    }

    void mangleFunctionParameters(Parameters* parameters, int varargs)
    {
        int numparams = 0;

        int paramsCppMangleDg(size_t n, Parameter fparam)
        {
            Type t = Target.cppParameterType(fparam);
            if (t.ty == Tsarray)
            {
                // Static arrays in D are passed by value; no counterpart in C++
                t.error(loc, "Internal Compiler Error: unable to pass static array `%s` to extern(C++) function, use pointer instead",
                    t.toChars());
                fatal();
            }
            headOfType(t);
            ++numparams;
            return 0;
        }

        if (parameters)
            Parameter._foreach(parameters, &paramsCppMangleDg);
        if (varargs)
            buf.writeByte('z');
        else if (!numparams)
            buf.writeByte('v'); // encode (void) parameters
    }

public:
    extern (D) this(OutBuffer* buf, Loc loc)
    {
        this.buf = buf;
        this.loc = loc;
    }

    /*****
     * Entry point. Append mangling to buf[]
     * Params:
     *  s = symbol to mangle
     */
    void mangleOf(Dsymbol s)
    {
        if (VarDeclaration vd = s.isVarDeclaration())
        {
            mangle_variable(vd, false);
        }
        else if (FuncDeclaration fd = s.isFuncDeclaration())
        {
            mangle_function(fd);
        }
        else
        {
            assert(0);
        }
    }

    /****** The rest is type mangling ************/

    void error(Type t)
    {
        const(char)* p;
        if (t.isImmutable())
            p = "`immutable` ";
        else if (t.isShared())
            p = "`shared` ";
        else
            p = "";
        t.error(loc, "Internal Compiler Error: %stype `%s` can not be mapped to C++\n", p, t.toChars());
        fatal(); //Fatal, because this error should be handled in frontend
    }

    /****************************
     * Mangle a type,
     * treating it as a Head followed by a Tail.
     * Params:
     *  t = Head of a type
     */
    void headOfType(Type t)
    {
        if (t.ty == Tclass)
        {
            mangleTypeClass(cast(TypeClass)t, true);
        }
        else
        {
            // For value types, strip const/immutable/shared from the head of the type
            t.mutableOf().unSharedOf().accept(this);
        }
    }

    override void visit(Type t)
    {
        error(t);
    }

    /******
     * Write out 1 or 2 character basic type mangling.
     * Handle const and substitutions.
     * Params:
     *  t = type to mangle
     *  p = if not 0, then character prefix
     *  c = mangling character
     */
    void writeBasicType(Type t, char p, char c)
    {
        if (p || t.isConst())
        {
            if (substitute(t))
                return;
            else
                append(t);
        }
        CV_qualifiers(t);
        if (p)
            buf.writeByte(p);
        buf.writeByte(c);
    }

    override void visit(TypeBasic t)
    {
        if (t.isImmutable() || t.isShared())
            return error(t);

        /* <builtin-type>:
         * v        void
         * w        wchar_t
         * b        bool
         * c        char
         * a        signed char
         * h        unsigned char
         * s        short
         * t        unsigned short
         * i        int
         * j        unsigned int
         * l        long
         * m        unsigned long
         * x        long long, __int64
         * y        unsigned long long, __int64
         * n        __int128
         * o        unsigned __int128
         * f        float
         * d        double
         * e        long double, __float80
         * g        __float128
         * z        ellipsis
         * Dd       64 bit IEEE 754r decimal floating point
         * De       128 bit IEEE 754r decimal floating point
         * Df       32 bit IEEE 754r decimal floating point
         * Dh       16 bit IEEE 754r half-precision floating point
         * Di       char32_t
         * Ds       char16_t
         * u <source-name>  # vendor extended type
         */
        char c;
        char p = 0;
        switch (t.ty)
        {
            case Tvoid:                 c = 'v';        break;
            case Tint8:                 c = 'a';        break;
            case Tuns8:                 c = 'h';        break;
            case Tint16:                c = 's';        break;
            case Tuns16:                c = 't';        break;
            case Tint32:                c = 'i';        break;
            case Tuns32:                c = 'j';        break;
            case Tfloat32:              c = 'f';        break;
            case Tint64:
                c = Target.c_longsize == 8 ? Target.int64Mangle : 'x';
                break;
            case Tuns64:
                c = Target.c_longsize == 8 ? Target.uint64Mangle : 'y';
                break;
            case Tint128:                c = 'n';       break;
            case Tuns128:                c = 'o';       break;
            case Tfloat64:               c = 'd';       break;
            case Tfloat80:               c = 'e';       break;
            case Tbool:                  c = 'b';       break;
            case Tchar:                  c = 'c';       break;
            case Twchar:                 c = 't';       break;  // unsigned short (perhaps use 'Ds' ?
            case Tdchar:                 c = 'w';       break;  // wchar_t (UTF-32) (perhaps use 'Di' ?
            case Timaginary32:  p = 'G'; c = 'f';       break;  // 'G' means imaginary
            case Timaginary64:  p = 'G'; c = 'd';       break;
            case Timaginary80:  p = 'G'; c = 'e';       break;
            case Tcomplex32:    p = 'C'; c = 'f';       break;  // 'C' means complex
            case Tcomplex64:    p = 'C'; c = 'd';       break;
            case Tcomplex80:    p = 'C'; c = 'e';       break;

            default:
                // Handle any target-specific basic types.
                if (auto tm = Target.cppTypeMangle(t))
                {
                    if (substitute(t))
                        return;
                    else
                        append(t);
                    CV_qualifiers(t);
                    buf.writestring(tm);
                    return;
                }
                return error(t);
        }
        writeBasicType(t, p, c);
    }

    override void visit(TypeVector t)
    {
        if (t.isImmutable() || t.isShared())
            return error(t);

        if (substitute(t))
            return;
        append(t);
        CV_qualifiers(t);

        // Handle any target-specific vector types.
        if (auto tm = Target.cppTypeMangle(t))
        {
            buf.writestring(tm);
        }
        else
        {
            assert(t.basetype && t.basetype.ty == Tsarray);
            assert((cast(TypeSArray)t.basetype).dim);
            version (none)
            {
                buf.writestring("Dv");
                buf.print((cast(TypeSArray *)t.basetype).dim.toInteger()); // -- Gnu ABI v.4
                buf.writeByte('_');
            }
            else
                buf.writestring("U8__vector"); //-- Gnu ABI v.3
            t.basetype.nextOf().accept(this);
        }
    }

    override void visit(TypeSArray t)
    {
        if (t.isImmutable() || t.isShared())
            return error(t);

        if (!substitute(t))
            append(t);
        CV_qualifiers(t);
        buf.writeByte('A');
        buf.print(t.dim ? t.dim.toInteger() : 0);
        buf.writeByte('_');
        t.next.accept(this);
    }

    override void visit(TypePointer t)
    {
        if (t.isImmutable() || t.isShared())
            return error(t);

        if (substitute(t))
            return;
        CV_qualifiers(t);
        buf.writeByte('P');
        t.next.accept(this);
        append(t);
    }

    override void visit(TypeReference t)
    {
        //printf("TypeReference %s\n", t.toChars());
        if (substitute(t))
            return;
        buf.writeByte('R');
        t.next.accept(this);
        append(t);
    }

    override void visit(TypeFunction t)
    {
        /*
         *  <function-type> ::= F [Y] <bare-function-type> E
         *  <bare-function-type> ::= <signature type>+
         *  # types are possible return type, then parameter types
         */
        /* ABI says:
            "The type of a non-static member function is considered to be different,
            for the purposes of substitution, from the type of a namespace-scope or
            static member function whose type appears similar. The types of two
            non-static member functions are considered to be different, for the
            purposes of substitution, if the functions are members of different
            classes. In other words, for the purposes of substitution, the class of
            which the function is a member is considered part of the type of
            function."

            BUG: Right now, types of functions are never merged, so our simplistic
            component matcher always finds them to be different.
            We should use Type.equals on these, and use different
            TypeFunctions for non-static member functions, and non-static
            member functions of different classes.
         */
        if (substitute(t))
            return;
        buf.writeByte('F');
        if (t.linkage == LINK.c)
            buf.writeByte('Y');
        Type tn = t.next;
        if (t.isref)
            tn = tn.referenceTo();
        tn.accept(this);
        mangleFunctionParameters(t.parameters, t.varargs);
        buf.writeByte('E');
        append(t);
    }

    override void visit(TypeStruct t)
    {
        if (t.isImmutable() || t.isShared())
            return error(t);

        /* magic structs __c_(u)long(long) get special mangling
         */
        const id = t.sym.ident;
        //printf("struct id = '%s'\n", id.toChars());
        if (id == Id.__c_long)
            return writeBasicType(t, 0, 'l');
        else if (id == Id.__c_ulong)
            return writeBasicType(t, 0, 'm');
        else if (id == Id.__c_longlong)
            return writeBasicType(t, 0, 'x');
        else if (id == Id.__c_ulonglong)
            return writeBasicType(t, 0, 'y');

        //printf("TypeStruct %s\n", t.toChars());
        doSymbol(t);
    }

    override void visit(TypeEnum t)
    {
        if (t.isImmutable() || t.isShared())
            return error(t);

        doSymbol(t);
    }

    /****************
     * Write structs and enums.
     * Params:
     *  t = TypeStruct or TypeEnum
     */
    final void doSymbol(Type t)
    {
        if (substitute(t))
            return;
        CV_qualifiers(t);

        // Handle any target-specific struct types.
        if (auto tm = Target.cppTypeMangle(t))
        {
            buf.writestring(tm);
        }
        else
        {
            Dsymbol s = t.toDsymbol(null);
            Dsymbol p = s.toParent();
            if (p && p.isTemplateInstance())
            {
                 /* https://issues.dlang.org/show_bug.cgi?id=17947
                  * Substitute the template instance symbol, not the struct/enum symbol
                  */
                if (substitute(p))
                    return;
            }
            if (!substitute(s))
            {
                cpp_mangle_name(s, t.isConst());
            }
        }
        if (t.isConst())
            append(t);
    }

    override void visit(TypeClass t)
    {
        mangleTypeClass(t, false);
    }

    /************************
     * Mangle a class type.
     * If it's the head, treat the initial pointer as a value type.
     * Params:
     *  t = class type
     *  head = true for head of a type
     */
    void mangleTypeClass(TypeClass t, bool head)
    {
        if (t.isImmutable() || t.isShared())
            return error(t);

        /* Mangle as a <pointer to><struct>
         */
        if (substitute(t))
            return;
        if (!head)
            CV_qualifiers(t);
        buf.writeByte('P');

        CV_qualifiers(t);

        {
            Dsymbol s = t.toDsymbol(null);
            Dsymbol p = s.toParent();
            if (p && p.isTemplateInstance())
            {
                 /* https://issues.dlang.org/show_bug.cgi?id=17947
                  * Substitute the template instance symbol, not the class symbol
                  */
                if (substitute(p))
                    return;
            }
        }

        if (!substitute(t.sym))
        {
            cpp_mangle_name(t.sym, t.isConst());
        }
        if (t.isConst())
            append(null);  // C++ would have an extra type here
        append(t);
    }
}
