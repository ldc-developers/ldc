/**
 * Compiler implementation of the
 * $(LINK2 http://www.dlang.org, D programming language).
 *
 * Copyright:   Copyright (C) 1999-2019 by The D Language Foundation, All Rights Reserved
 * Authors:     $(LINK2 http://www.digitalmars.com, Walter Bright)
 * License:     $(LINK2 http://www.boost.org/LICENSE_1_0.txt, Boost License 1.0)
 * Source:      $(LINK2 https://github.com/dlang/dmd/blob/master/src/dmd/semantic2.d, _semantic2.d)
 * Documentation:  https://dlang.org/phobos/dmd_semantic2.html
 * Coverage:    https://codecov.io/gh/dlang/dmd/src/master/src/dmd/semantic2.d
 */

module dmd.semantic2;

import core.stdc.stdio;
import core.stdc.string;

import dmd.aggregate;
import dmd.aliasthis;
import dmd.arraytypes;
import dmd.astcodegen;
import dmd.attrib;
import dmd.blockexit;
import dmd.clone;
import dmd.dcast;
import dmd.dclass;
import dmd.declaration;
import dmd.denum;
import dmd.dimport;
import dmd.dinterpret;
import dmd.dmodule;
import dmd.dscope;
import dmd.dstruct;
import dmd.dsymbol;
import dmd.dsymbolsem;
import dmd.dtemplate;
import dmd.dversion;
import dmd.errors;
import dmd.escape;
import dmd.expression;
import dmd.expressionsem;
import dmd.func;
import dmd.globals;
import dmd.id;
import dmd.identifier;
import dmd.init;
import dmd.initsem;
import dmd.hdrgen;
import dmd.mtype;
import dmd.nogc;
import dmd.nspace;
import dmd.objc;
import dmd.opover;
import dmd.parse;
import dmd.root.filename;
import dmd.root.outbuffer;
import dmd.root.rmem;
import dmd.root.rootobject;
import dmd.sideeffect;
import dmd.statementsem;
import dmd.staticassert;
import dmd.tokens;
import dmd.utf;
import dmd.utils;
import dmd.statement;
import dmd.target;
import dmd.templateparamsem;
import dmd.typesem;
import dmd.visitor;

enum LOG = false;


/*************************************
 * Does semantic analysis on initializers and members of aggregates.
 */
extern(C++) void semantic2(Dsymbol dsym, Scope* sc)
{
    scope v = new Semantic2Visitor(sc);
    dsym.accept(v);
}

private extern(C++) final class Semantic2Visitor : Visitor
{
    alias visit = Visitor.visit;
    Scope* sc;
    this(Scope* sc)
    {
        this.sc = sc;
    }

    override void visit(Dsymbol) {}

    override void visit(StaticAssert sa)
    {
        //printf("StaticAssert::semantic2() %s\n", sa.toChars());
        auto sds = new ScopeDsymbol();
        sc = sc.push(sds);
        sc.tinst = null;
        sc.minst = null;

        import dmd.staticcond;
        bool errors;
        bool result = evalStaticCondition(sc, sa.exp, sa.exp, errors);
        sc = sc.pop();
        if (errors)
        {
            errorSupplemental(sa.loc, "while evaluating: `static assert(%s)`", sa.exp.toChars());
        }
        else if (!result)
        {
            if (sa.msg)
            {
                sc = sc.startCTFE();
                sa.msg = sa.msg.expressionSemantic(sc);
                sa.msg = resolveProperties(sc, sa.msg);
                sc = sc.endCTFE();
                sa.msg = sa.msg.ctfeInterpret();
                if (StringExp se = sa.msg.toStringExp())
                {
                    // same with pragma(msg)
                    se = se.toUTF8(sc);
                    error(sa.loc, "static assert:  \"%.*s\"", cast(int)se.len, se.string);
                }
                else
                    error(sa.loc, "static assert:  %s", sa.msg.toChars());
            }
            else
                error(sa.loc, "static assert:  `%s` is false", sa.exp.toChars());
            if (sc.tinst)
                sc.tinst.printInstantiationTrace();
            if (!global.gag)
                fatal();
        }
    }

    override void visit(TemplateInstance tempinst)
    {
        if (tempinst.semanticRun >= PASS.semantic2)
            return;
        tempinst.semanticRun = PASS.semantic2;
        static if (LOG)
        {
            printf("+TemplateInstance.semantic2('%s')\n", tempinst.toChars());
        }
        if (!tempinst.errors && tempinst.members)
        {
            TemplateDeclaration tempdecl = tempinst.tempdecl.isTemplateDeclaration();
            assert(tempdecl);

            sc = tempdecl._scope;
            assert(sc);
            sc = sc.push(tempinst.argsym);
            sc = sc.push(tempinst);
            sc.tinst = tempinst;
            sc.minst = tempinst.minst;

            int needGagging = (tempinst.gagged && !global.gag);
            uint olderrors = global.errors;
            int oldGaggedErrors = -1; // dead-store to prevent spurious warning
            if (needGagging)
                oldGaggedErrors = global.startGagging();

            for (size_t i = 0; i < tempinst.members.dim; i++)
            {
                Dsymbol s = (*tempinst.members)[i];
                static if (LOG)
                {
                    printf("\tmember '%s', kind = '%s'\n", s.toChars(), s.kind());
                }
                s.semantic2(sc);
                if (tempinst.gagged && global.errors != olderrors)
                    break;
            }

            if (global.errors != olderrors)
            {
                if (!tempinst.errors)
                {
                    if (!tempdecl.literal)
                        tempinst.error(tempinst.loc, "error instantiating");
                    if (tempinst.tinst)
                        tempinst.tinst.printInstantiationTrace();
                }
                tempinst.errors = true;
            }
            if (needGagging)
                global.endGagging(oldGaggedErrors);

            sc = sc.pop();
            sc.pop();
        }
        static if (LOG)
        {
            printf("-TemplateInstance.semantic2('%s')\n", tempinst.toChars());
        }
    }

    override void visit(TemplateMixin tmix)
    {
        if (tmix.semanticRun >= PASS.semantic2)
            return;
        tmix.semanticRun = PASS.semantic2;
        static if (LOG)
        {
            printf("+TemplateMixin.semantic2('%s')\n", tmix.toChars());
        }
        if (tmix.members)
        {
            assert(sc);
            sc = sc.push(tmix.argsym);
            sc = sc.push(tmix);
            for (size_t i = 0; i < tmix.members.dim; i++)
            {
                Dsymbol s = (*tmix.members)[i];
                static if (LOG)
                {
                    printf("\tmember '%s', kind = '%s'\n", s.toChars(), s.kind());
                }
                s.semantic2(sc);
            }
            sc = sc.pop();
            sc.pop();
        }
        static if (LOG)
        {
            printf("-TemplateMixin.semantic2('%s')\n", tmix.toChars());
        }
    }

    override void visit(VarDeclaration vd)
    {
        if (vd.semanticRun < PASS.semanticdone && vd.inuse)
            return;

        //printf("VarDeclaration::semantic2('%s')\n", toChars());

        if (vd.aliassym)        // if it's a tuple
        {
            vd.aliassym.accept(this);
            vd.semanticRun = PASS.semantic2done;
            return;
        }

        if (vd._init && !vd.toParent().isFuncDeclaration())
        {
            vd.inuse++;

            /* https://issues.dlang.org/show_bug.cgi?id=20280
             *
             * Template instances may import modules that have not
             * finished semantic1.
             */
            if (!vd.type)
                vd.dsymbolSemantic(sc);


            // https://issues.dlang.org/show_bug.cgi?id=14166
            // Don't run CTFE for the temporary variables inside typeof
            vd._init = vd._init.initializerSemantic(sc, vd.type, sc.intypeof == 1 ? INITnointerpret : INITinterpret);
            vd.inuse--;
        }
        if (vd._init && vd.storage_class & STC.manifest)
        {
            /* Cannot initializer enums with CTFE classreferences and addresses of struct literals.
             * Scan initializer looking for them. Issue error if found.
             */
            if (ExpInitializer ei = vd._init.isExpInitializer())
            {
                static bool hasInvalidEnumInitializer(Expression e)
                {
                    static bool arrayHasInvalidEnumInitializer(Expressions* elems)
                    {
                        foreach (e; *elems)
                        {
                            if (e && hasInvalidEnumInitializer(e))
                                return true;
                        }
                        return false;
                    }

                    if (e.op == TOK.classReference)
                        return true;
                    if (e.op == TOK.address && (cast(AddrExp)e).e1.op == TOK.structLiteral)
                        return true;
                    if (e.op == TOK.arrayLiteral)
                        return arrayHasInvalidEnumInitializer((cast(ArrayLiteralExp)e).elements);
                    if (e.op == TOK.structLiteral)
                        return arrayHasInvalidEnumInitializer((cast(StructLiteralExp)e).elements);
                    if (e.op == TOK.assocArrayLiteral)
                    {
                        AssocArrayLiteralExp ae = cast(AssocArrayLiteralExp)e;
                        return arrayHasInvalidEnumInitializer(ae.values) ||
                               arrayHasInvalidEnumInitializer(ae.keys);
                    }
                    return false;
                }

                if (hasInvalidEnumInitializer(ei.exp))
                    vd.error(": Unable to initialize enum with class or pointer to struct. Use static const variable instead.");
            }
        }
        else if (vd._init && vd.isThreadlocal())
        {
            // Cannot initialize a thread-local class or pointer to struct variable with a literal
            // that itself is a thread-local reference and would need dynamic initialization also.
            if ((vd.type.ty == Tclass) && vd.type.isMutable() && !vd.type.isShared())
            {
                ExpInitializer ei = vd._init.isExpInitializer();
                if (ei && ei.exp.op == TOK.classReference)
                    vd.error("is a thread-local class and cannot have a static initializer. Use `static this()` to initialize instead.");
            }
            else if (vd.type.ty == Tpointer && vd.type.nextOf().ty == Tstruct && vd.type.nextOf().isMutable() && !vd.type.nextOf().isShared())
            {
                ExpInitializer ei = vd._init.isExpInitializer();
                if (ei && ei.exp.op == TOK.address && (cast(AddrExp)ei.exp).e1.op == TOK.structLiteral)
                    vd.error("is a thread-local pointer to struct and cannot have a static initializer. Use `static this()` to initialize instead.");
            }
        }
        vd.semanticRun = PASS.semantic2done;
    }

    override void visit(Module mod)
    {
        //printf("Module::semantic2('%s'): parent = %p\n", toChars(), parent);
        if (mod.semanticRun != PASS.semanticdone) // semantic() not completed yet - could be recursive call
            return;
        mod.semanticRun = PASS.semantic2;
        // Note that modules get their own scope, from scratch.
        // This is so regardless of where in the syntax a module
        // gets imported, it is unaffected by context.
        Scope* sc = Scope.createGlobal(mod); // create root scope
        //printf("Module = %p\n", sc.scopesym);
        // Pass 2 semantic routines: do initializers and function bodies
        for (size_t i = 0; i < mod.members.dim; i++)
        {
            Dsymbol s = (*mod.members)[i];
            s.semantic2(sc);
        }
        if (mod.userAttribDecl)
        {
            mod.userAttribDecl.semantic2(sc);
        }
        sc = sc.pop();
        sc.pop();
        mod.semanticRun = PASS.semantic2done;
        //printf("-Module::semantic2('%s'): parent = %p\n", toChars(), parent);
    }

    override void visit(FuncDeclaration fd)
    {
        import dmd.dmangle : mangleToFuncSignature;

        if (fd.semanticRun >= PASS.semantic2done)
            return;
        assert(fd.semanticRun <= PASS.semantic2);
        fd.semanticRun = PASS.semantic2;

        //printf("FuncDeclaration::semantic2 [%s] fd0 = %s %s\n", loc.toChars(), toChars(), type.toChars());

        // https://issues.dlang.org/show_bug.cgi?id=18385
        // Disable for 2.079, s.t. a deprecation cycle can be started with 2.080
        if (0)
        if (fd.overnext && !fd.errors)
        {
            OutBuffer buf1;
            OutBuffer buf2;

            // Always starts the lookup from 'this', because the conflicts with
            // previous overloads are already reported.
            auto f1 = fd;
            mangleToFuncSignature(buf1, f1);

            overloadApply(f1, (Dsymbol s)
            {
                auto f2 = s.isFuncDeclaration();
                if (!f2 || f1 == f2 || f2.errors)
                    return 0;

                // Don't have to check conflict between declaration and definition.
                if ((f1.fbody !is null) != (f2.fbody !is null))
                    return 0;

                /* Check for overload merging with base class member functions.
                 *
                 *  class B { void foo() {} }
                 *  class D : B {
                 *    override void foo() {}    // B.foo appears as f2
                 *    alias foo = B.foo;
                 *  }
                 */
                if (f1.overrides(f2))
                    return 0;

                // extern (C) functions always conflict each other.
                if (f1.ident == f2.ident &&
                    f1.toParent2() == f2.toParent2() &&
                    (f1.linkage != LINK.d && f1.linkage != LINK.cpp) &&
                    (f2.linkage != LINK.d && f2.linkage != LINK.cpp))
                {
                    /* Allow the hack that is actually used in druntime,
                     * to ignore function attributes for extern (C) functions.
                     * TODO: Must be reconsidered in the future.
                     *  BUG: https://issues.dlang.org/show_bug.cgi?id=18206
                     *
                     *  extern(C):
                     *  alias sigfn_t  = void function(int);
                     *  alias sigfn_t2 = void function(int) nothrow @nogc;
                     *  sigfn_t  bsd_signal(int sig, sigfn_t  func);
                     *  sigfn_t2 bsd_signal(int sig, sigfn_t2 func) nothrow @nogc;  // no error
                     */
                    if (f1.fbody is null || f2.fbody is null)
                        return 0;

                    auto tf1 = cast(TypeFunction)f1.type;
                    auto tf2 = cast(TypeFunction)f2.type;
                    error(f2.loc, "%s `%s%s` cannot be overloaded with %s`extern(%s)` function at %s",
                            f2.kind(),
                            f2.toPrettyChars(),
                            parametersTypeToChars(tf2.parameterList),
                            (f1.linkage == f2.linkage ? "another " : "").ptr,
                            linkageToChars(f1.linkage), f1.loc.toChars());
                    f2.type = Type.terror;
                    f2.errors = true;
                    return 0;
                }

                buf2.reset();
                mangleToFuncSignature(buf2, f2);

                auto s1 = buf1.peekChars();
                auto s2 = buf2.peekChars();

                //printf("+%s\n\ts1 = %s\n\ts2 = %s @ [%s]\n", toChars(), s1, s2, f2.loc.toChars());
                if (strcmp(s1, s2) == 0)
                {
                    auto tf2 = cast(TypeFunction)f2.type;
                    error(f2.loc, "%s `%s%s` conflicts with previous declaration at %s",
                            f2.kind(),
                            f2.toPrettyChars(),
                            parametersTypeToChars(tf2.parameterList),
                            f1.loc.toChars());
                    f2.type = Type.terror;
                    f2.errors = true;
                }
                return 0;
            });
        }
        if (!fd.type || fd.type.ty != Tfunction)
            return;
        TypeFunction f = cast(TypeFunction) fd.type;

        //semantic for parameters' UDAs
        foreach (i; 0 .. f.parameterList.length)
        {
            Parameter param = f.parameterList[i];
            if (param && param.userAttribDecl)
                param.userAttribDecl.semantic2(sc);
        }
    }

    override void visit(Import i)
    {
        //printf("Import::semantic2('%s')\n", toChars());
        if (i.mod)
        {
            i.mod.semantic2(null);
            if (i.mod.needmoduleinfo)
            {
                //printf("module5 %s because of %s\n", sc.module.toChars(), mod.toChars());
                if (sc)
                    sc._module.needmoduleinfo = 1;
            }
        }
    }

    override void visit(Nspace ns)
    {
        if (ns.semanticRun >= PASS.semantic2)
            return;
        ns.semanticRun = PASS.semantic2;
        static if (LOG)
        {
            printf("+Nspace::semantic2('%s')\n", ns.toChars());
        }
        if (ns.members)
        {
            assert(sc);
            sc = sc.push(ns);
            sc.linkage = LINK.cpp;
            foreach (s; *ns.members)
            {
                static if (LOG)
                {
                    printf("\tmember '%s', kind = '%s'\n", s.toChars(), s.kind());
                }
                s.semantic2(sc);
            }
            sc.pop();
        }
        static if (LOG)
        {
            printf("-Nspace::semantic2('%s')\n", ns.toChars());
        }
    }

    override void visit(AttribDeclaration ad)
    {
        Dsymbols* d = ad.include(sc);
        if (d)
        {
            Scope* sc2 = ad.newScope(sc);
            for (size_t i = 0; i < d.dim; i++)
            {
                Dsymbol s = (*d)[i];
                s.semantic2(sc2);
            }
            if (sc2 != sc)
                sc2.pop();
        }
    }

    /**
     * Run the DeprecatedDeclaration's semantic2 phase then its members.
     *
     * The message set via a `DeprecatedDeclaration` can be either of:
     * - a string literal
     * - an enum
     * - a static immutable
     * So we need to call ctfe to resolve it.
     * Afterward forwards to the members' semantic2.
     */
    override void visit(DeprecatedDeclaration dd)
    {
        getMessage(dd);
        visit(cast(AttribDeclaration)dd);
    }

    override void visit(AlignDeclaration ad)
    {
        ad.getAlignment(sc);
        visit(cast(AttribDeclaration)ad);
    }

    override void visit(UserAttributeDeclaration uad)
    {
        if (uad.decl && uad.atts && uad.atts.dim && uad._scope)
        {
            static void eval(Scope* sc, Expressions* exps)
            {
                foreach (ref Expression e; *exps)
                {
                    if (e)
                    {
                        e = e.expressionSemantic(sc);
                        if (definitelyValueParameter(e))
                            e = e.ctfeInterpret();
                        if (e.op == TOK.tuple)
                        {
                            TupleExp te = cast(TupleExp)e;
                            eval(sc, te.exps);
                        }
                    }
                }
            }

            uad._scope = null;
            eval(sc, uad.atts);
        }
        visit(cast(AttribDeclaration)uad);
    }

    override void visit(AggregateDeclaration ad)
    {
        //printf("AggregateDeclaration::semantic2(%s) type = %s, errors = %d\n", ad.toChars(), ad.type.toChars(), ad.errors);
        if (!ad.members)
            return;

        if (ad._scope)
        {
            ad.error("has forward references");
            return;
        }

        auto sc2 = ad.newScope(sc);

        ad.determineSize(ad.loc);

        for (size_t i = 0; i < ad.members.dim; i++)
        {
            Dsymbol s = (*ad.members)[i];
            //printf("\t[%d] %s\n", i, s.toChars());
            s.semantic2(sc2);
        }

        sc2.pop();
    }
}
