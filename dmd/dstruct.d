/**
 * Compiler implementation of the
 * $(LINK2 http://www.dlang.org, D programming language).
 *
 * Copyright:   Copyright (C) 1999-2018 by The D Language Foundation, All Rights Reserved
 * Authors:     $(LINK2 http://www.digitalmars.com, Walter Bright)
 * License:     $(LINK2 http://www.boost.org/LICENSE_1_0.txt, Boost License 1.0)
 * Source:      $(LINK2 https://github.com/dlang/dmd/blob/master/src/dmd/dstruct.d, _dstruct.d)
 * Documentation:  https://dlang.org/phobos/dmd_dstruct.html
 * Coverage:    https://codecov.io/gh/dlang/dmd/src/master/src/dmd/dstruct.d
 */

module dmd.dstruct;

import dmd.aggregate;
import dmd.arraytypes;
import dmd.declaration;
import dmd.dmodule;
import dmd.dscope;
import dmd.dsymbol;
import dmd.dsymbolsem;
import dmd.dtemplate;
import dmd.errors;
import dmd.expression;
import dmd.expressionsem;
import dmd.func;
import dmd.globals;
import dmd.id;
import dmd.identifier;
import dmd.mtype;
import dmd.opover;
import dmd.semantic3;
import dmd.target;
import dmd.tokens;
import dmd.typesem;
import dmd.typinf;
import dmd.visitor;

/***************************************
 * Search sd for a member function of the form:
 *   `extern (D) string toString();`
 * Params:
 *   sd = struct declaration to search
 * Returns:
 *   FuncDeclaration of `toString()` if found, `null` if not
 */
extern (C++) FuncDeclaration search_toString(StructDeclaration sd)
{
    Dsymbol s = search_function(sd, Id.tostring);
    FuncDeclaration fd = s ? s.isFuncDeclaration() : null;
    if (fd)
    {
        static __gshared TypeFunction tftostring;
        if (!tftostring)
        {
            tftostring = new TypeFunction(null, Type.tstring, 0, LINK.d);
            tftostring = tftostring.merge().toTypeFunction();
        }
        fd = fd.overloadExactMatch(tftostring);
    }
    return fd;
}

/***************************************
 * Request additional semantic analysis for TypeInfo generation.
 * Params:
 *      sc = context
 *      t = type that TypeInfo is being generated for
 */
extern (C++) void semanticTypeInfo(Scope* sc, Type t)
{
    extern (C++) final class FullTypeInfoVisitor : Visitor
    {
        alias visit = Visitor.visit;
    public:
        Scope* sc;

        override void visit(Type t)
        {
            Type tb = t.toBasetype();
            if (tb != t)
                tb.accept(this);
        }

        override void visit(TypeNext t)
        {
            if (t.next)
                t.next.accept(this);
        }

        override void visit(TypeBasic t)
        {
        }

        override void visit(TypeVector t)
        {
            t.basetype.accept(this);
        }

        override void visit(TypeAArray t)
        {
            t.index.accept(this);
            visit(cast(TypeNext)t);
        }

        override void visit(TypeFunction t)
        {
            visit(cast(TypeNext)t);
            // Currently TypeInfo_Function doesn't store parameter types.
        }

        override void visit(TypeStruct t)
        {
            //printf("semanticTypeInfo.visit(TypeStruct = %s)\n", t.toChars());
            StructDeclaration sd = t.sym;

            /* Step 1: create TypeInfoDeclaration
             */
            if (!sc) // inline may request TypeInfo.
            {
                Scope scx;
                scx._module = sd.getModule();
                getTypeInfoType(sd.loc, t, &scx);
                sd.requestTypeInfo = true;
            }
            else if (!sc.minst)
            {
                // don't yet have to generate TypeInfo instance if
                // the typeid(T) expression exists in speculative scope.
            }
            else
            {
                getTypeInfoType(sd.loc, t, sc);
                sd.requestTypeInfo = true;

                // https://issues.dlang.org/show_bug.cgi?id=15149
                // if the typeid operand type comes from a
                // result of auto function, it may be yet speculative.
                unSpeculative(sc, sd);
            }

            /* Step 2: If the TypeInfo generation requires sd.semantic3, run it later.
             * This should be done even if typeid(T) exists in speculative scope.
             * Because it may appear later in non-speculative scope.
             */
            if (!sd.members)
                return; // opaque struct
            if (!sd.xeq && !sd.xcmp && !sd.postblit && !sd.dtor && !sd.xhash && !search_toString(sd))
                return; // none of TypeInfo-specific members

            // If the struct is in a non-root module, run semantic3 to get
            // correct symbols for the member function.
            if (sd.semanticRun >= PASS.semantic3)
            {
                // semantic3 is already done
            }
            else if (TemplateInstance ti = sd.isInstantiated())
            {
                if (ti.minst && !ti.minst.isRoot())
                    Module.addDeferredSemantic3(sd);
            }
            else
            {
                if (sd.inNonRoot())
                {
                    //printf("deferred sem3 for TypeInfo - sd = %s, inNonRoot = %d\n", sd.toChars(), sd.inNonRoot());
                    Module.addDeferredSemantic3(sd);
                }
            }
        }

        override void visit(TypeClass t)
        {
        }

        override void visit(TypeTuple t)
        {
            if (t.arguments)
            {
                for (size_t i = 0; i < t.arguments.dim; i++)
                {
                    Type tprm = (*t.arguments)[i].type;
                    if (tprm)
                        tprm.accept(this);
                }
            }
        }
    }

    if (sc)
    {
        if (!sc.func)
            return;
        if (sc.intypeof)
            return;
        if (sc.flags & (SCOPE.ctfe | SCOPE.compile))
            return;
    }

    scope FullTypeInfoVisitor v = new FullTypeInfoVisitor();
    v.sc = sc;
    t.accept(v);
}

struct StructFlags
{
    alias Type = uint;

    enum Enum : int
    {
        hasPointers = 0x1, // NB: should use noPointers as in ClassFlags
    }

    alias hasPointers = Enum.hasPointers;
}

enum StructPOD : int
{
    no,    // struct is not POD
    yes,   // struct is POD
    fwd,   // POD not yet computed
}

/***********************************************************
 * All `struct` declarations are an instance of this.
 */
extern (C++) class StructDeclaration : AggregateDeclaration
{
    bool zeroInit;              // !=0 if initialize with 0 fill
    bool hasIdentityAssign;     // true if has identity opAssign
    bool hasIdentityEquals;     // true if has identity opEquals
    bool hasNoFields;           // has no fields
    FuncDeclarations postblits; // Array of postblit functions
    FuncDeclaration postblit;   // aggregate postblit

    FuncDeclaration xeq;        // TypeInfo_Struct.xopEquals
    FuncDeclaration xcmp;       // TypeInfo_Struct.xopCmp
    FuncDeclaration xhash;      // TypeInfo_Struct.xtoHash
    extern (C++) static __gshared FuncDeclaration xerreq;   // object.xopEquals
    extern (C++) static __gshared FuncDeclaration xerrcmp;  // object.xopCmp

    structalign_t alignment;    // alignment applied outside of the struct
    StructPOD ispod;            // if struct is POD

    // For 64 bit Efl function call/return ABI
    Type arg1type;
    Type arg2type;

    // Even if struct is defined as non-root symbol, some built-in operations
    // (e.g. TypeidExp, NewExp, ArrayLiteralExp, etc) request its TypeInfo.
    // For those, today TypeInfo_Struct is generated in COMDAT.
    bool requestTypeInfo;

    extern (D) this(const ref Loc loc, Identifier id, bool inObject)
    {
        super(loc, id);
        zeroInit = false; // assume false until we do semantic processing
        ispod = StructPOD.fwd;
        // For forward references
        type = new TypeStruct(this);

        if (inObject)
        {
            if (id == Id.ModuleInfo && !Module.moduleinfo)
                Module.moduleinfo = this;
        }
    }

    static StructDeclaration create(Loc loc, Identifier id, bool inObject)
    {
        return new StructDeclaration(loc, id, inObject);
    }

    override Dsymbol syntaxCopy(Dsymbol s)
    {
        StructDeclaration sd =
            s ? cast(StructDeclaration)s
              : new StructDeclaration(loc, ident, false);
        return ScopeDsymbol.syntaxCopy(sd);
    }

    final void semanticTypeInfoMembers()
    {
        if (xeq &&
            xeq._scope &&
            xeq.semanticRun < PASS.semantic3done)
        {
            uint errors = global.startGagging();
            xeq.semantic3(xeq._scope);
            if (global.endGagging(errors))
                xeq = xerreq;
        }

        if (xcmp &&
            xcmp._scope &&
            xcmp.semanticRun < PASS.semantic3done)
        {
            uint errors = global.startGagging();
            xcmp.semantic3(xcmp._scope);
            if (global.endGagging(errors))
                xcmp = xerrcmp;
        }

        FuncDeclaration ftostr = search_toString(this);
        if (ftostr &&
            ftostr._scope &&
            ftostr.semanticRun < PASS.semantic3done)
        {
            ftostr.semantic3(ftostr._scope);
        }

        if (xhash &&
            xhash._scope &&
            xhash.semanticRun < PASS.semantic3done)
        {
            xhash.semantic3(xhash._scope);
        }

        if (postblit &&
            postblit._scope &&
            postblit.semanticRun < PASS.semantic3done)
        {
            postblit.semantic3(postblit._scope);
        }

        if (dtor &&
            dtor._scope &&
            dtor.semanticRun < PASS.semantic3done)
        {
            dtor.semantic3(dtor._scope);
        }
    }

    override final Dsymbol search(const ref Loc loc, Identifier ident, int flags = SearchLocalsOnly)
    {
        //printf("%s.StructDeclaration::search('%s', flags = x%x)\n", toChars(), ident.toChars(), flags);
        if (_scope && !symtab)
            dsymbolSemantic(this, _scope);

        if (!members || !symtab) // opaque or semantic() is not yet called
        {
            error("is forward referenced when looking for `%s`", ident.toChars());
            return null;
        }

        return ScopeDsymbol.search(loc, ident, flags);
    }

    override const(char)* kind() const
    {
        return "struct";
    }

    override final void finalizeSize()
    {
        //printf("StructDeclaration::finalizeSize() %s, sizeok = %d\n", toChars(), sizeok);
        assert(sizeok != Sizeok.done);

        //printf("+StructDeclaration::finalizeSize() %s, fields.dim = %d, sizeok = %d\n", toChars(), fields.dim, sizeok);

        fields.setDim(0);   // workaround

        // Set the offsets of the fields and determine the size of the struct
        uint offset = 0;
        bool isunion = isUnionDeclaration() !is null;
        for (size_t i = 0; i < members.dim; i++)
        {
            Dsymbol s = (*members)[i];
            s.setFieldOffset(this, &offset, isunion);
        }
        if (type.ty == Terror)
            return;

        // 0 sized struct's are set to 1 byte
        if (structsize == 0)
        {
            hasNoFields = true;
            structsize = 1;
            alignsize = 1;
        }

        // Round struct size up to next alignsize boundary.
        // This will ensure that arrays of structs will get their internals
        // aligned properly.
        if (alignment == STRUCTALIGN_DEFAULT)
            structsize = (structsize + alignsize - 1) & ~(alignsize - 1);
        else
            structsize = (structsize + alignment - 1) & ~(alignment - 1);

        sizeok = Sizeok.done;

        //printf("-StructDeclaration::finalizeSize() %s, fields.dim = %d, structsize = %d\n", toChars(), fields.dim, structsize);

        if (errors)
            return;

        // Calculate fields[i].overlapped
        if (checkOverlappedFields())
        {
            errors = true;
            return;
        }

        // Determine if struct is all zeros or not
        zeroInit = true;
        foreach (vd; fields)
        {
            if (vd._init)
            {
                // Should examine init to see if it is really all 0's
                zeroInit = false;
                break;
            }
            else if (!vd.type.isZeroInit(loc))
            {
                zeroInit = false;
                break;
            }
        }

        auto tt = Target.toArgTypes(type);
        size_t dim = tt.arguments.dim;
        if (dim >= 1)
        {
            assert(dim <= 2);
            arg1type = (*tt.arguments)[0].type;
            if (dim == 2)
                arg2type = (*tt.arguments)[1].type;
        }
    }

    /***************************************
     * Fit elements[] to the corresponding types of the struct's fields.
     *
     * Params:
     *      loc = location to use for error messages
     *      sc = context
     *      elements = explicit arguments used to construct object
     *      stype = the constructed object type.
     * Returns:
     *      false if any errors occur,
     *      otherwise true and elements[] are rewritten for the output.
     */
    final bool fit(const ref Loc loc, Scope* sc, Expressions* elements, Type stype)
    {
        if (!elements)
            return true;

        size_t nfields = fields.dim - isNested();
        size_t offset = 0;
        for (size_t i = 0; i < elements.dim; i++)
        {
            Expression e = (*elements)[i];
            if (!e)
                continue;

            e = resolveProperties(sc, e);
            if (i >= nfields)
            {
                if (i == fields.dim - 1 && isNested() && e.op == TOK.null_)
                {
                    // CTFE sometimes creates null as hidden pointer; we'll allow this.
                    continue;
                }
                .error(loc, "more initializers than fields (%d) of `%s`", nfields, toChars());
                return false;
            }
            VarDeclaration v = fields[i];
            if (v.offset < offset)
            {
                .error(loc, "overlapping initialization for `%s`", v.toChars());
                return false;
            }
            offset = cast(uint)(v.offset + v.type.size());

            Type t = v.type;
            if (stype)
                t = t.addMod(stype.mod);
            Type origType = t;
            Type tb = t.toBasetype();

            const hasPointers = tb.hasPointers();
            if (hasPointers)
            {
                if ((stype.alignment() < Target.ptrsize ||
                     (v.offset & (Target.ptrsize - 1))) &&
                    (sc.func && sc.func.setUnsafe()))
                {
                    .error(loc, "field `%s.%s` cannot assign to misaligned pointers in `@safe` code",
                        toChars(), v.toChars());
                    return false;
                }
            }

            /* Look for case of initializing a static array with a too-short
             * string literal, such as:
             *  char[5] foo = "abc";
             * Allow this by doing an explicit cast, which will lengthen the string
             * literal.
             */
            if (e.op == TOK.string_ && tb.ty == Tsarray)
            {
                StringExp se = cast(StringExp)e;
                Type typeb = se.type.toBasetype();
                TY tynto = tb.nextOf().ty;
                if (!se.committed &&
                    (typeb.ty == Tarray || typeb.ty == Tsarray) &&
                    (tynto == Tchar || tynto == Twchar || tynto == Tdchar) &&
                    se.numberOfCodeUnits(tynto) < (cast(TypeSArray)tb).dim.toInteger())
                {
                    e = se.castTo(sc, t);
                    goto L1;
                }
            }

            while (!e.implicitConvTo(t) && tb.ty == Tsarray)
            {
                /* Static array initialization, as in:
                 *  T[3][5] = e;
                 */
                t = tb.nextOf();
                tb = t.toBasetype();
            }
            if (!e.implicitConvTo(t))
                t = origType; // restore type for better diagnostic

            e = e.implicitCastTo(sc, t);
        L1:
            if (e.op == TOK.error)
                return false;

            (*elements)[i] = doCopyOrMove(sc, e);
        }
        return true;
    }

    /***************************************
     * Determine if struct is POD (Plain Old Data).
     *
     * POD is defined as:
     *      $(OL
     *      $(LI not nested)
     *      $(LI no postblits, destructors, or assignment operators)
     *      $(LI no `ref` fields or fields that are themselves non-POD)
     *      )
     * The idea being these are compatible with C structs.
     *
     * Returns:
     *     true if struct is POD
     */
    final bool isPOD()
    {
        // If we've already determined whether this struct is POD.
        if (ispod != StructPOD.fwd)
            return (ispod == StructPOD.yes);

        ispod = StructPOD.yes;

        if (enclosing || postblit || dtor)
            ispod = StructPOD.no;

        // Recursively check all fields are POD.
        for (size_t i = 0; i < fields.dim; i++)
        {
            VarDeclaration v = fields[i];
            if (v.storage_class & STC.ref_)
            {
                ispod = StructPOD.no;
                break;
            }

            Type tv = v.type.baseElemOf();
            if (tv.ty == Tstruct)
            {
                TypeStruct ts = cast(TypeStruct)tv;
                StructDeclaration sd = ts.sym;
                if (!sd.isPOD())
                {
                    ispod = StructPOD.no;
                    break;
                }
            }
        }

        return (ispod == StructPOD.yes);
    }

    override final inout(StructDeclaration) isStructDeclaration() inout
    {
        return this;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * Unions are a variation on structs.
 */
extern (C++) final class UnionDeclaration : StructDeclaration
{
    extern (D) this(const ref Loc loc, Identifier id)
    {
        super(loc, id, false);
    }

    override Dsymbol syntaxCopy(Dsymbol s)
    {
        assert(!s);
        auto ud = new UnionDeclaration(loc, ident);
        return StructDeclaration.syntaxCopy(ud);
    }

    override const(char)* kind() const
    {
        return "union";
    }

    override inout(UnionDeclaration) isUnionDeclaration() inout
    {
        return this;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}
