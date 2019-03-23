/**
 * Compiler implementation of the
 * $(LINK2 http://www.dlang.org, D programming language).
 *
 * Copyright:   Copyright (C) 1999-2019 by The D Language Foundation, All Rights Reserved
 * Authors:     $(LINK2 http://www.digitalmars.com, Walter Bright)
 * License:     $(LINK2 http://www.boost.org/LICENSE_1_0.txt, Boost License 1.0)
 * Source:      $(LINK2 https://github.com/dlang/dmd/blob/master/src/dmd/objc.d, _objc.d)
 * Documentation:  https://dlang.org/phobos/dmd_objc.html
 * Coverage:    https://codecov.io/gh/dlang/dmd/src/master/src/dmd/objc.d
 */

module dmd.objc;

import dmd.aggregate;
import dmd.arraytypes;
import dmd.attrib;
import dmd.cond;
import dmd.dclass;
import dmd.declaration;
import dmd.dmangle;
import dmd.dmodule;
import dmd.dscope;
import dmd.dstruct;
import dmd.dsymbol;
import dmd.dsymbolsem;
import dmd.errors;
import dmd.expression;
import dmd.expressionsem;
import dmd.func;
import dmd.globals;
import dmd.gluelayer;
import dmd.id;
import dmd.identifier;
import dmd.mtype;
import dmd.root.outbuffer;
import dmd.root.stringtable;

struct ObjcSelector
{
    // MARK: Selector
    private __gshared StringTable stringtable;
    private __gshared StringTable vTableDispatchSelectors;
    private __gshared int incnum = 0;
    const(char)* stringvalue;
    size_t stringlen;
    size_t paramCount;

    extern (C++) static void _init()
    {
        stringtable._init();
    }

    extern (D) this(const(char)* sv, size_t len, size_t pcount)
    {
        stringvalue = sv;
        stringlen = len;
        paramCount = pcount;
    }

    extern (D) static ObjcSelector* lookup(const(char)* s)
    {
        size_t len = 0;
        size_t pcount = 0;
        const(char)* i = s;
        while (*i != 0)
        {
            ++len;
            if (*i == ':')
                ++pcount;
            ++i;
        }
        return lookup(s, len, pcount);
    }

    extern (D) static ObjcSelector* lookup(const(char)* s, size_t len, size_t pcount)
    {
        StringValue* sv = stringtable.update(s, len);
        ObjcSelector* sel = cast(ObjcSelector*)sv.ptrvalue;
        if (!sel)
        {
            sel = new ObjcSelector(sv.toDchars(), len, pcount);
            sv.ptrvalue = cast(char*)sel;
        }
        return sel;
    }

    extern (C++) static ObjcSelector* create(FuncDeclaration fdecl)
    {
        OutBuffer buf;
        size_t pcount = 0;
        TypeFunction ftype = cast(TypeFunction)fdecl.type;
        const id = fdecl.ident.toString();
        // Special case: property setter
        if (ftype.isproperty && ftype.parameterList.parameters && ftype.parameterList.parameters.dim == 1)
        {
            // rewrite "identifier" as "setIdentifier"
            char firstChar = id[0];
            if (firstChar >= 'a' && firstChar <= 'z')
                firstChar = cast(char)(firstChar - 'a' + 'A');
            buf.writestring("set");
            buf.writeByte(firstChar);
            buf.write(id.ptr + 1, id.length - 1);
            buf.writeByte(':');
            goto Lcomplete;
        }
        // write identifier in selector
        buf.write(id.ptr, id.length);
        // add mangled type and colon for each parameter
        if (ftype.parameterList.parameters && ftype.parameterList.parameters.dim)
        {
            buf.writeByte('_');
            Parameters* arguments = ftype.parameterList.parameters;
            size_t dim = Parameter.dim(arguments);
            for (size_t i = 0; i < dim; i++)
            {
                Parameter arg = Parameter.getNth(arguments, i);
                mangleToBuffer(arg.type, &buf);
                buf.writeByte(':');
            }
            pcount = dim;
        }
    Lcomplete:
        buf.writeByte('\0');
        return lookup(cast(const(char)*)buf.data, buf.size, pcount);
    }

    extern (D) const(char)[] toString() const pure
    {
        return stringvalue[0 .. stringlen];
    }
}

private __gshared Objc _objc;

Objc objc()
{
    return _objc;
}


/**
 * Contains all data for a class declaration that is needed for the Objective-C
 * integration.
 */
extern (C++) struct ObjcClassDeclaration
{
    /// `true` if this class is a metaclass.
    bool isMeta = false;

    /// `true` if this class is externally defined.
    bool isExtern = false;

    /// Name of this class.
    Identifier identifier;

    /// The class declaration this belongs to.
    ClassDeclaration classDeclaration;

    /// The metaclass of this class.
    ClassDeclaration metaclass;

    /// List of non-inherited methods.
    Dsymbols* methodList;

    extern (D) this(ClassDeclaration classDeclaration)
    {
        this.classDeclaration = classDeclaration;
        methodList = new Dsymbols;
    }

    bool isRootClass() const
    {
        return classDeclaration.classKind == ClassKind.objc &&
            !metaclass &&
            !classDeclaration.baseClass;
    }
}

// Should be an interface
extern(C++) abstract class Objc
{
    static void _init()
    {
        // IN_LLVM: if (global.params.isOSX && global.params.is64bit)
        if (global.params.hasObjectiveC)
            _objc = new Supported;
        else
            _objc = new Unsupported;
    }

    /**
     * Deinitializes the global state of the compiler.
     *
     * This can be used to restore the state set by `_init` to its original
     * state.
     */
    static void deinitialize()
    {
        _objc = _objc.init;
    }

    abstract void setObjc(ClassDeclaration cd);
    abstract void setObjc(InterfaceDeclaration);

    /**
     * Deprecate the given Objective-C interface.
     *
     * Representing an Objective-C class as a D interface has been deprecated.
     * Classes have now been properly implemented and the `class` keyword should
     * be used instead.
     *
     * In the future, `extern(Objective-C)` interfaces will be used to represent
     * Objective-C protocols.
     *
     * Params:
     *  interfaceDeclaration = the interface declaration to deprecate
     */
    abstract void deprecate(InterfaceDeclaration interfaceDeclaration) const;

    abstract void setSelector(FuncDeclaration, Scope* sc);
    abstract void validateSelector(FuncDeclaration fd);
    abstract void checkLinkage(FuncDeclaration fd);

    /**
     * Returns `true` if the given function declaration is virtual.
     *
     * Function declarations with Objective-C linkage and which are static or
     * final are considered virtual.
     *
     * Params:
     *  fd = the function declaration to check if it's virtual
     *
     * Returns: `true` if the given function declaration is virtual
     */
    abstract bool isVirtual(const FuncDeclaration fd) const;

    /**
     * Gets the parent of the given function declaration.
     *
     * Handles Objective-C static member functions, which are virtual functions
     * of the metaclass, by returning the parent class declaration to the
     * metaclass.
     *
     * Params:
     *  fd = the function declaration to get the parent of
     *  cd = the current parent, i.e. the class declaration the given function
     *      declaration belongs to
     *
     * Returns: the parent
     */
    abstract ClassDeclaration getParent(FuncDeclaration fd,
        ClassDeclaration cd) const;

    /**
     * Adds the given function to the list of Objective-C methods.
     *
     * This list will later be used output the necessary Objective-C module info.
     *
     * Params:
     *  fd = the function declaration to be added to the list
     *  cd = the class declaration the function belongs to
     */
    abstract void addToClassMethodList(FuncDeclaration fd,
        ClassDeclaration cd) const;

    /**
     * Returns the `this` pointer of the given function declaration.
     *
     * This is only used for class/static methods. For instance methods, no
     * Objective-C specialization is necessary.
     *
     * Params:
     *  funcDeclaration = the function declaration to get the `this` pointer for
     *
     * Returns: the `this` pointer of the given function declaration, or `null`
     *  if the given function declaration is not an Objective-C method.
     */
    abstract inout(AggregateDeclaration) isThis(inout FuncDeclaration funcDeclaration) const;

    /**
     * Creates the selector parameter for the given function declaration.
     *
     * Objective-C methods has an extra hidden parameter that comes after the
     * `this` parameter. The selector parameter is of the Objective-C type `SEL`
     * and contains the selector which this method was called with.
     *
     * Params:
     *  fd = the function declaration to create the parameter for
     *  sc = the scope from the semantic phase
     *
     * Returns: the newly created selector parameter or `null` for
     *  non-Objective-C functions
     */
    abstract VarDeclaration createSelectorParameter(FuncDeclaration fd, Scope* sc) const;

    /**
     * Creates and sets the metaclass on the given class/interface declaration.
     *
     * Will only be performed on regular Objective-C classes, not on metaclasses.
     *
     * Params:
     *  classDeclaration = the class/interface declaration to set the metaclass on
     */
    abstract void setMetaclass(InterfaceDeclaration interfaceDeclaration, Scope* sc) const;

    /// ditto
    abstract void setMetaclass(ClassDeclaration classDeclaration, Scope* sc) const;

    /**
     * Returns Objective-C runtime metaclass of the given class declaration.
     *
     * `ClassDeclaration.ObjcClassDeclaration.metaclass` contains the metaclass
     * from the semantic point of view. This function returns the metaclass from
     * the Objective-C runtime's point of view. Here, the metaclass of a
     * metaclass is the root metaclass, not `null`. The root metaclass's
     * metaclass is itself.
     *
     * Params:
     *  classDeclaration = The class declaration to return the metaclass of
     *
     * Returns: the Objective-C runtime metaclass of the given class declaration
     */
    abstract ClassDeclaration getRuntimeMetaclass(ClassDeclaration classDeclaration) const;

    ///
    abstract void addSymbols(AttribDeclaration attribDeclaration,
        ClassDeclarations* classes, ClassDeclarations* categories) const;

    ///
    abstract void addSymbols(ClassDeclaration classDeclaration,
        ClassDeclarations* classes, ClassDeclarations* categories) const;

    /**
     * Issues a compile time error if the `.offsetof`/`.tupleof` property is
     * used on a field of an Objective-C class.
     *
     * To solve the fragile base class problem in Objective-C, fields have a
     * dynamic offset instead of a static offset. The compiler outputs a
     * statically known offset which later the dynamic loader can update, if
     * necessary, when the application is loaded. Due to this behavior it
     * doesn't make sense to be able to get the offset of a field at compile
     * time, because this offset might not actually be the same at runtime.
     *
     * To get the offset of a field that is correct at runtime, functionality
     * from the Objective-C runtime can be used instead.
     *
     * Params:
     *  expression = the `.offsetof`/`.tupleof` expression
     *  aggregateDeclaration = the aggregate declaration the field of the
     *      `.offsetof`/`.tupleof` expression belongs to
     *  type = the type of the receiver of the `.tupleof` expression
     *
     * See_Also:
     *  $(LINK2 https://en.wikipedia.org/wiki/Fragile_binary_interface_problem,
     *      Fragile Binary Interface Problem)
     *
     * See_Also:
     *  $(LINK2 https://developer.apple.com/documentation/objectivec/objective_c_runtime,
     *      Objective-C Runtime)
     */
    abstract void checkOffsetof(Expression expression, AggregateDeclaration aggregateDeclaration) const;

    /// ditto
    abstract void checkTupleof(Expression expression, TypeClass type) const;
}

extern(C++) private final class Unsupported : Objc
{
    extern(D) final this()
    {
        version (IN_LLVM) {} else
        ObjcGlue.initialize();
    }

    override void setObjc(ClassDeclaration cd)
    {
        cd.error("Objective-C classes not supported");
    }

    override void setObjc(InterfaceDeclaration id)
    {
        id.error("Objective-C interfaces not supported");
    }

    override void deprecate(InterfaceDeclaration) const
    {
        // noop
    }

    override void setSelector(FuncDeclaration, Scope*)
    {
        // noop
    }

    override void validateSelector(FuncDeclaration)
    {
        // noop
    }

    override void checkLinkage(FuncDeclaration)
    {
        // noop
    }

    override bool isVirtual(const FuncDeclaration) const
    {
        assert(0, "Should never be called when Objective-C is not supported");
    }

    override ClassDeclaration getParent(FuncDeclaration, ClassDeclaration cd) const
    {
        return cd;
    }

    override void addToClassMethodList(FuncDeclaration, ClassDeclaration) const
    {
        // noop
    }

    override inout(AggregateDeclaration) isThis(inout FuncDeclaration funcDeclaration) const
    {
        return null;
    }

    override VarDeclaration createSelectorParameter(FuncDeclaration, Scope*) const
    {
        return null;
    }

    override void setMetaclass(InterfaceDeclaration, Scope*) const
    {
        // noop
    }

    override void setMetaclass(ClassDeclaration, Scope*) const
    {
        // noop
    }

    override ClassDeclaration getRuntimeMetaclass(ClassDeclaration classDeclaration) const
    {
        assert(0, "Should never be called when Objective-C is not supported");
    }

    override void addSymbols(AttribDeclaration attribDeclaration,
        ClassDeclarations* classes, ClassDeclarations* categories) const
    {
        // noop
    }

    override void addSymbols(ClassDeclaration classDeclaration,
        ClassDeclarations* classes, ClassDeclarations* categories) const
    {
        // noop
    }

    override void checkOffsetof(Expression expression, AggregateDeclaration aggregateDeclaration) const
    {
        // noop
    }

    override void checkTupleof(Expression expression, TypeClass type) const
    {
        // noop
    }
}

extern(C++) private final class Supported : Objc
{
    extern(D) final this()
    {
        VersionCondition.addPredefinedGlobalIdent("D_ObjectiveC");

        version (IN_LLVM)
        {
            objc_initSymbols();
        }
        else
        {
            ObjcGlue.initialize();
        }
        ObjcSelector._init();
    }

    override void setObjc(ClassDeclaration cd)
    {
        cd.classKind = ClassKind.objc;
        cd.objc.isExtern = (cd.storage_class & STC.extern_) > 0;
    }

    override void setObjc(InterfaceDeclaration id)
    {
        id.classKind = ClassKind.objc;
        id.objc.isExtern = true;
    }

    override void deprecate(InterfaceDeclaration id) const
    in
    {
        assert(id.classKind == ClassKind.objc);
    }
    body
    {
        // don't report deprecations for the metaclass to avoid duplicated
        // messages.
        if (id.objc.isMeta)
            return;

        id.deprecation("Objective-C interfaces have been deprecated");
        deprecationSupplemental(id.loc, "Representing an Objective-C class " ~
            "as a D interface has been deprecated. Please use "~
            "`extern (Objective-C) extern class` instead");
    }

    override void setSelector(FuncDeclaration fd, Scope* sc)
    {
        import dmd.tokens;

        if (!fd.userAttribDecl)
            return;
        Expressions* udas = fd.userAttribDecl.getAttributes();
        arrayExpressionSemantic(udas, sc, true);
        for (size_t i = 0; i < udas.dim; i++)
        {
            Expression uda = (*udas)[i];
            assert(uda);
            if (uda.op != TOK.tuple)
                continue;
            Expressions* exps = (cast(TupleExp)uda).exps;
            for (size_t j = 0; j < exps.dim; j++)
            {
                Expression e = (*exps)[j];
                assert(e);
                if (e.op != TOK.structLiteral)
                    continue;
                StructLiteralExp literal = cast(StructLiteralExp)e;
                assert(literal.sd);
                if (!isUdaSelector(literal.sd))
                    continue;
                if (fd.selector)
                {
                    fd.error("can only have one Objective-C selector per method");
                    return;
                }
                assert(literal.elements.dim == 1);
                StringExp se = (*literal.elements)[0].toStringExp();
                assert(se);
                fd.selector = ObjcSelector.lookup(cast(const(char)*)se.toUTF8(sc).string);
            }
        }
    }

    override void validateSelector(FuncDeclaration fd)
    {
        if (!fd.selector)
            return;
        TypeFunction tf = cast(TypeFunction)fd.type;
        if (fd.selector.paramCount != tf.parameterList.parameters.dim)
            fd.error("number of colons in Objective-C selector must match number of parameters");
        if (fd.parent && fd.parent.isTemplateInstance())
            fd.error("template cannot have an Objective-C selector attached");
    }

    override void checkLinkage(FuncDeclaration fd)
    {
        if (fd.linkage != LINK.objc && fd.selector)
            fd.error("must have Objective-C linkage to attach a selector");
    }

    override bool isVirtual(const FuncDeclaration fd) const
    in
    {
        assert(fd.selector);
        assert(fd.isMember);
    }
    body
    {
        // * final member functions are kept virtual with Objective-C linkage
        //   because the Objective-C runtime always use dynamic dispatch.
        // * static member functions are kept virtual too, as they represent
        //   methods of the metaclass.
        with (fd.protection)
            return !(kind == Prot.Kind.private_ || kind == Prot.Kind.package_);
    }

    override ClassDeclaration getParent(FuncDeclaration fd, ClassDeclaration cd) const
    out(metaclass)
    {
        assert(metaclass);
    }
    body
    {
        if (cd.classKind == ClassKind.objc && fd.isStatic && !cd.objc.isMeta)
            return cd.objc.metaclass;
        else
            return cd;
    }

    override void addToClassMethodList(FuncDeclaration fd, ClassDeclaration cd) const
    in
    {
        assert(fd.parent.isClassDeclaration);
    }
    body
    {
        if (cd.classKind != ClassKind.objc)
            return;

        if (!fd.selector)
            return;

        assert(fd.isStatic ? cd.objc.isMeta : !cd.objc.isMeta);

        cd.objc.methodList.push(fd);
    }

    override inout(AggregateDeclaration) isThis(inout FuncDeclaration funcDeclaration) const
    {
        with(funcDeclaration)
        {
            if (!selector)
                return null;

            // Use Objective-C class object as 'this'
            auto cd = isMember2().isClassDeclaration();

            if (cd.classKind == ClassKind.objc)
            {
                if (!cd.objc.isMeta)
                    return cd.objc.metaclass;
            }

            return null;
        }
    }

    override VarDeclaration createSelectorParameter(FuncDeclaration fd, Scope* sc) const
    in
    {
        assert(fd.selectorParameter is null);
    }
    body
    {
        if (!fd.selector)
            return null;

        auto var = new VarDeclaration(fd.loc, Type.tvoidptr, Identifier.anonymous, null);
        var.storage_class |= STC.parameter;
        var.dsymbolSemantic(sc);
        if (!sc.insert(var))
            assert(false);
        var.parent = fd;

        return var;
    }

    override void setMetaclass(InterfaceDeclaration interfaceDeclaration, Scope* sc) const
    {
        static auto newMetaclass(Loc loc, BaseClasses* metaBases)
        {
            return new InterfaceDeclaration(loc, null, metaBases);
        }

        .setMetaclass!newMetaclass(interfaceDeclaration, sc);
    }

    override void setMetaclass(ClassDeclaration classDeclaration, Scope* sc) const
    {
        auto newMetaclass(Loc loc, BaseClasses* metaBases)
        {
            return new ClassDeclaration(loc, null, metaBases, new Dsymbols(), 0);
        }

        .setMetaclass!newMetaclass(classDeclaration, sc);
    }

    override ClassDeclaration getRuntimeMetaclass(ClassDeclaration classDeclaration) const
    {
        if (!classDeclaration.objc.metaclass && classDeclaration.objc.isMeta)
        {
            if (classDeclaration.baseClass)
                return getRuntimeMetaclass(classDeclaration.baseClass);
            else
                return classDeclaration;
        }
        else
            return classDeclaration.objc.metaclass;
    }

    override void addSymbols(AttribDeclaration attribDeclaration,
        ClassDeclarations* classes, ClassDeclarations* categories) const
    {
        auto symbols = attribDeclaration.include(null);

        if (!symbols)
            return;

        foreach (symbol; *symbols)
            symbol.addObjcSymbols(classes, categories);
    }

    override void addSymbols(ClassDeclaration classDeclaration,
        ClassDeclarations* classes, ClassDeclarations* categories) const
    {
        with (classDeclaration)
            if (classKind == ClassKind.objc && !objc.isExtern && !objc.isMeta)
                classes.push(classDeclaration);
    }

    override void checkOffsetof(Expression expression, AggregateDeclaration aggregateDeclaration) const
    {
        if (aggregateDeclaration.classKind != ClassKind.objc)
            return;

        enum errorMessage = "no property `offsetof` for member `%s` of type " ~
            "`%s`";

        enum supplementalMessage = "`offsetof` is not available for members " ~
            "of Objective-C classes. Please use the Objective-C runtime instead";

        expression.error(errorMessage, expression.toChars(),
            expression.type.toChars());
        expression.errorSupplemental(supplementalMessage);
    }

    override void checkTupleof(Expression expression, TypeClass type) const
    {
        if (type.sym.classKind != ClassKind.objc)
            return;

        expression.error("no property `tupleof` for type `%s`", type.toChars());
        expression.errorSupplemental("`tupleof` is not available for members " ~
            "of Objective-C classes. Please use the Objective-C runtime instead");
    }

    extern(D) private bool isUdaSelector(StructDeclaration sd)
    {
        if (sd.ident != Id.udaSelector || !sd.parent)
            return false;
        Module _module = sd.parent.isModule();
        return _module && _module.isCoreModule(Id.attribute);
    }
}

/*
 * Creates and sets the metaclass on the given class/interface declaration.
 *
 * Will only be performed on regular Objective-C classes, not on metaclasses.
 *
 * Params:
 *  newMetaclass = a function that returns the metaclass to set. This should
 *      return the same type as `T`.
 *  classDeclaration = the class/interface declaration to set the metaclass on
 */
private void setMetaclass(alias newMetaclass, T)(T classDeclaration, Scope* sc)
if (is(T == ClassDeclaration) || is(T == InterfaceDeclaration))
{
    static if (is(T == ClassDeclaration))
        enum errorType = "class";
    else
        enum errorType = "interface";

    with (classDeclaration)
    {
        if (classKind != ClassKind.objc || objc.isMeta || objc.metaclass)
            return;

        if (!objc.identifier)
            objc.identifier = classDeclaration.ident;

        auto metaBases = new BaseClasses();

        foreach (base ; baseclasses.opSlice)
        {
            auto baseCd = base.sym;
            assert(baseCd);

            if (baseCd.classKind == ClassKind.objc)
            {
                assert(baseCd.objc.metaclass);
                assert(baseCd.objc.metaclass.objc.isMeta);
                assert(baseCd.objc.metaclass.type.ty == Tclass);

                auto metaBase = new BaseClass(baseCd.objc.metaclass.type);
                metaBase.sym = baseCd.objc.metaclass;
                metaBases.push(metaBase);
            }
            else
            {
                error("base " ~ errorType ~ " for an Objective-C " ~
                      errorType ~ " must be `extern (Objective-C)`");
            }
        }

        objc.metaclass = newMetaclass(loc, metaBases);
        objc.metaclass.storage_class |= STC.static_;
        objc.metaclass.classKind = ClassKind.objc;
        objc.metaclass.objc.isMeta = true;
        objc.metaclass.objc.isExtern = objc.isExtern;
        objc.metaclass.objc.identifier = objc.identifier;

        if (baseClass)
            objc.metaclass.baseClass = baseClass.objc.metaclass;

        members.push(objc.metaclass);
        objc.metaclass.addMember(sc, classDeclaration);

        objc.metaclass.dsymbolSemantic(sc);
    }
}
