/**
 * Compiler implementation of the
 * $(LINK2 http://www.dlang.org, D programming language).
 *
 * Copyright:   Copyright (c) 1999-2017 by Digital Mars, All Rights Reserved
 * Authors:     $(LINK2 http://www.digitalmars.com, Walter Bright)
 * License:     $(LINK2 http://www.boost.org/LICENSE_1_0.txt, Boost License 1.0)
 * Source:      $(DMDSRC _objc.d)
 */

module ddmd.objc;

import ddmd.arraytypes;
import ddmd.cond;
import ddmd.dclass;
import ddmd.dmangle;
import ddmd.dmodule;
import ddmd.dscope;
import ddmd.dstruct;
import ddmd.expression;
import ddmd.func;
import ddmd.globals;
import ddmd.gluelayer;
import ddmd.id;
import ddmd.identifier;
import ddmd.mtype;
import ddmd.root.outbuffer;
import ddmd.root.stringtable;

struct ObjcSelector
{
    // MARK: Selector
    extern (C++) static __gshared StringTable stringtable;
    extern (C++) static __gshared StringTable vTableDispatchSelectors;
    extern (C++) static __gshared int incnum = 0;
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

    extern (C++) static ObjcSelector* lookup(const(char)* s)
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

    extern (C++) static ObjcSelector* lookup(const(char)* s, size_t len, size_t pcount)
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
        if (ftype.isproperty && ftype.parameters && ftype.parameters.dim == 1)
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
        if (ftype.parameters && ftype.parameters.dim)
        {
            buf.writeByte('_');
            Parameters* arguments = ftype.parameters;
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
}

struct Objc_ClassDeclaration
{
    // true if this is an Objective-C class/interface
    bool objc;

    // MARK: Objc_ClassDeclaration
    extern (C++) bool isInterface()
    {
        return objc;
    }
}

struct Objc_FuncDeclaration
{
    FuncDeclaration fdecl;
    // Objective-C method selector (member function only)
    ObjcSelector* selector;

    extern (D) this(FuncDeclaration fdecl)
    {
        this.fdecl = fdecl;
    }
}

// MARK: semantic
extern (C++) void objc_ClassDeclaration_semantic_PASSinit_LINKobjc(ClassDeclaration cd)
{
    if (global.params.hasObjectiveC)
        cd.objc.objc = true;
    else
        cd.error("Objective-C classes not supported");
}

extern (C++) void objc_InterfaceDeclaration_semantic_objcExtern(InterfaceDeclaration id, Scope* sc)
{
    if (sc.linkage == LINKobjc)
    {
        if (global.params.hasObjectiveC)
            id.objc.objc = true;
        else
            id.error("Objective-C interfaces not supported");
    }
}

// MARK: semantic
extern (C++) void objc_FuncDeclaration_semantic_setSelector(FuncDeclaration fd, Scope* sc)
{
    import ddmd.tokens;

    if (!fd.userAttribDecl)
        return;
    Expressions* udas = fd.userAttribDecl.getAttributes();
    arrayExpressionSemantic(udas, sc, true);
    for (size_t i = 0; i < udas.dim; i++)
    {
        Expression uda = (*udas)[i];
        assert(uda);
        if (uda.op != TOKtuple)
            continue;
        Expressions* exps = (cast(TupleExp)uda).exps;
        for (size_t j = 0; j < exps.dim; j++)
        {
            Expression e = (*exps)[j];
            assert(e);
            if (e.op != TOKstructliteral)
                continue;
            StructLiteralExp literal = cast(StructLiteralExp)e;
            assert(literal.sd);
            if (!objc_isUdaSelector(literal.sd))
                continue;
            if (fd.objc.selector)
            {
                fd.error("can only have one Objective-C selector per method");
                return;
            }
            assert(literal.elements.dim == 1);
            StringExp se = (*literal.elements)[0].toStringExp();
            assert(se);
            fd.objc.selector = ObjcSelector.lookup(cast(const(char)*)se.toUTF8(sc).string);
        }
    }
}

extern (C++) bool objc_isUdaSelector(StructDeclaration sd)
{
    if (sd.ident != Id.udaSelector || !sd.parent)
        return false;
    Module _module = sd.parent.isModule();
    return _module && _module.isCoreModule(Id.attribute);
}

extern (C++) void objc_FuncDeclaration_semantic_validateSelector(FuncDeclaration fd)
{
    if (!fd.objc.selector)
        return;
    TypeFunction tf = cast(TypeFunction)fd.type;
    if (fd.objc.selector.paramCount != tf.parameters.dim)
        fd.error("number of colons in Objective-C selector must match number of parameters");
    if (fd.parent && fd.parent.isTemplateInstance())
        fd.error("template cannot have an Objective-C selector attached");
}

extern (C++) void objc_FuncDeclaration_semantic_checkLinkage(FuncDeclaration fd)
{
    if (fd.linkage != LINKobjc && fd.objc.selector)
        fd.error("must have Objective-C linkage to attach a selector");
}

// MARK: init
extern (C++) void objc_tryMain_dObjc()
{
    if (global.params.isOSX && global.params.is64bit)
    {
        global.params.hasObjectiveC = true;
        VersionCondition.addPredefinedGlobalIdent("D_ObjectiveC");
    }
}

extern (C++) void objc_tryMain_init()
{
    objc_initSymbols();
    ObjcSelector._init();
}
