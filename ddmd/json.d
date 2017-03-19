/**
 * Compiler implementation of the
 * $(LINK2 http://www.dlang.org, D programming language).
 *
 * Copyright:   Copyright (c) 1999-2016 by Digital Mars, All Rights Reserved
 * Authors:     $(LINK2 http://www.digitalmars.com, Walter Bright)
 * License:     $(LINK2 http://www.boost.org/LICENSE_1_0.txt, Boost License 1.0)
 * Source:      $(DMDSRC _json.d)
 */

module ddmd.json;

import core.stdc.stdio;
import core.stdc.string;
import ddmd.aggregate;
import ddmd.arraytypes;
import ddmd.attrib;
import ddmd.dclass;
import ddmd.declaration;
import ddmd.denum;
import ddmd.dimport;
import ddmd.dmodule;
import ddmd.dsymbol;
import ddmd.dtemplate;
import ddmd.expression;
import ddmd.func;
import ddmd.globals;
import ddmd.hdrgen;
import ddmd.id;
import ddmd.identifier;
import ddmd.mtype;
import ddmd.root.outbuffer;
import ddmd.visitor;

extern (C++) final class ToJsonVisitor : Visitor
{
    alias visit = super.visit;
public:
    OutBuffer* buf;
    int indentLevel;
    const(char)* filename;

    extern (D) this(OutBuffer* buf)
    {
        this.buf = buf;
    }

    void indent()
    {
        if (buf.offset >= 1 && buf.data[buf.offset - 1] == '\n')
            for (int i = 0; i < indentLevel; i++)
                buf.writeByte(' ');
    }

    void removeComma()
    {
        if (buf.offset >= 2 && buf.data[buf.offset - 2] == ',' && (buf.data[buf.offset - 1] == '\n' || buf.data[buf.offset - 1] == ' '))
            buf.offset -= 2;
    }

    void comma()
    {
        if (indentLevel > 0)
            buf.writestring(",\n");
    }

    void stringStart()
    {
        buf.writeByte('\"');
    }

    void stringEnd()
    {
        buf.writeByte('\"');
    }

    void stringPart(const(char)* s)
    {
        for (; *s; s++)
        {
            char c = cast(char)*s;
            switch (c)
            {
            case '\n':
                buf.writestring("\\n");
                break;
            case '\r':
                buf.writestring("\\r");
                break;
            case '\t':
                buf.writestring("\\t");
                break;
            case '\"':
                buf.writestring("\\\"");
                break;
            case '\\':
                buf.writestring("\\\\");
                break;
            case '\b':
                buf.writestring("\\b");
                break;
            case '\f':
                buf.writestring("\\f");
                break;
            default:
                if (c < 0x20)
                    buf.printf("\\u%04x", c);
                else
                {
                    // Note that UTF-8 chars pass through here just fine
                    buf.writeByte(c);
                }
                break;
            }
        }
    }

    // Json value functions
    /*********************************
     * Encode string into buf, and wrap it in double quotes.
     */
    void value(const(char)* s)
    {
        stringStart();
        stringPart(s);
        stringEnd();
    }

    void value(int value)
    {
        buf.printf("%d", value);
    }

    void valueBool(bool value)
    {
        buf.writestring(value ? "true" : "false");
    }

    /*********************************
     * Item is an intented value and a comma, for use in arrays
     */
    void item(const(char)* s)
    {
        indent();
        value(s);
        comma();
    }

    void item(int i)
    {
        indent();
        value(i);
        comma();
    }

    void itemBool(bool b)
    {
        indent();
        valueBool(b);
        comma();
    }

    // Json array functions
    void arrayStart()
    {
        indent();
        buf.writestring("[\n");
        indentLevel++;
    }

    void arrayEnd()
    {
        indentLevel--;
        removeComma();
        if (buf.offset >= 2 && buf.data[buf.offset - 2] == '[' && buf.data[buf.offset - 1] == '\n')
            buf.offset -= 1;
        else if (!(buf.offset >= 1 && buf.data[buf.offset - 1] == '['))
        {
            buf.writestring("\n");
            indent();
        }
        buf.writestring("]");
        comma();
    }

    // Json object functions
    void objectStart()
    {
        indent();
        buf.writestring("{\n");
        indentLevel++;
    }

    void objectEnd()
    {
        indentLevel--;
        removeComma();
        if (buf.offset >= 2 && buf.data[buf.offset - 2] == '{' && buf.data[buf.offset - 1] == '\n')
            buf.offset -= 1;
        else
        {
            buf.writestring("\n");
            indent();
        }
        buf.writestring("}");
        comma();
    }

    // Json object property functions
    void propertyStart(const(char)* name)
    {
        indent();
        value(name);
        buf.writestring(" : ");
    }

    void property(const(char)* name, const(char)* s)
    {
        if (s is null)
            return;
        propertyStart(name);
        value(s);
        comma();
    }

    void property(const(char)* name, int i)
    {
        propertyStart(name);
        value(i);
        comma();
    }

    void propertyBool(const(char)* name, bool b)
    {
        propertyStart(name);
        valueBool(b);
        comma();
    }

    void property(const(char)* name, TRUST trust)
    {
        switch (trust)
        {
        case TRUSTdefault:
            // Should not be printed
            //property(name, "default");
            break;
        case TRUSTsystem:
            property(name, "system");
            break;
        case TRUSTtrusted:
            property(name, "trusted");
            break;
        case TRUSTsafe:
            property(name, "safe");
            break;
        default:
            assert(false);
        }
    }

    void property(const(char)* name, PURE purity)
    {
        switch (purity)
        {
        case PUREimpure:
            // Should not be printed
            //property(name, "impure");
            break;
        case PUREweak:
            property(name, "weak");
            break;
        case PUREconst:
            property(name, "const");
            break;
        case PUREstrong:
            property(name, "strong");
            break;
        case PUREfwdref:
            property(name, "fwdref");
            break;
        default:
            assert(false);
        }
    }

    void property(const(char)* name, LINK linkage)
    {
        switch (linkage)
        {
        case LINKdefault:
            // Should not be printed
            //property(name, "default");
            break;
        case LINKd:
            // Should not be printed
            //property(name, "d");
            break;
        case LINKc:
            property(name, "c");
            break;
        case LINKcpp:
            property(name, "cpp");
            break;
        case LINKwindows:
            property(name, "windows");
            break;
        case LINKpascal:
            property(name, "pascal");
            break;
        default:
            assert(false);
        }
    }

    void propertyStorageClass(const(char)* name, StorageClass stc)
    {
        stc &= STCStorageClass;
        if (stc)
        {
            propertyStart(name);
            arrayStart();
            while (stc)
            {
                const(char)* p = stcToChars(stc);
                assert(p);
                item(p);
            }
            arrayEnd();
        }
    }

    void property(const(char)* linename, const(char)* charname, Loc* loc)
    {
        if (loc)
        {
            const(char)* filename = loc.filename;
            if (filename)
            {
                if (!this.filename || strcmp(filename, this.filename))
                {
                    this.filename = filename;
                    property("file", filename);
                }
            }
            if (loc.linnum)
            {
                property(linename, loc.linnum);
                if (loc.charnum)
                    property(charname, loc.charnum);
            }
        }
    }

    void property(const(char)* name, Type type)
    {
        if (type)
        {
            property(name, type.toChars());
        }
    }

    void property(const(char)* name, const(char)* deconame, Type type)
    {
        if (type)
        {
            if (type.deco)
                property(deconame, type.deco);
            else
                property(name, type.toChars());
        }
    }

    void property(const(char)* name, Parameters* parameters)
    {
        if (parameters is null || parameters.dim == 0)
            return;
        propertyStart(name);
        arrayStart();
        if (parameters)
        {
            for (size_t i = 0; i < parameters.dim; i++)
            {
                Parameter p = (*parameters)[i];
                objectStart();
                if (p.ident)
                    property("name", p.ident.toChars());
                property("type", "deco", p.type);
                propertyStorageClass("storageClass", p.storageClass);
                if (p.defaultArg)
                    property("default", p.defaultArg.toChars());
                objectEnd();
            }
        }
        arrayEnd();
    }

    /* ========================================================================== */
    void jsonProperties(Dsymbol s)
    {
        if (s.isModule())
            return;
        if (!s.isTemplateDeclaration()) // TemplateDeclaration::kind() acts weird sometimes
        {
            property("name", s.toChars());
            property("kind", s.kind());
        }
        if (s.prot().kind != PROTpublic) // TODO: How about package(names)?
            property("protection", protectionToChars(s.prot().kind));
        if (EnumMember em = s.isEnumMember())
        {
            if (em.origValue)
                property("value", em.origValue.toChars());
        }
        property("comment", s.comment);
        property("line", "char", &s.loc);
    }

    void jsonProperties(Declaration d)
    {
        jsonProperties(cast(Dsymbol)d);
        propertyStorageClass("storageClass", d.storage_class);
        property("type", "deco", d.type);
        // Emit originalType if it differs from type
        if (d.type != d.originalType && d.originalType)
        {
            const(char)* ostr = d.originalType.toChars();
            if (d.type)
            {
                const(char)* tstr = d.type.toChars();
                if (strcmp(tstr, ostr))
                {
                    //printf("tstr = %s, ostr = %s\n", tstr, ostr);
                    property("originalType", ostr);
                }
            }
            else
                property("originalType", ostr);
        }
    }

    void jsonProperties(TemplateDeclaration td)
    {
        jsonProperties(cast(Dsymbol)td);
        if (td.onemember && td.onemember.isCtorDeclaration())
            property("name", "this"); // __ctor -> this
        else
            property("name", td.ident.toChars()); // Foo(T) -> Foo
    }

    /* ========================================================================== */
    override void visit(Dsymbol s)
    {
    }

    override void visit(Module s)
    {
        objectStart();
        if (s.md)
            property("name", s.md.toChars());
        property("kind", s.kind());
        filename = s.srcfile.toChars();
        property("file", filename);
        property("comment", s.comment);
        propertyStart("members");
        arrayStart();
        for (size_t i = 0; i < s.members.dim; i++)
        {
            (*s.members)[i].accept(this);
        }
        arrayEnd();
        objectEnd();
    }

    override void visit(Import s)
    {
        if (s.id == Id.object)
            return;
        objectStart();
        propertyStart("name");
        stringStart();
        if (s.packages && s.packages.dim)
        {
            for (size_t i = 0; i < s.packages.dim; i++)
            {
                Identifier pid = (*s.packages)[i];
                stringPart(pid.toChars());
                buf.writeByte('.');
            }
        }
        stringPart(s.id.toChars());
        stringEnd();
        comma();
        property("kind", s.kind());
        property("comment", s.comment);
        property("line", "char", &s.loc);
        if (s.prot().kind != PROTpublic)
            property("protection", protectionToChars(s.prot().kind));
        if (s.aliasId)
            property("alias", s.aliasId.toChars());
        bool hasRenamed = false;
        bool hasSelective = false;
        for (size_t i = 0; i < s.aliases.dim; i++)
        {
            // avoid empty "renamed" and "selective" sections
            if (hasRenamed && hasSelective)
                break;
            else if (s.aliases[i])
                hasRenamed = true;
            else
                hasSelective = true;
        }
        if (hasRenamed)
        {
            // import foo : alias1 = target1;
            propertyStart("renamed");
            objectStart();
            for (size_t i = 0; i < s.aliases.dim; i++)
            {
                Identifier name = s.names[i];
                Identifier _alias = s.aliases[i];
                if (_alias)
                    property(_alias.toChars(), name.toChars());
            }
            objectEnd();
        }
        if (hasSelective)
        {
            // import foo : target1;
            propertyStart("selective");
            arrayStart();
            for (size_t i = 0; i < s.names.dim; i++)
            {
                Identifier name = s.names[i];
                if (!s.aliases[i])
                    item(name.toChars());
            }
            arrayEnd();
        }
        objectEnd();
    }

    override void visit(AttribDeclaration d)
    {
        Dsymbols* ds = d.include(null, null);
        if (ds)
        {
            for (size_t i = 0; i < ds.dim; i++)
            {
                Dsymbol s = (*ds)[i];
                s.accept(this);
            }
        }
    }

    override void visit(ConditionalDeclaration d)
    {
        if (d.condition.inc)
        {
            visit(cast(AttribDeclaration)d);
        }
    }

    override void visit(TypeInfoDeclaration d)
    {
    }

    override void visit(PostBlitDeclaration d)
    {
    }

    override void visit(Declaration d)
    {
        objectStart();
        //property("unknown", "declaration");
        jsonProperties(d);
        objectEnd();
    }

    override void visit(AggregateDeclaration d)
    {
        objectStart();
        jsonProperties(d);
        ClassDeclaration cd = d.isClassDeclaration();
        if (cd)
        {
            if (cd.baseClass && cd.baseClass.ident != Id.Object)
            {
                property("base", cd.baseClass.toPrettyChars(true));
            }
            if (cd.interfaces.length)
            {
                propertyStart("interfaces");
                arrayStart();
                foreach (b; cd.interfaces)
                {
                    item(b.sym.toPrettyChars(true));
                }
                arrayEnd();
            }
        }
        if (d.members)
        {
            propertyStart("members");
            arrayStart();
            for (size_t i = 0; i < d.members.dim; i++)
            {
                Dsymbol s = (*d.members)[i];
                s.accept(this);
            }
            arrayEnd();
        }
        objectEnd();
    }

    override void visit(FuncDeclaration d)
    {
        objectStart();
        jsonProperties(d);
        TypeFunction tf = cast(TypeFunction)d.type;
        if (tf && tf.ty == Tfunction)
            property("parameters", tf.parameters);
        property("endline", "endchar", &d.endloc);
        if (d.foverrides.dim)
        {
            propertyStart("overrides");
            arrayStart();
            for (size_t i = 0; i < d.foverrides.dim; i++)
            {
                FuncDeclaration fd = d.foverrides[i];
                item(fd.toPrettyChars());
            }
            arrayEnd();
        }
        if (d.fdrequire)
        {
            propertyStart("in");
            d.fdrequire.accept(this);
        }
        if (d.fdensure)
        {
            propertyStart("out");
            d.fdensure.accept(this);
        }
        objectEnd();
    }

    override void visit(TemplateDeclaration d)
    {
        objectStart();
        // TemplateDeclaration::kind returns the kind of its Aggregate onemember, if it is one
        property("kind", "template");
        jsonProperties(d);
        propertyStart("parameters");
        arrayStart();
        for (size_t i = 0; i < d.parameters.dim; i++)
        {
            TemplateParameter s = (*d.parameters)[i];
            objectStart();
            property("name", s.ident.toChars());
            TemplateTypeParameter type = s.isTemplateTypeParameter();
            if (type)
            {
                if (s.isTemplateThisParameter())
                    property("kind", "this");
                else
                    property("kind", "type");
                property("type", "deco", type.specType);
                property("default", "defaultDeco", type.defaultType);
            }
            TemplateValueParameter value = s.isTemplateValueParameter();
            if (value)
            {
                property("kind", "value");
                property("type", "deco", value.valType);
                if (value.specValue)
                    property("specValue", value.specValue.toChars());
                if (value.defaultValue)
                    property("defaultValue", value.defaultValue.toChars());
            }
            TemplateAliasParameter _alias = s.isTemplateAliasParameter();
            if (_alias)
            {
                property("kind", "alias");
                property("type", "deco", _alias.specType);
                if (_alias.specAlias)
                    property("specAlias", _alias.specAlias.toChars());
                if (_alias.defaultAlias)
                    property("defaultAlias", _alias.defaultAlias.toChars());
            }
            TemplateTupleParameter tuple = s.isTemplateTupleParameter();
            if (tuple)
            {
                property("kind", "tuple");
            }
            objectEnd();
        }
        arrayEnd();
        Expression expression = d.constraint;
        if (expression)
        {
            property("constraint", expression.toChars());
        }
        propertyStart("members");
        arrayStart();
        for (size_t i = 0; i < d.members.dim; i++)
        {
            Dsymbol s = (*d.members)[i];
            s.accept(this);
        }
        arrayEnd();
        objectEnd();
    }

    override void visit(EnumDeclaration d)
    {
        if (d.isAnonymous())
        {
            if (d.members)
            {
                for (size_t i = 0; i < d.members.dim; i++)
                {
                    Dsymbol s = (*d.members)[i];
                    s.accept(this);
                }
            }
            return;
        }
        objectStart();
        jsonProperties(d);
        property("base", "baseDeco", d.memtype);
        if (d.members)
        {
            propertyStart("members");
            arrayStart();
            for (size_t i = 0; i < d.members.dim; i++)
            {
                Dsymbol s = (*d.members)[i];
                s.accept(this);
            }
            arrayEnd();
        }
        objectEnd();
    }

    override void visit(EnumMember s)
    {
        objectStart();
        jsonProperties(cast(Dsymbol)s);
        property("type", "deco", s.origType);
        objectEnd();
    }

    override void visit(VarDeclaration d)
    {
        objectStart();
        jsonProperties(d);
        if (d._init)
            property("init", d._init.toChars());
        if (d.isField())
            property("offset", d.offset);
        if (d.alignment && d.alignment != STRUCTALIGN_DEFAULT)
            property("align", d.alignment);
        objectEnd();
    }

    override void visit(TemplateMixin d)
    {
        objectStart();
        jsonProperties(d);
        objectEnd();
    }
}

extern (C++) void json_generate(OutBuffer* buf, Modules* modules)
{
    scope ToJsonVisitor json = new ToJsonVisitor(buf);
    json.arrayStart();
    for (size_t i = 0; i < modules.dim; i++)
    {
        Module m = (*modules)[i];
        if (global.params.verbose)
            fprintf(global.stdmsg, "json gen %s\n", m.toChars());
        m.accept(json);
    }
    json.arrayEnd();
    json.removeComma();
}
