
/* Compiler implementation of the D programming language
 * Copyright (c) 1999-2014 by Digital Mars
 * All Rights Reserved
 * written by Walter Bright
 * http://www.digitalmars.com
 * Distributed under the Boost Software License, Version 1.0.
 * http://www.boost.org/LICENSE_1_0.txt
 * https://github.com/D-Programming-Language/dmd/blob/master/src/module.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "mars.h"
#include "module.h"
#include "parse.h"
#include "scope.h"
#include "identifier.h"
#include "id.h"
#include "import.h"
#include "dsymbol.h"
#include "expression.h"
#include "lexer.h"
#include "attrib.h"
#include "target.h"

AggregateDeclaration *Module::moduleinfo;

Module *Module::rootModule;
DsymbolTable *Module::modules;
Modules Module::amodules;

Dsymbols Module::deferred; // deferred Dsymbol's needing semantic() run on them
Dsymbols Module::deferred3;
unsigned Module::dprogress;

const char *lookForSourceFile(const char *filename);

void Module::init()
{
    modules = new DsymbolTable();
}

Module::Module(const char *filename, Identifier *ident, int doDocComment, int doHdrGen)
        : Package(ident)
{
    const char *srcfilename;

//    printf("Module::Module(filename = '%s', ident = '%s')\n", filename, ident->toChars());
    this->arg = filename;
    md = NULL;
    errors = 0;
    numlines = 0;
    members = NULL;
    isDocFile = 0;
    isPackageFile = false;
    needmoduleinfo = 0;
    selfimports = 0;
    rootimports = 0;
    insearch = 0;
    searchCacheIdent = NULL;
    searchCacheSymbol = NULL;
    searchCacheFlags = 0;
    decldefs = NULL;
#if IN_DMD
    massert = NULL;
    munittest = NULL;
    marray = NULL;
    sictor = NULL;
    sctor = NULL;
    sdtor = NULL;
    ssharedctor = NULL;
    sshareddtor = NULL;
    stest = NULL;
    sfilename = NULL;
#endif
    importedFrom = NULL;
    srcfile = NULL;
    objfile = NULL;
    docfile = NULL;
    hdrfile = NULL;

    debuglevel = 0;
    debugids = NULL;
    debugidsNot = NULL;
    versionlevel = 0;
    versionids = NULL;
    versionidsNot = NULL;

    macrotable = NULL;
    escapetable = NULL;
#if IN_DMD
    doppelganger = 0;
    cov = NULL;
    covb = NULL;
#endif

    nameoffset = 0;
    namelen = 0;

    srcfilename = FileName::defaultExt(filename, global.mars_ext);

    if (global.run_noext && global.params.run &&
        !FileName::ext(filename) &&
        FileName::exists(srcfilename) == 0 &&
        FileName::exists(filename) == 1)
    {
        FileName::free(srcfilename);
        srcfilename = FileName::removeExt(filename);    // just does a mem.strdup(filename)
    }
    else if (!FileName::equalsExt(srcfilename, global.mars_ext) &&
        !FileName::equalsExt(srcfilename, global.hdr_ext) &&
        !FileName::equalsExt(srcfilename, "dd"))
    {
        error("source file name '%s' must have .%s extension", srcfilename, global.mars_ext);
        fatal();
    }
    srcfile = new File(srcfilename);

#if IN_DMD
    objfile = setOutfile(global.params.objname, global.params.objdir, filename, global.obj_ext);

    if (doDocComment)
        setDocfile();

    if (doHdrGen)
        hdrfile = setOutfile(global.params.hdrname, global.params.hdrdir, arg, global.hdr_ext);

    //objfile = new File(objfilename);
#endif
#if IN_LLVM
    // LDC
    llvmForceLogging = false;
    noModuleInfo = false;
    this->doDocComment = doDocComment;
    this->doHdrGen = doHdrGen;
    this->arrayfuncs = 0;
    d_cover_valid = NULL;
    d_cover_data = NULL;
#endif
}

Module *Module::create(const char *filename, Identifier *ident, int doDocComment, int doHdrGen)
{
    return new Module(filename, ident, doDocComment, doHdrGen);
}

#if IN_DMD
void Module::setDocfile()
{
    docfile = setOutfile(global.params.docname, global.params.docdir, arg, global.doc_ext);
}

/*********************************************
 * Combines things into output file name for .html and .di files.
 * Input:
 *      name    Command line name given for the file, NULL if none
 *      dir     Command line directory given for the file, NULL if none
 *      arg     Name of the source file
 *      ext     File name extension to use if 'name' is NULL
 *      global.params.preservePaths     get output path from arg
 *      srcfile Input file - output file name must not match input file
 */

File *Module::setOutfile(const char *name, const char *dir, const char *arg, const char *ext)
{
    const char *docfilename;

    if (name)
    {
        docfilename = name;
    }
    else
    {
        const char *argdoc;
        if (global.params.preservePaths)
            argdoc = arg;
        else
            argdoc = FileName::name(arg);

        // If argdoc doesn't have an absolute path, make it relative to dir
        if (!FileName::absolute(argdoc))
        {   //FileName::ensurePathExists(dir);
            argdoc = FileName::combine(dir, argdoc);
        }
        docfilename = FileName::forceExt(argdoc, ext);
    }

    if (FileName::equals(docfilename, srcfile->name->str))
    {
        error("source file and output file have same name '%s'", srcfile->name->str);
        fatal();
    }

    return new File(docfilename);
}
#endif

void Module::deleteObjFile()
{
    if (global.params.obj)
        objfile->remove();
    //if (global.params.llvmBC)
    //bcfile->remove();
    if (doDocComment && docfile)
        docfile->remove();
}

const char *Module::kind()
{
    return "module";
}

Module *Module::load(Loc loc, Identifiers *packages, Identifier *ident)
{
    //printf("Module::load(ident = '%s')\n", ident->toChars());

    // Build module filename by turning:
    //  foo.bar.baz
    // into:
    //  foo\bar\baz
    char *filename = ident->toChars();
    if (packages && packages->dim)
    {
        OutBuffer buf;

        for (size_t i = 0; i < packages->dim; i++)
        {   Identifier *pid = (*packages)[i];

            buf.writestring(pid->toChars());
#if _WIN32
            buf.writeByte('\\');
#else
            buf.writeByte('/');
#endif
        }
        buf.writestring(filename);
        buf.writeByte(0);
        filename = (char *)buf.extractData();
    }

    Module *m = new Module(filename, ident, 0, 0);
    m->loc = loc;

    /* Look for the source file
     */
    const char *result = lookForSourceFile(filename);
    if (result)
        m->srcfile = new File(result);

    if (!m->read(loc))
        return NULL;

    if (global.params.verbose)
    {
        fprintf(global.stdmsg, "import    ");
        if (packages)
        {
            for (size_t i = 0; i < packages->dim; i++)
            {
                Identifier *pid = (*packages)[i];
                fprintf(global.stdmsg, "%s.", pid->toChars());
            }
        }
        fprintf(global.stdmsg, "%s\t(%s)\n", ident->toChars(), m->srcfile->toChars());
    }

    m = m->parse();

    Target::loadModule(m);

    return m;
}

bool Module::read(Loc loc)
{
    //printf("Module::read('%s') file '%s'\n", toChars(), srcfile->toChars());
    if (srcfile->read())
    {
        if (!strcmp(srcfile->toChars(), "object.d"))
        {
            ::error(loc, "cannot find source code for runtime library file 'object.d'");
#if IN_LLVM
            errorSupplemental(loc, "ldc2 might not be correctly installed.");
            errorSupplemental(loc, "Please check your ldc2.conf configuration file.");
            errorSupplemental(loc, "Installation instructions can be found at http://wiki.dlang.org/LDC.");
#else
            errorSupplemental(loc, "config file: %s", FileName::canonicalName(global.inifilename));
#endif
        }
        else
        {
            // if module is not named 'package' but we're trying to read 'package.d', we're looking for a package module
            bool isPackageMod = (strcmp(toChars(), "package") != 0) &&
                                (strcmp(srcfile->name->name(), "package.d") == 0);

            if (isPackageMod)
                ::error(loc, "importing package '%s' requires a 'package.d' file which cannot be found in '%s'",
                    toChars(), srcfile->toChars());
            else
                error(loc, "is in file '%s' which cannot be read", srcfile->toChars());
        }

        if (!global.gag)
        {
            /* Print path
             */
            if (global.path)
            {
                for (size_t i = 0; i < global.path->dim; i++)
                {
                    const char *p = (*global.path)[i];
                    fprintf(stderr, "import path[%llu] = %s\n", (ulonglong)i, p);
                }
            }
            else
                fprintf(stderr, "Specify path to file '%s' with -I switch\n", srcfile->toChars());
            fatal();
        }
        return false;
    }
    return true;
}

Module *Module::parse(bool gen_docs)
{
    //printf("Module::parse(srcfile='%s') this=%p\n", srcfile->name->toChars(), this);

    char *srcname = srcfile->name->toChars();
    //printf("Module::parse(srcname = '%s')\n", srcname);

    isPackageFile = (strcmp(srcfile->name->name(), "package.d") == 0);

    utf8_t *buf = (utf8_t *)srcfile->buffer;
    size_t buflen = srcfile->len;

    if (buflen >= 2)
    {
        /* Convert all non-UTF-8 formats to UTF-8.
         * BOM : http://www.unicode.org/faq/utf_bom.html
         * 00 00 FE FF  UTF-32BE, big-endian
         * FF FE 00 00  UTF-32LE, little-endian
         * FE FF        UTF-16BE, big-endian
         * FF FE        UTF-16LE, little-endian
         * EF BB BF     UTF-8
         */

        unsigned le;
        unsigned bom = 1;                // assume there's a BOM
        if (buf[0] == 0xFF && buf[1] == 0xFE)
        {
            if (buflen >= 4 && buf[2] == 0 && buf[3] == 0)
            {   // UTF-32LE
                le = 1;

            Lutf32:
                OutBuffer dbuf;
                unsigned *pu = (unsigned *)(buf);
                unsigned *pumax = &pu[buflen / 4];

                if (buflen & 3)
                {   error("odd length of UTF-32 char source %u", buflen);
                    fatal();
                }

                dbuf.reserve(buflen / 4);
                for (pu += bom; pu < pumax; pu++)
                {   unsigned u;

                    u = le ? Port::readlongLE(pu) : Port::readlongBE(pu);
                    if (u & ~0x7F)
                    {
                        if (u > 0x10FFFF)
                        {   error("UTF-32 value %08x greater than 0x10FFFF", u);
                            fatal();
                        }
                        dbuf.writeUTF8(u);
                    }
                    else
                        dbuf.writeByte(u);
                }
                dbuf.writeByte(0);              // add 0 as sentinel for scanner
                buflen = dbuf.offset - 1;       // don't include sentinel in count
                buf = (utf8_t *) dbuf.extractData();
            }
            else
            {   // UTF-16LE (X86)
                // Convert it to UTF-8
                le = 1;

            Lutf16:
                OutBuffer dbuf;
                unsigned short *pu = (unsigned short *)(buf);
                unsigned short *pumax = &pu[buflen / 2];

                if (buflen & 1)
                {   error("odd length of UTF-16 char source %u", buflen);
                    fatal();
                }

                dbuf.reserve(buflen / 2);
                for (pu += bom; pu < pumax; pu++)
                {   unsigned u;

                    u = le ? Port::readwordLE(pu) : Port::readwordBE(pu);
                    if (u & ~0x7F)
                    {   if (u >= 0xD800 && u <= 0xDBFF)
                        {   unsigned u2;

                            if (++pu > pumax)
                            {   error("surrogate UTF-16 high value %04x at EOF", u);
                                fatal();
                            }
                            u2 = le ? Port::readwordLE(pu) : Port::readwordBE(pu);
                            if (u2 < 0xDC00 || u2 > 0xDFFF)
                            {   error("surrogate UTF-16 low value %04x out of range", u2);
                                fatal();
                            }
                            u = (u - 0xD7C0) << 10;
                            u |= (u2 - 0xDC00);
                        }
                        else if (u >= 0xDC00 && u <= 0xDFFF)
                        {   error("unpaired surrogate UTF-16 value %04x", u);
                            fatal();
                        }
                        else if (u == 0xFFFE || u == 0xFFFF)
                        {   error("illegal UTF-16 value %04x", u);
                            fatal();
                        }
                        dbuf.writeUTF8(u);
                    }
                    else
                        dbuf.writeByte(u);
                }
                dbuf.writeByte(0);              // add 0 as sentinel for scanner
                buflen = dbuf.offset - 1;       // don't include sentinel in count
                buf = (utf8_t *) dbuf.extractData();
            }
        }
        else if (buf[0] == 0xFE && buf[1] == 0xFF)
        {   // UTF-16BE
            le = 0;
            goto Lutf16;
        }
        else if (buflen >= 4 && buf[0] == 0 && buf[1] == 0 && buf[2] == 0xFE && buf[3] == 0xFF)
        {   // UTF-32BE
            le = 0;
            goto Lutf32;
        }
        else if (buflen >= 3 && buf[0] == 0xEF && buf[1] == 0xBB && buf[2] == 0xBF)
        {   // UTF-8

            buf += 3;
            buflen -= 3;
        }
        else
        {
            /* There is no BOM. Make use of Arcane Jill's insight that
             * the first char of D source must be ASCII to
             * figure out the encoding.
             */

            bom = 0;
            if (buflen >= 4)
            {   if (buf[1] == 0 && buf[2] == 0 && buf[3] == 0)
                {   // UTF-32LE
                    le = 1;
                    goto Lutf32;
                }
                else if (buf[0] == 0 && buf[1] == 0 && buf[2] == 0)
                {   // UTF-32BE
                    le = 0;
                    goto Lutf32;
                }
            }
            if (buflen >= 2)
            {
                if (buf[1] == 0)
                {   // UTF-16LE
                    le = 1;
                    goto Lutf16;
                }
                else if (buf[0] == 0)
                {   // UTF-16BE
                    le = 0;
                    goto Lutf16;
                }
            }

            // It's UTF-8
            if (buf[0] >= 0x80)
            {   error("source file must start with BOM or ASCII character, not \\x%02X", buf[0]);
                fatal();
            }
        }
    }

    /* If it starts with the string "Ddoc", then it's a documentation
     * source file.
     */
    if (buflen >= 4 && memcmp(buf, "Ddoc", 4) == 0)
    {
        comment = buf + 4;
        isDocFile = 1;
#if IN_LLVM
        doDocComment = true;
#else
        if (!docfile)
            setDocfile();
#endif
        return this;
    }
    {
#if IN_LLVM
        Parser p(this, buf, buflen, gen_docs);
#else
        Parser p(this, buf, buflen, docfile != NULL);
#endif
        p.nextToken();
        members = p.parseModule();
        md = p.md;
        numlines = p.scanloc.linnum;
        if (p.errors)
            ++global.errors;
    }

    if (srcfile->ref == 0)
        ::free(srcfile->buffer);
    srcfile->buffer = NULL;
    srcfile->len = 0;

    /* The symbol table into which the module is to be inserted.
     */
    DsymbolTable *dst;

    if (md)
    {
        /* A ModuleDeclaration, md, was provided.
         * The ModuleDeclaration sets the packages this module appears in, and
         * the name of this module.
         */
        this->ident = md->id;
        Package *ppack = NULL;
        dst = Package::resolve(md->packages, &this->parent, &ppack);
        assert(dst);

        Module *m = ppack ? ppack->isModule() : NULL;
        if (m && strcmp(m->srcfile->name->name(), "package.d") != 0)
        {
            ::error(md->loc, "package name '%s' conflicts with usage as a module name in file %s",
                ppack->toPrettyChars(), m->srcfile->toChars());
        }
    }
    else
    {
        /* The name of the module is set to the source file name.
         * There are no packages.
         */
        dst = modules;          // and so this module goes into global module symbol table

        /* Check to see if module name is a valid identifier
         */
        if (!Lexer::isValidIdentifier(this->ident->toChars()))
            error("has non-identifier characters in filename, use module declaration instead");
    }

    // Add internal used functions in 'object' module members.
    if (!parent && ident == Id::object)
    {
        static const utf8_t code_ArrayEq[] =
            "bool _ArrayEq(T1, T2)(T1[] a, T2[] b) {\n"
            " if (a.length != b.length) return false;\n"
            " foreach (size_t i; 0 .. a.length) { if (a[i] != b[i]) return false; }\n"
            " return true; }\n";

        static const utf8_t code_ArrayPostblit[] =
            "void _ArrayPostblit(T)(T[] a) { foreach (ref T e; a) e.__xpostblit(); }\n";

        static const utf8_t code_ArrayDtor[] =
            "void _ArrayDtor(T)(T[] a) { foreach_reverse (ref T e; a) e.__xdtor(); }\n";

        static const utf8_t code_xopEquals[] =
            "bool _xopEquals(in void*, in void*) { throw new Error(\"TypeInfo.equals is not implemented\"); }\n";

        static const utf8_t code_xopCmp[] =
            "bool _xopCmp(in void*, in void*) { throw new Error(\"TypeInfo.compare is not implemented\"); }\n";

        Identifier *arreq = Id::_ArrayEq;
        Identifier *xopeq = Identifier::idPool("_xopEquals");
        Identifier *xopcmp = Identifier::idPool("_xopCmp");
        for (size_t i = 0; i < members->dim; i++)
        {
            Dsymbol *sx = (*members)[i];
            if (!sx) continue;
            if (arreq && sx->ident == arreq) arreq = NULL;
            if (xopeq && sx->ident == xopeq) xopeq = NULL;
            if (xopcmp && sx->ident == xopcmp) xopcmp = NULL;
        }

        if (arreq)
        {
            Parser p(loc, this, code_ArrayEq, strlen((const char *)code_ArrayEq), 0);
            p.nextToken();
            members->append(p.parseDeclDefs(0));
        }
        {
            Parser p(loc, this, code_ArrayPostblit, strlen((const char *)code_ArrayPostblit), 0);
            p.nextToken();
            members->append(p.parseDeclDefs(0));
        }
        {
            Parser p(loc, this, code_ArrayDtor, strlen((const char *)code_ArrayDtor), 0);
            p.nextToken();
            members->append(p.parseDeclDefs(0));
        }
        if (xopeq)
        {
            Parser p(loc, this, code_xopEquals, strlen((const char *)code_xopEquals), 0);
            p.nextToken();
            members->append(p.parseDeclDefs(0));
        }
        if (xopcmp)
        {
            Parser p(loc, this, code_xopCmp, strlen((const char *)code_xopCmp), 0);
            p.nextToken();
            members->append(p.parseDeclDefs(0));
        }
    }

    // Insert module into the symbol table
    Dsymbol *s = this;
    if (isPackageFile)
    {
        /* If the source tree is as follows:
         *     pkg/
         *     +- package.d
         *     +- common.d
         * the 'pkg' will be incorporated to the internal package tree in two ways:
         *     import pkg;
         * and:
         *     import pkg.common;
         *
         * If both are used in one compilation, 'pkg' as a module (== pkg/package.d)
         * and a package name 'pkg' will conflict each other.
         *
         * To avoid the conflict:
         * 1. If preceding package name insertion had occurred by Package::resolve,
         *    later package.d loading will change Package::isPkgMod to PKGmodule and set Package::mod.
         * 2. Otherwise, 'package.d' wrapped by 'Package' is inserted to the internal tree in here.
         */
        Package *p = new Package(ident);
        p->parent = this->parent;
        p->isPkgMod = PKGmodule;
        p->mod = this;
        p->symtab = new DsymbolTable();
        s = p;
    }
    if (!dst->insert(s))
    {
        /* It conflicts with a name that is already in the symbol table.
         * Figure out what went wrong, and issue error message.
         */
        Dsymbol *prev = dst->lookup(ident);
        assert(prev);
        if (Module *mprev = prev->isModule())
        {
            if (FileName::compare(srcname, mprev->srcfile->toChars()) != 0)
                error(loc, "from file %s conflicts with another module %s from file %s",
                    srcname, mprev->toChars(), mprev->srcfile->toChars());
            else if (isRoot() && mprev->isRoot())
                error(loc, "from file %s is specified twice on the command line",
                    srcname);
            else
                error(loc, "from file %s must be imported with 'import %s;'",
                    srcname, toPrettyChars());

            // Bugzilla 14446: Return previously parsed module to avoid AST duplication ICE.
            return mprev;
        }
        else if (Package *pkg = prev->isPackage())
        {
            if (pkg->isPkgMod == PKGunknown && isPackageFile)
            {
                /* If the previous inserted Package is not yet determined as package.d,
                 * link it to the actual module.
                 */
                pkg->isPkgMod = PKGmodule;
                pkg->mod = this;
            }
            else
                error(md ? md->loc : loc, "from file %s conflicts with package name %s",
                    srcname, pkg->toChars());
        }
        else
            assert(global.errors);
    }
    else
    {
        // Add to global array of all modules
        amodules.push(this);
    }
    return this;
}

void Module::importAll(Scope *prevsc)
{
    //printf("+Module::importAll(this = %p, '%s'): parent = %p\n", this, toChars(), parent);

    if (scope)
        return;                 // already done

    if (isDocFile)
    {
        error("is a Ddoc file, cannot import it");
        return;
    }

    if (md && md->msg)
    {
        if (StringExp *se = md->msg->toStringExp())
            md->msg = se;
        else
            md->msg->error("string expected, not '%s'", md->msg->toChars());
    }

    /* Note that modules get their own scope, from scratch.
     * This is so regardless of where in the syntax a module
     * gets imported, it is unaffected by context.
     * Ignore prevsc.
     */
    Scope *sc = Scope::createGlobal(this);      // create root scope

    // Add import of "object", even for the "object" module.
    // If it isn't there, some compiler rewrites, like
    //    classinst == classinst -> .object.opEquals(classinst, classinst)
    // would fail inside object.d.
    if (members->dim == 0 || ((*members)[0])->ident != Id::object)
    {
        Import *im = new Import(Loc(), NULL, Id::object, NULL, 0);
        members->shift(im);
    }

    if (!symtab)
    {
        // Add all symbols into module's symbol table
        symtab = new DsymbolTable();
        for (size_t i = 0; i < members->dim; i++)
        {
            Dsymbol *s = (*members)[i];
            s->addMember(sc, sc->scopesym);
        }
    }
    // anything else should be run after addMember, so version/debug symbols are defined

    /* Set scope for the symbols so that if we forward reference
     * a symbol, it can possibly be resolved on the spot.
     * If this works out well, it can be extended to all modules
     * before any semantic() on any of them.
     */
    setScope(sc);               // remember module scope for semantic
    for (size_t i = 0; i < members->dim; i++)
    {
        Dsymbol *s = (*members)[i];
        s->setScope(sc);
    }

    for (size_t i = 0; i < members->dim; i++)
    {
        Dsymbol *s = (*members)[i];
        s->importAll(sc);
    }

    sc = sc->pop();
    sc->pop();          // 2 pops because Scope::createGlobal() created 2
}

void Module::semantic()
{
    if (semanticRun != PASSinit)
        return;

    //printf("+Module::semantic(this = %p, '%s'): parent = %p\n", this, toChars(), parent);
    semanticRun = PASSsemantic;

    // Note that modules get their own scope, from scratch.
    // This is so regardless of where in the syntax a module
    // gets imported, it is unaffected by context.
    Scope *sc = scope;                  // see if already got one from importAll()
    if (!sc)
    {
        Scope::createGlobal(this);      // create root scope
    }

    //printf("Module = %p, linkage = %d\n", sc->scopesym, sc->linkage);

    // Pass 1 semantic routines: do public side of the definition
    for (size_t i = 0; i < members->dim; i++)
    {
        Dsymbol *s = (*members)[i];

        //printf("\tModule('%s'): '%s'.semantic()\n", toChars(), s->toChars());
        s->semantic(sc);
        runDeferredSemantic();
    }

    if (userAttribDecl)
    {
        userAttribDecl->semantic(sc);
    }

    if (!scope)
    {
        sc = sc->pop();
        sc->pop();              // 2 pops because Scope::createGlobal() created 2
    }
    semanticRun = PASSsemanticdone;
    //printf("-Module::semantic(this = %p, '%s'): parent = %p\n", this, toChars(), parent);
}

void Module::semantic2()
{
    //printf("Module::semantic2('%s'): parent = %p\n", toChars(), parent);
    if (semanticRun != PASSsemanticdone)       // semantic() not completed yet - could be recursive call
        return;
    semanticRun = PASSsemantic2;

    // Note that modules get their own scope, from scratch.
    // This is so regardless of where in the syntax a module
    // gets imported, it is unaffected by context.
    Scope *sc = Scope::createGlobal(this);      // create root scope
    //printf("Module = %p\n", sc.scopesym);

    // Pass 2 semantic routines: do initializers and function bodies
    for (size_t i = 0; i < members->dim; i++)
    {
        Dsymbol *s = (*members)[i];
        s->semantic2(sc);
    }

    if (userAttribDecl)
    {
        userAttribDecl->semantic2(sc);
    }

    sc = sc->pop();
    sc->pop();
    semanticRun = PASSsemantic2done;
    //printf("-Module::semantic2('%s'): parent = %p\n", toChars(), parent);
}

void Module::semantic3()
{
    //printf("Module::semantic3('%s'): parent = %p\n", toChars(), parent);
    if (semanticRun != PASSsemantic2done)
        return;
    semanticRun = PASSsemantic3;

    // Note that modules get their own scope, from scratch.
    // This is so regardless of where in the syntax a module
    // gets imported, it is unaffected by context.
    Scope *sc = Scope::createGlobal(this);      // create root scope
    //printf("Module = %p\n", sc.scopesym);

    // Pass 3 semantic routines: do initializers and function bodies
    for (size_t i = 0; i < members->dim; i++)
    {
        Dsymbol *s = (*members)[i];
        //printf("Module %s: %s.semantic3()\n", toChars(), s->toChars());
        s->semantic3(sc);
    }

    if (userAttribDecl)
    {
        userAttribDecl->semantic3(sc);
    }

    sc = sc->pop();
    sc->pop();
    semanticRun = PASSsemantic3done;
}

/**********************************
 * Determine if we need to generate an instance of ModuleInfo
 * for this Module.
 */

int Module::needModuleInfo()
{
    //printf("needModuleInfo() %s, %d, %d\n", toChars(), needmoduleinfo, global.params.cov);
    return needmoduleinfo;
}

Dsymbol *Module::search(Loc loc, Identifier *ident, int flags)
{
    /* Since modules can be circularly referenced,
     * need to stop infinite recursive searches.
     * This is done with the cache.
     */

    //printf("%s Module::search('%s', flags = %d) insearch = %d\n", toChars(), ident->toChars(), flags, insearch);
    if (insearch)
        return NULL;
    if (searchCacheIdent == ident && searchCacheFlags == flags)
    {
        //printf("%s Module::search('%s', flags = %d) insearch = %d searchCacheSymbol = %s\n",
        //        toChars(), ident->toChars(), flags, insearch, searchCacheSymbol ? searchCacheSymbol->toChars() : "null");
        return searchCacheSymbol;
    }

    unsigned int errors = global.errors;

    insearch = 1;
    Dsymbol *s = ScopeDsymbol::search(loc, ident, flags);
    insearch = 0;

    if (errors == global.errors)
    {
        // Bugzilla 10752: We can cache the result only when it does not cause
        // access error so the side-effect should be reproduced in later search.
        searchCacheIdent = ident;
        searchCacheSymbol = s;
        searchCacheFlags = flags;
    }
    return s;
}

Dsymbol *Module::symtabInsert(Dsymbol *s)
{
    searchCacheIdent = NULL;       // symbol is inserted, so invalidate cache
    return Package::symtabInsert(s);
}

void Module::clearCache()
{
    for (size_t i = 0; i < amodules.dim; i++)
    {
        Module *m = amodules[i];
        m->searchCacheIdent = NULL;
    }
}

/*******************************************
 * Can't run semantic on s now, try again later.
 */

void Module::addDeferredSemantic(Dsymbol *s)
{
    // Don't add it if it is already there
    for (size_t i = 0; i < deferred.dim; i++)
    {
        Dsymbol *sd = deferred[i];

        if (sd == s)
            return;
    }

    //printf("Module::addDeferredSemantic('%s')\n", s->toChars());
    deferred.push(s);
}


/******************************************
 * Run semantic() on deferred symbols.
 */

void Module::runDeferredSemantic()
{
    if (dprogress == 0)
        return;

    static int nested;
    if (nested)
        return;
    //if (deferred.dim) printf("+Module::runDeferredSemantic(), len = %d\n", deferred.dim);
    nested++;

    size_t len;
    do
    {
        dprogress = 0;
        len = deferred.dim;
        if (!len)
            break;

        Dsymbol **todo;
        Dsymbol **todoalloc = NULL;
        Dsymbol *tmp;
        if (len == 1)
        {
            todo = &tmp;
        }
        else
        {
            todo = (Dsymbol **)malloc(len * sizeof(Dsymbol *));
            assert(todo);
            todoalloc = todo;
        }
        memcpy(todo, deferred.tdata(), len * sizeof(Dsymbol *));
        deferred.setDim(0);

        for (size_t i = 0; i < len; i++)
        {
            Dsymbol *s = todo[i];

            s->semantic(NULL);
            //printf("deferred: %s, parent = %s\n", s->toChars(), s->parent->toChars());
        }
        //printf("\tdeferred.dim = %d, len = %d, dprogress = %d\n", deferred.dim, len, dprogress);
        if (todoalloc)
            free(todoalloc);
    } while (deferred.dim < len || dprogress);  // while making progress
    nested--;
    //printf("-Module::runDeferredSemantic(), len = %d\n", deferred.dim);
}

void Module::addDeferredSemantic3(Dsymbol *s)
{
    // Don't add it if it is already there
    for (size_t i = 0; i < deferred3.dim; i++)
    {
        Dsymbol *sd = deferred3[i];
        if (sd == s)
            return;
    }
    deferred3.push(s);
}

void Module::runDeferredSemantic3()
{
    Dsymbols *a = &Module::deferred3;
    for (size_t i = 0; i < a->dim; i++)
    {
        Dsymbol *s = (*a)[i];
        //printf("[%d] %s semantic3a\n", i, s->toPrettyChars());

        s->semantic3(NULL);

        if (global.errors)
            break;
    }
}

/************************************
 * Recursively look at every module this module imports,
 * return true if it imports m.
 * Can be used to detect circular imports.
 */

int Module::imports(Module *m)
{
    //printf("%s Module::imports(%s)\n", toChars(), m->toChars());
#if 0
    for (size_t i = 0; i < aimports.dim; i++)
    {
        Module *mi = (Module *)aimports.data[i];
        printf("\t[%d] %s\n", i, mi->toChars());
    }
#endif
    for (size_t i = 0; i < aimports.dim; i++)
    {
        Module *mi = aimports[i];
        if (mi == m)
            return true;
        if (!mi->insearch)
        {
            mi->insearch = 1;
            int r = mi->imports(m);
            if (r)
                return r;
        }
    }
    return false;
}

/*************************************
 * Return true if module imports itself.
 */

bool Module::selfImports()
{
    //printf("Module::selfImports() %s\n", toChars());
    if (selfimports == 0)
    {
        for (size_t i = 0; i < amodules.dim; i++)
            amodules[i]->insearch = 0;

        selfimports = imports(this) + 1;

        for (size_t i = 0; i < amodules.dim; i++)
            amodules[i]->insearch = 0;
    }
    return selfimports == 2;
}

/*************************************
 * Return true if module imports root module.
 */

bool Module::rootImports()
{
    //printf("Module::rootImports() %s\n", toChars());
    if (rootimports == 0)
    {
        for (size_t i = 0; i < amodules.dim; i++)
            amodules[i]->insearch = 0;

        rootimports = 1;
        for (size_t i = 0; i < amodules.dim; ++i)
        {
            Module *m = amodules[i];
            if (m->isRoot() && imports(m))
            {
                rootimports = 2;
                break;
            }
        }

        for (size_t i = 0; i < amodules.dim; i++)
            amodules[i]->insearch = 0;
    }
    return rootimports == 2;
}

/* =========================== ModuleDeclaration ===================== */

ModuleDeclaration::ModuleDeclaration(Loc loc, Identifiers *packages, Identifier *id)
{
    this->loc = loc;
    this->packages = packages;
    this->id = id;
    this->isdeprecated = false;
    this->msg = NULL;
}

char *ModuleDeclaration::toChars()
{
    OutBuffer buf;

    if (packages && packages->dim)
    {
        for (size_t i = 0; i < packages->dim; i++)
        {
            Identifier *pid = (*packages)[i];
            buf.writestring(pid->toChars());
            buf.writeByte('.');
        }
    }
    buf.writestring(id->toChars());
    return buf.extractString();
}

/* =========================== Package ===================== */

Package::Package(Identifier *ident)
        : ScopeDsymbol(ident)
{
    this->isPkgMod = PKGunknown;
    this->mod = NULL;
}


const char *Package::kind()
{
    return "package";
}

Module *Package::isPackageMod()
{
    if (isPkgMod == PKGmodule)
    {
        return mod;
    }
    return NULL;
}

/**
 * Checks if pkg is a sub-package of this
 *
 * For example, if this qualifies to 'a1.a2' and pkg - to 'a1.a2.a3',
 * this function returns 'true'. If it is other way around or qualified
 * package paths conflict function returns 'false'.
 *
 * Params:
 *  pkg = possible subpackage
 *
 * Returns:
 *  see description
 */
bool Package::isAncestorPackageOf(Package* pkg)
{
    while (pkg)
    {
        if (this == pkg)
            return true;

        if (!pkg->parent)
            break;

        pkg = pkg->parent->isPackage();
    }

    return false;
}

/****************************************************
 * Input:
 *      packages[]      the pkg1.pkg2 of pkg1.pkg2.mod
 * Returns:
 *      the symbol table that mod should be inserted into
 * Output:
 *      *pparent        the rightmost package, i.e. pkg2, or NULL if no packages
 *      *ppkg           the leftmost package, i.e. pkg1, or NULL if no packages
 */

DsymbolTable *Package::resolve(Identifiers *packages, Dsymbol **pparent, Package **ppkg)
{
    DsymbolTable *dst = Module::modules;
    Dsymbol *parent = NULL;

    //printf("Package::resolve()\n");
    if (ppkg)
        *ppkg = NULL;

    if (packages)
    {
        for (size_t i = 0; i < packages->dim; i++)
        {
            Identifier *pid = (*packages)[i];
            Package *pkg;
            Dsymbol *p = dst->lookup(pid);
            if (!p)
            {
                pkg = new Package(pid);
                dst->insert(pkg);
                pkg->parent = parent;
                pkg->symtab = new DsymbolTable();
            }
            else
            {
                pkg = p->isPackage();
                assert(pkg);
                // It might already be a module, not a package, but that needs
                // to be checked at a higher level, where a nice error message
                // can be generated.
                // dot net needs modules and packages with same name

                // But we still need a symbol table for it
                if (!pkg->symtab)
                    pkg->symtab = new DsymbolTable();
            }
            parent = pkg;
            dst = pkg->symtab;
            if (ppkg && !*ppkg)
                *ppkg = pkg;
            if (pkg->isModule())
            {
                // Return the module so that a nice error message can be generated
                if (ppkg)
                    *ppkg = (Package *)p;
                break;
            }
        }
    }
    if (pparent)
        *pparent = parent;
    return dst;
}

Dsymbol *Package::search(Loc loc, Identifier *ident, int flags)
{
    if (!isModule() && mod)
    {
        // Prefer full package name.
        Dsymbol *s = symtab ? symtab->lookup(ident) : NULL;
        if (s)
            return s;
        //printf("[%s] through pkdmod: %s\n", loc.toChars(), toChars());
        return mod->search(loc, ident, flags);
    }

    return ScopeDsymbol::search(loc, ident, flags);
}

/* ===========================  ===================== */

/********************************************
 * Look for the source file if it's different from filename.
 * Look for .di, .d, directory, and along global.path.
 * Does not open the file.
 * Input:
 *      filename        as supplied by the user
 *      global.path
 * Returns:
 *      NULL if it's not different from filename.
 */

const char *lookForSourceFile(const char *filename)
{

    /* Search along global.path for .di file, then .d file.
     */

    const char *sdi = FileName::forceExt(filename, global.hdr_ext);
    if (FileName::exists(sdi) == 1)
        return sdi;

    const char *sd  = FileName::forceExt(filename, global.mars_ext);
    if (FileName::exists(sd) == 1)
        return sd;

    if (FileName::exists(filename) == 2)
    {
        /* The filename exists and it's a directory.
         * Therefore, the result should be: filename/package.d
         * iff filename/package.d is a file
         */
        const char *n = FileName::combine(filename, "package.d");
        if (FileName::exists(n) == 1)
            return n;
        FileName::free(n);
    }

    if (FileName::absolute(filename))
        return NULL;

    if (!global.path)
        return NULL;

    for (size_t i = 0; i < global.path->dim; i++)
    {
        const char *p = (*global.path)[i];

        const char *n = FileName::combine(p, sdi);
        if (FileName::exists(n) == 1)
            return n;
        FileName::free(n);

        n = FileName::combine(p, sd);
        if (FileName::exists(n) == 1)
            return n;
        FileName::free(n);

        const char *b = FileName::removeExt(filename);
        n = FileName::combine(p, b);
        FileName::free(b);
        if (FileName::exists(n) == 2)
        {
            const char *n2 = FileName::combine(n, "package.d");
            if (FileName::exists(n2) == 1)
                return n2;
            FileName::free(n2);
        }
        FileName::free(n);
    }
    return NULL;
}
