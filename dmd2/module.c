
// Compiler implementation of the D programming language
// Copyright (c) 1999-2013 by Digital Mars
// All Rights Reserved
// written by Walter Bright
// http://www.digitalmars.com
// License for redistribution is by either the Artistic License
// in artistic.txt, or the GNU General Public License in gnu.txt.
// See the included readme.txt for details.

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
#include "hdrgen.h"
#include "lexer.h"

#ifdef IN_GCC
#include "d-dmd-gcc.h"
#endif

AggregateDeclaration *Module::moduleinfo;

Module *Module::rootModule;
DsymbolTable *Module::modules;
Modules Module::amodules;

Dsymbols Module::deferred; // deferred Dsymbol's needing semantic() run on them
unsigned Module::dprogress;

void Module::init()
{
    modules = new DsymbolTable();
}

Module::Module(char *filename, Identifier *ident, int doDocComment, int doHdrGen)
        : Package(ident)
{
    const char *srcfilename;
#if IN_DMD
    const char *symfilename;
#endif

//    printf("Module::Module(filename = '%s', ident = '%s')\n", filename, ident->toChars());
    this->arg = filename;
    md = NULL;
    errors = 0;
    numlines = 0;
    members = NULL;
    isDocFile = 0;
    needmoduleinfo = 0;
    selfimports = 0;
    insearch = 0;
    searchCacheIdent = NULL;
    searchCacheSymbol = NULL;
    searchCacheFlags = 0;
    semanticstarted = 0;
    semanticRun = 0;
    decldefs = NULL;
    vmoduleinfo = NULL;
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
    root = 0;
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
    safe = FALSE;
#if IN-DMD
    doppelganger = 0;
    cov = NULL;
    covb = NULL;
#endif

    nameoffset = 0;
    namelen = 0;

    srcfilename = FileName::defaultExt(filename, global.mars_ext);
    if (!FileName::equalsExt(srcfilename, global.mars_ext) &&
        !FileName::equalsExt(srcfilename, global.hdr_ext) &&
        !FileName::equalsExt(srcfilename, "dd"))
    {
        error("source file name '%s' must have .%s extension", srcfilename, global.mars_ext);
        fatal();
    }
    srcfile = new File(srcfilename);

#if IN_DMD
    objfile = setOutfile(global.params.objname, global.params.objdir, filename, global.obj_ext);

    symfilename = FileName::forceExt(filename, global.sym_ext);

    if (doDocComment)
        setDocfile();

    if (doHdrGen)
        hdrfile = setOutfile(global.params.hdrname, global.params.hdrdir, arg, global.hdr_ext);

    //objfile = new File(objfilename);
    symfile = new File(symfilename);
#endif
#if IN_LLVM
    // LDC
    llvmForceLogging = false;
    moduleInfoVar = NULL;
    this->doDocComment = doDocComment;
    this->doHdrGen = doHdrGen;
    this->isRoot = false;
    this->arrayfuncs = 0;
#endif
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
    {   error("Source file and output file have same name '%s'", srcfile->name->str);
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

Module::~Module()
{
}

const char *Module::kind()
{
    return "module";
}

Module *Module::load(Loc loc, Identifiers *packages, Identifier *ident)
{   Module *m;
    char *filename;

    //printf("Module::load(ident = '%s')\n", ident->toChars());

    // Build module filename by turning:
    //  foo.bar.baz
    // into:
    //  foo\bar\baz
    filename = ident->toChars();
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

    m = new Module(filename, ident, 0, 0);
    m->loc = loc;

    /* Search along global.path for .di file, then .d file.
     */
    const char *result = NULL;
    const char *fdi = FileName::forceExt(filename, global.hdr_ext);
    const char *fd  = FileName::forceExt(filename, global.mars_ext);
    const char *sdi = fdi;
    const char *sd  = fd;

    if (FileName::exists(sdi))
        result = sdi;
    else if (FileName::exists(sd))
        result = sd;
    else if (FileName::absolute(filename))
        ;
    else if (!global.path)
        ;
    else
    {
        for (size_t i = 0; i < global.path->dim; i++)
        {
            const char *p = (*global.path)[i];
            const char *n = FileName::combine(p, sdi);
            if (FileName::exists(n))
            {   result = n;
                break;
            }
            FileName::free(n);
            n = FileName::combine(p, sd);
            if (FileName::exists(n))
            {   result = n;
                break;
            }
            FileName::free(n);
        }
    }
    if (result)
        m->srcfile = new File(result);

    if (global.params.verbose)
    {
        printf("import    ");
        if (packages)
        {
            for (size_t i = 0; i < packages->dim; i++)
            {   Identifier *pid = (*packages)[i];
                printf("%s.", pid->toChars());
            }
        }
        printf("%s\t(%s)\n", ident->toChars(), m->srcfile->toChars());
    }

    if (!m->read(loc))
        return NULL;

    m->parse();

#ifdef IN_GCC
    d_gcc_magic_module(m);
#endif

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
            errorSupplemental(loc, "dmd might not be correctly installed. Run 'dmd -man' for installation instructions.");
        }
        else
        {
            error(loc, "is in file '%s' which cannot be read", srcfile->toChars());
        }

        if (!global.gag)
        {   /* Print path
             */
            if (global.path)
            {
                for (size_t i = 0; i < global.path->dim; i++)
                {
                    char *p = (*global.path)[i];
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

inline unsigned readwordLE(unsigned short *p)
{
#if IN_LLVM
#if __LITTLE_ENDIAN__
    return *p;
#else
    return (((unsigned char *)p)[1] << 8) | ((unsigned char *)p)[0];
#endif
#else
#if LITTLE_ENDIAN
    return *p;
#else
    return (((unsigned char *)p)[1] << 8) | ((unsigned char *)p)[0];
#endif
#endif
}

inline unsigned readwordBE(unsigned short *p)
{
#if IN_LLVM
#if __BIG_ENDIAN__
    return *p;
#else
    return (((unsigned char *)p)[0] << 8) | ((unsigned char *)p)[1];
#endif
#else
    return (((unsigned char *)p)[0] << 8) | ((unsigned char *)p)[1];
#endif
}

inline unsigned readlongLE(unsigned *p)
{
#if IN_LLVM
#if __LITTLE_ENDIAN__
    return *p;
#else
    return ((unsigned char *)p)[0] |
        (((unsigned char *)p)[1] << 8) |
        (((unsigned char *)p)[2] << 16) |
        (((unsigned char *)p)[3] << 24);
#endif
#else
#if LITTLE_ENDIAN
    return *p;
#else
    return ((unsigned char *)p)[0] |
        (((unsigned char *)p)[1] << 8) |
        (((unsigned char *)p)[2] << 16) |
        (((unsigned char *)p)[3] << 24);
#endif
#endif
}

inline unsigned readlongBE(unsigned *p)
{
#if IN_LLVM
#if __BIG_ENDIAN__
    return *p;
#else
    return ((unsigned char *)p)[3] |
        (((unsigned char *)p)[2] << 8) |
        (((unsigned char *)p)[1] << 16) |
        (((unsigned char *)p)[0] << 24);
#endif
#else
    return ((unsigned char *)p)[3] |
        (((unsigned char *)p)[2] << 8) |
        (((unsigned char *)p)[1] << 16) |
        (((unsigned char *)p)[0] << 24);
#endif
}

#if IN_LLVM
void Module::parse(bool gen_docs)
#else
void Module::parse()
#endif
{
    //printf("Module::parse()\n");

    char *srcname = srcfile->name->toChars();
    //printf("Module::parse(srcname = '%s')\n", srcname);

    unsigned char *buf = srcfile->buffer;
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

                    u = le ? readlongLE(pu) : readlongBE(pu);
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
                buf = (unsigned char *) dbuf.extractData();
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

                    u = le ? readwordLE(pu) : readwordBE(pu);
                    if (u & ~0x7F)
                    {   if (u >= 0xD800 && u <= 0xDBFF)
                        {   unsigned u2;

                            if (++pu > pumax)
                            {   error("surrogate UTF-16 high value %04x at EOF", u);
                                fatal();
                            }
                            u2 = le ? readwordLE(pu) : readwordBE(pu);
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
                buf = (unsigned char *) dbuf.extractData();
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

#ifdef IN_GCC
    // dump utf-8 encoded source
    if (global.params.dump_source)
    {   // %% srcname could contain a path ...
        d_gcc_dump_source(srcname, "utf-8", buf, buflen);
    }
#endif

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
        return;
    }
#if IN_LLVM
    Parser p(this, buf, buflen, gen_docs);
#else
    Parser p(this, buf, buflen, docfile != NULL);
#endif
    p.nextToken();
    members = p.parseModule();

    if (srcfile->ref == 0)
        ::free(srcfile->buffer);
    srcfile->buffer = NULL;
    srcfile->len = 0;

    md = p.md;
    numlines = p.loc.linnum;

    DsymbolTable *dst;

    if (md)
    {   this->ident = md->id;
        this->safe = md->safe;
        Package *ppack = NULL;
        dst = Package::resolve(md->packages, &this->parent, &ppack);
        if (ppack && ppack->isModule())
        {
            error(loc, "package name '%s' in file %s conflicts with usage as a module name in file %s",
                ppack->toChars(), srcname, ppack->isModule()->srcfile->toChars());
            dst = modules;
        }
    }
    else
    {
        dst = modules;

        /* Check to see if module name is a valid identifier
         */
        if (!Lexer::isValidIdentifier(this->ident->toChars()))
            error("has non-identifier characters in filename, use module declaration instead");
    }

    // Update global list of modules
    if (!dst->insert(this))
    {
        Dsymbol *prev = dst->lookup(ident);
        assert(prev);
        Module *mprev = prev->isModule();
        if (mprev)
        {
            if (strcmp(srcname, mprev->srcfile->toChars()) == 0)
                error(loc, "from file %s must be imported as module '%s'",
                    srcname, toPrettyChars());
            else
                error(loc, "from file %s conflicts with another module %s from file %s",
                    srcname, mprev->toChars(), mprev->srcfile->toChars());
        }
        else
        {
            Package *pkg = prev->isPackage();
            assert(pkg);
            error(pkg->loc, "from file %s conflicts with package name %s",
                srcname, pkg->toChars());
        }
    }
    else
    {
        amodules.push(this);
    }
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
            s->addMember(NULL, sc->scopesym, 1);
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
    {   Dsymbol *s = (*members)[i];
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
    if (semanticstarted)
        return;

    //printf("+Module::semantic(this = %p, '%s'): parent = %p\n", this, toChars(), parent);
    semanticstarted = 1;

    // Note that modules get their own scope, from scratch.
    // This is so regardless of where in the syntax a module
    // gets imported, it is unaffected by context.
    Scope *sc = scope;                  // see if already got one from importAll()
    if (!sc)
    {   printf("test2\n");
        Scope::createGlobal(this);      // create root scope
    }

    //printf("Module = %p, linkage = %d\n", sc->scopesym, sc->linkage);

#if 0
    // Add import of "object" if this module isn't "object"
    if (ident != Id::object)
    {
        Import *im = new Import(0, NULL, Id::object, NULL, 0);
        members->shift(im);
    }

    // Add all symbols into module's symbol table
    symtab = new DsymbolTable();
    for (size_t i = 0; i < members->dim; i++)
    {   Dsymbol *s = (Dsymbol *)members->data[i];
        s->addMember(NULL, sc->scopesym, 1);
    }

    /* Set scope for the symbols so that if we forward reference
     * a symbol, it can possibly be resolved on the spot.
     * If this works out well, it can be extended to all modules
     * before any semantic() on any of them.
     */
    for (size_t i = 0; i < members->dim; i++)
    {   Dsymbol *s = (Dsymbol *)members->data[i];
        s->setScope(sc);
    }
#endif

    // Do semantic() on members that don't depend on others
    for (size_t i = 0; i < members->dim; i++)
    {   Dsymbol *s = (*members)[i];

        //printf("\tModule('%s'): '%s'.semantic0()\n", toChars(), s->toChars());
        s->semantic0(sc);
    }

    // Pass 1 semantic routines: do public side of the definition
    for (size_t i = 0; i < members->dim; i++)
    {   Dsymbol *s = (*members)[i];

        //printf("\tModule('%s'): '%s'.semantic()\n", toChars(), s->toChars());
        s->semantic(sc);
        runDeferredSemantic();
    }

    if (!scope)
    {   sc = sc->pop();
        sc->pop();              // 2 pops because Scope::createGlobal() created 2
    }
    semanticRun = semanticstarted;
    //printf("-Module::semantic(this = %p, '%s'): parent = %p\n", this, toChars(), parent);
}

void Module::semantic2()
{
    if (deferred.dim)
    {
        for (size_t i = 0; i < deferred.dim; i++)
        {
            Dsymbol *sd = deferred[i];

            sd->error("unable to resolve forward reference in definition");
        }
        return;
    }
    //printf("Module::semantic2('%s'): parent = %p\n", toChars(), parent);
    if (semanticRun == 0)       // semantic() not completed yet - could be recursive call
        return;
    if (semanticstarted >= 2)
        return;
    assert(semanticstarted == 1);
    semanticstarted = 2;

    // Note that modules get their own scope, from scratch.
    // This is so regardless of where in the syntax a module
    // gets imported, it is unaffected by context.
    Scope *sc = Scope::createGlobal(this);      // create root scope
    //printf("Module = %p\n", sc.scopesym);

    // Pass 2 semantic routines: do initializers and function bodies
    for (size_t i = 0; i < members->dim; i++)
    {   Dsymbol *s;

        s = (*members)[i];
        s->semantic2(sc);
    }

    sc = sc->pop();
    sc->pop();
    semanticRun = semanticstarted;
    //printf("-Module::semantic2('%s'): parent = %p\n", toChars(), parent);
}

void Module::semantic3()
{
    //printf("Module::semantic3('%s'): parent = %p\n", toChars(), parent);
    if (semanticstarted >= 3)
        return;
    assert(semanticstarted == 2);
    semanticstarted = 3;

    // Note that modules get their own scope, from scratch.
    // This is so regardless of where in the syntax a module
    // gets imported, it is unaffected by context.
    Scope *sc = Scope::createGlobal(this);      // create root scope
    //printf("Module = %p\n", sc.scopesym);

    // Pass 3 semantic routines: do initializers and function bodies
    for (size_t i = 0; i < members->dim; i++)
    {   Dsymbol *s;

        s = (*members)[i];
        //printf("Module %s: %s.semantic3()\n", toChars(), s->toChars());
        s->semantic3(sc);
    }

    sc = sc->pop();
    sc->pop();
    semanticRun = semanticstarted;
}

void Module::inlineScan()
{
    if (semanticstarted >= 4)
        return;
    assert(semanticstarted == 3);
    semanticstarted = 4;

    // Note that modules get their own scope, from scratch.
    // This is so regardless of where in the syntax a module
    // gets imported, it is unaffected by context.
    //printf("Module = %p\n", sc.scopesym);

    for (size_t i = 0; i < members->dim; i++)
    {   Dsymbol *s = (*members)[i];
        //if (global.params.verbose)
            //printf("inline scan symbol %s\n", s->toChars());

        s->inlineScan();
    }
    semanticRun = semanticstarted;
}

/****************************************************
 */

#if IN_DMD
void Module::gensymfile()
{
    OutBuffer buf;
    HdrGenState hgs;

    //printf("Module::gensymfile()\n");

    buf.printf("// Sym file generated from '%s'", srcfile->toChars());
    buf.writenl();

    for (size_t i = 0; i < members->dim; i++)
    {   Dsymbol *s = (*members)[i];

        s->toCBuffer(&buf, &hgs);
    }

    // Transfer image to file
    symfile->setbuffer(buf.data, buf.offset);
    buf.data = NULL;

    symfile->writev();
}
#endif

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
    Dsymbol *s;
    if (insearch)
        s = NULL;
    else if (searchCacheIdent == ident && searchCacheFlags == flags)
    {
        s = searchCacheSymbol;
        //printf("%s Module::search('%s', flags = %d) insearch = %d searchCacheSymbol = %s\n", toChars(), ident->toChars(), flags, insearch, searchCacheSymbol ? searchCacheSymbol->toChars() : "null");
    }
    else
    {
        insearch = 1;
        s = ScopeDsymbol::search(loc, ident, flags);
        insearch = 0;

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
    {   Module *m = amodules[i];
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

/************************************
 * Recursively look at every module this module imports,
 * return TRUE if it imports m.
 * Can be used to detect circular imports.
 */

int Module::imports(Module *m)
{
    //printf("%s Module::imports(%s)\n", toChars(), m->toChars());
#if 0
    for (size_t i = 0; i < aimports.dim; i++)
    {   Module *mi = (Module *)aimports.data[i];
        printf("\t[%d] %s\n", i, mi->toChars());
    }
#endif
    for (size_t i = 0; i < aimports.dim; i++)
    {   Module *mi = aimports[i];
        if (mi == m)
            return TRUE;
        if (!mi->insearch)
        {
            mi->insearch = 1;
            int r = mi->imports(m);
            if (r)
                return r;
        }
    }
    return FALSE;
}

/*************************************
 * Return !=0 if module imports itself.
 */

int Module::selfImports()
{
    //printf("Module::selfImports() %s\n", toChars());
    if (!selfimports)
    {
        for (size_t i = 0; i < amodules.dim; i++)
        {   Module *mi = amodules[i];
            //printf("\t[%d] %s\n", i, mi->toChars());
            mi->insearch = 0;
        }

        selfimports = imports(this) + 1;

        for (size_t i = 0; i < amodules.dim; i++)
        {   Module *mi = amodules[i];
            //printf("\t[%d] %s\n", i, mi->toChars());
            mi->insearch = 0;
        }
    }
    return selfimports - 1;
}


/* =========================== ModuleDeclaration ===================== */

ModuleDeclaration::ModuleDeclaration(Identifiers *packages, Identifier *id, bool safe)
{
    this->packages = packages;
    this->id = id;
    this->safe = safe;
}

char *ModuleDeclaration::toChars()
{
    OutBuffer buf;

    if (packages && packages->dim)
    {
        for (size_t i = 0; i < packages->dim; i++)
        {   Identifier *pid = (*packages)[i];

            buf.writestring(pid->toChars());
            buf.writeByte('.');
        }
    }
    buf.writestring(id->toChars());
    buf.writeByte(0);
    return (char *)buf.extractData();
}

/* =========================== Package ===================== */

Package::Package(Identifier *ident)
        : ScopeDsymbol(ident)
{
}


const char *Package::kind()
{
    return "package";
}


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
        {   Identifier *pid = (*packages)[i];
            Dsymbol *p;

            p = dst->lookup(pid);
            if (!p)
            {
                p = new Package(pid);
                dst->insert(p);
                p->parent = parent;
                ((ScopeDsymbol *)p)->symtab = new DsymbolTable();
            }
            else
            {
                assert(p->isPackage());
                // It might already be a module, not a package, but that needs
                // to be checked at a higher level, where a nice error message
                // can be generated.
                // dot net needs modules and packages with same name
            }
            parent = p;
            dst = ((Package *)p)->symtab;
            if (ppkg && !*ppkg)
                *ppkg = (Package *)p;
            if (p->isModule())
            {   // Return the module so that a nice error message can be generated
                if (ppkg)
                    *ppkg = (Package *)p;
                break;
            }
        }
        if (pparent)
        {
            *pparent = parent;
        }
    }
    return dst;
}
