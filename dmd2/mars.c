
// Compiler implementation of the D programming language
// Copyright (c) 1999-2012 by Digital Mars
// All Rights Reserved
// written by Walter Bright
// http://www.digitalmars.com
// https://github.com/D-Programming-Language/dmd/blob/master/src/mars.c
// License for redistribution is by either the Artistic License
// in artistic.txt, or the GNU General Public License in gnu.txt.
// See the included readme.txt for details.

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <assert.h>
#include <limits.h>
#include <string>
#include <cstdarg>

#if linux || __APPLE__ || __FreeBSD__ || __OpenBSD__ || __sun
#include <errno.h>
#endif

#include "rmem.h"
#include "root.h"
#if !IN_LLVM
#include "async.h"
#endif
#include "target.h"

#include "mars.h"
#include "module.h"
#include "mtype.h"
#include "id.h"
#include "cond.h"
#include "expression.h"
#include "lexer.h"
#if !IN_LLVM
#include "lib.h"
#include "json.h"
#endif

#if WINDOWS_SEH
#include <windows.h>
long __cdecl __ehfilter(LPEXCEPTION_POINTERS ep);
#endif

#if !IN_LLVM
int response_expand(size_t *pargc, char ***pargv);
void browse(const char *url);
void getenv_setargv(const char *envvar, size_t *pargc, char** *pargv);

void obj_start(char *srcfile);
void obj_end(Library *library, File *objfile);
#endif

void printCtfePerformanceStats();

static bool parse_arch(size_t argc, char** argv, bool is64bit);

Global global;

void Global::init()
{
    mars_ext = "d";
    sym_ext  = "d";
    hdr_ext  = "di";
    doc_ext  = "html";
    ddoc_ext = "ddoc";
    json_ext = "json";
    map_ext  = "map";

#if IN_LLVM
    ll_ext  = "ll";
    bc_ext  = "bc";
    s_ext   = "s";
    obj_ext = "o";
    obj_ext_alt = "obj";
#else
#if TARGET_WINDOS
    obj_ext  = "obj";
#elif TARGET_LINUX || TARGET_OSX || TARGET_FREEBSD || TARGET_OPENBSD || TARGET_SOLARIS
    obj_ext  = "o";
#else
#error "fix this"
#endif

#if TARGET_WINDOS
    lib_ext  = "lib";
#elif TARGET_LINUX || TARGET_OSX || TARGET_FREEBSD || TARGET_OPENBSD || TARGET_SOLARIS
    lib_ext  = "a";
#else
#error "fix this"
#endif
#endif

    copyright = "Copyright (c) 1999-2012 by Digital Mars";
    written = "written by Walter Bright";
#if IN_DMD
    version = "v"
#include "verstr.h"
    ;
    compiler.vendor = "Digital Mars D";
#endif
#if IN_LLVM
    compiler.vendor = "LDC";
#endif

    main_d = "__main.d";

    // This should only be used as a global, so the other fields are
    // automatically initialized to zero when the program is loaded.
    // In particular, DO NOT zero-initialize .params here (like DMD
    // does) because command-line options initialize some of those
    // fields to non-zero defaults, and do so from constructors that
    // may run before this one.
}

unsigned Global::startGagging()
{
    ++gag;
    return gaggedErrors;
}

bool Global::endGagging(unsigned oldGagged)
{
    bool anyErrs = (gaggedErrors != oldGagged);
    --gag;
    // Restore the original state of gagged errors; set total errors
    // to be original errors + new ungagged errors.
    errors -= (gaggedErrors - oldGagged);
    gaggedErrors = oldGagged;
    return anyErrs;
}

bool Global::isSpeculativeGagging()
{
    return gag && gag == speculativeGag;
}


char *Loc::toChars()
{
    OutBuffer buf;

    if (filename)
    {
        buf.printf("%s", filename);
    }

    if (linnum)
        buf.printf("(%d)", linnum);
    buf.writeByte(0);
    return (char *)buf.extractData();
}

Loc::Loc(Module *mod, unsigned linnum)
{
    this->linnum = linnum;
    this->filename = mod ? mod->srcfile->toChars() : NULL;
}

bool Loc::equals(const Loc& loc)
{
    return linnum == loc.linnum && FileName::equals(filename, loc.filename);
}

/**************************************
 * Print error message
 */

void error(Loc loc, const char *format, ...)
{
    va_list ap;
    va_start(ap, format);
    verror(loc, format, ap);
    va_end( ap );
}

void warning(Loc loc, const char *format, ...)
{
    va_list ap;
    va_start(ap, format);
    vwarning(loc, format, ap);
    va_end( ap );
}

/**************************************
 * Print supplementary message about the last error
 * Used for backtraces, etc
 */
void errorSupplemental(Loc loc, const char *format, ...)
{
    va_list ap;
    va_start(ap, format);
    verrorSupplemental(loc, format, ap);
    va_end( ap );
}

void deprecation(Loc loc, const char *format, ...)
{
    va_list ap;
    va_start(ap, format);
    vdeprecation(loc, format, ap);

    va_end( ap );
}

// Just print, doesn't care about gagging
void verrorPrint(Loc loc, const char *header, const char *format, va_list ap,
                const char *p1, const char *p2)
{
    char *p = loc.toChars();

    if (*p)
        fprintf(stderr, "%s: ", p);
    mem.free(p);

    fputs(header, stderr);
    if (p1)
        fprintf(stderr, "%s ", p1);
    if (p2)
        fprintf(stderr, "%s ", p2);
#if _MSC_VER
    // MS doesn't recognize %zu format
    OutBuffer tmp;
    tmp.vprintf(format, ap);
    fprintf(stderr, "%s", tmp.toChars());
#else
    vfprintf(stderr, format, ap);
#endif
    fprintf(stderr, "\n");
    fflush(stderr);
}

// header is "Error: " by default (see mars.h)
void verror(Loc loc, const char *format, va_list ap,
                const char *p1, const char *p2, const char *header)
{
    if (!global.gag)
    {
        verrorPrint(loc, header, format, ap, p1, p2);
        if (global.errors >= 20)        // moderate blizzard of cascading messages
                fatal();
//halt();
    }
    else
    {
        global.gaggedErrors++;
    }
    global.errors++;
}

// Doesn't increase error count, doesn't print "Error:".
void verrorSupplemental(Loc loc, const char *format, va_list ap)
{
    if (!global.gag)
        verrorPrint(loc, "       ", format, ap);
}

void vwarning(Loc loc, const char *format, va_list ap)
{
    if (global.params.warnings && !global.gag)
    {
        verrorPrint(loc, "Warning: ", format, ap);
//halt();
        if (global.params.warnings == 1)
            global.warnings++;  // warnings don't count if gagged
    }
}

void vdeprecation(Loc loc, const char *format, va_list ap,
                const char *p1, const char *p2)
{
    static const char *header = "Deprecation: ";
    if (global.params.useDeprecated == 0)
        verror(loc, format, ap, p1, p2, header);
    else if (global.params.useDeprecated == 2 && !global.gag)
        verrorPrint(loc, header, format, ap, p1, p2);
}

/***************************************
 * Call this after printing out fatal error messages to clean up and exit
 * the compiler.
 */

void fatal()
{
#if 0
    halt();
#endif
    exit(EXIT_FAILURE);
}

/**************************************
 * Try to stop forgetting to remove the breakpoints from
 * release builds.
 */
void halt()
{
#ifdef DEBUG
    *(volatile char*)0=0;
#endif
}

#if !IN_LLVM

extern void backend_init();
extern void backend_term();

void usage()
{
#if TARGET_LINUX
    const char fpic[] ="\
  -fPIC          generate position independent code\n\
";
#else
    const char fpic[] = "";
#endif
    printf("DMD%llu D Compiler %s\n%s %s\n",
           (unsigned long long) sizeof(size_t) * 8,
        global.version, global.copyright, global.written);
    printf("\
Documentation: http://dlang.org/\n\
Usage:\n\
  dmd files.d ... { -switch }\n\
\n\
  files.d        D source files\n\
  @cmdfile       read arguments from cmdfile\n\
  -c             do not link\n\
  -cov           do code coverage analysis\n\
  -cov=nnn       require at least %%nnn code coverage\n\
  -D             generate documentation\n\
  -Dddocdir      write documentation file to docdir directory\n\
  -Dffilename    write documentation file to filename\n\
  -d             silently allow deprecated features\n\
  -dw            show use of deprecated features as warnings (default)\n\
  -de            show use of deprecated features as errors (halt compilation)\n\
  -debug         compile in debug code\n\
  -debug=level   compile in debug code <= level\n\
  -debug=ident   compile in debug code identified by ident\n\
  -debuglib=name    set symbolic debug library to name\n\
  -defaultlib=name  set default library to name\n\
  -deps=filename write module dependencies to filename\n%s"
"  -g             add symbolic debug info\n\
  -gc            add symbolic debug info, pretend to be C\n\
  -gs            always emit stack frame\n\
  -gx            add stack stomp code\n\
  -H             generate 'header' file\n\
  -Hddirectory   write 'header' file to directory\n\
  -Hffilename    write 'header' file to filename\n\
  --help         print help\n\
  -Ipath         where to look for imports\n\
  -ignore        ignore unsupported pragmas\n\
  -inline        do function inlining\n\
  -Jpath         where to look for string imports\n\
  -Llinkerflag   pass linkerflag to link\n\
  -lib           generate library rather than object files\n"
#if TARGET_LINUX || TARGET_OSX || TARGET_FREEBSD || TARGET_OPENBSD || TARGET_SOLARIS
"  -m32           generate 32 bit code\n\
  -m64           generate 64 bit code\n"
#endif
"  -main          add default main() (e.g. for unittesting)\n\
  -man           open web browser on manual page\n\
  -map           generate linker .map file\n\
  -noboundscheck turns off array bounds checking for all functions\n\
  -O             optimize\n\
  -o-            do not write object file\n\
  -odobjdir      write object & library files to directory objdir\n\
  -offilename    name output file to filename\n\
  -op            preserve source path for output files\n\
  -profile       profile runtime performance of generated code\n\
  -property      enforce property syntax\n\
  -quiet         suppress unnecessary messages\n\
  -release       compile release version\n\
  -run srcfile args...   run resulting program, passing args\n\
  -shared        generate shared library (DLL)\n\
  -transition=id show additional info about language change identified by 'id'\n\
  -transition=?  list all language changes\n\
  -unittest      compile in unit tests\n\
  -v             verbose\n\
  -version=level compile in version code >= level\n\
  -version=ident compile in version code identified by ident\n\
  -vtls          list all variables going into thread local storage\n\
  -w             warnings as errors (compilation will halt)\n\
  -wi            warnings as messages (compilation will continue)\n\
  -X             generate JSON file\n\
  -Xffilename    write JSON file to filename\n\
", fpic);
}

extern signed char tyalignsize[];

#if _WIN32 && __DMC__
extern "C"
{
    extern int _xi_a;
    extern int _end;
}
#endif

int tryMain(size_t argc, char *argv[])
{
    mem.init();                         // initialize storage allocator
    mem.setStackBottom(&argv);
#if _WIN32 && __DMC__
    mem.addroots((char *)&_xi_a, (char *)&_end);
#endif

    Strings files;
    Strings libmodules;
    char *p;
    Module *m;
    size_t argcstart = argc;
    int setdebuglib = 0;
    char noboundscheck = 0;
        int setdefaultlib = 0;
    const char *inifilename = NULL;
    global.init();

#ifdef DEBUG
    printf("DMD %s DEBUG\n", global.version);
#endif

    unittests();

    // Check for malformed input
    if (argc < 1 || !argv)
    {
      Largs:
        error(Loc(), "missing or null command line arguments");
        fatal();
    }
    for (size_t i = 0; i < argc; i++)
    {
        if (!argv[i])
            goto Largs;
    }

    if (response_expand(&argc,&argv))   // expand response files
        error(Loc(), "can't open response file");

    files.reserve(argc - 1);

    // Set default values
    global.params.argv0 = argv[0];
    global.params.link = 1;
    global.params.useAssert = 1;
    global.params.useInvariants = 1;
    global.params.useIn = 1;
    global.params.useOut = 1;
    global.params.useArrayBounds = 2;   // default to all functions
    global.params.useSwitchError = 1;
    global.params.useInline = 0;
    global.params.obj = 1;
    global.params.Dversion = 2;
    global.params.quiet = 1;
    global.params.useDeprecated = 2;

    global.params.linkswitches = new Strings();
    global.params.libfiles = new Strings();
    global.params.objfiles = new Strings();
    global.params.ddocfiles = new Strings();

    // Default to -m32 for 32 bit dmd, -m64 for 64 bit dmd
    global.params.is64bit = (sizeof(size_t) == 8);

#if TARGET_WINDOS
    global.params.is64bit = 0;
    global.params.defaultlibname = "phobos";
#elif TARGET_LINUX
    global.params.defaultlibname = "libphobos2.a";
#elif TARGET_OSX || TARGET_FREEBSD || TARGET_OPENBSD || TARGET_SOLARIS
    global.params.defaultlibname = "phobos2";
#else
#error "fix this"
#endif

    // Predefine version identifiers
    VersionCondition::addPredefinedGlobalIdent("DigitalMars");

#if TARGET_WINDOS
    VersionCondition::addPredefinedGlobalIdent("Windows");
    global.params.isWindows = 1;
#elif TARGET_LINUX
    VersionCondition::addPredefinedGlobalIdent("Posix");
    VersionCondition::addPredefinedGlobalIdent("linux");
    global.params.isLinux = 1;
#elif TARGET_OSX
    VersionCondition::addPredefinedGlobalIdent("Posix");
    VersionCondition::addPredefinedGlobalIdent("OSX");
    global.params.isOSX = 1;

    // For legacy compatibility
    VersionCondition::addPredefinedGlobalIdent("darwin");
#elif TARGET_FREEBSD
    VersionCondition::addPredefinedGlobalIdent("Posix");
    VersionCondition::addPredefinedGlobalIdent("FreeBSD");
    global.params.isFreeBSD = 1;
#elif TARGET_OPENBSD
    VersionCondition::addPredefinedGlobalIdent("Posix");
    VersionCondition::addPredefinedGlobalIdent("OpenBSD");
    global.params.isFreeBSD = 1;
#elif TARGET_SOLARIS
    VersionCondition::addPredefinedGlobalIdent("Posix");
    VersionCondition::addPredefinedGlobalIdent("Solaris");
    global.params.isSolaris = 1;
#else
#error "fix this"
#endif

    VersionCondition::addPredefinedGlobalIdent("LittleEndian");
    //VersionCondition::addPredefinedGlobalIdent("D_Bits");
#if DMDV2
    VersionCondition::addPredefinedGlobalIdent("D_Version2");
#endif
    VersionCondition::addPredefinedGlobalIdent("all");

#if _WIN32
    inifilename = inifile(argv[0], "sc.ini", "Environment");
#elif linux || __APPLE__ || __FreeBSD__ || __OpenBSD__ || __sun
    inifilename = inifile(argv[0], "dmd.conf", "Environment");
#else
#error "fix this"
#endif

    size_t dflags_argc = 0;
    char** dflags_argv = NULL;
    getenv_setargv("DFLAGS", &dflags_argc, &dflags_argv);

    bool is64bit = global.params.is64bit; // use default
    is64bit = parse_arch(argc, argv, is64bit);
    is64bit = parse_arch(dflags_argc, dflags_argv, is64bit);
    global.params.is64bit = is64bit;

    inifile(argv[0], inifilename, is64bit ? "Environment64" : "Environment32");

    getenv_setargv("DFLAGS", &argc, &argv);

#if 0
    for (size_t i = 0; i < argc; i++)
    {
        printf("argv[%d] = '%s'\n", i, argv[i]);
    }
#endif

    for (size_t i = 1; i < argc; i++)
    {
        p = argv[i];
        if (*p == '-')
        {
            if (strcmp(p + 1, "de") == 0)
                global.params.useDeprecated = 0;
            else if (strcmp(p + 1, "d") == 0)
                global.params.useDeprecated = 1;
            else if (strcmp(p + 1, "dw") == 0)
                global.params.useDeprecated = 2;
            else if (strcmp(p + 1, "c") == 0)
                global.params.link = 0;
            else if (memcmp(p + 1, "cov", 3) == 0)
            {
                global.params.cov = true;
                // Parse:
                //      -cov
                //      -cov=nnn
                if (p[4] == '=')
                {
                    if (isdigit((unsigned char)p[5]))
                    {   long percent;

                        errno = 0;
                        percent = strtol(p + 5, &p, 10);
                        if (*p || errno || percent > 100)
                            goto Lerror;
                        global.params.covPercent = percent;
                    }
                    else
                        goto Lerror;
                }
                else if (p[4])
                    goto Lerror;
            }
            else if (strcmp(p + 1, "shared") == 0
#if TARGET_OSX
                // backwards compatibility with old switch
                || strcmp(p + 1, "dylib") == 0
#endif
                )
                global.params.dll = 1;
#if TARGET_LINUX || TARGET_OSX || TARGET_FREEBSD || TARGET_OPENBSD || TARGET_SOLARIS
            else if (strcmp(p + 1, "fPIC") == 0)
                global.params.pic = 1;
#endif
            else if (strcmp(p + 1, "map") == 0)
                global.params.map = 1;
            else if (strcmp(p + 1, "multiobj") == 0)
                global.params.multiobj = 1;
            else if (strcmp(p + 1, "g") == 0)
                global.params.symdebug = 1;
            else if (strcmp(p + 1, "gc") == 0)
                global.params.symdebug = 2;
            else if (strcmp(p + 1, "gs") == 0)
                global.params.alwaysframe = 1;
            else if (strcmp(p + 1, "gx") == 0)
                global.params.stackstomp = true;
            else if (strcmp(p + 1, "gt") == 0)
            {   error(Loc(), "use -profile instead of -gt");
                global.params.trace = 1;
            }
            else if (strcmp(p + 1, "m32") == 0)
                global.params.is64bit = 0;
            else if (strcmp(p + 1, "m64") == 0)
                global.params.is64bit = 1;
            else if (strcmp(p + 1, "profile") == 0)
                global.params.trace = 1;
            else if (strcmp(p + 1, "v") == 0)
                global.params.verbose = 1;
#if DMDV2
            else if (strcmp(p + 1, "vtls") == 0)
                global.params.vtls = 1;
            else if (memcmp(p + 1, "transition", 10) == 0)
            {
                // Parse:
                //      -transition=number
                if (p[11] == '=')
                {
                    if (strcmp(p + 12, "?") == 0)
                    {
                        printf("\
Language changes listed by -transition=id:\n\
  =field,3449    do list all non-mutable fields occupies object instance\n\
  =tls           do list all variables going into thread local storage\n\
");
                        return EXIT_FAILURE;
                    }
                    if (isdigit((unsigned char)p[12]))
                    {   long num;

                        errno = 0;
                        num = strtol(p + 12, &p, 10);
                        if (*p || errno || num > INT_MAX)
                            goto Lerror;
                        switch (num)    // Bugzilla issue number
                        {
                            case 3449:
                                global.params.vfield = 1;
                                break;
                            default:
                                goto Lerror;
                        }
                    }
                    else if (Lexer::isValidIdentifier(p + 12))
                    {
                        if (strcmp(p + 12, "tls") == 0)
                            global.params.vtls = 1;
                        else if (strcmp(p + 12, "field") == 0)
                            global.params.vfield = 1;
                    }
                    else
                        goto Lerror;
                }
                else
                    goto Lerror;
            }
#endif
            else if (strcmp(p + 1, "v1") == 0)
            {
#if DMDV1
                global.params.Dversion = 1;
#else
                error(Loc(), "use DMD 1.0 series compilers for -v1 switch");
                break;
#endif
            }
            else if (strcmp(p + 1, "w") == 0)
                global.params.warnings = 1;
            else if (strcmp(p + 1, "wi") == 0)
                global.params.warnings = 2;
            else if (strcmp(p + 1, "O") == 0)
                global.params.optimize = 1;
            else if (p[1] == 'o')
            {
                switch (p[2])
                {
                    case '-':
                        global.params.obj = 0;
                        break;

                    case 'd':
                        if (!p[3])
                            goto Lnoarg;
                        global.params.objdir = p + 3;
                        break;

                    case 'f':
                        if (!p[3])
                            goto Lnoarg;
                        global.params.objname = p + 3;
                        break;

                    case 'p':
                        if (p[3])
                            goto Lerror;
                        global.params.preservePaths = 1;
                        break;

                    case 0:
                        error(Loc(), "-o no longer supported, use -of or -od");
                        break;

                    default:
                        goto Lerror;
                }
            }
            else if (p[1] == 'D')
            {   global.params.doDocComments = 1;
                switch (p[2])
                {
                    case 'd':
                        if (!p[3])
                            goto Lnoarg;
                        global.params.docdir = p + 3;
                        break;
                    case 'f':
                        if (!p[3])
                            goto Lnoarg;
                        global.params.docname = p + 3;
                        break;

                    case 0:
                        break;

                    default:
                        goto Lerror;
                }
            }
            else if (p[1] == 'H')
            {   global.params.doHdrGeneration = 1;
                switch (p[2])
                {
                    case 'd':
                        if (!p[3])
                            goto Lnoarg;
                        global.params.hdrdir = p + 3;
                        break;

                    case 'f':
                        if (!p[3])
                            goto Lnoarg;
                        global.params.hdrname = p + 3;
                        break;

                    case 0:
                        break;

                    default:
                        goto Lerror;
                }
            }
            else if (p[1] == 'X')
            {   global.params.doXGeneration = 1;
                switch (p[2])
                {
                    case 'f':
                        if (!p[3])
                            goto Lnoarg;
                        global.params.xfilename = p + 3;
                        break;

                    case 0:
                        break;

                    default:
                        goto Lerror;
                }
            }
            else if (strcmp(p + 1, "ignore") == 0)
                global.params.ignoreUnsupportedPragmas = 1;
            else if (strcmp(p + 1, "property") == 0)
                global.params.enforcePropertySyntax = 1;
            else if (strcmp(p + 1, "inline") == 0)
                global.params.useInline = 1;
            else if (strcmp(p + 1, "lib") == 0)
                global.params.lib = 1;
            else if (strcmp(p + 1, "nofloat") == 0)
                global.params.nofloat = 1;
            else if (strcmp(p + 1, "quiet") == 0)
                global.params.quiet = 1;
            else if (strcmp(p + 1, "release") == 0)
                global.params.release = 1;
            else if (strcmp(p + 1, "betterC") == 0)
                global.params.betterC = 1;
#if DMDV2
            else if (strcmp(p + 1, "noboundscheck") == 0)
                noboundscheck = 1;
#endif
            else if (strcmp(p + 1, "unittest") == 0)
                global.params.useUnitTests = 1;
            else if (p[1] == 'I')
            {
                if (!global.params.imppath)
                    global.params.imppath = new Strings();
                global.params.imppath->push(p + 2);
            }
            else if (p[1] == 'J')
            {
                if (!global.params.fileImppath)
                    global.params.fileImppath = new Strings();
                global.params.fileImppath->push(p + 2);
            }
            else if (memcmp(p + 1, "debug", 5) == 0 && p[6] != 'l')
            {
                // Parse:
                //      -debug
                //      -debug=number
                //      -debug=identifier
                if (p[6] == '=')
                {
                    if (isdigit((unsigned char)p[7]))
                    {   long level;

                        errno = 0;
                        level = strtol(p + 7, &p, 10);
                        if (*p || errno || level > INT_MAX)
                            goto Lerror;
                        DebugCondition::setGlobalLevel((int)level);
                    }
                    else if (Lexer::isValidIdentifier(p + 7))
                        DebugCondition::addGlobalIdent(p + 7);
                    else
                        goto Lerror;
                }
                else if (p[6])
                    goto Lerror;
                else
                    global.params.debuglevel = 1;
            }
            else if (memcmp(p + 1, "version", 7) == 0)
            {
                // Parse:
                //      -version=number
                //      -version=identifier
                if (p[8] == '=')
                {
                    if (isdigit((unsigned char)p[9]))
                    {   long level;

                        errno = 0;
                        level = strtol(p + 9, &p, 10);
                        if (*p || errno || level > INT_MAX)
                            goto Lerror;
                        VersionCondition::setGlobalLevel((int)level);
                    }
                    else if (Lexer::isValidIdentifier(p + 9))
                        VersionCondition::addGlobalIdent(p + 9);
                    else
                        goto Lerror;
                }
                else
                    goto Lerror;
            }
            else if (strcmp(p + 1, "-b") == 0)
                global.params.debugb = 1;
            else if (strcmp(p + 1, "-c") == 0)
                global.params.debugc = 1;
            else if (strcmp(p + 1, "-f") == 0)
                global.params.debugf = 1;
            else if (strcmp(p + 1, "-help") == 0)
            {   usage();
                exit(EXIT_SUCCESS);
            }
            else if (strcmp(p + 1, "-r") == 0)
                global.params.debugr = 1;
            else if (strcmp(p + 1, "-x") == 0)
                global.params.debugx = 1;
            else if (strcmp(p + 1, "-y") == 0)
                global.params.debugy = 1;
            else if (p[1] == 'L')
            {
                global.params.linkswitches->push(p + 2);
            }
            else if (memcmp(p + 1, "defaultlib=", 11) == 0)
            {
                setdefaultlib = 1;
                global.params.defaultlibname = p + 1 + 11;
            }
            else if (memcmp(p + 1, "debuglib=", 9) == 0)
            {
                setdebuglib = 1;
                global.params.debuglibname = p + 1 + 9;
            }
            else if (memcmp(p + 1, "deps=", 5) == 0)
            {
                global.params.moduleDepsFile = p + 1 + 5;
                if (!global.params.moduleDepsFile[0])
                    goto Lnoarg;
                global.params.moduleDeps = new OutBuffer;
            }
            else if (strcmp(p + 1, "main") == 0)
            {
                global.params.addMain = true;
            }
            else if (memcmp(p + 1, "man", 3) == 0)
            {
#if _WIN32
#if DMDV1
                browse("http://www.digitalmars.com/d/1.0/dmd-windows.html");
#else
                browse("http://dlang.org/dmd-windows.html");
#endif
#endif
#if linux
#if DMDV1
                browse("http://www.digitalmars.com/d/1.0/dmd-linux.html");
#else
                browse("http://dlang.org/dmd-linux.html");
#endif
#endif
#if __APPLE__
#if DMDV1
                browse("http://www.digitalmars.com/d/1.0/dmd-osx.html");
#else
                browse("http://dlang.org/dmd-osx.html");
#endif
#endif
#if __FreeBSD__
#if DMDV1
                browse("http://www.digitalmars.com/d/1.0/dmd-freebsd.html");
#else
                browse("http://dlang.org/dmd-freebsd.html");
#endif
#endif
#if __OpenBSD__
#if DMDV1
                browse("http://www.digitalmars.com/d/1.0/dmd-openbsd.html");
#else
                browse("http://dlang.org/dmd-openbsd.html");
#endif
#endif
                exit(EXIT_SUCCESS);
            }
            else if (strcmp(p + 1, "run") == 0)
            {   global.params.run = 1;
                global.params.runargs_length = ((i >= argcstart) ? argc : argcstart) - i - 1;
                if (global.params.runargs_length)
                {
                    const char *ext = FileName::ext(argv[i + 1]);
                    if (ext && FileName::equals(ext, "d") == 0
                            && FileName::equals(ext, "di") == 0)
                    {
                        error(Loc(), "-run must be followed by a source file, not '%s'", argv[i + 1]);
                        break;
                    }

                    files.push(argv[i + 1]);
                    global.params.runargs = &argv[i + 2];
                    i += global.params.runargs_length;
                    global.params.runargs_length--;
                }
                else
                {   global.params.run = 0;
                    goto Lnoarg;
                }
            }
            else
            {
             Lerror:
                error(Loc(), "unrecognized switch '%s'", argv[i]);
                continue;

             Lnoarg:
                error(Loc(), "argument expected for switch '%s'", argv[i]);
                continue;
            }
        }
        else
        {
#if TARGET_WINDOS
            const char *ext = FileName::ext(p);
            if (ext && FileName::compare(ext, "exe") == 0)
            {
                global.params.objname = p;
                continue;
            }
#endif
            files.push(p);
        }
    }

    if(global.params.is64bit != is64bit)
        error(Loc(), "the architecture must not be changed in the %s section of %s",
              is64bit ? "Environment64" : "Environment32", inifilename);

    if (global.errors)
    {
        fatal();
    }
    if (files.dim == 0)
    {   usage();
        return EXIT_FAILURE;
    }

    if (!setdebuglib)
        global.params.debuglibname = global.params.defaultlibname;

#if TARGET_OSX
    global.params.pic = 1;
#endif

#if TARGET_LINUX || TARGET_OSX || TARGET_FREEBSD || TARGET_OPENBSD || TARGET_SOLARIS
    if (global.params.lib && global.params.dll)
        error(Loc(), "cannot mix -lib and -shared");
#endif

    if (global.params.release)
    {   global.params.useInvariants = 0;
        global.params.useIn = 0;
        global.params.useOut = 0;
        global.params.useAssert = 0;
        global.params.useArrayBounds = 1;
        global.params.useSwitchError = 0;
    }
    if (noboundscheck)
        global.params.useArrayBounds = 0;

    if (global.params.run)
        global.params.quiet = 1;

    if (global.params.useUnitTests)
        global.params.useAssert = 1;

    if (!global.params.obj || global.params.lib)
        global.params.link = 0;

    if (global.params.link)
    {
        global.params.exefile = global.params.objname;
        global.params.oneobj = 1;
        if (global.params.objname)
        {
            /* Use this to name the one object file with the same
             * name as the exe file.
             */
            global.params.objname = const_cast<char *>(FileName::forceExt(global.params.objname, global.obj_ext));

            /* If output directory is given, use that path rather than
             * the exe file path.
             */
            if (global.params.objdir)
            {   const char *name = FileName::name(global.params.objname);
                global.params.objname = (char *)FileName::combine(global.params.objdir, name);
            }
        }
    }
    else if (global.params.lib)
    {
        global.params.libname = global.params.objname;
        global.params.objname = NULL;

        // Haven't investigated handling these options with multiobj
        if (!global.params.cov && !global.params.trace
#if 0 && TARGET_WINDOS
            /* multiobj causes class/struct debug info to be attached to init-data,
             * but this will not be linked into the executable, so this info is lost.
             * Bugzilla 4014
             */
            && !global.params.symdebug
#endif
           )
            global.params.multiobj = 1;
    }
    else if (global.params.run)
    {
        error(Loc(), "flags conflict with -run");
        fatal();
    }
    else
    {
        if (global.params.objname && files.dim > 1)
        {
            global.params.oneobj = 1;
            //error("multiple source files, but only one .obj name");
            //fatal();
        }
    }
    if (global.params.is64bit)
    {
        VersionCondition::addPredefinedGlobalIdent("D_InlineAsm_X86_64");
        VersionCondition::addPredefinedGlobalIdent("X86_64");
        VersionCondition::addPredefinedGlobalIdent("D_LP64");
        VersionCondition::addPredefinedGlobalIdent("D_SIMD");
#if TARGET_WINDOS
        VersionCondition::addPredefinedGlobalIdent("Win64");
        if (!setdefaultlib)
        {   global.params.defaultlibname = "phobos64";
            if (!setdebuglib)
                global.params.debuglibname = global.params.defaultlibname;
        }
#endif
    }
    else
    {
        VersionCondition::addPredefinedGlobalIdent("D_InlineAsm");
        VersionCondition::addPredefinedGlobalIdent("D_InlineAsm_X86");
        VersionCondition::addPredefinedGlobalIdent("X86");
#if TARGET_OSX
        VersionCondition::addPredefinedGlobalIdent("D_SIMD");
#endif
#if TARGET_WINDOS
        VersionCondition::addPredefinedGlobalIdent("Win32");
#endif
    }
    if (global.params.doDocComments)
        VersionCondition::addPredefinedGlobalIdent("D_Ddoc");
    if (global.params.cov)
        VersionCondition::addPredefinedGlobalIdent("D_Coverage");
    if (global.params.pic)
        VersionCondition::addPredefinedGlobalIdent("D_PIC");
#if DMDV2
    if (global.params.useUnitTests)
        VersionCondition::addPredefinedGlobalIdent("unittest");
    if (global.params.useAssert)
        VersionCondition::addPredefinedGlobalIdent("assert");
    if (noboundscheck)
        VersionCondition::addPredefinedGlobalIdent("D_NoBoundsChecks");
#endif

    VersionCondition::addPredefinedGlobalIdent("D_HardFloat");

    // Initialization
    Type::init();
    Id::initialize();
    Module::init();
    Target::init();
    Expression::init();
    initPrecedence();

    if (global.params.verbose)
    {   printf("binary    %s\n", argv[0]);
        printf("version   %s\n", global.version);
        printf("config    %s\n", inifilename ? inifilename : "(none)");
    }

    //printf("%d source files\n",files.dim);

    // Build import search path
    if (global.params.imppath)
    {
        for (size_t i = 0; i < global.params.imppath->dim; i++)
        {
            char *path = (*global.params.imppath)[i];
            Strings *a = FileName::splitPath(path);

            if (a)
            {
                if (!global.path)
                    global.path = new Strings();
                global.path->append(a);
            }
        }
    }

    // Build string import search path
    if (global.params.fileImppath)
    {
        for (size_t i = 0; i < global.params.fileImppath->dim; i++)
        {
            char *path = (*global.params.fileImppath)[i];
            Strings *a = FileName::splitPath(path);

            if (a)
            {
                if (!global.filePath)
                    global.filePath = new Strings();
                global.filePath->append(a);
            }
        }
    }

    if (global.params.addMain)
    {
        files.push(const_cast<char*>(global.main_d)); // a dummy name, we never actually look up this file
    }

    // Create Modules
    Modules modules;
    modules.reserve(files.dim);
    bool firstmodule = true;
    for (size_t i = 0; i < files.dim; i++)
    {
        const char *ext;
        char *name;

        p = files[i];

#if _WIN32
        // Convert / to \ so linker will work
        for (size_t j = 0; p[j]; j++)
        {
            if (p[j] == '/')
                p[j] = '\\';
        }
#endif

        p = (char *)FileName::name(p);          // strip path
        ext = FileName::ext(p);
        if (ext)
        {   /* Deduce what to do with a file based on its extension
             */
            if (FileName::equals(ext, global.obj_ext))
            {
                global.params.objfiles->push(files[i]);
                libmodules.push(files[i]);
                continue;
            }

            if (FileName::equals(ext, global.lib_ext))
            {
                global.params.libfiles->push(files[i]);
                libmodules.push(files[i]);
                continue;
            }

            if (strcmp(ext, global.ddoc_ext) == 0)
            {
                global.params.ddocfiles->push(files[i]);
                continue;
            }

            if (FileName::equals(ext, global.json_ext))
            {
                global.params.doXGeneration = 1;
                global.params.xfilename = files[i];
                continue;
            }

            if (FileName::equals(ext, global.map_ext))
            {
                global.params.mapfile = files[i];
                continue;
            }

#if TARGET_WINDOS
            if (FileName::equals(ext, "res"))
            {
                global.params.resfile = files[i];
                continue;
            }

            if (FileName::equals(ext, "def"))
            {
                global.params.deffile = files[i];
                continue;
            }

            if (FileName::equals(ext, "exe"))
            {
                assert(0);      // should have already been handled
            }
#endif

            /* Examine extension to see if it is a valid
             * D source file extension
             */
            if (FileName::equals(ext, global.mars_ext) ||
                FileName::equals(ext, global.hdr_ext) ||
                FileName::equals(ext, "dd"))
            {
                ext--;                  // skip onto '.'
                assert(*ext == '.');
                name = (char *)mem.malloc((ext - p) + 1);
                memcpy(name, p, ext - p);
                name[ext - p] = 0;              // strip extension

                if (name[0] == 0 ||
                    strcmp(name, "..") == 0 ||
                    strcmp(name, ".") == 0)
                {
                Linvalid:
                    error(Loc(), "invalid file name '%s'", files[i]);
                    fatal();
                }
            }
            else
            {   error(Loc(), "unrecognized file extension %s", ext);
                fatal();
            }
        }
        else
        {   name = p;
            if (!*name)
                goto Linvalid;
        }

        /* At this point, name is the D source file name stripped of
         * its path and extension.
         */

        Identifier *id = Lexer::idPool(name);
        m = new Module(files[i], id, global.params.doDocComments, global.params.doHdrGeneration);
        modules.push(m);

        if (firstmodule)
        {   global.params.objfiles->push((char *)m->objfile->name->str);
            firstmodule = false;
        }
    }

    // Read files

    /* Start by "reading" the dummy main.d file
     */
    if (global.params.addMain)
    {
        for (size_t i = 0; 1; i++)
        {
            assert(i != modules.dim);
            Module *m = modules[i];
            if (strcmp(m->srcfile->name->str, global.main_d) == 0)
            {
                static const char buf[] = "int main(){return 0;}";
                m->srcfile->setbuffer((void *)buf, sizeof(buf));
                m->srcfile->ref = 1;
                break;
            }
        }
    }

#define ASYNCREAD 1
#if ASYNCREAD
    // Multi threaded
    AsyncRead *aw = AsyncRead::create(modules.dim);
    for (size_t i = 0; i < modules.dim; i++)
    {
        m = modules[i];
        aw->addFile(m->srcfile);
    }
    aw->start();
#else
    // Single threaded
    for (size_t i = 0; i < modules.dim; i++)
    {
        m = modules[i];
        m->read(0);
    }
#endif

    // Parse files
    bool anydocfiles = false;
    size_t filecount = modules.dim;
    for (size_t filei = 0, modi = 0; filei < filecount; filei++, modi++)
    {
        m = modules[modi];
        if (global.params.verbose)
            printf("parse     %s\n", m->toChars());
        if (!Module::rootModule)
            Module::rootModule = m;
        m->importedFrom = m;
        if (!global.params.oneobj || modi == 0 || m->isDocFile)
            m->deleteObjFile();
#if ASYNCREAD
        if (aw->read(filei))
        {
            error(Loc(), "cannot read file %s", m->srcfile->name->toChars());
            fatal();
        }
#endif
        m->parse();
        if (m->isDocFile)
        {
            anydocfiles = true;
            m->gendocfile();

            // Remove m from list of modules
            modules.remove(modi);
            modi--;

            // Remove m's object file from list of object files
            for (size_t j = 0; j < global.params.objfiles->dim; j++)
            {
                if (m->objfile->name->str == (*global.params.objfiles)[j])
                {
                    global.params.objfiles->remove(j);
                    break;
                }
            }

            if (global.params.objfiles->dim == 0)
                global.params.link = 0;
        }
    }
#if ASYNCREAD
    AsyncRead::dispose(aw);
#endif

    if (anydocfiles && modules.dim &&
        (global.params.oneobj || global.params.objname))
    {
        error(Loc(), "conflicting Ddoc and obj generation options");
        fatal();
    }
    if (global.errors)
        fatal();
    if (global.params.doHdrGeneration)
    {
        /* Generate 'header' import files.
         * Since 'header' import files must be independent of command
         * line switches and what else is imported, they are generated
         * before any semantic analysis.
         */
        for (size_t i = 0; i < modules.dim; i++)
        {
            m = modules[i];
            if (global.params.verbose)
                printf("import    %s\n", m->toChars());
            m->genhdrfile();
        }
    }
    if (global.errors)
        fatal();

    // load all unconditional imports for better symbol resolving
    for (size_t i = 0; i < modules.dim; i++)
    {
       m = modules[i];
       if (global.params.verbose)
           printf("importall %s\n", m->toChars());
       m->importAll(NULL);
    }
    if (global.errors)
        fatal();

    backend_init();

    // Do semantic analysis
    for (size_t i = 0; i < modules.dim; i++)
    {
        m = modules[i];
        if (global.params.verbose)
            printf("semantic  %s\n", m->toChars());
        m->semantic();
    }
    if (global.errors)
        fatal();

    Module::dprogress = 1;
    Module::runDeferredSemantic();

    // Do pass 2 semantic analysis
    for (size_t i = 0; i < modules.dim; i++)
    {
        m = modules[i];
        if (global.params.verbose)
            printf("semantic2 %s\n", m->toChars());
        m->semantic2();
    }
    if (global.errors)
        fatal();

    // Do pass 3 semantic analysis
    for (size_t i = 0; i < modules.dim; i++)
    {
        m = modules[i];
        if (global.params.verbose)
            printf("semantic3 %s\n", m->toChars());
        m->semantic3();
    }
    if (global.errors)
        fatal();

    if (global.params.useInline)
    {
        /* The problem with useArrayBounds and useAssert is that the
         * module being linked to may not have generated them, so if
         * we inline functions from those modules, the symbols for them will
         * not be found at link time.
         * We must do this BEFORE generating the .deps file!
         */
        if (!global.params.useArrayBounds && !global.params.useAssert)
        {
            // Do pass 3 semantic analysis on all imported modules,
            // since otherwise functions in them cannot be inlined
            for (size_t i = 0; i < Module::amodules.dim; i++)
            {
                m = Module::amodules[i];
                if (global.params.verbose)
                    printf("semantic3 %s\n", m->toChars());
                m->semantic3();
            }
            if (global.errors)
                fatal();
        }
    }

    if (global.params.moduleDeps != NULL)
    {
        assert(global.params.moduleDepsFile != NULL);

        File deps(global.params.moduleDepsFile);
        OutBuffer* ob = global.params.moduleDeps;
        deps.setbuffer((void*)ob->data, ob->offset);
        deps.writev();
    }

    // Scan for functions to inline
    if (global.params.useInline)
    {
        for (size_t i = 0; i < modules.dim; i++)
        {
            m = modules[i];
            if (global.params.verbose)
                printf("inline scan %s\n", m->toChars());
            m->inlineScan();
        }
    }

    // Do not attempt to generate output files if errors or warnings occurred
    if (global.errors || global.warnings)
        fatal();

    printCtfePerformanceStats();

    Library *library = NULL;
    if (global.params.lib)
    {
        library = Library::factory();
        library->setFilename(global.params.objdir, global.params.libname);

        // Add input object and input library files to output library
        for (size_t i = 0; i < libmodules.dim; i++)
        {
            char *p = libmodules[i];
            library->addObject(p, NULL, 0);
        }
    }

    // Generate output files

    if (global.params.doXGeneration)
    {
        OutBuffer buf;
        json_generate(&buf, &modules);

        // Write buf to file
        const char *name = global.params.xfilename;

        if (name && name[0] == '-' && name[1] == 0)
        {   // Write to stdout; assume it succeeds
            int n = fwrite(buf.data, 1, buf.offset, stdout);
            assert(n == buf.offset);        // keep gcc happy about return values
        }
        else
        {
            /* The filename generation code here should be harmonized with Module::setOutfile()
             */

            const char *jsonfilename;

            if (name && *name)
            {
                jsonfilename = FileName::defaultExt(name, global.json_ext);
            }
            else
            {
                // Generate json file name from first obj name
                const char *n = (*global.params.objfiles)[0];
                n = FileName::name(n);

                //if (!FileName::absolute(name))
                    //name = FileName::combine(dir, name);

                jsonfilename = FileName::forceExt(n, global.json_ext);
            }

            FileName::ensurePathToNameExists(jsonfilename);

            File *jsonfile = new File(jsonfilename);

            jsonfile->setbuffer(buf.data, buf.offset);
            jsonfile->ref = 1;
            jsonfile->writev();
        }
    }

    if (global.params.oneobj)
    {
        for (size_t i = 0; i < modules.dim; i++)
        {
            m = modules[i];
            if (global.params.verbose)
                printf("code      %s\n", m->toChars());
            if (i == 0)
                obj_start(m->srcfile->toChars());
            m->genobjfile(0);
            if (!global.errors && global.params.doDocComments)
                m->gendocfile();
        }
        if (!global.errors && modules.dim)
        {
            obj_end(library, modules[0]->objfile);
        }
    }
    else
    {
        for (size_t i = 0; i < modules.dim; i++)
        {
            m = modules[i];
            if (global.params.verbose)
                printf("code      %s\n", m->toChars());
            if (global.params.obj)
            {   obj_start(m->srcfile->toChars());
                m->genobjfile(global.params.multiobj);
                obj_end(library, m->objfile);
                obj_write_deferred(library);
            }
            if (global.errors)
            {
                if (!global.params.lib)
                    m->deleteObjFile();
            }
            else
            {
                if (global.params.doDocComments)
                    m->gendocfile();
            }
        }
    }

    if (global.params.lib && !global.errors)
        library->write();

    backend_term();
    if (global.errors)
        fatal();

    int status = EXIT_SUCCESS;
    if (!global.params.objfiles->dim)
    {
        if (global.params.link)
            error(Loc(), "no object files to link");
    }
    else
    {
        if (global.params.link)
            status = runLINK();

        if (global.params.run)
        {
            if (!status)
            {
                status = runProgram();

                /* Delete .obj files and .exe file
                 */
                for (size_t i = 0; i < modules.dim; i++)
                {
                    Module *m = modules[i];
                    m->deleteObjFile();
                    if (global.params.oneobj)
                        break;
                }
                deleteExeFile();
            }
        }
    }

    return status;
}

int main(int argc, char *argv[])
{
    int status = -1;
#if WINDOWS_SEH
  __try
  {
    status = tryMain(argc, argv);
  }
  __except (__ehfilter(GetExceptionInformation()))
  {
    printf("Stack overflow\n");
    fatal();
  }
#else
  status = tryMain(argc, argv);
#endif
    return status;
}

#endif // !IN_LLVM

/***********************************
 * Parse and append contents of environment variable envvar
 * to argc and argv[].
 * The string is separated into arguments, processing \ and ".
 */

void getenv_setargv(const char *envvar, size_t *pargc, char** *pargv)
{
    char *p;

    int instring;
    int slash;
    char c;

    char *env = getenv(envvar);
    if (!env)
        return;

    env = mem.strdup(env);      // create our own writable copy

    size_t argc = *pargc;
    Strings *argv = new Strings();
    argv->setDim(argc);

    int argc_left = 0;
    for (int i = 0; i < argc; i++) {
        if (!strcmp((*pargv)[i], "-run") || !strcmp((*pargv)[i], "--run")) {
            // HACK: set flag to indicate we saw '-run' here
            global.params.run = true;
            // Don't eat -run yet so the program arguments don't get changed
            argc_left = argc - i;
            argc = i;
            *pargv = &(*pargv)[i];
            argv->setDim(i);
            break;
        } else {
        }
    }
    // HACK to stop required values from command line being drawn from DFLAGS
    argv->push((char*)"");
    argc++;

    size_t j = 1;               // leave argv[0] alone
    while (1)
    {
        int wildcard = 1;       // do wildcard expansion
        switch (*env)
        {
            case ' ':
            case '\t':
                env++;
                break;

            case 0:
                goto Ldone;

            case '"':
                wildcard = 0;
            default:
                argv->push(env);                // append
                //argv->insert(j, env);         // insert at position j
                j++;
                argc++;
                p = env;
                slash = 0;
                instring = 0;
                c = 0;

                while (1)
                {
                    c = *env++;
                    switch (c)
                    {
                        case '"':
                            p -= (slash >> 1);
                            if (slash & 1)
                            {   p--;
                                goto Laddc;
                            }
                            instring ^= 1;
                            slash = 0;
                            continue;

                        case ' ':
                        case '\t':
                            if (instring)
                                goto Laddc;
                            *p = 0;
                            //if (wildcard)
                                //wildcardexpand();     // not implemented
                            break;

                        case '\\':
                            slash++;
                            *p++ = c;
                            continue;

                        case 0:
                            *p = 0;
                            //if (wildcard)
                                //wildcardexpand();     // not implemented
                            goto Ldone;

                        default:
                        Laddc:
                            slash = 0;
                            *p++ = c;
                            continue;
                    }
                    break;
                }
        }
    }

Ldone:
    assert(argc == argv->dim);
    argv->reserve(argc_left);
    for (int i = 0; i < argc_left; i++)
        argv->data[argc++] = (void *)(*pargv)[i];

    *pargc = argc;
    *pargv = argv->tdata();
}

/***********************************
 * Parse command line arguments for -m32 or -m64
 * to detect the desired architecture.
 */

static bool parse_arch(size_t argc, char** argv, bool is64bit)
{
    for (size_t i = 0; i < argc; ++i)
    {   char* p = argv[i];
        if (p[0] == '-')
        {
            if (strcmp(p + 1, "m32") == 0)
                is64bit = 0;
            else if (strcmp(p + 1, "m64") == 0)
                is64bit = 1;
            else if (strcmp(p + 1, "run") == 0)
                break;
        }
    }
    return is64bit;
}

#if WINDOWS_SEH

long __cdecl __ehfilter(LPEXCEPTION_POINTERS ep)
{
    //printf("%x\n", ep->ExceptionRecord->ExceptionCode);
    if (ep->ExceptionRecord->ExceptionCode == STATUS_STACK_OVERFLOW)
    {
#if 1 //ndef DEBUG
        return EXCEPTION_EXECUTE_HANDLER;
#endif
    }
    return EXCEPTION_CONTINUE_SEARCH;
}

#endif
