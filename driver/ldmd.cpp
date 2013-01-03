//===-- ldmd.cpp - Drop-in DMD replacement wrapper for LDC ----------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license, except for the
// command line handling code, which originated from DMD. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Wrapper allowing use of LDC as drop-in replacement for DMD.
//
// The reason why full command line parsing is required instead of just
// rewriting the names of a few switches is an annoying impedance mismatch
// between the way how DMD handles arguments and the LLVM command line library:
// DMD allows all switches to be specified multiple times – in case of
// conflicts, the last one takes precedence. There is no easy way to replicate
// this behavior with LLVM, save parsing the switches and re-emitting a cleaned
// up string.
//
// DMD also reads switches from the DFLAGS enviroment variable, if present. This
// is contrary to what C compilers do, where CFLAGS is usually handled by the
// build system. Thus, conflicts like mentioned above occur quite frequently in
// practice in makefiles written for DMD, as DFLAGS is also a natural name for
// handling flags there.
//
// In order to maintain backwards compatibility with earlier versions of LDMD,
// unknown switches are passed through verbatim to LDC. Finding a better
// solution for this is tricky, as some of the LLVM arguments can be
// intentionally specified multiple times to get a certain effect (e.g. pass,
// linker options).
//
// Just as with the old LDMD script, arguments can be passed through unmodified
// to LDC by using -Csomearg.
//
// If maintaining this wrapper is deemed too messy at some point, an alternative
// would be to either extend the LLVM command line library to support the DMD
// semantics (unlikely to happen), or to abandon it altogether (except for
// passing the LLVM-defined flags to the various passes).
//
// Note: This program inherited ugly C-style string handling and memory leaks
// from DMD, but this should not be a problem due to the short-livedness of
// the process.
//
//===----------------------------------------------------------------------===//

#ifndef LDC_EXE_NAME
# error "Please define LDC_EXE_NAME to the name of the LDC executable to use."
#endif

#include <cassert>
#include <cerrno>
#include <climits>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SystemUtils.h"
#include "llvm/Support/raw_ostream.h"

#ifdef HAVE_SC_ARG_MAX
# include <unistd.h>
#endif

namespace ls = llvm::sys;

// We reuse DMD's response file parsing routine for maximum compatibilty - it
// handles quotes in a very peciuliar way.
int response_expand(int *pargc, char ***pargv);
void browse(const char *url);

/**
 * Prints a formatted error message to stderr and exits the program.
 */
void error(const char* fmt, ...)
{
    va_list argp;
    va_start(argp, fmt);
    fprintf(stderr, "Error: ");
    vfprintf(stderr, fmt, argp);
    fprintf(stderr, "\n");
    exit(EXIT_FAILURE);
    va_end(argp);
}

/**
 * Prints a formatted warning message to stderr.
 */
void warning(const char* fmt, ...)
{
    va_list argp;
    va_start(argp, fmt);
    fprintf(stderr, "Warning: ");
    vfprintf(stderr, fmt, argp);
    fprintf(stderr, "\n");
    va_end(argp);
}

char* concat(const char* a, const char* b)
{
    size_t na = strlen(a);
    size_t nb = strlen(b);
    char* result = static_cast<char*>(malloc(na + nb + 1));
    assert(result);
    memcpy(result, a, na);
    memcpy(result + na, b, nb + 1);
    return result;
}

char* concat(const char* a, int b)
{
    char bStr[14];
#if defined(_MSC_VER)
    _snprintf_s(bStr, _countof(bStr), sizeof(bStr), "%d", b);
#else
    snprintf(bStr, sizeof(bStr), "%d", b);
#endif
    return concat(a, bStr);
}

/**
 * Runs the given executable, returning its error code.
 */
int execute(ls::Path exePath, const char** args)
{
    std::string errorMsg;
    int rc = ls::Program::ExecuteAndWait(exePath, args, NULL, NULL,
        0, 0, &errorMsg);
    if (!errorMsg.empty())
    {
        error("Could not execute %s: %s", exePath.c_str(), errorMsg.c_str());
    }
    return rc;
}

/**
 * Prints usage information to stdout.
 */
void printUsage(const char* argv0, ls::Path ldcPath)
{
    // Print version information by actually invoking ldc -version.
    const char* args[] = { ldcPath.c_str(), "-version", NULL };
    execute(ldcPath, args);

    printf("\n\
Usage:\n\
  %s files.d ... { -switch }\n\
\n\
  files.d        D source files\n\
  @cmdfile       read arguments from cmdfile\n\
  -c             do not link\n"
#if 0
"  -cov           do code coverage analysis\n"
#endif
"  -D             generate documentation\n\
  -Dddocdir      write documentation file to docdir directory\n\
  -Dffilename    write documentation file to filename\n\
  -d             allow deprecated features\n\
  -debug         compile in debug code\n\
  -debug=level   compile in debug code <= level\n\
  -debug=ident   compile in debug code identified by ident\n\
  -debuglib=name    set symbolic debug library to name\n\
  -defaultlib=name  set default library to name\n\
  -deps=filename write module dependencies to filename\n\
  -fPIC          generate position independent code\n\
  -g             add symbolic debug info\n\
  -gc            add symbolic debug info, pretend to be C\n\
  -gs            always emit stack frame\n\
  -H             generate 'header' file\n\
  -Hddirectory   write 'header' file to directory\n\
  -Hffilename    write 'header' file to filename\n\
  --help         print help\n\
  -Ipath         where to look for imports\n\
  -ignore        ignore unsupported pragmas\n\
  -inline        do function inlining\n\
  -Jpath         where to look for string imports\n\
  -Llinkerflag   pass linkerflag to link\n\
  -lib           generate library rather than object files\n\
  -m32           generate 32 bit code\n\
  -m64           generate 64 bit code\n\
  -man           open web browser on manual page\n"
#if 0
"  -map           generate linker .map file\n"
#endif
#if DMDV2
"  -noboundscheck turns off array bounds checking for all functions\n"
#endif
"  -nofloat       do not emit reference to floating point\n\
  -O             optimize\n\
  -o-            do not write object file\n\
  -odobjdir      write object & library files to directory objdir\n\
  -offilename    name output file to filename\n\
  -op            do not strip paths from source file\n"
#if 0
"  -profile       profile runtime performance of generated code\n"
#endif
"  -property      enforce property syntax\n\
  -quiet         suppress unnecessary messages\n\
  -release       compile release version\n\
  -run srcfile args...   run resulting program, passing args\n\
  -shared        generate shared library\n\
  -unittest      compile in unit tests\n\
  -v             verbose\n\
  -vdmd          print the command used to invoke the underlying compiler\n\
  -version=level compile in version code >= level\n\
  -version=ident compile in version code identified by ident\n"
#if 0
"  -vtls          list all variables going into thread local storage\n"
#endif
"  -w             enable warnings\n\
  -wi            enable informational warnings\n\
  -X             generate JSON file\n\
  -Xffilename    write JSON file to filename\n\n", argv0
    );
}

/**
 * Parses an enviroment variable for flags and appends them to given list of
 * arguments.
 *
 * This corresponds to getenv_setargv() in DMD, but we need to duplicate it
 * here since it is defined in mars.c.
 */
void appendEnvVar(const char* envVarName, std::vector<char*>& args)
{
    char *env = getenv(envVarName);
    if (!env)
        return;

    env = strdup(env);      // create our own writable copy

    size_t j = 1;               // leave argv[0] alone
    while (1)
    {
        switch (*env)
        {
            case ' ':
            case '\t':
                env++;
                break;

            case 0:
                return;

            default:
                args.push_back(env);                // append
                j++;
                char* p = env;
                int slash = 0;
                int instring = 0;
                char c = 0;

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
                            break;

                        case '\\':
                            slash++;
                            *p++ = c;
                            continue;

                        case 0:
                            *p = 0;
                            return;

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
}

struct Debug
{
    enum Type
    {
        none,
        normal,
        pretendC
    };
};

struct Model
{
    enum Type
    {
        automatic,
        m32,
        m64
    };
};

struct Warnings
{
    enum Type
    {
        none,
        asErrors,
        informational
    };
};

struct Params
{
    bool allowDeprecated;
    bool compileOnly;
    bool coverage;
    bool emitSharedLib;
    bool pic;
    bool emitMap;
    bool multiObj;
    Debug::Type debugInfo;
    bool alwaysStackFrame;
    Model::Type targetModel;
    bool profile;
    bool verbose;
    bool vdmd;
    bool logTlsUse;
    Warnings::Type warnings;
    bool optimize;
    bool noObj;
    char* objDir;
    char* objName;
    bool preservePaths;
    bool generateDocs;
    char* docDir;
    char* docName;
    bool generateHeaders;
    char* headerDir;
    char* headerName;
    bool generateJson;
    char* jsonName;
    bool ignoreUnsupportedPragmas;
    bool enforcePropertySyntax;
    bool enableInline;
    bool emitStaticLib;
    bool noFloat;
    bool quiet;
    bool release;
    bool noBoundsChecks;
    bool emitUnitTests;
    std::vector<char*> modulePaths;
    std::vector<char*> importPaths;
    bool debugFlag;
    unsigned debugLevel;
    std::vector<char*> debugIdentifiers;
    unsigned versionLevel;
    std::vector<char*> versionIdentifiers;
    std::vector<char*> linkerSwitches;
    char* defaultLibName;
    char* debugLibName;
    char* moduleDepsFile;

    bool hiddenDebugB;
    bool hiddenDebugC;
    bool hiddenDebugF;
    bool hiddenDebugR;
    bool hiddenDebugX;
    bool hiddenDebugY;

    std::vector<char*> unknownSwitches;

    bool run;
    std::vector<char*> files;
    std::vector<char*> runArgs;
};

/**
 * Parses the flags from the given command line and the DFLAGS environment
 * variable into a Params struct.
 */
Params parseArgs(int originalArgc, char** originalArgv, ls::Path ldcPath)
{
    // Expand any response files present into the list of arguments.
    int argc = originalArgc;
    char** argv = originalArgv;
    if (response_expand(&argc, &argv))
    {
        error("Could not read response file.");
    }

    std::vector<char*> args(argv, argv + argc);

    appendEnvVar("DFLAGS", args);

    Params result = Params();
    for (size_t i = 1; i < args.size(); i++)
    {
        char* p = args[i];
        if (*p == '-')
        {
            if (strcmp(p + 1, "d") == 0)
                result.allowDeprecated = true;
            else if (strcmp(p + 1, "c") == 0)
                result.compileOnly = true;
            else if (strcmp(p + 1, "cov") == 0)
                result.coverage = true;
            else if (strcmp(p + 1, "shared") == 0
                // backwards compatibility with old switch
                || strcmp(p + 1, "dylib") == 0
                )
                result.emitSharedLib = true;
            else if (strcmp(p + 1, "fPIC") == 0)
                result.pic = true;
            else if (strcmp(p + 1, "map") == 0)
                result.emitMap = true;
            else if (strcmp(p + 1, "multiobj") == 0)
                result.multiObj = true;
            else if (strcmp(p + 1, "g") == 0)
                result.debugInfo = Debug::normal;
            else if (strcmp(p + 1, "gc") == 0)
                result.debugInfo = Debug::pretendC;
            else if (strcmp(p + 1, "gs") == 0)
                result.alwaysStackFrame = true;
            else if (strcmp(p + 1, "gt") == 0)
                error("use -profile instead of -gt\n");
            else if (strcmp(p + 1, "m32") == 0)
                result.targetModel = Model::m32;
            else if (strcmp(p + 1, "m64") == 0)
                result.targetModel = Model::m64;
            else if (strcmp(p + 1, "profile") == 0)
                result.profile = true;
            else if (strcmp(p + 1, "v") == 0)
                result.verbose = true;
            else if (strcmp(p + 1, "vdmd") == 0)
                result.vdmd = true;
#if DMDV2
            else if (strcmp(p + 1, "vtls") == 0)
                result.logTlsUse = true;
#endif
            else if (strcmp(p + 1, "v1") == 0)
            {
#if DMDV1
                // Just ignore, for backwards compatibility.
#else
                error("use DMD 1.0 series compilers for -v1 switch");
                break;
#endif
            }
            else if (strcmp(p + 1, "w") == 0)
                result.warnings = Warnings::asErrors;
            else if (strcmp(p + 1, "wi") == 0)
                result.warnings = Warnings::informational;
            else if (strcmp(p + 1, "O") == 0)
                result.optimize = true;
            else if (p[1] == 'o')
            {
                switch (p[2])
                {
                    case '-':
                        result.noObj = true;
                        break;

                    case 'd':
                        if (!p[3])
                            goto Lnoarg;
                        result.objDir = p + 3;
                        break;

                    case 'f':
                        if (!p[3])
                            goto Lnoarg;
                        result.objName = p + 3;
                        break;

                    case 'p':
                        if (p[3])
                            goto Lerror;
                        result.preservePaths = 1;
                        break;

                    case 0:
                        error("-o no longer supported, use -of or -od");
                        break;

                    default:
                        goto Lerror;
                }
            }
            else if (p[1] == 'D')
            {
                result.generateDocs = true;
                switch (p[2])
                {
                    case 'd':
                        if (!p[3])
                            goto Lnoarg;
                        result.docDir = p + 3;
                        break;
                    case 'f':
                        if (!p[3])
                            goto Lnoarg;
                        result.docName = p + 3;
                        break;

                    case 0:
                        break;

                    default:
                        goto Lerror;
                }
            }
            else if (p[1] == 'H')
            {
                result.generateHeaders = true;
                switch (p[2])
                {
                    case 'd':
                        if (!p[3])
                            goto Lnoarg;
                        result.headerDir = p + 3;
                        break;

                    case 'f':
                        if (!p[3])
                            goto Lnoarg;
                        result.headerName = p + 3;
                        break;

                    case 0:
                        break;

                    default:
                        goto Lerror;
                }
            }
            else if (p[1] == 'X')
            {
                result.generateJson = true;
                switch (p[2])
                {
                    case 'f':
                        if (!p[3])
                            goto Lnoarg;
                        result.jsonName = p + 3;
                        break;

                    case 0:
                        break;

                    default:
                        goto Lerror;
                }
            }
            else if (strcmp(p + 1, "ignore") == 0)
                result.ignoreUnsupportedPragmas = true;
            else if (strcmp(p + 1, "property") == 0)
                result.enforcePropertySyntax = true;
            else if (strcmp(p + 1, "inline") == 0)
                result.enableInline = true;
            else if (strcmp(p + 1, "lib") == 0)
                result.emitStaticLib = true;
            else if (strcmp(p + 1, "nofloat") == 0)
                result.noFloat = 1;
            else if (strcmp(p + 1, "quiet") == 0)
                result.quiet = 1;
            else if (strcmp(p + 1, "release") == 0)
                result.release = 1;
#if DMDV2
            else if (strcmp(p + 1, "noboundscheck") == 0)
                result.noBoundsChecks = 1;
#endif
            else if (strcmp(p + 1, "unittest") == 0)
                result.emitUnitTests = 1;
            else if (p[1] == 'I')
                result.modulePaths.push_back(p + 2);
            else if (p[1] == 'J')
                result.importPaths.push_back(p + 2);
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
                        result.debugLevel = (int)level;
                    }
                    else
                        result.debugIdentifiers.push_back(p + 7);
                }
                else if (p[6])
                    goto Lerror;
                else
                    result.debugFlag = true;
            }
            else if (memcmp(p + 1, "version", 5) == 0)
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
                        result.versionLevel = (int)level;
                    }
                    else
                        result.versionIdentifiers.push_back(p + 9);
                }
                else
                    goto Lerror;
            }
            else if (strcmp(p + 1, "-b") == 0)
                result.hiddenDebugB = 1;
            else if (strcmp(p + 1, "-c") == 0)
                result.hiddenDebugC = 1;
            else if (strcmp(p + 1, "-f") == 0)
                result.hiddenDebugF = 1;
            else if (strcmp(p + 1, "-help") == 0)
            {
                printUsage(originalArgv[0], ldcPath);
                exit(EXIT_SUCCESS);
            }
            else if (strcmp(p + 1, "-r") == 0)
                result.hiddenDebugR = 1;
            else if (strcmp(p + 1, "-x") == 0)
                result.hiddenDebugX = 1;
            else if (strcmp(p + 1, "-y") == 0)
                result.hiddenDebugY = 1;
            else if (p[1] == 'L')
            {
                result.linkerSwitches.push_back(p + 2);
            }
            else if (memcmp(p + 1, "defaultlib=", 11) == 0)
            {
                result.defaultLibName = p + 1 + 11;
            }
            else if (memcmp(p + 1, "debuglib=", 9) == 0)
            {
                result.debugLibName = p + 1 + 9;
            }
            else if (memcmp(p + 1, "deps=", 5) == 0)
            {
                result.moduleDepsFile = p + 1 + 5;
                if (!result.moduleDepsFile[0])
                    goto Lnoarg;
            }
            else if (memcmp(p + 1, "man", 3) == 0)
            {
                browse("http://ldc-developers.github.com");
                exit(EXIT_SUCCESS);
            }
            else if (strcmp(p + 1, "run") == 0)
            {
                result.run = true;
                int runargCount = (((int)i >= originalArgc) ? argc : originalArgc) - i - 1;
                if (runargCount)
                {
                    result.files.push_back(argv[i + 1]);
                    result.runArgs = std::vector<char*>(argv + i + 2, argv + i + runargCount + 1);
                    i += runargCount;
                }
                else
                {
                    result.run = false;
                    goto Lnoarg;
                }
            }
            else if (p[1] == 'C')
            {
                result.unknownSwitches.push_back(concat("-", p + 2));
            }
            else
            {
             Lerror:
                result.unknownSwitches.push_back(p);
                continue;

             Lnoarg:
                error("argument expected for switch '%s'", p);
                continue;
            }
        }
        else
        {
// FIXME: #if TARGET_WINDOS
            llvm::StringRef ext = ls::path::extension(p);
            if (ext.equals_lower("exe"))
            {
                result.objName = p;
                continue;
            }
// #endif
            result.files.push_back(p);
        }
    }
    if (result.files.empty())
    {
        printUsage(originalArgv[0], ldcPath);
        error("No source file specified.");
    }
    return result;
}

void pushSwitches(const char* prefix, const std::vector<char*>& vals, std::vector<const char*>& r)
{
    typedef std::vector<char*>::const_iterator It;
    for (It it = vals.begin(), end = vals.end(); it != end; ++it)
    {
        r.push_back(concat(prefix, *it));
    }
}

/**
 * Appends the LDC command line parameters corresponding to the given set of
 * parameters to r.
 */
void buildCommandLine(std::vector<const char*>& r, const Params& p)
{
    if (p.allowDeprecated) r.push_back("-d");
    if (p.compileOnly) r.push_back("-c");
    if (p.coverage) warning("Coverage report generation not yet supported by LDC.");
    if (p.emitSharedLib) r.push_back("-shared");
    if (p.pic) r.push_back("-relocation-model=pic");
    if (p.emitMap) warning("Map file generation not yet supported by LDC.");
#ifndef DMDV1
    // LDMD historically did not enable singleobj mode, so in order not to
    // break build systems as a D1 parting gift, don't change this right now.
    // This might change based on user feedback, though.
    if (!p.multiObj) r.push_back("-singleobj");
#endif
    if (p.debugInfo == Debug::normal) r.push_back("-g");
    else if (p.debugInfo == Debug::pretendC) r.push_back("-gc");
    if (p.alwaysStackFrame) r.push_back("-disable-fp-elim");
    if (p.targetModel == Model::m32) r.push_back("-m32");
    else if (p.targetModel == Model::m32) r.push_back("-m64");
    if (p.profile) warning("CPU profile generation not yet supported by LDC.");
    if (p.verbose) r.push_back("-v");
    if (p.logTlsUse) warning("-vtls not yet supported by LDC.");
    if (p.warnings == Warnings::asErrors) r.push_back("-w");
    else if (p.warnings == Warnings::informational) r.push_back("-wi");
    if (p.optimize) r.push_back("-O2");
    if (p.noObj) r.push_back("-o-");
    if (p.objDir) r.push_back(concat("-od=", p.objDir));
    if (p.objName) r.push_back(concat("-of=", p.objName));
    if (p.preservePaths) r.push_back("-op");
    if (p.generateDocs) r.push_back("-D");
    if (p.docDir) r.push_back(concat("-Dd=", p.docDir));
    if (p.docName) r.push_back(concat("-Df=", p.docName));
    if (p.generateHeaders) r.push_back("-H");
    if (p.headerDir) r.push_back(concat("-Hd=", p.headerDir));
    if (p.headerName) r.push_back(concat("-Hf=", p.headerName));
    if (p.generateJson) r.push_back("-X");
    if (p.jsonName) r.push_back(concat("-Xf=", p.jsonName));
    if (p.ignoreUnsupportedPragmas) r.push_back("-ignore");
    if (p.enforcePropertySyntax) r.push_back("-property");
    if (p.enableInline) r.push_back("-enable-inlining");
    if (p.emitStaticLib) r.push_back("-lib");
    if (p.noFloat) warning("-nofloat is ignored by LDC.");
    if (p.quiet) r.push_back("-quiet"); // Undocumented.
    if (p.release) r.push_back("-release"); // Also disables boundscheck.
    if (p.noBoundsChecks) r.push_back("-disable-boundscheck");
    if (p.emitUnitTests) r.push_back("-unittest");
    pushSwitches("-I=", p.modulePaths, r);
    pushSwitches("-J=", p.importPaths, r);
    if (p.debugFlag) r.push_back("-d-debug");
    if (p.debugLevel) r.push_back(concat("-d-debug=", p.debugLevel));
    pushSwitches("-d-debug=", p.debugIdentifiers, r);
    if (p.versionLevel) r.push_back(concat("-d-version=", p.versionLevel));
    pushSwitches("-d-version=", p.versionIdentifiers, r);
    pushSwitches("-L=", p.linkerSwitches, r);
    if (p.defaultLibName) r.push_back(concat("-defaultlib=", p.defaultLibName));
    if (p.debugLibName) r.push_back(concat("-debuglib=", p.debugLibName));
    if (p.hiddenDebugB) r.push_back("-hidden-debug-b");
    if (p.hiddenDebugC) r.push_back("-hidden-debug-c");
    if (p.hiddenDebugF) r.push_back("-hidden-debug-f");
    if (p.hiddenDebugR) r.push_back("-hidden-debug-r");
    if (p.hiddenDebugX) r.push_back("-hidden-debug-x");
    if (p.hiddenDebugY) r.push_back("-hidden-debug-y");
    r.insert(r.end(), p.unknownSwitches.begin(), p.unknownSwitches.end());
    if (p.run) r.push_back("-run");
    r.insert(r.end(), p.files.begin(), p.files.end());
    r.insert(r.end(), p.runArgs.begin(), p.runArgs.end());
}

/**
  * Returns the OS-dependent length limit for the command line when invoking
 * subprocesses.
 */
size_t maxCommandLineLen()
{
#if defined(HAVE_SC_ARG_MAX)
    // http://www.in-ulm.de/~mascheck/various/argmax – the factor 2 is just
    // a wild guess to account for the enviroment.
    return sysconf(_SC_ARG_MAX) / 2;
#elif defined(_WIN32)
    // http://blogs.msdn.com/b/oldnewthing/archive/2003/12/10/56028.aspx
    return 32767;
#else
# error "Do not know how to determine maximum command line length."
#endif
}

/**
 * Tries to locate an executable with the given name, or an invalid path if
 * nothing was found. Search paths: 1. Directory where this binary resides.
 * 2. System PATH.
 */
ls::Path locateBinary(std::string exeName, const char* argv0)
{
    ls::Path path = llvm::PrependMainExecutablePath(exeName,
        argv0, (void*)&locateBinary);
    if (path.canExecute()) return path;

    path = ls::Program::FindProgramByName(exeName);
    if (path.canExecute()) return path;

    return ls::Path();
}

static size_t addStrlen(size_t acc, const char* str)
{
    if (!str) return acc;
    return acc + strlen(str);
}

int main(int argc, char *argv[])
{
    ls::Path ldcPath = locateBinary(LDC_EXE_NAME, argv[0]);
    if (!ldcPath.isValid())
    {
        error("Could not locate "LDC_EXE_NAME" executable.");
    }

    // We need to manually set up argv[0] and the terminating NULL.
    std::vector<const char*> args;
    args.push_back(ldcPath.c_str());

    Params p = parseArgs(argc, argv, ldcPath);
    buildCommandLine(args, p);
    if (p.vdmd)
    {
        printf(" -- Invoking:");
        for (size_t i = 0; i < args.size(); ++i)
            printf(" %s", args[i]);
        puts("");
    }

    args.push_back(NULL);

    // Check if we need to write out a response file.
    size_t totalLen = std::accumulate(args.begin(), args.end(), 0, addStrlen);
    if (totalLen > maxCommandLineLen())
    {
        int rspFd;
        llvm::SmallString<128> rspPath;
        if (ls::fs::unique_file("ldmd-%%-%%-%%-%%.rsp", rspFd, rspPath) !=
            llvm::errc::success)
        {
            error("Could not open temporary response file.");
        }

        {
            llvm::raw_fd_ostream rspOut(rspFd, /*shouldClose=*/true);
            typedef std::vector<const char*>::const_iterator It;
            for (It it = args.begin(), end = args.end(); it != end; ++it)
            {
                rspOut << *it << '\n';
            }
        }

        std::string rspArg = "@";
        rspArg += rspPath.str();

        std::vector<const char*> newArgs;
        newArgs.push_back(argv[0]);
        newArgs.push_back(rspArg.c_str());
        newArgs.push_back(NULL);

        int rc = execute(ldcPath, &newArgs[0]);

        bool couldRemove;
        if (ls::fs::remove(rspPath.str(), couldRemove) != llvm::errc::success ||
            !couldRemove)
        {
            warning("Could not remove response file.");
        }

        return rc;
    }
    else
    {
        return execute(ldcPath, &args[0]);
    }
}
