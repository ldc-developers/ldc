
// Compiler implementation of the D programming language
// Copyright (c) 1999-2012 by Digital Mars
// All Rights Reserved
// written by Walter Bright
// http://www.digitalmars.com
// License for redistribution is by either the Artistic License
// in artistic.txt, or the GNU General Public License in gnu.txt.
// See the included readme.txt for details.

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <assert.h>
#include <limits.h>
#include <string.h>
#if IN_LLVM
#include <cstdarg>
#endif

#if POSIX
#include <errno.h>
#endif

#include "rmem.h"
#include "root.h"

#include "mars.h"
#include "module.h"
#include "mtype.h"
#include "id.h"
#include "cond.h"
#include "expression.h"
#include "lexer.h"
#include "json.h"

#if WINDOWS_SEH
#include <windows.h>
long __cdecl __ehfilter(LPEXCEPTION_POINTERS ep);
#endif


int response_expand(int *pargc, char ***pargv);
void browse(const char *url);
void getenv_setargv(const char *envvar, int *pargc, char** *pargv);

void obj_start(char *srcfile);
void obj_end(Library *library, File *objfile);

void printCtfePerformanceStats();

Global global;

Global::Global()
{
    mars_ext = "d";
    sym_ext  = "d";
    hdr_ext  = "di";
    doc_ext  = "html";
    ddoc_ext = "ddoc";
    json_ext = "json";
    map_ext  = "map";

// LDC
    ll_ext  = "ll";
    bc_ext  = "bc";
    s_ext   = "s";
    obj_ext = "o";
    obj_ext_alt = "obj";

    copyright = "Copyright (c) 1999-2012 by Digital Mars and Tomas Lindquist Olsen";
    written = "written by Walter Bright and Tomas Lindquist Olsen";
    version = "v1.075";
    ldc_version = "trunk";
    llvm_version = "LLVM "LDC_LLVM_VERSION_STRING;
    global.structalign = STRUCTALIGN_DEFAULT;

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


char *Loc::toChars() const
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

void verror(Loc loc, const char *format, va_list ap, const char *p1, const char *p2)
{
    if (!global.gag)
    {
        char *p = loc.toChars();

        if (*p)
            fprintf(stdmsg, "%s: ", p);
        mem.free(p);

        fprintf(stdmsg, "Error: ");
        if (p1)
            fprintf(stdmsg, "%s ", p1);
        if (p2)
            fprintf(stdmsg, "%s ", p2);
#if _MSC_VER
        // MS doesn't recognize %zu format
        OutBuffer tmp;
        tmp.vprintf(format, ap);
        fprintf(stdmsg, "%s", tmp.toChars());
#else
        vfprintf(stdmsg, format, ap);
#endif
        fprintf(stdmsg, "\n");
        fflush(stdmsg);
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
    {
        fprintf(stdmsg, "%s:        ", loc.toChars());
#if _MSC_VER
        // MS doesn't recognize %zu format
        OutBuffer tmp;
        tmp.vprintf(format, ap);
        fprintf(stdmsg, "%s", tmp.toChars());
#else
        vfprintf(stdmsg, format, ap);
#endif
        fprintf(stdmsg, "\n");
        fflush(stdmsg);
    }
}

void vwarning(Loc loc, const char *format, va_list ap)
{
    if (global.params.warnings && !global.gag)
    {
        char *p = loc.toChars();

        if (*p)
            fprintf(stdmsg, "%s: ", p);
        mem.free(p);

        fprintf(stdmsg, "Warning: ");
#if _MSC_VER
        // MS doesn't recognize %zu format
        OutBuffer tmp;
        tmp.vprintf(format, ap);
        fprintf(stdmsg, "%s", tmp.toChars());
#else
        vfprintf(stdmsg, format, ap);
#endif
        fprintf(stdmsg, "\n");
        fflush(stdmsg);
//halt();
        if (global.params.warnings == 1)
            global.warnings++;  // warnings don't count if gagged
    }
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

/***********************************
 * Parse and append contents of environment variable envvar
 * to argc and argv[].
 * The string is separated into arguments, processing \ and ".
 */

void getenv_setargv(const char *envvar, int *pargc, char** *pargv)
{
    char *p;

    int instring;
    int slash;
    char c;

    char *env = getenv(envvar);
    if (!env)
        return;

    env = mem.strdup(env);      // create our own writable copy

    int argc = *pargc;
    Strings *argv = new Strings();
    argv->setDim(argc);

    size_t argc_left = 0;
    for (size_t i = 0; i < argc; i++) {
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
            argv->data[i] = (void *)(*pargv)[i];
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
