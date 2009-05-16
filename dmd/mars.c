
// Compiler implementation of the D programming language
// Copyright (c) 1999-2009 by Digital Mars
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
#include <string>
#include <cstdarg>

#if POSIX
#include <errno.h>
#elif _WIN32
#include <windows.h>
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

#include "gen/revisions.h"

Global global;

Global::Global()
{
    mars_ext = "d";
    sym_ext  = "d";
    hdr_ext  = "di";
    doc_ext  = "html";
    ddoc_ext = "ddoc";

// LDC
    ll_ext  = "ll";
    bc_ext  = "bc";
    s_ext   = "s";
    obj_ext = "o";
#if _WIN32
    obj_ext_alt = "obj";
#endif

    copyright = "Copyright (c) 1999-2009 by Digital Mars and Tomas Lindquist Olsen";
    written = "written by Walter Bright and Tomas Lindquist Olsen";
    version = "v1.045";
    ldc_version = LDC_REV;
    llvm_version = LLVM_REV_STR;
    global.structalign = 8;

    // This should only be used as a global, so the other fields are
    // automatically initialized to zero when the program is loaded.
    // In particular, DO NOT zero-initialize .params here (like DMD
    // does) because command-line options initialize some of those
    // fields to non-zero defaults, and do so from constructors that
    // may run before this one.
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

/**************************************
 * Print error message and exit.
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
    if (global.params.warnings && !global.gag)
    {
        va_list ap;
        va_start(ap, format);
        vwarning(loc, format, ap);
        va_end( ap );
    }
}

void verror(Loc loc, const char *format, va_list ap)
{
    if (!global.gag)
    {
	char *p = loc.toChars();

	if (*p)
	    fprintf(stdmsg, "%s: ", p);
	mem.free(p);

	fprintf(stdmsg, "Error: ");
	vfprintf(stdmsg, format, ap);
	fprintf(stdmsg, "\n");
	fflush(stdmsg);
    }
    global.errors++;
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
        vfprintf(stdmsg, format, ap);
        fprintf(stdmsg, "\n");
        fflush(stdmsg);
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
    *(char*)0=0;
#endif
}

/***********************************
 * Parse and append contents of environment variable envvar
 * to argc and argv[].
 * The string is separated into arguments, processing \ and ".
 */

void getenv_setargv(const char *envvar, int *pargc, char** *pargv)
{
    char *env;
    char *p;
    Array *argv;
    int argc;

    int wildcard;		// do wildcard expansion
    int instring;
    int slash;
    char c;
    int j;

    env = getenv(envvar);
    if (!env)
	return;

    env = mem.strdup(env);	// create our own writable copy

    argc = *pargc;
    argv = new Array();
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
            argv->data[i] = (void *)(*pargv)[i];
        }
    }
    // HACK to stop required values from command line being drawn from DFLAGS
    argv->push((char*)"");
    argc++;

    j = 1;			// leave argv[0] alone
    while (1)
    {
	wildcard = 1;
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
		argv->push(env);		// append
		//argv->insert(j, env);		// insert at position j
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
			    {	p--;
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
				//wildcardexpand();	// not implemented
			    break;

			case '\\':
			    slash++;
			    *p++ = c;
			    continue;

			case 0:
			    *p = 0;
			    //if (wildcard)
				//wildcardexpand();	// not implemented
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
    *pargv = (char **)argv->data;
}
