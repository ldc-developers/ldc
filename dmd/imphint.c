

// Compiler implementation of the D programming language
// Copyright (c) 2010 by Digital Mars
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
#include <string.h>

/******************************************
 * Looks for undefined identifier s to see
 * if it might be undefined because an import
 * was not specified.
 * Not meant to be a comprehensive list of names in each module,
 * just the most common ones.
 */

const char *importHint(const char *s)
{
#if DMDV1
    static const char *modules[] =
    {   "tango.stdc.stdio",
        "tango.io.Stdout",
        "tango.math.Math",
    };
    static const char *names[] =
    {
        "printf", NULL,
        "Stdout", NULL,
        "sin", "cos", "sqrt", "abs", NULL,
    };
#else
    static const char *modules[] =
    {   "core.stdc.stdio",
        "std.stdio",
        "std.math",
    };
    static const char *names[] =
    {
        "printf", NULL,
        "writeln", NULL,
        "sin", "cos", "sqrt", "fabs", NULL,
    };
#endif
    int m = 0;
    for (int n = 0; n < sizeof(names)/sizeof(names[0]); n++)
    {
        const char *p = names[n];
        if (p == NULL)
        {   m++;
            continue;
        }
        assert(m < sizeof(modules)/sizeof(modules[0]));
        if (strcmp(s, p) == 0)
            return modules[m];
    }
    return NULL;        // didn't find it
}

#if UNITTEST

void unittest_importHint()
{
    const char *p;

    p = importHint("printf");
    assert(p);
    p = importHint("fabs");
    assert(p);
    p = importHint("xxxxx");
    assert(!p);
}

#endif
