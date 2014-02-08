
// Copyright (c) 2000-2012 by Digital Mars
// All Rights Reserved
// written by Walter Bright
// http://www.digitalmars.com
// License for redistribution is by either the Artistic License
// in artistic.txt, or the GNU General Public License in gnu.txt.
// See the included readme.txt for details.

#ifndef ROOT_MEM_H
#define ROOT_MEM_H

#if defined(IN_LLVM) && (LDC_LLVM_VER >= 303)
#include "llvm/Config/config.h"
#endif
#include <stddef.h>     // for size_t

struct Mem
{
    Mem() { }

    char *strdup(const char *s);
    void *malloc(size_t size);
    void *calloc(size_t size, size_t n);
    void *realloc(void *p, size_t size);
    void free(void *p);
    void *mallocdup(void *o, size_t size);
    void error();
};

extern Mem mem;

#endif /* ROOT_MEM_H */
