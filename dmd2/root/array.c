
// Copyright (c) 1999-2013 by Digital Mars
// All Rights Reserved
// written by Walter Bright
// http://www.digitalmars.com
// License for redistribution is by either the Artistic License
// in artistic.txt, or the GNU General Public License in gnu.txt.
// See the included readme.txt for details.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "port.h"
#include "root.h"
#include "rmem.h"


/********************************* Array ****************************/

Array::Array()
{
    data = SMALLARRAYCAP ? &smallarray[0] : NULL;
    dim = 0;
    allocdim = SMALLARRAYCAP;
}

Array::~Array()
{
    if (data != &smallarray[0])
        mem.free(data);
}

void Array::mark()
{
    mem.mark(data);
    for (size_t u = 0; u < dim; u++)
        mem.mark(data[u]);      // BUG: what if arrays of Object's?
}

void Array::reserve(size_t nentries)
{
    //printf("Array::reserve: dim = %d, allocdim = %d, nentries = %d\n", (int)dim, (int)allocdim, (int)nentries);
    if (allocdim - dim < nentries)
    {
        if (allocdim == 0)
        {   // Not properly initialized, someone memset it to zero
            if (nentries <= SMALLARRAYCAP)
            {   allocdim = SMALLARRAYCAP;
                data = SMALLARRAYCAP ? &smallarray[0] : NULL;
            }
            else
            {   allocdim = nentries;
                data = (void **)mem.malloc(allocdim * sizeof(*data));
            }
        }
        else if (allocdim == SMALLARRAYCAP)
        {
            allocdim = dim + nentries;
            data = (void **)mem.malloc(allocdim * sizeof(*data));
            memcpy(data, &smallarray[0], dim * sizeof(*data));
        }
        else
        {   allocdim = dim + nentries;
            data = (void **)mem.realloc(data, allocdim * sizeof(*data));
        }
    }
}

void Array::setDim(size_t newdim)
{
    if (dim < newdim)
    {
        reserve(newdim - dim);
    }
    dim = newdim;
}

void Array::fixDim()
{
    if (dim != allocdim)
    {
        if (allocdim >= SMALLARRAYCAP)
        {
            if (dim <= SMALLARRAYCAP)
            {
                memcpy(&smallarray[0], data, dim * sizeof(*data));
                mem.free(data);
            }
            else
                data = (void **)mem.realloc(data, dim * sizeof(*data));
        }
        allocdim = dim;
    }
}

void Array::push(void *ptr)
{
    reserve(1);
    data[dim++] = ptr;
}

void *Array::pop()
{
    return data[--dim];
}

void Array::shift(void *ptr)
{
    reserve(1);
    memmove(data + 1, data, dim * sizeof(*data));
    data[0] = ptr;
    dim++;
}

void Array::insert(size_t index, void *ptr)
{
    reserve(1);
    memmove(data + index + 1, data + index, (dim - index) * sizeof(*data));
    data[index] = ptr;
    dim++;
}


void Array::insert(size_t index, Array *a)
{
    if (a)
    {
        size_t d = a->dim;
        reserve(d);
        if (dim != index)
            memmove(data + index + d, data + index, (dim - index) * sizeof(*data));
        memcpy(data + index, a->data, d * sizeof(*data));
        dim += d;
    }
}


/***********************************
 * Append array a to this array.
 */

void Array::append(Array *a)
{
    insert(dim, a);
}

void Array::remove(size_t i)
{
    if (dim - i - 1)
        memmove(data + i, data + i + 1, (dim - i - 1) * sizeof(data[0]));
    dim--;
}

char *Array::toChars()
{
    char **buf = (char **)malloc(dim * sizeof(char *));
    assert(buf);
    size_t len = 2;
    for (size_t u = 0; u < dim; u++)
    {
        buf[u] = ((Object *)data[u])->toChars();
        len += strlen(buf[u]) + 1;
    }
    char *str = (char *)mem.malloc(len);

    str[0] = '[';
    char *p = str + 1;
    for (size_t u = 0; u < dim; u++)
    {
        if (u)
            *p++ = ',';
        len = strlen(buf[u]);
        memcpy(p,buf[u],len);
        p += len;
    }
    *p++ = ']';
    *p = 0;
    free(buf);
    return str;
}

void Array::zero()
{
    memset(data,0,dim * sizeof(data[0]));
}

void *Array::tos()
{
    return dim ? data[dim - 1] : NULL;
}

int
#if _WIN32
  __cdecl
#endif
        Array_sort_compare(const void *x, const void *y)
{
    Object *ox = *(Object **)x;
    Object *oy = *(Object **)y;

    return ox->compare(oy);
}

void Array::sort()
{
    if (dim)
    {
        qsort(data, dim, sizeof(Object *), Array_sort_compare);
    }
}

Array *Array::copy()
{
    Array *a = new Array();

    a->setDim(dim);
    memcpy(a->data, data, dim * sizeof(void *));
    return a;
}

