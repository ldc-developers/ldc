/* Copyright (C) 2011-2024 by The D Language Foundation, All Rights Reserved
 * written by Walter Bright
 * https://www.digitalmars.com
 * Distributed under the Boost Software License, Version 1.0.
 * https://www.boost.org/LICENSE_1_0.txt
 * https://github.com/dlang/dmd/blob/master/src/dmd/root/array.h
 */

#pragma once

#include "dsystem.h"
#include "object.h"
#include "rmem.h"

#if IN_LLVM
#include "llvm/Support/Compiler.h"
#include <iterator>
#endif

template <typename TYPE>
struct Array
{
    d_size_t length;

  private:
    DArray<TYPE> data;
    #define SMALLARRAYCAP       1
    TYPE smallarray[SMALLARRAYCAP];    // inline storage for small arrays

#if !IN_LLVM
    Array(const Array&);
#endif

  public:
    Array()
    {
        data.ptr = NULL;
        length = 0;
        data.length = 0;
    }

    ~Array()
    {
        if (data.ptr != &smallarray[0])
            mem.xfree(data.ptr);
    }

    char *toChars() const
    {
        const char **buf = (const char **)mem.xmalloc(length * sizeof(const char *));
        d_size_t len = 2;
        for (d_size_t u = 0; u < length; u++)
        {
            buf[u] = ((RootObject *)data.ptr[u])->toChars();
            len += strlen(buf[u]) + 1;
        }
        char *str = (char *)mem.xmalloc(len);

        str[0] = '[';
        char *p = str + 1;
        for (d_size_t u = 0; u < length; u++)
        {
            if (u)
                *p++ = ',';
            len = strlen(buf[u]);
            memcpy(p,buf[u],len);
            p += len;
        }
        *p++ = ']';
        *p = 0;
        mem.xfree(buf);
        return str;
    }

    void push(TYPE ptr)
    {
        reserve(1);
        data.ptr[length++] = ptr;
    }

    void append(Array *a)
    {
        insert(length, a);
    }

    void reserve(d_size_t nentries)
    {
        //printf("Array::reserve: length = %d, data.length = %d, nentries = %d\n", (int)length, (int)data.length, (int)nentries);
        if (data.length - length < nentries)
        {
            if (data.length == 0)
            {
                // Not properly initialized, someone memset it to zero
                if (nentries <= SMALLARRAYCAP)
                {
                    data.length = SMALLARRAYCAP;
                    data.ptr = SMALLARRAYCAP ? &smallarray[0] : NULL;
                }
                else
                {
                    data.length = nentries;
                    data.ptr = (TYPE *)mem.xmalloc(data.length * sizeof(TYPE));
                }
            }
            else if (data.length == SMALLARRAYCAP)
            {
                data.length = length + nentries;
                data.ptr = (TYPE *)mem.xmalloc(data.length * sizeof(TYPE));
                memcpy(data.ptr, &smallarray[0], length * sizeof(TYPE));
            }
            else
            {
                /* Increase size by 1.5x to avoid excessive memory fragmentation
                 */
                d_size_t increment = length / 2;
                if (nentries > increment)       // if 1.5 is not enough
                    increment = nentries;
                data.length = length + increment;
                data.ptr = (TYPE *)mem.xrealloc(data.ptr, data.length * sizeof(TYPE));
            }
        }
    }

    void remove(d_size_t i)
    {
        if (length - i - 1)
            memmove(data.ptr + i, data.ptr + i + 1, (length - i - 1) * sizeof(TYPE));
        length--;
    }

    void insert(d_size_t index, Array *a)
    {
        if (a)
        {
            d_size_t d = a->length;
            reserve(d);
            if (length != index)
                memmove(data.ptr + index + d, data.ptr + index, (length - index) * sizeof(TYPE));
            memcpy(data.ptr + index, a->data.ptr, d * sizeof(TYPE));
            length += d;
        }
    }

    void insert(d_size_t index, TYPE ptr)
    {
        reserve(1);
        memmove(data.ptr + index + 1, data.ptr + index, (length - index) * sizeof(TYPE));
        data.ptr[index] = ptr;
        length++;
    }

    void setDim(d_size_t newdim)
    {
        if (length < newdim)
        {
            reserve(newdim - length);
        }
        length = newdim;
    }

    d_size_t find(TYPE ptr) const
    {
        for (d_size_t i = 0; i < length; i++)
        {
            if (data.ptr[i] == ptr)
                return i;
        }
        return SIZE_MAX;
    }

    bool contains(TYPE ptr) const
    {
        return find(ptr) != SIZE_MAX;
    }

    TYPE& operator[] (d_size_t index)
    {
#ifdef DEBUG
        assert(index < length);
#endif
        return data.ptr[index];
    }

    TYPE *tdata()
    {
        return data.ptr;
    }

    Array *copy()
    {
        Array *a = new Array();
        a->setDim(length);
        memcpy(a->data.ptr, data.ptr, length * sizeof(TYPE));
        return a;
    }

    void shift(TYPE ptr)
    {
        reserve(1);
        memmove(data.ptr + 1, data.ptr, length * sizeof(TYPE));
        data.ptr[0] = ptr;
        length++;
    }

    void zero()
    {
        memset(data.ptr, 0, length * sizeof(TYPE));
    }

    TYPE pop()
    {
        return data.ptr[--length];
    }

#if IN_LLVM
    // Define members and types like std::vector
    typedef size_t size_type;

    Array(const Array &a) : length(0), data()
    {
        setDim(a.length);
        memcpy(data.ptr, a.data.ptr, length * sizeof(TYPE));
    }

    Array &operator=(Array &a)
    {
        setDim(a.length);
        memcpy(data.ptr, a.data.ptr, length * sizeof(TYPE));
        return *this;
    }

    Array(Array &&a)
    {
        if (data.ptr != &smallarray[0])
            mem.xfree(data.ptr);
        length = a.length;
        if (a.data.ptr == &a.smallarray[0])
        {
            data.ptr = &smallarray[0];
            data.length = a.data.length;
            memcpy(data.ptr, a.data.ptr, length * sizeof(TYPE));
        }
        else
        {
            data = a.data;
            a.data.ptr = nullptr;
        }
        a.length = 0;
        a.data.length = 0;
    }

    Array &operator=(Array<TYPE> &&a)
    {
        if (data.ptr != &smallarray[0])
            mem.xfree(data.ptr);
        length = a.length;
        if (a.data.ptr == &a.smallarray[0])
        {
            data.ptr = &smallarray[0];
            data.length = a.data.length;
            memcpy(data.ptr, a.data.ptr, length * sizeof(TYPE));
        }
        else
        {
            data = a.data;
            a.data.ptr = nullptr;
        }
        a.length = 0;
        a.data.length = 0;
        return *this;
    }

    const TYPE &operator[](d_size_t index) const
    {
#ifdef DEBUG
        assert(index < length);
#endif
        return data.ptr[index];
    }

    size_type size() const
    {
        return static_cast<size_type>(length);
    }

    bool empty() const
    {
        return length == 0;
    }

    TYPE front() const
    {
        return data.ptr[0];
    }

    TYPE back() const
    {
        return data.ptr[length-1];
    }

    void push_back(TYPE a)
    {
        push(a);
    }

    void pop_back()
    {
        pop();
    }

    typedef TYPE *iterator;
    typedef const TYPE *const_iterator;
    typedef std::reverse_iterator<iterator> reverse_iterator;
    typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

    iterator begin() { return static_cast<iterator>(data.ptr); }
    iterator end() { return static_cast<iterator>(&data.ptr[length]); }
    reverse_iterator rbegin() { return reverse_iterator(end()); }
    reverse_iterator rend() { return reverse_iterator(begin()); }

    const_iterator begin() const { return static_cast<const_iterator>(data.ptr); }
    const_iterator end() const { return static_cast<const_iterator>(&data.ptr[length]); }
    const_reverse_iterator rbegin() const { return const_reverse_iterator(end()); }
    const_reverse_iterator rend() const { return const_reverse_iterator(begin()); }

    iterator erase(iterator pos)
    {
        size_t index = pos - data.ptr;
        remove(index);
        return static_cast<iterator>(&data.ptr[index]);
    }
#endif // IN_LLVM
};
