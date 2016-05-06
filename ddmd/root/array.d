// Compiler implementation of the D programming language
// Copyright (c) 1999-2015 by Digital Mars
// All Rights Reserved
// written by Walter Bright
// http://www.digitalmars.com
// Distributed under the Boost Software License, Version 1.0.
// http://www.boost.org/LICENSE_1_0.txt

module ddmd.root.array;

import core.stdc.string;

import ddmd.root.rmem;

extern (C++) struct Array(T)
{
public:
    size_t dim;
    T* data;

private:
    size_t allocdim;
    enum SMALLARRAYCAP = 1;
    T[SMALLARRAYCAP] smallarray; // inline storage for small arrays

public:
    ~this()
    {
        if (data != &smallarray[0])
            mem.xfree(data);
    }

    const(char)* toChars()
    {
        static if (is(typeof(T.init.toChars())))
        {
            const(char)** buf = cast(const(char)**)mem.xmalloc(dim * (char*).sizeof);
            size_t len = 2;
            for (size_t u = 0; u < dim; u++)
            {
                buf[u] = data[u].toChars();
                len += strlen(buf[u]) + 1;
            }
            char* str = cast(char*)mem.xmalloc(len);

            str[0] = '[';
            char* p = str + 1;
            for (size_t u = 0; u < dim; u++)
            {
                if (u)
                    *p++ = ',';
                len = strlen(buf[u]);
                memcpy(p, buf[u], len);
                p += len;
            }
            *p++ = ']';
            *p = 0;
            mem.xfree(buf);
            return str;
        }
        else
        {
            assert(0);
        }
    }

    void push(T ptr)
    {
        reserve(1);
        data[dim++] = ptr;
    }

    void append(typeof(this)* a)
    {
        insert(dim, a);
    }

    void reserve(size_t nentries)
    {
        //printf("Array::reserve: dim = %d, allocdim = %d, nentries = %d\n", (int)dim, (int)allocdim, (int)nentries);
        if (allocdim - dim < nentries)
        {
            if (allocdim == 0)
            {
                // Not properly initialized, someone memset it to zero
                if (nentries <= SMALLARRAYCAP)
                {
                    allocdim = SMALLARRAYCAP;
                    data = SMALLARRAYCAP ? smallarray.ptr : null;
                }
                else
                {
                    allocdim = nentries;
                    data = cast(T*)mem.xmalloc(allocdim * (*data).sizeof);
                }
            }
            else if (allocdim == SMALLARRAYCAP)
            {
                allocdim = dim + nentries;
                data = cast(T*)mem.xmalloc(allocdim * (*data).sizeof);
                memcpy(data, smallarray.ptr, dim * (*data).sizeof);
            }
            else
            {
                allocdim = dim + nentries;
                data = cast(T*)mem.xrealloc(data, allocdim * (*data).sizeof);
            }
        }
    }

    void remove(size_t i)
    {
        if (dim - i - 1)
            memmove(data + i, data + i + 1, (dim - i - 1) * (data[0]).sizeof);
        dim--;
    }

    void insert(size_t index, typeof(this)* a)
    {
        if (a)
        {
            size_t d = a.dim;
            reserve(d);
            if (dim != index)
                memmove(data + index + d, data + index, (dim - index) * (*data).sizeof);
            memcpy(data + index, a.data, d * (*data).sizeof);
            dim += d;
        }
    }

    void insert(size_t index, T ptr)
    {
        reserve(1);
        memmove(data + index + 1, data + index, (dim - index) * (*data).sizeof);
        data[index] = ptr;
        dim++;
    }

    void setDim(size_t newdim)
    {
        if (dim < newdim)
        {
            reserve(newdim - dim);
        }
        dim = newdim;
    }

    ref inout(T) opIndex(size_t i) inout
    {
        return data[i];
    }

    inout(T)* tdata() inout
    {
        return data;
    }

    Array!T* copy() const
    {
        auto a = new Array!T();
        a.setDim(dim);
        memcpy(a.data, data, dim * (void*).sizeof);
        return a;
    }

    void shift(T ptr)
    {
        reserve(1);
        memmove(data + 1, data, dim * (*data).sizeof);
        data[0] = ptr;
        dim++;
    }

    void zero()
    {
        data[0 .. dim] = T.init;
    }

    T pop()
    {
        return data[--dim];
    }

    int apply(int function(T, void*) fp, void* param)
    {
        static if (is(typeof(T.init.apply(fp, null))))
        {
            for (size_t i = 0; i < dim; i++)
            {
                T e = data[i];
                if (e)
                {
                    if (e.apply(fp, param))
                        return 1;
                }
            }
            return 0;
        }
        else
            assert(0);
    }

    extern (D) inout(T)[] opSlice() inout
    {
        return data[0 .. dim];
    }

    extern (D) inout(T)[] opSlice(size_t a, size_t b) inout
    {
        assert(a <= b && b <= dim);
        return data[a .. b];
    }
}

struct BitArray
{
    size_t length() const
    {
        return len;
    }

    void length(size_t nlen)
    {
        immutable obytes = (len + 7) / 8;
        immutable nbytes = (nlen + 7) / 8;
        ptr = cast(size_t*)mem.xrealloc(ptr, nbytes);
        if (nbytes > obytes)
            (cast(ubyte*)ptr)[obytes .. nbytes] = 0;
        len = nlen;
    }

    bool opIndex(size_t idx) const
    {
        import core.bitop : bt;

        assert(idx < length);
        return !!bt(ptr, idx);
    }

    void opIndexAssign(bool val, size_t idx)
    {
        import core.bitop : btc, bts;

        assert(idx < length);
        if (val)
            bts(ptr, idx);
        else
            btc(ptr, idx);
    }

    @disable this(this);

    ~this()
    {
        mem.xfree(ptr);
    }

private:
    size_t len;
    size_t *ptr;
}
