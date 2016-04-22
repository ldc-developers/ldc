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

    char* toChars()
    {
        static if (is(typeof(T.init.toChars())))
        {
            char** buf = cast(char**)mem.xmalloc(dim * (char*).sizeof);
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

    ref T opIndex(size_t i)
    {
        return data[i];
    }

    T* tdata()
    {
        return data;
    }

    typeof(this)* copy()
    {
        auto a = new typeof(this)();
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

    extern (D) T[] opSlice()
    {
        return data[0 .. dim];
    }
}

version(IN_LLVM)
{

/+ A container that preserves the order in which elements were added
 + and that can do sub-linear search for whether an element is already in the container.
 + The sub-linear search behavior requires extra memory and is only enabled when
 + constructed with fastSearch = true.
 +/
extern (C++) struct ArrayTree(T)
{
private:
    Array!T *array;

    import std.container.rbtree : RedBlackTree;
    alias TreeType = RedBlackTree!(void*); // RedBlackTree!(T, "cast(void*)a < cast(void*)b") is broken in Phobos version 2.070
    TreeType tree;

public:
    static __gshared bool fastSearch = false; // Enable sub-linear lookup

    // Disable default construction, for default construction use `new ArrayTree!T(null)`.
    @disable this();

    // Construct and use the array as initializer. No deep copy is made.
    this(Array!T *a)
    {
        if (a)
        {
            array = a;
        }
        else
        {
            array = new Array!T();
        }

        if (fastSearch)
        {
            tree = new TreeType();
            foreach (elem; *array)
            {
                tree.stableInsert(cast(void*)elem);
            }
        }
    }

    ~this()
    {
        destroy(array);
        destroy(tree);
    }

    // Construct a new ArrayTree is a!=null, otherwise return null. No deep copy is made.
    static typeof(this)* convert(Array!T *a)
    {
        if (a)
        {
            return new typeof(this)(a);
        }
        else
        {
            return null;
        }
    }

    size_t dim() @property
    {
        return array.dim;
    }

    // Necessary because mangling of push(T) is broken for 2.070
    void push(void* ptr)
    {
        push(cast(T)ptr);
    }

    void push(T ptr)
    {
        array.push(ptr);
        if (tree)
            tree.stableInsert(cast(void*)ptr);
    }

    // Insert element iff it is not inside yet. Returns true if the element was inserted.
    // This is sublinear if ArrayTree was constructed with fastSearch = true,
    // otherwise a linear search is performed.
    bool pushIfAbsent(T ptr)
    {
        if (tree)
        {
            size_t inserted = tree.stableInsert(cast(void*)ptr);
            if (inserted > 0)
                array.push(ptr);
            return inserted > 0;
        }
        else
        {
            foreach (elem; *array)
            {
                if (elem == ptr)
                {
                    return false;
                }
            }
            push(ptr);
            return true;
        }
    }

    void append(typeof(this) at)
    {
        append(at.getArrayPtr());
    }

    void append(Array!T* at)
    {
        array.append(at);
        if (tree)
        {
            foreach (elem; *at)
            {
                tree.stableInsert(cast(void*)elem);
            }
        }
    }

    void remove(size_t i)
    {
        array.remove(i);
        if (tree)
            tree.removeKey(cast(void*)(*array)[i]);
    }

    void insert(size_t index, typeof(this) at)
    {
        array.insert(index, at.array);
        if (tree)
        {
            foreach (elem; at)
            {
                tree.stableInsert(cast(void*)elem);
            }
        }
    }

    void insert(size_t index, T ptr)
    {
        array.insert(index, ptr);
        if (tree)
            tree.stableInsert(cast(void*)ptr);
    }

    ref T opIndex(size_t i)
    {
        return (*array)[i];
    }

    // Gets a pointer (!) to the array, not a copy of the array
    typeof(array) getArrayPtr()
    {
        return array;
    }

    // Makes a deep copy
    typeof(this)* copy()
    {
        auto at = new typeof(this)(null);
        at.array.setDim(array.dim);
        memcpy(at.array.data, array.data, array.dim * (void*).sizeof);
        at.tree = tree ? tree.dup() : null;
        return at;
    }

    void shift(T ptr)
    {
        array.shift(ptr);
        if (tree)
            tree.stableInsert(cast(void*)ptr);
    }

    int apply(int function(T, void*) fp, void* param)
    {
        return array.apply(fp, param);
    }

    extern (D) T[] opSlice()
    {
        return (*array)[];
    }
}

}
