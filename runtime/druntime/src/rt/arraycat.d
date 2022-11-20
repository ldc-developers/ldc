/**
 * Implementation of array copy support routines.
 *
 * Copyright: Copyright Digital Mars 2004 - 2016.
 * License:   Distributed under the
 *            $(LINK2 http://www.boost.org/LICENSE_1_0.txt, Boost Software License 1.0).
 * Authors:   Walter Bright, Sean Kelly
 * Source:    $(DRUNTIMESRC rt/_arraycat.d)
 */

module rt.arraycat;

private
{
    import core.stdc.string;
    import core.internal.util.array;
    debug(PRINTF) import core.stdc.stdio;
}

extern (C) @trusted nothrow:

version (LDC)
{
    void _d_array_slice_copy(void* dst, size_t dstlen, void* src, size_t srclen, size_t elemsz)
    {
        import ldc.intrinsics : llvm_memcpy;

        enforceRawArraysConformable("copy", elemsz, src[0..srclen], dst[0..dstlen]);
        llvm_memcpy!size_t(dst, src, dstlen * elemsz, 0);
    }
}
else
{
    void[] _d_arraycopy(size_t size, void[] from, void[] to)
    {
        debug(PRINTF) printf("f = %p,%d, t = %p,%d, size = %d\n",
                     from.ptr, from.length, to.ptr, to.length, size);

        enforceRawArraysConformable("copy", size, from, to);
        memcpy(to.ptr, from.ptr, to.length * size);
        return to;
    }
}
