// RUN: %ldc -o- -mdcompute-targets=cuda-430 %s
// REQUIRES: target_NVPTX
@compute(CompileFor.deviceOnly) module dcompute_enum;
pragma(LDC_no_moduleinfo);
import ldc.dcompute;
template isUnsigned(T)
{
    static if (!__traits(isUnsigned, T))
        enum isUnsigned = false;
    else static if (is(T U == enum))
        enum isUnsigned = isUnsigned!U;
    else
        enum isUnsigned = __traits(isZeroInit, T) // Not char, wchar, or dchar.
            && !is(immutable T == immutable bool) && !is(T == __vector);
}
@kernel void tst (uint* dst)
{
    dst[0] = isUnsigned!(typeof(dst[0]));
}
