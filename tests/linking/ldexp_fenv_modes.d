// All-mode numerical coverage is separate from ldexp_fenv.d, whose minimal
// shape is needed to expose result reuse across rounding-mode changes.
// REQUIRES: LTO
// RUN: %ldc -O0 %s %S/../../runtime/druntime/src/core/math.d -of=%t-O0%exe && %t-O0%exe
// RUN: %ldc -O2 -flto=full %s %S/../../runtime/druntime/src/core/math.d -of=%t-full%exe && %t-full%exe
// RUN: %ldc -O2 -flto=thin %s %S/../../runtime/druntime/src/core/math.d -of=%t-thin%exe && %t-thin%exe

import core.math : ldexp;
import core.stdc.fenv;

void check(T)()
{
    static if (is(T == float)) alias Bits = uint;
    else alias Bits = ulong;
    enum Bits normal = Bits(1) << (T.mant_dig - 1);
    enum Bits sign = Bits(1) << (Bits.sizeof * 8 - 1);
    enum Bits two = Bits(T.max_exp) << (T.mant_dig - 1);
    enum int tinyExponent = T.min_exp - T.mant_dig;
    // FE_ALL_EXCEPT includes the nonstandard denormal-operand flag on some
    // x86 runtimes. The unchanged subnormal-input normalization can set it.
    // Check the five standard exceptions, including underflow and inexact.
    enum standardExceptions = FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW
        | FE_UNDERFLOW | FE_INEXACT;

    T fromBits(Bits bits) { return *cast(T*)&bits; }
    struct Case { T value; int exponent; Bits[4] expected; }
    // Expected magnitudes in nearest, upward, downward, toward-zero order.
    // The tiny cases are exact rational multiples of the minimum subnormal.
    Case[] cases = [
        Case(T(0.609375), tinyExponent + 3, [5, 5, 4, 4]), // 4.875 units
        Case(T(0.5625), tinyExponent + 3, [4, 5, 4, 4]), // 4.5: even down
        Case(T(0.6875), tinyExponent + 3, [6, 6, 5, 5]), // 5.5: even up
        Case(T(0.625), tinyExponent + 3, [5, 5, 5, 5]),
        Case(T(1), tinyExponent - 2, [0, 1, 0, 0]),
        Case(T(1), int.min, [0, 1, 0, 0]),
        Case(T(1) + T.epsilon, tinyExponent - 1, [1, 1, 0, 0]),
        // Around 2, scaling by 2^(min_exp-2) gives magnitudes in
        // minimum-subnormal units: normal-2, normal-1.5, normal-1,
        // normal-0.5, normal, normal+1. Derive inputs from bits so
        // their construction cannot depend on the active rounding mode.
        Case(fromBits(two - 4), T.min_exp - 2,
             [normal - 2, normal - 2, normal - 2, normal - 2]),
        Case(fromBits(two - 3), T.min_exp - 2,
             [normal - 2, normal - 1, normal - 2, normal - 2]),
        Case(fromBits(two - 2), T.min_exp - 2,
             [normal - 1, normal - 1, normal - 1, normal - 1]),
        Case(fromBits(two - 1), T.min_exp - 2,
             [normal, normal, normal - 1, normal - 1]),
        Case(fromBits(two), T.min_exp - 2,
             [normal, normal, normal, normal]),
        Case(fromBits(two + 1), T.min_exp - 2,
             [normal + 1, normal + 1, normal + 1, normal + 1]),
        Case(fromBits(1), 1, [2, 2, 2, 2]),
        Case(fromBits(normal - 1), 1,
             [2 * normal - 2, 2 * normal - 2, 2 * normal - 2, 2 * normal - 2]),
        Case(T.min_normal, 0, [normal, normal, normal, normal]),
        Case(T(0), -1, [0, 0, 0, 0]),
    ];
    foreach (negative; [false, true])
    foreach (mi, mode; [FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO])
    foreach (c; cases)
    foreach (oldFlags; [0, FE_DIVBYZERO, FE_DIVBYZERO | FE_INEXACT | FE_UNDERFLOW])
    {
        Bits inputBits = *cast(Bits*)&c.value | (negative ? sign : 0);
        T input = fromBits(inputBits);
        assert(fesetround(mode) == 0);
        assert(feclearexcept(FE_ALL_EXCEPT) == 0);
        assert(feraiseexcept(oldFlags) == 0);
        int initialFlags = fetestexcept(standardExceptions);
        assert(initialFlags == oldFlags,
            "failed to initialize exception flags");
        T result = ldexp(input, c.exponent);
        int flags = fetestexcept(standardExceptions);
        auto magnitudeMode = negative && (mi == 1 || mi == 2) ? 3 - mi : mi;
        Bits expected = c.expected[magnitudeMode] | (negative ? sign : 0);
        assert(*cast(Bits*)&result == expected);
        // Preserve this implementation's float/double behavior and old flags.
        // This is not a promise about arbitrary ldexp paths or wider formats.
        assert(flags == oldFlags);
    }
}

int main()
{
    fenv_t saved;
    assert(fegetenv(&saved) == 0);
    scope(exit) fesetenv(&saved);
    check!float();
    check!double();
    return 0;
}
