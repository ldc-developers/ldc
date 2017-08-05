// Tests CTFE-ability of some common std.math functions.

// RUN: %ldc -c %s

import std.math;

mixin template Func1(string name) {
    enum r = mixin(name ~ "(0.5L)");
    pragma(msg, name ~ "(0.5L) = " ~ r.stringof);
    enum d = mixin(name ~ "(0.5)");
    pragma(msg, name ~ "(0.5)  = " ~ d.stringof);
    enum f = mixin(name ~ "(0.5f)");
    pragma(msg, name ~ "(0.5f) = " ~ f.stringof);
}

mixin template Func2(string name) {
    enum r = mixin(name ~ "(0.5L, -2.5L)");
    pragma(msg, name ~ "(0.5L, -2.5L) = " ~ r.stringof);
    enum d = mixin(name ~ "(0.5, -2.5)");
    pragma(msg, name ~ "(0.5,  -2.5)  = " ~ d.stringof);
    enum f = mixin(name ~ "(0.5f, -2.5f)");
    pragma(msg, name ~ "(0.5f, -2.5f) = " ~ f.stringof);
}

mixin template Func2_int(string name) {
    enum r = mixin(name ~ "(0.5L, 2)");
    pragma(msg, name ~ "(0.5L, 2) = " ~ r.stringof);
    enum d = mixin(name ~ "(0.5, 2)");
    pragma(msg, name ~ "(0.5,  2) = " ~ d.stringof);
    enum f = mixin(name ~ "(0.5f, 2)");
    pragma(msg, name ~ "(0.5f, 2) = " ~ f.stringof);
}

mixin template Func3(string name) {
    enum r = mixin(name ~ "(0.5L, -2.5L, 6.25L)");
    pragma(msg, name ~ "(0.5L, -2.5L, 6.25L) = " ~ r.stringof);
    enum d = mixin(name ~ "(0.5, -2.5, 6.25)");
    pragma(msg, name ~ "(0.5,  -2.5,  6.25)  = " ~ d.stringof);
    enum f = mixin(name ~ "(0.5f, -2.5f, 6.25f)");
    pragma(msg, name ~ "(0.5f, -2.5f, 6.25f) = " ~ f.stringof);
}

void main()
{
    { mixin Func1!"isNaN"; }
    { mixin Func1!"isInfinity"; }
    { mixin Func1!"isFinite"; }

    { mixin Func1!"abs"; }
    { mixin Func1!"fabs"; }
    { mixin Func1!"sqrt"; }

    { mixin Func1!"sin"; }
    { mixin Func1!"cos"; }
    { mixin Func1!"tan"; }
    { mixin Func1!"cosh"; }
    { mixin Func1!"asinh"; }
    { mixin Func1!"acosh"; }
    { mixin Func1!"atanh"; }

    { mixin Func1!"ceil"; }
    { mixin Func1!"floor"; }
    { mixin Func1!"round"; }
    { mixin Func1!"trunc"; }
    { mixin Func1!"rint"; }
    { mixin Func1!"nearbyint"; }

    { mixin Func1!"exp"; }
    { mixin Func1!"exp2"; }
    { mixin Func2_int!"ldexp"; }
    { mixin Func1!"log"; }
    { mixin Func1!"log2"; }
    { mixin Func1!"log10"; }

    { mixin Func2!"pow"; }
    { mixin Func2_int!"pow"; }

    { mixin Func2!"fmin"; }
    { mixin Func2!"fmax"; }
    { mixin Func2!"copysign"; }

    { mixin Func3!"fma"; }

    static assert(2.0 ^^ 8 == 256.0);
}
