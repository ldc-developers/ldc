/**
 * Contains convenience functionality for use with libFuzzer.
 *
 * Copyright: Authors 2018
 * License:   $(LINK2 http://www.boost.org/LICENSE_1_0.txt, Boost License 1.0)
 * Authors:   LDC Team
 */

module ldc.libfuzzer;

import std.typecons : Flag, Yes;

/**
 * Defines the necessary code to initialize the D runtime and calls the
 * FuzzTarget function.
 * FuzzTarget needs to be a function with signature:
 *     int function(in ubyte[] data)
 * The returned value should be 0. Non-zero values are reserved for future
 * use. See https://llvm.org/docs/LibFuzzer.html#fuzz-target .
 *
 * Example:
 * ---
 * import ldc.libfuzzer;
 * mixin DefineTestOneInput!fuzz_target;
 *
 * int fuzz_target(in ubyte[] data)
 * {
 *     // Write your test here...
 *     return 0;
 * }
 * ---
 */

mixin template DefineTestOneInput(alias FuzzTarget, Flag!"initializeDRuntime" initializeDRuntime = Yes.initializeDRuntime)
    if (is(typeof(&FuzzTarget) == int function(in ubyte[])))
{
    static if (initializeDRuntime)
    {
        // Do druntime initialization and de-initialization through module ctor/dtor.
        // Choose a low priority = 10, to hopefully init before (de-init after) standard D module ctors (dtors).
        __gshared static bool runtimeInitialized = false;
        pragma(crt_constructor, 10)
        void initDRuntime()
        {
            import core.runtime : rt_init;
            rt_init();
            runtimeInitialized = true;
        }

        pragma(crt_destructor, 10)
        void terminateDRuntime()
        {
            import core.runtime : rt_term;
            if (runtimeInitialized)
                runtimeInitialized = !rt_term();
        }
    }

    // libFuzzer's user entry point
    pragma(mangle, "LLVMFuzzerTestOneInput") // Work around https://issues.dlang.org/show_bug.cgi?id=12575
    extern (C) int LLVMFuzzerTestOneInput(const(ubyte*) data, size_t size)
    {
        try
        {
            // Call the user's function
            return FuzzTarget(data[0 .. size]);
        }
        catch (Throwable t)
        {
            _d_print_throwable(t);
        }
        // We only reach here when an exception was caught.
        assert(0);
    }

    pragma(mangle, "_d_print_throwable") // Work around https://issues.dlang.org/show_bug.cgi?id=12575
    extern (C) void _d_print_throwable(Throwable t);
}
