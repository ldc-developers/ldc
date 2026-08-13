//===-- tools/runtime-adapt/source/ldcmods/libfuzzer.d ------------*- D -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// ldc/libfuzzer.di ← driver/cl_options_sanitizers.cpp (-fsanitize=fuzzer).
//
//===----------------------------------------------------------------------===//

module ldcmods.libfuzzer;

import compilerparse;
import ldcmods.common;

bool wantLibfuzzer(const CompilerModel m)
{
    return hasLdcModule(m, "libfuzzer") || m.hasFuzzer || m.hasSanitizer;
}

string renderLibfuzzer(const CompilerModel, string tag)
{
    return banner(tag) ~ q"D
module ldc.libfuzzer;

mixin template DefineTestOneInput(alias FuzzTarget, bool initializeDRuntime = true)
    if (is(typeof(&FuzzTarget) == int function(in ubyte[])))
{
    static if (initializeDRuntime)
    {
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

    pragma(mangle, "LLVMFuzzerTestOneInput")
    extern (C) int LLVMFuzzerTestOneInput(const(ubyte*) data, size_t size)
    {
        try
            return FuzzTarget(data[0 .. size]);
        catch (Throwable t)
            _d_print_throwable(t);
        assert(0);
    }

    pragma(mangle, "_d_print_throwable")
    extern (C) void _d_print_throwable(Throwable t);
}
D";
}
