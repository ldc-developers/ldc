// Test inlining of some standard library functions

// RUN: %ldc %s -c -output-ll -release -O0 -of=%t.O0.ll && FileCheck %s --check-prefix OPT0 < %t.O0.ll
// RUN: %ldc %s -c -output-ll -release -O3 -enable-cross-module-inlining -of=%t.O3.ll && FileCheck %s --check-prefix OPT3 < %t.O3.ll

extern (C): // simplify mangling for easier matching

// OPT0-LABEL: define{{.*}} @foo(
// OPT3-LABEL: define{{.*}} @foo(
int foo(size_t i)
{
    // core.bitop.bsf() is force-inlined
    import core.bitop;
    // FIXME: The OPT0 check is disabled for now, because cross-module inlining is disabled fully (also for `pragma(inline, true)` functions).
    // O PT0: call {{.*}} @llvm.cttz
    // OPT3: call {{.*}} @llvm.cttz
    return bsf(i);
    // OPT0: ret
    // OPT3: ret
}

// OPT0-LABEL: define{{.*}} @ggg(
// OPT3-LABEL: define{{.*}} @ggg(
char[] ggg(char* str)
{
    // std.string.fromStringz() is inlined when optimizing
    import std.string;
    // OPT0: call {{.*}} @{{.*}}D3std6string__T11fromStringz
    // OPT3: call {{.*}}strlen
    return fromStringz(str);
    // OPT0: ret
    // OPT3: ret
}
// OPT0: declare {{.*}}D3std6string__T11fromStringz
// OPT3: declare {{.*}}strlen
