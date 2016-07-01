// Test inlining of some standard library functions

// REQUIRES: atleast_llvm307

// Test also that the tested functions are indeed not inlined at -O0 (basically verifying that we are testing something real)
// RUN: %ldc %s -c -output-ll -release -O0 -of=%t.O0.ll && FileCheck %s --check-prefix OPT0 < %t.O0.ll
// RUN: %ldc %s -c -output-ll -release -O3 -of=%t.O3.ll && FileCheck %s --check-prefix OPT3 < %t.O3.ll

extern (C): // simplify mangling for easier matching

// OPT0-LABEL: define{{.*}} @foo(
// OPT3-LABEL: define{{.*}} @foo(
int foo(size_t i)
{
    import core.bitop;
    // OPT0: call {{.*}} @{{.*}}core5bitop3bsf
    // OPT3: call {{.*}} @llvm.cttz
    return bsf(i);
    // OPT0: ret
    // OPT3: ret
}
// OPT0: declare {{.*}}core5bitop3bsf

// OPT0-LABEL: define{{.*}} @ggg(
// OPT3-LABEL: define{{.*}} @ggg(
char[] ggg(char* str)
{
    import std.string;
    // OPT0: call {{.*}} @{{.*}}std6string11fromStringz
    // OPT3: call {{.*}}strlen
    return fromStringz(str);
    // OPT0: ret
    // OPT3: ret
}
// OPT0: declare {{.*}}std6string11fromStringz
// OPT3: declare {{.*}}strlen
