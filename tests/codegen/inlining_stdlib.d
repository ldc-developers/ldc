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
    // OPT0: call {{.*}} @llvm.cttz
    // OPT3: call {{.*}} @llvm.cttz
    return bsf(i);
    // OPT0: ret
    // OPT3: ret
}

// OPT0-LABEL: define{{.*}} @ggg(
// OPT3-LABEL: define{{.*}} @ggg(
double ggg(double r)
{
    // std.math.nextDown() is inlined when optimizing
    import std.math;
    // OPT0: call {{.*}} @{{.*}}D3std4math10operations8nextDown
    // OPT3: call {{.*}} @{{.*}}D3std4math10operations6nextUp
    return nextDown(r);
    // OPT0: ret
    // OPT3: ret
}
// OPT0: declare {{.*}}D3std4math10operations8nextDown
// OPT3: declare {{.*}}D3std4math10operations6nextUp
