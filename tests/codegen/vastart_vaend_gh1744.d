// Test that va_end is called when returning from a variadic function.

// Because the IR is kind of ugly, testing at -O0 is very brittle. Instead we
// test that at -O3, LLVM was able to analyse the function correctly and
// optimize-out the va_start and va_end calls and remove the call to
// return_two (Github #1744).
// RUN: %ldc %s -c -output-ll -O3 -of=%t.O3.ll \
// RUN:   && FileCheck %s --check-prefix OPT3 < %t.O3.ll

module mod;

// OPT3-LABEL: define {{.*}} @{{.*}}void_three_return_paths
void void_three_return_paths(int a, ...)
{
    // OPT3: call void @llvm.va_start({{.*}} %[[VA:[0-9]+]])
    // OPT3-NOT: return_two
    return_two();

    if (a > 0)
    {
        throw new Exception("");
        return;
    }

    // There are two control paths (normal return, exception resume) that
    // should call va_end.
    // OPT3: call void @llvm.va_end({{.*}} %[[VA]])
    // OPT3: call void @llvm.va_end({{.*}} %[[VA]])
}

// OPT3-LABEL: define {{.*}} @{{.*}}return_two
int return_two(...)
{
    // OPT3-NOT: va_start
    // OPT3-NOT: va_end
    // OPT3: ret i32 2
    return 2;
}

// Github #1744:
// OPT3-LABEL: define {{.*}} @{{.*}}two_four
int two_four()
{
    // OPT3-NOT: return_two
    // OPT3: ret i32 8
    return return_two() * 4;
}
