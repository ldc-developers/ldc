// REQUIRES: target_X86

// RUN: %ldc -mtriple=x86_64-linux-gnu -output-ll -of=%t_64.ll %s && FileCheck %s --check-prefix=M64 < %t_64.ll
// RUN: %ldc -mtriple=i686-linux-gnu   -output-ll -of=%t_32.ll %s && FileCheck %s --check-prefix=M32 < %t_32.ll

void foo()
{
    // M64: %1 = getelementptr inbounds i8,{{.*}}_D6gh28653fooFZv{{.*}}, i64 -10
    // M32: %1 = getelementptr inbounds i8,{{.*}}_D6gh28653fooFZv{{.*}}, i32 -10
    // M64-NEXT: %2 = ptrtoint ptr %1 to i64
    // M32-NEXT: %2 = ptrtoint ptr %1 to i32
    auto addr = (cast(size_t) &foo) - 10;
}
