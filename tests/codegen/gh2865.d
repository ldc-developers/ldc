// RUN: %ldc -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

void foo()
{
    // CHECK: %1 = getelementptr {{.*}}_D6gh28653fooFZv{{.*}} to i8*), i32 -10
    // CHECK-NEXT: %2 = ptrtoint i8* %1 to i{{32|64}}
    auto addr = (cast(size_t) &foo) - 10;
}
