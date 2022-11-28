// RUN: %ldc -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

void foo()
{
    // CHECK: %1 = getelementptr inbounds i8,{{.*}}_D6gh28653fooFZv{{.*}}, i64 -10
    // CHECK-NEXT: %2 = ptrtoint {{i8\*|ptr}} %1 to i{{32|64}}
    auto addr = (cast(size_t) &foo) - 10;
}
