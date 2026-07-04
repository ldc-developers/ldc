// RUN: %ldc -O0 -c -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

// CHECK: declare {{.*}}@_d_assert({{.*}}) #[[ASSERT:[0-9]+]]
void g(bool b)
{
    assert(b);
}

// CHECK: attributes #[[ASSERT]] ={{.*}} cold{{.*}}noreturn
