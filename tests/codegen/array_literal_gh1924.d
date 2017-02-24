// RUN: %ldc -c -O3 -output-ll -of=%t.ll %s && FileCheck %s < %t.ll
// RUN: %ldc -d-version=RUN -run %s

// CHECK-LABEL: define{{.*}} @{{.*}}simple2d
auto simple2d()
{
    // CHECK: _d_newarrayU
    // CHECK: _d_newarrayU
    // CHECK-NOT: _d_newarrayU
    // CHECK: ret {
    return [[1.0]];
}

// GitHub issue #1925
// CHECK-LABEL: define{{.*}} @{{.*}}make2d
auto make2d()
{
    // CHECK: _d_newarrayU
    // CHECK-NOT: _d_newarrayU
    double[][1] a = [[1.0]];
    // CHECK: ret
    return a;
}

// CHECK-LABEL: define{{.*}} @{{.*}}make3d
auto make3d()
{
    // CHECK: _d_newarrayU
    // CHECK: _d_newarrayU
    // CHECK-NOT: _d_newarrayU
    int[][1][] a = [[[1]]];
    // CHECK: ret {
    return a;
}

struct S
{
    auto arr = [[321]];
}

// CHECK-LABEL: define{{.*}} @{{.*}}makeS
auto makeS()
{
    // CHECK: _d_newarrayU
    // CHECK: _d_newarrayU
    // CHECK-NOT: _d_newarrayU
    // CHECK: ret
    return S();
}

mixin template A()
{
    auto a = [1, 2, 3];
    auto b = [[1, 2, 3]];
}

version (RUN)
{
    void main()
    {
        {
            auto a = simple2d();
            auto b = simple2d();
            assert(a.ptr !is b.ptr);
            assert(a[0].ptr !is b[0].ptr);
        }
        {
            auto a = make2d();
            auto b = make2d();
            assert(a.ptr !is b.ptr);
            assert(a[0].ptr !is b[0].ptr);
        }
        {
            auto a = make3d();
            auto b = make3d();
            assert(a.ptr !is b.ptr);
            assert(a[0].ptr !is b[0].ptr);
            assert(a[0][0].ptr !is b[0][0].ptr);
        }
        {
            enum e = [[1.0]];
            auto a = e;
            auto b = e;
            assert(a.ptr !is b.ptr);
            assert(a[0].ptr !is b[0].ptr);
        }
        {
            auto a = makeS();
            auto b = makeS();
            assert(a.arr.ptr !is b.arr.ptr);
            assert(a.arr[0].ptr !is b.arr[0].ptr);
        }
        {
            mixin A!() a0;
            mixin A!() a1;
            assert(a0.a.ptr !is a1.a.ptr);
            assert(a0.b.ptr !is a1.b.ptr);
            assert(a0.b[0].ptr !is a1.b[0].ptr);
        }
    }
}
