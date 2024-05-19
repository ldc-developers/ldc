// REQUIRES: Windows

// RUN: %ldc -output-ll -dllimport=all -of=%t.ll %s && FileCheck %s < %t.ll

import std.variant : Variant; // pre-instantiated template

void foo()
{
    // define TypeInfo_Struct referencing dllimported Variant vtable and init symbol
    auto t = typeid(Variant);
}

// no direct init symbol refs:
// CHECK-NOT: @_D3std7variant__T8VariantN{{.*}}6__initZ

// dllimported init symbol:
// CHECK: @_D3std7variant__T8VariantN{{.*}}6__initZ = external dllimport

// no direct init symbol refs:
// CHECK-NOT: @_D3std7variant__T8VariantN{{.*}}6__initZ

// check generated CRT constructor:
// CHECK:      define private void @ldc.dllimport_relocation()

// should set the vptr:
// CHECK:      if:
// CHECK-NEXT: store ptr @_D15TypeInfo_Struct6__vtblZ,

// should set m_init.ptr:
// CHECK:      if1:
// CHECK-NEXT: store ptr {{.*}}@_D3std7variant__T8VariantN{{.*}}6__initZ, {{.*}}TypeInfo_S3std7variant__T8VariantN{{.*}}6__initZ
