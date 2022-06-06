// Test disabling of cross-module inlining

// RUN: %ldc %s -I%S -c -output-ll  -enable-cross-module-inlining -O0 -of=%t.ENA.ll && FileCheck %s < %t.ENA.ll
// RUN: %ldc %s -I%S -c -output-ll -disable-cross-module-inlining -O3 -of=%t.DIS.ll && FileCheck %s < %t.DIS.ll

import inputs.inlinables;

extern (C): // simplify mangling for easier matching

// CHECK-LABEL: define{{.*}} @call_easily_inlinable(
int call_easily_inlinable(int i)
{
    // CHECK: call {{.*}} @easily_inlinable(
    return easily_inlinable(i);
    // CHECK: ret
}

// CHECK-DAG: declare {{.*}} @easily_inlinable(
