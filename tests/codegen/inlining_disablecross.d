// Test disabling/enabling of cross-module inlining

// REQUIRES: atleast_llvm307

// RUN: %ldc %s -I%S -c -output-ll  -enable-cross-module-inlining -O0 -of=%t.ENA.ll && FileCheck %s --check-prefix ENABLED  < %t.ENA.ll
// RUN: %ldc %s -I%S -c -output-ll -disable-cross-module-inlining -O3 -of=%t.DIS.ll && FileCheck %s --check-prefix DISABLED < %t.DIS.ll

import inputs.inlinables;

extern (C): // simplify mangling for easier matching

// DISABLED-LABEL: define{{.*}} @call_easily_inlinable(
// ENABLED-LABEL: define{{.*}} @call_easily_inlinable(
int call_easily_inlinable(int i)
{
    // DISABLED: call {{.*}} @easily_inlinable(
    return easily_inlinable(i);
    // DISABLED: ret
    // ENABLED: ret
}

// ENABLED-DAG: define {{.*}} @easily_inlinable(
// DISABLED-DAG: declare {{.*}} @easily_inlinable(
