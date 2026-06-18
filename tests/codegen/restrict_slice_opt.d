// RUN: %ldc -O3 -boundscheck=off -ffast-math -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

import ldc.attributes;

// Without restrict: LLVM must assume slices may overlap → scalar code path
// CHECK-LABEL: define {{.*}}@{{.*}}noRestrict
// CHECK-NOT: load <4 x float>
void noRestrict(float[] o, const float[] a,
                const float[] b, const float[] c) {
    foreach (i; 0..16) o[i] = b[i] * a[i] + c[i];
}

// With restrict: LLVM knows slices don't overlap → auto-vectorized
// CHECK-LABEL: define {{.*}}@{{.*}}withRestrict
// CHECK: load <4 x float>
void withRestrict(@restrict float[] o, @restrict const float[] a,
                  @restrict const float[] b, @restrict const float[] c) {
    foreach (i; 0..16) o[i] = b[i] * a[i] + c[i];
}
