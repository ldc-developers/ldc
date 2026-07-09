// RUN: %ldc -O3 -boundscheck=off -ffast-math -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

import ldc.attributes;

// Without restrict on tuple: LLVM must assume slices overlap → scalar.
// CHECK-LABEL: define {{.*}}@{{.*}}noRestrictTuple
// CHECK-NOT: load <4 x float>
auto noRestrictTuple(Args...)(float[] res, const Args args) {
    foreach (i; 0 .. 16)
        res[i] = args[0][i] * args[1][i] + args[2][i];
}

// With restrict on tuple (arrayOp pattern from issue #4991): vectorized.
// CHECK-LABEL: define {{.*}}@{{.*}}withRestrictTuple
// CHECK: load <4 x float>
auto withRestrictTuple(Args...)(@restrict float[] res, @restrict Args args) {
    foreach (i; 0 .. 16)
        res[i] = args[0][i] * args[1][i] + args[2][i];
}

void test() {
    float[] res, a, b, c;
    res = new float[16]; a = new float[16]; b = new float[16]; c = new float[16];
    noRestrictTuple(res, a, b, c);
    withRestrictTuple(res, a, b, c);
}
