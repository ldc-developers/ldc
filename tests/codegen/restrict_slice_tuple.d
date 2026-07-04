// RUN: %ldc -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

import ldc.attributes;

// Two restrict slices via a tuple parameter:
// the @restrict UDA on the tuple should propagate to both
// expanded slice elements, generating pairwise separate_storage.
// CHECK-LABEL: define {{.*}}@{{.*}}tupleTwoSlices
// CHECK: separate_storage
auto tupleTwoSlices(Args...)(@restrict Args args) {
    args[0][0] = 1;
    args[1][0] = 2;
}

// Mixed tuple: slices + pointer.
// Slices get separate_storage, pointer gets noalias on the LLVM param.
// CHECK-LABEL: define {{.*}}@{{.*}}tupleMixed
// CHECK-SAME: ptr noalias
// CHECK: separate_storage
auto tupleMixed(Args...)(@restrict Args args) {
    args[0][0] = *args[2];
    args[1][0] = 3;
}

// arrayOp-like: non-tuple restrict slice + tuple restrict slices.
// @restrict on both the non-tuple param and the tuple param should
// generate separate_storage among all slices.  This is the pattern
// from druntime's core.internal.array.operations.arrayOp (issue #4991).
// CHECK-LABEL: define {{.*}}@{{.*}}arrayOpLike
// CHECK: separate_storage
auto arrayOpLike(Args...)(@restrict float[] res, @restrict Args args) {
    res[0] = args[0][0] + args[1][0];
}

void test() {
    int[] a; double[] b;
    tupleTwoSlices(a, b);

    int[] x; float[] y; int* p;
    tupleMixed(x, y, p);

    float[] res, fa, fb;
    res = new float[16]; fa = new float[16]; fb = new float[16];
    arrayOpLike(res, fa, fb);
}
