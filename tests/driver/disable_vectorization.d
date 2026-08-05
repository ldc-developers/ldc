// https://github.com/ldc-developers/ldc/issues/5233

// REQUIRES: target_X86

// RUN: %ldc -mtriple=x86_64-linux-gnu -output-s -of=%t.s -O %s
// RUN: FileCheck %s --check-prefix=ALL --check-prefix=ENABLED < %t.s

// RUN: %ldc -mtriple=x86_64-linux-gnu -output-s -of=%t.s -O -disable-loop-vectorization %s
// RUN: FileCheck %s --check-prefix=ALL --check-prefix=NOLOOP < %t.s

// `-Oz` is equivalent to `-O -disable-loop-vectorization` wrt. vectorization defaults
// RUN: %ldc -mtriple=x86_64-linux-gnu -output-s -of=%t.s -Oz %s
// RUN: FileCheck %s --check-prefix=ALL --check-prefix=NOLOOP < %t.s

// RUN: %ldc -mtriple=x86_64-linux-gnu -output-s -of=%t.s -O -disable-loop-vectorization -disable-slp-vectorization %s
// RUN: FileCheck %s --check-prefix=ALL --check-prefix=DISABLED < %t.s

// `-O1` disables both
// RUN: %ldc -mtriple=x86_64-linux-gnu -output-s -of=%t.s -O1 %s
// RUN: FileCheck %s --check-prefix=ALL --check-prefix=DISABLED < %t.s


import ldc.attributes : restrict;


// loop test:
// ALL: _D21disable_vectorization11add_dynamicFAfxAfxQdZv:
void add_dynamic(@restrict float[] a, @restrict const float[] b, @restrict const float[] c) {
    // ENABLED: addps
    // NOLOOP-NOT: addps
    // DISABLED-NOT: addps
    foreach (i; 0 .. a.length)
        a[i] = b[i] + c[i];
    // ALL: .size	_D21disable_vectorization11add_dynamicFAfxAfxQdZv
}


// SLP test:
// ALL: _D21disable_vectorization10add_staticFKG4fKxG4fKxQfZv:
void add_static(@restrict ref float[4] a, @restrict const ref float[4] b, @restrict const ref float[4] c) {
    // ENABLED: addps
    // NOLOOP: addps
    // DISABLED-NOT: addps
    foreach (i; 0 .. a.length)
        a[i] = b[i] + c[i];
    // ALL: .size	_D21disable_vectorization10add_staticFKG4fKxG4fKxQfZv
}
