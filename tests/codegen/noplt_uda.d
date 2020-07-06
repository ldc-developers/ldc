// REQUIRES: atleast_llvm500

// RUN: %ldc --output-ll -of=%t.ll %s && FileCheck %s -check-prefixes=CHECK-NOPLT,CHECK-NOPLT-METADATA < %t.ll

import ldc.attributes : noplt;

// CHECK-NOPLT: Function Attrs: nonlazybind
// CHECK-NOPLT-NEXT: declare {{.*}} @{{.*}}3foo
// CHECK-NOPLT-METADATA: !"RtLibUseGOT"
@noplt int foo();

int bar() {
    return foo();
}
