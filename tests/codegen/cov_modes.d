// Test the different modes of -cov instrumentation code

// RUN: %ldc --cov                                                 --output-ll -of=%t.ll         %s && FileCheck --check-prefix=ALL --check-prefix=DEFAULT %s < %t.ll
// RUN: %ldc --cov --cov-increment=boolean --cov-increment=default --output-ll -of=%t.default.ll %s && FileCheck --check-prefix=ALL --check-prefix=DEFAULT %s < %t.default.ll

// RUN: %ldc --cov --cov-increment=atomic     --output-ll -of=%t.atomic.ll    %s && FileCheck --check-prefix=ALL --check-prefix=ATOMIC    %s < %t.atomic.ll
// RUN: %ldc --cov --cov-increment=non-atomic --output-ll -of=%t.nonatomic.ll %s && FileCheck --check-prefix=ALL --check-prefix=NONATOMIC %s < %t.nonatomic.ll
// RUN: %ldc --cov --cov-increment=boolean    --output-ll -of=%t.boolean.ll   %s && FileCheck --check-prefix=ALL --check-prefix=BOOLEAN   %s < %t.boolean.ll


// REQUIRES: Linux
// RUN: mkdir %t
// RUN: mkdir %t/atomic    && %ldc --cov --cov-increment=atomic     --run %s --DRT-covopt="dstpath:%t/atomic"
// RUN: mkdir %t/nonatomic && %ldc --cov --cov-increment=non-atomic --run %s --DRT-covopt="dstpath:%t/nonatomic"
// RUN: mkdir %t/boolean   && %ldc --cov --cov-increment=boolean    --run %s --DRT-covopt="dstpath:%t/boolean"
// Some sed xargs magic to replace '/' with '-' in the filename, and replace the extension '.d' with '.lst'
// RUN: echo %s | sed -e "s,/,-,g" -e "s,\(.*\).d,\1.lst," | xargs printf "%%s%%s" "%t/atomic/"    | xargs cat | FileCheck --check-prefix=ATOMIC_LST %s
// RUN: echo %s | sed -e "s,/,-,g" -e "s,\(.*\).d,\1.lst," | xargs printf "%%s%%s" "%t/nonatomic/" | xargs cat | FileCheck --check-prefix=NONATOMIC_LST %s
// RUN: echo %s | sed -e "s,/,-,g" -e "s,\(.*\).d,\1.lst," | xargs printf "%%s%%s" "%t/boolean/"   | xargs cat | FileCheck --check-prefix=BOOLEAN_LST %s

void f2()
{
}

// ALL-LABEL: define{{.*}} void @{{.*}}f1
void f1()
{
    // DEFAULT: atomicrmw add {{.*}}@_d_cover_data, {{.*}} monotonic
    // ATOMIC: atomicrmw add {{.*}}@_d_cover_data, {{.*}} monotonic
    // NONATOMIC: load {{.*}}@_d_cover_data, {{.*}} !nontemporal
    // NONATOMIC: store {{.*}}@_d_cover_data, {{.*}} !nontemporal
    // BOOLEAN: store {{.*}}@_d_cover_data, {{.*}} !nontemporal
    // ALL-LABEL: call{{.*}} @{{.*}}f2
    f2();
}

void main()
{
    foreach (i; 0..10)
        f1();
    // ATOMIC_LST: {{^ *}}10|        f1();
    // NONATOMIC_LST: {{^ *}}10|        f1();
    // BOOLEAN_LST: {{^ *}}1|        f1();
}
