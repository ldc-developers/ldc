// See Github issue #1860

// RUN: %ldc -c -output-ll -of=%t.ll -float-abi=soft   %s && FileCheck --check-prefix=SOFT %s < %t.ll
// RUN: %ldc -c -output-ll -of=%t.ll -float-abi=softfp %s && FileCheck --check-prefix=HARD %s < %t.ll

// SOFT: define{{.*}} @{{.*}}3fooFZv{{.*}} #[[KEYVALUE:[0-9]+]]
// HARD: define{{.*}} @{{.*}}3fooFZv{{.*}} #[[KEYVALUE:[0-9]+]]
void foo()
{
}

// SOFT: attributes #[[KEYVALUE]]
// SOFT-DAG: "target-features"="{{.*}}+soft-float{{.*}}"
// HARD: attributes #[[KEYVALUE]]
// HARD-NOT: "target-features"="{{.*}}+soft-float{{.*}}"
