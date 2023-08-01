// Tests that our TargetMachine options are added as function attributes

// RUN: %ldc -c -output-ll -of=%t.ll %s
// RUN: FileCheck %s --check-prefix=COMMON --check-prefix=WITH_FP < %t.ll
// RUN: %ldc -c -output-ll -of=%t.ll %s -O2
// RUN: FileCheck %s --check-prefix=COMMON --check-prefix=NO_FP < %t.ll
// RUN: %ldc -c -output-ll -of=%t.ll %s -O2 -frame-pointer=all
// RUN: FileCheck %s --check-prefix=COMMON --check-prefix=WITH_FP < %t.ll
// RUN: %ldc -c -output-ll -of=%t.ll %s -frame-pointer=none -mattr=test
// RUN: FileCheck %s --check-prefix=COMMON --check-prefix=NO_FP --check-prefix=ATTR < %t.ll

// COMMON: define{{.*}} @{{.*}}3fooFZv{{.*}} #[[KEYVALUE:[0-9]+]]
void foo()
{
}

// COMMON: attributes #[[KEYVALUE]]
// COMMON-DAG: "target-cpu"=

// WITH_FP-DAG: "frame-pointer"="all"
// NO_FP-DAG:   "frame-pointer"="none"

// ATTR-DAG: "target-features"="{{[^"]*}}+test
