// Tests that our TargetMachine options are added as function attributes

// RUN: %ldc -c -output-ll -of=%t.ll %s
// RUN: FileCheck %s --check-prefix=COMMON --check-prefix=WITH_FP < %t.ll
// RUN: %ldc -c -output-ll -of=%t.ll %s -O2
// RUN: FileCheck %s --check-prefix=COMMON --check-prefix=NO_FP < %t.ll
// RUN: %ldc -c -output-ll -of=%t.ll %s -O2 -disable-fp-elim
// RUN: FileCheck %s --check-prefix=COMMON --check-prefix=WITH_FP < %t.ll
// RUN: %ldc -c -output-ll -of=%t.ll %s -disable-fp-elim=false -mattr=test
// RUN: FileCheck %s --check-prefix=COMMON --check-prefix=NO_FP --check-prefix=ATTR < %t.ll

// COMMON: define{{.*}} @{{.*}}3fooFZv{{.*}} #[[KEYVALUE:[0-9]+]]
void foo()
{
}

// COMMON: attributes #[[KEYVALUE]]
// COMMON-DAG: "target-cpu"=
// COMMON-DAG: "unsafe-fp-math"="false"
// COMMON-DAG: "less-precise-fpmad"="false"
// COMMON-DAG: "no-infs-fp-math"="false"
// COMMON-DAG: "no-nans-fp-math"="false"

// WITH_FP-DAG: "no-frame-pointer-elim"="true"
// NO_FP-DAG:   "no-frame-pointer-elim"="false"

// ATTR-DAG: "target-features"="{{.*}}+test{{.*}}"
