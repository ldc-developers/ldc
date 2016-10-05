// Tests that our TargetMachine options are added as function attributes

// RUN: %ldc -c -output-ll -of=%t.ll %s && FileCheck %s --check-prefix=DEFAULT < %t.ll
// RUN: %ldc -c -output-ll -of=%t.ll %s -disable-fp-elim && FileCheck %s --check-prefix=FRAMEPTR < %t.ll
// RUN: %ldc -c -output-ll -of=%t.ll %s -mattr=test && FileCheck %s --check-prefix=ATTR < %t.ll

// DEFAULT: define{{.*}} @{{.*}}3fooFZv{{.*}} #[[KEYVALUE:[0-9]+]]
// FRAMEPTR: define{{.*}} @{{.*}}3fooFZv{{.*}} #[[KEYVALUE:[0-9]+]]
// ATTR: define{{.*}} @{{.*}}3fooFZv{{.*}} #[[KEYVALUE:[0-9]+]]
void foo()
{
}

// DEFAULT: attributes #[[KEYVALUE]]
// DEFAULT-DAG: "target-cpu"=
// DEFAULT-DAG: "use-soft-float"="{{(true|false)}}"
// DEFAULT-DAG: "no-frame-pointer-elim"="false"
// DEFAULT-DAG: "unsafe-fp-math"="false"
// DEFAULT-DAG: "less-precise-fpmad"="false"
// DEFAULT-DAG: "no-infs-fp-math"="false"
// DEFAULT-DAG: "no-nans-fp-math"="false"

// FRAMEPTR: attributes #[[KEYVALUE]]
// FRAMEPTR-DAG: "no-frame-pointer-elim"="true"

// ATTR: attributes #[[KEYVALUE]]
// ATTR-DAG: "target-features"="{{.*}}+test{{.*}}"
