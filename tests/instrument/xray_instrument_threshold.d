// RUN: %ldc -c -output-ll -fxray-instrument -fxray-instruction-threshold=543 -of=%t.ll %s && FileCheck %s < %t.ll

// CHECK-LABEL: define{{.*}} @{{.*}}10instrument
// CHECK-SAME: #[[INSTR:[0-9]+]]
void instrument()
{
}

// CHECK-DAG: attributes #[[INSTR]] ={{.*}} "xray-instruction-threshold"="543"
