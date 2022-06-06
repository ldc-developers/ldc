// RUN: %ldc -c -output-ll -fxray-instrument -of=%t.ll %s && FileCheck %s < %t.ll

import ldc.attributes;

// CHECK-LABEL: define{{.*}} @{{.*}}10instrument
// CHECK-SAME: #[[INSTR:[0-9]+]]
void instrument()
{
}

// CHECK-LABEL: define{{.*}} @{{.*}}15dont_instrument
// CHECK-SAME: #[[DONT_INSTR:[0-9]+]]
void dont_instrument()
{
    pragma(LDC_profile_instr, false);
}

// CHECK-DAG: attributes #[[INSTR]] ={{.*}} "xray-instruction-threshold"=
// CHECK-DAG: attributes #[[DONT_INSTR]] ={{.*}} "function-instrument"="xray-never"
