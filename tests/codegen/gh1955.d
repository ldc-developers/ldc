// RUN: %ldc -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

import rt.monitor_ : Mutex;

struct D_CRITICAL_SECTION // private symbol in rt.critical_
{
    D_CRITICAL_SECTION* next;
    Mutex mtx;
}

void main()
{
    /* The synchronized-block uses a global buffer for the D_CRITICAL_SECTION.
     * Match its size and alignment.
     */
    // CHECK: __critsec{{.*}} = global {{\[}}[[SIZEOF:[0-9]+]] x i8{{\]}} zeroinitializer
    // CHECK-SAME: align [[ALIGNOF:[0-9]+]]
    synchronized {}

    /* Verify size and alignment of the global against a manual D_CRITICAL_SECTION.
     */
    // CHECK: %cs = alloca %gh1955.D_CRITICAL_SECTION, align [[ALIGNOF]]
    // CHECK-SAME: size/byte = [[SIZEOF]]
    D_CRITICAL_SECTION cs;
}
