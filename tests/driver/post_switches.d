// RUN: not %ldc -I=%runtimedir/src -conf=%S/inputs/post_switches.conf %s -v -L-user-passed-switch | FileCheck %s

// CHECK: -normal-switch
// CHECK-SAME: -normal-two-switch
// CHECK-SAME: -user-passed-switch
// CHECK-SAME: -post-switch
// CHECK-SAME: -post-two-switch


// RUN: not %ldc -I=%runtimedir/src -conf=%S/inputs/post_switches.conf -v -L-user-passed-switch -run %s -L-after-run | FileCheck %s --check-prefix=WITHrUN

// WITHrUN: -normal-switch
// WITHrUN-SAME: -normal-two-switch
// WITHrUN-SAME: -user-passed-switch
// WITHrUN-SAME: -post-switch
// WITHrUN-SAME: -post-two-switch
// WITHrUN-NOT: -after-run

void main()
{
}
