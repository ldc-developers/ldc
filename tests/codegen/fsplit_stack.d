// Test --fsplit-stack

// REQUIRES: target_X86

// RUN: %ldc -mtriple=x86_64-linux --fsplit-stack -c --output-ll -of=%t.ll %s && FileCheck --check-prefix=IR  %s < %t.ll
// RUN: %ldc -mtriple=x86_64-linux --fsplit-stack -c --output-s  -of=%t.s  %s && FileCheck --check-prefix=ASM %s < %t.s

import ldc.attributes;

// Extern C disables mangling, for easier function name matching.
extern (C):

// IR-LABEL: define{{.*}} @foofoofoofoo
// IR-SAME: #[[ATTR:[0-9]+]]
// ASM-LABEL: foofoofoofoo
// ASM: callq  __morestack
// ASM-LABEL: .cfi_endproc
void foofoofoofoo()
{
    int[100] a;
}

// IR-LABEL: define{{.*}} @g_nosplitstack_g
// IR-SAME: #[[ATTR_DISABLED:[0-9]+]]
// ASM-LABEL: g_nosplitstack_g
// ASM-NOT: callq  __morestack
// ASM-LABEL: .cfi_endproc
@noSplitStack
void g_nosplitstack_g()
{
    int[100] a;
}


// IR-NOT: attributes #[[ATTR_DISABLED]] ={{.*}}split-stack
// IR-DAG: attributes #[[ATTR]] ={{.*}}split-stack
// IR-NOT: attributes #[[ATTR_DISABLED]] ={{.*}}split-stack
