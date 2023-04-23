// Test --fsplit-stack

// REQUIRES: target_X86

// RUN: %ldc -mtriple=x86_64-linux --fsplit-stack -c --output-ll -of=%t.ll %s && FileCheck --check-prefix=IR  %s < %t.ll
// RUN: %ldc -mtriple=x86_64-linux --fsplit-stack -c --output-s  -of=%t.s  %s && FileCheck --check-prefix=ASM %s < %t.s

// IR-LABEL: define{{.*}} @foofoofoofoo
// IR-SAME: #[[ATTR:[0-9]+]]

// ASM-LABEL: foofoofoofoo
// ASM: callq  __morestack
// ASM-LABEL: .cfi_endproc

extern (C) void foofoofoofoo()
{
    int[100] a;
}

// IR-DAG: attributes #[[ATTR]] ={{.*}}split-stack
