// Example user-implemented --fsplit-stack __morestack handling that detects stack overflow.
// Also serves as an example for users when they want to use this in their project.

// REQUIRES: Linux
// REQUIRES: host_X86

// RUN: %ldc -g --fsplit-stack %s -of=%t%exe
// RUN: not %t%exe 2>&1 | FileCheck %s

import ldc.attributes;

// CHECK: Local variables would grow the stack beyond limit!
// CHECK: fsplit-stack-runtest.d:[[@LINE+1]]
void foo() {
    byte[2_000] a;
}

@noSplitStack
void set_stacksize_in_TCB_relative_to_rsp(size_t stack_size) {
    asm { "mov %%rsp, %%r11;
           sub %0, %%r11;
           mov %%r11, %%fs:0x70;" // fs:0x70 is what split-stack uses as minimum stack address
          : //output operands
          : "r" (stack_size) //input operands
          : "r11", "memory"; // clobbers
    }
}

// Override C runtime __morestack and abort (which lets druntime print a stack trace)
@noSplitStack
extern(C) void __morestack()
{
    throw new Error("Local variables would grow the stack beyond limit!");
    import ldc.intrinsics;
    llvm_trap(); // just in case
}

void main() {
    // GC must be initialized before calling __morestack, perhaps because of funny stack setup when __morestack is called?
    int[] a = new int[1];

    set_stacksize_in_TCB_relative_to_rsp(1_000);

    foo();
}
