// Example user-implemented --fsplit-stack __morestack handling that detects stack overflow.
// Also serves as an example for users when they want to use this in their project.

// REQUIRES: Linux || FreeBSD || Windows
// REQUIRES: host_X86

// RUN: %ldc -g --fsplit-stack %s -of=%t%exe
// RUN: not %t%exe 2>&1 | FileCheck %s

// CHECK: Stack overflow!
// CHECK: foo argument bytes:
// CHECK: foo local variables bytes: {{2...}}
void foo() {
    byte[2_000] a;
}

void main() {
    set_stacksize_in_TCB_relative_to_rsp(1_000);

    foo();
}


import ldc.attributes;

@noSplitStack
void set_stacksize_in_TCB_relative_to_rsp(size_t stack_size) {
    // fs:0x70 is what split-stack uses as minimum stack address on Linux,
    // this can be checked by looking at the split-stack codegen assembly (rsp is compared with it).
    version (linux)
        enum stack_limit_str = "%%fs:0x70";
    else version (FreeBSD)
        enum stack_limit_str = "%%fs:0x18";
    else version (Windows)
        enum stack_limit_str = "%%gs:0x28";

    asm { "mov %%rsp, %%r11;
           sub %0, %%r11;
           mov %%r11, " ~ stack_limit_str ~ ";"
          : //output operands
          : "r" (stack_size) //input operands
          : "r11", "memory"; // clobbers
    }
}

// Override C runtime __morestack and abort (which lets druntime print a stack trace)
// With this implementation, standard backtracing will not work correctly because `__morestack`
// is called from `foo` _before_ the stack frame of `foo` is setup correctly (e.g before `push rbp`).
// See the implementation of libgcc's `__morestack` for the CFI trickery that is needed for unwinding.
@noSplitStack
extern(C) void __morestack()
{
    uint local_variables_bytes;
    uint foo_parameters;
    // r10d contains local variable size in bytes
    // r11d contains parameters on stack in bytes
    asm { "mov %%r10d, %0;
           mov %%r11d, %1;"
          : "=m" (local_variables_bytes), //output operands
            "=m" (foo_parameters)
          : // no input operands
          : ; // no clobbers
    }

    import core.stdc.stdio;
    printf("Stack overflow!\n");
    printf("foo argument bytes: %d\n", foo_parameters);
    printf("foo local variables bytes: %d\n", local_variables_bytes);
    fflush(stdout);

    // Abort execution without stack unwinding.
    import core.stdc.stdlib;
    _Exit(1);
}
