// Make sure an invalid asm instruction causes a non-zero exit code.

// RUN: not %ldc -c %s 2> %t.stderr
// RUN: FileCheck %s < %t.stderr

void main()
{
    asm { "some_garbage"; }
}

// CHECK: error:
