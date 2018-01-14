// Make sure llvm_expect() is supported during CTFE.

// RUN: %ldc -c %s

import ldc.intrinsics : llvm_expect;

int notZero(int x)
{
    if (llvm_expect(x == 0, false))
        return 666;
    return x;
}

void main()
{
    static assert(notZero(0) == 666);
    static assert(notZero(1) == 1);
}
