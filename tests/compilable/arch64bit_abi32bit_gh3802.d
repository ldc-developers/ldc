// Tests compilation for 64-bit architecture with 32-bit word size ABI.
// Triple examples: mips64el-linux-gnuabin32, x86_64-linux-gnux32
// Targeting x86 because that's the most widely available target in our CI/developer systems.

// REQUIRES: target_X86
// RUN: %ldc -mtriple=x86_64-linux-gnux32 -O -c %s

static assert((void*).sizeof == 4);
static assert(size_t.sizeof == 4);
static assert(ptrdiff_t.sizeof == 4);

version (D_LP64) static assert(0);
version (D_X32) { /* expected */ } else static assert(0);

bool equals(string lhs, string rhs)
{
    foreach (const i; 0 .. lhs.length) {}
    return false;
}

// test _d_array_slice_copy optimization IR pass (requires -O)
auto test_ArraySliceCopyOpt()
{
    static void copy(int[] a, int[] b)
    {
        a[] = b[];
    }

    int[2] a = [1, 2];
    int[2] b = [3, 4];
    copy(a, b);
    return a;
}
