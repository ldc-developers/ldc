// Tests compilation for 64-bit architecture with 32-bit word size ABI.
// Triple examples: mips64el-linux, x86_64-linux-gnux32, aarch64-linux-gnu_ilp32
// Targeting x86 because that's the most widely available target in our CI/developer systems.

// REQUIRES: target_X86
// RUN: %ldc -mtriple=x86_64-linux-gnux32 -c %s

bool equals(string lhs, string rhs)
{
    foreach (const i; 0 .. lhs.length) {}
    return false;
}
