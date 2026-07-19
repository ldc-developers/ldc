// REQUIRES: target_RISCV

// Default float ABI is hard (lp64d-style).
// RUN: %ldc -c -o- %s -mtriple=riscv64-unknown-linux-gnu -d-version=HARD
// RUN: %ldc -c -o- %s -mtriple=riscv64-unknown-linux-gnu -float-abi=soft -d-version=SOFT

version (HARD)
{
    version (D_HardFloat) {} else static assert(0);
    version (D_SoftFloat) static assert(0);
}
else version (SOFT)
{
    version (D_SoftFloat) {} else static assert(0);
    version (D_HardFloat) static assert(0);
}
else
    static assert(0);
