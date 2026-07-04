// https://github.com/dlang/dmd/issues/21757

// LDC: ignore MS linker warnings for 32-bit Windows
// TRANSFORM_OUTPUT(windows32): remove_lines("warning LNK4318: Very long symbol name encountered while producing debug information")

struct S
{
    void*[300_000] arr; // generates stupidly large symbols for RTImfoImpl!() that crash MS link.exe
}

void main()
{
}
