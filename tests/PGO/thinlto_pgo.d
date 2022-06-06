// Test execution path for ThinLTO when PGO data is available.
// I manually verified that PGO data is added to the ThinLTO module summary, but do not know how to automatically test this reliably.

// REQUIRES: LTO
// REQUIRES: PGO_RT

// RUN: %ldc -fprofile-instr-generate=%t.profraw -run %s  \
// RUN:   &&  %profdata merge %t.profraw -o %t.profdata \
// RUN:   &&  %ldc -c -flto=thin -of=%t2%obj -fprofile-instr-use=%t.profdata %s

void coldfunction()
{
}

void hotfunction()
{
}

void main()
{
    coldfunction();
    foreach (i; 0 .. 1000)
    {
        hotfunction();
    }
}
