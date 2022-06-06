// Test function-level enabling/disabling of instrumentation of switch statements.

// RUN: %ldc -boundscheck=off -c -output-ll -fprofile-instr-generate -of=%t.ll %s && FileCheck %s --check-prefix=PROFGEN < %t.ll

extern (C): // simplify name mangling for simpler string matching

// PROFGEN-LABEL: @enabled(
int enabled(int i)
{
    pragma(LDC_profile_instr, true);

    switch (i)
    {
        // PROFGEN: casecntr
    case 1:
        return 1;
        // PROFGEN: casecntr
    case 2:
        return 2;
        // PROFGEN: defaultcntr
    default:
        return 3;
    }
}

// PROFGEN-LABEL: @disabled(
// PROFGEN-NOT: casecntr
// PROFGEN-NOT: defaultcntr
int disabled(int i)
{
    pragma(LDC_profile_instr, false);

    switch (i)
    {
    case 1:
        return 1;
    case 2:
        return 2;
    default:
        return 3;
    }
}

// PROFGEN-LABEL: @bunch_of_branches_enabled(
int bunch_of_branches_enabled(int i, const int two)
{
    pragma(LDC_profile_instr, true);

    switch (i)
    {
// PROFGEN: casecntr
    case 1:
        return 1;
// PROFGEN: casecntr
    case two:
        return 2;
// PROFGEN: defaultcntr
    default:
        return 3;
    }
}

// PROFGEN-LABEL: @bunch_of_branches_disabled(
// PROFGEN-NOT: casecntr
// PROFGEN-NOT: defaultcntr
int bunch_of_branches_disabled(int i, const int two)
{
    pragma(LDC_profile_instr, false);

    switch (i)
    {
    case 1:
        return 1;
    case two:
        return 2;
    default:
        return 3;
    }
}
