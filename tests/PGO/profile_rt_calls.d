// Tests runtime profile-rt access.

// REQUIRES: PGO_RT

// RUN: %ldc -fprofile-instr-generate=%t.profraw -run %s

import ldc.profile;

bool foo(bool a, bool b) { return a ? a : b; }
bool bar(bool a, bool b) { return a ? a : b; }
extern(C) bool fooC(bool a, bool b) { return a; }
extern(C++) bool fooCpp(bool a, bool b) { return a; }

bool notinstrumented(bool a, bool b) {
    pragma(LDC_profile_instr, false)
    return a ? a : b;
}

extern(C) void getdataprofile() {
    assert( getData!foo != null );
    assert( getData!fooC != null );
    assert( getData!fooCpp != null );
    assert( getData!notinstrumented == null );
}

void check_counters() {
    resetAll();
    foo(true, true);
    foo(false, true);
    assert( getCount!(foo)(0) == 2 );
    assert( getCount!(foo)(1) == 1 );
    assert( getCount!(foo)(2) == ulong.max );
    assert( getCount!notinstrumented(0) == ulong.max );

    bar(true, true); bar(true, true); bar(true, true);
    assert( getCallCount!bar == 3 );
    assert( getCallCount!notinstrumented == ulong.max );

    setCount!bar(0, 123);
    assert( getCount!(bar)(0) == 123 );
    setCount!bar(3, 123);

    resetCounts!foo;
    assert( getCount!(foo)(0) == 0 );
    assert( getCount!(foo)(1) == 0 );
    assert( getCount!(bar)(0) == 123 );
}

void main() {
    getdataprofile();
    check_counters();

    // prevent linker stripping
    fooC(true, false);
    fooCpp(true, false);
}
