// RUN: %ldc -run %s

void main()
{
    int[string] aa = [ "one": 123 ];
    typeof(null) nul;

    auto sum = nul + nul;
    auto diff = nul - nul;

    assert(aa + nul == aa);
    assert(nul + aa == aa);
    assert(aa - nul == aa);
    assert(nul - aa == aa);

    static assert(!__traits(compiles, nul * nul));
    static assert(!__traits(compiles, aa * nul));
    static assert(!__traits(compiles, nul / nul));
    static assert(!__traits(compiles, aa / nul));
    static assert(!__traits(compiles, nul % nul));
    static assert(!__traits(compiles, aa % nul));

    static assert(!__traits(compiles, nul & nul));
    static assert(!__traits(compiles, aa | nul));
}
