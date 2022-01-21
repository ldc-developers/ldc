// https://github.com/ldc-developers/ldc/issues/3883
// RUN: %ldc -run %s

struct S {
    int x;
    ~this() {}
}

__gshared S* ptr;

S foo() {
    auto result = S(123);
    (() @trusted { result.x++; ptr = &result; })();
    return result;
}

void main() {
    auto r = foo();
    assert(r.x == 124);
    assert(&r == ptr);
}
