// RUN: %ldc -run %s

struct S
{
    __gshared int numDtor;
    int a;
    ~this() { ++numDtor; a = 0; }
    ref int val() return { return a; }
}

S make() { return S(2); }

int call() { return make().val; }
int literal() { return S(123).val; }

void main()
{
    assert(call() == 2);
    assert(literal() == 123);
    assert(S.numDtor == 2);
}
