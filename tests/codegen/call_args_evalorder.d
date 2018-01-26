// RUN: %ldc -run %s

void checkInt(int a, int b, int c)
{
    assert(a == 1);
    assert(b == 2);
    assert(c == 3);
}

int incrementBy2AndReturn2(ref int a)
{
    a += 2;
    return 2;
}

// ---

struct BigStruct
{
    long[33] blub;
    int v;
    this(int v) { this.v = v; }
}

void checkBigStruct(BigStruct a, BigStruct b, BigStruct c)
{
    assert(a.v == 1);
    assert(b.v == 2);
    assert(c.v == 3);
}

BigStruct incrementBy2AndReturn2(ref BigStruct a)
{
    a.v += 2;
    return BigStruct(2);
}

// ---

void main()
{
    int a = 1;
    checkInt(a, incrementBy2AndReturn2(a), a);

    auto s = BigStruct(1);
    checkBigStruct(s, incrementBy2AndReturn2(s), s);
}
