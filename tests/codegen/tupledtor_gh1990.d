// RUN: %ldc -run %s

int checksum = 1;

struct Foo
{
    int val = 0;
    ~this()
    {
        checksum = (checksum + val) * 3;
    }
}

void bar(ARGS...)()
{
    ARGS tup;
    checksum = 7;
    tup[0].val = 1; // this one's dtor must be called second
    tup[1].val = 2; // this one's dtor must be called first
}

void main()
{
    bar!(Foo, Foo)();

    assert(checksum == ((7 + 2) * 3 + 1) * 3);
}
