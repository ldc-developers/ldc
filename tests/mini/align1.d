module tangotests.align1;

extern(C) int printf(char*, ...);

struct TLA
{
    char[3] tla;
    char[] toString() { return tla; }
    void dump()
    {
        printf("%.*s\n", 3, tla.ptr);
    }
}

void main()
{
    TLA fbi = TLA("FBI");
    fbi.dump();
}
