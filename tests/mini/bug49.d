module bug49;

pragma(LLVM_internal, "notypeinfo")
struct S
{
    int i;
    long l;
}

void main()
{
    S s;
    s.i = 0x__FFFF_FF00;
    s.l = 0xFF00FF_FF00;
    s.i &= s.l;
}
