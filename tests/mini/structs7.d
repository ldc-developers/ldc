module structs7;

pragma(LLVM_internal, "notypeinfo")
struct S
{
    int i;
    long l;
}

void main()
{
    S s = void;
    int i = s.i;
    long l = s.l;
}
