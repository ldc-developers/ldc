module union1;

pragma(LLVM_internal, "notypeinfo")
union U
{
    float f;
    int i;
}

void main()
{
    float f = 2;
    U u = U(f);
    assert(u.i == *cast(int*)&f);
}
