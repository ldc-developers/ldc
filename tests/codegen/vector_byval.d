
// Passing big vector by value cause issues on win64
// RUN: %ldc -run %s

pragma(inline,false)
auto foo(Args...)(Args args)
{
    typeof(args[0][0]) ret = 0;
    foreach (arg; args)
    {
        foreach (v; arg[])
        {
            ret += v;
        }
    }
    return ret;
}

auto createVec(T, size_t Len)()
{
    import core.simd;
    Vector!(T[Len]) ret;
    ret[] = 1;
    return ret;
}

void main()
{
    void test(T)()
    {
        assert(foo(createVec!(T,16)) == cast(T)16);
        assert(foo(createVec!(T,32)) == cast(T)32);
        assert(foo(createVec!(T,16),createVec!(T,32)) == cast(T)48);
        assert(foo(createVec!(T,16),createVec!(T,8),createVec!(T,8)) == cast(T)32);
        assert(foo(createVec!(T,16),createVec!(T,16),createVec!(T,16)) == cast(T)48);
        assert(foo(createVec!(T,16),createVec!(T,32),createVec!(T,32)) == cast(T)80);
        assert(foo(createVec!(T,16),createVec!(T,128)) == cast(T)144);
        assert(foo(createVec!(T,128),createVec!(T,128),createVec!(T,128),createVec!(int,128)) == cast(T)512);
    }
    test!byte();
    test!short();
    test!int();
    test!float();
    test!double();
}
