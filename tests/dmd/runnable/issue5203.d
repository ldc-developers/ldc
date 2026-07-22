// https://github.com/ldc-developers/ldc/issues/5203

void main()
{
    auto s = Struct();
    auto dg = s.nestedThunk();
    assert(dg() == 42);
}

struct Struct
{
    int field = 42;
    auto nestedThunk()
    {
        int inner()
        {
            return __traits(getMember, this, "field");
        }
        return &inner;
    }
}
