module ldc_enum;

enum EF : float  { a = 1.1, b = 1, c = 2 }
enum ED : double { a = 1.2, b, c }

auto test1_a()
{
    auto t = typeid(EF);
    assert(t.name == "ldc_enum.EF");
    auto init = *cast(float*)t.initializer.ptr;
    assert(init == 1.1f);
}

auto test1_b()
{
    auto t = typeid(ED);
    assert(t.name == "ldc_enum.ED");
    auto init = *cast(double*)t.initializer.ptr;
    assert(init == 1.2);
}

void main()
{
    test1_a();
    test1_b();
}
