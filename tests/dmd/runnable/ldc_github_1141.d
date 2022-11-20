class DClass { int a = 1; }

extern (C++)
class CppClass { int a = 1; }

void main()
{
    auto d = new DClass();
    auto cpp = new CppClass();
    assert(d.a == 1);
    assert(cpp.a == 1);
}
