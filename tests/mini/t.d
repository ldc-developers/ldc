interface MyInterface
{
    int func();
}

class MyClass : MyInterface
{
    int var;
    int func()
    {
        return var;
    }
}

void func1(MyInterface i)
{
    int delegate() dg = &i.func;
    func2(dg);
}

extern(C) int printf(char*, ...);

void func2(int delegate() dg)
{
    int i = dg();
    printf("%d\n", i);
}

void main()
{
    auto c = new MyClass;
    c.var = 42;
    func1(c);
}
