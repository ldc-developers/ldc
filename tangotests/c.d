class Foo
{
    int i;
}

class Bar : Foo
{
    int j;
}

void func()
{
    scope c = new Bar;
    func2(c);
}

void func2(Bar c)
{
    c.i = 123;
}

void main()
{
    func();
}