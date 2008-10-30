module constructors;

import tango.io.Console;

class C
{
    this()
    {
        Cout("C()").newline;
    }
    this(char[] str)
    {
        Cout("C(")(str)(")").newline;
    }
}

class D : C
{
    this()
    {
        super("D");
        Cout("D()").newline;
    }
}

void main()
{
    auto c1 = new C();
    auto c2 = new C("C");
    auto d = new D();
}
