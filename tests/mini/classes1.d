module tangotests.classes1;

class Outer
{
    int data;

    class Inner
    {
        long data;

        this(long d)
        {
            data = d*2;
        }
    }

    void func()
    {
        auto i = new Inner(data);
        data += (i.data/4);
    }

    this(int d)
    {
        data = d;
    }
}

void main()
{
    scope c = new Outer(100);
    c.func();
    int d = c.data;
    printf("150 = %d\n", d);
}

extern(C) int printf(char*, ...);
