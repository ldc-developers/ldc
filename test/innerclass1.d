module innerclass1;

class Outer
{
    int i;
    class Inner
    {
        int func()
        {
            return i;
        }
    }
}

void main()
{
    Outer o = new Outer;
    {
        o.i = 42;
        {
            auto i = o.new Inner;
            {
                int x = i.func();
                assert(x == 42);
            }
        }
    }
    printf("SUCCESS\n");
}
