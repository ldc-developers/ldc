interface Inter
{
    int func();
}

extern(C) int printf(char*, ...);

class InterClass : Inter
{
    int func()
    {
        return printf("InterClass.func()\n");
    }
}

alias int delegate() dg_t;

void main()
{
    scope c = new InterClass;

    {
    Inter i = cast(Inter)c;
        {
        dg_t dg = &i.func;
            {
            int j = dg();
            }
        }
    }
}
