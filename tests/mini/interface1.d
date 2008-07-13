module interface1;

extern(C) int printf(char*,...);

interface Inter
{
    void func();
}

class Class : Inter
{
    override void func()
    {
        printf("hello world\n");
    }
}

void main()
{
    scope c = new Class;
    c.func();
    Inter i = c;
    i.func();
}
