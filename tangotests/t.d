interface MyInterface
{
    void func();
}

abstract class MyBaseClass : MyInterface
{
    abstract void func();
}

class MyClass : MyBaseClass
{
    void func()
    {
    }

    MyBaseClass toBase()
    {
        return this;
    }
}

void main()
{
    printf("STARTING\n");
    auto c = new MyClass;
    printf("c = %p\n", c);
    auto b = c.toBase;
    printf("b = %p\n", b);
    printf("FINISHED\n");
}

extern(C) int printf(char*, ...);
