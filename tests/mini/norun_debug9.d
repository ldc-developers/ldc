module mini.norun_debug9;

struct Foo
{
    int a,b,c;

    void func()
    {
        int* fail = cast(int*) 1;
        *fail = 0;
    }
}

void main()
{
    Foo foo = Foo(1,10,73);
    foo.func();
}
