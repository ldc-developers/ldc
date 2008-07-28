module mini.norun_debug8;

struct Str
{
    size_t len;
    void* ptr;
}

struct Foo
{
    long l;
    Bar bar;
}

struct Bar
{
    float x,y,z;
    Foo* foo;
}

void main()
{
    Str str;
    Foo foo;
    foo.l = 42;
    foo.bar.y = 3.1415;
    foo.bar.foo = &foo;

    int* fail;
    *fail = 0;
}
