module s;

interface Inter
{
    void inter();
}

interface Inter2
{
    void inter2();
}

interface InterOne : Inter
{
    void interOne();
}

abstract class ClassAbstract : InterOne
{
    abstract void inter();
    abstract void interOne();
}

class TheClassOne : ClassAbstract
{
    void inter()
    {
    }
    void interOne()
    {
    }
}

class TheClassTwo : TheClassOne, Inter2
{
    long l;
    double d;

    void inter2()
    {
    }
}

extern(C) int printf(char*, ...);

void main()
{
    printf("classinfo test\n");
    {
        auto c = new TheClassOne;
        {
            auto ci = c.classinfo;
            printf("ci = %.*s\n", ci.name.length, ci.name.ptr);
            printf("ci.interfaces.length = %lu\n", ci.interfaces.length);
            printf("i[0] = %.*s\n", ci.interfaces[0].classinfo.name.length, ci.interfaces[0].classinfo.name.ptr);
            printf("i[1] = %.*s\n", ci.interfaces[1].classinfo.name.length, ci.interfaces[1].classinfo.name.ptr);
        }
    }
    {
        auto c = new TheClassTwo;
        {
            auto ci = c.classinfo;
            printf("ci = %.*s\n", ci.name.length, ci.name.ptr);
            printf("ci.interfaces.length = %lu\n", ci.interfaces.length);
            printf("i[0] = %.*s\n", ci.interfaces[0].classinfo.name.length, ci.interfaces[0].classinfo.name.ptr);
            printf("i[1] = %.*s\n", ci.interfaces[1].classinfo.name.length, ci.interfaces[1].classinfo.name.ptr);
            printf("i[2] = %.*s\n", ci.interfaces[2].classinfo.name.length, ci.interfaces[2].classinfo.name.ptr);
        }
        auto i = cast(InterOne)c;
        {
            auto ci = i.classinfo;
            printf("ci = %.*s\n", ci.name.length, ci.name.ptr);
        }
        auto i2 = cast(Inter2)i;
        {
            auto ci = i2.classinfo;
            printf("ci = %.*s\n", ci.name.length, ci.name.ptr);
        }
        auto o = cast(Object)i2;
        {
            auto ci = o.classinfo;
            printf("ci = %.*s\n", ci.name.length, ci.name.ptr);
        }
    }
}
