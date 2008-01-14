interface Iin
{
    void[] read(size_t n);
}

interface Iout
{
    size_t write(void[] d);
}

class C : Iin
{
    void[] read(size_t n)
    {
        return null;
    }

    size_t write(void[] d)
    {
        return 0;
    }
}

void func()
{
    scope c = new C;
}
