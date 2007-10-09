module throw1;

extern(C) int rand();

class C
{
}

void func()
{
    if (rand() & 1)
        throw new C;
}

int main()
{
    try
    {
        func();
    }
    catch(Object)
    {
        return 1;
    }
    return 0;
}
