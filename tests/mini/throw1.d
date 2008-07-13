module throw1;

extern(C) int rand();

class C
{
}

void func(bool b)
{
    if (b)
        throw new C;
}

int main()
{
    bool b = true;
    try
    {
        func(b);
    }
    catch(Object)
    {
        return 0;
    }
    return 1;
}
