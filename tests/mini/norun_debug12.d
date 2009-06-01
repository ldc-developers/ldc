module mini.norun_debug12;

interface I
{
    int foo();
}

class C : I
{
    int i = 24;
    int foo()
    {
        return i;
    }
}

void main()
{
    scope c = new C;
    I i = c;

    int* fail = cast(int*) 1;
    *fail = 0;
}
