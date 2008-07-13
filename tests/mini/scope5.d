module scope5;

int i;

void func(int a, int b)
{
    i = 0;
    {
        scope(exit) i++;
        if (a) {
            scope(exit) i++;
            if (b) return;
            i++;
        }
    }
    i++;
}

void main()
{
    func(0,0);
    assert(i == 2);
    func(1,1);
    assert(i == 2);
    func(1,0);
    assert(i == 4);
}
