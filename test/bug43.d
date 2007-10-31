module bug43;

struct S
{
    ubyte[3] vals;
}

void func(ubyte[3] v)
{
}

void main()
{
    S s;
    func(s.vals);
}
