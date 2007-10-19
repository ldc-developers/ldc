module bug8;

void main()
{
    s = newS();
}

S* s;

struct S
{
    int i;
}

S* newS()
{
    auto tmp = new S;
    tmp.i = 4;
    return tmp;
}

