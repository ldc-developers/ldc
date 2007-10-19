module bug18;

struct S {
    int[9] i;
}

void main()
{
    int[9] i;
    auto s = S(i);
}
