module bug38;

void func(int* p)
{
    p++;
}

void main()
{
    int i;
    func(&i);
}
