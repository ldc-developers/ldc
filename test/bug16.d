module bug16;

void func(long val)
{
    val >>= 32;
}

void main()
{
    func(64L);
}
