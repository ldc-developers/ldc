module bug59;

void main()
{
    int[2] a = 0;
    //func(a);
    a[0] = 1;
    int i = a[0];
    int* p = &a[0];
}

void func(int[2] a)
{
    int* p = cast(int*)a;
}

void func2(int[4] a)
{
    int* p = 3+cast(int*)a;
}
