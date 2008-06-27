module arrays;

extern(C) int printf(char*, ...);

void integer()
{
    auto arr = new int[16];
    arr[1] = 42;
    arr[6] = 555;
    print_int(arr);
    delete arr;
}

void floating()
{
    auto arr = new float[6];
    arr[1] = 3.14159265;
    arr[3] = 1.61803399;
    print_float(arr);
    delete arr;
}

void print_int(int[] arr)
{
    printf("arr[%lu] = [", arr.length);
    for (auto i=0; i<arr.length; i++)
        printf("%d,", arr[i]);
    printf("\b]\n");
}

void print_float(float[] arr)
{
    printf("arr[%lu] = [", arr.length);
    for (auto i=0; i<arr.length; i++)
        printf("%f,", arr[i]);
    printf("\b]\n");
}

void main()
{
    integer();
    floating();
}
