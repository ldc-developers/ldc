module tangotests.gc1;

void main()
{
    int[] arr;

    for (int i=0; i<100; ++i)
    {
        arr ~= new int[1000];
    }

    printf("arr.length = %u\n", arr.length);
}

extern(C) int printf(char*, ...);