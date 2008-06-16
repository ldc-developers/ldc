module bug29;
extern(C) int printf(char*, ...);

void main()
{
    int[] arr16 = new int[4];
    {
        void[] arr = arr16;
        {
            printf("%lu\n", arr.length);
            {
                assert(arr.length == 16);
            }
        }
    }
}
