module bug29;

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
