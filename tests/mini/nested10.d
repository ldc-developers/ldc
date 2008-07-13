module nested10;

extern(C) int printf(char*, ...);

void main()
{
    int j = 3;
    void F()
    {
        int i = j;
        printf("F: i = %d, j = %d\n", i, j);
        void G()
        {
            printf("G: i = %d, j = %d\n", i, j);
            j += i;
        }
        G();
    }
    F();
    printf("6 = %d\n", j);
    assert(j == 6);
}
