extern(C) int printf(char*, ...);

void main()
{
    int[3] a = [1, 2, 3];
    int[3] b = [4, 5, 6];
    int[3] c;

    c[] = a[] + b[];

    printf("c.ptr = %p\n", c.ptr);
    printf("c.length = %lu\n", c.length);

    assert(c[0] == 5);
    assert(c[1] == 7);
    assert(c[2] == 9);
}
