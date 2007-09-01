module a;

int i = 42;

void main()
{
    int j;
    j = i;
    int* p;
    p = &j;
    int k = *p;
    assert(k == 42);

    byte b = -1;
    int i = b;
    assert(i == -1);

    int* p2 = p;

    printf("Done\n");
}
