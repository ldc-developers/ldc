module bug22;
extern(C) int printf(char*, ...);

void main()
{
    int i;
    delegate {
        i = 42;
    }();
    printf("%d\n", i);
    assert(i == 42);
}
