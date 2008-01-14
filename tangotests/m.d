void main()
{
    int* e = __errno_location();
    printf("&errno = %p\n", e);
    printf("errno = %d\n", *e);
}

extern(C):
int* __errno_location();
int printf(char*,...);
