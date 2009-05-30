void main()
{
    int* e = __errno_location();
    printf("&errno = %p\n", e);
    printf("errno = %d\n", *e);
}

extern(C):
version(darwin) {
    int* __error();
    alias __error __errno_location;
} else version (mingw32) {
    int* strerror();
    alias strerror __errno_location;
} else {
    int* __errno_location();
}
int printf(char*,...);
