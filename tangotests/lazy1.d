module tangotests.lazy1;

extern(C) int printf(char*, ...);

void main()
{
    lazystr("whee\n");
}

void lazystr(lazy char[] msg)
{
    printf("%.*s", msg.length, msg.ptr);
}
