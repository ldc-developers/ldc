module tangotests.lazy2;

extern(C) int printf(char*, ...);

void main()
{
    lazy1("hello\n");
}

void lazy1(lazy char[] str)
{
    lazy2(str);
}

void lazy2(lazy char[] msg)
{
    printf("%.*s", msg.length, msg.ptr);
}
