module globals1;

char[] gstr = "hello world";

void main()
{
    printf("%.*s\n", gstr.length, gstr.ptr);
    char[] str = gstr;
    printf("%.*s\n", str.length, str.ptr);
}
