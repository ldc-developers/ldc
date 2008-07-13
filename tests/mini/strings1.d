module strings1;
extern(C) int printf(char*, ...);
void f(char[11] buffer)
{
    printf("%.*s\n", buffer.length, buffer.ptr);
}

void main()
{
    char[11] buffer;
    char[] hello = "hello world";
    {buffer[] = hello[];}
    {f(buffer);}
    {f("eleven char");}
}
