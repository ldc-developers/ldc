module bug63;
extern(C) int printf(char*, ...);

void main()
{
    static void notnested()
    {
        printf("hello world\n");
    }
    notnested();
}
