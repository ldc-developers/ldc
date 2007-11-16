module bug63;

void main()
{
    static void notnested()
    {
        printf("hello world\n");
    }
    notnested();
}
