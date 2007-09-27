module scope1;

void main()
{
    printf("1\n");
    {
        scope(exit) printf("2\n");
    }
    printf("3\n");
}
