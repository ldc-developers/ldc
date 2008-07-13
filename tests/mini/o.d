extern(C) int printf(char*, ...);

void func()
{
    try
    {
        printf("try\n");
        return 0;
    }
    catch
    {
        printf("catch\n");
    }
    finally
    {
        printf("finally\n");
    }
    return 0;
}

void main()
{
    func();
}
