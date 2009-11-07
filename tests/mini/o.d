extern(C) int printf(char*, ...);

void func()
{
    try
    {
        printf("try\n");
        return;
    }
    catch
    {
        printf("catch\n");
    }
    finally
    {
        printf("finally\n");
    }
    return;
}

void main()
{
    func();
}
