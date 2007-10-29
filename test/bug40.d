module bug40;

char[] func(void* p)
{
    return null;
}

void main()
{
    char[] function(void*) fp = &func;
    assert(fp(null) is null);
}
