module bug48;

size_t func(void *p)
{
    return cast(size_t)*cast(void* *)p;
}

void main()
{
}
