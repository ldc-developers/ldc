module bug34;

class MyTypeInfo_Pointer
{
    char[] toString() { return m_next.toString() ~ "*"; }

    int opEquals(Object o)
    {   TypeInfo_Pointer c;

    return this is o ||
        ((c = cast(TypeInfo_Pointer)o) !is null &&
         this.m_next == c.m_next);
    }

    hash_t getHash(void *p)
    {
        return cast(uint)*cast(void* *)p;
    }

    int equals(void *p1, void *p2)
    {
        return cast(int)(*cast(void* *)p1 == *cast(void* *)p2);
    }

    int compare(void *p1, void *p2)
    {
    if (*cast(void* *)p1 < *cast(void* *)p2)
        return -1;
    else if (*cast(void* *)p1 > *cast(void* *)p2)
        return 1;
    else
        return 0;
    }

    size_t tsize()
    {
    return (void*).sizeof;
    }

    void swap(void *p1, void *p2)
    {   void* tmp;
    tmp = *cast(void**)p1;
    *cast(void**)p1 = *cast(void**)p2;
    *cast(void**)p2 = tmp;
    }

    TypeInfo next() { return m_next; }
    uint flags() { return 1; }

    TypeInfo m_next;
}

void main()
{
}
