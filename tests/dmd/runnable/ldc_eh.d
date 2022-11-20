struct S1
{
    int x;

    this(this)
    {
    }

    ~this()
    {
    }
}

S1 createS1()
{
    // dmdfe creates try-finally there to call destructor for s1,
    // later the statement is rewrited to try-catch, because
    // s1 is a nrvo variable. Test goto in such case.
    S1 s1;
    s1.x = 1;
    if(true)
        goto Lexit;
    s1.x = 2;
    Lexit:
        return s1;
}

void test1()
{
    auto s1 = createS1();
    assert(s1.x == 1);
}


void main()
{
    test1();
}

