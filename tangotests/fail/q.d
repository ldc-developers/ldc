class E : Exception
{
    this(char[] msg)
    {
        super(msg);
    }

    char[] toString()
    {
        return super.toString();
    }
}

extern(C) int printf(char*, ...);

void main()
{
    auto e = new E("hello world");
    auto msg = e.toString();
    printf("message should be: '%.*s'\n", msg.length, msg.ptr);
    throw e;
}
