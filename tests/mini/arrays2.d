char[] get()
{
    return "hello";
}

void param(char[] s)
{
}

void refparam(ref char[] s)
{
}

void main()
{
    char[] dstr = get();
    param(dstr);
    refparam(dstr);
}
