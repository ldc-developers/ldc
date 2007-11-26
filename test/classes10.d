module classes10;

class C
{
    int i;
    override char[] toString()
    {
        return "foobar";
    }
}

void main()
{
    Object o = new C;
    char[] s = o.toString();
    assert(s == "foobar");
}
