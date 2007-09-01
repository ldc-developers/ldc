module virtcall;

class C
{
    override char[] toString()
    {
        return "overridden";
    }
}

void main()
{
    C c = new C;
    auto s = c.toString();
}
