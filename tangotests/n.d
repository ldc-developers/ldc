struct Structure
{
    static void static_method()
    {
    }

    void method()
    {
    }
}

void main()
{
    //Structure.static_method();

    Structure s;
    s.method();
}
