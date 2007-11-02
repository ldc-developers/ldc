module templ1;

T func1(T)(T a)
{
    static T b = a;
    return b;
}

void main()
{
}
