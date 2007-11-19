module complex2;

void main()
{
    cdouble c = 3.0 + 0i;
    cdouble d = 2.0 + 0i;
    {
        cdouble c1 = c + 3.0;
        cdouble c2 = c - 3.0i;
    }
    {
        cdouble c1 = c / 2.0;
    }
}
