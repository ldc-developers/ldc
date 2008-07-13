module bug30;

void main()
{
    int[] a = new int[4];
    {a[0] = 1;
    a[1] = 2;
    a[2] = 3;
    a[3] = 4;}
    int[] b = new int[4];
    {b[0] = 1;
    b[1] = 2;
    b[2] = 3;
    b[3] = 4;}
    int[] c = new int[4];
    {c[0] = 1;
    c[1] = 2;
    c[2] = 4;
    c[3] = 3;}
    {assert(a == b);}
    {assert(a != c);}
}
