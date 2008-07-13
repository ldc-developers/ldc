module arrays11;

void ints()
{
    int[] a = [1,2,3,4,5,6];
    {assert(a == a);}

    int[] b = [4,5,6,7,8,9];
    {assert(a != b);}
    {assert(a[3..$] == b[0..3]);}
}

void floats()
{
    float[] a = [1.0f, 2.0f, 3.0f, 4.0f];
    {assert(a == a);}

    float[] b = [2.0f, 3.0f, 5.0f];
    {assert(a != b);}
    {assert(a[1..3] == b[0..2]);}
}

struct S
{
    int i;
    int j;

    int opEquals(S s)
    {
        return (i == s.i) && (j == s.j);
    }
}

void structs()
{
    S[] a = [S(0,0), S(1,0), S(2,0), S(3,0)];
    {assert(a == a);}
    S[] b = [S(0,1), S(1,0), S(2,0), S(3,1)];
    {assert(a != b);}
    {assert(a[1..3] == b[1..3]);}

    S[2] c = [S(2,0), S(3,1)];
    {assert(c == b[2..$]);}
}

void main()
{
    ints();
    floats();
    structs();
}
