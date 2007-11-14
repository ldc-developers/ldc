module arrays12;

void ints()
{
    int[3] a = [1,2,3];
    int[3] b = [2,3,4];
    int[3] c = [2,5,0];
    {assert(a < b);}
    {assert(b > a);}
    {assert(a < c);}
    {assert(c > a);}
    {assert(b < c);}
    {assert(c > b);}
}

void main()
{
    ints();
}
