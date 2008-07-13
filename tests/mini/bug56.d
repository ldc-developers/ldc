module bug56;

void main()
{
    int[] a;
    a = [1,2,3];
    {int[] b = [4,5,6];}
}