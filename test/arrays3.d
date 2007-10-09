module arrays3;

void main()
{
    int[] arr;
    {arr = new int[25];}
    {assert(arr.length == 25);}
    {arr.length = arr.length + 5;}
    {assert(arr.length == 30);}
}
