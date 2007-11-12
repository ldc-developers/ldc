module arrays5;
void main()
{
    auto arr = new float[5];
    {arr[4] = 1f;}
    {assert(arr[0] !<>= 0f);}
    {assert(arr[1] !<>= 0f);}
    {assert(arr[2] !<>= 0f);}
    {assert(arr[3] !<>= 0f);}
    {assert(arr[4] == 1f);}
}
