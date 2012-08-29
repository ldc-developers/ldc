module tangotests.vararg1;

import tango.stdc.stdio;

void func(int[] arr...)
{
    printf("1,2,4,5,6 == %d,%d,%d,%d,%d\n", arr[0],arr[1],arr[2],arr[3],arr[4]);
}

void main()
{
    func(1,2,4,5,6);
}
