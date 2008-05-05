module tangotests.vararg1;

import tango.io.Stdout;

void func(int[] arr...)
{
    Stdout.formatln("1,2,4,5,6 == {},{},{},{},{}", arr[0],arr[1],arr[2],arr[3],arr[4]);
}

void main()
{
    func(1,2,4,5,6);
}
