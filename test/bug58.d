module bug58;
import std.stdio;
void main()
{
    int[16] arr = [1,16,2,15,3,14,4,13,5,12,6,11,7,10,8,9];
    writefln("arr = ",arr);
    arr.sort;
    writefln("arr.sort = ",arr);
    assert(arr == [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]);
}