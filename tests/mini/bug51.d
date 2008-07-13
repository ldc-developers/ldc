module bug51;
const ubyte[3] arr1 = 1;
const ubyte[3] arr2 = [1];
const ubyte[3] arr3 = [1:1];
void main()
{
    assert(arr1 == [cast(ubyte)1,1,1][]);
    assert(arr2 == [cast(ubyte)1,0,0][]);
    assert(arr3 == [cast(ubyte)0,1,0][]);
}
