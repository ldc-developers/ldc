module foreach5;
extern(C) int printf(char*, ...);
void main()
{
    int[3] arr = [1,2,3];

    foreach(v;arr) {
        v++;
    }

    printf("%d\n", arr[0]);
    assert(arr[0] == 1);
    assert(arr[1] == 2);
    assert(arr[2] == 3);

    foreach(ref v;arr) {
        v++;
    }

    printf("%d\n", arr[0]);
    assert(arr[0] == 2);
    assert(arr[1] == 3);
    assert(arr[2] == 4);
}
