module bug62;
extern(C) int printf(char*, ...);

void main()
{
    int[] arr = [1,2,5,7,9];
    int i = 0;
    foreach(v; arr) {
        i += v;
    }
    printf("sum = %d\n", i);
    assert(i == 24);
}