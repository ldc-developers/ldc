module foreach4;
extern(C) int printf(char*, ...);
void main()
{
    int[] arr = new int[4];
    foreach(i,v;arr) {
        printf("arr[%u] = %d\n",i,v);
    }
}
