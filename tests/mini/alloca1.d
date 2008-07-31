module alloca1;

pragma(alloca) void* alloca(uint);

extern(C) int printf(char*, ...);

void main()
{
    int n = 16;
    int* p = cast(int*)alloca(n*int.sizeof);
    int[] a = p[0..n];
    a[] = 0;
    foreach(i,v; a) {
        printf("a[%2d] = %d\n", i, v);
    }
}
