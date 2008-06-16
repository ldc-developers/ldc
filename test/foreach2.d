module foreach2;
extern(C) int printf(char*, ...);
void main()
{
    static arr = [1.0, 2.0, 4.0, 8.0, 16.0];
    foreach(i,v; arr)
    {
        printf("arr[%d] = %f\n", i, v);
    }
    printf("-------------------------------\n");
    foreach_reverse(i,v; arr)
    {
        printf("arr[%d] = %f\n", i, v);
    }
}
