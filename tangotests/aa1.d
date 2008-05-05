module tangotests.aa1;

extern(C) int printf(char*,...);

void main()
{
    int[int] map;
    map[1] = 1;
    map[10] = 1;
    map[11] = 11;
    map[14] = 41;
    map[21] = 12;
    map[23] = 32;
    map[32] = 23;
    map[201] = 102;
    foreach(k,v; map)
    {
        printf("%d:%d ", k,v);
    }
    printf("\n");
}
