module aa6;

void main()
{
    int[int] aa;
    aa = [1:1, 2:4, 3:9, 4:16];
    printf("---\n");
    foreach(int k, int v; aa)
        printf("aa[%d] = %d\n", k, v);
    aa.rehash;
    printf("---\n");
    foreach(int k, int v; aa)
        printf("aa[%d] = %d\n", k, v);
    size_t n = aa.length;
    assert(n == 4);
    int[] keys = aa.keys;
    assert(keys[] == [1,2,3,4][]);
    int[] vals = aa.values;
    assert(vals[] == [1,4,9,16][]);
    aa.remove(3);
    printf("---\n");
    foreach(int k, int v; aa)
        printf("aa[%d] = %d\n", k, v);
    assert(aa.length == 3);
}
