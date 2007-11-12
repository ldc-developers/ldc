module bug61;

void main()
{
    int[3] a = [42,4,141414];
    printf("empty:\n");
    foreach(v; a[3..$]) {
        printf("int = %d\n", v);
    }
    printf("one element:\n");
    foreach(v; a[2..$]) {
        printf("int = %d\n", v);
    }
    printf("all elements:\n");
    foreach(v; a) {
        printf("int = %d\n", v);
    }
    printf("empty reversed:\n");
    foreach_reverse(v; a[3..$]) {
        printf("int = %d\n", v);
    }
    printf("all elements reversed:\n");
    foreach_reverse(v; a) {
        printf("int = %d\n", v);
    }
}
