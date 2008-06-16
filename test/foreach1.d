module foreach1;
extern(C) int printf(char*, ...);

void main()
{
    static arr = [1,2,3,4,5];

    printf("forward");
    foreach(v;arr) {
        printf(" %d",v);
    }
    printf("\nreverse");
    foreach_reverse(v;arr) {
        printf(" %d",v);
    }
    printf("\n");
}
