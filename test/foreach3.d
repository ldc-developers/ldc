module foreach3;

void main()
{
    static str = ['h','e','l','l','o'];
    foreach(i,v; str) {
        printf("%c",v);
    }
    printf("\n");

    foreach(i,v; str) {
        v++;
    }

    foreach(i,v; str) {
        printf("%c",v);
    }
    printf("\n");
}
