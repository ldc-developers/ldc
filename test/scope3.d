module scope3;
extern(C) int printf(char*, ...);
void main()
{
    int i;
    while (i < 10) {
        scope(success) i++;
    }
    printf("%d\n", i);
}
