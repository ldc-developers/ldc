module scope3;

void main()
{
    int i;
    while (i < 10) {
        scope(success) i++;
    }
    printf("%d\n", i);
}
