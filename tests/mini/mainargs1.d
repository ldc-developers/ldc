module mainargs1;

extern(C) int printf(char*,...);

void main(char[][] args)
{
    foreach(v; args)
    {
        printf("%.*s\n", v.length, v.ptr);
    }
}
