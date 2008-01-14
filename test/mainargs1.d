module mainargs1;

extern(C) int printf(char*,...);

void main(string[] args)
{
    foreach(v; args)
    {
        printf("%.*s\n", v.length, v.ptr);
    }
}
