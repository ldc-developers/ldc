module mainargs1;

void main(string[] args)
{
    foreach(v; args)
    {
        printf("%.*s\n", v.length, v.ptr);
    }
}
