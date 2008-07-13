module moduleinfo2;

extern(C) int printf(char*, ...);

void main()
{
    printf("listing modules:\n");
    foreach(m; ModuleInfo)
    {
        printf("  %.*s\n", m.name.length, m.name.ptr);
    }
}
