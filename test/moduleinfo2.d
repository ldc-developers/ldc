module moduleinfo2;

extern(C) int printf(char*, ...);

void main()
{
    ModuleInfo[] mi = ModuleInfo.modules();
    printf("listing %u modules:\n");
    foreach(m; mi)
    {
        printf("  %s\n", m.name.length, m.name.ptr);
    }
    assert(mi.length > 50);
}
