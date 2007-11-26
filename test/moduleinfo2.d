module moduleinfo2;
import std.stdio;
void main()
{
    ModuleInfo[] mi = ModuleInfo.modules();
    writefln("listing ",mi.length," modules");
    foreach(m; mi)
    {
        writefln("  ",m.name);
    }
    assert(mi.length > 50);
}
