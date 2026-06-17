module rt.sections_wasm;

version (WebAssembly):

import rt.deh;
import rt.minfo;

struct SectionGroup
{
    static int opApply(scope int delegate(ref SectionGroup) dg)
    {
        return dg(_sections);
    }

    static int opApplyReverse(scope int delegate(ref SectionGroup) dg)
    {
        return dg(_sections);
    }

    @property immutable(ModuleInfo*)[] modules() const nothrow @nogc
    {
        return _moduleGroup.modules;
    }

    @property ref inout(ModuleGroup) moduleGroup() inout return nothrow @nogc
    {
        return _moduleGroup;
    }

    //@property immutable(FuncTable)[] ehTables() const nothrow @nogc
    //{
        //auto pbeg = cast(immutable(FuncTable)*)&__start_deh;
        //auto pend = cast(immutable(FuncTable)*)&__stop_deh;
        //return pbeg[0 .. pend - pbeg];
    //    return null;
    //}

    @property inout(void[])[] gcRanges() inout return nothrow @nogc
    {
        //return _gcRanges[];
        return null;
    }

private:
    ModuleGroup _moduleGroup;
    //void[][1] _gcRanges;
}

void initSections() nothrow @nogc
{
    auto mbeg = cast(immutable ModuleInfo**)&__start___minfo;
    auto mend = cast(immutable ModuleInfo**)&__stop___minfo;
    _sections.moduleGroup = ModuleGroup(mbeg[0 .. mend - mbeg]);

    //auto pbeg = cast(void*)&__dso_handle;
    //auto pend = cast(void*)&_end;
    //_sections._gcRanges[0] = pbeg[0 .. pend - pbeg];
}

void finiSections() nothrow @nogc
{
}

void[] initTLSRanges() nothrow @nogc
{
    //auto pbeg = cast(void*)&_tlsstart;
    //auto pend = cast(void*)&_tlsend;
    //return pbeg[0 .. pend - pbeg];
    return null;
}

void finiTLSRanges(void[] rng) nothrow @nogc
{
}

void scanTLSRanges(void[] rng, scope void delegate(void* pbeg, void* pend) nothrow dg) nothrow
{
    dg(rng.ptr, rng.ptr + rng.length);
}

private:

__gshared SectionGroup _sections;

extern(C)
{
    extern __gshared
    {
        //void* __start_deh;
        //void* __stop_deh;
        void* __start___minfo;
        void* __stop___minfo;
        //int __dso_handle;
        //int _end;
    }

    //extern
    //{
    //    void* _tlsstart;
    //    void* _tlsend;
    //}
}
