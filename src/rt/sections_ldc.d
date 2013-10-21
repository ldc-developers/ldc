/**
 * This module provides support for the legacy _Dmodule_ref-based ModuleInfo
 * discovery mechanism. Multiple images (i.e. shared libraries) are not handled
 * at all.
 *
 * It is expected to fade away as work on the compiler-provided functionality
 * required for proper shared library support continues.
 *
 * Copyright: Copyright David Nadlinger 2013.
 * License: Distributed under the
 *      $(LINK2 http://www.boost.org/LICENSE_1_0.txt, Boost Software License 1.0).
 *    (See accompanying file LICENSE)
 * Authors: David Nadlinger, Martin Nowak
 * Source: $(DRUNTIMESRC src/rt/_sections_win64.d)
 */

module rt.sections_ldc;

version (linux) {} else version(LDC):

import core.stdc.stdlib : alloca;
import rt.minfo;

version (FreeBSD)
{
    version = UseELF;

    import core.sys.freebsd.sys.elf;
    import core.sys.freebsd.sys.link_elf;
}

struct SectionGroup
{
    static int opApply(scope int delegate(ref SectionGroup) dg)
    {
        return dg(globalSectionGroup);
    }

    static int opApplyReverse(scope int delegate(ref SectionGroup) dg)
    {
        return dg(globalSectionGroup);
    }

    @property inout(ModuleInfo*)[] modules() inout
    {
        return _moduleGroup.modules;
    }

    @property ref inout(ModuleGroup) moduleGroup() inout
    {
        return _moduleGroup;
    }

    @property inout(void[])[] gcRanges() inout
    {
        return _gcRanges[];
    }

private:
    ModuleGroup _moduleGroup;

    import rt.util.container;
    Array!(void[]) _gcRanges;
}
private __gshared SectionGroup globalSectionGroup;

private
{
    version (OSX)
    {
        import core.sys.osx.mach.dyld;
        import core.sys.osx.mach.getsect;
        import core.sys.osx.mach.loader;

        struct Section
        {
            immutable(char)* segment;
            immutable(char)* section;
        }

        immutable Section[3] dataSections = [
            Section(SEG_DATA, SECT_DATA),
            Section(SEG_DATA, SECT_BSS),
            Section(SEG_DATA, SECT_COMMON)
        ];
    }
    else version (Win64)
    {
        extern extern (C) __gshared
        {
            void* _data_start__;
            void* _data_end__;
            void* _bss_start__;
            void* _bss_end__;
        }
    }
    else version (Windows)
    {
        extern extern (C) __gshared
        {
            int _data_start__;
            int _bss_end__;
        }
    }
    else version (UseELF)
    {
        nothrow
        void findDataSection(void[]* data)
        {
            static extern(C) nothrow
            int callback(dl_phdr_info* info, size_t sz, void* arg)
            {
                auto range = cast(void[]*) arg;
                foreach (i, ref phdr; info.dlpi_phdr[0 .. info.dlpi_phnum])
                {
                    if (phdr.p_type == PT_LOAD && phdr.p_flags == (PF_W|PF_R))
                    {
                        *range = (cast(void*)phdr.p_vaddr)[0 .. phdr.p_memsz];
                        return 1;
                    }
                }
                return 0;
            }

            dl_iterate_phdr(&callback, data);
        }

        struct TLSInfo { size_t moduleId, size; }
        TLSInfo getTLSInfo(in ref dl_phdr_info info)
        {
            foreach (ref phdr; info.dlpi_phdr[0 .. info.dlpi_phnum])
            {
                if (phdr.p_type == PT_TLS)
                {
                    return TLSInfo(info.dlpi_tls_modid, phdr.p_memsz);
                }
            }
            assert(0, "Failed to determine TLS size.");
        }

        nothrow
        bool findPhdrForAddr(in void* addr, dl_phdr_info* result=null)
        {
            static struct DG { const(void)* addr; dl_phdr_info* result; }

            extern(C) nothrow
            int callback(dl_phdr_info* info, size_t sz, void* arg)
            {
                auto p = cast(DG*)arg;
                if (findSegmentForAddr(*info, p.addr))
                {
                    if (p.result !is null) *p.result = *info;
                    return 1; // break;
                }
                return 0; // continue iteration
            }

            auto dg = DG(addr, result);
            return dl_iterate_phdr(&callback, &dg) != 0;
        }

        nothrow
        bool findSegmentForAddr(in ref dl_phdr_info info, in void* addr, ElfW!"Phdr"* result=null)
        {
            if (addr < cast(void*)info.dlpi_addr) // quick reject
                return false;

            foreach (ref phdr; info.dlpi_phdr[0 .. info.dlpi_phnum])
            {
                auto beg = cast(void*)(info.dlpi_addr + phdr.p_vaddr);
                if (cast(size_t)(addr - beg) < phdr.p_memsz)
                {
                    if (result !is null) *result = phdr;
                    return true;
                }
            }
            return false;
        }

        struct tls_index
        {
            size_t ti_module;
            size_t ti_offset;
        }

        extern(C) void* __tls_get_addr(tls_index* ti);

        void[] getTLSRange(TLSInfo info)
        {
            if (info.moduleId == 0) return null;
            auto ti = tls_index(info.moduleId, 0);
            return __tls_get_addr(&ti)[0 .. info.size];
        }
    }
    else version (Solaris)
    {
        extern extern(C) __gshared
        {
            int _environ;
            int _end;
        }
    }
}

void initSections()
{
    globalSectionGroup.moduleGroup = ModuleGroup(getModuleInfos());

    static void pushRange(void* start, void* end)
    {
        globalSectionGroup._gcRanges.insertBack(start[0 .. (end - start)]);
    }

    version (OSX)
    {
        static extern(C) void scanSections(in mach_header* hdr, ptrdiff_t slide)
        {
            foreach (s; dataSections)
            {
                // Should probably be decided at runtime by actual image bitness
                // (mach_header.magic) rather than at build-time?
                version (D_LP64)
                    auto sec = getsectbynamefromheader_64(
                        cast(mach_header_64*)hdr, s.segment, s.section);
                else
                    auto sec = getsectbynamefromheader(hdr, s.segment, s.section);

                if (sec == null || sec.size == 0)
                    continue;

                globalSectionGroup._gcRanges.insertBack(
                    (cast(void*)(sec.addr + slide))[0 .. sec.size]);
            }
        }
        _dyld_register_func_for_add_image(&scanSections);
    }
    else version (Win64)
    {
        pushRange(&_data_start__, &_data_end__);
        if (_bss_start__ != null)
        {
            pushRange(&_bss_start__, &_bss_end__);
        }
    }
    else version (Windows)
    {
        pushRange(&_data_start__, &_bss_end__);
    }
    else version (UseELF)
    {
        // Add data section based on ELF image
        void[] data = void;
        findDataSection(&data);
        globalSectionGroup._gcRanges.insertBack(data);

        // Explicitly add TLS range for main thread.
        dl_phdr_info phdr = void;
        findPhdrForAddr(&globalSectionGroup, &phdr) || assert(0);
        globalSectionGroup._gcRanges.insertBack(getTLSRange(getTLSInfo(phdr)));
    }
    else version (Solaris)
    {
        pushRange(&_environ, &_end);
    }
}

void finiSections()
{
    import core.stdc.stdlib : free;
    free(globalSectionGroup.modules.ptr);
}

private
{
    version (OSX)
    {
        extern(C) void _d_dyld_getTLSRange(void*, void**, size_t*);
        private ubyte dummyTlsSymbol;
    }
    else version (Windows)
    {
        extern(C) extern
        {
            int _tls_start;
            int _tls_end;
        }
    }
}

void[] initTLSRanges()
{
    version (OSX)
    {
        void* start = null;
        size_t size = 0;
        _d_dyld_getTLSRange(&dummyTlsSymbol, &start, &size);
        assert(start && size, "Could not determine TLS range.");
        return start[0 .. size];
    }
    else version (linux)
    {
        // glibc allocates the TLS area for each new thread at the stack of
        // the stack, so we only need to do something for the main thread.
        return null;
    }
    else version (FreeBSD)
    {
        return null;
    }
    else version (Solaris)
    {
        static assert(0, "TLS range detection not implemented on Solaris.");
    }
    else version (Windows)
    {
        auto pbeg = cast(void*)&_tls_start;
        auto pend = cast(void*)&_tls_end;
        return pbeg[0 .. pend - pbeg];
    }
    else static assert(0, "TLS range detection not implemented for this OS.");

}

void finiTLSRanges(void[] rng)
{
}

void scanTLSRanges(void[] rng, scope void delegate(void* pbeg, void* pend) dg)
{
    if (rng) dg(rng.ptr, rng.ptr + rng.length);
}

extern (C) __gshared ModuleReference* _Dmodule_ref;   // start of linked list

private:

// This linked list is created by a compiler generated function inserted
// into the .ctor list by the compiler.
struct ModuleReference
{
    ModuleReference* next;
    ModuleInfo*      mod;
}

ModuleInfo*[] getModuleInfos()
{
    import core.stdc.stdlib : malloc;

    size_t len = 0;
    for (auto mr = _Dmodule_ref; mr; mr = mr.next)
        len++;

    auto result = (cast(ModuleInfo**)malloc(len * size_t.sizeof))[0 .. len];

    auto tip = _Dmodule_ref;
    foreach (ref r; result)
    {
        r = tip.mod;
        tip = tip.next;
    }

    return result;
}
