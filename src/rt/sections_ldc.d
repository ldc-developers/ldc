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

version (linux) {} else version (FreeBSD) {} else version(LDC):

import core.stdc.stdlib : alloca;
import rt.minfo;
debug(PRINTF) import core.stdc.stdio : printf;

version (Solaris)
{
    version = UseELF;

    import core.sys.solaris.link;
    import core.sys.solaris.sys.elf;
}

alias SectionGroup DSO;
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

    @property immutable(ModuleInfo*)[] modules() const
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

    import rt.util.container.array;
    Array!(void[]) _gcRanges;

    version(Solaris)
    {
        size_t _tlsSize;
    }
    else version(UseELF)
    {
        size_t _tlsMod;
        size_t _tlsSize;
    }
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
    else version (CRuntime_Microsoft)
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
        /************
         * Scan segments in Linux dl_phdr_info struct and store
         * the TLS and writeable data segments in *pdso.
         */
        void scanSegments(in ref dl_phdr_info info, DSO* pdso)
        {
            foreach (ref phdr; info.dlpi_phdr[0 .. info.dlpi_phnum])
            {
                switch (phdr.p_type)
                {
                case PT_LOAD:
                    if (phdr.p_flags & PF_W) // writeable data segment
                    {
                        auto beg = cast(void*)(info.dlpi_addr + phdr.p_vaddr);
                        pdso._gcRanges.insertBack(beg[0 .. phdr.p_memsz]);
                    }
                    break;

                case PT_TLS: // TLS segment
                    assert(!pdso._tlsSize); // is unique per DSO
                    version(Solaris)
                    {
                        pdso._tlsSize = phdr.p_memsz;
                    }
                    else
                    {
                        pdso._tlsMod = info.dlpi_tls_modid;
                        pdso._tlsSize = phdr.p_memsz;
                    }
                    break;

                default:
                    break;
                }
            }
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

        version(Solaris)
        {
            /* Solaris does not support the dl_phdr_info.dlpi_tls_modid field.
             * The static TLS range is placed immediately preceding the thread
             * pointer. Accesses to this TLS data is based off of subtractions
             * from the current thread pointer.
             * See: https://docs.oracle.com/cd/E26502_01/html/E26507/gentextid-23191.html#chapter8-7
             */

            /*
             * The following data structures are private to libc. They are not
             * required for the implementation but they help in debugging this
             * stuff.
             */
            struct mutex_t
            {
                ubyte[24] __fill;
            }

            struct tls_t
            {
                void*  tls_data;
                size_t tls_size;
            }

            struct tls_metadata_t
            {
                mutex_t                                      tls_lock;
                tls_t                                        tls_modinfo;
                tls_t                                        static_tls;
                byte[64 - mutex_t.sizeof -2 * tls_t.sizeof]  tls_pad;
            }

            struct uberdata_t
            {
                byte[10496] __fill;
                tls_metadata_t tls_metadata;
                /* incomplete */
            }

            struct ulwp_t
            {
                version(SPARC)
                {
                    uint     ul_dinstr;
                    uint[15] ul_padsparc0;
                    uint     ul_dsave;
                    uint     ul_drestore;
                    uint     ul_dftret;
                    uint     ul_dreturn;
                }
                version(SPARC64)
                {
                    uint     ul_dinstr;
                    uint[15] ul_padsparc0;
                    uint     ul_dsave;
                    uint     ul_drestore;
                    uint     ul_dftret;
                    uint     ul_dreturn;
                }
                ulwp_t* ul_self;
                version (D_LP64)
                    ubyte[56] ul_dinstr;
                else
                    ubyte[40] ul_dinstr;
                uberdata_t*  ul_uberdata;
                tls_t        ul_tls;
                ulwp_t*      ul_forw;
                ulwp_t*      ul_back;
                ulwp_t*      ul_next;
                ulwp_t*      ul_hash;
                void*        ul_rval;
                /* incomplete */
            }

            // Return the current thread pointer.
            private ulwp_t* curthread() nothrow
            {
                import ldc.llvmasm;

                version (X86_64)
                {
                    return __asm!(ulwp_t*)("movq %fs:0, $0", "=r");
                }
                else version (X86)
                {
                    return __asm!(ulwp_t*)("movl %gs:0, $0", "=r");
                }
                else
                {
                     static assert(0, "TLS range detection not implemented on Solaris for this architecture.");
                }
            }

            // See: http://src.illumos.org/source/xref/illumos-gate/usr/src/cmd/sgs/include/i386/machdep_x86.h#102
            version (D_LP64)
                enum M_TLSSTATALIGN = 0x10;
            else
                enum M_TLSSTATALIGN = 0x08;

            void[] getTLSRange(DSO* pdso)
            {
                // See: http://src.illumos.org/source/xref/illumos-gate/usr/src/cmd/sgs/libld/common/machrel.intel.c#996
                //     tlsstatsize = S_ROUND(ofl->ofl_tlsphdr->p_memsz, M_TLSSTATALIGN);
                void* thptr = curthread();
                size_t sz = (pdso._tlsSize + (M_TLSSTATALIGN-1) + 512) & ~(M_TLSSTATALIGN-1);
                return (thptr - sz)[0 .. sz];
            }
        }
        else
        {
            struct tls_index
            {
                size_t ti_module;
                size_t ti_offset;
            }

            extern(C) void* __tls_get_addr(tls_index* ti);

            /* The dynamic thread vector (DTV) pointers may point 0x8000 past the start of
             * each TLS block. This is at least true for PowerPC and Mips platforms.
             */
            version(PPC)
                enum TLS_DTV_OFFSET = 0x8000;
            else version(PPC64)
                enum TLS_DTV_OFFSET = 0x8000;
            else version(MIPS)
                enum TLS_DTV_OFFSET = 0x8000;
            else version(MIPS64)
                enum TLS_DTV_OFFSET = 0x8000;
            else
                enum TLS_DTV_OFFSET = 0x;

            void[] getTLSRange(DSO* pdso)
            {
                if (pdso._tlsMod == 0) return null;
                auto ti = tls_index(pdso._tlsMod, 0);
                return (__tls_get_addr(&ti)-TLS_DTV_OFFSET)[0 .. pdso._tlsSize];
            }
        }
    }
}

/****
 * Gets called on program startup just before GC is initialized.
 */
void initSections()
{
    debug(PRINTF) printf("initSections called\n");
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
    else version (CRuntime_Microsoft)
    {
        pushRange(_data_start__, _data_end__);
        if (_bss_start__ != null)
        {
            pushRange(_bss_start__, _bss_end__);
        }
    }
    else version (Windows)
    {
        pushRange(&_data_start__, &_bss_end__);
    }
    else version (UseELF)
    {
        dl_phdr_info phdr = void;
        findPhdrForAddr(&globalSectionGroup, &phdr) || assert(0);

        scanSegments(phdr, &globalSectionGroup);
    }
}

/***
 * Gets called on program shutdown just after GC is terminated.
 */
void finiSections()
{
    debug(PRINTF) printf("finiSections called\n");
    import core.stdc.stdlib : free;
    free(cast(void*)globalSectionGroup.modules.ptr);
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

/***
 * Called once per thread; returns array of thread local storage ranges
 */
void[] initTLSRanges()
{
    debug(PRINTF) printf("initTLSRanges called\n");
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
    else version (UseELF)
    {
        auto rng = getTLSRange(&globalSectionGroup);
        debug(PRINTF) printf("Add range %p %d\n", rng ? rng.ptr : cast(void*)0, rng ? rng.length : 0);
        return rng;
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
    debug(PRINTF) printf("finiTLSRanges called\n");
}

void scanTLSRanges(void[] rng, scope void delegate(void* pbeg, void* pend) nothrow dg) nothrow
{
    debug(PRINTF) printf("scanTLSRanges called (rng = %p %d)\n", rng ? rng.ptr : cast(void*)0, rng ? rng.length : 0);
    if (rng) dg(rng.ptr, rng.ptr + rng.length);
}

extern (C) __gshared ModuleReference* _Dmodule_ref;   // start of linked list

private:

// This linked list is created by a compiler generated function inserted
// into the .ctor list by the compiler.
struct ModuleReference
{
    ModuleReference* next;
    immutable(ModuleInfo)* mod;
}

immutable(ModuleInfo*)[] getModuleInfos()
out (result)
{
    foreach(m; result)
        assert(m !is null);
}
body
{
    import core.stdc.stdlib : malloc;

    size_t len = 0;
    for (auto mr = _Dmodule_ref; mr; mr = mr.next)
        len++;

    auto result = (cast(immutable(ModuleInfo)**)malloc(len * size_t.sizeof))[0 .. len];

    auto tip = _Dmodule_ref;
    foreach (ref r; result)
    {
        r = tip.mod;
        tip = tip.next;
    }

    return cast(immutable)result;
}
