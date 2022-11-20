/**
 * This is a special module and linked into each DSO (Dynamic Shared Object,
 * i.e., DLL/.so/.dylib or executable), registering the binary with druntime.
 *
 * When linking against static druntime (or linking the shared druntime library
 * itself), rt.sections_elf_shared pulls in this object file automatically.
 *
 * When linking against shared druntime, the compiler links in this
 * object file automatically.
 */
module rt.dso;

pragma(LDC_no_moduleinfo); // not needed; avoid collision across shared druntime and other binaries

import ldc.attributes : hidden;
import rt.sections_elf_shared; // : CompilerDSOData, _d_dso_registry;

static if (is(CompilerDSOData)): // only for targets supporting rt.sections_elf_shared

version (OSX)
    version = Darwin;
else version (iOS)
    version = Darwin;
else version (TVOS)
    version = Darwin;
else version (WatchOS)
    version = Darwin;

private @hidden:

// dummy variable used to link this object file into the binary (ref'd from
// rt.sections_elf_shared)
extern(C) __gshared int __rt_dso_ref = 0;

__gshared CompilerDSOData dsoData;
__gshared void* dsoSlot;

version (Posix)
{
    // Automatically registers this DSO with druntime.
    pragma(crt_constructor)
    void register_dso()
    {
        dsoData._version = 1;
        dsoData._slot = &dsoSlot;
        dsoData._minfo_beg = &__start___minfo;
        dsoData._minfo_end = &__stop___minfo;
        version (Darwin)
            dsoData._getTLSAnchor = &getTLSAnchor;

        _d_dso_registry(&dsoData);
    }

    // Automatically unregisters this DSO from druntime.
    pragma(crt_destructor)
    void unregister_dso()
    {
        _d_dso_registry(&dsoData);
    }

    // special symbols created by the linker:
    extern(C) extern __gshared
    {
        version (Darwin)
        {
            pragma(mangle, "\1section$start$__DATA$.minfo")
            immutable ModuleInfo* __start___minfo;
            pragma(mangle, "\1section$end$__DATA$.minfo")
            immutable ModuleInfo* __stop___minfo;
        }
        else
        {
            immutable ModuleInfo* __start___minfo;
            immutable ModuleInfo* __stop___minfo;
        }
    }

    version (Darwin)
    {
        align(16) byte tlsAnchor = 1;
        extern(C) void* getTLSAnchor() nothrow @nogc
        {
            return &tlsAnchor;
        }
    }
}
else version (Windows)
{
    // Automatically registers this DSO with druntime.
    pragma(crt_constructor)
    void register_dso()
    {
        dsoData._version = 1;
        dsoData._slot = &dsoSlot;
        dsoData._imageBase = cast(typeof(dsoData._imageBase)) &__ImageBase;
        dsoData._getTLSRange = &getTLSRange;

        _d_dso_registry(&dsoData);

        // The default MSVCRT DllMain appears not to call `pragma(crt_destructor)`
        // functions, not even when explicitly unloading via `FreeLibrary()`.
        // Fortunately, registering an atexit handler inside the DSO seems to do
        // the job just as well.
        import core.stdc.stdlib : atexit;
        atexit(&unregister_dso);
    }

    // Unregisters this DSO from druntime.
    extern(C) void unregister_dso()
    {
        _d_dso_registry(&dsoData);
    }

    extern(C) extern __gshared
    {
        // special symbol created by the linker
        void* __ImageBase;

        // special symbols provided by the MSVC runtime
        uint _tls_index;
        void*[2] _tls_used; // start, end
    }

    // Returns the TLS range for the executing thread and this DSO.
    void[] getTLSRange() nothrow @nogc
    {
        void** _tls_array;
        version (Win32)
            asm nothrow @nogc { "mov %%fs:(0x2C), %0" : "=r" (_tls_array); }
        else version (Win64)
            asm nothrow @nogc { "mov %%gs:0(%1),  %0" : "=r" (_tls_array) : "r" (0x58); }
        else
            static assert(0);

        void* pbeg = _tls_array[_tls_index];
        const size = _tls_used[1] - _tls_used[0];
        return pbeg[0 .. size];
    }
}
