/**
 * This is a special module and linked into each Windows executable/DLL,
 * registering the binary with druntime.
 *
 * When linking against static druntime (or linking the shared druntime DLL
 * itself), rt.sections_elf_shared pulls in this object file automatically.
 *
 * When linking against shared druntime, the compiler links in this
 * object file automatically.
 */
module rt.dso_windows;

version (Windows):

pragma(LDC_no_moduleinfo); // not needed; avoid collision for druntime.dll and other binaries

import rt.sections_elf_shared : CompilerDSOData, _d_dso_registry;

private:

__gshared CompilerDSOData dsoData;
__gshared void* dsoSlot;

// Automatically registers this DSO (Dynamic Shared Object, i.e., DLL or
// executable) with druntime.
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
extern (C) void unregister_dso()
{
    _d_dso_registry(&dsoData);
}

// dummy variable used to link this object file into the binary (ref'd from
// rt.sections_elf_shared)
extern(C) __gshared int __dso_windows_ref = 0;

// special symbols created by the linker:
extern(C) extern __gshared
{
    void* __ImageBase;
    void*[2] _tls_used; // start, end
    int _tls_index;
}

// Returns the TLS range for the executing thread and this DSO.
void[] getTLSRange() nothrow @nogc
{
    void* pbeg;
    void* pend;
    // with VS2017 15.3.1, the linker no longer puts TLS segments into a
    //  separate image section. That way _tls_start and _tls_end no
    //  longer generate offsets into .tls, but DATA.
    // Use the TEB entry to find the start of TLS instead and read the
    //  length from the TLS directory
    version (D_InlineAsm_X86)
    {
        asm @nogc nothrow
        {
            mov EAX, _tls_index;
            mov ECX, FS:[0x2C];     // _tls_array
            mov EAX, [ECX+4*EAX];
            mov pbeg, EAX;
            add EAX, [_tls_used+4]; // end
            sub EAX, [_tls_used+0]; // start
            mov pend, EAX;
        }
    }
    else version (D_InlineAsm_X86_64)
    {
        asm @nogc nothrow
        {
            xor RAX, RAX;
            mov EAX, _tls_index;
            mov RCX, 0x58;
            mov RCX, GS:[RCX];      // _tls_array (immediate value causes fixup)
            mov RAX, [RCX+8*RAX];
            mov pbeg, RAX;
            add RAX, [_tls_used+8]; // end
            sub RAX, [_tls_used+0]; // start
            mov pend, RAX;
        }
    }
    else
        static assert(false, "Architecture not supported.");

    return pbeg[0 .. pend - pbeg];
}
