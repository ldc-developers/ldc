/**
 * Written in the D programming language.
 * This module provides ELF-specific support for sections with shared libraries.
 *
 * Copyright: Copyright Martin Nowak 2012-2013.
 * License:   $(HTTP www.boost.org/LICENSE_1_0.txt, Boost License 1.0).
 * Authors:   Martin Nowak
 * Source: $(DRUNTIMESRC rt/_sections_linux.d)
 */

module rt.sections_elf_shared;

version (OSX)
    version = Darwin;
else version (iOS)
    version = Darwin;
else version (TVOS)
    version = Darwin;
else version (WatchOS)
    version = Darwin;

version (CRuntime_Glibc) enum SharedELF = true;
else version (CRuntime_Musl) enum SharedELF = true;
else version (FreeBSD) enum SharedELF = true;
else version (NetBSD) enum SharedELF = true;
else version (DragonFlyBSD) enum SharedELF = true;
else version (CRuntime_UClibc) enum SharedELF = true;
else enum SharedELF = false;

version (Darwin) enum SharedDarwin = true;
else enum SharedDarwin = false;

version (Windows) enum IsWindows = true;
else enum IsWindows = false;

static if (SharedELF || SharedDarwin || IsWindows):

version (MIPS32)  version = MIPS_Any;
version (MIPS64)  version = MIPS_Any;
version (RISCV32) version = RISCV_Any;
version (RISCV64) version = RISCV_Any;

// debug = PRINTF;
import core.internal.elf.dl;
import core.memory;
import core.stdc.config;
import core.stdc.stdio;
import core.stdc.stdlib : calloc, exit, free, malloc, EXIT_FAILURE;
import core.stdc.string : strlen;
version (linux)
{
    import core.sys.linux.dlfcn;
    import core.sys.linux.elf;
    import core.sys.linux.link;
}
else version (FreeBSD)
{
    import core.sys.freebsd.dlfcn;
    import core.sys.freebsd.sys.elf;
    import core.sys.freebsd.sys.link_elf;
}
else version (Darwin)
{
    import core.sys.darwin.dlfcn;
    import core.sys.darwin.mach.dyld;
    import core.sys.darwin.mach.getsect;

    version (D_LP64)
        import rt.sections_darwin_64 : getTLSRange, foreachDataSection, getSection;
    else
        static assert(0, "Not implemented for this architecture");

    extern(C) intptr_t _dyld_get_image_slide(const mach_header*) nothrow @nogc;
}
else version (NetBSD)
{
    import core.sys.netbsd.dlfcn;
    import core.sys.netbsd.sys.elf;
    import core.sys.netbsd.sys.link_elf;
}
else version (DragonFlyBSD)
{
    import core.sys.dragonflybsd.dlfcn;
    import core.sys.dragonflybsd.sys.elf;
    import core.sys.dragonflybsd.sys.link_elf;
}
else version (Windows)
{
    import core.sys.windows.winbase;
    import core.sys.windows.windef;
    import rt.sections_win64 : IMAGE_DOS_HEADER, findImageSection, getModuleInfos;
}
else
{
    static assert(0, "unimplemented");
}
import rt.deh;
import rt.dmain2;
import rt.minfo;
import core.internal.container.array;
import core.internal.container.hashtab;
import rt.util.utility : safeAssert;
version (Shared) import core.sync.mutex;

alias DSO SectionGroup;
struct DSO
{
    static int opApply(scope int delegate(ref DSO) dg)
    {
        foreach (dso; _loadedDSOs)
        {
            if (auto res = dg(*dso))
                return res;
        }
        return 0;
    }

    static int opApplyReverse(scope int delegate(ref DSO) dg)
    {
        foreach_reverse (dso; _loadedDSOs)
        {
            if (auto res = dg(*dso))
                return res;
        }
        return 0;
    }

    @property immutable(ModuleInfo*)[] modules() const nothrow @nogc
    {
        return _moduleGroup.modules;
    }

    @property ref inout(ModuleGroup) moduleGroup() inout return nothrow @nogc
    {
        return _moduleGroup;
    }

    version (DigitalMars)
    @property immutable(FuncTable)[] ehTables() const nothrow @nogc
    {
        return null;
    }

    @property inout(void[])[] gcRanges() inout nothrow @nogc
    {
        return _gcRanges[];
    }

private:

    invariant()
    {
        safeAssert(_moduleGroup.modules.length > 0, "No modules for DSO.");
        version (CRuntime_UClibc) {} else
        static if (SharedELF)
        {
            safeAssert(_tlsMod || !_tlsSize, "Inconsistent TLS fields for DSO.");
        }
    }

    void** _slot;
    ModuleGroup _moduleGroup;
    Array!(void[]) _gcRanges;
    static if (SharedELF)
    {
        size_t _tlsMod;
        size_t _tlsSize;
        version (LDC)
            size_t _tlsAlignment;
    }
    else static if (SharedDarwin)
    {
        GetTLSAnchor _getTLSAnchor;
    }
    else version (Windows)
    {
        GetTLSRange _getTLSRange;
    }

    version (Shared)
    {
        Array!(void[]) _codeSegments; // array of code segments
        Array!(DSO*) _deps; // D libraries needed by this DSO
        void* _handle; // corresponding handle
    }

    // get the TLS range for the executing thread
    void[] tlsRange() const nothrow @nogc
    {
        static if (SharedELF)
        {
            version (LDC)
                return getTLSRange(_tlsMod, _tlsSize, _tlsAlignment);
            else
                return getTLSRange(_tlsMod, _tlsSize);
        }
        else static if (SharedDarwin)
        {
            auto range = getTLSRange(_getTLSAnchor());
            safeAssert(range !is null, "Could not determine TLS range.");
            return range;
        }
        else version (Windows)
        {
            return _getTLSRange();
        }
        else static assert(0, "unimplemented");
    }
}

/****
 * Boolean flag set to true while the runtime is initialized.
 */
__gshared bool _isRuntimeInitialized;


version (FreeBSD) private __gshared void* dummy_ref;
version (DragonFlyBSD) private __gshared void* dummy_ref;
version (NetBSD) private __gshared void* dummy_ref;

/****
 * Gets called on program startup just before GC is initialized.
 */
void initSections() nothrow @nogc
{
    _isRuntimeInitialized = true;
    // reference symbol to support weak linkage
    version (FreeBSD) dummy_ref = &_d_dso_registry;
    version (DragonFlyBSD) dummy_ref = &_d_dso_registry;
    version (NetBSD) dummy_ref = &_d_dso_registry;
}


/***
 * Gets called on program shutdown just after GC is terminated.
 */
void finiSections() nothrow @nogc
{
    _isRuntimeInitialized = false;
}

alias ScanDG = void delegate(void* pbeg, void* pend) nothrow;

version (Shared)
{
    /***
     * Called once per thread; returns array of thread local storage ranges
     */
    Array!(ThreadDSO)* initTLSRanges() @nogc nothrow
    {
        return &_loadedDSOs();
    }

    void finiTLSRanges(Array!(ThreadDSO)* tdsos) @nogc nothrow
    {
        // Nothing to do here. tdsos used to point to the _loadedDSOs instance
        // in the dying thread's TLS segment and as such is not valid anymore.
        // The memory for the array contents was already reclaimed in
        // cleanupLoadedLibraries().
    }

    void scanTLSRanges(Array!(ThreadDSO)* tdsos, scope ScanDG dg) nothrow
    {
        foreach (ref tdso; *tdsos)
            dg(tdso._tlsRange.ptr, tdso._tlsRange.ptr + tdso._tlsRange.length);
    }

    size_t sizeOfTLS() nothrow @nogc
    {
        auto tdsos = initTLSRanges();
        size_t sum;
        foreach (ref tdso; *tdsos)
            sum += tdso._tlsRange.length;
        return sum;
    }

    // interface for core.thread to inherit loaded libraries
    void* pinLoadedLibraries() nothrow @nogc
    {
        auto res = cast(Array!(ThreadDSO)*)calloc(1, Array!(ThreadDSO).sizeof);
        res.length = _loadedDSOs.length;
        foreach (i, ref tdso; _loadedDSOs)
        {
            (*res)[i] = tdso;
            if (tdso._addCnt)
            {
                // Increment the refcount of explicitly loaded libraries to pin them.
                version (Windows)
                {
                    HMODULE hModule;
                    const success = GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
                                                       cast(const(wchar)*) tdso._pdso._slot, &hModule) == TRUE;
                }
                else
                {
                    const success = .dlopen(nameForDSO(tdso._pdso), RTLD_LAZY) !is null;
                }
                safeAssert(success, "Failed to increment library refcount.");
                (*res)[i]._addCnt = 1; // new array takes over the additional ref count
            }
        }
        return res;
    }

    void unpinLoadedLibraries(void* p) nothrow @nogc
    {
        auto pary = cast(Array!(ThreadDSO)*)p;
        // In case something failed we need to undo the pinning.
        foreach (ref tdso; *pary)
        {
            if (tdso._addCnt)
            {
                auto handle = tdso._pdso._handle;
                safeAssert(handle !is null, "Invalid library handle.");
                version (Windows)
                    FreeLibrary(handle);
                else
                    .dlclose(handle);
            }
        }
        pary.reset();
        .free(pary);
    }

    // Called before TLS ctors are ran, copy over the loaded libraries
    // of the parent thread.
    void inheritLoadedLibraries(void* p) nothrow @nogc
    {
        safeAssert(_loadedDSOs.empty, "DSOs have already been registered for this thread.");
        _loadedDSOs.swap(*cast(Array!(ThreadDSO)*)p);
        .free(p);
        foreach (ref dso; _loadedDSOs)
        {
            // the copied _tlsRange corresponds to parent thread
            dso.updateTLSRange();
        }
    }

    // Called after all TLS dtors ran, decrements all remaining dlopen refs.
    void cleanupLoadedLibraries() nothrow @nogc
    {
        foreach (ref tdso; _loadedDSOs)
        {
            if (tdso._addCnt == 0) continue;

            auto handle = tdso._pdso._handle;
            safeAssert(handle !is null, "Invalid DSO handle.");
            for (; tdso._addCnt > 0; --tdso._addCnt)
            {
                version (Windows)
                    FreeLibrary(handle);
                else
                    .dlclose(handle);
            }
        }

        // Free the memory for the array contents.
        _loadedDSOs.reset();
    }
}
else
{
    /***
     * Called once per thread; returns array of thread local storage ranges
     */
    Array!(void[])* initTLSRanges() nothrow @nogc
    {
        auto rngs = &_tlsRanges();
        if (rngs.empty)
        {
            foreach (ref pdso; _loadedDSOs)
                rngs.insertBack(pdso.tlsRange());
        }
        return rngs;
    }

    void finiTLSRanges(Array!(void[])* rngs) nothrow @nogc
    {
        rngs.reset();
        .free(rngs);
    }

    void scanTLSRanges(Array!(void[])* rngs, scope ScanDG dg) nothrow
    {
        foreach (rng; *rngs)
            dg(rng.ptr, rng.ptr + rng.length);
    }

    size_t sizeOfTLS() nothrow @nogc
    {
        auto rngs = initTLSRanges();
        size_t sum;
        foreach (rng; *rngs)
            sum += rng.length;
        return sum;
    }
}

private:

// start of linked list for ModuleInfo references
version (FreeBSD) deprecated extern (C) __gshared void* _Dmodule_ref;
version (DragonFlyBSD) deprecated extern (C) __gshared void* _Dmodule_ref;
version (NetBSD) deprecated extern (C) __gshared void* _Dmodule_ref;

version (Shared)
{
    /*
     * Array of thread local DSO metadata for all libraries loaded and
     * initialized in this thread.
     *
     * Note:
     *     A newly spawned thread will inherit these libraries.
     * Note:
     *     We use an array here to preserve the order of
     *     initialization.  If that became a performance issue, we
     *     could use a hash table and enumerate the DSOs during
     *     loading so that the hash table values could be sorted when
     *     necessary.
     */
    struct ThreadDSO
    {
        DSO* _pdso;
        static if (_pdso.sizeof == 8) uint _refCnt, _addCnt;
        else static if (_pdso.sizeof == 4) ushort _refCnt, _addCnt;
        else static assert(0, "unimplemented");
        void[] _tlsRange;
        alias _pdso this;
        // update the _tlsRange for the executing thread
        void updateTLSRange() nothrow @nogc
        {
            _tlsRange = _pdso.tlsRange();
        }
    }
    @property ref Array!(ThreadDSO) _loadedDSOs() @nogc nothrow { static Array!(ThreadDSO) x; return x; }
    //Array!(ThreadDSO) _loadedDSOs;

    /*
     * Set to true during rt_loadLibrary/rt_unloadLibrary calls.
     */
    bool _rtLoading;

    /*
     * Hash table to map the native handle (as returned by dlopen)
     * to the corresponding DSO*, protected by a mutex.
     */
    __gshared Mutex _handleToDSOMutex;
    @property ref HashTab!(void*, DSO*) _handleToDSO() @nogc nothrow { __gshared HashTab!(void*, DSO*) x; return x; }
    //__gshared HashTab!(void*, DSO*) _handleToDSO;
}
else
{
    /*
     * Static DSOs loaded by the runtime linker. This includes the
     * executable. These can't be unloaded.
     */
    @property ref Array!(DSO*) _loadedDSOs() @nogc nothrow { __gshared Array!(DSO*) x; return x; }
    //__gshared Array!(DSO*) _loadedDSOs;

    /*
     * Thread local array that contains TLS memory ranges for each
     * library initialized in this thread.
     */
    @property ref Array!(void[]) _tlsRanges() @nogc nothrow {
        static Array!(void[])* x = null;
        if (x is null)
            x = cast(Array!(void[])*).calloc(1, Array!(void[]).sizeof);
        safeAssert(x !is null, "Failed to allocate TLS ranges");
        return *x;
    }
    //Array!(void[]) _tlsRanges;

    enum _rtLoading = false;
}

///////////////////////////////////////////////////////////////////////////////
// Compiler to runtime interface.
///////////////////////////////////////////////////////////////////////////////

// link in rt.dso:
import ldc.attributes : assumeUsed;
extern(C) extern __gshared int __rt_dso_ref;
@assumeUsed __gshared dummy = &__rt_dso_ref;

version (Darwin)
    private alias ImageHeader = mach_header*;
else version (Windows)
    private alias ImageHeader = IMAGE_DOS_HEADER*;
else
    private alias ImageHeader = SharedObject;

version (Darwin)
{
    extern(C) alias GetTLSAnchor = void* function() nothrow @nogc;
}
else version (Windows)
{
    alias GetTLSRange = void[] function() nothrow @nogc;

    extern(C) bool gc_isProxied() nothrow @nogc; // in core.internal.gc.proxy
}

/*
 * This data structure is generated by the compiler, and then passed to
 * _d_dso_registry().
 */
package struct CompilerDSOData
{
    size_t _version;                                       // currently 1
    void** _slot;                                          // can be used to store runtime data
    version (Windows)
    {
        IMAGE_DOS_HEADER* _imageBase; // &__ImageBase
        GetTLSRange _getTLSRange;
    }
    else
    {
        immutable(object.ModuleInfo*)* _minfo_beg, _minfo_end; // array of modules in this object file
        static if (SharedDarwin) GetTLSAnchor _getTLSAnchor;
    }
}

T[] toRange(T)(T* beg, T* end) { return beg[0 .. end - beg]; }

/* For each shared library and executable, the compiler generates code that
 * sets up CompilerDSOData and calls _d_dso_registry().
 * A pointer to that code is inserted into both the .init_array and .fini_array
 * segment so it gets called by the loader on startup and shutdown.
 */
package extern(C) void _d_dso_registry(void* arg)
{
    auto data = cast(CompilerDSOData*) arg;

    // only one supported currently
    safeAssert(data._version >= 1, "Incompatible compiler-generated DSO data version.");

    // no backlink => register
    if (*data._slot is null)
    {
        immutable firstDSO = _loadedDSOs.empty;
        if (firstDSO) initLocks();

        DSO* pdso = cast(DSO*).calloc(1, DSO.sizeof);
        static assert(__traits(isZeroInit, DSO));
        pdso._slot = data._slot;
        *data._slot = pdso; // store backlink in library record

        version (Windows)
        {
            ImageHeader header = data._imageBase;
            pdso._getTLSRange = data._getTLSRange;
            pdso._moduleGroup = ModuleGroup(getModuleInfos(header));
        }
        else
        {
            version (LDC)
            {
                auto minfoBeg = data._minfo_beg;
                while (minfoBeg < data._minfo_end && !*minfoBeg) ++minfoBeg;
                auto minfoEnd = minfoBeg;
                while (minfoEnd < data._minfo_end && *minfoEnd) ++minfoEnd;
                pdso._moduleGroup = ModuleGroup(toRange(minfoBeg, minfoEnd));
            }
            else
                pdso._moduleGroup = ModuleGroup(toRange(data._minfo_beg, data._minfo_end));

            static if (SharedDarwin) pdso._getTLSAnchor = data._getTLSAnchor;

            ImageHeader header = void;
            const headerFound = findImageHeaderForAddr(data._slot, header);
            safeAssert(headerFound, "Failed to find image header.");
        }

        scanSegments(header, pdso);

        version (Shared)
        {
            auto handle = handleForAddr(data._slot);

            getDependencies(header, pdso._deps);
            pdso._handle = handle;
            setDSOForHandle(pdso, pdso._handle);

            if (!_rtLoading)
            {
                /* This DSO was not loaded by rt_loadLibrary which
                 * happens for all dependencies of an executable or
                 * the first dlopen call from a C program.
                 * In this case we add the DSO to the _loadedDSOs of this
                 * thread with a refCnt of 1 and call the TlsCtors.
                 */
                immutable ushort refCnt = 1, addCnt = 0;
                _loadedDSOs.insertBack(ThreadDSO(pdso, refCnt, addCnt, pdso.tlsRange()));
            }
        }
        else
        {
            version (LDC)
            {
                // We don't want to depend on __tls_get_addr in non-Shared builds
                // so we can actually link statically, so there must be only one
                // D shared object.
                import core.internal.abort;
                _loadedDSOs.empty ||
                    abort("Only one D shared object allowed for static runtime. " ~
                          "Link with shared runtime via LDC switch '-link-defaultlib-shared'.");
            }
            foreach (p; _loadedDSOs)
                safeAssert(p !is pdso, "DSO already registered.");
            _loadedDSOs.insertBack(pdso);
            _tlsRanges.insertBack(pdso.tlsRange());
        }

        // don't initialize modules before rt_init was called (see Bugzilla 11378)
        if (_isRuntimeInitialized)
        {
            registerGCRanges(pdso);
            // rt_loadLibrary will run tls ctors, so do this only for dlopen
            immutable runTlsCtors = !_rtLoading;
            runModuleConstructors(pdso, runTlsCtors);
        }
    }
    // has backlink => unregister
    else
    {
        DSO* pdso = cast(DSO*)*data._slot;
        *data._slot = null;

        // don't finalizes modules after rt_term was called (see Bugzilla 11378)
        bool doFinalize = _isRuntimeInitialized;
        version (Shared) {} else version (Windows)
        {
            /* If a DLL with its own static druntime has been loaded via
             * rt_loadLibrary() [from an .{exe,dll} host with static druntime],
             * its GC is 'replaced' by a proxy to the host's GC.
             * If it's still proxied, the DLL isn't being unloaded explicitly
             * via rt_unloadLibrary(), and is so probably being unloaded
             * implicitly when terminating the process. Chances are high that
             * the host's druntime has already been terminated, in which case
             * trying to access its GC causes a segfault.
             */
            if (doFinalize)
                doFinalize = !gc_isProxied();
        }
        if (doFinalize)
        {
            // rt_unloadLibrary already ran tls dtors, so do this only for dlclose
            immutable runTlsDtors = !_rtLoading;
            runModuleDestructors(pdso, runTlsDtors);
            unregisterGCRanges(pdso);
            // run finalizers after module dtors (same order as in rt_term)
            version (Shared) runFinalizers(pdso);
        }

        version (Shared)
        {
            if (!_rtLoading)
            {
                /* This DSO was not unloaded by rt_unloadLibrary so we
                 * have to remove it from _loadedDSOs here.
                 */
                foreach (i, ref tdso; _loadedDSOs)
                {
                    if (tdso._pdso == pdso)
                    {
                        _loadedDSOs.remove(i);
                        break;
                    }
                }
            }

            unsetDSOForHandle(pdso, pdso._handle);
        }
        else
        {
            // static DSOs are unloaded in reverse order
            safeAssert(pdso == _loadedDSOs.back, "DSO being unregistered isn't current last one.");
            _loadedDSOs.popBack();
        }

        freeDSO(pdso);

        // last DSO being unloaded => shutdown registry
        if (_loadedDSOs.empty)
        {
            version (Shared)
            {
                safeAssert(_handleToDSO.empty, "_handleToDSO not in sync with _loadedDSOs.");
                _handleToDSO.reset();
            }
            finiLocks();
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
// dynamic loading
///////////////////////////////////////////////////////////////////////////////

// Shared D libraries are only supported when linking against a shared druntime library.

version (Shared)
{
    ThreadDSO* findThreadDSO(DSO* pdso) nothrow @nogc
    {
        foreach (ref tdata; _loadedDSOs)
            if (tdata._pdso == pdso) return &tdata;
        return null;
    }

    void incThreadRef(DSO* pdso, bool incAdd)
    {
        if (auto tdata = findThreadDSO(pdso)) // already initialized
        {
            if (incAdd && ++tdata._addCnt > 1) return;
            ++tdata._refCnt;
        }
        else
        {
            foreach (dep; pdso._deps)
                incThreadRef(dep, false);
            immutable ushort refCnt = 1, addCnt = incAdd ? 1 : 0;
            _loadedDSOs.insertBack(ThreadDSO(pdso, refCnt, addCnt, pdso.tlsRange()));
            pdso._moduleGroup.runTlsCtors();
        }
    }

    void decThreadRef(DSO* pdso, bool decAdd)
    {
        auto tdata = findThreadDSO(pdso);
        safeAssert(tdata !is null, "Failed to find thread DSO.");
        safeAssert(!decAdd || tdata._addCnt > 0, "Mismatching rt_unloadLibrary call.");

        if (decAdd && --tdata._addCnt > 0) return;
        if (--tdata._refCnt > 0) return;

        pdso._moduleGroup.runTlsDtors();
        foreach (i, ref td; _loadedDSOs)
            if (td._pdso == pdso) _loadedDSOs.remove(i);
        foreach (dep; pdso._deps)
            decThreadRef(dep, false);
    }

    void* loadLibraryImpl(T)(const T* name)
    {
        immutable save = _rtLoading;
        _rtLoading = true;
        scope (exit) _rtLoading = save;

        version (Windows)
        {
            static if (T.sizeof == 1)
                auto handle = LoadLibraryA(name);
            else
                auto handle = LoadLibraryW(name);
        }
        else
            auto handle = .dlopen(name, RTLD_LAZY);

        if (handle is null) return null;

        // if it's a D library
        if (auto pdso = dsoForHandle(handle))
            incThreadRef(pdso, true);
        return handle;
    }

    extern(C) void* rt_loadLibrary(const char* name)
    {
        return loadLibraryImpl(name);
    }

    version (Windows)
    extern(C) void* rt_loadLibraryW(const wchar* name)
    {
        return loadLibraryImpl(name);
    }

    extern(C) int rt_unloadLibrary(void* handle)
    {
        if (handle is null) return false;

        immutable save = _rtLoading;
        _rtLoading = true;
        scope (exit) _rtLoading = save;

        // if it's a D library
        if (auto pdso = dsoForHandle(handle))
            decThreadRef(pdso, true);

        version (Windows)
            return FreeLibrary(handle);
        else
            return .dlclose(handle) == 0;
    }
}

///////////////////////////////////////////////////////////////////////////////
// helper functions
///////////////////////////////////////////////////////////////////////////////

void initLocks() nothrow @nogc
{
    version (Shared)
    {
        import core.lifetime : emplace;

        _handleToDSOMutex = cast(Mutex) malloc(__traits(classInstanceSize, Mutex));
        safeAssert(_handleToDSOMutex !is null, "Could not allocate mutex.");
        _handleToDSOMutex.emplace();
    }
}

void finiLocks() nothrow @nogc
{
    version (Shared)
    {
        //destroy(_handleToDSOMutex);   not @nogc
        _handleToDSOMutex.__dtor();
        free(cast(void*) _handleToDSOMutex);
        _handleToDSOMutex = null;
    }
}

void runModuleConstructors(DSO* pdso, bool runTlsCtors)
{
    pdso._moduleGroup.sortCtors();
    pdso._moduleGroup.runCtors();
    if (runTlsCtors) pdso._moduleGroup.runTlsCtors();
}

void runModuleDestructors(DSO* pdso, bool runTlsDtors)
{
    if (runTlsDtors) pdso._moduleGroup.runTlsDtors();
    pdso._moduleGroup.runDtors();
}

void registerGCRanges(DSO* pdso) nothrow @nogc
{
    foreach (rng; pdso._gcRanges)
        GC.addRange(rng.ptr, rng.length);
}

void unregisterGCRanges(DSO* pdso) nothrow @nogc
{
    foreach (rng; pdso._gcRanges)
        GC.removeRange(rng.ptr);
}

version (Shared) void runFinalizers(DSO* pdso)
{
    foreach (seg; pdso._codeSegments)
        GC.runFinalizers(seg);
}

void freeDSO(DSO* pdso) nothrow @nogc
{
    pdso._gcRanges.reset();
    version (Shared)
    {
        pdso._codeSegments.reset();
        pdso._deps.reset();
        pdso._handle = null;
    }
    .free(pdso);
}

version (Shared)
{
@nogc nothrow:
    version (Windows) {} else
    const(char)* nameForDSO(in DSO* pdso)
    {
        Dl_info info = void;
        const success = dladdr(pdso._slot, &info) != 0;
        safeAssert(success, "Failed to get DSO info.");
        return info.dli_fname;
    }

    DSO* dsoForHandle(void* handle)
    {
        DSO* pdso;
        _handleToDSOMutex.lock_nothrow();
        if (auto ppdso = handle in _handleToDSO)
            pdso = *ppdso;
        _handleToDSOMutex.unlock_nothrow();
        return pdso;
    }

    void setDSOForHandle(DSO* pdso, void* handle)
    {
        _handleToDSOMutex.lock_nothrow();
        safeAssert(handle !in _handleToDSO, "DSO already registered.");
        _handleToDSO[handle] = pdso;
        _handleToDSOMutex.unlock_nothrow();
    }

    void unsetDSOForHandle(DSO* pdso, void* handle)
    {
        _handleToDSOMutex.lock_nothrow();
        safeAssert(_handleToDSO[handle] == pdso, "Handle doesn't match registered DSO.");
        _handleToDSO.remove(handle);
        _handleToDSOMutex.unlock_nothrow();
    }

    static if (SharedELF) void getDependencies(const scope ref SharedObject object, ref Array!(DSO*) deps)
    {
        // get the entries of the .dynamic section
        ElfW!"Dyn"[] dyns;
        foreach (ref phdr; object)
        {
            if (phdr.p_type == PT_DYNAMIC)
            {
                auto p = cast(ElfW!"Dyn"*)(object.baseAddress + (phdr.p_vaddr & ~(size_t.sizeof - 1)));
                dyns = p[0 .. phdr.p_memsz / ElfW!"Dyn".sizeof];
                break;
            }
        }
        // find the string table which contains the sonames
        const(char)* strtab;
        foreach (dyn; dyns)
        {
            if (dyn.d_tag == DT_STRTAB)
            {
                version (CRuntime_Musl)
                    enum relocate = true;
                else version (linux)
                {
                    // This might change in future glibc releases (after 2.29) as dynamic sections
                    // are not required to be read-only on RISC-V. This was copy & pasted from MIPS
                    // while upstreaming RISC-V support. Otherwise MIPS is the only arch which sets
                    // in glibc: #define DL_RO_DYN_SECTION 1
                    version (RISCV_Any)     enum relocate = true;
                    else version (MIPS_Any) enum relocate = true;
                    else                    enum relocate = false;
                }
                else version (FreeBSD)
                    enum relocate = true;
                else version (NetBSD)
                    enum relocate = true;
                else version (DragonFlyBSD)
                    enum relocate = true;
                else
                    static assert(0, "unimplemented");

                const base = relocate ? cast(const char*) object.baseAddress : null;
                strtab = base + dyn.d_un.d_ptr;

                break;
            }
        }
        foreach (dyn; dyns)
        {
            immutable tag = dyn.d_tag;
            if (!(tag == DT_NEEDED || tag == DT_AUXILIARY || tag == DT_FILTER))
                continue;

            // soname of the dependency
            auto name = strtab + dyn.d_un.d_val;
            // get handle without loading the library
            auto handle = handleForName(name);
            // the runtime linker has already loaded all dependencies
            safeAssert(handle !is null, "Failed to get library handle.");
            // if it's a D library
            if (auto pdso = dsoForHandle(handle))
                deps.insertBack(pdso); // append it to the dependencies
        }
    }
    else static if (SharedDarwin) void getDependencies(in ImageHeader info, ref Array!(DSO*) deps)
    {
        // FIXME: Not implemented yet.
    }
    else version (Windows) void getDependencies(in ImageHeader info, ref Array!(DSO*) deps)
    {
        // FIXME: Not implemented yet.
    }

    version (Windows) {} else
    void* handleForName(const char* name)
    {
        auto handle = .dlopen(name, RTLD_NOLOAD | RTLD_LAZY);
        if (handle !is null) .dlclose(handle); // drop reference count
        return handle;
    }
}

///////////////////////////////////////////////////////////////////////////////
// Elf program header iteration
///////////////////////////////////////////////////////////////////////////////

/************
 * Scan segments in the image header and store
 * the TLS and writeable data segments in *pdso.
 */
version (Windows)
{
    void scanSegments(IMAGE_DOS_HEADER* image, DSO* pdso) nothrow @nogc
    {
        version (Shared)
            pdso._codeSegments.insertBack(image.findImageSection(".text"));

        // the ".data" image section includes both object file sections ".data" and ".bss"
        pdso._gcRanges.insertBack(image.findImageSection(".data"));
    }
}
else
{
    static if (SharedELF) void scanSegments(const scope ref SharedObject object, DSO* pdso) nothrow @nogc
    {
        foreach (ref phdr; object)
        {
            switch (phdr.p_type)
            {
            case PT_LOAD:
                if (phdr.p_flags & PF_W) // writeable data segment
                {
                    auto beg = object.baseAddress + (phdr.p_vaddr & ~(size_t.sizeof - 1));
                    pdso._gcRanges.insertBack(beg[0 .. phdr.p_memsz]);
                }
                version (Shared) if (phdr.p_flags & PF_X) // code segment
                {
                    auto beg = object.baseAddress + (phdr.p_vaddr & ~(size_t.sizeof - 1));
                    pdso._codeSegments.insertBack(beg[0 .. phdr.p_memsz]);
                }
                break;

            case PT_TLS: // TLS segment
                safeAssert(!pdso._tlsSize, "Multiple TLS segments in image header.");
                version (CRuntime_UClibc)
                {
                    // uClibc doesn't provide a 'dlpi_tls_modid' definition
                }
                else
                    pdso._tlsMod = object.info.dlpi_tls_modid;
                pdso._tlsSize = phdr.p_memsz;
                version (LDC)
                    pdso._tlsAlignment = phdr.p_align;
                break;

            default:
                break;
            }
        }
    }
    else static if (SharedDarwin) void scanSegments(mach_header* info, DSO* pdso)
    {
        intptr_t slide = 0;
        auto command = cast(const(load_command)*)(cast(ubyte*)info + mach_header_64.sizeof);

        foreach(_; 0 .. info.ncmds) {

            // Scan 64 bit segment
            // See: getSlide() in https://opensource.apple.com/source/dyld/dyld-851.27/dyld3/MachOLoaded.cpp.auto.html
            if (command.cmd == LC_SEGMENT_64) 
            {
                auto segment = cast(const(segment_command_64)*) command;
                
                if (segment.filesize != 0) 
                {
                    slide = cast(uintptr_t)info - segment.vmaddr;
                    break;
                }
            }

            command = cast(const(load_command)*)(cast(ubyte*)command + command.cmdsize);
        }

        foreachDataSection(info, slide, (sectionData) { pdso._gcRanges.insertBack(sectionData); });

        version (Shared)
        {
            void[] text = getSection(info, slide, "__TEXT", "__text");
            if (!text)
                assert(0, "Failed to get text section.");
            pdso._codeSegments.insertBack(text);
        }
    }

    bool findImageHeaderForAddr(in void* addr, out ImageHeader result) nothrow @nogc
    {
        version (Darwin)
        {
            Dl_info info;
            const res = dladdr(addr, &info);
            result = cast(ImageHeader) info.dli_fbase;

            return res != 0;
        }
        else
        {
            return SharedObject.findForAddress(addr, result);
        }
    }
}

/**************************
 * Input:
 *      addr  an internal address of a DSO
 * Returns:
 *      the dlopen handle for that DSO or null if addr is not within a loaded DSO
 */
version (Shared) void* handleForAddr(void* addr) nothrow @nogc
{
    version (Windows)
    {
        void* hModule;
        if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                cast(const(wchar)*) addr, &hModule))
            return null;
        return hModule;
    }
    else
    {
        Dl_info info = void;
        if (dladdr(addr, &info) != 0)
            return handleForName(info.dli_fname);
        return null;
    }
}

///////////////////////////////////////////////////////////////////////////////
// TLS module helper
///////////////////////////////////////////////////////////////////////////////

version (Windows) {} else:

/*
 * Returns: the TLS memory range for a given module and the calling
 * thread or null if that module has no TLS.
 *
 * Note: This will cause the TLS memory to be eagerly allocated.
 */
struct tls_index
{
    version (CRuntime_Glibc)
    {
        // For x86_64, fields are of type uint64_t, this is important for x32
        // where tls_index would otherwise have the wrong size.
        // See https://sourceware.org/git/?p=glibc.git;a=blob;f=sysdeps/x86_64/dl-tls.h
        version (X86_64)
        {
            ulong ti_module;
            ulong ti_offset;
        }
        else
        {
            c_ulong ti_module;
            c_ulong ti_offset;
        }
    }
    else
    {
        size_t ti_module;
        size_t ti_offset;
    }
}

static if (SharedDarwin)
{
    version (LDC)
    {
        private align(16) ubyte dummyTlsSymbol = 42;
        // By initializing dummyTlsSymbol with something non-zero and aligning
        // to 16-bytes, section __thread_data will be aligned as a workaround
        // for https://github.com/ldc-developers/ldc/issues/1252
    }
}
else
{

version (LDC)
{
    version (PPC)
    {
        extern(C) void* __tls_get_addr_opt(tls_index* ti) nothrow @nogc;
        alias __tls_get_addr = __tls_get_addr_opt;
    }
    else version (PPC64)
    {
        extern(C) void* __tls_get_addr_opt(tls_index* ti) nothrow @nogc;
        alias __tls_get_addr = __tls_get_addr_opt;
    }
    else
        extern(C) void* __tls_get_addr(tls_index* ti) nothrow @nogc;
}
else
extern(C) void* __tls_get_addr(tls_index* ti) nothrow @nogc;

/* The dynamic thread vector (DTV) pointers may point 0x8000 past the start of
 * each TLS block. This is at least true for PowerPC and Mips platforms.
 * See: https://sourceware.org/git/?p=glibc.git;a=blob;f=sysdeps/powerpc/dl-tls.h;h=f7cf6f96ebfb505abfd2f02be0ad0e833107c0cd;hb=HEAD#l34
 *      https://sourceware.org/git/?p=glibc.git;a=blob;f=sysdeps/mips/dl-tls.h;h=93a6dc050cb144b9f68b96fb3199c60f5b1fcd18;hb=HEAD#l32
 *      https://sourceware.org/git/?p=glibc.git;a=blob;f=sysdeps/riscv/dl-tls.h;h=ab2d860314de94c18812bc894ff6b3f55368f20f;hb=HEAD#l32
 */
version (X86)
    enum TLS_DTV_OFFSET = 0x0;
else version (X86_64)
    enum TLS_DTV_OFFSET = 0x0;
else version (ARM)
    enum TLS_DTV_OFFSET = 0x0;
else version (AArch64)
    enum TLS_DTV_OFFSET = 0x0;
else version (RISCV_Any)
    enum TLS_DTV_OFFSET = 0x800;
else version (HPPA)
    enum TLS_DTV_OFFSET = 0x0;
else version (SPARC)
    enum TLS_DTV_OFFSET = 0x0;
else version (SPARC64)
    enum TLS_DTV_OFFSET = 0x0;
else version (PPC)
    enum TLS_DTV_OFFSET = 0x8000;
else version (PPC64)
    enum TLS_DTV_OFFSET = 0x8000;
else version (MIPS_Any)
    enum TLS_DTV_OFFSET = 0x8000;
else version (LoongArch64)
    enum TLS_DTV_OFFSET = 0x0;
else
    static assert( false, "Platform not supported." );

version (LDC)
{
    // We do not want to depend on __tls_get_addr for non-Shared builds to support
    // linking against a static C runtime.
    version (X86)    version = X86_Any;
    version (X86_64) version = X86_Any;
    version (Shared) {} else version (linux) version (X86_Any)
        version = Static_Linux_X86_Any;
}

// LDC: added `alignment` param
void[] getTLSRange(size_t mod, size_t sz, size_t alignment) nothrow @nogc
{
    version (Static_Linux_X86_Any)
    {
        void* tcb;
        version (X86)
            asm nothrow @nogc { "mov %%gs:0, %0" : "=r" (tcb); }
        else version (X86_64)
            asm nothrow @nogc { "mov %%fs:0, %0" : "=r" (tcb); }

        void* start = tcb - sz;

        // For non-ld.gold linkers, it seems to be necessary to align the
        // pointer down according to the TLS alignment.
        start = cast(void*) ((cast(size_t) start) & ~(alignment - 1));

        version (unittest) // verify against __tls_get_addr for static druntime unittest runners
        {
            void* reference;
            if (mod != 0)
            {
                auto ti = tls_index(mod, 0);
                reference = __tls_get_addr(&ti);
            }

            if (reference != start)
            {
                import core.stdc.stdlib : abort;
                fprintf(stderr, "ERROR: getTLSRange mismatch - %p\n", start);
                fprintf(stderr, "                          vs. %p\n", reference);
                abort();
            }
        }

        return start[0 .. sz];
    }
    else
    {
        if (mod == 0)
            return null;

        // base offset
        auto ti = tls_index(mod, 0);
        return (__tls_get_addr(&ti)-TLS_DTV_OFFSET)[0 .. sz];
    }
}

} // !SharedDarwin
