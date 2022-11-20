version (DLL)
{
    import core.sys.windows.dll;
    import core.sys.windows.windef : HINSTANCE;

    // like SimpleDllMain mixin, except for DLL_PROCESS_DETACH
    extern(Windows)
    bool DllMain(HINSTANCE hInstance, uint ulReason, void* reserved)
    {
        import core.sys.windows.winnt;
        switch (ulReason)
        {
            default: assert(0);

            case DLL_PROCESS_ATTACH:
                // initialize DLL druntime
                return dll_process_attach(hInstance, true);

            case DLL_PROCESS_DETACH:
                version (NoUnload)
                {
                    // skip terminating the DLL druntime - the proxied GC has already been destroyed
                }
                else
                {
                    dll_process_detach(hInstance, true);
                }
                return true;

            case DLL_THREAD_ATTACH:
                return dll_thread_attach(true, true);

            case DLL_THREAD_DETACH:
                return dll_thread_detach(true, true);
        }
    }
}
else
{
    void main()
    {
        import core.runtime;

        // dynamically load the DLL
        version (NoUnload)
            auto dll = Runtime.loadLibrary("dll_gc_proxy_teardown_nounload.dll");
        else
            auto dll = Runtime.loadLibrary("dll_gc_proxy_teardown.dll");

        assert(dll);

        version (NoUnload) {} else
        Runtime.unloadLibrary(dll);

        // .exe druntime is terminated right after this, before implicitly unloading the DLL for version(NoUnload)
    }
}
