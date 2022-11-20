/**
 * Contains forward references to the AddressSanitizer interface.
 *
 * Copyright: Authors 2019-2019
 * License:   $(LINK2 http://www.boost.org/LICENSE_1_0.txt, Boost License 1.0)
 * Authors:   LDC Developers
 */

module ldc.sanitizers_optionally_linked;

version (SupportSanitizers)
{
    version (OSX)
        version = Darwin;
    else version (iOS)
        version = Darwin;
    else version (TVOS)
        version = Darwin;
    else version (WatchOS)
        version = Darwin;

    version (Darwin) {}
    else version (Posix)
    {
        version = ELF;
    }

    // Forward declarations of sanitizer functions (only ELF supports optional static linking).
    extern(C) @system @nogc nothrow
    {
        version (ELF)
            enum pragmastring = "pragma(LDC_extern_weak):\n";
        else
            enum pragmastring = "";

        mixin(pragmastring ~ q{
            void __sanitizer_start_switch_fiber(void** fake_stack_save, const(void)* bottom, size_t size);
            void __sanitizer_finish_switch_fiber(void* fake_stack_save, const(void)** bottom_old, size_t* size_old);
            void* __asan_get_current_fake_stack();
            void* __asan_addr_is_in_fake_stack(void *fake_stack, void *addr, void **beg, void **end);
        });
    }


    nothrow @nogc
    void informSanitizerOfStartSwitchFiber(void** fake_stack_save, const(void)* bottom, size_t size)
    {
        auto fptr = getOptionalSanitizerFunc!"__sanitizer_start_switch_fiber"();
        if (fptr)
            fptr(fake_stack_save, bottom, size);
    }

    nothrow @nogc
    void informSanitizerOfFinishSwitchFiber(void* fake_stack_save, const(void)** bottom_old, size_t* size_old)
    {
        auto fptr = getOptionalSanitizerFunc!"__sanitizer_finish_switch_fiber"();
        if (fptr)
            fptr(fake_stack_save, bottom_old, size_old);
    }

    nothrow @nogc
    void* asanGetCurrentFakeStack()
    {
        auto fptr = getOptionalSanitizerFunc!"__asan_get_current_fake_stack"();
        if (fptr)
            return fptr();
        else
            return null;
    }

    nothrow @nogc
    void* asanAddressIsInFakeStack(void *fake_stack, void *addr, void **beg, void **end)
    {
        auto fptr = getOptionalSanitizerFunc!"__asan_addr_is_in_fake_stack"();
        if (fptr)
            return fptr(fake_stack, addr, beg, end);
        else
            return null;
    }

    // This uses the forward declaration of `functionName` and returns a pointer to that function
    // if it is found in the executable, and `null` otherwise. Templated such that it can internally
    // cache the function pointer. Thread-safe.
    private auto getOptionalSanitizerFunc(string functionName)()
    {
        import ldc.intrinsics: llvm_expect;
        import core.atomic: atomicLoad, atomicStore, MemoryOrder;

        // If `fptr` is null, it's not initialized yet.
        // If `fptr` is 1, the function has not been found.
        // Otherwise, `fptr` is a valid function pointer.
        static shared typeof(mixin("&" ~ functionName)) fptr = null;
        enum FUNC_NOT_FOUND = cast(void*) 1;

        // Because `fptr` will never change after it's been initialized, we only have to make sure
        // that the read is atomic for thread safety.
        void* foundptr = atomicLoad!(MemoryOrder.raw)(fptr);

        if (llvm_expect(foundptr is null, false))
        {
            // Multiple threads may enter this branch. It is fine to do the redundant work.
            // The obtained `foundptr` should be the same for all threads and we can safely store it
            // atomically and use the local value afterwards.
            version (Darwin)
            {
                // On Darwin, ASan is always dynamically linked.
                import core.sys.posix.dlfcn : dlsym, dlopen;
                foundptr = dlsym(dlopen(null, 0), functionName);
            }
            else version (ELF)
            {
                // Check statically linked symbols
                foundptr = mixin("&" ~ functionName);
                if (!foundptr) {
                    // Check dynamically linked symbols
                    import core.sys.posix.dlfcn : dlsym, dlopen;
                    foundptr = dlsym(dlopen(null, 0), functionName);
                }
            }
            else version (Windows)
            {
                import core.sys.windows.windows : GetModuleHandleA, GetProcAddress;
                foundptr = GetProcAddress(GetModuleHandleA(null), functionName);
            }

            if (foundptr is null)
                foundptr = FUNC_NOT_FOUND;

            // It's ok if all threads write to `fptr` because it's the same value anyway, as long as
            // the write is atomic.
            atomicStore!(MemoryOrder.raw)(fptr, cast(typeof(fptr))foundptr);
        }

        // Expect false to maximize performance when sanitizers are not active.
        if (llvm_expect(foundptr != FUNC_NOT_FOUND, false))
            return cast(typeof(fptr)) foundptr;
        else
            return null;
    }

} // version (SupportSanitizers)
