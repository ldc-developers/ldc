/**
 * Contains forward references to the AddressSanitizer interface.
 * See compiler-rt/include/sanitizer/asan_interface.h
 *
 * Copyright: Authors 2017-2017
 * License:   $(LINK2 http://www.boost.org/LICENSE_1_0.txt, Boost License 1.0)
 * Authors:   LDC Developers
 */
module ldc.asan;

@system:
@nogc:
nothrow:
extern (C):

// Poisons memory region [addr, addr+size) for AddressSanitizer.
// Method is NOT thread-safe in the sense that no two threads can
// (un)poison memory in the same memory region simultaneously.
void __asan_poison_memory_region(const(void*) addr, size_t size);

// Unpoisons memory region [addr, addr+size) for AddressSanitizer.
// Method is NOT thread-safe in the sense that no two threads can
// (un)poison memory in the same memory region simultaneously.
void __asan_unpoison_memory_region(const(void*) addr, size_t size);

// Returns 1 if the byte at addr is poisoned for AddressSanitizer.
// Otherwise returns 0.
int __asan_address_is_poisoned(const(void*) addr);
