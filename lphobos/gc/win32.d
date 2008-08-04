
// Copyright (C) 2001-2002 by Digital Mars
// All Rights Reserved
// www.digitalmars.com
// Written by Walter Bright

import std.c.windows.windows;

alias int pthread_t;

/***********************************
 * Map memory.
 */

void *os_mem_map(uint nbytes)
{
    return VirtualAlloc(null, nbytes, MEM_RESERVE, PAGE_READWRITE);
}

/***********************************
 * Commit memory.
 * Returns:
 *	0	success
 *	!=0	failure
 */

int os_mem_commit(void *base, uint offset, uint nbytes)
{
    void *p;

    p = VirtualAlloc(base + offset, nbytes, MEM_COMMIT, PAGE_READWRITE);
    return cast(int)(p == null);
}


/***********************************
 * Decommit memory.
 * Returns:
 *	0	success
 *	!=0	failure
 */

int os_mem_decommit(void *base, uint offset, uint nbytes)
{
    return cast(int)VirtualFree(base + offset, nbytes, MEM_DECOMMIT) == 0; 
}

/***********************************
 * Unmap memory allocated with os_mem_map().
 * Memory must have already been decommitted.
 * Returns:
 *	0	success
 *	!=0	failure
 */

int os_mem_unmap(void *base, uint nbytes)
{
    return cast(int)VirtualFree(base, 0, MEM_RELEASE) == 0; 
}


/********************************************
 */

pthread_t pthread_self()
{
    //printf("pthread_self() = %x\n", GetCurrentThreadId());
    return cast(pthread_t) GetCurrentThreadId();
}

/**********************************************
 * Determine "bottom" of stack (actually the top on Win32 systems).
 */

void *os_query_stackBottom()
{
    asm
    {
	naked			;
	mov	EAX,FS:4	;
	ret			;
    }
}

/**********************************************
 * Determine base address and size of static data segment.
 */

version (GNU)
{
// This is MinGW specific
extern (C)
{
    // TODO: skip the .rdata between .data and .bss?
    extern int _data_start__;
    extern int _bss_end__;
}

void os_query_staticdataseg(void **base, uint *nbytes)
{
    *base = cast(void *)&_data_start__;
    *nbytes = cast(uint)(cast(char *)&_bss_end__ - cast(char *)&_data_start__);
}

}
else
{
    
extern (C)
{
    extern int _xi_a;	// &_xi_a just happens to be start of data segment
    extern int _edata;	// &_edata is start of BSS segment
    extern int _end;	// &_end is past end of BSS
}

void os_query_staticdataseg(void **base, uint *nbytes)
{
    *base = cast(void *)&_xi_a;
    *nbytes = cast(uint)(cast(char *)&_end - cast(char *)&_xi_a);
}

}
/++++

void os_query_staticdataseg(void **base, uint *nbytes)
{
    static char dummy = 6;
    SYSTEM_INFO si;
    MEMORY_BASIC_INFORMATION mbi;
    char *p;
    void *bottom = null;
    uint size = 0;

    // Tests show the following does not work reliably.
    // The reason is that the data segment is arbitrarily divided
    // up into PAGE_READWRITE and PAGE_WRITECOPY.
    // This means there are multiple regions to query, and
    // can even wind up including the code segment.
    assert(0);				// fix implementation

    GetSystemInfo(&si);
    p = (char *)((uint)(&dummy) & ~(si.dwPageSize - 1));
    while (VirtualQuery(p, &mbi, sizeof(mbi)) == sizeof(mbi) &&
        mbi.Protect & (PAGE_READWRITE | PAGE_WRITECOPY) &&
        !(mbi.Protect & PAGE_GUARD) &&
        mbi.AllocationBase != 0)
    {
	bottom = (void *)mbi.BaseAddress;
	size = (uint)mbi.RegionSize;

	printf("dwPageSize        = x%x\n", si.dwPageSize);
	printf("&dummy            = %p\n", &dummy);
	printf("BaseAddress       = %p\n", mbi.BaseAddress);
	printf("AllocationBase    = %p\n", mbi.AllocationBase);
	printf("AllocationProtect = x%x\n", mbi.AllocationProtect);
	printf("RegionSize        = x%x\n", mbi.RegionSize);
	printf("State             = x%x\n", mbi.State);
	printf("Protect           = x%x\n", mbi.Protect);
	printf("Type              = x%x\n\n", mbi.Type);

	p -= si.dwPageSize;
    }

    *base = bottom;
    *nbytes = size;
}

++++/
