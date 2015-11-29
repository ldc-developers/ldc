/**
 * Implementation of support routines for synchronized blocks.
 *
 * Copyright: Copyright The LDC Developers 2012
 * License:   <a href="http://www.boost.org/LICENSE_1_0.txt">Boost License 1.0</a>.
 * Authors:   Kai Nacke <kai@redstar.de>
 */

/*          Copyright The LDC Developers 2012.
 * Distributed under the Boost Software License, Version 1.0.
 *    (See accompanying file LICENSE or copy at
 *          http://www.boost.org/LICENSE_1_0.txt)
 */

/* ================================= Win32 ============================ */

#if _WIN32

#if _MSC_VER || __MINGW64__

#include <Windows.h>
#include <string.h>

const void* _data_start__;
const void* _data_end__;
const void* _bss_start__;
const void* _bss_end__;

static void init_data_seg(void)
{
    // Get handle to module
    HMODULE hModule = GetModuleHandle(NULL);

    // Store the base address the loaded Module
    char* dllImageBase = (char*) hModule; //suppose hModule is the handle to the loaded Module (.exe or .dll)

    // Get the DOS header
    IMAGE_DOS_HEADER* dos_header = (PIMAGE_DOS_HEADER) hModule;

    // Get the address of NT Header
    IMAGE_NT_HEADERS *pNtHdr = (PIMAGE_NT_HEADERS)((char*) hModule + dos_header->e_lfanew);

    // After Nt headers comes the table of section, so get the addess of section table
    IMAGE_SECTION_HEADER *pSectionHdr = (IMAGE_SECTION_HEADER *) (pNtHdr + 1);

    // Iterate through the list of all sections, and check the section name in the if conditon. etc
    int i;
    for ( i = 0 ; i < pNtHdr->FileHeader.NumberOfSections ; i++ )
    {
         char *name = (char*) pSectionHdr->Name;
         if (memcmp(name, ".data", 6) == 0)
         {
            _data_start__ = dllImageBase + pSectionHdr->VirtualAddress;
            _data_end__ = (char *) _data_start__ + pSectionHdr->Misc.VirtualSize;
         }
         else if (memcmp(name, ".bss", 5) == 0)
         {
            _bss_start__ = dllImageBase + pSectionHdr->VirtualAddress;
            _bss_end__ = (char *) _bss_start__ + pSectionHdr->Misc.VirtualSize;
         }
         pSectionHdr++;
    }
}


typedef int  (__cdecl *_PF)(void);

static int __cdecl ctor(void)
{
    init_data_seg();
    return 0;
}

static int __cdecl dtor(void)
{
    return 0;
}


#pragma data_seg(push)

#pragma section(".CRT$XIY", long, read)
#pragma section(".CRT$XTY", long, read)

#pragma data_seg(".CRT$XIY")
__declspec(allocate(".CRT$XIY")) static _PF _ctor = &ctor;

#pragma data_seg(".CRT$XTY")
__declspec(allocate(".CRT$XTY")) static _PF _dtor = &dtor;

#pragma data_seg(pop)
#endif

#endif

