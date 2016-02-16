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

const char* _data_start__;
const char* _data_end__;
const char* _bss_start__;
const char* _bss_end__;

EXTERN_C IMAGE_DOS_HEADER __ImageBase;

static void init_data_seg(void)
{
    // Get handle to this module (.exe/.dll)
    HMODULE hModule = (HMODULE) &__ImageBase;
    char* imageBase = (char*) hModule;

    // Get the DOS header
    PIMAGE_DOS_HEADER pDosHeader = (PIMAGE_DOS_HEADER) hModule;

    // Get the address of the NT headers
    PIMAGE_NT_HEADERS pNtHeaders = (PIMAGE_NT_HEADERS) (imageBase + pDosHeader->e_lfanew);

    // After the NT headers comes the sections table
    PIMAGE_SECTION_HEADER pSectionHeader = (PIMAGE_SECTION_HEADER) (pNtHeaders + 1);

    // Iterate over all sections
    for (int i = 0; i < pNtHeaders->FileHeader.NumberOfSections; i++)
    {
         BYTE* name = pSectionHeader->Name;
         if (memcmp(name, ".data", 6) == 0)
         {
            _data_start__ = imageBase + pSectionHeader->VirtualAddress;
            _data_end__ = _data_start__ + pSectionHeader->Misc.VirtualSize;
         }
         else if (memcmp(name, ".bss", 5) == 0)
         {
            _bss_start__ = imageBase + pSectionHeader->VirtualAddress;
            _bss_end__ = _bss_start__ + pSectionHeader->Misc.VirtualSize;
         }

         pSectionHeader++;
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

