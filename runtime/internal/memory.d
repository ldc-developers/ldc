/**
 * This module exposes functionality for inspecting and manipulating memory.
 *
 * Copyright: Copyright (C) 2005-2006 Digital Mars, www.digitalmars.com.
 *            All rights reserved.
 * License:
 *  This software is provided 'as-is', without any express or implied
 *  warranty. In no event will the authors be held liable for any damages
 *  arising from the use of this software.
 *
 *  Permission is granted to anyone to use this software for any purpose,
 *  including commercial applications, and to alter it and redistribute it
 *  freely, in both source and binary form, subject to the following
 *  restrictions:
 *
 *  o  The origin of this software must not be misrepresented; you must not
 *     claim that you wrote the original software. If you use this software
 *     in a product, an acknowledgment in the product documentation would be
 *     appreciated but is not required.
 *  o  Altered source versions must be plainly marked as such, and must not
 *     be misrepresented as being the original software.
 *  o  This notice may not be removed or altered from any source
 *     distribution.
 * Authors:   Walter Bright, Sean Kelly
 */
module memory;

version = GC_Use_Dynamic_Ranges;

version(Posix)
{
    version = GC_Use_Data_Proc_Maps;
}
version(solaris)
{
    version = GC_Use_Data_Proc_Maps;
}

version(GC_Use_Data_Proc_Maps)
{
    version(Posix) {} else {
        static assert(false, "Proc Maps only supported on Posix systems");
    }
    private import tango.stdc.posix.unistd;
    private import tango.stdc.posix.fcntl;
    private import tango.stdc.string;

    version = GC_Use_Dynamic_Ranges;
}

private
{
    version( linux )
    {
        //version = SimpleLibcStackEnd;

        version( SimpleLibcStackEnd )
        {
            extern (C) extern void* __libc_stack_end;
        }
        else
        {
            import tango.stdc.posix.dlfcn;
        }
    }
    version(LDC)
    {
        pragma(intrinsic, "llvm.frameaddress")
        {
                void* llvm_frameaddress(uint level=0);
        }
    }
}


/**
 *
 */

version( solaris ) {	
    version(X86_64) {
        extern (C) void* _userlimit;
    }
}

extern (C) void* rt_stackBottom()
{
    version( Win32 )
    {
        void* bottom;
        asm
        {
            mov EAX, FS:4;
            mov bottom, EAX;
        }
        return bottom;
    }
    else version( linux )
    {
        version( SimpleLibcStackEnd )
        {
            return __libc_stack_end;
        }
        else
        {
            // See discussion: http://autopackage.org/forums/viewtopic.php?t=22
                static void** libc_stack_end;

                if( libc_stack_end == libc_stack_end.init )
                {
                    void* handle = dlopen( null, RTLD_NOW );
                    libc_stack_end = cast(void**) dlsym( handle, "__libc_stack_end" );
                    dlclose( handle );
                }
                return *libc_stack_end;
        }
    }
    else version( darwin )
    {
        // darwin has a fixed stack bottom
        return cast(void*) 0xc0000000;
    }
    else version( solaris )
    {
        version(X86_64) {
            return _userlimit;
        }
        else {
            // <sys/vmparam.h>
            return cast(void*) 0x8048000;
        }
    }
    else
    {
        static assert( false, "Operating system not supported." );
    }
}


/**
 *
 */
extern (C) void* rt_stackTop()
{
    version(LDC)
    {
        return llvm_frameaddress();
    }
    else version( D_InlineAsm_X86 )
    {
        asm
        {
            naked;
            mov EAX, ESP;
            ret;
        }
    }
    else
    {
            static assert( false, "Architecture not supported." );
    }
}


private
{
    version( Win32 )
    {
        extern (C)
        {
            extern int _data_start__;
            extern int _bss_end__;
        }

        alias _data_start__ Data_Start;
        alias _bss_end__    Data_End;
    }
    else version( linux )
    {
        extern (C)
        {
            extern int _data;
            extern int __data_start;
            extern int _end;
            extern int _data_start__;
            extern int _data_end__;
            extern int _bss_start__;
            extern int _bss_end__;
            extern int __fini_array_end;
        }

        alias __data_start  Data_Start;
        alias _end          Data_End;
    }
    else version( solaris )
    {
        extern(C)
        {
            extern int _edata;
            extern int _end;
        }

        alias _edata        Data_Start;
        alias _end          Data_End;
    }

    version( GC_Use_Dynamic_Ranges )
    {
        private import tango.stdc.stdlib;

        struct DataSeg
        {
            void* beg;
            void* end;
        }

        DataSeg* allSegs = null;
        size_t   numSegs = 0;

        extern (C) void _d_gc_add_range( void* beg, void* end )
        {
            void* ptr = realloc( allSegs, (numSegs + 1) * DataSeg.sizeof );

            if( ptr ) // if realloc fails, we have problems
            {
                allSegs = cast(DataSeg*) ptr;
                allSegs[numSegs].beg = beg;
                allSegs[numSegs].end = end;
                numSegs++;
            }
        }

        extern (C) void _d_gc_remove_range( void* beg )
        {
            for( size_t pos = 0; pos < numSegs; ++pos )
            {
                if( beg == allSegs[pos].beg )
                {
                    while( ++pos < numSegs )
                    {
                        allSegs[pos-1] = allSegs[pos];
                    }
                    numSegs--;
                    return;
                }
            }
        }
    }

    alias void delegate( void*, void* ) scanFn;

    void* dataStart,  dataEnd;
}


/**
 *
 */
extern (C) void rt_scanStaticData( scanFn scan )
{
    scan( dataStart, dataEnd );

    version( GC_Use_Dynamic_Ranges )
    {
        for( size_t pos = 0; pos < numSegs; ++pos )
        {
            scan( allSegs[pos].beg, allSegs[pos].end );
        }
    }
}

void initStaticDataPtrs()
{
    const int S = (void*).sizeof;

    // Can't assume the input addresses are word-aligned
    static void* adjust_up( void* p )
    {
        return p + ((S - (cast(size_t)p & (S-1))) & (S-1)); // cast ok even if 64-bit
    }

    static void * adjust_down( void* p )
    {
        return p - (cast(size_t) p & (S-1));
    }

    version( Win32 )
    {
        dataStart = adjust_up( &Data_Start );
        dataEnd   = adjust_down( &Data_End );
    }
    else version(linux)
    {
        dataStart = adjust_up( &Data_Start );
        dataEnd   = adjust_down( &Data_End );
    }
    else version(solaris)
    {
        dataStart = adjust_up( &Data_Start );
        dataEnd   = adjust_down( &Data_End );
    }
    else
    {
        static assert( false, "Operating system not supported." );
    }

    version( GC_Use_Data_Proc_Maps )
    {
        parseDataProcMaps();
    }
}

version( GC_Use_Data_Proc_Maps )
{
version(solaris)
{
    typedef long offset_t;
    enum : uint { PRMAPSZ = 64, MA_WRITE = 0x02 }
    extern(C)
    {
        struct prmap {
            uintptr_t pr_vaddr;         /* virtual address of mapping */
            size_t pr_size;             /* size of mapping in bytes */
            char[PRMAPSZ]  pr_mapname;  /* name in /proc/<pid>/object */
            private offset_t pr_offset; /* offset into mapped object, if any */
            int pr_mflags;              /* protection and attribute flags (see below) */
            int pr_pagesize;            /* pagesize (bytes) for this mapping */
            int pr_shmid;               /* SysV shmid, -1 if not SysV shared memory */

            private int[1] pr_filler;
        }
    }

    void parseDataProcMaps()
    {
        debug (ProcMaps) printf("initStaticDataPtrs()\n");
        // http://docs.sun.com/app/docs/doc/816-5174/proc-4
        prmap pr;

        int   fd = open("/proc/self/map", O_RDONLY);
        scope (exit) close(fd);

        while (prmap.sizeof == read(fd, &pr, prmap.sizeof))
        if (pr.pr_mflags & MA_WRITE)
        {
            void* start = cast(void*) pr.pr_vaddr;
            void* end   = cast(void*)(pr.pr_vaddr + pr.pr_size);
            debug (ProcMaps) printf("  vmem at %p - %p with size %d bytes\n", start, end, pr.pr_size);

            // Exclude stack  and  dataStart..dataEnd
            if ( ( !dataEnd ||
                !( dataStart >= start && dataEnd <= end ) ) &&
                !( &pr >= start && &pr < end ) )
            {
                // we already have static data from this region.  anything else
                // is heap (%% check)
                debug (ProcMaps) printf("  Adding map range %p - %p\n", start, end);
                _d_gc_add_range(start, end);
            }
        }
    }
}
else
{
    const int S = (void*).sizeof;

    // TODO: This could use cleanup!
    void parseDataProcMaps()
    {
        // TODO: Exclude zero-mapped regions

        int   fd = open("/proc/self/maps", O_RDONLY);
        ptrdiff_t   count; // %% need to configure ret for read..
        char  buf[2024];
        char* p;
        char* e;
        char* s;
        void* start;
        void* end;

        p = buf.ptr;
        if (fd != -1)
        {
            while ( (count = read(fd, p, buf.sizeof - (p - buf.ptr))) > 0 )
            {
                e = p + count;
                p = buf.ptr;
                while (true)
                {
                    s = p;
                    while (p < e && *p != '\n')
                        p++;
                    if (p < e)
                    {
                        // parse the entry in [s, p)
                        static if( S == 4 )
                        {
                            enum Ofs
                            {
                                Write_Prot = 19,
                                Start_Addr = 0,
                                End_Addr   = 9,
                                Addr_Len   = 8,
                            }
                        }
                        else static if( S == 8 )
                        {
                            //X86-64 only has 12 bytes address space(in PAE mode) - not 16
                            //We also need the 32 bit offsets for 32 bit apps
                            version(X86_64) {
                                enum Ofs
                                {
                                    Write_Prot = 27,
                                    Start_Addr = 0,
                                    End_Addr   = 13,
                                    Addr_Len   = 12,
                                    Write_Prot_32 = 19,
                                    Start_Addr_32 = 0,
                                    End_Addr_32   = 9,
                                    Addr_Len_32   = 8,
                                }
                            }
                            else
                            {
                                enum Ofs
                                {
                                    Write_Prot = 35,
                                    Start_Addr = 0,
                                    End_Addr   = 9,
                                    Addr_Len   = 17,
                                }
                            }
                        }
                        else
                        {
                            static assert( false );
                        }

                        // %% this is wrong for 64-bit:
                        // long strtoul(const char*,char**,int);
                        // but seems to work on x86-64:
                        // probably because C's long is 64 bit there

                        if( s[Ofs.Write_Prot] == 'w' )
                        {
                            s[Ofs.Start_Addr + Ofs.Addr_Len] = '\0';
                            s[Ofs.End_Addr + Ofs.Addr_Len] = '\0';
                            start = cast(void*) strtoul(s + Ofs.Start_Addr, null, 16);
                            end   = cast(void*) strtoul(s + Ofs.End_Addr, null, 16);

                            // 1. Exclude anything overlapping [dataStart, dataEnd)
                            // 2. Exclude stack
                            if ( ( !dataEnd ||
                                !( dataStart >= start && dataEnd <= end ) ) &&
                                !( &buf[0] >= start && &buf[0] < end ) )
                            {
                                // we already have static data from this region.  anything else
                                // is heap (%% check)
                                debug (ProcMaps) printf("Adding map range %p 0%p\n", start, end);
                                _d_gc_add_range(start, end);
                            }
                        }
                        version(X86_64)
                        {
                            //We need to check here for 32 bit apps like ldc produces
                            //and add them to the gc scan range
                            if( s[Ofs.Write_Prot_32] == 'w' )
                            {
                                s[Ofs.Start_Addr_32 + Ofs.Addr_Len_32] = '\0';
                                s[Ofs.End_Addr_32 + Ofs.Addr_Len_32] = '\0';
                                start = cast(void*) strtoul(s + Ofs.Start_Addr_32, null, 16);
                                end   = cast(void*) strtoul(s + Ofs.End_Addr_32, null, 16);
                                if ( ( !dataEnd ||
                                    !( dataStart >= start && dataEnd <= end ) ) &&
                                    !( &buf[0] >= start && &buf[0] < end ) )
                                {
                                    _d_gc_add_range(start, end);
                                }
                            }
                        }

                        p++;
                    }
                    else
                    {
                        count = p - s;
                        memmove(buf.ptr, s, cast(size_t)count);
                        p = buf.ptr + count;
                        break;
                    }
                }
            }
            close(fd);
        }
    }
}
}