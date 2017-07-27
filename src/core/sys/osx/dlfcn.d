/**
 * D header file for OS X
 *
 * $(LINK2 https://opensource.apple.com/source/dyld/dyld-360.22/include/dlfcn.h, macOS dlfcn.h)
 *
 * Copyright: Copyright David Nadlinger 2016.
 * License:   $(WEB www.boost.org/LICENSE_1_0.txt, Boost License 1.0).
 * Authors:   David Nadlinger
 */
module core.sys.osx.dlfcn;

public import core.sys.posix.dlfcn;

version (OSX):

struct Dl_info
{
    const(char)* dli_fname;
    void*        dli_fbase;
    const(char)* dli_sname;
    void*        dli_saddr;
}

extern(C) int dladdr(in void* addr, Dl_info* info) nothrow @nogc;

enum RTLD_NOLOAD = 0x10;
enum RTLD_NODELETE = 0x80;
enum RTLD_FIRST = 0x100;
