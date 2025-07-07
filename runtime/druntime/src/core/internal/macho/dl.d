/**
 * Simplifies working with shared Macho-O objects of the current process.
 *
 * Copyright: Copyright Kitsunebi Games 2025
 * License:   $(HTTP www.boost.org/LICENSE_1_0.txt, Boost License 1.0).
 * Authors:   Luna (the Foxgirl) Nielsen
 * Source: $(DRUNTIMESRC core/internal/macho/dl.d)
 */

module core.internal.macho.dl;

version (OSX)
    version = Darwin;
else version (iOS)
    version = Darwin;
else version (TVOS)
    version = Darwin;
else version (WatchOS)
    version = Darwin;

version (Darwin):

import core.sys.darwin.mach.getsect : mach_header_64, getsectiondata, getsectbynamefromheader_64;
import core.sys.darwin.dlfcn;
import core.sys.darwin.fcntl;
import core.sys.darwin.sys.mman;
import core.sys.posix.sys.stat;
import core.sys.darwin.mach.loader;
import core.sys.darwin.mach.dyld : 
    _dyld_image_count, 
    _dyld_get_image_name, 
    _dyld_get_image_header, 
    _dyld_get_image_vmaddr_slide;

/**
    Enables iterating over the process' currently loaded shared objects.
*/
struct SharedObjects {
@nogc nothrow:
    ///
    alias Callback = int delegate(SharedObject);

    ///
    static int opApply(scope Callback dg)
    {
        foreach(i; 0.._dyld_image_count) {
            if (int result = dg(SharedObject.fromIndex(i)))
                return result;
        }
        return 0;
    }
}

/**
    A loaded mach-o binary.
*/
struct SharedObject {
@nogc nothrow:
private:
    mach_header_64* _header;
    ptrdiff_t vmaddr_slide;
    const(char)* _name;

public:

    /**
        Returns the shared object with the given index.
    */
    static SharedObject fromIndex(uint idx) {
        return SharedObject(
            cast(mach_header_64*)_dyld_get_image_header(idx),
            _dyld_get_image_vmaddr_slide(idx),
            _dyld_get_image_name(idx)
        );
    }

    /**
        Returns the shared object with the given dlopen handle.

        Params:
            dlhandle = Handle returned by dlopen
        
        Returns:
            A SharedObject instance matching the given dlhandle,
            or an empty handle on failure.
    */
    static SharedObject fromHandle(void* dlhandle) {
        foreach(so; SharedObjects) {
            if (auto hndl = dlopen(so.name, RTLD_NOLOAD)) {
                dlclose(hndl);

                if (hndl is dlhandle)
                    return so;
            }
        }
        return SharedObject.init;
    }

    /**
        Returns the shared object with the given dlopen handle.

        Params:
            path = Path to the object to load.
        
        Returns:
            A SharedObject instance matching the given path,
            or an empty handle on failure.
    */
    static SharedObject fromFile(const(char)* path) {
        if (auto hndl = dlopen(path, 0))
            return SharedObject.fromHandle(hndl);

        mach_header_64* base_header = cast(mach_header_64*)_dyld_get_image_header(0);

        // Try opening and mapping the file.
        int fd = open(path, O_RDONLY);
        if (fd == -1)
            return SharedObject.init;
        
        stat_t fdInfo;
        if (fstat(fd, &fdInfo) == -1)
            return SharedObject.init;
        
        void* data = mmap(null, fdInfo.st_size, PROT_READ, MAP_SHARED, fd, 0);
        if (data == MAP_FAILED)
            return SharedObject.init;

        // Non-fat mach-o object.
        mach_header* hdr = cast(mach_header*)data;
        if (hdr.magic == MH_MAGIC || hdr.magic == MH_MAGIC_64) {
            if (hdr.cputype != base_header.cputype || hdr.cpusubtype != base_header.cpusubtype) {
                munmap(data, fdInfo.st_size);
                return SharedObject.init;
            }

            return SharedObject(cast(mach_header_64*)data, -1, path);
        }

        // Fat binary.
        fat_header* fat = cast(fat_header*)data;
        if (fat.magic == [0xca, 0xfe, 0xba, 0xbe]) {
            fat_entry* entry = cast(fat_entry*)(data+fat_header.sizeof);
            foreach(i; 0..fat.count) {
                if (entry.cputype == base_header.cputype && entry.cpusubtype == base_header.cpusubtype) {
                    return SharedObject(cast(mach_header_64*)(data+entry.file_offset), -1, path);
                }

                entry++;
            }
        }

        // Remember to unmap the file.
        munmap(data, fdInfo.st_size);
        return SharedObject.init;
    }

    /**
        Returns the object of the current process' executable.
    */
    static SharedObject thisExecutable() {
        return SharedObject.fromIndex(0);
    }

    /**
        Whether the object is valid.
    */
    @property bool isValid() {
        return _header !is null;
    }

    /**
        The name of this object.
    */
    @property const(char)* name() {
        return _name;
    }

    /**
        The base address of this object.
    */
    @property void* baseAddress() {
        return cast(void*)_header;
    }

    /**
        The mach header of the image.
    */
    @property mach_header_64* header() {
        return _header;
    }

    /**
        The virtual memory slide for the image.
    */
    @property ptrdiff_t slide() {
        return vmaddr_slide;
    }

    /**
        Gets whether the given section is present.
    */
    bool hasSection(const(char)* segname, const(char)* sectname) {
        return getSection(segname, sectname).length > 0;
    }

    /**
        Gets the given section within the shared object/image.
    */
    ubyte[] getSection(const(char)* segname, const(char)* sectname) {

        // mmapped mach-o
        if (vmaddr_slide == -1 && _header) {
            if (auto sect = getsectbynamefromheader_64(_header, segname, sectname)) {
                return (cast(ubyte*)_header+sect.offset)[0..sect.size];
            }
            return null;
        }

        // linked mach-o
        if (_header) {
            size_t len;
            if (auto data = getsectiondata(_header, segname, sectname, &len))
                return data[0..len];
        }

        return null;
    }
}
