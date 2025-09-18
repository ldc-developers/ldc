/**
 * Provides (read-only) memory-mapped I/O for Mach-O files.
 *
 *
 * Copyright: Copyright Kitsunebi Games 2025
 * License:   $(HTTP www.boost.org/LICENSE_1_0.txt, Boost License 1.0).
 * Authors:   Luna (the Foxgirl) Nielsen
 * Source: $(DRUNTIMESRC core/internal/macho/io.d)
 */

module core.internal.macho.io;

version (OSX)
    version = Darwin;
else version (iOS)
    version = Darwin;
else version (TVOS)
    version = Darwin;
else version (WatchOS)
    version = Darwin;

version (Darwin):

import core.stdc.stdio : snprintf;
import core.stdc.stdlib : free, malloc;
import core.sys.darwin.mach.dyld : _NSGetExecutablePath;

/**
    Returns the path to the process' executable as newly allocated slice.
*/
char[] thisExePath() {
    uint len;
    if (_NSGetExecutablePath(null, &len) != -1)
        return null;

    auto buffer = cast(char*) malloc(len);
    if (!buffer)
        return null;

    if (_NSGetExecutablePath(buffer, &len) != 0) {
        free(buffer);
        return null;
    }

    return buffer[0..len];
}

/**
    Gets the base file name of the given path.
*/
char[] getBaseName(char[] path) {
    size_t i;
    foreach_reverse(j; 0..path.length) {
        if (path[j] == '/') {
            i = j;
            break;
        }
    }
    return path[i+1..$];
}

/**
    Gets the default path of a dSYM file.
*/
char[] getDsymDefaultPath() {
    enum DSYM_SUBPATH_FMT = "%s.dSYM/Contents/Resources/DWARF/%s";

    char[] exePath = thisExePath();
    char[] exeBase = exePath.getBaseName();
    int len = snprintf(null, 0, DSYM_SUBPATH_FMT, exePath.ptr, exeBase.ptr);

    char* fullname = cast(char*)malloc(len+1);
    len = snprintf(fullname, len+1, DSYM_SUBPATH_FMT, exePath.ptr, exeBase.ptr);

    free(exePath.ptr);
    return fullname[0..len];
}