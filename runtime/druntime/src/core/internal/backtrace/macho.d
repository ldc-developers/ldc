/**
 * This module extracts debug info from the currently running Mach-O executable.
 *
 * Copyright: Copyright Jacob Carlborg 2018.
 * License:   $(LINK2 http://www.boost.org/LICENSE_1_0.txt, Boost License 1.0)
 * Authors:   Jacob Carlborg
 * Source:    $(DRUNTIMESRC core/internal/backtrace/macho.d)
 */
module core.internal.backtrace.macho;

version (OSX)
    version = Darwin;
else version (iOS)
    version = Darwin;
else version (TVOS)
    version = Darwin;
else version (WatchOS)
    version = Darwin;

version (Darwin):

import core.stdc.config : c_ulong;
import core.stdc.stdlib : free;
import core.internal.macho.dl;
import core.internal.macho.io;

struct Image {
    SharedObject self;
    SharedObject debugObj;

    this(SharedObject self) {
        this.self = self;
        this.debugObj = self; 

        if (!self.hasSection("__DWARF", "__debug_line")) {
            auto dsymPath = getDsymDefaultPath();
            this.debugObj = SharedObject.fromFile(dsymPath.ptr);
        }
    }

    T processDebugLineSectionData(T)(scope T delegate(const(ubyte)[]) processor) {
        return processor(debugObj.getSection("__DWARF", "__debug_line"));
    }

    static Image openSelf() {
        return Image(SharedObject.thisExecutable());
    }

    @property bool isValid() {
        return self.isValid;
    }

    @property size_t baseAddress() {
        return cast(size_t)self.baseAddress;
    }
}
