/**
 * This module provides OSX-specific support for sections.
 *
 * Copyright: Copyright The D Language Foundation 2008 - 2016.
 * License: Distributed under the
 *      $(LINK2 http://www.boost.org/LICENSE_1_0.txt, Boost Software License 1.0).
 *    (See accompanying file LICENSE)
 * Authors: Walter Bright, Sean Kelly, Martin Nowak, David Nadlinger
 * Source: $(DRUNTIMESRC src/rt/_mach_utils.d)
 */
module rt.mach_utils;

version (OSX):

import core.sys.osx.mach.dyld;
import core.sys.osx.mach.getsect;

struct SectionRef
{
    immutable(char)* seg;
    immutable(char)* sect;
}

static immutable SectionRef[] dataSections =
[
    {SEG_DATA, SECT_DATA},
    {SEG_DATA, SECT_BSS},
    {SEG_DATA, SECT_COMMON}
];

ubyte[] getSection(in mach_header* header, intptr_t slide,
                   in char* segmentName, in char* sectionName)
{
    version (X86)
    {
        assert(header.magic == MH_MAGIC);
        auto sect = getsectbynamefromheader(header,
                                            segmentName,
                                            sectionName);
    }
    else version (X86_64)
    {
        assert(header.magic == MH_MAGIC_64);
        auto sect = getsectbynamefromheader_64(cast(mach_header_64*)header,
                                            segmentName,
                                            sectionName);
    }
    else
        static assert(0, "unimplemented");

    if (sect !is null && sect.size > 0)
        return (cast(ubyte*)sect.addr + slide)[0 .. cast(size_t)sect.size];
    return null;
}
