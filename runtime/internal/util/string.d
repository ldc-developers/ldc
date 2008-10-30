/*******************************************************************************

        copyright:      Copyright (c) 2004 Tango group. All rights reserved

        license:        BSD style: $(LICENSE)

        version:        Initial release: July 2006


        Various char[] utilities

*******************************************************************************/

module util.string;

private import tango.stdc.string;

// convert uint to char[], within the given buffer
// Returns a valid slice of the populated buffer
char[] intToUtf8 (char[] tmp, size_t val)
in {
   assert (tmp.length > 20, "atoi buffer should be 20 or more chars wide");
   }
body
{
    char* p = tmp.ptr + tmp.length;

    do {
       *--p = cast(char)((val % 10) + '0');
       } while (val /= 10);

    return tmp [cast(size_t)(p - tmp.ptr) .. $];
}


// function to compare two strings
int stringCompare (char[] s1, char[] s2)
{
    auto len = s1.length;

    if (s2.length < len)
        len = s2.length;

    int result = memcmp(s1.ptr, s2.ptr, len);

    if (result == 0)
        result = cast(int)s1.length - cast(int)s2.length;

    return result;
}
