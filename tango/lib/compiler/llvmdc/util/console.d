/*******************************************************************************

        copyright:      Copyright (c) 2004 Tango group. All rights reserved

        license:        BSD style: $(LICENSE)

        version:        Initial release: July 2006


        Various low-level console oriented utilities

*******************************************************************************/

module util.console;

private import util.string;

version (Win32)
        {
        private extern (Windows) int GetStdHandle (int);
        private extern (Windows) int WriteFile (int, char*, int, int*, void*);
        }

else

version (Posix)
        {
        private extern (C) ptrdiff_t write (int, void*, size_t);
        }

/+
// emit a char[] to the console. Note that Win32 does not handle utf8, but
// then neither does fprintf (stderr). This will handle redirection though.
// May need to remedy the utf8 issue
int console (char[] s)
{
        version (Win32)
                {
                int count;
                if (WriteFile (GetStdHandle(0xfffffff5), s.ptr, s.length, &count, null))
                    return count;
                return -1;
                }
        else
        version (Posix)
                {
                return write (2, s.ptr, s.length);
                }
}

// emit an integer to the console
int console (uint i)
{
        char[10] tmp = void;

        return console (intToUtf8 (tmp, i));
}
+/

struct Console
{
    Console opCall (char[] s)
    {
            version (Win32)
                    {
                    int count;
                    WriteFile (GetStdHandle(0xfffffff5), s.ptr, s.length, &count, null);
                    }
            else
            version (Posix)
                    {
                    write (2, s.ptr, s.length);
                    }
            return *this;
    }

    // emit an integer to the console
    Console opCall (size_t i)
    {
            char[20] tmp = void;

            return console (intToUtf8 (tmp, i));
    }
}

Console console;
