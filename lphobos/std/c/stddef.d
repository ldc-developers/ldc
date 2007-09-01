
/**
 * C's &lt;stddef.h&gt;
 * Authors: Walter Bright, Digital Mars, www.digitalmars.com
 * License: Public Domain
 * Macros:
 *	WIKI=Phobos/StdCStddef
 */

module std.c.stddef;

version (Win32)
{
    alias wchar wchar_t;
}
else version (linux)
{
    alias dchar wchar_t;
}
else
{
    static assert(0);
}
