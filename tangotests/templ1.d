module tangotests.templ1;

import Util = tango.text.Util;

extern(C) int printf(char*, ...);

void main()
{
    foreach (line; Util.lines("a\nb\nc"))
    {
        printf("%.*s\n", line.length, line.ptr);
    }
}
