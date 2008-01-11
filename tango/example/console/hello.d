/*******************************************************************************

        Hello World using tango.io

        This illustrates bare console output, with no fancy formatting. 

        Console I/O in Tango is UTF-8 across both linux and Win32. The
        conversion between various unicode representations is handled
        by higher level constructs, such as Stdout and Stderr

        Note that Cerr is tied to the console error output, and Cin is
        tied to the console input. 

*******************************************************************************/

import tango.io.Console;

void main()
{
        Cout ("hello, sweetheart \u263a").newline;
}
