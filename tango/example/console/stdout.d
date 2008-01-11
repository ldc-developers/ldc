/*******************************************************************************

        Illustrates the basic console formatting. This is different than
        the use of tango.io.Console, in that Stdout supports a variety of
        printf-style formatting, and has unicode-conversion support

*******************************************************************************/

private import tango.io.Stdout;

void main()
{
        Stdout ("hello, sweetheart \u263a").newline;
}
