private import tango.stdc.stdio;

private import tango.io.Console;

/*******************************************************************************

        Demonstrates buffered output. Console output (and Stdout etc) is
        buffered, requiring a flush or newline to render on the console.

*******************************************************************************/

void main (char[][] args)
{
        Cout ("how now ");
        printf ("printf\n");
        Cout ("brown cow").newline;
}
