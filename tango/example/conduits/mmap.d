
private import  tango.io.Console,
                tango.io.FileConduit,
                tango.io.MappedBuffer;

/*******************************************************************************

        open a file, map it into memory, and copy to console

*******************************************************************************/

void main (char[][] args)
{
        if (args.length is 2)
           {
           // open a file for reading
           auto mmap = new MappedBuffer (new FileConduit (args[1]));

           // copy content to console
           Cout (cast(char[]) mmap.slice) ();
           }
        else
           Cout ("usage is: mmap filename").newline;
}
