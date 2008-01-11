private import  tango.io.Stdout,
                tango.io.FileScan;

/*******************************************************************************

        List ".d" files and enclosing folders visible via a directory given
        as a command-line argument. In this example we're also postponing a
        flush on Stdout until output is complete. Stdout is usually flushed
        on each invocation of newline or formatln, but here we're using '\n'
        to illustrate how to avoid flushing many individual lines
        
*******************************************************************************/

void main(char[][] args)
{       
        char[] root = args.length < 2 ? "." : args[1];
        Stdout.formatln ("Scanning '{}'", root);

        auto scan = (new FileScan)(root, ".d");

        Stdout.format ("\n{} Folders\n", scan.folders.length);
        foreach (folder; scan.folders)
                 Stdout.format ("{}\n", folder);

        Stdout.format ("\n{0} Files\n", scan.files.length);
        foreach (file; scan.files)
                 Stdout.format ("{}\n", file);

        Stdout.formatln ("\n{} Errors", scan.errors.length);
        foreach (error; scan.errors)
                 Stdout (error).newline;
}
