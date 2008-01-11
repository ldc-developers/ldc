/*******************************************************************************
  copyright:   Copyright (c) 2006 Juan Jose Comellas. All rights reserved
  license:     BSD style: $(LICENSE)
  author:      Juan Jose Comellas <juanjo@comellas.com.ar>
*******************************************************************************/

private import tango.io.Stdout;
private import tango.sys.Process;
private import tango.core.Exception;

private import tango.text.stream.LineIterator;


/**
 * Example program for the tango.sys.Process class.
 */
void main()
{
    version (Windows)
        char[] command = "ping -n 4 localhost";
    else version (Posix)
        char[] command = "ping -c 4 localhost";
    else
        assert(false, "Unsupported platform");

    try
    {
        auto p = new Process(command, null);

        Stdout.formatln("Executing {0}", p.toString());
        p.execute();

        Stdout.formatln("Output from process: {0} (pid {1})\n---",
                        p.programName, p.pid);

        foreach (line; new LineIterator!(char)(p.stdout))
        {
            Stdout.formatln("{0}", line);
        }

        Stdout.print("---\n");

        auto result = p.wait();

        Stdout.formatln("Process '{0}' ({1}) finished: {2}",
                        p.programName, p.pid, result.toString());
    }
    catch (ProcessException e)
    {
        Stdout.formatln("Process execution failed: {0}", e.toString());
    }
    catch (IOException e)
    {
        Stdout.formatln("Input/output exception caught: {0}", e.toString());
    }
    catch (Exception e)
    {
        Stdout.formatln("Unexpected exception caught: {0}", e.toString());
    }
}
