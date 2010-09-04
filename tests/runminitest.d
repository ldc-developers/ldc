module runminitest;

import tango.sys.Environment,
       tango.io.Stdout, 
       tango.io.vfs.FileFolder;
import Path = tango.io.Path;
import Util = tango.text.Util;
import tango.text.convert.Format;
import tango.stdc.stdlib, 
       tango.stdc.stringz;

int main(char[][] args)
{
    enum : int
    {
        COMPILE,
        NOCOMPILE,
        RUN,
        NORUN
    }

    char[][] compilefailed;
    char[][] nocompilefailed;
    char[][] runfailed;
    char[][] norunfailed;

    Environment.cwd("mini");

    if (!Path.exists("obj"))
        Path.createFolder("obj");

    foreach(f; Path.children("./obj"))
    {
        Path.remove(f.path ~ f.name);
    }

    static int classify(char[] name)
    {
        char[] tail;
        char[] desc = Util.head(name, "_", tail);
        if ("compile" == desc)
            return COMPILE;
        else if ("nocompile" == desc)
            return NOCOMPILE;
        else if ("run" == desc)
            return RUN;
        else if ("norun" == desc)
            return NORUN;
        return RUN;
    }

    auto scan = new FileFolder (".");
    auto contents = scan.tree.catalog("*.d");
    foreach(c; contents) {
        auto testname = Path.parse(c.name).name;
        Stdout.formatln("TEST NAME: {}", testname);

        char[] cmd = Format.convert("ldc {} -quiet -L-s -ofobj/{}", c, testname);
        foreach(v; args[1..$]) {
            cmd ~= ' ';
            cmd ~= v;
        }
        int cl = classify(testname);
        if (cl == COMPILE || cl == NOCOMPILE)
            cmd ~= " -c";
        Stdout(cmd).newline;
        if (system(toStringz(cmd)) != 0) {
            if (cl != NOCOMPILE)
                compilefailed ~= c.toString;
        }
        else if (cl == RUN || cl == NORUN) {
            if (system(toStringz(Path.native("obj/" ~ testname))) != 0) {
                if (cl == RUN)
                    runfailed ~= c.toString;
            }
            else {
                if (cl == NORUN)
                    norunfailed ~= c.toString;
            }
        }
        else {
            if (cl == NOCOMPILE)
                nocompilefailed ~= c.toString;
        }
    }

    size_t nerrors = 0;

    if (compilefailed.length > 0)
    {
        Stdout.formatln("{}{}{}{}", compilefailed.length, '/', contents.files, " of the tests failed to compile:");
        foreach(b; compilefailed) {
            Stdout.formatln(" {}",b);
        }
        nerrors += compilefailed.length;
    }

    if (nocompilefailed.length > 0)
    {
        Stdout.formatln("{}{}{}{}", nocompilefailed.length, '/', contents.files, " of the tests failed to NOT compile:");
        foreach(b; nocompilefailed) {
            Stdout.formatln(" {}",b);
        }
        nerrors += nocompilefailed.length;
    }

    if (runfailed.length > 0)
    {
        Stdout.formatln("{}{}{}{}", runfailed.length, '/', contents.files, " of the tests failed to run:");
        foreach(b; runfailed) {
            Stdout.formatln("  {}",b);
        }
        nerrors += runfailed.length;
    }

    if (norunfailed.length > 0)
    {
        Stdout.formatln("{}{}{}{}", norunfailed.length, '/', contents.files, " of the tests failed to NOT run:");
        foreach(b; norunfailed) {
            Stdout.formatln(" {}",b);
        }
        nerrors += norunfailed.length;
    }

    Stdout.formatln("{}{}{}{}", contents.files - nerrors, '/', contents.files, " of the tests passed");

    return nerrors ? 1 : 0;
}
