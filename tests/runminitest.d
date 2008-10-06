module runminitest;

import std.file;
import std.path;
import std.process;
import std.stdio;
import std.string;

int main(string[] args)
{
    enum : int
    {
        COMPILE,
        NOCOMPILE,
        RUN,
        NORUN
    }

    string[] compilefailed;
    string[] nocompilefailed;
    string[] runfailed;
    string[] norunfailed;

    chdir("mini");

    if (!exists("obj"))
        mkdir("obj");

    foreach(f; listdir("./obj", "*"))
    {
        std.file.remove(f);
    }

    static int classify(char[] name)
    {
        if (find(name, "compile_") == 0)
            return COMPILE;
        else if (find(name, "nocompile_") == 0)
            return NOCOMPILE;
        else if (find(name, "run_") == 0)
            return RUN;
        else if (find(name, "norun_") == 0)
            return NORUN;
        return RUN;
    }

    auto contents = listdir(".", "*.d");
    foreach(c; contents) {
        auto testname = getName(getBaseName(c));
        writefln("TEST NAME: ", testname);

        string cmd = format("ldc %s -quiet -ofobj" ~ std.path.sep ~ "%s -odobj", c, testname);
        foreach(v; args[1..$]) {
            cmd ~= ' ';
            cmd ~= v;
        }
        int cl = classify(testname);
        if (cl == COMPILE || cl == NOCOMPILE)
            cmd ~= " -c";
        writefln(cmd);
        if (system(cmd) != 0) {
            if (cl != NOCOMPILE)
                compilefailed ~= c;
        }
        else if (cl == RUN || cl == NORUN) {
            if (system("obj" ~ std.path.sep ~ testname) != 0) {
                if (cl == RUN)
                    runfailed ~= c;
            }
            else {
                if (cl == NORUN)
                    norunfailed ~= c;
            }
        }
        else {
            if (cl == NOCOMPILE)
                nocompilefailed ~= c;
        }
    }

    size_t nerrors = 0;

    if (compilefailed.length > 0)
    {
        writefln(compilefailed.length, '/', contents.length, " of the tests failed to compile:");
        foreach(b; compilefailed) {
            writefln("  ",b);
        }
        nerrors += compilefailed.length;
    }

    if (nocompilefailed.length > 0)
    {
        writefln(nocompilefailed.length, '/', contents.length, " of the tests failed to NOT compile:");
        foreach(b; nocompilefailed) {
            writefln("  ",b);
        }
        nerrors += nocompilefailed.length;
    }

    if (runfailed.length > 0)
    {
        writefln(runfailed.length, '/', contents.length, " of the tests failed to run:");
        foreach(b; runfailed) {
            writefln("  ",b);
        }
        nerrors += runfailed.length;
    }

    if (norunfailed.length > 0)
    {
        writefln(norunfailed.length, '/', contents.length, " of the tests failed to NOT run:");
        foreach(b; norunfailed) {
            writefln("  ",b);
        }
        nerrors += norunfailed.length;
    }

    writefln(contents.length - nerrors, '/', contents.length, " of the tests passed");

    return nerrors ? 1 : 0;
}
