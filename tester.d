module tester;

import std.file;
import std.path;
import std.process;
import std.stdio;
import std.string;

void printUsage(string cmd)
{
    writefln("Usage:");
    writefln("  ",cmd," %%name %%cmd %%...");
    writefln("%%name:");
    writefln("  name of test without path or extension. eg: bug1");
    writefln("%%cmd:");
    writefln("  c   = compile module");
    writefln("  gdb = same as 'c' but launches compiler in gdb");
    writefln("  ll  = compile module and print the disassemled bitcode");
    writefln("  llo = compile and optimize module, then print the disassemled bitcode");
    writefln("%%...");
    writefln("  the rest of the command line options are passed directly to llvmdc");
}

string testFileName(string test, string ext="")
{
    return "tangotests/"~test~ext;
}

// couldnt get execvp to work
int execute(string cmd)
{
    return system(cmd);
}
int execute(string cmd, string[] args)
{
    char[] c = cmd.dup;
    foreach(v; args) {
        c ~= ' ';
        c ~= v;
    }
    writefln(c);
    return system(c);
}

void compileTest(string test, string[] args)
{
    args = [testFileName(test,".d")] ~ args;
    if (execute("llvmdc", args) != 0) {
        throw new Exception("Failed to compile test: "~test);
    }
}

void disassembleTest(string test, bool print)
{
    string[] args = ["-f",testFileName(test,".bc")];
    if (execute("llvm-dis", args) != 0) {
        throw new Exception("Failed to disassemble test: "~test);
    }
    if (print) {
        execute("cat "~testFileName(test,".ll"));
    }
}

void debugTest(string test, string[] common)
{
    string[] args = ["--args", "llvmdc", testFileName(test,".d")];
    args ~= common;
    if (execute("gdb", args) != 0) {
        throw new Exception("Failed to compile test: '"~test~"' for debugging");
    }
}

void optimizeTest(string test)
{
    string bc = testFileName(test,".bc");
    if (execute("opt -std-compile-opts -f -o="~bc~" "~bc)) {
        throw new Exception("Failed to optimize test: "~test);
    }
}

void runTest(string test)
{
    if (execute(testFileName(test))) {
        throw new Exception("Failed to run test: "~test);
    }
}

int main(string[] args)
{
    if (args.length < 3) {
        printUsage(args[0]);
        return 1;
    }

    string test = args[1];
    string kind = args[2];

    string[] compilelink = ["-Itangotests","-odtangotests"];
    compilelink ~= args[3..$];
    string[] compileonly = compilelink.dup;

    compileonly ~= "-c";
    compilelink ~= "-of"~testFileName(test);

    switch(kind) {
    case "c":
        compileTest(test,compileonly);
        break;
    case "gdb":
        debugTest(test,compileonly);
        break;
    case "ll":
        compileTest(test,compileonly);
        disassembleTest(test,true);
        break;
    case "llo":
        compileTest(test,compileonly);
        optimizeTest(test);
        disassembleTest(test,true);
        break;
    case "run":
        compileTest(test,compilelink);
        runTest(test);
        break;
    default:
        throw new Exception("Invalid command: "~kind);
    }
    return 0;
}
