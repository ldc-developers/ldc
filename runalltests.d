module runalltests;

import std.file;
import std.path;
import std.process;
import std.stdio;

int main(string[] args) {
    string[] bad;
    string[] badrun;

    chdir("test");

    auto contents = listdir(".", "*.d");
    foreach(c; contents) {
        auto cmd = "llvmdc -quiet "~c;
        writefln(cmd);
        if (system(cmd) != 0) {
            bad ~= c;
        }
        else if (system(getName(c)) != 0) {
            badrun ~= c;
        }
    }

    int ret = 0;
    if (bad.length > 0 || badrun.length > 0) {
        writefln(bad.length, '/', contents.length, " of the tests failed to compile:");
        foreach(b; bad) {
            writefln("  ",b);
        }
        writefln(badrun.length, '/', contents.length - bad.length, " of the compiled tests failed to run:");
        foreach(b; badrun) {
            writefln("  ",b);
        }
        ret = 1;
    }

    writefln(contents.length - bad.length - badrun.length, '/', contents.length, " of the tests passed");
    return ret;
}
