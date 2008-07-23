module runminitest;

import std.file;
import std.path;
import std.process;
import std.stdio;
import std.string;

int main(string[] args) {
    string[] bad;
    string[] badrun;

    chdir("mini");
    if(!exists("obj"))
        mkdir("obj");

    auto contents = listdir(".", "*.d");
    foreach(c; contents) {
        string cmd = format("llvmdc %s -quiet -ofobj/%s", c, getName(c));
        foreach(v; args[1..$]) {
            cmd ~= ' ';
            cmd ~= v;
        }
        writefln(cmd);
        if (system(cmd) != 0) {
            bad ~= c;
        }
        else if (system("obj/" ~ getName(c)) != 0) {
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
