module runalltests;

import std.file;
import std.path;
import std.process;
import std.stdio;

int main(string[] args) {
    string[] bad;
    string[] badrun;

    auto contents = listdir("test", "*.d");
    foreach(c; contents) {
        auto cmd = "./tester.sh "~getName(c);
        if (system(cmd~" ll") != 0) {
            bad ~= c;
        }
        else if (system(cmd~" run") != 0) {
            badrun ~= c;
        }
    }

    int ret = 0;
    if (bad.length > 0) {
        writefln(bad.length, '/', contents.length, " tests failed to compile:");
        foreach(b; bad) {
            writefln("  ",b);
        }
        writefln(badrun.length, '/', contents.length, " tests failed to run:");
        foreach(b; badrun) {
            writefln("  ",b);
        }
        ret = 1;
    }

    writefln(contents.length - bad.length - badrun.length, '/', contents.length, " tests passed");
    return ret;
}
