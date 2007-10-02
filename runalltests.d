module runalltests;

import std.file;
import std.path;
import std.process;
import std.stdio;

int main(string[] args) {
    string[] good;
    string[] bad;

    auto contents = listdir("test", "*.d");
    foreach(c; contents) {
        if (system("./tester.sh "~getName(c)~" ll") != 0) {
            bad ~= c;
        }
        else {
            good ~= c;
        }
    }

    int ret = 0;
    if (bad.length > 0) {
        writefln(bad.length, '/', contents.length, " tests failed:");
        foreach(b; bad) {
            writefln("  ",b);
        }
        ret = 1;
    }

    writefln(good.length, '/', contents.length, " tests passed");
    return ret;
}
