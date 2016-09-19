//===-- removefile.d ------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Usage: removefile <filename>
// Does nothing if the filename does not exist.
//
//===----------------------------------------------------------------------===//

static import std.file;
import std.stdio;

int main(string[] args)
{
    if (args.length != 2)
    {
        writeln("Error: requires exactly 1 commandline argument.");
        return -1;
    }

    if (std.file.exists(args[1]))
        std.file.remove(args[1]);

    return 0;
}
