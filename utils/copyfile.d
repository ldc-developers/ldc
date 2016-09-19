//===-- copyfile.d --------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Usage: copyfile <source> <target>
// Source and target must be filenames.
// Overwrites target.
//
//===----------------------------------------------------------------------===//

static import std.file;
import std.stdio;

int main(string[] args)
{
    if (args.length != 3)
    {
        writeln("Error: requires exactly 2 commandline arguments.");
        return -1;
    }

    std.file.copy(args[1], args[2]);

    return 0;
}
