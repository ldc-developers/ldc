//===-- driver/exe_path.d - Executable path management ----------*- D -*-====//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Stores the program's executable path and provides some helpers to generate
// derived paths.
//
//===----------------------------------------------------------------------===//

module driver.exe_path;


private string _exePath;

static this()
{
    import std.file : thisExePath;
    _exePath = thisExePath();
}

/// path to the current executable
/// <baseDir>/bin/ldc2
public @property string exePath()
{
    return _exePath;
}

/// path to the bin dir of the current executable
/// <baseDir>/bin
public @property string binDir()
{
    import std.path : dirName;
    return dirName(_exePath);
}

/// <baseDir>
public @property string baseDir()
{
    import std.path : dirName;
    return dirName(binDir);
}

/// <baseDir>/bin/suffix
public string prependBinDir(string suffix)
{
    import std.path : chainPath;
    import std.conv : to;

    return chainPath(binDir, suffix).to!string;
}
