//===-- tools/runtime-adapt/source/walk.d -------------------------*- D -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

module walk;

import std.algorithm : endsWith, sort;
import std.array : replace, split;
import std.file : dirEntries, exists, SpanMode;
import std.path : baseName, buildPath, dirSeparator, extension, relativePath;

struct RelFile
{
    string rel;
    string abs;
}

bool isDModuleName(string name)
{
    auto ext = extension(name);
    return ext == ".d" || ext == ".di";
}

/// Non-D runtime sources that still belong in the emitted tree.
bool isPassthroughName(string name)
{
    auto ext = extension(name);
    switch (ext)
    {
    case ".c", ".h", ".S", ".asm", ".inc", ".dd":
        return true;
    default:
        return false;
    }
}

bool isSourceName(string name)
{
    return isDModuleName(name) || isPassthroughName(name);
}

bool skipDirName(string name)
{
    switch (name)
    {
    case ".git", "test", "tests", "changelog", "__dummy":
        return true;
    default:
        return false;
    }
}

bool skipFileName(string name)
{
    switch (name)
    {
    case "test_runner.d":
        return true;
    default:
        return name.endsWith(".obj");
    }
}

private RelFile[] collectUnder(string root, string prefix)
{
    RelFile[] outp;
    if (!exists(root))
        return outp;
    foreach (e; dirEntries(root, SpanMode.depth))
    {
        if (e.isDir)
            continue;
        auto relFromRoot = relativePath(e.name, root).replace(dirSeparator, "/");
        bool skipped;
        foreach (part; relFromRoot.split("/"))
        {
            if (skipDirName(part))
            {
                skipped = true;
                break;
            }
        }
        if (skipped || skipFileName(baseName(e.name)) || !isSourceName(e.name))
            continue;
        auto rel = prefix.length ? prefix ~ "/" ~ relFromRoot : relFromRoot;
        outp ~= RelFile(rel, e.name);
    }
    return outp;
}

private RelFile[] collectRuntime(string ldcRoot, bool includeLdcPackage)
{
    import paths : druntimeSrc, phobosRoot;
    RelFile[] files;
    foreach (f; collectUnder(druntimeSrc(ldcRoot), ""))
    {
        if (!includeLdcPackage && (f.rel == "ldc" || f.rel.length > 4
                && f.rel[0 .. 4] == "ldc/"))
            continue;
        files ~= f;
    }
    auto ph = phobosRoot(ldcRoot);
    files ~= collectUnder(buildPath(ph, "std"), "std");
    files ~= collectUnder(buildPath(ph, "etc"), "etc");
    files.sort!((a, b) => a.rel < b.rel);
    return files;
}

/// Official/stock druntime + phobos. Never includes LDC's `ldc/` package.
RelFile[] walkStockRuntime(string root)
{
    return collectRuntime(root, false);
}

/// Full LDC runtime including `ldc/` — goal/guide only, never an emit source.
RelFile[] walkLdcRuntime(string ldcRoot)
{
    return collectRuntime(ldcRoot, true);
}

/// Merged tree (adapted overlay or LDC import/): object.d, core/, rt/, ldc/, std/
RelFile[] walkMergedTree(string root)
{
    auto files = collectUnder(root, "");
    files.sort!((a, b) => a.rel < b.rel);
    return files;
}

RelFile* findRel(RelFile[] files, string rel)
{
    foreach (ref f; files)
        if (f.rel == rel)
            return &f;
    return null;
}
