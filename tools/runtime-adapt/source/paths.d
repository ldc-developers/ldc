//===-- tools/runtime-adapt/source/paths.d ------------------------*- D -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

module paths;

import std.file : exists, getcwd, thisExePath;
import std.path : absolutePath, baseName, buildNormalizedPath, buildPath, dirName;

struct Roots
{
    string ldc;           /// this LDC checkout
    string reference;     /// git ref (v1.36.0) or directory
    string adapted;       /// existing adapted tree (object.d + core/ + std/)
    string target;        /// git ref or directory to upgrade toward (default: this checkout)
    string outDir;
    string reportMd;
    string reportJson;
}

string findLdcRoot()
{
    string[] candidates = [getcwd()];
    auto cwd = getcwd();
    if (baseName(cwd) == "runtime-adapt" && baseName(dirName(cwd)) == "tools")
        candidates ~= dirName(dirName(cwd));

    auto p = dirName(thisExePath);
    foreach (_; 0 .. 6)
    {
        candidates ~= p;
        p = dirName(p);
    }

    foreach (c; candidates)
    {
        if (exists(buildPath(c, "runtime", "druntime", "src", "object.d"))
            && exists(buildPath(c, "driver", "main.cpp"))
            && exists(buildPath(c, "gen", "llvmhelpers.cpp")))
            return buildNormalizedPath(c);
    }
    return buildNormalizedPath(cwd);
}

/// Overlay lives only under this LDC tree (never a sibling project).
string guessAdapted(string ldcRoot)
{
    foreach (p; [
        buildPath(ldcRoot, "tools", "runtime-adapt", "overlays", "current"),
    ])
    {
        if (exists(buildPath(p, "object.d")))
            return p;
    }
    return "";
}

string toolRoot(string ldcRoot)
{
    return buildPath(ldcRoot, "tools", "runtime-adapt");
}

Roots defaultRoots()
{
    Roots r;
    r.ldc = findLdcRoot();
    r.reference = "HEAD";
    r.adapted = guessAdapted(r.ldc);
    r.target = r.ldc;
    r.outDir = defaultOutputDir(r.ldc);
    r.reportMd = buildPath(workDir(r.ldc), "report.md");
    r.reportJson = buildPath(workDir(r.ldc), "report.json");
    return r;
}

/// Generate output under this LDC checkout. Relative `--output` is from the repo root.
string defaultOutputDir(string ldcRoot)
{
    return buildPath(workDir(ldcRoot), "generated");
}

string resolveOutput(string ldcRoot, string given)
{
    import std.path : isAbsolute;
    if (!given.length)
        return defaultOutputDir(ldcRoot);
    if (isAbsolute(given))
        return abs(given);
    return abs(buildPath(ldcRoot, given));
}

unittest
{
    auto p = resolveOutput("E:/ldc", "tools/runtime-adapt/.work/generated");
    assert(p.length);
    assert(defaultOutputDir("E:/ldc").length >= 9);
}

string druntimeSrc(string ldcRoot)
{
    return buildPath(ldcRoot, "runtime", "druntime", "src");
}

string phobosRoot(string ldcRoot)
{
    return buildPath(ldcRoot, "runtime", "phobos");
}

void requireExists(string path, string what)
{
    import std.exception : enforce;
    enforce(exists(path), what ~ " not found: " ~ path);
}

string abs(string p)
{
    return buildNormalizedPath(absolutePath(p));
}

string workDir(string ldcRoot)
{
    return buildPath(ldcRoot, "tools", "runtime-adapt", ".work");
}

/// Gitignored caches filled by generate / --prefetch-refs / --reference TAG.
string[] cacheDirs(string ldcRoot)
{
    auto tool = toolRoot(ldcRoot);
    return [
        workDir(ldcRoot),
        buildPath(tool, "workspace", "refs"),
        buildPath(tool, "workspace", "stock"),
        buildPath(tool, "workspace", "overlays"),
        buildPath(tool, "workspace", ".tmp"),
        buildPath(tool, "clones"),
        buildPath(tool, "overlays"),
        buildPath(tool, "bin"),
        buildPath(tool, ".dub"),
    ];
}

/// Remove cache dirs. Tracked source, tests, and workspace/*.ps1 stay.
int cleanCaches(string ldcRoot)
{
    import std.file : exists, rmdirRecurse;
    int n;
    foreach (p; cacheDirs(ldcRoot))
    {
        if (!p.length || !exists(p))
            continue;
        rmdirRecurse(p);
        ++n;
    }
    return n;
}

unittest
{
    auto d = cacheDirs("X");
    assert(d.length >= 7);
    bool sawWork, sawRefs;
    foreach (p; d)
    {
        if (p.length >= 5 && p[$ - 5 .. $] == ".work")
            sawWork = true;
        if (p.length >= 4 && p[$ - 4 .. $] == "refs")
            sawRefs = true;
    }
    assert(sawWork && sawRefs);
}
