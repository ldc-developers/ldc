//===-- tools/runtime-adapt/tests/consecutive.d -------------------*- D -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

module tests.consecutive;

import consecutive;
import parseutil;
import paths;
import resolve;
import versions;
import walk;

import std.file : exists, getcwd;
import std.path : buildPath, dirName;

private string toolCwd()
{
    auto cwd = getcwd();
    if (exists(buildPath(cwd, "source", "app.d")) && exists(buildPath(cwd, "dub.sdl")))
        return cwd;
    auto p = buildPath(cwd, "tools", "runtime-adapt");
    if (exists(buildPath(p, "source", "app.d")))
        return p;
    return cwd;
}

private string ldcFromTool(string tool)
{
    return dirName(dirName(tool));
}

unittest
{
    assert(consecutiveTags.length == 13);
    foreach (i, tag; consecutiveTags[0 .. $ - 1])
    {
        auto cur = matchConstraint(minorLadder, tag);
        auto nxt = matchConstraint(minorLadder, consecutiveTags[i + 1]);
        assert(cur !is null && nxt !is null);
        assert(cur.reference == tag);
        assert(!appliesTo(*cur, consecutiveTags[i + 1]));
        assert(!appliesTo(*nxt, tag));
        assert(versionKey(consecutiveTags[i + 1]) > versionKey(tag));
    }
}

unittest
{
    auto ldc = ldcFromTool(toolCwd());
    int ran;
    // Sample the ends and the 1.36 pin so dub test stays bounded; full
    // ladder is `dub run -- --consecutive`.
    static immutable pairs = [
        ["v1.30.0", "v1.31.0"],
        ["v1.36.0", "v1.37.0"],
        ["v1.41.0", "v1.42.0"],
    ];
    foreach (pair; pairs)
    {
        auto tag = pair[0];
        auto next = pair[1];
        if (!isLdcRoot(buildPath(workDir(ldc), "ref-" ~ tag))
            && !isLdcRoot(tag))
        {
            // Release path: extract via git archive; skip if this host has no tag.
            try
            {
                materializeReference(ldc, tag);
                materializeReference(ldc, next);
            }
            catch (Exception)
                continue;
        }
        auto st = runConsecutiveStep(ldc, tag, next);
        assert(!st.skipped, st.skipReason);
        assert(st.refFiles > 0);
        assert(st.reconstructMismatches == 0, st.note);
        assert(st.astParseFail == 0, st.note);
        assert(st.versionOk, st.note);
        assert(st.identical == st.refFiles);
        // libdparse: object.d at both tags is module object
        import std.file : readText;
        auto objFrom = parseDSource(
            readText(buildPath(st.fromRoot, "runtime", "druntime", "src", "object.d")),
            "object.d");
        auto objTo = parseDSource(
            readText(buildPath(st.toRoot, "runtime", "druntime", "src", "object.d")),
            "object.d");
        assert(objFrom.parsed && objTo.parsed);
        assert(objFrom.moduleName == "object");
        assert(objTo.moduleName == "object");
        ++ran;
    }
    // Zero pairs is OK when git tags cannot be archived on this host.
}
