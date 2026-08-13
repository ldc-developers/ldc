//===-- tools/runtime-adapt/source/resolve.d ----------------------*- D -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// A reference is (in order): a directory the caller already has,
// workspace/refs/<tag> (optional cache of official druntime/phobos), or
// a git archive of this LDC repo into .work/ref-<tag>/. Generate logic
// never lives in workspace/ — only the reference input may.
//
//===----------------------------------------------------------------------===//

module resolve;

import paths;

import std.array : replace;
import std.file : exists, mkdirRecurse;
import std.path : buildPath;
import std.process : execute;
import std.stdio : stderr, writeln;
import std.string : strip;

/// Optional workspace cache of an LDC checkout (compiler + goal runtime).
string workspaceRef(string ldcRoot, string tag)
{
    return buildPath(toolRoot(ldcRoot), "workspace", "refs", tag);
}

/// Optional official/stock druntime/phobos (never LDC's runtime).
string workspaceStock(string ldcRoot, string tag)
{
    return buildPath(toolRoot(ldcRoot), "workspace", "stock", tag);
}

/// LDC compiler checkout (driver/ + gen/), used as a guide — not as runtime input.
bool isLdcCheckout(string p)
{
    return p.length
        && exists(buildPath(p, "driver", "main.cpp"))
        && exists(buildPath(p, "gen", "llvmhelpers.cpp"));
}

/// This LDC checkout (HEAD). Generate uses this unless `--reference TAG`.
bool isThisCheckout(string ldcRoot, string reference)
{
    if (!reference.length || reference == "HEAD" || reference == "."
        || reference == "latest")
        return true;
    if (ldcRoot.length && (reference == ldcRoot || abs(reference) == abs(ldcRoot)))
        return true;
    return false;
}

/// Directory, workspace/refs/<tag>, or git ref → tree with runtime/druntime/src.
/// Caches land in untracked `workspace/refs/` or `.work/ref-<tag>/`.
string materializeReference(string ldcRoot, string reference)
{
    if (isThisCheckout(ldcRoot, reference))
        return abs(ldcRoot);
    if (reference.length && isLdcRoot(reference))
        return abs(reference);

    auto ws = workspaceRef(ldcRoot, reference);
    if (isLdcRoot(ws))
        return ws;

    auto dest = buildPath(workDir(ldcRoot), "ref-" ~ sanitize(reference));
    if (isLdcRoot(dest))
        return dest;

    writeln("extracting   ", reference, " via git archive → ", dest);
    mkdirRecurse(dest);
    archiveTree(ldcRoot, reference, dest, "runtime/druntime");

    if (!exists(buildPath(dest, "runtime", "druntime", "src", "object.d")))
        archiveGitlink(ldcRoot, reference, dest, "runtime/druntime",
            buildPath(ldcRoot, "runtime", "druntime"));

    auto sha = gitlinkSha(ldcRoot, reference, "runtime/phobos");
    auto phobosGit = buildPath(ldcRoot, "runtime", "phobos");
    if (sha.length)
        archiveCommit(phobosGit, sha, buildPath(dest, "runtime", "phobos"), ["std", "etc"]);
    else
        archiveTree(ldcRoot, reference, dest, "runtime/phobos");

    if (!exists(buildPath(dest, "runtime", "druntime", "src", "object.d")))
        throw new Exception("reference extract missing object.d: " ~ dest
            ~ " (need a git tag in this LDC repo or a directory with runtime/druntime/src)");
    return dest;
}

/// Official stock tree. Prefers workspace/stock/<tag>. An LDC checkout is
/// accepted only so walkStockRuntime can skip `ldc/`; its runtime is not
/// the reference product.
string materializeStockReference(string ldcRoot, string reference)
{
    if (isThisCheckout(ldcRoot, reference))
        return abs(ldcRoot);
    if (reference.length && isLdcRoot(reference) && !isLdcCheckout(reference))
        return abs(reference);
    auto stock = workspaceStock(ldcRoot, reference);
    if (isLdcRoot(stock))
        return stock;
    return materializeReference(ldcRoot, reference);
}

/// Workspace LDC checkout at `tag` (compiler source + goal). Never an emit body.
string materializeGuide(string ldcRoot, string tag)
{
    // Generate always parses *this* checkout's compiler unless the one
    // optional --reference TAG is itself an LDC compiler tree.
    if (isThisCheckout(ldcRoot, tag) && isLdcCheckout(ldcRoot))
        return abs(ldcRoot);
    if (tag.length && isLdcCheckout(tag))
        return abs(tag);
    auto ws = workspaceRef(ldcRoot, tag);
    if (isLdcCheckout(ws))
        return ws;
    if (isLdcCheckout(ldcRoot))
        return ldcRoot;
    return "";
}

string materializeTarget(string ldcRoot, string target)
{
    if (target.length == 0 || target == "HEAD" || target == ldcRoot)
        return ldcRoot;
    if (isLdcRoot(target) || isLdcCheckout(target))
        return abs(target);
    return materializeReference(ldcRoot, target);
}

bool isLdcRoot(string p)
{
    return p.length && exists(buildPath(p, "runtime", "druntime", "src", "object.d"));
}

string sanitize(string s)
{
    return s.replace("/", "_").replace("\\", "_").replace(":", "_");
}

private void archiveTree(string repo, string rev, string dest, string prefix)
{
    auto zip = dest ~ "-" ~ sanitize(prefix) ~ ".zip";
    auto r = execute(["git", "-C", repo, "archive", "--format=zip", "-o", zip,
        rev, "--", prefix]);
    if (r.status != 0)
        return;
    auto ux = execute(["tar", "-xf", zip, "-C", dest]);
    if (ux.status != 0)
        stderr.writeln("warning: extract ", prefix, " failed:\n", ux.output);
}

private void archiveGitlink(string ldcRoot, string rev, string dest, string path,
    string submoduleGit)
{
    auto sha = gitlinkSha(ldcRoot, rev, path);
    if (!sha.length)
        return;
    archiveCommit(submoduleGit, sha, buildPath(dest, path), null);
}

private void archiveCommit(string repo, string sha, string dest, string[] paths)
{
    if (!exists(buildPath(repo, ".git")) && !exists(repo))
    {
        stderr.writeln("warning: no git dir ", repo);
        return;
    }
    mkdirRecurse(dest);
    auto zip = dest ~ "-" ~ sha[0 .. (sha.length < 8 ? sha.length : 8)] ~ ".zip";
    string[] cmd = ["git", "-C", repo, "archive", "--format=zip", "-o", zip, sha];
    if (paths.length)
    {
        cmd ~= "--";
        cmd ~= paths;
    }
    auto r = execute(cmd);
    if (r.status != 0)
    {
        stderr.writeln("warning: git archive ", sha, " in ", repo, ":\n", r.output);
        return;
    }
    auto ux = execute(["tar", "-xf", zip, "-C", dest]);
    if (ux.status != 0)
        stderr.writeln("warning: extract ", sha, " failed:\n", ux.output);
}

private string gitlinkSha(string ldcRoot, string rev, string path)
{
    auto r = execute(["git", "-C", ldcRoot, "ls-tree", rev, path]);
    if (r.status != 0)
        return "";
    auto parts = r.output.strip.splitWs;
    // 160000 commit <sha>\tpath
    if (parts.length >= 3 && parts[1] == "commit")
        return parts[2];
    return "";
}

private string[] splitWs(string s)
{
    import std.algorithm : filter;
    import std.array : array, split;
    return s.split().filter!(a => a.length > 0).array;
}
