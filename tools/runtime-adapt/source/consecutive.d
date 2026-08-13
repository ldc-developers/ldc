//===-- tools/runtime-adapt/source/consecutive.d ------------------*- D -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Walk consecutive git tags (materializeReference → .work/ref-*). Identity
// overlay at N, reconstruct, map upgrade to N+1. No workspace/ required.
//
//===----------------------------------------------------------------------===//

module consecutive;

import adapt;
import classify;
import generate;
import parseutil;
import paths;
import resolve;
import versions;
import walk;

import std.file : exists, mkdirRecurse, rmdirRecurse;
import std.format : format;
import std.path : buildPath;

struct ConsecutiveStep
{
    string fromTag;
    string toTag;
    string fromRoot;
    string toRoot;
    bool skipped;
    string skipReason;
    int refFiles;
    int toFiles;
    int identical;
    int rebase;
    int stillIdenticalAtTarget;
    int parseFailures;
    int reconstructMismatches;
    int astParseFail;
    int astRules;
    bool versionOk;
    string note;
}

ConsecutiveStep runConsecutiveStep(string ldcRoot, string fromTag, string toTag)
{
    ConsecutiveStep s;
    s.fromTag = fromTag;
    s.toTag = toTag;
    try
        s.fromRoot = materializeReference(ldcRoot, fromTag);
    catch (Exception e)
    {
        s.skipped = true;
        s.skipReason = e.msg;
        return s;
    }
    try
        s.toRoot = materializeReference(ldcRoot, toTag);
    catch (Exception e)
    {
        s.skipped = true;
        s.skipReason = e.msg;
        return s;
    }

    auto fromFiles = walkLdcRuntime(s.fromRoot);
    auto toFiles = walkLdcRuntime(s.toRoot);
    s.refFiles = cast(int) fromFiles.length;
    s.toFiles = cast(int) toFiles.length;

    // Identity overlay: stock runtime at fromTag is both reference and adapted.
    auto inv = classifyTrees(fromFiles, fromFiles, toFiles);
    auto c = countKinds(inv.rows);
    s.identical = c.identical;
    foreach (r; inv.rows)
    {
        if (r.kind == Kind.identical && r.targetPresent && !r.targetSameAsRef)
            s.rebase++;
        if (r.kind == Kind.identical && r.targetSameAsRef)
            s.stillIdenticalAtTarget++;
    }
    s.parseFailures = cast(int) inv.parseFailures.length;

    auto tmp = buildPath(workDir(ldcRoot), "id-" ~ fromTag);
    if (exists(tmp))
        rmdirRecurse(tmp);
    generateTree(inv, tmp, AdaptMode.verifyOnly, fromTag);
    s.reconstructMismatches = verifyGeneratedVsLdc(inv, tmp);

    auto astDir = buildPath(workDir(ldcRoot), "ast-" ~ fromTag);
    if (exists(astDir))
        rmdirRecurse(astDir);
    mkdirRecurse(astDir);
    foreach (rel; ["object.d", "rt/lifetime.d", "core/exception.d", "core/memory.d"])
    {
        import std.file : readText, write;
        import std.path : dirName;
        import std.array : split;
        string srcPath;
        foreach (f; fromFiles)
            if (f.rel == rel)
            {
                srcPath = f.abs;
                break;
            }
        if (!srcPath.length)
            continue;
        auto ad = adaptSource(readText(srcPath), rel, fromTag, AdaptMode.overlay);
        auto dest = buildPath(astDir ~ rel.split("/"));
        mkdirRecurse(dirName(dest));
        write(dest, ad.output);
        s.astRules += cast(int) ad.applied.length;
        if (!ad.parsedOut)
            s.astParseFail++;
    }

    s.versionOk = versioningHolds(fromTag) && versioningHolds(toTag);
    auto cons = matchConstraint(minorLadder, fromTag);
    s.note = format("constraint=%s files=%s→%s rebase=%s mismatch=%s astFail=%s astRules=%s ver=%s",
        cons !is null ? cons.id : "?", s.refFiles, s.toFiles, s.rebase,
        s.reconstructMismatches, s.astParseFail, s.astRules, s.versionOk);
    return s;
}

/// Compare generated overlay files back to the identity sources in `inv`.
int verifyGeneratedVsLdc(const Inventory inv, string generatedDir)
{
    import std.file : readText;
    import std.path : buildPath;
    import std.array : split;

    int bad;
    foreach (r; inv.rows)
    {
        if (r.action == Action.omit)
            continue;
        auto dest = buildPath(generatedDir ~ r.rel.split("/"));
        if (!exists(dest))
        {
            ++bad;
            continue;
        }
        auto src = r.action == Action.copyReference ? r.refPath : r.adaptedPath;
        if (normalizeSource(readText(src)) != normalizeSource(readText(dest)))
            ++bad;
    }
    return bad;
}

ConsecutiveStep[] runConsecutiveLadder(string ldcRoot, const string[] tags = null)
{
    auto seq = tags.length ? tags : consecutiveTags;
    ConsecutiveStep[] outp;
    foreach (i, tag; seq)
    {
        if (i + 1 >= seq.length)
            break;
        outp ~= runConsecutiveStep(ldcRoot, tag, seq[i + 1]);
    }
    return outp;
}
