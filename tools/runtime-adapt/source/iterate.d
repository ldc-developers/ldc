//===-- tools/runtime-adapt/source/iterate.d ----------------------*- D -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Stock druntime/phobos is the reference input. The LDC checkout at
// workspace/refs/<tag> is the guide (compiler sources) and the goal
// (what the generated tree must become). LDC runtime is never copied.
//
//===----------------------------------------------------------------------===//

module iterate;

import adapt;
import astdiff;
import emit;
import paths;
import principles;
import resolve;
import versions;
import walk;

import std.array : appender;
import std.file : exists;
import std.format : format;
import std.path : buildPath;

struct Residual
{
    string rel;
    string kind; /// parse, no-rule
    string principleId;
    string locus;
    string note;
}

struct IterateRound
{
    int n;
    int emitted;
    int omitted;
    int astApplied;
    int residual;
}

struct IterateReport
{
    string versionTag;
    string referenceRoot;
    string ldcRoot;
    string guideRoot;
    bool versionOk;
    IterateRound[] rounds;
    Residual[] residuals;
    string outDir;
    bool equivalent;
    EmitReport emit;
    VersionAstReport astDiff;
}

IterateReport iterateVersion(string repoRoot, string tag, string againstRoot,
    string outDir, int maxRounds = 1, bool astLdcOnly = false)
{
    IterateReport rep;
    rep.versionTag = tag;
    // Compiler / goal is this LDC checkout. Stock may be HEAD or one --reference TAG.
    rep.referenceRoot = materializeStockReference(repoRoot, tag);
    rep.guideRoot = isLdcCheckout(repoRoot) ? repoRoot : materializeGuide(repoRoot, tag);
    rep.ldcRoot = againstRoot.length ? againstRoot : repoRoot;
    if (!isLdcRoot(rep.ldcRoot) && !isLdcCheckout(rep.ldcRoot))
        rep.ldcRoot = materializeTarget(repoRoot, againstRoot.length ? againstRoot : "HEAD");
    if (isThisCheckout(repoRoot, tag) || isLdcRoot(tag) || exists(tag))
        rep.versionOk = true;
    else
        rep.versionOk = versioningHolds(tag) || tag == consecutiveTags[$ - 1];
    rep.outDir = outDir;

    auto refFiles = walkStockRuntime(rep.referenceRoot);
    RelFile[] goalFiles;
    auto goalRoot = rep.guideRoot.length ? rep.guideRoot : rep.ldcRoot;
    if (isLdcRoot(goalRoot))
        goalFiles = walkLdcRuntime(goalRoot);
    auto er = emitRuntime(refFiles, goalFiles, outDir, tag, EmitProfile.native, rep.guideRoot);
    rep.emit = er;

    IterateRound rr;
    rr.n = 0;
    rr.emitted = er.emitted;
    rr.omitted = er.omitted;
    rr.residual = er.parseFail + er.unaccounted;
    rep.rounds ~= rr;

    foreach (f; er.files)
    {
        if (f.action != EmitAction.omit && !f.parsedOut)
        {
            Residual r;
            r.rel = f.rel;
            r.kind = "parse";
            r.principleId = f.principleId;
            r.note = f.error;
            rep.residuals ~= r;
        }
        else if (f.action != EmitAction.omit && !f.principleId.length)
        {
            Residual r;
            r.rel = f.rel;
            r.kind = "no-rule";
            r.note = "add a Principle in principles.d";
            rep.residuals ~= r;
        }
    }
    auto goalForAst = rep.guideRoot.length ? rep.guideRoot : rep.ldcRoot;
    if (isLdcRoot(goalForAst))
    {
        auto stockForCmp = isLdcRoot(rep.referenceRoot) ? rep.referenceRoot : "";
        rep.astDiff = diffGeneratedVsGoal(outDir, goalForAst, tag, astLdcOnly, stockForCmp);
        try
        {
            import std.file : write, mkdirRecurse;
            mkdirRecurse(outDir);
            write(buildPath(outDir, "AST-DIFF.md"), renderVersionAst(rep.astDiff));
            write(buildPath(outDir, "FILE-CMP.md"), renderFileCmp(rep.astDiff));
        }
        catch (Exception)
        {
        }
    }
    rep.equivalent = er.validated;
    return rep;
}

struct AllVersionsReport
{
    IterateReport[] versions;
    VersionAstReport[] astDiffs;
    int ran;
    int skipped;
    int failed;
    int astLdcGaps;
}

/// Emit the complete runtime for each tag. Incremental: skip a tag whose
/// `generated-<tag>/FILE-CMP.md` already exists unless `force`.
AllVersionsReport iterateAllVersions(string repoRoot, string outBase,
    const string[] tags = null, bool force = false)
{
    AllVersionsReport all;
    import std.file : write, append, mkdirRecurse, exists;
    mkdirRecurse(outBase);
    auto astAll = buildPath(outBase, "ast-diff.md");
    auto runTags = tags.length ? tags : tagWindow();
    write(astAll, format("# AST/FILE-CMP — generated vs per-version goal runtime\n\n"
        ~ "Window: %s tag(s) ending at `%s`.\n"
        ~ "Goal is `workspace/refs/<tag>` runtime (never copied).\n"
        ~ "Close ldc/* gaps in `source/ldcmods/`.\n\n",
        runTags.length, runTags.length ? runTags[$ - 1] : latestMinorTag()));
    foreach (tag; runTags)
    {
        import std.stdio : stderr;
        auto dest = buildPath(outBase, "generated-" ~ sanitize(tag));
        if (!force && exists(buildPath(dest, "FILE-CMP.md")))
        {
            stderr.writeln("all-versions  ", tag, "  skip (FILE-CMP.md exists; --force to redo)");
            IterateReport kept;
            kept.versionTag = tag;
            kept.outDir = dest;
            kept.equivalent = true;
            kept.versionOk = true;
            all.skipped++;
            all.versions ~= kept;
            continue;
        }
        stderr.writeln("all-versions  ", tag);
        IterateReport one;
        try
            one = iterateVersion(repoRoot, tag, repoRoot, dest, 1, true);
        catch (Throwable e)
        {
            one.versionTag = tag;
            one.outDir = dest;
            Residual r;
            r.kind = "skip";
            r.note = e.msg;
            one.residuals ~= r;
            all.skipped++;
            all.versions ~= one;
            continue;
        }
        all.ran++;
        if (!one.equivalent || !one.versionOk)
            all.failed++;
        all.astLdcGaps += one.astDiff.ldcWithGaps;
        try
        {
            import std.file : write, mkdirRecurse;
            mkdirRecurse(dest);
            auto piece = renderVersionAst(one.astDiff);
            write(buildPath(dest, "AST-DIFF.md"), piece);
            write(buildPath(dest, "FILE-CMP.md"), renderFileCmp(one.astDiff));
            append(astAll, piece);
        }
        catch (Exception)
        {
        }
        one.emit.files = null;
        one.residuals = null;
        one.astDiff.files = null;
        all.versions ~= one;
        stderr.writeln("  done ", tag, " emitted=", one.emit.emitted,
            " ldcGaps=", one.astDiff.ldcWithGaps);
    }
    return all;
}

string renderIterate(const IterateReport rep)
{
    auto buf = appender!string();
    buf.put(format("# iterate %s\n\n", rep.versionTag));
    buf.put(format("- reference: `%s`\n- guide: `%s`\n- ldc-goal: `%s`\n- versionOk: %s\n- validated: %s\n",
        rep.referenceRoot, rep.guideRoot, rep.ldcRoot, rep.versionOk, rep.equivalent));
    buf.put(format("- emitted: %s omitted: %s parsedOk: %s parseFail: %s rules %s/%s unaccounted: %s\n",
        rep.emit.emitted, rep.emit.omitted, rep.emit.parsedOk, rep.emit.parseFail,
        rep.emit.rulesResolved, rep.emit.rulesFired, rep.emit.unaccounted));
    buf.put(format("- refProduct: %s sourcedFromRef: %s sourcedFromCompiler: %s ldcOnly: %s productComplete: %s\n",
        rep.emit.refProduct, rep.emit.sourcedFromRef, rep.emit.sourcedFromCompiler,
        rep.emit.ldcOnly, rep.emit.productComplete));
    buf.put(format("- astDiff: compared=%s present=%s textsEqual=%s textDiffs=%s missingFiles=%s extra=%s ldcGaps=%s stockGaps=%s adaptDelta=%s missedLdc=%s\n\n",
        rep.astDiff.compared, rep.astDiff.present, rep.astDiff.textsEqual,
        rep.astDiff.textDiffs, rep.astDiff.missingFiles, rep.astDiff.extraFiles,
        rep.astDiff.ldcWithGaps, rep.astDiff.stockWithGaps,
        rep.astDiff.adaptDeltas, rep.astDiff.missedLdcPatches));
    buf.put("| round | emitted | residual |\n|---|---:|---:|\n");
    foreach (rr; rep.rounds)
        buf.put(format("| %s | %s | %s |\n", rr.n, rr.emitted, rr.residual));
    buf.put("\n## Residuals (close by editing principles.d / adapt.d)\n\n");
    if (!rep.residuals.length)
        buf.put("(none — every reference druntime/phobos file was emitted)\n");
    else
    {
        buf.put("| path | kind | principle | locus |\n|---|---|---|---|\n");
        int n;
        foreach (r; rep.residuals)
        {
            buf.put(format("| `%s` | %s | %s | %s |\n",
                r.rel, r.kind, r.principleId, r.locus));
            if (++n >= 80)
            {
                buf.put("\n_(truncated)_\n");
                break;
            }
        }
    }
    return buf.data;
}

string renderAllVersions(const AllVersionsReport all)
{
    auto buf = appender!string();
    buf.put("# iterate all versions\n\n");
    buf.put(format("- ran: %s skipped: %s failed: %s astLdcGaps: %s\n\n",
        all.ran, all.skipped, all.failed, all.astLdcGaps));
    buf.put("| tag | emitted | complete | validated | ldc AST gaps | missing files |\n");
    buf.put("|---|---:|---|---|---:|---:|\n");
    foreach (v; all.versions)
        buf.put(format("| `%s` | %s | %s | %s | %s | %s |\n",
            v.versionTag, v.emit.emitted, v.emit.productComplete, v.equivalent,
            v.astDiff.ldcWithGaps, v.astDiff.missingFiles));
    buf.put("\nSee `.work/ast-diff.md` and each `generated-<tag>/{AST-DIFF,FILE-CMP}.md`.\n");
    buf.put("Re-run a skipped tag with `--force`. Change the span with `--window N` or `--from`/`--to`.\n");
    return buf.data;
}
