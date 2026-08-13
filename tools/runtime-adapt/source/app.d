//===-- tools/runtime-adapt/source/app.d --------------------------*- D -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

module app;

import adapt;
import astdiff;
import classify;
import consecutive;
import generate;
import iterate;
import paths;
import report;
import resolve;
import versions;
import walk;

import std.conv : to;
import std.file : exists, mkdirRecurse, write;
import std.format : format;
import std.path : buildPath, dirName;
import std.stdio;

int main(string[] args)
{
    version (unittest)
    {
        if (args.length <= 1)
            return 0;
    }
    auto roots = defaultRoots();
    bool overlayGenerate = true;
    bool doUpgrade = true;
    bool help;
    bool doSync;
    bool doConsecutive;
    bool doAst;
    string diffVersion;
    bool doGenerate;
    bool doAllVersions;
    bool doClean;
    bool outputSet;
    bool adaptedSet;
    string outputArg;
    size_t window = defaultVersionWindow;
    string fromTag, toTag, rangeSpec;
    bool referenceSet;

    for (size_t i = 1; i < args.length; ++i)
    {
        auto a = args[i];
        string next()
        {
            if (i + 1 >= args.length)
                throw new Exception("missing value for " ~ a);
            return args[++i];
        }

        if (a == "-h" || a == "--help")
            help = true;
        else if (a == "--ldc-root")
            roots.ldc = abs(next());
        else if (a == "--reference")
        {
            roots.reference = next();
            referenceSet = true;
            if (parseTagSpec(roots.reference).length > 1)
                doAllVersions = true;
        }
        else if (a == "--latest")
        {
            roots.reference = "HEAD";
            referenceSet = true;
        }
        else if (a == "--adapted")
        {
            roots.adapted = abs(next());
            adaptedSet = true;
        }
        else if (a == "--target")
            roots.target = next();
        else if (a == "--output" || a == "--out-dir")
        {
            outputArg = next();
            outputSet = true;
        }
        else if (a == "--report")
            roots.reportMd = abs(next());
        else if (a == "--json")
            roots.reportJson = abs(next());
        else if (a == "--no-generate")
            overlayGenerate = false;
        else if (a == "--no-upgrade")
            doUpgrade = false;
        else if (a == "--sync-workspace" || a == "--prefetch-refs")
            doSync = true;
        else if (a == "--consecutive")
            doConsecutive = true;
        else if (a == "--adapt-ast")
            doAst = true;
        else if (a == "--diff-version")
        {
            diffVersion = next();
            referenceSet = true;
        }
        else if (a == "--generate" || a == "--iterate")
            doGenerate = true;
        else if (a == "--all-versions")
            doAllVersions = true;
        else if (a == "--clean")
            doClean = true;
        else if (a == "--window")
        {
            window = to!size_t(next());
            doAllVersions = true;
        }
        else if (a == "--from")
        {
            fromTag = next();
            doAllVersions = true;
        }
        else if (a == "--to")
        {
            toTag = next();
            doAllVersions = true;
        }
        else if (a == "--range")
        {
            rangeSpec = next();
            doAllVersions = true;
        }
        else
        {
            stderr.writeln("unknown flag: ", a);
            help = true;
        }
    }

    if (outputSet)
        roots.outDir = resolveOutput(roots.ldc, outputArg);

    immutable noAction = !doGenerate && !doClean && !doSync && !doAllVersions
        && !doConsecutive && !diffVersion.length && !adaptedSet;
    if (help || noAction)
    {
        writeln(usageHelp());
        return help && args.length > 2 ? 1 : 0;
    }

    if (doClean)
    {
        auto n = cleanCaches(roots.ldc);
        writeln("clean        removed ", n, " cache dir(s) under ", toolRoot(roots.ldc));
        if (!doSync && !doAllVersions && !doGenerate && !doConsecutive
            && !diffVersion.length && !roots.adapted.length && !referenceSet)
            return 0;
    }

    auto tags = selectedTags(roots.reference, referenceSet, fromTag, toTag,
        rangeSpec, window, doAllVersions);

    if (doSync)
    {
        auto prefetch = doAllVersions || fromTag.length || toTag.length
            || rangeSpec.length ? tags : tagWindow(window);
        foreach (tag; prefetch)
        {
            writeln("prefetch ", tag);
            auto p = materializeReference(roots.ldc, tag);
            writeln("  ", p);
        }
        if (!doConsecutive && !doAllVersions && !doGenerate && !diffVersion.length
            && !roots.adapted.length)
            return 0;
    }

    if (doConsecutive)
    {
        auto steps = runConsecutiveLadder(roots.ldc, tags.length > 1 ? tags : tagWindow(window));
        int ran, skip, bad;
        foreach (st; steps)
        {
            if (st.skipped)
            {
                ++skip;
                writeln("skip ", st.fromTag, "→", st.toTag, " ", st.skipReason);
                continue;
            }
            ++ran;
            writeln(st.fromTag, "→", st.toTag, " ", st.note);
            if (st.reconstructMismatches || st.astParseFail || !st.versionOk)
                ++bad;
        }
        writefln("consecutive ran=%s skip=%s mismatch-steps=%s", ran, skip, bad);
        return bad ? 4 : 0;
    }

    if (doAllVersions)
    {
        // Window/range never emits. Pull tags into untracked workspace/ or .work/.
        writeln("prefetch window  ", tags.length, " tag(s) ",
            tags.length ? tags[0] : "", "…", tags.length ? tags[$ - 1] : "",
            " (generate is at most one --reference TAG from this checkout)");
        foreach (tag; tags)
        {
            writeln("prefetch ", tag);
            writeln("  ", materializeReference(roots.ldc, tag));
        }
        if (!doGenerate && !diffVersion.length && !roots.adapted.length)
            return 0;
    }

    if (diffVersion.length || doGenerate)
    {
        auto tag = diffVersion.length ? diffVersion : roots.reference;
        if (tag == "latest" || !tag.length)
            tag = "HEAD";
        auto one = parseTagSpec(tag);
        if (one.length > 1)
        {
            stderr.writeln("error: generate accepts at most one tag; got ", tag);
            stderr.writeln("       use --prefetch-refs --range ", tag, " to cache only");
            return 2;
        }
        if (one.length == 1)
            tag = one[0];
        auto against = roots.target == roots.ldc ? "" : roots.target;
        if (!outputSet)
            roots.outDir = defaultOutputDir(roots.ldc);
        auto outD = roots.outDir;
        writeln("iterate      ", tag, " against ", against.length ? against : roots.ldc);
        auto rep = iterateVersion(roots.ldc, tag, against, outD);
        auto md = renderIterate(rep);
        mkdirRecurse(dirName(roots.reportMd));
        write(roots.reportMd, md);
        writeln(md);
        if (rep.emit.emitted)
            writeln("generated    ", outD, " files=", rep.emit.emitted);
        writeln("file-cmp     ", buildPath(outD, "FILE-CMP.md"));
        writeln("ast-diff     ", buildPath(outD, "AST-DIFF.md"),
            " ldcGaps=", rep.astDiff.ldcWithGaps,
            " missingFiles=", rep.astDiff.missingFiles);
        return (rep.equivalent && rep.versionOk) ? 0 : 5;
    }

    if (!roots.adapted.length || !exists(buildPath(roots.adapted, "object.d")))
    {
        stderr.writeln("error: --adapted DIR must be an overlay with object.d ");
        stderr.writeln("       (optional overlay reconstruct; not --generate)");
        return 2;
    }

    auto refRoot = materializeReference(roots.ldc, roots.reference);
    requireExists(buildPath(refRoot, "runtime", "druntime", "src", "object.d"),
        "reference druntime");
    auto tgtRoot = materializeTarget(roots.ldc, roots.target);

    auto cs = loadConstraints(toolRoot(roots.ldc));
    auto cons = matchConstraint(cs, roots.reference);
    writeln("ldc          ", roots.ldc);
    writeln("reference    ", refRoot, " (", roots.reference, ")");
    writeln("adapted      ", roots.adapted);
    writeln("target       ", tgtRoot);
    if (cons !is null)
        writeln("constraint   ", cons.id, " [", cons.referenceMin, " .. ", cons.referenceMax, "]");
    else
        writeln("constraint   (none recorded for ", roots.reference, ")");

    writeln("walking…");
    auto refFiles = walkLdcRuntime(refRoot);
    auto adFiles = walkMergedTree(roots.adapted);
    auto tgtFiles = walkLdcRuntime(tgtRoot);
    writeln("  reference ", refFiles.length);
    writeln("  adapted   ", adFiles.length);
    writeln("  target    ", tgtFiles.length);

    writeln("parsing + classifying (libdparse)…");
    auto inv = classifyTrees(refFiles, adFiles, tgtFiles);
    auto c = countKinds(inv.rows);
    writefln("  identical=%s adapted=%s stub-adapt=%s extra=%s missing=%s total=%s",
        c.identical, c.adapted, c.stubAdapt, c.extra, c.missing, inv.rows.length);

    UpgradeRow[] up;
    if (doUpgrade)
        up = upgradeMap(inv.rows);

    mkdirRecurse(dirName(roots.reportMd));
    write(roots.reportMd, renderMarkdown(inv, up, roots.reference, roots.adapted,
        tgtRoot, cons));
    write(roots.reportJson, renderJson(inv, up, cons));
    writeln("report      ", roots.reportMd);

    if (overlayGenerate)
    {
        writeln("generating  ", roots.outDir);
        auto st = generateTree(inv, roots.outDir,
            doAst ? AdaptMode.overlay : AdaptMode.verifyOnly, roots.reference);
        writeManifest(inv, buildPath(workDir(roots.ldc), "MANIFEST.md"),
            roots.reference, roots.adapted);
        writefln("  copy-reference %s  copy-adapted %s  omitted %s",
            st.copiedReference, st.copiedAdapted, st.omitted);
        auto vr = verifyAgainstAdapted(roots.outDir, roots.adapted);
        writefln("verify       compared=%s mismatches=%s missing=%s extra=%s",
            vr.compared, vr.mismatches, vr.missingInGenerated, vr.extraInGenerated);
        foreach (d; vr.details)
        {
            if (vr.mismatches + vr.missingInGenerated + vr.extraInGenerated > 8)
                break;
            writeln("  ", d);
        }
        if (vr.mismatches || vr.missingInGenerated)
            return 3;
    }
    return 0;
}

/// Tags implied by the generate / window / range flags.
string[] selectedTags(string reference, bool referenceSet, string fromTag, string toTag,
    string rangeSpec, size_t window, bool multi)
{
    if (rangeSpec.length)
        return parseTagSpec(rangeSpec);
    if (fromTag.length || toTag.length)
        return tagRange(fromTag, toTag);
    if (referenceSet)
    {
        auto spec = parseTagSpec(reference);
        if (spec.length > 1)
            return spec;
        if (multi)
            return tagWindow(window, spec.length ? spec[0] : latestMinorTag());
        return spec.length ? spec : [reference];
    }
    if (multi)
        return tagWindow(window);
    if (reference == "latest" || !reference.length)
        return [latestMinorTag()];
    return [reference];
}

unittest
{
    assert(selectedTags(latestMinorTag(), false, "", "", "", 12, false) == [latestMinorTag()]);
    assert(selectedTags("", false, "", "", "", 12, true) == tagWindow());
    assert(selectedTags("", false, "v1.40.0", "v1.42.0", "", 12, true)
        == ["v1.40.0", "v1.41.0", "v1.42.0"]);
    assert(selectedTags("", false, "", "", "v1.36.0..v1.38.0", 12, true)
        == ["v1.36.0", "v1.37.0", "v1.38.0"]);
    assert(selectedTags("v1.40.0", true, "", "", "", 3, true)
        == tagWindow(3, "v1.40.0"));
}

string usageHelp()
{
    auto latest = latestMinorTag();
    auto w = tagWindow();
    return format(q"HELP
runtime-adapt — this LDC checkout's stock runtime → complete LDC runtime.

Always run from the LDC repo root. Generate uses this tree (HEAD). At most
one extra tag may be named; ranges only fill the untracked workspace.

  dub build --root=tools/runtime-adapt --compiler=ldc2
  dub test  --root=tools/runtime-adapt --compiler=ldc2
  dub run   --root=tools/runtime-adapt --compiler=ldc2 -- [args]

  (no args / --help)    this help

Generate this checkout (at most one --reference TAG):

  --generate
  --generate --output tools/runtime-adapt/.work/generated
  --generate --reference v1.36.0

  --output DIR          write here (relative to the LDC repo root)
                        default: tools/runtime-adapt/.work/generated

Cache only (no emit; last %s minors %s … %s):

  --prefetch-refs
  --all-versions
  --all-versions --window 8
  --from v1.36.0 --to v1.42.0
  --range v1.36.0..v1.42.0
  --clean               delete .work/, workspace/{refs,stock}, bin/, clones/

Paths:
  --ldc-root DIR        this LDC checkout (auto-detected)
  --report / --json     summary reports

Overlay reconstruct (not the product): --adapted --adapt-ast --consecutive

Tracked on this branch: source/, tests/, README, EXTENDING.md.
Untracked: .work/, bin/, workspace/refs/, workspace/stock/, clones/.

When LDC adds an intrinsic, UDA, or ldc/*.d, see EXTENDING.md.
HELP", defaultVersionWindow, w.length ? w[0] : latest, w.length ? w[$ - 1] : latest);
}
