//===-- tools/runtime-adapt/source/generate.d ---------------------*- D -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Reconstruct an adapted druntime/phobos tree *from* --reference:
//   identical  → emit the reference (stock) body
//   adapted    → emit the overlay body
//   extra      → emit the overlay-only file
//   missing    → omit
// Sidecars (MANIFEST, JS stubs) go to metaDir so the D tree can match the
// overlay file-for-file after newline normalize.
//
//===----------------------------------------------------------------------===//

module generate;

import adapt;
import classify;
import parseutil;

import std.array : appender, replace, split;
import std.file : exists, mkdirRecurse, readText, write;
import std.format : format;
import std.path : buildPath, dirName;

struct GenerateStats
{
    int copiedReference;
    int copiedAdapted;
    int omitted;
    int astApplied;
    int astParseFail;
}

GenerateStats generateTree(const Inventory inv, string outDir,
    AdaptMode mode = AdaptMode.verifyOnly, string referenceTag = "")
{
    GenerateStats st;
    mkdirRecurse(outDir);

    foreach (r; inv.rows)
    {
        if (r.action == Action.omit)
        {
            st.omitted++;
            continue;
        }
        string src;
        if (r.action == Action.copyReference)
        {
            src = r.refPath;
            st.copiedReference++;
        }
        else
        {
            src = r.adaptedPath;
            st.copiedAdapted++;
        }
        if (!src.length || !exists(src))
        {
            st.copiedReference -= r.action == Action.copyReference ? 1 : 0;
            st.copiedAdapted -= r.action == Action.copyAdapted ? 1 : 0;
            continue;
        }
        auto dest = buildPath(outDir ~ r.rel.split("/"));
        mkdirRecurse(dirName(dest));
        auto text = normalizeSource(readText(src));
        if (mode == AdaptMode.overlay && r.action == Action.copyReference)
        {
            auto ad = adaptSource(text, r.rel, referenceTag, AdaptMode.overlay);
            text = ad.output;
            if (ad.applied.length)
                st.astApplied++;
            if (!ad.parsedOut)
                st.astParseFail++;
        }
        else
        {
            auto chk = parseDSource(text, r.rel);
            if (!chk.parsed)
                st.astParseFail++;
        }
        write(dest, text);
    }
    return st;
}

void writeManifest(const Inventory inv, string dest, string reference, string adapted)
{
    auto buf = appender!string();
    buf.put("# Reconstructed adapted runtime\n\n");
    buf.put(format("- reference: `%s`\n- adapted: `%s`\n\n", reference, adapted));
    buf.put("| path | action | kind | group | prompt |\n|---|---|---|---|---|\n");
    foreach (r; inv.rows)
    {
        buf.put(format("| `%s` | %s | %s | %s | %s |\n",
            r.rel, actionName(r.action), kindName(r.kind),
            groupNameOf(r), r.prompt.replace("|", "/")));
    }
    mkdirRecurse(dirName(dest));
    write(dest, buf.data);
}

private string groupNameOf(const FileRow r)
{
    import kernel : groupName;
    return groupName(r.group);
}

/// Normalized-text compare of generated D tree vs the overlay.
struct VerifyResult
{
    int compared;
    int mismatches;
    int missingInGenerated;
    int extraInGenerated;
    string[] details;
}

VerifyResult verifyAgainstAdapted(string generatedDir, string adaptedDir)
{
    import walk : RelFile, walkMergedTree;

    VerifyResult vr;
    auto want = walkMergedTree(adaptedDir);
    auto got = walkMergedTree(generatedDir);
    bool[string] seen;
    foreach (w; want)
    {
        seen[w.rel] = true;
        vr.compared++;
        RelFile* g;
        foreach (ref x; got)
            if (x.rel == w.rel)
            {
                g = &x;
                break;
            }
        if (g is null)
        {
            vr.missingInGenerated++;
            vr.details ~= "missing " ~ w.rel;
            continue;
        }
        if (normalizeSource(readText(w.abs)) != normalizeSource(readText(g.abs)))
        {
            vr.mismatches++;
            vr.details ~= "content " ~ w.rel;
        }
    }
    foreach (g; got)
        if (g.rel !in seen)
        {
            vr.extraInGenerated++;
            vr.details ~= "extra " ~ g.rel;
        }
    return vr;
}
