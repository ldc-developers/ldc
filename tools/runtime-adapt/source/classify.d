//===-- tools/runtime-adapt/source/classify.d ---------------------*- D -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

module classify;

import justify;
import kernel;
import parseutil;
import walk;

import std.algorithm : sort;
import std.array : array;
import std.digest : toHexString;
import std.digest.sha : sha1Of;
import std.file : readText;

enum Kind
{
    identical,  /// adapted == reference (newline-normalized)
    adapted,    /// both exist, different — overlay body
    stubAdapt,  /// adapted and looks shrunk/stubbed
    extra,      /// only in adapted
    missing,    /// only in reference — omitted kernel
}

string kindName(Kind k)
{
    final switch (k)
    {
    case Kind.identical: return "identical";
    case Kind.adapted: return "adapted";
    case Kind.stubAdapt: return "stub-adapt";
    case Kind.extra: return "extra";
    case Kind.missing: return "missing";
    }
}

/// Reconstruct action: produce the adapted tree *from* the reference.
enum Action
{
    copyReference, /// stock file from --reference (identical)
    copyAdapted,   /// overlay body
    omit,          /// not in the adapted tree
}

string actionName(Action a)
{
    final switch (a)
    {
    case Action.copyReference: return "copy-reference";
    case Action.copyAdapted: return "copy-adapted";
    case Action.omit: return "omit";
    }
}

struct FileRow
{
    string rel;
    Kind kind;
    Action action;
    KernelGroup group;
    string refPath;
    string adaptedPath;
    string targetPath;
    ulong refBytes;
    ulong adaptedBytes;
    bool targetSameAsRef;
    bool targetPresent;
    ParseFacts refParse;
    ParseFacts adaptedParse;
    string note;
    string prompt; /// LDC concept that prompted this decision
}

struct Inventory
{
    FileRow[] rows;
    string[] parseFailures;
}

private string digestOf(string text)
{
    return sha1Of(cast(const(ubyte)[]) text).toHexString.idup;
}

private string tryRead(string path)
{
    if (!path.length)
        return "";
    try
        return normalizeSource(readText(path));
    catch (Exception)
        return "";
}

Inventory classifyTrees(RelFile[] refFiles, RelFile[] adaptedFiles, RelFile[] targetFiles)
{
    Inventory inv;
    bool[string] seen;

    foreach (rf; refFiles)
    {
        seen[rf.rel] = true;
        FileRow row;
        row.rel = rf.rel;
        row.refPath = rf.abs;
        row.group = classifyPath(rf.rel);
        auto ad = findRel(adaptedFiles, rf.rel);
        auto tf = findRel(targetFiles, rf.rel);
        auto refText = tryRead(rf.abs);
        row.refBytes = refText.length;
        row.refParse = parseDSource(refText, rf.rel);
        if (tf !is null)
        {
            row.targetPath = tf.abs;
            row.targetPresent = true;
            row.targetSameAsRef = digestOf(refText) == digestOf(tryRead(tf.abs));
        }

        if (ad is null)
        {
            row.kind = Kind.missing;
            row.action = Action.omit;
            row.prompt = filePrompt(rf.rel, row.group);
            row.note = "not in adapted tree — omit so reconstruct stays exact";
        }
        else
        {
            row.adaptedPath = ad.abs;
            auto ours = tryRead(ad.abs);
            row.adaptedBytes = ours.length;
            row.adaptedParse = parseDSource(ours, ad.rel);
            if (digestOf(refText) == digestOf(ours))
            {
                row.kind = Kind.identical;
                row.action = Action.copyReference;
                row.prompt = filePrompt(rf.rel, row.group);
                row.note = "identical to reference after newline normalize — emit stock --reference body";
            }
            else
            {
                auto stub = looksStubbed(ours, row.refBytes, row.adaptedBytes);
                row.kind = stub ? Kind.stubAdapt : Kind.adapted;
                row.action = Action.copyAdapted;
                row.prompt = filePrompt(rf.rel, row.group);
                if (looksAdapted(row.adaptedParse, row.refParse))
                    row.note = "custom CRuntime_* / WebAssembly overlay — keep adapted body";
                else if (stub)
                    row.note = "shrunk or stub-marked vs reference — keep adapted body";
                else
                    row.note = "differs from reference — keep adapted body";
            }
        }
        if (row.refPath.length && !row.refParse.parsed)
            inv.parseFailures ~= rf.rel ~ " (reference): " ~ row.refParse.error;
        if (row.adaptedPath.length && !row.adaptedParse.parsed)
            inv.parseFailures ~= rf.rel ~ " (adapted): " ~ row.adaptedParse.error;
        inv.rows ~= row;
    }

    foreach (af; adaptedFiles)
    {
        if (af.rel in seen)
            continue;
        FileRow row;
        row.rel = af.rel;
        row.adaptedPath = af.abs;
        row.group = classifyPath(af.rel);
        auto ours = tryRead(af.abs);
        row.adaptedBytes = ours.length;
        row.adaptedParse = parseDSource(ours, af.rel);
        row.kind = Kind.extra;
        row.action = Action.copyAdapted;
        row.prompt = filePrompt(af.rel, row.group);
        row.note = "adapted-only (not in reference) — copy verbatim";
        auto tf = findRel(targetFiles, af.rel);
        if (tf !is null)
        {
            row.targetPath = tf.abs;
            row.targetPresent = true;
        }
        inv.rows ~= row;
    }
    inv.rows.sort!((a, b) => a.rel < b.rel);
    return inv;
}

struct Counts
{
    int identical, adapted, stubAdapt, extra, missing;
}

Counts countKinds(const FileRow[] rows)
{
    Counts c;
    foreach (r; rows)
    {
        final switch (r.kind)
        {
        case Kind.identical: c.identical++; break;
        case Kind.adapted: c.adapted++; break;
        case Kind.stubAdapt: c.stubAdapt++; break;
        case Kind.extra: c.extra++; break;
        case Kind.missing: c.missing++; break;
        }
    }
    return c;
}

struct UpgradeRow
{
    string rel;
    string verdict;
    string note;
}

/// How to move the overlay from --reference toward --target (newer LDC).
UpgradeRow[] upgradeMap(const FileRow[] rows)
{
    UpgradeRow[] outp;
    foreach (r; rows)
    {
        UpgradeRow u;
        u.rel = r.rel;
        if (r.kind == Kind.missing)
        {
            u.verdict = r.targetPresent ? "still-omit" : "drop";
            u.note = r.prompt;
        }
        else if (r.kind == Kind.extra)
        {
            u.verdict = "keep-extra";
            u.note = r.prompt;
        }
        else if (r.kind == Kind.identical)
        {
            u.verdict = r.targetPresent ? "rebase-from-target" : "keep";
            u.note = r.targetSameAsRef
                ? "still identical at target"
                : "was stock at reference; take newer LDC body";
        }
        else
        {
            u.verdict = r.targetPresent ? "reapply-adapt" : "keep-adapted";
            u.note = "replay overlay edits onto target; " ~ r.prompt;
        }
        outp ~= u;
    }
    return outp;
}
