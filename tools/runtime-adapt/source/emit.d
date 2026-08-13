//===-- tools/runtime-adapt/source/emit.d -------------------------*- D -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Stock druntime/phobos is the emit source. ldc/*.di is generated from
// LDC compiler sources (driver/, gen/) at the workspace tag. The LDC
// runtime is the goal only — never copied.
//
//===----------------------------------------------------------------------===//

module emit;

import adapt;
import compilerparse;
import kernel;
import ldcmods;
import parseutil;
import principles;
import walk;

import std.array : appender, split;
import std.file : exists, mkdirRecurse, readText, write, rmdirRecurse;
import std.format : format;
import std.path : buildPath, dirName, extension;

struct FileEmit
{
    string rel;
    EmitAction action;
    string principleId;
    bool fromReference;
    bool fromLdc;
    bool equalTexts;
    bool parsedOut;
    bool passthrough;
    string[] astRules;
    string error;
}

struct EmitReport
{
    FileEmit[] files;
    int emitted;
    int omitted;
    int parsedOk;
    int parseFail;
    int rulesFired;
    int rulesResolved;
    int unaccounted;
    int sourcedFromRef;
    int sourcedFromCompiler;
    int refProduct;
    int ldcOnly;
    bool productComplete;
    bool validated;
}

/// Stock source that libdparse cannot parse is still valid output.
private bool acceptParserGap(ref FileEmit fe, string body, string original, ParseFacts chk)
{
    if (chk.parsed)
        return false;
    if (body != original)
        return false;
    fe.parsedOut = true;
    fe.error = "libdparse-gap: " ~ chk.error;
    return true;
}

private string tryRead(string p)
{
    if (!p.length || !exists(p))
        return "";
    try
        return normalizeSource(readText(p));
    catch (Exception)
        return "";
}

/// Which compiler hooks already appear as identifiers in a file set (guide).
private string[] hooksNamedIn(RelFile[] files, const string[] hooks)
{
    import std.algorithm : canFind, sort, uniq;
    import std.array : array;
    bool[string] hit;
    foreach (f; files)
    {
        auto t = tryRead(f.abs);
        if (!t.length)
            continue;
        foreach (h; hooks)
            if (h.length && !(h in hit) && t.canFind(h))
                hit[h] = true;
    }
    return hit.keys.sort.uniq.array;
}

private bool isDModule(string rel)
{
    auto ext = extension(rel);
    return ext == ".d" || ext == ".di";
}

private void writeEmitted(ref EmitReport rep, ref FileEmit fe, string outDir, string body)
{
    auto dest = buildPath(outDir ~ fe.rel.split("/"));
    mkdirRecurse(dirName(dest));
    write(dest, body);
    rep.emitted++;
    if (fe.parsedOut)
        rep.parsedOk++;
    else
        rep.parseFail++;
    rep.files ~= fe;
}

private void accountRule(ref EmitReport rep, bool hasPrinciple, bool resolved)
{
    if (hasPrinciple)
    {
        rep.rulesFired++;
        if (resolved)
            rep.rulesResolved++;
    }
    else
        rep.unaccounted++;
}

/// Emit one reference file at its version-faithful path. Body is always
/// the reference (adapted). `ldcTxt` is comparison-only.
private void emitReferenceFile(ref EmitReport rep, string outDir, string tag,
    string destRel, string refTxt, string ldcTxt, string principleRel,
    KernelGroup g, EmitProfile profile, const(CompilerModel)* model)
{
    auto refF = parseDSource(refTxt, principleRel);
    auto ldcF = ldcTxt.length ? parseDSource(ldcTxt, destRel) : ParseFacts.init;
    auto p = selectPrinciple(principleRel, g, refF, ldcF, profile);
    EmitAction act = p is null ? EmitAction.takeReference : p.action;

    FileEmit fe;
    fe.rel = destRel;
    fe.fromReference = true;
    fe.fromLdc = ldcTxt.length > 0;
    fe.equalTexts = ldcTxt.length && refTxt == ldcTxt;
    fe.principleId = p is null ? "take.reference" : p.id;
    fe.action = act;
    fe.passthrough = !isDModule(destRel);

    if (profile == EmitProfile.native && act == EmitAction.omit)
    {
        act = EmitAction.takeReference;
        fe.action = act;
        fe.principleId = "take.reference";
    }

    if (profile == EmitProfile.strip && act == EmitAction.omit)
    {
        rep.omitted++;
        accountRule(rep, fe.principleId.length > 0, true);
        rep.files ~= fe;
        return;
    }

    // Overlay stubs are opt-in (strip). Native product is the reference body.
    if (act == EmitAction.astOverlay && profile == EmitProfile.native)
    {
        act = EmitAction.takeReference;
        fe.action = act;
        fe.principleId = "take.reference";
    }

    // Never take an LDC body for a file that exists in the reference.
    if (act == EmitAction.takeLdc)
    {
        act = EmitAction.takeReference;
        fe.action = act;
        fe.principleId = "take.reference";
    }

    rep.sourcedFromRef++;

    string body = refTxt;
    if (fe.passthrough)
    {
        fe.parsedOut = true;
        accountRule(rep, fe.principleId.length > 0, true);
        writeEmitted(rep, fe, outDir, body);
        return;
    }

    const original = body;
    AdaptMode mode = AdaptMode.verifyOnly;
    if (act == EmitAction.astOverlay)
        mode = AdaptMode.overlay;
    else if (profile == EmitProfile.native && model !is null && model.ok)
        mode = AdaptMode.fromCompiler;
    auto ad = adaptSource(body, destRel, tag, mode, model);
    body = ad.output;
    fe.astRules = ad.applied;
    fe.parsedOut = ad.parsedOut;
    if (!ad.parsedOut)
    {
        if (!acceptParserGap(fe, body, original, parseDSource(body, destRel)))
            fe.error = ad.error;
    }
    if (ad.applied.length)
    {
        rep.rulesFired += cast(int) ad.applied.length;
        if (fe.parsedOut)
            rep.rulesResolved += cast(int) ad.applied.length;
    }
    else
        accountRule(rep, fe.principleId.length > 0, fe.parsedOut);

    writeEmitted(rep, fe, outDir, body);
}

/// Emit every stock reference file, then ldc/*.di from compiler sources.
/// `goalFiles` is comparison only — LDC runtime bodies are never copied.
EmitReport emitRuntime(RelFile[] refFiles, RelFile[] goalFiles, string outDir,
    string tag, EmitProfile profile = EmitProfile.native, string guideRoot = "")
{
    if (exists(outDir))
        rmdirRecurse(outDir);
    mkdirRecurse(outDir);

    EmitReport rep;
    bool[string] done;
    rep.refProduct = cast(int) refFiles.length;

    CompilerModel model;
    const(CompilerModel)* modelPtr;
    if (guideRoot.length)
    {
        model = parseCompiler(guideRoot);
        if (model.ok)
        {
            model.hooksPresentInStock = hooksNamedIn(refFiles, model.runtimeHooks);
            model.hooksPresentInGoal = hooksNamedIn(goalFiles, model.runtimeHooks);
            modelPtr = &model;
        }
    }

    foreach (rf; refFiles)
    {
        auto destRel = rf.rel;
        auto refTxt = tryRead(rf.abs);
        auto mapped = remapRel(rf.rel);
        auto lf = findRel(goalFiles, destRel);
        if (lf is null && mapped != destRel)
            lf = findRel(goalFiles, mapped);
        auto goalTxt = lf is null ? "" : tryRead(lf.abs);
        auto g = classifyPath(rf.rel);
        emitReferenceFile(rep, outDir, tag, destRel, refTxt, goalTxt, rf.rel, g, profile, modelPtr);
        done[destRel] = true;
        done[mapped] = true;
    }

    if (profile == EmitProfile.native && modelPtr !is null)
    {
        foreach (di; generateImpliedModules(model, done, tag))
        {
            FileEmit fe;
            fe.rel = di.rel;
            fe.action = EmitAction.generateFromCompiler;
            fe.principleId = di.principleId;
            fe.astRules = di.corrections;
            auto ad = adaptSource(di.body, di.rel, tag, AdaptMode.verifyOnly, modelPtr);
            auto body = ad.output;
            fe.parsedOut = ad.parsedOut;
            if (!ad.parsedOut)
            {
                if (!acceptParserGap(fe, body, di.body, parseDSource(body, di.rel)))
                    fe.error = ad.error;
            }
            accountRule(rep, true, fe.parsedOut);
            writeEmitted(rep, fe, outDir, body);
            rep.sourcedFromCompiler++;
            done[di.rel] = true;
        }
        foreach (di; generateLdcDi(guideRoot, tag))
        {
            FileEmit fe;
            fe.rel = di.rel;
            fe.action = EmitAction.generateFromCompiler;
            fe.principleId = di.principleId;
            fe.passthrough = false;
            fe.astRules = di.corrections;
            auto ad = adaptSource(di.body, di.rel, tag, AdaptMode.verifyOnly);
            auto body = ad.output;
            fe.parsedOut = ad.parsedOut;
            if (!ad.parsedOut)
            {
                if (!acceptParserGap(fe, body, di.body, parseDSource(body, di.rel)))
                    fe.error = ad.error;
            }
            accountRule(rep, true, fe.parsedOut);
            writeEmitted(rep, fe, outDir, body);
            rep.sourcedFromCompiler++;
            done[di.rel] = true;
        }
    }

    foreach (lf; goalFiles)
    {
        if (lf.rel in done || inverseRemap(lf.rel) in done)
            continue;
        FileEmit fe;
        fe.rel = lf.rel;
        fe.action = EmitAction.omit;
        fe.principleId = "omit.goal-only";
        fe.fromLdc = true;
        // Goal-only: implement in source (see ldcmods.d). Never copy LDC runtime.
        rep.ldcOnly++;
        rep.files ~= fe;
    }

    const want = profile == EmitProfile.strip
        ? (rep.refProduct - rep.omitted)
        : (rep.refProduct + rep.sourcedFromCompiler);
    rep.productComplete = rep.emitted == want && rep.parseFail == 0
        && (profile != EmitProfile.native || rep.omitted == 0);
    rep.validated = rep.parseFail == 0 && rep.unaccounted == 0
        && rep.rulesFired == rep.rulesResolved && rep.productComplete;
    return rep;
}

string renderEmit(const EmitReport rep)
{
    auto buf = appender!string();
    buf.put("# emit runtime\n\n");
    buf.put(format("- emitted: %s\n- omitted: %s\n- parsedOk: %s\n- parseFail: %s\n",
        rep.emitted, rep.omitted, rep.parsedOk, rep.parseFail));
    buf.put(format("- refProduct: %s\n- sourcedFromRef: %s\n- sourcedFromCompiler: %s\n- ldcOnly: %s\n- productComplete: %s\n",
        rep.refProduct, rep.sourcedFromRef, rep.sourcedFromCompiler, rep.ldcOnly, rep.productComplete));
    buf.put(format("- rulesFired: %s\n- rulesResolved: %s\n- unaccounted: %s\n- validated: %s\n\n",
        rep.rulesFired, rep.rulesResolved, rep.unaccounted, rep.validated));
    if (rep.parseFail || rep.unaccounted || !rep.productComplete)
    {
        buf.put("## Unresolved\n\n");
        foreach (f; rep.files)
        {
            if (f.action == EmitAction.omit)
                continue;
            if (f.parsedOut && f.principleId.length)
                continue;
            if (!f.parsedOut)
                buf.put(format("- PARSE `%s` (%s) %s\n", f.rel, f.principleId, f.error));
            else if (!f.principleId.length)
                buf.put(format("- NO-RULE `%s` action=%s\n", f.rel, f.action));
        }
    }
    return buf.data;
}
