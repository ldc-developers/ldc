//===-- tools/runtime-adapt/source/adapt.d ------------------------*- D -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// AST-driven adjustments. libdparse (serve-d) builds the module AST from the
// *reference* source; we splice the original text at AST/token spans so the
// result is still valid D, then re-parse to prove it.
//
//===----------------------------------------------------------------------===//

module adapt;

import dparse.ast;
import dparse.lexer;
import dparse.parser : parseModule;
import dparse.rollback_allocator : RollbackAllocator;

import compilerparse;
import parseutil;
import versions;

import std.algorithm : canFind, sort;
import std.array : appender, join;

enum AdaptMode
{
    verifyOnly,    /// parse only; emit original (newline-normalized)
    overlay,       /// apply CRT / hook / import splices (strip overlay)
    fromCompiler,  /// generic stock → LDC using a parsed CompilerModel
}

struct AdaptResult
{
    bool ok;
    bool parsedIn;
    bool parsedOut;
    string output;
    string error;
    string[] applied; /// rule ids that fired
    ParseFacts inFacts;
    ParseFacts outFacts;
}

struct Splice
{
    size_t start;
    size_t end;
    string insert;
}

/// Apply AST-informed splices to one reference (or overlay) source file.
AdaptResult adaptSource(string source, string rel, string referenceTag,
    AdaptMode mode, const(CompilerModel)* model = null)
{
    AdaptResult r;
    auto src = normalizeSource(source);
    auto inn = parseDSource(src, rel);
    r.parsedIn = inn.parsed;
    r.inFacts = inn;
    if (!inn.parsed)
    {
        r.error = "input parse: " ~ inn.error;
        r.output = src;
        return r;
    }

    if (mode == AdaptMode.verifyOnly)
    {
        r.output = src;
        r.ok = true;
        r.parsedOut = true;
        r.outFacts = inn;
        return r;
    }

    if (mode == AdaptMode.fromCompiler)
    {
        return adaptFromCompiler(src, rel, referenceTag, inn, model);
    }

    LexerConfig config;
    config.fileName = rel;
    config.stringBehavior = StringBehavior.source;
    auto cache = StringCache(StringCache.defaultBucketCount);
    auto tokens = getTokensForParser(cast(ubyte[]) src, config, &cache);
    string lastErr;
    void msg(string, size_t, size_t, string m, bool isError)
    {
        if (isError)
            lastErr = m;
    }
    RollbackAllocator rba;
    uint errors;
    auto mod = parseModule(tokens, rel, &rba, &msg, &errors);
    if (errors || mod is null)
    {
        r.error = lastErr.length ? lastErr : "re-lex parse failed";
        r.output = src;
        return r;
    }

    auto vis = new AdaptVisitor(tokens);
    vis.visit(mod);

    Splice[] splices;
    if (rel == "object.d" || inn.moduleName == "object")
    {
        auto at = afterModuleDecl(tokens);
        if (at != size_t.max && !src.canFind("CRuntime_OVERLAY"))
        {
            splices ~= Splice(at, at,
                "\n\n// runtime-adapt (" ~ referenceTag ~ "): CRT gate from DtoThrow/WASI predefs\n"
                ~ "version (CRuntime_WASI) {}\n"
                ~ "else version (CRuntime_OVERLAY) {}\n"
                ~ "else static assert(0, \"runtime-adapt: CRuntime_WASI or CRuntime_OVERLAY required\");\n");
            r.applied ~= "object.crt-gate";
        }
    }

    foreach (fn; vis.functions)
    {
        if (fn.bodyStart == size_t.max || fn.bodyEnd == size_t.max)
            continue;
        if (fn.name == "_d_allocmemory" || fn.name == "_d_allocmemoryT")
        {
            splices ~= Splice(fn.bodyStart, fn.bodyEnd,
                "\n    {\n        // runtime-adapt " ~ referenceTag
                ~ ": stock body is GC.malloc (rt/lifetime.d). Overlay bump.\n"
                ~ "        return null;\n    }");
            r.applied ~= "hook." ~ fn.name;
        }
        else if (fn.name == "_d_throw_exception")
        {
            splices ~= Splice(fn.bodyStart, fn.bodyEnd,
                "\n    {\n        // runtime-adapt " ~ referenceTag
                ~ ": DtoThrow always calls this (gen/llvmhelpers.cpp).\n"
                ~ "        assert(0, \"runtime-adapt: _d_throw_exception\");\n    }");
            r.applied ~= "hook._d_throw_exception";
        }
    }

    foreach (imp; vis.imports)
    {
        if (!shouldDropImport(imp.moduleName))
            continue;
        splices ~= Splice(imp.start, imp.end, "/* runtime-adapt drop import "
            ~ imp.moduleName ~ " */");
        r.applied ~= "drop-import." ~ imp.moduleName;
    }

    r.output = applySplices(src, splices);
    auto outp = parseDSource(r.output, rel);
    r.parsedOut = outp.parsed;
    r.outFacts = outp;
    r.ok = outp.parsed;
    if (!outp.parsed)
        r.error = "output parse: " ~ outp.error;
    return r;
}

/// Stock → LDC: versions the compiler predefines, hooks it calls.
private AdaptResult adaptFromCompiler(string src, string rel, string tag,
    ParseFacts inn, const(CompilerModel)* model)
{
    AdaptResult r;
    r.parsedIn = true;
    r.inFacts = inn;
    auto body = src;

    // LDC predefines "LDC", not "DigitalMars" (dmd/target.d's DigitalMars
    // add is version (IN_LLVM) skipped; driver/main.cpp never adds it).
    // Assigning version = DigitalMars would make std.compiler.Vendor
    // digitalMars and fire DMD-only workarounds the goal tree leaves false.

    if (model !is null)
    {
        foreach (hook; model.runtimeHooks)
        {
            if (hookHome(hook) != rel)
                continue;
            if (body.canFind(hook))
                continue;
            // Only splice a stub when the goal tree has the symbol and the
            // stock file does not. Compiler-only createFwdDecl names stay out
            // (they are not D source in LDC's runtime).
            immutable inStock = model.hooksPresentInStock.canFind(hook);
            immutable inGoal = model.hooksPresentInGoal.canFind(hook);
            if (inStock)
                continue;
            if (model.hooksPresentInGoal.length && !inGoal)
                continue;
            body ~= "\n\n// runtime-adapt " ~ tag ~ ": compiler getRuntimeFunction\n"
                ~ "extern (C) void " ~ hook ~ "();\n";
            r.applied ~= "hook-decl." ~ hook;
        }
    }

    r.output = body;
    auto outp = parseDSource(body, rel);
    r.parsedOut = outp.parsed;
    r.outFacts = outp;
    r.ok = outp.parsed;
    if (!outp.parsed)
        r.error = "output parse: " ~ outp.error;
    return r;
}



private struct FnSpan
{
    string name;
    size_t bodyStart = size_t.max;
    size_t bodyEnd = size_t.max;
}

private struct ImpSpan
{
    string moduleName;
    size_t start;
    size_t end;
}

private final class AdaptVisitor : ASTVisitor
{
    alias visit = ASTVisitor.visit;
    const(Token)[] tokens;
    FnSpan[] functions;
    ImpSpan[] imports;

    this(const(Token)[] t)
    {
        tokens = t;
    }

    override void visit(const ImportDeclaration imp)
    {
        foreach (si; imp.singleImports)
        {
            ImpSpan s;
            s.moduleName = chainName(si.identifierChain);
            if (imp.startIndex || imp.endIndex)
            {
                s.start = imp.startIndex;
                s.end = imp.endIndex;
            }
            else
            {
                s.start = firstIdentIndex(si.identifierChain);
                s.end = s.start;
            }
            if (s.moduleName.length)
                imports ~= s;
        }
        imp.accept(this);
    }

    override void visit(const FunctionDeclaration fn)
    {
        FnSpan s;
        s.name = fn.name.text;
        braceSpanAfterName(s.name, s.bodyStart, s.bodyEnd);
        functions ~= s;
        fn.accept(this);
    }

    private void braceSpanAfterName(string name, ref size_t a, ref size_t b)
    {
        bool seenName;
        int depth;
        foreach (t; tokens)
        {
            if (!seenName)
            {
                if (t.type == tok!"identifier" && t.text == name)
                    seenName = true;
                continue;
            }
            if (t.type == tok!";" && depth == 0 && a == size_t.max)
                return;
            if (t.type == tok!"{")
            {
                if (depth == 0)
                    a = t.index;
                ++depth;
            }
            else if (t.type == tok!"}")
            {
                --depth;
                if (depth == 0 && a != size_t.max)
                {
                    auto closeLen = t.text.length ? t.text.length : 1;
                    b = t.index + closeLen;
                    return;
                }
            }
        }
    }
}

private string chainName(const IdentifierChain chain)
{
    if (chain is null)
        return "";
    string[] p;
    foreach (t; chain.identifiers)
        if (t.text.length)
            p ~= t.text;
    return p.join(".");
}

private size_t firstIdentIndex(const IdentifierChain chain)
{
    if (chain is null || !chain.identifiers.length)
        return 0;
    return chain.identifiers[0].index;
}

private size_t afterModuleDecl(const(Token)[] tokens)
{
    bool sawModule;
    foreach (t; tokens)
    {
        if (t.type == tok!"module")
            sawModule = true;
        else if (sawModule && t.type == tok!";")
            return t.index + 1;
    }
    return size_t.max;
}

private bool shouldDropImport(string modName)
{
    if (!modName.length)
        return false;
    static immutable drop = [
        "core.thread", "core.sync", "rt.minfo", "rt.dmain2", "rt.sections",
        "rt.sections_elf_shared", "rt.sections_win64", "rt.sections_osx",
        "core.runtime", "rt.deh", "rt.dwarfeh",
    ];
    foreach (d; drop)
        if (modName == d || (modName.length > d.length && modName[0 .. d.length] == d
                && modName[d.length] == '.'))
            return true;
    return false;
}

private string applySplices(string src, Splice[] splices)
{
    if (!splices.length)
        return src;
    splices.sort!((a, b) => a.start > b.start);
    string s = src;
    foreach (sp; splices)
    {
        if (sp.start > s.length || sp.end > s.length || sp.end < sp.start)
            continue;
        s = s[0 .. sp.start] ~ sp.insert ~ s[sp.end .. $];
    }
    return s;
}

/// Versioning check: constraint for `tag` must not apply to the next minor.
bool versioningHolds(string tag)
{
    auto i = size_t.max;
    foreach (k, t; consecutiveTags)
        if (t == tag)
        {
            i = k;
            break;
        }
    auto c = matchConstraint(minorLadder, tag);
    if (c is null)
        return false;
    if (i == size_t.max || i + 1 >= consecutiveTags.length)
        return c.reference == tag;
    return !appliesTo(*c, consecutiveTags[i + 1]) && appliesTo(*c, tag);
}
