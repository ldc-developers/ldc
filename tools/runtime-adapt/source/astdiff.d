//===-- tools/runtime-adapt/source/astdiff.d ----------------------*- D -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Full-file comparison of generated druntime/phobos vs the version's LDC
// goal tree, plus a conservative declaration-surface AST for ldc/*.
// Stock files that match after newline normalize are not scanned.
//
//===----------------------------------------------------------------------===//

module astdiff;

public import filecmp;

import dparse.lexer;

import parseutil;
import walk;

import std.algorithm : canFind, sort, uniq;
import std.array : appender, array, join, replace, split;
import std.file : exists, readText;
import std.format : format;
import std.path : extension;

struct AstShape
{
    bool parsed;
    string error;
    string moduleName;
    string[] symbols;   /// types + module-level enums/immutables
    string[] functions;
    string[] types;
    string[] fields;    /// Struct.field
    string[] pragmas;   /// LDC_intrinsic:llvm.xxx or pragma ident
    string[] imports;
}

struct FileAstDiff
{
    string rel;
    string section; /// ldc or stock
    bool missingFile;
    bool extraFile;
    bool parseGoal;
    bool parseGen;
    string[] missingSymbols;
    string[] extraSymbols;
    string[] missingFields;
    string[] missingPragmas;
    string moduleMismatch;
    bool textsEqual;
    string[] notes;
}

struct VersionAstReport
{
    string tag;
    string generatedDir;
    string goalRoot;
    string stockRoot;
    int compared;
    int missingFiles;
    int extraFiles;
    int ldcFiles;
    int ldcWithGaps;
    int stockWithGaps;
    int parseGaps;
    int textsEqual;
    int present;
    int textDiffs;
    int adaptDeltas;
    int missedLdcPatches;
    FileAstDiff[] files;
    FileCmp[] cmps;
}

AstShape extractAst(string source, string fileName)
{
    AstShape s;
    auto src = normalizeSource(source);
    LexerConfig config;
    config.fileName = fileName;
    config.stringBehavior = StringBehavior.source;
    auto cache = StringCache(StringCache.defaultBucketCount);
    auto tokens = getTokensForParser(cast(ubyte[]) src, config, &cache);
    collectFromTokens(s, tokens);
    s.parsed = s.moduleName.length > 0 || s.types.length > 0 || tokens.length > 0;

    import std.regex : ctRegex, matchAll;
    auto pre = ctRegex!`pragma\s*\(\s*LDC_intrinsic\s*,\s*"([^"]+)"`;
    foreach (cap; matchAll(src, pre))
    {
        auto p = "LDC_intrinsic:" ~ cap[1];
        if (!s.pragmas.canFind(p))
            s.pragmas ~= p;
    }
    auto p2 = ctRegex!`pragma\s*\(\s*(LDC_[A-Za-z0-9_]+)`;
    foreach (cap; matchAll(src, p2))
    {
        if (!s.pragmas.canFind(cap[1]))
            s.pragmas ~= cap[1];
    }
    s.functions = s.functions.sort.uniq.array;
    s.types = s.types.sort.uniq.array;
    s.fields = s.fields.sort.uniq.array;
    s.pragmas = s.pragmas.sort.uniq.array;
    s.imports = s.imports.sort.uniq.array;
    s.symbols = (s.types ~ s.symbols).sort.uniq.array;
    return s;
}

private bool isStringy(const Token t)
{
    if (t.type == tok!"stringLiteral" || t.type == tok!"wstringLiteral"
        || t.type == tok!"dstringLiteral")
        return true;
    auto tx = t.text;
    if (!tx.length)
        return false;
    auto c0 = tx[0];
    return c0 == '"' || c0 == '`' || c0 == '\''
        || (tx.length >= 2 && (tx[0 .. 2] == "q{" || tx[0 .. 2] == "q\""
            || tx[0 .. 2] == "r\"" || tx[0 .. 2] == "x\""));
}

private void collectFromTokens(ref AstShape s, const(Token)[] tokens)
{
    string currentType;
    int brace;
    int skipTo = -1;
    bool afterModule, afterImport, afterTypeKw;
    bool sawUnittest, sawVersion;
    for (size_t i; i < tokens.length; ++i)
    {
        auto t = tokens[i];
        if (isStringy(t))
            continue;
        if (t.type == tok!"{")
        {
            ++brace;
            if (sawUnittest && skipTo < 0)
                skipTo = brace - 1;
            afterTypeKw = false;
            sawUnittest = false;
            continue;
        }
        if (t.type == tok!"}")
        {
            if (brace > 0)
                --brace;
            if (skipTo >= 0 && brace == skipTo)
                skipTo = -1;
            if (brace == 0)
                currentType = "";
            continue;
        }
        if (skipTo >= 0)
            continue;
        if (t.type == tok!"unittest")
        {
            sawUnittest = true;
            continue;
        }
        if (t.type == tok!"version")
        {
            sawVersion = true;
            continue;
        }
        if (sawVersion && t.type == tok!"identifier" && t.text == "unittest")
        {
            sawUnittest = true;
            sawVersion = false;
            continue;
        }
        if (t.type == tok!"module")
        {
            afterModule = true;
            sawVersion = false;
            continue;
        }
        if (t.type == tok!"import")
        {
            afterImport = true;
            sawVersion = false;
            continue;
        }
        if (t.type == tok!";")
        {
            afterModule = afterImport = afterTypeKw = sawVersion = false;
            continue;
        }
        if (t.type == tok!"struct" || t.type == tok!"class" || t.type == tok!"enum"
            || t.type == tok!"union" || t.type == tok!"template" || t.type == tok!"interface")
        {
            afterTypeKw = true;
            sawVersion = false;
            continue;
        }
        if (t.type == tok!"immutable" || t.type == tok!"const" || t.type == tok!"alias")
        {
            afterTypeKw = false;
            sawVersion = false;
            continue;
        }
        if (t.type != tok!"identifier" || !okIdent(t.text))
        {
            if (t.type != tok!"identifier")
                sawVersion = false;
            continue;
        }
        if (afterModule)
        {
            s.moduleName ~= s.moduleName.length ? "." ~ t.text : t.text;
            continue;
        }
        if (afterImport)
        {
            if (!s.imports.length || s.imports[$ - 1].length == 0)
                s.imports ~= t.text;
            else if (i && tokens[i - 1].type == tok!".")
                s.imports[$ - 1] ~= "." ~ t.text;
            else
                s.imports ~= t.text;
            continue;
        }
        if (afterTypeKw && brace <= 1)
        {
            auto nxt = (i + 1 < tokens.length) ? tokens[i + 1].type : tok!";";
            // A type name is followed by `{`, `(`, or `:` (inheritance).
            // `enum name =` is a value; `enum int n` has a builtin in between.
            if (nxt != tok!"{" && nxt != tok!"(" && nxt != tok!":")
            {
                afterTypeKw = false;
                continue;
            }
            s.types ~= t.text;
            if (brace == 0)
                currentType = t.text;
            afterTypeKw = false;
            continue;
        }
        // Field: Type name; / Type name = inside a struct. Not mask.length.
        if (currentType.length && brace == 1 && i + 1 < tokens.length
            && (tokens[i + 1].type == tok!";" || tokens[i + 1].type == tok!"=")
            && t.text != currentType
            && !(i && tokens[i - 1].type == tok!"."))
        {
            s.fields ~= currentType ~ "." ~ t.text;
            continue;
        }
    }
}

FileAstDiff diffShapes(string rel, const AstShape goal, const AstShape gen, bool extraFile)
{
    FileAstDiff d;
    d.rel = rel;
    d.section = (rel.length >= 4 && rel[0 .. 4] == "ldc/") ? "ldc" : "stock";
    d.extraFile = extraFile;
    if (!goal.parsed && !extraFile)
        d.parseGoal = true;
    if (!gen.parsed && !extraFile)
        d.parseGen = true;
    if (extraFile)
    {
        d.notes ~= "generated only (not in this version's goal runtime)";
        return d;
    }
    if (goal.moduleName.length && gen.moduleName.length && goal.moduleName != gen.moduleName)
        d.moduleMismatch = goal.moduleName ~ " vs " ~ gen.moduleName;
    foreach (sy; goal.symbols)
        if (okDeclName(sy) && !gen.symbols.canFind(sy) && d.missingSymbols.length < 40)
            d.missingSymbols ~= sy;
    foreach (sy; gen.symbols)
        if (okDeclName(sy) && !goal.symbols.canFind(sy) && d.extraSymbols.length < 16)
            d.extraSymbols ~= sy;
    foreach (f; goal.fields)
        if (!gen.fields.canFind(f))
            d.missingFields ~= f;
    foreach (p; goal.pragmas)
        if (!pragmaCovered(p, gen.pragmas) && d.missingPragmas.length < 30)
            d.missingPragmas ~= p;
    return d;
}

/// llvm.memcpy.p0i8.p0i8.i# and llvm.memcpy.p0.p0.i# are the same intrinsic family.
private bool pragmaCovered(string want, const string[] have)
{
    if (have.canFind(want))
        return true;
    auto ws = pragmaStem(want);
    if (!ws.length)
        return false;
    foreach (h; have)
        if (pragmaStem(h) == ws)
            return true;
    return false;
}

private string pragmaStem(string p)
{
    auto s = p;
    enum pre = "LDC_intrinsic:";
    if (s.length > pre.length && s[0 .. pre.length] == pre)
        s = s[pre.length .. $];
    static immutable drop = [".p0i8", ".p0", ".p1", ".i#", ".f#", ".i32", ".i64", ".i8", ".i16"];
    bool more = true;
    while (more)
    {
        more = false;
        foreach (d; drop)
            if (s.length > d.length && s[$ - d.length .. $] == d)
            {
                s = s[0 .. $ - d.length];
                more = true;
            }
        if (s.length && s[$ - 1] == '.')
        {
            s = s[0 .. $ - 1];
            more = true;
        }
    }
    return s;
}

bool hasGaps(const FileAstDiff d)
{
    // FILE-CMP is the source of truth for text. AST gaps are missing
    // files, fields, and LDC pragmas — not token-scan "symbols".
    return d.missingFile || d.parseGen || d.moduleMismatch.length
        || d.missingFields.length || d.missingPragmas.length;
}

private string tryReadNorm(string p)
{
    if (!p.length || !exists(p))
        return "";
    try
        return normalizeSource(readText(p));
    catch (Exception)
        return "";
}

/// Compare generated tree to the version's LDC runtime (goal). Never copies.
VersionAstReport diffGeneratedVsGoal(string generatedDir, string goalRoot, string tag,
    bool ldcOnly = false, string stockRoot = "")
{
    VersionAstReport r;
    r.tag = tag;
    r.generatedDir = generatedDir;
    r.goalRoot = goalRoot;
    r.stockRoot = stockRoot;
    if (!generatedDir.length || !exists(generatedDir) || !isWalkable(goalRoot))
        return r;

    auto genFiles = walkMergedTree(generatedDir);
    auto goalFiles = walkLdcRuntime(goalRoot);
    RelFile[] stockFiles;
    if (stockRoot.length && isWalkable(stockRoot))
        stockFiles = walkStockRuntime(stockRoot);
    bool[string] seen;

    foreach (gf; goalFiles)
    {
        immutable isLdc = gf.rel.length >= 4 && gf.rel[0 .. 4] == "ldc/";
        if (ldcOnly && !isLdc)
            continue;
        seen[gf.rel] = true;
        r.compared++;
        if (isLdc)
            r.ldcFiles++;
        auto found = findRel(genFiles, gf.rel);
        FileAstDiff d;
        d.rel = gf.rel;
        d.section = isLdc ? "ldc" : "stock";
        if (found is null)
        {
            d.missingFile = true;
            d.notes ~= "missing in generated — add source/ldcmods/<name>.d and a row in ldcmods/package.d";
            r.missingFiles++;
            r.files ~= d;
            r.cmps ~= classifyFile(gf.rel, "", tryReadNorm(gf.abs), "");
            if (d.section == "ldc")
                r.ldcWithGaps++;
            else
                r.stockWithGaps++;
            continue;
        }
        r.present++;
        string gtxt, ntxt, stxt;
        try
        {
            gtxt = tryReadNorm(gf.abs);
            ntxt = tryReadNorm(found.abs);
        }
        catch (Exception e)
        {
            d.notes ~= e.msg;
            r.files ~= d;
            continue;
        }
        auto stockHit = findRel(stockFiles, gf.rel);
        if (stockHit !is null)
            stxt = tryReadNorm(stockHit.abs);
        auto cmp = classifyFile(gf.rel, ntxt, gtxt, stxt);
        if (cmp.klass == "adapt-delta")
            r.adaptDeltas++;
        else if (cmp.klass == "missed-ldc")
            r.missedLdcPatches++;
        if (cmp.textsEqual)
        {
            d.textsEqual = true;
            r.textsEqual++;
            continue;
        }
        r.textDiffs++;
        r.cmps ~= cmp;
        if (!isD(gf.rel))
        {
            d.notes ~= "native source differs (generated stub, not copied from goal)";
            r.files ~= d;
            continue;
        }
        if (!isLdc)
        {
            d.notes ~= "stock text differs: " ~ cmp.klass;
            if (cmp.firstDiffLine)
                d.notes ~= format("first hunk L%s gen=`%s` goal=`%s`",
                    cmp.firstDiffLine, cmp.genLine, cmp.goalLine);
            r.files ~= d;
            r.stockWithGaps++;
            continue;
        }
        if (gtxt.length > 400_000 || ntxt.length > 400_000)
        {
            d.notes ~= "large file; text differs (skipped deep scan)";
            r.files ~= d;
            r.ldcWithGaps++;
            continue;
        }
        AstShape gs, ns;
        try
        {
            gs = extractAst(gtxt, gf.rel);
            ns = extractAst(ntxt, found.rel);
        }
        catch (Exception e)
        {
            d.notes ~= "ast extract: " ~ e.msg;
            r.files ~= d;
            r.ldcWithGaps++;
            continue;
        }
        d = diffShapes(gf.rel, gs, ns, false);
        d.missingSymbols = keepDeclaredNames(d.missingSymbols, gtxt);
        d.extraSymbols = keepDeclaredNames(d.extraSymbols, ntxt);
        d.notes ~= "ldc-emit (generated from compiler, not copied)";
        if (cmp.firstDiffLine)
            d.notes ~= format("first hunk L%s", cmp.firstDiffLine);
        if (!gs.parsed)
            r.parseGaps++;
        if (!ns.parsed)
            r.parseGaps++;
        r.files ~= d;
        if (hasGaps(d))
            r.ldcWithGaps++;
        else if (d.missingFile)
            r.ldcWithGaps++;
    }
    foreach (nf; genFiles)
    {
        if (ldcOnly && (nf.rel.length < 4 || nf.rel[0 .. 4] != "ldc/"))
            continue;
        if (nf.rel in seen)
            continue;
        if (nf.rel == "AST-DIFF.md" || nf.rel == "FILE-CMP.md")
            continue;
        FileAstDiff d;
        d.rel = nf.rel;
        d.section = (nf.rel.length >= 4 && nf.rel[0 .. 4] == "ldc/") ? "ldc" : "stock";
        d.extraFile = true;
        d.notes ~= "generated only";
        r.extraFiles++;
        r.files ~= d;
        r.cmps ~= classifyFile(nf.rel, tryReadNorm(nf.abs), "", "");
    }
    return r;
}

string renderVersionAst(const VersionAstReport r, int maxFiles = 60)
{
    auto buf = appender!string();
    buf.put(format("## %s\n\n", r.tag));
    buf.put(format("- goal: `%s`\n- generated: `%s`\n", r.goalRoot, r.generatedDir));
    if (r.stockRoot.length)
        buf.put(format("- stock: `%s`\n", r.stockRoot));
    buf.put(format("- compared: %s  present: %s  missingFiles: %s  extraFiles: %s  textsEqual: %s  textDiffs: %s\n",
        r.compared, r.present, r.missingFiles, r.extraFiles, r.textsEqual, r.textDiffs));
    buf.put(format("- ldc files: %s  ldcWithGaps: %s  stockWithGaps: %s  parseGaps: %s\n",
        r.ldcFiles, r.ldcWithGaps, r.stockWithGaps, r.parseGaps));
    buf.put(format("- adapt-delta: %s  missed-ldc-patches: %s\n\n",
        r.adaptDeltas, r.missedLdcPatches));

    void section(string title, string want, int cap)
    {
        buf.put("### ");
        buf.put(title);
        buf.put("\n\n");
        int n;
        foreach (f; r.files)
        {
            if (f.section != want)
                continue;
            if (!hasGaps(f) && !f.extraFile && !f.missingFile && !f.notes.length)
                continue;
            buf.put(renderFileDiff(f));
            if (++n >= cap)
            {
                buf.put("_(truncated)_\n\n");
                break;
            }
        }
        if (!n)
            buf.put("(none)\n\n");
    }
    section("ldc/* (fix in `source/ldcmods/<name>.d`)", "ldc", 40);
    section("stock / other (fix in `source/adapt.d`)", "stock", 20);
    return buf.data;
}

string renderFileCmp(const VersionAstReport r)
{
    auto buf = appender!string();
    buf.put(format("# FILE-CMP %s — generated vs goal\n\n", r.tag));
    buf.put("Full-file compare after newline normalize. Goal is never copied.\n\n");
    buf.put(format("- compared %s  textsEqual %s  textDiffs %s  missing %s  extra %s\n",
        r.compared, r.textsEqual, r.textDiffs, r.missingFiles, r.extraFiles));
    buf.put(format("- adapt-delta %s (we changed stock the goal left alone)\n", r.adaptDeltas));
    buf.put(format("- missed-ldc %s (stock==gen, goal has an LDC patch we did not apply)\n\n",
        r.missedLdcPatches));
    buf.put("| path | class | gen | goal | first hunk |\n");
    buf.put("|---|---|---:|---:|---|\n");
    int n;
    foreach (c; r.cmps)
    {
        if (c.klass == "match")
            continue;
        buf.put(format("| `%s` | %s | %s | %s | L%s `%s` |\n",
            c.rel, c.klass, c.genLen, c.goalLen, c.firstDiffLine,
            c.genLine.replace("|", "\\|")));
        if (++n >= 120)
        {
            buf.put("\n_(truncated)_\n");
            break;
        }
    }
    if (!n)
        buf.put("_(every compared source file matches the goal after newline normalize)_\n");
    return buf.data;
}

string renderAllAstDiffs(const VersionAstReport[] reps)
{
    auto buf = appender!string();
    buf.put("# AST diff — generated vs per-version goal runtime\n\n");
    buf.put("Goal is `workspace/refs/<tag>` runtime (never copied). ");
    buf.put("Use the ldc/* section to close codegen gaps.\n\n");
    buf.put("| tag | compared | missing files | extra | ldc gaps | stock gaps | parse |\n");
    buf.put("|---|---:|---:|---:|---:|---:|---:|\n");
    foreach (r; reps)
        buf.put(format("| `%s` | %s | %s | %s | %s | %s | %s |\n",
            r.tag, r.compared, r.missingFiles, r.extraFiles,
            r.ldcWithGaps, r.stockWithGaps, r.parseGaps));
    buf.put("\n");
    foreach (r; reps)
        buf.put(renderVersionAst(r));
    return buf.data;
}

private string renderFileDiff(const FileAstDiff f)
{
    auto buf = appender!string();
    buf.put(format("#### `%s`\n\n", f.rel));
    if (f.missingFile)
        buf.put("- **missing file** — add `source/ldcmods/<name>.d` + a `ldcEmitters` row\n");
    if (f.extraFile)
        buf.put("- extra file (not in goal)\n");
    if (f.parseGoal)
        buf.put("- goal did not parse (libdparse); skipped some facts\n");
    if (f.parseGen)
        buf.put("- generated did not parse\n");
    if (f.moduleMismatch.length)
        buf.put(format("- module name: `%s`\n", f.moduleMismatch));
    void list(string title, const string[] xs, int cap = 24)
    {
        if (!xs.length)
            return;
        buf.put("- **");
        buf.put(title);
        buf.put(":** ");
        string[] show = xs.length > cap ? xs[0 .. cap].dup : xs.dup;
        buf.put(show.join(", "));
        if (xs.length > cap)
            buf.put(format(" (+%s)", xs.length - cap));
        buf.put("\n");
    }
    list("missing symbols", f.missingSymbols);
    list("extra symbols", f.extraSymbols, 12);
    list("missing fields", f.missingFields);
    list("missing pragmas", f.missingPragmas);
    foreach (n; f.notes)
    {
        buf.put("- ");
        buf.put(n);
        buf.put("\n");
    }
    buf.put("\n");
    return buf.data;
}

private string[] keepDeclaredNames(const string[] names, string src)
{
    string[] outp;
    foreach (n; names)
    {
        if (!okDeclName(n) || n.length < 3)
            continue;
        if (src.canFind("struct " ~ n) || src.canFind("class " ~ n)
            || src.canFind("union " ~ n) || src.canFind("template " ~ n)
            || src.canFind("interface " ~ n) || src.canFind("enum " ~ n))
            outp ~= n;
    }
    return outp;
}

private bool okDeclName(string t)
{
    if (!okIdent(t))
        return false;
    foreach (ch; t)
        if (ch <= ' ' || ch == '`' || ch == '"' || ch == '\'')
            return false;
    return true;
}

private bool okIdent(string t)
{
    if (t.length < 2)
        return false;
    auto c0 = t[0];
    if (!((c0 >= 'A' && c0 <= 'Z') || (c0 >= 'a' && c0 <= 'z') || c0 == '_'))
        return false;
    foreach (ch; t[1 .. $])
        if (!((ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z')
                || (ch >= '0' && ch <= '9') || ch == '_'))
            return false;
    switch (t)
    {
    case "int", "uint", "long", "ulong", "void", "bool", "byte", "ubyte",
        "short", "ushort", "char", "wchar", "dchar", "float", "double",
        "real", "string", "size_t", "if", "for", "while", "switch",
        "version", "debug", "mixin", "typeof", "cast", "this", "super",
        "null", "true", "false", "return", "auto", "enum", "alias",
        "inout", "shared", "scope", "pure", "nothrow", "ref", "out",
        "lazy", "align", "pragma", "import", "module", "unittest":
        return false;
    default:
        return true;
    }
}

private bool isD(string rel)
{
    auto e = extension(rel);
    return e == ".d" || e == ".di";
}

private bool isWalkable(string root)
{
    import paths : druntimeSrc;
    return root.length && exists(druntimeSrc(root));
}

unittest
{
    enum goal = q"D
module ldc.attributes;
struct allocSize { int sizeArgIdx; int numArgIdx; }
immutable assumeUsed = 1;
unittest { auto junk = "seven"; enum turco = 1; }
D";
    enum gen = q"D
module ldc.attributes;
struct allocSize { int sizeArgIdx; }
D";
    auto g = extractAst(goal, "ldc/attributes.d");
    auto n = extractAst(gen, "ldc/attributes.d");
    assert(g.parsed && n.parsed);
    assert(g.types.canFind("allocSize"));
    assert(g.fields.canFind("allocSize.sizeArgIdx"));
    assert(!g.symbols.canFind("turco"), g.symbols.join(","));
    assert(!g.symbols.canFind("seven"));
    auto d = diffShapes("ldc/attributes.d", g, n, false);
    assert(d.section == "ldc");
    assert(d.missingFields.canFind("allocSize.numArgIdx"));
    assert(g.fields.canFind("allocSize.numArgIdx"));
    assert(hasGaps(d));
    assert(pragmaCovered("LDC_intrinsic:llvm.memcpy.p0i8.p0i8.i#",
        ["LDC_intrinsic:llvm.memcpy.p0.p0.i#"]));

    auto c = classifyFile("object.d", "a\nb\n", "a\nc\n", "a\nc\n");
    assert(c.klass == "adapt-delta");
    assert(c.firstDiffLine == 2);
    auto c2 = classifyFile("core/bitop.d", "stock\n", "goal\n", "stock\n");
    assert(c2.klass == "missed-ldc");
    auto c3 = classifyFile("ldc/simd.di", "gen\n", "goal\n", "");
    assert(c3.klass == "ldc-emit");
}
