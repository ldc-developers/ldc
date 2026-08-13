//===-- tools/runtime-adapt/source/parseutil.d --------------------*- D -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

module parseutil;

import dparse.ast;
import dparse.lexer;
import dparse.parser : parseModule;
import dparse.rollback_allocator : RollbackAllocator;

import std.algorithm : canFind;
import std.array : join;

struct ParseFacts
{
    bool parsed;
    string error;
    string moduleName;
    string[] imports;
    string[] externCNames;
    string[] versions;
    bool mentionsCustomCrt; /// CRuntime_* other than stock WASI/Glibc/Microsoft/…
    bool mentionsWebAssembly;
    bool mentionsWasi;
}

string normalizeSource(string raw)
{
    import std.array : replace;
    return raw.replace("\r\n", "\n").replace("\r", "\n");
}

ParseFacts parseDSource(string source, string fileName)
{
    ParseFacts f;
    auto src = normalizeSource(source);
    f.mentionsWebAssembly = src.canFind("WebAssembly");
    f.mentionsWasi = src.canFind("version(WASI)") || src.canFind("version (WASI)")
        || src.canFind("CRuntime_WASI");

    LexerConfig config;
    config.fileName = fileName;
    config.stringBehavior = StringBehavior.source;
    auto cache = StringCache(StringCache.defaultBucketCount);
    auto tokens = getTokensForParser(cast(ubyte[]) src, config, &cache);

    foreach (t; tokens)
    {
        if (t.type == tok!"identifier" && t.text.length)
        {
            auto id = t.text;
            if (id == "WebAssembly" || id == "WASI" || id == "CRuntime_WASI"
                || (id.length > 9 && id[0 .. 9] == "CRuntime_"))
            {
                if (!f.versions.canFind(id))
                    f.versions ~= id;
                if (id.length > 9 && id[0 .. 9] == "CRuntime_"
                    && id != "CRuntime_WASI" && id != "CRuntime_Glibc"
                    && id != "CRuntime_Microsoft" && id != "CRuntime_Musl"
                    && id != "CRuntime_Bionic" && id != "CRuntime_DigitalMars"
                    && id != "CRuntime_Newlib" && id != "CRuntime_UClibc")
                    f.mentionsCustomCrt = true;
            }
        }
    }

    string lastErr;
    void msg(string, size_t, size_t, string message, bool isError)
    {
        if (isError)
            lastErr = message;
    }

    RollbackAllocator rba;
    uint errors;
    auto mod = parseModule(tokens, fileName, &rba, &msg, &errors);
    if (errors || mod is null)
    {
        f.parsed = false;
        f.error = lastErr.length ? lastErr : "parse failed";
        return f;
    }
    f.parsed = true;
    auto v = new FactVisitor();
    v.visit(mod);
    f.moduleName = v.moduleName;
    f.imports = v.imports;
    f.externCNames = v.externC;
    return f;
}

private string idChainToString(const IdentifierChain chain)
{
    if (chain is null)
        return "";
    string[] parts;
    foreach (t; chain.identifiers)
        if (t.text.length)
            parts ~= t.text;
    return parts.join(".");
}

private bool isExternC(const Attribute attr)
{
    return attr !is null && attr.linkageAttribute !is null
        && attr.linkageAttribute.identifier.text == "C";
}

private final class FactVisitor : ASTVisitor
{
    alias visit = ASTVisitor.visit;
    string moduleName;
    string[] imports;
    string[] externC;

    override void visit(const ModuleDeclaration m)
    {
        if (m !is null)
            moduleName = idChainToString(m.moduleName);
        m.accept(this);
    }

    override void visit(const ImportDeclaration imp)
    {
        foreach (si; imp.singleImports)
        {
            auto n = idChainToString(si.identifierChain);
            if (n.length && !imports.canFind(n))
                imports ~= n;
        }
        imp.accept(this);
    }

    override void visit(const FunctionDeclaration fn)
    {
        bool cLinkage;
        foreach (a; fn.attributes)
            if (isExternC(a))
                cLinkage = true;
        if (cLinkage && fn.name.text.length && !externC.canFind(fn.name.text))
            externC ~= fn.name.text;
        fn.accept(this);
    }
}

bool looksStubbed(string src, ulong refBytes, ulong oursBytes)
{
    auto n = normalizeSource(src);
    if (refBytes > 0 && oursBytes * 100 / refBytes < 35)
        return true;
    return n.canFind("unimplemented") || n.canFind("TODO: stub") || n.canFind("STUB:");
}

bool looksAdapted(const ParseFacts ours, const ParseFacts stock)
{
    if (ours.mentionsCustomCrt && !stock.mentionsCustomCrt)
        return true;
    if (ours.mentionsWebAssembly && !stock.mentionsWebAssembly)
        return true;
    return false;
}
