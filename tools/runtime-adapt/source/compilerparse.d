//===-- tools/runtime-adapt/source/compilerparse.d ----------------*- D -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Parse LDC *compiler* sources (dmd/, gen/, driver/, ir/) into a model of
// the runtime modules the frontend expects. Never reads runtime/.
//
//===----------------------------------------------------------------------===//

module compilerparse;

import std.algorithm : canFind, sort, uniq;
import std.array : appender, array;
import std.file : dirEntries, exists, readText, SpanMode;
import std.path : baseName, buildPath, extension;
import std.regex : ctRegex, matchAll;
import std.string : strip;

struct IdPair
{
    string ident; /// compiler Id::* name
    string str;   /// D identifier the runtime must declare
}

struct AttrShape
{
    string name;
    string[] fieldTypes; /// "int", "string", …
}

struct CompilerModel
{
    bool ok;
    string guideRoot;
    IdPair[] ids;
    string[] pragmas;
    string[] magicModules;     /// attributes, dcompute, opencl, …
    string[] attrNames;        /// allocSize, section, _assumeUsed, …
    string[] dcomputeNames;    /// compute, _kernel, Pointer, …
    string[] ldcModules;       /// bare names after "ldc."
    string[] llvmIntrinsics;   /// llvm.returnaddress, …
    string[] ldcIntrinsics;    /// convertvector, bitop.bt, …
    AttrShape[] attrShapes;
    bool llvmIdentIsMajor;
    int llvmLo = 11;
    int llvmHi = 24;
    bool hasWasm;
    bool hasFuzzer;
    bool hasSanitizer;
    bool definesLdc;               /// addPredefinedGlobalIdent("LDC")
    string[] predefinedVersions;   /// LDC, CRuntime_WASI, WebAssembly, …
    string[] runtimeHooks;         /// _d_throw_exception, _d_allocmemory, …
    string[] hooksPresentInStock;  /// already declared in the reference tree
    string[] hooksPresentInGoal;   /// already declared in the goal tree (guide)
    string[] implicitModules;      /// core.interpolation, …
    string[string] implicitSymbols; /// InterpolationHeader → core.interpolation
    string[] notes;
}

/// Parse driver/ + gen/ + dmd/ of an LDC checkout. Skips runtime/.
/// Locus map (LDC commit usually touches the left file first):
///   gen/pragma.cpp      → pragmas, LDC_intrinsic, inline_{asm,ir}
///   gen/uda.cpp         → attrNames / attrShapes
///   gen/runtime.cpp     → getRuntimeFunction / createFwdDecl hooks
///   driver/main.cpp     → addPredefinedGlobalIdent, LLVM version ident
///   dmd/id.d            → Id::* tables
///   dmd/imphint.d       → implicitModules
CompilerModel parseCompiler(string guideRoot)
{
    CompilerModel m;
    m.guideRoot = guideRoot;
    if (!guideRoot.length)
        return m;
    auto pragmaCpp = buildPath(guideRoot, "gen", "pragma.cpp");
    auto mainCpp = buildPath(guideRoot, "driver", "main.cpp");
    if (!exists(pragmaCpp) || !exists(mainCpp))
        return m;
    m.ok = true;

    parseIdTables(m, guideRoot);
    parsePragmaCpp(m, pragmaCpp);
    parseUdaCpp(m, buildPath(guideRoot, "gen", "uda.cpp"));
    parseMainCpp(m, mainCpp);
    scanCompilerTree(m, guideRoot);
    parseImphint(m, buildPath(guideRoot, "dmd", "imphint.d"));
    dedupe(m);
    return m;
}

/// Stock path that should declare `hook` if the compiler calls it.
string hookHome(string hook)
{
    if (hook == "_d_throw_exception" || hook == "_d_throwc")
        return "object.d";
    if (hook == "_d_assert" || hook == "_d_assert_msg" || hook == "_d_arraybounds"
        || hook == "_d_unittest" || hook == "_d_unittest_msg")
        return "core/exception.d";
    if (hook.length >= 6 && hook[0 .. 3] == "_d_"
        && (hook.canFind("alloc") || hook.canFind("del") || hook.canFind("new")))
        return "rt/lifetime.d";
    if (hook == "_d_cover_register2" || hook == "_d_cover_register")
        return "rt/cover.d";
    return "";
}

/// D path for a dotted module name (core.interpolation → core/interpolation.d).
string moduleToRel(string dotted)
{
    if (!dotted.length)
        return "";
    if (dotted == "__importc_builtins" || dotted == "importc_builtins")
        return "__importc_builtins.di";
    auto buf = appender!string();
    foreach (ch; dotted)
        buf.put(ch == '.' ? '/' : ch);
    if (dotted == "ldc.intrinsics" || dotted == "ldc.llvmasm" || dotted == "ldc.simd"
        || dotted == "ldc.opencl" || dotted == "ldc.profile" || dotted == "ldc.libfuzzer")
        return buf.data ~ ".di";
    return buf.data ~ ".d";
}

bool hasPragma(const CompilerModel m, string id)
{
    foreach (p; m.pragmas)
        if (p == id)
            return true;
    return false;
}

bool hasLdcModule(const CompilerModel m, string name)
{
    foreach (n; m.ldcModules)
        if (n == name)
            return true;
    foreach (n; m.magicModules)
        if (n == name)
            return true;
    return false;
}

/// runtime-adapt path for a compiler-facing ldc.* module.
string moduleRel(string ldcName)
{
    if (ldcName == "intrinsics" || ldcName == "llvmasm" || ldcName == "simd"
        || ldcName == "opencl" || ldcName == "profile" || ldcName == "libfuzzer")
        return "ldc/" ~ ldcName ~ ".di";
    return "ldc/" ~ ldcName ~ ".d";
}

private void parseIdTables(ref CompilerModel m, string root)
{
    auto re = ctRegex!`\{\s*"([^"]+)"\s*(?:,\s*"([^"]*)")?\s*\}`;
    void eat(string text)
    {
        foreach (cap; matchAll(text, re))
        {
            IdPair p;
            p.ident = cap[1];
            p.str = cap[2].length ? cap[2] : cap[1];
            m.ids ~= p;
            if (p.ident.length >= 4 && p.ident[0 .. 4] == "LDC_")
                m.pragmas ~= p.ident;
            else if (p.ident.length >= 3 && p.ident[0 .. 3] == "uda")
                m.attrNames ~= p.str;
            else if (p.ident == "attributes" || p.ident == "dcompute"
                || p.ident == "opencl" || p.ident == "ldc")
                m.magicModules ~= p.str;
            else if (p.ident == "dcPointer" || p.ident == "dcReflect"
                || p.ident == "udaCompute" || p.ident == "udaKernel")
                m.dcomputeNames ~= p.str;
        }
    }
    foreach (rel; ["dmd/id.d", "dmd/id.h"])
    {
        auto p = buildPath(root, rel);
        if (exists(p))
            eat(readText(p));
    }
}

private void parsePragmaCpp(ref CompilerModel m, string path)
{
    if (!exists(path))
        return;
    const t = readText(path);
    auto idRe = ctRegex!`Id::(LDC_[A-Za-z0-9_]+)`;
    foreach (cap; matchAll(t, idRe))
        m.pragmas ~= cap[1];
    auto lit = ctRegex!`"((?:llvm|ldc)\.[A-Za-z0-9_.#]+)"`;
    foreach (cap; matchAll(t, lit))
    {
        auto s = cap[1];
        if (s.length >= 5 && s[0 .. 5] == "llvm.")
            m.llvmIntrinsics ~= s;
        else if (s.length >= 4 && s[0 .. 4] == "ldc.")
            m.ldcIntrinsics ~= s[4 .. $];
    }
}

private void parseUdaCpp(ref CompilerModel m, string path)
{
    if (!exists(path))
        return;
    const t = readText(path);
    auto attr = ctRegex!`@ldc\.attributes\.([A-Za-z_][A-Za-z0-9_]*)`;
    foreach (cap; matchAll(t, attr))
        m.attrNames ~= cap[1];
    auto dc = ctRegex!`@ldc\.dcompute\.([A-Za-z_][A-Za-z0-9_]*)`;
    foreach (cap; matchAll(t, dc))
        m.dcomputeNames ~= cap[1];

    // checkStructElems(sle, {Type::tint32, Type::tstring}) near applyAttrX
    auto fn = ctRegex!`void applyAttr([A-Za-z]+)\s*\(`;
    auto elems = ctRegex!`checkStructElems\s*\(\s*sle\s*,\s*\{([^}]+)\}`;
    string current;
    import std.string : splitLines;
    foreach (line; t.splitLines)
    {
        auto fm = matchAll(line, fn);
        if (!fm.empty)
            current = fm.front[1];
        auto em = matchAll(line, elems);
        if (!em.empty && current.length)
        {
            AttrShape s;
            s.name = current[0 .. 1].toLower ~ current[1 .. $];
            foreach (tok; em.front[1].splitTypeList)
                s.fieldTypes ~= tok;
            m.attrShapes ~= s;
        }
    }
}

private string toLower(string s)
{
    if (!s.length)
        return s;
    char c = s[0];
    if (c >= 'A' && c <= 'Z')
        c = cast(char)(c + 32);
    return c ~ s[1 .. $];
}

private string[] splitTypeList(string inner)
{
    string[] outp;
    foreach (part; inner.splitByComma)
    {
        auto p = part.strip;
        if (p.canFind("tint32") || p.canFind("tuns32"))
            outp ~= "int";
        else if (p.canFind("tint64"))
            outp ~= "long";
        else if (p.canFind("tstring") || p.canFind("String"))
            outp ~= "string";
        else if (p.canFind("tbool"))
            outp ~= "bool";
        else if (p.length)
            outp ~= "string";
    }
    return outp;
}

private string[] splitByComma(string s)
{
    string[] outp;
    size_t a;
    foreach (i, ch; s)
    {
        if (ch == ',')
        {
            outp ~= s[a .. i];
            a = i + 1;
        }
    }
    outp ~= s[a .. $];
    return outp;
}

private void parseMainCpp(ref CompilerModel m, string path)
{
    const t = readText(path);
    m.llvmIdentIsMajor = t.canFind("XSTR(LLVM_VERSION_MAJOR)");
    m.hasWasm = t.canFind("WebAssembly") || t.canFind("isWasm");
    auto ver = ctRegex!`addPredefinedGlobalIdent\s*\(\s*"([A-Za-z_][A-Za-z0-9_]*)"`;
    foreach (cap; matchAll(t, ver))
    {
        m.predefinedVersions ~= cap[1];
        if (cap[1] == "LDC")
            m.definesLdc = true;
        if (cap[1] == "WebAssembly")
            m.hasWasm = true;
    }
}

private void parseImphint(ref CompilerModel m, string path)
{
    if (!exists(path))
        return;
    const t = readText(path);
    auto hint = ctRegex!`"([A-Za-z_][A-Za-z0-9_]*)"\s*:\s*"([a-zA-Z0-9_.]+)"`;
    foreach (cap; matchAll(t, hint))
    {
        if (cap[2].canFind("."))
        {
            m.implicitModules ~= cap[2];
            m.implicitSymbols[cap[1]] = cap[2];
        }
    }
}

private void scanCompilerTree(ref CompilerModel m, string root)
{
    auto modRe = ctRegex!`ldc\.([A-Za-z_][A-Za-z0-9_]*)`;
    auto llvmRe = ctRegex!`"llvm\.([A-Za-z0-9_.#]+)"`;
    auto verRe = ctRegex!`LDC_LLVM_VER\s*(?:>=|<=|>|<|==)\s*(\d+)`;
    auto hookRe = ctRegex!`getRuntimeFunction\s*\([^;]{0,200}"(_d_[A-Za-z0-9_]+)"`;
    auto fwdRe = ctRegex!`createFwdDecl\s*\([^;]{0,200}"(_d_[A-Za-z0-9_]+)"`;
    auto importedRe = ctRegex!`imported!"([a-zA-Z0-9_.]+)"`;
    auto coreModRe = ctRegex!`"(core\.[a-zA-Z0-9_.]+)"`;
    int lo = int.max, hi;

    void walkDir(string dir)
    {
        if (!exists(dir))
            return;
        foreach (e; dirEntries(dir, SpanMode.depth))
        {
            if (e.isDir)
                continue;
            auto ext = extension(e.name);
            if (ext != ".cpp" && ext != ".h" && ext != ".d" && ext != ".c")
                continue;
            // Never read the LDC runtime — compiler sources only.
            auto bn = e.name;
            if (bn.canFind("runtime") && bn.canFind("druntime"))
                continue;
            string text;
            try
                text = readText(e.name);
            catch (Exception)
                continue;
            foreach (cap; matchAll(text, modRe))
            {
                auto n = cap[1];
                if (isLdcRuntimeModule(n))
                    m.ldcModules ~= n;
            }
            foreach (cap; matchAll(text, llvmRe))
                m.llvmIntrinsics ~= "llvm." ~ cap[1];
            foreach (cap; matchAll(text, hookRe))
                m.runtimeHooks ~= cap[1];
            foreach (cap; matchAll(text, fwdRe))
                m.runtimeHooks ~= cap[1];
            foreach (cap; matchAll(text, importedRe))
                m.implicitModules ~= cap[1];
            foreach (cap; matchAll(text, coreModRe))
                if (cap[1] == "core.interpolation" || cap[1] == "core.stdc.stdatomic")
                    m.implicitModules ~= cap[1];
            foreach (cap; matchAll(text, verRe))
            {
                import std.conv : to;
                auto v = to!int(cap[1]);
                if (v < 100)
                    v *= 100;
                if (v < lo)
                    lo = v;
                if (v > hi)
                    hi = v;
            }
        }
    }
    foreach (sub; ["dmd", "gen", "driver", "ir"])
        walkDir(buildPath(root, sub));

    if (lo != int.max)
    {
        m.llvmLo = lo / 100;
        m.llvmHi = hi / 100;
        if (m.llvmHi < m.llvmLo)
            m.llvmHi = m.llvmLo + 6;
    }
    auto cmake = buildPath(root, "CMakeLists.txt");
    if (exists(cmake))
    {
        auto cre = ctRegex!`find_package\s*\(\s*LLVM\s+(\d+)`;
        foreach (cap; matchAll(readText(cmake), cre))
        {
            import std.conv : to;
            auto maj = to!int(cap[1]);
            if (maj < m.llvmLo)
                m.llvmLo = maj;
        }
    }
    if (m.llvmIdentIsMajor && m.llvmHi < 18)
        m.llvmHi = 24;
    m.hasSanitizer = exists(buildPath(root, "driver", "cl_options_sanitizers.cpp"));
    m.hasFuzzer = m.hasSanitizer || hasLdcModule(m, "libfuzzer");
}

/// Modules the LDC frontend actually talks to in druntime (not filenames).
bool isLdcRuntimeModule(string n)
{
    switch (n)
    {
    case "attributes", "dcompute", "intrinsics", "llvmasm", "simd",
        "opencl", "profile", "libfuzzer", "eh_msvc", "eh_wasm", "asan",
        "sanitizer_common", "sanitizers_optionally_linked",
        "sanitizers_flag", "traits", "dynamic_compile":
        return true;
    default:
        return false;
    }
}

private void dedupe(ref CompilerModel m)
{
    m.pragmas = m.pragmas.sort.uniq.array;
    m.magicModules = m.magicModules.sort.uniq.array;
    m.attrNames = m.attrNames.sort.uniq.array;
    m.dcomputeNames = m.dcomputeNames.sort.uniq.array;
    m.ldcModules = m.ldcModules.sort.uniq.array;
    m.llvmIntrinsics = m.llvmIntrinsics.sort.uniq.array;
    m.ldcIntrinsics = m.ldcIntrinsics.sort.uniq.array;
    m.predefinedVersions = m.predefinedVersions.sort.uniq.array;
    m.runtimeHooks = m.runtimeHooks.sort.uniq.array;
    m.implicitModules = m.implicitModules.sort.uniq.array;
}
