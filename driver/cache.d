//===-- driver/cache.d --------------------------------------------*- D -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

module driver.cache;

import ddmd.arraytypes;
import ddmd.dmodule;
import ddmd.errors;
import ddmd.globals;
import ddmd.root.filename;

import gen.logger;

import std.digest.md;

extern (C++, cache)
{
    void recoverObjectFile(const(char)* cacheFile, size_t cacheFileLen,
                           const(char)* objectFile, size_t objectFileLen);
    bool canDoSourceCachedBuild(ref Modules modules);
}

struct ManifestDependency
{
    string filename;
    string hash;
}

string getManifestFileName(const(char)[] hash)
{
    import std.path;
    import std.string;
    import core.stdc.string;
    import std.conv;

    string cachePath = to!string(global.params.useCompileCache);
    return buildNormalizedPath(expandTilde(cachePath), "manifest_" ~ hash);
}

extern (C++) void cacheManifest(const(char)* hash, const(char)* cacheObjFile)
{
    import std.stdio;
    import std.string;
    import std.path;

    // Make sure it is still valid to cache this build. This takes care of disabling
    // caching when string mixins contain __TIME__ (which is not caught during parsing).
    if (!isCachingValidAfterSemanticAnalysis())
    {
        Log.printfln("Not writing cache manifest: caching is proven invalid by semantic analysis.");
        return;
    }

    auto filename = getManifestFileName(fromStringz(hash));
    auto f = File(filename, "w");

    Log.printfln("Write cache manifest: %s", filename);

    f.writeln("Cached object file:");
    f.writeln(fromStringz(cacheObjFile));

    f.writeln("Non-existant paths:");
    foreach (fname; nonExistantPaths)
    {
        f.writeln(fname);
    }

    f.writeln("Imported:");
    foreach (i, fname; Module.allTextImports)
    {
        f.writeln(fromStringz(fname));
        f.writeln(fromStringz(Module.allTextImportsHash[i]));
    }
    foreach (m; Module.amodules)
    {
        if (!m.isRoot)
        {
            auto fname = fromStringz(m.srcfile.toChars());
            f.writeln(fname.asAbsolutePath);
            f.writeln(fromStringz(m.srcfile.hashToChars()));
        }
    }

    f.close();
}

void touchFile(string filename)
{
    import std.stdio : File;

    auto f = File(filename, "r+");
    f.close();
}

// Return true when successful.
bool attemptRecoverFromCache(string hash, const(char)* outputObjFile)
{
    static import std.file;
    import std.string;
    import core.stdc.string;

    string cacheObjFile;
    ManifestDependency[] deps;
    string[] nonexistants;
    if (!readManifest(hash, cacheObjFile, deps, nonexistants))
    {
        Log.printfln("No cache manifest found for this build.");
        return false;
    }

    // The cached file may have been removed (e.g. by cache pruning).
    if (!std.file.exists(cacheObjFile))
    {
        Log.printfln("Cached object file was deleted from cache.");
        return false;
    }

    if (!checkManifestNonexistants(nonexistants))
        return false;

    if (!checkManifestDependencies(deps))
        return false;

    // It all checks out! Let's recover the cached file and be happy :-)
    Log.printfln("Cache manifest checks out.\nRecovering outputs from cache (source cached).");
    recoverObjectFile(cacheObjFile.ptr, cacheObjFile.length, outputObjFile, strlen(outputObjFile));
    return true;
}

struct CachingState
{
    const(char)* dateUsed; // the date string used by calculateModulesHash.
    const(char)* timeUsed; // the time string used by calculateModulesHash.
}
CachingState state;

// Checks whether caching is still valid after doing semantic analysis.
//
// String mixins are only evaluated/parsed/... during semantic analysis, and
// code inside string mixins is therefore not taken into account by
// calculateModulesHash. This function is used to check whether the module hash
// calculated by calculateModulesHash should have been different (and if so,
// caching makes no sense because cache lookup is based on the hash returned by
// calculateModulesHash).
bool isCachingValidAfterSemanticAnalysis()
{
    return (global.params.timeUsedByLexer == state.timeUsed)
        && (global.params.dateUsedByLexer == state.dateUsed);
}

string calculateModulesHash(ref Modules modules)
{
    import std.string;
    import std.file;

    MD5 md5;
    md5.start();

    // First add the compiler version to the hash
    md5.put(cast(const(ubyte)[]) fromStringz(global.ldc_version));
    md5.put(cast(const(ubyte)[]) fromStringz(global._version));
    md5.put(cast(const(ubyte)[]) fromStringz(global.llvm_version));

    addCommandlineToHash(md5);

    // The current directory is also an input (import lookup).
    md5.put(cast(const(ubyte)[]) getcwd());

    // Add the date and/or time as compile "input" in case the Lexer needed it.
    state.dateUsed = global.params.dateUsedByLexer;
    state.timeUsed = global.params.timeUsedByLexer;
    if (state.dateUsed)
    {
        auto str = fromStringz(state.dateUsed);
        Log.printfln("Add date to hash, \"%s\"", str);
        md5.put(cast(const(ubyte)[]) str);
    }
    if (state.timeUsed)
    {
        auto str = fromStringz(state.timeUsed);
        Log.printfln("Add time to hash, \"%s\"", str);
        md5.put(cast(const(ubyte)[]) str);
    }

    foreach (ref m; modules)
    {
        md5.put(m.srcfile.buffer[0 .. m.srcfile.len]);
        // Also add the module source filenames to the hash (because it is an input to the compiler: e.g. __FILE__)
        md5.put(cast(const(ubyte)[]) fromStringz(m.srcfile.name.toChars()));
    }

    auto hash = md5.finish();
    return toHexString!(LetterCase.lower)(hash).dup;
}

private void addCommandlineToHash(ref MD5 md5)
{
    import core.runtime;
    import std.string;

    // Add _all_ commandline flags to the hash, except the ones that are proven to not matter.
    // TODO: make the hash independent of things that don't matter:
    //       - order of cmdline flags (e.g. "-g -c" == "-c -g")
    //       - and more...
    auto args = Runtime.cArgs();
    for (size_t i = 0; i < args.argc; i++)
    {
        md5.put(cast(const(ubyte)[]) fromStringz(args.argv[i]));
    }
}

private bool checkManifestNonexistants(string[] nonexistants)
{
    static import std.file;

    foreach (fname; nonexistants)
    {
        if (std.file.exists(fname))
        {
            Log.printfln(
                "Cache manifest fail: previously non-existant file now exists (%s).",
                fname);
            return false;
        }
    }

    return true;
}

private bool checkManifestDependencies(ManifestDependency[] deps)
{
    foreach (ref d; deps)
    {
        if (!checkManifestDependency(d))
        {
            Log.printfln("Manifest hash failure for %s", d.filename);
            return false;
        }
    }

    return true;
}

// true if we have a match
bool checkManifestDependency(ref ManifestDependency dep)
{
    import std.digest.md;
    import std.file;
    import std.stdio;
    import std.string;

    if (!exists(dep.filename))
        return false;

    auto f = File(dep.filename, "rb");
    auto md5 = md5Of(f.byChunk(4096));
    return toHexString!(LetterCase.lower)(md5) == dep.hash;
}

bool readManifest(string hash, ref string cacheObjFile,
    ref ManifestDependency[] deps, ref string[] nonexistant)
{
    import std.file;
    import std.stdio;
    import std.string;
    import std.array;
    import std.conv;

    auto filename = getManifestFileName(hash);
    if (!exists(filename))
        return false;
    auto f = File(filename, "r");
    scope (exit)
        f.close();

    char[] buf;
    f.readln(buf);
    if (buf != "Cached object file:\n")
    {
        warning(Loc(), "Corrupt manifest (%s) 1", filename.toStringz());
        return false;
    }

    f.readf("%s\n", &cacheObjFile);
    if (cacheObjFile.empty)
    {
        warning(Loc(), "Corrupt manifest (%s) 2", filename.toStringz());
        return false;
    }

    f.readln(buf);
    if (buf != "Non-existant paths:\n")
    {
        warning(Loc(), "Corrupt manifest (%s) 3", filename.toStringz());
        return false;
    }

    string fname;
    while (!f.eof)
    {
        auto status = f.readf("%s\n", &fname);
        if (status != 1)
            break;
        if (fname == "Imported:")
            break;

        nonexistant ~= fname;
    }

    if (fname != "Imported:")
    {
        warning(Loc(), "Corrupt manifest (%s) 4", filename.toStringz());
        return false;
    }

    while (!f.eof)
    {
        ManifestDependency dep;
        auto status = f.readf("%s\n%s\n", &dep.filename, &dep.hash);
        if (status != 2)
            break;

        deps ~= dep;
    }

    return true;
}
