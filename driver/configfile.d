//===-- driver/configfile.d - LDC config file handling ------------*- D -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Handles reading and parsing of an LDC config file (ldc.conf/ldc2.conf).
//
//===----------------------------------------------------------------------===//
module driver.configfile;

import dmd.globals;
import dmd.root.array;
import dmd.root.string : toDString, toCString, toCStringThen;
import driver.config;
import core.stdc.stdio;


string normalizeSlashes(const(char)* binDir)
{
    auto res = binDir.toDString.dup;
    foreach (ref c; res)
    {
        if (c == '\\') c = '/';
    }
    return cast(string)res; // assumeUnique
}

void findArraySetting(GroupSetting[] sections, string name, scope void delegate(const ArraySetting as) callback)
{
    foreach (section; sections)
    {
        foreach (c; section.children)
        {
            if (c.type == Setting.Type.array && c.name == name)
            {
                auto as = cast(ArraySetting) c;
                callback(as);
            }
        }
    }
}

string findScalarSetting(GroupSetting[] sections, string name)
{
    string result = null;
    foreach (section; sections)
    {
        foreach (c; section.children)
        {
            if (c.type == Setting.Type.scalar && c.name == name)
                result = (cast(ScalarSetting) c).val;
        }
    }
    return result;
}

string replace(string str, string pattern, string replacement)
{
    string res;
    size_t cap = str.length;
    if (replacement.length > pattern.length)
        cap += replacement.length - pattern.length;
    reserve(res, cap);

    while (str.length)
    {
        if (str.length < pattern.length)
        {
            res ~= str;
            str = null;
        }
        else if (str[0 .. pattern.length] == pattern)
        {
            res ~= replacement;
            str = str[pattern.length .. $];
        }
        else
        {
            res ~= str[0];
            str = str[1 .. $];
        }
    }
    return res;
}

unittest
{
    enum pattern = "pattern";
    enum test1 = "find the pattern in a sentence";
    enum test2 = "find the pattern";
    enum test3 = "pattern in a sentence";
    enum test4 = "a pattern, yet other patterns";

    assert(replace(test1, pattern, "word") == "find the word in a sentence");
    assert(replace(test2, pattern, "word") == "find the word");
    assert(replace(test3, pattern, "word") == "word in a sentence");
    assert(replace(test4, pattern, "word") == "a word, yet other words");
}

struct CfgPaths
{
    string cfgBaseDir; /// ldc2.conf directory
    string ldcBinaryDir; /// ldc2.exe binary dir

    this(const(char)* cfPath, const(char)* binDir)
    {
        import dmd.root.filename: FileName;

        cfgBaseDir = normalizeSlashes(FileName.path(cfPath));
        ldcBinaryDir = normalizeSlashes(binDir);
    }
}

string replacePlaceholders(string str, CfgPaths cfgPaths)
{
    return str
        .replace("%%ldcbinarypath%%", cfgPaths.ldcBinaryDir)
        .replace("%%ldcconfigpath%%", cfgPaths.cfgBaseDir)
        .replace("%%ldcversion%%", cast(string) global.ldc_version);
}

/++ Check that a section only contains known config keys

 ldc recognizes:
 - switches
 - post-switches
 - lib-dirs
 - rpath
+/
void validateSettingNames(const GroupSetting group, const char* filePath) {
    static void fail(const Setting setting, const char* filePath) {
        string fmt(Setting.Type type) {
            final switch(type) {
                static foreach (mem; __traits(allMembers, Setting.Type))
                case __traits(getMember, Setting.Type, mem):
                    return mem;
            }
        }

        import dmd.root.string : toDString;
        string msg;
        if (setting.type == Setting.Type.group)
            msg = "Nested group " ~ setting.name ~ " is unsupported";
        else
            msg = "Unknown " ~ fmt(setting.type) ~ " setting named " ~ setting.name;

        throw new Exception(msg);
    }

    alias ST = Setting.Type;
    static immutable knownSettings = [
        new Setting("switches", ST.array),
        new Setting("post-switches", ST.array),
        new Setting("lib-dirs", ST.array),
        new Setting("rpath", ST.scalar),
    ];
    outer: foreach (setting; group.children) {
        foreach (known; knownSettings)
            if (setting.name == known.name && setting.type == known.type)
                continue outer;
        fail(setting, filePath);
    }
}

extern(C++) struct ConfigFile
{
    __gshared ConfigFile instance;

private:

    // representation

    const(char)* pathcstr;
    Array!(const(char)*) switches;
    Array!(const(char)*) postSwitches;
    Array!(const(char)*) _libDirs;
    const(char)* rpathcstr;

    static bool sectionMatches(const(char)* section, const(char)* triple) nothrow;

    bool readConfig(const(char)* cfPath, const(char)* triple, const(char)* binDir)
    {
        const cfgPaths = CfgPaths(cfPath, binDir);

        try
        {
            GroupSetting[] sections; // in lexical order
            foreach (s; parseConfigFile(cfPath))
            {
                if (s.type == Setting.Type.group &&
                    (s.name == "default" || s.name.toCStringThen!(name => sectionMatches(name.ptr, triple))))
                {
                    sections ~= cast(GroupSetting) s;
                }
            }

            foreach (group; sections)
                validateSettingNames(group, cfPath);

            void readArraySetting(GroupSetting[] sections, string name, ref Array!(const(char)*) output)
            {
                void applyArray(const ArraySetting input)
                {
                    if (!input.isAppending)
                        output.setDim(0);

                    output.reserve(input.vals.length);
                    foreach (sw; input.vals)
                    {
                        const finalSwitch = sw.replacePlaceholders(cfgPaths).toCString;
                        output.push(finalSwitch.ptr);
                    }
                }
                findArraySetting(sections, name, &applyArray);
            }

            readArraySetting(sections, "switches", switches);
            readArraySetting(sections, "post-switches", postSwitches);
            readArraySetting(sections, "lib-dirs", _libDirs);
            const rpath = findScalarSetting(sections, "rpath");
            // A missing rpath => do nothing
            // An empty rpath => clear the setting
            if (rpath.ptr !is null)
                this.rpathcstr = rpath.length == 0 ? null : rpath.replacePlaceholders(cfgPaths).toCString.ptr;

            return true;
        }
        catch (Exception ex)
        {
            fprintf(stderr, "Error while reading config file: %s\n%.*s\n", cfPath, cast(int) ex.msg.length, ex.msg.ptr);
            return false;
        }
    }
}

unittest
{
    assert(ConfigFile.sectionMatches("i[3-6]86-.*-windows-msvc", "i686-pc-windows-msvc"));
    assert(ConfigFile.sectionMatches("86(_64)?-.*-linux", "x86_64--linux-gnu"));
    assert(!ConfigFile.sectionMatches("^linux", "x86_64--linux-gnu"));
}
