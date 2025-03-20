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

const(string)[] findArraySetting(GroupSetting[] sections, string name)
{
    const(string)[] result = null;
    foreach (section; sections)
    {
        foreach (c; section.children)
        {
            if (c.type == Setting.Type.array && c.name == name)
            {
                auto as = cast(ArraySetting) c;
                if (as.isAppending)
                    result ~= as.vals;
                else
                    result = as.vals;
            }
        }
    }
    return result;
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

            if (sections.length == 0)
            {
                throw new Exception("No matching section for triple '" ~ cast(string) triple.toDString
                                    ~ "'");
            }

            const switches = findArraySetting(sections, "switches");
            const postSwitches = findArraySetting(sections, "post-switches");
            if (switches.length + postSwitches.length == 0)
                throw new Exception("Could not look up switches");

            void applyArray(ref Array!(const(char)*) output, const(string)[] input)
            {
                output.setDim(0);

                output.reserve(input.length);
                foreach (sw; input)
                {
                    const finalSwitch = sw.replacePlaceholders(cfgPaths).toCString;
                    output.push(finalSwitch.ptr);
                }
            }

            applyArray(this.switches, switches);
            applyArray(this.postSwitches, postSwitches);

            const libDirs = findArraySetting(sections, "lib-dirs");
            applyArray(_libDirs, libDirs);

            const rpath = findScalarSetting(sections, "rpath");
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
