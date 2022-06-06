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

import dmd.root.array;
import driver.config;
import core.stdc.stdio;
import core.stdc.string;


string prepareBinDir(const(char)* binDir)
{
    immutable len = strlen(binDir);
    auto res = binDir[0 .. len].dup;
    foreach (ref c; res)
    {
        if (c == '\\') c = '/';
    }
    return cast(string)res; // assumeUnique
}

T findSetting(T)(GroupSetting[] sections, Setting.Type type, string name)
{
    // lexically later sections dominate earlier ones
    foreach_reverse (section; sections)
    {
        foreach (c; section.children)
        {
            if (c.type == type && c.name == name)
                return cast(T) c;
        }
    }
    return null;
}

ArraySetting findArraySetting(GroupSetting[] sections, string name)
{
    return findSetting!ArraySetting(sections, Setting.Type.array, name);
}

ScalarSetting findScalarSetting(GroupSetting[] sections, string name)
{
    return findSetting!ScalarSetting(sections, Setting.Type.scalar, name);
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

    static bool sectionMatches(const(char)* section, const(char)* triple);

    bool readConfig(const(char)* cfPath, const(char)* triple, const(char)* binDir)
    {
        switches.setDim(0);
        postSwitches.setDim(0);

        immutable dBinDir = prepareBinDir(binDir);

        try
        {
            GroupSetting[] sections; // in lexical order
            foreach (s; parseConfigFile(cfPath))
            {
                if (s.type == Setting.Type.group &&
                    (s.name == "default" || sectionMatches((s.name ~ '\0').ptr, triple)))
                {
                    sections ~= cast(GroupSetting) s;
                }
            }

            if (sections.length == 0)
            {
                const dTriple = triple[0 .. strlen(triple)];
                const dCfPath = cfPath[0 .. strlen(cfPath)];
                throw new Exception("No matching section for triple '" ~ cast(string) dTriple
                                    ~ "' in " ~ cast(string) dCfPath);
            }

            auto switches = findArraySetting(sections, "switches");
            auto postSwitches = findArraySetting(sections, "post-switches");
            if (!switches && !postSwitches)
            {
                const dCfPath = cfPath[0 .. strlen(cfPath)];
                throw new Exception("Could not look up switches in " ~ cast(string) dCfPath);
            }

            void applyArray(ref Array!(const(char)*) output, ArraySetting input)
            {
                if (!input)
                    return;

                output.reserve(input.vals.length);
                foreach (sw; input.vals)
                {
                    const finalSwitch = sw.replace("%%ldcbinarypath%%", dBinDir) ~ '\0';
                    output.push(finalSwitch.ptr);
                }
            }

            applyArray(this.switches, switches);
            applyArray(this.postSwitches, postSwitches);

            auto libDirs = findArraySetting(sections, "lib-dirs");
            applyArray(_libDirs, libDirs);

            if (auto rpath = findScalarSetting(sections, "rpath"))
                this.rpathcstr = (rpath.val.replace("%%ldcbinarypath%%", dBinDir) ~ '\0').ptr;

            return true;
        }
        catch (Exception ex)
        {
            fprintf(stderr, "Error: %.*s\n", cast(int) ex.msg.length, ex.msg.ptr);
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
