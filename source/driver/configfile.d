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

import ddmd.root.array;
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


ArraySetting findArraySetting(GroupSetting section, string name)
{
    if (!section) return null;
    foreach (c; section.children)
    {
        if (c.type == Setting.Type.array && c.name == name)
            return cast(ArraySetting) c;
    }
    return null;
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
private:

    // representation

    const(char)* pathcstr;
    Array!(const(char)*) switches;
    Array!(const(char)*) postSwitches;

    bool readConfig(const(char)* cfPath, const(char)* sectionName, const(char)* binDir)
    {
        switches.setDim(0);
        postSwitches.setDim(0);

        immutable dBinDir = prepareBinDir(binDir);
        const dSec = sectionName[0 .. strlen(sectionName)];

        try
        {
            GroupSetting section, defaultSection;
            foreach (s; parseConfigFile(cfPath))
            {
                if (s.type != Setting.Type.group)
                    continue;
                if (s.name == dSec)
                    section = cast(GroupSetting) s;
                else if (s.name == "default")
                    defaultSection = cast(GroupSetting) s;
            }

            if (!section && !defaultSection)
            {
                const dCfPath = cfPath[0 .. strlen(cfPath)];
                if (sectionName)
                    throw new Exception("Could not look up section '" ~ cast(string) dSec
                                        ~ "' nor the 'default' section in " ~ cast(string) dCfPath);
                else
                    throw new Exception("Could not look up 'default' section in " ~ cast(string) dCfPath);
            }

            ArraySetting findArray(string name)
            {
                auto r = findArraySetting(section, name);
                if (!r)
                    r = findArraySetting(defaultSection, name);
                return r;
            }

            auto switches = findArray("switches");
            auto postSwitches = findArray("post-switches");
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

            return true;
        }
        catch (Exception ex)
        {
            fprintf(stderr, "Error: %.*s\n", ex.msg.length, ex.msg.ptr);
            return false;
        }
    }
}
