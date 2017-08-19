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


ArraySetting findSwitches(Setting s)
{
    auto grp = cast(GroupSetting)s;
    if (!grp) return null;
    foreach (c; grp.children)
    {
        if (c.name == "switches")
        {
            return cast(ArraySetting)c;
        }
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

    bool readConfig(const(char)* cfPath, const(char)* section, const(char)* binDir)
    {
        switches.setDim(0);

        immutable dBinDir = prepareBinDir(binDir);
        const dSec = section[0 .. strlen(section)];

        try
        {
            auto settingSections = parseConfigFile(cfPath);

            bool sectionFound;
            ArraySetting secSwitches;
            ArraySetting defSwitches;

            foreach (s; settingSections)
            {
                if (s.name == dSec)
                {
                    sectionFound = true;
                    secSwitches = findSwitches(s);
                }
                else if (s.name == "default")
                {
                    sectionFound = true;
                    defSwitches = findSwitches(s);
                }
            }

            if (!sectionFound)
            {
                const dCfPath = cfPath[0 .. strlen(cfPath)];
                if (section)
                    throw new Exception("Could not look up section '" ~ cast(string) dSec
                                        ~ "' nor the 'default' section in " ~ cast(string) dCfPath);
                else
                    throw new Exception("Could not look up 'default' section in " ~ cast(string) dCfPath);
            }

            auto usedSwitches = secSwitches ? secSwitches : defSwitches;
            if (!usedSwitches)
            {
                const dCfPath = cfPath[0 .. strlen(cfPath)];
                throw new Exception("Could not look up switches in " ~ cast(string) dCfPath);
            }

            switches.reserve(usedSwitches.vals.length);
            foreach (i, sw; usedSwitches.vals)
            {
                const finalSwitch = sw.replace("%%ldcbinarypath%%", dBinDir) ~ '\0';
                switches.push(finalSwitch.ptr);
            }

            return true;
        }
        catch (Exception ex)
        {
            fprintf(stderr, "Error: %.*s\n", ex.msg.length, ex.msg.ptr);
            return false;
        }
    }
}
