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


struct ConfigFile
{
public:

    alias s_iterator = const(char)**;

private:

    // representation

    const(char)* pathcstr;
    s_iterator switches_b;
    s_iterator switches_e;

    extern(C++)
    bool readConfig(const(char)* cfPath, const(char)* section, const(char)* binDir)
    {
        immutable dBinDir = prepareBinDir(binDir);
        const dSec = section[0 .. strlen(section)];

        try
        {
            auto settings = parseConfigFile(cfPath);

            ArraySetting secSwitches;
            ArraySetting defSwitches;

            foreach (s; settings)
            {
                if (s.name == dSec)
                {
                    secSwitches = findSwitches(s);
                }
                else if (s.name == "default")
                {
                    defSwitches = findSwitches(s);
                }
            }

            auto switches = secSwitches ? secSwitches : defSwitches;
            if (!switches)
            {
                const dCfPath = cfPath[0 .. strlen(cfPath)];
                throw new Exception("could not look up switches in " ~ cast(string) dCfPath);
            }

            auto finalSwitches = new const(char)*[switches.vals.length];
            foreach (i, sw; switches.vals)
            {
                const finalSwitch = sw.replace("%%ldcbinarypath%%", dBinDir) ~ '\0';
                finalSwitches[i] = finalSwitch.ptr;
            }

            switches_b = finalSwitches.ptr;
            switches_e = finalSwitches.ptr + finalSwitches.length;

            return true;
        }
        catch (Exception ex)
        {
            fprintf(stderr, "%.*s\n", ex.msg.length, ex.msg.ptr);
            return false;
        }
    }
}
