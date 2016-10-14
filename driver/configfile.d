module driver.configfile;

import config.config : Config, ConfigException;

import std.algorithm : map;
import std.string : fromStringz, toStringz;
import std.stdio : stderr;
import std.array : replace;

import core.stdc.stdlib;

/// Struct that hold data fetched from the config file
extern (C++)
struct ConfigData {
    const(char)** switches_beg;
    const(char)** switches_end;
}

/// Read data from the config file
/// Returns a boolean indicating if data was succesfully read.
/// Resulting data is stored in data.
extern (C++)
bool readDataFromConfigFile (   const(char)* pathcstr,
                                const(char)* sectioncstr,
                                const(char)* bindircstr,
                                out ConfigData data)
in
{
    assert(pathcstr);
    assert(bindircstr);
}
body
{
    auto path = fromStringz(pathcstr).idup;
    auto section = sectioncstr ? fromStringz(sectioncstr).idup : "default";
    auto bindir = fromStringz(bindircstr).replace("\\", "/");

    try
    {
        auto conf = Config.readFile(path);

        auto setting = conf.lookUp(section~".switches");
        if (!setting && section != "default")
        {
            section = "default";
            setting = conf.lookUp(section~".switches");
        }

        if (!setting)
        {
            stderr.writeln("could not look up setting \""~section~".switches\" in config file "~path);
            return false;
        }

        auto switches = setting.asArray;
        if (!switches)
        {
            stderr.writeln("ill-formed config file "~path~":\n\""~section~".switches\" should be an array.");
            return false;
        }

        auto slice = new const(char)*[switches.children.length];
        foreach (i, sw; switches.children)
        {
            auto swstr = sw.asScalar.value!string;
            slice[i] = toStringz(swstr.replace("%%ldcbinarypath%%", bindir));
        }

        data.switches_beg = slice.ptr;
        data.switches_end = slice.ptr+slice.length;

        return true;
    }
    catch (ConfigException ex)
    {
        stderr.writeln("could not read switches from config file \""~path~"\":\n"~ex.msg);
        return false;
    }
    catch (Exception ex)
    {
        stderr.writeln("unexpected error while reading config file \""~path~"\":\n"~ex.msg);
        return false;
    }
}

