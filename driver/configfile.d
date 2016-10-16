module driver.configfile;

import config.config : Config, ConfigException;

import std.algorithm : map;
import std.string : fromStringz, toStringz;
import std.stdio : stderr;
import std.array : replace;
import std.file : exists;


extern(C++)
const(char)* getExePathBinDirCStr();


struct ConfigFile
{
public:

    alias s_iterator = const(char)**;

    /// Read data from the config file
    /// Returns a boolean indicating if data was succesfully read.
    extern(C++)
    bool read(const(char)* explicitConfFile, const(char)* section)
    {
        // explicitly provided by user in command line?
        if (explicitConfFile)
        {
            auto cfPath = fromStringz(explicitConfFile);

            // treat an empty path (`-conf=`) as missing command-line option,
            // defaulting to an auto-located config file, analogous to DMD
            if (cfPath.length && !exists(cfPath))
            {
                stderr.writefln("Warning: configuration file '%s' not found, falling " ~
                    "back to default", cfPath);
                cfPath = null;
            }
            else if (cfPath.length)
            {
                pathcstr = toStringz(cfPath);
            }
        }

        // locate file automatically if path is not set yet
        if (!pathcstr)
        {
            if (!locate()) return false;
        }

        // retrieve data from config file
        return readConfig(fromStringz(section).idup);
    }

private:

    // impl in C++
    extern(C++) bool locate();

    bool readConfig(string section)
    {
        auto cfPath = fromStringz(pathcstr).idup;
        auto bindir = fromStringz(getExePathBinDirCStr()).replace("\\", "/");

        try
        {
            auto conf = Config.readFile(cfPath);

            auto setting = conf.lookUp(section~".switches");
            if (!setting && section != "default")
            {
                section = "default";
                setting = conf.lookUp(section~".switches");
            }

            if (!setting)
            {
                stderr.writeln("could not look up setting \""~section~".switches\" in config file "~cfPath);
                return false;
            }

            auto switches = setting.asArray;
            if (!switches)
            {
                stderr.writeln("ill-formed config file "~cfPath~":\n\""~section~".switches\" should be an array.");
                return false;
            }

            auto slice = new const(char)*[switches.children.length];
            foreach (i, sw; switches.children)
            {
                auto swstr = sw.asScalar.value!string;
                slice[i] = toStringz(swstr.replace("%%ldcbinarypath%%", bindir));
            }

            switches_beg = slice.ptr;
            switches_end = slice.ptr+slice.length;

            return true;
        }
        catch (ConfigException ex)
        {
            stderr.writeln("could not read switches from config file \""~cfPath~"\":\n"~ex.msg);
            return false;
        }
        catch (Exception ex)
        {
            stderr.writeln("unexpected error while reading config file \""~cfPath~"\":\n"~ex.msg);
            return false;
        }
    }

    const(char)* pathcstr;
    s_iterator switches_beg;
    s_iterator switches_end;
}

