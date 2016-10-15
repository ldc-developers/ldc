module driver.configfile;

import config.config : Config, ConfigException;

import std.algorithm : map;
import std.string : fromStringz, toStringz;
import std.stdio : stderr;
import std.array : replace;


extern(C++)
struct ConfigFile
{
public:

    alias s_iterator = const(char)**;

    // impl in C++
    final bool read(const char *explicitConfFile, const char *section);

private:

    // impl in C++
    final bool locate();

    final bool readConfig(const(char)* sectioncstr, const(char)* bindircstr)
    in
    {
        assert(this.pathcstr);
        assert(bindircstr);
    }
    body
    {
        auto path = fromStringz(this.pathcstr).idup;
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

            switches_beg = slice.ptr;
            switches_end = slice.ptr+slice.length;

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

    const(char)* pathcstr;
    s_iterator switches_beg;
    s_iterator switches_end;
}

