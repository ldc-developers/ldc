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


struct ConfigFile
{
public:

    alias s_iterator = const(char)**;

    /// Read data from the config file
    /// Returns a boolean indicating if data was succesfully read.
    extern(C++)
    bool read(const(char)* explicitConfFile, const(char)* section)
    {
        import std.string : fromStringz, toStringz;
        import std.file : exists;
        import std.stdio : stderr;

        string cfPath;

        // explicitly provided by user in command line?
        if (explicitConfFile)
        {
            cfPath = fromStringz(explicitConfFile).idup;

            // treat an empty path (`-conf=`) as missing command-line option,
            // defaulting to an auto-located config file, analogous to DMD
            if (cfPath.length && !exists(cfPath))
            {
                stderr.writefln("Warning: configuration file '%s' not found, falling " ~
                    "back to default", cfPath);
                cfPath = null;
            }
        }

        // locate file automatically if path is not set yet
        if (!cfPath.length)
        {
            cfPath = locate();
            if (!cfPath.length) return false;
        }

        pathcstr = toStringz(cfPath);

        // retrieve data from config file
        return readConfig(cfPath, fromStringz(section).idup);
    }

private:

    // representation

    const(char)* pathcstr;
    s_iterator switches_b;
    s_iterator switches_e;


    // private methods

    string locate()
    {
        import exePath = driver.exe_path;
        import std.file : exists, getcwd;
        import std.path : chainPath, dirName;
        import std.stdio : stderr;
        import std.conv : to;

        enum filename = "ldc2.conf";

        // try the current working dir
        string p = chainPath(getcwd(), filename).to!string;
        if (exists(p)) return p;

        immutable binDir = exePath.binDir;

        // try next to the executable
        p = chainPath(binDir, filename).to!string;
        if (exists(p)) return p;

        // user configuration
        immutable home = getUserHomeDirectory();

        // try ~/.ldc
        p = chainPath(home, ".ldc", filename).to!string;
        if (exists(p)) return p;

        version (Windows)
        {
            // try home dir
            p = chainPath(home, filename).to!string;
            if (exists(p)) return p;
        }

        // system configuration
        // try in etc relative to the executable: exe\..\etc
        // do not use .. in path because of security risks
        p = chainPath(dirName(binDir), filename).to!string;
        if (exists(p)) return p;

        version(Windows)
        {
            // try reading path from the registry
            p = readPathFromRegistry();
            if (p) {
                p = chainPath(p, "etc", filename).to!string;
                if (exists(p)) return p;
            }
        }
        else
        {
            immutable lip = getLdcInstallPrefix();

            // try install-prefix/etc
            p = chainPath(lip, "etc", filename).to!string;
            if (exists(p)) return p;

            // try install-prefix/etc/ldc
            p = chainPath(lip, "etc", "ldc", filename).to!string;
            if (exists(p)) return p;

            // try /etc (absolute path)
            p = chainPath("/etc", filename).to!string;
            if (exists(p)) return p;

            // try /etc/ldc (absolute path)
            p = chainPath("/etc/ldc", filename).to!string;
            if (exists(p)) return p;
        }

        stderr.writefln("Warning, failed to locate the configuration file %s\n", filename);
        return null;
    }


    bool readConfig(string cfPath, string section)
    {
        import exePath = driver.exe_path;
        import std.json : parseJSON, JSONValue, JSON_TYPE, JSONException;
        import std.array : replace, uninitializedArray;
        import std.string : toStringz;
        import std.stdio : stderr;
        import std.file : read;

        immutable binDir = exePath.binDir.replace("\\", "/");

        try
        {
            auto confStr = cast(string)read(cfPath);
            auto conf = parseJSON(confStr);
            JSONValue group;
            JSONValue switchesVal;
            JSONValue[] switchesArr;

            try
            {
                group = conf[section];
            }
            catch(JSONException ex)
            {
                if (section != "default")
                {
                    section = "default";
                    group = conf["default"];
                }
                else
                {
                    throw ex;
                }
            }

            try
            {
                switchesVal = group["switches"];
            }
            catch(JSONException)
            {
                stderr.writeln("could not look up setting \""~section~".switches\" in config file "~cfPath);
                return false;
            }

            if (switchesVal.type != JSON_TYPE.ARRAY)
            {
                stderr.writeln("ill-formed config file "~cfPath~":\n\""~section~".switches\" should be an array.");
                return false;
            }

            switchesArr = switchesVal.array;

            auto slice = uninitializedArray!(const(char)*[])(switchesArr.length);
            foreach (i, sw; switchesArr)
            {
                if (sw.type != JSON_TYPE.STRING)
                {
                    stderr.writeln("ill-formed config file "~cfPath~":\n\""~section~".switches\"" ~
                            " should be an array of strings.");
                    return false;
                }
                slice[i] = toStringz(sw.str.replace("%%ldcbinarypath%%", binDir));
            }

            switches_b = slice.ptr;
            switches_e = slice.ptr+slice.length;

            return true;
        }
        catch (JSONException ex)
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
}



private {


    string getUserHomeDirectory()
    {
        version (Windows)
        {
            import core.sys.windows.windows : S_OK;
            import std.string : fromStringz;
            import std.conv : to;

            char[1024] buf;
            auto res = SHGetFolderPathA(null, CSIDL_FLAG_CREATE | CSIDL_APPDATA,
                    null, SHGFP_TYPE_CURRENT, buf.ptr);
            assert(res == S_OK, "Failed to get user home directory");
            return fromStringz(buf.ptr).idup;
        }
        else
        {
            import std.process : environment;

            return environment.get("HOME", "/");
        }
    }

    version(Windows)
    {
        string readPathFromRegistry()
        {
            import std.windows.registry : Registry;

            scope(failure) return null;
            auto HKLM = Registry.localMachine;
            // FIXME: version number should be obtained from CMake
            auto key = HKLM.getKey(`SOFTWARE\ldc-developers\LDC\0.11.0`);
            return key.getValue("Path").value_SZ;
        }
    }


    extern(C++)
    const(char)* getLdcInstallPrefixCStr();

    string getLdcInstallPrefix()
    {
        import std.exception : assumeUnique;
        import std.string : fromStringz;

        return assumeUnique(fromStringz(getLdcInstallPrefixCStr()));
    }


    version(Windows)
    {
        // bindings missing in LDC LTS
        import core.sys.windows.windows : HRESULT, HWND, HANDLE, DWORD, LPSTR;

        extern(Windows)
        HRESULT SHGetFolderPathA(HWND, int, HANDLE, DWORD, LPSTR);

        enum {
            SHGFP_TYPE_CURRENT = 0,
        }
        enum {
            CSIDL_APPDATA       = 26,
            CSIDL_FLAG_CREATE   = 0x8000,
        }
    }
}
