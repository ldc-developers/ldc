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
        import std.file : exists, getcwd;
        import std.path : chainPath, dirName;
        import std.stdio : stderr;
        import std.conv : to;

        enum filename = "ldc2.conf";

        // try the current working dir
        string p = chainPath(getcwd(), filename).to!string;
        if (exists(p)) return p;

        immutable binDir = getExePathBinDir();

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
        import config.config : Config, ConfigException;
        import std.array : replace;
        import std.string : toStringz;
        import std.stdio : stderr;

        auto bindir = getExePathBinDir().replace("\\", "/");

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

            switches_b = slice.ptr;
            switches_e = slice.ptr+slice.length;

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
    const(char)* getExePathBinDirCStr();

    string getExePathBinDir()
    {
        import std.exception : assumeUnique;
        import std.string : fromStringz;

        return assumeUnique(fromStringz(getExePathBinDirCStr()));
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
        import core.sys.windows.windows : HRESULT, HWND, HANDLE, DWORD, LPSTR, LPWSTR;

        extern(Windows)
        HRESULT SHGetFolderPathA(HWND, int, HANDLE, DWORD, LPSTR);
        extern(Windows)
        HRESULT SHGetFolderPathW(HWND, int, HANDLE, DWORD, LPWSTR);

        enum {
            SHGFP_TYPE_CURRENT = 0,
            SHGFP_TYPE_DEFAULT = 1,
        }

        enum {
            CSIDL_DESKTOP            =  0,
            CSIDL_INTERNET,
            CSIDL_PROGRAMS,
            CSIDL_CONTROLS,
            CSIDL_PRINTERS,
            CSIDL_PERSONAL,
            CSIDL_FAVORITES,
            CSIDL_STARTUP,
            CSIDL_RECENT,
            CSIDL_SENDTO,
            CSIDL_BITBUCKET,
            CSIDL_STARTMENU,      // = 11
            CSIDL_MYMUSIC            = 13,
            CSIDL_MYVIDEO,        // = 14
            CSIDL_DESKTOPDIRECTORY   = 16,
            CSIDL_DRIVES,
            CSIDL_NETWORK,
            CSIDL_NETHOOD,
            CSIDL_FONTS,
            CSIDL_TEMPLATES,
            CSIDL_COMMON_STARTMENU,
            CSIDL_COMMON_PROGRAMS,
            CSIDL_COMMON_STARTUP,
            CSIDL_COMMON_DESKTOPDIRECTORY,
            CSIDL_APPDATA,
            CSIDL_PRINTHOOD,
            CSIDL_LOCAL_APPDATA,
            CSIDL_ALTSTARTUP,
            CSIDL_COMMON_ALTSTARTUP,
            CSIDL_COMMON_FAVORITES,
            CSIDL_INTERNET_CACHE,
            CSIDL_COOKIES,
            CSIDL_HISTORY,
            CSIDL_COMMON_APPDATA,
            CSIDL_WINDOWS,
            CSIDL_SYSTEM,
            CSIDL_PROGRAM_FILES,
            CSIDL_MYPICTURES,
            CSIDL_PROFILE,
            CSIDL_SYSTEMX86,
            CSIDL_PROGRAM_FILESX86,
            CSIDL_PROGRAM_FILES_COMMON,
            CSIDL_PROGRAM_FILES_COMMONX86,
            CSIDL_COMMON_TEMPLATES,
            CSIDL_COMMON_DOCUMENTS,
            CSIDL_COMMON_ADMINTOOLS,
            CSIDL_ADMINTOOLS,
            CSIDL_CONNECTIONS,  // = 49
            CSIDL_COMMON_MUSIC     = 53,
            CSIDL_COMMON_PICTURES,
            CSIDL_COMMON_VIDEO,
            CSIDL_RESOURCES,
            CSIDL_RESOURCES_LOCALIZED,
            CSIDL_COMMON_OEM_LINKS,
            CSIDL_CDBURN_AREA,  // = 59
            CSIDL_COMPUTERSNEARME  = 61,
            CSIDL_FLAG_DONT_VERIFY = 0x4000,
            CSIDL_FLAG_CREATE      = 0x8000,
            CSIDL_FLAG_MASK        = 0xFF00
        }
    }
}