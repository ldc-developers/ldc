#include <iostream>
#include <string>
#include <cassert>
#include <cstring>

#include "llvm/System/Path.h"

#include "libconfig.h++"

#include "gen/configfile.h"

#include "mars.h"

#if _WIN32
#include <windows.h>
#undef GetCurrentDirectory
#endif

namespace sys = llvm::sys;

ConfigFile::ConfigFile()
{
    cfg = new libconfig::Config;
}

ConfigFile::~ConfigFile()
{
    delete cfg;
}

#if _WIN32
sys::Path ConfigGetExePath(sys::Path p)
{
    char buf[MAX_PATH];
    GetModuleFileName(NULL, buf, MAX_PATH);
    p = buf;
    p.eraseComponent();
    return p;
}
#endif


bool ConfigFile::read(const char* argv0, void* mainAddr, const char* filename)
{

#if _WIN32
    std::string exeDirectoryName;
#endif
    // try to find the config file

    // 1) try the current working dir
    sys::Path p = sys::Path::GetCurrentDirectory();
    p.appendComponent(filename);

    if (!p.exists())
    {
        // 2) try the user home dir
        p = sys::Path::GetUserHomeDirectory();
        p.appendComponent(filename);

        if (!p.exists())
        {
            // 3) try the install-prefix/etc
            p = sys::Path(LDC_INSTALL_PREFIX);
        #if !_WIN32
            // Does Window need something similar?
            p.appendComponent("etc");
        #endif
            p.appendComponent(filename);

            if (!p.exists())
            {
            #if _WIN32
                p = ConfigGetExePath(p);
                exeDirectoryName = p.toString();
            #else
                // 4) try next to the executable
                p = sys::Path::GetMainExecutable(argv0, mainAddr);
                p.eraseComponent();
            #endif
                p.appendComponent(filename);
                if (!p.exists())
                {        
                    // 5) fail load cfg, users still have the DFLAGS environment var
                    std::cerr << "Error failed to locate the configuration file: " << filename << std::endl;
                    return false;
                }
            }
        }
    }

    // save config file path for -v output
    pathstr = p.toString();

    try
    {
        // read the cfg
        cfg->readFile(p.c_str());

        // make sure there's a default group
        if (!cfg->exists("default"))
        {
            std::cerr << "no default settings in configuration file" << std::endl;
            return false;
        }
        libconfig::Setting& root = cfg->lookup("default");
        if (!root.isGroup())
        {
            std::cerr << "default is not a group" << std::endl;
            return false;
        }

        // handle switches
        if (root.exists("switches"))
        {
            std::string binpathkey = "%%ldcbinarypath%%";

        #if _WIN32
            std::string binpath;
            //This will happen if ldc.conf is found somewhere other than
            //beside the ldc executable
            if (exeDirectoryName == "")
            {
                sys::Path p;
                p = ConfigGetExePath(p);
                exeDirectoryName = p.toString();                
            }
            binpath = exeDirectoryName;
        #else
            std::string binpath = sys::Path::GetMainExecutable(argv0, mainAddr).getDirname();
        #endif
        

            libconfig::Setting& arr = cfg->lookup("default.switches");
            int len = arr.getLength();
            for (int i=0; i<len; i++)
            {
                std::string v = arr[i];
                
                // replace binpathkey with binpath
                size_t p;
                while (std::string::npos != (p = v.find(binpathkey)))
                    v.replace(p, binpathkey.size(), binpath);
                
                switches.push_back(strdup(v.c_str()));
            }
        }

    }
    catch(libconfig::FileIOException& fioe)
    {
        std::cerr << "Error reading configuration file: " << filename << std::endl;
        return false;
    }
    catch(libconfig::ParseException& pe)
    {
        std::cerr << "Error parsing configuration file: " << filename
            << "(" << pe.getLine() << "): " << pe.getError() << std::endl;
        return false;
    }
    catch(...)
    {
        std::cerr << "Unknown exception caught!" << std::endl;
        return false;
    }

    return true;
}

