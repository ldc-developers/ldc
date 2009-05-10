#include <iostream>
#include <string>
#include <cassert>
#include <cstring>

#include "llvm/System/Path.h"

#include "libconfig.h++"

#include "gen/configfile.h"

#include "mars.h"

namespace sys = llvm::sys;

ConfigFile::ConfigFile()
{
    cfg = new libconfig::Config;
}

ConfigFile::~ConfigFile()
{
    delete cfg;
}

bool ConfigFile::read(const char* argv0, void* mainAddr, const char* filename)
{

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
                // 4) try next to the executable
                p = sys::Path::GetMainExecutable(argv0, mainAddr);
                p.eraseComponent();
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
            std::string binpath = sys::Path::GetMainExecutable(argv0, mainAddr).getDirname();

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

