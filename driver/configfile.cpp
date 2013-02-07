//===-- configfile.cpp ----------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "driver/configfile.h"
#include "mars.h"
#include "libconfig.h++"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include <cassert>
#include <cstring>
#include <iostream>
#include <string>

namespace sys = llvm::sys;

ConfigFile::ConfigFile()
{
    cfg = new libconfig::Config;
}

ConfigFile::~ConfigFile()
{
   // delete cfg;
}


bool ConfigFile::locate(sys::Path& p, const char* argv0, void* mainAddr, const char* filename)
{
    // temporary configuration

    // try the current working dir
    p = sys::Path::GetCurrentDirectory();
    p.appendComponent(filename);
    if (sys::fs::exists(p.str()))
        return true;

    // try next to the executable
    p = sys::Path::GetMainExecutable(argv0, mainAddr);
    p.eraseComponent();
    p.appendComponent(filename);
    if (sys::fs::exists(p.str()))
        return true;

    // user configuration

    // try ~/.ldc
    p = sys::Path::GetUserHomeDirectory();
    p.appendComponent(".ldc");
    p.appendComponent(filename);
    if (sys::fs::exists(p.str()))
        return true;

#if _WIN32
    // try home dir
    p = sys::Path::GetUserHomeDirectory();
    p.appendComponent(filename);
    if (sys::fs::exists(p.str()))
        return true;
#endif

    // system configuration

    // try in etc relative to the executable: exe\..\etc
    // do not use .. in path because of security risks
    p = sys::Path::GetMainExecutable(argv0, mainAddr);
    p.eraseComponent();
    p.eraseComponent();
    if (!p.isEmpty())
    {
        p.appendComponent("etc");
        p.appendComponent(filename);
        if (sys::fs::exists(p.str()))
            return true;
    }

#if _WIN32
    // try the install-prefix
    p = sys::Path(LDC_INSTALL_PREFIX);
    p.appendComponent(filename);
    if (sys::fs::exists(p.str()))
        return true;
#else
    // try the install-prefix/etc
    p = sys::Path(LDC_INSTALL_PREFIX);
    p.appendComponent("etc");
    p.appendComponent(filename);
    if (sys::fs::exists(p.str()))
        return true;

    // try the install-prefix/etc/ldc
    p = sys::Path(LDC_INSTALL_PREFIX);
    p.appendComponent("etc");
    p.appendComponent("ldc");
    p.appendComponent(filename);
    if (sys::fs::exists(p.str()))
        return true;

    // try /etc (absolute path)
    p = sys::Path("/etc");
    p.appendComponent(filename);
    if (sys::fs::exists(p.str()))
        return true;

    // try /etc/ldc (absolute path)
    p = sys::Path("/etc/ldc");
    p.appendComponent(filename);
    if (sys::fs::exists(p.str()))
        return true;
#endif

    return false;
}

bool ConfigFile::read(const char* argv0, void* mainAddr, const char* filename)
{
    sys::Path p;
    if (!locate(p, argv0, mainAddr, filename))
    {
        // failed to find cfg, users still have the DFLAGS environment var
        std::cerr << "Error failed to locate the configuration file: " << filename << std::endl;
        return false;
    }

    // save config file path for -v output
    pathstr = p.str();

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

            std::string binpath = sys::path::parent_path(sys::Path::GetMainExecutable(argv0, mainAddr).str());

            libconfig::Setting& arr = cfg->lookup("default.switches");
            int len = arr.getLength();
            for (int i=0; i<len; i++)
            {
                std::string v = arr[i].operator std::string();

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

