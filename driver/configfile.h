//===-- driver/configfile.h - LDC config file handling ----------*- C++ -*-===//
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

#ifndef LDC_CONF_CONFIGFILE_H
#define LDC_CONF_CONFIGFILE_H

#include <vector>
#include <string>

namespace libconfig
{
    class Config;
}

class ConfigFile
{
public:
    typedef std::vector<const char*>    s_vector;
    typedef s_vector::iterator          s_iterator;

public:
    ConfigFile();
    ~ConfigFile();

    bool read(const char* argv0, void* mainAddr, const char* filename);

    s_iterator switches_begin()   { return switches.begin(); }
    s_iterator switches_end()     { return switches.end(); }

    const std::string& path()     { return pathstr; }

private:
    bool locate(llvm::sys::Path& path, const char* argv0, void* mainAddr, const char* filename);

    libconfig::Config* cfg;
    std::string pathstr;

    s_vector switches;
};

#endif // LDC_CONF_CONFIGFILE_H
