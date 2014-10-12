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

#ifndef LDC_DRIVER_CONFIGFILE_H
#define LDC_DRIVER_CONFIGFILE_H

#include <string>
#include <vector>
#include "llvm/ADT/SmallString.h"
#include "libconfig.h"

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
    bool locate(llvm::SmallString<128> &path, const char* argv0, void* mainAddr, const char* filename);

    config_t* cfg;
    std::string pathstr;

    s_vector switches;
};

#endif // LDC_DRIVER_CONFIGFILE_H
