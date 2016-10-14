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

struct ConfigData {
    const char** switches_beg;
    const char** switches_end;
};

class ConfigFile {
public:
  typedef const char ** s_iterator;

public:
  ConfigFile();

  bool read(const char *explicitConfFile, const char* section);

  s_iterator switches_begin() { return data.switches_beg; }
  s_iterator switches_end() { return data.switches_end; }

  const std::string &path() { return pathstr; }

private:
  bool locate();


  std::string pathstr;
  ConfigData data;
};

#endif // LDC_DRIVER_CONFIGFILE_H
