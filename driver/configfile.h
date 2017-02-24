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

class ConfigFile {
public:
  using s_iterator = const char **;

public:

  bool read(const char *explicitConfFile, const char *section);

  s_iterator switches_begin() { return switches_b; }
  s_iterator switches_end() { return switches_e; }

  std::string path() { return std::string(pathcstr); }

private:
  bool locate(std::string& pathstr);

  // implemented in D
  bool readConfig(const char* cfPath, const char* section, const char* binDir);

  const char *pathcstr  =nullptr;
  s_iterator switches_b =nullptr;
  s_iterator switches_e =nullptr;
};

#endif // LDC_DRIVER_CONFIGFILE_H
