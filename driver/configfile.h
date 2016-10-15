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
  typedef const char **s_iterator;

public:
  ConfigFile();

  /// Read data from the config file
  /// Returns a boolean indicating if data was succesfully read.
  bool read(const char *explicitConfFile, const char *section);

  s_iterator switches_begin() { return switches_b; }
  s_iterator switches_end() { return switches_e; }

  std::string path() { return std::string(pathcstr); }

private:
  bool locate();

  // impl in D
  bool readConfig(const char *sectioncstr, const char *bindircstr);

  const char *pathcstr;
  s_iterator switches_b;
  s_iterator switches_e;
};

#endif // LDC_DRIVER_CONFIGFILE_H
