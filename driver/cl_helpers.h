//===-- gen/cl_helpers.h - Command line processing helpers ------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Helpers to augment the LLVM command line parsing library with some extra
// functionality.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "dmd/globals.h" // for CHECKENABLE enum
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"

template <typename TYPE> struct Array;
typedef Array<const char *> Strings;

namespace opts {
namespace cl = llvm::cl;

/// Duplicate the string (incl. null-termination) and replace '/' with '\' on
/// Windows.
DString dupPathString(llvm::StringRef src);

/// Helper function to handle -of, -od, etc.
DString fromPathString(const cl::opt<std::string> &src);

/// Helper class to determine values
template <class DT> struct FlagParserDataType {};

template <> struct FlagParserDataType<bool> {
  static bool true_val() { return true; }
  static bool false_val() { return false; }
};

template <> struct FlagParserDataType<cl::boolOrDefault> {
  static cl::boolOrDefault true_val() { return cl::BOU_TRUE; }
  static cl::boolOrDefault false_val() { return cl::BOU_FALSE; }
};

inline bool getFlagOrDefault(cl::boolOrDefault value, bool defaultValue) {
  return value == cl::BOU_UNSET ? defaultValue : value == cl::BOU_TRUE;
}

template <> struct FlagParserDataType<CHECKENABLE> {
  static CHECKENABLE true_val() { return CHECKENABLEon; }
  static CHECKENABLE false_val() { return CHECKENABLEoff; }
};

template <class DataType> class FlagParser : public cl::generic_parser_base {
protected:
  llvm::SmallVector<std::pair<std::string, DataType>, 2> switches;
  cl::Option &owner() const { return Owner; }

public:
  FlagParser(cl::Option &O) : generic_parser_base(O) {}
  typedef DataType parser_data_type;

  void initialize() {
    std::string Name(owner().ArgStr);
    switches.push_back(
        make_pair("enable-" + Name, FlagParserDataType<DataType>::true_val()));
    switches.push_back(make_pair("disable-" + Name,
                                 FlagParserDataType<DataType>::false_val()));
    // Replace <foo> with -enable-<foo> and register -disable-<foo>.
    // A literal option can only registered if the argstr is empty -
    // just do this first.
    owner().setArgStr("");
    AddLiteralOption(Owner, strdup(switches[1].first.data()));
    owner().setArgStr(switches[0].first.data());
  }

  enum cl::ValueExpected getValueExpectedFlagDefault() const {
    return cl::ValueOptional;
  }

  // Implement virtual functions needed by generic_parser_base
  unsigned getNumOptions() const override { return 0; }

  llvm::StringRef getOption(unsigned N) const override {
    llvm_unreachable("Unexpected call");
    return "";
  }

  llvm::StringRef getDescription(unsigned N) const override {
    llvm_unreachable("Unexpected call");
    return "";
  }

private:
  struct OptionValue : cl::OptionValueBase<DataType, false> {
    OptionValue(){};
  };
  const OptionValue EmptyOptionValue;

public:
  // getOptionValue - Return the value of option name N.
  const cl::GenericOptionValue &getOptionValue(unsigned N) const override {
    return EmptyOptionValue;
  }

  // parse - Return true on error.
  bool parse(cl::Option &O, llvm::StringRef ArgName, llvm::StringRef Arg,
             DataType &Val) {
    for (const auto &pair : switches) {
      const auto &name = pair.first;
      if (name == ArgName || (name.size() < ArgName.size() &&
                              ArgName.substr(0, name.size()) == name &&
                              ArgName[name.size()] == '=')) {
        if (!parse(owner(), Arg, Val)) {
          Val = (Val == pair.second)
                    ? FlagParserDataType<DataType>::true_val()
                    : FlagParserDataType<DataType>::false_val();
          return false;
        }
        // Invalid option value
        break;
      }
    }
    return true;
  }

  void getExtraOptionNames(llvm::SmallVectorImpl<llvm::StringRef> &Names) {
    for (auto I = switches.begin() + 1, E = switches.end(); I != E; ++I) {
      Names.push_back(I->first.data());
    }
  }

private:
  static bool parse(cl::Option &O, llvm::StringRef Arg, DataType &Val) {
    if (Arg == "" || Arg == "true" || Arg == "TRUE" || Arg == "True" ||
        Arg == "1") {
      Val = FlagParserDataType<DataType>::true_val();
      return false;
    }

    if (Arg == "false" || Arg == "FALSE" || Arg == "False" || Arg == "0") {
      Val = FlagParserDataType<DataType>::false_val();
      return false;
    }
    return O.error("'" + Arg +
                   "' is invalid value for boolean argument! Try 0 or 1");
  }
};

/// Helper class for options that set multiple flags
class MultiSetter {
  std::vector<CHECKENABLE *> locations;
  bool invert;
  explicit MultiSetter(bool); // not implemented, disable auto-conversion
public:
  // end with a nullptr
  MultiSetter(bool invert, CHECKENABLE *p, ...);
  MultiSetter() = default;

  void operator=(bool val);
};

/// Helper class to fill Strings with char* when given strings
/// (Errors on empty strings)
class StringsAdapter {
  const char *name;
  Strings *arrp;

public:
  StringsAdapter(const char *name_, Strings &arr) {
    name = name_;
    arrp = &arr;
    assert(name);
  }

  void push_back(const char *cstr);

  void push_back(const std::string &str) { push_back(str.c_str()); }
};
}
