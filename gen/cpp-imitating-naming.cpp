//===-- gen/cpp-imitating-naming.cpp ----------------------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include <string>
#include <cctype>

#include "driver/cl_options-llvm.h"
#include "gen/cpp-imitating-naming.h"

////////////////////////////////////////////////////////////////////////////////
namespace cl = llvm::cl;

////////////////////////////////////////////////////////////////////////////////
static cl::opt<bool>
    cppImitatingNaming("di-imitate-cpp-naming",
                       cl::desc("Imitate C++ type names for debugger"),
                       cl::ZeroOrMore);

////////////////////////////////////////////////////////////////////////////////

std::string replaceAllGeneric(const std::string &haystack,
                              const std::string &search,
                              const std::string &replace) {
  size_t index = 0;

  size_t lengthSearch = search.size();
  size_t lengthReplace = replace.size();

  std::string result = haystack;

  while (true) {
    index = result.find(search, index);

    if (index == std::string::npos) {
      break;
    }

    result.replace(index, lengthSearch, replace);
    index += lengthReplace;
  }

  return result;
}

////////////////////////////////////////////////////////////////////////////////

std::string convertSingleDTemplate(const std::string &haystack) {
  size_t start = 0;
  size_t index = haystack.find('!');

  std::string result;

  if (index != std::string::npos) {
    result += haystack.substr(start, index);
    result += '<';

    start = index + 1;

    if (haystack[index + 1] == '(') {
      start++;
      index++;

      size_t enclosedParentheses = 0;

      while (true) {
        switch (haystack[index]) {
        case '(':
          enclosedParentheses++;
          break;
        case ')':
          enclosedParentheses--;
          break;
        }

        if (enclosedParentheses == 0) {
          break;
        }

        if (index == haystack.size() - 1) {
          break;
        }

        index++;
      }

      if (haystack[index] != ')') {
        index++;
      }
    } else {
      index = haystack.find_first_of(" ,.:<>()[]", start);
    }

    result += haystack.substr(start, index - start);
    result += '>';

    start = index;

    if (haystack[start] == ')') {
      start++;
    }
  }

  if (start != std::string::npos) {
    result += haystack.substr(start);
  }

  return result;
}

////////////////////////////////////////////////////////////////////////////////

std::string convertSingleDArray(const std::string &haystack) {
  size_t index = haystack.find('[');

  while (index != std::string::npos && std::isdigit(haystack[index + 1])) {
    index = haystack.find('[', index + 1);
  }

  std::string result = haystack;

  if (index != std::string::npos && index > 0) {
    index--;

    if (haystack[index] == '>' || haystack[index] == ')') {
      size_t enclosedParentheses = 0;
      size_t enclosedChevrons = 0;

      while (true) {
        switch (haystack[index]) {
        case '(':
          enclosedParentheses++;
          break;
        case ')':
          enclosedParentheses--;
          break;
        case '>':
          enclosedChevrons++;
          break;
        case '<':
          enclosedChevrons--;
          break;
        }

        if (enclosedParentheses == 0 && enclosedChevrons == 0) {
          break;
        }

        if (index == 0) {
          break;
        }

        index--;
      }
    }

    if (index > 0) {
      index = haystack.find_last_of(" ,<(", index - 1);
      index = index != std::string::npos ? index + 1 : 0;
    }

    size_t bracketsStart = haystack.find('[', index);
    size_t bracketsIndex = bracketsStart;

    size_t enclosedSquareBrackets = 0;

    while (true) {
      switch (haystack[bracketsIndex]) {
      case '[':
        enclosedSquareBrackets++;
        break;
      case ']':
        enclosedSquareBrackets--;
        break;
      }

      if (enclosedSquareBrackets == 0) {
        break;
      }

      if (bracketsIndex == haystack.size() - 1) {
        break;
      }

      bracketsIndex++;
    }

    bracketsIndex++;

    size_t lengthSearch = bracketsIndex - index;
    std::string search = haystack.substr(index, lengthSearch);

    size_t lengthValue = bracketsStart - index;
    std::string value = haystack.substr(index, lengthValue);

    if (haystack[bracketsIndex - 1] == ']') {
      bracketsIndex--;
    }

    bracketsStart++;

    size_t lengthKey = bracketsIndex - bracketsStart;
    std::string key = haystack.substr(bracketsStart, lengthKey);

    if (key.empty()) {
      std::string replace = "slice<" + value + ">";

      result = replaceAllGeneric(haystack, search, replace);
    } else {
      std::string pairKeyValue = key + ", " + value;
      std::string replace = "associative_array<" + pairKeyValue + ">";

      result = replaceAllGeneric(haystack, search, replace);
    }
  }

  return result;
}

////////////////////////////////////////////////////////////////////////////////

std::string convertDTemplate(const std::string &haystack) {
  const size_t haystackLength = haystack.size();

  std::string result = haystack;

  size_t resultLength = result.size();
  size_t previousLength = 0;

  do {
    previousLength = resultLength;

    result = convertSingleDTemplate(result);
    resultLength = result.size();
  } while (resultLength != previousLength);

  return result;
}

////////////////////////////////////////////////////////////////////////////////

std::string convertDArray(const std::string &haystack) {
  const size_t haystackLength = haystack.size();

  std::string result = haystack;

  size_t resultLength = result.size();
  size_t previousLength = 0;

  do {
    previousLength = resultLength;

    result = convertSingleDArray(result);
    resultLength = result.size();
  } while (resultLength != previousLength);

  return result;
}

////////////////////////////////////////////////////////////////////////////////

std::string convertDTypeName(const std::string &originalName) {
  std::string result;

  result = replaceAllGeneric(originalName, ".", "::");
  result = convertDTemplate(result);
  result = convertDArray(result);

  return result;
}

////////////////////////////////////////////////////////////////////////////////

std::string processDITypeName(const std::string &originalName) {
  return cppImitatingNaming ? convertDTypeName(originalName)
                                  : originalName;
}
