#include "pgo.h"

#include <algorithm>
#include <cassert>

//#include <iostream>


#include <llvm/ProfileData/InstrProf.h>
#include <llvm/ProfileData/InstrProfReader.h>
#include <llvm/ProfileData/InstrProfWriter.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>

#include "context.h"
#include "utils.h"

using IntPtrT = void *;

enum ValueKind {
#define VALUE_PROF_KIND(Enumerator, Value) Enumerator = Value,
#include "llvm/ProfileData/InstrProfData.inc"
};

typedef struct alignas(INSTR_PROF_DATA_ALIGNMENT)
    __llvm_profile_data {
#define INSTR_PROF_DATA(Type, LLVMType, Name, Initializer) Type Name;
#include "llvm/ProfileData/InstrProfData.inc"
} __llvm_profile_data;

namespace {

template<typename T>
struct AddressRange {
  T *Begin = nullptr;
  T *End = nullptr;

  void addAddress(T *addr) {
    assert(addr != nullptr);
    if (Begin == nullptr) {
      assert(End == nullptr);
      Begin = addr;
      End = addr + 1;
    } else {
      Begin = std::min(Begin, addr);
      End = std::max(End, addr + 1);
      assert(End > Begin);
    }
  }

  void reset() {
    Begin = nullptr;
    End = nullptr;
  }

  llvm::ArrayRef<T> range() const {
    return llvm::ArrayRef<T>(Begin, End);
  }

  bool empty() const {
    return Begin == End;
  }
};

struct PGOState {
  AddressRange<__llvm_profile_data> ProfData;
  llvm::StringRef NamesString;

  void reset() {
    ProfData.reset();
    NamesString = llvm::StringRef();
  }

  bool empty() const {
    return ProfData.empty();
  }

  llvm::Error write(llvm::InstrProfWriter &writer) {
    assert(!empty());
    if (auto res = writer.setIsIRLevelProfile(true)) {
      return res;
    }
    llvm::InstrProfSymtab symTab;
    if (auto res = symTab.create(NamesString)) {
      return res;
    }

    llvm::Error error = llvm::Error::success();
    for (auto &&data : ProfData.range()) {
      auto name = symTab.getFuncName(data.NameRef);
      auto countersPtr = static_cast<const uint64_t*>(data.CounterPtr);
      std::vector<uint64_t> counters(countersPtr, countersPtr + data.NumCounters);
      llvm::NamedInstrProfRecord record(name, data.FuncHash, std::move(counters));
      writer.addRecord(std::move(record), [&](llvm::Error err){
        llvm::consumeError(std::move(error));
        error = std::move(err);
      });
      if (error) {
        return error;
      }
    }

    return llvm::Error::success();
  }
};

PGOState &getState() {
  static PGOState state;
  return state;
}

void __llvm_profile_register_function(void *data_) {
  assert(data_ != nullptr);
  __llvm_profile_data *data = static_cast<__llvm_profile_data *>(data_);
  getState().ProfData.addAddress(data);

//  llvm::NamedInstrProfRecord r;
//  std::cerr << "\t\tCounterPtr :"      << data->CounterPtr << " "
//            << "\t\tFuncHash :"        << data->FuncHash << " "
//            << "\t\tFunctionPointer :" << data->FunctionPointer << " "
//            << "\t\tNameRef :"         << data->NameRef << " "
//            << "\t\tNumCounters :"     << data->NumCounters << " "
//            << "\t\tNumValueSites :"   << data->NumValueSites[0] << "-" << Data->NumValueSites[1] << " "
//            << "\t\tValues          :" << data->Values << std::endl;
}

void __llvm_profile_register_names_function(void *NamesStart,
                                            uint64_t NamesSize) {
  getState().NamesString = llvm::StringRef(static_cast<const char*>(NamesStart),
                                           NamesSize);
}
void __llvm_profile_instrument_target(uint64_t TargetValue, void *Data,
                                      uint32_t CounterIndex) {

}
int __llvm_profile_runtime = 0;

void addInstrumentationSymbols(
    const Context &context,
    std::unordered_map<std::string, void *> &symbols) {
#define PGO_SYM(name) {#name, &name}
  symbols.insert(PGO_SYM(__llvm_profile_register_function));
  symbols.insert(PGO_SYM(__llvm_profile_register_names_function));
  symbols.insert(PGO_SYM(__llvm_profile_instrument_target));
  symbols.insert(PGO_SYM(__llvm_profile_runtime));
#undef PGO_SYM
}

} // anon namespace

PgoHandler::PgoHandler(const Context &context,
                       std::unordered_map<std::string, void *> &symbols,
                       llvm::PassManagerBuilder &builder) {
  if (context.genInstrumentation) {
    addInstrumentationSymbols(context, symbols);
    builder.EnablePGOInstrGen = true;
  }
  if (context.useInstrumentation && !getState().empty()) {
    llvm::InstrProfWriter writer;
    if (auto err = getState().write(writer)) {
      fatal(context, "Get PGO state :" + llvm::toString(std::move(err)));
    }

    int fd = 0;
    llvm::SmallVector<char, 64> path;
    if (auto err = llvm::sys::fs::createTemporaryFile("", "", fd, path)) {
        fatal(context, "Cannot create PGO temp file:" + err.message());
    }
    Filename = std::string(path.data(), path.size());

    llvm::raw_fd_ostream os(fd, true, false);
    writer.write(os);
    os.flush();
    if (os.has_error()) {
      llvm::sys::fs::remove(Filename);
      fatal(context, "Cannot write to PGO file");
    }

    builder.PGOInstrUse = Filename;
  }
}

PgoHandler::~PgoHandler() {
  if (!Filename.empty()) {
    llvm::sys::fs::remove(Filename);
  }
}
