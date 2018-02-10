#include "pgo.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <limits>
#include <unordered_map>

#include <iostream>

#ifdef _MSC_VER
#include <intrin.h> // atomic ops
#endif

#include <llvm/IR/Module.h>
#include <llvm/ProfileData/InstrProf.h>
#include <llvm/ProfileData/InstrProfReader.h>
#include <llvm/ProfileData/InstrProfWriter.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>

#include "context.h"
#include "utils.h"

namespace {

template<typename T>
bool my_cmpxchg(T **Ptr, void *OldV, void *NewV) {
#ifdef _MSC_VER
  auto Ret = _InterlockedCompareExchangePointer(reinterpret_cast<void **>(Ptr),
                                                NewV, OldV);
  return Ret == OldV;
#else
  return __sync_bool_compare_and_swap(reinterpret_cast<void **>(Ptr),
                                      OldV, NewV);
#endif
}

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

struct ValueProfNode;
using PtrToNodeT = ValueProfNode *;
struct ValueProfNode {
#define INSTR_PROF_VALUE_NODE(Type, LLVMType, Name, Initializer) Type Name;
#include "llvm/ProfileData/InstrProfData.inc"
};

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
  std::vector<std::unique_ptr<ValueProfNode*[]>> ValueProfCounters;
  std::vector<ValueProfNode> ValueProfNodes;
  std::atomic<ValueProfNode*> ValueProfNodesStart = {nullptr};
  uint32_t VPMaxNumValsPerSite = 8; //TODO: settings

  std::vector<std::pair<std::string, uint64_t>> Symbols;
  std::unordered_map<void*, uint64_t> SymTable; //sym->hash

  void reset() {
    ProfData.reset();
    NamesString = llvm::StringRef();
    ValueProfCounters.clear();
    ValueProfNodes.clear();
    ValueProfNodesStart = nullptr;
    Symbols.clear();
    SymTable.clear();
  }

  bool empty() const {
    return ProfData.empty();
  }

  void registerProfData(__llvm_profile_data *data) {
    assert(data != nullptr);
    if (data->Values == nullptr) {
      uint64_t numVSites = 0;
      for (uint32_t kind = IPVK_First; kind <= IPVK_Last; ++kind) {
        numVSites += data->NumValueSites[kind];
      }
      if (numVSites > 0) {
        ValueProfCounters.emplace_back(llvm::make_unique<ValueProfNode*[]>(numVSites));
        data->Values = ValueProfCounters.back().get();
      }
    }
    ProfData.addAddress(data);
  }

  void registerFuncNames(const void *names, uint64_t namesSize) {
    NamesString = llvm::StringRef(static_cast<const char*>(names), namesSize);

    // Assume this is called last
    uint64_t numVSites = 0;
    for (auto &&data : ProfData.range()) {
      for (uint32_t kind = IPVK_First; kind <= IPVK_Last; ++kind) {
        numVSites += data.NumValueSites[kind];
      }
    }
    ValueProfNodes.resize(std::max(numVSites,
                                   static_cast<decltype(numVSites)>(10)));
    ValueProfNodesStart = ValueProfNodes.data();
  }

  ValueProfNode* allocNode() {
    assert(ValueProfNodesStart.load() != nullptr);
    assert(!ValueProfNodes.empty());
    auto ValueProfNodesEnd = ValueProfNodes.data() + ValueProfNodes.size();
    if (ValueProfNodesStart >= ValueProfNodesEnd) {
      // TODO: warning
      return nullptr;
    }
    auto NewNode = ValueProfNodesStart++;
    if (NewNode >= ValueProfNodesEnd) {
      return nullptr;
    }
    *NewNode = {};
    return NewNode;
  }

  void fillSymList(const llvm::Module &module) {
    assert(Symbols.empty());
    for (auto &&func : module.functions()) {
      if (func.isDeclaration()) {
        continue;
      }
      auto name = func.getName();
      Symbols.push_back({std::string(name.data()),
                         llvm::IndexedInstrProf::ComputeHash(name)});
    }
  }

  void fillSymTable(llvm::function_ref<void*(llvm::StringRef)> getter) {
    assert(SymTable.empty());
    for (auto &&sym : Symbols) {
      auto &name = sym.first;
      auto addr = getter(name);
      if (addr != nullptr) {
        auto hash = sym.second;
        SymTable.insert({addr, hash});
      }
    }
    Symbols.clear();
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
    std::vector<InstrProfValueData> tempData;
    for (auto &&data : ProfData.range()) {
      auto name = symTab.getFuncName(data.NameRef);
      auto countersPtr = static_cast<const uint64_t*>(data.CounterPtr);
      std::vector<uint64_t> counters(countersPtr, countersPtr + data.NumCounters);
      llvm::NamedInstrProfRecord record(name, data.FuncHash, std::move(counters));
      uint32_t siteOffset = 0;
      for (uint32_t kind = IPVK_First; kind <= IPVK_Last; ++kind) {
        const auto numSites = data.NumValueSites[kind];
        record.reserveSites(kind, numSites);
        for (uint32_t site = 0; site < numSites; ++site) {
          auto values = static_cast<ValueProfNode**>(data.Values);
          tempData.clear();
          auto currSite = siteOffset + site;
          if (values != nullptr && values[currSite] != nullptr) {
            for (ValueProfNode *currNode = values[currSite];
                 currNode != nullptr;
                 currNode = currNode->Next) {
              InstrProfValueData data{};
              data.Count = currNode->Count;
              if (kind == IPVK_IndirectCallTarget) {
                auto it =
                    SymTable.find(reinterpret_cast<void *>(currNode->Value));
                if (it == SymTable.end()) {
                  continue;
                }
                data.Value = it->second;
              } else {
                data.Value = currNode->Value;
              }
              tempData.emplace_back(data);
            }
          }
          record.addValueData(kind, site, tempData.data(),
                              static_cast<uint32_t>(tempData.size()), nullptr);
        }
        siteOffset += numSites;
      }

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
  getState().registerProfData(data);

  std::cerr << "\t\tCounterPtr :"      << data->CounterPtr << " "
            << "\t\tFuncHash :"        << data->FuncHash << " "
            << "\t\tFunctionPointer :" << data->FunctionPointer << " "
            << "\t\tNameRef :"         << data->NameRef << " "
            << "\t\tNumCounters :"     << data->NumCounters << " "
            << "\t\tNumValueSites :"   << data->NumValueSites[0] << "-" << data->NumValueSites[1] << " "
            << "\t\tValues          :" << data->Values << std::endl;
}

void __llvm_profile_register_names_function(void *NamesStart,
                                            uint64_t NamesSize) {
  getState().registerFuncNames(NamesStart, NamesSize);
}
void __llvm_profile_instrument_target(uint64_t TargetValue, void *Data_,
                                      uint32_t CounterIndex) {
  __llvm_profile_data *PData = static_cast<__llvm_profile_data *>(Data_);
  std::cerr << "\t\tCounterPtr :"      << PData->CounterPtr << " "
            << "\t\tFuncHash :"        << PData->FuncHash << " "
            << "\t\tFunctionPointer :" << PData->FunctionPointer << " "
            << "\t\tNameRef :"         << PData->NameRef << " "
            << "\t\tNumCounters :"     << PData->NumCounters << " "
            << "\t\tNumValueSites :"   << PData->NumValueSites[0] << "-" << PData->NumValueSites[1] << " "
            << "\t\tValues          :" << PData->Values << " "
            << "\t\tTargetValue :"     << (void*)TargetValue << " "
            << "\t\tCounterIndex :"     << CounterIndex << std::endl;

  assert(PData->Values != nullptr);

  ValueProfNode **ValueCounters = static_cast<ValueProfNode **>(PData->Values);
  ValueProfNode *PrevVNode = nullptr;
  ValueProfNode *MinCountVNode = nullptr;
  ValueProfNode *CurVNode = ValueCounters[CounterIndex];
  auto MinCount = std::numeric_limits<uint64_t>::max();

  uint8_t VDataCount = 0;
  while (CurVNode) {
    if (TargetValue == CurVNode->Value) {
      CurVNode->Count++;
      return;
    }
    if (CurVNode->Count < MinCount) {
      MinCount = CurVNode->Count;
      MinCountVNode = CurVNode;
    }
    PrevVNode = CurVNode;
    CurVNode = CurVNode->Next;
    ++VDataCount;
  }

  auto &state = getState();
  if (VDataCount >= state.VPMaxNumValsPerSite) {
    /* Bump down the min count node's count. If it reaches 0,
     * evict it. This eviction/replacement policy makes hot
     * targets more sticky while cold targets less so. In other
     * words, it makes it less likely for the hot targets to be
     * prematurally evicted during warmup/establishment period,
     * when their counts are still low. In a special case when
     * the number of values tracked is reduced to only one, this
     * policy will guarantee that the dominating target with >50%
     * total count will survive in the end. Note that this scheme
     * allows the runtime to track the min count node in an adaptive
     * manner. It can correct previous mistakes and eventually
     * lock on a cold target that is alread in stable state.
     *
     * In very rare cases,  this replacement scheme may still lead
     * to target loss. For instance, out of \c N value slots, \c N-1
     * slots are occupied by luke warm targets during the warmup
     * period and the remaining one slot is competed by two or more
     * very hot targets. If those hot targets occur in an interleaved
     * way, none of them will survive (gain enough weight to throw out
     * other established entries) due to the ping-pong effect.
     * To handle this situation, user can choose to increase the max
     * number of tracked values per value site. Alternatively, a more
     * expensive eviction mechanism can be implemented. It requires
     * the runtime to track the total number of evictions per-site.
     * When the total number of evictions reaches certain threshold,
     * the runtime can wipe out more than one lowest count entries
     * to give space for hot targets.
     */
    if (!MinCountVNode->Count || !(--MinCountVNode->Count)) {
      CurVNode = MinCountVNode;
      CurVNode->Value = TargetValue;
      CurVNode->Count++;
    }
    return;
  }

  CurVNode = state.allocNode();
  if (CurVNode == nullptr) {
    return;
  }
  CurVNode->Value = TargetValue;
  CurVNode->Count++;

  bool Success = false;
  if (ValueCounters[CounterIndex] == nullptr) {
    Success =
        my_cmpxchg(&ValueCounters[CounterIndex], nullptr, CurVNode);
  }
  else if (PrevVNode != nullptr && PrevVNode->Next == nullptr)
    Success = my_cmpxchg(&(PrevVNode->Next), nullptr, CurVNode);
  (void)Success;
}

void __llvm_profile_instrument_range(
    uint64_t TargetValue, void *Data, uint32_t CounterIndex,
    int64_t PreciseRangeStart, int64_t PreciseRangeLast, int64_t LargeValue) {

  if (LargeValue != std::numeric_limits<int64_t>::min() &&
      static_cast<int64_t>(TargetValue) >= LargeValue) {
    TargetValue = static_cast<decltype(TargetValue)>(LargeValue);
  }
  else if (static_cast<int64_t>(TargetValue) < PreciseRangeStart ||
           static_cast<int64_t>(TargetValue) > PreciseRangeLast) {
    TargetValue = static_cast<decltype(TargetValue)>(PreciseRangeLast + 1);
  }

  __llvm_profile_instrument_target(TargetValue, Data, CounterIndex);
}

int __llvm_profile_runtime = 0;

void addInstrumentationSymbols(
    const Context &context,
    std::unordered_map<std::string, void *> &symbols) {
#define PGO_SYM(name) {#name, &name}
  symbols.insert(PGO_SYM(__llvm_profile_register_function));
  symbols.insert(PGO_SYM(__llvm_profile_register_names_function));
  symbols.insert(PGO_SYM(__llvm_profile_instrument_target));
  symbols.insert(PGO_SYM(__llvm_profile_instrument_range));
  symbols.insert(PGO_SYM(__llvm_profile_runtime));
#undef PGO_SYM
}

} // anon namespace

PgoHandler::PgoHandler(const Context &context, llvm::Module &module,
                       std::unordered_map<std::string, void *> &symbols,
                       llvm::PassManagerBuilder &builder) {
  if (context.genInstrumentation) {
    addInstrumentationSymbols(context, symbols);
    builder.EnablePGOInstrGen = true;
  }
  auto &state = getState();
  if (context.useInstrumentation && !state.empty()) {
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
  state.reset();
  state.fillSymList(module);
}

PgoHandler::~PgoHandler() {
  if (!Filename.empty()) {
    llvm::sys::fs::remove(Filename);
  }
}

void bindPgoSymbols(const Context &context,
                    llvm::function_ref<void *(llvm::StringRef)> getter)
{
  if (context.genInstrumentation) {
    getState().fillSymTable(getter);
  }
}
