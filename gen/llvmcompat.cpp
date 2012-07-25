#include "gen/llvmcompat.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/ADT/Triple.h"
#include <string>

#if LDC_LLVM_VER == 300
namespace llvm {
    namespace sys {
        std::string getDefaultTargetTriple() {
            return LLVM_HOSTTRIPLE;
        }
    }

    Triple Triple__get32BitArchVariant(const std::string& triple) {
        Triple T(triple);
        switch (T.getArch()) {
            case Triple::UnknownArch:
            case Triple::msp430:
                T.setArch(Triple::UnknownArch);
                break;

            case Triple::amdil:
            case Triple::arm:
            case Triple::cellspu:
            case Triple::le32:
            case Triple::mblaze:
            case Triple::mips:
            case Triple::mipsel:
            case Triple::ppc:
            case Triple::sparc:
            case Triple::tce:
            case Triple::thumb:
            case Triple::x86:
            case Triple::xcore:
                // Already 32-bit.
                break;

            case Triple::mips64:    T.setArch(Triple::mips);    break;
            case Triple::mips64el:  T.setArch(Triple::mipsel);  break;
            case Triple::ppc64:     T.setArch(Triple::ppc);   break;
            case Triple::sparcv9:   T.setArch(Triple::sparc);   break;
            case Triple::x86_64:    T.setArch(Triple::x86);     break;
        }
        return T;
    }

    Triple Triple__get64BitArchVariant(const std::string& triple) {
        Triple T(triple);
        switch (T.getArch()) {
            case Triple::UnknownArch:
            case Triple::amdil:
            case Triple::arm:
            case Triple::cellspu:
            case Triple::le32:
            case Triple::mblaze:
            case Triple::msp430:
            case Triple::tce:
            case Triple::thumb:
            case Triple::xcore:
                T.setArch(Triple::UnknownArch);
                break;

            case Triple::mips64:
            case Triple::mips64el:
            case Triple::ppc64:
            case Triple::sparcv9:
            case Triple::x86_64:
                // Already 64-bit.
                break;

            case Triple::mips:    T.setArch(Triple::mips64);    break;
            case Triple::mipsel:  T.setArch(Triple::mips64el);  break;
            case Triple::ppc:     T.setArch(Triple::ppc64);     break;
            case Triple::sparc:   T.setArch(Triple::sparcv9);   break;
            case Triple::x86:     T.setArch(Triple::x86_64);    break;
        }
        return T;
    }

}
#endif


