ifdef IN_LDC
# need OS for the conditions below
include ../../../../dmd/osmodel.mak
endif

TESTS := importc_compare

# FIXME: fails on Alpine v3.21 with conflicting struct declarations in the C headers:
# /usr/include/asm-generic/fcntl.h(195): Error: struct `importc_includes.flock` conflicts with struct `importc_includes.flock` at /usr/include/fcntl.h(24)
ifdef IS_MUSL
    TESTS :=
endif

# FIXME: fails on macOS arm64, e.g., due to unsupported `_Float16`
ifeq ($(OS)-$(ARCH),osx-aarch64)
    TESTS :=
endif

include ../common.mak

extra_dflags += -d
