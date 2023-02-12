# set from top makefile
# LDC: we have no top makefile, include osmodel.mak for OS
include ../../../../dmd/osmodel.mak
#OS:=
MODEL:=
BUILD:=
DMD:=
DRUNTIME:=
DRUNTIMESO:=
LINKDL:=
QUIET:=
TIMELIMIT:=
LDL:=$(subst -L,,$(LINKDL)) # -ldl

SRC:=src
GENERATED:=./generated
ROOT:=$(GENERATED)/$(OS)/$(BUILD)/$(MODEL)

ifneq (default,$(MODEL))
	MODEL_FLAG:=-m$(MODEL)
endif
CFLAGS_BASE:= $(MODEL_FLAG) $(PIC) -Wall
#ifeq (osx,$(OS))
#	ifeq (64,$(MODEL))
#		CFLAGS_BASE+=--target=x86_64-darwin-apple  # ARM cpu is not supported by dmd
#	endif
#endif
# LDC: use -defaultlib=druntime-ldc instead of `-defaultlib= -L$(DRUNTIME[SO])`
DFLAGS:=$(MODEL_FLAG) $(PIC) -w -I../../src -I../../import -I$(SRC) -defaultlib=druntime-ldc -preview=dip1000
ifeq (,$(findstring win,$(OS)))
	DFLAGS += -L-lpthread -L-lm $(LINKDL)
endif
# LINK_SHARED may be set by importing makefile
# LDC: -link-defaultlib-shared takes care of rpath, linking ldc_rt.dso.o etc.
DFLAGS+=$(if $(LINK_SHARED),-link-defaultlib-shared,)
ifeq ($(BUILD),debug)
	# LDC: link against debug druntime
	DFLAGS += -g -debug -link-defaultlib-debug
	ifeq (,$(findstring win,$(OS)))
		CFLAGS := $(CFLAGS_BASE) -g
	else
		CFLAGS := $(CFLAGS_BASE) /Zi
	endif
else
	DFLAGS += -O -release
	ifeq (,$(findstring win,$(OS)))
		CFLAGS := $(CFLAGS_BASE) -O3
	else
		CFLAGS := $(CFLAGS_BASE) /O2
	endif
endif
CXXFLAGS_BASE := $(CFLAGS_BASE)
CXXFLAGS:=$(CFLAGS)
