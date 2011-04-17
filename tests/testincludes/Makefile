# Copied from tango runtime makefile.
# Designed to work with GNU make
# Targets:
#	make
#		Same as make all
#	make lib
#		Build the common library
#   make doc
#       Generate documentation
#	make clean
#		Delete unneeded files created by build process

LIB_TARGET=libtangobos-partial.a
LIB_MASK=libtangobos-partial.a*

CP=cp -f
RM=rm -f
MD=mkdir -p

ADD_CFLAGS=
ADD_DFLAGS=

#CFLAGS=-O3 $(ADD_CFLAGS)
CFLAGS=-g $(ADD_CFLAGS)

#DFLAGS=-release -O3 -inline -w $(ADD_DFLAGS)
DFLAGS=-g -w -noasm $(ADD_DFLAGS)

#TFLAGS=-O3 -inline -w $(ADD_DFLAGS)
TFLAGS=-g -w -noasm $(ADD_DFLAGS)

DOCFLAGS=-version=DDoc

CC=gcc
LC=llvm-ar rsv
CLC=ar rsv
DC=ldc
LLC=llvm-as
LLVMLINK=llvm-link
LLC=llc

INC_DEST=
LIB_DEST=
DOC_DEST=

.SUFFIXES: .d .bc

.d.bc:
	$(DC) -c $(DFLAGS) $< -of$@
#	$(DC) -c $(DFLAGS) $< -of$@

targets : lib
all     : lib
lib     : tangobos.lib

######################################################

OBJ_CORE= \
    std/gc.bc \
    std/outofmemory.bc \
    std/IEEE.bc \
    std/stdarg.bc \
#    std/asserterror.bc \
#    std/format.bc \

ALL_OBJS= \
    $(OBJ_CORE)

######################################################

tangobos.lib : $(LIB_TARGET)

$(LIB_TARGET) : $(ALL_OBJS)
	$(RM) $(LIB_MASK)
	$(LLVMLINK) -o=$(LIB_TARGET).bc `find . -name "*.bc"`
	$(LLC) -o=$(LIB_TARGET).s $(LIB_TARGET).bc
	$(CC) -c -o $(LIB_TARGET) $(LIB_TARGET).s

clean :
	$(RM) $(ALL_OBJS)
	find . -name "$(LIB_MASK)" | xargs $(RM)

