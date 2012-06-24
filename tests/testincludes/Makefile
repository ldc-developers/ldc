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
LIB_MASK=libtangobos-partial.*

CP=cp -f
RM=rm -f
MD=mkdir -p

ADD_CFLAGS=
ADD_DFLAGS=

#CFLAGS=-O3 $(ADD_CFLAGS)
CFLAGS=-g $(ADD_CFLAGS)

#DFLAGS=-release -O3 -inline -w $(ADD_DFLAGS)
DFLAGS=-g -w -noasm $(ADD_DFLAGS)

DC=ldc

targets : lib
all     : lib
lib     : tangobos.lib

SOURCE= \
    std/gc.d \
    std/outofmemory.d \
    std/IEEE.d \
    std/stdarg.d \
#    std/asserterror.d \
#    std/format.d \

tangobos.lib : $(LIB_TARGET)

$(LIB_TARGET) : $(ALL_OBJS)
	$(DC) -lib -of$(LIB_TARGET) $(DFLAGS) $(SOURCE)

clean :
	$(RM) $(ALL_OBJS)
	find . -name "$(LIB_MASK)" | xargs $(RM)

