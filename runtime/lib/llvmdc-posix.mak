# Makefile to build the composite D runtime library for Linux
# Designed to work with GNU make
# Targets:
#	make
#		Same as make all
#	make lib
#		Build the runtime library
#   make doc
#       Generate documentation
#	make clean
#		Delete unneeded files created by build process

LIB_TARGET=libtango-base-llvmdc.a
LIB_MASK=libtango-base-llvmdc*.a
LIB_TARGET_C=libtango-base-c-llvmdc.a
LIB_MASK_C=libtango-base-c-llvmdc*.a
LIB_NAME_NATIVE=libtango-base-llvmdc-native
LIB_TARGET_NATIVE=$(LIB_NAME_NATIVE).a

DIR_CC=./common/tango
DIR_RT=../../runtime/internal
DIR_GC=./gc/basic
#DIR_GC=./gc/stub

CP=cp -f
RM=rm -f
MD=mkdir -p

CC=gcc
LC=llvm-ar rsv
CLC=ar rsv
DC=llvmdc
LLVMLINK=llvm-link
LLC=llc

ADD_CFLAGS=
#ADD_DFLAGS=
ADD_DFLAGS=-I`pwd`/common/

targets : lib sharedlib doc
all     : lib sharedlib doc

######################################################

ALL_OBJS=

######################################################

ALL_DOCS=

######################################################

lib : $(ALL_OBJS)
	make -C $(DIR_CC) -fllvmdc.mak lib DC=$(DC) ADD_DFLAGS="$(ADD_DFLAGS)" ADD_CFLAGS="$(ADD_CFLAGS)"
	make -C $(DIR_RT) -fllvmdc.mak lib
	make -C $(DIR_GC) -fllvmdc.mak lib DC=$(DC) ADD_DFLAGS="$(ADD_DFLAGS)" ADD_CFLAGS="$(ADD_CFLAGS)"
	# could link the three parts into one here, but why should we

sharedlib : $(ALL_OBJS)
	make -C $(DIR_CC) -fllvmdc.mak sharedlib DC=$(DC) ADD_DFLAGS="$(ADD_DFLAGS)" ADD_CFLAGS="$(ADD_CFLAGS)"
	make -C $(DIR_RT) -fllvmdc.mak sharedlib
	make -C $(DIR_GC) -fllvmdc.mak sharedlib DC=$(DC) ADD_DFLAGS="$(ADD_DFLAGS)" ADD_CFLAGS="$(ADD_CFLAGS)"
	# could link the three parts into one here, but why should we

doc : $(ALL_DOCS)
	make -C $(DIR_CC) -fllvmdc.mak doc
	make -C $(DIR_RT) -fllvmdc.mak doc
	make -C $(DIR_GC) -fllvmdc.mak doc

######################################################

clean :
	find . -name "*.di" | xargs $(RM)
	$(RM) $(ALL_OBJS)
	$(RM) $(ALL_DOCS)
	make -C $(DIR_CC) -fllvmdc.mak clean
	make -C $(DIR_RT) -fllvmdc.mak clean
	make -C $(DIR_GC) -fllvmdc.mak clean
	$(RM) $(LIB_MASK)
	$(RM) $(LIB_MASK_C)
	$(RM) $(LIB_NAME_NATIVE)*

install :
	make -C $(DIR_CC) -fllvmdc.mak install
	make -C $(DIR_RT) -fllvmdc.mak install
	make -C $(DIR_GC) -fllvmdc.mak install
	$(CP) $(LIB_MASK) $(LIB_DEST)/.
	$(CP) $(LIB_MASK_C) $(LIB_DEST)/.
