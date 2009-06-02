# Makefile to build the composite D runtime library for Linux
# Designed to work with GNU make
# Targets:
#	make
#		same as make lib
#	make all
#		make lib-release lib-debug and doc
#	make lib
#		Build the compiler runtime library (which version depends on VERSION, name on LIB_BUILD)
#	make lib-release
#		Build the release version of the compiler runtime library
#	make lib-debug
#		Build the debug version of the compiler runtime library
#   make doc
#       Generate documentation
#	make clean
#		Delete unneeded files created by build process
#	make clean-all
#		Delete unneeded files created by build process and the libraries
#	make unittest
#		Performs the unittests of the runtime library

ifeq ($(SHARED),yes)
LIB_EXT=so
LC_CMD=$(CC) -shared -o
else
LIB_EXT=a
LC_CMD=$(CLC)
endif

LIB_BASE=libtango-base-ldc
LIB_BUILD=
LIB_TARGET=$(LIB_BASE)$(LIB_BUILD).$(LIB_EXT)
LIB_BC=$(LIB_BASE)$(LIB_BUILD)-bc.a
LIB_C=$(LIB_BASE)$(LIB_BUILD)-c.a
LIB_MASK=$(LIB_BASE)*.a $(LIB_BASE)*.so

DIR_CC=./common/tango
DIR_RT=./compiler/ldc
DIR_RT2=./compiler/shared
DIR_GC=./gc/basic
MAKEFILE=ldc-posix.mak
targets : libs
all     : lib-release lib-debug doc

include ldcCommonFlags.mak

.PHONY : libs lib-release lib-debug unittest all doc clean install clean-all targets

######################################################

ALL_OBJS=

######################################################

ALL_DOCS=

libs : $(LIB_TARGET) $(LIB_BC) $(LIB_C)
$(LIB_TARGET) : $(ALL_OBJS)
	$(MAKE) -C $(DIR_CC) -fldc.mak libs DC=$(DC) ADD_DFLAGS="$(ADD_DFLAGS)" ADD_CFLAGS="$(ADD_CFLAGS)" \
	 	VERSION="$(VERSION)" LIB_BUILD="$(LIB_BUILD)" SHARED="$(SHARED)"
	$(MAKE) -C $(DIR_RT) -fldc.mak libs DC=$(DC) ADD_DFLAGS="$(ADD_DFLAGS)" ADD_CFLAGS="$(ADD_CFLAGS)" \
	 	VERSION="$(VERSION)" LIB_BUILD="$(LIB_BUILD)" SHARED="$(SHARED)"
	$(MAKE) -C $(DIR_GC) -fldc.mak libs DC=$(DC) ADD_DFLAGS="$(ADD_DFLAGS)" ADD_CFLAGS="$(ADD_CFLAGS)" \
                VERSION="$(VERSION)" LIB_BUILD="$(LIB_BUILD)" SHARED="$(SHARED)"
	$(RM) $@
	$(LC_CMD) $@ `find $(DIR_CC) -name "*.o" | xargs echo`
	$(LC_CMD) $@ `find $(DIR_RT) -name "*.o" | xargs echo`
	$(LC_CMD) $@ `find $(DIR_RT2) -name "*.o" | xargs echo`
	$(LC_CMD) $@ `find $(DIR_GC) -name "*.o" | xargs echo`
ifneq ($(RANLIB),)
	$(RANLIB) $@
endif

$(LIB_BC): $(LIB_TARGET)
	$(RM) $@
	$(LC) $@ `find $(DIR_CC) -name "*.bc" | xargs echo`
	$(LC) $@ `find $(DIR_RT) -name "*.bc" | xargs echo`
	$(LC) $@ `find $(DIR_RT2) -name "*.bc" | xargs echo`
	$(LC) $@ `find $(DIR_GC) -name "*.bc" | xargs echo`
ifneq ($(RANLIB),)
	$(RANLIB) $@
endif

LIB_C_OBJS= $(DIR_CC)/libtango-cc-tango-c-only$(LIB_BUILD).a $(DIR_RT)/libtango-rt-ldc$(LIB_BUILD)-c.a 

$(LIB_C): $(LIB_TARGET) $(LIB_C_OBJS)
	$(CLC) $@ $(LIB_C_OBJS)

doc : $(ALL_DOCS)
	$(MAKE) -C $(DIR_CC) -fldc.mak doc DC=$(DC)
	$(MAKE) -C $(DIR_RT) -fldc.mak doc DC=$(DC)
	$(MAKE) -C $(DIR_GC) -fldc.mak doc DC=$(DC)

######################################################

#	find . -name "*.di" | xargs $(RM)
clean :
	$(RM) $(ALL_OBJS)
	$(MAKE) -C $(DIR_CC) -fldc.mak clean
	$(MAKE) -C $(DIR_RT) -fldc.mak clean
	$(MAKE) -C $(DIR_GC) -fldc.mak clean

clean-all : clean
	$(MAKE) -C $(DIR_CC) -fldc.mak clean-all
	$(MAKE) -C $(DIR_RT) -fldc.mak clean-all
	$(MAKE) -C $(DIR_GC) -fldc.mak clean-all
	$(RM) $(ALL_DOCS)
	$(RM) $(LIB_MASK)
	find $(DIR_CC) -name "*.bc" | xargs rm -rf
	find $(DIR_RT) -name "*.bc" | xargs rm -rf
	find $(DIR_RT2) -name "*.bc"| xargs rm -rf
	find $(DIR_GC) -name "*.bc" | xargs rm -rf
	find $(DIR_CC) -name "*.o"  | xargs rm -rf
	find $(DIR_RT) -name "*.o"  | xargs rm -rf
	find $(DIR_RT2) -name "*.o" | xargs rm -rf
	find $(DIR_GC) -name "*.o"  | xargs rm -rf

install :
	$(MAKE) -C $(DIR_CC) -fldc.mak install
	$(MAKE) -C $(DIR_RT) -fldc.mak install
	$(MAKE) -C $(DIR_GC) -fldc.mak install
#	$(CP) $(LIB_MASK) $(LIB_DEST)/.
