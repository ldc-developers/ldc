include ../common.mak

DFLAGS+=-cov

NORMAL_TESTS:=$(addprefix $(ROOT)/,$(addsuffix .done,basic))
MERGE_TESTS:=$(addprefix $(ROOT)/,$(addsuffix .done,merge merge_true))

DIFF:=diff --strip-trailing-cr

.PHONY: all clean
all: $(NORMAL_TESTS) $(MERGE_TESTS)

$(NORMAL_TESTS): $(ROOT)/%.done: $(ROOT)/%
	@echo Testing $*
	@rm -f $(ROOT)/src-$*.lst
	$(QUIET)$(ROOT)/$* $(ROOT) $(RUN_ARGS)
ifneq (,$(findstring win,$(OS)))
	sed -i 's:^src\\:src/:g' $(ROOT)/src-$*.lst
endif
	$(QUIET)$(DIFF) src-$*.lst.exp $(ROOT)/src-$*.lst
	@touch $@

$(MERGE_TESTS): $(ROOT)/%.done: $(ROOT)/%
	@echo Testing $*
	@rm -f $(ROOT)/src-$*.lst
	$(QUIET)$(ROOT)/$* $(ROOT) $(RUN_ARGS)
	$(QUIET)$(ROOT)/$* $(ROOT) $(RUN_ARGS)
ifneq (,$(findstring win,$(OS)))
	sed -i 's:^src\\:src/:g' $(ROOT)/src-$*.lst
endif
	$(QUIET)$(DIFF) src-$*.lst.exp $(ROOT)/src-$*.lst
	@touch $@

$(ROOT)/no_code.done: $(ROOT)/%.done: $(ROOT)/%
	@echo Testing $*
	@rm -f $(ROOT)/src-$*.lst
	$(QUIET)$(ROOT)/$* $(ROOT) $(RUN_ARGS)
	$(QUIET)$(ROOT)/$* $(ROOT) $(RUN_ARGS)
ifneq (,$(findstring win,$(OS)))
	sed -i 's:^src\\:src/:g' $(ROOT)/src-$*.lst
	sed -i 's:^src\\:src/:g' $(ROOT)/src-$*_imp.lst
endif
	$(QUIET)$(DIFF) src-$*.lst.exp $(ROOT)/src-$*.lst
	$(QUIET)$(DIFF) src-$*_imp.lst.exp $(ROOT)/src-$*_imp.lst
	@touch $@

$(ROOT)/no_code: $(SRC)/no_code_imp.d
$(ROOT)/%: $(SRC)/%.d
	$(QUIET)$(DMD) $(DFLAGS) -of$(ROOT)/$* $^

clean:
	rm -rf $(GENERATED) *.lst
