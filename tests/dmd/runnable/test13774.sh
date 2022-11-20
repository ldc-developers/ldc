#!/usr/bin/env bash

set -e

# LDC: llvm-lib (v8.0) doesn't support merging static libs, need MS lib.exe
ldc_ar=''
if [[ "$OS" == win* ]]; then ldc_ar='-ar=lib.exe'; fi

$DMD -m${MODEL} -I${EXTRA_FILES} -lib -of${OUTPUT_BASE}a${LIBEXT} ${EXTRA_FILES}${SEP}lib13774a.d
$DMD -m${MODEL} -I${EXTRA_FILES} -lib $ldc_ar -of${OUTPUT_BASE}b${LIBEXT} ${EXTRA_FILES}${SEP}lib13774b.d ${OUTPUT_BASE}a${LIBEXT}

# Windows: make sure b.lib contains both object files
if [[ "$OS" == win* ]]; then
    $DMD -m${MODEL} -I${EXTRA_FILES} -of${OUTPUT_BASE}link${EXE} ${EXTRA_FILES}${SEP}link13774.d ${OUTPUT_BASE}b${LIBEXT}
    rm_retry ${OUTPUT_BASE}link${EXE}
fi

rm_retry ${OUTPUT_BASE}{a${LIBEXT},b${LIBEXT}}
