#!/usr/bin/env bash

dir=${RESULTS_DIR}/compilable
output_file=${dir}/ldc_output_filenames.sh.out

rm -f ${output_file}

function bailout {
    cat ${output_file}
    rm -f ${output_file}
    exit 1
}

# 3 object files, 2 with same name

# -of (implying -singleobj); additionally make sure object file extension is NOT enforced
$DMD -m${MODEL} -Icompilable/extra-files/ldc_output_filenames -of${dir}/myObj.myExt -c compilable/extra-files/ldc_output_filenames/{main.d,foo.d,imp/foo.d} >> ${output_file}
if [ $? -ne 0 ]; then bailout; fi;
rm ${dir}/myObj.myExt >> ${output_file}
if [ $? -ne 0 ]; then bailout; fi;

# -op
$DMD -m${MODEL} -Icompilable/extra-files/ldc_output_filenames -od${dir} -c -op compilable/extra-files/ldc_output_filenames/{main.d,foo.d,imp/foo.d} >> ${output_file}
if [ $? -ne 0 ]; then bailout; fi;
rm ${dir}/compilable/extra-files/ldc_output_filenames/{main${OBJ},foo${OBJ},imp/foo${OBJ}} >> ${output_file}
if [ $? -ne 0 ]; then bailout; fi;

# -oq
$DMD -m${MODEL} -Icompilable/extra-files/ldc_output_filenames -od${dir} -c -oq compilable/extra-files/ldc_output_filenames/{main.d,foo.d,imp/foo.d} >> ${output_file}
if [ $? -ne 0 ]; then bailout; fi;
rm ${dir}/{ldc_output_filenames${OBJ},foo${OBJ},imp.foo${OBJ}} >> ${output_file}
if [ $? -ne 0 ]; then bailout; fi;

# -o-
$DMD -m${MODEL} -Icompilable/extra-files/ldc_output_filenames -o- compilable/extra-files/ldc_output_filenames/{main.d,foo.d,imp/foo.d} >> ${output_file}
if [ $? -ne 0 ]; then bailout; fi;


# Make sure the default file extension is appended if the user's
# -of doesn't contain any (LDMD only).

# args: <extra command-line options> <-of filename> <expected output filename>
function buildAndDelete {
    $DMD -m${MODEL} -Icompilable/extra-files/ldc_output_filenames -of${dir}/$2 "$1" compilable/extra-files/ldc_output_filenames/{main.d,foo.d,imp/foo.d} >> ${output_file}
    if [ $? -ne 0 ]; then bailout; fi;
    rm ${dir}/$3 >> ${output_file}
    if [ $? -ne 0 ]; then bailout; fi;
}

# executable
EXE_EXTENSION=
if [[ "$OS" == win* ]]; then EXE_EXTENSION=.exe; fi;
buildAndDelete "" executable executable${EXE_EXTENSION}
buildAndDelete "" executable.myExt executable.myExt

# static library
LIB_EXTENSION=.a
if [[ "$OS" == win* ]]; then LIB_EXTENSION=.lib; fi;
buildAndDelete "-lib" staticLib staticLib${LIB_EXTENSION}
buildAndDelete "-lib" staticLib.myExt staticLib.myExt

# shared library
SO_EXTENSION=.so
if [[ "$OS" == win* ]]; then SO_EXTENSION=.dll;
elif [ "$OS" == "osx" ]; then SO_EXTENSION=.dylib;
fi;
buildAndDelete "-shared" sharedLib sharedLib${SO_EXTENSION}
buildAndDelete "-shared" sharedLib.myExt sharedLib.myExt
