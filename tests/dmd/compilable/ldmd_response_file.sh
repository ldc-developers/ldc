#!/usr/bin/env bash

# Make sure LDMD forwards a huge command line correctly to LDC.

dir=${RESULTS_DIR}/compilable

# generate a ~100K response file for LDMD
rsp_file=${dir}/ldmd_response_file.rsp
echo "-version=FirstLine" > ${rsp_file}
for i in {1..1000}
do
   echo "-I=Some/lengthy/string/Some/lengthy/string/Some/lengthy/string/Some/lengthy/string/Some/lengthy/string/" >> ${rsp_file}
done
echo "-version=LastLine" >> ${rsp_file}

# statically assert that both versions are set
src_file=${dir}/ldmd_response_file.d
echo "version (FirstLine) {" > ${src_file}
echo "    version (LastLine) {} else static assert(0);" >> ${src_file}
echo "} else" >> ${src_file}
echo "    static assert(0);" >> ${src_file}

# LDMD errors if there's no source file.
$DMD @${rsp_file} -c -o- ${src_file}
if [ $? -ne 0 ]; then exit 1; fi;
