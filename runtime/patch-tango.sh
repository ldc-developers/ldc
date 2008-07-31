#!/bin/bash

cd ../tango
patch -p0 < ../runtime/llvmdc.diff
