#!/bin/bash

cd ../druntime
patch -p0 < ../runtime/ldc2.diff
