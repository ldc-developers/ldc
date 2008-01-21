#!/bin/bash
llvmdc lib.d -c -g -dis
llvmdc app.d lib.bc -g -dis -ofapp
