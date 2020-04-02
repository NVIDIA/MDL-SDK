#!/usr/bin/env python
#*****************************************************************************
# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#*****************************************************************************

# This script generates libbsdf_bitcode.h from libbsdf.bc

import sys
import os
import struct

copyright_str = """/******************************************************************************
 * Copyright 2020 NVIDIA Corporation. All rights reserved.
 *****************************************************************************/
"""

def process_data_set(table_data_filename, fout):
    f = open(table_data_filename, "rb")
    name_we = os.path.basename(table_data_filename)
    name = os.path.splitext(name_we)[0]

    print("- processing multi-scatter data set: %s" % name)
    fout.write("\n// Automatically generated from %s\n" % name_we)

    # check header
    res_roughness, = struct.unpack('i', f.read(4))
    res_theta, = struct.unpack('i', f.read(4))
    res_ior, = struct.unpack('i', f.read(4))

    if (res_roughness != 64) or (res_theta != 64) or ((res_ior != 16) and (res_ior != 0)):
        print("unexpected resolutions in dataset %s" % name)
        print("- res_roughness: %d" % res_roughness)
        print("- res_theta: %d" % res_theta)
        print("- res_ior: %d" % res_ior)
        return -1

    expected_buffer_size = (res_ior * 2 + 1) * res_roughness * (res_theta + 1) * 4
    bytes = f.read()
    buffer_size = len(bytes)
    if expected_buffer_size != buffer_size:
        print("unexpected size of dataset %s" % name)
        print("- expected_buffer_size: %d" % expected_buffer_size)
        print("- buffer_size: %d" % buffer_size)
        return -1

    fout.write("unsigned const libbsdf_multiscatter_res_theta_%s = %d;\n" % (name, res_theta))
    fout.write("unsigned const libbsdf_multiscatter_res_roughness_%s = %d;\n" % (name, res_roughness))
    fout.write("unsigned const libbsdf_multiscatter_res_ior_%s = %d;\n" % (name, res_ior))

    # process the actual data after the header
    fout.write("unsigned char const libbsdf_multiscatter_data_%s[] = {\n" % name)
    i = 0
    fout.write("    ")
    for byte in bytes:
        if isinstance(byte, str):
            byte = ord(byte)
        fout.write("0x%02x, " % byte)
        if i == 7:
            fout.write("\n    ")
            i = 0
        else:
            i += 1

    fout.write("};\n")
    return 0

def usage():
    print("Usage: %s <output directory> <inputfile1> <inputfile2> ..." % sys.argv[0])
    return 1

def main(args):
    if len(args) < 3:
        return usage()

    with open(args[1], "w") as f:
        f.write(copyright_str)
        f.write("\n")
        f.write("#include <mdl/jit/generator_jit/generator_jit_libbsdf_data.h>\n")
        f.write("\n")
        f.write("namespace mi {\n")
        f.write("namespace mdl {\n")
        f.write("namespace libbsdf_data {\n")

        # process all files
        for x in range(2, len(args)):
            res = process_data_set(args[x], f)
            if (res < 0):
                print("res: %s" % res)
                return -1

        f.write("}\n")
        f.write("}\n")
        f.write("}\n")

    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
