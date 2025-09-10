#!/usr/bin/env python
#*****************************************************************************
# Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
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
 * Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *****************************************************************************/
"""

def process_data_set(table_data_filename, fout, mode = "multiscatter"):
    f = open(table_data_filename, "rb")
    name_we = os.path.basename(table_data_filename)
    name = os.path.splitext(name_we)[0]

    fout.write("\n// Automatically generated from %s\n" % name_we)

    # check header
    expected_buffer_size = 0
    if mode == "multiscatter":
        print("- processing multi-scatter data set: %s" % name)
        res_u, = struct.unpack('i', f.read(4))
        res_v, = struct.unpack('i', f.read(4))
        res_w, = struct.unpack('i', f.read(4))

        if (res_u != 64) or (res_v != 64) or ((res_w != 16) and (res_w != 0)):
            print("unexpected resolutions in dataset %s" % name)
            print("- res_roughness: %d" % res_u)
            print("- res_theta: %d" % res_v)
            print("- res_ior: %d" % res_w)
            return -1
        
        expected_buffer_size = (res_w * 2 + 1) * res_u * (res_v + 1) * 4
    elif mode == "general":
        print("- processing general data set: %s" % name)
        res_u, = struct.unpack('i', f.read(4))
        res_v, = struct.unpack('i', f.read(4))
        res_w, = struct.unpack('i', f.read(4))
        expected_buffer_size = res_u * res_v * res_w * 3 * 4  # float3 elements
    else:
        print("unexpected mode '%s' for dataset %s" % (mode, name))
        return -1
    
    # read the binary data 
    bytes = f.read()
    buffer_size = len(bytes)
    if expected_buffer_size != buffer_size:
        print("unexpected size of dataset %s" % name)
        print("- expected_buffer_size: %d" % expected_buffer_size)
        print("- buffer_size: %d" % buffer_size)
        return -1

    # write the C constants
    # and process the actual data after the header
    if mode == "multiscatter":
        fout.write("unsigned const libbsdf_multiscatter_res_theta_%s = %d;\n" % (name, res_v))
        fout.write("unsigned const libbsdf_multiscatter_res_roughness_%s = %d;\n" % (name, res_u))
        fout.write("unsigned const libbsdf_multiscatter_res_ior_%s = %d;\n" % (name, res_w))
        fout.write("unsigned char const libbsdf_multiscatter_data_%s[] = {\n" % name)
    elif mode == "general":
        fout.write("unsigned const libbsdf_general_res_u_%s = %d;\n" % (name, res_u))
        fout.write("unsigned const libbsdf_general_res_v_%s = %d;\n" % (name, res_v))
        fout.write("unsigned const libbsdf_general_res_w_%s = %d;\n" % (name, res_w))
        fout.write("unsigned char const libbsdf_general_data_%s[] = {\n" % name)
    else:
        return -1
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
    print("Usage: %s <output directory> <multiscatter|general> <inputfile1> <inputfile2> ..." % sys.argv[0])
    return 1

def main(args):
    if len(args) < 4:
        return usage()

    with open(args[2], "w") as f:
        f.write(copyright_str)
        f.write("\n")
        f.write("#include <mdl/jit/generator_jit/generator_jit_libbsdf_data.h>\n")
        f.write("\n")
        f.write("namespace mi {\n")
        f.write("namespace mdl {\n")
        f.write("namespace libbsdf_data {\n")

        # process all files
        for x in range(3, len(args)):
            res = process_data_set(args[x], f, args[1])
            if (res < 0):
                print("res: %s" % res)
                return -1

        f.write("}\n")
        f.write("}\n")
        f.write("}\n")

    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
