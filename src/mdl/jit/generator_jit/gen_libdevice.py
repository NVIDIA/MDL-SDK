#!/usr/bin/env python
#*****************************************************************************
# Copyright (c) 2014-2020, NVIDIA CORPORATION. All rights reserved.
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

import sys
import os

def xxd(filename, fout):
	f = open(filename, "rb")
	bytes = f.read()

	i = 0
	fout.write("\t")
	for byte in bytes:
		if isinstance(byte, str):
			byte = ord(byte)
		fout.write("0x%02x, " % byte)
		if i == 7:
			fout.write("\n\t")
			i = 0
		else:
			i += 1

def main(args):
	filter         = args[1]
	#CUDA_DIR      = args[2].replace("\\","/") #re-enable in case we update to newer LLVM
	MISRC_DIR      = args[2].replace("\\","/")
	IntDir         = args[3]

	#LIBDEVICE_DIR = CUDA_DIR + "/nvvm/libdevice" #re-enable in case we update to newer LLVM
	LIBDEVICE_DIR = MISRC_DIR + "/libdevice"

	# to support old python versions having problems executing
	# relative path with forward slashes under windows
	if filter[0] == '.':
		filter = os.getcwd() + "/" + filter
	version = 10
	bc_name  = "libdevice.%u.bc" % (version)
	out_name = "glue_libdevice.h"

	print("Stripping %s ..." % bc_name)
	cmd_line = (filter +
		" \"" + LIBDEVICE_DIR + "/" + bc_name + "\" \"" +
		IntDir + "/" + bc_name + "\"")
	exit_code = os.system(cmd_line)
	if exit_code != 0:
		sys.stderr.write ("ERROR: command %s exited unexpectedly, exitcode %d\n" % (cmd_line, exit_code))
		sys.exit (1)

	print("Generating %s ..." % out_name)
	f = open(IntDir + "/" + out_name, "w")
	f.write("static unsigned char const glue_bitcode[] = {\n")
	xxd(IntDir + "/" + bc_name, f)
	f.write("};\n")
	f.close()

if __name__ == "__main__":
	main(sys.argv)
