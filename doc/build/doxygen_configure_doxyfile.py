#!/usr/bin/env python3
#*****************************************************************************
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

'''Modify some settings in Doxyfile according to dot availability and directory structure.'''

import re
import sys

assert len(sys.argv) >= 1+5

input_file     = sys.argv[1]
output_file    = sys.argv[2]
have_dot_arg   = sys.argv[3]
dot_path_arg   = sys.argv[4]
image_path_arg = sys.argv[5]

# A non-empty path causes a warning, even if dot support itself is disabled.
if have_dot_arg == "NO":
    dot_path_arg = ""

buffer = open(input_file, 'r', encoding="utf-8").readlines()
text = ''.join(buffer)

text = re.compile('(HAVE_DOT\\s*=).*$',   re.M).sub('\\1 %s' % have_dot_arg,   text)
text = re.compile('(DOT_PATH\\s*=).*$',   re.M).sub('\\1 %s' % dot_path_arg,   text)
text = re.compile('(IMAGE_PATH\\s*=).*$', re.M).sub('\\1 %s' % image_path_arg, text)

open(output_file, 'w', encoding="utf-8").write(text)
