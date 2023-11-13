#!/usr/bin/env python
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

import argparse
import shutil
import re


# Note, this scipt is used to remove unnecessary and potentially misused functions from the binding.
# In the future this script or the generation itself can change in order to avoid post processing
# like this. But in the meantime we want to make sure that nobody relies on functions that are not
# meant to be used.
def run(args, binding_file_path):

    # create a backup
    shutil.copyfile(binding_file_path, args.outdir + '/' + args.module + '.prepost.py' )

    # read the file
    with open(binding_file_path, 'r') as binding_file:
        Lines = binding_file.readlines()

    ProcessedLines = []

    strip: bool = False
    strip_n_lines: int = 0
    strip_end: str = ""
    class_name: str = ""
    is_iinterface: bool = False
    for line in Lines:
        # keep class name for logs
        lineNoWhiteSpaces = line.strip()
        if line.startswith('class '):
            class_name = re.search(r'^class (.*?)\(.*', line).group(1)
            is_iinterface = class_name.startswith("I")

        # find interface classes with a leading _
        if line.startswith('class _I'):
            strip_end = f'_pymdlsdk.{class_name}_swigregister'
            strip = True
        elif strip and line.startswith(strip_end):
            strip = False
            strip_n_lines = 2
            if args.verbose:
                print(f"removed class: '{class_name}'")

        # remove the __ref__ and __deref__ functions added by swig.
        # They are not used by the bindings and should not be used by binding users.
        elif lineNoWhiteSpaces.startswith('def __ref__(self, *args):'):
            strip_n_lines = 3
            if args.verbose:
                print(f"removed __ref__ from class: '{class_name}'")
        elif lineNoWhiteSpaces.startswith('def __deref__(self, *args):'):
            strip_n_lines = 3
            if args.verbose:
                print(f"removed __deref__ from class: '{class_name}'")

        # for the 367100 branch
        elif lineNoWhiteSpaces.startswith('def _retain(self)'):
            strip_n_lines = 12
            if args.verbose:
                print(f"removed _retain from class: '{class_name}'")
        elif lineNoWhiteSpaces.startswith('def _release(self)'):
            strip_n_lines = 13
            if args.verbose:
                print(f"removed _release from class: '{class_name}'")

        # there is also an additional golbal IID function that is redundant, side effection from creating a static function
        elif lineNoWhiteSpaces.startswith('def I') and lineNoWhiteSpaces.startswith(f'def {class_name}_IID()'):
            strip = True
            strip_end = f'return _pymdlsdk.{class_name}_IID()'
        elif strip and lineNoWhiteSpaces.startswith(strip_end):
            strip = False
            strip_n_lines = 2
            if args.verbose:
                print(f"removed unused global function: def {class_name}_IID()")
        # and the same with the get_interface function that is needed on the type (with an instance as parameter)
        elif lineNoWhiteSpaces.startswith('def I') and lineNoWhiteSpaces.startswith(f'def {class_name}__get_interface(iface)'):
            strip = True
            strip_end = f'return _pymdlsdk.{class_name}__get_interface(iface)'
        elif strip and lineNoWhiteSpaces.startswith(strip_end):
            strip = False
            strip_n_lines = 2
            if args.verbose:
                print(f"removed unused global function: def {class_name}__get_interface(...)")

        # make the init function invalid because we don't allow to call them for now
        elif is_iinterface and lineNoWhiteSpaces.startswith(f'__swig_destroy__ = _pymdlsdk.delete_{class_name}'):
            # we pop the last added function which is the call we don't want
            previousLine = ProcessedLines.pop().strip()
            ProcessedLines.append(f"        raise RuntimeError(\"Direct construction of type '{class_name}' is not allowed. Please use API functions instead.\")    # pragma: no cover\n")
            ProcessedLines.append(f"        # {previousLine}\n")
            if args.verbose:
                print(f"made the constructor of {class_name} raise an exception")

        # skip lines or not
        if not strip:
            if strip_n_lines > 0:
                strip_n_lines = strip_n_lines - 1
            else:
                ProcessedLines.append(line)

    # pverwrite the file
    with open(binding_file_path, 'w') as binding_file:
        binding_file.writelines(ProcessedLines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-outdir', help='the directory of the create python binding module')
    parser.add_argument('-module', help='the name of the create python binding module without extension')
    parser.add_argument('-v', dest='verbose', action='store_true', default=False, help='print verbose output to log the changed functions and classes')

    args = parser.parse_args()
    binding_file_path = args.outdir + '/' + args.module + '.py'
    print(f"post processing binding moduke: {binding_file_path}")

    run(parser.parse_args(), binding_file_path)
