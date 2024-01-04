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

    # list of global constnats to preserve
    global_names_to_preserve: list[str] = [
        'MI_NEURAYLIB_API_VERSION',
        'MI_NEURAYLIB_PLUGIN_TYPE',
        'MI_NEURAYLIB_PRODUCT_VERSION_STRING',
        'MI_NEURAYLIB_VERSION_MAJOR',
        'MI_NEURAYLIB_VERSION_MINOR',
        'MI_NEURAYLIB_VERSION_QUALIFIER',
        'MI_NEURAYLIB_VERSION_QUALIFIER_EMPTY',
        'STANDALONE_SUPPORT_ENABLED',
    ]

    # list of global constants for enum values to be removed
    global_names_to_remove: list[str] = [
        'BAKE_',
        'BAKER_',  # belongs to BAKE_
        'CLIP_',
        'ELEMENT_TYPE_',
        'FILTER_',
        'LIGHTPROFILE_',
        'MDL_VERSION_',
        'MDL_REPAIR_',
        'MDL_REMOVE_',  # belongs to REMOVE_
        'MESSAGE_SEVERITY_',
        'OPACITY_',
        'PROPAGATION_',  # since we removed the IAttribute_set from the bindings
        'SLOT_',
        'TEXTURE_',
        'UVTILE_MODE_',
    ]
    global_names_patters: list[re.Pattern] = []
    for n in global_names_to_remove:
        global_names_patters.append(re.compile(rf"^({n})(\w+)\s=\s_pymdlsdk.(\1)(\2)\s*$"))

    global_name_pattern:re.Pattern = re.compile(r"^(\w+)\s=\s_pymdlsdk.(\1)\s*$")

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
        elif lineNoWhiteSpaces.startswith('def __ref__(self, *args) ->'):
            strip_n_lines = 3
            if args.verbose:
                print(f"removed __ref__ from class: '{class_name}'")
        elif lineNoWhiteSpaces.startswith('def __deref__(self, *args) ->'):
            strip_n_lines = 3
            if args.verbose:
                print(f"removed __deref__ from class: '{class_name}'")

        # there is also an additional golbal IID function that is redundant, side effection from creating a static function
        elif lineNoWhiteSpaces.startswith('def I') and \
            lineNoWhiteSpaces.startswith(f'def {class_name}_IID() -> "mi::base::Uuid const":'):
            strip = True
            strip_end = f'return _pymdlsdk.{class_name}_IID()'
        elif strip and lineNoWhiteSpaces.startswith(strip_end):
            strip = False
            strip_n_lines = 2
            if args.verbose:
                print(f"removed unused global function: def {class_name}_IID()")
        # and the same with the get_interface function that is needed on the type (with an instance as parameter)
        elif lineNoWhiteSpaces.startswith('def I') and \
            lineNoWhiteSpaces.startswith(f'def {class_name}__get_interface(iface: "mi::base::IInterface *") ->'):
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
            ProcessedLines.append(f"        raise RuntimeError(\"Direct construction of type '{class_name}' is not allow. Please use API functions instead.\")    # pragma: no cover\n")
            ProcessedLines.append(f"        # {previousLine}\n")
            if args.verbose:
                print(f"made the constructor of {class_name} raise an exception")

        # remove global enum constants (we wrapped all of them into classes to provide correct scoping)
        elif not line.startswith(' '):  # global names start at the beginning of the line
            constant_process: bool = False
            for p in global_names_patters:
                if p.search(line) is not None:
                    strip_n_lines = 1 if 'FORCE_32_BIT' in lineNoWhiteSpaces else 2
                    constant_process = True
                    if args.verbose:
                        print(f"removed global constant: {lineNoWhiteSpaces}")
            if not constant_process:
                general: re.Match = global_name_pattern.match(line)
                if general and general.group(1) not in global_names_to_preserve:
                    print(f"Error: MDL Python Bindings contain global constant '{general.group(1)}' that isn't handled.")

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
