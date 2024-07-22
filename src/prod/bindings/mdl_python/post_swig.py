#!/usr/bin/env python
#*****************************************************************************
# Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
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
from ast import literal_eval as make_tuple

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
        'BSDF_',  # Bsdf_type used in IBsdf_isotropic_data
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

    manualTypeHintMap: dict = {}
    manualTypeHintMap['-> "char const *":'] = '-> str:'
    manualTypeHintMap['-> "mi::Sint64":'] = '-> int:'
    manualTypeHintMap[': "mi::Sint64"'] = ': int'
    manualTypeHintMap['-> "mi::Sint32":'] = '-> int:'
    manualTypeHintMap[': "mi::Sint32"'] = ': int'
    manualTypeHintMap['-> "mi::Sint16":'] = '-> int:'
    manualTypeHintMap[': "mi::Sint16"'] = ': int'
    manualTypeHintMap['-> "mi::Sint8":'] = '-> int:'
    manualTypeHintMap[': "mi::Sint8"'] = ': int'
    manualTypeHintMap['-> "mi::Uint64":'] = '-> int:'
    manualTypeHintMap[': "mi::Uint64"'] = ': int'
    manualTypeHintMap['-> "mi::Uint32":'] = '-> int:'
    manualTypeHintMap[': "mi::Uint32"'] = ': int'
    manualTypeHintMap['-> "mi::Uint16":'] = '-> int:'
    manualTypeHintMap[': "mi::Uint16"'] = ': int'
    manualTypeHintMap['-> "mi::Uint8":'] = '-> int:'
    manualTypeHintMap[': "mi::Uint8"'] = ': int'
    manualTypeHintMap['-> "mi::Float32":'] = '-> float:'
    manualTypeHintMap[': "mi::Float32"'] = ': float'
    manualTypeHintMap['-> "mi::Float64":'] = '-> float:'
    manualTypeHintMap[': "mi::Float64"'] = ': float'
    manualTypeHintMap['-> "mi::Size":'] = '-> int:'
    manualTypeHintMap[': "mi::Size"'] = ': int'
    manualTypeHintMap['-> "mi::Difference":'] = '-> int:'
    manualTypeHintMap[': "mi::Difference"'] = ': int'
    manualTypeHintMap['-> "size_t":'] = '-> int:'
    manualTypeHintMap[': "size_t"'] = ': int'
    manualTypeHintMap['-> "uint64_t":'] = '-> int:'
    manualTypeHintMap[': "uint64_t"'] = ': int'
    manualTypeHintMap[': "char const *"'] = ': str'
    manualTypeHintMap['-> "bool":'] = '-> bool:'
    manualTypeHintMap[': "bool"'] = ': bool'
    manualTypeHintMap['-> "mi::math::Color_struct":'] = '-> "Color_struct":'
    manualTypeHintMap['-> "void":'] = '-> None:'
    manualTypeHintMap['-> "std::string":'] = '-> str:'

    # used for __exit__ functions
    manualTypeHintMap['exc_type: "void const *"'] = 'exc_type'
    manualTypeHintMap['exc_value: "void const *"'] = 'exc_value'
    manualTypeHintMap['exc_traceback: "void const *"'] = 'exc_traceback'

    # since we renamed them to ovoid conflicts in swig
    manualTypeHintMap[': "_IType_struct"'] = ': "IType_structure"'
    manualTypeHintMap[': "_IType_enum"'] = ': "IType_enumeration"'


    def add_type_map_iinterface(map: dict, map_from: str, map_to: str):
        map[f'-> "_{map_from}":'] =                             f'-> "{map_to}":'
        map[f': "_{map_from}"'] =                               f': "{map_to}"'
        map[f'-> "mi::{map_from} *":'] =                        f'-> "{map_to}":'
        map[f': "mi::{map_from} *"'] =                          f': "{map_to}"'
        map[f'-> "mi::{map_from} const *":'] =                  f'-> "{map_to}":'
        map[f': "mi::{map_from} const *"'] =                    f': "{map_to}"'
        map[f'-> "mi::base::{map_from} *":'] =                  f'-> "{map_to}":'
        map[f': "mi::base::{map_from} *"'] =                    f': "{map_to}"'
        map[f'-> "mi::base::{map_from} const *":'] =            f'-> "{map_to}":'
        map[f': "mi::base::{map_from} const *"'] =              f': "{map_to}"'
        map[f'-> "mi::neuraylib::{map_from} *":'] =             f'-> "{map_to}":'
        map[f': "mi::neuraylib::{map_from} *"'] =               f': "{map_to}"'
        map[f'-> "mi::neuraylib::{map_from} const *":'] =       f'-> "{map_to}":'
        map[f': "mi::neuraylib::{map_from} const *"'] =         f': "{map_to}"'
        map[f'-> "SmartPtr< mi::{map_from} > *":'] =            f'-> "{map_to}":'
        map[f'-> "SmartPtr< mi::base::{map_from} > *":'] =      f'-> "{map_to}":'
        map[f'-> "SmartPtr< mi::neuraylib::{map_from} > *":'] = f'-> "{map_to}":'

    def add_type_map_primitive(map: dict, map_from: str, map_to: str):
        map[f'-> "{map_from}":'] =                      f'-> "{map_to}":'
        map[f': "{map_from}"'] =                        f': "{map_to}"'
        map[f'-> "{map_from} const":'] =                f'-> "{map_to}":'
        map[f': "{map_from} const"'] =                  f': "{map_to}"'
        map[f'-> "mi::{map_from}":'] =                  f'-> "{map_to}":'
        map[f': "mi::{map_from}"'] =                    f': "{map_to}"'
        map[f'-> "mi::{map_from} const":'] =            f'-> "{map_to}":'
        map[f': "mi::{map_from} const"'] =              f': "{map_to}"'
        map[f'-> "mi::base::{map_from}":'] =            f'-> "{map_to}":'
        map[f': "mi::base::{map_from}"'] =              f': "{map_to}"'
        map[f'-> "mi::base::{map_from} const":'] =      f'-> "{map_to}":'
        map[f': "mi::base::{map_from} const"'] =        f': "{map_to}"'

    strip: bool = False
    strip_n_lines: int = 0
    class_name: str = ""
    append_to_end_of_class_lines: list[str] = []
    is_iinterface: bool = False
    num_lines = len(Lines)
    line_index: int = 0
    while line_index < num_lines:
        line: str = Lines[line_index]

        if not class_name:
            # drop the unused swig functions in the beginning of the module
            # (this works as long as there are no empty lines in the functions to strip)
            if line.startswith('def _swig_setattr_nondynamic_instance_variable(set):') or \
               line.startswith('def _swig_setattr_nondynamic_class_variable(set):') or \
               line.startswith('def _swig_add_metaclass(metaclass):') or \
               line.startswith('class _SwigNonDynamicMeta(type):'):
                line = line.strip()
                while line != "":  # drop everything until we find the next empty line
                    line_index = line_index + 1
                    line = Lines[line_index].strip()
                line_index = line_index + 2  # drop the empty line as well
                continue

            if line.startswith('if _swig_python_version_info < (2, 7, 0):') or \
               line.startswith('    from . import _pymdlsdk') or \
               line.startswith('except ImportError:') or \
               line.startswith('    except __builtin__.Exception:'):
                line = line.rstrip() + "  # pragma: no cover\n"

        # keep class name for logs
        lineNoWhiteSpaces = line.strip()
        if line.startswith('class '):
            class_name = re.search(r'^class (.*?)\(.*', line).group(1)
            is_iinterface = class_name.startswith("I")
            if not line.startswith('class _'):
                if is_iinterface:
                    add_type_map_iinterface(manualTypeHintMap, class_name, class_name)
                else:
                    add_type_map_primitive(manualTypeHintMap, class_name, class_name)
            
            assert(len(append_to_end_of_class_lines) == 0)

        # end of class
        if line.startswith("# Register "):
            # append the lines we collected for "end of class"
            for l in append_to_end_of_class_lines:
                ProcessedLines.append(l)
            append_to_end_of_class_lines.clear()

        # find interface classes with a leading _
        if line.startswith('class _I'):
            strip = True
        elif strip and line.startswith(f'_pymdlsdk.{class_name}_swigregister'):
            strip = False
            strip_n_lines = 1
            if args.verbose:
                print(f"removed class: '{class_name}'")
        # add empty lines after regular classes that somehow are missing in swig 4.2.1
        elif line.startswith(f'_pymdlsdk.{class_name}_swigregister'):
            ProcessedLines.append(line+"\n")
            line_index = line_index + 1
            continue # stop here since we changed the line_index

        # handle type hint mappings
        elif lineNoWhiteSpaces.startswith('@post_swig_add_type_hint_mapping('):
            deco_args: tuple = make_tuple(lineNoWhiteSpaces[32:])
            manualTypeHintMap[f'-> "{deco_args[0]}":'] = f'-> "{deco_args[1]}":'
            manualTypeHintMap[f': "{deco_args[0]}"'] = f': "{deco_args[1]}"'
            strip_n_lines = 1  # drop this line

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
        elif strip and lineNoWhiteSpaces.startswith( f'return _pymdlsdk.{class_name}_IID()'):
            strip = False
            strip_n_lines = 2
            if args.verbose:
                print(f"removed unused global function: def {class_name}_IID()")
        # and the same with the get_interface function that is needed on the type (with an instance as parameter)
        elif lineNoWhiteSpaces.startswith('def I') and \
            lineNoWhiteSpaces.startswith(f'def {class_name}__get_interface(iface: "mi::base::IInterface *") ->'):
            strip = True
        elif strip and lineNoWhiteSpaces.startswith(f'return _pymdlsdk.{class_name}__get_interface(iface)'):
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
                    print(f"Error: Bindings contain global constant '{general.group(1)}' that isn't handled.")

        # move (extended) functions to the end of the class to make sure regular functions are defined earlier
        elif lineNoWhiteSpaces == '@post_swig_move_to_end_of_class':
            found_function_begin: bool = False
            while(True):
                line_index = line_index + 1
                line = Lines[line_index]
                if line.startswith('    def '):
                    if found_function_begin: # stop when reaching the next function
                        break
                    else:
                        found_function_begin = True
                if line.startswith('    @') and found_function_begin: # stop when reaching a decorator of the next function
                    break
                if line.startswith("# Register "): # stop when reaching the end of the class
                    break
                append_to_end_of_class_lines.append(line)
            continue  # Note, we need skip processing here

        # skip lines or not
        if not strip:
            if strip_n_lines > 0:
                strip_n_lines = strip_n_lines - 1
            else:
                ProcessedLines.append(line)

        # to read the next line.
        # can't use a range iterator here because we need to manipulate the intex
        line_index = line_index + 1


    # second loop over all lines to replace type hints until swig is able to produce correct python names
    # also add some sanity checks
    if args.verbose:
        for key, value in manualTypeHintMap.items():
            print(f"replacing type hints:    {key}    by    {value}")

    num_lines = len(ProcessedLines)
    for i in range(num_lines):
        line: str = ProcessedLines[i]
        lineNoWhiteSpaces: str = line.lstrip()
        # for now, look at signatures only
        if lineNoWhiteSpaces.startswith("def "):
            for key, value in manualTypeHintMap.items():
                line = line.replace(key, value)
            ProcessedLines[i] = line

            # check for deprecated c++ functions that should not be accessible in the bindings
            # note, the python bindings stubs have a leading underscore
            if lineNoWhiteSpaces.startswith("def deprecated_"):
                print(f"Error: Bindings contain a deprecated function: {line.strip()}")
                continue

            # out parameters are not supported directly, they are in handled in 'mdl_python_swig.i'
            if ': "mi::Sint32 *"' in line:
                print(f"Error: Bindings contains an out parameter: {line.strip()}")
                continue

            # missing interface that need to be added to the bindings. Ignoring the function is also possible if desired.
            if line.find("mi::") >= 0:  # issue a warning for now. this could potentially mean an error
                print(f"Error: Bindings contains unhandled type hints: {line.strip()}")

            if ' *"' in line:
                print(f"Error: Bindings contains a pointer in the signature: {line.strip()}")
                continue

            if ' &"' in line:
                print(f"Error: Bindings contains a reference in the signature: {line.strip()}")
                continue


    # write the file
    with open(binding_file_path, 'w') as binding_file:
        binding_file.writelines(ProcessedLines)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-outdir', help='the directory of the create python binding module')
    parser.add_argument('-module', help='the name of the create python binding module without extension')
    parser.add_argument('-v', dest='verbose', action='store_true', default=False, help='print verbose output to log the changed functions and classes')

    args = parser.parse_args()
    binding_file_path = args.outdir + '/' + args.module + '.py'
    print(f"Post-processing binding module: {binding_file_path}")

    run(parser.parse_args(), binding_file_path)
