#!/usr/bin/env python
#*****************************************************************************
# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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
# Preprocessing script for DiCE and IndeX header files to handle SWIG template
# instantiation. It parses SWIG .i files recursively, looking for definitions of
# interfaces (using "NVINDEX_INTERFACE()") as well as for all .h files that are
# included. It then searches the class declarations of these interfaces in the
# .h files and inserts SWIG "%template" statements, writing the modified .h
# files to a separate output directory.

from __future__ import print_function

import errno
import os
import os.path
import re
import sys


# Creates directories recursively, does not fail when they already exist (same
# functionality as exists_ok=True in Python 3.5)
def makedirs_safe(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


class Processor:
    start_file             = None
    base_path_index        = None
    base_path_dice         = None
    output_path            = None
    output_dependency_file = None

    headers            = set()
    nvindex_interfaces = {}
    dice_interfaces    = {}
    log                = set()

    dependencies = []

    # .i files provided by SWIG, these will be ignored
    SWIG_FILES = [
        "exception.i",
        "typemaps.i",
        "carrays.i"
    ]

    # Matches the UUID template argument to "Interface_declare" with optional base class as last parameter
    RE_UUID = '([0-9a-fxA-FX,\s]+)([^0-9\s][\w_:<>,\s]*)?'


    def __init__(self, start_file, base_path_index, base_path_dice, output_path, output_dependency_file):
        self.start_file       = start_file
        self.base_path_index = base_path_index
        self.base_path_dice  = base_path_dice
        self.output_path     = output_path
        self.output_dependency_file = output_dependency_file


    def process_swig_file(self, filename, level = 0):
        if not os.path.exists(filename):
            if filename in self.SWIG_FILES:
                return # skip standard .i files from SWIG

            sys.stderr.write("Included file not found: %s\n" % filename)
            if filename.endswith(".i"):
                sys.stderr.write("If this is a file provided by SWIG then please adapt 'swig_files'.")
            sys.exit(1)

        self.dependencies.append(filename)

        regex_swig_include = re.compile('^\s*%include\s*"([^"]+\.i)"')
        regex_header_include = re.compile('^\s*%include\s*"([^"]+\.h)"')
        regex_header_include_nvindex = re.compile('^\s*NVINDEX_INCLUDE\(([^\)]+\.h)\)')

        regex_nvindex_interface = re.compile('^\s*NVINDEX_INTERFACE[1]?\s*\(([\w_]+)\s*\)')
        regex_dice_interface = re.compile('^\s*DICE_INTERFACE(_BASE|_MI)?\s*\(([\w_]+)\s*\)')
        regex_dice_implement  = re.compile('^\s*DICE_IMPLEMENT\s*\(([\w_]+)\s*,\s*([\w_:]+)\)')
        regex_index_implement = re.compile('^\s*NVINDEX_IMPLEMENT\s*\(([\w_]+)\s*,\s*([\w_:]+)\)')

        lines = open(filename).readlines()
        for line in lines:
            match = regex_swig_include.match(line)
            if match:
               self.process_swig_file(match.group(1), level + 1)

            match = regex_header_include.match(line)
            if match:
                self.headers.add(match.group(1))

            match = regex_header_include_nvindex.match(line)
            if match:
                self.headers.add("nv/index/" + match.group(1))

            match = regex_nvindex_interface.match(line)
            if match and match.group(1) != "CLASS_TYPE":  # skip internal definition
                self.nvindex_interfaces[match.group(1)] = 'nv::index'

            match = regex_dice_interface.match(line)
            if match:
                cmd = match.group(1)
                if not cmd: prefix = 'mi::neuraylib'
                elif cmd == '_BASE': prefix = 'mi::base'
                elif cmd == '_MI': prefix = 'mi'
                self.dice_interfaces[match.group(2)] = prefix

            match = regex_dice_implement.match(line)
            if match:
                self.dice_interfaces[match.group(1)] = 'mi::neuraylib'

            match = regex_index_implement.match(line)
            if match:
                self.nvindex_interfaces[match.group(1)] = 'nv::index'


    def process_header_file(self, filename, base_dir, interfaces):
        self.dependencies.append(base_dir + "/" + filename)

        RE_CLASS = 'class\s*([\w_]+)'

        # Matches the beginning of a template declaration
        regex_template = re.compile('^\s*template\s*<')
        # Matches the beginning of a class declaration (but not a forward declaration)
        regex_class_only = re.compile('^\s*' + RE_CLASS + '\s*[^;]*\s*$')
        # Matches a class declaration using "Interface_declare"
        regex_class_interface_declare = re.compile('^\s*' + RE_CLASS + '\s*:\s*public\s+' +
                                      '([\w::]*)Interface_declare<\s*' + self.RE_UUID + '\s*>' )
        # Matches a class declaration using "Interface_implement"
        regex_class_interface_implement = re.compile('^\s*' + RE_CLASS + '\s*:\s*public\s+' +
                                                     '([\w::]*)Interface_implement<\s*([\w_:]+)\s*>' )
        # Matche the start of a "{"-block
        regex_block_begin = re.compile('^([^{]*){')

        class_declaration = ""
        class_name        = ""
        class_prefix      = ""
        template          = ""
        namespace_begin   = "";
        namespace_end     = "";
        namespace_prefix  = "";
        output            = ""
        file_modified     = False

        # skip neuraylib.h or mdl_sdk.h
        filename_to_open = base_dir + "/" + filename
        if filename_to_open.endswith("/mi/neuraylib.h") or filename_to_open.endswith("/mi/mdl_sdk.h"):
            return

        # Iterate over the lines in the header files
        lines = open(base_dir + "/" + filename).readlines()
        for line in lines:
            # We are currently not inside a class declaration
            if class_name == "":
                # Look for class declaration
                match = regex_class_only.match(line)
                if match and match.group(1) in interfaces:
                    # Found declaration of a class that should be wrapped
                    class_declaration = line
                    class_name = match.group(1)
                    class_prefix = interfaces[class_name] # e.g. "nv::index" or "mi::base"

                    # Set up the current namespace
                    namespace_prefix = class_prefix + "::"
                    namespace_begin  = ""
                    namespace_end    = ""
                    for i in class_prefix.split("::"):
                        namespace_begin += "namespace " + i + " {\n";
                        namespace_end += "}\n";

                else:
                    # Look for a template declaration
                    match = regex_template.match(line)
                    if match:
                        # Template declaration found: Delay output, as a
                        # relevant class declaration may follow, and the SWIG
                        # statements need to be added before the template
                        output += template
                        template = line
                    else:
                        # Neither class not template declaration, just output directly.
                        # First output any previous template declaration.
                        if template != "":
                            output += template
                            template = ""
                        output += line;
            else:
                # We are inside a class declaration for a class that should be wrapped

                # Look for first "{" after "class"
                match = regex_block_begin.match(line)
                if match:
                    # Collect everything starting at "class"
                    output_todo = class_declaration + line
                    class_declaration += match.group(1)

                    done = False

                    def fixup_base(base):
                        if not base:
                            base = ""
                        if base.startswith("base::"): #FIXME: remove
                            base = "mi::" + base
                        if base.startswith("neuraylib::"): #FIXME: remove
                            base = "mi::" + base
                        if base != "" and not (base.startswith("mi::") or base.startswith("nv::")):
                            base = namespace_prefix + base
                        return base

                    def remove_extra_whitespace(s):
                        return re.sub('\s+', ' ', s).strip()

                    #
                    # Handle Interface_declare
                    #
                    match = regex_class_interface_declare.match(class_declaration)
                    if match and not done:
                        (name, uuid, base) = match.group(1, 3, 4)
                        base = fixup_base(base)
                        uuid = remove_extra_whitespace(uuid)

                        instantiation = ""

                        if "<" in base:
                            instantiation += "%template(Interface_declare_" + name + "_base_template) " + base + ";\n"
                            instantiation += "%nodefaultctor Interface_declare_" + name + "_base_template;\n"
                            instantiation += "%nodefaultdtor Interface_declare_" + name + "_base_template;\n"

                        instantiation += "%template(Interface_declare_" + name + ") mi::base::Interface_declare<" + uuid + base + ">;\n"
                        instantiation += "%nodefaultctor Interface_declare_" + name + ";\n"
                        instantiation += "%nodefaultdtor Interface_declare_" + name + ";\n"

                        # #instantiation += "%ignore " + name + ";\n"
                        instantiation += "%rename(_" + name + ") " + name + ";\n"
                        instantiation += "%nodefaultctor " + name + ";\n"
                        instantiation += "%nodefaultdtor " + name + ";\n"
                        # 
                        # if base:
                        # 
                        #     #instantiation += "%ignore mi::base::Interface_declare<" + uuid + base + ">;\n"
                        # 
                        #     base_wo_prefix = base[len(class_prefix) + 2:]
                        #     prefix_current = ""
                        #     for i in reversed(class_prefix.split("::")):
                        #        prefix_current = i + "::" + prefix_current
                        #        instantiation += "%ignore mi::base::Interface_declare<" + uuid + prefix_current + base_wo_prefix + ">;\n"
                        # 
                        # instantiation += "%rename(\"%s\", %$isenum) " + class_prefix + "::" + name + "::Kind;\n"
                        # instantiation += "%template(Interface_declare_" + name + ") mi::base::Interface_declare<" + uuid + base + ">;\n"

                        self.log.add(instantiation)
                        output += namespace_end + instantiation + namespace_begin;
                        file_modified = True
                        done = True

                    #
                    # Handle Interface_implement
                    #
                    match = regex_class_interface_implement.match(class_declaration)
                    if match and not done:
                        (name, base) = match.group(1, 3)
                        base = fixup_base(base)
                        base = remove_extra_whitespace(base)

                        implementation = "%template(Interface_implement_" + name + ") mi::base::Interface_implement<" + base + ">;\n"
                        instantiation += "%nodefaultctor Interface_implement_" + name + ";\n"
                        instantiation += "%nodefaultdtor Interface_implement_" + name + ";\n"
                        # implementation = "%ignore mi::base::Interface_implement<" + base + ">;\n"

                        self.log.add(implementation)
                        output += namespace_end + implementation + namespace_begin;
                        file_modified = True
                        done = True

                    if not done:
                        # Found "class" but couldn't understand declaration
                        raise Exception("Could not parse declaration of class '" + class_name +
                                        "' in file '" + filename + "': " + class_declaration)

                    del interfaces[class_name] # This has been handled
                    class_declaration = ""
                    class_name = ""

                    # Output any previous template declaration.
                    if template != "":
                        output += template
                        template = ""

                    output += output_todo
                else:
                    # Still inside the class declaration, before "{"
                    class_declaration += line

        # Write file if it was modified
        if file_modified:
            output_filename = self.output_path + "/" + filename
            makedirs_safe(os.path.dirname(output_filename))
            open(output_filename, "w").write(output)


    def process(self):
        self.process_swig_file(self.start_file)

        # Iterate over all found headers
        for h in self.headers:
            if h.startswith("nv/"):
                # Handle IndeX headers
                interfaces = self.nvindex_interfaces
                base = self.base_path_index
            elif h.startswith("mi/"):
                # Handle DiCE headers
                interfaces = self.dice_interfaces
                base = self.base_path_dice
            else:
                # Skip other headers (i.e. headers defined as part of the SWIG wrapper)
                continue

            self.process_header_file(h, base, interfaces)

        # Debug info about what interfaces haven been found
        sys.stderr.write("\nprocessed interfaces\n")
        self.log = sorted(self.log)
        # for interface in self.log:
        #     sys.stderr.write("\n%s" % interface)

        # Check for interfaces that were defined (e.g. with NVINDEX_INTERFACE()) but not found in the header files
        if self.nvindex_interfaces:
            sys.stderr.write("Warning: Defined IndeX interfaces not found in headers: %s\n" % self.nvindex_interfaces)
        if self.dice_interfaces:
            sys.stderr.write("Warning: Defined DiCE interfaces not found in headers: %s\n" % self.dice_interfaces)

        # Write dependency file
        with open(self.output_dependency_file + ".d", "w") as f:
            f.write("# dependency file generated by " + sys.argv[0] + "\n")
            f.write(os.path.abspath(self.output_dependency_file) + ": \\\n")
            for d in self.dependencies:
                f.write("  " + os.path.abspath(d) + " \\\n")
            f.write("\n\n")

        # Placeholder file must be written after dependency file
        with open(self.output_dependency_file, "w") as f:
            f.write("/* placeholder file generated by " + sys.argv[0] + " */\n")


if __name__ == "__main__":
    if len(sys.argv) != 6:
        sys.stderr.write("usage: %s <swig-file> <index-header-dir> <dice-header-dir> <output-dir> <output-dependencies>\n" % sys.argv[0])
        sys.exit(1)

    # pass parameters unpacked from argument list
    Processor(*sys.argv[1:]).process()
