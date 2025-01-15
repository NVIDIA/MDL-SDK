#!/bin/env python
#*****************************************************************************
# Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
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

# This script generates the get_semantics_name() function to get enum names for enum values.

import os
import sys


class EnumParser:
    def __init__(self, header_path, enum_name, imported_enum=None, import_prefix=None):
        self.header_path = header_path
        self.enum_name = enum_name
        self.enum_values = []
        self.enum_value_map = {}
        if imported_enum:
            for name, val in imported_enum.enum_values:
                self.enum_value_map[import_prefix + name] = val

        self.parse()

    # Normalize an enum entry into the form "<name>['='<value>('+'<value>)*]"
    def normalize_entry(self, line, f):
        comment_idx = line.find("//")
        if comment_idx != -1:
            line = line[:comment_idx].strip()

        # ignore directives so far
        if line.startswith("#"):
            line = ""

        end_of_enum = False
        while True:
            # end of entry?
            if line.endswith(","):
                line = line[:-1].strip()
                break

            # end of enum?
            rbrace_idx = line.find("}")
            if rbrace_idx != -1:
                line = line[:rbrace_idx].strip()
                end_of_enum = True
                break

            # add next line
            nextline = next(f, "") #.next()

            # ignore directives so far
            if nextline.startswith("#"):
                nextline = ""

            comment_idx = nextline.find("//")
            if comment_idx != -1:
                nextline = nextline[:comment_idx].strip()
            line += " " + nextline.strip()

        # remove any spaces
        line = line.replace(" ","")
        return line, end_of_enum

    # Parse the header and extract the name/value mapping of the enum name given in the constructor.
    def parse(self):
        in_enum = False
        cur_val = 0

        with open(self.header_path, "rt") as f:
            for line in f:
                line = line.strip()
                if not in_enum and line.startswith("enum " + self.enum_name):
                    in_enum = True
                elif in_enum:
                    entry, end_of_enum = self.normalize_entry(line, f)

                    if entry:
                        if "=" in entry:
                            # "<name>'='<value>('+'<value>)*" case
                            name, val = entry.split("=")
                            cur_val = 0
                            for added_val_name in val.split("+"):
                                try:
                                    cur_val += int(added_val_name, 0)
                                except:
                                    # not a number, so it should be a known enum
                                    cur_val += self.enum_value_map[added_val_name]
                        else:
                            # "<name>" case
                            name = entry

                        val = cur_val
                        cur_val += 1

                        self.enum_value_map[name] = val
                        self.enum_values.append((name, val))

                    if end_of_enum:
                        break

#
# Main function
#
def main():
    if len(sys.argv) < 3 or len(sys.argv) > 4 or (len(sys.argv) == 4 and sys.argv[3] != "--include-nvidia-dfs"):
        print("Usage: %s public_header_dir output_file [--include-nvidia-dfs]" % sys.argv[0])
        sys.exit(1)

    public_header_dir = sys.argv[1]
    output_file = sys.argv[2]
    include_nvidia_dfs = len(sys.argv) == 4

    expressions_path = os.path.join(public_header_dir, "mi", "mdl", "mdl_expressions.h")
    definitions_path = os.path.join(public_header_dir, "mi", "mdl", "mdl_definitions.h")

    if not os.path.exists(expressions_path) or not os.path.exists(definitions_path):
        print('Invalid public_header_dir parameter, "%s" or "%s" not found!' %
            (expressions_path, definitions_path))
        sys.exit(1)


    # IDefintion::Semantics depends of IExpression::Operator, so parse "Operator" first
    # and use it when parsing "Semantics".
    # Note: this will use the first "Operator" enum it finds in mdl_expresions.h.
    operator_enums = EnumParser(expressions_path, "Operator")
    semantics_enums = EnumParser(definitions_path, "Semantics", operator_enums, "IExpression::")

    with open(output_file, "wt") as f:
        f.write(
            "//*****************************************************************************\n"
            "// Copyright 2024 NVIDIA Corporation. All rights reserved.\n"
            "//*****************************************************************************\n"
            "// Generated by gen_mdltl_enum_names.py\n"
            "\n"
            "#include <mi/mdl/mdl_definitions.h>  // mi::mdl::IDefinition::Semantics\n"
            "\n"
            "char const *get_semantics_name(mi::mdl::IDefinition::Semantics semantics)\n"
            "{\n"
            "    switch (semantics) {\n")

        for name, val in semantics_enums.enum_values:
            # ignore enum entries ending with "_FIRST" or "_LAST" in entry list
            # to avoid duplicate switch cases
            if not name.endswith("_FIRST") and not name.endswith("_LAST") and (include_nvidia_dfs or (not ("NVIDIA_DF" in name))):
                f.write('    case mi::mdl::IDefinition::%s: return "%s";\n' % (name, name))

        # mdltlc depends on a nullptr return value for unknown
        # semantic values because of extended Distiller semantics.
        f.write('    default: return nullptr;\n');
        f.write(
            "    }\n"
            '    return nullptr;\n'
            "}\n")


if __name__ == '__main__':
    main()
