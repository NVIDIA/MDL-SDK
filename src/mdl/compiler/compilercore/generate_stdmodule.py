#!/usr/bin/env python
#*****************************************************************************
# Copyright (c) 2012-2020, NVIDIA CORPORATION. All rights reserved.
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

#
# Convert a mdl file to C++ string literals
#

import glob, os, string, re
import time
import sys

try: # Python 2.2 has no optparse
    from optparse import OptionParser
    no_optparse = False
except:
    no_optparse = True


class Block:
    def __init__(self):
        self.name = ""
        self.comment = []
        self.text = []


def scan_src_file(prefix, filename):
    """
    Read a .mdl file and build a Block(name,comment,text).
    The name is the filename without extension, with module_
    prepended
    """
    block = Block()

    # remove extension and add prefix
    block.name, ext = os.path.splitext(os.path.basename(filename))
    block.name = prefix + "_" + block.name

    in_file = open(filename)
    state = "header"
    for line in in_file:
        # rstrip removes the '\n' too
        line = line.rstrip()

        if state == "header":
            # read header, lines must start with "//"
            res = re.search(r"\s*//(.*)", line)
            if res:
                block.comment.append(res.group(1))
            else:
                block.text.append(line)
                state = "in_text"

        elif state == "in_text":
            block.text.append(line)

    return block


def check_whitespace(src):
    """Check a given file for whitespace errors."""
    f = open(src)
    lineno = 1
    module = os.path.basename(src)
    s_re = re.compile("^[ ]*\t+[ ]*.*")
    e_re = re.compile(".*[ \t]+$")
    bad = False
    for line in f.readlines():
        if s_re.match(line):
            os.sys.stderr.write("%s(%d): error: has whitespace errors at start\n" % (module, lineno))
            bad = True
        elif e_re.match(line):
            os.sys.stderr.write("%s(%d): error: has whitespace errors at end\n" % (module, lineno))
            bad = True
        lineno = lineno + 1
    if bad:
        sys.exit("Module has whitespace errors")


def check_module(checker, src):
    """Run a checker on a given module."""
    path, name = os.path.split(src)
    module_name, ext = header_filename, ext = os.path.splitext(name)
    if module_name == "distilling_support":
        # does not work with this simple logic because it is in nvidia AND requieres base import, so ignore
        return
    retval = os.system(checker + " " + path + " " + module_name)
    if retval != 0:
        sys.exit("Checking module '" + module_name + "' failed! Aborting.")

def write_cpp_header(module, blocks, target):
    # write header
    head = time.strftime( """/******************************************************************************
 * Copyright %Y NVIDIA Corporation. All rights reserved.
 *****************************************************************************/

""")
    target.write(head)

def generate_cpp_file(files,
                      dst,
                      pch,
                      prefix,
                      module,
                      namespace,
                      do_escape,
                      silent,
                      key,
                      checker):
    """
    Generate C++ files (.cpp/.h) from parsed blocks.
    """

    blocks = []
    for src in files:
        check_whitespace(src)
        if checker:
            check_module(checker, src)
        block = scan_src_file(prefix, src)
        blocks.append(block);

    header_filename, ext = os.path.splitext(dst)
    header_filename += '.h'

    # generate .cpp file
    target = open(dst, "w+")

    write_cpp_header(module, blocks, target)

    target.write('#include ' + pch + '\n')
    target.write('#include "' + os.path.basename(header_filename) + '"\n\n')
    target.write('namespace mi {\n')
    target.write('namespace ' + namespace + ' {\n\n')

    # write all blocks
    for block in blocks:
        for str in block.comment:
            target.write("// " + str + '\n')

        text = "\n".join(block.text) + "\n"
        l = len(text)
        kl = len(key)
        target.write("unsigned char const " + block.name + "[%u] = {" % (l))
        first = False
        if key:
            for i in range(l):
                first = False
                if i % 8 == 0:
                    target.write('\n  ')
                    first = True
                code = ord(text[i]) ^ ord(key[i % kl]) ^ (i & 0xFF)
                start = code
                if not first:
                    target.write(' ')
                target.write("0x%02x," % code)
            target.write('\n};\n\n');
        else:
            for i in range(l):
                first = False
                if i % 8 == 0:
                    target.write('\n  ')
                    first = True
                if not first:
                    target.write(' ')
                target.write("0x%02x," % ord(text[i]))
            target.write('\n};\n\n');

    # write footer
    target.write('}  // mi\n}  // ' + namespace + '\n')
    target.close()


    # generate header file
    target = open(header_filename, "w+")
    write_cpp_header(module, blocks, target)

    guard = header_filename[:].replace("\\", "/")
    guard = os.path.basename(guard).upper().replace('.', '_')
    target.write('#ifndef ' + guard + '\n')
    target.write('#define ' + guard + '\n\n')
    target.write('namespace mi {\n')
    target.write('namespace ' + namespace + ' {\n\n')

    # write all blocks
    for block in blocks:
        text = "\n".join(block.text) + "\n"
        l = len(text)
        target.write("extern unsigned char const " + block.name + "[%u];\n" % (l))

    target.write('\n}  // mi\n}  // ' + namespace + '\n')
    target.write('#endif // ' + guard + '\n')
    target.close()


def main():
    try:
        mi_src = os.environ["MI_SRC"];
    except:
        mi_src = "../../.."
    default_dst = os.path.normpath(mi_src + "/mdl/compiler/compilercore")

    if no_optparse:
        class Options:
            def __init__(self):
                self.dst_path  = default_dst
                self.silent    = False
                self.namespace = "mdl"
        options = Options()
        args = []
        #print sys.argv
        state = "default"
        for arg in sys.argv[1:]:
            if   state == "args":
                args.append(arg)
                state = "args" # keep looking for 'em
            elif state == "expect_dst":
                options.dst_path = arg
                state = "default"
            elif state == "expect_namespace":
                options.namespace = arg
                state = "default"
            elif state == "expect_module":
                options.module = arg
                state = "default"
            elif state == "expect_prefix":
                options.prefix = arg
                state = "default"
            elif state == "expect_pch":
                options.pch = arg
                state = "default"
            elif state == "expect_key":
                options.encode = True
                options.key = arg
                state = "default"
            elif state == "default":
                if arg == "-d": state = "expect_dst"
                elif arg == "-n": state = "expect_namespace"
                elif arg == "-s": options.silent = True
                elif arg == "-e": options.do_escape = True
                elif arg == "--escape": options.do_escape = True
                elif arg == "-m": state = "expect_module"
                elif arg == "-p": state = "expect_prefix"
                elif arg == "--prefix": state = "expect_prefix"
                elif arg == "--pch": state = "expect_pch"
                elif arg == "-E": state = "expect_key"
                elif arg == "--encode": state = "expect_key"
                else:
                    args.append(arg)
                    state = "args"
    else:
        parser = OptionParser()

        parser.add_option("-d", "--dst",
                          help="dst C++ file to generate",
                          dest="dst_path")
        parser.add_option("-s", "--silent",
                          help="suppress messages",
                          action="store_true", dest="silent")
        parser.add_option("-e", "--escape",
                          help="escape quotes and backslashes found in files",
                          action="store_true", dest="do_escape")
        parser.add_option("-n", "--namespace",
                          help="namespace to use for generated cpp files",
                          dest="namespace",
                          default="mdl")
        parser.add_option("-m", "--module",
                          help="module name to put into source files",
                          dest="module",
                          default="mdl/compiler/compilercore")
        parser.add_option("-p", "--prefix",
                          help="prefix given value to const char * declarations",
                          dest="prefix",
                          default="mdl_module")
        parser.add_option("--pch",
                          help="precompiled header to use",
                          dest="pch",
                          default='"pch.h"')
        parser.add_option("-E", "--encode",
                          help="encode string using simple XOR chiffre",
                          dest="key",
                          default=None)
        parser.add_option("-c", "--check",
                          help="enforce module check",
                          dest="checker",
                          default=None)
        (options, args) = parser.parse_args()

    if len(args) == 0:
        print("Must supply at least one mdl file as input")
        sys.exit(1)

    if not options.silent:
        print("Creating '%s' from '%s'" % (options.dst_path, ' '.join(args)))

    generate_cpp_file(args,
                      options.dst_path,
                      options.pch,
                      options.prefix,
                      options.module,
                      options.namespace,
                      options.do_escape,
                      options.silent,
                      options.key,
                      options.checker)


if __name__ == "__main__":
    main()
