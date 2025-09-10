#!/usr/bin/env python3
#*****************************************************************************
# Copyright (c) 2009-2025, NVIDIA CORPORATION. All rights reserved.
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

"""Checks argument and return types of virtual methods in public API."""

# pylint: disable-msg=C0103

import os
import re
import sys

# global variables
filename = ""
methodname = ""
blacklist = [ "int", "float", "double", "size_t" ]
whitelist = []
exitcode = 0

def remove_comments (buffer):
    """Removes C- and C++-style comments from buffer.

    Newlines in C++-style comments are preserved (otherwise line numbers in
    error messages are wrong). Also replaces string constants by '...'/"..."
    to simplify searching of "," and ")" in parameter lists.
    """
    def replacer (m):
        """Replaces subexpressions matched by RE below."""
        s = m.group (0)
        if s.startswith ('/*'):
            return "" + "\n" * s.count ('\n')
        if s.startswith ('/'):
            return ""
        if s.startswith ('"'):
            return "..."
        if s.startswith ('\''):
            return "'...'"
        assert False
        return s

    text = ''.join (buffer)
    pattern = \
        re.compile (r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE)
    text = re.sub (pattern, replacer, text)
    return text.split ('\n')

def find_next_non_whitespace (buffer, line, col):
    """Finds the next non-whitespace character.

     Starts at position (line, col). Returns (-1,-1) if none is found.
     """
    while line < len (buffer):
        m = re.search (r'\S', buffer[line][col:])
        if m:
            return (line, col + m.start())
        line += 1
        col = 0
    return (-1, -1)

def find_character (buffer, character, line, col):
    """Finds the next occurrence of the given character.

    Starts at position (line,col). Returns (-1,-1) if none is found.
    """
    while line < len (buffer):
        index = buffer[line].find (character, col)
        if index != -1:
            return (line, index)
        line += 1
        col = 0
    return (-1, -1)

def join_lines (buffer, line_start, col_start, line_end, col_end):
    """Joins buffer contents.

    Joins contents from (line_start,col_start) to (line_end,col_end)
    (excluded) to a single string. Replaces sequences of whitespace by a
    single space.
    """
    assert line_start <= line_end
    assert col_start <= col_end or line_start < line_end
    if line_start == line_end:
        return buffer[line_start][col_start:col_end]
    s = buffer[line_start][col_start:]
    for line in range (line_start+1, line_end):
        s += buffer[line].rstrip ('\n')
    s += buffer[line_end][:col_end]
    s = re.sub (r'\s+', ' ', s)
    return s

def report_error (buffer, line, col, param):
    """Prints an error message for blacklist/non-POD types.

    The error concerns the given parameter at buffer position (line,col).
    """
    (line_error, col_error) = find_next_non_whitespace (buffer, line, col)
    if param in blacklist:
        print('%s:%d:%d: unwanted type `%s\' at API boundary' \
            % (filename, line_error+1, col_error+1, param))
    else:
        print('%s:%d:%d: non-POD type `%s\' at API boundary' \
            % (filename, line_error+1, col_error+1, param))
    print(buffer[line_error])
    print('%s' % (' '*col_error + '^'))
    global exitcode
    exitcode = 1

def report_mi_warning (buffer, line, col):
    """Prints a warning for the mi:: qualifier.

    The warning concerns the given parameter at buffer position (line,col).
    """
    (line_error, col_error) = find_next_non_whitespace (buffer, line, col)
    print("%s:%d:%d: please consider removing the 'mi::' qualifier " \
        "unless necessary" % (filename, line_error+1, col_error+1))
    print(buffer[line_error])
    print('%s' % (' '*col_error + '^'))

def process_return_type (buffer, line_start, col_start, line_end, col_end):
    """Processes the return type for a virtual method.

    The 'virtual' keyword starts at (line_start,col_start). The "(" of the
    parameter list is at (line_end,col_end).
    """
    param = join_lines (buffer, line_start, col_start, line_end, col_end)
    assert param.startswith ('virtual')
    param = param[len('virtual'):]            # strip 'virtual' keyword
    col_start += len('virtual')
    param = param.strip()
    if param[0] == '~':                       # virtual destructor
        return
    index = param.rfind (' ')                 # strip method name
    assert index != -1
    global methodname
    methodname = param[index+1:]
    param = param[:index]
    param = re.sub (r'\\ *', '', param)       # remove leading backslash
                                              # (line continuation)
    param = re.sub (r' +\*', '*', param)      # strip space in front of *
    param = re.sub (r' +\&', '&', param)      # strip space in front of &
    param = re.sub (r'\*+ *$', '', param)     # strip trailing *'s
    param = re.sub (r'\& *$', '', param)      # strip trailing &
    param = re.sub (r'\bconst\b', '', param)  # strip const modifier
    param = re.sub (r'\binline\b', '', param) # strip inline modifier
    param = param.strip()
    if param.startswith ('mi::'):
        report_mi_warning (buffer, line_start, col_start)
        param = param[4:]
    if param not in whitelist:
        report_error (buffer, line_start, col_start, param)


def process_parameter (buffer, line_start, col_start, line_end, col_end):
    """Processes a parameter.

    The parameter starts at (line_start,col_start) and ends (comma or closing
    parenthesis) at (line_end,col_end)."""
    param = join_lines (buffer, line_start, col_start, line_end, col_end)
    param = param.strip()
    if param == "":                           # void argument list
        return
    param = re.sub (r'\\ *', '', param)       # remove leading backslash
                                              # (line continuation)
    param = re.sub (r'=.*$', '', param)       # strip default argument
    param = re.sub (r' +\*', '*', param)      # strip space in front of *
    param = re.sub (r' +\&', '&', param)      # strip space in front of &
    param = re.sub (r'\*[^\*]*$', '*', param) # strip parameter name after *
    param = re.sub (r'\&[^\&]*$', '&', param) # strip parameter name after &
    param = re.sub (r'\*+$', '', param)       # strip trailing *'s
    param = re.sub (r'\&$', '', param)        # strip trailing &
    param = re.sub (r'const ', '', param)     # strip const modifier
    param = re.sub (r' .*$', '', param)       # strip ' ' and remainder
    param = re.sub (r'\*+$', '', param)       # strip trailing *'s
    if param.startswith ('mi::'):
        report_mi_warning (buffer, line_start, col_start)
        param = param[4:]
    if param not in whitelist:
        report_error (buffer, line_start, col_start, param)

def process_parameter_list (buffer, line_open, col_open, line_close, col_close):
    """Processes a parameter list.

    The parameter list starts at (line_open,col_open) (which is actually the
    next character after "(")) and ends at (line_close,col_close) (the ")").
    """
    (line_start, col_start) = (line_open, col_open)
    (line_end, col_end) = find_character (buffer, ',', line_start, col_start)
    while line_end != -1 and col_end != -1 \
        and ((line_end == line_close and col_end < col_close) \
            or (line_end < line_close)):
        process_parameter (buffer, line_start, col_start, line_end, col_end)
        line_start = line_end
        col_start = col_end+1
        (line_end, col_end) = \
            find_character (buffer, ',', line_start, col_start)
    process_parameter (buffer, line_start, col_start, line_close, col_close)

def process_virtual_method (buffer, line, col):
    """Processes a virtual method.

    The 'virtual' keyword starts at (line,col).
    """
    m = re.search (r'^(.*)MI_NEURAYLIB_DEPRECATED_METHOD_[1-9]*._.\(([^(]*)\)(.*)$', buffer[line])
    if m:
        buffer[line] = m.group (1) + m.group (2) + m.group (3)
    (line_open, col_open) = find_character (buffer, '(', line, col+1)
    assert line_open != -1
    assert col_open != -1
    (line_close, col_close) = \
        find_character (buffer, ')', line_open, col_open+1)
    assert line_close != -1
    assert col_close != -1
    process_return_type (buffer, line, col, line_open, col_open)
    process_parameter_list (
        buffer, line_open, col_open+1, line_close, col_close)

def process_buffer (buffer):
    """Processes the buffer contents (read from a file)."""
    buffer = remove_comments (buffer)
    line = 0
    col = 0
    while line < len (buffer):
        m = re.search (r'(^|\W)(virtual)($|\W)', buffer[line][col:])
        if not m:
            line += 1
            col = 0
        else:
            col += m.start (2)
            process_virtual_method (buffer, line, col)
            col += 1

def process_file():
    """Processes the file given by the global variable 'filename'."""
    f = open (filename, 'r', encoding="utf-8")
    buffer = f.readlines()
    f.close()
    process_buffer (buffer)

def usage():
    """Prints a usage message."""
    print('Usage: test_types_at_api_boundary.py <include_dir> <whitelist>')
    print('  Checks that virtual methods only use types as arguments and return')
    print('  values that are listed in the file \'whitelist\'.')
    print()
    print('  The examined files are base.h, math.h, and neuraylib.h,')
    print('  including files in the corresponding subdiretories. Needs $MI_SRC.')
    print()
    print('  Returns 0 if all used types are on the whitelist, 1 if some types')
    print('  are not on the whitelist, and 2 in case of other errors.')
    print()
    print('  Limitations:')
    print('  - No support for lookup rules and namespaces.')
    print('  - Checks only virtual methods (crucial for the simplicity of the')
    print('    parser).')
    print('  - Constructor calls in default arguments not handled correctly')
    print('    (could be handled by tracking "(" while searching for ")").')
    print('  - Type names of parameters consisting of several tokens (const,')
    print('    *, & do not count here) are not handled correctly, unless there')
    print('    is a final *, &, or a variable name (could be handled by ')
    print('    matching prefixes of the parameter against the whitelist).')
    print()
    print('  Note:')
    print('  - Column numbers in error messages might be larger than or equal')
    print('    to the length of the corresponding line (meaning: first column')
    print('    of next line).')

def main():
    """Checks a set of predefined files in $MI_SRC/public."""
    if len (sys.argv) != 3:
        usage()
        sys.exit(2)
    if sys.argv[1] == "-h" or sys.argv[1] == "--help":
        usage()
        sys.exit(0)

    print('Warning: see the -h or --help output for known limitations.')
    print()

    global whitelist
    f = open (sys.argv[2], 'r')
    whitelist = f.readlines()
    f.close()
    for i, _ in enumerate(whitelist):
        whitelist[i] = whitelist[i].strip ('\n')

    include_dir = sys.argv[1]
    filenames = []
    for f in [ "base.h", "math.h", "mdl_sdk.h",
            ]:
        filenames += [ "mi/" + f ]
    for directory in [ "base", "math", "neuraylib",
            ]:
        filenames += [ "mi/" + directory + "/" + f
            for f in os.listdir (include_dir + "/mi/" + directory)
            if f != ".svn" and f != "iterator.h" ]
    filenames.sort()
    os.chdir (include_dir)

    global filename
    for filename in filenames:
        process_file()
    sys.exit (exitcode)

if __name__ == "__main__":
    main()
