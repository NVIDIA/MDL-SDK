#!/usr/bin/env python3
#*****************************************************************************
# Copyright (c) 2010-2023, NVIDIA CORPORATION. All rights reserved.
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

"""Checks that interface IDs and their hashes in API headers are unique."""

# pylint: disable-msg=C0103

import os
import re
import sys

# global variables
directories = [
    "../include",
    "api/api/mdl",
    "api/api/neuray",
    "prod/lib/neuray",
    "prod/lib/mdl_sdk",
    ]
mixin_names = r'Interface_declare|User_class|Element|Job|Fragmented_job'
filename = ""
iids = []
exitcode = 0

class IID:
    """Stores an interface ID.

    Stores the 11 template IDs, their hash, the filename, line number,
    and textual representation."""

    def __init__ (self):
        self.id = [ None for i in range (0, 11) ]
        self.filename = ""
        self.hash = -1
        self.line = -1
        self.buffer = ""

    def __lt__ (self, other):
        """The comparison operator is defined in terms of the hash."""
        assert self.hash != -1
        assert other.hash != -1
        return self.hash < other.hash

    def set_id (self, index, value):
        """Sets the template id 'index' to 'value'."""
        assert index >= 0
        assert index <= 10
        if value[:2] == "0x":
            self.id[index] = int (value, 16)
        else:
            self.id[index] = int (value, 10)

    def get_id (self, index):
        """Returns the template id 'index'."""
        assert index >= 0
        assert index <= 10
        return self.id[index]

    def compute_hash (self):
        """Computes and stores hash of the interface ID."""
        m_id1 =  self.id[0]
        m_id2 =  self.id[1]        + (self.id[ 2] << 16)
        m_id3 =  self.id[3]        + (self.id[ 4] <<  8) \
              + (self.id[5] << 16) + (self.id[ 6] << 24)
        m_id4 =  self.id[7]        + (self.id[ 8] <<  8) \
              + (self.id[9] << 16) + (self.id[10] << 24)
        self.hash = m_id1 ^ m_id2 ^ m_id3 ^ m_id4

    def get_hash (self):
        """Returns the hash of the interface ID."""
        return self.hash

    def set_filename (self, filename_):
        """Sets the filename of the file that contains the interface ID."""
        self.filename = filename_

    def get_filename (self):
        """Returns the filename of the file that contains the interface ID."""
        return self.filename

    def set_line (self, line):
        """Sets the line number that contains the interface ID."""
        self.line = line

    def get_line (self):
        """Returns the line number that contains the interface ID."""
        return self.line

    def set_buffer (self, buffer):
        """Sets the textual representation of the interface ID."""
        self.buffer = buffer

    def get_buffer (self):
        """Returns the textual representation of the interface ID."""
        return self.buffer

def remove_comments (buffer):
    """Removes C- and C++-style comments from buffer.

    Newlines in C++-style comments are preserved (otherwise line numbers in
    error messages are wrong). Also replaces string constants by '...'/"..."
    to simplify searching of "<" and ">".
    """
    def replacer (match):
        """Replaces subexpressions matched by RE below."""
        s = match.group (0)
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

def remove_if_zero (buffer):
    """Removes '#if 0' sections from buffer.

    Note that only literal '0' is matched, no expressions are evaluated. Blocks
    following #else or #elif are not processed. Newlines are preserved
    (otherwise line numbers in error messages are wrong)."""

    start_pattern = re.compile (r'^#\s*if\s+0($|\s)')
    end_pattern   = re.compile (r'^#\s*(elif|else|endif)')
    replace       = False
    line          = 0
    while line < len (buffer):
        if start_pattern.match (buffer[line]):
            replace = True
        elif end_pattern.match (buffer[line]):
            replace = False
        elif replace:
            buffer[line] = '\n'
        line += 1
    return buffer

def find_next_non_whitespace (buffer, line, col):
    """Finds the next non-whitespace character.

     Starts at position (line, col). Returns (-1,-1) if none is found.
     """
    while line < len (buffer):
        match = re.search (r'\S', buffer[line][col:])
        if match:
            return (line, col + match.start())
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

def process_match (buffer, line_start, col_start, offset):
    """Processes a potential interface ID."""

    global exitcode

    (line, col) \
        = find_next_non_whitespace (buffer, line_start, col_start + offset)
    if buffer[line][col] != "<":
#        print "skipping %s:%d:%d: %s (no template)" \
#            % (filename, line_start, col_start, buffer[line_start][col_start:])
        return

    (line_end, col_end) = find_character (buffer, ">", line_start, col_start)
    if line_end == -1 or col_end == -1:
        print("error %s:%d:%d: %s" \
            % (filename, line_start, col_start, buffer[line_start][col_start:]))
        exitcode = 1
        return

    s = join_lines (buffer, line_start, col_start, line_end, col_end+1)

    match = re.search (r'^(' + mixin_names + r')\s*<([^,]*),([^,]*),([^,]*),([^,]*),([^,]*),([^,]*),([^,]*),([^,]*),([^,]*),([^,]*),([^,]*)(,.*>|>)$', s)
    if not match:
        print("skipping %s:%d: %s (regular expression does not match)" \
            % (filename, line_start, s))
        return

    global iids
    iid = IID()
    iid.set_filename (filename)
    iid.set_line (line_start)
    iid.set_buffer (s)
    try:
        for i in range (0, 11):
            iid.set_id (i, match.group (i+2).strip())
        iid.compute_hash()
        iids += [ iid ]
    except ValueError:
        print("skipping %s:%d: %s (template arguments are no numbers)" \
            % (filename, line_start, s))

def process_buffer (buffer):
    """Processes the buffer contents (read from a file)."""
    buffer = remove_comments (buffer)
    buffer = remove_if_zero (buffer)
    line = 0
    col = 0
    while line < len (buffer):
        match = re.search (r'(^|\W)(' + mixin_names + r')($|\W)', buffer[line][col:])
        if not match:
            line += 1
            col = 0
        else:
            col += match.start (2)
            process_match (buffer, line, col, len (match.group (2)))
            col += 1

def process_file():
    """Processes the file given by the global variable 'filename'."""
    f = open (filename, 'r', encoding="utf-8")
    buffer = f.readlines()
    f.close()
    process_buffer (buffer)

def usage():
    """Prints a usage message."""
    print('Usage: test_unique_IIDs.py <src_dir>')
    print('  Checks that all interface IDs and their hashes are unique.')
    print()
    print('  Returns 0 if all interface IDS and their hashes are unique,')
    print('  1 if not, and 2 in case of other errors.')
    print()
    print('  Limitations:')
    print('  - All searches a set of of hard-coded directories.')
    print('  - Understands only a few hard-coded mixin classes, namely')
    print('    %s.' % mixin_names)
    print()
    print('  Note:')
    print('  - Column numbers in error messages might be larger than or equal')
    print('    to the length of the corresponding line (meaning: first column')
    print('    of next line).')

def main():
    """Checks a set of predefined files in $MI_SRC."""
    if len (sys.argv) != 2:
        usage()
        sys.exit(2)
    if len (sys.argv) == 2 and (sys.argv[1] == "-h" or sys.argv[1] == "--help"):
        usage()
        sys.exit(0)

    src_dir = sys.argv[1]
    os.chdir (src_dir)
    filenames = []
    for directory in directories:
        for root, _, files in os.walk (directory):
            if root.find (".svn") == -1:
                filenames += [ root + os.path.sep + f for f in files ]
    filenames.sort()

    global filename
    for filename in filenames:
        if filename.endswith (".h") or filename.endswith (".cpp"):
            process_file()
    assert len (iids) > 300

    print()
    print("%d interface IDs found" % len (iids))
    print()

    iids.sort()

    global exitcode
    i = 0
    while i < len (iids):
        j = i + 1
        while (j < len (iids)) and (iids[i].get_hash() == iids[j].get_hash()):
            j += 1
        if j > i + 1:
            print("Non unique interface IDs and/or hashes:")
            for k in range (i, j):
                id_ = iids[k]
                print("%s:%d: %s (hash: 0x%08x)" \
                    % (id_.get_filename(), id_.get_line(), id_.get_buffer(), id_.get_hash()))
            print()
            exitcode = 1
        i = j

    if exitcode == 0:
        print("All checked interface IDs and hashes are unique.")

    sys.exit (exitcode)

if __name__ == "__main__":
    main()
