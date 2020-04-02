#!/bin/env python
#*****************************************************************************
# Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
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

# This script generates the MDL runtime header file for libbsdf.
#
# Call it like this:
# python gen_libbsdf_runtime_header.py ../../compiler/stdmodule ../libbsdf/libbsdf_runtime.h
#
# python 2.6 or higher is needed
#
import sys
import re
import os

from gen_intrinsic_func import SignatureParser, error

reference_parameter_types = {
	"bool2",
	"bool3",
	"bool4",
	"color",
	"double2",
	"double3",
	"double4",
	"float2",
	"float3",
	"float4",
	"int2",
	"int3",
	"int4"
}


def eat_until(token_set, tokens):
	"""eat tokens until token_kind is found and return them, handle parenthesis"""
	r = 0
	e = 0
	g = 0
	a = 0
	l = len(tokens)
	eaten_tokens = []
	while l > 0:
		tok = tokens[0]
		if r == 0 and e == 0 and g == 0 and a == 0 and tok in token_set:
			return eaten_tokens, tokens
		if tok == '(':
			r += 1
		elif tok == ')':
			r -= 1
		elif tok == '[':
			e += 1
		elif tok == ']':
			e -= 1
		elif tok == '{':
			g += 1
		elif tok == '}':
			g -= 1
		elif tok == '[[':
			a += 1
		elif tok == ']]':
			a -= 1
		eaten_tokens.append(tokens[0])
		tokens = tokens[1:]
		l -= 1

	# do not return empty tokens, the parser do not like that
	return eaten_tokens, [None]


def format_param(param):
	typename, name, defparam = param

	if typename in reference_parameter_types:
		res = "%s const &%s" % (typename, name)
	else:
		res = "%s %s" % (typename, name)

	if defparam:
		res += " = %s" % defparam
	return res


def parse_prototype(sigparser, decl, prototypes):
	"""Get the C++ prototype for a given function declaration."""
	# poor man's scanner :-)
	tokens = re.sub(r'[,()]', lambda m: ' ' + m.group(0) + ' ', decl).split()

	tokens, ret_type = sigparser.get_type(tokens)

	name = tokens[0]

	if tokens[1] != '(':
		error("unknown token '" + tokens[1] + "' while processing '" + decl + "': '(' expected")
		sys.exit(1)

	tokens = tokens[2:]

	params = []

	if tokens[0] != ')':
		while True:
			tokens, t = sigparser.get_type(tokens)
			paramname = tokens[0]
			tokens = tokens[1:]

			if tokens[0] == '=':
				# default argument
				defarg, tokens = eat_until({',':None, ')':None}, tokens[1:])
			else:
				defarg = []

			params.append((t, paramname, ''.join(defarg)))

			if tokens[0] == ')':
				break
			if tokens[0] != ',':
				error("unknown token '" + tokens[1] + "' while processing '"
					+ decl + "': ',' expected")
				sys.exit(1)
			# skip the comma
			tokens = tokens[1:]

	# For array returns, add one pointer parameter per array element
	if "[" in ret_type:
		match = re.match("([^[]+)\[(\d+)\]", ret_type)
		if match:
			elem_type = match.group(1)
			num_elems = int(match.group(2))
			for i in range(num_elems):
				params.append((elem_type + "*", "res_%d" % i, []))
			ret_type = "void"

	prototype = "%s %s(%s);" % (ret_type, name, ", ".join(map(format_param, params)))
	if "[" in ret_type or name == "transpose" or "[<N>]" in prototype:
		prototype = "// %s  (not supported yet)" % prototype
	prototypes.append(prototype)


def print_wrapped(parser, fileobj, line, wrap_pos = 99):
	"""print the given line (provided without newline at end) and wrap it at wrap_pos,
	   splitting the line at commas. Also handles commented out lines."""
	orig_line = line
	prefix = ""
	next_prefix = "//     " if line.startswith("//") else "    "

	while parser.indent * 4 + len(prefix) + len(line) >= wrap_pos:
		splitpos = line.rfind(',', 0, wrap_pos - parser.indent * 4 - len(prefix))
		if splitpos == -1:
			raise Exception("Unable to split line: %s" % orig_line)
		parser.write(fileobj, prefix + line[:splitpos + 1] + "\n")
		line = line[splitpos + 1:].lstrip()
		prefix = next_prefix

	parser.write(fileobj, prefix + line + "\n")


def usage(args):
	"""print usage info and exit"""
	print "Usage: %s stdlib_directory outputfile" % args[0]
	return 1


def main(args):
	"""Process one file and generate signatures."""
	if len(args) != 3:
		return usage(args)

	stdlib_dir = args[1]
	out_name   = args[2]
	strict     = True

	prototypes = []

	# monkey patch SignatureParser to generate signature names suitable for C++ header files
	SignatureParser.get_signature = (
		lambda self, decl: parse_prototype(self, decl, prototypes))

	try:
		parser = SignatureParser(args[0], stdlib_dir, out_name, strict)

		# copy the copyright from first 3 lines of libbsdf.h
		libbsdf_h_path = os.path.join(os.path.dirname(out_name), "libbsdf.h")
		with open(libbsdf_h_path) as f:
			copyright = "".join([next(f) for x in xrange(3)])

		with open(out_name, "w") as f:
			parser.write(f, copyright)
			parser.write(f, "\n#ifndef MDL_LIBBSDF_RUNTIME_H\n"
				"#define MDL_LIBBSDF_RUNTIME_H\n")

			for module_name in ["math", "debug"]:
				# clear list before parsing next module
				del prototypes[:]

				parser.parse(module_name)

				parser.write(f, "\nnamespace %s\n" % module_name)
				parser.write(f, "{\n")
				parser.indent += 1
				for prototype in prototypes:
					print_wrapped(parser, f, prototype)
				parser.indent -= 1
				parser.write(f, "}\n")

			parser.write(f, "\n#endif  // MDL_LIBBSDF_RUNTIME_H\n")

	except Exception as e:
		error(str(e))
		return 1
	return 0


if __name__ == "__main__":
	sys.exit(main(sys.argv))
