#!/bin/env python
#
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
#
# This script generated signatures for compiler known functions.
#
# python 2.3 or higher is needed 
#
import sys
import os
import re

def error(msg):
	"""Write a message to stderr"""
	sys.stderr.write("gen_intrinsic_eval: Error: " + msg + "\n")

def warning(msg):
	"""Write a message to stderr"""
	sys.stderr.write("gen_intrinsic_eval: Warning: " + msg + "\n")

def make_temp_file():
	"""Return a temporary file name"""
	fd, name = tempfile.mkstemp()
	os.close(fd)
	return name

class SignatureParser:
	"""main signature parser"""

	def __init__(self, script_name, indir, out_name, strict):
		"""constructor"""
		self.debug = False
		self.indir  = indir
		self.out_name = out_name
		self.r_intrinsic = re.compile(r"\[\[\s+intrinsic\(\)[^]]*\]\];")
		self.curr_module = ""
		self.m_intrinsics = {}
		self.m_intrinsic_mods = {}
		self.m_signatures = {}
		self.indent = 0
		self.strict = strict
		
		self.intrinsic_modes = {}
		
		#
		# ADD NEW TYPES HERE!
		#
		self.m_types = {
			"bool"       : "BB",
			"bool2"      : "B2",
			"bool3"      : "B3",
			"bool4"      : "B4",
			"int"        : "II",
			"int2"       : "I2",
			"int3"       : "I3",
			"int4"       : "I4",
			"float"      : "FF",
			"float2"     : "F2",
			"float3"     : "F3",
			"float4"     : "F4",
			"double"     : "DD",
			"double2"    : "D2",
			"double3"    : "D3",
			"double4"    : "D4",
			"color"      : "CC",
			"float2x2"   : "F22",
			"float2x3"   : "F23",
			"float2x4"   : "F24",
			"float3x2"   : "F32",
			"float3x3"   : "F33",
			"float3x4"   : "F34",
			"float4x2"   : "F42",
			"float4x3"   : "F43",
			"float4x4"   : "F44",
			"double2x2"  : "D22",
			"double2x3"  : "D23",
			"double2x4"  : "D24",
			"double3x2"  : "D32",
			"double3x3"  : "D33",
			"double3x4"  : "D34",
			"double4x2"  : "D42",
			"double4x3"  : "D43",
			"double4x4"  : "D44",
			
			"float[2]"   : "FA2",
			"float2[2]"  : "F2A2",
			"float3[2]"  : "F3A2",
			"float4[2]"  : "F4A2",
			"double[2]"  : "DA2",
			"double2[2]" : "D2A2",
			"double3[2]" : "D3A2",
			"double4[2]" : "D4A2",
			"float[<N>]" : "FAN",
			"float[N]"   : "FAn",
		}
		
		# create inverse mapping
		self.m_inv_types = {}
		
		for type, code in self.m_types.items():
			old_type = self.m_inv_types.setdefault(code, type)
			if type != old_type:
				error("type code %s is not unique, used by '%s' and '%s'" % (code, old_type, type))

	def split_signature(self, signature):
		"""Split a signature into return type and parameter types."""
		params = signature.split('_')
		ret_type = params[0]
		params = params[1:]
		
		return ret_type, params

	def get_atomic_value_kind(self, type_code):
		"""If type_code is an atomic value, return its value kind, else None."""
		cases = {
			"bool":   "IValue::VK_BOOL",
			"int":    "IValue::VK_INT",
			"float":  "IValue::VK_FLOAT",
			"double": "IValue::VK_DOUBLE",
			"color":  "IValue::VK_RGB_COLOR",
			"string": "IValue::VK_STRING"
		}
		return cases.get(self.m_inv_types[type_code], None)

	def get_vector_type_kind(self, type_code):
		"""If type_code is an vector value, return its type kind, else None."""
		cases = {
			"bool2":   "IType::TK_BOOL",
			"bool3":   "IType::TK_BOOL",
			"bool4":   "IType::TK_BOOL",
			"int2":    "IType::TK_INT",
			"int3":    "IType::TK_INT",
			"int4":    "IType::TK_INT",
			"float2":  "IType::TK_FLOAT",
			"float3":  "IType::TK_FLOAT",
			"float4":  "IType::TK_FLOAT",
			"double2": "IType::TK_DOUBLE",
			"double3": "IType::TK_DOUBLE",
			"double4": "IType::TK_DOUBLE",
		}
		return cases.get(self.m_inv_types[type_code], None)

	def get_vector_type_and_size(self, type_code):
		"""If type_code is an vector value, return its (element type, size) pair else None."""
		cases = {
			"bool2":   ("bool",   2),
			"bool3":   ("bool",   3),
			"bool4":   ("bool",   4),
			"int2":    ("int",    2),
			"int3":    ("int",    3),
			"int4":    ("int",    4),
			"float2":  ("float",  2),
			"float3":  ("float",  3),
			"float4":  ("float",  4),
			"double2": ("double", 2),
			"double3": ("double", 3),
			"double4": ("double", 4),
			"color":   ("float",  3)
		}
		return cases.get(self.m_inv_types[type_code], None)

	def get_matrix_type_kind(self, type_code):
		"""If type_code is an matrix value, return its type kind, else None."""
		cases = {
			"float2x2"   : "IType::TK_FLOAT",
			"float2x3"   : "IType::TK_FLOAT",
			"float2x4"   : "IType::TK_FLOAT",
			"float3x2"   : "IType::TK_FLOAT",
			"float3x3"   : "IType::TK_FLOAT",
			"float3x4"   : "IType::TK_FLOAT",
			"float4x2"   : "IType::TK_FLOAT",
			"float4x3"   : "IType::TK_FLOAT",
			"float4x4"   : "IType::TK_FLOAT",
			"double2x2"  : "IType::TK_DOUBLE",
			"double2x3"  : "IType::TK_DOUBLE",
			"double2x4"  : "IType::TK_DOUBLE",
			"double3x2"  : "IType::TK_DOUBLE",
			"double3x3"  : "IType::TK_DOUBLE",
			"double3x4"  : "IType::TK_DOUBLE",
			"double4x2"  : "IType::TK_DOUBLE",
			"double4x3"  : "IType::TK_DOUBLE",
			"double4x4"  : "IType::TK_DOUBLE",
		}
		return cases.get(self.m_inv_types[type_code], None)

	def write(self, f, s):
		"""write string s to file f after doing indent."""
		for i in range(self.indent):
			f.write("    ")
		f.write(s)

	def parse(self, mdl_name):
		"""Parse a mdl module."""
		self.curr_module = mdl_name
		fname = self.indir + "/" + mdl_name + ".mdl"
		f = open(fname, "r")
		o = self.parse_file(f)
		f.close()
		
	def as_intrinsic_function(self, decl):
		"""Check if the given declaration is an intrinsic function declaration."""
		if decl[:5] == "const":
			return None
		if decl[:4] == "enum":
			return None
		if decl[:5] == "struct":
			return None
		if decl[:8] == "material":
			return None
		m = self.r_intrinsic.search(decl)
		if m:
			decl = decl[:m.start()]
			# kill all other annotations
			return re.sub(r'\[\[[^]]*\]\]', "", decl).strip()
		return None
		
	def get_type(self, tokens):
		"""decode a type"""
		start = 0
		end   = 1
		if tokens[0] == "uniform" or tokens[0] == "varying":
			# skip uniform and varying modifier
			end += 1
			start += 1
		ret_type = " ".join(tokens[start:end])
		
		return tokens[end:], ret_type

	def do_get_type_code(self, s):
		"""get the type code"""
		try:
			return self.m_types[s]
		except KeyError as e:
			error("Unsupported type '" + s + "' found")
			sys.exit(1)
		
	def get_type_code(self, s):
		"""get the type code"""
		c = self.do_get_type_code(s)
		
		return c

	def create_signature(self, ret_type, args):
		"""create the signature"""
		ret_tp = self.get_type_code(ret_type)
		sig    = ""
		
		for arg in args:
			sig += '_' + self.get_type_code(arg)
			
		return ret_tp + sig

	def is_float_type(self, type_code):
		"""If type_code is an float type, return True, else False."""
		cases = {
			"float":  True,
			"double": True,
		}
		return cases.get(self.m_inv_types[type_code], False)
		
	def is_int_type(self, type_code):
		"""If type_code is an int type, return True, else False."""
		return self.m_inv_types[type_code] == "int"
		
	def is_bool_type(self, type_code):
		"""If type_code is a bool type, return True, else False."""
		return self.m_inv_types[type_code] == "bool"
		
	def is_atomic_type(self, type_code):
		"""If type_code is a bool, int, or float type, return True, else False."""
		return self.is_bool_type(type_code) or self.is_int_type(type_code) or self.is_float_type(type_code)
		
	def is_math_supported(self, name, signature):
		"""Checks if the given math intrinsic is supported."""
		ret_type, params = self.split_signature(signature)

		base = None
		dim  = 0
		vt = self.get_vector_type_and_size(ret_type)
		if vt:
			base = vt[0]
			dim  = vt[1]

		all_atomic    = self.is_atomic_type(ret_type)
		all_base_same = base != None
		for param in params:
			if not self.is_atomic_type(param):
				all_atomic = False
			if self.m_inv_types[param] != base:
				vt = self.get_vector_type_and_size(param)
				if not vt or vt[0] != base or vt[1] != dim:
					all_base_same = False

		if len(params) == 1:
			if name == "blackbody" and params[0] == "FF" :
				self.intrinsic_modes[name + signature] = "math::blackbody"
				return True
			elif name == "average":
				# support average with one argument
				self.intrinsic_modes[name + signature] = "math::average"
				return True
			elif name == "DX" or name == "DY":
				vt = self.get_vector_type_and_size(params[0])
				if (params[0] == "FF" or params[0] == "DD" or
					(vt and (vt[0] == "float" or vt[0] == "double"))):
					# support DX(floatX) and DY(floatX)
					self.intrinsic_modes[name + signature] = "math::DX|DY"
					return True
				return False  # Not yet supported derivations cannot be handled component_wise

		if len(params) == 2:
			if name == "emission_color" and params[0] == "FAN" and params[1] == "FAn":
				self.intrinsic_modes[name + signature] = "math::emission_color_spectrum"
				return True

		if all_atomic and self.is_atomic_type(ret_type):
			# simple all float/int/bool functions
			self.intrinsic_modes[name + signature] = "all_atomic"
			return True

		if len(params) == 1:
			if name == "any" or name == "all":
				# support any and all with one argument
				self.intrinsic_modes[name + signature] = "math::any|all"
				return True
			elif name == "isnan" or name == "isfinite":
				if self.get_vector_type_and_size(params[0]) or self.is_atomic_type(params[0]):
					# support all isnan/isfinite with one argument
					self.intrinsic_modes[name + signature] = "math::isnan|isfinite"
					return True
			elif name == "luminance":
				if params[0] == "F3" or params[0] == "CC":
					# support luminance(float3) and luminance(color)
					self.intrinsic_modes[name + signature] = "math::luminance"
					return True
			elif name == "max_value" or name == "min_value":
				if params[0] == "CC":
					# support max_value(color) and min_value(color)
					self.intrinsic_modes[name + signature] = "math::max_value|min_value"
					return True
				else:
					vt = self.get_vector_type_and_size(params[0])
					if vt and (vt[0] == "float" or vt[0] == "double"):
						# support max_value(floatX) and min_value(floatX)
						self.intrinsic_modes[name + signature] = "math::max_value|min_value"
						return True
			elif name == "max_value_wavelength" or name == "min_value_wavelength":
				# support max_value_wavelength(color) and min_value_wavelength(color)
				self.intrinsic_modes[name + signature] = "math::max_value_wavelength|min_value_wavelength"
				return True
			elif name == "length" or name == "normalize":
				vt = self.get_vector_type_and_size(params[0])
				if params[0] != "CC" and vt and (vt[0] == "float" or vt[0] == "double"):
					# support length(floatX) and normalize(floatX)
					self.intrinsic_modes[name + signature] = "math::length|normalize"
					return True
				return False  # Not yet supported modes may not be handled via component_wise
			elif name == "transpose":
				if self.get_matrix_type_kind(params[0]):
					# support length(floatX)
					self.intrinsic_modes[name + signature] = "math::transpose"
					return True
			elif name == "emission_color":
				if params[0] == "CC":
					self.intrinsic_modes[name + signature] = "math::emission_color_color"
					return True
		if name == "distance" or name == "dot":
			if len(params) == 2:
				if params[0] == params[1] and params[0] != "CC":
					vt = self.get_vector_type_and_size(params[0])
					if vt and (vt[0] == "float" or vt[0] == "double"):
						# support distance(floatX)
						self.intrinsic_modes[name + signature] = "math::distance|dot"
						return True

		if name == "cross":
			if signature == "F3_F3_F3" or signature == "D3_D3_D3":
				# the only supported cross variant
				self.intrinsic_modes[name + signature] = "math::cross"
				return True
			else:
				return False

		if name == "sincos":
			if len(params) == 1:
				arg_tp = params[0]
				if self.is_float_type(arg_tp):
					# support sincos for float types
					self.intrinsic_modes[name + signature] = "math::sincos"
					return True
				vt = self.get_vector_type_and_size(arg_tp)
				if vt and (vt[0] == "float" or vt[0] == "double"):
					# support sincos for float vector types
					self.intrinsic_modes[name + signature] = "math::sincos"
					return True
			return False

		if name == "modf":
			if len(params) == 1:
				arg_tp = params[0]
				if self.is_float_type(arg_tp):
					# support modf for float types
					self.intrinsic_modes[name + signature] = "math::modf"
					return True
				vt = self.get_vector_type_and_size(arg_tp)
				if vt and (vt[0] == "float" or vt[0] == "double"):
					# support modf for float vector types
					self.intrinsic_modes[name + signature] = "math::modf"
					return True
			return False

		if name == "eval_at_wavelength":
			self.intrinsic_modes[name + signature] = "math::eval_at_wavelength"
			return True

		if all_base_same:
			# assume component operation
			self.intrinsic_modes[name + signature] = "math::component_wise"
			return True

		return False
		
	def is_supported(self, modname, name, signature):
		"""Checks if the given intrinsic is supported."""
		if modname == "math":
			return self.is_math_supported(name, signature)
		return False

	def get_signature(self, decl):
		"""Get the signature for a given function declaration."""
		# poor man's scanner :-)
		tokens = re.sub(r'[,()]', lambda m: ' ' + m.group(0) + ' ', decl).split()
		
		tokens, ret_type = self.get_type(tokens)
		
		name = tokens[0]
		
		self.m_intrinsic_mods[name] = self.curr_module
		
		if tokens[1] != '(':
			error("unknown token '" + tokens[1] + "' while processing '" + decl + "': '(' expected")
			sys.exit(1)
			
		tokens = tokens[2:]
		
		args = []
		
		if tokens[0] != ')':
			while True:
				tokens, t = self.get_type(tokens)
				args.append(t)
				
				# throw away the name
				tokens = tokens[1:]
				
				if tokens[0] == ')':
					break
				if tokens[0] != ',':
					error("unknown token '" + tokens[1] + "' while processing '"
					      + decl + "': ',' expected")
					sys.exit(1)
				# skip the comma
				tokens = tokens[1:]

		signature = self.create_signature(ret_type, args)

		if self.debug:
			print("%s %s" % (decl, signature))

		if self.is_supported(self.curr_module, name, signature):
			# insert the new signature for the given name
			sigs = self.m_intrinsics.setdefault(name, {})
			sigs[signature] = True
			
			# remember the signature (without return type)
			_, params = self.split_signature(signature)
			self.m_signatures["_".join(params)] = True
		else:
			warning("Cannot evaluate %s" % decl)

		return ""
		
	def parse_file(self, f):
		"""Parse a file and retrieve intrinsic function definitions."""
		
		start = False
		curr_line = ""
		for line in f.readlines():
			l = line.strip();
			
			if not start:
				if l[:6] == "export":
					start = True
					curr_line = l[7:].strip()
			else:
				curr_line += l
			if start:
				if l[-1] == ";":
					start = False
					decl = self.as_intrinsic_function(curr_line)
					if not decl:
						continue
					if self.debug:
						print(decl)
					sig = self.get_signature(decl)
					if self.debug:
						print(sig)

	def gen_condition(self, f, params, as_assert, pre_if = ""):
		"""Generate the condition for the parameter type check."""
		if as_assert:
			self.write(f, "MDL_ASSERT(check_sig_%s(arguments));\n" % "_".join(params))
		else:
			self.write(f, "%sif (check_sig_%s(arguments)) {\n" % (pre_if, "_".join(params)))

	def create_evaluation(self, f, intrinsic, signature):
		"""Create the evaluation call for a given intrinsic, signature pair."""
		ret_type, params = self.split_signature(signature)

		mode = self.intrinsic_modes.get(intrinsic + signature)

		if mode == "all_atomic":
			self.write(f, "// atomic\n")
			idx = 0
			for param in params:
				kind = self.m_inv_types[param]
				self.write(f, "%s const %s = cast<IValue_%s>(arguments[%d])->get_value();\n" % (
					kind, chr(ord('a') + idx), kind, idx))
				idx += 1
			kind = self.m_inv_types[ret_type]
			call = ("return value_factory->create_%s(" + intrinsic + "(") % kind
			idx = 0
			comma = ""
			for param in params:
				call += comma
				comma = ", "
				call += chr(ord('a') + idx)
				idx += 1
			call += "));\n"
			self.write(f, call);
			return

		elif mode == "math::eval_at_wavelength":
			# FIXME: not supported yet
			self.write(f, "return value_factory->create_float(0.0f);\n")
			return

		elif mode == "math::blackbody":
			self.write(f, "float sRGB[3];\n")
			self.write(f, "spectral::mdl_blackbody(sRGB, cast<IValue_float>(arguments[0])->get_value());\n")
			self.write(f, "IValue_float const *r = value_factory->create_float(sRGB[0]);\n")
			self.write(f, "IValue_float const *g = value_factory->create_float(sRGB[1]);\n")
			self.write(f, "IValue_float const *b = value_factory->create_float(sRGB[2]);\n")
			self.write(f, "return value_factory->create_rgb_color(r, g, b);\n")
			return

		elif mode == "math::emission_color_spectrum":
			# FIXME: so far black
			self.write(f, "IValue_float const *zero = value_factory->create_float(0.0f);\n")
			self.write(f, "return value_factory->create_rgb_color(zero, zero, zero);\n")
			return

		elif mode == "math::emission_color_color":
			# FIXME: so far no-op
			self.write(f, "return arguments[0];\n")
			return

		elif mode == "math::DX|DY":
			# always zero IF called on a constant
			self.write(f, "IType const *arg_tp = arguments[0]->get_type()->skip_type_alias();\n")
			self.write(f, "return value_factory->create_zero(arg_tp);\n")
			return

		elif mode == "math::cross":
			vt = self.get_vector_type_and_size(params[0])
			if vt[0] == "float":
				self.write(f, "return do_cross<float>(value_factory, arguments);\n")
			else:
				self.write(f, "return do_cross<double>(value_factory, arguments);\n")
			return

		elif mode == "math::sincos":
			if len(params) == 1:
				arg_tp = params[0]
				if self.is_float_type(arg_tp):
					# support sincos for float types
					kind = self.m_inv_types[arg_tp]
					self.write(f, "IValue_%s const *a = cast<IValue_%s>(arguments[0]);\n" % (kind, kind))
					self.write(f, "%s t_s, t_c;\n" % kind)
					self.write(f, "sincos(a->get_value(), t_s, t_c);\n")
					self.write(f, "IValue const *res[2] = {\n")
					self.indent += 1
					self.write(f, "value_factory->create_%s(t_s),\n" % kind)
					self.write(f, "value_factory->create_%s(t_c)};\n" % kind)
					self.indent -= 1
					self.write(f, "IType_factory *type_factory = value_factory->get_type_factory();\n")
					self.write(f, "IType const *a_type = type_factory->create_array(a->get_type(), 2);\n")
					self.write(f, "return value_factory->create_array(as<IType_array>(a_type), res, 2);\n")
					return
				vt = self.get_vector_type_and_size(arg_tp)
				if vt and (vt[0] == "float" or vt[0] == "double"):
					# support sincos for float vector types
					kind = vt[0]
					self.write(f, "IValue const *r_s[%d];\n" % vt[1])
					self.write(f, "IValue const *r_c[%d];\n" % vt[1])
					self.write(f, "IValue_vector const *arg = cast<IValue_vector>(arguments[0]);\n")
					self.write(f, "for (int j = 0; j < %d; ++j) {\n" % vt[1])
					self.indent += 1
					self.write(f, "IValue_%s const *a = cast<IValue_%s>(arg->get_value(j));\n" % (kind, kind))
					self.write(f, "%s t_s, t_c;\n" % kind)
					self.write(f, "sincos(a->get_value(), t_s, t_c);\n")
					self.write(f, "r_s[j] = value_factory->create_%s(t_s);\n" % kind)
					self.write(f, "r_c[j] = value_factory->create_%s(t_c);\n" % kind)
					self.indent -= 1
					self.write(f, "}\n")
					self.write(f, "IType_vector const *v_type = arg->get_type();\n")
					self.write(f, "IValue const *res[2] = {\n")
					self.indent += 1
					self.write(f, "value_factory->create_vector(v_type, r_s, %d),\n" % vt[1])
					self.write(f, "value_factory->create_vector(v_type, r_c, %d)};\n" % vt[1])
					self.indent -= 1
					self.write(f, "IType_factory *type_factory = value_factory->get_type_factory();\n")
					self.write(f, "IType const *a_type = type_factory->create_array(v_type, 2);\n")
					self.write(f, "return value_factory->create_array(as<IType_array>(a_type), res, 2);\n")
					return

		elif mode == "math::modf":
			if len(params) == 1:
				arg_tp = params[0]
				if self.is_float_type(arg_tp):
					# support modf for float types
					kind = self.m_inv_types[arg_tp]
					self.write(f, "IValue_%s const *a = cast<IValue_%s>(arguments[0]);\n" % (kind, kind))
					self.write(f, "%s t_fractional, t_integral;\n" % kind)
					self.write(f, "t_fractional = modf(a->get_value(), t_integral);\n")
					self.write(f, "IValue const *res[2] = {\n")
					self.indent += 1
					self.write(f, "value_factory->create_%s(t_integral),\n" % kind)
					self.write(f, "value_factory->create_%s(t_fractional)};\n" % kind)
					self.indent -= 1
					self.write(f, "IType_factory *type_factory = value_factory->get_type_factory();\n")
					self.write(f, "IType const *a_type = type_factory->create_array(a->get_type(), 2);\n")
					self.write(f, "return value_factory->create_array(as<IType_array>(a_type), res, 2);\n")
					return
				vt = self.get_vector_type_and_size(arg_tp)
				if vt and (vt[0] == "float" or vt[0] == "double"):
					# support modf for float vector types
					kind = vt[0]
					self.write(f, "IValue const *r_fractional[%d];\n" % vt[1])
					self.write(f, "IValue const *r_integral[%d];\n" % vt[1])
					self.write(f, "IValue_vector const *arg = cast<IValue_vector>(arguments[0]);\n")
					self.write(f, "for (int j = 0; j < %d; ++j) {\n" % vt[1])
					self.indent += 1
					self.write(f, "IValue_%s const *a = cast<IValue_%s>(arg->get_value(j));\n" % (kind, kind))
					self.write(f, "%s t_fractional, t_integral;\n" % kind)
					self.write(f, "t_fractional = modf(a->get_value(), t_integral);\n")
					self.write(f, "r_fractional[j] = value_factory->create_%s(t_fractional);\n" % kind)
					self.write(f, "r_integral[j] = value_factory->create_%s(t_integral);\n" % kind)
					self.indent -= 1
					self.write(f, "}\n")
					self.write(f, "IType_vector const *v_type = arg->get_type();\n")
					self.write(f, "IValue const *res[2] = {\n")
					self.indent += 1
					self.write(f, "value_factory->create_vector(v_type, r_integral, %d),\n" % vt[1])
					self.write(f, "value_factory->create_vector(v_type, r_fractional, %d)};\n" % vt[1])
					self.indent -= 1
					self.write(f, "IType_factory *type_factory = value_factory->get_type_factory();\n")
					self.write(f, "IType const *a_type = type_factory->create_array(v_type, 2);\n")
					self.write(f, "return value_factory->create_array(as<IType_array>(a_type), res, 2);\n")
					return

		elif mode == "math::any|all":
			vt = self.get_vector_type_and_size(params[0])
			need_or = intrinsic == "any"
			if need_or:
				self.write(f, "bool res = false;\n")
			else:
				self.write(f, "bool res = true;\n")

			self.write(f, "for (int j = 0; j < %d; ++j) {\n" % vt[1])
			self.indent += 1
			self.write(f, "IValue const *tmp;\n")
			idx = 0
			for param in params:
				kind = self.m_inv_types[param]
				self.write(f, "tmp = cast<IValue_vector>(arguments[%d])->get_value(j);\n" % idx)
				self.write(f, "%s const %s = cast<IValue_%s>(tmp)->get_value();\n" % (
					vt[0], chr(ord('a') + idx), vt[0]))
				idx += 1

			call = "res = res"
			idx = 0
			cases = { "bool" : "false", "float" : "1.0f", "double" : "1.0" }
			zero = cases.get(vt[0], "0")
			if need_or:
				comma = " | "
			else:
				comma = " & "
			for param in params:
				call += comma + "("
				call += chr(ord('a') + idx)
				call += " != %s)" % zero
				idx += 1
			call += ";\n"
			self.write(f, call);

			self.indent -= 1
			self.write(f, "}\n")
			self.write(f, "return value_factory->create_bool(res);\n")
			return

		elif mode == "math::average":
			if params[0] == "CC":
				size = 3
				type = "float"
				vec  = "rgb_color"
			else:
				vt = self.get_vector_type_and_size(params[0])
				if not vt:
					self.write(f, "return arguments[0];\n");
					return
				size = vt[1]
				type = vt[0]
				vec  = "vector"

			self.write(f, "IValue_%s const *arg = cast<IValue_%s>(arguments[0]);\n" % (vec, vec))
			self.write(f, "IValue const *sum = arg->get_value(0);\n")
			self.write(f, "for (int j = 1; j < %d; ++j) {\n" % size)
			self.indent += 1
			self.write(f, "sum = sum->add(value_factory, arg->get_value(j));\n")
			self.indent -= 1
			self.write(f, "}\n")
			self.write(f, "IValue const *c = value_factory->create_%s(%s(%d));\n" % (type, type, size))

			self.write(f, "return sum->divide(value_factory, c);\n")
			return

		elif mode == "math::isnan|isfinite":
			if len(params) == 1:
				vt = self.get_vector_type_and_size(params[0])
				if vt:
					self.write(f, "IValue const *res[%d];\n" % (vt[1]))

					self.write(f, "IValue_vector const *v = cast<IValue_vector>(arguments[0]);\n")
					self.write(f, "for (int j = 0; j < %d; ++j) {\n" % vt[1])
					self.indent += 1
					self.write(f, "IValue_%s const *a = cast<IValue_%s>(v->get_value(j));\n" % (vt[0], vt[0]))

					self.write(f, "res[j] = value_factory->create_bool(" + intrinsic + "(a->get_value()));\n");

					self.indent -= 1
					self.write(f, "}\n")
					self.write(f, "IType_factory *type_factory = value_factory->get_type_factory();\n")
					self.write(f, "IType_bool const *b_type = type_factory->create_bool();\n")
					self.write(f, "IType_vector const *v_type = type_factory->create_vector(b_type, %d);\n" % vt[1])
					self.write(f, "return value_factory->create_vector(v_type, res, %d);\n" % vt[1])
				else:
					kind = self.m_inv_types(params[0])
					self.write(f, "%s a = cast<IValue_%s>(arguments[0])->get_value());\n" % (kind))
					self.write(f, "return value_factory->create_bool(" + intrinsic + "(a)));\n")
				return

		elif mode == "math::luminance":
			if len(params) == 1:
				vt = self.get_vector_type_and_size(params[0])
				if params[0] == "F3":
					self.write(f, "return do_luminance_sRGB(value_factory, arguments);\n")
				elif params[0] == "CC":
					self.write(f, "return do_luminance_color(value_factory, arguments);\n")
				return

		elif mode == "math::max_value|min_value":
			if len(params) == 1:
				if params[0] == "CC":
					# color argument currently unsupported
					self.write(f, "return do_%s_rgb_color(value_factory, arguments);\n" % intrinsic)
				else:
					vt = self.get_vector_type_and_size(params[0])
					if vt:
						if vt[0] == "float":
							self.write(f, "return do_%s<float>(value_factory, arguments);\n" % intrinsic)
						elif vt[0] == "double":
							self.write(f, "return do_%s<double>(value_factory, arguments);\n" % intrinsic)
				return

		elif mode == "math::max_value_wavelength|min_value_wavelength":
			# FIXME: so far black
			self.write(f, "return value_factory->create_float(0.0f);\n")
			return

		elif mode == "math::distance|dot":
			if len(params) == 2 and params[0] == params[1]:
					vt = self.get_vector_type_and_size(params[0])
					if vt:
						if vt[0] == "float":
							self.write(f, "return do_%s<float>(value_factory, arguments);\n" % intrinsic)
						elif vt[0] == "double":
							self.write(f, "return do_%s<double>(value_factory, arguments);\n" % intrinsic)

		elif mode == "math::length|normalize":
			if len(params) == 1:
				vt = self.get_vector_type_and_size(params[0])
				if vt and params[0] != "CC":
					if vt[0] == "float":
						self.write(f, "return do_%s<float>(value_factory, arguments);\n" % intrinsic)
					elif vt[0] == "double":
						self.write(f, "return do_%s<double>(value_factory, arguments);\n" % intrinsic)
				return

		elif mode == "math::transpose":
			if len(params) == 1:
				mk = self.get_matrix_type_kind(params[0])
				if mk:
					if mk == "IType::TK_FLOAT":
						self.write(f, "return do_%s<float>(value_factory, arguments);\n" % intrinsic)
					elif mk == "IType::TK_DOUBLE":
						self.write(f, "return do_%s<double>(value_factory, arguments);\n" % intrinsic)
				return

		elif mode == "math::component_wise":
			vt = self.get_vector_type_and_size(ret_type)
			# vector/color all same base arguments

			if ret_type == "CC":
				self.write(f, "IValue_float const *res[3];\n")
			else:
				if self.get_vector_type_and_size(params[0]):
					self.write(f, "IType_vector const *v_type = cast<IValue_vector>(arguments[0])->get_type();\n")
				else:
					self.write(f, "IType_vector const *v_type = cast<IValue_vector>(arguments[1])->get_type();\n")
				self.write(f, "IValue const *res[%d];\n" % (vt[1]))

			idx = 0
			for param in params:
				if self.is_atomic_type(param):
					self.write(f, "IValue_%s const *%s = cast<IValue_%s>(arguments[%d]);\n" % 
						(vt[0], chr(ord('a') + idx), vt[0], idx))
				elif param == "CC":
					self.write(f, "IValue_rgb_color const *v_%s = cast<IValue_rgb_color>(arguments[%d]);\n" % 
						(chr(ord('a') + idx), idx))
				else:
					self.write(f, "IValue_vector const *v_%s = cast<IValue_vector>(arguments[%d]);\n" % 
						(chr(ord('a') + idx), idx))
				idx += 1

			self.write(f, "for (int j = 0; j < %d; ++j) {\n" % vt[1])
			self.indent += 1
			idx = 0
			for param in params:
				if not self.is_atomic_type(param):
					self.write(f, "IValue_%s const *%s = cast<IValue_%s>(v_%s->get_value(j));\n" % 
								(vt[0], chr(ord('a') + idx), vt[0], chr(ord('a') + idx)))
				idx += 1

			call = ("res[j] = value_factory->create_%s(" + intrinsic + "(") % vt[0]
			idx = 0
			comma = ""
			for param in params:
				call += comma
				comma = ", "
				call += chr(ord('a') + idx)
				call += "->get_value()"
				idx += 1
			call += "));\n"
			self.write(f, call);

			self.indent -= 1
			self.write(f, "}\n")
			if ret_type == "CC":
				self.write(f, "return value_factory->create_rgb_color(res[0], res[1], res[2]);\n")
			else:
				self.write(f, "return value_factory->create_vector(v_type, res, %d);\n" % vt[1])
			return

		elif mode == None:
			error("Mode not set for intrinsic: %s %s" % (intrinsic, signature))

		else:
			error("Unsupported mode for intrinsic: %s %s %s" % (intrinsic, signature, mode))

		self.write(f, "//Unsupported\n")

	def handle_signatures(self, f, intrinsic, signatures):
		"""Create code all sigtatures of one intrinsic."""
		if len(signatures) == 1:
			# no overloads
			params = signatures[0].split('_')[1:]
			
			if self.strict:
				self.gen_condition(f, params, False)
				self.indent += 1
			else:
				self.gen_condition(f, params, True)
				
			self.create_evaluation(f, intrinsic, signatures[0])
			
			if self.strict:
				self.indent -= 1
				self.write(f, "}\n")
		else:
			# have overloads
			signatures.sort()
			pre_if = ""
			for sig in signatures:
				params = sig.split('_')[1:]
				self.gen_condition(f, params, False, pre_if)
				pre_if = "} else "
				self.indent += 1
				self.create_evaluation(f, intrinsic, sig)
				self.indent -= 1
			self.write(f, "}\n")

	def handle_intrinsic(self, f, intrinsic):
		"""Create code for one intrinsic."""
		sigs = self.m_intrinsics[intrinsic]
		
		# order all signatures by ascending lenght
		l = {}
		for sig in sigs:
			sig_token = sig.split('_')
			n_params = len(sig_token) - 1
			
			l.setdefault(n_params, []).append(sig)

		k = list(l.keys())
		if len(k) == 1:
			# typical case: all signatures have the same length
			n_param = k[0]
			if self.strict:
				self.write(f, "if (n_arguments == %d) {\n" % n_param)
				self.indent += 1
			else:
				# create just an assertion
				self.write(f, "MDL_ASSERT(n_arguments == %d);\n" % n_param)

			for n_param in k:
				self.handle_signatures(f, intrinsic, l[n_param])

			if self.strict:
				self.indent -= 1
				self.write(f, "}\n")
		else:
			# overloads with different signature length
			self.write(f, "switch (n_arguments) {\n")
			n_params = k
			n_params.sort()
			for n_param in n_params:
				self.write(f, "case %d:\n" % n_param)
				
				self.indent += 1
				self.write(f, "{\n")
				self.indent += 1
				self.handle_signatures(f, intrinsic, l[n_param])
				self.indent -= 1
				self.write(f, "}\n")
				
				self.write(f, "break;\n")
				self.indent -= 1
				
			self.write(f, "}\n")

	def create_type_sig_tuple(self, params):
		"""Create a type signature tuple (a, b) for a signature a_b."""
		res = []
		comma = ""
		for param in params:
			res.append(self.m_inv_types[param])
		return "(" + ", ".join(res) + ")"

	def gen_type_check(self, f, idx, type_code):
		"""Create a check for the idx parameter to be of given type."""
		atomic_chk = self.get_atomic_value_kind(type_code)
		
		if atomic_chk:
			self.write(f, "if (arguments[%s]->get_kind() != %s)\n" % (idx, atomic_chk))
			self.indent += 1
			self.write(f, "return false;\n")
			self.indent -= 1
		else:
			vector_chk = self.get_vector_type_kind(type_code)
			matrix_chk = self.get_matrix_type_kind(type_code)
			if vector_chk:
				self.write(f, "if (IValue_vector const *v = as<IValue_vector>(arguments[%s])) {\n" % (idx))
				self.indent += 1

				self.write(f, "if (v->get_component_count() != %s)\n" % type_code[-1])
				self.indent += 1
				self.write(f, "return false;\n")
				self.indent -= 1
				
				self.write(f, "IType_vector const *v_type = v->get_type();\n")
				self.write(f, "IType_atomic const *e_type = v_type->get_element_type();\n")
				self.write(f, "if (e_type->get_kind() != %s)\n" % vector_chk)
				self.indent += 1
				self.write(f, "return false;\n")
				self.indent -= 1
				
				self.indent -= 1

				self.write(f, "} else {\n")

				self.indent += 1
				self.write(f, "return false;\n")
				self.indent -= 1

				self.write(f, "}\n")
			elif matrix_chk:
				self.write(f, "if (IValue_matrix const *v = as<IValue_matrix>(arguments[%s])) {\n" % (idx))
				self.indent += 1

				self.write(f, "if (v->get_component_count() != %s)\n" % type_code[-1])
				self.indent += 1
				self.write(f, "return false;\n")
				self.indent -= 1
				
				self.write(f, "IType_matrix const *m_type = v->get_type();\n")
				self.write(f, "IType_vector const *v_type = m_type->get_element_type();\n")
				self.write(f, "if (v_type->get_size() != %s)\n" % type_code[-2])
				self.indent += 1
				self.write(f, "return false;\n")
				self.indent -= 1
				
				self.write(f, "IType_atomic const *e_type = v_type->get_element_type();\n")
				self.write(f, "if (e_type->get_kind() != %s)\n" % matrix_chk)
				self.indent += 1
				self.write(f, "return false;\n")
				self.indent -= 1
				
				self.indent -= 1

				self.write(f, "} else {\n")

				self.indent += 1
				self.write(f, "return false;\n")
				self.indent -= 1

				self.write(f, "}\n")
			else:
				self.write(f, "// Unsupported\n");
				self.write(f, "return false;\n")
			
	def create_signature_checker(self, f):
		"""Create all signature checker functions."""
		signatures = list(self.m_signatures.keys())
		signatures.sort()
		for sig in signatures:
			params = sig.split('_')
			self.write(f, "/// Check that the given arguments have the signature %s.\n" % self.create_type_sig_tuple(params))
			self.write(f, "///\n")
			self.write(f, "/// \\param arguments  the values, must be of length %d\n" % len(params))
			self.write(f, "static bool check_sig_%s(IValue const * const arguments[])\n" % sig)
			self.write(f, "{\n")
			
			self.indent += 1
			
			all_equal = True
			first_p = params[0]
			for param in params:
				if first_p != param:
					all_equal = False
					break
			
			if all_equal and len(params) > 1:
				self.write(f, "for (size_t i = 0; i < %d; ++i) {\n" % (len(params)))
				self.indent += 1
				self.gen_type_check(f, 'i', first_p)
				self.indent -= 1
				self.write(f, "}\n")
			else:
				for i in range(len(params)):
					self.gen_type_check(f, str(i), params[i])
			
			self.write(f, "return true;\n")
			
			self.indent -= 1
			
			self.write(f, "}\n\n")

	def finalize(self):
		"""Create output."""
		f = open(self.out_name, "w")
		
		self.create_signature_checker(f)

		self.write(f, "/// Evaluates an intrinsic function called on constant arguments.\n")
		self.write(f, "///\n")
		self.write(f, "/// \\param value_factory  The value factory used to create new values\n")
		self.write(f, "/// \\param sema           The semantic of the intrinsic function\n")
		self.write(f, "/// \\param arguments      The values of the function arguments\n")
		self.write(f, "/// \\param n_arguments    The number of arguments\n")
		self.write(f, "///\n")
		self.write(f, "/// \\return The function result or IValue_bad if the function could\n")
		self.write(f, "///         not be evaluated\n")
		self.write(f, "IValue const *evaluate_intrinsic_function(\n")
		self.indent += 1
		self.write(f, "IValue_factory         *value_factory,\n")
		self.write(f, "IDefinition::Semantics sema,\n")
		self.write(f, "IValue const * const   arguments[],\n")
		self.write(f, "size_t                 n_arguments)\n")
		self.indent -= 1
		self.write(f, "{\n")

		self.indent += 1
		
		self.write(f, "switch (sema) {\n")

		keys = list(self.m_intrinsics.keys())
		keys.sort()
		
		for intrinsic in keys:
			mod_name = self.m_intrinsic_mods[intrinsic]
			self.write(f, "case IDefinition::DS_INTRINSIC_%s_%s:\n" % (mod_name.upper(), intrinsic.upper()))
			self.indent += 1
			
			self.handle_intrinsic(f, intrinsic)
			self.write(f, "break;\n");
			
			self.indent -= 1

		self.write(f, "default:\n")
		self.indent += 1
		self.write(f, "break;\n");
		self.indent -= 1

		self.write(f, "}\n")
		self.write(f, "// cannot evaluate\n")
		self.write(f, "return value_factory->create_bad();\n")

		self.indent -= 1

		self.write(f, "}\n")
		f.close()

	def add_support(self, decl):
		"""The given declaration is supported."""
		decl = self.as_intrinsic_function(decl)
		# NYI
		pass
		
	def add_simple_math(self):
		pass
		
def usage(args):
	"""print usage info and exit"""
	print("Usage: %s stdlib_directory outputfile" % args[0])
	return 1

def main(args):
	"""Process one file and generate signatures."""
	if len(args) != 3:
		return usage(args)

	stdlib_dir = args[1]
	out_name   = args[2]
	strict     = True
	
	try:
		parser = SignatureParser(args[0], stdlib_dir, out_name, strict)
		parser.parse("math")
		parser.finalize()
		
	except IOError as e:
		error(str(e))
		return 1
	return 0

if __name__ == "__main__":
	sys.exit(main(sys.argv))

