#
# generate HLSL intrinsics
#
import sys

###############################################################################
# HLSL-specific information.                                                  #
###############################################################################

class db_hlsl_attribute(object):
    "An HLSL attribute declaration"
    def __init__(self, title_name, scope, args, doc):
        self.name = title_name.lower()  # lowercase attribute name
        self.title_name = title_name    # title-case attribute name
        self.scope = scope              # one of l (loop), c (condition), s (switch), f (function)
        self.args = args                # list of arguments
        self.doc = doc                  # documentation

class db_hlsl_intrinsic(object):
    "An HLSL intrinsic declaration"
    def __init__(self, name, idx, opname, params, ns, ns_idx, doc, ro, rn, unsigned_op, overload_idx):
        self.name = name                                # Function name
        self.idx = idx                                  # Unique number within namespace
        self.opname = opname                            # D3D-style name
        self.params = params                            # List of parameters
        self.ns = ns                                    # Function namespace
        self.ns_idx = ns_idx                            # Namespace index
        self.doc = doc                                  # Documentation
        id_prefix = "DS_IOP" if ns == "Intrinsics" else "DS_MOP"
        self.enum_name = "%s_%s" % (id_prefix, name)    # enum name
        self.readonly = ro                              # Only read memory
        self.readnone = rn                              # Not read memory
        self.unsigned_op = unsigned_op                  # Unsigned opcode if exist
        if unsigned_op != "":
            self.unsigned_op = "%s_%s" % (id_prefix, unsigned_op)
        self.overload_param_index = overload_idx        # Parameter determines the overload type, -1 means ret type
        self.key = ("%3d" % ns_idx) + "!" + name + "!" + ("%2d" % len(params)) + "!" + ("%3d" % idx)    # Unique key
        self.vulkanSpecific = ns.startswith("Vk")       # Vulkan specific intrinsic - SPIRV change

class db_hlsl_namespace(object):
    "A grouping of HLSL intrinsics"
    def __init__(self, name):
        self.name = name
        self.intrinsics = []

class db_hlsl_intrisic_param(object):
    "An HLSL parameter declaration for an intrinsic"
    def __init__(self, name, param_qual, template_id, template_list, component_id, component_list, rows, cols, type_name, idx, template_id_idx, component_id_idx):
        self.name = name                             # Parameter name
        self.param_qual = param_qual                 # Parameter qualifier expressions
        self.template_id = template_id               # Template ID (possibly identifier)
        self.template_list = template_list           # Template list (possibly identifier)
        self.component_id = component_id             # Component ID (possibly identifier)
        self.component_list = component_list         # Component list (possibly identifier)
        self.rows = rows                             # Row count for parameter, possibly identifier
        self.cols = cols                             # Row count for parameter, possibly identifier
        self.type_name = type_name                   # Type name
        self.idx = idx                               # Argument index
        self.template_id_idx = template_id_idx       # Template ID numeric value
        self.component_id_idx = component_id_idx     # Component ID numeric value

class db_hlsl(object):
    "A database of HLSL language data"
    def __init__(self, intrinsic_defs):
        self.base_types = {
            "bool": "LICOMPTYPE_BOOL",
            "int": "LICOMPTYPE_INT",
            "int16_t": "LICOMPTYPE_INT16",
            "uint": "LICOMPTYPE_UINT",
            "uint16_t": "LICOMPTYPE_UINT16",
            "u64": "LICOMPTYPE_UINT64",
            "any_int": "LICOMPTYPE_ANY_INT",
            "any_int32": "LICOMPTYPE_ANY_INT32",
            "uint_only": "LICOMPTYPE_UINT_ONLY",
            "float16_t": "LICOMPTYPE_FLOAT16",
            "float": "LICOMPTYPE_FLOAT",
            "fldbl": "LICOMPTYPE_FLOAT_DOUBLE",
            "any_float": "LICOMPTYPE_ANY_FLOAT",
            "float_like": "LICOMPTYPE_FLOAT_LIKE",
            "double": "LICOMPTYPE_DOUBLE",
            "double_only": "LICOMPTYPE_DOUBLE_ONLY",
            "numeric": "LICOMPTYPE_NUMERIC",
            "numeric16_only": "LICOMPTYPE_NUMERIC16_ONLY",
            "numeric32": "LICOMPTYPE_NUMERIC32",
            "numeric32_only": "LICOMPTYPE_NUMERIC32_ONLY",
            "any": "LICOMPTYPE_ANY",
            "sampler1d": "LICOMPTYPE_SAMPLER1D",
            "sampler2d": "LICOMPTYPE_SAMPLER2D",
            "sampler3d": "LICOMPTYPE_SAMPLER3D",
            "sampler_cube": "LICOMPTYPE_SAMPLERCUBE",
            "sampler_cmp": "LICOMPTYPE_SAMPLERCMP",
            "sampler": "LICOMPTYPE_SAMPLER",
            "ray_desc" : "LICOMPTYPE_RAYDESC",
            "acceleration_struct" : "LICOMPTYPE_ACCELERATION_STRUCT",
            "udt" : "LICOMPTYPE_USER_DEFINED_TYPE",
            "void": "LICOMPTYPE_VOID",
            "string": "LICOMPTYPE_STRING",
            "wave": "LICOMPTYPE_WAVE"}
        self.trans_rowcol = {
            "r": "IA_R",
            "c": "IA_C",
            "r2": "IA_R2",
            "c2": "IA_C2"}
        self.param_qual = {
            "in": "AR_QUAL_IN",
            "inout": "AR_QUAL_IN | AR_QUAL_OUT",
            "out": "AR_QUAL_OUT",
            "col_major": "AR_QUAL_COLMAJOR",
            "row_major": "AR_QUAL_ROWMAJOR"}
        self.intrinsics = []
        self.load_intrinsics(intrinsic_defs)
        self.create_namespaces()
        self.populate_attributes()
        self.opcode_namespace = "Def_function"

    def create_namespaces(self):
        last_ns = None
        self.namespaces = {}
        for i in sorted(self.intrinsics, key=lambda x: x.key):
            if last_ns is None or last_ns.name != i.ns:
                last_ns = db_hlsl_namespace(i.ns)
                self.namespaces[i.ns] = last_ns
            last_ns.intrinsics.append(i)

    def load_intrinsics(self, intrinsic_defs):
        import re
        blank_re = re.compile(r"^\s*$")
        comment_re = re.compile(r"^\s*//")
        namespace_beg_re = re.compile(r"^namespace\s+(\w+)\s*{\s*$")
        namespace_end_re = re.compile(r"^}\s*namespace\s*$")
        intrinsic_re = re.compile(r"^\s*([^(]+)\s+\[\[(\S*)\]\]\s+(\w+)\s*\(\s*([^)]*)\s*\)\s*(:\s*\w+\s*)?;$")
        operand_re = re.compile(r"^:\s*(\w+)\s*$")
        bracket_cleanup_re = re.compile(r"<\s*(\S+)\s*,\s*(\S+)\s*>") # change <a,b> to <a@> to help split params and parse
        params_split_re = re.compile(r"\s*,\s*")
        ws_split_re = re.compile(r"\s+")
        typeref_re = re.compile(r"\$type(\d+)$")
        type_matrix_re = re.compile(r"(\S+)<(\S+)@(\S+)>$")
        type_vector_re = re.compile(r"(\S+)<(\S+)>$")
        type_any_re = re.compile(r"(\S+)<>$")
        digits_re = re.compile(r"^\d+$")
        opt_param_match_re = re.compile(r"^\$match<(\S+)@(\S+)>$")
        ns_idx = 0
        num_entries = 0

        def add_flag(val, new_val):
            if val == "" or val == "0":
                return new_val
            return val + " | " + new_val

        def translate_rowcol(val):
            digits_match = digits_re.match(val)
            if digits_match:
                return val
            assert val in self.trans_rowcol, "unknown row/col %s" % val
            return self.trans_rowcol[val]

        def process_arg(desc, idx, done_args, intrinsic_name):
            "Process a single parameter description."
            opt_list = []
            desc = desc.strip()
            if desc == "...":
                param_name = "..."
                type_name = "..."
            else:
                opt_list = ws_split_re.split(desc)
                assert len(opt_list) > 0, "malformed parameter desc %s" % (desc)
                param_name = opt_list.pop()      # last token is name
                type_name = opt_list.pop() # next-to-last is type specifier

            param_qual = "0"
            template_id = str(idx)
            template_list = "LITEMPLATE_ANY"
            component_id = str(idx)
            component_list = "LICOMPTYPE_ANY"
            rows = "1"
            cols = "1"
            if type_name == "$unspec":
                assert idx == 0, "'$unspec' can only be used as the return type"
                # template_id may be -1 in other places other than return type, for example in Stream.Append().
                # $unspec is a shorthand for return types only though.
                template_id = "-1"
                component_id = "0"
                type_name = "void"
            elif type_name == "...":
                assert idx != 0, "'...' can only be used in the parameter list"
                template_id = "-2"
                component_id = "0"
                type_name = "void"
            else:
                typeref_match = typeref_re.match(type_name)
                if typeref_match:
                    template_id = typeref_match.group(1)
                    component_id = template_id
                    assert idx != 1, "Can't use $type on the first argument"
                    assert template_id != "0", "Can't match an input to the return type"
                    done_idx = int(template_id) - 1
                    assert done_idx <= len(args) + 1, "$type must refer to a processed arg"
                    done_arg = done_args[done_idx]
                    type_name = done_arg.type_name
            # Determine matrix/vector/any/scalar type names.
            type_matrix_match = type_matrix_re.match(type_name)
            if type_matrix_match:
                base_type = type_matrix_match.group(1)
                rows = type_matrix_match.group(2)
                cols = type_matrix_match.group(3)
                template_list = "LITEMPLATE_MATRIX"
            else:
                type_vector_match = type_vector_re.match(type_name)
                if type_vector_match:
                    base_type = type_vector_match.group(1)
                    cols = type_vector_match.group(2)
                    template_list = "LITEMPLATE_VECTOR"
                else:
                    type_any_match = type_any_re.match(type_name)
                    if type_any_match:
                        base_type = type_any_match.group(1)
                        rows = "r"
                        cols = "c"
                        template_list = "LITEMPLATE_ANY"
                    else:
                        base_type = type_name
                        if base_type.startswith("sampler") or base_type.startswith("string") or base_type.startswith("wave") or base_type.startswith("acceleration_struct") or base_type.startswith("ray_desc"):
                            template_list = "LITEMPLATE_OBJECT"
                        else:
                            template_list = "LITEMPLATE_SCALAR"
            assert base_type in self.base_types, "Unknown base type '%s' in '%s'" % (base_type, desc)
            component_list = self.base_types[base_type]
            rows = translate_rowcol(rows)
            cols = translate_rowcol(cols)
            for opt in opt_list:
                if opt in self.param_qual:
                    param_qual = add_flag(param_qual, self.param_qual[opt])
                else:
                    opt_param_match_match = opt_param_match_re.match(opt)
                    assert opt_param_match_match, "Unknown parameter qualifier '%s'" % (opt)
                    template_id = opt_param_match_match.group(1)
                    component_id = opt_param_match_match.group(2)
            if component_list == "LICOMPTYPE_VOID":
                if type_name == "void":
                    template_list = "LITEMPLATE_VOID"
                    rows = "0"
                    cols = "0"
                    if template_id == "0":
                        param_qual = "0"
            # Keep these as numeric values.
            template_id_idx = int(template_id)
            component_id_idx = int(component_id)
            # Verify that references don't point to the right (except for the return value).
            assert idx == 0 or template_id_idx <= int(idx), "Argument '%s' has a forward reference" % (param_name)
            assert idx == 0 or component_id_idx <= int(idx), "Argument '%s' has a forward reference" % (param_name)
            if template_id == "-1":
                template_id = "INTRIN_TEMPLATE_FROM_TYPE"
            elif template_id == "-2":
                template_id = "INTRIN_TEMPLATE_VARARGS"
            if component_id == "-1":
                component_id = "INTRIN_COMPTYPE_FROM_TYPE_ELT0"
            return db_hlsl_intrisic_param(param_name, param_qual, template_id, template_list, component_id, component_list, rows, cols, type_name, idx, template_id_idx, component_id_idx)

        def process_attr(attr):
            attrs = attr.split(',')
            readonly = False          # Only read memory
            readnone = False          # Not read memory
            unsigned_op = ""          # Unsigned opcode if exist
            overload_param_index = -1 # Parameter determines the overload type, -1 means ret type.
            for a in attrs:
                if (a == ""):
                    continue
                if (a == "ro"):
                    readonly = True
                    continue
                if (a == "rn"):
                    readnone = True
                    continue
                assign = a.split('=')

                if (len(assign) != 2):
                    assert False, "invalid attr %s" % (a)
                    continue
                d = assign[0]
                v = assign[1]
                if (d == "unsigned_op"):
                    unsigned_op = v
                    continue
                if (d == "overload"):
                    overload_param_index = int(v)
                    continue
                assert False, "invalid attr %s" % (a)

            return readonly, readnone, unsigned_op, overload_param_index

        current_namespace = None
        for line in intrinsic_defs:
            if blank_re.match(line): continue
            if comment_re.match(line): continue
            match_obj = namespace_beg_re.match(line)
            if match_obj:
                assert not current_namespace, "cannot open namespace without closing prior one"
                current_namespace = match_obj.group(1)
                num_entries = 0
                ns_idx += 1
                continue
            if namespace_end_re.match(line):
                assert current_namespace, "cannot close namespace without previously opening it"
                current_namespace = None
                continue
            match_obj = intrinsic_re.match(line)
            if match_obj:
                assert current_namespace, "instruction missing namespace %s" % (line)
                # Get a D3D-style operand name for the instruction.
                # Unused for DXIL.
                opts = match_obj.group(1)
                attr = match_obj.group(2)
                name = match_obj.group(3)
                params = match_obj.group(4)
                op = match_obj.group(5)
                if op:
                    operand_match = operand_re.match(op)
                    if operand_match:
                        op = operand_match.group(1)
                if not op:
                    op = name
                readonly, readnone, unsigned_op, overload_param_index = process_attr(attr)
                # Add an entry for this intrinsic.
                if bracket_cleanup_re.search(opts):
                    opts = bracket_cleanup_re.sub(r"<\1@\2>", opts)
                if bracket_cleanup_re.search(params):
                    params = bracket_cleanup_re.sub(r"<\g<1>@\2>", params)
                ret_desc = "out " + opts + " " + name
                if len(params) > 0:
                    in_args = params_split_re.split(params)
                else:
                    in_args = []
                arg_idx = 1
                args = []
                for in_arg in in_args:
                    args.append(process_arg(in_arg, arg_idx, args, name))
                    arg_idx += 1
                # We have to process the return type description last
                # to match the compiler's handling of it and allow
                # the return type to match an input type.
                # It needs to be the first entry, so prepend it.
                args.insert(0, process_arg(ret_desc, 0, args, name))
                # TODO: verify a single level of indirection
                self.intrinsics.append(db_hlsl_intrinsic(
                    name, num_entries, op, args, current_namespace, ns_idx, "pending doc for " + name,
                    readonly, readnone, unsigned_op, overload_param_index))
                num_entries += 1
                continue
            assert False, "cannot parse line %s" % (line)

    def populate_attributes(self):
        "Populate basic definitions for attributes."
        attributes = []
        def add_attr(title_name, scope, doc):
            attributes.append(db_hlsl_attribute(title_name, scope, [], doc))
        def add_attr_arg(title_name, scope, args, doc):
            attributes.append(db_hlsl_attribute(title_name, scope, args, doc))
        add_attr("Allow_UAV_Condition", "l", "Allows a compute shader loop termination condition to be based off of a UAV read. The loop must not contain synchronization intrinsics")
        add_attr("Branch", "c", "Evaluate only one side of the if statement depending on the given condition")
        add_attr("Call", "s", "The bodies of the individual cases in the switch will be moved into hardware subroutines and the switch will be a series of subroutine calls")
        add_attr("EarlyDepthStencil", "f", "Forces depth-stencil testing before a shader executes")
        add_attr("FastOpt", "l", "Reduces the compile time but produces less aggressive optimizations")
        add_attr("Flatten", "c", "Evaluate both sides of the if statement and choose between the two resulting values")
        add_attr("ForceCase", "s", "Force a switch statement in the hardware")
        add_attr("Loop", "l", "Generate code that uses flow control to execute each iteration of the loop")
        add_attr_arg("ClipPlanes", "f", "Optional list of clip planes", [{"name":"ClipPlane", "type":"int", "count":6}])
        add_attr_arg("Domain", "f", "Defines the patch type used in the HS", [{"name":"DomainType", type:"string"}])
        add_attr_arg("Instance", "f", "Use this attribute to instance a geometry shader", [{"name":"Count", "type":"int"}])
        add_attr_arg("MaxTessFactor", "f", "Indicates the maximum value that the hull shader would return for any tessellation factor.", [{"name":"Count", "type":"int"}])
        add_attr_arg("MaxVertexCount", "f", "maxvertexcount doc", [{"name":"Count", "type":"int"}])
        add_attr_arg("NumThreads", "f", "Defines the number of threads to be executed in a single thread group.", [{"name":"x", "type":"int"},{"name":"z", "type":"int"},{"name":"y", "type":"int"}])
        add_attr_arg("OutputControlPoints", "f", "Defines the number of output control points per thread that will be created in the hull shader", [{"name":"Count", "type":"int"}])
        add_attr_arg("OutputTopology", "f", "Defines the output primitive type for the tessellator", [{"name":"Topology", "type":"string"}])
        add_attr_arg("Partitioning", "f", "Defines the tesselation scheme to be used in the hull shader", [{"name":"Scheme", "type":"scheme"}])
        add_attr_arg("PatchConstantFunc", "f", "Defines the function for computing patch constant data", [{"name":"FunctionName", "type":"string"}])
        add_attr_arg("RootSignature", "f", "RootSignature doc", [{"name":"SignatureName", "type":"string"}])
        add_attr_arg("Unroll", "l", "Unroll the loop until it stops executing or a max count", [{"name":"Count", "type":"int"}])
        self.attributes = attributes

g_db_hlsl = None
g_templ_name = None

def get_db_hlsl():
    global g_db_hlsl
    if g_db_hlsl is None:
        with open(g_templ_name, "rU") as f:
            g_db_hlsl = db_hlsl(f)
    return g_db_hlsl

def get_hlsl_intrinsics():
    db = get_db_hlsl()
    result = ""
    last_ns = ""
    ns_table = ""
    is_vk_table = False  # SPIRV Change
    id_prefix = ""
    arg_idx = 0
    opcode_namespace = db.opcode_namespace
    for i in sorted(db.intrinsics, key=lambda x: x.key):
        if last_ns != i.ns:
            last_ns = i.ns
            id_prefix = "DS_IOP" if last_ns == "Intrinsics" else "DS_MOP"
            if (len(ns_table)):
                result += ns_table + "};\n"
                # SPIRV Change Starts
                if is_vk_table:
                    result += "\n#endif // ENABLE_SPIRV_CODEGEN\n"
                    is_vk_table = False
                # SPIRV Change Ends
            result += "\n//\n// Start of %s\n//\n\n" % (last_ns)
            # This used to be qualified as __declspec(selectany), but that's no longer necessary.
            ns_table = "static HLSL_intrinsic const g_%s[] =\n{\n" % (last_ns)
            # SPIRV Change Starts
            if (i.vulkanSpecific):
                is_vk_table = True
                result += "#ifdef ENABLE_SPIRV_CODEGEN\n\n"
            # SPIRV Change Ends
            arg_idx = 0
        ma = "MA_WRITE"
        if i.readonly:
            ma = "MA_READ_ONLY"
        elif i.readnone:
            ma = "MA_READ_NONE"

        ns_table += "    { %s::%s_%s, %s, %d, %d, g_%s_Args%s },\n" % (opcode_namespace, id_prefix, i.name, ma, i.overload_param_index,len(i.params), last_ns, arg_idx)
        result += "static HLSL_intrinsic_argument const g_%s_Args%s[] =\n{\n" % (last_ns, arg_idx)
        for p in i.params:
            result += "    {\"%s\", %s, %s, %s, %s, %s, %s, %s},\n" % (
                p.name, p.param_qual, p.template_id, p.template_list,
                p.component_id, p.component_list, p.rows, p.cols)
        result += "};\n\n"
        arg_idx += 1
    result += ns_table + "};\n"
    result += "\n#endif // ENABLE_SPIRV_CODEGEN\n" if is_vk_table else ""  # SPIRV Change
    return result

# SPIRV Change Starts
def wrap_with_ifdef_if_vulkan_specific(intrinsic, text):
    if intrinsic.vulkanSpecific:
        return "#ifdef ENABLE_SPIRV_CODEGEN\n" + text + "#endif // ENABLE_SPIRV_CODEGEN\n"
    return text
# SPIRV Change Ends

def enum_hlsl_intrinsics():
    db = get_db_hlsl()
    result = "  DS_HLSL_INTRINSIC_FIRST,\n"
    first = " = DS_HLSL_INTRINSIC_FIRST"
    enumed = []
    last = None
    for i in sorted(db.intrinsics, key=lambda x: x.key):
        if (i.enum_name not in enumed):
            enumerant = "  %s%s,\n" % (i.enum_name, first)
            first =""
            result += wrap_with_ifdef_if_vulkan_specific(i, enumerant)  # SPIRV Change
            enumed.append(i.enum_name)
            last = i.enum_name
    # unsigned
    result += "  // unsigned\n"

    for i in sorted(db.intrinsics, key=lambda x: x.key):
        if (i.unsigned_op != ""):
          if (i.unsigned_op not in enumed):
            result += "  %s%s,\n" % (i.unsigned_op, first)
            first = ""
            enumed.append(i.unsigned_op)
            last = i.unsigned_op

    result += "  DS_HLSL_INTRINSIC_LAST = %s,\n" % last
    return result

def error(msg):
    """Write a message to stderr"""
    sys.stderr.write("gen_intrinsic_eval: Error: " + msg + "\n")

def usage(args):
    """print usage info and exit"""
    print("Usage: %s outputfile specification" % args[0])
    return 1

def main(args):
    global g_templ_name

    """Process one file and generate signatures."""
    g_templ_name = args[1]
    out_enum     = args[2]
    out_defs     = args[3]

    try:
        if out_enum:
            f = open(out_enum, "wt")
        else:
            f = sys.stderr
        f.write("// Generated by gen_hlsl_intrinsics.py.\n")
        f.write(enum_hlsl_intrinsics())
        f.close()

        if out_defs:
            f = open(out_defs, "wt")
        else:
            f = sys.stderr
        f.write("// Generated by gen_hlsl_intrinsics.py.\n")
        f.write(get_hlsl_intrinsics())
        f.close()
    except IOError as e:
        error(str(e))
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
