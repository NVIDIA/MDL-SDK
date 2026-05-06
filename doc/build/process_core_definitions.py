#!/usr/bin/python

# pylint: disable-msg=C0114,C0116

import collections
import errno
import os
import re
import sys

def extract_using(source):
    using_pat = re.compile(r'export\s+using\s+::(\w+)\s+import\s+(\w+)', re.S)
    result = {}
    for module, element in using_pat.findall(source):
        result[element] = f"::{module}::{element}"
    return result

def expand_using(defs, src):
    result = src
    for abbrev, fullname in defs.items():
        pattern = re.compile(rf"\b{re.escape(abbrev)}\b")
        result = pattern.sub(fullname, result)
    return result

def extract_annotations(annotations):
    annotation_pat = re.compile(r'anno::(\w+)\("(.*?)"\)')
    result = collections.OrderedDict()
    for name, value in annotation_pat.findall(annotations):
        if name != "author":
            result[name] = value
    return result

def extract_parameters(parameters):
    result = collections.OrderedDict()
    parameter_pat \
        = re.compile(r'(uniform)?\s+(\w+)\s+(\w+)\s*(?:=\s*(.*?))?\s+\[\[\s*(.*?)\s*\]\]', re.S)
    for uni, ptype, varname, default, anno in parameter_pat.findall(parameters):
        result[varname] = [uni == "uniform", ptype, default, extract_annotations(anno)]
    return result

def extract_signatures(src):
    materials = collections.OrderedDict()
    functions = collections.OrderedDict()
    signature_pat \
        = re.compile(r'^export\s+(\w+)\s+(\w+)\s*\((.*?\]\])\s*\)\s*\[\[(.*?)\]\]', re.S|re.M)
    using = extract_using(src)
    for return_type, name, sig, anno in signature_pat.findall(src):
        sig = expand_using(using, sig)
        if return_type == "material":
            materials[name] = [extract_annotations(anno), extract_parameters(sig)]
        else:
            functions[name] = [return_type, extract_annotations(anno), extract_parameters(sig)]
    return materials, functions

def extract_fields(fields):
    result = collections.OrderedDict()
    field_pat = re.compile(r'(\w+)\s+(?:=\s*(.*?))?\s*\[\[\s*(.*?)\s*\]\]', re.S)
    for name, index, anno in field_pat.findall(fields):
        result[name] = [index, extract_annotations(anno)]
    return result

def extract_enums(src):
    result = collections.OrderedDict()
    enum_pat = re.compile(r'export\s+enum\s+(\w+)\s+\[\[\s+(.*?)\s+\]\]\s+\{\s+(.*?)\s+\};', re.S)
    for name, anno, fields in enum_pat.findall(src):
        result[name] = [extract_annotations(anno), extract_fields(fields)]
    return result

def dictappend(dictionary, name, value):
    if value:
        if dictionary.get(name):
            dictionary[name] += value
        else:
            dictionary[name] = value

def extract(mdl_input, remove_deprecated=True):

    if not os.path.exists(mdl_input):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), mdl_input)

    with open(mdl_input, encoding="utf-8") as fp:
        src = fp.read()

    # remove annotation_exports
    src = re.compile('^export +annotation +.*?; *$', re.M).sub('', src)

    # convert quotes
    src = re.compile(r'\\"(\w)').sub(r'`\1', src)
    src = re.compile(r'(\w)\\"').sub(r'\1`', src)

    group_pat = re.compile(r'^//group\s+(.*?)$(.*?)\s+//endgroup', re.S|re.M)
    groups = group_pat.findall(src)

    materials = collections.OrderedDict()
    functions = collections.OrderedDict()
    enums = collections.OrderedDict()

    for name, text in groups:

        m, f = extract_signatures(text)
        e = extract_enums(text)

        if remove_deprecated and "Deprecated" in name:
            continue

        dictappend(materials, name, m)
        dictappend(functions, name, f)
        dictappend(enums, name, e)

    return materials, functions, enums

def description(anno):
    return anno.get("description", "")

def display_name(anno, name):
    result = anno.get("display_name")
    if result is None:
        result = re.sub("_", " ", name).capitalize()
    return result

def description_table(parameters):
    result  = "| Display name | Description |\n"
    result += "| ------------ | ----------- |\n"
    for name, value in parameters.items():
        anno = value[-1]
        result += f"| {display_name(anno, name)} | {description(anno)} |\n"
    result += "\n"
    return result

def floatify(s):
    result = s
    pat = re.compile(r'(\d\.)(?!\d)', re.M)
    m = pat.search(result)
    if m:
        result = pat.sub(f"{m.group(1)}0", result)
        result = re.sub(",", ", ", result)
    pat = re.compile(r'(?<!\d)(\.\d)', re.M)
    m = pat.search(" " + result)
    if m:
        result = pat.sub(f"0{m.group(1)}", result)
    return result

def generate_thumbnail(anno, caption):
    if "thumbnail" not in anno:
        return ""
    filename = anno["thumbnail"]
    filename = re.compile(r'^.*/').sub("images/", filename)
    result = f'![{caption}]({filename} "{caption}")\n\n'
    if not os.path.exists(filename):
        print(f'Warning: thumbail "{filename}" is missing')
    return result

def generate_parameters(parameters, separate_table=True):
    result = ""
    if separate_table:
        result += description_table(parameters)
    desc_title = "" if separate_table else "| Description "
    desc_ruler = "" if separate_table else "| ----------- "
    result += f"| Display name | Type | Parameter | Default {desc_title}|\n"
    result += f"| ------------ | ---- | --------- | ------- {desc_ruler}|\n"
    for name, value in parameters.items():
        uni, ptype, default, anno = value
        display = display_name(anno, name)
        uni_str = "uniform " if uni else ""
        default = re.sub("diffuse_color: ", "", default)
        default = floatify(default)
        default = f"`{default}`" if default else "(none)"
        desc = "" if separate_table else f"| {description(anno)}"
        result += f"| {display} | `{uni_str}{ptype}` | `{name}` | {default} {desc}\n"
    result += "\n"
    return result

def generate_fields(enum_fields):
    result  = "| Field | Index | Description |\n"
    result += "| ----- | ----- | ----------- |\n"
    i = 0
    for name, value in enum_fields.items():
        index, anno = value
        index_value = i if index == "" else index
        result += f"| `{name}` | `{index_value}` | {description(anno)} |\n"
        i += 1
    result += "\n"
    return result

def generate(materials, functions, enums, md_header, md_output):

    with open(md_header, encoding="utf-8") as fp:
        result = fp.read()
    result += "\n"

    result += "# Materials and building blocks\n\n"
    for group_name, group_value in materials.items():
        single_group = group_name == ""
        if not single_group:
            result += f"## {group_name}\n\n"
        for name, value in group_value.items():
            anno, params = value
            result += f"##{'#' if single_group else ''} {display_name(anno, name)}\n\n"
            result += f"MDL identifier: `core_definitions::{name}`\n\n"
            result += generate_thumbnail(anno, name)
            result += f"{description(anno)}\n\n"
            result += generate_parameters(params)
            result += "\n"

    result += "# Texturing functions\n\n"
    for group_name, group_value in functions.items():
        # Only one group so far, so no separate section title
        for name, value in group_value.items():
            _, anno, params = value
            result += f"## {display_name(anno, name)}\n\n"
            result += f"MDL identifier: `core_definitions::{name}`\n\n"
            result += generate_thumbnail(anno, name)
            result += f"{description(anno)}\n\n"
            result += generate_parameters(params)
            result += "\n"

    result += "# Enumerations\n\n"
    for group_name, group_value in enums.items():
        # Only one group so far, so no separate section title
        for name, value in group_value.items():
            anno, fields = value
            result += f"## {display_name(anno, name)}\n\n"
            result += f"MDL identifier: `core_definitions::{name}`\n\n"
            result += generate_thumbnail(anno, name)
            result += f"{description(anno)}\n\n"
            result += generate_fields(fields)
            result += "\n"

    with open(md_output, "w", encoding="utf-8") as fp:
        fp.write(result)

def main():
    if len(sys.argv) != 1+3:
        print(f"Usage: {sys.argv[0]} <mdl_input> <md_intro> <md_output>")
        raise SystemExit(1)

    mdl_input = sys.argv[1]
    md_header = sys.argv[2]
    md_output = sys.argv[3]

    materials, functions, enums = extract(mdl_input)
    assert len(materials) >= 3

    generate(materials, functions, enums, md_header, md_output)

if __name__ == "__main__":
    main()
