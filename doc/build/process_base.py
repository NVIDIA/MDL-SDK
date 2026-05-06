#!/usr/bin/python

# pylint: disable-msg=C0114,C0116

import functools
import re
import sys

STR_FUNCTION_ANNOTATION = r'\[\[\s*(.*?)\s*\]\]'

RE_CONSTANT = re.compile(rf'''
export
\s+
const
\s+
(?P<type>\w+)
\s+
(?P<name>\w+)
\s*
\(
(?P<value>.*?)
\s*
\)
{STR_FUNCTION_ANNOTATION}
;
''', re.S|re.M|re.VERBOSE)

RE_FUNCTION_DEFINITION = re.compile(rf'''
export
\s+
(uniform\s+)?
(?P<type>\w+)
\s+
(?P<name>[\w_]+)
\s*
\(
\s*
(?P<params>.*?) # parameters
\s*
^\)  # beginning of line?
\s*
{STR_FUNCTION_ANNOTATION}
(\s+uniform\s+)?
\s*
^\{{
.*?
^\}}
''', re.S|re.M|re.VERBOSE)

def field_container_pat(name):
    return re.compile(rf'''
export
\s+
{name}
\s+
(?P<name>[\w_]+)
\s*
{STR_FUNCTION_ANNOTATION}
\s*
\{{
\s*
(?P<fields>[^}}]+)
\s*
\}}\s*
;
''', re.S|re.M|re.VERBOSE)


RE_ENUM = field_container_pat('enum')
RE_STRUCT = field_container_pat('struct')

RE_PARAM = re.compile(r'''
(?P<qual>uniform(?:\s))?
\s*
(?P<type>[\w_<>:[\]]+)
\s+
(?P<name>[\w+]+)
\s*
(?P<tail>.*)
''', re.S|re.M|re.VERBOSE)

# --- Input processing ---

def symbols_to_ignore(src):
    symbols = re.compile(r'//@ignore\s+(.*)\s*').findall(src)
    if not symbols:
        return []
    return functools.reduce(lambda x,y: x+y, [e.split() for e in symbols])

def uncomment(src):
    result = src
    result = re.compile(r'/\*.+?\*/', re.S|re.M).sub('', result)
    result = "\n".join(s.split("//")[0] for s in result.split("\n"))
    return result

def untab(src):
    return re.sub(r'\t', ' '*4, src)

def reformat_floats(src):
    result = re.compile(r'([0-9])\.([^0-9])').sub(r'\1.0\2', src)
    result = re.compile(r'(?P<prefix>[^0-9])\.([0-9])', re.S|re.M).sub(r'\g<prefix>0.\2', result)
    result = re.compile(r'^\.([0-9])', re.S|re.M).sub(r'0.\1', result)
    # But:
    result = re.sub(r'channel (\d).0', r'channel `\1`.', result)
    return result

def reformat_arglist(src):
    return re.compile(r'\( ([0-9])').sub(r'(\1', src)

def modify_module_source(src):
    intro_pat = re.compile(r'/\*@(.*?)@\*/', re.S|re.M)
    result = intro_pat.sub('', src)
    result = re.compile(r'\\"', re.S|re.M).sub('"', result)
    result = uncomment(result)
    result = untab(result)
    result = reformat_floats(result)
    result = reformat_arglist(result)
    return result

# --- Table of contents ---

def list_remove(values, to_remove):
    for rem in to_remove:
        if rem in values:
            values.remove(rem)
    return values

def constant_names(src):
    return [e[1] for e in RE_CONSTANT.findall(src)]

def enum_names(src):
    return [e[0] for e in RE_ENUM.findall(src)]

def struct_names(src):
    return [e[0] for e in RE_STRUCT.findall(src)]

def function_names(src):
    return [e[2] for e in RE_FUNCTION_DEFINITION.findall(src)]

def get_link(text, target):
    return f"[`{text}`](#{target.lower()})"

LINKED_ITEMS = []

def table_of_contents(src, ignore):
    data_constants = list_remove(constant_names(src), ignore)
    data_enums = list_remove(enum_names(src), ignore)
    data_structs = list_remove(struct_names(src), ignore)
    data_functions = list_remove(function_names(src), ignore)

    result = "# Language elements\n\n"

    sections = zip(
        ['Constants', 'Enums', 'Structs', 'Functions'],
        [data_constants, data_enums, data_structs, data_functions])

    for title, data in sections:
        if data:
            result += f"## {title}\n\n"
            for item in data:
                result += f"{get_link(item,item)}\n\n"

    global LINKED_ITEMS
    LINKED_ITEMS = data_constants + data_enums + data_structs + data_functions

    result += "\n"
    return result

# --- Utilities ---

def set_format_for_description(s):
    result = s
    # Add spaces around * to avoid being interpreted as markup for italics.
    result = re.sub(r'([^ ])\*', r'\1 *', result)
    result = re.sub(r'\*([^ ])', r'* \1', result)
    # Add italic markup.
    result = re.sub(r'([uv])\s+direction', r'*\1* direction', result)
    result = re.sub(r'\suvw', ' `uvw`', result)
    result = re.sub(r'"u" direction', '*u* direction', result)
    result = re.sub(r'"v" direction', '*v* direction', result)
    # Add code markup.
    result = re.compile(r'(\w+_[\w_()]+)').sub(r'`\1`', result)
    result = re.sub(r'"color1"', '`color1`', result)
    result = re.sub(r'"color2"', '`color2`', result)
    return result

def extract_description(s):
    if not s:
        return "", s
    pat = re.compile(r'anno::description\s*\(\s*"(.*?)\s*"\s*\)', re.S|re.M)
    description = "\n".join(pat.findall(s))
    description = set_format_for_description(description)
    remaining = pat.sub('', s)
    return description, remaining

def extract_annotations(s):
    if not s:
        return {}
    pat = re.compile(r'anno::(\w+)\s*\((.*)', re.M|re.S)
    split_pat = re.compile(r'\)\s*,', re.S|re.M)
    result = {}
    n = 1
    parts = [e.strip() for e in split_pat.split(s)]
    parts[-1] = parts[-1][:-1]
    for anno in parts:
        match = pat.match(anno)
        if not match:
            msg = f'Unexpected annotation format: {anno}'
            raise AssertionError(msg)
        name = match.group(1)
        value = match.group(2)
        if name == 'description':
            value = set_format_for_description(value[1:-1]).capitalize()
        elif name == 'hard_range':
            low, high = [e.strip() for e in value.split(',')]
            value = f'{low}..{high}'
        result[name] = value
        n += 1
    return result

def inner_comma_break(s, breaklevel=1):
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r',\s*', ', ', s)
    e = 0
    i = 0
    result = ''
    breakchar = '` <br> `'
    spacer = '  '
    while i < len(s):
        if s[i] == ',' and e == breaklevel:
            result += ',' + breakchar
            if s[i+1] != ')':
                result += spacer
        elif s[i] == '(':
            e += 1
            result += '('
            if e == 1:
                result += breakchar
                result += spacer
                while s[i] == ' ':
                    s += 1
        elif s[i] == ')':
            if e == 1:
                result += breakchar
            e -= 1
            result += ')'
        else:
            result += s[i]
        i += 1
    return result


def break_long_value(s, maxlength=40):
    if not s:
        return ''
    result = s
    if s.count('(') > 1 or len(s) > maxlength:
        result = inner_comma_break(s)
    elif len(s) > maxlength:
        result = re.sub(r'\s*\(\s*', '( ', result)
        result = re.sub(r'\s*\)\s*', ' )', result)
        result = re.sub(r'\s*,\s*', ', ', result)
        result = "<br>".join(result.split())
    return result

def param_match(name, p):
    p = p.strip()
    m = RE_PARAM.match(p)
    if not m:
        msg = f'Match failure for {name}:\n  {p}'
        raise AssertionError(msg)
    return m

def attrs(name, param):
    m = param_match(name, param)
    pname = m.group('name')
    type_ = m.group('type')
    qual = 'no' if m.group('qual') else 'yes'
    default, anno = default_and_annotation(m.group('tail'))
    return pname, type_, qual, default, anno

def has_a_range(name, params):
    result = False
    for p in params:
        anno = attrs(name, p)[-1]
        anno = extract_annotations(anno)
        range_ = anno.get('hard_range', '')
        if range_:
            result = True
            break
    return result

def add_link(s):
    name = s.split('(')[0]
    if name in LINKED_ITEMS:
        return get_link(s,name)
    return f"`{s}`"

def parameter_split(s, delimiter=','):
    # Assume every parameter has an annotation block
    parts = s.split(f']]{delimiter}')
    return [p + ']]' for p in parts[:-1]] + [parts[-1]]

def default_and_annotation(s):
    s = re.sub(r'\s+', ' ', s)
    # Protect escaped quotes
    s = re.sub(r'\\"', '\1', s)
    s = s.strip('=')
    parts = s.split('[[')
    if len(parts) == 2:
        default, anno = parts
        anno = anno.strip(']]')
        default = default.strip()
        anno = anno.strip()
    else:
        default = parts[0]
        anno = ''

    return default, anno

# --- Output generation ---

def constants(src, ignore):
    rows = ""
    for type_, name, value, desc in RE_CONSTANT.findall(src):
        if name in ignore:
            continue
        rows += f"| `{name}` | `{type_}` |  `{value}` | {desc} |"
    if not rows:
        return ""
    result  = "| Name | Type | Value | Description |\n"
    result += "| ---- | ---- | ----- | ----------- |\n"
    result += rows
    result += "\n"
    return result

def enums(src, ignore):
    result = "# Enums\n\n"

    for name, desc, fields in field_container_pat('enum').findall(src):
        if name in ignore:
            continue
        anno = extract_annotations(desc)
        desc = anno.get('description', '')
        fields = parameter_split(fields, ',')

        result += f"## {name}\n\n"
        result += f"{desc}\n\n"
        result += "| Name | Description |\n"
        result += "| ---- | ----------- |\n"

        for p in fields:
            if not p.strip():
                continue
            if '=' in p:
                pat = re.compile(r'\s*(\w+)\s*=\s*([\w().]+)\s*\[\[\s*(.*?)\s*\]\]\s*', re.M|re.S)
                m = pat.match(p)
                if not m:
                    msg = f'Pattern: {pat.patter}\n    does not match:\n{p}'
                    raise AssertionError(msg)
                pname, _, anno = m.groups()
            else:
                pat = re.compile(r'\s*(\w+)\s*\[\[(.*?)\s*\]\]', re.M|re.S)
                m = pat.match(p)
                if not m:
                    msg = f'Pattern: {pat.patter}\n    does not match:\n{p}'
                    raise AssertionError(msg)
                pname, anno = m.groups()
            desc = extract_annotations(anno).get('description', '')
            result += f"| `{pname}` | {desc} |\n"
        result += "\n"

    result += "\n"
    return result

def structs(src, ignore):
    result = "# Structs\n\n"

    for name, desc, fields, in field_container_pat('struct').findall(src):
        if name in ignore:
            continue
        desc = extract_annotations(desc).get('description', '')

        result += f"## {name}\n\n"
        result += f"{desc}\n\n"
        result += "| Name | Type | Default | Description |\n"
        result += "| ---- | ---- | ------- | ----------- |\n"

        for p in fields.strip().strip(';').split(';'):
            p = p.strip()
            pat = re.compile(
                r'\s*((?:uniform\s+)?\w+)\s*(\w+)\s*=\s*([\w().:]+)\s*\[\[\s*(.*?)\s*\]\]\s*',
                re.M|re.S)
            m = pat.match(p)
            if not m:
                msg = f'Pattern: {pat.pattern}\n    does not match:\n{p}'
                raise AssertionError(msg)
            ptype, pname, default, anno = m.groups()
            desc = extract_annotations(anno).get('description', '')
            result += f"| `{pname}` | `{ptype}` | `{default}` | {desc} |\n"

        result += "\n"
    result += "\n"
    return result

def functions(src, ignore):
    result = "# Functions\n\n"

    for _, type_, name, params, func_anno, _ in RE_FUNCTION_DEFINITION.findall(src):
        if name in ignore:
            continue
        params = parameter_split(params)
        func_anno = extract_annotations(func_anno)
        desc = func_anno.get('description', '')
        include_range = has_a_range(name, params)
        range_title = "| Range " if include_range else ""
        range_ruler = "| ----- " if include_range else ""

        result += f"## {name}\n\n"
        result += f"Returns {add_link(type_)}\n\n"
        result += f"{desc}\n\n"
        result += f"| Name | Type | Default {range_title} | Varying? | Description |\n"
        result += f"| ---- | ---- | ------- {range_ruler} | ---------| ----------- |\n"

        for p in params:
            pname, type_, qual, default, anno = attrs(name, p)
            type_ = add_link(type_)
            anno = extract_annotations(anno)
            description = anno.get('description', '')
            range_ = anno.get('hard_range', '')
            if default:
                default = f"{add_link(break_long_value(default))}"
            else:
                default = "(none)"
            if include_range:
                if range_:
                    range_ = f"| `{range_}` "
                else:
                    range_ = f"| "
            else:
                range_ = ""
            result += f"| `{pname}` | {type_} | {default} {range_} | {qual} | {description} |\n"

        result += "\n"

    result += "\n"
    return result

def main():
    if len(sys.argv) != 1+3:
        print(f"Usage: {sys.argv[0]} <mdl_input> <md_intro> <md_output>")
        raise SystemExit(1)

    mdl_input = sys.argv[1]
    md_header = sys.argv[2]
    md_output = sys.argv[3]

    with open(md_header, encoding="utf-8") as fp:
        result = fp.read()
    result += "\n\n"

    with open(mdl_input, encoding="utf-8") as fp:
        src = fp.read()

    ignore = symbols_to_ignore(src)
    src = modify_module_source(src)

    result += table_of_contents(src, ignore)

    for func in [constants, enums, structs, functions]:
        text = func(src, ignore)
        if text:
            result += text

    with open(md_output, 'w', encoding="utf-8") as fp:
        fp.write(result)

if __name__ == "__main__":
    main()
