/******************************************************************************
 * Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *****************************************************************************/

#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>

#include <mi/mdl_sdk.h>
#include <mi/base/ilogger.h>

#include "mdl_assert.h"
#include "mdl_printer.h"
#include "mdl_distiller_utils.h"

using mi::IString;

using mi::base::Handle;

using mi::neuraylib::IExpression_direct_call;
using mi::neuraylib::IExpression_list;
using mi::neuraylib::IExpression;
using mi::neuraylib::IExpression_constant;
using mi::neuraylib::IExpression_temporary;
using mi::neuraylib::IExpression_parameter;

using mi::neuraylib::IFunction_definition;
using mi::neuraylib::IMdl_configuration;

using mi::neuraylib::IValue;
using mi::neuraylib::IValue_compound;
using mi::neuraylib::IValue_bool;
using mi::neuraylib::IValue_int;
using mi::neuraylib::IValue_enum;
using mi::neuraylib::IValue_float;
using mi::neuraylib::IValue_double;
using mi::neuraylib::IValue_string;
using mi::neuraylib::IValue_vector;
using mi::neuraylib::IValue_matrix;
using mi::neuraylib::IValue_color;
using mi::neuraylib::IValue_array;
using mi::neuraylib::IValue_struct;
using mi::neuraylib::IValue_invalid_df;
using mi::neuraylib::IValue_texture;
using mi::neuraylib::IValue_light_profile;
using mi::neuraylib::IValue_bsdf_measurement;

using mi::neuraylib::IType;
using mi::neuraylib::IType_atomic;
using mi::neuraylib::IType_alias;
using mi::neuraylib::IType_array;
using mi::neuraylib::IType_matrix;
using mi::neuraylib::IType_int;
using mi::neuraylib::IType_float;
using mi::neuraylib::IType_double;
using mi::neuraylib::IType_string;
using mi::neuraylib::IType_enum;
using mi::neuraylib::IType_struct;
using mi::neuraylib::IType_vector;
using mi::neuraylib::IType_texture;
using mi::neuraylib::IType_light_profile;
using mi::neuraylib::IType_bsdf_measurement;
using mi::neuraylib::IType_reference;

char const *utf8_to_unicode_char(char const *up, unsigned &res)
{
    bool error = false;
    unsigned char ch = up[0];

    // find start code: either 0xxxxxxx or 11xxxxxx
    while ((ch >= 0x80) && ((ch & 0xC0) != 0xC0)) {
        ++up;
        ch = up[0];
    }

    if (ch <= 0x7F) {
        // 0xxxxxxx
        res = ch;
        up += 1;
    } else if ((ch & 0xF8) == 0xF0) {
        // 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
        unsigned c1 = ch & 0x07; ch = up[1]; error |= (ch & 0xC0) != 0x80;
        unsigned c2 = ch & 0x3F; ch = up[2]; error |= (ch & 0xC0) != 0x80;
        unsigned c3 = ch & 0x3F; ch = up[3]; error |= (ch & 0xC0) != 0x80;
        unsigned c4 = ch & 0x3F;
        res = (c1 << 18) | (c2 << 12) | (c3 << 6) | c4;

        // must be U+10000 .. U+10FFFF
        error |= (res < 0x1000) || (res > 0x10FFFF);

        // Because surrogate code points are not Unicode scalar values, any UTF-8 byte
        // sequence that would otherwise map to code points U+D800..U+DFFF is illformed
        error |= (0xD800 <= res) && (res <= 0xDFFF);

        if (!error) {
            up += 4;
        } else {
            res = 0xFFFD;  // replacement character
            up += 1;
        }
    } else if ((ch & 0xF0) == 0xE0) {
        // 1110xxxx 10xxxxxx 10xxxxxx
        unsigned c1 = ch & 0x0F; ch = up[1]; error |= (ch & 0xC0) != 0x80;
        unsigned c2 = ch & 0x3F; ch = up[2]; error |= (ch & 0xC0) != 0x80;
        unsigned c3 = ch & 0x3F;
        res = (c1 << 12) | (c2 << 6) | c3;

        // must be U+0800 .. U+FFFF
        error |= res < 0x0800;

        // Because surrogate code points are not Unicode scalar values, any UTF-8 byte
        // sequence that would otherwise map to code points U+D800..U+DFFF is illformed
        error |= (0xD800 <= res) && (res <= 0xDFFF);

        if (!error) {
            up += 3;
        } else {
            res = 0xFFFD;  // replacement character
            up += 1;
        }
    } else if ((ch & 0xE0) == 0xC0) {
        // 110xxxxx 10xxxxxx
        unsigned c1 = ch & 0x1F; ch = up[1]; error |= (ch & 0xC0) != 0x80;
        unsigned c2 = ch & 0x3F;
        res = (c1 << 6) | c2;

        // must be U+0080 .. U+07FF
        error |= res < 0x80;

        if (!error) {
            up += 2;
        } else {
            res = 0xFFFD;  // replacement character
            up += 1;
        }
    } else {
        // error
        res = 0xFFFD;  // replacement character
        up += 1;
    }
    return up;
}

/// Return whether a function of the given return type can be baked.
///
/// \param t type to check
///
/// \return true of the given type can be baked, false otherwise.
static bool bakable_type(IType const *t)
{
    switch (t->get_kind()) {
    case IType::TK_BOOL:
    case IType::TK_INT:
    case IType::TK_COLOR:
    case IType::TK_FLOAT:
    case IType::TK_DOUBLE:
        return true;

    case IType::TK_VECTOR:
    {
        Handle<IType_vector const> vec_t(t->get_interface<IType_vector>());
        mdl_assert(vec_t.is_valid_interface());

        Handle<IType const> elem_t(vec_t->get_element_type());

        switch (elem_t->get_kind()) {
        case IType::TK_FLOAT:
        case IType::TK_DOUBLE:
            return true;

        default:
            return false;
        }
    }
    default:
        return false;
    }
}

/// Return the texture type used for baking for the given type `t`.
///
/// \param t type to determine baking texture type name for.
///
/// \return string naming the texture type for baking, or the empty
/// string if the type cannot be baked.
std::string get_baking_type(IType const *t) {
    switch (t->get_kind()) {
    case IType::TK_BOOL:
        return "bool";

    case IType::TK_INT:
        return "int";

    case IType::TK_COLOR:
        return "color";

    case IType::TK_FLOAT:
        return "float";

    case IType::TK_VECTOR:
    {
        Handle<IType_vector const> vec_t(t->get_interface<IType_vector>());
        mdl_assert(vec_t.is_valid_interface());

        Handle<IType const> elem_t(vec_t->get_element_type());

        switch (elem_t->get_kind()) {
        case IType::TK_FLOAT:
            return "float" + std::to_string(vec_t->get_size());

        case IType::TK_DOUBLE:
            return "double" + std::to_string(vec_t->get_size());

        default:
            std::cerr << "*** baking type: " << vec_t->get_kind() << "\n";
            mdl_assert(!"unsupported type in get_baking_type *(1)");
        return "";
        }
    }
    default:
        std::cerr << "*** baking type: " << t->get_kind() << "\n";
        mdl_assert(!"unsupported type in get_baking_type *(2)");
        return "";
    }
}

/// Remove leading scope and replace all others with '_' to form valid
/// identifier
std::string derived_material_name(std::string mdl_material_name) {
    std::string output;
    mdl_material_name.replace(0, 2, "");
    char const *p_end = mdl_material_name.c_str() + mdl_material_name.size();
    for (char const *p = mdl_material_name.c_str(); p < p_end ;) {
        unsigned unicode_char;
        p = utf8_to_unicode_char(p, unicode_char);

        if ((unicode_char >= 'a' && unicode_char <= 'z') ||
            (unicode_char >= 'A' && unicode_char <= 'Z') ||
            (unicode_char >= '0' && unicode_char <= '9') || 
            (unicode_char == '_')) {
            output += char(unicode_char);
        } else if (unicode_char == ':') {
            output += '_';
        } else {
            char buf[32];
            snprintf(buf, sizeof(buf), "%08x", unicode_char);
            output += "_U";
            output += buf;
        }
    }
    return output;
    // while (true) {
    //     std::string::size_type scope = mdl_material_name.find("::");
    //     if (scope == std::string::npos)
    //         break;
    //     mdl_material_name.replace(scope, 2, "_");
    // }
    // return mdl_material_name;
}

/// Replace dots with underscores.
std::string clean_identifier(std::string name) {
    while (true) {
        std::string::size_type idx = name.find(".");
        if (idx == std::string::npos)
            break;
        name.replace(idx, 1, "_");
    }
    return name;
}

/// Concatenate the given strings, with a dot in the middle if both
/// are non-empty.
static std::string dot_sep(std::string const &a, std::string const &b) {
    if (a.size() == 0) {
        return b;
    } else if (b.size() == 0) {
        return a;
    } else {
        return a + "." + b;
    }
}

/// Given a field selection function name, return the corresponding
/// field name (including the leading dot).
static std::string get_struct_selector(std::string const& function_name) {
    std::string::size_type last_dot = function_name.rfind(".");
    return std::string(function_name,last_dot);
}

/// Returns the type modifier as string.
static const char* modifier_string( const IType* type) {
    mi::Uint32 m = type->get_all_type_modifiers();
    if ( m & IType::MK_VARYING)
        return "varying ";
    if ( m & IType::MK_UNIFORM)
        return "uniform ";
    return "";
}

/// Returns the type of an elemental type as string.
static const char* elem_type_name(const IType_atomic* elem_type) {
    switch ( elem_type->get_kind()) {
    case IType::TK_BOOL:
        return "bool";
    case IType::TK_INT:
        return "int";
    case IType::TK_FLOAT:
        return "float";
    case IType::TK_DOUBLE:
        return "double";
    default:
        break;
    }
    return "UNKNOWN_TYPE";
}

/// Remove the "mdl" prefix from a database name and return the rest
/// of the string (including the leading "::").
static std::string drop_prefix(std::string const &ident)
{
    if (ident.find("mdl::") != std::string::npos) {
        return std::string(ident, 3);
    } else {
        return ident;
    }
}

static std::string drop_module_prefix(std::string qualified_name)
{
    std::string::size_type last_scope = qualified_name.rfind("::");
    if (last_scope == std::string::npos)
        return qualified_name;
    return std::string(qualified_name, last_scope + 2);
}

/// Mdl_printer Constructor.
Mdl_printer::Mdl_printer(
    INeuray* neuray,
    mi::neuraylib::IMdl_factory *mdl_factory,
    mi::neuraylib::IExpression_factory *expr_factory,
    std::ostream &out,
    Options const* options,
    char const *target,
    std::string const &material_name,
    std::string const &print_material_name,
    ICompiled_material const *material,
    Bake_paths* bake_paths,
    bool do_bake,
    ITransaction *transaction)
    : m_neuray(neuray)
    , m_mdl_factory(mdl_factory)
    , m_expr_factory(expr_factory)
    , m_out(out)
    , m_options(options)
    , m_target(target)
    , m_material_name(material_name)
    , m_derived_material_name(derived_material_name(material_name))
    , m_print_material_name(print_material_name)
    , m_material(material)
    , m_bake_paths(bake_paths)
    , m_do_bake(do_bake)
    , m_transaction(transaction)
    , m_required_mdl_version(mdl_spec_1_3)
    , m_used_modules()
    , m_used_identifiers()
    , m_used_parameters()
    , m_temporary_stack()
    , m_baked_temporaries()
    , m_messages()
    , m_error_count(0)
    , m_suppress_trivial_temps(true)
    , m_unfold_temps(false)
    , m_outline_mode(false)
    , m_suppress_default_parameters(true)
    , m_emit_messages(true)
    , m_warn_on_parameters(false)
{
}

/// Return true if the given expression should enclosed by parentheses
/// if printed as a subexpression.
bool Mdl_printer::expression_needs_parens(IExpression const *expr)
{
    switch (expr->get_kind()) {
    case IExpression::EK_CONSTANT:
        return false;

    case IExpression::EK_CALL:
        return true;

    case IExpression::EK_PARAMETER:
        return false;

    case IExpression::EK_DIRECT_CALL:
    {
        Handle<const IExpression_direct_call> call(
            expr->get_interface<const IExpression_direct_call>());
        mdl_assert( call.is_valid_interface());

        Handle<const IExpression_list> arguments(call->get_arguments());

        Handle<const IFunction_definition> function(
            m_transaction->access<IFunction_definition>(call->get_definition()));


        IFunction_definition::Semantics semantic = function->get_semantic();
        bool is_array = function->is_array_constructor();
        if (is_array)
            return false;

        if (semantic == IFunction_definition::IFunction_definition::DS_ARRAY_INDEX) {
            Handle<IExpression const> array_expr(arguments->get_expression(mi::Size(0)));
            Handle<IExpression const> index_expr(arguments->get_expression(1));
            return expression_needs_parens(array_expr.get());
        }

        if ((semantic == IFunction_definition::DS_SELECT) 
                || (semantic == IFunction_definition::DS_INTRINSIC_DAG_FIELD_ACCESS)) {
            Handle<IExpression const> struct_expr(arguments->get_expression(mi::Size(0)));
            return expression_needs_parens(struct_expr.get());
        }
        if (semantic == IFunction_definition::DS_TERNARY) {
            return true;
        }

        if ((semantic >= IFunction_definition::DS_UNARY_FIRST) 
                && (semantic < IFunction_definition::DS_BINARY_FIRST)){
            return true;
        }
        if ((semantic >= IFunction_definition::DS_BINARY_FIRST) 
                && (semantic < IFunction_definition::DS_TERNARY)) {
            return true;
        }

        return false;
    }

    case IExpression::EK_TEMPORARY:
        return false;

    default:
        mdl_assert(!"unreachable");
        return true;
    }
}

/// Return true if `expr` is a trivial expression, which is a constant
/// of numeric, boolean, string, color or enum type.
bool Mdl_printer::trivial_expression(IExpression const *expr)
{
    if (m_suppress_trivial_temps) {
        if (expr->get_kind() == IExpression::EK_CONSTANT 
                || expr->get_kind() == IExpression::EK_PARAMETER) {
            Handle<IType const> t(expr->get_type());
            switch (t->get_kind()) {
            case IType::TK_INT:
            case IType::TK_FLOAT:
            case IType::TK_DOUBLE:
            case IType::TK_STRING:
            case IType::TK_BOOL:
            case IType::TK_ENUM:
            case IType::TK_COLOR:
                return true;

            default:
                return false;
            }
        }
    }
    return false;
}

/// Return true if `expr` contains an "interesting" expression. An
/// interesting expression is one that should not be omitted from the
/// output in outline mode, e.g. BSDFs and all BSDF
/// combinators/transformers.
bool Mdl_printer::contains_interesting_expression(IExpression const *expr)
{
    Handle<IType const> t(expr->get_type());
    IType::Kind kind =t->get_kind();
    if (kind == IType::TK_BSDF || kind == IType::TK_VDF || kind == IType::TK_EDF 
            || kind == IType::TK_HAIR_BSDF) {
        return true;
    }

    switch (expr->get_kind()) {
    case IExpression::EK_CONSTANT:
    {
        return false;
    }

    case IExpression::EK_CALL:
    {
        // TODO: add indirect call expression support if needed
        error("material '" + m_material_name + "' contains indirect call.");
        return false;
    }

    case IExpression::EK_PARAMETER:
    {
        return false;
    }

    case IExpression::EK_DIRECT_CALL:
    {
        Handle<const IExpression_direct_call> call(
            expr->get_interface<const IExpression_direct_call>());
        mdl_assert(call.is_valid_interface());
        Handle<const IExpression_list> arguments(call->get_arguments());

        Handle<const IFunction_definition> function(
            m_transaction->access<IFunction_definition>(call->get_definition()));

        bool interesting = false;
        mi::Size argc = arguments->get_size();
        for (mi::Size i = 0; i < argc; i++) {
            Handle<IExpression const> arg_expr(arguments->get_expression(i));
            interesting = interesting || contains_interesting_expression(arg_expr.get());
        }
        return interesting;
    }

    case IExpression::EK_TEMPORARY:
    {
        Handle<const IExpression_temporary> temp(
            expr->get_interface<const IExpression_temporary>());
        mdl_assert(temp.is_valid_interface());

        Handle<const IExpression> temp_expr(
            m_material->get_temporary(temp->get_index()));
        return contains_interesting_expression(temp_expr.get());
    }

    default:
        mdl_assert(!"unreachable");
        return true;
    }
}

bool Mdl_printer::suppress_expression(IExpression const *expr)
{
    if (m_outline_mode) {
        return !contains_interesting_expression(expr);
    } else {
        return false;
    }
}

void Mdl_printer::warning(std::string msg)
{
    m_messages.push_back("warning: " + msg);
}

void Mdl_printer::error(std::string msg)
{
    m_messages.push_back("error: " + msg);
    m_error_count++;
}

void Mdl_printer::ind(int indent) {
    for (int i = 0; i < indent; i++) {
        m_out << "  ";
    }
}

bool Mdl_printer::is_unicode_identifier(std::string const &identifier) {
    for (auto &c : identifier) {
        if (c < ' ' || c >= 127) {
            return true;
        }
    }
    return false;
}

void Mdl_printer::print_identifier(std::string const &identifier) {
    std::string ident = identifier;

    // We will change a couple of settings while printing, so back
    // up the current settings.
    std::ios::fmtflags old_settings = m_out.flags();

    size_t c_pos = std::string::npos;
    do {
        c_pos = ident.find("::");
        std::string elem;
        if (c_pos == std::string::npos) {
            elem = ident;
        } else {
            elem = ident.substr(0, c_pos);
            ident = ident.substr(c_pos + 2);
        }
        if (is_unicode_identifier(elem)) {
            m_out << '\'';
            char const *p_end = elem.c_str() + elem.size();
            for (char const *p = elem.c_str(); p < p_end ;) {
                unsigned unicode_char;
                p = utf8_to_unicode_char(p, unicode_char);

                switch (unicode_char) {
                case '\a':  m_out << "\\a";   break;
                case '\b':  m_out <<  "\\b";   break;
                case '\f':  m_out << "\\f";   break;
                case '\n':  m_out << "\\n";   break;
                case '\r':  m_out << "\\r";   break;
                case '\t':  m_out << "\\t";   break;
                case '\v':  m_out << "\\v";   break;
                case '\\':  m_out << "\\\\";  break;
                case '\'':  m_out << "\\\'";  break;
                case '"':   m_out << "\\\"";  break;
                default:

                    if (unicode_char >= 32 && unicode_char <= 0x7F) {
                        m_out << char(unicode_char);
                    } else if (unicode_char <= 0xFFFF) {
                        m_out << "\\u" << std::setw(4) << std::setfill('0') << std::hex 
                            << unsigned(unicode_char);
                    } else {
                        m_out << "\\U" << std::setw(8) << std::setfill('0') << std::hex 
                            << unsigned(unicode_char);
                    }
                    break;
                }
            }
            m_out << '\'';
        } else {
            m_out << elem;
        }
        if (c_pos != std::string::npos) {
            m_out << "::";
        }
    } while (c_pos != std::string::npos);

    // Restore formmatting settings.
    m_out.flags(old_settings);
}

// Returns a vector type as string.
void Mdl_printer::print_vector_type(IType const* type) {
    Handle<const IType_vector> vector_type(type->get_interface<const IType_vector>());
    mdl_assert(vector_type.is_valid_interface());
    Handle<const IType_atomic> elem_type(vector_type->get_element_type());
    mi::Size n = vector_type->get_size();
    m_out << elem_type_name(elem_type.get()) << n;
}

// Returns a matrix type as string
void Mdl_printer::print_matrix_type(IType const* type) {
    Handle<const IType_matrix> matrix_type(type->get_interface<const IType_matrix>());
    mdl_assert(matrix_type.is_valid_interface());
    Handle<const IType_vector> vector_type(matrix_type->get_element_type());
    Handle<const IType_atomic> elem_type( vector_type->get_element_type());
    mi::Size ni = matrix_type->get_size();
    mi::Size nj = vector_type->get_size();
    m_out << elem_type_name(elem_type.get()) << ni << 'x' << nj;
}

void Mdl_printer::print_type(const IType* type) {
    switch (type->get_kind()) {
    case IType::TK_ALIAS:
    {
        Handle<const IType_alias> alias_type(type->get_interface<IType_alias>());
        mdl_assert(alias_type.is_valid_interface());

        warning("material '" + m_material_name + "' contains an alias type.");

        m_out << modifier_string(type);
        Handle<const IType> subtype( alias_type->get_aliased_type());
        print_type(subtype.get());
        break;
    }

    case IType::TK_BOOL:
        m_out << "bool";
        break;

    case IType::TK_INT:
        m_out << "int";
        break;

    case IType::TK_ENUM:
    {
        Handle<const IType_enum> enum_type(type->get_interface<IType_enum>());
        mdl_assert(enum_type.is_valid_interface());

        m_out << enum_type->get_symbol();
        break;
    }

    case IType::TK_FLOAT:
        m_out << "float";
        break;

    case IType::TK_DOUBLE:
        m_out << "double";
        break;

    case IType::TK_STRING:
        m_out << "string";
        break;

    case IType::TK_VECTOR:
        print_vector_type(type);
        break;

    case IType::TK_MATRIX:
        print_matrix_type(type);
        break;

    case IType::TK_COLOR:
        m_out << "color";
        break;

    case IType::TK_ARRAY:
    {
        Handle<const IType_array> array_type(type->get_interface<IType_array>());
        mdl_assert(array_type.is_valid_interface());

        if (!array_type->is_immediate_sized()) {
            error("material '" + m_material_name + "' contains deferred-size array type.");
        } else {
            Handle<const IType> elem_type(array_type->get_element_type());
            print_type(elem_type.get());
            m_out << '[' << array_type->get_size() << ']';
        }
        break;
    }


    case IType::TK_STRUCT:
    {
        Handle<const IType_struct> struct_type(type->get_interface<IType_struct>());
        mdl_assert(struct_type.is_valid_interface());

        m_out << struct_type->get_symbol();
        break;
    }

    case IType::TK_TEXTURE:
    {
        Handle<const IType_texture> texture_type(type->get_interface<const IType_texture>());
        mdl_assert(texture_type.is_valid_interface());

        switch (texture_type->get_shape()) {
        case IType_texture::TS_2D:
            m_out << "texture_2d";
            break;
        case IType_texture::TS_3D:
            m_out << "texture_3d";
            break;
        case IType_texture::TS_CUBE:
            m_out << "texture_cube";
            break;
        case IType_texture::TS_PTEX:
            m_out << "texture_ptex";
            break;
        case IType_texture::TS_BSDF_DATA:
            mdl_assert(!"bsdf data should not occur in MDL source code");
            m_out << "texture_bsdf_data";
            break;
        default:
            mdl_assert(!"unreachable");
            break;
        }
        break;
    }
    case IType::TK_LIGHT_PROFILE:
        m_out << "light_profile";
        break;

    case IType::TK_BSDF_MEASUREMENT:
        m_out << "bsdf_measurement";
        break;

    case IType::TK_BSDF:
        m_out << "bsdf";
        break;

    case IType::TK_HAIR_BSDF:
        m_out << "hair_bsdf";
        break;

    case IType::TK_EDF:
        m_out << "edf";
        break;

    case IType::TK_VDF:
        m_out << "vdf";
        break;

    default:
        break;
    }
}

void Mdl_printer::print_value(IValue const *value, std::string const &path_prefix) {
    Handle<IType const> vt(value->get_type());
    if (m_do_bake && bakable_type(vt.get())) {
        std::string type = get_baking_type(vt.get());
        bool b = true;
        if (m_options->all_textures) {
            m_bake_paths->push_back(Bake_path(type, path_prefix));
        } else {
            b = (m_bake_paths->end() != std::find_if(m_bake_paths->begin(), m_bake_paths->end(),
                                                     Bake_path_cmp(path_prefix)));
        }
        if (b) {
            std::string texname = baked_texture_file_name(m_derived_material_name.c_str(), 
                    path_prefix);

            if (is_normal_map_path(path_prefix))
            {
                m_out << type << "(::base::tangent_space_normal_texture("
                    << "texture: texture_2d(\"" << texname << "\", tex::gamma_linear)))";
            }
            else
            {
                m_out << type << "(::base::file_texture("
                    << "texture: texture_2d(\"" << texname << "\", tex::gamma_linear), "
                    << "mono_source: ::base::mono_average)."
                    << (type == "float" ? "mono" : "tint") << ")";
            }
            return;
        }
    }
    switch (value->get_kind()) {
    case IValue::VK_BOOL:
    {
        Handle<IValue_bool const> b(value->get_interface<IValue_bool const>());
        mdl_assert(b.is_valid_interface());

        m_out << (b->get_value() ? "true" : "false");
        break;
    }

    case IValue::VK_INT:
    {
        Handle<IValue_int const> i(value->get_interface<IValue_int const>());
        mdl_assert(i.is_valid_interface());

        m_out << i->get_value();
        break;
    }

    case IValue::VK_ENUM:
    {
        Handle<IValue_enum const> e(value->get_interface<IValue_enum const>());
        mdl_assert(e.is_valid_interface());

        mi::Size idx = e->get_index();
        Handle<const IType_enum> enum_type(e->get_type());
        std::string variant_name(get_module_from_qualified_name(enum_type->get_symbol())
                                 + "::" + enum_type->get_value_name(idx));
        m_out << variant_name;
        break;
    }

    case IValue::VK_FLOAT:
    {
        Handle<IValue_float const> float_val(value->get_interface<IValue_float const>());
        mdl_assert(float_val.is_valid_interface());

        float f = float_val->get_value();
        if (std::isnan(f)) {
            m_out << "(0.0f/0.0f)";
            break;
        }
        if (std::isinf(f)) {
            if (f < 0.0f) {
                m_out << "(-1.0f/0.0f)";
            } else {
                m_out << "(1.0f/0.0f)";
            }
            break;
        }
        if (f == float(int(f))) {
            m_out << int(f) << ".0f";
        } else {
            m_out << f << "f";
        }
        break;
    }

    case IValue::VK_DOUBLE:
    {
        Handle<IValue_double const> double_val(value->get_interface<IValue_double const>());
        mdl_assert(double_val.is_valid_interface());

        double d = double_val->get_value();
        if (std::isnan(d)) {
            m_out << "(0.0/0.0)";
            break;
        }
        if (std::isinf(d)) {
            if (d < 0.0) {
                m_out << "(-1.0/0.0)";
            } else {
                m_out << "(1.0/0.0)";
            }
            break;
        }

        m_out << d;
        break;
    }

    case IValue::VK_STRING:
    {
        Handle<IValue_string const> val_string(value->get_interface<IValue_string const>());
        mdl_assert(val_string.is_valid_interface());

        m_out << "\"" << val_string->get_value() << "\"";
        break;
    }

    case IValue::VK_VECTOR:
    {
        Handle<IValue_vector const> vec(value->get_interface<IValue_vector const>());
        mdl_assert(vec.is_valid_interface());

        Handle<const IType_vector> vector_type(vec->get_type());
        print_vector_type(vector_type.get());
        m_out << '(';
        mi::Size n = vector_type->get_size();
        for ( mi::Size i = 0; i < n; ++i) {
            if ( i > 0) {
                m_out << ',';
            }
            Handle<const IValue> s(vec->get_value(i));
            std::stringstream stream;
            stream << i;
            print_value(s.get(), dot_sep(path_prefix, stream.str()));
        }
        m_out << ')';
        break;
    }

    case IValue::VK_MATRIX:
    {
        Handle<IValue_matrix const> val_matrix(value->get_interface<IValue_matrix const>());
        mdl_assert(val_matrix.is_valid_interface());

        Handle<const IType_matrix> matrix_type(val_matrix->get_type());
        Handle<const IType_vector> vector_type(matrix_type->get_element_type());
        mi::Size ni = matrix_type->get_size();
        mi::Size nj = vector_type->get_size();
        print_matrix_type(matrix_type.get());
        m_out << '(';
        for ( mi::Size i = 0; i < ni; ++i) {
            Handle<const IValue_vector> vec(val_matrix->get_value(i));
            for ( mi::Size j = 0; j < nj; ++j) {
                if ( i > 0 || j > 0) {
                    m_out << ',';
                }
                Handle<const IValue> s(vec->get_value(j));
                std::stringstream stream;
                stream << i << ", " << j;
                print_value(s.get(), dot_sep(path_prefix, stream.str()));
            }
        }
        m_out << ')';
        break;
    }

    case IValue::VK_COLOR:
    {
        Handle<IValue_color const> d(value->get_interface<IValue_color const>());
        mdl_assert(d.is_valid_interface());

        Handle<IValue_float const> r(d->get_value(0));
        Handle<IValue_float const> g(d->get_value(1));
        Handle<IValue_float const> b(d->get_value(2));
        m_out << "color(";
        if (std::isfinite(r->get_value()) && r->get_value() == g->get_value() 
                && r->get_value() == b->get_value()) {
            if (r->get_value() != 0.0) {
                print_value(r.get(), path_prefix); // FIXME: what prefix is correct?
            }
        } else {
            print_value(r.get(), dot_sep(path_prefix, "r"));
            m_out << ",";
            print_value(g.get(), dot_sep(path_prefix, "g"));
            m_out << ",";
            print_value(b.get(), dot_sep(path_prefix, "b"));
        }
        m_out << ")";
        break;
    }

    case IValue::VK_ARRAY:
    {
        Handle<const IValue_compound> val_array(value->get_interface<const IValue_compound>());
        mdl_assert(val_array.is_valid_interface());

        Handle<const IType_array> array_type(value->get_type<IType_array>());
        Handle<const IType> elem_type(array_type->get_element_type());
        print_type(elem_type.get());
        m_out << "[](";
        mi::Size n = val_array->get_size();
        for (mi::Size i = 0; i < n; ++i) {
            Handle<const IValue> elem(val_array->get_value(i));
            if (i > 0) {
                m_out << ',';
            }
            std::stringstream stream;
            stream << i;
            print_value(elem.get(), dot_sep(path_prefix, stream.str()));
        }
        m_out << ')';
        break;
    }

    case IValue::VK_STRUCT:
    {
        Handle<IValue_struct const> s(value->get_interface<IValue_struct const>());
        mdl_assert(s.is_valid_interface());

        Handle<const IType_struct> struct_type(s->get_type());
        const char* struct_name = struct_type->get_symbol();
        m_out << struct_name << '(';
        mi::Size n = s->get_size();
        for (mi::Size i = 0; i < n; ++i) {
            Handle<const IValue> elem(s->get_value(i));
            char const *sel_name = struct_type->get_field_name(i);
            if ( i > 0) {
                m_out << ',';
            }
            print_value(elem.get(), dot_sep(path_prefix, sel_name)); // FIXME: Is path creation 
                                                                     //        correct?
        }
        m_out << ')';
        break;
    }

    case IValue::VK_INVALID_DF:
    {
        Handle<const IValue_invalid_df> val_ref(
            value->get_interface<const IValue_invalid_df>());
        mdl_assert(val_ref.is_valid_interface());

        Handle<const IType_reference> ref_type(val_ref->get_type());
        print_type(ref_type.get());
        m_out << "()";
        break;
    }

    case IValue::VK_TEXTURE:
    {
        Handle<const IValue_texture> val_texture(value->get_interface<const IValue_texture>());
        mdl_assert(val_texture.is_valid_interface());

        Handle<const IType_texture> texture_type(val_texture->get_type());
        print_type(texture_type.get());
        m_out << '(';
        Handle<const mi::neuraylib::ITexture> tex(
            m_transaction->access<mi::neuraylib::ITexture>(val_texture->get_value()));
        if (tex.is_valid_interface()) {
            const char *url = val_texture->get_file_path();
            if (url != NULL) {
                m_out << '\"' << url << '\"';
            } else {
                Handle<const mi::neuraylib::IImage> img(
                    m_transaction->access<mi::neuraylib::IImage>(tex->get_image()));
                // TODO add uvtile/animated texture support
                m_out << '\"' << strip_path(m_neuray, img->get_filename(0, 0)) << '\"';
            }
            float gamma = tex->get_gamma();
            if (gamma == 1.0) {
                m_out << ", gamma: ::tex::gamma_linear";
            }
            if ( gamma == 2.2) {
                m_out << ", gamma: ::tex::gamma_srgb";
            }
        }
        m_out << ')';
        break;
    }

    case IValue::VK_LIGHT_PROFILE:
    {
        Handle<const IValue_light_profile> val_light_profile(
            value->get_interface<const IValue_light_profile>());
        m_out << "light_profile(";
        Handle<const mi::neuraylib::ILightprofile> lp(
            m_transaction->access<mi::neuraylib::ILightprofile>(
                val_light_profile->get_value()));
        if (lp.is_valid_interface()) {
            m_out << '\"' << strip_path(m_neuray, lp->get_filename()) << '\"';
        }
        m_out << ')';
        break;
    }

    case IValue::VK_BSDF_MEASUREMENT:
    {
        Handle<const IValue_bsdf_measurement> val_bsdf_measurement(
            value->get_interface<const IValue_bsdf_measurement>());
        m_out << "bsdf_measurement(";
        Handle<const mi::neuraylib::IBsdf_measurement> mbsdf(
            m_transaction->
            access<mi::neuraylib::IBsdf_measurement>(val_bsdf_measurement->get_value()));
        if (mbsdf.is_valid_interface()) {
            m_out << '\"' << strip_path(m_neuray, mbsdf->get_filename()) << '\"';
        }
        m_out << ')';
        break;
    }

    default:
        mdl_assert(!"unsupported variant");
        break;
    }
}

void Mdl_printer::print_direct_call(IExpression_direct_call const *call, 
        std::string const &path_prefix, int indent)
{
    Handle<const IExpression_list> arguments(call->get_arguments());

    Handle<const IFunction_definition> function(
        m_transaction->access<IFunction_definition>(call->get_definition()));
    std::string function_name = drop_signature(function->get_mdl_name());

    Handle<const IString> s(m_mdl_factory->decode_name(function_name.c_str()));
    function_name = s->get_c_str();

    if (function_name.substr(0,30) == std::string("::nvidia::distilling_support::")) {
        function_name = m_options->dist_supp_file + function_name.substr(28);
    }

    IFunction_definition::Semantics semantic = function->get_semantic();

    bool is_array = function->is_array_constructor();
    if (is_array) {
        if (suppress_expression(call)) {
            m_out << "...";
            return;
        }
        Handle<const IExpression> arg0(arguments->get_expression(mi::Size(0)));
        Handle<const IType> arg0_type(arg0->get_type());
        print_type(arg0_type.get());
        m_out << "[](";
        mi::Size argc = arguments->get_size();
        if (argc > 1) {
            m_out << "\n";
            ind(indent + 1);
        }
        for (mi::Size i = 0; i < argc; i++) {
            Handle<IExpression const> arg_expr(arguments->get_expression(i));
            if (i > 0) {
                m_out << ",";
                m_out << "\n";
                ind(indent + 1);
            }
            std::stringstream s;
            s << i;
            print_expression(arg_expr.get(), dot_sep(path_prefix, s.str()), // FIXME: Does path
                             indent + 1);                                   // extension make
                                                                            // sense here?
        }
        if (argc > 1) {
            m_out << "\n";
            ind(indent);
        }
        m_out << ")";
    } else if (semantic == IFunction_definition::IFunction_definition::DS_ARRAY_INDEX) {
        if (suppress_expression(call)) {
            m_out << "...";
            return;
        }
        Handle<IExpression const> array_expr(arguments->get_expression(mi::Size(0)));
        Handle<IExpression const> index_expr(arguments->get_expression(1));
        bool arg0_parens = expression_needs_parens(array_expr.get());

        if (arg0_parens)
            m_out << "(";
        print_expression(array_expr.get(), path_prefix, indent); // FIXME: path_prefix correct?
        if (arg0_parens)
            m_out << ")";
        m_out << "[";
        print_expression(index_expr.get(), path_prefix, indent); // FIXME: path_prefix correct?
        m_out << "]";
    } else if ((semantic == IFunction_definition::DS_SELECT) 
            || (semantic == IFunction_definition::DS_INTRINSIC_DAG_FIELD_ACCESS)) {
        if (suppress_expression(call)) {
            m_out << "...";
            return;
        }
        Handle<IExpression const> struct_expr(arguments->get_expression(mi::Size(0)));

        bool arg0_parens = expression_needs_parens(struct_expr.get());
        if (arg0_parens)
            m_out << "(";
        print_expression(struct_expr.get(),
                         dot_sep(path_prefix, get_struct_selector(function_name)), // FIXME:Is path
                         indent);                                                  // creation 
                                                                                   // correct?
        if (arg0_parens)
            m_out << ")";
        print_identifier(get_struct_selector(function_name));
    } else if (semantic == IFunction_definition::DS_TERNARY) {
        if (suppress_expression(call)) {
            m_out << "...";
            return;
        }
        Handle<IExpression const> arg0_expr(arguments->get_expression(mi::Size(0)));
        char const *arg0_name = arguments->get_name(mi::Size(0));
        Handle<IExpression const> arg1_expr(arguments->get_expression(1));
        char const *arg1_name = arguments->get_name(1);
        Handle<IExpression const> arg2_expr(arguments->get_expression(2));
        char const *arg2_name = arguments->get_name(2);

        bool arg0_parens = expression_needs_parens(arg0_expr.get());
        bool arg1_parens = expression_needs_parens(arg1_expr.get());
        bool arg2_parens = expression_needs_parens(arg2_expr.get());
        if (arg0_parens)
            m_out << "(";
        print_expression(arg0_expr.get(), dot_sep(path_prefix, arg0_name), indent);
        if (arg0_parens)
            m_out << ")";
        m_out << " ? ";
        if (arg1_parens)
            m_out << "(";
        print_expression(arg1_expr.get(), dot_sep(path_prefix, arg1_name), indent);
        if (arg1_parens)
            m_out << ")";
        m_out << " : ";
        if (arg2_parens)
            m_out << "(";
        print_expression(arg2_expr.get(), dot_sep(path_prefix, arg2_name), indent);
        if (arg2_parens)
            m_out << ")";
    } else if ((semantic >= IFunction_definition::DS_UNARY_FIRST) 
            && (semantic < IFunction_definition::DS_CAST)){
        if (suppress_expression(call)) {
            m_out << "...";
            return;
        }
        Handle<IExpression const> arg0_expr(arguments->get_expression(mi::Size(0)));
        char const *arg0_name = arguments->get_name(mi::Size(0));

        char const *operator_symbol = function_name.c_str() + 8; // length of "operator"

        bool arg0_parens = expression_needs_parens(arg0_expr.get());

        if (arg0_parens)
            m_out << "(";
        m_out << operator_symbol;
        print_expression(arg0_expr.get(), dot_sep(path_prefix, arg0_name),
                         indent);
        if (arg0_parens)
            m_out << ")";
    } else if ((semantic >= IFunction_definition::DS_BINARY_FIRST) 
            && (semantic < IFunction_definition::DS_TERNARY)) {
        if (suppress_expression(call)) {
            m_out << "...";
            return;
        }
        Handle<IExpression const> arg0_expr(arguments->get_expression(mi::Size(0)));
        char const *arg0_name = arguments->get_name(mi::Size(0));
        Handle<IExpression const> arg1_expr(arguments->get_expression(1));
        char const *arg1_name = arguments->get_name(1);

        char const *operator_symbol = function_name.c_str() + 8; // length of "operator"

        bool arg0_parens = expression_needs_parens(arg0_expr.get());
        bool arg1_parens = expression_needs_parens(arg1_expr.get());

        if (arg0_parens)
            m_out << "(";
        print_expression(arg0_expr.get(), dot_sep(path_prefix, arg0_name),
                         indent);
        if (arg0_parens)
            m_out << ")";
        m_out << " " << operator_symbol << " ";
        if (arg1_parens)
            m_out << "(";
        print_expression(arg1_expr.get(), dot_sep(path_prefix, arg1_name),
                         indent);
        if (arg1_parens)
            m_out << ")";
    } else {
        if (semantic == IFunction_definition::DS_CONV_OPERATOR) {
            // The int(enum_val) conversion operator is prefixed with
            // the fully qualified name of the enum, but only
            // internally handled. Therefore, we simply drop the
            // prefix and use the "int" operator, which works in this
            // case.
            std::string fname = drop_module_prefix(function_name);
            if (fname == "int") {
                m_out << fname << "(";
            }
            else {
                print_identifier(drop_prefix(function_name));
                m_out << "(";
            }
        } else if (semantic == IFunction_definition::DS_CAST) {
            Handle<const mi::neuraylib::IType> ret_ty(call->get_type());
            m_out << "cast<";
            print_type(ret_ty.get());
            m_out << ">(";
        } else {
            print_identifier(drop_prefix(function_name));
            m_out << "(";
        }

        Handle<IExpression_list const> defaults(function->get_defaults());

        mi::Size argc = arguments->get_size();
        bool param_printed = false;

        if (argc > 1) {
            m_out << "\n";
            ind(indent + 1);
        }

        for (mi::Size i = 0; i < argc; i++) {
            char const *name = arguments->get_name(i);
            Handle<IExpression const> arg_expr(arguments->get_expression(i));
            Handle<IExpression const> arg_def(defaults->get_expression(name));
            char const *arg_name = arguments->get_name(i);

            if (m_suppress_default_parameters
                && name
                && arg_def
                && m_expr_factory->compare(arg_expr.get(), arg_def.get()) == 0) {
                continue;
            }
            if (param_printed) {
                m_out << ",\n";
                ind(indent + 1);
            }

            param_printed = true;

            if (suppress_expression(arg_expr.get())) {
                m_out << "...";
                continue;
            }

            if (semantic != IFunction_definition::DS_CAST) {
                m_out << name << ": ";
            }
            print_expression(arg_expr.get(), dot_sep(path_prefix, arg_name),
                             indent + 1);
        }
        if (argc > 1) {
            m_out << "\n";
            ind(indent);
        }
        m_out << ")";
    }
}

void Mdl_printer::print_expression(IExpression const *expr, std::string const &path_prefix, 
        int indent)
{
    switch (expr->get_kind()) {
    case IExpression::EK_CONSTANT:
    {
        if (suppress_expression(expr)) {
            m_out << "...";
            return;
        }
        Handle<IExpression_constant const> c(expr->get_interface<IExpression_constant const>());
        mdl_assert(c.is_valid_interface());

        Handle<IValue const> val(c->get_value());
        print_value(val.get(), path_prefix);
        break;
    }

    case IExpression::EK_CALL:
        // TODO: add indirect call expression support if needed
        break;

    case IExpression::EK_PARAMETER:
    {
        if (suppress_expression(expr)) {
            m_out << "...";
            return;
        }
        // add parameter support here if class compilation needs to be supported
        Handle<const IExpression_parameter> 
            parameter(expr->get_interface<const IExpression_parameter>());
        mdl_assert(parameter.is_valid_interface());

        mi::Size parameter_idx = parameter->get_index();

        std::string arg_name = clean_identifier(m_material->get_parameter_name(parameter_idx));
#if 0
        Handle< const IValue> argument(
            m_material->get_argument(parameter_idx));
        print_value(argument.get(), dot_sep(path_prefix, arg_name));
#else
        print_identifier(arg_name);
#endif
        break;
    }

    case IExpression::EK_DIRECT_CALL:
    {
        Handle<const IExpression_direct_call> call( 
            expr->get_interface<const IExpression_direct_call>());
        mdl_assert( call.is_valid_interface());

        print_direct_call(call.get(), path_prefix, indent);

        break;
    }

    case IExpression::EK_TEMPORARY:
    {
        if (suppress_expression(expr)) {
            m_out << "...";
            return;
        }
        if (!preserve_temporaries()) {
            Handle<const IExpression_temporary> temp( 
                expr->get_interface<const IExpression_temporary>());
            mdl_assert(temp.is_valid_interface());

            Handle<const IExpression> temp_expr(
                m_material->get_temporary(temp->get_index()));
            print_expression(temp_expr.get(), path_prefix, indent);
        } else {
            Handle<const IExpression_temporary> 
                temp(expr->get_interface<const IExpression_temporary>());
            mdl_assert(temp.is_valid_interface());

            Handle<IExpression const> temp_expr(m_material->get_temporary(temp->get_index()));
            mdl_assert(temp_expr.is_valid_interface());
            if (trivial_expression(temp_expr.get())) {
                print_expression(temp_expr.get(), path_prefix, indent);
            } else {
                m_out << "t" << temp->get_index();
            }
        }
        break;
    }

    default:
        mdl_assert(!"unreachable");
        break;
    }
}

void Mdl_printer::analyze_value(IValue const *value, std::string const &path_prefix)
{
    if (m_do_bake) {
        Handle<IType const> vt(value->get_type());
        bool b = true;
        if (!m_options->all_textures) {
            b = (m_bake_paths->end() != std::find_if(m_bake_paths->begin(), m_bake_paths->end(),
                                                     Bake_path_cmp(path_prefix)));
        }
        if (b) {
            for (auto &r : m_temporary_stack) {
                m_baked_temporaries.insert(r);
            }
        }
    }

    switch (value->get_kind()) {
    case IValue::VK_BOOL:
    case IValue::VK_INT:
        break;

    case IValue::VK_ENUM:
    {
        Handle<IValue_enum const> e(value->get_interface<IValue_enum const>());
        mdl_assert(e.is_valid_interface());

        Handle<const IType_enum> enum_type( e->get_type());
        add_identifier(enum_type->get_symbol());
        break;
    }

    case IValue::VK_FLOAT:
    case IValue::VK_DOUBLE:
    case IValue::VK_STRING:
        break;

    case IValue::VK_VECTOR:
    {
        Handle<IValue_vector const> vec(value->get_interface<IValue_vector const>());
        mdl_assert(vec.is_valid_interface());

        Handle<const IType_vector> vector_type(vec->get_type());
        mi::Size n = vector_type->get_size();
        for ( mi::Size i = 0; i < n; ++i) {
            Handle<const IValue> s(vec->get_value(i));
            analyze_value(s.get(), path_prefix); // FIXME: Correct prefix?
        }
        break;
    }

    case IValue::VK_MATRIX:
    {
        Handle<IValue_matrix const> val_matrix(value->get_interface<IValue_matrix const>());
        mdl_assert(val_matrix.is_valid_interface());

        Handle<const IType_matrix> matrix_type( val_matrix->get_type());
        Handle<const IType_vector> vector_type( matrix_type->get_element_type());
        mi::Size ni = matrix_type->get_size();
        mi::Size nj = vector_type->get_size();
        for ( mi::Size i = 0; i < ni; ++i) {
            Handle<const IValue_vector> vec( val_matrix->get_value(i));
            for ( mi::Size j = 0; j < nj; ++j) {
                Handle<const IValue> s(vec->get_value(j));
                analyze_value(s.get(), path_prefix); // FIXME: Correct prefix?
            }
        }
        break;
    }

    case IValue::VK_COLOR:
    {
        Handle<IValue_color const> d(value->get_interface<IValue_color const>());
        mdl_assert(d.is_valid_interface());

        Handle<IValue_float const> r(d->get_value(0));
        Handle<IValue_float const> g(d->get_value(1));
        Handle<IValue_float const> b(d->get_value(2));
        analyze_value(r.get(), dot_sep(path_prefix, "r"));
        analyze_value(g.get(), dot_sep(path_prefix, "g"));
        analyze_value(b.get(), dot_sep(path_prefix, "b"));
        break;
    }

    case IValue::VK_ARRAY:
    {
        Handle<IValue_array const> array_value(value->get_interface<IValue_array const>());
        mdl_assert(array_value.is_valid_interface());

        Handle<IType_array const> array_type(array_value->get_type());
        Handle<IType const> elem_type(array_type->get_element_type());

        mi::Size n = array_value->get_size();
        for (mi::Size i = 0; i < n; ++i) {
            Handle<const IValue> elem(array_value->get_value(i));
            analyze_value(elem.get(), path_prefix); // FIXME: Correct prefix?
        }
        break;
    }

    case IValue::VK_STRUCT:
    {
        Handle<IValue_struct const> s(value->get_interface<IValue_struct const>());
        mdl_assert(s.is_valid_interface());

        Handle<const IType_struct> struct_type(s->get_type());

        const char* struct_name = struct_type->get_symbol();
        add_identifier(struct_name);

        mi::Size n = s->get_size();
        for (mi::Size i = 0; i < n; ++i) {
            Handle<const IValue> elem(s->get_value(i));
            analyze_value(elem.get(), path_prefix);
        }
        break;
    }

    case IValue::VK_TEXTURE:
    {
        Handle<const IValue_texture> val_texture(value->get_interface<const IValue_texture>());
        mdl_assert(val_texture.is_valid_interface());

        Handle<const mi::neuraylib::ITexture> tex(
            m_transaction->access<mi::neuraylib::ITexture>(val_texture->get_value()));
        if (tex.is_valid_interface()) {
            // Force ::tex module to be imported.
            add_identifier( "::tex::gamma_linear");
        }
        break;
    }

    case IValue::VK_INVALID_DF:
    case IValue::VK_LIGHT_PROFILE:
    case IValue::VK_BSDF_MEASUREMENT:
        break;

    default:
        mdl_assert(!"unsupported variant");
        break;
    }
}

void Mdl_printer::analyze_expression(IExpression const *expr, std::string const &path_prefix)
{
    switch (expr->get_kind()) {
    case IExpression::EK_CONSTANT:
    {
        Handle<IExpression_constant const> c(expr->get_interface<IExpression_constant const>());
        mdl_assert(c.is_valid_interface());
        Handle<IValue const> val(c->get_value());
        analyze_value(val.get(), path_prefix);
        break;
    }

    case IExpression::EK_CALL:
    {
        // TODO: add indirect call expression support if needed
        error("material '" + m_material_name + "' contains indirect call.");
        break;
    }

    case IExpression::EK_PARAMETER:
    {
        // add parameter support here if class compilation needs to be supported
        Handle<const IExpression_parameter> 
            parameter(expr->get_interface<const IExpression_parameter>());
        mdl_assert(parameter.is_valid_interface());

        mi::Size parameter_idx = parameter->get_index();
        const char* pname = m_material->get_parameter_name(parameter_idx);
        if (m_used_parameters.find(pname) == m_used_parameters.end()) {
            m_used_parameters.insert(pname);

            if (get_warn_on_parameters()) {
                warning("material '" + m_material_name + "' contains the parameter '"
                        + (pname ? pname : "<unnamed>")
                        + "', treating material as instance and exporting value.");
            }
        }

        break;
    }

    case IExpression::EK_DIRECT_CALL:
    {
        Handle<const IExpression_direct_call> call( 
            expr->get_interface<const IExpression_direct_call>());
        mdl_assert(call.is_valid_interface());
        Handle<const IExpression_list> arguments(call->get_arguments());

        Handle<const IFunction_definition> function(
            m_transaction->access<IFunction_definition>(call->get_definition()));
        std::string function_name = drop_signature( function->get_mdl_name());

        Handle<const IString> s(m_mdl_factory->decode_name( function_name.c_str()));
        function_name = s->get_c_str();

        if ( function_name.substr(0,30) == std::string("::nvidia::distilling_support::")) {
            function_name = m_options->dist_supp_file + function_name.substr(28);
        }

        check_mdl_version_requirement(function_name, call.get(),
                                      arguments.get());

        add_identifier(function_name.c_str());

        mi::Size argc = arguments->get_size();
        for (mi::Size i = 0; i < argc; i++) {
            Handle<IExpression const> arg_expr(arguments->get_expression(i));
            char const *arg_name = arguments->get_name(i);
            analyze_expression(arg_expr.get(), dot_sep(path_prefix, arg_name));
        }
        break;
    }

    case IExpression::EK_TEMPORARY:
    {
        Handle<const IExpression_temporary> 
            temp(expr->get_interface<const IExpression_temporary>());
        mdl_assert(temp.is_valid_interface());

        Handle<IExpression const> temp_expr(m_material->get_temporary(temp->get_index()));

        m_temporary_stack.push_back(temp->get_index());

        analyze_expression(temp_expr.get(), path_prefix);

        m_temporary_stack.pop_back();

        break;
    }

    default:
        mdl_assert(!"unreachable");
        break;
    }
}

static std::string mdl_1_8_functions[] =
{
    "::df::dusty_diffuse_reflection_bsdf",
    "::df::fog_vdf",
};

static std::string mdl_1_7_functions[] =
{
    "::df::unbounded_mix",
    "::df::color_unbounded_mix",
    "::df::color_normalized_mix",
    "::df::color_edf_component",
    "::df::color_vdf_component"
};

static std::string mdl_1_6_functions[] =
{
    "::df::measured_factor",    // MDL 1.5, handled with 1.6.
    "::df::chiang_hair_bsdf",   // MDL 1.5, handled with 1.6.
    "operator_cast"             // MDL 1.5, handled with 1.6.
};

void Mdl_printer::check_mdl_version_requirement(
    std::string const &function_name,
    IExpression_direct_call const *call,
    IExpression_list const *arguments)
{
    if (is_unicode_identifier(function_name)) {
            set_min_mdl_version(mdl_spec_1_8);
            return;
    }
    for (auto &fn : mdl_1_8_functions) {
        if (function_name == fn) {
            set_min_mdl_version(mdl_spec_1_8);
            return;
        }
    }

    for (auto &fn : mdl_1_7_functions) {
        if (function_name == fn) {
            set_min_mdl_version(mdl_spec_1_7);
            return;
        }
    }

    for (auto &fn : mdl_1_6_functions) {
        if (function_name == fn) {
            set_min_mdl_version(mdl_spec_1_6);
            return;
        }
    }

    if (function_name == "::df::tint") {
        if (arguments->get_size() > 2) {
            set_min_mdl_version(mdl_spec_1_6);
            return;
        }
        Handle<IExpression_list const> args(call->get_arguments());
        Handle<IExpression const> df_arg(args->get_expression(1));
        Handle<IType const> df_t(df_arg->get_type());
        if (df_t->get_kind() == IType::TK_VDF) {
            set_min_mdl_version(mdl_spec_1_7);
            return;
        }
        if (df_t->get_kind() == IType::TK_HAIR_BSDF) {
            set_min_mdl_version(mdl_spec_1_6);
            return;
        }
    }

    if (function_name == "::df::directional_factor") {
        Handle<IType const> ret_t(call->get_type());
        if (ret_t->get_kind() == IType::TK_EDF) {
            set_min_mdl_version(mdl_spec_1_7);
            return;
        }
    }

    if (function_name == "material_volume") {
        if (arguments->get_size() > 3) {
            set_min_mdl_version(mdl_spec_1_7);
            return;
        }
    }
}

void Mdl_printer::set_min_mdl_version(mdl_spec version) {
    if (m_required_mdl_version < version) {
        m_required_mdl_version = version;
    }
}

bool Mdl_printer::preserve_temporaries()
{
    return !m_unfold_temps && !m_do_bake && !m_outline_mode;
}

void Mdl_printer::analyze_material()
{
    Handle<const IExpression_direct_call> body(m_material->get_body());

    mi::Size param_count = m_material->get_parameter_count();
   
    for (mi::Size i = 0; i < param_count; i++) {
        Handle <IValue const> param_value(m_material->get_argument(i));
        analyze_value(param_value.get(), "");
    }
    
    if (preserve_temporaries()) {
        mi::Size temp_count = m_material->get_temporary_count();
        if (temp_count > 0) {
            for (mi::Size i = 0; i < temp_count; i++) {
                Handle<IExpression const> temp(m_material->get_temporary(i));
                analyze_expression(temp.get(), "");
            }
        }
    }

    analyze_expression(body.get(), "");

    for (auto &id : m_used_identifiers) {
        if (id.substr(0, 2) == "::") {
            auto mod = get_module_from_qualified_name(id);
            if (mod.size() > 0)
                m_used_modules.insert(mod);
        }
    }
}

void Mdl_printer::print_prolog()
{
    switch (m_required_mdl_version) {
    case mdl_spec_1_3:
        m_out << "mdl 1.3;\n";
        break;

    case mdl_spec_1_6:
        m_out << "mdl 1.6;\n";
        break;

    case mdl_spec_1_7:
        m_out << "mdl 1.7;\n";
        break;

    case mdl_spec_1_8:
        m_out << "mdl 1.8;\n";
        break;

    case mdl_spec_1_9:
        m_out << "mdl 1.9;\n";
        break;

    default:
        mdl_assert(!"minimum MDL spec version not determined");
        m_out << "mdl 1.7;\n";
        break;
    }

    m_out << "\n\n// Imports\n";

    for (auto &mdl : m_used_modules) {
        m_out << "import " << mdl << "::*;\n";
    }
    m_out << "\n";

}

void Mdl_printer::print_epilog()
{
#if 0
    m_out << "// referenced identifiers\n";
    for (auto &id : m_used_identifiers) {
        m_out << "//   " << id << "\n";
    }
    m_out << "// referenced parameters\n";
    for (auto &id : m_used_parameters) {
        m_out << "//   " << id << "\n";
    }
#endif
}

void Mdl_printer::print_material()
{
    Handle<const IExpression_direct_call> body(m_material->get_body());
    Handle<const IExpression_list> arguments(body->get_arguments());

    m_out << "export material ";
    print_identifier(m_print_material_name);
    m_out << "(";

    mi::Size param_count = m_material->get_parameter_count();
    if (param_count == 0) {
        m_out << ") = ";
    } else {
        for (mi::Size i = 0; i < param_count; i++) {
            std::string param_name = clean_identifier(m_material->get_parameter_name(i));
            Handle <IValue const> param_value(m_material->get_argument(i));
            Handle <IType const> param_type(param_value->get_type());

            if (i > 0)
                m_out << ",\n";
            else
                m_out << "\n";
            m_out << "    uniform ";
            print_type(param_type.get());
            m_out << " " << param_name << " = ";
            print_value(param_value.get(), "");
        }
        m_out << "\n  ) = ";
    }

    if (preserve_temporaries()) {
        mi::Size temp_count = m_material->get_temporary_count();
        mi::Size non_triv_temp_count = 0;

        for (mi::Size i = 0; i < temp_count; i++) {
            Handle<IExpression const> temp(m_material->get_temporary(i));
            if (!trivial_expression(temp.get()))
                non_triv_temp_count++;
        }
        if (non_triv_temp_count > 0) {
            m_out << "let {\n";
            for (mi::Size i = 0; i < temp_count; i++) {
                Handle<IExpression const> temp(m_material->get_temporary(i));
                if (!trivial_expression(temp.get())) {
                    Handle<IType const> temp_type(temp->get_type());
                    ind(1);
                    print_type(temp_type.get());
                    m_out << " t" << i << " = ";
                    print_expression(temp.get(), "", 2); // FIXME: Correct prefix value?
                    m_out << ";\n";
                }
            }
            m_out << "} in\n";
        }
        ind(1);
    }
    print_expression(body.get(), "", 1);
    m_out << ";\n";
}

void Mdl_printer::add_identifier(char const *identifier)
{
    m_used_identifiers.insert(identifier);
}

void Mdl_printer::print_module()
{
    m_used_modules.insert("::base"); // ::base is always imported.
    m_used_modules.insert(m_options->dist_supp_file);

    analyze_material();

    print_prolog();
    print_material();
    print_epilog();

    if (m_emit_messages && m_messages.size() > 0) {
        for (auto &msg : m_messages) {
            std::cerr << msg << "\n";
        }
        if (m_messages.size() - m_error_count > 0) {
            std::cerr << (m_messages.size() - m_error_count) << " warning(s)\n";
        }
        if (m_error_count > 0) {
            std::cerr << m_error_count << " error(s)\n";
        }
    }
}
