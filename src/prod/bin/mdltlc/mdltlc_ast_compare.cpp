/******************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
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

#include "pch.h"

#include <cstring>
#include <string>

#include "mdltlc_ast_compare.h"
#include "mdltlc_values.h"

namespace {

static char const *safe(char const *s)
{
    return s ? s : "<null>";
}

static std::string str(unsigned value)
{
    return std::to_string(value);
}

static std::string str(size_t value)
{
    return std::to_string(value);
}

static bool same_string(char const *a, char const *b)
{
    if (a == nullptr || b == nullptr) {
        return a == b;
    }
    return std::strcmp(a, b) == 0;
}

static void mismatch(
    Mdltlc_ast_compare_result &result,
    std::string const         &path,
    std::string const         &expected,
    std::string const         &actual,
    Location const            &loc)
{
    if (!result.equal) {
        return;
    }
    result.equal = false;
    result.path = path;
    result.expected = expected;
    result.actual = actual;
    result.line = loc.get_line();
    result.column = loc.get_column();
}

static bool compare_location(
    Location const            &expected,
    Location const            &actual,
    std::string const         &path,
    Mdltlc_ast_compare_result &result)
{
    if (expected.get_file_id() == actual.get_file_id() &&
        expected.get_line() == actual.get_line() &&
        expected.get_column() == actual.get_column()) {
        return true;
    }

    mismatch(
        result,
        path + ".location",
        "line " + str(expected.get_line()) + ", column " + str(expected.get_column()),
        "line " + str(actual.get_line()) + ", column " + str(actual.get_column()),
        actual);
    return false;
}

static char const *expr_kind_name(Expr::Kind kind)
{
    switch (kind) {
    case Expr::EK_INVALID:         return "invalid";
    case Expr::EK_LITERAL:         return "literal";
    case Expr::EK_REFERENCE:       return "reference";
    case Expr::EK_UNARY:           return "unary";
    case Expr::EK_BINARY:          return "binary";
    case Expr::EK_CONDITIONAL:     return "conditional";
    case Expr::EK_CALL:            return "call";
    case Expr::EK_TYPE_ANNOTATION: return "type_annotation";
    case Expr::EK_ATTRIBUTE:       return "attribute";
    }
    return "<unknown>";
}

static char const *value_kind_name(Value::Kind kind)
{
    switch (kind) {
    case Value::VK_BOOL:   return "bool";
    case Value::VK_INT:    return "int";
    case Value::VK_FLOAT:  return "float";
    case Value::VK_STRING: return "string";
    }
    return "<unknown>";
}

static char const *type_kind_name(Type::Kind kind)
{
    switch (kind) {
    case Type::TK_BOOL:              return "bool";
    case Type::TK_INT:               return "int";
    case Type::TK_ENUM:              return "enum";
    case Type::TK_FLOAT:             return "float";
    case Type::TK_DOUBLE:            return "double";
    case Type::TK_STRING:            return "string";
    case Type::TK_LIGHT_PROFILE:     return "light_profile";
    case Type::TK_BSDF:              return "bsdf";
    case Type::TK_HAIR_BSDF:         return "hair_bsdf";
    case Type::TK_EDF:               return "edf";
    case Type::TK_VDF:               return "vdf";
    case Type::TK_VECTOR:            return "vector";
    case Type::TK_MATRIX:            return "matrix";
    case Type::TK_ARRAY:             return "array";
    case Type::TK_COLOR:             return "color";
    case Type::TK_FUNCTION:          return "function";
    case Type::TK_STRUCT:            return "struct";
    case Type::TK_TEXTURE:           return "texture";
    case Type::TK_BSDF_MEASUREMENT:  return "bsdf_measurement";
    case Type::TK_MATERIAL_EMISSION: return "material_emission";
    case Type::TK_MATERIAL_SURFACE:  return "material_surface";
    case Type::TK_MATERIAL_VOLUME:   return "material_volume";
    case Type::TK_MATERIAL_GEOMETRY: return "material_geometry";
    case Type::TK_MATERIAL:          return "material";
    case Type::TK_VAR:               return "type_variable";
    case Type::TK_ERROR:             return "error";
    }
    return "<unknown>";
}

static std::string type_summary(Type const *type)
{
    if (type == nullptr) {
        return "<null>";
    }

    if (Type_var const *tv = as<Type_var>(type)) {
        if (!tv->is_bound()) {
            return "type_variable";
        }
        type = tv->get_type();
        if (type == nullptr) {
            return "type_variable";
        }
    }

    if (Type_struct const *ts = as<Type_struct>(type)) {
        return std::string("struct ") + safe(ts->get_name()->get_name());
    }
    if (Type_enum const *te = as<Type_enum>(type)) {
        return std::string("enum ") + safe(te->get_name()->get_name());
    }
    return type_kind_name(type->get_kind());
}

static bool compare_type(
    Type const                 *expected,
    Type const                 *actual,
    std::string const         &path,
    Mdltlc_ast_compare_result &result,
    Location const            &loc)
{
    if (expected == nullptr || actual == nullptr) {
        if (expected == actual) {
            return true;
        }
        mismatch(result, path, type_summary(expected), type_summary(actual), loc);
        return false;
    }

    if (expected->get_kind() == Type::TK_VAR && actual->get_kind() == Type::TK_VAR) {
        return true;
    }

    if (expected->get_kind() != actual->get_kind()) {
        mismatch(result, path, type_summary(expected), type_summary(actual), loc);
        return false;
    }

    if (Type_struct const *expected_struct = as<Type_struct>(expected)) {
        Type_struct const *actual_struct = cast<Type_struct>(actual);
        if (!same_string(
                expected_struct->get_name()->get_name(),
                actual_struct->get_name()->get_name())) {
            mismatch(result, path, type_summary(expected), type_summary(actual), loc);
            return false;
        }
    }
    if (Type_enum const *expected_enum = as<Type_enum>(expected)) {
        Type_enum const *actual_enum = cast<Type_enum>(actual);
        if (!same_string(
                expected_enum->get_name()->get_name(),
                actual_enum->get_name()->get_name())) {
            mismatch(result, path, type_summary(expected), type_summary(actual), loc);
            return false;
        }
    }
    if (Type_vector const *expected_vector = as<Type_vector>(expected)) {
        Type_vector const *actual_vector = cast<Type_vector>(actual);
        if (expected_vector->get_size() != actual_vector->get_size()) {
            mismatch(result, path, type_summary(expected), type_summary(actual), loc);
            return false;
        }
        return compare_type(
            expected_vector->get_element_type(),
            actual_vector->get_element_type(),
            path + ".element_type",
            result,
            loc);
    }
    if (Type_matrix const *expected_matrix = as<Type_matrix>(expected)) {
        Type_matrix const *actual_matrix = cast<Type_matrix>(actual);
        if (expected_matrix->get_column_count() != actual_matrix->get_column_count()) {
            mismatch(result, path, type_summary(expected), type_summary(actual), loc);
            return false;
        }
        return compare_type(
            expected_matrix->get_element_type(),
            actual_matrix->get_element_type(),
            path + ".element_type",
            result,
            loc);
    }
    if (Type_array const *expected_array = as<Type_array>(expected)) {
        Type_array const *actual_array = cast<Type_array>(actual);
        return compare_type(
            expected_array->get_element_type(),
            actual_array->get_element_type(),
            path + ".element_type",
            result,
            loc);
    }
    return true;
}

static std::string value_summary(Value const *value)
{
    if (value == nullptr) {
        return "<null>";
    }

    switch (value->get_kind()) {
    case Value::VK_BOOL:
        return cast<Value_bool>(value)->get_value() ? "true" : "false";
    case Value::VK_INT:
        return str(unsigned(cast<Value_int>(value)->get_value()));
    case Value::VK_FLOAT:
        return safe(cast<Value_float>(value)->get_s_value());
    case Value::VK_STRING:
        return safe(cast<Value_string>(value)->get_value());
    }
    return "<unknown>";
}

static bool compare_value(
    Value const                *expected,
    Value const                *actual,
    std::string const         &path,
    Mdltlc_ast_compare_result &result,
    Location const            &loc)
{
    if (expected == nullptr || actual == nullptr) {
        if (expected == actual) {
            return true;
        }
        mismatch(result, path, value_summary(expected), value_summary(actual), loc);
        return false;
    }

    if (expected->get_kind() != actual->get_kind()) {
        mismatch(
            result,
            path + ".kind",
            value_kind_name(expected->get_kind()),
            value_kind_name(actual->get_kind()),
            loc);
        return false;
    }

    switch (expected->get_kind()) {
    case Value::VK_BOOL:
        if (cast<Value_bool>(expected)->get_value() != cast<Value_bool>(actual)->get_value()) {
            mismatch(result, path, value_summary(expected), value_summary(actual), loc);
            return false;
        }
        break;
    case Value::VK_INT:
        if (cast<Value_int>(expected)->get_value() != cast<Value_int>(actual)->get_value()) {
            mismatch(result, path, value_summary(expected), value_summary(actual), loc);
            return false;
        }
        break;
    case Value::VK_FLOAT:
        if (!same_string(
                cast<Value_float>(expected)->get_s_value(),
                cast<Value_float>(actual)->get_s_value())) {
            mismatch(result, path, value_summary(expected), value_summary(actual), loc);
            return false;
        }
        break;
    case Value::VK_STRING:
        if (!same_string(
                cast<Value_string>(expected)->get_value(),
                cast<Value_string>(actual)->get_value())) {
            mismatch(result, path, value_summary(expected), value_summary(actual), loc);
            return false;
        }
        break;
    }
    return true;
}

static bool compare_expr(
    Expr const                 *expected,
    Expr const                 *actual,
    std::string const         &path,
    Mdltlc_ast_compare_result &result);

static bool compare_argument_list(
    Argument_list const       &expected,
    Argument_list const       &actual,
    std::string const         &path,
    Mdltlc_ast_compare_result &result)
{
    Argument_list::const_iterator expected_it(expected.begin());
    Argument_list::const_iterator actual_it(actual.begin());
    Argument_list::const_iterator expected_end(expected.end());
    Argument_list::const_iterator actual_end(actual.end());

    size_t index = 0;
    for (; expected_it != expected_end && actual_it != actual_end; ++expected_it, ++actual_it, ++index) {
        if (!compare_expr(
                expected_it->get_expr(),
                actual_it->get_expr(),
                path + "[" + str(index) + "]",
                result)) {
            return false;
        }
    }

    if (expected_it != expected_end || actual_it != actual_end) {
        mismatch(
            result,
            path + ".count",
            expected_it == expected_end ? str(index) : "more than " + str(index),
            actual_it == actual_end ? str(index) : "more than " + str(index),
            Location(Location::OWNER_FILE_IDX, 1, 1));
        return false;
    }
    return true;
}

static bool compare_expr(
    Expr const                 *expected,
    Expr const                 *actual,
    std::string const         &path,
    Mdltlc_ast_compare_result &result)
{
    if (expected == nullptr || actual == nullptr) {
        if (expected == actual) {
            return true;
        }
        mismatch(result, path, expected ? "expression" : "<null>", actual ? "expression" : "<null>",
            actual ? actual->get_location() : Location(Location::OWNER_FILE_IDX, 1, 1));
        return false;
    }

    if (expected->get_kind() != actual->get_kind()) {
        mismatch(
            result,
            path + ".kind",
            expr_kind_name(expected->get_kind()),
            expr_kind_name(actual->get_kind()),
            actual->get_location());
        return false;
    }
    if (!compare_location(expected->get_location(), actual->get_location(), path, result)) {
        return false;
    }
    if (expected->in_parenthesis() != actual->in_parenthesis()) {
        mismatch(
            result,
            path + ".parenthesis",
            expected->in_parenthesis() ? "true" : "false",
            actual->in_parenthesis() ? "true" : "false",
            actual->get_location());
        return false;
    }
    if (!compare_type(
            expected->get_type(),
            actual->get_type(),
            path + ".type",
            result,
            actual->get_location())) {
        return false;
    }

    switch (expected->get_kind()) {
    case Expr::EK_INVALID:
        return true;

    case Expr::EK_LITERAL:
        return compare_value(
            cast<Expr_literal>(expected)->get_value(),
            cast<Expr_literal>(actual)->get_value(),
            path + ".value",
            result,
            actual->get_location());

    case Expr::EK_REFERENCE:
    {
        Expr_ref const *expected_ref = cast<Expr_ref>(expected);
        Expr_ref const *actual_ref = cast<Expr_ref>(actual);
        if (!same_string(expected_ref->get_name()->get_name(), actual_ref->get_name()->get_name())) {
            mismatch(
                result,
                path + ".name",
                safe(expected_ref->get_name()->get_name()),
                safe(actual_ref->get_name()->get_name()),
                actual->get_location());
            return false;
        }
        return true;
    }

    case Expr::EK_TYPE_ANNOTATION:
    {
        Expr_type_annotation const *expected_annot = cast<Expr_type_annotation>(expected);
        Expr_type_annotation const *actual_annot = cast<Expr_type_annotation>(actual);
        if (!same_string(
                expected_annot->get_type_name()->get_name(),
                actual_annot->get_type_name()->get_name())) {
            mismatch(
                result,
                path + ".type_name",
                safe(expected_annot->get_type_name()->get_name()),
                safe(actual_annot->get_type_name()->get_name()),
                actual->get_location());
            return false;
        }
        return compare_expr(
            expected_annot->get_argument(),
            actual_annot->get_argument(),
            path + ".arg",
            result);
    }

    case Expr::EK_UNARY:
    {
        Expr_unary const *expected_unary = cast<Expr_unary>(expected);
        Expr_unary const *actual_unary = cast<Expr_unary>(actual);
        if (expected_unary->get_operator() != actual_unary->get_operator()) {
            mismatch(
                result,
                path + ".operator",
                str(unsigned(expected_unary->get_operator())),
                str(unsigned(actual_unary->get_operator())),
                actual->get_location());
            return false;
        }
        return compare_expr(
            expected_unary->get_argument(),
            actual_unary->get_argument(),
            path + ".arg",
            result);
    }

    case Expr::EK_BINARY:
    {
        Expr_binary const *expected_binary = cast<Expr_binary>(expected);
        Expr_binary const *actual_binary = cast<Expr_binary>(actual);
        if (expected_binary->get_operator() != actual_binary->get_operator()) {
            mismatch(
                result,
                path + ".operator",
                str(unsigned(expected_binary->get_operator())),
                str(unsigned(actual_binary->get_operator())),
                actual->get_location());
            return false;
        }
        return compare_expr(
                expected_binary->get_left_argument(),
                actual_binary->get_left_argument(),
                path + ".lhs",
                result) &&
            compare_expr(
                expected_binary->get_right_argument(),
                actual_binary->get_right_argument(),
                path + ".rhs",
                result);
    }

    case Expr::EK_CONDITIONAL:
    {
        Expr_conditional const *expected_cond = cast<Expr_conditional>(expected);
        Expr_conditional const *actual_cond = cast<Expr_conditional>(actual);
        return compare_expr(
                expected_cond->get_condition(),
                actual_cond->get_condition(),
                path + ".condition",
                result) &&
            compare_expr(
                expected_cond->get_true(),
                actual_cond->get_true(),
                path + ".true",
                result) &&
            compare_expr(
                expected_cond->get_false(),
                actual_cond->get_false(),
                path + ".false",
                result);
    }

    case Expr::EK_CALL:
    {
        Expr_call const *expected_call = cast<Expr_call>(expected);
        Expr_call const *actual_call = cast<Expr_call>(actual);
        if (expected_call->get_argument_count() != actual_call->get_argument_count()) {
            mismatch(
                result,
                path + ".arg_count",
                str(expected_call->get_argument_count()),
                str(actual_call->get_argument_count()),
                actual->get_location());
            return false;
        }
        if (!compare_expr(
                expected_call->get_callee(),
                actual_call->get_callee(),
                path + ".callee",
                result)) {
            return false;
        }
        for (size_t i = 0; i < expected_call->get_argument_count(); ++i) {
            if (!compare_expr(
                    expected_call->get_argument(i),
                    actual_call->get_argument(i),
                    path + ".arg[" + str(i) + "]",
                    result)) {
                return false;
            }
        }
        return true;
    }

    case Expr::EK_ATTRIBUTE:
    {
        Expr_attribute const *expected_attr = cast<Expr_attribute>(expected);
        Expr_attribute const *actual_attr = cast<Expr_attribute>(actual);
        if (!same_string(expected_attr->get_node_name(), actual_attr->get_node_name())) {
            mismatch(
                result,
                path + ".node_name",
                safe(expected_attr->get_node_name()),
                safe(actual_attr->get_node_name()),
                actual->get_location());
            return false;
        }
        if (!compare_expr(
                expected_attr->get_argument(),
                actual_attr->get_argument(),
                path + ".arg",
                result)) {
            return false;
        }

        Expr_attribute::Expr_attribute_vector const &expected_entries =
            expected_attr->get_attributes();
        Expr_attribute::Expr_attribute_vector const &actual_entries =
            actual_attr->get_attributes();
        if (expected_entries.size() != actual_entries.size()) {
            mismatch(
                result,
                path + ".attr_count",
                str(expected_entries.size()),
                str(actual_entries.size()),
                actual->get_location());
            return false;
        }
        for (size_t i = 0; i < expected_entries.size(); ++i) {
            Expr_attribute::Expr_attribute_entry const &expected_entry = expected_entries[i];
            Expr_attribute::Expr_attribute_entry const &actual_entry = actual_entries[i];
            std::string entry_path = path + ".attr[" + str(i) + "]";
            if (!same_string(expected_entry.name->get_name(), actual_entry.name->get_name())) {
                mismatch(
                    result,
                    entry_path + ".name",
                    safe(expected_entry.name->get_name()),
                    safe(actual_entry.name->get_name()),
                    actual->get_location());
                return false;
            }
            if (expected_entry.is_pattern != actual_entry.is_pattern) {
                mismatch(
                    result,
                    entry_path + ".is_pattern",
                    expected_entry.is_pattern ? "true" : "false",
                    actual_entry.is_pattern ? "true" : "false",
                    actual->get_location());
                return false;
            }
            if (!compare_type(
                    expected_entry.type,
                    actual_entry.type,
                    entry_path + ".type",
                    result,
                    actual->get_location())) {
                return false;
            }
            if (!compare_expr(
                    expected_entry.expr,
                    actual_entry.expr,
                    entry_path + ".expr",
                    result)) {
                return false;
            }
        }
        return true;
    }
    }

    return true;
}

template <typename List>
static size_t list_size(List const &list)
{
    size_t size = 0;
    for (typename List::const_iterator it(list.begin()), end(list.end()); it != end; ++it) {
        ++size;
    }
    return size;
}

static bool compare_imports(
    Import_list const          &expected,
    Import_list const          &actual,
    std::string const          &path,
    Mdltlc_ast_compare_result  &result)
{
    if (list_size(expected) != list_size(actual)) {
        mismatch(
            result,
            path + ".import_count",
            str(list_size(expected)),
            str(list_size(actual)),
            Location(Location::OWNER_FILE_IDX, 1, 1));
        return false;
    }

    Import_list::const_iterator expected_it(expected.begin());
    Import_list::const_iterator actual_it(actual.begin());
    for (size_t i = 0; expected_it != expected.end(); ++expected_it, ++actual_it, ++i) {
        std::string item_path = path + ".import[" + str(i) + "]";
        if (!compare_location(
                expected_it->get_location(),
                actual_it->get_location(),
                item_path,
                result)) {
            return false;
        }
        if (!same_string(expected_it->get_name(), actual_it->get_name())) {
            mismatch(
                result,
                item_path + ".name",
                safe(expected_it->get_name()),
                safe(actual_it->get_name()),
                actual_it->get_location());
            return false;
        }
    }
    return true;
}

static bool compare_debug_outs(
    Debug_out_list const       &expected,
    Debug_out_list const       &actual,
    std::string const          &path,
    Mdltlc_ast_compare_result  &result)
{
    if (list_size(expected) != list_size(actual)) {
        mismatch(
            result,
            path + ".debug_count",
            str(list_size(expected)),
            str(list_size(actual)),
            Location(Location::OWNER_FILE_IDX, 1, 1));
        return false;
    }

    Debug_out_list::const_iterator expected_it(expected.begin());
    Debug_out_list::const_iterator actual_it(actual.begin());
    for (size_t i = 0; expected_it != expected.end(); ++expected_it, ++actual_it, ++i) {
        std::string item_path = path + ".debug[" + str(i) + "]";
        if (!compare_location(
                expected_it->get_location(),
                actual_it->get_location(),
                item_path,
                result)) {
            return false;
        }
        if (!same_string(expected_it->get_name(), actual_it->get_name())) {
            mismatch(
                result,
                item_path + ".name",
                safe(expected_it->get_name()),
                safe(actual_it->get_name()),
                actual_it->get_location());
            return false;
        }
    }
    return true;
}

static bool compare_postcond(
    Postcond const             &expected,
    Postcond const             &actual,
    std::string const          &path,
    Mdltlc_ast_compare_result  &result)
{
    if (expected.is_empty() != actual.is_empty()) {
        mismatch(
            result,
            path + ".empty",
            expected.is_empty() ? "true" : "false",
            actual.is_empty() ? "true" : "false",
            Location(Location::OWNER_FILE_IDX, 1, 1));
        return false;
    }
    if (expected.is_empty()) {
        return true;
    }
    return compare_expr(expected.get_expr(), actual.get_expr(), path + ".expr", result);
}

static bool compare_rules(
    Rule_list const            &expected,
    Rule_list const            &actual,
    std::string const          &path,
    Mdltlc_ast_compare_result  &result)
{
    if (list_size(expected) != list_size(actual)) {
        mismatch(
            result,
            path + ".rule_count",
            str(list_size(expected)),
            str(list_size(actual)),
            Location(Location::OWNER_FILE_IDX, 1, 1));
        return false;
    }

    Rule_list::const_iterator expected_it(expected.begin());
    Rule_list::const_iterator actual_it(actual.begin());
    for (size_t i = 0; expected_it != expected.end(); ++expected_it, ++actual_it, ++i) {
        std::string rule_path = path + ".rule[" + str(i) + "]";
        if (!compare_location(
                expected_it->get_location(),
                actual_it->get_location(),
                rule_path,
                result)) {
            return false;
        }
        if (!same_string(expected_it->get_rule_name(), actual_it->get_rule_name())) {
            mismatch(
                result,
                rule_path + ".name",
                safe(expected_it->get_rule_name()),
                safe(actual_it->get_rule_name()),
                actual_it->get_location());
            return false;
        }
        if (expected_it->get_result_code() != actual_it->get_result_code()) {
            mismatch(
                result,
                rule_path + ".result_code",
                str(unsigned(expected_it->get_result_code())),
                str(unsigned(actual_it->get_result_code())),
                actual_it->get_location());
            return false;
        }
        if (expected_it->get_dead_rule() != actual_it->get_dead_rule()) {
            mismatch(
                result,
                rule_path + ".dead_rule",
                str(unsigned(expected_it->get_dead_rule())),
                str(unsigned(actual_it->get_dead_rule())),
                actual_it->get_location());
            return false;
        }
        if (!compare_expr(expected_it->get_lhs(), actual_it->get_lhs(), rule_path + ".lhs", result) ||
            !compare_expr(expected_it->get_rhs(), actual_it->get_rhs(), rule_path + ".rhs", result) ||
            !compare_expr(expected_it->get_guard(), actual_it->get_guard(), rule_path + ".guard", result) ||
            !compare_argument_list(
                expected_it->get_bindings(),
                actual_it->get_bindings(),
                rule_path + ".where",
                result) ||
            !compare_debug_outs(
                expected_it->get_debug_out(),
                actual_it->get_debug_out(),
                rule_path,
                result)) {
            return false;
        }
    }
    return true;
}

static bool compare_ruleset(
    Ruleset const             &expected,
    Ruleset const             &actual,
    std::string const         &path,
    Mdltlc_ast_compare_result &result)
{
    if (!compare_location(expected.get_location(), actual.get_location(), path, result)) {
        return false;
    }
    if (!same_string(expected.get_name(), actual.get_name())) {
        mismatch(result, path + ".name", safe(expected.get_name()), safe(actual.get_name()),
            actual.get_location());
        return false;
    }
    if (expected.get_strategy() != actual.get_strategy()) {
        mismatch(
            result,
            path + ".strategy",
            str(unsigned(expected.get_strategy())),
            str(unsigned(actual.get_strategy())),
            actual.get_location());
        return false;
    }
    return compare_imports(expected.get_imports(), actual.get_imports(), path, result) &&
        compare_rules(expected.get_rules(), actual.get_rules(), path, result) &&
        compare_postcond(expected.get_postcond(), actual.get_postcond(), path + ".postcond", result);
}

}  // namespace

std::string Mdltlc_ast_compare_result::message() const
{
    std::string msg = "hand-written MDLTL parser AST mismatch at ";
    msg += path.empty() ? "<root>" : path;
    msg += ": expected ";
    msg += expected;
    msg += ", actual ";
    msg += actual;
    return msg;
}

bool mdltlc_compare_asts(
    Ruleset_list const        &expected,
    Ruleset_list const        &actual,
    Mdltlc_ast_compare_result &result)
{
    result = Mdltlc_ast_compare_result();

    if (list_size(expected) != list_size(actual)) {
        mismatch(
            result,
            "ruleset_count",
            str(list_size(expected)),
            str(list_size(actual)),
            Location(Location::OWNER_FILE_IDX, 1, 1));
        return false;
    }

    Ruleset_list::const_iterator expected_it(expected.begin());
    Ruleset_list::const_iterator actual_it(actual.begin());
    for (size_t i = 0; expected_it != expected.end(); ++expected_it, ++actual_it, ++i) {
        if (!compare_ruleset(*expected_it, *actual_it, "ruleset[" + str(i) + "]", result)) {
            return false;
        }
    }
    return true;
}
