/******************************************************************************
 * Copyright (c) 2012-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "pch.h"

#include <cstring>

#include <mi/base/handle.h>

#include <mi/mdl/mdl_modules.h>
#include <base/system/stlext/i_stlext_restore.h>

#include "generator_dag_code_printer.h"
#include "generator_dag_tools.h"

namespace mi {
namespace mdl {

namespace {

/// RAII-helper for indentation
class Indent_scope {
public:
    /// Constructor.
    Indent_scope(int &depth)
    : m_depth(depth)
    {
        ++depth;
    }

    /// Destructor.
    ~Indent_scope() { --m_depth; }

private:
    int &m_depth;
};

static bool need_vertical_params(
    IGenerated_code_dag const *code_dag,
    size_t                    idx,
    size_t                    n_params)
{
    if (n_params > 3) {
        // more than three parameters
        return true;
    }
    if (idx & 1) {
        idx = idx >> 1;
        for (size_t i = 0; i < n_params; ++i) {
             if (code_dag->get_function_parameter_annotation_count(idx, i) > 0) {
                 // has annotations
                 return true;
             }
             if (code_dag->get_function_parameter_default(idx, i) != NULL) {
                 // has an default argument
                 return true;
             }
        }
    } else {
        idx = idx >> 1;
        for (size_t i = 0; i < n_params; ++i) {
             if (code_dag->get_material_parameter_annotation_count(idx, i) > 0) {
                 // has annotations
                 return true;
             }
             if (code_dag->get_material_parameter_default(idx, i) != NULL) {
                 // has an default argument
                 return true;
             }
        }
    }
    return false;
}

}  // anonymous

// Print the code to the given printer.
void DAG_code_printer::print(Printer *printer, mi::base::IInterface const *code) const
{
    mi::base::Handle<IGenerated_code_dag const> code_dag(
        code->get_interface<IGenerated_code_dag>());

    if (!code_dag.is_valid_interface())
        return;

    MI::STLEXT::Store<Printer *> tmp(m_printer, printer);

    keyword("dag");
    printer->printf(" \"%s\";\n", code_dag->get_module_name());

    // print imports
    size_t import_count = code_dag->get_import_count();
    if (import_count > 0) {
        print("\n");
        for (size_t i = 0; i < import_count; ++i) {
            keyword("import");
            print(" \"");
            print(code_dag->get_import(i));
            print("\";\n");
        }
    }

    if (size_t annotation_count = code_dag->get_module_annotation_count()) {
        print("\nmodule [[\n");
        for (size_t i = 0; i < annotation_count; ++i) {
            indent(1);
            print_exp(1, code_dag.get(), 2 * i + 1, code_dag->get_module_annotation(i));
            if (i < annotation_count - 1)
                print(", ");
            print("\n");
        }
        print("]];\n");
    }

    print_types(code_dag.get());
    print_constants(code_dag.get());
    print_annotations(code_dag.get());
    print_functions(code_dag.get());
    print_materials(code_dag.get());
}

/// Check if the given call is a material constructor call.
static bool is_material_constructor_call(DAG_call const *call)
{
    if (call->get_semantic() == IDefinition::DS_ELEM_CONSTRUCTOR) {
        if (IType_struct const *s = as<IType_struct>(call->get_type())) {
            return s->get_predefined_id() == IType_struct::SID_MATERIAL;
        }
    }
    return false;
}

// Print a DAG IR node inside a material or function definition.
void DAG_code_printer::print_exp(
    int                       depth,
    IGenerated_code_dag const *dag,
    size_t                    def_index,
    DAG_node const            *node) const
{
    switch(node->get_kind()) {
    case DAG_node::EK_CONSTANT:
        print(cast<DAG_constant>(node)->get_value());
        break;
    case DAG_node::EK_TEMPORARY:
        {
            DAG_temporary const *temporary = cast<DAG_temporary>(node);
            size_t index = temporary->get_index();
            char const *name;
            if (def_index & 1) {
                name = dag->get_function_temporary_name(def_index >> 1, index);
            } else {
                name = dag->get_material_temporary_name(def_index >> 1, index);
            }
            push_color(IPrinter::C_ENTITY);
            if (*name) {
                m_printer->print(name);
            } else {
                m_printer->printf("t_" FMT_SIZE_T, index);
            }
            pop_color();
        }
        break;
    case DAG_node::EK_CALL:
        {
            DAG_call const *call = cast<DAG_call>(node);
            push_color(IPrinter::C_ENTITY);
            if (is_material_constructor_call(call)) {
                m_printer->print("material");
            } else {
                m_printer->print("'");
                m_printer->print(call->get_name());
                m_printer->print("'");
            }
            pop_color();
            m_printer->print("(");
            int count = call->get_argument_count();
            for (int i = 0; i < count; ++i) {
                print("\n");
                indent(depth+1);
                print(call->get_parameter_name(i));
                print(": ");
                print_exp(depth+1, dag, def_index, call->get_argument(i));
                if (i < count - 1)
                    print(", ");
            }
            print(")");
        }
        break;
    case DAG_node::EK_PARAMETER:
        {
            int index = cast<DAG_parameter>(node)->get_index();
            push_color(IPrinter::C_ENTITY);
            if (def_index & 1) {
                print(dag->get_function_parameter_name(def_index >> 1, index));
            } else {
                print(dag->get_material_parameter_name(def_index >> 1, index));
            }
            pop_color();
        }
        break;
    }
}

// Print the semantics if known as a comment.
void DAG_code_printer::print_sema(IDefinition::Semantics sema) const
{
    char const *s = NULL;
    bool be_generated = false;

    switch (sema) {
    case IDefinition::DS_INTRINSIC_DAG_FIELD_ACCESS:
        s = "field access function";
        be_generated = true;
        break;
    case IDefinition::DS_INTRINSIC_DAG_ARRAY_CONSTRUCTOR:
        s = "array constructor";
        be_generated = true;
        break;
    case IDefinition::DS_INTRINSIC_DAG_ARRAY_LENGTH:
        s = "array length operator";
        be_generated = true;
        break;
    case IDefinition::DS_COPY_CONSTRUCTOR:
        s = "copy constructor";
        break;
    case IDefinition::DS_CONV_CONSTRUCTOR:
        s = "conversion constructor";
        break;
    case IDefinition::DS_ELEM_CONSTRUCTOR:
        s = "elemental constructor";
        break;
    case IDefinition::DS_COLOR_SPECTRUM_CONSTRUCTOR:
        s = "color from spectrum constructor";
        break;
    case IDefinition::DS_MATRIX_ELEM_CONSTRUCTOR:
        s = "matrix elemental constructor";
        break;
    case IDefinition::DS_MATRIX_DIAG_CONSTRUCTOR:
        s = "matrix diagonal constructor";
        break;
    case IDefinition::DS_INVALID_REF_CONSTRUCTOR:
        s = "invalid reference constructor";
        break;
    case IDefinition::DS_DEFAULT_STRUCT_CONSTRUCTOR:
        s = "default constructor for a struct";
        break;
    case IDefinition::DS_TEXTURE_CONSTRUCTOR:
        s = "texture constructor";
        break;
    case IDefinition::DS_CONV_OPERATOR:
        s = "conversion operator";
        break;
    default:
        if (semantic_is_operator(sema)) {
            switch (semantic_to_operator(sema)) {

            // unary
            case IExpression::OK_BITWISE_COMPLEMENT:
                s = "bitwise complement operator";
                break;
            case IExpression::OK_LOGICAL_NOT:
                s = "unary logical negation operator";
                break;
            case IExpression::OK_POSITIVE:
                s = "unary arithmetic positive operator";
                break;
            case IExpression::OK_NEGATIVE:
                s = "unary arithmetic negation operator";
                break;
            case IExpression::OK_PRE_INCREMENT:
                s = "pre-increment operator";
                break;
            case IExpression::OK_PRE_DECREMENT:
                s = "pre-decrement operator";
                break;
            case IExpression::OK_POST_INCREMENT:
                s = "post-increment operator";
                break;
            case IExpression::OK_POST_DECREMENT:
                s = "post-decrement operator";
                break;
            case IExpression::OK_CAST:
                s = "cast<> operator";
                break;

            // binary
            case IExpression::OK_SELECT:
                s = "binary select operator";
                break;
            case IExpression::OK_ARRAY_INDEX:
                s = "array index operator";
                break;
            case IExpression::OK_MULTIPLY:
                s = "multiplication operator";
                break;
            case IExpression::OK_DIVIDE:
                s = "division operator";
                break;
            case IExpression::OK_MODULO:
                s = "modulus operator";
                break;
            case IExpression::OK_PLUS:
                s = "binary addition operator";
                break;
            case IExpression::OK_MINUS:
                s = "binary subtraction operator";
                break;
            case IExpression::OK_SHIFT_LEFT:
                s = "shift-left operator";
                break;
            case IExpression::OK_SHIFT_RIGHT:
                s = "arithmetic (signed) shift-right operator";
                break;
            case IExpression::OK_UNSIGNED_SHIFT_RIGHT:
                s = "logical (unsigned) shift-right operator";
                break;
            case IExpression::OK_LESS:
                s = "less operator";
                break;
            case IExpression::OK_LESS_OR_EQUAL:
                s = "less-or-equal operator";
                break;
            case IExpression::OK_GREATER_OR_EQUAL:
                s = "greater-or-equal operator";
                break;
            case IExpression::OK_GREATER:
                s = "greater operator";
                break;
            case IExpression::OK_EQUAL:
                s = "equal operator";
                break;
            case IExpression::OK_NOT_EQUAL:
                s = "not-equal operator";
                break;
            case IExpression::OK_BITWISE_AND:
                s = "bitwise and operator";
                break;
            case IExpression::OK_BITWISE_XOR:
                s = "bitwise xor operator";
                break;
            case IExpression::OK_BITWISE_OR:
                s = "bitwise or operator";
                break;
            case IExpression::OK_LOGICAL_AND:
                s = "logical and operator";
                break;
            case IExpression::OK_LOGICAL_OR:
                s = "logical or operator";
                break;
            case IExpression::OK_ASSIGN:
                s = "assign operator";
                break;
            case IExpression::OK_MULTIPLY_ASSIGN:
                s = "multiplication-assign operator";
                break;
            case IExpression::OK_DIVIDE_ASSIGN:
                s = "division-assign operator";
                break;
            case IExpression::OK_MODULO_ASSIGN:
                s = "modulus-assign operator";
                break;
            case IExpression::OK_PLUS_ASSIGN:
                s = "plus-assign operator";
                break;
            case IExpression::OK_MINUS_ASSIGN:
                s = "minus-assign operator";
                break;
            case IExpression::OK_SHIFT_LEFT_ASSIGN:
                s = "shift-left-assign operator";
                break;
            case IExpression::OK_SHIFT_RIGHT_ASSIGN:
                s = "arithmetic (signed) shift-right-assign operator";
                break;
            case IExpression::OK_UNSIGNED_SHIFT_RIGHT_ASSIGN:
                s = "logical (unsigned) shift-right-assign operator";
                break;
            case IExpression::OK_BITWISE_OR_ASSIGN:
                s = "bitwise or-assign operator";
                break;
            case IExpression::OK_BITWISE_XOR_ASSIGN:
                s = "bitwise xor-assign operator";
                break;
            case IExpression::OK_BITWISE_AND_ASSIGN:
                s = "bitwise and-assign operator";
                break;
            case IExpression::OK_SEQUENCE:
                s = "sequence (comma) operator";
                break;

            // ternary
            case IExpression::OK_TERNARY:
                s = "ternary operator (conditional)";
                break;

            // variadic
            case IExpression::OK_CALL:
                s = "call operator";
                break;
            }
        } else {
            // not an operator
            return;
        }
    }
    push_color(ISyntax_coloring::C_COMMENT);
    print(be_generated ? "// DAG-BE generated " : "// ");
    print(s);
    print(".\n");
    pop_color();
}

// Print a keyword.
void DAG_code_printer::keyword(const char *w) const
{
    push_color(ISyntax_coloring::C_KEYWORD);
    print(w);
    pop_color();
}

// Print only the last part of an absolute MDL name.
void DAG_code_printer::print_short(
    char const *abs_name) const
{
    char const *p = strrchr(abs_name, ':');
    if (p != NULL && p > abs_name && p[-1] == ':')
        abs_name = p + 1;
    print(abs_name);
}

// Print a type in MDL syntax.
void DAG_code_printer::print_mdl_type(
    IType const *type,
    bool full) const
{
    char const    *tn  = NULL;
    ISymbol const *sym = NULL;

    IType::Kind tk = type->get_kind();

    switch (tk) {
    case IType::TK_ERROR:
        push_color(ISyntax_coloring::C_ERROR);
        print("<ERROR>");
        pop_color();
        break;
    case IType::TK_INCOMPLETE:
        push_color(ISyntax_coloring::C_ERROR);
        print("<INCOMPLETE>");
        pop_color();
        break;
    case IType::TK_BOOL:             keyword("bool"); break;
    case IType::TK_INT:              keyword("int"); break;
    case IType::TK_FLOAT:            keyword("float"); break;
    case IType::TK_DOUBLE:           keyword("double"); break;
    case IType::TK_STRING:           keyword("string"); break;
    case IType::TK_COLOR:            keyword("color"); break;
    case IType::TK_LIGHT_PROFILE:    keyword("light_profile"); break;
    case IType::TK_BSDF_MEASUREMENT: keyword("bsdf_measurement"); break;
    case IType::TK_ENUM:
        {
            IType_enum const *e_type = cast<IType_enum>(type);
            ISymbol const *sym = e_type->get_symbol();

            if (full) {
                keyword("enum");
                print(" ");
            }
            print_short(sym->get_name());
        }
        break;
    case IType::TK_ALIAS:
        {
            if (full) {
                keyword("typedef");
                print(" ");
            }
            IType_alias const *a_type = cast<IType_alias>(type);
            sym = a_type->get_symbol();
            type = a_type->get_aliased_type();
            if (IType_alias const *aa_type = as<IType_alias>(type)) {
                // alias of an aliased type, go to base type
                IType::Modifiers mod = a_type->get_type_modifiers();
                if (mod & IType::MK_VARYING) {
                    keyword("varying");
                    print(" ");
                } else if (mod & IType::MK_UNIFORM) {
                    keyword("uniform");
                    print(" ");
                }
                type = aa_type->skip_type_alias();
            }
            print_mdl_type(type);

            if (full) {
                print(" ");
                print_short(sym->get_name());
            }
            return;
        }
    case IType::TK_BSDF:      keyword("bsdf"); break;
    case IType::TK_HAIR_BSDF: keyword("hair_bsdf"); break;
    case IType::TK_EDF:       keyword("edf"); break;
    case IType::TK_VDF:       keyword("vdf"); break;
    case IType::TK_STRUCT:
        {
            IType_struct const *s_type = cast<IType_struct>(type);
            ISymbol const *sym = s_type->get_symbol();

            if (full) {
                keyword("struct");
                print(" ");
            }
            print_short(sym->get_name());
        }
        break;

    case IType::TK_VECTOR:
        {
            IType_vector const *v_type = cast<IType_vector>(type);
            IType        const *e_type = v_type->get_element_type();

            push_color(ISyntax_coloring::C_KEYWORD);
            print_mdl_type(e_type);
            print(long(v_type->get_size()));
            pop_color();
            break;
        }
    case IType::TK_MATRIX:
        {
            IType_matrix const *m_type = cast<IType_matrix>(type);
            IType_vector const *e_type = m_type->get_element_type();
            IType_atomic const *a_type = e_type->get_element_type();

            push_color(ISyntax_coloring::C_KEYWORD);
            print_mdl_type(a_type);
            print(long(m_type->get_columns()));
            print("x");
            print(long(e_type->get_size()));
            pop_color();
            break;
        }
    case IType::TK_ARRAY:
        {
            IType_array const *a_type = cast<IType_array>(type);
            IType       const *e_type = a_type->get_element_type();

            print_mdl_type(e_type);
            print("[");
            if (a_type->is_immediate_sized()) {
                push_color(ISyntax_coloring::C_LITERAL);
                print(long(a_type->get_size()));
                pop_color();
            } else {
                print(a_type->get_deferred_size()->get_size_symbol());
            }
            print("]");
            break;
        }
    case IType::TK_FUNCTION:
        {
            // should not happen
            IType_function const *f_type   = cast<IType_function>(type);
            IType const          *ret_type = f_type->get_return_type();

            if (ret_type != NULL)
                print_mdl_type(ret_type);
            else
                print("void");

            print(" ");
            if (tn != NULL)
                print(tn);
            else
                print("(*)");
            print("(");

            for (size_t i = 0, n = f_type->get_parameter_count(); i < n; ++i) {
                if (i > 0)
                    print(", ");

                IType const   *p_type;
                ISymbol const *sym;
                f_type->get_parameter(i, p_type, sym);
                print(p_type);
                print(" ");
                print(sym);
            }
            print(")");
            break;
        }
    case IType::TK_TEXTURE:
        {
            IType_texture const *t_type = cast<IType_texture>(type);

            push_color(ISyntax_coloring::C_KEYWORD);
            print("texture_");

            switch(t_type->get_shape()) {
            case IType_texture::TS_2D:
                print("2d");
                break;
            case IType_texture::TS_3D:
                print("3d");
                break;
            case IType_texture::TS_CUBE:
                print("cube");
                break;
            case IType_texture::TS_PTEX:
                print("ptex");
                break;
            case IType_texture::TS_BSDF_DATA:
                print("bsdf_data");
                break;
            }
            pop_color();
            break;
        }
    }
}

// Print all types of the code dag.
void DAG_code_printer::print_types(IGenerated_code_dag const *code_dag) const
{
    size_t types_count = code_dag->get_type_count();
    for (size_t i = 0; i < types_count; ++i) {
        IType const *type = code_dag->get_type(i);

        print("\n");

        char const *orig_name = code_dag->get_original_type_name(i);
        if (orig_name != NULL) {
            push_color(ISyntax_coloring::C_COMMENT);
            print("// Alias of \"");
            print(orig_name);
            print("\"\n");
            pop_color();
        }

        keyword("export");
        print(" ");
        print_mdl_type(type, /*full=*/true);

        size_t sub_count = code_dag->get_type_sub_entity_count(i);
        if (sub_count >= 0) {
            print(" {\n");

            for (size_t j = 0; j < sub_count; ++j) {
                indent(1);

                if (IType const *e_tp = code_dag->get_type_sub_entity_type(i, j)) {
                    print_mdl_type(e_tp);
                    print(' ');
                }

                push_color(ISyntax_coloring::C_ENTITY);
                print(code_dag->get_type_sub_entity_name(i, j));
                pop_color();

                if (size_t annotation_count = code_dag->get_type_sub_entity_annotation_count(i, j)) {
                    print('\n');
                    indent(1);
                    print("[[\n");
                    for (size_t k = 0; k < annotation_count; ++k) {
                        indent(2);
                        print_exp(
                            2, code_dag, 0, code_dag->get_type_sub_entity_annotation(i, j, k));
                        if (k < annotation_count - 1)
                            print(',');
                        print("\n");
                    }
                    indent(1);
                    print("]]");
                }

                if (j < sub_count - 1)
                    print(',');
                print('\n');
            }

            print('}');
        }

        if (size_t annotation_count = code_dag->get_type_annotation_count(i)) {
            print("\n[[\n");
            for (size_t k = 0; k < annotation_count; ++k) {
                indent(1);
                print_exp(1, code_dag, 0, code_dag->get_type_annotation(i, k));
                if (k < annotation_count - 1)
                    print(',');
                print("\n");
            }
            print("]]");
        }
        print(";\n");
    }
}

// Print all constants of the code dag.
void DAG_code_printer::print_constants(IGenerated_code_dag const *code_dag) const
{
    size_t constants_count = code_dag->get_constant_count();
    if (constants_count > 0)
        print("\n");

    for (size_t i = 0; i < constants_count; ++i) {
        const char         *constant_name = code_dag->get_constant_name(i);
        DAG_constant const *init          = code_dag->get_constant_value(i);
        IType const        *type          = init->get_type();

        keyword("export");
        print(" ");
        print(type);
        print(" ");
        print(constant_name);
        print(" = ");
        print_exp(1, code_dag, 0, init);

        if (size_t annotation_count = code_dag->get_constant_annotation_count(i)) {
            print("\n[[\n");
            for (size_t k = 0; k < annotation_count; ++k) {
                indent(1);
                print_exp(1, code_dag, 0, code_dag->get_constant_annotation(i, k));
                if (k < annotation_count - 1)
                    print(", ");
                print("\n");
            }
            print("]]");
        }
        print(";\n");
    }
}

// Print all functions of the code dag.
void DAG_code_printer::print_functions(IGenerated_code_dag const *code_dag) const
{
    size_t function_count = code_dag->get_function_count();
    for (size_t i = 0; i < function_count; ++i) {
        print("\n");
        IDefinition::Semantics sema = code_dag->get_function_semantics(i);
        print_sema(sema);

        if (code_dag->get_function_property(i, IGenerated_code_dag::FP_IS_NATIVE)) {
            push_color(ISyntax_coloring::C_COMMENT);
            print("// declared \"native\"\n");
            pop_color();
        }

        char const *orig_name = code_dag->get_original_function_name(i);
        if (orig_name != NULL) {
            push_color(ISyntax_coloring::C_COMMENT);
            print("// Alias of \"");
            print(orig_name);
            print("\"\n");
            pop_color();
        }
        char const *cloned_name = code_dag->get_cloned_function_name(i);
        if (cloned_name != NULL) {
            push_color(ISyntax_coloring::C_COMMENT);
            print("// Clone of \"");
            print(cloned_name);
            print("\"\n");
            pop_color();
        }

        size_t n_refs = code_dag->get_function_references_count(i);
        if (n_refs > 0) {
            push_color(ISyntax_coloring::C_COMMENT);
            print("// Cross references:\n");

            for (size_t j = 0; j < n_refs; ++j) {
                char const *ref = code_dag->get_function_reference(i, j);
                print("//   '");
                print(ref);
                print("'\n");
            }
            pop_color();
        }

        bool is_exported = code_dag->get_function_property(i, IGenerated_code_dag::FP_IS_EXPORTED);
        if (is_exported) {
            keyword("export");
        } else {
            push_color(ISyntax_coloring::C_COMMENT);
            print("/* local */");
            pop_color();
        }
        print(" ");

        IType const *return_type = code_dag->get_function_return_type(i);
        print(return_type);
        if (size_t annotation_count = code_dag->get_function_return_annotation_count(i)) {
            print(" [[\n");
            for (size_t k = 0; k < annotation_count; ++k) {
                indent(1);
                print_exp(
                    1, code_dag, 2 * i + 1, code_dag->get_function_return_annotation(i, k));
                if (k < annotation_count - 1)
                    print(", ");
                print("\n");
            }
            print("]]");
        }
        print(" '");

        char const *function_name = code_dag->get_function_name(i);
        print(function_name);
        print("'(");
        size_t parameter_count = code_dag->get_function_parameter_count(i);
        bool vertical = need_vertical_params(code_dag, 2 * i + 1, parameter_count);

        int depth = 0;
        if (vertical) {
            ++depth;
            print("\n");
            indent(depth);
        }

        for (int k = 0; k < parameter_count; ++k) {
            if (k > 0) {
                if (vertical) {
                    print(",\n");
                    indent(depth);
                } else {
                    print(", ");
                }
            }

            IType const *parameter_type = code_dag->get_function_parameter_type(i, k);
            print(parameter_type);
            print(" ");

            char const *parameter_name = code_dag->get_function_parameter_name(i, k);
            print(parameter_name);
            if (size_t annotation_count = code_dag->get_function_parameter_annotation_count(i, k)) {
                print(" [[\n");
                for (size_t l = 0; l < annotation_count; ++l) {
                    DAG_call const *anno =
                        cast<DAG_call>(code_dag->get_function_parameter_annotation(i, k, l));

                    Indent_scope scope(depth);
                    indent(depth);
                    if (anno->get_semantic() == IDefinition::DS_ENABLE_IF_ANNOTATION) {
                        m_printer->print("enable_if: ");
                        print_exp(
                            depth,
                            code_dag,
                            2 * i + 1,
                            code_dag->get_function_parameter_enable_if_condition(i, k));
                    } else {
                        print_exp(depth, code_dag, 2 * i + 1, anno);
                    }

                    if (l < annotation_count - 1)
                        print(", ");
                    print("\n");
                }
                indent(depth);
                print("]]");
            }

            if (DAG_node const *init = code_dag->get_function_parameter_default(i, k)) {
                print(" = ");
                print_exp(depth, code_dag, 2 * i + 1, init);
            }
        }
        print(")");
        if (vertical)
            --depth;

        if (size_t annotation_count = code_dag->get_function_annotation_count(i)) {
            print("\n[[\n");
            for (size_t k = 0; k < annotation_count; ++k) {
                indent(1);
                print_exp(1, code_dag, 2 * i + 1, code_dag->get_function_annotation(i, k));
                if (k < annotation_count - 1)
                    print(", ");
                print("\n");
            }
            print("]]");
        }

        if (DAG_node const *body = code_dag->get_function_body(i)) {
            print(" = ");
            if (size_t temporary_count = code_dag->get_function_temporary_count(i)) {
                keyword("let");
                print(" {\n");
                for (size_t k = 0; k < temporary_count; k++) {
                    Indent_scope scope(depth);

                    indent(depth);
                    DAG_node const *temporary = code_dag->get_function_temporary(i, k);
                    char const     *name      = code_dag->get_function_temporary_name(i, k);
                    if (*name)
                        m_printer->printf("%s = ", name);
                    else
                       m_printer->printf("t_%d = ", k);
                    print_exp(depth, code_dag, 2 * i + 1, temporary);
                    print(";\n");
                }
                indent(depth);
                print("} ");
                keyword("in");
                print(" ");
            }
            print_exp(depth, code_dag, 2 * i + 1, body);
        }
        print(";\n");
    }
}

// Print all materials of the code dag.
void DAG_code_printer::print_materials(IGenerated_code_dag const *code_dag) const
{
    size_t material_count = code_dag->get_material_count();
    for (size_t mat_idx = 0; mat_idx < material_count; ++mat_idx) {
        print("\n");

        char const *orig_name = code_dag->get_original_material_name(mat_idx);
        if (orig_name != NULL) {
            push_color(ISyntax_coloring::C_COMMENT);
            print("// Alias of \"");
            print(orig_name);
            print("\"\n");
            pop_color();
        }
        char const *cloned_name = code_dag->get_cloned_material_name(mat_idx);
        if (cloned_name != NULL) {
            push_color(ISyntax_coloring::C_COMMENT);
            print("// Clone of \"");
            print(cloned_name);
            print("\"\n");
            pop_color();
        }

        bool is_exported = code_dag->get_material_exported(mat_idx);
        if (is_exported) {
            keyword("export");
        } else {
            push_color(ISyntax_coloring::C_COMMENT);
            print("/* local */");
            pop_color();
        }
        print(" ");

        keyword("material");
        print(" ");

        const char *material_name = code_dag->get_material_name(mat_idx);
        print(material_name);
        print("(");
        size_t parameter_count = code_dag->get_material_parameter_count(mat_idx);
        bool vertical = need_vertical_params(code_dag, 2 * mat_idx, parameter_count);

        int depth = 0;
        if (vertical) {
            ++depth;
            print("\n");
            indent(depth);
        }
        for (size_t k = 0; k < parameter_count; ++k) {
            if (k > 0) {
                if (vertical) {
                    print(",\n");
                    indent(depth);
                } else {
                    print(", ");
                }
            }

            IType const *p_type = code_dag->get_material_parameter_type(mat_idx, k);
            print(p_type);
            print(" ");
            char const *p_name = code_dag->get_material_parameter_name(mat_idx, k);
            print(p_name);

            if (size_t n_annos = code_dag->get_material_parameter_annotation_count(mat_idx, k)) {
                print(" [[\n");
                for (size_t l = 0; l < n_annos; ++l) {
                    DAG_call const *anno = cast<DAG_call>(
                        code_dag->get_material_parameter_annotation(mat_idx, k, l));

                    Indent_scope scope(depth);
                    indent(depth);

                    if (anno->get_semantic() == IDefinition::DS_ENABLE_IF_ANNOTATION) {
                        m_printer->print("enable_if: ");
                        print_exp(
                            depth,
                            code_dag,
                            2 * mat_idx,
                            code_dag->get_material_parameter_enable_if_condition(mat_idx, k));
                    } else {
                        print_exp(
                            depth,
                            code_dag,
                            2 * mat_idx,
                            anno);
                    }
                    if (l < n_annos - 1)
                        print(", ");
                    print("\n");
                }
                indent(depth);
                print("]]");
            }
            if (DAG_node const *init = code_dag->get_material_parameter_default(mat_idx, k)) {
                print(" = ");
                print_exp(depth, code_dag, 2 * mat_idx, init);
            }
        }
        print(")\n");
        if (vertical)
            --depth;
        print("= ");
        if (size_t temporary_count = code_dag->get_material_temporary_count(mat_idx)) {
            keyword("let");
            print(" {\n");
            for (size_t k = 0; k < temporary_count; k++) {
                Indent_scope scope(depth);

                indent(depth);
                DAG_node const *temporary = code_dag->get_material_temporary(mat_idx, k);
                char const     *name      = code_dag->get_material_temporary_name(mat_idx, k);
                if (*name)
                    m_printer->printf("%s = ", name);
                else
                   m_printer->printf("t_%d = ", k);
                print_exp(depth, code_dag, 2 * mat_idx, temporary);
                print(";\n");
            }
            indent(depth);
            print("} ");
            keyword("in");
            print(" ");
        }
        print_exp(depth, code_dag, 2 * mat_idx, code_dag->get_material_value(mat_idx));
        if (size_t n_annos = code_dag->get_material_annotation_count(mat_idx)) {
            print("\n[[\n");
            for (size_t k = 0; k < n_annos; ++k) {
                indent(depth);
                print_exp(
                    depth, code_dag, 2 * mat_idx, code_dag->get_material_annotation(mat_idx,k));
                if (k < n_annos - 1)
                    print(", ");
                print("\n");
            }
            print("]]");
        }
        print(";\n");
    }
}

// Print all annotations of the code dag.
void DAG_code_printer::print_annotations(IGenerated_code_dag const *code_dag) const
{
    size_t annotation_count = code_dag->get_annotation_count();
    for (size_t i = 0; i < annotation_count; ++i) {
        print("\n");
        IDefinition::Semantics sema = code_dag->get_annotation_semantics(i);
        print_sema(sema);

        char const *orig_name = code_dag->get_original_annotation_name(i);
        if (orig_name != NULL) {
            push_color(ISyntax_coloring::C_COMMENT);
            print("// Alias of \"");
            print(orig_name);
            print("\"\n");
            pop_color();
        }

        bool is_exported = code_dag->get_annotation_property(
            i, IGenerated_code_dag::AP_IS_EXPORTED);
        if (is_exported) {
            keyword("export");
        } else {
            push_color(ISyntax_coloring::C_COMMENT);
            print("/* local */");
            pop_color();
        }
        print(" ");

        keyword("annotation");
        print(" '");

        char const *annotation_name = code_dag->get_annotation_name(i);
        print(annotation_name);
        print("'(");
        size_t parameter_count = code_dag->get_annotation_parameter_count(i);
        bool vertical = need_vertical_params(code_dag, 2 * i + 1, parameter_count);

        int depth = 0;
        if (vertical) {
            ++depth;
            print("\n");
            indent(depth);
        }

        for (size_t k = 0; k < parameter_count; ++k) {
            if (k > 0) {
                if (vertical) {
                    print(",\n");
                    indent(depth);
                } else {
                    print(", ");
                }
            }

            IType const *parameter_type = code_dag->get_annotation_parameter_type(i, k);
            print(parameter_type);
            print(" ");

            char const *parameter_name = code_dag->get_annotation_parameter_name(i, k);
            print(parameter_name);

            if (DAG_node const *init = code_dag->get_annotation_parameter_default(i, k)) {
                print(" = ");
                print_exp(depth, code_dag, 2 * i + 1, init);
            }
        }
        print(")");
        if (vertical)
            --depth;

        if (size_t annotation_count = code_dag->get_annotation_annotation_count(i)) {
            print("\n[[\n");
            for (size_t k = 0; k < annotation_count; ++k) {
                indent(1);
                print_exp(1, code_dag, 2 * i + 1, code_dag->get_annotation_annotation(i, k));
                if (k < annotation_count - 1)
                    print(", ");
                print("\n");
            }
            print("]]");
        }
        print(";\n");
    }
}

// Print the code to the given printer.
void Material_instance_printer::print(Printer *printer, mi::base::IInterface const *inst) const
{
    mi::base::Handle<IGenerated_code_dag::IMaterial_instance const> instance(
        inst->get_interface<IGenerated_code_dag::IMaterial_instance>());

    if (!instance.is_valid_interface())
        return;

    MI::STLEXT::Store<Printer *> tmp(m_printer, printer);

    keyword("instance");
    size_t n_params = instance->get_parameter_count();
    if (n_params > 0) {
        print("(");
        for (size_t i = 0; i < n_params; ++i) {
            if (i > 0)
                print(", ");

            IValue const *def_value = instance->get_parameter_default(i);
            IType const  *p_type    = def_value->get_type();

            print(p_type);
            print(" ");
            print(instance->get_parameter_name(i));
            print(" = ");
            print(def_value);
        }
        print(")");
    }
    print(" =\n");
    indent(1);
    if (size_t temporary_count = instance->get_temporary_count()) {
        keyword("let");
        print(" {\n");
        for (size_t k = 0; k < temporary_count; ++k) {
            indent(2);
            m_printer->printf("t_%u = ", unsigned(k));
            print_exp(2, instance.get(), instance->get_temporary_value(k));
            print(";\n");
        }
        indent(1);
        print("} in ");
    }
    print_exp(1, instance.get(), instance->get_constructor());
    print(";\n");
}

void Material_instance_printer::print_exp(
    int                                           depth,
    IGenerated_code_dag::IMaterial_instance const *instance,
    DAG_node const                                *node) const
{
    switch (node->get_kind()) {
    case DAG_node::EK_CONSTANT:
        print(cast<DAG_constant>(node)->get_value());
        break;
    case DAG_node::EK_TEMPORARY:
        {
            int index = cast<DAG_temporary>(node)->get_index();
            color(IPrinter::C_ENTITY);
            m_printer->printf("t_%d", index);
            color(IPrinter::C_DEFAULT);
        }
        break;
    case DAG_node::EK_CALL:
        {
            DAG_call const *call = cast<DAG_call>(node);
            color(IPrinter::C_ENTITY);
            if (is_material_constructor_call(call)) {
                keyword("material");
            } else {
                print(call->get_name());
            }
            color(IPrinter::C_DEFAULT);
            print("(");
            int count = call->get_argument_count();
            for(int i = 0; i < count; i++) {
                print("\n");
                indent(depth+1);
                print(call->get_parameter_name(i));
                print(": ");
                print_exp(depth+1,instance,call->get_argument(i));
                if (i < count - 1)
                    print(", ");
            }
            print(")");
        }
        break;
    case DAG_node::EK_PARAMETER:
        {
            int index = cast<DAG_parameter>(node)->get_index();
            color(IPrinter::C_ENTITY);
            print(instance->get_parameter_name(index));
            color(IPrinter::C_DEFAULT);
        }
        break;
    }
}

// Print a keyword.
void Material_instance_printer::keyword(const char *w) const
{
    color(IPrinter::C_KEYWORD);
    print(w);
    color(IPrinter::C_DEFAULT);
}

} // mdl
} // mi
