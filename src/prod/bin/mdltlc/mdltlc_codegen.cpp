/******************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION. All rights reserved.
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

#include <sstream>
#include <string>
#include <fstream>

#include "mdltlc_codegen.h"
#include "mdltlc_analysis.h"
#include "mdltlc_compilation_unit.h"

#define MDLTLC_DEBUG_POSTCOND 0

static char const *copyright_string =
R"(/******************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION. All rights reserved.
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
)";

/// Return the name of the identifier at the head of `expr`. It can be
/// a call or reference expression. Alias expressions and attributes will
/// be skipped while looking for the node name. Return nullptr otherwise.
char const* node_name(Expr const* expr) {
    while (true) {
        if (Expr_attribute const* att = as<Expr_attribute>(expr)) {
            expr = att->get_argument();
        }
        else if (Expr_binary const* eb = as<Expr_binary>(expr)) {
            if (eb->get_operator() == Expr_binary::Operator::OK_TILDE) {
                expr = eb->get_right_argument();
            }
            else {
                break;
            }
        }
        else if (Expr_call const* ec = as<Expr_call>(expr)) {
            expr = ec->get_callee();
        }
        else if (Expr_ref const* er = as<Expr_ref>(expr)) {
            return er->get_name()->get_name();
        }
        else {
            break;
        }
    }

    return nullptr;
}

/// Returns arity of a mixer node (1, 2, 3, or 4), or 0 otherwise.
int get_n_ary_mixer(mi::mdl::Node_types *node_types, Type_function const *tf) {
    int n = 0;
    mi::mdl::Node_type const *node_type = tf->get_node_type();

    if (!node_type)
        return 0;

    int node_type_idx = node_types->idx_from_type(node_type->type_name);

    switch (node_type_idx) {
    case mi::mdl::bsdf_mix_1:
    case mi::mdl::bsdf_mix_2:
    case mi::mdl::bsdf_mix_3:
    case mi::mdl::bsdf_mix_4:
        n = 1 + node_type_idx - mi::mdl::bsdf_mix_1;
        break;
    case mi::mdl::bsdf_clamped_mix_1:
    case mi::mdl::bsdf_clamped_mix_2:
    case mi::mdl::bsdf_clamped_mix_3:
    case mi::mdl::bsdf_clamped_mix_4:
        n = 1 + node_type_idx - mi::mdl::bsdf_clamped_mix_1;
        break;
    case mi::mdl::bsdf_unbounded_mix_1:
    case mi::mdl::bsdf_unbounded_mix_2:
    case mi::mdl::bsdf_unbounded_mix_3:
    case mi::mdl::bsdf_unbounded_mix_4:
        n = 1 + node_type_idx - mi::mdl::bsdf_unbounded_mix_1;
        break;
    case mi::mdl::bsdf_color_mix_1:
    case mi::mdl::bsdf_color_mix_2:
    case mi::mdl::bsdf_color_mix_3:
    case mi::mdl::bsdf_color_mix_4:
        n = 1 + node_type_idx - mi::mdl::bsdf_color_mix_1;
        break;
    case mi::mdl::bsdf_color_clamped_mix_1:
    case mi::mdl::bsdf_color_clamped_mix_2:
    case mi::mdl::bsdf_color_clamped_mix_3:
    case mi::mdl::bsdf_color_clamped_mix_4:
        n = 1 + node_type_idx - mi::mdl::bsdf_color_clamped_mix_1;
        break;
    case mi::mdl::bsdf_color_unbounded_mix_1:
    case mi::mdl::bsdf_color_unbounded_mix_2:
    case mi::mdl::bsdf_color_unbounded_mix_3:
    case mi::mdl::bsdf_color_unbounded_mix_4:
        n = 1 + node_type_idx - mi::mdl::bsdf_color_unbounded_mix_1;
        break;
    case mi::mdl::edf_mix_1:
    case mi::mdl::edf_mix_2:
    case mi::mdl::edf_mix_3:
    case mi::mdl::edf_mix_4:
        n = 1 + node_type_idx - mi::mdl::edf_mix_1;
        break;
    case mi::mdl::edf_clamped_mix_1:
    case mi::mdl::edf_clamped_mix_2:
    case mi::mdl::edf_clamped_mix_3:
    case mi::mdl::edf_clamped_mix_4:
        n = 1 + node_type_idx - mi::mdl::edf_clamped_mix_1;
        break;
    case mi::mdl::edf_unbounded_mix_1:
    case mi::mdl::edf_unbounded_mix_2:
    case mi::mdl::edf_unbounded_mix_3:
    case mi::mdl::edf_unbounded_mix_4:
        n = 1 + node_type_idx - mi::mdl::edf_unbounded_mix_1;
        break;
    case mi::mdl::edf_color_mix_1:
    case mi::mdl::edf_color_mix_2:
    case mi::mdl::edf_color_mix_3:
    case mi::mdl::edf_color_mix_4:
        n = 1 + node_type_idx - mi::mdl::edf_color_mix_1;
        break;
    case mi::mdl::edf_color_clamped_mix_1:
    case mi::mdl::edf_color_clamped_mix_2:
    case mi::mdl::edf_color_clamped_mix_3:
    case mi::mdl::edf_color_clamped_mix_4:
        n = 1 + node_type_idx - mi::mdl::edf_color_clamped_mix_1;
        break;
    case mi::mdl::edf_color_unbounded_mix_1:
    case mi::mdl::edf_color_unbounded_mix_2:
    case mi::mdl::edf_color_unbounded_mix_3:
    case mi::mdl::edf_color_unbounded_mix_4:
        n = 1 + node_type_idx - mi::mdl::edf_color_unbounded_mix_1;
        break;
    case mi::mdl::vdf_mix_1:
    case mi::mdl::vdf_mix_2:
    case mi::mdl::vdf_mix_3:
    case mi::mdl::vdf_mix_4:
        n = 1 + node_type_idx - mi::mdl::vdf_mix_1;
        break;
    case mi::mdl::vdf_clamped_mix_1:
    case mi::mdl::vdf_clamped_mix_2:
    case mi::mdl::vdf_clamped_mix_3:
    case mi::mdl::vdf_clamped_mix_4:
        n = 1 + node_type_idx - mi::mdl::vdf_clamped_mix_1;
        break;
    case mi::mdl::vdf_unbounded_mix_1:
    case mi::mdl::vdf_unbounded_mix_2:
    case mi::mdl::vdf_unbounded_mix_3:
    case mi::mdl::vdf_unbounded_mix_4:
        n = 1 + node_type_idx - mi::mdl::vdf_unbounded_mix_1;
        break;
    case mi::mdl::vdf_color_mix_1:
    case mi::mdl::vdf_color_mix_2:
    case mi::mdl::vdf_color_mix_3:
    case mi::mdl::vdf_color_mix_4:
        n = 1 + node_type_idx - mi::mdl::vdf_color_mix_1;
        break;
    case mi::mdl::vdf_color_clamped_mix_1:
    case mi::mdl::vdf_color_clamped_mix_2:
    case mi::mdl::vdf_color_clamped_mix_3:
    case mi::mdl::vdf_color_clamped_mix_4:
        n = 1 + node_type_idx - mi::mdl::vdf_color_clamped_mix_1;
        break;
    case mi::mdl::vdf_color_unbounded_mix_1:
    case mi::mdl::vdf_color_unbounded_mix_2:
    case mi::mdl::vdf_color_unbounded_mix_3:
    case mi::mdl::vdf_color_unbounded_mix_4:
        n = 1 + node_type_idx - mi::mdl::vdf_color_unbounded_mix_1;
        break;
    default:
        n = 0;
        break;
    }
    return n;
}

/// Returns true if the node is mixer with color weights.
bool is_color_mixer(mi::mdl::Node_types *node_types, Type_function const *tf) {
    mi::mdl::Node_type const* node_type = tf->get_node_type();

    if (!node_type)
        return false;

    int node_type_idx = node_types->idx_from_type(node_type->type_name);

    switch (node_type_idx) {
    case mi::mdl::bsdf_color_mix_1:
    case mi::mdl::bsdf_color_mix_2:
    case mi::mdl::bsdf_color_mix_3:
    case mi::mdl::bsdf_color_mix_4:
    case mi::mdl::bsdf_color_clamped_mix_1:
    case mi::mdl::bsdf_color_clamped_mix_2:
    case mi::mdl::bsdf_color_clamped_mix_3:
    case mi::mdl::bsdf_color_clamped_mix_4:
    case mi::mdl::bsdf_color_unbounded_mix_1:
    case mi::mdl::bsdf_color_unbounded_mix_2:
    case mi::mdl::bsdf_color_unbounded_mix_3:
    case mi::mdl::bsdf_color_unbounded_mix_4:
    case mi::mdl::edf_color_mix_1:
    case mi::mdl::edf_color_mix_2:
    case mi::mdl::edf_color_mix_3:
    case mi::mdl::edf_color_mix_4:
    case mi::mdl::edf_color_clamped_mix_1:
    case mi::mdl::edf_color_clamped_mix_2:
    case mi::mdl::edf_color_clamped_mix_3:
    case mi::mdl::edf_color_clamped_mix_4:
    case mi::mdl::edf_color_unbounded_mix_1:
    case mi::mdl::edf_color_unbounded_mix_2:
    case mi::mdl::edf_color_unbounded_mix_3:
    case mi::mdl::edf_color_unbounded_mix_4:
    case mi::mdl::vdf_color_mix_1:
    case mi::mdl::vdf_color_mix_2:
    case mi::mdl::vdf_color_mix_3:
    case mi::mdl::vdf_color_mix_4:
    case mi::mdl::vdf_color_clamped_mix_1:
    case mi::mdl::vdf_color_clamped_mix_2:
    case mi::mdl::vdf_color_clamped_mix_3:
    case mi::mdl::vdf_color_clamped_mix_4:
    case mi::mdl::vdf_color_unbounded_mix_1:
    case mi::mdl::vdf_color_unbounded_mix_2:
    case mi::mdl::vdf_color_unbounded_mix_3:
    case mi::mdl::vdf_color_unbounded_mix_4:
        return true;
        break;
    default:
        break;
    }
    return false;
}

void Compilation_unit::output_cpp_match_variables(
    pp::Pretty_print &p,
    Expr const *expr,
    mi::mdl::string const &prefix,
    Var_set &used_vars)
{
    if (Expr_call const *call = as<Expr_call>(expr)) {

        Expr const *callee = call->get_callee();
        Type_function const *callee_type =
            cast<Type_function>(callee->get_type());

        for (int i = 0; i < call->get_argument_count(); i++) {

            Expr const *arg = call->get_argument(i);
            char idx_str[32];
            snprintf(idx_str, sizeof(idx_str), "%d", i);


            mi::mdl::string prefix2(m_arena.get_allocator());

            if (get_n_ary_mixer(m_node_types, callee_type) > 0) {
                prefix2 = "e.get_remapped_argument(";
            } else {
                prefix2 = "e.get_compound_argument(";
            }
            prefix2 += prefix + ", " + idx_str + ")";

            output_cpp_match_variables(p, arg, prefix2, used_vars);
        }
    } else if (Expr_ref const *er = as<Expr_ref>(expr)) {
        if (strcmp(er->get_name()->get_name(), "_") &&
            used_vars.find(er->get_name()) != used_vars.end()) {
            p.string("const DAG_node* v_");
            p.string(er->get_name()->get_name());
            p.string(" = ");
            p.string(prefix.c_str());
            p.string(";");
            p.nl();
        }
    } else if (Expr_binary const *eb = as<Expr_binary>(expr)) {
        if (eb->get_operator() == Expr_binary::Operator::OK_TILDE) {
            output_cpp_match_variables(p, eb->get_left_argument(), prefix, used_vars);
            output_cpp_match_variables(p, eb->get_right_argument(), prefix, used_vars);
        } else {
            MDL_ASSERT(!"unexpected binary operator in output_cpp_match_variables");
        }
    } else if (Expr_attribute const *attr = as<Expr_attribute>(expr)) {
        /* Extract variables from wrapped expression first. */
        output_cpp_match_variables(p, attr->get_argument(), prefix, used_vars);

        /* Now extract variables from attributes. */

        Expr_attribute::Expr_attribute_vector const &attrs = attr->get_attributes();
        for (Expr_attribute::Expr_attribute_entry const &ap : attrs) {

            // Find out whether we need to emit a binding for this
            // attribute. This is the case if either the attribute
            // variable itself is used, or if the pattern it binds
            // defines any variables.
            //
            // FIXME: This is not fully correct, it fails to detect
            // when neither the attribute name nor any variables bound
            // in it's pattern are used. This will only result in C++
            // compiler warnings, which is acceptable for now.

            Var_set locally_defined(m_arena.get_allocator());
            if (ap.expr)
                defined_vars(ap.expr, locally_defined);

            if (used_vars.find(ap.name) == used_vars.end() &&
                locally_defined.size() == 0)
                continue;

            mi::mdl::string arg_pfx(m_arena.get_allocator());

            if (ap.expr) {
                mi::mdl::string tmp(m_arena.get_allocator());
                tmp = std::to_string(next_attr_counter()).c_str();

                arg_pfx += "vv_";
                arg_pfx += tmp.c_str();
                arg_pfx += "_";
                arg_pfx += ap.name->get_name();

                p.string("const DAG_node *");
                p.string(arg_pfx.c_str());
                p.string(" = e.get_attribute(");
                p.string(prefix.c_str());
                p.string(", \"");
                p.escaped_string(ap.name->get_name());
                p.string("\");");
                p.nl();
                output_cpp_match_variables(p, ap.expr, arg_pfx, used_vars);
            } else {
                arg_pfx += "v_";
                arg_pfx += ap.name->get_name();

                p.string("const DAG_node *");
                p.string(arg_pfx.c_str());
                p.string(" = e.get_attribute(");
                p.string(prefix.c_str());
                p.string(", \"");
                p.escaped_string(ap.name->get_name());
                p.string("\");");
                p.nl();
            }
        }
    } else if (Expr_type_annotation const *ta = as <Expr_type_annotation>(expr)) {
        output_cpp_match_variables(p, ta->get_argument(), prefix, used_vars);
    } else {
        MDL_ASSERT(!"unexpected node in output_cpp_match_variables");
    }
}

/// Return true if all arguments of the call expressions are constant
/// (e.g., literals).
bool Compilation_unit::all_arguments_constants(Expr_call const *expr_call) {
    for (size_t i = 0; i < expr_call->get_argument_count(); i++) {
        Expr const *arg = expr_call->get_argument(i);
        if (!is<Expr_literal>(arg))
            return false;
    }
    return true;
}

// Output the C++ equivalent of the given literal.
void Compilation_unit::output_cpp_literal_value(pp::Pretty_print &p,Expr_literal const *lit, bool for_color) {
    switch (lit->get_type()->get_kind()) {
    case Type::Kind::TK_INT:
    {
        int i = cast<Value_int>(lit->get_value())->get_value();
        p.integer(i);
        break;
    }
    case Type::Kind::TK_FLOAT:
    {
        float i = cast<Value_float>(lit->get_value())->get_value();
        if (!for_color) {
            // COMPATIBILITY: We are printing the float value in the form
            // the user entered it in the mdltl file.
            p.string(cast<Value_float>(lit->get_value())->get_s_value());

            // COMPATIBILITY: Printing "f" suffix.
            p.string("f");
        } else {
            p.floating_point(i);
        }
        break;
    }
    case Type::Kind::TK_BOOL:
    {
        bool i = cast<Value_bool>(lit->get_value())->get_value();
        p.string(i ? "true" : "false");
        break;
    }
    case Type::Kind::TK_STRING:
    {
        char const *i = cast<Value_string>(lit->get_value())->get_value();
        p.string("\"");
        p.escaped_string(i);
        p.string("\"");
        break;
    }
    default:
        error(lit->get_location(), "[BUG] unsupperted literal type in output_cpp_literal_value");
        MDL_ASSERT(!"[BUG] unsupperted literal type in output_cpp_literal_value");
        break;
    }
}

// Output the code to generate a constant matching the literal
// expression `expr`.
void Compilation_unit::output_cpp_literal(pp::Pretty_print &p,Expr const *expr) {
    Expr_literal const *e = cast<Expr_literal>(expr);
    switch (e->get_type()->get_kind()) {
    case Type::Kind::TK_INT:
    {
        p.string("e.create_int_constant");
        p.with_parens([&] (pp::Pretty_print &p) {
                output_cpp_literal_value(p, e);
            });
        break;
    }
    case Type::Kind::TK_FLOAT:
    {
        p.string("e.create_float_constant");
        p.with_parens([&] (pp::Pretty_print &p) {
                output_cpp_literal_value(p, e);
            });
        break;
    }
    case Type::Kind::TK_BOOL:
    {
        p.string("e.create_bool_constant");
        p.with_parens([&] (pp::Pretty_print &p) {
                output_cpp_literal_value(p, e);
            });
        break;
    }
    case Type::Kind::TK_STRING:
    {
        p.string("e.create_string_constant");
        p.with_parens([&] (pp::Pretty_print &p) {
                output_cpp_literal_value(p, e);
            });
        break;
    }
    default:
        error(expr->get_location(), "[BUG] unsupperted literal type in output_cpp_literal");
        MDL_ASSERT(!"[BUG] unsupperted literal type in output_cpp_literal");
        break;
    }
}

// Output a call to a constant creation function named `call_name`.
void Compilation_unit::output_cpp_constant_call(
    pp::Pretty_print &p,
    Expr_call const *call,
    char const *call_name)
{
    int n = call->get_argument_count();

    p.string("e.create_");
    p.string(call_name);
    p.string("_constant");
    p.with_parens(
        [&] (pp::Pretty_print &p) {

            // Output arguments as C++ literals.

            for (size_t i = 0; i < n; i++) {
                Expr const *arg = call->get_argument(i);
                Expr_literal const *lit = cast<Expr_literal>(arg);
                output_cpp_literal_value(p, lit, true);
                if (i < n - 1)
                    p.comma();
            }

            // Special case: for `color` and `float3` constructors with one
            // argument, we expand the first argument to three actual
            // arguments.  These are also output as C++ literals.

            if ((!strcmp(call_name, "color") || !strcmp(call_name, "float3")) && n == 1) {
                Expr const *arg = call->get_argument(0);
                Expr_literal const *lit = cast<Expr_literal>(arg);
                for (size_t i = n; i < 3; i++) {
                    p.comma();
                    output_cpp_literal_value(p, lit, true);
                }
            }
        });
}

// Output a call to a mixer function.
void Compilation_unit::output_cpp_mixer_call(
    pp::Pretty_print &p,
    Expr_call const *call,
    int n_ary_mixer,
    Type_function const *callee_type)
{
    p.string("e.create_");
    if (is_color_mixer(m_node_types, callee_type))
        p.string("color_");
    p.string("mixer_call");
    p.with_parens([&](pp::Pretty_print &p) {
        p.with_indent([&](pp::Pretty_print p) {
                p.string("Args_wrapper<");
                p.integer(2 * n_ary_mixer);
                p.string(">::mk_args");
                p.with_parens([&](pp::Pretty_print &p) {
                    p.with_indent([&](pp::Pretty_print p) {
                        p.string("e, m_node_types,");
                        p.space();
                        p.string("node_null,");
                        p.space();

                        // Create n argument pairs (weight,component).

                        for (int k = 0; k < n_ary_mixer; ++k) {
                            Expr const *arg_k = call->get_argument(2 * k);
                            Expr const *arg_k_1 = call->get_argument(2 * k + 1);

                            output_cpp_expr(p, arg_k);
                            p.string(",");
                            p.space();
                            output_cpp_expr(p, arg_k_1);
                            if (k + 1 < n_ary_mixer) {
                                p.comma();
                                p.space();
                            }
                        }
                        });
                    });
                p.string(".args,");
            p.space();
            p.integer(2 * n_ary_mixer);
            });
        });
}

// Output a function call expression.
void Compilation_unit::output_cpp_function_call(
    pp::Pretty_print &p,
    Expr_call const *call,
    Expr_ref const *callee_ref)
{
    int n = call->get_argument_count();

    p.string("e.create_function_call");
    p.with_parens([&] (pp::Pretty_print &p) {
            p.with_indent([&] (pp::Pretty_print &p) {
                    char const *name = callee_ref->get_name()->get_name();
                    p.string("\"");
                    if (strncmp("::", name, 2))
                        p.string("::");
                    p.string(name);
                    p.string("\",");
                    p.space();
                    p.string("Nodes_wrapper<");
                    p.integer(n);
                    p.string(">");
                    p.with_parens([&] (pp::Pretty_print &p) {
                            p.with_indent([&] (pp::Pretty_print &p) {
                                    for (size_t i = 0; i < n; i++) {
                                        Expr const *arg = call->get_argument(i);
                                        output_cpp_expr(p, arg);
                                        if (i < n - 1) {
                                            p.comma();
                                            p.space();
                                        }
                                    }
                                });
                        });
                    p.string(".data(),");
                    p.space();
                    // Re-use the formatted `n`.
                    p.integer(n);
                    p.string(", root_dbg_info");
                });
        });
}

void Compilation_unit::output_cpp_bsdf_call(
    pp::Pretty_print &p,
    Expr_call const *call,
    Expr_ref const *callee_ref,
    mi::mdl::Node_type const *node_type,
    int node_type_idx)
{
    Type const *return_type = call->get_type();
    int arg_n = call->get_argument_count();
    int param_n = node_type->parameters.size();
    char param_b[32];

    snprintf(param_b, sizeof(param_b), "%d", param_n);

    char const *ref_sig = callee_ref->get_signature();
    MDL_ASSERT(ref_sig && "signature required");
    MDL_ASSERT(node_type->get_signature() == ref_sig && "signatures must match");
    p.string("e.create_call");
    p.with_parens([&] (pp::Pretty_print &p) {
            p.with_indent([&] (pp::Pretty_print &p) {
                    p.string("\"");
                    p.string(ref_sig);
                    p.string("\",");
                    p.space();
                    char const *mdl_semantics_name =
                        get_semantics_name(node_type->semantics);
                    if (mdl_semantics_name) {
                        // Normal MDL semantics value.
                        p.string("IDefinition::");
                        p.string(mdl_semantics_name);
                    } else {
                        // Extended Distiller semantics value.
                        p.string("static_cast<mi::mdl::IDefinition::Semantics>(");
                        p.integer(find_semantics(call));
                        p.string(")");
                    }
                    p.comma();
                    p.space();
                    p.with_indent([&] (pp::Pretty_print &p) {

                            p.string("Args_wrapper<");
                            p.integer(param_n);
                            p.string_with_nl(">::mk_args");
                            p.with_parens([&] (pp::Pretty_print &p) {
                                    p.softbreak();
                                    p.string("e, m_node_types,");
                                    p.space();
                                    p.string(callee_ref->get_name()->get_name());
                                    for (int i = 0; i < arg_n; ++i) {
                                        p.comma();
                                        p.space();
                                        output_cpp_expr(p, call->get_argument(i));
                                    }
                                });
                        });
                    p.string(".args,");
                    p.space();
                    p.integer(param_n);
                    p.comma();
                    p.space();
                    if ((node_type_idx >= mi::mdl::material)
                        && (node_type_idx <= mi::mdl::material_geometry)) {
                        p.string("e.get_type_factory()->");
                        p.softbreak();
                        p.string("get_predefined_struct(");
                        p.softbreak();
                        switch (node_type_idx) {
                        case mi::mdl::material:
                            p.string("IType_struct::SID_MATERIAL");
                            break;
                        case mi::mdl::material_surface:
                            p.string("IType_struct::SID_MATERIAL_SURFACE");
                            break;
                        case mi::mdl::material_emission:
                            p.string("IType_struct::SID_MATERIAL_EMISSION");
                            break;
                        case mi::mdl::material_volume:
                            p.string("IType_struct::SID_MATERIAL_VOLUME");
                            break;
                        case mi::mdl::material_geometry:
                            p.string("IType_struct::SID_MATERIAL_GEOMETRY");
                            break;
                        }
                        p.string(")");
                    } else if (callee_ref->get_name() == m_symbol_table->get_symbol("local_normal")
                               || callee_ref->get_name() == m_symbol_table->get_symbol("::nvidia::distilling_support::local_normal")) {

                        // COMPATIBILITY: Calls to local_normal are treated as a
                        // color-valued function in the distiller. We are internally
                        // handling it as a bsdf, therefore we need to convert back on
                        // code generation to match old compiler behaviour.

                        p.string("e.get_type_factory()->create_color()");
                    } else {
                        p.string("e.get_type_factory()->create_");
                        switch (return_type->get_kind()) {
                        case Type::Kind::TK_BSDF:
                            p.string("bsdf");
                            break;
                        case Type::Kind::TK_EDF:
                            p.string("edf");
                            break;
                        case Type::Kind::TK_VDF:
                            p.string("vdf");
                            break;
                        case Type::Kind::TK_HAIR_BSDF:
                            p.string("hair_bsdf");
                            break;
                        default:
                            error(call->get_location(),
                                  "[BUG] Inconsistent return type in output_cpp_bsdf_call");
                            MDL_ASSERT(!"[BUG] Inconsistent return type in output_cpp_bsdf_call");
                            break;
                        }
                        p.string("()");
                    }
                    p.string(", root_dbg_info");
                });
        });
}

/// Generate the code to construct the DAG for a call
/// expression. Depending on the kind of called entity, this may
/// result in the creation of constants, in function calls or BSDF
/// calls.
void Compilation_unit::output_cpp_call(pp::Pretty_print &p, Expr_call const *call) {
    Expr const *callee = call->get_callee();
    Expr_ref const *callee_ref = cast<Expr_ref>(callee);
    Type_function const *callee_type = cast<Type_function>(callee_ref->get_type());

    // If all arguments are constant (literals), and we are calling
    // one of the basic constructors for `color` or `float3`, generate
    // the constant creation expression directly.

    if (all_arguments_constants(call)) {
        if (callee_ref->get_name() == m_symbol_table->get_symbol("color")) {
            output_cpp_constant_call(p, call, "color");
            return;
        }
        if (callee_ref->get_name() == m_symbol_table->get_symbol("float3")) {
            output_cpp_constant_call(p, call, "float3");
            return;
        }
    }

    // If the name of the callee is defined as one of the node types,
    // we generate either a BSDF constant or a node call.

    if (m_node_types->is_type(callee_ref->get_name()->get_name())) {

        int n_ary_mixer = get_n_ary_mixer(m_node_types, callee_type);
        if (n_ary_mixer > 0) {
            output_cpp_mixer_call(p, call, n_ary_mixer, callee_type);
            return;
        }


        mi::mdl::Node_type const *node_type = callee_type->get_node_type();

        if (node_type) {
            int node_type_idx = m_node_types->idx_from_type(node_type->type_name);

            if ((node_type_idx == mi::mdl::bsdf
                 || node_type_idx == mi::mdl::edf
                 || node_type_idx == mi::mdl::vdf
                 || node_type_idx == mi::mdl::hair_bsdf)) {
                p.string("e.create_");
                switch (node_type_idx) {
                case mi::mdl::bsdf:
                    p.string("bsdf");
                    break;
                case mi::mdl::edf:
                    p.string("edf");
                    break;
                case mi::mdl::vdf:
                    p.string("vdf");
                    break;
                case mi::mdl::hair_bsdf:
                    p.string("hair_bsdf");
                    break;
                default:
                    error(call->get_location(),
                          "[BUG] encountered invalid return type kind in output_cpp_call");
                    MDL_ASSERT(!"[BUG] encountered invalid return type kind in output_cpp_call");
                    break;
                }
                p.string("_constant()");
                return;
            }

            output_cpp_bsdf_call(p, call, callee_ref, node_type, node_type_idx);
            return;
        }

        error(call->get_location(),
              "[BUG] inconsistency in output_cpp_call");
        MDL_ASSERT(!"[BUG] inconsistency in output_cpp_call");
        return;
    }

    // If none of the other cases matched, we create a function call
    // expression.
    output_cpp_function_call(p, call, callee_ref);
}

void Compilation_unit::output_cpp_expr_bindings(
    pp::Pretty_print &p,
    Expr const *expr)
{
    switch (expr->get_kind()) {
    case Expr::Kind::EK_INVALID:
    case Expr::Kind::EK_LITERAL:
    case Expr::Kind::EK_REFERENCE:
    {
        break;
    }
    case Expr::Kind::EK_UNARY:
    {
        Expr_unary const *e = cast<Expr_unary>(expr);

        output_cpp_expr_bindings(p, e->get_argument());
        break;
    }
    case Expr::Kind::EK_BINARY:
    {
        Expr_binary const *e = cast<Expr_binary>(expr);
        output_cpp_expr_bindings(p, e->get_left_argument());
        output_cpp_expr_bindings(p, e->get_right_argument());
        break;
    }
    case Expr::Kind::EK_CONDITIONAL:
    {
        Expr_conditional const *e = cast<Expr_conditional>(expr);
        output_cpp_expr_bindings(p, e->get_condition());
        output_cpp_expr_bindings(p, e->get_true());
        output_cpp_expr_bindings(p, e->get_false());
        break;
    }
    case Expr::Kind::EK_CALL:
    {
        Expr_call const *e = cast<Expr_call>(expr);
        size_t n = e->get_argument_count();
        for (size_t i = 0; i < n; i++) {
            Expr const *arg = e->get_argument(i);
            output_cpp_expr_bindings(p, arg);
        }
        break;
    }
    case Expr::Kind::EK_TYPE_ANNOTATION:
    {
        Expr_type_annotation const *e = cast<Expr_type_annotation>(expr);
        output_cpp_expr_bindings(p, e->get_argument());
        break;
    }
    case Expr::Kind::EK_ATTRIBUTE:
    {
        Expr_attribute const *e = cast<Expr_attribute>(expr);

        // Output argument expression first, because it may define
        // variables used in the current node's attributes.
        output_cpp_expr_bindings(p, e->get_argument());

        p.string("DAG_node const *");
        p.string(e->get_node_name());
        p.string(" = ");
        p.with_indent([&] (pp::Pretty_print &p) {
                output_cpp_expr(p, e->get_argument());
            });
        p.semicolon();
        p.nl();

        for (Expr_attribute::Expr_attribute_entry entry : e->get_attributes()) {

            // Expression attribute pairs MUST have an expression.
            MDL_ASSERT(entry.expr);

            // Generate code for each attribute expression, put it
            // into map for this node.
            mi::mdl::string tmp_expr(m_arena.get_allocator());
            tmp_expr = e->get_node_name();
            tmp_expr += "_";
            tmp_expr += entry.name->get_name();

            p.string("DAG_node const *");
            p.string(tmp_expr.c_str());
            p.string(" = ");
            p.with_indent([&] (pp::Pretty_print &p) {
                    output_cpp_expr(p, entry.expr);
                });
            p.semicolon();
            p.nl();
            p.string("e.set_attribute");
            p.with_parens([&] (pp::Pretty_print &p) {
                    p.string(e->get_node_name());
                    p.comma();
                    p.space();
                    p.string("\"");
                    p.string(entry.name->get_name());
                    p.string("\"");
                    p.comma();
                    p.string(tmp_expr.c_str());
                });
            p.semicolon();
            p.nl();
        }

        break;
    }
    }
}

void Compilation_unit::output_cpp_expr(
    pp::Pretty_print &p,
    Expr const *expr)
{
    mi::mdl::string tmp(m_arena.get_allocator());
    char b[32];

    switch (expr->get_kind()) {
    case Expr::Kind::EK_INVALID:
        error(expr->get_location(),
              "[BUG] encountered invalid expression in output_cpp_expr");
        MDL_ASSERT(!"[BUG] encountered invalid expression in output_cpp_expr");
        break;
    case Expr::Kind::EK_LITERAL:
    {
        output_cpp_literal(p, expr);
        break;
    }
    case Expr::Kind::EK_REFERENCE:
    {
        Expr_ref const *e = cast<Expr_ref>(expr);


        switch (e->get_type()->get_kind()) {
        case Type::Kind::TK_ENUM:
        {
            Type_enum const *t = cast<Type_enum>(e->get_type());

            // FIXME: Instead of hard-coding these, come up with
            // better solution.

            if (!strcmp(t->get_name()->get_name(), "scatter_mode") ||
                !strcmp(t->get_name()->get_name(), "::df::scatter_mode")) {
                p.string("e.create_scatter_enum_constant");
                p.with_parens([&] (pp::Pretty_print &p) {
                        int code = t->lookup_variant(e->get_name());
                        snprintf(b, sizeof(b), "%d", code);
                        p.integer(code);
                    });
            } else
                if (!strcmp(t->get_name()->get_name(), "emission_mode") ||
                    !strcmp(t->get_name()->get_name(), "::df::emission_mode")) {
                    p.string("e.create_emission_enum_constant");
                    p.with_parens([&] (pp::Pretty_print &p) {
                            int code = t->lookup_variant(e->get_name());
                            snprintf(b, sizeof(b), "%d", code);
                            p.integer(code);
                        });
                } else
                    if (!strcmp(t->get_name()->get_name(), "wrap_mode") ||
                        !strcmp(t->get_name()->get_name(), "::tex::wrap_mode")) {
                        p.string("e.create_wrap_mode_enum_constant");
                        p.with_parens([&] (pp::Pretty_print &p) {
                                int code = t->lookup_variant(e->get_name());
                                snprintf(b, sizeof(b), "%d", code);
                                p.integer(code);
                            });
                    } else {
                        error(expr->get_location(), "[BUG] unsupported enum type in output_cpp_expr");
                        MDL_ASSERT(!"[BUG] unsupported enum type in output_cpp_expr");
                    }
            break;
        }
        default:
            p.string("v_");
            p.string(e->get_name()->get_name());
            break;
        }
        break;
    }
    case Expr::Kind::EK_UNARY:
    {
        Expr_unary const *e = cast<Expr_unary>(expr);

        // Special handling for option access.

        // FIXME: Design a more general, data-driven way of generating
        // code for options.

        if (e->get_operator() == Expr_unary::Operator::OK_OPTION) {

            Expr const *arg = e->get_argument();
            while (arg->get_kind() == Expr::Kind::EK_TYPE_ANNOTATION) {
                arg = cast<Expr_type_annotation>(arg)->get_argument();
            }
            Symbol const *option_name = cast<Expr_ref>(arg)->get_name();
            if (option_name == m_symbol_table->get_symbol("top_layer_weight")) {
                p.string("e.create_float_constant(options->top_layer_weight)");
            } else if (option_name == m_symbol_table->get_symbol("global_ior")) {
                p.string("e.create_global_ior()");
            } else if (option_name == m_symbol_table->get_symbol("global_float_ior")) {
                p.string("e.create_global_float_ior()");
            } else if (option_name == m_symbol_table->get_symbol("merge_metal_and_base_color")) {
                p.string("e.create_bool_constant(options->merge_metal_and_base_color)");
            } else if (option_name == m_symbol_table->get_symbol("merge_transmission_and_base_color")) {
                p.string("e.create_bool_constant(options->merge_transmission_and_base_color)");
            } else {
                error(expr->get_location(), "[BUG] unsupported option name");
                MDL_ASSERT(!"[BUG] unsupported option name");
            }
            return;
        }

        p.string("e.create_unary");
        p.with_parens([&] (pp::Pretty_print &p) {
                p.nl();
                p.with_indent([&] (pp::Pretty_print &p) {
                        p.string(m_api_class);
                        p.string("::");
                        switch (e->get_operator()) {
                        case Expr_unary::Operator::OK_BITWISE_COMPLEMENT:
                            p.string("OK_BITWISE_COMPLEMENT");
                            break;
                        case Expr_unary::Operator::OK_LOGICAL_NOT:
                            p.string("OK_LOGICAL_NOT");
                            break;
                        case Expr_unary::Operator::OK_POSITIVE:
                            p.string("OK_POSITIVE");
                            break;
                        case Expr_unary::Operator::OK_NEGATIVE:
                            p.string("OK_NEGATIVE");
                            break;
                        case Expr_unary::Operator::OK_PRE_INCREMENT:
                            p.string("OK_PRE_INCREMENT");
                            break;
                        case Expr_unary::Operator::OK_PRE_DECREMENT:
                            p.string("OK_PRE_DECREMENT");
                            break;
                        case Expr_unary::Operator::OK_POST_INCREMENT:
                            p.string("OK_POST_INCREMENT");
                            break;
                        case Expr_unary::Operator::OK_POST_DECREMENT:
                            p.string("OK_POST_DECREMENT");
                            break;
                        default:
                            error(expr->get_location(),
                                  "[BUG] unsupported unary operator in output_cpp_expr");
                            MDL_ASSERT(!"[BUG] unsupported unary operator in output_cpp_expr");
                            break;
                        }
                        p.comma();
                        p.nl();
                        output_cpp_expr(p, e->get_argument());
                    });
            });
        break;
    }
    case Expr::Kind::EK_BINARY:
    {
        Expr_binary const *e = cast<Expr_binary>(expr);


        // Special handling of struct field access.
        if (e->get_operator() == Expr_binary::Operator::OK_SELECT) {
            p.string("e.create_select");
            p.with_parens([&] (pp::Pretty_print &p) {
                    p.with_indent([&] (pp::Pretty_print &p) {
                            p.nl();
                            output_cpp_expr(p, e->get_left_argument());
                            p.comma();
                            p.nl();
                            p.string("\"");
                            p.string(cast<Expr_ref>(e->get_right_argument())->get_name()->get_name());
                            p.string("\"");
                        });
                });
            return;
        }

        p.string("e.create_binary");
        p.with_parens([&] (pp::Pretty_print &p) {
                p.nl();
                p.with_indent([&] (pp::Pretty_print &p) {

                        p.string(m_api_class);
                        p.string("::");
                        switch (e->get_operator()) {
                        case Expr_binary::Operator::OK_ARRAY_SUBSCRIPT:
                            p.string("OK_ARRAY_INDEX");
                            break;
                        case Expr_binary::Operator::OK_MULTIPLY:
                            p.string("OK_MULTIPLY");
                            break;
                        case Expr_binary::Operator::OK_DIVIDE:
                            p.string("OK_DIVIDE");
                            break;
                        case Expr_binary::Operator::OK_MODULO:
                            p.string("OK_MODULO");
                            break;
                        case Expr_binary::Operator::OK_PLUS:
                            p.string("OK_PLUS");
                            break;
                        case Expr_binary::Operator::OK_MINUS:
                            p.string("OK_MINUS");
                            break;
                        case Expr_binary::Operator::OK_SHIFT_LEFT:
                            p.string("OK_SHIFT_LEFT");
                            break;
                        case Expr_binary::Operator::OK_SHIFT_RIGHT:
                            p.string("OK_SHIFT_RIGHT");
                            break;
                        case Expr_binary::Operator::OK_SHIFT_RIGHT_ARITH:
                            p.string("OK_SHIFT_RIGHT_ARITH");
                            break;
                        case Expr_binary::Operator::OK_LESS:
                            p.string("OK_LESS");
                            break;
                        case Expr_binary::Operator::OK_LESS_OR_EQUAL:
                            p.string("OK_LESS_OR_EQUAL");
                            break;
                        case Expr_binary::Operator::OK_GREATER_OR_EQUAL:
                            p.string("OK_GREATER_OR_EQUAL");
                            break;
                        case Expr_binary::Operator::OK_GREATER:
                            p.string("OK_GREATER");
                            break;
                        case Expr_binary::Operator::OK_EQUAL:
                            p.string("OK_EQUAL");
                            break;
                        case Expr_binary::Operator::OK_NOT_EQUAL:
                            p.string("OK_NOT_EQUAL");
                            break;
                        case Expr_binary::Operator::OK_BITWISE_AND:
                            p.string("OK_BITWISE_AND");
                            break;
                        case Expr_binary::Operator::OK_BITWISE_OR:
                            p.string("OK_BITWISE_OR");
                            break;
                        case Expr_binary::Operator::OK_BITWISE_XOR:
                            p.string("OK_BITWISE_XOR");
                            break;
                        case Expr_binary::Operator::OK_LOGICAL_AND:
                            p.string("OK_LOGICAL_AND");
                            break;
                        case Expr_binary::Operator::OK_LOGICAL_OR:
                            p.string("OK_LOGICAL_OR");
                            break;
                        default:
                            error(expr->get_location(),
                                  "[BUG] unsupported binary operator in output_cpp_expr");
                            MDL_ASSERT(!"[BUG] unsupported binary operator in output_cpp_expr");
                            break;
                        }
                        p.comma();
                        p.nl();
                        output_cpp_expr(p, e->get_left_argument());
                        p.comma();
                        p.nl();
                        output_cpp_expr(p, e->get_right_argument());
                    });
            });
        break;
    }
    case Expr::Kind::EK_CONDITIONAL:
    {
        Expr_conditional const *e = cast<Expr_conditional>(expr);
        p.string("e.create_ternary");
        p.with_parens([&] (pp::Pretty_print &p) {
                p.nl();
                p.with_indent([&] (pp::Pretty_print &p) {

                        output_cpp_expr(p, e->get_condition());
                        p.comma();
                        p.nl();
                        output_cpp_expr(p, e->get_true());
                        p.comma();
                        p.nl();
                        output_cpp_expr(p, e->get_false());
                    });
            });
        break;
    }
    case Expr::Kind::EK_CALL:
    {
        Expr_call const *e = cast<Expr_call>(expr);

        output_cpp_call(p, e);

        break;
    }
    case Expr::Kind::EK_TYPE_ANNOTATION:
    {
        // Type annotations are simply skipped. Output the argument.
        Expr_type_annotation const *e = cast<Expr_type_annotation>(expr);
        output_cpp_expr(p, e->get_argument());
        break;
    }
    case Expr::Kind::EK_ATTRIBUTE:
    {
        Expr_attribute const *e = cast<Expr_attribute>(expr);

        // When this is reached, the code for constructing the
        // argument expression has been emitted and the result bound
        // to a variable. We just need to output the variable name
        // here.

        p.string(e->get_node_name());

        break;
    }
    }
}

void Compilation_unit::output_reversed(pp::Pretty_print &p,
                                       Argument_list::const_iterator it,
                                       Argument_list::const_iterator end) {
    if (it != end) {
        Argument_list::const_iterator next = it;
        ++next;
        output_reversed(p, next, end);

        Argument const &argument = *it;
        Expr_binary const *binding = cast<Expr_binary>(argument.get_expr());
        Expr_ref const *var = cast<Expr_ref>(binding->get_left_argument());
        Expr const *value = binding->get_right_argument();

        p.string("DAG_node const* v_");
        p.string(var->get_name()->get_name());
        p.string(" = ");
        output_cpp_expr(p, value);
        p.semicolon();
        p.nl();
    }
}

void Compilation_unit::output_cpp_matcher_rhs(
    pp::Pretty_print &p,
    Rule const &rule,
    size_t rule_index,
    size_t node_index,
    size_t cont_index)
{

    auto emit_call_cont = [&] () {
        p.with_braces([&] (pp::Pretty_print &p) {
            p.with_indent([&](pp::Pretty_print &p) {
                p.nl();
                p.string("return match_rule");
                p.integer(cont_index);
                p.string("(node");
                p.integer(node_index);
                p.string(", node_props");
                p.integer(rule_index);
                p.string(");");
            });
            p.nl();
        });
    };

    p.string("DAG_DbgInfo root_dbg_info = node");
    p.integer(node_index);
    p.string("->get_dbg_info();");
    p.nl();
    p.string("(void) root_dbg_info;");
    p.nl();

    // Generate code for creating where bindings (where expressions
    // can refere to these).
    Argument_list const &bindings = rule.get_bindings();
    if (!bindings.empty()) {
        output_reversed(p, bindings.begin(), bindings.end());
    }

    // Generate code for the rule guard (if given).
    if (Expr const *guard = rule.get_guard()) {
        Expr_unary const *g = cast<Expr_unary>(guard);
        Expr const *arg = g->get_argument();
        p.string("if (!");

        if (g->get_operator() == Expr_unary::OK_MAYBE_GUARD) {
            p.string("e.eval_maybe_if(");
        } else {
            p.string("e.eval_if(");
        }
        p.with_indent([&] (pp::Pretty_print &p) {
                output_cpp_expr(p, arg);
            });
        p.string(")) ");
        emit_call_cont();
    }

    // Generate code for debug output.
    Debug_out_list const &deb_outs = rule.get_debug_out();
    if (!deb_outs.empty()) {
        p.nl();
        p.string("if (event_handler != nullptr && options != nullptr && options->debug_print)");
        p.space();
        p.with_braces([&] (pp::Pretty_print &p) {
            p.with_indent([&] (pp::Pretty_print &p) {
                for (mi::mdl::Ast_list<Debug_out>::const_iterator it(deb_outs.begin()),
                         end(deb_outs.end());
                     it != end;
                     ++it) {
                    p.nl();
                    char const *var_name = it->get_name();
                    p.string("fire_debug_print(e, *event_handler,");
                    p.space();
                    p.integer(rule_index);
                    p.comma();
                    p.space();
                    p.string("\"");
                    p.string(var_name);
                    p.string("\", v_");
                    p.string(var_name);
                    p.string(");");
                }
            });
            p.nl();
        });
    }

    // Generate tracer code.
    p.nl();
    p.string("if (event_handler != nullptr)");
    p.with_indent([&] (pp::Pretty_print &p) {
            p.nl();
            p.string("fire_match_event(*event_handler, ");
            p.integer(rule_index);
            p.string(");");
        });
    p.nl();


    switch (rule.get_result_code()) {
    case Rule::Result_code::RC_NO_RESULT_CODE:
        break;
    case Rule::Result_code::RC_SKIP_RECURSION:
        p.string_with_nl("result_code = RULE_SKIP_RECURSION;\n");
        break;
    case Rule::Result_code::RC_REPEAT_RULES:
        p.string_with_nl("result_code = RULE_REPEAT_RULES;\n");
        break;
    }

    output_cpp_expr_bindings(p, rule.get_rhs());

    p.string("return ");
    output_cpp_expr(p, rule.get_rhs());
    p.semicolon();
}

void Compilation_unit::output_cpp_rule_match1(
    pp::Pretty_print &p,
    Rule const &rule,
    bool first_matcher,
    size_t skip_tl,
    size_t rule_index,
    size_t node_index,
    size_t fail_node_index,
    size_t cont_index,
    Expr const *node,
    size_t &tmp_index)
{
    tmp_index += 1;

    // Destructure LHS into
    // 1. Name of top-level node (if any)
    // 2. Call expression for top-level node
    // 3. Attribute expression (if any).
    Expr const *node_expr = nullptr;
    Expr_attribute const *attr_expr = nullptr;
    Expr_ref const *name_expr = nullptr;
    {
        Expr const *e = node;
        while (true) {
            if (Expr_attribute const* a = as<Expr_attribute>(e)) {
                attr_expr = a;
                e = a->get_argument();
            } else if (Expr_binary const* b = as<Expr_binary>(e)) {
                if (b->get_operator() == Expr_binary::Operator::OK_TILDE) {
                    name_expr = cast<Expr_ref>(b->get_left_argument());
                    e = b->get_right_argument();
                } else {
                    break;
                }
            } else if (Expr_call const* c = as<Expr_call>(e)) {
                node_expr = c;
                break;
            } else if (Expr_type_annotation const* a = as<Expr_type_annotation>(e)) {
                e = a->get_argument();
            } else if (Expr_ref const* r = as<Expr_ref>(e)) {
                node_expr = r;
                break;
            } else {
                printf("kind: %d\n", e->get_kind());
                MDL_ASSERT(!"call, attr or alias expression expected");
            }
        }
        MDL_ASSERT(node_expr && "no call or reference expression found");
    }

    auto emit_call_cont = [&] (size_t to_skip = 0) {
        p.with_braces([&] (pp::Pretty_print &p) {
            p.with_indent([&] (pp::Pretty_print&p) {
                p.nl();
                p.string("return match_rule");
                p.integer(cont_index + to_skip);
                p.string("(node");
                p.integer(fail_node_index);
                p.string(", node_props");
                p.integer(fail_node_index);
                p.string(");");
            });
            p.nl();
        });
    };

    if (name_expr != nullptr) {
        p.string("DAG_node const *v_");
        p.string(name_expr->get_name()->get_name());
        p.string(" = node");
        p.integer(node_index);
        p.string(";");
    }

    switch (node_expr->get_kind()) {
    case Expr::EK_CALL:
    {
        Expr_call const *call = cast<Expr_call>(node_expr);
        Type const *ret_type = call->get_type();
        Expr const *callee = call->get_callee();
        Type_function const *callee_type = cast<Type_function>(callee->get_type());

        p.nl();
        p.string("// ");
        if (!first_matcher) {
            p.string("continued ");
        }
        p.string("match for ");
        {
            std::stringstream s_out;
            pp::Pretty_print p1(m_arena, s_out, pp::Pretty_print::LARGE_LINE_WIDTH);
            node_expr->pp(p1);
            p.string(s_out.str().c_str());
        }

        if (first_matcher) {
            // First, check a node's semantics.
            mi::mdl::IDefinition::Semantics sema =
                mi::mdl::IDefinition::Semantics(find_semantics(call));
            bool is_ternary = sema >= mi::mdl::IDefinition::DS_OP_BASE && sema <= mi::mdl::IDefinition::DS_OP_END;


            char const *mdl_semantics_name = get_semantics_name(sema);
            p.nl();
            p.string("if (node_props"); 
            p.integer(node_index); 
            p.string(".sema != ");
            if (mdl_semantics_name) {
                p.string("IDefinition::");
                p.string(mdl_semantics_name);
            } else if (is_ternary) {
                mi::mdl::IExpression::Operator op = mi::mdl::IExpression::Operator(sema - mi::mdl::IDefinition::DS_OP_BASE);
                switch (op) {
                case mi::mdl::IExpression::OK_TERNARY:
                    p.string("(IDefinition::DS_OP_BASE + IExpression::OK_TERNARY)");
                    break;
                default:
                    p.integer(sema);
                    p.string("/* unhandled */");
                    break;
                }
            } else {

//                MDL_ASSERT(!"unhandled semantics in sema->string conversion");
                p.integer(sema);
                p.string("/* unhandled */");
            }

            // Check the node's type if needed:
            if (sema == mi::mdl::IDefinition::DS_INVALID_REF_CONSTRUCTOR) {
                p.softbreak();
                p.string(" || node_props"); p.integer(node_index); p.string(".type_kind != ");
                switch (ret_type->get_kind()) {
                case Type::TK_BSDF:
                    p.string("IType::TK_BSDF");
                    break;
                case Type::TK_EDF:
                    p.string("IType::TK_EDF");
                    break;
                case Type::TK_VDF:
                    p.string("IType::TK_VDF");
                    break;
                case Type::TK_HAIR_BSDF:
                    p.string("IType::TK_HAIR_BSDF");
                    break;
                case Type::TK_STRUCT:
                    p.string("IType::TK_STRUCT");
                    break;
                default:
                    p.integer(int(ret_type->get_kind()));
                    break;
                }
            } else if (sema == mi::mdl::IDefinition::DS_ELEM_CONSTRUCTOR) {
                p.softbreak();
                p.string(" || node_props"); p.integer(node_index); p.string(".type_kind != ");
                switch (ret_type->get_kind()) {
                case Type::TK_STRUCT:
                    p.string("IType::TK_STRUCT");
                    break;
                case Type::TK_MATERIAL:
                    p.string("IType::TK_STRUCT || node_props"); p.integer(node_index); p.string(".struct_id != IType_struct::SID_MATERIAL");
                    break;
                case Type::TK_MATERIAL_SURFACE:
                    p.string("IType::TK_STRUCT || node_props"); p.integer(node_index); p.string(".struct_id != IType_struct::SID_MATERIAL_SURFACE");
                    break;
                case Type::TK_MATERIAL_VOLUME:
                    p.string("IType::TK_STRUCT || node_props"); p.integer(node_index); p.string(".struct_id != IType_struct::SID_MATERIAL_VOLUME");
                    break;
                case Type::TK_MATERIAL_GEOMETRY:
                    p.string("IType::TK_STRUCT || node_props"); p.integer(node_index); p.string(".struct_id != IType_struct::SID_MATERIAL_GEOMETRY");
                    break;
                case Type::TK_MATERIAL_EMISSION:
                    p.string("IType::TK_STRUCT || node_props"); p.integer(node_index); p.string(".struct_id != IType_struct::SID_MATERIAL_EMISSION");
                    break;
                default:
                    MDL_ASSERT(!"unexpected type");
                    p.integer(int(ret_type->get_kind()));
                    break;
                }
            } else if ((sema == mi::mdl::IDefinition::DS_INTRINSIC_DF_NORMALIZED_MIX)
                || (sema == mi::mdl::IDefinition::DS_INTRINSIC_DF_CLAMPED_MIX)
                || (sema == mi::mdl::IDefinition::DS_INTRINSIC_DF_COLOR_NORMALIZED_MIX)
                || (sema == mi::mdl::IDefinition::DS_INTRINSIC_DF_COLOR_CLAMPED_MIX)
                || (sema == mi::mdl::IDefinition::DS_INTRINSIC_DF_UNBOUNDED_MIX)
                || (sema == mi::mdl::IDefinition::DS_INTRINSIC_DF_COLOR_UNBOUNDED_MIX)) {
                p.softbreak();
                p.string(" || node_props"); p.integer(node_index); p.string(".arity != ");
                p.integer(call->get_argument_count() / 2);
                p.softbreak();
                p.string(" || node_props"); p.integer(node_index); p.string(".type_kind != ");
                switch (ret_type->get_kind()) {
                case Type::TK_BSDF:
                    p.string("IType::TK_BSDF");
                    break;
                case Type::TK_EDF:
                    p.string("IType::TK_EDF");
                    break;
                case Type::TK_VDF:
                    p.string("IType::TK_VDF");
                    break;
                default:
                    MDL_ASSERT(!"unhandled type kind in mixer");
                    break;
                }
            } else if ((sema == mi::mdl::IDefinition::DS_INTRINSIC_DF_TINT)
                || (sema == mi::mdl::IDefinition::DS_INTRINSIC_DF_DIRECTIONAL_FACTOR)) {
                p.string(" || node_props"); p.integer(node_index); p.string(".arity != ");
                p.integer(call->get_argument_count());
                p.string(" || node_props"); p.integer(node_index); p.string(".type_kind != ");
                switch (ret_type->get_kind()) {
                case Type::TK_BSDF:
                    p.string("IType::TK_BSDF");
                    break;
                case Type::TK_EDF:
                    p.string("IType::TK_EDF");
                    break;
                case Type::TK_VDF:
                    p.string("IType::TK_VDF");
                    break;
                default:
                    MDL_ASSERT(!"unhandled type kind in tint/directional_factor");
                    break;
                }
            } else if (is_ternary) {
                p.softbreak();
                p.string(" || node_props"); p.integer(node_index); p.string(".type_kind != ");
                switch (ret_type->get_kind()) {
                case Type::TK_BSDF:
                    p.string("IType::TK_BSDF");
                    break;
                case Type::TK_EDF:
                    p.string("IType::TK_EDF");
                    break;
                case Type::TK_VDF:
                    p.string("IType::TK_VDF");
                    break;
                case Type::TK_HAIR_BSDF:
                    p.string("IType::TK_HAIR_BSDF");
                    break;
                case Type::TK_MATERIAL: // For material structs.
                    p.string("IType::TK_STRUCT || node_props"); p.integer(node_index); p.string(".struct_id != IType_struct::SID_MATERIAL");
                    break;
                default:
                    MDL_ASSERT(!"unhandled type kind in ternary");
                    break;
                }
            }
            p.string(") ");
            emit_call_cont(skip_tl);
        }

        for (int i = 0; i < call->get_argument_count(); i++) {
            Expr const *arg = call->get_argument(i);
            if (Expr_ref const *ref = as<Expr_ref>(arg)) {
                if (strcmp(ref->get_name()->get_name(), "_") == 0) {
                    continue;
                }
            }
            p.nl();
            p.string("DAG_node const *node");
            p.integer(tmp_index);
            p.string(" = ");
            if (get_n_ary_mixer(m_node_types, callee_type) > 0) {
                p.string("e.get_remapped_argument(");
            } else {
                p.string("e.get_compound_argument(");
            }
            p.string("node");
            p.integer(node_index);
            p.string(", ");
            p.integer(i);
            p.string(");");
            if (arg->get_kind() != Expr::EK_REFERENCE) {
                p.string_with_nl("\nIDistiller_plugin_api::Match_properties node_props");
                p.integer(tmp_index);
                p.string_with_nl(";\n");
                p.string("e.get_match_properties(node");
                p.integer(tmp_index);
                p.string(", node_props");
                p.integer(tmp_index);
                p.string("); ");
            }
            output_cpp_rule_match1(p, rule, true, 0, rule_index, tmp_index,
                                   fail_node_index,
                                   cont_index, arg,
                                   tmp_index);
            tmp_index++;
        }
        break;
    }

    case Expr::EK_REFERENCE:
    {
        Expr_ref const *ref = cast<Expr_ref>(node_expr);
        if (strcmp(ref->get_name()->get_name(), "_") != 0) {
            p.nl();
            p.string("DAG_node const *v_");
            p.escaped_string(ref->get_name()->get_name());
            p.string(" = node");
            p.integer(node_index);
            p.semicolon();
            p.string(" (void)v_");
            p.escaped_string(ref->get_name()->get_name());
            p.semicolon();
        }
        break;
    }
    default:
        MDL_ASSERT(false && "unexpected expression kind");
    }

    // Match all attributes and bind their values to variables.
    if (attr_expr != nullptr) {
        Expr_attribute::Expr_attribute_vector const &attrs = attr_expr->get_attributes();
        for (Expr_attribute::Expr_attribute_entry const &ap : attrs) {
            p.nl();
            p.string("if (!e.attribute_exists(");
            p.string("node");
            p.integer(node_index);
            p.string(", \"");
            p.string(ap.name->get_name());
            p.string("\")) ");
            emit_call_cont();

            if (ap.expr) {
                p.nl();
                p.string("const DAG_node *node");
                p.integer(tmp_index);
                p.string(" = e.get_attribute(node");
                p.integer(node_index);
                p.string(", \"");
                p.escaped_string(ap.name->get_name());
                p.string("\");");
                p.string(" (void)node");
                p.integer(tmp_index);
                p.semicolon();
                size_t ti = tmp_index;
                tmp_index++;
                output_cpp_rule_match1(p, rule, true, 0, rule_index, ti,
                                       fail_node_index,
                                       cont_index,
                                       ap.expr,
                                       tmp_index);
            }
        }
    }

}

void Compilation_unit::output_cpp_rule_matcher(
    pp::Pretty_print &p,
    Rule const &rule,
    bool first_matcher,
    size_t skip_tl,
    size_t rule_index,
    size_t cont_index)
{
    Expr const *lhs = rule.get_lhs();
    size_t tmp_index = rule_index + 1;

    p.without_indent([&] (pp::Pretty_print &p) {
        p.string("// ");
        p.string(m_filename_only);
        p.string(":");

        p.integer(rule.get_location().get_line());
        p.nl();
        p.string("//");
        if (rule.get_dead_rule() == Rule::Dead_rule::DR_DEAD)
            p.string(" deadrule ");
        p.string("RUID ");
        p.integer(rule.get_uid());
    });
    p.nl();
    p.string("auto match_rule");
    p.integer(rule_index);
    p.string(" = [&] (DAG_node const *node");
    p.integer(rule_index);
    p.string(", IDistiller_plugin_api::Match_properties &node_props");
    p.integer(rule_index);
    p.string(") -> const DAG_node * ");
    p.with_braces([&] (pp::Pretty_print &p) {
        p.with_indent([&] (pp::Pretty_print &p) {
            p.nl();
            output_cpp_rule_match1(p, rule, first_matcher, skip_tl,
                                   /*rule_index=*/rule_index,
                                   /*node_index=*/rule_index,
                                   /*fail_node_index=*/rule_index,
                                   /*cont_index=*/cont_index,
                                   lhs,
                                   tmp_index);
            p.nl();
            mi::mdl::string s(m_arena.get_allocator());
            output_cpp_matcher_rhs(p, rule, rule_index, rule_index, cont_index);
        });
        p.nl();
    });
    p.semicolon();
    p.nl();
    p.string("(void)match_rule");
    p.integer(rule_index);
    p.semicolon();
    p.nl();
}

static mi::mdl::IType::Kind map_kind(Type::Kind k) {
    switch (k) {
    case Type::TK_ERROR:
        return mi::mdl::IType::TK_ERROR;
    case Type::TK_BOOL:
        return mi::mdl::IType::TK_BOOL;
    case Type::TK_INT:
        return mi::mdl::IType::TK_INT;
    case Type::TK_ENUM:
        return mi::mdl::IType::TK_ENUM;
    case Type::TK_FLOAT:
        return mi::mdl::IType::TK_FLOAT;
    case Type::TK_DOUBLE:
        return mi::mdl::IType::TK_DOUBLE;
    case Type::TK_STRING:
        return mi::mdl::IType::TK_STRING;
    case Type::TK_LIGHT_PROFILE:
        return mi::mdl::IType::TK_LIGHT_PROFILE;
    case Type::TK_BSDF:
        return mi::mdl::IType::TK_BSDF;
    case Type::TK_HAIR_BSDF:
        return mi::mdl::IType::TK_HAIR_BSDF;
    case Type::TK_EDF:
        return mi::mdl::IType::TK_EDF;
    case Type::TK_VDF:
        return mi::mdl::IType::TK_VDF;
    case Type::TK_VECTOR:
        return mi::mdl::IType::TK_VECTOR;
    case Type::TK_MATRIX:
        return mi::mdl::IType::TK_MATRIX;
    case Type::TK_ARRAY:
        return mi::mdl::IType::TK_ARRAY;
    case Type::TK_COLOR:
        return mi::mdl::IType::TK_COLOR;
    case Type::TK_FUNCTION:
        return mi::mdl::IType::TK_FUNCTION;
    case Type::TK_STRUCT:
        return mi::mdl::IType::TK_STRUCT;
    case Type::TK_TEXTURE:
        return mi::mdl::IType::TK_TEXTURE;
    case Type::TK_BSDF_MEASUREMENT:
        return mi::mdl::IType::TK_BSDF_MEASUREMENT;
    case Type::TK_MATERIAL_EMISSION:
        return mi::mdl::IType::TK_STRUCT;
    case Type::TK_MATERIAL_SURFACE:
        return mi::mdl::IType::TK_STRUCT;
    case Type::TK_MATERIAL_VOLUME:
        return mi::mdl::IType::TK_STRUCT;
    case Type::TK_MATERIAL_GEOMETRY:
        return mi::mdl::IType::TK_STRUCT;
    case Type::TK_MATERIAL:
        return mi::mdl::IType::TK_STRUCT;
    default:
        return mi::mdl::IType::TK_ERROR;
    }
}

// Calculate the matching properties of the top-level call node of `rule`.
// They are returned in the given out parameters.
static void get_node_properties(Rule const &rule,
                            mi::mdl::IDefinition::Semantics &sema,
                            mi::mdl::IType::Kind &tkind,
                            mi::mdl::IType_struct::Predefined_id &struct_id,
                            size_t &arity)
{
    Expr const *expr = rule.get_lhs();

    sema = mi::mdl::IDefinition::DS_UNKNOWN;
    tkind = mi::mdl::IType::TK_ERROR;
    struct_id = mi::mdl::IType_struct::SID_USER;
    arity = 0;

    // Skip all attributes, node aliases and type annotations.
    for (;;) {
        if (Expr_attribute const* call = as<Expr_attribute>(expr)) {
            expr = call->get_argument();
        } else if (Expr_binary const* eb = as<Expr_binary>(expr)) {
            if (eb->get_operator() == Expr_binary::Operator::OK_TILDE) {
                expr = eb->get_right_argument();
            }
            else {
                break;
            }
        } else if (Expr_type_annotation const* a = as<Expr_type_annotation>(expr)) {
            expr = a->get_argument();
        } else {
            break;
        }
    }

    if (Expr_call const* call = as<Expr_call>(expr)) {
        expr = call->get_callee();
    } else {
        return;
    }

    if (is<Expr_ref>(expr)) {
        Type* expr_t = expr->get_type();
        if (Type_function* tf = as<Type_function>(expr_t)) {
            arity = tf->get_parameter_count();
            char const* sel = tf->get_selector();
            if (sel) {
                sema = tf->get_semantics();
                Type::Kind k = tf->get_return_type()->get_kind();
                switch (k) {
                case Type::TK_MATERIAL:
                    tkind = mi::mdl::IType::TK_STRUCT;
                    struct_id = mi::mdl::IType_struct::SID_MATERIAL;
                    break;
                case Type::TK_MATERIAL_SURFACE:
                    tkind = mi::mdl::IType::TK_STRUCT;
                    struct_id = mi::mdl::IType_struct::SID_MATERIAL_SURFACE;
                    break;
                case Type::TK_MATERIAL_GEOMETRY:
                    tkind = mi::mdl::IType::TK_STRUCT;
                    struct_id = mi::mdl::IType_struct::SID_MATERIAL_GEOMETRY;
                    break;
                case Type::TK_MATERIAL_VOLUME:
                    tkind = mi::mdl::IType::TK_STRUCT;
                    struct_id = mi::mdl::IType_struct::SID_MATERIAL_VOLUME;
                    break;
                case Type::TK_MATERIAL_EMISSION:
                    tkind = mi::mdl::IType::TK_STRUCT;
                    struct_id = mi::mdl::IType_struct::SID_MATERIAL_EMISSION;
                    break;
                default:
                    tkind = map_kind(k);
                    break;
                }
                return;
            } else {
                MDL_ASSERT(!"null selector found when looking up selector");
                return;
            }
        } else {
            MDL_ASSERT(!"invalid (non-function) argument found when looking up selector");
            return;
        }
    } else {
        MDL_ASSERT(!"invalid (non-node) argument found when looking up selector");
        return;
    }
}

void Compilation_unit::output_cpp_matcher(
    pp::Pretty_print &p,
    Ruleset &ruleset, mi::mdl::vector<Rule const *>::Type &rules) 
{
    // Write function header for matcher function and start switch
    // statement on rules.

    p.string("// Run the matcher.");
    p.nl();
    p.string("DAG_node const* ");
    p.string(ruleset.get_name());
    p.string("::matcher");
    p.with_parens([&] (pp::Pretty_print &p) {
        p.with_indent([&] (pp::Pretty_print &p) {
            p.nl();
            p.string_with_nl("IRule_matcher_event *event_handler,\n");
            p.string(m_api_class);
            p.string_with_nl(
                " &e,\n"
                "DAG_node const *node,\n"
                "const mi::mdl::Distiller_options *options,\n"
                "Rule_result_code &result_code");
        });
    });
    p.string_with_nl(" const\n");
    p.with_braces([&] (pp::Pretty_print &p) {
        p.with_indent([&] (pp::Pretty_print &p) {
            p.nl();
            {
                size_t i = 0;
                mi::mdl::IDefinition::Semantics last_sema =
                    mi::mdl::IDefinition::Semantics::DS_UNKNOWN;
                mi::mdl::IType::Kind last_tkind =
                    mi::mdl::IType::TK_ERROR;
                mi::mdl::IType_struct::Predefined_id last_struct_id = mi::mdl::IType_struct::SID_USER;
                size_t last_arity = 0;
                mi::mdl::vector<bool>::Type first_matcher(get_allocator());
                mi::mdl::vector<size_t>::Type skip_tl(get_allocator());

                // Calculate, for each rule, the node matching properties of the LHS. Then 
                // collect information on which rules can be combined because their top-level 
                // match properties are equal. The entry first_matcher[i] is true if rule i
                // is the first with a certain set of matching properties, and needs the 
                // matching code.
                for (auto it(rules.begin()), end(rules.end()); it != end; ++it, ++i) {
                    Rule const &rule = **it;
                    mi::mdl::IDefinition::Semantics sema =
                        mi::mdl::IDefinition::Semantics::DS_UNKNOWN;
                    mi::mdl::IType::Kind tkind =
                        mi::mdl::IType::TK_ERROR;
                    mi::mdl::IType_struct::Predefined_id struct_id = mi::mdl::IType_struct::SID_USER;
                    size_t arity = 0;
                    get_node_properties(rule, sema, tkind, struct_id, arity);
                    if (sema != last_sema || tkind != last_tkind || struct_id != last_struct_id || arity != last_arity) {
                        last_sema = sema;
                        last_tkind = tkind;
                        last_arity = arity;
                        first_matcher.push_back(true);
                    } else {
                        first_matcher.push_back(false);
                    }
                    skip_tl.push_back(0);
                }

                // Calculate for each rule i where first_matcher[i] is true the number of 
                // rules to skip if the top-level matching fails.
                for (int i = 0; i < first_matcher.size(); ++i) {
                    if (first_matcher[i]) {
                        for (int j = i; j < first_matcher.size() - 1; ++j) {
                            if (first_matcher[j + 1] == false) {
                                skip_tl[i]++;
                            } else {
                                break;
                            }
                        }
                    }
                }

                p.string("auto match_rule");
                p.integer(rules.size());
                p.string(" = [&] (DAG_node const *node, \
IDistiller_plugin_api::Match_properties &node_props) -> \
const DAG_node * { return node; };");
                p.nl();
                i = rules.size();
                for (auto it(rules.rbegin()), end(rules.rend()); it != end; ++it, --i) {
                    p.nl();

                    Rule const &rule = **it;
                    output_cpp_rule_matcher(p, rule,
                                            first_matcher[i - 1],
                                            skip_tl[i - 1],
                                            i - 1, i);
                }
                p.nl();
                p.string_with_nl("IDistiller_plugin_api::Match_properties node_props;\n");
                p.string_with_nl("e.get_match_properties(node, node_props);\n");
                p.string("return match_rule");
                p.integer(0);
                p.string("(node, node_props);");
                p.nl();
            }

        });
        p.nl();
    });
    p.nl();
    p.nl();
}

void Compilation_unit::output_cpp_pattern_condition(
    pp::Pretty_print &p,
    Expr const *expr,
    mi::mdl::string const &prefix)
{
    if (Expr_call const *call = as<Expr_call>(expr)) {
        Expr const *callee = call->get_callee();
        Type_function const *callee_type = cast<Type_function>(callee->get_type());
        mi::mdl::Node_type const *nt = callee_type->get_node_type();
//        printf("nt: %s\n", nt->mdl_type_name.c_str());
        if (!nt) {
            error(expr->get_location(),
                  "[BUG] NULL node_type in output_cpp_pattern_condition");
            MDL_ASSERT(!"[BUG] NULL node_type in output_cpp_pattern_condition");
        }

        for (int i = 0; i < call->get_argument_count(); i++) {
            Expr const *arg = call->get_argument(i);

            char b[32];
            snprintf(b, sizeof(b), "%d", i);

            mi::mdl::string prefix2(m_arena.get_allocator());

            if (get_n_ary_mixer(m_node_types, callee_type) > 0) {
                prefix2 = "e.get_remapped_argument(";
            } else {
                prefix2 = "e.get_compound_argument(";
            }
            prefix2 += prefix + ", " + b + ")";

            if (is<Expr_call>(arg)) {
                p.nl();
                p.oper("&&");
                p.space();
                p.with_parens([&] (pp::Pretty_print &p) {
                    p.string("e.get_selector(");
                    p.string(prefix2.c_str());
                    p.string(") == ");
                    p.string(find_selector(arg));
                });
                output_cpp_pattern_condition(p, arg, prefix2);
            } else {
                output_cpp_pattern_condition(p, arg, prefix2);
            }
        }
    } else if (Expr_binary const *eb = as<Expr_binary>(expr)) {
        if (eb->get_operator() == Expr_binary::Operator::OK_TILDE) {
            output_cpp_pattern_condition(p, eb->get_right_argument(), prefix);
        } else {
            MDL_ASSERT(!"unexpected binary operator in Compilation_unit::output_cpp_pattern_condition");
        }
    } else if (Expr_attribute const *attr = as<Expr_attribute>(expr)) {
        Expr const *arg = attr->get_argument();

        if (is<Expr_call>(arg)) {
            p.nl();
            p.oper("&&");
            p.space();
            p.with_parens([&] (pp::Pretty_print &p) {
                p.string("e.get_selector(");
                p.string(prefix.c_str());
                p.string(") == ");
                p.string(find_selector(arg));
            });
            output_cpp_pattern_condition(p, arg, prefix);
        } else {
            output_cpp_pattern_condition(p, arg, prefix);
        }

        for (Expr_attribute::Expr_attribute_entry const & ap : attr->get_attributes()) {

            const std::string &attr_name = ap.name->get_name();
            Expr const *attr_pat = ap.expr;

            p.nl();
            p.oper("&&");
            p.space();
            p.with_parens([&] (pp::Pretty_print &p) {
                p.string("e.attribute_exists(");
                p.string(prefix.c_str());
                p.string(", \"");
                p.string(attr_name.c_str());
                p.string("\")");
                if (attr_pat) {
                    if (is<Expr_call>(attr_pat)) {
                        p.nl();
                        p.oper("&&");
                        p.space();
                        p.with_parens([&] (pp::Pretty_print &p) {
                            p.string("e.get_selector(e.get_attribute(");
                            p.string(prefix.c_str());
                            p.string(", \"");
                            p.string(attr_name.c_str());
                            p.string("\")");
                            p.string(") == ");
                            p.string(find_selector(attr_pat));
                        });
                    }
                    output_cpp_pattern_condition(p, attr_pat, prefix);
                }
            });
        }
    } else if (Expr_type_annotation const *ea = as<Expr_type_annotation>(expr)) {
        output_cpp_pattern_condition(p, ea->get_argument(), prefix);
    } else if (is<Expr_ref>(expr)) {
        /* Ignore pattern variables - they always match. */
    } else {
        {
            pp::Pretty_print p(m_arena, std::cerr, 80);
            expr->pp(p);
        }
        MDL_ASSERT(!"unexpected node kind in Compilation_unit::output_cpp_pattern_condition");
    }
}

int Compilation_unit::find_semantics(Expr const *expr) {
    // Ignore all attributes and node aliases.
    for (;;) {
        if (Expr_attribute const *call = as<Expr_attribute>(expr)) {
            expr = call->get_argument();
        } else if (Expr_binary const *eb = as<Expr_binary>(expr)) {
            if (eb->get_operator() == Expr_binary::Operator::OK_TILDE) {
                expr = eb->get_right_argument();
            } else {
                break;
            }
        } else {
            break;
        }
    }

    if (Expr_call const *call = as<Expr_call>(expr)) {
        expr = call->get_callee();
    }

    MDL_ASSERT(is<Expr_ref>(expr) && "non-ref expression");
    Type *expr_t = expr->get_type();
    Type_function *tf = cast<Type_function>(expr_t);
    return tf->get_semantics();
}


char const *Compilation_unit::find_selector(Expr const *expr) {
    // Ignore all attributes and node aliases.
    for (;;) {
        if (Expr_attribute const *call = as<Expr_attribute>(expr)) {
            expr = call->get_argument();
        } else if (Expr_binary const *eb = as<Expr_binary>(expr)) {
            if (eb->get_operator() == Expr_binary::Operator::OK_TILDE) {
                expr = eb->get_right_argument();
            } else {
                break;
            }
        } else {
            break;
        }
    }

    if (Expr_call const *call = as<Expr_call>(expr)) {
        expr = call->get_callee();
    }

    MDL_ASSERT(is<Expr_ref>(expr) && "non-reference expression");
    Type *expr_t = expr->get_type();
    Type_function *tf = cast<Type_function>(expr_t);
    char const *sel = tf->get_selector();
    MDL_ASSERT(sel && "non-null selector expected");
    return sel;
}

void Compilation_unit::output_cpp_postcond_helpers(pp::Pretty_print &p,
                                                   Ruleset &ruleset,
                                                   Expr *expr,
                                                   int &idx)
{
    switch (expr->get_kind()) {
    case Expr::Kind::EK_UNARY:
    {
        Expr_unary *e = cast<Expr_unary>(expr);
        switch (e->get_operator()) {
        case Expr_unary::Operator::OK_MATCH:
        case Expr_unary::Operator::OK_NONODE:
        {

            p.string("bool ");
            p.string(ruleset.get_name());
            p.string("::checker_");
            p.integer(idx);
            p.lparen();
            p.with_indent([&] (pp::Pretty_print &p) {
                    p.nl();
                    p.string("mi::mdl::");
                    p.string(m_api_class);
                    p.string_with_nl(" &e,\nmi::mdl::DAG_node const *node)");
                });
            p.nl();
            p.with_braces([&] (pp::Pretty_print &p) {
                    p.with_indent([&] (pp::Pretty_print &p) {
                            p.string_with_nl("\nint selector = e.get_selector(node);\n");

                            Expr *argument = e->get_argument();

                            char const *selector = find_selector(argument);

                            p.string("bool result = ");
                            if ( e->get_operator() == Expr_unary::Operator::OK_MATCH) {
                                p.string(" ((selector == ");
                                p.string(selector);
                                p.string(")");

                                mi::mdl::string prefix(m_arena.get_allocator());
                                prefix = "node";
                                output_cpp_pattern_condition(p, argument, prefix);

                                p.rparen();
                                p.semicolon();
                                p.nl();
#if MDLTLC_DEBUG_POSTCOND
                                p.string("printf(\"[");
                                p.string(ruleset.get_name());
                                p.string(": checking match(");
                                p.string(selector);
                                p.string(")=%d]\\n\", result);");
                                p.nl();
#endif
                            } else { // NONODE_KIND
                                p.string("selector != ");
                                p.string(selector);
                                p.semicolon();
                                p.nl();

#if MDLTLC_DEBUG_POSTCOND
                                p.string("printf(\"[");
                                p.string(ruleset.get_name());
                                p.string(": checking nonode(");
                                p.string(selector);
                                p.string(")=%d]\\n\", result);");
                                p.nl();
#endif
                            }
                            p.nl();
                            p.string_with_nl("return result;\n");
                        });
                    p.nl();
                });
            p.nl();
            p.nl();

            idx++;

            break;
        }
        default:
            error(expr->get_location(), 
                  "[BUG] invalid unary operator encountered in postcondition expression");
            MDL_ASSERT(!"[BUG] invalid unary operator encountered in postcondition expression");
            break;
        }
        break;
    }

    case Expr::Kind::EK_BINARY:
    {
        Expr_binary *e = cast<Expr_binary>(expr);

        output_cpp_postcond_helpers(p, ruleset, e->get_left_argument(), idx);

        switch (e->get_operator()) {
        case Expr_binary::OK_LOGICAL_AND:
            break;

        case Expr_binary::OK_LOGICAL_OR:
            break;

        default:
            error(expr->get_location(),
                  "[BUG] invalid binary operator encountered in postcondition expression");
            MDL_ASSERT(!"[BUG] invalid binary operator encountered in postcondition expression");
            break;
        }

        output_cpp_postcond_helpers(p, ruleset, e->get_right_argument(), idx);

        break;
    }
    case Expr::Kind::EK_INVALID:
    case Expr::Kind::EK_LITERAL:
    case Expr::Kind::EK_REFERENCE:
    case Expr::Kind::EK_CONDITIONAL:
    case Expr::Kind::EK_CALL:
    case Expr::Kind::EK_TYPE_ANNOTATION:
    case Expr::Kind::EK_ATTRIBUTE:
        error(expr->get_location(),
              "[BUG] invalid expression encountered in postcondition");
        MDL_ASSERT(!"[BUG] invalid expression encountered in postcondition");
        break;
    }

}

void Compilation_unit::output_cpp_postcond_expr(pp::Pretty_print &p,
                                                Ruleset &ruleset,
                                                Expr *expr,
                                                int &idx) {
    switch (expr->get_kind()) {
    case Expr::Kind::EK_UNARY:
    {
        Expr_unary *e = cast<Expr_unary>(expr);

        switch (e->get_operator()) {
        case Expr_unary::Operator::OK_MATCH:
            p.string("checker_");
            p.integer(idx);
            p.string("(e, root)");
            idx++;
            break;

        case Expr_unary::Operator::OK_NONODE:
            p.string( "e.all_nodes(checker_");
            p.integer(idx);
            p.string(", root)");
            idx++;

            break;

        default:
            error(expr->get_location(),
                  "[BUG] invalid unary operator encountered in postcondition expression");
            MDL_ASSERT(!"[BUG] invalid unary operator encountered in postcondition expression");
            break;
        }
        break;
    }

    case Expr::Kind::EK_BINARY:
    {
        Expr_binary *e = cast<Expr_binary>(expr);
        bool paren_left =  e->get_left_argument()->get_prio() > 0 && e->get_left_argument()->get_prio() > e->get_prio();
        bool paren_right = e->get_right_argument()->get_prio() > 0 && e->get_right_argument()->get_prio() >= e->get_prio();

        if (paren_left) {
            p.lparen();
        }
        output_cpp_postcond_expr(p, ruleset, e->get_left_argument(), idx);
        if (paren_left) {
            p.rparen();
        }

        switch (e->get_operator()) {
        case Expr_binary::OK_LOGICAL_AND:
            p.oper("&&", true);
            break;

        case Expr_binary::OK_LOGICAL_OR:
            p.oper("||", true);
            break;

        default:
            error(expr->get_location(),
                  "[BUG] invalid binary operator encountered in postcondition expression");
            MDL_ASSERT(!"[BUG] invalid binary operator encountered in postcondition expression");
            break;
        }

        if (paren_right) {
            p.lparen();
        }
        output_cpp_postcond_expr(p, ruleset, e->get_right_argument(), idx);
        if (paren_right) {
            p.rparen();
        }

        break;
    }
    case Expr::Kind::EK_INVALID:
    case Expr::Kind::EK_LITERAL:
    case Expr::Kind::EK_REFERENCE:
    case Expr::Kind::EK_CONDITIONAL:
    case Expr::Kind::EK_CALL:
    case Expr::Kind::EK_TYPE_ANNOTATION:
    case Expr::Kind::EK_ATTRIBUTE:
        error(expr->get_location(),
              "[BUG] invalid expression encountered in postcondition");
        MDL_ASSERT(!"[BUG] invalid expression encountered in postcondition");
        break;
    }

}

void Compilation_unit::output_cpp_postcond(pp::Pretty_print &p,
                                           Ruleset &ruleset) {
    int idx = 0;

    if (!ruleset.get_postcond().is_empty())
        output_cpp_postcond_helpers(p, ruleset, ruleset.get_postcond().get_expr(), idx);

    p.string("bool ");
    p.string(ruleset.get_name());
    p.string("::postcond");
    p.with_indent([&] (pp::Pretty_print &p) {
            p.with_parens([&] (pp::Pretty_print &p) {
                    p.nl();
                    p.string("IRule_matcher_event *event_handler,");
                    p.nl();
                    p.string(m_api_class);
                    p.string(" &e,");
                    p.nl();
                    p.string("DAG_node const *root,");
                    p.nl();
                    p.string("const mi::mdl::Distiller_options *options");
                });
            p.string(" const");
        });
    p.nl();
    p.with_braces([&] (pp::Pretty_print &p) {
            p.with_indent([&] (pp::Pretty_print &p) {
                    p.nl();

                    p.string("(void)e; (void)root; // no unused variable warnings");
                    p.nl();
                    p.string("bool result = ");

                    if (ruleset.get_postcond().is_empty()) {
                        p.string("true");
                    } else {
                        int idx2 = 0;

                        p.with_indent([&] (pp::Pretty_print &p) {
                                output_cpp_postcond_expr(p,
                                                         ruleset,
                                                         ruleset.get_postcond().get_expr(),
                                                         idx2);
                            });
                    }

                    p.semicolon();
                    p.nl();
                    p.string("if (!result && event_handler != NULL)");
                    p.with_indent([&] (pp::Pretty_print &p) {
                            p.nl();

                            p.string("fire_postcondition_event(*event_handler);");
                        });
                    p.nl();
                    p.string("return result;");
                });
            p.nl();
        });
    p.nl();
    p.nl();
}

void Compilation_unit::output_cpp_event_handler(pp::Pretty_print &p,
                                                Ruleset &ruleset) {
    p.string("void ");
    p.string(ruleset.get_name());
    p.string("::fire_match_event");
    p.with_parens([&] (pp::Pretty_print &p) {
            p.with_indent([&] (pp::Pretty_print &p) {
                    p.nl();
                    p.string("mi::mdl::IRule_matcher_event &event_handler,");
                    p.nl();
                    p.string("std::size_t id");
                });
        });
    p.nl();
    p.with_braces([&] (pp::Pretty_print &p) {
            p.with_indent([&] (pp::Pretty_print &p) {
                    p.nl();
                    p.string("Rule_info const &ri = g_rule_info[id];");
                    p.nl();
                    p.string("event_handler.rule_match_event");
                    p.with_indent([&] (pp::Pretty_print &p) {
                            p.with_parens([&] (pp::Pretty_print &p) {
                                    p.string("\"");
                                    p.string(ruleset.get_name());
                                    p.string("\",");
                                    p.space();
                                    p.string("ri.ruid,");
                                    p.space();
                                    p.string("ri.rname,");
                                    p.space();
                                    p.string("ri.fname,");
                                    p.space();
                                    p.string("ri.fline");
                                });
                            p.semicolon();
                        });
                });
            p.nl();
        });
    p.nl();
    p.nl();

    p.string( "void ");
    p.string(ruleset.get_name());
    p.string("::fire_postcondition_event");
    p.with_parens([&] (pp::Pretty_print &p) {
            p.nl();
            p.string("mi::mdl::IRule_matcher_event &event_handler");
        });
    p.nl();
    p.with_braces([&] (pp::Pretty_print &p) {
            p.with_indent([&] (pp::Pretty_print &p) {
                    p.nl();
                    p.string("event_handler.postcondition_failed(\"");
                    p.string(ruleset.get_name());
                    p.string("\");");
                });
            p.nl();
        });
    p.nl();
    p.nl();

    p.string("void ");
    p.string(ruleset.get_name());
    p.string("::fire_debug_print");
    p.with_parens([&] (pp::Pretty_print &p) {
        p.with_indent([&] (pp::Pretty_print &p) {
            p.nl();
            p.string("mi::mdl::IDistiller_plugin_api &plugin_api,");
            p.nl();
            p.string("mi::mdl::IRule_matcher_event &event_handler,");
            p.nl();
            p.string("std::size_t idx,");
            p.nl();
            p.string("char const *var_name,");
            p.nl();
            p.string("DAG_node const *value");
        });
    });
    p.nl();
    p.with_braces([&] (pp::Pretty_print &p) {
        p.with_indent([&] (pp::Pretty_print &p) {
            p.nl();
            p.string("Rule_info const &ri = g_rule_info[idx];");
            p.nl();
            p.string("event_handler.debug_print");
            p.with_indent([&] (pp::Pretty_print &p) {
                p.with_parens([&] (pp::Pretty_print &p) {
                    p.string("plugin_api,");
                    p.space();
                    p.string("\"");
                    p.string(ruleset.get_name());
                    p.string("\",");
                    p.space();
                    p.string("ri.ruid,");
                    p.space();
                    p.string("ri.rname,");
                    p.space();
                    p.string("ri.fname,");
                    p.space();
                    p.string("ri.fline,");
                    p.space();
                    p.string("var_name,");
                    p.space();
                    p.string("value");
                });
                p.semicolon();
            });
        });
        p.nl();
    });
    p.nl();
    p.nl();

}

/// Comparison functor for sorting rules by the root node value of the pattern
struct Cmp_rule {
   bool operator()(Rule const *const &a, Rule const *const &b) const {
       Expr const *lhs_a = a->get_lhs();
       for (;;) {
           if (lhs_a->get_kind() == Expr::Kind::EK_ATTRIBUTE) {
               lhs_a = cast<Expr_attribute>(lhs_a)->get_argument();
           } else if (Expr_binary const *eb = as<Expr_binary>(lhs_a)) {
               lhs_a = eb->get_right_argument();
           } else {
               break;
           }
       }
       Expr_call const *call_a = cast<Expr_call>(lhs_a);
       Expr_ref const *callee_a = cast<Expr_ref>(call_a->get_callee());
       Symbol const *name_a = callee_a->get_name();

       Expr const *lhs_b = b->get_lhs();
       for (;;) {
           if (lhs_b->get_kind() == Expr::Kind::EK_ATTRIBUTE) {
               lhs_b = cast<Expr_attribute>(lhs_b)->get_argument();
           } else if (Expr_binary const *eb = as<Expr_binary>(lhs_b)) {
               lhs_b = eb->get_right_argument();
           } else {
               break;
           }
       }
       Expr_call const *call_b = cast<Expr_call>(lhs_b);
       Expr_ref const *callee_b = cast<Expr_ref>(call_b->get_callee());
       Symbol const *name_b = callee_b->get_name();

       return std::strcmp(name_a->get_name(), name_b->get_name()) < 0;
   }
};

void Compilation_unit::sort_rules(mi::mdl::vector<Rule const *>::Type &rules, Ruleset const &ruleset) {
    for (Rule_list::const_iterator rit(ruleset.get_rules().begin()), rend(ruleset.get_rules().end());
         rit != rend; ++rit) {
        rules.push_back(rit);
    }

    std::stable_sort( rules.begin(), rules.end(), Cmp_rule());

}

void Compilation_unit::output_cpp(mi::mdl::string const &stem_name, mi::mdl::string const &cpp_name)
{
    std::fstream cpp_stream(cpp_name.c_str(), std::ios_base::out);
    if (!cpp_stream) {
        fprintf(stderr, "%s: unable to create output file\n", cpp_name.c_str());
        m_error_count++;
        return;
    }

    pp::Pretty_print p(m_arena, cpp_stream, 80);

    // Write copyright line and include the generated header file.

    p.string_with_nl(copyright_string);
    p.string_with_nl(
        "// Generated by mdltlc\n"
        "\n"
        "#include \"pch.h\"\n"
        "\n"
        "#include \"");
    p.string(stem_name.c_str());
    p.string_with_nl(".h\"\n\n");

    p.string_with_nl(
        "#include <mi/mdl/mdl_distiller_plugin_api.h>\n"
        "#include <mi/mdl/mdl_distiller_plugin_helper.h>\n\n");

    // Open namespace.

    p.string_with_nl(
        "using namespace mi::mdl;\n"
        "\n"
        "namespace MI {\n"
        "namespace DIST {\n\n"
        );

    // For each rule set, generate methods and rule table.
    for (Ruleset_list::iterator it(m_rulesets.begin()), end(m_rulesets.end());
         it != end; ++it) {

        // Write out code for strategy and name accessors.

        p.string_with_nl(
            "// Return the strategy to be used with this rule set.\n"
            "Rule_eval_strategy ");
        p.string(it->get_name());
        p.string("::get_strategy() const {");
        p.with_indent([&] (pp::Pretty_print &p) {
                p.string_with_nl("\nreturn ");
                char const *strategy = "RULE_EVAL_BOTTOM_UP";
                switch (it->get_strategy()) {
                case Ruleset::Strategy::STRAT_BOTTOMUP:
                    strategy = "RULE_EVAL_BOTTOM_UP";
                    break;
                case Ruleset::Strategy::STRAT_TOPDOWN:
                    strategy = "RULE_EVAL_TOP_DOWN";
                    break;
                }

                p.string(strategy);
                p.string(";\n");
            });
        p.string_with_nl(
            "}\n"
            "\n"
            "// Return the name of the rule set.\n"
            "char const * ");
        p.string(it->get_name());
        p.string("::get_rule_set_name() const {");
        p.with_indent([&] (pp::Pretty_print &p) {
                p.nl();
                p.string("return \"");
                p.string(it->get_name());
                p.string("\"");
                p.semicolon();
            });
        p.nl();
        p.rbrace();
        p.nl();
        p.nl();

        Var_set used_materials(get_allocator());
        for (Rule_list::const_iterator rit(it->get_rules().begin()),
                 rend(it->get_rules().end());
             rit != rend; ++rit)
            {
                used_target_materials(m_arena, rit->get_rhs(), used_materials);
            }

        p.string_with_nl(
            "// Return the number of imports of this rule set.\n"
            "size_t ");
        p.string(it->get_name());
        p.string("::get_target_material_name_count() const {");
        p.with_indent([&] (pp::Pretty_print &p) {
                p.string_with_nl("\nreturn ");
                p.integer(used_materials.size());
                p.string(";");
            });
        p.string_with_nl(
            "\n}\n"
            "\n");
        p.string_with_nl(
            "// Return the name of the import at index i.\n"
            "char const *");
        p.string(it->get_name());
        p.string("::get_target_material_name(size_t i) const {");
        p.with_indent([&] (pp::Pretty_print &p) {
                if (used_materials.size() > 0) {
                    p.string_with_nl(
                        "\nstatic char const *materials[] = "
                        );
                    p.with_braces([&] (pp::Pretty_print &p) {
                            p.with_indent([&] (pp::Pretty_print &p) {
                                    bool first = true;
                                    for (Var_set::const_iterator iit = used_materials.begin();
                                         iit != used_materials.end();
                                         ++iit) {
                                        if (!first)
                                            p.comma();
                                        else
                                            first = false;
                                        p.nl();
                                        p.string("\"");
                                        p.string((*iit)->get_name());
                                        p.string("\"");
                                    }
                                });
                            p.nl();
                        });
                    p.string_with_nl(";\nreturn materials[i];");
                } else {
                    p.string_with_nl("\nreturn nullptr;");
                }
            });
        p.string_with_nl(
            "\n}\n"
            "\n");

        // Sort the list of rules in the same way as the old compiler,
        // so that we can compare the outputs.

        mi::mdl::vector<Rule const *>::Type rules(m_arena.get_allocator());

        sort_rules(rules, *it);

        // Print out matcher code.

        output_cpp_matcher(p, *it, rules);

        // Print out postcondition code.

        output_cpp_postcond(p, *it);

        // Print event handler.

        output_cpp_event_handler(p, *it);

        // Print out rule table.

        size_t n = it->get_rules().size();

        p.string_with_nl("\n// Rule info table.\n");
                    p.string(it->get_name());
                    p.string("::Rule_info const ");
                    p.string(it->get_name());
                    p.string("::g_rule_info[");
                    p.integer(n);
                    p.string("] = {");
                    p.with_indent([&] (pp::Pretty_print &p) {
                            p.nl();

                            bool first = true;
                            for (mi::mdl::vector<Rule const *>::Type::const_iterator it(rules.begin()), end(rules.end());
                                 it != end; ++it) {
                                if (!first) {
                                    p.comma();
                                    p.nl();
                                } else {
                                    first = false;
                                }
                                Rule const &rule = **it;
                                p.lbrace();
                                p.space();
                                p.integer(rule.get_uid());
                                p.string(", \"");

                                char const *rule_name = rule.get_rule_name();
                                if (rule_name == nullptr) {
                                    Expr const *lhs = rule.get_lhs();
                                   
                                    char const *lhs_node_name = node_name(lhs);
                                    MDL_ASSERT(lhs_node_name);
                                    rule_name = lhs_node_name;
                                }

                                p.string(rule_name);
                                p.string("\", \"");
                                p.string(m_filename_only);
                                p.string("\", ");
                                p.integer(rule.get_location().get_line());
                                p.space();
                                p.rbrace();
                            }
                        });
                    p.nl();
                    p.rbrace();
                    p.semicolon();
                    p.nl();
                    p.nl();
    }

    // Close namespace and finish file.
    p.string_with_nl(
        "\n} // DIST\n"
        "} // MI\n"
        "// End of generated code\n");

    if (m_comp_options->get_verbosity() >= 1) {
        printf("[info] Generated %s.\n", cpp_name.c_str());
    }
}

/// Output the signatures of the helper functions generated for
/// postcondition checks.
void Compilation_unit::output_h_postcond_expr(
    pp::Pretty_print &p,
    Expr *expr,
    int &idx)
{
    switch (expr->get_kind()) {
    case Expr::Kind::EK_UNARY:
    {
        Expr_unary *e = cast<Expr_unary>(expr);
        switch (e->get_operator()) {
        case Expr_unary::Operator::OK_MATCH:
        case Expr_unary::Operator::OK_NONODE:
            p.string("\n"
                      "    static bool checker_");
            p.integer(idx);
            p.string("(\n"
                      "        mi::mdl::");
            p.string(m_api_class);
            p.string("      &e,\n"
                      "        mi::mdl::DAG_node const   *node);");
            p.nl();

            idx++;

            break;
        default:
            error(expr->get_location(),
                  "[BUG] invalid unary operator encountered in postcondition expression");
            MDL_ASSERT(!"[BUG] invalid unary operator encountered in postcondition expression");
            break;
        }
        break;
    }

    case Expr::Kind::EK_BINARY:
    {
        Expr_binary *e = cast<Expr_binary>(expr);

        output_h_postcond_expr(p, e->get_left_argument(), idx);

        switch (e->get_operator()) {
        case Expr_binary::OK_LOGICAL_AND:
            break;

        case Expr_binary::OK_LOGICAL_OR:
            break;

        default:
            error(expr->get_location(),
                  "[BUG] invalid binary operator encountered in postcondition expression");
            MDL_ASSERT(!"[BUG] invalid binary operator encountered in postcondition expression");
            break;
        }

        output_h_postcond_expr(p, e->get_right_argument(), idx);

        break;
    }
    case Expr::Kind::EK_INVALID:
    case Expr::Kind::EK_LITERAL:
    case Expr::Kind::EK_REFERENCE:
    case Expr::Kind::EK_CONDITIONAL:
    case Expr::Kind::EK_CALL:
    case Expr::Kind::EK_TYPE_ANNOTATION:
    case Expr::Kind::EK_ATTRIBUTE:
        error(expr->get_location(),
              "[BUG] invalid expression encountered in postcondition");
        MDL_ASSERT(!"[BUG] invalid expression encountered in postcondition");
        break;
    }

}

/// Output the header file for the generated C++ code of the current
/// mdltl file.
void Compilation_unit::output_h(
    mi::mdl::string const &stemname,
    mi::mdl::string const &h_name)
{
    mi::mdl::string upper_stemname(stemname);

    std::transform(upper_stemname.begin(),
                   upper_stemname.end(),
                   upper_stemname.begin(),
                   toupper);

    std::fstream h_stream(h_name.c_str(), std::ios_base::out);
    if (!h_stream) {
        fprintf(stderr, "%s: unable to create output file\n", h_name.c_str());
        m_error_count++;
        return;
    }

    pp::Pretty_print p(m_arena, h_stream, 80);

    // Write out copyright and header guard.

    p.string_with_nl(copyright_string);
    p.string_with_nl(
        "// Generated by mdltlc\n"
        "\n"
        "#ifndef MDL_DISTILLER_DIST_");
    p.string(upper_stemname.c_str());
    p.string_with_nl
        ("_H\n"
         "#define MDL_DISTILLER_DIST_");
    p.string(upper_stemname.c_str());
    p.string_with_nl("_H\n\n");

    // Write include statements and open namespace.

    p.string_with_nl(
        "#include \"mdl_assert.h\"\n\n"
        "#include <mi/mdl/mdl_distiller_rules.h>\n"
        "#include <mi/mdl/mdl_distiller_node_types.h>\n"
        "\n"
        "namespace MI {\n"
        "namespace DIST {\n\n"
        );

    // For each rule set, declare a class deriving from the rule
    // engine matcher class.

    for (Ruleset_list::iterator it(m_rulesets.begin()), end(m_rulesets.end());
         it != end; ++it) {

        // Print class header and prototypes of rule engine interface functions.

        p.string("class ");
        p.string(it->get_name());
        p.string(" : public mi::mdl::");
        p.string(m_rule_matcher_class);
        p.space();
        p.lbrace();
        p.with_indent([&] (pp::Pretty_print &p) {
                p.nl();
                p.string("public:");
            });
        p.with_indent([&] (pp::Pretty_print &p) {
                p.nl();
                p.string_with_nl(
                    "virtual mi::mdl::Rule_eval_strategy get_strategy() const;\n"
                    "virtual size_t get_target_material_name_count() const;\n"
                    "virtual char const *get_target_material_name(size_t i) const;\n"
                    "void set_node_types(mi::mdl::Node_types *node_types) {\n"
                    "  m_node_types = node_types;\n"
                    "};\n"
                    "\n"
                    "virtual mi::mdl::DAG_node const *matcher(");
                p.with_indent([&] (pp::Pretty_print &p) {
                        p.string_with_nl(
                            "\n"
                            "mi::mdl::IRule_matcher_event *event_handler,\n"
                            "mi::mdl::");
                        p.string(m_api_class);
                        p.string_with_nl(" &engine,\n"
                                         "mi::mdl::DAG_node const *node,\n"
                                         "const mi::mdl::Distiller_options *options,\n"
                                         "mi::mdl::Rule_result_code &result_code) const;");
                    });
                p.string_with_nl(
                    "\n\n"
                    "virtual bool postcond(");
                p.with_indent([&] (pp::Pretty_print &p) {
                        p.nl();
                        p.string_with_nl(
                            "mi::mdl::IRule_matcher_event *event_handler,\n"
                            "mi::mdl::");
                        p.string(m_api_class);
                        p.string_with_nl(
                            "&engine,\n"
                            "mi::mdl::DAG_node const *root,\n"
                            "const mi::mdl::Distiller_options *options) const;");
                    });
                p.string_with_nl(
                    "\n\n"
                    "virtual char const * get_rule_set_name() const;\n"
                    "\n"
                    "static void fire_match_event(");
                p.with_indent([&] (pp::Pretty_print &p) {
                        p.string_with_nl(
                            "\n"
                            "mi::mdl::IRule_matcher_event &event_handler,\n"
                            "std::size_t id);");
                    });
                p.string_with_nl(
                    "\n\n"
                    "static void fire_postcondition_event(");
                p.with_indent([&] (pp::Pretty_print &p) {
                        p.string_with_nl(
                            "\nmi::mdl::IRule_matcher_event &event_handler);");
                    });
                p.nl();
                p.string_with_nl(
                    "\n\n"
                    "static void fire_debug_print(");
                p.with_indent([&] (pp::Pretty_print &p) {
                    p.nl();
                    p.string("mi::mdl::IDistiller_plugin_api &plugin_api,");
                    p.nl();
                    p.string("mi::mdl::IRule_matcher_event &event_handler,");
                    p.nl();
                    p.string("std::size_t idx,");
                    p.nl();
                    p.string("char const *var_name,");
                    p.nl();
                    p.string("mi::mdl::DAG_node const *value);");
                });


                // Print prototypes for postcondition support functions.

                if (!it->get_postcond().is_empty()) {
                    int idx = 0;
                    output_h_postcond_expr(p, it->get_postcond().get_expr(), idx);
                }

            });
        // Print rule table.

        size_t n = it->get_rules().size();
        p.with_indent([&] (pp::Pretty_print &p) {
                p.nl();
                p.string_with_nl("private:");
            });
        p.with_indent([&] (pp::Pretty_print &p) {
                p.nl();
                p.string_with_nl(
                    "struct Rule_info ");
                p.lbrace();
                p.with_indent([&] (pp::Pretty_print &p) {
                        p.string_with_nl(
                            "\n"
                            "unsigned ruid;\n"
                            "char const *rname;\n"
                            "char const *fname;\n"
                            "unsigned fline;");
                    });
                p.nl();
                p.rbrace();
                p.semicolon();
                p.nl();
                p.nl();
                p.string("static Rule_info const g_rule_info[");
                p.integer(n);
                p.string("];\n");
                p.string("mi::mdl::Node_types *m_node_types = nullptr;\n");
            });
        // Finish class definition.

        p.rbrace();
        p.semicolon();
        p.nl();
        p.nl();
    }

    // Close namespace and finish header file.

    p.string_with_nl(
        "\n"
        "} // DIST\n"
        "} // MI\n"
        "\n"
        "#endif\n"
        "// End of generated code\n");
    if (m_comp_options->get_verbosity() >= 1) {
        printf("[info] Generated %s.\n", h_name.c_str());
    }
}

void Compilation_unit::output() {

    mi::mdl::IAllocator *allocator = m_arena.get_allocator();

    // This is the complete input file path, as given on the command
    // line.
    mi::mdl::string filename(allocator);
    filename = m_filename;

    // This is the directory portion of the input file path, without
    // the trailing slash. Empty if a relative filename without any
    // directory is given.
    mi::mdl::string dirname(allocator);

    // This is the filename portion of the input file path.
    mi::mdl::string basename(allocator);

    basename = m_filename_only;

    size_t slash_pos = filename.rfind('/');
    if (slash_pos != std::string::npos) {
        dirname = filename.substr(0, slash_pos);
    } else {
        slash_pos = filename.rfind('\\');
        if (slash_pos != std::string::npos) {
            dirname = filename.substr(0, slash_pos);
        } else {
            dirname = "";
        }
    }

    // This is the part of the input filename without any directory
    // and without the last filename extension.
    mi::mdl::string stemname(allocator);

    size_t dot_pos = basename.rfind('.');
    if (dot_pos != std::string::npos)
        stemname = basename.substr(0, dot_pos);
    else
        stemname = basename;

    // This is the output directory. It is the output directory from
    // the command line option --output-dir (if given), otherwise the
    // directory portion of the input file name.
    mi::mdl::string outdir_name(allocator);
    if (m_comp_options->get_output_dir())
        outdir_name = m_comp_options->get_output_dir();
    else
        outdir_name = dirname;

    // Full name of the output C++ file.
    mi::mdl::string cpp_name(allocator);

    // Full name of the output header file.
    mi::mdl::string h_name(allocator);

    if (outdir_name.size() > 0) {
        cpp_name += outdir_name + "/" + stemname + ".cpp";
        h_name += outdir_name +"/" + stemname + ".h";
    } else {
        cpp_name += stemname + ".cpp";
        h_name += stemname + ".h";
    }

    output_h(stemname, h_name);
    output_cpp(stemname, cpp_name);
}
