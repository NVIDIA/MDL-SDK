/******************************************************************************
 * Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
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
/// \file mdl_printer.h
/// \brief Printer for compiled materials that produces MDL code.

#pragma once

#include <set>
#include <vector>

#include <mi/mdl_sdk.h>

#include "options.h"
#include "mdl_distiller_utils.h"

using mi::neuraylib::INeuray;
using mi::neuraylib::IType;
using mi::neuraylib::IValue;
using mi::neuraylib::IExpression;
using mi::neuraylib::ICompiled_material;
using mi::neuraylib::ITransaction;
using mi::neuraylib::IExpression_direct_call;
using mi::neuraylib::IExpression_list;

/// The Mdl_printer class can print MDL source code for a compiled
/// material.
class Mdl_printer {
  public:
    Mdl_printer(INeuray* neuray,
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
                ITransaction *transaction);

    /// Print out the compiled material as a complete MDL module.
    void print_module();

    /// Print out the given type.
    ///
    /// \param type MDL type to print out.
    void print_type(IType const* type);

    /// Print out the given value.
    ///
    /// \param value MDL value to print out.
    void print_value(IValue const *value,
                     std::string const &path_prefix);

    /// Print out the given identifier. Format unicode identifiers
    /// using MDL 1.8 syntax.
    void print_identifier(std::string const &identifier);

    /// Return true if the given identifiers is a unicode identifier.
    bool is_unicode_identifier(std::string const &identifier);

    /// Print out the given MDL expression.
    ///
    /// The `indent` parameter tells the printer how many indentation
    /// steps (currently two spaces) to print out after each newline.
    ///
    /// \param expr MDL expression to print out.
    /// \param indent starting indentation for the print operation.
    void print_expression(IExpression const *expr,
                          std::string const &path_prefix,
                          int indent = 0);

    void analyze_material();

    mdl_spec get_required_mdl_version() {
        return m_required_mdl_version;
    }

    /// Set the option to suppress binding and usage of trivial
    /// temporaries when printing. When a temporary is encountered
    /// that is simply a numeric, color or string literal, or a
    /// parameter, it will be inlined in the output instead of
    /// generating a let binding for it, even if the value is used
    /// multiple times. This loses sharing but makes the printout more
    /// readable.
    ///
    /// The default of this option is "true".
    ///
    /// \param b new value of the setting.
    void set_suppress_trivial_temps(bool b) {
        m_suppress_trivial_temps = b;
    }

    /// Return the current setting for suppressing trivial
    /// temporaries.
    bool get_suppress_trivial_temps() {
        return m_suppress_trivial_temps;
    }

    /// Set the option to unfold temporaries in the printout. If set
    /// to true, all temporaries are inlined and no let bindings are
    /// generated for the material.
    ///
    /// The default of this option is "false".
    ///
    /// \param b new value of the setting.
    void set_unfold_temps(bool b) {
        m_unfold_temps = b;
    }

    /// Return the current setting for unfolding temporaries.
    bool get_unfold_temps() {
        return m_unfold_temps;
    }

    /// Set the option to replace all "non-interesting" expression
    /// from the output. This allows a quick overview of the material
    /// that only shows material components, BSDFs and some
    /// structs. All other expressions are replaced with three dots
    /// "...".
    ///
    /// The default of this option is "false".
    ///
    /// \param b new value of the setting.
    void set_outline_mode(bool b) {
        m_outline_mode = b;
    }

    /// Return the current setting for outline mode.
    bool get_outline_mode() {
        return m_outline_mode;
    }

    /// Set the option to suppress all arguments to functions that are
    /// equal to the corresponding default parameters. This shortens
    /// the output to make it more readable, while remaining
    /// compatible with the full output.
    ///
    /// The default of this option is "true".
    ///
    /// \param b new value of the setting.
    void set_suppress_default_parameters(bool b) {
        m_suppress_default_parameters = b;
    }

    /// Return the current setting for default parameter suppression.
    bool get_suppress_default_parameters() {
        return m_suppress_default_parameters;
    }

    /// Set the option to emit warnings and errors to standard error.
    ///
    /// The default of this option is "true".
    ///
    /// \param b new value of the setting.
    void set_emit_messages(bool b) {
        m_emit_messages = b;
    }

    /// Return the current setting of warning and error message
    /// output.
    bool get_emit_messages() {
        return m_emit_messages;
    }

    /// Set the option to emit warnings when parameters are encountered.
    ///
    /// The default of this option is "false".
    ///
    /// \param b new value of the setting.
    void set_warn_on_parameters(bool b) {
        m_warn_on_parameters = b;
    }

    /// Return the current setting of parameter occurrence warning.
    /// output.
    bool get_warn_on_parameters() {
        return m_warn_on_parameters;
    }

  private:
    void warning(std::string msg);
    void error(std::string msg);

    void ind(int indent);
    void add_identifier(char const *identifier);

    void print_vector_type(IType const* type);
    void print_matrix_type(IType const* type);
    void print_direct_call(IExpression_direct_call const *expr, 
            std::string const &path_prefix, int indent);

    void analyze_value(IValue const *value, std::string const &path_prefix);
    void analyze_expression(IExpression const *expr, std::string const &path_prefix);
    void check_mdl_version_requirement(
        std::string const &function_name,
        IExpression_direct_call const *call_expr,
        IExpression_list const *arguments);
    void set_min_mdl_version(mdl_spec version);

    void print_prolog();
    void print_epilog();
    void print_material();

    bool trivial_expression(IExpression const *expr);
    bool expression_needs_parens(IExpression const *expr);
    bool suppress_expression(IExpression const *expr);
    bool contains_interesting_expression(IExpression const *expr);
    bool preserve_temporaries();

    INeuray *m_neuray;
    mi::neuraylib::IMdl_factory *m_mdl_factory;
    mi::neuraylib::IExpression_factory *m_expr_factory;
    std::ostream &m_out;
    Options const *m_options;
    char const *m_target;
    std::string const m_material_name;
    std::string const m_derived_material_name;
    std::string const m_print_material_name;
    ICompiled_material const *m_material;
    Bake_paths* m_bake_paths;
    bool m_do_bake;
    ITransaction *m_transaction;
    mdl_spec m_required_mdl_version;

    std::set<std::string> m_used_modules;
    std::set<std::string> m_used_identifiers;
    std::set<std::string> m_used_parameters;
    std::vector<mi::Size> m_temporary_stack;
    std::set<mi::Size> m_baked_temporaries;
    std::vector<std::string> m_messages;
    int m_error_count;

    bool m_suppress_trivial_temps;
    bool m_unfold_temps;
    bool m_outline_mode;
    bool m_suppress_default_parameters;
    bool m_emit_messages;
    bool m_warn_on_parameters;
};

std::string derived_material_name(std::string mdl_material_name);
