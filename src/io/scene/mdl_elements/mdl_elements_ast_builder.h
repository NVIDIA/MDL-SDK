/***************************************************************************************************
 * Copyright (c) 2016-2023, NVIDIA CORPORATION. All rights reserved.
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
 **************************************************************************************************/
/// \file
/// \brief      Module-internal utilities for building MDL AST from neuray
///             expressions/types

#ifndef IO_SCENE_MDL_ELEMENTS_MDL_ELEMENTS_AST_BUILDER_H
#define IO_SCENE_MDL_ELEMENTS_MDL_ELEMENTS_AST_BUILDER_H

#include <map>
#include <set>
#include <string>

#include <mi/base/handle.h>
#include <mi/mdl/mdl_definitions.h>
#include <mi/mdl/mdl_mdl.h>
#include <base/data/db/i_db_tag.h>

#include "i_mdl_elements_type.h"
#include "i_mdl_elements_expression.h"
#include "i_mdl_elements_value.h"
#include "mdl_elements_utilities.h"

namespace mi {
namespace mdl {

class IModule;
class IType_name;
class IExpression;
class IExpression_factory;
class IExpression_reference;
class IName_factory;
class IQualified_name;
class ISimple_name;
class ISymbol;
class IType;
class IType_enum;
class IType_factory;
class IValue;
class IValue_factory;
class Module;
class Symbol_table;

}  // mdl
}  // mi

namespace MI {

namespace DB { class Transaction; }

namespace MDL {

class IType;

/// Helper class that converts neuray type/expressions into MDL AST.
class Mdl_ast_builder
{
public:
    /// Constructor.
    ///
    /// \param owner                   The MDL module that will own the newly constructed entities.
    /// \param transaction             The current transaction.
    /// \param traverse_ek_parameter   Indicates whether parameter references should be traversed
    ///                                and resolved via \p args, or converted to a parameter
    ///                                reference pointing to the declared parameters.
    /// \param args                    The arguments for occurring parameter references (only used
    ///                                if \p traverse_ek_parameter is \c true).
    /// \param name_manger             Name mangler, converts namespace and module names to MDL
    ///                                identifiers.
    /// \param avoid_resource_urls     Indicates whether to create tag-based or string-based
    ///                                resources.
    Mdl_ast_builder(
        mi::mdl::IModule* owner,
        DB::Transaction* transaction,
        bool traverse_ek_parameter,
        const IExpression_list* args,
        Name_mangler& name_mangler,
        bool avoid_resource_urls);

    /// Create a simple name from a string without signature.
    ///
    /// \param name  the core name
    const mi::mdl::ISimple_name* create_simple_name( const std::string& name);

    /// Create a qualified name from a string without signature.
    ///
    /// \param name  the core name
    ///
    /// Handles :: as scope operator.
    mi::mdl::IQualified_name* create_qualified_name( const std::string& name);

    /// Create a qualified name (containing the scope) from a string.
    ///
    /// \param name  the core name
    ///
    /// Handles :: as scope operator, might create a qualified name without a component.
    mi::mdl::IQualified_name* create_scope_name( const std::string& name);

    /// Construct a Type_name AST element for a neuray type.
    ///
    /// \param type   the neuray type
    mi::mdl::IType_name* create_type_name( const mi::base::Handle<const IType>& type);

    /// Retrieve the filed symbol from a DS_INTRINSIC_DAG_FIELD_ACCESS call.
    ///
    /// \param def  the unmangled MDL DAG name of the call
    const mi::mdl::ISymbol* get_field_sym( const std::string& def);

    /// Transform a call.
    ///
    /// \param ret_type     the (neuray) return type of the call
    /// \param sema         the semantic of the callee
    /// \param callee_name  the unmangled DAG name of the callee
    /// \param n_params     number of parameters of the callee
    /// \param args         callee arguments
    /// \param named_args   if true, create a call with name arguments, else with
    ///                     positional arguments
    const mi::mdl::IExpression* transform_call(
        const IType* ret_type,
        mi::mdl::IDefinition::Semantics sema,
        const std::string& callee_name,
        mi::Size n_params,
        const IExpression_list* args,
        bool named_args);

    /// Transform a MDL expression from neuray representation to MDL representation.
    ///
    /// \param expr  the neuray expression
    const mi::mdl::IExpression* transform_expr( const IExpression* expr);

    /// Transform a MDL expression from neuray representation to MDL representation.
    ///
    /// \param value  the neuray value
    const mi::mdl::IExpression* transform_value( const IValue* value);

    /// Transform a (non-user defined) MDL type from neuray representation to MDL representation.
    ///
    /// \param type  the neuray type
    const mi::mdl::IType* transform_type( const IType* type);

    /// Create a new temporary symbol.
    const mi::mdl::ISymbol* get_temporary_symbol();

    /// Create a simple name for a given Symbol.
    const mi::mdl::ISimple_name* to_simple_name( const mi::mdl::ISymbol* sym);

    /// Create a simple name for a given name.
    const mi::mdl::ISimple_name* to_simple_name( const char* name);

    /// Create a reference expression for a qualified name.
    ///
    /// \param qname  the qualified name
    /// \param type   if non-NULL, the MDL type
    mi::mdl::IExpression_reference* to_reference(
        mi::mdl::IQualified_name* qname,
        const mi::mdl::IType* type = nullptr);

    /// Create a reference expression for a given Symbol.
    ///
    /// \param sym  the symbol
    mi::mdl::IExpression_reference* to_reference( const mi::mdl::ISymbol* sym);

    /// Declare a parameter.
    ///
    /// \param sym   the name of the new parameter
    /// \param init  the expression that will be replaced by this parameter
    void declare_parameter( const mi::mdl::ISymbol* sym, const IExpression* init);

    /// Remove all declared parameter mappings.
    void remove_parameters();

    /// Convert an neuray enum type into a MDL enum type.
    const mi::mdl::IType_enum* convert_enum_type(const IType_enum* e_tp);

    /// Get the list of used user types.
    const std::set<std::string>& get_used_user_types() const { return m_used_user_types; }

private:
    /// Given a call name and a list of arguments, add a multi_scatter parameter
    /// and create a call.
    const mi::mdl::IExpression* add_multiscatter_param(
        const std::string& callee_name,
        mi::Size n_params,
        bool named_args,
        const IExpression_list* args);

    /// The MDL module that will own the newly constructed entities.
    mi::mdl::Module* m_owner;

    /// The current transaction.
    DB::Transaction* m_trans;

    /// Indicates whether parameter references should be traversed and resolved via \c m_args, or
    /// converted to a parameter reference pointing to the declared parameters.
    bool m_traverse_ek_parameter;

    /// The name factory of \c m_owner.
    mi::mdl::IName_factory& m_nf;

    /// The value factory of \c m_owner.
    mi::mdl::IValue_factory& m_vf;

    /// The expression factory of \c m_owner.
    mi::mdl::IExpression_factory& m_ef;

    /// The type factory of \c m_owner.
    mi::mdl::IType_factory& m_tf;

    /// The Symbol table of \c m_owner.
    mi::mdl::Symbol_table& m_st;

    /// The MDL type factory.
    mi::base::Handle<MDL::IType_factory> m_mdl_tf;

    /// The count for temporary generation.
    unsigned m_tmp_idx;

    template<typename T>
    struct Handle_less {
        bool operator() (const mi::base::Handle<T>& a, const mi::base::Handle<T>& b) const {
            std::less<typename mi::base::Handle<T>::pointer> less;
            return less(a.get(), b.get());
        }
    };

    using Param_map = std::map<
        mi::base::Handle<const IExpression>,
        const mi::mdl::ISymbol*,
        Handle_less<const IExpression>>;

    /// The parameter map.
    Param_map m_param_map;

    using Param_vector = std::vector<const mi::mdl::ISymbol*>;

    /// The parameter vector.
    Param_vector m_param_vector;

    /// The arguments of the original entity (only used if \c m_traverse_ek_parameter is \c true).
    mi::base::Handle<IExpression_list const> m_args;

    /// Set of used user types (decoded).
    std::set<std::string> m_used_user_types;

    mi::mdl::IMDL::MDL_version m_owner_version;

    /// Name mangler.
    Name_mangler& m_name_mangler;

    /// Create tag- or string-based resources.
    bool m_avoid_resource_urls;

    /// Set of indirect calls in the current call stack, used to check for cycles.
    std::set<DB::Tag> m_set_indirect_calls;
};

} // namespace MDL

} // namespace MI

#endif // IO_SCENE_MDL_ELEMENTS_MDL_ELEMENTS_AST_BUILDER_H
