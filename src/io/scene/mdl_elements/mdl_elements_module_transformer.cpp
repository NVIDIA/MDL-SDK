/***************************************************************************************************
 * Copyright (c) 2020-2025, NVIDIA CORPORATION. All rights reserved.
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

#include "pch.h"

#include "i_mdl_elements_module_transformer.h"

#include <map>
#include <memory>
#include <regex>
#include <set>

#include <mi/mdl/mdl_modules.h>
#include <mi/mdl/mdl_names.h>
#include <mi/mdl/mdl_symbols.h>
#include <mi/mdl/mdl_thread_context.h>

#include <base/lib/config/config.h>
#include <base/util/registry/i_config_registry.h>
#include <base/util/string_utils/i_string_utils.h>
#include <base/data/db/i_db_transaction.h>
#include <mdl/compiler/compilercore/compilercore_serializer.h>
#include <mdl/compiler/compilercore/compilercore_analysis.h>
#include <mdl/compiler/compilercore/compilercore_def_table.h>
#include <mdl/compiler/compilercore/compilercore_modules.h>
#include <mdl/compiler/compilercore/compilercore_module_transformer.h>
#include <mdl/compiler/compilercore/compilercore_tools.h>
#include <mdl/integration/mdlnr/i_mdlnr.h>

#include "i_mdl_elements_module.h"
#include "i_mdl_elements_utilities.h"
#include "mdl_elements_utilities.h"

using mi::mdl::as;
using mi::mdl::cast;
using mi::mdl::impl_cast;

namespace MI {

namespace MDL {

User_constant_remover::User_constant_remover(
    mi::mdl::IModule* module)
  : m_module( impl_cast<mi::mdl::Module>( module))
{
    m_ef = m_module->get_expression_factory();
    m_vf = m_module->get_value_factory();
    m_nf = m_module->get_name_factory();
}

// Process it.
void User_constant_remover::process()
{
    visit(m_module);
}

// Creates a qualified name from a scope.
mi::mdl::IQualified_name *User_constant_remover::create_qualified_name( mi::mdl::Scope const *scope)
{
    mi::mdl::Definition_table const &def_tab = m_module->get_definition_table();
    mi::mdl::Scope const *global = def_tab.get_global_scope();
    mi::mdl::Scope const *owner_scope = scope;

    // Collect scope symbols from bottom to top
    for( ; scope != global && scope != nullptr; scope = scope->get_parent()) {
        if( mi::mdl::ISymbol const *sym = scope->get_scope_name()) {
            m_sym_stack.push( sym);
        }
    }

    mi::mdl::IQualified_name *qname = m_nf->create_qualified_name();
    // need "::" in front?
    qname->set_absolute( m_sym_stack.empty() && owner_scope == global);

    // Add the in reverse order to the qualified name
    while( !m_sym_stack.empty()) {
        mi::mdl::ISymbol const *sym = m_sym_stack.top();

        mi::mdl::ISimple_name *sname = m_nf->create_simple_name( sym);
        qname->add_component( sname);

        m_sym_stack.pop();
    }

    return qname;
}

// Converts an enum value to a reference expression.
mi::mdl::IExpression *User_constant_remover::convert_enum_value( mi::mdl::IValue_enum const *value)
{
    mi::mdl::IType_enum const *type = value->get_type();
    mi::mdl::Definition_table const &def_tab = m_module->get_definition_table();
    mi::mdl::Scope const *scope = def_tab.get_type_scope( type);
    MDL_ASSERT( scope);

    // Enums are referenced without mentioning the enum type name, so skip it
    scope = scope->get_parent();

    mi::mdl::IQualified_name *qname = create_qualified_name( scope);

    // Get name of enum value and add it to the qualified name
    mi::mdl::IType_enum::Value const *e_value = type->get_value( value->get_index());

    mi::mdl::ISimple_name *value_sname = m_nf->create_simple_name( e_value->get_symbol());
    qname->add_component( value_sname);

    mi::mdl::IType_name *tname = m_nf->create_type_name( qname);
    mi::mdl::IExpression_reference *ref = m_ef->create_reference( tname);
    ref->set_type( type);
    return ref;
}

// Converts a struct or array value to a call expression to the corresponding constructor.
mi::mdl::IExpression* User_constant_remover::convert_compound_value(
    const mi::mdl::IValue_compound*value)
{
    const mi::mdl::IType_compound* type = value->get_type();
    const mi::mdl::Definition_table& def_tab = m_module->get_definition_table();
    const mi::mdl::Scope* scope;
    if( auto const* a_tp = as<mi::mdl::IType_array>( type)) {
        scope = def_tab.get_type_scope( a_tp->get_element_type());
    } else {
        scope = def_tab.get_type_scope( type);
    }
    MDL_ASSERT( scope);

    mi::mdl::IQualified_name* qname = create_qualified_name( scope);
    mi::mdl::IType_name* tname = m_nf->create_type_name( qname);

    // Adapt type for arrays, with a value, the size of the array is known
    if( mi::mdl::is<mi::mdl::IType_array>( type)) {
        mi::mdl::IExpression_literal *array_size = m_ef->create_literal(
            m_vf->create_int( type->get_compound_size()));
        tname->set_array_size( array_size);
        //tname->set_incomplete_array();
    }

    mi::mdl::IExpression_reference* constructor = m_ef->create_reference( tname);
    constructor->set_type( type);

    // Adapt constructor for arrays, if needed
    if( mi::mdl::is<mi::mdl::IType_array>( type)) {
        constructor->set_array_constructor();
    }

    mi::mdl::IExpression_call* call = m_ef->create_call( constructor);
    call->set_type( type);

    // Convert all compound elements and add to constructor arguments
    for( int i = 0, n = value->get_component_count(); i < n; ++i) {
        const mi::mdl::IValue* arg_value = value->get_value( i);
        const mi::mdl::IExpression* arg_expr = convert_user_value( arg_value);
        // Was not a user value? Use original value
        if( arg_expr == nullptr) {
            arg_expr = m_ef->create_literal( arg_value);
        }
        const mi::mdl::IArgument* arg = m_ef->create_positional_argument( arg_expr);
        call->add_argument( arg);
    }

    return call;
}

// Converts a value, if it contains a user constant, otherwise returns nullptr.
mi::mdl::IExpression *User_constant_remover::convert_user_value( mi::mdl::IValue const *value)
{
    mi::mdl::IType const *type = value->get_type();
    if( !mi::mdl::is_user_type( type))
        return nullptr;

    switch( type->get_kind()) {
    case mi::mdl::IType::TK_ENUM:
        return convert_enum_value( cast<mi::mdl::IValue_enum>( value));

    case mi::mdl::IType::TK_STRUCT:
        return convert_compound_value( cast<mi::mdl::IValue_compound>( value));

    case mi::mdl::IType::TK_ARRAY:
        {
            mi::mdl::IType const *e_type = cast<mi::mdl::IType_array>( type)->get_element_type();
            if( !mi::mdl::is_user_type( e_type))
                return nullptr;
            return convert_compound_value( cast<mi::mdl::IValue_compound>( value));
        }

    default:
        return nullptr;
    }
}

// Converts the literal, if it contains a user constant, otherwise keeps it unmodified.
mi::mdl::IExpression *User_constant_remover::post_visit( mi::mdl::IExpression_literal* expr)
{
    mi::mdl::IExpression *converted_expr = convert_user_value( expr->get_value());

    // Was not a user value? Return original expression
    if( converted_expr == nullptr)
        return expr;

    return converted_expr;
}


namespace {

/// Converts an absolute qualified name given as vector of strings into a string representation.
std::string stringify(
    const std::vector<std::string>& name, bool ignore_last_component = false)
{
    std::string result;
    mi::Size n = name.size() - (ignore_last_component ? 1 : 0);
    for( mi::Size i = 0; i < n; ++i)
       result += "::" + name[i];
    return result;
}

/// Converts a qualified name into a string representation.
std::string stringify( const mi::mdl::IQualified_name* name)
{
    std::string result;
    if( name->is_absolute())
        result += "::";
    for( mi::Uint32 i = 0, n = name->get_component_count(); i < n; ++i) {
        const mi::mdl::ISimple_name* simple = name->get_component( i);
        if( i > 0)
            result += "::";
        result += simple->get_symbol()->get_name();
    }
    return result;
}

/// Restores import entries in the constructor and drops them in the destructor.
class Import_entries_holder : public boost::noncopyable
{
public:
    Import_entries_holder(
        SYSTEM::Access_module<MDLC::Mdlc_module>& mdlc_module,
        DB::Transaction* transaction,
        mi::mdl::Module* module)
      : m_mdlc_module( mdlc_module),
        m_transaction( transaction),
        m_module( module, mi::base::DUP_INTERFACE)
    {
        Module_cache module_cache(
            m_transaction, m_mdlc_module->get_module_wait_queue(), {});
        m_module->restore_import_entries( &module_cache);
    }

    ~Import_entries_holder()
    {
        m_module->drop_import_entries();
    }

private:
    SYSTEM::Access_module<MDLC::Mdlc_module>& m_mdlc_module;
    DB::Transaction* m_transaction;
    mi::base::Handle<mi::mdl::Module> m_module;
};


/// Upgrades the MDL version of the module including all its call references.
///
/// There must be no weak relative import declarations or weak relative resource file paths.
class Version_upgrader : public User_constant_remover, public mi::mdl::IClone_modifier
{
    using Base = User_constant_remover;
public:
    Version_upgrader(
        mi::mdl::IMDL* mdl,
        mi::mdl::Module* module,
        mi::neuraylib::Mdl_version to_version);

    void run();

private:
    /// Promotes call references and its arguments.
    mi::mdl::IExpression* post_visit( mi::mdl::IExpression_call* expr) final;

    /// Implement IClone_modifier by returning the argument.
    mi::mdl::IExpression* clone_expr_reference( const mi::mdl::IExpression_reference* ref) final
    { return const_cast<mi::mdl::IExpression_reference*>( ref); }
    mi::mdl::IExpression* clone_expr_call( const mi::mdl::IExpression_call* call) final
    { return const_cast<mi::mdl::IExpression_call*>( call); }
    mi::mdl::IExpression* clone_literal( const mi::mdl::IExpression_literal* lit) final
    { return const_cast<mi::mdl::IExpression_literal*>( lit); }
    mi::mdl::IQualified_name* clone_name( const mi::mdl::IQualified_name* qname) final
    { return const_cast<mi::mdl::IQualified_name*>( qname); }

    mi::mdl::IMDL* m_mdl;

    mi::mdl::IMDL::MDL_version m_from_version;

    mi::mdl::IMDL::MDL_version m_to_version;
    int m_to_major = 1;
    int m_to_minor = 0;

    /// Newly required imports due to promoted call expressions.
    std::set<std::string> m_new_imports;
};

Version_upgrader::Version_upgrader(
    mi::mdl::IMDL* mdl,
    mi::mdl::Module* module,
    mi::neuraylib::Mdl_version to_version)
  : Base( module),
    m_mdl( mdl)
{
    m_from_version = m_module->get_mdl_version();
    m_to_version = convert_mdl_version( to_version);
    std::tie( m_to_major, m_to_minor) = split_mdl_version( m_to_version);

    ASSERT( M_SCENE, m_to_version > m_from_version);
}

void Version_upgrader::run()
{
    // Update the MDL version first (queried by Module_inliner::promote_call_reference())
    m_module->set_version(
        m_to_major,
        m_to_minor,
        /*enable_mdl_next*/ true,       // allow version upgrade to MDL next, ...
        /*enable_experimental*/ false); // ... but not to experimental

    // Promote call expressions and its arguments.
    visit( m_module);

    // Add newly required imports due to promoted call expressions.
    for( const auto& i: m_new_imports)
        m_module->add_import( i.c_str());
}

mi::mdl::IExpression* Version_upgrader::post_visit( mi::mdl::IExpression_call* expr)
{
    const auto* ref = as<mi::mdl::IExpression_reference>( expr->get_reference());
    if( !ref)
        return expr;

    mi::Uint32 rules = mi::mdl::Module::PR_NO_CHANGE;
    ref = mi::mdl::Module_inliner::promote_call_reference(
        *m_module, m_from_version, ref, this, rules);
    if( rules == mi::mdl::Module::PR_NO_CHANGE)
        return expr;

    if( rules & mi::mdl::Module::PR_FRESNEL_LAYER_TO_COLOR)
        m_new_imports.insert( "::df::color_fresnel_layer");
    if( rules & mi::mdl::Module::PR_MEASURED_EDF_ADD_TANGENT_U)
        m_new_imports.insert( "::state::texture_tangent_u");

    mi::mdl::IExpression_call* call = m_ef->create_call( ref);
    for( int i = 0, j = 0, n = expr->get_argument_count(); i < n; ++i, ++j) {
        const mi::mdl::IArgument* arg = m_module->clone_arg( expr->get_argument( i), this);
        call->add_argument( arg);
        j = m_module->promote_call_arguments( call, arg, j, rules, /*creator=*/nullptr);
    }
    return call;
}

/// Removes all namespace alias declarations.
///
/// Expands import declarations and type names using namespace aliases.
class Alias_remover : protected mi::mdl::Module_visitor, public mi::mdl::IClone_modifier
{
public:
    Alias_remover( mi::mdl::Module* module, const std::vector<std::string>& module_name);

    void run();

private:
    /// Adapts qualified names in import declarations to avoid aliases.
    void handle_import_name( mi::mdl::IQualified_name* name);

    /// Adapts qualified names in type names to avoid aliases.
    void post_visit( mi::mdl::IType_name* type_name) final;

    /// Implement IClone_modifier by returning the argument.
    mi::mdl::IExpression* clone_expr_reference( const mi::mdl::IExpression_reference* ref) final
    { return const_cast<mi::mdl::IExpression_reference*>( ref); }
    mi::mdl::IExpression* clone_expr_call( const mi::mdl::IExpression_call* call) final
    { return const_cast<mi::mdl::IExpression_call*>( call); }
    mi::mdl::IExpression* clone_literal( const mi::mdl::IExpression_literal* lit) final
    { return const_cast<mi::mdl::IExpression_literal*>( lit); }
    mi::mdl::IQualified_name* clone_name( const mi::mdl::IQualified_name* qname) final
    { return const_cast<mi::mdl::IQualified_name*>( qname); }

    /// Returns a qualified name where the last component has been removed.
    mi::mdl::IQualified_name* strip_last_component( mi::mdl::IQualified_name* name);

    /// Returns an absolute name for \p name.
    ///
    /// Requires that \p name does not use any namespace aliases.
    std::vector<std::string> get_absolute_name( const mi::mdl::IQualified_name* name) const;

    /// Removes leading '.' and '..' components from relative names.
    mi::mdl::IQualified_name* get_reference_name( mi::mdl::IQualified_name* name) const;

    mi::mdl::Module* m_module;
    const std::vector<std::string>& m_module_name;

    mi::mdl::IName_factory* m_nf;

    /// Stores for each modified import declaration the mapping from its import index to the newly
    /// created name (suitable for later references, i.e., without leading '.' and '..'
    /// components).
    std::map<mi::Size, mi::mdl::IQualified_name*> m_mapping;
};

Alias_remover::Alias_remover(
    mi::mdl::Module* module, const std::vector<std::string>& module_name)
  : m_module( module),
    m_module_name( module_name)
{
    m_nf = m_module->get_name_factory();
}

void Alias_remover::run()
{
    // Check whether the module contains any aliases.
    bool aliases_found = false;
    for( size_t i = 0, n = m_module->get_declaration_count(); i < n; ++i) {
        const mi::mdl::IDeclaration* decl = m_module->get_declaration( i);
        mi::mdl::IDeclaration::Kind kind = decl->get_kind();
        if(    kind == mi::mdl::IDeclaration::DK_MODULE
            || kind == mi::mdl::IDeclaration::DK_IMPORT)
            continue;
        if( kind == mi::mdl::IDeclaration::DK_NAMESPACE_ALIAS) {
            aliases_found = true;
            break;
        }
        break;
    }
    if( !aliases_found)
        return;

    // Adapt import declarations to avoid aliases.
    for( size_t i = 0, n = m_module->get_declaration_count(); i < n; ++i) {

        const mi::mdl::IDeclaration* decl = m_module->get_declaration( i);
        mi::mdl::IDeclaration::Kind kind = decl->get_kind();
        if(    kind == mi::mdl::IDeclaration::DK_MODULE
            || kind == mi::mdl::IDeclaration::DK_NAMESPACE_ALIAS)
            continue;
        if( kind != mi::mdl::IDeclaration::DK_IMPORT)
            break;

        const auto* import = cast<mi::mdl::IDeclaration_import>( decl);

        auto* using_name = const_cast<mi::mdl::IQualified_name*>( import->get_module_name());
        if( using_name) {
            handle_import_name( using_name);
            std::vector<std::string> absolute_name = get_absolute_name( using_name);
            std::string absolute_name_str = stringify( absolute_name);
            mi::Size index = m_module->get_import_index( absolute_name_str.c_str());
            ASSERT( M_SCENE, index != 0);
            mi::mdl::IQualified_name* reference_name = get_reference_name( using_name);
            m_mapping[index] = reference_name;
            continue;
        }

        for( int j = 0, n2 = import->get_name_count(); j < n2; ++j) {
            auto* import_name = const_cast<mi::mdl::IQualified_name*>( import->get_name( j));
            handle_import_name( import_name);
            mi::mdl::IQualified_name* module_name = strip_last_component( import_name);
            std::vector<std::string> absolute_name = get_absolute_name( module_name);
            std::string absolute_name_str = stringify( absolute_name);
            mi::Size index = m_module->get_import_index( absolute_name_str.c_str());
            ASSERT( M_SCENE, index != 0);
            mi::mdl::IQualified_name* reference_name = get_reference_name( module_name);
            m_mapping[index] = reference_name;
        }
    }

    // Remove all alias declarations.
    std::vector<const mi::mdl::IDeclaration*> declarations;
    declarations.reserve( m_module->get_declaration_count());
    for( size_t i = 0, n = m_module->get_declaration_count(); i < n; ++i) {
        const mi::mdl::IDeclaration* decl = m_module->get_declaration( i);
        switch( decl->get_kind()) {
            case mi::mdl::IDeclaration::DK_NAMESPACE_ALIAS:
                break;
            default:
                declarations.push_back( decl);
                break;
        }
    }
    m_module->replace_declarations( declarations.data(), declarations.size());

    // Adapt names of expression references to avoid aliases.
    visit( m_module);
}

void Alias_remover::handle_import_name( mi::mdl::IQualified_name* name)
{
    // Push components of \p name onto a stack.
    std::stack<const mi::mdl::ISymbol*> components;
    for( int i = name->get_component_count()-1; i >= 0; --i)
        components.push( name->get_component( i)->get_symbol());

    name->clear_components();
    mi::mdl::Definition_table& def_table = m_module->get_definition_table();

    // Loop over components on stack.
    while( !components.empty()) {

        const mi::mdl::ISymbol* sym = components.top();
        components.pop();

        const mi::mdl::Definition* def = def_table.get_namespace_alias( sym);

        // Re-add component symbols which are not a namespace alias.
        if( !def) {
            name->add_component( m_nf->create_simple_name( sym));
            continue;
        }

        // Re-add component symbol without declaration in this module.
        const auto* decl = as<mi::mdl::IDeclaration_namespace_alias>( def->get_declaration());
        if( !decl) {
            name->add_component( m_nf->create_simple_name( sym));
            continue;
        }

        // Push components of namespace aliases onto the stack.
        const mi::mdl::IQualified_name* namespace_name = decl->get_namespace();
        if( namespace_name->is_absolute())
            name->set_absolute();
        for( int i = namespace_name->get_component_count()-1; i >= 0; --i)
            components.push( namespace_name->get_component( i)->get_symbol());
    }
}

void Alias_remover::post_visit( mi::mdl::IType_name* type_name)
{
    mi::mdl::IQualified_name* old_name = type_name->get_qualified_name();

    // Skip unqualified names.
    if( old_name->get_component_count() < 2)
       return;

    // Skip expression references without definition.
    const mi::mdl::IDefinition* def = old_name->get_definition();
    if( !def)
        return;

    // Skip non-imported definitions or definitions with unmodified import index.
    mi::Size index = impl_cast<mi::mdl::Definition>( def)->get_original_import_idx();
    if( index == 0)
        return;
    auto it = m_mapping.find( index);
    if( it == m_mapping.end())
        return;

    mi::mdl::IQualified_name* module_name = it->second;

    // Construct new name for the expression reference based on the new name from the import
    // declaration and the symbol of the expression reference.
    mi::mdl::IQualified_name* new_name = m_nf->create_qualified_name();
    if( module_name->is_absolute())
        new_name->set_absolute();
    for( int i = 0, n = module_name->get_component_count(); i < n; ++i)
        new_name->add_component( module_name->get_component( i));
    new_name->add_component( m_nf->create_simple_name( def->get_symbol()));

    type_name->set_qualified_name( new_name);
}

mi::mdl::IQualified_name* Alias_remover::strip_last_component( mi::mdl::IQualified_name* name)
{
    mi::mdl::IQualified_name* result = m_nf->create_qualified_name();
    if( name->is_absolute())
        result->set_absolute();

    int n = name->get_component_count();
    ASSERT( M_SCENE, n > 1);
    for( int i = 0; i < n-1; ++i)
        result->add_component( name->get_component( i));

    return result;
}

std::vector<std::string> Alias_remover::get_absolute_name(
    const mi::mdl::IQualified_name* name) const
{
    std::vector<std::string> result;

    if( !name->is_absolute()) {
        result = m_module_name;
        result.pop_back();
    }

    for( mi::Uint32 i = 0, n = name->get_component_count(); i < n; ++i) {
        const mi::mdl::ISymbol* sym = name->get_component( i)->get_symbol();
        size_t id = sym->get_id();
        if( id == mi::mdl::ISymbol::SYM_DOT) {
            // pass
        } else if( id == mi::mdl::ISymbol::SYM_DOTDOT) {
            result.pop_back();
        } else {
            result.emplace_back( sym->get_name());
        }
    }

    return result;
}

mi::mdl::IQualified_name* Alias_remover::get_reference_name(
    mi::mdl::IQualified_name* name) const
{
    if( name->is_absolute())
        return name;

    mi::mdl::IQualified_name* result = m_nf->create_qualified_name();
    for( int i = 0, n = name->get_component_count(); i < n; ++i) {
        const mi::mdl::ISimple_name* sn = name->get_component( i);
        const mi::mdl::ISymbol* sym = sn->get_symbol();
        size_t id = sym->get_id();
        if( id == mi::mdl::ISymbol::SYM_DOT) {
            // pass
        } else if( id == mi::mdl::ISymbol::SYM_DOTDOT) {
            // pass
        } else {
            result->add_component( sn);
        }
    }

    return result;
}

/// Provides common infrastructure for adjusting import declarations.
class Import_declaration_replacer : public User_constant_remover
{
    using Base = User_constant_remover;
public:
    Import_declaration_replacer(
        mi::mdl::IMDL* mdl,
        mi::mdl::Module* module,
        const std::vector<std::string>& module_name)
      : Base( module),
        m_mdl( mdl),
        m_module_name( module_name)

    {
        int major = 0;
        int minor = 0;
        m_module->get_version( major, minor);
        m_namespace_aliases_legal = major == 1 && minor <= 7;
    }

    void run();

protected:
    /// To be implemented in derived classes to modify the given name.
    ///
    /// The flag \p name_refers_to_module indicates whether \p name is the name of a module, or the
    /// name of an entity inside a module.
    virtual void handle_name( mi::mdl::IQualified_name* name, bool name_refers_to_module) = 0;

    /// To be implemented in derived classes to indicate which alias declarations need to be
    /// removed.
    ///
    /// Indicates whether the alias declaration is correct after the transformation, i.e., does not
    /// contradict the transformation.
    virtual bool is_correct_alias_decl( const mi::mdl::IDeclaration_namespace_alias* alias) = 0;

    /// Converts \p name to an absolute qualified name for import declarations.
    ///
    /// The new name is added to \c m_mapping if \p update_mapping is set.
    void do_create_absolute_import_declaration(
        mi::mdl::IQualified_name* name,
        bool name_refers_to_module,
        const std::vector<std::string>& absolute_name,
        mi::Size import_index,
        bool update_mapping);

    /// Converts \p name to a strict relative qualified name for import declarations.
    ///
    /// The new name is added to \c m_mapping.
    void do_create_relative_import_declaration(
        mi::mdl::IQualified_name* name,
        bool name_refers_to_module,
        const std::vector<std::string>& absolute_name,
        mi::Size import_index,
        Execution_context* context);

    /// Creates an absolute name as combination of \c m_module_name and \p name.
    ///
    /// If \p force_absolute_name is set, then it overrides name->is_absolute().
    std::vector<std::string> create_absolute_name(
        const mi::mdl::IQualified_name* name,
        bool force_absolute_name,
        bool& was_absolute,
        bool& was_strict_relative) const;

    /// Creates aliases for components of \p name as necessary and modifies \p name accordingly.
    void create_aliases( std::vector<std::string>& name);

    /// Indicates whether the symbol represents "." or "..".
    static bool is_dot_or_dotdot( const mi::mdl::ISymbol* sym);

private:
    /// Adapts the names of the expression references if its import declaration is recorded in
    /// \c m_mapping.
    mi::mdl::IExpression* post_visit( mi::mdl::IExpression_reference* expr) final;

    /// Indicates whether namespace aliases are legal (up to MDL 1.7).
    bool m_namespace_aliases_legal;

protected:
    mi::mdl::IMDL* m_mdl;

    const std::vector<std::string>& m_module_name;

    /// Stores for each modified import declaration the mapping from its import index to the newly
    /// created name.
    std::map<mi::Size, mi::mdl::IQualified_name*> m_mapping;

    /// Stores the mapping (namespace name, alias name) of existing alias declarations.
    ///
    /// Used to avoid the creation of redundant declarations. Only those alias declarations where
    /// the namespace does not contain ".", "..", or "::" are stored.
    std::map<std::string, std::string> m_old_aliases;

    /// Stores the mapping (namespace name, alias name) of new alias declarations.
    std::map<std::string, std::string> m_new_aliases;

    /// Counter to generate new alias names.
    mi::Size m_counter = 0;
};

void Import_declaration_replacer::run()
{
    // Get existing alias declarations, but only if the namespace does not contain ".", "..", or
    // "::" (the others will not be re-used anyway), and only the first one for each namespace.
    for( mi::Size i = 0, n = m_module->get_declaration_count(); i < n; ++i) {

        const mi::mdl::IDeclaration* decl = m_module->get_declaration( static_cast<int>( i));
        const auto* alias = as<mi::mdl::IDeclaration_namespace_alias>( decl);
        if( !alias)
            continue;

        const mi::mdl::IQualified_name* qn = alias->get_namespace();
        if( qn->is_absolute() || qn->get_component_count() > 1)
            continue;

        const mi::mdl::ISymbol* sym = qn->get_component( 0)->get_symbol();
        if( is_dot_or_dotdot( sym))
            continue;

        auto& s = m_old_aliases[sym->get_name()];
        if( s.empty())
            s = alias->get_alias()->get_symbol()->get_name();
    }

    // Modify all import declarations if necessary, keep track of the mapping in m_mapping.
    for( mi::Size i = 0, n = m_module->get_declaration_count(); i < n; ++i) {

        const mi::mdl::IDeclaration* decl = m_module->get_declaration( static_cast<int>( i));
        const auto* import = as<mi::mdl::IDeclaration_import>( decl);
        if( !import)
            continue;

        auto* using_name = const_cast<mi::mdl::IQualified_name*>( import->get_module_name());
        if( using_name) {
            handle_name( using_name, /*name_refers_to_module*/ true);
            continue;
        }

        for( int j = 0, n2 = import->get_name_count(); j < n2; ++j) {
            auto* import_name = const_cast<mi::mdl::IQualified_name*>( import->get_name( j));
            handle_name( import_name, /*name_refers_to_module*/ false);
        }
    }

    // Adapt names of expression references based on \c m_mapping.
    visit( m_module);

    // Add new alias declarations, remove incorrect alias declarations. We would like to remove all
    // now unused alias declarations, but since this is not easy we remove at least the incorrect
    // ones (which are unused at this point).

    std::vector<const mi::mdl::IDeclaration*> declarations;
    declarations.reserve( m_module->get_declaration_count() + m_new_aliases.size());
    mi::mdl::IDeclaration_factory* df = m_module->get_declaration_factory();

    for( size_t i = 0, n = m_module->get_declaration_count(); i < n; ++i) {

        const mi::mdl::IDeclaration* decl = m_module->get_declaration( i);
        switch( decl->get_kind()) {

            // Add new alias declarations in front of the first import declaration.
            case mi::mdl::IDeclaration::DK_IMPORT: {
                for( auto& new_alias : m_new_aliases) {
                    mi::mdl::ISimple_name* alias_sn
                        = m_nf->create_simple_name( m_nf->create_symbol( new_alias.second.c_str()));
                    mi::mdl::ISimple_name* namespace_sn
                        = m_nf->create_simple_name( m_nf->create_symbol( new_alias.first.c_str()));
                    mi::mdl::IQualified_name* namespace_qn = m_nf->create_qualified_name();
                    namespace_qn->add_component( namespace_sn);
                    const mi::mdl::IDeclaration* alias_decl
                        = df->create_namespace_alias( alias_sn, namespace_qn);
                    declarations.push_back( alias_decl);
                }
                m_new_aliases.clear();
                declarations.push_back( decl);
                break;
            }

            // Remove incorrect alias declarations.
            case mi::mdl::IDeclaration::DK_NAMESPACE_ALIAS: {
                const auto* alias = cast<mi::mdl::IDeclaration_namespace_alias>( decl);
                if( is_correct_alias_decl( alias))
                    declarations.push_back( decl);
                break;
            }

            default:
                declarations.push_back( decl);
                break;
        }
    }

    m_module->replace_declarations( declarations.data(), declarations.size());
}

void Import_declaration_replacer::do_create_absolute_import_declaration(
    mi::mdl::IQualified_name* name,
    bool name_refers_to_module,
    const std::vector<std::string>& absolute_name,
    mi::Size import_index,
    bool update_mapping)
{
    // Update \p name.
    name->clear_components();
    name->set_absolute();
    for( const auto& n : absolute_name)
        name->add_component(
            m_nf->create_simple_name( m_nf->create_symbol( n.c_str())));

    // Create IQualified_name for absolute module name.
    mi::mdl::IQualified_name* absolute_module_name_qn = m_nf->create_qualified_name();
    absolute_module_name_qn->set_absolute();
    for( mi::Size i = 0, n = absolute_name.size() - (name_refers_to_module ? 0 : 1); i < n; ++i)
        absolute_module_name_qn->add_component(
            m_nf->create_simple_name( name->get_component( static_cast<int>( i))->get_symbol()));

    // Map import index to the absolute module name.
    if( update_mapping) {
        auto it = m_mapping.find( import_index);
        if( it == m_mapping.end())
            m_mapping[import_index] = absolute_module_name_qn;
        else
            ASSERT( M_SCENE, stringify( it->second) == stringify( absolute_module_name_qn));
    }
}

void Import_declaration_replacer::do_create_relative_import_declaration(
    mi::mdl::IQualified_name* name,
    bool name_refers_to_module,
    const std::vector<std::string>& absolute_name,
    mi::Size import_index,
    Execution_context* context)
{
    // Compute common prefix of \c m_module_name and absolute_name.
    mi::Size len = 0;
    mi::Size n_module = m_module_name.size();
    mi::Size n_name   = absolute_name.size();
    while( len < n_module && len < n_name && m_module_name[len] == absolute_name[len])
        ++len;

    // Skip modules in different search paths.
    mi::base::Handle<const mi::mdl::Module> imported_module( m_module->get_import(
        static_cast<int>( import_index-1)));
    if( !Mdl_module_transformer::same_search_path(
        m_module, imported_module->get_filename(), n_module-1 - len)) {
        ASSERT( M_SCENE, context);
        add_warning_message( context,
            std::string( "Skipping ") + imported_module->get_name() + " since it was found in a "
            "different search path. Use filters to suppress this message.");
        return;
    }

    // Update \p name.
    name->clear_components();
    name->set_absolute( false);
    if( len == n_module - 1) {
        const mi::mdl::ISymbol* symbol
            = mi::mdl::Symbol_table::get_predefined_symbol( mi::mdl::ISymbol::SYM_DOT);
        name->add_component( m_nf->create_simple_name( symbol));
    } else {
        const mi::mdl::ISymbol* sym
            = mi::mdl::Symbol_table::get_predefined_symbol( mi::mdl::ISymbol::SYM_DOTDOT);
        for( mi::Size i = len; i < n_module - 1; ++i)
            name->add_component( m_nf->create_simple_name( sym));
    }
    for( mi::Size i = len; i < n_name; ++i)
       name->add_component(
           m_nf->create_simple_name( m_nf->create_symbol( absolute_name[i].c_str())));

    // Create IQualified_name for relative module name (without "." and ".." components).
    mi::mdl::IQualified_name* relative_module_name_qn = m_nf->create_qualified_name();
    for( mi::Size i = len; i < n_name - (name_refers_to_module ? 0 : 1); ++i)
        relative_module_name_qn->add_component(
            m_nf->create_simple_name( m_nf->create_symbol( absolute_name[i].c_str())));

    // Map import index to the relative module name.
    auto it = m_mapping.find( import_index);
    if( it == m_mapping.end())
        m_mapping[import_index] = relative_module_name_qn;
    else
        ASSERT( M_SCENE, stringify( it->second) == stringify( relative_module_name_qn));
}

namespace {

void handle_namespace_alias(
    const mi::mdl::IDefinition* alias,
    std::vector<std::string>& result,
    bool& was_absolute,
    bool& was_strict_relative)
{
    const mi::mdl::ISymbol* sym = alias->get_namespace();
    const char* name = sym->get_name();
    if( name[0] == ':') {
        name = name + 2;
        result.clear();
        was_absolute = true;
    }

    std::vector<std::string> tmp;
    STRING::split( name, ":", tmp);

    for( mi::Size i = 0, n = tmp.size(); i < n; ++i) {
        if( tmp[i].empty())
            ; // by-product of splitting at ":" instead of "::"
        else if( tmp[i] == ".") {
            if( i == 0)
                was_strict_relative = true;
        } else if( tmp[i] == "..") {
            result.pop_back();
            if( i == 0)
                was_strict_relative = true;
        } else
            result.push_back( tmp[i]);
    }
}

} // namespace

std::vector<std::string> Import_declaration_replacer::create_absolute_name(
    const mi::mdl::IQualified_name* name,
    bool force_absolute_name,
    bool& was_absolute,
    bool& was_strict_relative) const
{
    std::vector<std::string> result;

    if( name->is_absolute() || force_absolute_name) {
        ASSERT( M_SCENE, !is_dot_or_dotdot( name->get_component( 0)->get_symbol()));
        was_absolute = true;
    } else {
        result = m_module_name;
        result.pop_back();
        was_absolute = false;
    }
    was_strict_relative = false;

    for( mi::Uint32 i = 0, n = name->get_component_count(); i < n; ++i) {
        const mi::mdl::ISymbol* sym = name->get_component( i)->get_symbol();
        size_t id       = sym->get_id();
        if( id == mi::mdl::ISymbol::SYM_DOT) {
            if( i == 0)
                was_strict_relative = true;
        } else if( id == mi::mdl::ISymbol::SYM_DOTDOT) {
            result.pop_back();
            if( i == 0)
                was_strict_relative = true;
        } else {
            const mi::mdl::Definition* def
                = m_module->get_definition_table().get_namespace_alias( sym);
            if( def) {
                bool dummy1 = false;
                bool dummy2 = false;
                handle_namespace_alias(
                    def,
                    result,
                    i == 0 ? was_absolute : dummy1,
                    i == 0 ? was_strict_relative : dummy2);
            } else
                result.emplace_back( sym->get_name());
        }
    }

    return result;
}

void Import_declaration_replacer::create_aliases( std::vector<std::string>& absolute_name)
{
   if( !m_namespace_aliases_legal)
       return;

    mi::mdl::Symbol_table* st = &m_module->get_symbol_table();

    for( std::string& s: absolute_name) {

        // No need to create aliases for "*" and identifiers.
        if( s == "*" || m_mdl->is_valid_mdl_identifier( s.c_str()))
            continue;

        // Use existing (old) alias declarations.
        auto it = m_old_aliases.find( s);
        if( it != m_old_aliases.end()) {
            s = it->second;
            continue;
        }

        // Use previously created (new) alias declarations.
        it = m_new_aliases.find( s);
        if( it != m_new_aliases.end()) {
            s = it->second;
            continue;
        }

        // Construct unique name.
        std::string alias;
        while( true) {
            std::ostringstream ss;
            ss << "alias" << m_counter++;
            alias = ss.str();
            if( !st->lookup_symbol( alias.c_str()))
                break;
        }

        m_new_aliases[s] = alias;
        s = alias;
    }
}

bool Import_declaration_replacer::is_dot_or_dotdot( const mi::mdl::ISymbol* sym)
{
    size_t id = sym->get_id();
    return id == mi::mdl::ISymbol::SYM_DOT || id == mi::mdl::ISymbol::SYM_DOTDOT;
}

mi::mdl::IExpression* Import_declaration_replacer::post_visit( mi::mdl::IExpression_reference* expr)
{
    // Skip expression references without definition.
    const mi::mdl::IDefinition* def = expr->get_definition();
    if( !def)
        return expr;

    // Skip unqualified expression references.
    if( expr->get_name()->get_qualified_name()->get_component_count() < 2)
       return expr;

    // Skip non-imported definitions or definitions with unmodified import index.
    mi::Size index = impl_cast<mi::mdl::Definition>( def)->get_original_import_idx();
    if( index == 0)
        return expr;
    auto it = m_mapping.find( index);
    if( it == m_mapping.end())
        return expr;

    mi::mdl::IQualified_name* module_name = it->second;

    // Construct new name for the expression reference based on the new name from the import
    // declaration and the symbol of the expression reference.
    mi::mdl::IQualified_name* expr_name = m_nf->create_qualified_name();
    for( mi::Size i = 0, n = mi::Size( module_name->get_component_count()); i < n; ++i) {
        const mi::mdl::ISymbol* symbol =
            module_name->get_component( static_cast<int>( i))->get_symbol();
        expr_name->add_component( m_nf->create_simple_name( symbol));
    }
    expr_name->add_component( m_nf->create_simple_name( def->get_symbol()));

    // Set new name for expression reference.
    auto* type_name = const_cast<mi::mdl::IType_name*>( expr->get_name());
    type_name->set_qualified_name( expr_name);

    return expr;
}

/// Creates absolute import declarations.
class Absolute_import_declaration_creator : public Import_declaration_replacer
{
public:
    Absolute_import_declaration_creator(
        mi::mdl::IMDL* mdl,
        mi::mdl::Module* module,
        const std::vector<std::string>& module_name,
        std::wregex* include_regex,
        std::wregex* exclude_regex)
      : Import_declaration_replacer( mdl, module, module_name),
        m_include_regex( include_regex),
        m_exclude_regex( exclude_regex) { }

private:
    void handle_name( mi::mdl::IQualified_name* name, bool name_refers_to_module) final
    {
        // Compute absolute name from \c module_name and and \p name.
        bool was_absolute        = false;
        bool was_strict_relative = false;
        std::vector<std::string> absolute_name = create_absolute_name(
            name, /*force_absolute_name*/ false, was_absolute, was_strict_relative);

        // Skip names that are already absolute.
        if( was_absolute)
            return;

        // Compute corresponding module name and its import index.
        std::string absolute_module_name( stringify(
            absolute_name, /*ignore_last_component*/ !name_refers_to_module));
        mi::Size index = m_module->get_import_index( absolute_module_name.c_str());

        // The import index can be invalid if weak relative names should be interpreted as absolute
        // names. Do that now. In that case, we do not need to update expression references.
        bool update_mapping = true;
        if( index == 0) {
            absolute_name = create_absolute_name(
                name, /*force_absolute_name*/ true, was_absolute, was_strict_relative);
            absolute_module_name = stringify(
                absolute_name, /*ignore_last_component*/ !name_refers_to_module);
            index = m_module->get_import_index( absolute_module_name.c_str());
            ASSERT( M_SCENE, index > 0);
            update_mapping = false;
        }

        // Apply filters.
        std::string mdl_module_name = MDL::encode_module_name( absolute_module_name);
        std::wstring mdl_module_name_wstr = STRING::utf8_to_wchar( mdl_module_name.c_str());
        if( m_include_regex && !std::regex_match( mdl_module_name_wstr, *m_include_regex))
            return;
        if( m_exclude_regex &&  std::regex_match( mdl_module_name_wstr, *m_exclude_regex))
            return;

        create_aliases( absolute_name);
        do_create_absolute_import_declaration(
            name, name_refers_to_module, absolute_name, index, update_mapping);
    }

    bool is_correct_alias_decl( const mi::mdl::IDeclaration_namespace_alias* alias) final
    {
        const mi::mdl::IQualified_name* qn = alias->get_namespace();
        return !is_dot_or_dotdot( qn->get_component( 0)->get_symbol());
    }

    std::wregex* m_include_regex;
    std::wregex* m_exclude_regex;
};

/// Creates relative import declarations.
class Relative_import_declaration_creator : public Import_declaration_replacer
{
public:
    Relative_import_declaration_creator(
        mi::mdl::IMDL* mdl,
        mi::mdl::Module* module,
        const std::vector<std::string>& module_name,
        std::wregex* include_regex,
        std::wregex* exclude_regex,
        Execution_context* context)
      : Import_declaration_replacer( mdl, module, module_name),
        m_include_regex( include_regex),
        m_exclude_regex( exclude_regex),
        m_context( context) { }

private:
    void handle_name( mi::mdl::IQualified_name* name, bool name_refers_to_module) final
    {
        // Compute absolute name from \c module_name and and \p name.
        bool was_absolute        = false;
        bool was_strict_relative = false;
        std::vector<std::string> absolute_name = create_absolute_name(
            name, /*force_absolute_name*/ false, was_absolute, was_strict_relative);

        // Skip names that are already strict relative.
        if( was_strict_relative)
            return;

        // Compute corresponding module name and its import index.
        std::string absolute_module_name( stringify(
            absolute_name, /*ignore_last_component*/ !name_refers_to_module));
        mi::Size index = m_module->get_import_index( absolute_module_name.c_str());

        // The import index can be invalid if weak relative names should be interpreted as absolute
        // names. Do that now.
        if( index == 0) {
            absolute_name = create_absolute_name(
                name, /*force_absolute_name*/ true, was_absolute, was_strict_relative);
            absolute_module_name = stringify(
                absolute_name, /*ignore_last_component*/ !name_refers_to_module);
            index = m_module->get_import_index( absolute_module_name.c_str());
            ASSERT( M_SCENE, index > 0);
        }

        // Skip builtin modules.
        mi::base::Handle<const mi::mdl::Module> imported_module( m_module->get_import(
            static_cast<int>( index-1)));
        if( imported_module->is_compiler_owned() || imported_module->is_native())
            return;

        // Apply filters.
        std::string mdl_module_name = MDL::encode_module_name( absolute_module_name);
        std::wstring mdl_module_name_wstr = STRING::utf8_to_wchar( mdl_module_name.c_str());
        if( m_include_regex && !std::regex_match( mdl_module_name_wstr, *m_include_regex))
            return;
        if( m_exclude_regex &&  std::regex_match( mdl_module_name_wstr, *m_exclude_regex))
            return;

        create_aliases( absolute_name);
        do_create_relative_import_declaration(
            name, name_refers_to_module, absolute_name, index, m_context);
    }

    bool is_correct_alias_decl( const mi::mdl::IDeclaration_namespace_alias* alias) final
    {
        const mi::mdl::IQualified_name* qn = alias->get_namespace();
        const mi::mdl::ISymbol* sym = qn->get_component( 0)->get_symbol();
        if( is_dot_or_dotdot( sym))
            return true;

        if( !qn->is_absolute())
            return true;

        // Absolute names referrring to modules are ok if these modules are builtin.
        mi::Size index = m_module->get_import_index( stringify( qn).c_str());
        if( index > 0) {
            mi::base::Handle<const mi::mdl::Module> imported_module( m_module->get_import(
                static_cast<int>( index-1)));
            return imported_module->is_compiler_owned() || imported_module->is_native();
        }

        return false;
    }

    std::wregex* m_include_regex;
    std::wregex* m_exclude_regex;
    Execution_context* m_context;
};

/// Creates non-weak import declarations.
class Non_weak_relative_import_declaration_creator : public Import_declaration_replacer
{
public:
    Non_weak_relative_import_declaration_creator(
        mi::mdl::IMDL* mdl,
        mi::mdl::Module* module,
        const std::vector<std::string>& module_name)
      : Import_declaration_replacer( mdl, module, module_name) { }

private:
    void handle_name( mi::mdl::IQualified_name* name, bool name_refers_to_module) final
    {
        // Compute absolute name from \c module_name and and \p name.
        bool was_absolute        = false;
        bool was_strict_relative = false;
        std::vector<std::string> absolute_name = create_absolute_name(
            name, /*force_absolute_name*/ false, was_absolute, was_strict_relative);

        // Skip names that are already absolute or strict relative.
        if( was_absolute || was_strict_relative)
            return;

        // Compute corresponding module name and its import index.
        std::string absolute_module_name( stringify(
            absolute_name, /*ignore_last_component*/ !name_refers_to_module));
        mi::Size index = m_module->get_import_index( absolute_module_name.c_str());

        if( index > 0) {
            // Weak relative is interpreted as (strict) relative
            create_aliases( absolute_name);
            do_create_relative_import_declaration(
                name, name_refers_to_module, absolute_name, index, /*context*/ nullptr);
        } else {
            // Weak relative is interpreted as absolute
            absolute_name = create_absolute_name(
                name, /*force_absolute_name*/ true, was_absolute, was_strict_relative);
            absolute_module_name = stringify(
                absolute_name, /*ignore_last_component*/ !name_refers_to_module);
            index = m_module->get_import_index( absolute_module_name.c_str());
            ASSERT( M_SCENE, index > 0);

            create_aliases( absolute_name);
            do_create_absolute_import_declaration(
                name, name_refers_to_module, absolute_name, index, /*update_mapping*/ false);
        }
    }

    bool is_correct_alias_decl( const mi::mdl::IDeclaration_namespace_alias* alias) final
    {
        // This class is only used when upgrading to MDL 1.6. And for older versions there should
        // be no alias declarations.
        ASSERT( M_SCENE, false);
        return true;
    }
};

/// Provides common infrastructure for adjusting resource file paths.
class Resource_file_path_replacer : public User_constant_remover
{
    using Base = User_constant_remover;
public:
    Resource_file_path_replacer(
        mi::mdl::Module* module,
        const std::vector<std::string>& module_name)
      : Base( module),
        m_module_name( module_name)
    {
    }

    void run() { visit( m_module); }

protected:
    /// Creates an absolute file path as combination of \p module_name and \p file_path.
    /// \p force_absolute_name treats weak relative file paths as absolute.
    std::string create_absolute_file_path(
        const std::string& file_path,
        bool force_absolute_name) const;

    /// Creates a strict relative file path from an absolute file path.
    std::string create_relative_file_path(
        const std::string& file_path,
        mi::Size& n_module,
        mi::Size& len) const;

    /// Returns the resource index for a given absolute file path, or -1 if not found.
    mi::Size get_resource_index( const std::string& absolute_file_path) const;

    /// Creates a resource value by cloning \p value, but with the string value set to \p s.
    const mi::mdl::IValue_resource* create_resource(
        const mi::mdl::IValue_resource* value, const char* s);

    const std::vector<std::string>& m_module_name;
};

/// Creates an absolute file path as combination of \c m_module_name and \p file_path.
/// \p force_absolute_name treats weak relative file paths as absolute.
std::string Resource_file_path_replacer::create_absolute_file_path(
    const std::string& file_path,
    bool force_absolute_name) const
{
    if( force_absolute_name) {
        ASSERT( M_SCENE, file_path[0] != '/');
        return "/" + file_path;
    }

    if( file_path[0] == '/')
        return file_path;

    std::vector<std::string> components;
    STRING::split( file_path, "/", components);
    ASSERT( M_SCENE, !force_absolute_name || (components[0] != "." && components[0] != ".."));

    std::vector<std::string> result( m_module_name);
    result.pop_back();

    for( const auto& c: components) {
        if( c == ".")
            ; // nothing to do
        else if( c == "..")
            result.pop_back();
        else
            result.push_back( c);
    }

    std::string str;
    for( const auto& r: result)
        str += "/" + r;

    return str;
}

/// Creates a strict relative file path from an absolute file path.
std::string Resource_file_path_replacer::create_relative_file_path(
    const std::string& file_path,
    mi::Size& n_module,
    mi::Size& len) const
{
    ASSERT( M_SCENE, file_path[0] == '/');

    std::vector<std::string> components;
    STRING::split( file_path.substr( 1), "/", components);
    ASSERT( M_SCENE, components[0] != "." && components[0] != "..");

    // Compute common prefix of \c m_module_name and \p file_path.
    len                  = 0;
    n_module             = m_module_name.size();
    mi::Size n_file_path = components.size();
    while( len < n_module && len < n_file_path && m_module_name[len] == components[len])
        ++len;

    std::string result;

    if( len == n_module - 1)
        result = ".";
    else {
        for( mi::Size i = len; i < n_module - 1; ++i) {
            if( i > len)
                result += "/";
            result += "..";
        }
    }

    for( mi::Size i = len; i < n_file_path; ++i)
        result += "/" + components[i];

    return result;
}

/// Returns the resource index for a given absolute file path, or -1 if not found.
mi::Size Resource_file_path_replacer::get_resource_index(
    const std::string& absolute_file_path) const
{
    mi::Size n = m_module->get_referenced_resources_count();
    for( mi::Size i = 0; i < n; ++i)
        if( m_module->get_referenced_resource_url( i) == absolute_file_path)
            return i;

    return static_cast<mi::Size>( -1);
}

/// Creates a resource value by cloning \p value, but with the string value set to \p s.
const mi::mdl::IValue_resource* Resource_file_path_replacer::create_resource(
    const mi::mdl::IValue_resource* value, const char* s)
{
    switch( value->get_kind()) {
        case mi::mdl::IValue::VK_TEXTURE: {
            const auto* t = cast<mi::mdl::IValue_texture>( value);
            return m_vf->create_texture(
                t->get_type(),
                s,
                t->get_gamma_mode(),
                t->get_selector(),
                t->get_tag_value(),
                t->get_tag_version());
        }
        case mi::mdl::IValue::VK_LIGHT_PROFILE: {
            const auto* l = cast<mi::mdl::IValue_light_profile>( value);
            return m_vf->create_light_profile(
                l->get_type(),
                s,
                l->get_tag_value(),
                l->get_tag_version());
        }
        case mi::mdl::IValue::VK_BSDF_MEASUREMENT: {
            const auto* b = cast<mi::mdl::IValue_bsdf_measurement>( value);
            return m_vf->create_bsdf_measurement(
                b->get_type(),
                s,
                b->get_tag_value(),
                b->get_tag_version());
        }
        default:
            ASSERT( M_SCENE, false);
    }

    return nullptr;
}

/// Creates absolute resource file paths.
class Absolute_resource_file_path_creator : public Resource_file_path_replacer
{
    using Base = Resource_file_path_replacer;
public:
    Absolute_resource_file_path_creator(
        mi::mdl::Module* module,
        const std::vector<std::string>& module_name,
        std::wregex* include_regex,
        std::wregex* exclude_regex)
      : Resource_file_path_replacer( module, module_name),
        m_include_regex( include_regex),
        m_exclude_regex( exclude_regex) { }

private:
    mi::mdl::IExpression* post_visit( mi::mdl::IExpression_literal* expr) final
    {
        // Handled by base class?
        mi::mdl::IExpression *new_expr = Base::post_visit( expr);
        if( new_expr != expr) {
            return new_expr;
        }

        // Skip non-resources.
        const mi::mdl::IValue_resource* resource = as<mi::mdl::IValue_resource>( expr->get_value());
        if( !resource)
            return expr;

        // Get file path from resource or string value.
        const char* file_path = resource->get_string_value();

        // Skip resources without file path.
        if( !file_path || file_path[0] == '\0')
             return expr;

        // Skip absolute file paths.
        if( file_path[0] == '/')
            return expr;

        // Create absolute file path.
        std::string absolute_file_path = create_absolute_file_path(
            file_path, /*force_absolute*/ false);
        mi::Size index = get_resource_index( absolute_file_path);
        if( index == static_cast<mi::Size>( -1)) {
            absolute_file_path = create_absolute_file_path( file_path, /*force_absolute*/ true);
            index = get_resource_index( absolute_file_path);
            ASSERT( M_SCENE, index != static_cast<mi::Size>( -1));
        }

        // Apply filters.
        std::wstring absolute_file_path_wstr = STRING::utf8_to_wchar( absolute_file_path.c_str());
        if( m_include_regex && !std::regex_match( absolute_file_path_wstr, *m_include_regex))
            return expr;
        if( m_exclude_regex &&  std::regex_match( absolute_file_path_wstr, *m_exclude_regex))
            return expr;

        // Create new value.
        const mi::mdl::IValue* absolute_value = create_resource(
            resource, absolute_file_path.c_str());

        expr->set_value( absolute_value);
        return expr;
    }

    std::wregex* m_include_regex;
    std::wregex* m_exclude_regex;
};

/// Creates strict relative resource file paths.
class Relative_resource_file_path_creator : public Resource_file_path_replacer
{
    using Base = Resource_file_path_replacer;
public:
    Relative_resource_file_path_creator(
        mi::mdl::Module* module,
        const std::vector<std::string>& module_name,
        std::wregex* include_regex,
        std::wregex* exclude_regex,
        Execution_context* context)
      : Resource_file_path_replacer( module, module_name),
        m_include_regex( include_regex),
        m_exclude_regex( exclude_regex),
        m_context( context) { }

private:
    mi::mdl::IExpression* post_visit( mi::mdl::IExpression_literal* expr) final
    {
        // Handled by base class?
        mi::mdl::IExpression *new_expr = Base::post_visit( expr);
        if( new_expr != expr) {
            return new_expr;
        }

        // Skip non-resources.
        const mi::mdl::IValue_resource* resource = as<mi::mdl::IValue_resource>( expr->get_value());
        if( !resource)
            return expr;

        // Get file path from resource or string value.
        const char* file_path = resource->get_string_value();

        // Skip resources without file path.
        if( !file_path || file_path[0] == '\0')
             return expr;

        // Skip strict relative file paths.
        if( file_path[0] == '.')
            return expr;

        // Create absolute file path.
        std::string absolute_file_path = create_absolute_file_path(
            file_path, /*force_absolute*/ false);
        mi::Size index = get_resource_index( absolute_file_path);
        if( index == static_cast<mi::Size>( -1)) {
            absolute_file_path = create_absolute_file_path( file_path, /*force_absolute*/ true);
            index = get_resource_index( absolute_file_path);
            ASSERT( M_SCENE, index != static_cast<mi::Size>( -1));
        }

        // Apply filters.
        std::wstring absolute_file_path_wstr = STRING::utf8_to_wchar( absolute_file_path.c_str());
        if( m_include_regex && !std::regex_match( absolute_file_path_wstr, *m_include_regex))
            return expr;
        if( m_exclude_regex &&  std::regex_match( absolute_file_path_wstr, *m_exclude_regex))
            return expr;

        mi::Size n_module = 0;
        mi::Size len = 0;
        std::string relative_file_path = create_relative_file_path(
            absolute_file_path, n_module, len);

        // Skip modules in different search paths.
        const char* file_name = m_module->get_referenced_resource_file_name( index);
        if( file_name && !Mdl_module_transformer::same_search_path(
            m_module, file_name, n_module-1 - len)) {
            add_warning_message( m_context,
                std::string( "Skipping ") + absolute_file_path + " since it was found in a "
                "different search path. Use filters to suppress this message.");
            return expr;
        }

        // Create new value.
        const mi::mdl::IValue* absolute_value = create_resource(
            resource, relative_file_path.c_str());

        expr->set_value( absolute_value);
        return expr;
    }

    std::wregex* m_include_regex;
    std::wregex* m_exclude_regex;
    Execution_context* m_context;
};

/// Creates non-weak relative resource file paths.
class Non_weak_relative_resource_file_path_creator : public Resource_file_path_replacer
{
    using Base = Resource_file_path_replacer;
public:
    Non_weak_relative_resource_file_path_creator(
        mi::mdl::Module* module,
        const std::vector<std::string>& module_name,
        Execution_context* context)
      : Resource_file_path_replacer( module, module_name)/*,
        m_context( context)*/ { }

private:
    mi::mdl::IExpression* post_visit( mi::mdl::IExpression_literal* expr) final
    {
        // Handled by base class?
        mi::mdl::IExpression *new_expr = Base::post_visit( expr);
        if( new_expr != expr) {
            return new_expr;
        }

        // Skip non-resources.
        const mi::mdl::IValue_resource* resource = as<mi::mdl::IValue_resource>( expr->get_value());
        if( !resource)
            return expr;

        // Get file path from resource or string value.
        const char* file_path = resource->get_string_value();

        // Skip resources without file path.
        if( !file_path || file_path[0] == '\0')
             return expr;

        // Skip absolute or strict relative file paths.
        if( file_path[0] == '/' || file_path[0] == '.')
            return expr;

        // Create absolute file path.
        std::string absolute_file_path = create_absolute_file_path(
            file_path, /*force_absolute*/ false);

        std::string new_file_path;
        mi::Size index = get_resource_index( absolute_file_path);
        if( index != static_cast<mi::Size>( -1)) {
            // Weak relative is interpreted as (strict) relative
            mi::Size n_module = 0;
            mi::Size len = 0;
            new_file_path = create_relative_file_path(
                absolute_file_path, n_module, len);
        } else {
            // Weak relative is interpreted as absolute
            absolute_file_path = create_absolute_file_path( file_path, /*force_absolute*/ true);
            index = get_resource_index( absolute_file_path);
            ASSERT( M_SCENE, index != static_cast<mi::Size>( -1));
            new_file_path = absolute_file_path;
        }

        // Create new value.
        const mi::mdl::IValue* absolute_value = create_resource(
            resource, new_file_path.c_str());

        expr->set_value( absolute_value);
        return expr;
    }

    //Execution_context* m_context;
};

} // namespace

Mdl_module_transformer::Mdl_module_transformer(
    DB::Transaction* transaction, mi::mdl::IModule* module)
  : m_transaction( transaction)
{
    m_transaction->pin();

    m_mdlc_module.set();
    m_mdl = m_mdlc_module->get_mdl();

    // No cloning/serialization of module.
    m_module = make_handle_dup( impl_cast<mi::mdl::Module>( module));

    const mi::mdl::IQualified_name* name = m_module->get_qualified_name();
    for( mi::Uint32 i = 0, n = name->get_component_count(); i < n; ++i) {
        const mi::mdl::ISimple_name* simple = name->get_component( i);
        m_module_name.emplace_back( simple->get_symbol()->get_name());
    }
}

Mdl_module_transformer::Mdl_module_transformer(
    DB::Transaction* transaction, const mi::mdl::IModule* module)
  : m_transaction( transaction)
{
    m_transaction->pin();

    m_mdlc_module.set();
    m_mdl = m_mdlc_module->get_mdl();

    mi::mdl::Buffer_serializer serializer( m_mdl->get_mdl_allocator());
    m_mdl->serialize_module( module, &serializer, /*include_dependencies*/ false);

    mi::mdl::Buffer_deserializer deserializer(
        m_mdl->get_mdl_allocator(), serializer.get_data(), serializer.get_size());
    m_module = impl_cast<mi::mdl::Module>( const_cast<mi::mdl::IModule*>(
        m_mdl->deserialize_module( &deserializer)));

    const mi::mdl::IQualified_name* name = m_module->get_qualified_name();
    for( mi::Uint32 i = 0, n = name->get_component_count(); i < n; ++i) {
        const mi::mdl::ISimple_name* simple = name->get_component( i);
        m_module_name.emplace_back( simple->get_symbol()->get_name());
    }
}

Mdl_module_transformer::~Mdl_module_transformer()
{
    m_transaction->unpin();
}

mi::Sint32 Mdl_module_transformer::upgrade_mdl_version(
    mi::neuraylib::Mdl_version to_version, Execution_context* context)
{
    if( to_version < mi::neuraylib::MDL_VERSION_1_3) {
        add_error_message( context, "The new MDL version needs to be at least 1.3", -6);
        return -1;
    }

    if( to_version == mi::neuraylib::MDL_VERSION_INVALID) {
        add_error_message( context, "Invalid new MDL version.", -6);
        return -1;
    }

    mi::neuraylib::Mdl_version from_version
        = convert_mdl_version( m_module->get_version());
    if( to_version < from_version) {
        add_error_message(
            context, "The new MDL version needs to be higher than the current one.", -6);
        return -1;
    }

    if( to_version == from_version)
        return 0;

    if( !is_module_valid( context))
        return -1;

    Import_entries_holder import_entries( m_mdlc_module, m_transaction, m_module.get());

    if(    from_version <= mi::neuraylib::MDL_VERSION_1_5
        && to_version   >= mi::neuraylib::MDL_VERSION_1_6) {

        // Convert all weak relative import declarations and resource file paths.
        Non_weak_relative_import_declaration_creator visitor1(
            m_mdl.get(), m_module.get(), m_module_name);
        visitor1.run();

        Non_weak_relative_resource_file_path_creator visitor2(
            m_module.get(), m_module_name, context);
        visitor2.run();

        // Do not run analyze_module() here. We might have created strict relative import
        // declarations and/or resource file paths, and the module version might still be below 1.3.
    }

    if(    (   from_version >= mi::neuraylib::MDL_VERSION_1_6
            && from_version <= mi::neuraylib::MDL_VERSION_1_7)
        && to_version >= mi::neuraylib::MDL_VERSION_1_8) {

        // Remove all aliases.
        Alias_remover visitor( m_module.get(), m_module_name);
        visitor.run();

        // Do not run analyze_module() here. We might have created Unicode identifiers, and the
        // module version might still be below 1.8.
    }

    Version_upgrader upgrader( m_mdl.get(), m_module.get(), to_version);
    upgrader.run();

    analyze_module( context);

    return context->get_result() == 0 ? 0 : -1;
}

mi::Sint32 Mdl_module_transformer::use_absolute_import_declarations(
    const char* include_filter,
    const char* exclude_filter,
    Execution_context* context)
{
    if( !is_module_valid( context))
        return -1;

    std::unique_ptr<std::wregex> include_regex, exclude_regex;
    if( !convert_filters( include_filter, exclude_filter, include_regex, exclude_regex, context))
        return -1;

    Import_entries_holder import_entries( m_mdlc_module, m_transaction, m_module.get());

    Absolute_import_declaration_creator visitor(
        m_mdl.get(), m_module.get(), m_module_name, include_regex.get(), exclude_regex.get());
    visitor.run();

    analyze_module( context);

    return context->get_result() == 0 ? 0 : -1;
}

mi::Sint32 Mdl_module_transformer::use_relative_import_declarations(
    const char* include_filter,
    const char* exclude_filter,
    Execution_context* context)
{
    if( !is_module_valid( context))
        return -1;

    std::unique_ptr<std::wregex> include_regex, exclude_regex;
    if( !convert_filters( include_filter, exclude_filter, include_regex, exclude_regex, context))
        return -1;

    int major, minor;
    m_module->get_version( major, minor);
    if( major == 1 && minor < 3) {
        add_error_message( context, "Transformation to (strict) relative import declarations "
            "requires MDL version >= 1.3", -5);
        return -1;
    }

    Import_entries_holder import_entries( m_mdlc_module, m_transaction, m_module.get());

    Relative_import_declaration_creator visitor(
        m_mdl.get(), m_module.get(), m_module_name, include_regex.get(), exclude_regex.get(),
        context);
    visitor.run();

    analyze_module( context);

    return context->get_result() == 0 ? 0 : -1;
}

mi::Sint32 Mdl_module_transformer::use_absolute_resource_file_paths(
    const char* include_filter,
    const char* exclude_filter,
    Execution_context* context)
{
    if( !is_module_valid( context))
        return -1;

    std::unique_ptr<std::wregex> include_regex, exclude_regex;
    if( !convert_filters( include_filter, exclude_filter, include_regex, exclude_regex, context))
        return -1;

    Import_entries_holder import_entries( m_mdlc_module, m_transaction, m_module.get());

    Absolute_resource_file_path_creator visitor(
        m_module.get(), m_module_name, include_regex.get(), exclude_regex.get());
    visitor.run();

    analyze_module( context);

    return context->get_result() == 0 ? 0 : -1;
}

mi::Sint32 Mdl_module_transformer::use_relative_resource_file_paths(
    const char* include_filter,
    const char* exclude_filter,
    Execution_context* context)
{
    if( !is_module_valid( context))
        return -1;

    std::unique_ptr<std::wregex> include_regex, exclude_regex;
    if( !convert_filters( include_filter, exclude_filter, include_regex, exclude_regex, context))
        return -1;

    int major, minor;
    m_module->get_version( major, minor);
    if( major == 1 && minor < 3) {
        add_error_message( context, "Transformation to (strict) relative resource file paths "
            "requires MDL version >= 1.3", -5);
        return -1;
    }

    Import_entries_holder import_entries( m_mdlc_module, m_transaction, m_module.get());

    Relative_resource_file_path_creator visitor(
        m_module.get(), m_module_name, include_regex.get(), exclude_regex.get(), context);
    visitor.run();

    analyze_module( context);

    return context->get_result() == 0 ? 0 : -1;
}

namespace {

class Inline_import_callback : public mi::mdl::IInline_import_callback
{
public:
    Inline_import_callback(
        const char* module_name,
        std::wregex* include_regex,
        std::wregex* exclude_regex)
      : m_module_name( module_name),
        m_include_regex( include_regex),
        m_exclude_regex( exclude_regex) { }

    bool inline_import( const mi::mdl::IModule* module) final
    {
        const mi::mdl::Module* module_impl = impl_cast<mi::mdl::Module>( module);
        if( module_impl->is_compiler_owned() || module_impl->is_native())
            return false;

        // For the top-level case of computing the minimum required MDL version.
        std::string module_name = module->get_name();
        if( module_name == m_module_name)
            return true;

        std::string mdl_module_name = MDL::encode_module_name( module_name);
        std::wstring module_name_wstr = STRING::utf8_to_wchar( mdl_module_name.c_str());
        if( m_include_regex && !std::regex_match( module_name_wstr, *m_include_regex))
            return false;
        if( m_exclude_regex &&  std::regex_match( module_name_wstr, *m_exclude_regex))
            return false;

        return true;
}

private:
    std::string m_module_name;
    std::wregex* m_include_regex;
    std::wregex* m_exclude_regex;
};

} // namespace

mi::Sint32 Mdl_module_transformer::inline_imported_modules(
    const char* include_filter,
    const char* exclude_filter,
    bool omit_anno_origin,
    Execution_context* context)
{
    if( !is_module_valid( context))
        return -1;

    std::unique_ptr<std::wregex> include_regex, exclude_regex;
    if( !convert_filters( include_filter, exclude_filter, include_regex, exclude_regex, context))
        return -1;

    Import_entries_holder import_entries( m_mdlc_module, m_transaction, m_module.get());

    // Compute the MDL version of the module.
    Inline_import_callback callback(
        m_module->get_name(), include_regex.get(), exclude_regex.get());
    std::set<const mi::mdl::Module*> done;
    // The ::anno::origin annotation requires MDL >= 1.5.
    mi::mdl::IMDL::MDL_version version
        = omit_anno_origin ? mi::mdl::IMDL::MDL_VERSION_1_0 : mi::mdl::IMDL::MDL_VERSION_1_5;
    // The computation here is conservative in the sense that we just look at the module version,
    // not at the features from that module that are actually used.
    get_min_required_mdl_version( m_module.get(), &callback, done, version);

    mi::base::Handle<mi::mdl::Module> new_module( impl_cast<mi::mdl::Module>(
        m_mdl->create_module( /*context*/ nullptr, m_module->get_name(), version)));

    mi::mdl::Def_set exports(
        0, mi::mdl::Def_set::hasher(), mi::mdl::Def_set::key_equal(), m_module->get_allocator());
    mi::mdl::Module_inliner::Reference_map references(
        mi::mdl::Module_inliner::Reference_map::key_compare(), m_module->get_allocator());
    mi::mdl::Module_inliner::Import_set imports(
        mi::mdl::Module_inliner::Import_set::key_compare(), m_module->get_allocator());
    size_t counter = 0;

    mi::mdl::Module_inliner inliner(
        m_module->get_allocator(),
        m_module.get(),
        new_module.get(),
        &callback,
        omit_anno_origin,
        /*is_root_module*/ true,
        references,
        imports,
        exports,
        counter);
    inliner.run();

    // copy the file name from the original module here: this is necessary to support
    // relative path of resources, which is otherwise not allowed for modules without
    // a file name (aka string modules)
    new_module->set_filename(m_module->get_filename());
    m_module = new_module;

    analyze_module( context);

    // remove the file name again
    m_module->set_filename(nullptr);

    return context->get_result() == 0 ? 0 : -1;
}

void Mdl_module_transformer::analyze_module( MDL::Execution_context* context)
{
    // Note that the AST dump is not guaranteed to be valid MDL (even for valid modules), e.g., it
    // generates empty selector strings for MDL < 1.7.
    SYSTEM::Access_module<CONFIG::Config_module> config_module( false);
    const CONFIG::Config_registry& registry = config_module->get_configuration();
    bool dump_ast = false;
    registry.get_value( "mdl_dump_ast_in_module_transformer", dump_ast);
    if( dump_ast)
        mi::mdl::dump_ast( m_module.get());

    mi::base::Handle<mi::mdl::IThread_context> thread_context(
        MDL::create_thread_context( m_mdl.get(), context));

    mi::mdl::Options& options = thread_context->access_options();
    options.set_option( MDL_OPTION_KEEP_ORIGINAL_RESOURCE_FILE_PATHS, "true");
    options.set_option( MDL_OPTION_OPT_LEVEL, "0");

    m_module->analyze( /*module_cache*/ nullptr, thread_context.get());
    if( !m_module->is_valid()) {
        convert_and_log_messages( m_module->access_messages(), context);
        add_error_message( context, "Module transformation failed.", -4);
    }
}

bool Mdl_module_transformer::is_module_valid( MDL::Execution_context* context)
{
    if( !m_module->is_valid()) {
        add_error_message( context,
            "A previous transformation failed and left this transformer's copy of the module in an "
            " invalid state.", -7);
        return false;
    }

    return true;
}

const mi::mdl::IModule* Mdl_module_transformer::get_module() const
{
    m_module->retain();
    return m_module.get();
}

bool Mdl_module_transformer::convert_filters(
    const char* include_filter,
    const char* exclude_filter,
    std::unique_ptr<std::wregex>& include_regex,
    std::unique_ptr<std::wregex>& exclude_regex,
    MDL::Execution_context* context)
{
    try {
        if( include_filter) {
            std::wstring include_filter_wstr = STRING::utf8_to_wchar( include_filter);
            include_regex = std::make_unique<std::wregex>(
                include_filter_wstr, std::wregex::extended);
        }
        if( exclude_filter) {
            std::wstring exclude_filter_wstr = STRING::utf8_to_wchar( exclude_filter);
            exclude_regex = std::make_unique<std::wregex>(
                exclude_filter_wstr, std::wregex::extended);
        }
        return true;
    } catch( const std::regex_error& ) {
        add_error_message( context, "Invalid regular expression.", -3);
        return false;
    }
}

namespace {

// Remove n filename/directories from the end of s.
bool remove_components( std::string& s, mi::Size n)
{
    while( n > 0) {
        size_t pos = s.find_last_of( "/\\");
        if( pos == std::string::npos)
            return false;
        s = s.substr( 0, pos);
        --n;
    }
    return true;
}

}

bool Mdl_module_transformer::same_search_path(
    const mi::mdl::IModule* module,
    const std::string& referenced_filename,
    mi::Size up_levels)
{
    std::string module_filename = module->get_filename();
    if( module_filename.empty() ^ referenced_filename.empty())
        return false;
    if( module_filename.empty() && referenced_filename.empty())
        return true;

    // Compute common prefix if both are in the same search path.
    std::string prefix = module_filename;
    if( !remove_components( prefix, up_levels+1)) {
        ASSERT( M_SCENE, false);
        return false;
    }

    // Check that it actually is a common prefix.
    mi::Size n = prefix.size();
    return referenced_filename.substr( 0, n) == prefix
        && referenced_filename.size() > n
        && (referenced_filename[n] == '/' || referenced_filename[n] == '\\');
}

void Mdl_module_transformer::get_min_required_mdl_version(
    const mi::mdl::Module* module,
    mi::mdl::IInline_import_callback* callback,
    std::set<const mi::mdl::Module*>& done,
    mi::mdl::IMDL::MDL_version& version)
{
    if( done.find( module) != done.end())
        return;

    if( callback->inline_import( module)) {
        mi::mdl::IMDL::MDL_version v = module->get_version();
        if( v > version)
            version = v;
    }

    for( size_t i = 0, n = module->get_import_count(); i < n; ++i) {
        mi::base::Handle<const mi::mdl::Module> import( module->get_import( i));
        get_min_required_mdl_version( import.get(), callback, done, version);
    }

    done.insert( module);
}

} // namespace MDL

} // namespace MI
