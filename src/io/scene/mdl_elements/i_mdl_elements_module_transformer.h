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

#ifndef IO_SCENE_MDL_ELEMENTS_I_MDL_ELEMENTS_MODULE_TRANSFORMER_H
#define IO_SCENE_MDL_ELEMENTS_I_MDL_ELEMENTS_MODULE_TRANSFORMER_H

#include <memory>
#include <regex>
#include <set>
#include <string>
#include <vector>
#include <stack>

#include <boost/core/noncopyable.hpp>

#include <mi/base/handle.h>
#include <mi/mdl/mdl_mdl.h>
#include <mi/neuraylib/imodule.h> // for Mdl_version

#include <base/system/main/access_module.h>
#include <mdl/compiler/compilercore/compilercore_visitor.h>

namespace mi {
namespace mdl {
class IInline_import_callback;
class IMDL;
class IModule;
class IName_factory;
class IQualified_name;
class Module;
class Scope;
}  // mdl
}  // mi

namespace MI {

namespace DB { class Transaction; }
namespace MDLC { class Mdlc_module; }

namespace MDL {

class Execution_context;

/// The module transformer allows to apply certain transformations on an MDL module.
///
/// All regular expressions related to module names use MDL names. For documentation, see the
/// counterparts in the public API in <mi/neuraylib/imdl_module_transformer.h>.
class Mdl_module_transformer : public boost::noncopyable
{
public:
    Mdl_module_transformer( DB::Transaction* transaction, const mi::mdl::IModule* module);

    ~Mdl_module_transformer();

    // public API methods

    mi::Sint32 upgrade_mdl_version(
        mi::neuraylib::Mdl_version version, Execution_context* context);

    mi::Sint32 use_absolute_import_declarations(
        const char* include_filter,
        const char* exclude_filter,
        Execution_context* context);

    mi::Sint32 use_relative_import_declarations(
        const char* include_filter,
        const char* exclude_filter,
        Execution_context* context);

    mi::Sint32 use_absolute_resource_file_paths(
        const char* include_filter,
        const char* exclude_filter,
        Execution_context* context);

    mi::Sint32 use_relative_resource_file_paths(
        const char* include_filter,
        const char* exclude_filter,
        Execution_context* context);

    mi::Sint32 inline_imported_modules(
        const char* include_filter,
        const char* exclude_filter,
        bool omit_anno_origin,
        Execution_context* context);

    // internal methods

    // Analyses the module being transformed.
    void analyze_module( Execution_context* context);

    /// Indicates whether the module is valid. Adds a suitable error message to the context if not.
    bool is_module_valid( Execution_context* context);

    /// Returns the module being transformed.
    const mi::mdl::IModule* get_module() const;

    /// Sets up regular expressions for both filters.
    ///
    /// The smart pointer is left unchanged if the corresponding filter is \c NULL. Returns \c true
    /// in case of success, or \c false if any filter is not a valid regular expression.
    ///
    /// \note The conversion function used here (and for the strings that are matched later) does
    ///       not support code points beyond U+FFFF, even if the underlying wchar implementation
    ///       would support such code points, e.g., on Linux.
    static bool convert_filters(
        const char* include_filter,
        const char* exclude_filter,
        std::unique_ptr<std::wregex>& include_regex,
        std::unique_ptr<std::wregex>& exclude_regex,
        Execution_context* context);

    /// Indicates whether the a module and some file (typically another module or a resource) are in
    /// the same search path.
    ///
    /// \param up_levels   The minimum number of ".." references needed to reach the file from the
    ///                    importing module.
    static bool same_search_path(
        const mi::mdl::IModule* module,
        const std::string& referenced_filename,
        mi::Size up_levels);

    /// Returns the maximum of the MDL version of the given module and all its (direct and indirect)
    /// imports for which the callback returns \c true (and the initial value of \p version).
    ///
    /// Pass an empty set for \p done.
    static void get_min_required_mdl_version(
        const mi::mdl::Module* module,
        mi::mdl::IInline_import_callback* callback,
        std::set<const mi::mdl::Module*>& done,
        mi::mdl::IMDL::MDL_version& version);

private:
    SYSTEM::Access_module<MDLC::Mdlc_module> m_mdlc_module;

    DB::Transaction* m_transaction;
    std::vector<std::string> m_module_name; ///< Core module name.

    mi::base::Handle<mi::mdl::IMDL> m_mdl;
    mi::base::Handle<mi::mdl::Module> m_module;
};

/// Helper class to convert constants of user types back to references and constructor calls.
/// This is necessary whenever the analyzer is called again after a module transformation,
/// because the analyzer will add new instances of the user types, while the user constants
/// would still point to the old instances.
class User_constant_remover : protected mi::mdl::Module_visitor
{
public:
    /// Constructor.
    ///
    /// \param imodule  the module to process
    User_constant_remover(
        mi::mdl::IModule *imodule);

    /// Process it.
    void process();

protected:
    /// Converts the given literal, if it contains a user constant, to a reference,
    /// otherwise keeps it unmodified.
    mi::mdl::IExpression *post_visit(mi::mdl::IExpression_literal *expr) override;

private:
    /// Converts a value, if it contains a user constant, otherwise returns nullptr.
    mi::mdl::IExpression *convert_user_value(mi::mdl::IValue const *value);

    // Converts an enum value to a reference expression.
    mi::mdl::IExpression *convert_enum_value(mi::mdl::IValue_enum const *value);

    /// Converts a struct or array value to a call expression to the corresponding constructor.
    mi::mdl::IExpression *convert_compound_value(mi::mdl::IValue_compound const *value);

    /// Creates a qualified name from a scope.
    mi::mdl::IQualified_name *create_qualified_name(mi::mdl::Scope const *scope);

protected:
    mi::mdl::Module *m_module;
    mi::mdl::IExpression_factory *m_ef;
    mi::mdl::IValue_factory *m_vf;
    mi::mdl::IName_factory *m_nf;

private:
    using Symbol_stack = std::stack<const mi::mdl::ISymbol*>;

    /// Temporarily used symbol stack.
    Symbol_stack m_sym_stack;
};

} // namespace MDL

} // namespace MI

#endif // IO_SCENE_MDL_ELEMENTS_I_MDL_ELEMENTS_MODULE_TRANSFORMER_H
