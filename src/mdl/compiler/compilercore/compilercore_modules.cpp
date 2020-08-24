/******************************************************************************
 * Copyright (c) 2011-2020, NVIDIA CORPORATION. All rights reserved.
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

#include <mi/base/interface_implement.h>
#include <mi/base/handle.h>
#include <mi/mdl/mdl_modules.h>
#include <mi/mdl/mdl_names.h>

#include <cstring>

#include "compilercore_cc_conf.h"
#include "compilercore_bitset.h"
#include "compilercore_mdl.h"
#include "compilercore_modules.h"
#include "compilercore_positions.h"
#include "compilercore_analysis.h"
#include "compilercore_optimizer.h"
#include "compilercore_def_table.h"
#include "compilercore_tools.h"
#include "compilercore_assert.h"
#include "compilercore_checker.h"
#include "compilercore_dynamic_memory.h"
#include "compilercore_serializer.h"
#include "compilercore_mangle.h"
#include "compilercore_manifest.h"
#include "compilercore_overload.h"
#include "compilercore_tools.h"
#include "compilercore_fatal.h"
#include "compilercore_file_resolution.h"
#include "compilercore_func_hash.h"

#ifndef M_PIf
#   define M_PIf        3.14159265358979323846f
#endif


namespace mi {
namespace mdl {

namespace {

/// Helper class to match the parameters of a signature.
class Signature_matcher {
public:
    /// Constructor.
    ///
    /// \param alloc     the allocator
    /// \param compiler  the MDL compiler
    Signature_matcher(
        IAllocator   *alloc,
        IMDL         *compiler,
        string const &signature)
    : m_dag_mangler(alloc, compiler)
    , m_signature(signature)
    {
    }

    /// Check if parameters of a given definition match.
    ///
    /// \param def  the definition to check
    bool match_param(IDefinition const *def)
    {
        IType_function const *ftype = as<IType_function>(def->get_type());
        if (ftype == NULL) {
            // not a function type
            return false;
        }

        string mangled_name = m_dag_mangler.mangle(def, (char const *)NULL, true);
        return mangled_name == m_signature;
    }

private:
    /// The DAG mangler.
    DAG_mangler m_dag_mangler;

    /// The signature.
    string      m_signature;
};

}  // anonymous


// Get the major version.
int Semantic_version::get_major() const
{
    return m_major;
}

// Get the minor version.
int Semantic_version::get_minor() const
{
    return m_minor;
}

// Get the patch version.
int Semantic_version::get_patch() const
{
    return m_patch;
}

// Get the pre-release string.
char const *Semantic_version::get_prerelease() const
{
    return m_prelease;
}

// Compare for less or equal.
bool Semantic_version::operator<=(Semantic_version const &o) const
{
    if (m_major == o.m_major) {
        if (m_minor == o.m_minor) {
            if (m_patch == o.m_patch) {
                // "no prerelease" is always >= the "with prerelease"
                if (m_prelease == NULL || m_prelease[0] == '\0') {
                    // they are == if the other has NO prerelease, otherwise this is >
                    return o.m_prelease == NULL || o.m_prelease[0] == '\0';
                }
                if (o.m_prelease == NULL || o.m_prelease[0] == '\0') {
                    // they are == if this has NO prerelease, otherwise this is <
                    return true;
                }
                return strcmp(m_prelease, o.m_prelease) <= 0;
            }
            return m_patch < o.m_patch;
        }
        return m_minor < o.m_minor;
    }
    return m_major < o.m_major;
}

// Constructor.
Semantic_version::Semantic_version(
    int        major,
    int        minor,
    int        patch,
    char const *prerelease)
: m_major(major)
, m_minor(minor)
, m_patch(patch)
, m_prelease(prerelease == NULL ? "" : prerelease)
{
}

// Constructor.
Semantic_version::Semantic_version(
    ISemantic_version const &ver)
: m_major(ver.get_major())
, m_minor(ver.get_minor())
, m_patch(ver.get_patch())
, m_prelease(ver.get_prerelease())
{
}

// Constructor.
Module::Module(
    IAllocator        *alloc,
    MDL               *compiler,
    size_t            unique_id,
    char const        *module_name,
    char const        *file_name,
    IMDL::MDL_version version,
    unsigned          flags)
: Base(alloc)
, m_arena(alloc)
, m_compiler(compiler)
, m_unique_id(unique_id)
, m_absname(NULL)
, m_filename(Arena_strdup(m_arena, file_name))
, m_qual_name(NULL)
, m_is_analyzed(false)
, m_is_valid(false)
, m_is_stdlib((flags & MF_IS_STDLIB) != 0)
, m_is_builtins((flags & MF_IS_BUILTIN) != 0)
, m_is_mdle((flags & MF_IS_MDLE) != 0)
, m_is_native((flags & MF_IS_NATIVE) != 0)
, m_is_compiler_owned((flags & (MF_IS_STDLIB|MF_IS_OWNED)) != 0)
, m_is_debug((flags & MF_IS_DEBUG) != 0)
, m_is_hashed((flags & MF_IS_HASHED) != 0)
, m_sema_version(NULL)
, m_mdl_version(version)
, m_msg_list(alloc, file_name)
, m_sym_tab(m_arena)
, m_name_factory(m_sym_tab, m_arena)
, m_decl_factory(m_arena)
, m_expr_factory(m_arena)
, m_stmt_factory(m_arena)
, m_type_factory(m_arena, compiler, &m_sym_tab)
, m_value_factory(m_arena, m_type_factory)
, m_anno_factory(m_arena)
, m_def_tab(*this)
, m_declarations(&m_arena)
, m_imported_modules(alloc)
, m_exported_definitions(alloc)
, m_builtin_definitions(alloc)
, m_deprecated_msg_map(Deprecated_msg_map::key_compare(), alloc)
, m_arc_mdl_version(IMDL::MDL_LATEST_VERSION)
, m_archive_versions(&m_arena)
, m_res_table(&m_arena)
, m_func_hashes(Func_hash_map::key_compare(), alloc)
{
    MDL_ASSERT(file_name != NULL);
    if (!m_is_compiler_owned) {
        // "owned" by the compiler itself, so do not reference it here
        // to avoid dependency cycles
        compiler->retain();
    }
    set_name(module_name);
}

// Destructor.
Module::~Module()
{
    if (!m_is_compiler_owned) {
        // "owned" by the compiler itself, so do not reference it here
        // to avoid dependency cycles
        m_compiler->release();
    }
}

// Get the absolute name of the module.
char const *Module::get_name() const
{
    return m_absname;
}

// Get the owner file name of the message list.
char const *Module::get_msg_name() const
{
    return m_msg_list.get_fname(0);
}

// Set the owner file name of the message list.
void Module::set_msg_name(char const *msg_name)
{
    m_msg_list.set_fname(0, msg_name);
}

// Get the absolute name of the module as a qualified name.
IQualified_name const *Module::get_qualified_name() const
{
    return m_qual_name;
}

// Set the absolute name of the module.
void Module::set_name(const char *name)
{
    // internalize and remember
    m_absname   = name ? Arena_strdup(m_arena, name) : "";
    m_qual_name = qname_from_cstring(m_absname);
}

// Get the absolute name of the file from which the module was loaded.
char const *Module::get_filename() const
{
    return m_filename;
}

// Convert a MDL version into major, minor pair.
void Module::get_version(IMDL::MDL_version version, int &major, int &minor)
{
    switch (version) {
    case IMDL::MDL_VERSION_1_0:     major = 1; minor = 0; return;
    case IMDL::MDL_VERSION_1_1:     major = 1; minor = 1; return;
    case IMDL::MDL_VERSION_1_2:     major = 1; minor = 2; return;
    case IMDL::MDL_VERSION_1_3:     major = 1; minor = 3; return;
    case IMDL::MDL_VERSION_1_4:     major = 1; minor = 4; return;
    case IMDL::MDL_VERSION_1_5:     major = 1; minor = 5; return;
    case IMDL::MDL_VERSION_1_6:     major = 1; minor = 6; return;
    case IMDL::MDL_VERSION_1_7:     major = 1; minor = 7; return;
    }
    MDL_ASSERT(!"MDL version not known");
    major = 0;
    minor = 0;
}

// Get the language version.
void Module::get_version(int &major, int &minor) const
{
    get_version(m_mdl_version, major, minor);
}

// Set the language version.
bool Module::set_version(MDL *compiler, int major, int minor, bool enable_experimental_features)
{
    return compiler->check_version(major, minor, m_mdl_version, enable_experimental_features);
}

// Analyze the module.
bool Module::analyze(
    IModule_cache   *cache,
    IThread_context *context)
{
    mi::base::Handle<Thread_context> hctx;

    Thread_context *ctx = impl_cast<Thread_context>(context);

    if (ctx == NULL) {
        // user does not pass a context, create a temporary one
        hctx = mi::base::make_handle(m_compiler->create_thread_context());
        ctx  = hctx.get();
    }

    NT_analysis nt_analysis(m_compiler, *this, *ctx, cache);
    nt_analysis.run();

    Sema_analysis sema_analysis(m_compiler, *this, *ctx);
    sema_analysis.run();

    Optimizer::run(
        m_compiler,
        *this,
        *ctx,
        nt_analysis,
        sema_analysis.get_statement_info_data());

    // run the checker
    MDL_ASSERT(
        Module_checker::check(m_compiler, this, /*verbose=*/false) && "Module check failed");

    // we have analyzed it
    set_analyze_result(access_messages().get_error_message_count() == 0);

    if (has_function_hashes()) {
        // compute function hashes if necessary
        Sema_hasher::run(get_allocator(), this, m_compiler);
    }

    // drop the reference count of all imports, it was increased
    // during load_module_to_import() inside NT_analysis::run().
    // Note that this does NOT drop all imports, it just sets the count
    // to one again.
    drop_import_entries();

    return m_is_valid;
}

// Get all known function hashes.
void Module::get_all_function_hashes(
    Function_hash_set &hashes) const
{
    hashes.clear();

    for (Func_hash_map::const_iterator it(m_func_hashes.begin()), end(m_func_hashes.end());
        it != end;
        ++it)
    {
        IModule::Function_hash const &fh = it->second;

        hashes.insert(fh);
    }
}


// Check if the module has been analyzed.
bool Module::is_analyzed() const
{
    return m_is_analyzed;
}

// Check if the module contents are valid.
bool Module::is_valid() const
{
    return m_is_valid;
}

// Get the number of imports.
int Module::get_import_count() const
{
    return int(m_imported_modules.size());
}

// Get the import at index.
Module const *Module::get_import(int index) const
{
    if (index < 0 || index >= get_import_count())
        return NULL;
    Import_entry const &entry = m_imported_modules[index];

    Module const *res = entry.get_module();

    if (res == NULL) {
        MDL_ASSERT(! "Reload of imported modules NYI");
        return res;
    }
    res->retain();
    return res;
}

// Get the number of exported definitions.
int Module::get_exported_definition_count() const
{
    return int(m_exported_definitions.size());
}

// Get the exported definition at index.
Definition const *Module::get_exported_definition(int index) const
{
    if (index < 0 || index >= get_exported_definition_count())
        return NULL;
    return m_exported_definitions[index];
}

// Get the number of declarations.
int Module::get_declaration_count() const
{
    return int(m_declarations.size());
}

// Get the declaration at index.
IDeclaration const *Module::get_declaration(int index) const
{
    return m_declarations[index];
}

// Add a declaration.
void Module::add_declaration(IDeclaration const *decl)
{
    MDL_ASSERT(decl != NULL);
    m_declarations.push_back(decl);

    // module is modified and must be re-analyzed
    m_is_analyzed = m_is_valid = false;
}

/// Add an import at the end of all other imports or namespace aliases.
void Module::add_import(char const *name)
{
    IQualified_name     *qname              = qname_from_cstring(name);
    IDeclaration_import *import_declaration = m_decl_factory.create_import(0);
    import_declaration->add_name(qname);

    Declaration_vector::iterator it(m_declarations.begin()), end(m_declarations.end());
    while (it != end &&
        ((*it)->get_kind() == IDeclaration::DK_IMPORT ||
         (*it)->get_kind() == IDeclaration::DK_NAMESPACE_ALIAS))
        ++it;
    m_declarations.insert(it, import_declaration);

    // module is modified and must be re-analyzed
    m_is_analyzed = m_is_valid = false;
}

// Get the name factory.
Name_factory *Module::get_name_factory() const
{
    return &m_name_factory;
}

// Get the expression factory.
Expression_factory *Module::get_expression_factory() const
{
    return &m_expr_factory;
}

// Get the statement factory.
Statement_factory *Module::get_statement_factory() const
{
    return &m_stmt_factory;
}

// Get the declaration factory.
Declaration_factory *Module::get_declaration_factory() const
{
    return &m_decl_factory;
}

// Get the type factory.
Type_factory *Module::get_type_factory() const
{
    return &m_type_factory;
}

// Get the value factory.
Value_factory *Module::get_value_factory() const
{
    return &m_value_factory;
}

// Get the annotation factory.
Annotation_factory *Module::get_annotation_factory() const
{
    return &m_anno_factory;
}

// Access messages.
Messages const &Module::access_messages() const
{
    return m_msg_list;
}

// Get the absolute name of the module a definitions belongs to.
char const *Module::get_owner_module_name(IDefinition const *idef) const
{
    Definition const *def = impl_cast<Definition>(idef);

    if (def->has_flag(Definition::DEF_IS_PREDEFINED))
        return "";
    size_t import_idx = def->get_original_import_idx();
    if (import_idx == 0) {
        // from this module
        return m_absname;
    }
    if (Import_entry const *entry = get_import_entry(import_idx)) {
        return entry->get_absolute_name();
    }
    return NULL;
}

// Get the module a definitions belongs to.
Module const *Module::get_owner_module(IDefinition const *idef) const
{
    Definition const *def = impl_cast<Definition>(idef);

    if (def->has_flag(Definition::DEF_IS_PREDEFINED)) {
        this->retain();
        return this;
    }
    size_t import_idx = def->get_original_import_idx();
    if (import_idx == 0) {
        // not imported
        this->retain();
        return this;
    }
    if (Import_entry const *entry = get_import_entry(import_idx)) {
        Module const *owner = entry->get_module();
        if (owner)
            owner->retain();
        return owner;
    }
    return NULL;
}

// Get the original definition (if imported from another module).
Definition const *Module::get_original_definition(IDefinition const *idef) const
{
    Module const *unused = NULL;
    return get_original_definition(idef, unused);
}

// Set the absolute name of the file from which the module was loaded.
void Module::set_filename(char const *name)
{
    m_filename = name != NULL ? Arena_strdup(m_arena, name) : "";

    // also use this name for error reports
    set_msg_name(m_filename);
}

// Get the original definition (if imported from another module).
Definition const *Module::get_original_definition(
    IDefinition const *idef,
    Module const      *&mod_owner) const
{
    Definition const *def = impl_cast<Definition>(idef);
    mod_owner = NULL;

    if (def->has_flag(Definition::DEF_IS_PREDEFINED)) {
        Definition const *def_def = def->get_definite_definition();
        return def_def != NULL ? def_def : def;
    }
    size_t import_idx = def->get_original_import_idx();
    if (import_idx == 0) {
        // not imported
        Definition const *def_def = def->get_definite_definition();
        return def_def != NULL ? def_def : def;
    }
    if (Import_entry const *entry = get_import_entry(import_idx)) {
        if (Module const *owner = entry->get_module()) {
            def = owner->lookup_original_definition(def);
            if (def != NULL) {
                Definition const *def_def = def->get_definite_definition();
                if (def_def != NULL)
                    def = def_def;
                mod_owner = owner;
            }
            return def;
        }
    }
    return NULL;
}

/// Get the first constructor of a given type.
Definition const *Module::get_first_constructor(IType const *type) const
{
    if (Scope const *scope = m_def_tab.get_type_scope(type)) {
        ISymbol const *sym = scope->get_scope_name();

        return scope->find_definition_in_scope(sym);
    }
    return NULL;
}

// Get the next constructor of a type or NULL.
Definition const *Module::get_next_constructor(Definition const *constr_def) const
{
    if (constr_def != NULL && constr_def->get_kind() == Definition::DK_CONSTRUCTOR) {
        return constr_def->get_prev_def();
    }
    return NULL;
}

// Get the first conversion operator of a given type.
Definition const *Module::get_first_conversion_operator(IType const *type) const
{
    if (Scope const *scope = m_def_tab.get_type_scope(type)) {
        // Note: cannot be overloaded
        for (Definition const *def = scope->get_first_definition_in_scope();
            def != NULL;
            def = def->get_next_def_in_scope())
        {
            if (def->get_kind() == IDefinition::DK_CONSTRUCTOR &&
                def->get_semantics() == IDefinition::DS_CONV_OPERATOR) {
                return def;
            }
        }
    }
    return NULL;
}

// Get the next conversion operator of a type or NULL.
Definition const *Module::get_next_conversion_operator(Definition const *op_def) const
{
    if (op_def == NULL)
        return NULL;

    if (op_def->get_kind() == Definition::DK_CONSTRUCTOR &&
        op_def->get_semantics() == IDefinition::DS_CONV_OPERATOR)
    {
        // Note: cannot be overloaded
        for (Definition const *def = op_def->get_next_def_in_scope();
            def != NULL;
            def = def->get_next_def_in_scope())
        {
            if (def->get_kind() == IDefinition::DK_CONSTRUCTOR &&
                def->get_semantics() == IDefinition::DS_CONV_OPERATOR) {
                    return def;
            }
        }
    }
    return NULL;

}

// Get the the number of constructors of a given type.
int Module::get_type_constructor_count(IType const *type) const
{
    Definition const *def = get_first_constructor(type);
    if (def == NULL)
        return -1;
    int count = 0;
    do {
        def = get_next_constructor(def);
        ++count;
    } while (def != NULL);
    return count;
}

// Get the i'th constructor of a type or NULL.
IDefinition const *Module::get_type_constructor(IType const *type, int index) const
{
    if (index < 0)
        return NULL;
    
    Definition const *def = get_first_constructor(type);
    for (; index > 0 && def != NULL; def = get_next_constructor(def)) {
        --index;
    }
    return def;
}

// Get the the number of conversion operators of a given type.
int Module::get_conversion_operator_count(IType const *type) const
{
    Definition const *def = get_first_conversion_operator(type);
    if (def == NULL)
        return -1;
    int count = 0;
    do {
        def = get_next_conversion_operator(def);
        ++count;
    } while (def != NULL);
    return count;
}

// Get the i'th conversion operator of a type or NULL.
IDefinition const *Module::get_conversion_operator(IType const *type, int index) const
{
    if (index < 0)
        return NULL;

    Definition const *def = get_first_conversion_operator(type);
    for (; index > 0 && def != NULL; def = get_next_conversion_operator(def)) {
        --index;
    }
    return def;
}

// Check if a given identifier is defined at global scope in this module.
bool Module::is_name_defined(char const *name) const
{
    if (ISymbol const *sym = m_sym_tab.lookup_symbol(name)) {
        // not nice, but we do not modify the table here
        Definition_table *def_tab = const_cast<Definition_table *>(&m_def_tab);

        Definition_table::Scope_transition(*def_tab, def_tab->get_global_scope());

        Definition const *def = def_tab->get_definition(sym);
        return def != NULL;
    }
    return false;
}

// Get the number of builtin definitions.
int Module::get_builtin_definition_count() const
{
    return int(m_builtin_definitions.size());
}

// Get the builtin definition at index.
IDefinition const *Module::get_builtin_definition(int index) const
{
    if (0 <= index && size_t(index) < m_builtin_definitions.size())
        return m_builtin_definitions[index];
    return NULL;
}

// Returns true if this is a module from the standard library.
bool Module::is_stdlib() const
{
    return m_is_stdlib;
}

// Returns true if this is a module is the one and only builtins module.
bool Module::is_builtins() const
{
    return m_is_builtins;
}

// Returns true if this is an MDLE module.
bool Module::is_mdle() const
{
    return m_is_mdle;
}

// Returns the amount of used memory by this module.
size_t Module::get_memory_size() const
{
    size_t res = sizeof(*this);

    res += m_arena.get_chunks_size();
    res += m_def_tab.get_memory_size() - sizeof(m_def_tab);
    res += dynamic_memory_consumption(m_msg_list);
    res += dynamic_memory_consumption(m_declarations);
    res += dynamic_memory_consumption(m_imported_modules);
    res += dynamic_memory_consumption(m_exported_definitions);
    res += dynamic_memory_consumption(m_builtin_definitions);

    return res;
}

// Drop all import entries.
void Module::drop_import_entries() const
{
    // Don't do that for standard library modules, it is pointless
    // because stdlib module import only other stdlib modules ...
    // Moreover, to deserialize a module, at least state must be complete ...
    if (is_stdlib())
        return;

    mi::base::Lock &weak_lock = m_compiler->get_weak_module_lock();

    for (size_t i = 0, n = m_imported_modules.size(); i < n; ++i) {
        Import_entry &entry = m_imported_modules[i];

        if (Module const *import = entry.get_module())
            import->drop_import_entries();
        entry.drop_module(weak_lock);
    }
}

// Restore all import entries using a module cache.
bool Module::restore_import_entries(IModule_cache *cache) const
{
    // don't do that for standard library modules, it is pointless
    // because stdlib module import only other stdlib modules ...
    if (is_stdlib())
        return true;

    bool result = true;

    mi::base::Lock &weak_lock = m_compiler->get_weak_module_lock();

    for (size_t i = 0, n = m_imported_modules.size(); i < n; ++i) {
        Import_entry &entry = m_imported_modules[i];

        Module const *import = entry.lock_module(weak_lock);
        if (import == NULL) {
            if (cache == NULL)
                return false;

            mi::base::Handle<IModule const> imod(cache->lookup(entry.get_absolute_name(), NULL));
            if (imod.is_valid_interface()) {
                import = impl_cast<Module>(imod.get());
                entry.enter_module(weak_lock, import);
            } else {
                // restoration failed, really bad
                result = false;
            }
        }
        if (import != NULL) {
            result = result && import->restore_import_entries(cache);
        }
    }
    return result;
}

/// Lookup a type given by its symbol.
static IType const *lookup_type(
    Definition_table const &def_tab,
    vector<ISymbol const *>::Type const &syms)
{
    Definition const *def   = NULL;
    Scope const      *scope = def_tab.get_global_scope();

    size_t l = syms.size();
    if (l > 1) {
        // name spaces
        for (size_t i = 0; i < l - 1; ++i) {
            scope = scope->find_named_subscope(syms[i]);
            if (scope == NULL) {
                return NULL;
            }
        }
        def = scope->find_definition_in_scope(syms[l - 1]);
    } else {
        // could be a global name like "int", search also in parent
        def = scope->find_def_in_scope_or_parent(syms[0]);
    }

    if (def != NULL) {
        if (def->get_kind() == Definition::DK_TYPE) {
            return def->get_type();
        }
    }
    return NULL;
}

static bool parse_parameter_signature(
    Module const                *owner,
    char const * const          param_type_names[],
    size_t                      num_param_type_names,
    vector<IType const *>::Type &arg_types)
{
#define SKIP_SPACE while (isspace(start[0])) ++start

    IAllocator             *alloc   = owner->get_allocator();
    Symbol_table const     &st      = owner->get_symbol_table();
    Definition_table const &def_tab = owner->get_definition_table();
    Type_factory const     &tf      = *owner->get_type_factory();
    char const             *absname = owner->get_name();
    size_t                 l        = strlen(absname);

    for (size_t i = 0; i < num_param_type_names; ++i) {

        char const *start = &param_type_names[i][0];
        char const *end   = start + strlen(param_type_names[i]);

        bool has_brackets = *(end-1) == ']';
        char const *left_bracket  = nullptr;
        char const *right_bracket = nullptr;
        if (has_brackets) {
            right_bracket = end-1;
            left_bracket  = right_bracket;
            while(left_bracket != start && left_bracket[0] != '[')
                --left_bracket;
            if (left_bracket == start)
                return false;
        }

        // short-cut: if we see here the "module name", skip it:
        // this is because user defined types uses always the full name in DAG signatures ...
        if (strncmp(absname, start, l) == 0 && start[l] == ':' && start[l + 1] == ':')
            start += l + 2;

        vector<ISymbol const *>::Type syms(alloc);

        if (start[0] == ':' && start[1] == ':') {
            do {
                start += 2;

                char const *p = start;

                ++p;
                while (p[0] != '\0' && p[0] != ':' && p != left_bracket && p != end)
                    ++p;

                string name(start, p, alloc);
                start = p;

                ISymbol const *sym = st.lookup_symbol(name.c_str());
                if (sym == NULL)
                    return false;
                syms.push_back(sym);
            } while (start[0] == ':' && start[1] == ':');
        } else {
            char const *p = start;

            // IDENT = LETTER { LETTER | DIGIT | '_' } .
            if (!isalpha(p[0]))
                return false;
            ++p;
            while (isalnum(p[0]) || p[0] == '_')
                ++p;

            string name(start, p, alloc);
            start = p;

            ISymbol const *sym = st.lookup_symbol(name.c_str());
            if (sym == NULL)
                return false;

            syms.push_back(sym);
        }
        SKIP_SPACE;

        if (has_brackets && start != left_bracket)
            return false;
        if (!has_brackets && start != end)
            return false;

        IType const *type = lookup_type(def_tab, syms);
        if (type == NULL)
            return false;

        if (!has_brackets) {
            arg_types.push_back(type);
            continue;
        }

        start = left_bracket + 1;
        SKIP_SPACE;

        if (isdigit(start[0])) {

            // immediate sized array
            int size = 0;
            do {
                size = 10 * size;
                switch (start[0]) {
                case '0':           break;
                case '1': size += 1; break;
                case '2': size += 2; break;
                case '3': size += 3; break;
                case '4': size += 4; break;
                case '5': size += 5; break;
                case '6': size += 6; break;
                case '7': size += 7; break;
                case '8': size += 8; break;
                case '9': size += 9; break;
                }
                ++start;
            }  while (isdigit(start[0]));

            IType const *a_type = tf.find_array(type, size);
            if (a_type == NULL) {
                // If this fails, this module does NOT have an immediate sized array type of
                // the given size.
                // That means, exact overloads WILL fail. However, there might be overloads
                // of deferred size that would allow calling with immediate size ...
                // Two possible solutions:
                //
                // 1) Create the array type. bad, as this would litter the type factory with
                //    types no one uses
                // 2) Return one matching deferred array type. This does not litter the type
                //    factory AND matches exact the same cases, so this seems to be a good
                //    solution ...
                a_type = tf.find_any_deferred_array(type);
                if (a_type == NULL) {
                    // none exists, so there is then no possible overload
                    return false;
                }
            }
            type = a_type;


        } else {

            // deferred sized array
            if (start == right_bracket) {
                // accept T[] as deferred sized array
            } else if (isalpha(start[0])) {
                // accept T[identifier] as deferred sized array
                ++start;
                while (isalnum(start[0]) || start[0] == '_') {
                    ++start;
                }
            }

            // just use any deferred type
            type = tf.find_any_deferred_array(type);
            if (type == NULL) {
                // none exists, so there is then no possible overload
                return false;
            }
        }

        SKIP_SPACE;
        if (start != right_bracket)
            return false;

        // found one
        arg_types.push_back(type);
   }

   return true;
}

namespace {

    class Overload_result_set : public Allocator_interface_implement<IOverload_result_set> {
        typedef Allocator_interface_implement<IOverload_result_set> Base;
    public:
        /// Constructor.
        ///
        /// \param owner         the owner module of this overload result
        /// \param compiler      the MDL compiler interface
        /// \param reverse_list  the list of overload results in reversed order
        Overload_result_set(
            Module const *owner,
            MDL          *compiler,
            Overload_solver::Definition_list const &reverse_list)
        : Base(owner->get_allocator())
        , m_reverse_list(reverse_list)
        , m_owner(owner, mi::base::DUP_INTERFACE)
        , m_it(m_reverse_list.begin())
        , m_ctx(compiler->create_thread_context())
        {
        }

        /// Get the first result or NULL if no results.
        Definition const *first() const MDL_FINAL
        {
            m_it = m_reverse_list.rbegin();
            return next();
        }

        /// Get the next result or NULL if no more results
        Definition const *next() const MDL_FINAL
        {
            if (m_it != m_reverse_list.rend()) {
                Definition const *res = *m_it;
                ++m_it;

                return res;
            }
            return NULL;
        }

        /// Get the first result as a DAG signature.
        char const *first_signature() const MDL_FINAL
        {
            return signature(first());
        }

        /// Get the first result as a DAG signature.
        char const *next_signature() const MDL_FINAL
        {
            return signature(next());
        }

    private:
        /// Convert a definition into a (DAG) signature.
        char const *signature(Definition const *def) const {
            if (def != NULL) {
                return m_owner->mangle_dag_name(def, m_ctx.get());
            }
            return NULL;
        }

    private:
        Overload_solver::Definition_list                                 m_reverse_list;
        mi::base::Handle<Module const>                                   m_owner;
        mutable Overload_solver::Definition_list::const_reverse_iterator m_it;
        mutable mi::base::Handle<IThread_context>                        m_ctx;
    };

}  // anonymous

// Lookup a function definition given by its name and an array
// of (positional) parameter types.
IOverload_result_set const *Module::find_overload_by_signature(
    char const         *func_name,
    char const * const param_type_names[],
    size_t             num_param_type_names) const
{
    if (!is_builtins()) {
        // func_name must be an absolute name, so check the module name first
        size_t l = strlen(m_absname);
        if (strncmp(func_name, m_absname, l) != 0)
            return NULL;
        if (func_name[l] != ':' || func_name[l + 1] != ':')
            return NULL;
        func_name += l + 2;
    } else {
        if (func_name[0] != ':' || func_name[1] != ':')
            return NULL;
        func_name += 2;
    }

    // the rest must be an identifier, we do not allow to access an imported entity in
    // a namespace
    ISymbol const *sym = m_sym_tab.lookup_symbol(func_name);
    if (sym == NULL)
        sym = m_sym_tab.lookup_operator_symbol(func_name, num_param_type_names);
    if (sym == NULL)
        return NULL;

    // create a type list from the parameter signature
    IAllocator *alloc = get_allocator();
    vector<IType const *>::Type arg_types(alloc);
    if (num_param_type_names > 0) {
        if (!parse_parameter_signature(this, param_type_names, num_param_type_names, arg_types))
            return NULL;
    }

    Definition const *def = NULL;

    // check for conversion operator; currently only conversion to int is supported
    if (sym->get_id() == ISymbol::SYM_TYPE_INT) {
        if (num_param_type_names == 1) {
            char const *tname = param_type_names[0];
            size_t     l      = strlen(m_absname);
            bool       error  = false;

            if (strncmp(tname, m_absname, l) == 0 && tname[l] == ':' && tname[l + 1] == ':') {
                tname = tname + l + 2;
            } else {
                error = !is_builtins() || strchr(tname, ':') != NULL;
            }
            if (!error) {
                ISymbol const *tsym = m_sym_tab.lookup_symbol(tname);
                if (tsym != NULL) {
                    Scope const *search_scope =
                        is_builtins() ?
                        m_def_tab.get_predef_scope() :
                        m_def_tab.get_global_scope();
                    def = search_scope->find_definition_in_scope(tsym);
                }
            }
        }

        if (def != NULL && def->get_kind() == IDefinition::DK_TYPE) {
            Scope *search_scope = def->get_own_scope();

            Definition const *def = search_scope->find_definition_in_scope(sym);
            if (def != NULL) {
                for (Definition const *cdef = def; cdef != NULL; cdef = cdef->get_prev_def()) {
                    Definition::Kind kind = cdef->get_kind();
                    if (kind != IDefinition::DK_FUNCTION &&
                        kind != IDefinition::DK_CONSTRUCTOR &&
                        kind != IDefinition::DK_OPERATOR)
                    {
                        // neither a function nor a constructor nor an operator
                        continue;
                    }

                    bool match = false;
                    if (IType_function const *f_tp = as<IType_function>(cdef->get_type())) {
                        if (f_tp->get_parameter_count() == 1) {
                            ISymbol const *p_sym;
                            IType const *p_type;

                            f_tp->get_parameter(0, p_type, p_sym);

                            match = arg_types[0]->skip_type_alias() == p_type->skip_type_alias();
                        }
                    }
                    if (match) {
                        if (cdef->has_flag(Definition::DEF_IS_EXPORTED)) {
                            // found it
                            Overload_solver::Definition_list list(alloc);
                            list.push_back(cdef);

                            Allocator_builder builder(alloc);

                            return builder.create<Overload_result_set>(
                                this,
                                m_compiler,
                                list);
                        } else {
                            // found it, but is not exported
                            break;
                        }
                    }
                }
            }
        }
    }

    // we have a symbol, look into the definition table
    Scope const *search_scope =
        is_builtins() ? m_def_tab.get_predef_scope() : m_def_tab.get_global_scope();
    def = search_scope->find_definition_in_scope(sym);
    if (def == NULL)
        return NULL;

    if (def->get_kind() == IDefinition::DK_TYPE) {
        // a type, check for constructors
        if (Scope *type_scope = def->get_own_scope()) {
            // search for constructors
            def = type_scope->find_definition_in_scope(sym);
            if (def == NULL)
                return NULL;
        }
    }

    Definition::Kind kind = def->get_kind();
    if (kind != IDefinition::DK_FUNCTION && kind != IDefinition::DK_CONSTRUCTOR && kind != IDefinition::DK_OPERATOR) {
        // neither a function, nor a constructor, nor an operator
        return NULL;
    }

    // unfortunately overload resolution needs type binding which might create new types,
    // so it cannot operate on const modules; however the "effect" itself *is*
    // const.
    Overload_solver os(*const_cast<Module *>(this));

    size_t num_args = arg_types.size();
    Allocator_builder builder(alloc);

    return builder.create<Overload_result_set>(
        this,
        m_compiler,
        os.find_positional_overload(def, num_args > 0 ? &arg_types[0] : NULL, num_args)
    );
}

// Returns the mangled MDL name of a definition that is owned by the current module if one exists.
char const *Module::mangle_mdl_name(
    IDefinition const *def,
    IThread_context   *context) const
{
    if (Thread_context *ctx = impl_cast<Thread_context>(context)) {
        MDL_ASSERT(
            impl_cast<Definition>(def)->get_owner_module_id() == get_unique_id() &&
            "definition not owned by this module");

        string res(get_allocator());
        MDL_name_mangler mangler(get_allocator(), res);

        mangler.mangle(is_builtins() ? NULL : get_name(), def);

        return ctx->set_string_buffer(Thread_context::SBI_DAG_MANGLE_BUFFER, res.c_str());
    }
    return NULL;
}

// Returns the mangled DAG name of a definition that is owned by the current module if one exists.
char const *Module::mangle_dag_name(
    IDefinition const *def,
    IThread_context   *context) const
{
    if (Thread_context *ctx = impl_cast<Thread_context>(context)) {
        switch (def->get_kind()) {
        case IDefinition::DK_ERROR:
        case IDefinition::DK_CONSTANT:
        case IDefinition::DK_ENUM_VALUE:
        case IDefinition::DK_TYPE:
        case IDefinition::DK_VARIABLE:
        case IDefinition::DK_MEMBER:
        case IDefinition::DK_PARAMETER:
        case IDefinition::DK_ARRAY_SIZE:
        case IDefinition::DK_NAMESPACE:
            // no mangled name
            return NULL;
        case IDefinition::DK_ANNOTATION:
        case IDefinition::DK_FUNCTION:
        case IDefinition::DK_CONSTRUCTOR:
        case IDefinition::DK_OPERATOR:
            // mangle it
            break;
        }

        MDL_ASSERT(
            impl_cast<Definition>(def)->get_owner_module_id() == get_unique_id() &&
            "definition not owned by this module");

        DAG_mangler mangler(get_allocator(), m_compiler);

        string mangled_name(mangler.mangle(def, is_builtins() ? NULL : get_name()));
        return ctx->set_string_buffer(Thread_context::SBI_DAG_MANGLE_BUFFER, mangled_name.c_str());
    }
    return NULL;
}

// Helper function to parse definition and the parameter types from a signature.
Definition const *Module::parse_annotation_params(
    char const                  *anno_name,
    char const * const          param_type_names[],
    int                         num_param_type_names,
    vector<IType const *>::Type &arg_types) const
{
    if (!is_builtins()) {
        // func_name must be an absolute name, so check the module name first
        size_t l = strlen(m_absname);
        if (strncmp(anno_name, m_absname, l) != 0)
            return NULL;
        if (anno_name[l] != ':' || anno_name[l + 1] != ':')
            return NULL;
        anno_name += l + 2;
    } else {
        if (anno_name[0] != ':' || anno_name[1] != ':')
            return NULL;
        anno_name += 2;
    }

    // the rest must be an identifier, we do not allow to access an imported entity in
    // a namespace
    ISymbol const *sym = m_sym_tab.lookup_symbol(anno_name);
    if (sym == NULL)
        return NULL;

    // we have a symbol, look into the definition table
    Scope const *search_scope =
        is_builtins() ? m_def_tab.get_predef_scope() : m_def_tab.get_global_scope();
    Definition const *def = search_scope->find_definition_in_scope(sym);
    if (def == NULL)
        return NULL;

    Definition::Kind kind = def->get_kind();
    if (kind != IDefinition::DK_ANNOTATION) {
        // not an annotation
        return NULL;
    }

    if (num_param_type_names > 0) {
        if (!parse_parameter_signature(this, param_type_names, num_param_type_names, arg_types))
            return NULL;
    }
    return def;
}

// Lookup an exact annotation definition given by its name and an array
// of all (positional) parameter types.
IDefinition const *Module::find_annotation(
    char const         *anno_name,
    char const * const param_type_names[],
    size_t             num_param_type_names) const
{
    // create a type list from the parameter signature
    IAllocator *alloc = get_allocator();
    vector<IType const *>::Type arg_types(alloc);

    Definition const *def
        = parse_annotation_params(anno_name, param_type_names, num_param_type_names, arg_types);

    if (def == NULL) {
        // name not found or signature parsing error
        return NULL;
    }

    // unfortunately overload resolution needs type binding which might create new types,
    // so it cannot operate on const modules; however the "effect" itself *is*
    // const.
    Overload_solver os(*const_cast<Module *>(this));

    size_t num_args = arg_types.size();
    Allocator_builder builder(alloc);

    Overload_solver::Definition_list l = os.find_positional_overload(
        def, num_args > 0 ? &arg_types[0] : NULL, num_args);

    if (l.empty())
        return NULL;

    def = l.front();
    l.pop_front();

    if (!l.empty())
        return NULL;

    IType_function const *ft = cast<IType_function>(def->get_type());
    int n = ft->get_parameter_count();

    if (size_t(n) != num_args) {
        // not exact match
        return NULL;
    }

    for (int i = 0; i < n; ++i) {
        ISymbol const *p_sym;
        IType const   *p_type;

        ft->get_parameter(i, p_type, p_sym);

        p_type = p_type->skip_type_alias();

        if (p_type != arg_types[i]) {
            // not exact match
            return NULL;
        }
    }
    return def;
}

// Lookup an annotation definition given by its name and an array
// of (positional) parameter types.
IOverload_result_set const *Module::find_annotation_by_signature(
    char const         *anno_name,
    char const * const param_type_names[],
    size_t             num_param_type_names) const
{
    // create a type list from the parameter signature
    IAllocator *alloc = get_allocator();
    vector<IType const *>::Type arg_types(alloc);

    Definition const *def
        = parse_annotation_params(anno_name, param_type_names, num_param_type_names, arg_types);

    if (def == NULL) {
        // name not found or signature parsing error
        return NULL;
    }

    // unfortunately overload resolution needs type binding which might create new types,
    // so it cannot operate on const modules; however the "effect" itself *is*
    // const.
    Overload_solver os(*const_cast<Module *>(this));

    size_t num_args = arg_types.size();
    Allocator_builder builder(alloc);

    return builder.create<Overload_result_set>(
        this,
        m_compiler,
        os.find_positional_overload(def, num_args > 0 ? &arg_types[0] : NULL, num_args)
    );
}

// Get the module declaration of this module if any.
IDeclaration_module const *Module::get_module_declaration() const
{
    for (size_t i = 0, n = m_declarations.size(); i < n; ++i) {
        IDeclaration const *decl = m_declarations[i];

        if (decl->get_kind() == IDeclaration::DK_MODULE)
            return cast<IDeclaration_module>(decl);
    }
    return NULL;
}

// Get the semantic version of this module if one was set.
ISemantic_version const *Module::get_semantic_version() const
{
    return m_sema_version;
}

// Get the number of referenced resources in this module.
size_t Module::get_referenced_resources_count() const
{
    return m_res_table.size();
}

// Get the absolute URL of the i'th referenced resource.
char const *Module::get_referenced_resource_url(size_t i) const
{
    if (i < m_res_table.size()) {
        return m_res_table[i].m_url;
    }
    return NULL;
}

// Get the type of the i'th referenced resource.
IType_resource const *Module::get_referenced_resource_type(size_t i) const

{
    if (i < m_res_table.size()) {
        return m_res_table[i].m_type;
    }
    return NULL;
}

// Get the exists flag of the i'th referenced resource.
bool Module::get_referenced_resource_exists(size_t i) const
{
    if (i < m_res_table.size()) {
        return m_res_table[i].m_exists;
    }
    return false;
}

// Get the OS-specific filename of the i'th referenced resource if any.
char const *Module::get_referenced_resource_file_name(size_t i) const
{
    if (i < m_res_table.size()) {
        return m_res_table[i].m_filename;
    }
    return NULL;
}

// Returns true if this module supports function hashes.
bool Module::has_function_hashes() const
{
    return m_is_hashed;
}

// Get the function hash for a given function definition if any.
Module::Function_hash const *Module::get_function_hash(IDefinition const *def) const
{
    Func_hash_map::const_iterator it = m_func_hashes.find(def);
    if (it != m_func_hashes.end())
        return &it->second;
    return NULL;
}

// Access messages.
Messages_impl &Module::access_messages_impl()
{
    return m_msg_list;
}

// Access messages.
Messages_impl const &Module::access_messages_impl() const
{
    return m_msg_list;
}

// Get the module symbol for a foreign symbol if it exists.
ISymbol const *Module::lookup_symbol(ISymbol const *sym) const
{
    return m_sym_tab.find_equal_symbol(sym);
}

// Import a symbol from another module.
ISymbol const *Module::import_symbol(ISymbol const *sym)
{
    // if it's not a predefined ID, simply create a copy, no need for a proxy here.
    // Note that this might change the ID, so NEVER compare symbols
    // from different modules.
    Symbol::Predefined_id id = Symbol::Predefined_id(sym->get_id());
    if (id == Symbol::SYM_OPERATOR) {
        // operator symbols are global
        return sym;
    }
    ISymbol const *psym = m_sym_tab.get_predefined_symbol(id);
    if (psym != NULL)
        return psym;
    return m_sym_tab.get_symbol(sym->get_name());
}

// Import a type from another module.
IType const *Module::import_type(IType const *type)
{
    // For imported types we could either create proxies, OR
    // copy them ... import() chooses the second variant.
    return m_type_factory.import(type);
}

// Import a value from another module.
IValue const *Module::import_value(IValue const *value) const
{
    return m_value_factory.import(value);
}

// Import a position from another module.
Position *Module::import_position(Position const *pos)
{
    if (pos != NULL) {
        struct Position_allocator {
            Position_allocator(Memory_arena &arena) : m_builder(arena) {}

            Position_impl *alloc_position(Position const *pos) {
                return m_builder.create<Position_impl>(pos);
            }

            Arena_builder m_builder;
        };
        Position_allocator allocator(m_arena);

        // Simply create a copy on the arena of the module, no need for proxy here.
        return allocator.alloc_position(pos);
    }
    return NULL;
}

// Register a module to be imported into this module.
size_t Module::register_import(
    Module const *imp_mod,
    bool         *first)
{
    bool dummy;
    if (first == NULL)
        first = &dummy;
    *first = false;

    size_t idx = get_import_index(imp_mod);
    if (idx == 0) {
        // first import from this module, put it into the import table

        // get and internalize the file name and the absolute name of the imported module
        char *f_name   = Arena_strdup(m_arena, imp_mod->m_filename);
        char *abs_name = Arena_strdup(m_arena, imp_mod->m_absname);

        size_t mod_id = imp_mod->get_unique_id();
        m_imported_modules.push_back(Import_entry(mod_id, imp_mod, f_name, abs_name));
        idx = m_imported_modules.size();
        *first = true;
    }

    // increase the ref-count of this module here, because the analysis will decrease it by one
    // at its end
    m_imported_modules[idx - 1].lock_module(m_compiler->get_weak_module_lock());

    return idx;
}

// Register a module to be imported into this module lazy.
size_t Module::register_import(char const *abs_name, char const *fname, bool is_stdlib)
{
    size_t idx = get_import_index(abs_name);
    if (idx > 0) {
        // already known
        return idx;
    }
    // first import from this module, put it into the import table

    // get and internalize the file name and the absolute name of the imported module
    fname    = Arena_strdup(m_arena, fname);
    abs_name = Arena_strdup(m_arena, abs_name);

    m_imported_modules.push_back(Import_entry(fname, abs_name, is_stdlib));
    return m_imported_modules.size();
}

// Register an exported entity.
void Module::register_exported(Definition const *def)
{
    m_exported_definitions.push_back(def);
}

// Allocate initializers for a function definition.
void Module::allocate_initializers(Definition *def, size_t num)
{
    MDL_ASSERT(num > 0);
    Definition::Initializers *init = reinterpret_cast<Definition::Initializers *>(
        m_arena.allocate(
            sizeof(Definition::Initializers) + (num - 1) *
            sizeof(init->exprs[0])));
    std::fill_n(&init->exprs[0], num, (IExpression *)0);

    init->count            = num;
    def->m_parameter_inits = init;
}

// Clone the given parameter.
IParameter const *Module::clone_param(
    IParameter const *param,
    bool             clone_init,
    bool             clone_anno,
    IClone_modifier  *modifier)
{
    IType_name *tn = clone_name(param->get_type_name(), modifier);
    ISimple_name const *sn = clone_name(param->get_name());

    IExpression *new_init = NULL;
    if (clone_init)
        if (IExpression const *init_expr = param->get_init_expr()) {
            new_init = clone_expr(init_expr, modifier);
        }
    IAnnotation_block *new_annos = NULL;
    if (clone_anno)
        new_annos = clone_annotation_block(param->get_annotations(), modifier);

    return m_decl_factory.create_parameter(tn, sn, new_init, new_annos);
}

// Clone the given variable declaration.
IDeclaration *Module::clone_decl(
    IDeclaration_variable const *decl,
    IClone_modifier             *modifier)
{
    IType_name const *tn = clone_name(decl->get_type_name(), modifier);
    IDeclaration_variable *new_decl = m_decl_factory.create_variable(tn);

    for (int i = 0, n = decl->get_variable_count(); i < n; ++i) {
        IExpression *new_init = NULL;
        if (IExpression const *init_expr = decl->get_variable_init(i)) {
            new_init = clone_expr(init_expr, modifier);
        }
        ISimple_name const *new_name  = clone_name(decl->get_variable_name(i));
        IAnnotation_block  *new_annos = clone_annotation_block(decl->get_annotations(i), modifier);
        new_decl->add_variable(new_name, new_init, new_annos);
    }
    return new_decl;
}

// Clone the given annotation block.
IAnnotation_block *Module::clone_annotation_block(
    IAnnotation_block const *anno_block,
    IClone_modifier         *modifier)
{
    if (!anno_block) return NULL;

    IAnnotation_block *new_block = m_anno_factory.create_annotation_block();
    for (int i = 0, n = anno_block->get_annotation_count(); i < n; ++i) {

        IAnnotation const *anno = anno_block->get_annotation(i);
        IDefinition const *def = anno->get_name()->get_definition();

        if (def->get_semantics() == IDefinition::DS_VERSION_NUMBER_ANNOTATION &&
            m_mdl_version >= IMDL::MDL_VERSION_1_3) {
                continue; // skip
        }
        IAnnotation *new_anno = clone_annotation(anno_block->get_annotation(i), modifier);
        new_block->add_annotation(new_anno);
    }
    return new_block;
}

// Clone the given annotation.
IAnnotation *Module::clone_annotation(
    IAnnotation const *anno,
    IClone_modifier   *modifier)
{
    IQualified_name const *new_name = clone_name(anno->get_name(), modifier);
    IAnnotation *new_anno = m_anno_factory.create_annotation(new_name);
    for (int i = 0, n = anno->get_argument_count(); i < n; ++i) {
        IArgument const *arg = anno->get_argument(i);
        IArgument const *new_arg = clone_arg(arg, modifier);
        new_anno->add_argument(new_arg);
    }
    return new_anno;
}

// Clone the given expression.
IExpression *Module::clone_expr(
    IExpression const *expr,
    IClone_modifier   *modifier)
{
    IExpression *res = NULL;

    // import the type first, so importing values will not fail
    IType const *old_type = expr->get_type();
    IType const *new_type = old_type != NULL ? import_type(old_type) : NULL;

    switch (expr->get_kind()) {
    case IExpression::EK_INVALID:
        res = m_expr_factory.create_invalid();
        break;
    case IExpression::EK_LITERAL:
        // handled later
        res = const_cast<IExpression *>(expr);
        break;
    case IExpression::EK_REFERENCE:
        {
            IExpression_reference const *r_expr = cast<IExpression_reference>(expr);
            if (modifier != NULL) {
                res = modifier->clone_expr_reference(r_expr);
            } else {
                IType_name const *name = clone_name(r_expr->get_name(), modifier);
                IExpression_reference *ref = m_expr_factory.create_reference(name);
                if (r_expr->is_array_constructor()) {
                    ref->set_array_constructor();
                } else {
                    ref->set_definition(r_expr->get_definition());
                }
                res = ref;
            }
            break;
        }
    case IExpression::EK_UNARY:
        {
            IExpression_unary const *unexpr = cast<IExpression_unary>(expr);
            IExpression const *arg = clone_expr(unexpr->get_argument(), modifier);
            IExpression_unary *res_unexpr = m_expr_factory.create_unary(unexpr->get_operator(), arg);
            if (res_unexpr->get_operator() == IExpression_unary::OK_CAST) {
                res_unexpr->set_type_name(clone_name(unexpr->get_type_name(), modifier));
            }
            res = res_unexpr;
            break;
        }
    case IExpression::EK_BINARY:
        {
            IExpression_binary const *binexpr = cast<IExpression_binary>(expr);
            IExpression const *lhs = clone_expr(binexpr->get_left_argument(), modifier);
            IExpression const *rhs = clone_expr(binexpr->get_right_argument(), modifier);
            res = m_expr_factory.create_binary(binexpr->get_operator(), lhs, rhs);
            break;
        }
    case IExpression::EK_CONDITIONAL:
        {
            IExpression_conditional const *c_expr = cast<IExpression_conditional>(expr);
            IExpression const *cond = clone_expr(c_expr->get_condition(), modifier);
            IExpression const *t_ex = clone_expr(c_expr->get_true(), modifier);
            IExpression const *f_ex = clone_expr(c_expr->get_false(), modifier);
            res = m_expr_factory.create_conditional(cond, t_ex, f_ex);
            break;
        }
    case IExpression::EK_CALL:
        {
            IExpression_call const *c_expr = cast<IExpression_call>(expr);
            if (modifier != NULL) {
                res = modifier->clone_expr_call(c_expr);
            } else {
                IExpression const *ref = clone_expr(c_expr->get_reference(), modifier);
                IExpression_call *call = m_expr_factory.create_call(ref);

                for (int i = 0, n = c_expr->get_argument_count(); i < n; ++i) {
                    IArgument const *arg = clone_arg(c_expr->get_argument(i), modifier);
                    call->add_argument(arg);
                }
                res = call;
            }
            break;
        }
    case IExpression::EK_LET:
        {
            IExpression_let const *l_expr = cast<IExpression_let>(expr);
            IExpression const     *exp = clone_expr(l_expr->get_expression(), modifier);
            IExpression_let       *let = m_expr_factory.create_let(exp);

            for (int i = 0, n = l_expr->get_declaration_count(); i < n; ++i) {
                IDeclaration const *decl =
                    clone_decl(cast<IDeclaration_variable>(l_expr->get_declaration(i)), modifier);
                let->add_declaration(decl);
            }
            res = let;
            break;
        }
    }
    MDL_ASSERT(res != NULL && "Unsupported expression kind");

    IExpression::Kind kind = res->get_kind();
    if (kind == IExpression::EK_LITERAL) {
        // already a literal, import it
        IExpression_literal const *lit = cast<IExpression_literal>(res);
        if (modifier != NULL) {
            res = modifier->clone_literal(lit);
        } else {
            IValue const *old_value = lit->get_value();
            IValue const *new_value = import_value(old_value);
            return create_literal(new_value, &lit->access_position());
        }
    } else {
        // no literal
        res->set_type(new_type);
    }
    return res;
}

// Clone the given argument.
IArgument const *Module::clone_arg(
    IArgument const *arg,
    IClone_modifier *modifier)
{
    switch (arg->get_kind()) {
    case IArgument::AK_POSITIONAL:
        {
            IArgument_positional const *pos = cast<IArgument_positional>(arg);
            IExpression const *expr = clone_expr(pos->get_argument_expr(), modifier);
            return m_expr_factory.create_positional_argument(expr);
        }
    case IArgument::AK_NAMED:
        {
            IArgument_named const *named = cast<IArgument_named>(arg);
            ISimple_name const *sname = clone_name(named->get_parameter_name());
            IExpression const *expr = clone_expr(named->get_argument_expr(), modifier);

            return m_expr_factory.create_named_argument(sname, expr);
        }
    }
    return NULL;
}

// Clone the given type name.
IType_name *Module::clone_name(
    IType_name const *type_name,
    IClone_modifier  *modifier)
{
    IQualified_name *qname = clone_name(type_name->get_qualified_name(), modifier);
    IType_name      *res   = m_name_factory.create_type_name(qname);

    if (type_name->is_absolute()) {
        res->set_absolute();
    }

    res->set_qualifier(type_name->get_qualifier());

    if (type_name->is_incomplete_array()) {
        res->set_incomplete_array();
    }

    if (IExpression const *array_size = type_name->get_array_size()) {
        array_size = clone_expr(array_size, modifier);
        res->set_array_size(array_size);
    }

    if (ISimple_name const *sname = type_name->get_size_name()) {
        res->set_size_name(clone_name(sname));
    }
    return res;
}

// Clone the given simple name.
ISimple_name const *Module::clone_name(ISimple_name const *sname)
{
    ISymbol const *sym = sname->get_symbol();

    // symbol can come from another symbol table, so check it
    if (m_sym_tab.get_symbol_for_id(sym->get_id()) != sym) {
        sym = import_symbol(sym);
    }

    ISimple_name *res = const_cast<ISimple_name*>(m_name_factory.create_simple_name(sym));
    res->set_definition(sname->get_definition());
    return res;
}

// Clone the given qualified name.
IQualified_name *Module::clone_name(
    IQualified_name const *qname,
    IClone_modifier       *modifier)
{
    if (modifier != NULL) {
        return modifier->clone_name(qname);
    }

    IQualified_name *res = m_name_factory.create_qualified_name();

    for (int i = 0, n = qname->get_component_count(); i < n; ++i) {
        ISimple_name const *sname = clone_name(qname->get_component(i));
        res->add_component(sname);
    }
    res->set_definition(qname->get_definition());
    return res;
}

// Get the import entry for a given import index.
Module::Import_entry const *Module::get_import_entry(size_t idx) const
{
    if (idx > 0 && idx <= m_imported_modules.size()) {
        return &m_imported_modules[idx - 1];
    }
    return NULL;
}

// Get the import index for a given (already imported) module.
size_t Module::get_import_index(Module const *mod) const
{
    // PERFORMANCE: the linear search is not expected to be a bottleneck in the compiler itself.
    for (size_t i = 0, n = m_imported_modules.size(); i < n; ++i) {
        Import_entry const &entry = m_imported_modules[i];

        if (entry.get_module() == mod) {
            return i + 1;
        }
    }
    return 0;
}

// Get the import index for a given (already imported) module.
size_t Module::get_import_index(char const *abs_name) const
{
    // PERFORMANCE: the linear search is not expected to be a bottleneck in the compiler itself.
    for (size_t i = 0, n = m_imported_modules.size(); i < n; ++i) {
        Import_entry const &entry = m_imported_modules[i];

        if (strcmp(entry.get_absolute_name(), abs_name) == 0) {
            return i + 1;
        }
    }
    return 0;
}

// Get the unique id of the original owner module of a definition.
size_t Module::get_original_owner_id(Definition const *def) const
{
    size_t import_idx = def->get_original_import_idx();
    if (import_idx == 0) {
        return def->get_owner_module_id();
    }

    if (get_unique_id() == def->get_owner_module_id()) {
        // typical case: from out module
        Module::Import_entry const *entry = get_import_entry(import_idx);
        return entry->get_id();
    }

    // bad, this module is not the owner of def, try to find its owner in our import table
    Module::Import_entry const *from_entry = find_import_entry(def->get_owner_module_id());

    Module::Import_entry const *entry = from_entry->get_module()->get_import_entry(import_idx);
    return entry->get_id();
}

// Find the import entry for a given module ID.
Module::Import_entry const *Module::find_import_entry(size_t module_id) const
{
    // check if we already know it.
    // PERFORMANCE: the linear search is not expected to be a bottleneck in the compiler itself,
    // but maybe in the DAG backend.
    for (size_t i = 0, n = m_imported_modules.size(); i < n; ++i) {
        Import_entry const &entry = m_imported_modules[i];
        if (entry.get_id() == module_id) {
            return &entry;
        }
    }
    return NULL;
}

// Find the import entry for a given absolute module name.
Module::Import_entry const *Module::find_import_entry(char const *absname) const
{
    // check if we already know it.
    // PERFORMANCE: the linear search is not expected to be a bottleneck in the compiler itself,
    // but maybe in the DAG backend.
    for (size_t i = 0, n = m_imported_modules.size(); i < n; ++i) {
        Import_entry const &entry = m_imported_modules[i];
        if (strcmp(entry.get_absolute_name(), absname) == 0) {
            return &entry;
        }
    }
    return NULL;
}

// Creates a qualified name from a C-string.
IQualified_name *Module::qname_from_cstring(char const *name)
{
    IQualified_name *qualified_name = m_name_factory.create_qualified_name();

    if (name[0] == ':' && name[1] == ':') {
        qualified_name->set_absolute();
        name = name + 2;
    }

    IAllocator *alloc = get_allocator();
    while (name[0] != '\0') {
        ISymbol const *symbol = NULL;
        if (char const *pos = strstr(name, "::")) {
            string component(name, pos - name, alloc);
            symbol = m_name_factory.create_symbol(component.c_str());
            name = pos + 2;
        } else {
            symbol = m_name_factory.create_symbol(name);
            name = "";
        }
        ISimple_name const *simple_name = m_name_factory.create_simple_name(symbol);
        qualified_name->add_component(simple_name);
    }
    return qualified_name;
}

// Return the filename of an imported module.
char const *Module::get_import_fname(size_t mod_id) const
{
    Import_entry const *entry = find_import_entry(mod_id);
    if (entry != NULL) {
        return entry->get_file_name();
    }
    return NULL;
}

// Return the value create by a const default constructor of a type.
IValue const *Module::create_default_value(
    IValue_factory *factory,
    IType const    *type) const
{
    IDefinition::Semantics constr_sema = IDefinition::DS_ELEM_CONSTRUCTOR;

restart:
    switch (type->get_kind()) {
    case IType::TK_INCOMPLETE:
    case IType::TK_ERROR:
    case IType::TK_FUNCTION:
        return factory->create_bad();

    case IType::TK_ALIAS:
        type = type->skip_type_alias();
        goto restart;

    case IType::TK_ARRAY:
        // special case for array types
        {
            IType_array const *a_type = cast<IType_array>(type);
            if (! a_type->is_immediate_sized()) {
                MDL_ASSERT(!"cannot create default value for abstract array");
                return factory->create_bad();
            }
            IValue const *e_val = create_default_value(
                get_value_factory(), a_type->get_element_type());

            size_t a_size = a_type->get_size();
            VLA<IValue const *> values(get_allocator(), a_size);

            for (size_t i = 0; i < a_size; ++i) {
                values[i] = e_val;
            }
            return factory->create_array(a_type, values.data(), a_size);
        }

    case IType::TK_LIGHT_PROFILE:
    case IType::TK_TEXTURE:
    case IType::TK_BSDF_MEASUREMENT:
    case IType::TK_BSDF:
    case IType::TK_HAIR_BSDF:
    case IType::TK_EDF:
    case IType::TK_VDF:
        // create an invalid reference for those
        return factory->create_invalid_ref(as<IType_reference>(type));

    case IType::TK_BOOL:
    case IType::TK_INT:
    case IType::TK_ENUM:
    case IType::TK_FLOAT:
    case IType::TK_DOUBLE:
    case IType::TK_STRING:
        // those types have an copy constructor with a default argument
        // that is used as default constructor
        constr_sema = IDefinition::DS_COPY_CONSTRUCTOR;
        break;

    case IType::TK_COLOR:
        // the color type has a conversion operator with a default argument
        constr_sema = IDefinition::DS_CONV_CONSTRUCTOR;
        break;

    case IType::TK_VECTOR:
    case IType::TK_MATRIX:
    case IType::TK_STRUCT:
        // those have an elementary constructor
        constr_sema = IDefinition::DS_ELEM_CONSTRUCTOR;
        break;
    }

    // first step: find the elementary constructor
    Definition const               *c_def = NULL;
    mi::base::Handle<Module const> c_mod  = mi::base::make_handle_dup(this);

    for (c_def = get_first_constructor(type); c_def != NULL; c_def = get_next_constructor(c_def)) {
        if (c_def->get_semantics() == constr_sema) {
            // resolve for imports, struct constructors may be imported
            Definition const *orig = get_original_definition(c_def);
            if (constr_sema != IDefinition::DS_CONV_CONSTRUCTOR ||
                orig->get_default_param_initializer(0) != NULL)
            {
                // found
                c_mod = mi::base::make_handle(get_owner_module(c_def));
                c_def = orig;
                break;
            }
        }
    }
    if (c_def == NULL) {
        MDL_ASSERT(!"Module::create_default_value(): could not find default constructor");
        return factory->create_bad();
    }

    IType_function const *func_type = cast<IType_function>(c_def->get_type());

    switch (type->get_kind()) {
    case IType::TK_ALIAS:
    case IType::TK_INCOMPLETE:
    case IType::TK_ERROR:
    case IType::TK_ARRAY:
    case IType::TK_LIGHT_PROFILE:
    case IType::TK_TEXTURE:
    case IType::TK_BSDF_MEASUREMENT:
    case IType::TK_BSDF:
    case IType::TK_HAIR_BSDF:
    case IType::TK_EDF:
    case IType::TK_VDF:
        // already handled
        break;
    case IType::TK_FUNCTION:
        // should not happen
        MDL_ASSERT(!"cannot create a default value for a function type");
        break;
    case IType::TK_BOOL:
    case IType::TK_INT:
    case IType::TK_ENUM:
    case IType::TK_FLOAT:
    case IType::TK_DOUBLE:
    case IType::TK_STRING:
        // atomic types: constructor has one argument
        MDL_ASSERT(func_type->get_parameter_count() == 1 && "wrong atomic type constructor");
        return import_value(
            c_def->get_default_param_initializer(0)->
            fold(c_mod.get(), c_mod->get_value_factory(), NULL));

    case IType::TK_COLOR:
        {
            IExpression const *expr = c_def->get_default_param_initializer(0);
            IValue const      *pval = import_value(
                expr->fold(c_mod.get(), c_mod->get_value_factory(), NULL));
            if (is<IValue_bad>(pval))
                return pval;

            IValue const *values[3] = { pval, pval, pval };
            return factory->create_compound(cast<IType_compound>(type), values, 3);
        }
    case IType::TK_VECTOR:
    case IType::TK_MATRIX:
    case IType::TK_STRUCT:
        {
            // compound types
            size_t param_count = func_type->get_parameter_count();

            VLA<IValue const *> values(get_allocator(), param_count);

            for (size_t i = 0; i < param_count; ++i) {
                IValue const *pval;
                if (IExpression const *expr = c_def->get_default_param_initializer(i)) {
                    // this compound element has an initializer, fold it
                    pval = import_value(expr->fold(c_mod.get(), factory, /*handler=*/NULL));
                } else {
                    // this compound element is default initialized
                    IType const   *ptype;
                    ISymbol const *psym;

                    func_type->get_parameter(i, ptype, psym);

                    pval = create_default_value(factory, ptype);
                }

                if (is<IValue_bad>(pval))
                    return pval;
                
                values[i] = pval;
            }
            return factory->create_compound(cast<IType_compound>(type), values.data(), param_count);
        }
    }
    return factory->create_bad();
}

// Replace the global declarations by new collection.
void Module::replace_declarations(IDeclaration const * const decls[], size_t len)
{
    m_declarations.resize(len);
    for (size_t i = 0; i < len; ++i)
        m_declarations[i] = decls[i];
}

// Get the imported definition for an outside definition if it is imported.
Definition const *Module::find_imported_definition(Definition const *def) const
{
    if (def->get_owner_module_id() == m_unique_id) {
        // this definition is owned by this module (i.e. either created
        // by this module or already imported)
        return def;
    }

    ISymbol const *imp_sym = m_sym_tab.find_equal_symbol(def->get_sym());
    if (imp_sym == NULL) {
        // this definition is not imported, otherwise its symbol would be
        return NULL;
    }

    size_t original_id = get_original_owner_id(def);
    Import_entry const *entry = find_import_entry(original_id);
    if (entry == NULL) {
        // the owner module is not imported at all
        return NULL;
    }
    size_t import_id = get_import_index(entry->get_module());
    if (import_id == 0) {
        // not in our import table
        return NULL;
    }

    Scope *scope = NULL;

    if (def->get_kind() == Definition::DK_CONSTRUCTOR) {
        // a constructor: search in the type scope
        IType_function const *fct_type = as<IType_function>(def->get_type());
        if (fct_type == NULL)
            return NULL;

        IType const *ret_type = fct_type->get_return_type();
        ret_type = m_type_factory.get_equal(ret_type);
        if (ret_type == NULL)
            return NULL;

        scope = m_def_tab.get_type_scope(ret_type);
        if (scope == NULL)
            return NULL;
    } else if (def->has_flag(Definition::DEF_IS_PREDEFINED)) {
        // a predefined entity, search for it in predefined scope
        scope = m_def_tab.get_predef_scope();
    }

    if (scope != NULL) {
        // we found a matching scope, check if it contains the definition
        Definition const *imp = scope->find_definition_in_scope(imp_sym);

        // check if its ours
        for (; imp != NULL; imp = imp->get_prev_def()) {
            if (imp->has_flag(Definition::DEF_IS_PREDEFINED)) {
                // predefined are never imported, hence do not check for the module id
                if (imp->get_original_unique_id() == def->get_original_unique_id()) {
                    // found it
                    return imp;
                }
            } else if (imp->has_flag(Definition::DEF_IS_IMPORTED)) {
                // imported once must have equal ID's
                if (imp->get_original_unique_id() == def->get_original_unique_id() &&
                    imp->get_original_import_idx() == import_id) {
                    // found it
                    return imp;
                }
            }
        }
        return NULL;
    }

    // the module itself is imported, try to find the imported definition
    // start with the full qualified name and search the way back
    Module const *imp_mod = entry->get_module();

    IQualified_name const *imp_qname = imp_mod->get_qualified_name();
    int n_comp = imp_qname->get_component_count();

    for (int i = 0; i <= n_comp; ++i) {
        Scope *scope = m_def_tab.get_global_scope();

        for (int j = i; j < n_comp; ++j) {
            ISymbol const *s     = imp_qname->get_component(j)->get_symbol();
            ISymbol const *imp_s = m_sym_tab.find_equal_symbol(s);

            if (imp_s == NULL) {
                // not known
                break;
            }
            scope = scope->find_named_subscope(imp_s);
            if (scope == NULL) {
                // not known
                break;
            }
        }

        if (scope != NULL) {
            // we found a matching scope, check if it contains the definition
            Definition const *imp = scope->find_definition_in_scope(imp_sym);

            // check if its our
            for (; imp != NULL; imp = imp->get_prev_def()) {
                if (!imp->has_flag(Definition::DEF_IS_IMPORTED))
                    continue;

                if (imp->get_original_unique_id() == def->get_original_unique_id() &&
                    imp->get_original_import_idx() == import_id) {
                    // found it
                    return imp;
                }
            }
        }
    }
    // not found
    return NULL;
}

// Get the imported definition of a type.
Definition const *Module::find_imported_type_definition(IType const *type) const
{
    type = type->skip_type_alias();
    if (Scope const *scope = m_def_tab.get_type_scope(type)) {
        // inside this module
        return scope->get_owner_definition();
    }

    // check if we already know it.
    // PERFORMANCE: the linear search is not expected to be a bottleneck in the compiler itself,
    // but maybe in the DAG backend.
    for (size_t i = 0, n = m_imported_modules.size(); i < n; ++i) {
        Import_entry const &entry = m_imported_modules[i];
        Module const *tp_owner = entry.get_module();

        if (Scope const *scope = tp_owner->m_def_tab.get_type_scope(type)) {
            // inside this module
            return scope->get_owner_definition();
        }
    }

    // not found
    return NULL;
}

// Get the module for a given module id.
Module const *Module::do_find_imported_module(size_t id, bool &direct) const
{
    if (Import_entry const *entry = find_import_entry(id)) {
        direct = true;
        return entry->get_module();
    }

    for (size_t i = 0, n = m_imported_modules.size(); i < n; ++i) {
        Import_entry const &entry = m_imported_modules[i];

        Module const *imp_mod = entry.get_module()->find_imported_module(id, direct);
        if (imp_mod != NULL) {
            direct = false;
            return imp_mod;
        }
    }
    return NULL;
}

// Get the module for a given module name.
Module const *Module::do_find_imported_module(char const *absname, bool &direct) const
{
    if (Import_entry const *entry = find_import_entry(absname)) {
        direct = true;
        return entry->get_module();
    }

    for (size_t i = 0, n = m_imported_modules.size(); i < n; ++i) {
        Import_entry const &entry = m_imported_modules[i];

        Module const *imp_mod = entry.get_module()->find_imported_module(absname, direct);
        if (imp_mod != NULL) {
            direct = false;
            return imp_mod;
        }
    }
    return NULL;
}


// Get the module for a given module id.
Module const *Module::find_imported_module(size_t id, bool &direct) const
{
    if (Module const *res = do_find_imported_module(id, direct)) {
        return res;
    }
    direct = false;
    return m_compiler->find_builtin_module(id);
}

// Get the module for a given module name.
Module const *Module::find_imported_module(char const *absname, bool &direct) const
{
    if (Module const *res = do_find_imported_module(absname, direct)) {
        return res;
    }
    direct = false;
    return m_compiler->find_builtin_module(string(absname, get_allocator()));
}

// Add an auto-import.
void Module::add_auto_import(IDeclaration_import const *import)
{
    Declaration_vector::iterator it(m_declarations.begin()), end(m_declarations.end());
    while (it != end && (*it)->get_kind() == IDeclaration::DK_IMPORT)
        ++it;
    m_declarations.insert(it, import);
}

// Find the definition of a of a standard library entity.
Definition const *Module::find_stdlib_symbol(
    char const    *mod_name,
    ISymbol const *sym) const
{
    Module const *stdlib_mod = NULL;

    if (is_stdlib() && strcmp(m_absname, mod_name) == 0) {
        // ensure the state module finds itself
        stdlib_mod = this;
    } else {
        // should be already registered
        stdlib_mod = m_compiler->find_builtin_module(string(mod_name, get_allocator()));
    }

    if (stdlib_mod == NULL) {
        // looking up the ::state module should never fail, others could
        MDL_ASSERT(strcmp("::state", mod_name) != 0 && "built in module state not found");
        return NULL;
    }
    Scope const *scope = stdlib_mod->m_def_tab.get_global_scope();

    Definition const *def = scope->find_definition_in_scope(sym);
    if (def != NULL) {
        if (def->has_flag(Definition::DEF_IS_EXPORTED))
            return def;
    }
    return NULL;
}

// Create a literal or a scoped literal from a value.
IExpression_literal *Module::create_literal(IValue const *value, Position const *pos)
{
    IExpression_literal *res = m_expr_factory.create_literal(value);

    if (pos != NULL) {
        Position &lpos = res->access_position();

        lpos.set_start_line(pos->get_start_line());
        lpos.set_start_column(pos->get_start_column());
        lpos.set_end_line(pos->get_end_line());
        lpos.set_end_column(pos->get_end_column());
        lpos.set_filename_id(pos->get_filename_id());
    }
    return res;
}

// Lookup the original definition on an imported one.
Definition const *Module::lookup_original_definition(Definition const *imported) const
{
    Definition const *orig = NULL;

    // we lookup an exported definition, so it must be at global scope or it is a constructor,
    // then it might be in a type scope.
    Scope const *scope = m_def_tab.get_global_scope();

    bool look_inside = false;
    Definition::Kind kind = imported->get_kind();
    if (kind == IDefinition::DK_OPERATOR) {
        // operators of struct and enum types are inside the type scope, operators for predefined
        // are out-side
        if (imported->get_semantics() == operator_to_semantic(IExpression_binary::OK_ASSIGN)) {
            if (IType_function const *op_tp = as<IType_function>(imported->get_type())) {
                IType const *ret_tp = op_tp->get_return_type();
                IType::Kind kind = ret_tp->get_kind();

                look_inside = kind == IType::TK_STRUCT || kind == IType::TK_ENUM;
            }
        }
    }
    if (look_inside || kind == IDefinition::DK_CONSTRUCTOR || kind == IDefinition::DK_MEMBER) {
        // Look at type scope: Unfortunately we cannot retrieve the type from the
        // definition, because it is owned by another module. We could "import" it back,
        // but this quick search is hopefully fast enough.
        for (Scope const *s = scope->get_first_subscope(); s != NULL; s = s->get_next_subscope()) {
            if (s->get_scope_type() == NULL) {
                // not a type scope
                continue;
            }
            orig = s->find_definition_in_scope(imported->get_original_unique_id());
            if (orig != NULL)
                break;
        }
    } else {
        // search only at global scope
        orig = scope->find_definition_in_scope(imported->get_original_unique_id());
    }

    MDL_ASSERT(orig != NULL);
    return orig;
}

// Insert a constant declaration after all other constant declarations.
void Module::insert_constant_declaration(IDeclaration_constant *decl)
{
    Declaration_vector::iterator it(m_declarations.begin()), end(m_declarations.end());

    for (; it != end; ++it) {
        IDeclaration const *decl = *it;

        switch (decl->get_kind()) {
        case IDeclaration::DK_IMPORT:
        case IDeclaration::DK_CONSTANT:
            // skip
            break;
        default:
            // stop here
            goto exit_loop;
        }
    }
exit_loop:
    m_declarations.insert(it, decl);
}


static IDefinition const *find_matching_def(
    Scope const       *search_scope,
    ISymbol const     *sym,
    Signature_matcher &matcher,
    bool              only_exported)
{
    Definition const *def = search_scope->find_definition_in_scope(sym);
    if (def == NULL)
        return NULL;

    if (def->get_kind() == IDefinition::DK_TYPE) {
        // a type, check for constructors
        if (Scope *type_scope = def->get_own_scope()) {
            // search for constructors
            def = type_scope->find_definition_in_scope(sym);
            if (def == NULL)
                return NULL;
        }
    }

    for (Definition const *cdef = def; cdef != NULL; cdef = cdef->get_prev_def()) {
        Definition::Kind kind = cdef->get_kind();
        if (kind != IDefinition::DK_FUNCTION && kind != IDefinition::DK_CONSTRUCTOR &&
            kind != IDefinition::DK_OPERATOR)
        {
            // neither a function nor a constructor nor an operator
            continue;
        }

        // check all overloads
        if (matcher.match_param(cdef)) {
            if (!only_exported || cdef->has_flag(Definition::DEF_IS_EXPORTED))
                return cdef;
            else {
                // found it, but is not exported
                break;
            }
        }
    }

    // not found
    return NULL;
}

static bool is_latin_alpha(char c) {
    return c >= 0 && c < 0x80 && isalpha(c);
}

static bool is_latin_alnum(char c) {
    return c >= 0 && c < 0x80 && isalnum(c);
}

static bool is_latin_digit(char c) {
    return c >= 0 && c < 0x80 && isdigit(c);
}

// Find the definition of a signature.
IDefinition const *Module::find_signature(
    char const *signature,
    bool       only_exported) const
{
    // skip module name if the signature starts with it
    size_t l = strlen(m_absname);
    if (strncmp(signature, m_absname, l) == 0 && signature[l] == ':' && signature[l + 1] == ':')
        signature = signature + l + 2;

    char const *start = NULL, *end = NULL;

    char const *p = start = end = signature;

    bool is_operator = false;

    // a valid entity name is LETTER(_|LETTER|DIGIT)+($DIGIT+.DIGIT+)?
    char c = *p;
    if (is_latin_alpha(c)) {
        do {
            ++p;
            c = *p;
        } while (c == '_' || is_latin_alnum(c));

        end = p;
        if (c == '$') {
            // a version suffix: $major.minor
            do {
                ++p;
                c = *p;
            } while (is_latin_digit(c));
            if (c == '.') {
                do {
                    ++p;
                    c = *p;
                } while (is_latin_digit(c));
            }
            // the version suffix is part of the signature, but not of the name
            p = end;
        }
    }
    if (end - start == 8 && strncmp(start, "operator", 8) == 0) {
        // might be an operator name
        is_operator = true;

        if (end[0] == '(' && end[1] == ')') {
            // operator()
            end += 2;
            p = end;
            c = *p;
        } else if (end[0] == '[' && end[1] == ']') {
            // operator[]
            end += 2;
            p = end;
            c = *p;
        } else {
            bool flag = true;
            for (p = end; flag;) {
                switch (*p) {
                case '~':
                case '!':
                case '.':
                case '*':
                case '/':
                case '%':
                case '+':
                case '-':
                case '<':
                case '>':
                case '=':
                case '^':
                case '&':
                case '|':
                case '?':
                    ++p;
                    break;
                default:
                    flag = false;
                    break;
                }
            }
            end = p;
            c = *p;
        }
    }
    if (c != '(' && c != '\0') {
        // wrong signature
        return NULL;
    }

    size_t len = end - start;

    if (len == 0) {
        // wrong signature
        return NULL;
    }

    string name(start, len, get_allocator());

    // filter out the deprecated suffix
    string mod_signature = name;
    mod_signature += p;

    Signature_matcher matcher(get_allocator(), m_compiler, mod_signature);

    ISymbol const *sym = NULL, *second_sym = NULL;

    if (is_operator) {
        // so far there are only definitions for unary and binary operators, do not try others
        ISymbol const *sym_unary  = m_sym_tab.lookup_operator_symbol(name.c_str(), 1);
        ISymbol const *sym_binary = m_sym_tab.lookup_operator_symbol(name.c_str(), 2);

        if (sym_binary == NULL) {
            sym = sym_unary;
        } else {
            sym = sym_binary;
            if (sym_unary != NULL) {
                second_sym = sym_unary;
            }
        }
    } else {
        sym = m_sym_tab.lookup_symbol(name.c_str());
    }

    if (sym == NULL) {
        // name does not exist in this module
        return NULL;
    }

    // check for conversion operator; currently only conversion to int is supported
    if (sym->get_id() == ISymbol::SYM_TYPE_INT && *p == '(') {
        Definition const *def = NULL;

        size_t l = strlen(p);
        if (p[l - 1] == ')') {

            bool error = false;
            string tname(p + 1, p + l - 1, get_allocator());

            size_t idx = tname.rfind("::");
            if (idx != string::npos) {
                if (tname.substr(0, idx) == m_absname) {
                    tname = tname.substr(idx + 2);
                } else {
                    error = true;
                }
            } else {
                error = !is_builtins();
            }

            if (!error) {
                ISymbol const *tsym = m_sym_tab.lookup_symbol(tname.c_str());
                if (tsym != NULL) {
                    Scope const *search_scope =
                        is_builtins() ?
                        m_def_tab.get_predef_scope() :
                        m_def_tab.get_global_scope();
                    def = search_scope->find_definition_in_scope(tsym);
                }
            }
        }

        if (def != NULL && def->get_kind() == IDefinition::DK_TYPE) {
            Scope *search_scope = def->get_own_scope();

            Definition const *def = search_scope->find_definition_in_scope(sym);
            if (def != NULL) {
                for (Definition const *cdef = def; cdef != NULL; cdef = cdef->get_prev_def()) {
                    Definition::Kind kind = cdef->get_kind();
                    if (kind != IDefinition::DK_FUNCTION &&
                        kind != IDefinition::DK_CONSTRUCTOR &&
                        kind != IDefinition::DK_OPERATOR)
                    {
                        // neither a function nor a constructor nor an operator
                        continue;
                    }

                    // check all overloads
                    if (matcher.match_param(cdef)) {
                        if (!only_exported || cdef->has_flag(Definition::DEF_IS_EXPORTED))
                            return cdef;
                        else {
                            // found it, but is not exported
                            break;
                        }
                    }
                }
            }
        }
    }

    // we have a symbol, look into the definition table
    Scope const *search_scope =
        is_builtins() ? m_def_tab.get_predef_scope() : m_def_tab.get_global_scope();

    IDefinition const *def = find_matching_def(search_scope, sym, matcher, only_exported);
    if (def != NULL)
        return def;
    if (second_sym != NULL)
        def = find_matching_def(search_scope, second_sym, matcher, only_exported);
    return def;
}

// Find the definition of a signature of a standard library function.
IDefinition const *Module::find_stdlib_signature(
    char const *module_name,
    char const *signature) const
{
    return m_compiler->find_stdlib_signature(module_name, signature);
}

// Set a deprecated message for a given definition.
void Module::set_deprecated_message(Definition const *def, IValue_string const *msg)
{
    m_deprecated_msg_map[def] = msg;
}

// Get the deprecated message for a given definition if any.
IValue_string const *Module::get_deprecated_message(Definition const *def) const
{
    Deprecated_msg_map::const_iterator it = m_deprecated_msg_map.find(def);
    if (it != m_deprecated_msg_map.end())
        return it->second;
    return NULL;
}

// Serialize this module.
void Module::serialize(Module_serializer &serializer) const
{
    serializer.write_section_tag(Serializer::ST_MODULE_START);
    DOUT(("Module START\n"));
    INC_SCOPE();

    if (serializer.is_module_registered(this)) {
        // this module was already serialized in this serializer
        Tag_t t = serializer.get_module_tag(this);
        serializer.write_encoded_tag(t);
        DOUT(("Tag %u\n", unsigned(t)));
        serializer.write_section_tag(Serializer::ST_MODULE_END);
        DEC_SCOPE();
        DOUT(("Module END (already serialized '%s')\n", m_absname));

        return;
    }

    Tag_t t = serializer.register_module(this);
    serializer.write_encoded_tag(t);
    DOUT(("Tag %u (id %u)\n", unsigned(t), unsigned(m_unique_id)));

    // do not write the unique ID, it can be changed on the deserialization site

    // write the name of this module
    serializer.write_cstring(m_absname);
    DOUT(("absname: '%s'\n", m_absname));

    // write the file name of this module
    serializer.write_cstring(m_filename);
    DOUT(("filename: '%s'\n", m_filename));

    // do not write m_qual_name, is generated by the absolute name

    serializer.write_bool(m_is_analyzed);
    DOUT(("analyzed: %s\n", m_is_analyzed ? "true" : "false"));
    serializer.write_bool(m_is_valid);
    DOUT(("valid: %s\n", m_is_valid ? "true" : "false"));
    serializer.write_bool(m_is_stdlib);
    DOUT(("stdlib: %s\n", m_is_stdlib ? "true" : "false"));
    serializer.write_bool(m_is_mdle);
    DOUT(("mdle: %s\n", m_is_mdle ? "true" : "false"));
    serializer.write_bool(m_is_builtins);
    DOUT(("builtin: %s\n", m_is_builtins ? "true" : "false"));
    serializer.write_bool(m_is_native);
    DOUT(("native: %s\n", m_is_native ? "true" : "false"));
    serializer.write_bool(m_is_compiler_owned);
    DOUT(("compiler owned: %s\n", m_is_compiler_owned ? "true" : "false"));
    serializer.write_bool(m_is_debug);
    DOUT(("debug: %s\n", m_is_debug ? "true" : "false"));
    serializer.write_bool(m_is_hashed);
    DOUT(("func hashes: %s\n", m_is_hashed ? "true" : "false"));

    serializer.write_encoded_tag(size_t(m_mdl_version));
    DOUT(("version: %u\n", m_mdl_version));

    // serialize the message list
    m_msg_list.serialize(serializer);

    // serialize first the symbol table
    m_sym_tab.serialize(serializer);

    // next the type factory: it uses the symbol table
    m_type_factory.serialize(serializer);

    // next the value factory: it uses the type factory
    m_value_factory.serialize(serializer);

    // now we can serialize the definition table
    m_def_tab.serialize(serializer);

    // serialize the import table
    size_t n_imports = m_imported_modules.size();
    serializer.write_encoded_tag(n_imports);
    DOUT(("#imports: %u\n", unsigned(n_imports)));
    INC_SCOPE();

    for (size_t i = 0; i < n_imports; ++i) {
        Import_entry const &import = m_imported_modules[i];

        char const *imp_absname = import.get_absolute_name();
        serializer.write_cstring(imp_absname);

        char const *imp_fname = import.get_file_name();
        serializer.write_cstring(imp_fname);

        Module const *imp_mod = import.get_module();
        if (serializer.is_known_module(imp_mod))
            t = serializer.get_module_tag(import.get_module());
        else
            t = Tag_t(0);
        serializer.write_encoded_tag(t);
        DOUT(("imported '%s', tag %u\n", imp_absname, unsigned(t)));
    }
    DEC_SCOPE();

    // serialize the exported definition list
    size_t n_exports = m_exported_definitions.size();
    serializer.write_encoded_tag(n_exports);
    DOUT(("#exports: %u\n", unsigned(n_exports)));
    INC_SCOPE();

    for (size_t i = 0; i < n_exports; ++i) {
        Definition const *export_def = m_exported_definitions[i];

        t = serializer.get_definition_tag(export_def);
        serializer.write_encoded_tag(t);
        DOUT(("exported def %u\n", unsigned(t)));
    }
    DEC_SCOPE();

    // serialize the deprecated messages
    size_t n_deprecated_msgs = m_deprecated_msg_map.size();
    serializer.write_encoded_tag(n_deprecated_msgs);
    DOUT(("#deprecated msgs: %u\n", unsigned(n_deprecated_msgs)));
    INC_SCOPE();
    for (Deprecated_msg_map::const_iterator
        it(m_deprecated_msg_map.begin()), end(m_deprecated_msg_map.end());
        it != end;
        ++it)
    {
        Definition const    *def = it->first;
        IValue_string const *msg = it->second;

        t = serializer.get_definition_tag(def);
        serializer.write_encoded_tag(t);
        DOUT(("deprecated def %u\n", unsigned(t)));

        t = serializer.get_value_tag(msg);
        serializer.write_encoded_tag(t);
        DOUT(("deprecated msg %u\n", unsigned(t)));
    }
    DEC_SCOPE();

    // serialize the archive info
    size_t n_arc_info = m_archive_versions.size();
    serializer.write_encoded_tag(n_arc_info);
    DOUT(("#arc_versions: %u\n", unsigned(n_arc_info)));
    INC_SCOPE();
    for (Arc_version_vec::const_iterator
        it(m_archive_versions.begin()), end(m_archive_versions.end());
        it != end;
        ++it)
    {
        Archive_version const  &v  = *it;
        Semantic_version const &sv = v.get_version();

        serializer.write_cstring(v.get_name());
        serializer.write_int(sv.get_major());
        serializer.write_int(sv.get_minor());
        serializer.write_int(sv.get_patch());
        serializer.write_cstring(sv.get_prerelease());
    }
    DEC_SCOPE();

    if (n_arc_info > 0) {
        serializer.write_encoded_tag(size_t(m_arc_mdl_version));
        DOUT(("arc version: %u\n", m_arc_mdl_version));
    }

    // serialize the resource table
    size_t n_res_tables = m_res_table.size();
    serializer.write_encoded_tag(n_res_tables);
    DOUT(("#res_table_entries: %u\n", unsigned(n_res_tables)));
    INC_SCOPE();

    for (size_t i = 0; i < n_res_tables; ++i) {
        Resource_entry const &re = m_res_table[i];

        char const *url = re.m_url;
        serializer.write_cstring(url);

        // Note: filename is NOT serialized
        IType_resource const *type = re.m_type;
        t = serializer.get_type_tag(type);
        serializer.write_encoded_tag(t);

        bool exists = re.m_exists;
        serializer.write_bool(exists);

        DOUT(("url '%s', type %u, esists %u\n", url, t, unsigned(exists)));
    }
    DEC_SCOPE();

    // serialize the function hashes
    if (m_is_hashed) {
        size_t n_func_hashes = m_func_hashes.size();
        serializer.write_encoded_tag(n_func_hashes);
        DOUT(("#rfunc_hashes: %u\n", unsigned(n_func_hashes)));
        INC_SCOPE();

        for (Func_hash_map::const_iterator it(m_func_hashes.begin()), end(m_func_hashes.end());
            it != end;
            ++it)
        {
            IDefinition const   *def  = it->first;
            Function_hash const &hash = it->second;

            t = serializer.get_definition_tag(def);
            serializer.write_encoded_tag(t);
            DOUT(("hashed def %u\n", unsigned(t)));

            for (size_t i = 0, n = dimension_of(hash.hash); i < n; ++i) {
                serializer.write_byte(hash.hash[i]);
            }
        }
        DEC_SCOPE();
    }

    // last the AST
    serialize_ast(serializer);

    serializer.write_section_tag(Serializer::ST_MODULE_END);
    DEC_SCOPE();
    DOUT(("Module END\n"));
}

/// visit all default arguments of the material constructors.
static void replace_in_material_geometry(
    Module *mod,
    Definition const *old_def,
    Definition const *new_def)
{
    class Replacer : public Module_visitor {
    public:
        /// Constructor.
        Replacer(Definition const *old_def, Definition const *new_def)
        : m_old_def(old_def)
        , m_new_def(new_def)
        {
        }

    private:
        /// Replace the definition old by new.
        IExpression *post_visit(IExpression_reference *expr) MDL_FINAL
        {
            if (expr->get_definition() == m_old_def) {
                expr->set_definition(m_new_def);
            }
            return expr;
        }

    private:
        Definition const *m_old_def;
        Definition const *m_new_def;
    };

    Replacer visitor(old_def, new_def);

    IType_struct const *mg_type =
        mod->get_type_factory()->get_predefined_struct(IType_struct::SID_MATERIAL_GEOMETRY);

    Scope *scope = mod->get_definition_table().get_type_scope(mg_type);

    for (Definition const *def = scope->get_first_definition_in_scope();
        def != NULL;
        def = def->get_next_def_in_scope())
    {
        if (def->get_kind() != Definition::DK_CONSTRUCTOR)
            continue;

        for (Definition const *cd = def; cd != NULL; cd = cd->get_next_def()) {
            IType_function const *ftype = cast<IType_function>(cd->get_type());

            for (int i = 0, n = ftype->get_parameter_count(); i < n; ++i) {
                MDL_ASSERT(
                    cd->get_original_import_idx() == 0 &&
                    "material definition not from this module");

                if (IExpression const *expr = cd->get_default_param_initializer(i)) {
                    visitor.visit(expr);
                }
            }
        }
    }
}


// Fix the state namespace after the deserialization.
void Module::fix_state_import_after_deserialization()
{
    // After the deserialization there might exists two "state" namespaces.
    // One that was created AFTER the empty module was initialized (created by auto-import
    // for the default material) and one that was deserialized.
    // Merge them and update the default material.

    ISymbol const *state_sym = m_sym_tab.get_predefined_symbol(ISymbol::SYM_CNST_STATE);
    Scope *global = m_def_tab.get_global_scope();

    Scope *old_state_ns = NULL, *state_ns = NULL;
    for (Scope const *s = global->get_first_subscope(); s != NULL; s = s->get_next_subscope()) {
        if (s->get_scope_name() == state_sym) {
            if (old_state_ns == NULL) {
                old_state_ns = const_cast<Scope *>(s);
            } else {
                MDL_ASSERT(state_ns == NULL);
                state_ns = const_cast<Scope *>(s);
            }
        }
    }
    if (old_state_ns != NULL && state_ns != NULL) {
        // first should be the auto-imported one
        MDL_ASSERT(old_state_ns->get_first_subscope() == NULL);
        Definition const *auto_normal = old_state_ns->get_first_definition_in_scope();
        MDL_ASSERT(auto_normal != NULL);
        MDL_ASSERT(auto_normal->get_next_def_in_scope() == NULL);

        // search normal in next
        Definition const *normal = state_ns->find_definition_in_scope(auto_normal->get_sym());
        MDL_ASSERT(normal != NULL);

        // now replace the old auto generated normal with the deserialized one
        replace_in_material_geometry(this, auto_normal, normal);

        // and kick the old state name scope
        old_state_ns->remove_from_parent();
    }
}

namespace {

class Owned_module_version_checker : public IDefinition_visitor
{
public:
    /// Called for every visited definition.
    ///
    /// \param def  the definition
    void visit(Definition const *def) const MDL_FINAL
    {
        if (m_mismatch) {
            // mismatch already found, stop checker
            return;
        }

        if (def->has_flag(Definition::DEF_IS_IMPORTED)) {
            size_t id = def->get_original_import_idx();
            MDL_ASSERT(id > 0 && "import index of imported entry is equal zero");

            if (m_owned_idxs.test_bit(id)) {
                // found a base import
                Definition const *odef = m_mod.get_original_definition(def);
                if (odef == NULL) {
                    m_mismatch = true;
                } else {
                    if (!check_sema(def, odef)) {
                        m_mismatch = true;
                    }
                }
            }
        }
    }

    /// Check matching semantics, checks name, types, NO default values.
    bool check_sema(Definition const *def, Definition const *odef) const
    {
        if (def->get_kind() != odef->get_kind())
            return false;

        if (def->get_semantics() != odef->get_semantics())
            return false;

        if (strcmp(def->get_sym()->get_name(), odef->get_sym()->get_name()) != 0) {
            return false;
        }
        return check_types(def->get_type(), odef->get_type());
    }

    /// Check matching types.
    bool check_types(IType const *tp, IType const *otp) const
    {
        if (tp->get_kind() != otp->get_kind())
            return false;

        switch (tp->get_kind()) {
        case IType::TK_ALIAS:
            {
                IType_alias const *atp  = cast<IType_alias>(tp);
                IType_alias const *oatp = cast<IType_alias>(otp);

                if (atp->get_type_modifiers() != oatp->get_type_modifiers())
                    return false;
                return check_types(atp->get_aliased_type(), oatp->get_aliased_type());
            }

        case IType::TK_BOOL:
        case IType::TK_INT:
        case IType::TK_FLOAT:
        case IType::TK_DOUBLE:
        case IType::TK_STRING:
        case IType::TK_LIGHT_PROFILE:
        case IType::TK_BSDF:
        case IType::TK_HAIR_BSDF:
        case IType::TK_EDF:
        case IType::TK_VDF:
        case IType::TK_TEXTURE:
        case IType::TK_BSDF_MEASUREMENT:
        case IType::TK_INCOMPLETE:
        case IType::TK_COLOR:
        case IType::TK_ERROR:
            // atomic types
            return true;

        case IType::TK_VECTOR:
            {
                IType_vector const *vtp  = cast<IType_vector>(tp);
                IType_vector const *ovtp = cast<IType_vector>(otp);

                if (vtp->get_size() != ovtp->get_size())
                    return false;
                return check_types(vtp->get_element_type(), ovtp->get_element_type());
            }

        case IType::TK_MATRIX:
            {
                IType_matrix const *mtp  = cast<IType_matrix>(tp);
                IType_matrix const *omtp = cast<IType_matrix>(otp);

                if (mtp->get_columns() != omtp->get_columns())
                    return false;
                return check_types(mtp->get_element_type(), omtp->get_element_type());
            }

        case IType::TK_ARRAY:
            {
                IType_array const *atp  = cast<IType_array>(tp);
                IType_array const *oatp = cast<IType_array>(otp);

                bool imm = atp->is_immediate_sized();
                if (imm != oatp->is_immediate_sized())
                    return false;
                if (imm) {
                    if (atp->get_size() != oatp->get_size()) {
                        return false;
                    }
                } else {
                    IType_array_size const *asz  = atp->get_deferred_size();
                    IType_array_size const *oasz = oatp->get_deferred_size();

                    if (strcmp(
                        asz->get_size_symbol()->get_name(),
                        oasz->get_size_symbol()->get_name()) != 0)
                    {
                        return false;
                    }
                }
                return check_types(atp->get_element_type(), oatp->get_element_type());
            }

        case IType::TK_FUNCTION:
            {
                IType_function const *ftp  = cast<IType_function>(tp);
                IType_function const *oftp = cast<IType_function>(otp);

                int n_params = ftp->get_parameter_count();
                if (n_params != oftp->get_parameter_count())
                    return false;

                for (int i = 0; i < n_params; ++i) {
                    ISymbol const *p_sym, *op_sym;
                    IType const *p_tp, *op_tp;

                    ftp->get_parameter(i, p_tp, p_sym);
                    oftp->get_parameter(i, op_tp, op_sym);

                    if (strcmp(p_sym->get_name(), op_sym->get_name()) != 0)
                        return false;

                    if (!check_types(p_tp, op_tp))
                        return false;
                }

                if (ftp->get_return_type() == NULL) {
                    if (oftp->get_return_type() != NULL)
                        return false;
                } else {
                    if (!check_types(ftp->get_return_type(), oftp->get_return_type()))
                        return false;
                }
                return true;
            }

        case IType::TK_ENUM:
            {
                IType_enum const *etp  = cast<IType_enum>(tp);
                IType_enum const *oetp = cast<IType_enum>(otp);

                if (etp->get_predefined_id() != oetp->get_predefined_id())
                    return false;

                int n_vals = etp->get_value_count();
                if (n_vals != oetp->get_value_count())
                    return false;

                if (strcmp(etp->get_symbol()->get_name(), oetp->get_symbol()->get_name()) != 0)
                    return false;

                for (int i = 0; i < n_vals; ++i) {
                    ISymbol const *v_sym, *ov_sym;
                    int code, ocode;

                    etp->get_value(i, v_sym, code);
                    oetp->get_value(i, ov_sym, ocode);

                    if (code != ocode)
                        return false;
                    if (strcmp(v_sym->get_name(), ov_sym->get_name()) != 0)
                        return false;
                }
                return true;
            }

        case IType::TK_STRUCT:
            {
                IType_struct const *stp  = cast<IType_struct>(tp);
                IType_struct const *ostp = cast<IType_struct>(otp);

                if (stp->get_predefined_id() != ostp->get_predefined_id())
                    return false;

                int n_fields = stp->get_field_count();
                if (n_fields != ostp->get_field_count())
                    return false;

                if (strcmp(stp->get_symbol()->get_name(), ostp->get_symbol()->get_name()) != 0)
                    return false;

                for (int i = 0; i < n_fields; ++i) {
                    ISymbol const *s_f, *os_f;
                    IType const *f_tp, *of_tp;

                    stp->get_field(i, f_tp, s_f);
                    ostp->get_field(i, of_tp, os_f);

                    if (strcmp(s_f->get_name(), os_f->get_name()) != 0)
                        return false;
                    if (!check_types(f_tp, of_tp))
                        return false;
                }
                return true;
            }
        }
        return false;
    }

    /// Returns true if mismatch.
    bool is_mismatch() const { return m_mismatch; }

    /// Reset the checker.
    void reset() { m_mismatch = false; }

    /// Constructor.
    Owned_module_version_checker(
        Module               &mod,
        Dynamic_bitset const &owned_idxs)
    : m_mod(mod)
    , m_owned_idxs(owned_idxs)
    , m_mismatch(false)
    {
    }

private:
    /// The current module.
    Module &m_mod;

    /// The set of owned module import indexes.
    Dynamic_bitset const &m_owned_idxs;

    /// Set to true if a mismatch was found.
    mutable bool m_mismatch;
};

}  // anonymous

// Check that owned imports match the builtin version after deserialization.
void Module::check_owned_imports()
{
    Dynamic_bitset owned_idxs(get_allocator());

    // check which modules are imported
    bool has_owned_import = false;
    for (size_t i = 0, n = m_imported_modules.size(); i < n; ++i) {
        Import_entry const &e = m_imported_modules[i];

        Module const *imp_mod = e.get_module();

        // if the checker is called inside deserialization, non-builtin modules
        // are not set, so ignore these entries.
        if (imp_mod == NULL)
            continue;

        if (imp_mod->m_is_compiler_owned) {
            // found owned module
            owned_idxs.set_bit(i + 1);
            has_owned_import = true;
        }
    }

    if (!has_owned_import)
        return;

    Owned_module_version_checker checker(*this, owned_idxs);

    // check if it matches.
    Scope *global = m_def_tab.get_global_scope();

    global->walk(&checker);

    if (checker.is_mismatch()) {
        MDL_ASSERT(!"owned module mismatch after deserialization");
        fatal_error("deserialization detected version mismatch of built-in modules");
    }
}

// Deserialize this module.
Module const *Module::deserialize(Module_deserializer &deserializer)
{
    DOUT(("Module START\n"));
    INC_SCOPE();
    Tag_t t;

    t = deserializer.read_encoded_tag();
    DOUT(("Tag %u\n", unsigned(t)));

    if (deserializer.is_known_module(t)) {
        // we already know this module

        Module const *mod = deserializer.get_module(t);

        t = deserializer.read_section_tag();
        MDL_ASSERT(t == Serializer::ST_MODULE_END);
        DEC_SCOPE();
        DOUT(("Module END (already deserialized '%s' %u)\n",
            mod->get_name(), unsigned(mod->get_unique_id())));

        mod->retain();
        return mod;
    }
    // otherwise create a new one

    // get the name of this module
    string abs_name(deserializer.read_cstring(), deserializer.get_allocator());
    DOUT(("absname: '%s'\n", abs_name.c_str()));
    // do NOT set the name here, as this will modify the symbol table which is NOT yet
    // deserialized, set it AFTER the table is ready

    // get the file name of this module
    string filename(deserializer.read_cstring(), deserializer.get_allocator());
    DOUT(("filename: '%s'\n", filename.c_str()));

    // do not read m_qual_name, is generated by the absolute name

    bool is_analyzed = deserializer.read_bool();
    DOUT(("analyzed: %s\n", is_analyzed ? "true" : "false"));
    bool is_valid    = deserializer.read_bool();
    DOUT(("valid: %s\n", is_valid ? "true" : "false"));
    bool is_stdlib   = deserializer.read_bool();
    DOUT(("stdlib: %s\n", is_stdlib ? "true" : "false"));
    bool is_mdle   = deserializer.read_bool();
    DOUT(("mdle: %s\n", is_mdle ? "true" : "false"));
    bool is_builtins = deserializer.read_bool();
    DOUT(("builtin: %s\n", is_builtins ? "true" : "false"));
    bool is_native = deserializer.read_bool();
    DOUT(("native: %s\n", is_native ? "true" : "false"));
    bool is_compiler_owned = deserializer.read_bool();
    DOUT(("compiler owned: %s\n", is_compiler_owned ? "true" : "false"));
    bool is_debug = deserializer.read_bool();
    DOUT(("debug: %s\n", is_debug ? "true" : "false"));
    bool is_hashed = deserializer.read_bool();
    DOUT(("func hashes: %s\n", is_hashed ? "true" : "false"));

    IMDL::MDL_version mdl_version = IMDL::MDL_version(deserializer.read_encoded_tag());
    DOUT(("version: %u\n", mdl_version));

    Module *mod = deserializer.create_module(mdl_version, is_analyzed);
    deserializer.register_module(t, mod);

    mod->set_filename(filename.c_str());

    mod->m_is_analyzed       = is_analyzed;
    mod->m_is_valid          = is_valid;
    mod->m_is_stdlib         = is_stdlib;
    mod->m_is_mdle           = is_mdle;
    mod->m_is_builtins       = is_builtins;
    mod->m_is_native         = is_native;
    mod->m_is_compiler_owned = is_compiler_owned;
    mod->m_is_debug          = is_debug;
    mod->m_is_hashed         = is_hashed;

    // deserialize the message list
    mod->m_msg_list.deserialize(deserializer);

    // deserialize first the symbol table
    mod->m_sym_tab.deserialize(deserializer);

    // next the type factory: it uses the symbol table
    mod->m_type_factory.deserialize(deserializer);

    // next the value factory: it uses the type factory
    mod->m_value_factory.deserialize(deserializer);

    // now we can deserialize the definition table
    mod->m_def_tab.deserialize(deserializer);

    // deserialize the import table
    size_t n_imports = deserializer.read_encoded_tag();
    DOUT(("#imports: %u\n", unsigned(n_imports)));
    INC_SCOPE();

    // clear the import table here: Because the (empty) module was analyzed
    // to fill the predefined entities, the table is not empty here
    mod->m_imported_modules.clear();

    mod->m_imported_modules.reserve(n_imports);
    for (size_t i = 0; i < n_imports; ++i) {
        string imp_absname(deserializer.read_cstring(), deserializer.get_allocator());
        string imp_fname(deserializer.read_cstring(), deserializer.get_allocator());

        t = deserializer.read_encoded_tag();
        DOUT(("imported '%s', tag %u\n", imp_absname.c_str(), unsigned(t)));

        if (t != Tag_t(0)) {
            Module const *imp_mod = deserializer.get_module(t);
            mod->register_import(imp_mod);
        } else {
            // cannot be a stdlib module if not always available
            mod->register_import(imp_absname.c_str(), imp_fname.c_str(), /*is_stdlib*/false);
        }
    }
    DEC_SCOPE();

    // deserialize the exported definition list
    size_t n_exports = deserializer.read_encoded_tag();
    DOUT(("#exports: %u\n", unsigned(n_exports)));
    INC_SCOPE();

    mod->m_exported_definitions.reserve(n_exports);
    for (size_t i = 0; i < n_exports; ++i) {
        t = deserializer.read_encoded_tag();
        DOUT(("exported def %u\n", unsigned(t)));

        Definition const *export_def = deserializer.get_definition(t);
        mod->m_exported_definitions.push_back(export_def);
    }
    DEC_SCOPE();

    // deserialize the deprecated messages
    size_t n_deprecated_msgs = deserializer.read_encoded_tag();
    DOUT(("#deprecated msgs: %u\n", unsigned(n_deprecated_msgs)));
    INC_SCOPE();
    for (size_t i = 0; i < n_deprecated_msgs; ++i) {
        t = deserializer.read_encoded_tag();
        DOUT(("deprecated def %u\n", unsigned(t)));
        Definition const *def = deserializer.get_definition(t);

        t = deserializer.read_encoded_tag();
        DOUT(("deprecated msg %u\n", unsigned(t)));
        IValue_string const *msg = cast<IValue_string>(deserializer.get_value(t));

        MDL_ASSERT(def != NULL && msg != NULL);
        mod->m_deprecated_msg_map[def] = msg;
    }
    DEC_SCOPE();

    // deserialize the archive info
    size_t n_arc_info = deserializer.read_encoded_tag();
    DOUT(("#arc_versions: %u\n", unsigned(n_arc_info)));
    INC_SCOPE();
    mod->m_archive_versions.clear();
    mod->m_archive_versions.reserve(n_arc_info);
    for (size_t i = 0; i < n_arc_info; ++i) {
        char const *name = Arena_strdup(mod->m_arena, deserializer.read_cstring());

        int major = deserializer.read_int();
        int minor = deserializer.read_int();
        int patch = deserializer.read_int();
        char const *prerelease = Arena_strdup(mod->m_arena, deserializer.read_cstring());

        Semantic_version ver(major, minor, patch, prerelease);
        mod->m_archive_versions.push_back(Archive_version(name, &ver));
    }
    DEC_SCOPE();

    if (n_arc_info > 0) {
        mod->m_arc_mdl_version = IMDL::MDL_version(deserializer.read_encoded_tag());
        DOUT(("arc version: %u\n", mod->m_arc_mdl_version));
    }

    // deserialize the resource table
    size_t n_res_tables = deserializer.read_encoded_tag();
    DOUT(("#res_table_entries: %u\n", unsigned(n_res_tables)));
    mod->m_res_table.clear();
    mod->m_res_table.reserve(n_res_tables);

    INC_SCOPE();

    for (size_t i = 0; i < n_res_tables; ++i) {
        char const *url = Arena_strdup(mod->m_arena, deserializer.read_cstring());

        // Note: filename was not serialized

        t = deserializer.read_encoded_tag();
        IType_resource const *type = cast<IType_resource>(deserializer.get_type(t));

        bool exists = deserializer.read_bool();

        DOUT(("url '%s', type %u, esists %u\n", url, t, unsigned(exists)));

        mod->m_res_table.push_back(Module::Resource_entry(url, /*filename=*/NULL, type, exists));
    }
    DEC_SCOPE();

    // deserialize the function hashes
    if (is_hashed) {
        size_t n_func_hashes = deserializer.read_encoded_tag();
        DOUT(("#rfunc_hashes: %u\n", unsigned(n_func_hashes)));
        INC_SCOPE();

        for (size_t i = 0; i < n_func_hashes; ++i) {
            t = deserializer.read_encoded_tag();
            DOUT(("hashed def %u\n", unsigned(t)));
            IDefinition const *def = deserializer.get_definition(t);

            Function_hash hash;
            for (size_t i = 0, n = dimension_of(hash.hash); i < n; ++i) {
                hash.hash[i] = deserializer.read_byte();
            }
            mod->m_func_hashes[def] = hash;
        }
        DEC_SCOPE();
    }

    // last the AST
    mod->deserialize_ast(deserializer);

    // now it is safe to set the name
    mod->set_name(abs_name.c_str());

    t = deserializer.read_section_tag();
    MDL_ASSERT(t == Serializer::ST_MODULE_END);
    DEC_SCOPE();
    DOUT(("Module END\n"));

    mod->fix_state_import_after_deserialization();

    mod->check_owned_imports();

    return mod;
}

// Serialize the AST of this module.
void Module::serialize_ast(Module_serializer &serializer) const
{
    serializer.write_section_tag(Serializer::ST_AST);

    DOUT(("AST\n"));

    size_t n_decls = m_declarations.size();
    serializer.write_encoded_tag(n_decls);

    DOUT(("#decls %u\n", unsigned(n_decls)));
    INC_SCOPE();

    // write all declarations
    for (size_t i = 0; i < n_decls; ++i) {
        IDeclaration const *decl = m_declarations[i];

        serializer.write_decl(decl);
    }

    // write all expressions that are only referenced by initializers
    serializer.write_unreferenced_init_expressions();
    DEC_SCOPE();
}

// Deserialize the AST of this module.
void Module::deserialize_ast(Module_deserializer &deserializer)
{
    Tag_t t;

    t = deserializer.read_section_tag();
    MDL_ASSERT(t == Serializer::ST_AST);

    DOUT(("AST\n"));

    size_t n_decls = deserializer.read_encoded_tag();

    DOUT(("#decls %u\n", unsigned(n_decls)));
    INC_SCOPE();

    // clear the declaration list first: we might have analyzed the empty module,
    // which generates an "import state::normal"
    m_declarations.clear();

    // read all declarations
    m_declarations.reserve(n_decls);
    for (size_t i = 0; i < n_decls; ++i) {
        IDeclaration *decl = deserializer.read_decl(*this);

        m_declarations.push_back(decl);
    }

    // read all unreferenced init expressions
    deserializer.read_unreferenced_init_expressions(*this);
    DEC_SCOPE();
}

// Get the compiler of this module.
MDL *Module::get_compiler() const {
    m_compiler->retain();
    return m_compiler;
}

// Enter a new semantic version for given module.
bool Module::set_semantic_version(
    int            major,
    int            minor,
    int            patch,
    char const     *prelease)
{
    if (m_sema_version != NULL)
        return false;

    Arena_builder builder(m_arena);

    m_sema_version = builder.create<Semantic_version>(
        major, minor, patch, Arena_strdup(m_arena, prelease));
    return true;
}

// Set the module info from an archive manifest.
void Module::set_archive_info(
    IArchive_manifest const *manifest)
{
    m_arc_mdl_version = manifest->get_mdl_version();

    size_t n = 1;
    for (IArchive_manifest_dependency const *dep = manifest->get_first_dependency();
         dep != NULL;
         dep = dep->get_next())
    {
        ++n;
    }

    m_archive_versions.reserve(n);

    /// The version of THIS module is always the first
    m_archive_versions.push_back(
        Archive_version(
            Arena_strdup(m_arena, manifest->get_archive_name()),
            manifest->get_sema_version()));

    for (IArchive_manifest_dependency const *dep = manifest->get_first_dependency();
        dep != NULL;
        dep = dep->get_next())
    {
        m_archive_versions.push_back(
            Archive_version(
            Arena_strdup(m_arena, dep->get_archive_name()),
            dep->get_version()));
    }
}

// Get the owner archive version if any.
Module::Archive_version const *Module::get_owner_archive_version() const
{
    if (m_archive_versions.size() > 0)
        return &m_archive_versions[0];
    return NULL;
}

// Get the number of archive dependencies.
size_t Module::get_archive_dependencies_count() const
{
    size_t res = m_archive_versions.size();

    if (res > 0)
        res -= 1;
    return res;
}

// Get the i'th archive dependency.
Module::Archive_version const *Module::get_archive_dependency(size_t i) const
{
    if (i < get_archive_dependencies_count())
        return &m_archive_versions[i + 1];
    return NULL;
}

// Add a new resource entry.
void Module::add_resource_entry(
    char const           *url,
    char const           *file_name,
    IType_resource const *type,
    bool                 exists)
{
    m_res_table.push_back(
        Resource_entry(
            Arena_strdup(m_arena, url),
            Arena_strdup(m_arena, file_name),
            type,
            exists));
}

// Alters one call argument according to the given promotion rules.
int Module::promote_call_arguments(
    IExpression_call *call,
    IArgument const  *arg,
    int              param_index,
    unsigned         rules)
{
    if (rules & PR_SPOT_EDF_ADD_SPREAD_PARAM) {
        if (param_index == 0) {
            // add M_PI as 1. argument
            IValue_float const  *v    = m_value_factory.create_float(M_PIf);
            IExpression const   *e    = m_expr_factory.create_literal(v);
            IArgument const     *narg = m_expr_factory.create_positional_argument(e);
            call->add_argument(narg);
            return param_index + 1;
        }
    }
    if (rules & PC_MEASURED_EDF_ADD_MULTIPLIER) {
        if (param_index == 0) {
            // add 1.0 as 1. argument
            IValue_float const  *v    = m_value_factory.create_float(1.0);
            IExpression const   *e    = m_expr_factory.create_literal(v);
            IArgument const     *narg = m_expr_factory.create_positional_argument(e);
            call->add_argument(narg);
            return param_index + 1;
        }
    }
    if (rules & PR_MEASURED_EDF_ADD_TANGENT_U) {
        if (param_index == 3) {
            // add state::tangent_u(0) as 4. argument

            ISymbol const *sym_state = m_sym_tab.get_symbol("state");
            ISymbol const *sym_tanu = m_sym_tab.get_symbol("texture_tangent_u");

            ISimple_name *sn_state = m_name_factory.create_simple_name(sym_state);
            ISimple_name *sn_tanu = m_name_factory.create_simple_name(sym_tanu);

            IQualified_name *qn = m_name_factory.create_qualified_name();
            qn->add_component(sn_state);
            qn->add_component(sn_tanu);
            qn->set_absolute();

            IType_name *tn = m_name_factory.create_type_name(qn);

            IExpression_reference *ref = m_expr_factory.create_reference(tn);
            IExpression_call      *tu_call = m_expr_factory.create_call(ref);

            IValue_int const  *v = m_value_factory.create_int(0);
            IExpression const *e = m_expr_factory.create_literal(v);
            IArgument const   *a = m_expr_factory.create_positional_argument(e);

            tu_call->add_argument(a);

            IArgument const *narg = m_expr_factory.create_positional_argument(tu_call);
            call->add_argument(narg);
            return param_index + 1;
        }
    }
    if (rules & PR_FRESNEL_LAYER_TO_COLOR) {
        if (param_index == 1) {
            // wrap the 1. argument by a color constructor
            ISymbol const *sym_color = m_sym_tab.get_symbol("color");

            ISimple_name *sn_color = m_name_factory.create_simple_name(sym_color);
            IQualified_name *qn = m_name_factory.create_qualified_name();
            qn->add_component(sn_color);

            IType_name *tn = m_name_factory.create_type_name(qn);

            IExpression_reference *ref = m_expr_factory.create_reference(tn);
            IExpression_call      *color_call = m_expr_factory.create_call(ref);

            IExpression const *e = arg->get_argument_expr();
            IArgument const   *a = m_expr_factory.create_positional_argument(e);

            color_call->add_argument(a);

            const_cast<IArgument *>(arg)->set_argument_expr(color_call);
            return param_index;
        }
    }
    if (rules & PR_WIDTH_HEIGHT_ADD_UV_TILE) {
        if (param_index == 0) {
            // add int2(0) as 1. argument
            ISymbol const *sym_int2 = m_sym_tab.get_symbol("int2");

            ISimple_name *sn_int2 = m_name_factory.create_simple_name(sym_int2);

            IQualified_name *qn = m_name_factory.create_qualified_name();
            qn->add_component(sn_int2);

            IType_name *tn = m_name_factory.create_type_name(qn);

            IExpression_reference *ref = m_expr_factory.create_reference(tn);
            IExpression_call      *int2_call = m_expr_factory.create_call(ref);

            IValue_int const  *v = m_value_factory.create_int(0);
            IExpression const *e = m_expr_factory.create_literal(v);
            IArgument const   *a = m_expr_factory.create_positional_argument(e);

            int2_call->add_argument(a);

            IArgument const *narg = m_expr_factory.create_positional_argument(int2_call);
            call->add_argument(narg);
            return param_index + 1;
        }
    }
    if (rules & PR_TEXEL_ADD_UV_TILE) {
        if (param_index == 1) {
            // add int2(0) as 2. argument
            ISymbol const *sym_int2 = m_sym_tab.get_symbol("int2");

            ISimple_name *sn_int2 = m_name_factory.create_simple_name(sym_int2);

            IQualified_name *qn = m_name_factory.create_qualified_name();
            qn->add_component(sn_int2);

            IType_name *tn = m_name_factory.create_type_name(qn);

            IExpression_reference *ref = m_expr_factory.create_reference(tn);
            IExpression_call      *int2_call = m_expr_factory.create_call(ref);

            IValue_int const  *v = m_value_factory.create_int(0);
            IExpression const *e = m_expr_factory.create_literal(v);
            IArgument const   *a = m_expr_factory.create_positional_argument(e);

            int2_call->add_argument(a);

            IArgument const *narg = m_expr_factory.create_positional_argument(int2_call);
            call->add_argument(narg);
            return param_index + 1;
        }
    }
    if (rules & PR_ROUNDED_CORNER_ADD_ROUNDNESS) {
        if (param_index == 1) {
            // add roundness as 2. argument
            IValue_float const  *v    = m_value_factory.create_float(1.0f);
            IExpression const   *e    = m_expr_factory.create_literal(v);
            IArgument const     *narg = m_expr_factory.create_positional_argument(e);
            call->add_argument(narg);
            return param_index + 1;
        }
    }
    if (rules & PR_MATERIAL_ADD_HAIR) {
        if (param_index == 5) {
            // MDL 1.4 -> 1.5: add default hair bsdf
            ISymbol const *s = m_name_factory.create_symbol("hair_bsdf");
            ISimple_name const *sn = m_name_factory.create_simple_name(s);
            IQualified_name *qn = m_name_factory.create_qualified_name();
            qn->add_component(sn);
            IType_name *tn = m_name_factory.create_type_name(qn);
            IExpression_reference *hair_ref = m_expr_factory.create_reference(tn);
            IExpression_call *hair_call = m_expr_factory.create_call(hair_ref);

            IArgument const *a    = call->get_argument(5);
            IArgument const *narg = NULL;
            if (a->get_kind() == IArgument::AK_POSITIONAL) {
                narg = m_expr_factory.create_positional_argument(hair_call);
            } else {
                ISymbol const      *sh      = m_name_factory.create_symbol("hair");
                ISimple_name const *sn_hair = m_name_factory.create_simple_name(sh);
                narg = m_expr_factory.create_named_argument(sn_hair, hair_call);
            }
            call->add_argument(narg);

            return param_index + 1;
        }
    }
    if (rules & PR_GLOSSY_ADD_MULTISCATTER) {
        if (param_index == 2) {
            // add color(0) as 3. argument for glossy bsdfs
            ISymbol const *sym_color = m_sym_tab.get_symbol("color");

            ISimple_name *sn_color = m_name_factory.create_simple_name(sym_color);

            IQualified_name *qn = m_name_factory.create_qualified_name();
            qn->add_component(sn_color);

            IType_name *tn = m_name_factory.create_type_name(qn);

            IExpression_reference *ref = m_expr_factory.create_reference(tn);
            IExpression_call      *color_call = m_expr_factory.create_call(ref);

            IValue_float const *v = m_value_factory.create_float(0.0f);
            IExpression const *e  = m_expr_factory.create_literal(v);
            IArgument const   *a  = m_expr_factory.create_positional_argument(e);

            color_call->add_argument(a);

            IArgument const *narg = m_expr_factory.create_positional_argument(color_call);
            call->add_argument(narg);
            return param_index + 1;
        }
    }
    return param_index;
}

namespace {

/// Helper class to enumerate referenced resources.
class Resource_enumerator : public Module_visitor {
    typedef Module_visitor Base;

public:
    /// Constructor.
    ///
    /// \param module    the current module
    /// \param ana       the calling analysis
    /// \param resolver  the file resolver to be used
    /// \param rrh       the resource restriction handler
    Resource_enumerator(
        Module                        *module,
        Analysis                      &ana,
        File_resolver                 &resolver,
        IResource_restriction_handler &rrh)
    : m_module(module)
    , m_ana(ana)
    , m_resolver(resolver)
    , m_rrh(rrh)
    {
    }

private:
    /// Check restrictions on a resource URL.
    void check_restriction(
        char const     *url,
        Position const &pos)
    {
        IAllocator *alloc = m_module->get_allocator();

        mi::base::Handle<IMDL_resource_set> result(m_resolver.resolve_resource(
            pos,
            url,
            m_module->get_name(),
            m_module->get_filename()));

        if (result.is_valid_interface()) {
            for (size_t i = 0, n = result->get_count(); i < n; ++i) {
                IResource_restriction_handler::Resource_restriction rr =
                    m_rrh.process(m_module, result->get_filename(i));

                switch (rr) {
                case IResource_restriction_handler::RR_OK:
                    break;
                case IResource_restriction_handler::RR_NOT_EXISTANT:
                    m_ana.warning(
                        MISSING_RESOURCE,
                        pos,
                        Error_params(alloc).add(url));
                    break;
                case IResource_restriction_handler::RR_OUTSIDE_ARCHIVE:
                    m_ana.warning(
                        RESOURCE_OUTSIDE_ARCHIVE,
                        pos,
                        Error_params(alloc).add(url));
                    break;
                }
            }
        } else {
            m_ana.warning(
                MISSING_RESOURCE,
                pos,
                Error_params(alloc).add(url));
        }
    }

    /// Visit literals.
    IExpression *post_visit(IExpression_literal *expr) MDL_OVERRIDE
    {
        IValue const *v = expr->get_value();

        if (IValue_resource const *res = as<IValue_resource>(v)) {
            char const *url = res->get_string_value();

            check_restriction(url, expr->access_position());
        }
        return expr;
    }

    /// Visit calls.
    IExpression *post_visit(IExpression_call *call) MDL_FINAL
    {
        IExpression_reference const *ref = as<IExpression_reference>(call->get_reference());
        if (ref == NULL || ref->is_array_constructor()) {
            return call;
        }

        if (!is<IType_resource>(call->get_type()->skip_type_alias())) {
            return call;
        }

        IDefinition const      *def = ref->get_definition();
        IDefinition::Semantics sema = def->get_semantics();

        char const *url = NULL;

        if (sema == IDefinition::DS_TEXTURE_CONSTRUCTOR) {
            // texture constructor
            IArgument const           *name = call->get_argument(0);
            IExpression_literal const *lit  = as<IExpression_literal>(name->get_argument_expr());

            if (lit != NULL) {
                IValue_string const *s = as<IValue_string>(lit->get_value());
                if (s == NULL) {
                    return call;
                }

                url = s->get_value();
            }
        } else if (sema == IDefinition::DS_CONV_CONSTRUCTOR) {
            // conversion from string to resource
            IArgument const           *value = call->get_argument(0);
            IExpression_literal const *lit   = as<IExpression_literal>(value->get_argument_expr());

            if (lit != NULL) {
                IValue_string const *s = as<IValue_string>(lit->get_value());
                if (s == NULL) {
                    return call;
                }

                url = s->get_value();
            }
        }

        if (url != NULL) {
            check_restriction(url, call->access_position());
        }
        return call;
    }

private:
    /// The current module.
    Module *m_module;

    /// The calling analysis.
    Analysis &m_ana;

    /// The file resolver to be used to resolve resources.
    File_resolver &m_resolver;

    /// The restriction handler.
    IResource_restriction_handler &m_rrh;
};

}  // anonymous

// Enumerate all referenced resources by this module.
void Module::check_referenced_resources(
    Analysis                      &ana,
    File_resolver                 &res_resolver,
    IResource_restriction_handler &rrh)
{
    Resource_enumerator numerator(this, ana, res_resolver, rrh);

    numerator.visit(this);
}

/// Construct a Type_name AST element for an MDL type.
///
/// \param type   the MDL type
/// \param owner  the MDL module that will own the newly constructed type name
///
/// \note The type \c type must be owned by the module \c owner.
static IType_name *construct_type_name(
    IType const *type,
    Module      *owner)
{
    IType::Modifiers modifiers = type->get_type_modifiers();
    type = type->skip_type_alias();

    switch (type->get_kind()) {
    case IType::TK_ALIAS:
    case IType::TK_FUNCTION:
    case IType::TK_INCOMPLETE:
    case IType::TK_ERROR:
        // should not happen
        MDL_ASSERT(!"unexpected type kind");
        return NULL;
    case IType::TK_BOOL:
    case IType::TK_INT:
    case IType::TK_ENUM:
    case IType::TK_FLOAT:
    case IType::TK_DOUBLE:
    case IType::TK_STRING:
    case IType::TK_LIGHT_PROFILE:
    case IType::TK_BSDF:
    case IType::TK_HAIR_BSDF:
    case IType::TK_EDF:
    case IType::TK_VDF:
    case IType::TK_VECTOR:
    case IType::TK_MATRIX:
    case IType::TK_COLOR:
    case IType::TK_STRUCT:
    case IType::TK_TEXTURE:
    case IType::TK_BSDF_MEASUREMENT:
        {
            IAllocator *alloc = owner->get_allocator();
            Allocator_builder builder(alloc);
            mi::base::Handle<Buffer_output_stream> buffer(
                builder.create<Buffer_output_stream>(alloc));
            mi::base::Handle<IMDL>     compiler(owner->get_compiler());
            mi::base::Handle<IPrinter> printer(compiler->create_printer(buffer.get()));

            printer->print(type);

            char *name = buffer->get_data();

            Name_factory *nf = owner->get_name_factory();

            IQualified_name *qname = nf->create_qualified_name();
            IType_name      *tn    = nf->create_type_name(qname);

            if (name[0] == ':' && name[1] == ':') {
                qname->set_absolute();
                tn->set_absolute();
                name += 2;
            }

            Symbol_table &st = owner->get_symbol_table();

            for (;;) {
                char *p = strstr(name, "::");
                if (p == NULL)
                    break;

                *p = '\0';

                ISymbol const      *sym   = st.get_symbol(name);
                ISimple_name const *sname = nf->create_simple_name(sym);

                qname->add_component(sname);

                name = p + 2;
            }

            ISymbol const      *sym   = st.get_symbol(name);
            ISimple_name const *sname = nf->create_simple_name(sym);

            qname->add_component(sname);

            if (modifiers & IType::MK_UNIFORM) {
                tn->set_qualifier(FQ_UNIFORM);
            } else if (modifiers & IType::MK_VARYING) {
                tn->set_qualifier(FQ_VARYING);
            }

            return tn;
        }
        break;
    case IType::TK_ARRAY:
        {
            IType_array const *a_tp = cast<IType_array>(type);

            IType const *e_tp = a_tp->get_element_type();

            IType_name *tn = construct_type_name(e_tp, owner);

            if (modifiers & IType::MK_UNIFORM) {
                tn->set_qualifier(FQ_UNIFORM);
            } else if (modifiers & IType::MK_VARYING) {
                tn->set_qualifier(FQ_VARYING);
            }

            Expression_factory *ef = owner->get_expression_factory();
            if (a_tp->is_immediate_sized()) {
                size_t size = a_tp->get_size();

                Value_factory *vf     = owner->get_value_factory();
                IValue const  *v_size = vf->create_int(int(size));

                IExpression_literal *lit = ef->create_literal(v_size);

                tn->set_array_size(lit);
            } else {
                IType_array_size const *size = a_tp->get_deferred_size();
                Name_factory           *nf   = owner->get_name_factory();

                ISymbol const      *sym   = size->get_size_symbol();
                ISimple_name const *sname = nf->create_simple_name(sym);
                IQualified_name    *qname = nf->create_qualified_name();

                qname->add_component(sname);

                IType_name const            *tname = nf->create_type_name(qname);
                IExpression_reference const *ref   = ef->create_reference(tname);

                tn->set_array_size(ref);
            }
            return tn;
        }
    }
    MDL_ASSERT(!"unsupported type kind");
    return NULL;
}

static IExpression const *promote_expr(
    Module            &mod,
    IExpression const *expr);

/// Promote the given argument.
static IArgument const *promote_argument(
    Module           &mod,
    IArgument const *arg)
{
    switch (arg->get_kind()) {
    case IArgument::AK_POSITIONAL:
        {
            IArgument_positional const *pos = cast<IArgument_positional>(arg);
            IExpression const *expr = promote_expr(mod, pos->get_argument_expr());
            return mod.get_expression_factory()->create_positional_argument(expr);
        }
    case IArgument::AK_NAMED:
        {
            IArgument_named const *named = cast<IArgument_named>(arg);
            ISimple_name const *sname = named->get_parameter_name();
            IExpression const *expr = promote_expr(mod, named->get_argument_expr());

            return mod.get_expression_factory()->create_named_argument(sname, expr);
        }
    }
    MDL_ASSERT(!"unssuported argument kind");
    return arg;
}

///< Promote a given function call to the version of a given module.
///
/// \param[in]  mod    the destination module
/// \param[in]  tn     the type name of the call reference
/// \param[out] rules  the promotion rule set that must be applied to the call arguments
static IType_name const *promote_name(
    Module           &mod,
    IType_name const *tn,
    unsigned         &rules)
{
    rules = Module::PR_NO_CHANGE;
    if (tn->is_array())
        return tn;

    int mod_major = 0, mod_minor = 0;
    mod.get_version(mod_major, mod_minor);

    IQualified_name const *qn = tn->get_qualified_name();

    int n_components = qn->get_component_count();
    ISymbol const *sym;
    ISimple_name const *sn;
    string n(mod.get_allocator());
    if (n_components == 1) {
        sn = qn->get_component(0);
        sym = sn->get_symbol();
        if (strcmp(sym->get_name(), "material$1.4") == 0) {
            if (mod_major > 1 || (mod_major == 1 && mod_minor > 4)) {
                rules = Module::PR_MATERIAL_ADD_HAIR;
                n = "material";
            } else {
                n = "material";
            }
        } else {
            return tn;
        }
    } else {
        sn               = qn->get_component(n_components -1);
        sym              = sn->get_symbol();
        char const *name = sym->get_name();

        char const *suffix = strrchr(name, '$');
        if (suffix == NULL)
            return tn;
        int major = 0, minor = 0;

        char const *p = suffix;
        for (++p; *p != '.' && *p != '\0'; ++p) {
            major *= 10;
            switch (*p) {
            case '0': major += 0; break;
            case '1': major += 1; break;
            case '2': major += 2; break;
            case '3': major += 3; break;
            case '4': major += 4; break;
            case '5': major += 5; break;
            case '6': major += 6; break;
            case '7': major += 7; break;
            case '8': major += 8; break;
            case '9': major += 9; break;
            default:
                return tn;
            }
        }
        if (*p != '.')
            return tn;

        for (++p; *p != '\0'; ++p) {
            minor *= 10;
            switch (*p) {
            case '0': minor += 0; break;
            case '1': minor += 1; break;
            case '2': minor += 2; break;
            case '3': minor += 3; break;
            case '4': minor += 4; break;
            case '5': minor += 5; break;
            case '6': minor += 6; break;
            case '7': minor += 7; break;
            case '8': minor += 8; break;
            case '9': minor += 9; break;
            default:
                return tn;
            }
        }
        if (*p != '\0')
            return tn;

        n = string(name, suffix, mod.get_allocator());

        if (major < mod_major || (major == mod_major && minor < mod_minor)) {
            // the symbol was removed BEFORE the module version, we need promotion
            // which might change the name
            if (n_components == 2) {
                ISimple_name const *sn = qn->get_component(0);
                ISymbol const *package = sn->get_symbol();

                if (strcmp(package->get_name(), "df") == 0) {
                    if (major == 1) {
                        if (minor == 0) {
                            // all functions deprecated after MDL 1.0
                            if (n == "spot_edf") {
                                rules = Module::PR_SPOT_EDF_ADD_SPREAD_PARAM;
                            } else if (n == "measured_edf") {
                                if (mod_major > 1 || (mod_major == 1 && mod_minor >= 1)) {
                                    rules |= Module::PC_MEASURED_EDF_ADD_MULTIPLIER;
                                }
                                if (mod_major > 1 || (mod_major == 1 && mod_minor >= 2)) {
                                    rules |= Module::PR_MEASURED_EDF_ADD_TANGENT_U;
                                }
                            }
                        } else if (minor == 1) {
                            // all functions deprecated after MDL 1.1
                            if (n == "measured_edf") {
                                rules = Module::PR_MEASURED_EDF_ADD_TANGENT_U;
                            }
                        } else if (minor == 3) {
                            // all functions deprecated after MDL 1.3
                            if (n == "fresnel_layer") {
                                rules = Module::PR_FRESNEL_LAYER_TO_COLOR;
                                n = "color_fresnel_layer";
                            }
                        } else if (minor == 5) {
                            // all functions deprecated after MDL 1.5
                            if (n == "simple_glossy_bsdf" ||
                                n == "backscattering_glossy_reflection_bsdf" ||
                                n == "microfacet_beckmann_smith_bsdf" ||
                                n == "microfacet_ggx_smith_bsdf" ||
                                n == "microfacet_beckmann_vcavities_bsdf" ||
                                n == "microfacet_ggx_vcavities_bsdf" ||
                                n == "ward_geisler_moroder_bsdf")
                            {
                                rules = Module::PR_GLOSSY_ADD_MULTISCATTER;
                            }
                        }
                    }
                } else if (strcmp(package->get_name(), "state") == 0) {
                    if (major == 1) {
                        if (minor == 2) {
                            // all functions deprecated after MDL 1.2
                            if (n == "rounded_corner_normal") {
                                rules = Module::PR_ROUNDED_CORNER_ADD_ROUNDNESS;
                            }
                        }
                    }
                } else if (strcmp(package->get_name(), "tex") == 0) {
                    if (major == 1) {
                        if (minor == 3) {
                            // all functions deprecated after MDL 1.3
                            if (n == "width") {
                                rules = Module::PR_WIDTH_HEIGHT_ADD_UV_TILE;
                            } else if (n == "height") {
                                rules = Module::PR_WIDTH_HEIGHT_ADD_UV_TILE;
                            } else if (n == "texel_float") {
                                rules = Module::PR_TEXEL_ADD_UV_TILE;
                            } else if (n == "texel_float2") {
                                rules = Module::PR_TEXEL_ADD_UV_TILE;
                            } else if (n == "texel_float3") {
                                rules = Module::PR_TEXEL_ADD_UV_TILE;
                            } else if (n == "texel_float4") {
                                rules = Module::PR_TEXEL_ADD_UV_TILE;
                            } else if (n == "texel_color") {
                                rules = Module::PR_TEXEL_ADD_UV_TILE;
                            }
                        }
                    }
                }
            }
        }
    }
    sym = mod.get_symbol_table().get_symbol(n.c_str());

    Name_factory *name_fact = mod.get_name_factory();

    IQualified_name *n_qn = name_fact->create_qualified_name();

    for (int i = 0; i < n_components - 1; ++i) {
        n_qn->add_component(qn->get_component(i));
    }

    sn = name_fact->create_simple_name(sym);
    n_qn->add_component(sn);

    if (qn->is_absolute())
        n_qn->set_absolute();
    n_qn->set_definition(qn->get_definition());

    IType_name *n_tn = name_fact->create_type_name(n_qn);

    if (tn->is_absolute())
        n_tn->set_absolute();
    if (IExpression const *array_size = tn->get_array_size())
        n_tn->set_array_size(promote_expr(mod, array_size));
    if (tn->is_incomplete_array())
        n_tn->set_incomplete_array();
    n_tn->set_qualifier(tn->get_qualifier());
    if (ISimple_name const *size_name = tn->get_size_name())
        n_tn->set_size_name(size_name);
    n_tn->set_type(tn->get_type());
    return n_tn;
}

/// Promote the given expression.
static IExpression const *promote_expr(
    Module            &mod,
    IExpression const *expr)
{
    switch (expr->get_kind()) {
    case IExpression::EK_INVALID:
    case IExpression::EK_LITERAL:
    case IExpression::EK_REFERENCE:
        return expr;
    case IExpression::EK_UNARY:
        {
            IExpression_unary const *unexpr = cast<IExpression_unary>(expr);
            IExpression const *arg = promote_expr(mod, unexpr->get_argument());
            IExpression_unary *u =
                mod.get_expression_factory()->create_unary(unexpr->get_operator(), arg);
            u->set_type_name(unexpr->get_type_name());
            return u;
        }
    case IExpression::EK_BINARY:
        {
            IExpression_binary const *binexpr = cast<IExpression_binary>(expr);
            IExpression const *lhs = promote_expr(mod, binexpr->get_left_argument());
            IExpression const *rhs = promote_expr(mod, binexpr->get_right_argument());
            IExpression_binary *b =
                mod.get_expression_factory()->create_binary(binexpr->get_operator(), lhs, rhs);
            b->set_type(expr->get_type());
            return b;
        }
    case IExpression::EK_CONDITIONAL:
        {
            IExpression_conditional const *c_expr = cast<IExpression_conditional>(expr);
            IExpression const *cond = promote_expr(mod, c_expr->get_condition());
            IExpression const *t_ex = promote_expr(mod, c_expr->get_true());
            IExpression const *f_ex = promote_expr(mod, c_expr->get_false());
            IExpression_conditional *c =
                mod.get_expression_factory()->create_conditional(cond, t_ex, f_ex);
            c->set_type(expr->get_type());
            return c;
        }
    case IExpression::EK_CALL:
        {
            IExpression_call const *c_expr = cast<IExpression_call>(expr);
            IExpression const *r           = c_expr->get_reference();

            unsigned rules = Module::PR_NO_CHANGE;
            if (IExpression_reference const *ref = as<IExpression_reference>(r)) {
                IType_name const *tn = ref->get_name();

                tn = promote_name(mod, tn, rules);

                r = mod.get_expression_factory()->create_reference(tn);
            }

            IExpression_call *call = mod.get_expression_factory()->create_call(r);

            for (int i = 0, j = 0, n = c_expr->get_argument_count(); i < n; ++i, ++j) {
                IArgument const *arg = promote_argument(mod, c_expr->get_argument(i));
                call->add_argument(arg);

                j = mod.promote_call_arguments(call, arg, j, rules);
            }
            call->set_type(expr->get_type());
            return call;
        }
    case IExpression::EK_LET:
        {
            IExpression_let const *l_expr = cast<IExpression_let>(expr);
            IExpression const *exp = promote_expr(mod, l_expr->get_expression());
            IExpression_let *let = mod.get_expression_factory()->create_let(exp);

            for (int i = 0, n = l_expr->get_declaration_count(); i < n; ++i) {
                // FIXME: clone the declaration
                MDL_ASSERT(!"cloning of declaration not implemented");
                IDeclaration const *decl = l_expr->get_declaration(i);
                let->add_declaration(decl);
            }
            let->set_type(expr->get_type());
            return let;
        }
    }
    MDL_ASSERT(!"Unsupported expression kind");
    return expr;
}


// Construct a Type_name AST element for an MDL type.
IType_name *create_type_name(
    IType const *type,
    IModule     *owner)
{
    Module       *mod = impl_cast<Module>(owner);
    Type_factory *tf  = mod->get_type_factory();

    type = tf->import(type);
    return construct_type_name(type, mod);
}

// Promote a given expression to the MDL version of the owner module.
IExpression const *promote_expressions_to_mdl_version(
    IModule           *owner,
    IExpression const *expr)
{
    Module *mod = impl_cast<Module>(owner);

    return promote_expr(*mod, expr);
}

// Compare two function hashes.
bool operator<(IModule::Function_hash const &a, IModule::Function_hash const &b)
{
    return memcmp(a.hash, b.hash, dimension_of(a.hash)) < 0;
}

}  // mdl
}  // mi

