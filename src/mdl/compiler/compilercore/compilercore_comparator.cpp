/******************************************************************************
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "compilercore_mdl.h"
#include "compilercore_cstring_hash.h"
#include "compilercore_errors.h"
#include "compilercore_streams.h"
#include "compilercore_tools.h"
#include "compilercore_comparator.h"
#include "compilercore_thread_context.h"
#include "compilercore_modules.h"
#include "compilercore_positions.h"
#include "compilercore_archiver.h"
#include "compilercore_file_utils.h"
#include "compilercore_mangle.h"

//#include <cstring>

//#include <mi/mdl/mdl_declarations.h>
//#include <mi/mdl/mdl_expressions.h>
//#include <mi/mdl/mdl_modules.h>
//#include <mi/mdl/mdl_statements.h>
//#include <mi/mdl/mdl_symbols.h>
//#include <mi/mdl/mdl_values.h>

namespace mi {
namespace mdl {

/// A simple LRU module cache for the comparator.
class Module_cache : public IModule_cache {
    class Cache_entry {
        friend class Module_cache;
    public:
        /// Get the module.
        mi::base::Handle<IModule const> &get() {
            return m_module;
        }

    public:
        /// Constructor.
        Cache_entry(IModule const *module)
        : m_module(module, mi::base::DUP_INTERFACE)
        , m_prev(NULL)
        , m_next(NULL)
        {
        }

    private:
        mi::base::Handle<IModule const> m_module;

        /// Points to the previous entry.
        Cache_entry *m_prev;

        /// Points to the next entry.
        Cache_entry *m_next;
    };

public:
    /// Create a IModule_cache_lookup_handle for this IModule_cache implementation.
    IModule_cache_lookup_handle *create_lookup_handle() const MDL_FINAL {
        // we do not support parallel loading in the comparator yet
        return NULL;
    }

    /// Free a handle created by create_lookup_handle().
    void free_lookup_handle(
        IModule_cache_lookup_handle *handle) const MDL_FINAL {
        // we do not support parallel loading in the comparator yet
    }

    /// Lookup a module.
    ///
    /// \param absname      the absolute name of a MDL module as returns my the module
    ///                     resolver
    /// \param handle       a handle used throughout the loading process of a model or NULL in case
    ///                     the goal is to just check if a module is loaded.
    ///
    /// \return  If this module is already known, return it, otherwise NULL.
    ///
    /// \note  The module must be returned with increased reference count.
    IModule const *lookup(
        char const                  *absname,
        IModule_cache_lookup_handle *handle) const MDL_FINAL;

    /// Get the module loading callback
    IModule_loaded_callback *get_module_loading_callback() const MDL_FINAL {
        // we do not support parallel loading in the comparator yet
        return NULL;
    }

    /// Enters a module into the cache.
    void enter(IModule const *module);

private:
    /// Find a cache entry.
    ///
    /// \param absname  the absolute name of a MDL module as returns my the module resolver
    ///
    /// \return  If this module is already known, return it, otherwise NULL.
    ///
    /// \note  The cache entry.
    Cache_entry *find(char const *absname) const;

    /// frees a list.
    void free_list(Cache_entry *list);

    /// Remove from active list.
    void remove_active(Cache_entry *p);

    /// Insert at head.
    void insert_active(Cache_entry *p);

public:
    /// Constructor.
    Module_cache(
        IAllocator *alloc,
        size_t     max_n_entries,
        char const *cache_name)
    : m_builder(alloc)
    , m_head(NULL)
    , m_last(NULL)
    , m_free(NULL)
    , m_n_entries(0u)
    , m_max_n_entries(max_n_entries)
    , m_cache_name(cache_name)
    {
    }

    /// Destructor.
    ~Module_cache() {
        free_list(m_head);
        free_list(m_free);
    }

private:
    /// The builder.
    Allocator_builder m_builder;

    /// The Head.
    Cache_entry *m_head;

    /// The last entry in the active list.
    Cache_entry *m_last;

    /// The free list.
    Cache_entry *m_free;

    /// Number of entries.
    size_t m_n_entries;

    /// Maximum number of entries.
    size_t const m_max_n_entries;

    /// For debugging: name of the cache.
    char const * const m_cache_name;
};

/// Base class for all comparators.
class Comparator_base
{
protected:
    typedef ptr_hash_map<char const, size_t, cstring_hash, cstring_equal_to>::Type ID_map;

protected:
    /// Constructor.
    ///
    /// \param alloc  the allocator
    /// \param ctx    the thread context to be used
    Comparator_base(
        IAllocator     *alloc,
        Thread_context &ctx)
    : m_alloc(alloc)
    , m_arena(alloc)
    , m_ctx(ctx)
    , m_last_msg_idx(0)
    , zero(0, 0, 0, 0)
    , m_id_map(0, ID_map::hasher(), ID_map::key_equal(), alloc)
    , m_msgs(alloc, "<owner>")
    {
        fill_id_table(ctx);
    }

protected:
    /// Register a module in the error table.
    size_t associate_file_name(char const *fname);

    /// Register a module in the error table.
    size_t associate_file_name(string const &fname);

    /// Fill the ID table from the file names already known by the context's message list.
    ///
    /// \param ctx  the context
    void fill_id_table(Thread_context &ctx);

    /// Get the message id for a given file name.
    size_t get_id_for_fname(char const *fname);

    /// Creates a new error.
    ///
    /// \param owner   the owner file name (where the error is reported)
    /// \param code    the error code
    /// \param loc     the location of the error
    /// \param params  additional parameters
    void error(
        char const         *owner,
        int                code,
        Err_location const &loc,
        Error_params const &params);

    /// Creates a new warning.
    ///
    /// \param owner   the owner file name (where the error is reported)
    /// \param code    the error code
    /// \param loc     the location of the error
    /// \param params  additional parameters
    void warning(
        char const         *owner,
        int                code,
        Err_location const &loc,
        Error_params const &params);

    /// Adds a new note to the previous message.
    ///
    /// \param owner   the owner file name (where the error is reported)
    /// \param code    the error code
    /// \param loc     the location of the error
    /// \param params  additional parameters
    void add_note(
        char const         *owner,
        int                code,
        Err_location const &loc,
        Error_params const &params);

    /// Enter a module and all of its imports into a cache.
    ///
    /// \param cache   a module cache
    /// \param module  the module to enter
    void cache_module(
        Module_cache  &cache,
        IModule const *module);

    /// Append all messages from the context to the current message block.
    void append_ctx_messages();

public:
    /// Get the allocator.
    IAllocator *get_allocator() { return m_alloc;  }

protected:
    /// The allocator.
    IAllocator *m_alloc;

    /// Arena for strings.
    Memory_arena m_arena;

    /// The thread context to be used.
    Thread_context &m_ctx;

    /// Last message index.
    int m_last_msg_idx;

    /// A zero position.
    Position_impl const zero;

    /// Map modules to IDs.
    ID_map m_id_map;

    /// Acuumulated messages;
    Messages_impl m_msgs;
};

/// Helper class to compare two modules.
///
/// Note that this helper class itself does not enforce that both modules have the same name.
class Comparator : public Comparator_base
{
    typedef Comparator_base Base;

public:
    /// Compare the two modules.
    void compare_modules();

private:
    /// Compare two types.
    ///
    /// \param ctx      current thread context
    /// \param def      the definition of the original type
    /// \param symB     the symbol of the type (in the new module) if any
    /// \param symtabA  the symbol table of the original module
    /// \param symtabB  the symbol table of the new module
    /// \param deftabB  the definition table of the new module
    void compare_type(
        Definition const       *def,
        ISymbol const          *symB,
        Symbol_table const     &symtabA,
        Symbol_table const     &symtabB,
        Definition_table const &deftabB);

    /// Compare two constants.
    ///
    /// \param defA     the definition of the constant from the original module
    /// \param symB     the symbol of the constant (in the new module) if any
    /// \param deftabB  the definition table of the new module
    void compare_constant(
        Definition const       *defA,
        ISymbol const          *symB,
        Definition_table const &deftabB);

    /// Compare two functions or annotations.
    ///
    /// \param defA     the definition of the function from the original module
    /// \param symB     the symbol of the function (in the new module) if any
    /// \param deftabB  the definition table of the new module
    void compare_function_or_annotation(
        Definition const       *defA,
        ISymbol const          *symB,
        Definition_table const &deftabB);

    /// Find a global definition of given kind for given symbol.
    ///
    /// \param def_kind  the definition kind
    /// \param sym       the symbol
    /// \param deftab    the definition table
    ///
    /// \return the definition if one exists of the given kind
    Definition const *find_definition_of_kind(
        IDefinition::Kind      def_kind,
        ISymbol const          *sym,
        Definition_table const &deftab);

    /// Finds the best possible overload that matches the given function type.
    ///
    /// \param f_type         a function type from the original module
    /// \param def            the list head of an function or annotation overload set
    /// \param is_overloaded  true, if we expect an overload set
    ///
    /// \note This is not an overload resolution, we just allow extra parameters at the end
    ///       but do NOT check the return type.
    Definition const *find_best_overload(
        IType_function const *f_type,
        Definition const     *def,
        bool                 is_overloaded);

    /// Compare two symbols (from different modules) for equality.
    ///
    /// \param symA  the symbol from the original module
    /// \param symB  the symbol from the new module
    static bool compare_symbols(
        ISymbol const *symA,
        ISymbol const *symB);

    /// Compare two types (from different modules) for equality.
    ///
    /// \param typeA  the type from the original module
    /// \param typeB  the type from the new module
    static bool compare_types(
        IType const *typeA,
        IType const *typeB);

    /// Compare two values (from different modules) for equality.
    ///
    /// \param valueA  the value from the original module
    /// \param valueB  the value from the new module
    static bool compare_values(
        IValue const *valueA,
        IValue const *valueB);

    /// Compare two expressions (from different modules) for syntactical equality.
    ///
    /// \param exprA  the expression from the original module
    /// \param exprB  the expression from the new module
    static bool compare_expressions(
        IExpression const *exprA,
        IExpression const *exprB);

    /// Compare two type names (from different modules) for syntactical equality.
    ///
    /// \param nameA  the type name from the original module
    /// \param nameB  the type name from the new module
    static bool compare_type_names(
        IType_name const *nameA,
        IType_name const *nameB);

    /// Compare two definitions (from different modules) for syntactical equality.
    ///
    /// \param defA  the definition from the original module
    /// \param defB  the definition from the new module
    static bool compare_definitions(
        Definition const *defA,
        Definition const *defB);

    /// Compare call arguments (from different modules) for syntactical equality.
    ///
    /// \param argA  the argument from the original module
    /// \param argB  the argument from the new module
    static bool compare_arguments(
        IArgument const *argA,
        IArgument const *argB);

    /// Compare two struct types.
    ///
    /// \param defA     the definition of the struct type in the original module
    /// \param defB     the definition of the struct type in the new module
    /// \param symtabA  the symbol table of the original module
    /// \param symtabB  the symbol table of the new module
    void compare_struct(
        Definition const   *defA,
        Definition const   *defB,
        Symbol_table const &symtabA,
        Symbol_table const &symtabB);

    /// Compare two enum types.
    ///
    /// \param defA     the definition of the enum type in the original module
    /// \param defB     the definition of the enum type in the new module
    /// \param symtabA  the symbol table of the original module
    /// \param symtabB  the symbol table of the new module
    void compare_enum(
        Definition const   *defA,
        Definition const   *defB,
        Symbol_table const &symtabA,
        Symbol_table const &symtabB);

    /// Prints a signature to the printer.
    ///
    /// \param def  the signature to print
    void print_signature(Definition const *def);

public:
    /// Constructor.
    ///
    /// \param alloc     the allocator
    /// \param compiler  the MDL compiler
    /// \param modA      the original module while comparing
    /// \param modB      the new module while comparing
    /// \param ctx       the current thread context
    /// \param cb        the event callback if any
    Comparator(
        IAllocator            *alloc,
        IMDL                  *compiler,
        Module const          *modA,
        Module const          *modB,
        Thread_context        &ctx,
        IMDL_comparator_event *cb)
    : Base(alloc, ctx)
    , m_modA(modA)
    , m_modB(modB)
    , m_fnameA("<No file source>")
    , m_fnameB("<No file source>")
    , m_cb(cb)
    , m_os()
    , m_printer()
    {
        if (char const *fnameA = modA->get_filename())
            m_fnameA = fnameA;
        if (char const *fnameB = modB->get_filename())
            m_fnameB = fnameB;

        associate_file_name(m_fnameA);
        associate_file_name(m_fnameB);

        if (cb != NULL) {
            m_os = Allocator_builder(alloc).create<String_output_stream>(alloc);
            m_printer = compiler->create_printer(m_os.get());
        }
    }

private:
    /// The original module while comparing.
    Module const *const m_modA;

    /// The new module while comparing.
    Module const *const m_modB;

    /// Non-NULL filename of modA.
    char const *m_fnameA;

    /// Non-NULL filename of modB.
    char const *m_fnameB;

    /// If Non-NULL, the event callback.
    IMDL_comparator_event *m_cb;

    /// If non-NULL, a string output stream.
    mi::base::Handle<String_output_stream> m_os;

    /// If non-NULL, a printer.
    mi::base::Handle<IPrinter> m_printer;
};

/// Helper class to compare two archives.
class Archive_comparator : public Comparator_base
{
    typedef Comparator_base Base;

public:
    /// Compare the two archives.
    void compare_archives();

private:
    /// Replace the archive prefix of the given file by another archive prefix given.
    string replace_archive_prefix(
        char const   *f_name,
        string const &archive_fnameA,
        string const &archive_fnameB);

public:
    /// Constructor.
    ///
    /// \param alloc     the allocator
    /// \param modA      the original module while comparing
    /// \param modB      the new module while comparing
    /// \param ctx       the current thread context
    Archive_comparator(
        IAllocator            *alloc,
        MDL                   *compiler,
        char const            *archiveA,
        char const            *archiveB,
        Thread_context        &ctx,
        IMDL_search_path      *sp,
        IMDL_comparator_event *cb)
    : Base(alloc, ctx)
    , m_compiler(compiler, mi::base::DUP_INTERFACE)
    , m_fnameA(archiveA, alloc)
    , m_fnameB(archiveB, alloc)
    , m_repl_sp(sp)
    , m_cb(cb)
    , m_cache_A(alloc, 8, "A")
    , m_cache_B(alloc, 8, "B")
    {
        m_fnameA = convert_slashes_to_os_separators(m_fnameA);
        m_fnameB = convert_slashes_to_os_separators(m_fnameB);

        associate_file_name(m_fnameA);
        associate_file_name(m_fnameB);

        if (m_repl_sp == NULL)
            m_repl_sp = compiler->get_search_path().get();
    }

private:
    /// The MDL compiler to be used.
    mi::base::Handle<MDL> m_compiler;

    /// Filename of archiveA.
    string m_fnameA;

    /// Filename of archiveB.
    string m_fnameB;

    /// The search path for replacement archives.
    IMDL_search_path *m_repl_sp;

    /// The event callback if any.
    IMDL_comparator_event *m_cb;

    /// Module cache for the A archive.
    Module_cache m_cache_A;

    /// Module cache for the B archive.
    Module_cache m_cache_B;
};

// ---------------------------- Module_cache ----------------------------

// Lookup a module.
IModule const *Module_cache::lookup(
    char const *absname,
    IModule_cache_lookup_handle *handle) const
{
    if (Cache_entry *p = find(absname)) {
        // move to front
        if (p != m_head) {
            Module_cache *self = const_cast<Module_cache *>(this);

            self->remove_active(p);
            self->insert_active(p);
        }

        mi::base::Handle<IModule const> &m = p->get();
        m->retain();

        return m.get();
    }
    return NULL;
}

// Enters a module into the cache.
void Module_cache::enter(IModule const *module)
{
    Module const *m = impl_cast<Module>(module);

    if (m->is_compiler_owned()) {
        // do not cache compiler owned modules
        return;
    }

    Cache_entry *p = find(module->get_name());
    if (p != NULL)
        return;

    if (m_n_entries >= m_max_n_entries) {
        // remove the last one
        Cache_entry *p = m_last;

        if (p != NULL) {
            remove_active(p);

            // put to free list
            p->m_module.reset();

            p->m_next = m_free;
            m_free = p;
        }
    }

    // get a free
    p = m_free;
    if (p != NULL) {
        m_free = p->m_next;

        new(p) Cache_entry(module);
    } else {
        p = m_builder.create<Cache_entry>(module);
    }

    insert_active(p);
}

// Find a cache entry.
Module_cache::Cache_entry *Module_cache::find(char const *absname) const
{
    for (Cache_entry *p = m_head; p != NULL; p = p->m_next) {
        mi::base::Handle<IModule const> &m    = p->get();
        char const                      *name = m->get_name();

        if (strcmp(name, absname) == 0)
            return p;
    }
    return NULL;
}

// frees a list.
void Module_cache::free_list(Cache_entry *list) {
    for (Cache_entry *n = NULL, *p = list; p != NULL; p = n) {
        n = p->m_next;
        m_builder.destroy(p);
    }
    list = NULL;
}

// Remove from active list.
void Module_cache::remove_active(Cache_entry *p)
{
    if (p == m_last)
        m_last = p->m_prev;

    if (m_head == p)
        m_head = p->m_next;

    if (p->m_prev != NULL)
        p->m_prev->m_next = p->m_next;
    if (p->m_next != NULL)
        p->m_next->m_prev = p->m_prev;

    p->m_next = p->m_prev = NULL;

    --m_n_entries;
}

// Insert at head.
void Module_cache::insert_active(Cache_entry *p)
{
    if (m_last == NULL)
        m_last = p;
    if (m_head != NULL)
        m_head->m_prev = p;
    p->m_next = m_head;
    p->m_prev = NULL;
    m_head = p;

    ++m_n_entries;
}

// ---------------------------- MDL_comparator ----------------------------

// Constructor.
MDL_comparator::MDL_comparator(
    IAllocator *alloc,
    MDL        *compiler)
: Base(alloc)
, m_compiler(compiler, mi::base::DUP_INTERFACE)
, m_repl_sp()
, m_options(alloc)
, m_cb(NULL)
{
}

// Load an "original" module with a given name.
IModule const *MDL_comparator::load_module(
    IThread_context *context,
    char const      *module_name)
{
    if (context == NULL)
        return NULL;
    return m_compiler->load_module(context, module_name, /*cache=*/NULL);
}

// Load a "replacement" module with a given name.
IModule const *MDL_comparator::load_replacement_module(
    IThread_context *context,
    char const      *module_name,
    char const      *file_name)
{
    if (context == NULL)
        return NULL;

    Thread_context &tc = *impl_cast<Thread_context>(context);

    mi::base::Handle<IMDL_search_path> old_sp = m_compiler->get_search_path();

    tc.set_module_replacement_path(module_name, file_name);
    if (m_repl_sp.is_valid_interface()) {
        m_repl_sp->retain();
        m_compiler->install_search_path(m_repl_sp.get());
    }

    IModule const *res = m_compiler->load_module(context, module_name, /*cache=*/NULL);

    old_sp->retain();
    m_compiler->install_search_path(old_sp.get());
    return res;
}

// Compare two modules and evaluate, if modB can be used as a replacement for modA.
void MDL_comparator::compare_modules(
    IThread_context *pCtx,
    IModule const   *imodA,
    IModule const   *imodB)
{
    IThread_context *ctx = pCtx;
    mi::base::Handle<IThread_context> tmp;
    if (ctx == NULL) {
        tmp = mi::base::make_handle(m_compiler->create_thread_context());
        ctx = tmp.get();
    }

    Thread_context &tc = *impl_cast<Thread_context>(ctx);
    tc.clear_messages();

    Comparator cmp(
        get_allocator(),
        m_compiler.get(),
        impl_cast<Module>(imodA),
        impl_cast<Module>(imodB),
        tc,
        m_cb);

    cmp.compare_modules();
}

// Compare two archives and evaluate, if archivB can be used as a replacement for archivA.
void MDL_comparator::compare_archives(
    IThread_context *pCtx,
    char const      *archivA,
    char const      *archivB)
{
    IThread_context *ctx = pCtx;
    mi::base::Handle<IThread_context> tmp;
    if (ctx == NULL) {
        tmp = mi::base::make_handle(m_compiler->create_thread_context());
        ctx = tmp.get();
    }

    Thread_context &tc = *impl_cast<Thread_context>(ctx);
    tc.clear_messages();

    Archive_comparator comparator(
        get_allocator(), m_compiler.get(), archivA, archivB, tc, m_repl_sp.get(), m_cb);

    comparator.compare_archives();
}

// Install a MDL search path helper for all replacement modules/archives.
void MDL_comparator::install_replacement_search_path(IMDL_search_path *search_path)
{
    m_repl_sp = mi::base::make_handle_dup(search_path);
}

// Access options.
Options &MDL_comparator::access_options()
{
    return m_options;
}

// Set an event callback.
void MDL_comparator::set_event_cb(IMDL_comparator_event *cb)
{
    m_cb = cb;
}

// ------------------------------------------------------------------------------

// Register a module in the error table.
size_t Comparator_base::associate_file_name(char const *fname)
{
    ID_map::const_iterator it = m_id_map.find(fname);
    if (it != m_id_map.end()) {
        return it->second;
    }

    fname = Arena_strdup(m_arena, fname);

    Messages_impl &msgs = m_msgs;
    size_t        ID = msgs.register_fname(fname);

    m_id_map[fname] = ID;

    return ID;
}

// Register a module in the error table.
size_t Comparator_base::associate_file_name(string const &fname)
{
    return associate_file_name(fname.c_str());
}

// Fill the ID table from the file names already known by the context's message list
void Comparator_base::fill_id_table(Thread_context &ctx)
{
    Messages_impl const &msgs = m_msgs;
    for (size_t i = 0, n = msgs.get_fname_count(); i < n; ++i) {
        char const *fname = msgs.get_fname(i);

        if (fname != NULL)
            m_id_map[fname] = i;
    }
}

// Get the message id for a given file name.
size_t Comparator_base::get_id_for_fname(char const *fname)
{
    size_t file_id = 0;
    if (fname != NULL) {
        ID_map::const_iterator it = m_id_map.find(fname);
        if (it != m_id_map.end())
            file_id = it->second;
    }
    return file_id;
}

// Creates a new error.
void Comparator_base::error(
    char const         *owner,
    int                code,
    Err_location const &loc,
    Error_params const &params)
{
    Position const *pos = loc.get_position();
    if (pos == NULL)
        pos = &zero;

    size_t file_id = get_id_for_fname(owner);

    Messages_impl &msg_list = m_msgs;

    string msg(msg_list.format_msg(code, MDL_comparator::MESSAGE_CLASS, params));
    m_last_msg_idx = msg_list.add_error_message(
        code, MDL_comparator::MESSAGE_CLASS, file_id, pos, msg.c_str());
}

// Creates a new warning.
void Comparator_base::warning(
    char const         *owner,
    int                code,
    Err_location const &loc,
    Error_params const &params)
{
    Position const *pos = loc.get_position();
    if (pos == NULL)
        pos = &zero;

    size_t file_id = get_id_for_fname(owner);

    Messages_impl &msg_list = m_msgs;

    string msg(msg_list.format_msg(code, MDL_comparator::MESSAGE_CLASS, params));
    m_last_msg_idx = msg_list.add_warning_message(
        code, MDL_comparator::MESSAGE_CLASS, file_id, pos, msg.c_str());
}

// Adds a new note to the previous message.
void Comparator_base::add_note(
    char const         *owner,
    int                code,
    Err_location const &loc,
    Error_params const &params)
{
    Position const *pos = loc.get_position();
    if (pos == NULL)
        pos = &zero;

    size_t file_id = get_id_for_fname(owner);

    Messages_impl &msg_list = m_msgs;

    string msg(msg_list.format_msg(code, MDL_comparator::MESSAGE_CLASS, params));
    msg_list.add_note(
        m_last_msg_idx,
        IMessage::MS_INFO,
        code,
        MDL_comparator::MESSAGE_CLASS,
        file_id,
        pos,
        msg.c_str());
}

// Enter a module and all of its imports into a cache.
void Comparator_base::cache_module(Module_cache &cache, IModule const *module)
{
    // cache imports transitively
    for (int i = 0, n = module->get_import_count(); i < n; ++i) {
        mi::base::Handle<IModule const> imp(module->get_import(i));

        cache_module(cache, imp.get());

        cache.enter(imp.get());
    }
    cache.enter(module);
}

// Append all messages from the context to the current message block.
void Comparator_base::append_ctx_messages()
{
    m_msgs.copy_messages(m_ctx.access_messages_impl());
}

// ------------------------------------------------------------------------------

// Find a global definition of given kind for given symbol.
Definition const *Comparator::find_definition_of_kind(
    IDefinition::Kind      def_kind,
    ISymbol const          *sym,
    Definition_table const &deftab)
{
    Definition const *def = NULL;

    if (sym != NULL) {
        Scope const *global = deftab.get_global_scope();
        def = global->find_definition_in_scope(sym);

        if (def != NULL && def->get_kind() != def_kind)
            def = NULL;
    }
    return def;
}

// Compare two modules.
void Comparator::compare_modules()
{
    IMDL::MDL_version verA = m_modA->get_mdl_version();
    IMDL::MDL_version verB = m_modB->get_mdl_version();

    if (verA != verB) {
        warning(
            NULL,
            DIFFERENT_MDL_VERSIONS,
            zero,
            Error_params(m_alloc).add_mdl_version(verA).add_mdl_version(verB)
        );
    }

    Symbol_table const     &symtabA = m_modA->get_symbol_table();

    Definition_table const &deftabB = m_modB->get_definition_table();
    Symbol_table const     &symtabB = m_modB->get_symbol_table();

    for (int i = 0, n = m_modA->get_exported_definition_count(); i < n; ++i) {
        Definition const *defA = m_modA->get_exported_definition(i);
        ISymbol const    *symA = defA->get_sym();

        if (m_cb != NULL) {
            print_signature(defA);
            string sig = m_os->get_buffer();
            m_os->clear();

            m_cb->fire_event(IMDL_comparator_event::EV_COMPARING_EXPORT, sig.c_str());
        }

        // find matching symbol in B
        ISymbol const    *symB = symtabB.lookup_symbol(symA->get_name());

        switch (defA->get_kind()) {
        case IDefinition::DK_ERROR:
            // should never be exported.
            MDL_ASSERT(!"unexpected exporported error definition");
            break;
        case IDefinition::DK_CONSTANT:
            compare_constant(defA, symB, deftabB);
            break;
        case IDefinition::DK_ENUM_VALUE:
            break;
        case IDefinition::DK_TYPE:
            compare_type(defA, symB, symtabA, symtabB, deftabB);
            break;
        case IDefinition::DK_FUNCTION:
        case IDefinition::DK_ANNOTATION:
            compare_function_or_annotation(defA, symB, deftabB);
            break;
        case IDefinition::DK_VARIABLE:
        case IDefinition::DK_MEMBER:
        case IDefinition::DK_CONSTRUCTOR:
        case IDefinition::DK_PARAMETER:
        case IDefinition::DK_ARRAY_SIZE:
        case IDefinition::DK_OPERATOR:
        case IDefinition::DK_NAMESPACE:
            break;
        }
    }

    // copy messages to ctx
    m_ctx.access_messages_impl().copy_messages(m_msgs);
}

// Compare two types.
void Comparator::compare_type(
    Definition const       *defA,
    ISymbol const          *symB,
    Symbol_table const     &symtabA,
    Symbol_table const     &symtabB,
    Definition_table const &deftabB)
{
    Definition const *defB = find_definition_of_kind(IDefinition::DK_TYPE, symB, deftabB);

    if (defB == NULL) {
        error(
            m_fnameA,
            TYPE_DOES_NOT_EXISTS,
            defA,
            Error_params(get_allocator())
                .add_signature(defA)
                .add(m_fnameB)
        );
        return;
    }

    IType const *typeA = defA->get_type();
    IType const *typeB = defB->get_type();

    IType::Kind tp_kind = typeA->get_kind();
    if (tp_kind != typeB->get_kind()) {
        error(
            m_fnameA,
            TYPES_DIFFERENT,
            defA,
            Error_params(get_allocator())
                .add_signature(defA)
        );
        add_note(
            m_fnameB,
            OTHER_DEFINED_AT,
            defB,
            Error_params(get_allocator()).add_signature(defB)
        );
        return;
    }

    if (tp_kind == IType::TK_STRUCT) {
        compare_struct(defA, defB, symtabA, symtabB);
    } else if (tp_kind == IType::TK_ENUM) {
        compare_enum(defA, defB, symtabA, symtabB);
    }
}

// Compare two symbols (from different modules) for equality.
bool Comparator::compare_symbols(
    ISymbol const *symA,
    ISymbol const *symB)
{
    return strcmp(symA->get_name(), symB->get_name()) == 0;
}

// Compare two types (from different modules) for equality.
bool Comparator::compare_types(
    IType const *typeA,
    IType const *typeB)
{
    IType::Modifiers modA = typeA->get_type_modifiers();
    IType::Modifiers modB = typeB->get_type_modifiers();

    if (modA != modB)
        return false;

    typeA = typeA->skip_type_alias();
    typeB = typeB->skip_type_alias();

    IType::Kind kind = typeA->get_kind();
    if (kind != typeB->get_kind())
        return false;

    switch (kind) {
    case IType::TK_ALIAS:
        MDL_ASSERT(!"unexpected nested alias type");
        return false;

    case IType::TK_BOOL:
    case IType::TK_INT:
    case IType::TK_FLOAT:
    case IType::TK_DOUBLE:
    case IType::TK_STRING:
    case IType::TK_COLOR:
    case IType::TK_LIGHT_PROFILE:
    case IType::TK_BSDF:
    case IType::TK_HAIR_BSDF:
    case IType::TK_EDF:
    case IType::TK_VDF:
    case IType::TK_BSDF_MEASUREMENT:
    case IType::TK_INCOMPLETE:
    case IType::TK_ERROR:
        return true;
    case IType::TK_ENUM:
        {
            IType_enum const *enumA = cast<IType_enum>(typeA);
            IType_enum const *enumB = cast<IType_enum>(typeB);

            int n_values = enumA->get_value_count();
            if (n_values != enumB->get_value_count())
                return false;

            for (int i = 0; i < n_values; ++i) {
                ISymbol const *symA;
                int           codeA;

                enumA->get_value(i, symA, codeA);

                ISymbol const *symB;
                int           codeB;

                enumB->get_value(i, symB, codeB);

                // all enum values must match name and value
                if (strcmp(symA->get_name(), symB->get_name()) != 0)
                    return false;
                if (codeA != codeB)
                    return false;
            }
            return true;
        }
    case IType::TK_VECTOR:
        {
            IType_vector const *vectorA = cast<IType_vector>(typeA);
            IType_vector const *vectorB = cast<IType_vector>(typeB);

            if (vectorA->get_size() != vectorB->get_size())
                return false;

            return compare_types(vectorA->get_element_type(), vectorB->get_element_type());
        }
    case IType::TK_MATRIX:
        {
            IType_matrix const *matrixA = cast<IType_matrix>(typeA);
            IType_matrix const *matrixB = cast<IType_matrix>(typeB);

            if (matrixA->get_columns() != matrixB->get_columns())
                return false;

            return compare_types(matrixA->get_element_type(), matrixB->get_element_type());
        }
    case IType::TK_ARRAY:
        {
            IType_array const *arA = cast<IType_array>(typeA);
            IType_array const *arB = cast<IType_array>(typeB);

            return compare_types(arA->get_element_type(), arB->get_element_type());
        }
    case IType::TK_FUNCTION:
        {
            IType_function const *funcA = cast<IType_function>(typeA);
            IType_function const *funcB = cast<IType_function>(typeB);

            IType const *retA = funcA->get_return_type();
            IType const *retB = funcB->get_return_type();

            int n_params = funcA->get_parameter_count();
            if (n_params != funcB->get_parameter_count())
                return false;

            if ((retA == NULL || retB == NULL) && retA != retB)
                return false;

            if (!compare_types(retA, retB))
                return false;

            for (int i = 0; i < n_params; ++i) {
                ISymbol const *symA;
                IType const   *paramA;

                funcA->get_parameter(i, paramA, symA);

                ISymbol const *symB;
                IType const   *paramB;

                funcB->get_parameter(i, paramB, symB);

                // we have call be name, so check names too
                if (strcmp(symA->get_name(), symB->get_name()) != 0)
                    return false;

                if (!compare_types(paramA, paramB))
                    return false;
            }
            return true;
        }
    case IType::TK_STRUCT:
        {
            IType_struct const *structA = cast<IType_struct>(typeA);
            IType_struct const *structB = cast<IType_struct>(typeB);

            if (structA->get_field_count() != structB->get_field_count())
                return false;

            for (int i = 0, n = structA->get_field_count(); i < n; ++i) {
                ISymbol const *symA;
                IType const   *memA;

                structA->get_field(i, memA, symA);

                ISymbol const *symB;
                IType const   *memB;

                structB->get_field(i, memB, symB);

                if (!compare_types(memA, memB))
                    return false;
            }
            return true;
        }
    case IType::TK_TEXTURE:
        {
            IType_texture const *txA = cast<IType_texture>(typeA);
            IType_texture const *txB = cast<IType_texture>(typeB);

            // coordinate type depends on the shape
            return txA->get_shape() == txB->get_shape();
        }
    }
    MDL_ASSERT(!"unsupported type kind");
    return false;
}

// Compare two constants.
void Comparator::compare_constant(
    Definition const       *defA,
    ISymbol const          *symB,
    Definition_table const &deftabB)
{
    Definition const *defB = find_definition_of_kind(IDefinition::DK_CONSTANT, symB, deftabB);

    if (defB == NULL) {
        error(
            m_fnameA,
            CONSTANT_DOES_NOT_EXISTS,
            defA,
            Error_params(get_allocator())
            .add_signature(defA)
            .add(m_fnameB)
        );
        return;
    }

    IType const *typeA = defA->get_type();
    IType const *typeB = defB->get_type();

    IType::Kind tp_kind = typeA->get_kind();
    if (tp_kind != typeB->get_kind()) {
        error(
            m_fnameA,
            CONSTANT_OF_DIFFERENT_TYPE,
            defA,
            Error_params(get_allocator())
                .add_signature(defA)
                .add(typeA)
                .add(typeB)
        );
        add_note(
            m_fnameB,
            OTHER_DEFINED_AT,
            defB,
            Error_params(get_allocator()).add_signature(defB)
        );
        return;
    }

    IValue const *initA = defA->get_constant_value();
    IValue const *initB = defB->get_constant_value();

    if (!compare_values(initA, initB)) {
        error(
            m_fnameA,
            CONSTANT_OF_DIFFERENT_VALUE,
            defA,
            Error_params(get_allocator())
            .add_signature(defA)
            .add(initA)
            .add(initB)
        );
        add_note(
            m_fnameB,
            OTHER_DEFINED_AT,
            defB,
            Error_params(get_allocator()).add_signature(defB)
        );
        return;
    }
}

// Finds the best possible overload that matches the given function type.
Definition const *Comparator::find_best_overload(
    IType_function const *f_type,
    Definition const     *last_def,
    bool                 is_overloaded)
{
    if (!is_overloaded) {
        if (last_def->get_prev_def() == NULL) {
            // there is NO overload set, just one function, and this is expected, return
            // it here
            return last_def;
        }
    }

    // either we expect an overload set or we found one, find the best match

    typedef ptr_hash_set<Definition const>::Type Cand_set;

    Cand_set candidates(0, Cand_set::hasher(), Cand_set::key_equal(), get_allocator());

    int n_paramsA = f_type->get_parameter_count();

    if (n_paramsA == 0) {
        // a rare case in MDL, a parameterless function

        for (Definition const *def = last_def; def != NULL; def = def->get_prev_def()) {
            IType_function const *f_typeB = cast<IType_function>(def->get_type());

            int n_paramsB = f_typeB->get_parameter_count();

            if (n_paramsB == 0) {
                // we found our candidate, because we have no return type overloads
                return def;
            }
            if (def->get_default_param_initializer(0) != NULL) {
                // we have more the zero parameters, but the first one has a default, hence all
                // must have one. This function can be called parameterless.
                candidates.insert(def);
            }
        }

        // next try: check if we find one definition with ALL parameters have defaults
        if (candidates.empty()) {
            // probably removed
            return NULL;
        }

        IType const *ret_type = f_type->get_return_type();
        if (ret_type == NULL) {
            // we are looking for an annotation, because ALL annotations have NO return type,
            // return NULL if more that one annotation exists, else the only existing one
            if (candidates.size() == 1)
                return *candidates.begin();
            return NULL;
        }

        // select the candidate with matching return type
        Definition const *best_cand = NULL;
        for (Cand_set::const_iterator it(candidates.begin()), end(candidates.end());
            it != end;
            ++it)
        {
            Definition const     *cand    = *it;
            IType_function const *f_typeB = cast<IType_function>(cand->get_type());

            if (compare_types(ret_type, f_typeB->get_return_type())) {
                if (best_cand != NULL) {
                    // ambiguous
                    return NULL;
                }
                best_cand = cand;
            }
        }
        return best_cand;
    }

    // We have at least one parameter here ...

    // first, assume we find a good match, i.e. all parameters still there, just some was
    // added at the end
    for (Definition const *def = last_def; def != NULL; def = def->get_prev_def()) {
        IType_function const *f_typeB = cast<IType_function>(def->get_type());

        int n_paramsB = f_typeB->get_parameter_count();
        if (n_paramsB < n_paramsA)
            continue;

        bool have_match = true;
        for (int i = 0; i < n_paramsA; ++i) {
            IType const   *param_typeA = NULL;
            ISymbol const *param_symA = NULL;

            IType const   *param_typeB = NULL;
            ISymbol const *param_symB = NULL;

            f_type->get_parameter(i, param_typeA, param_symA);
            f_typeB->get_parameter(i, param_typeB, param_symB);

            if (!compare_symbols(param_symA, param_symB)) {
                have_match = false;
                break;
            }

            if (!compare_types(param_typeA, param_typeB)) {
                have_match = false;
                break;
            }
        }
        if (have_match) {
            candidates.insert(def);
        }
    }

    if (candidates.size() == 1) {
        // only one candidate
        return *candidates.begin();
    }

    // check return type if more that one candidate
    Definition const *best_cand = NULL;
    IType const *ret_type = f_type->get_return_type();

    for (Cand_set::const_iterator it(candidates.begin()), end(candidates.end()); it != end; ++it) {
        Definition const     *cand    = *it;
        IType_function const *f_typeB = cast<IType_function>(cand->get_type());

        if (compare_types(ret_type, f_typeB->get_return_type())) {
            if (best_cand != NULL) {
                // ambiguous
                best_cand = NULL;
                break;
            }
            best_cand = cand;
        }
    }
    if (best_cand != NULL)
        return best_cand;

    // use heuristics to find a match, NYI
    return NULL;
}

// Compare two functions or annotations.
void Comparator::compare_function_or_annotation(
    Definition const       *defA,
    ISymbol const          *symB,
    Definition_table const &deftabB)
{
    bool is_anno = defA->get_kind() == IDefinition::DK_ANNOTATION;

    IType_function const *typeA = cast<IType_function>(defA->get_type());
    Definition const *defB = find_definition_of_kind(
        is_anno ? IDefinition::DK_ANNOTATION : IDefinition::DK_FUNCTION,
        symB,
        deftabB);

    if (defB != NULL) {
        defB = find_best_overload(typeA, defB, /*is_overloaded=*/defA->get_prev_def() != NULL);
    }

    if (defB == NULL) {
        error(
            m_fnameA,
            is_anno ? ANNOTATION_DOES_NOT_EXISTS : FUNCTION_DOES_NOT_EXISTS,
            defA,
            Error_params(get_allocator())
            .add_signature(defA)
            .add(m_fnameB)
        );
        return;
    }

    IType_function const *typeB = cast<IType_function>(defB->get_type());

    if (!is_anno && !compare_types(typeA->get_return_type(), typeB->get_return_type())) {
        error(
            m_fnameA,
            FUNCTION_RET_TYPE_DIFFERENT,
            defA,
            Error_params(get_allocator())
                .add_signature_no_rt(defA)
                .add(typeA->get_return_type())
                .add(typeB->get_return_type())
        );
        add_note(
            m_fnameB,
            OTHER_DEFINED_AT,
            defB,
            Error_params(get_allocator())
                .add_signature_no_rt(defA)
        );
    }

    // check if all defaults are still there
    int n_paramsA = typeA->get_parameter_count();
    int n_paramsB = typeB->get_parameter_count();

    if (n_paramsA > n_paramsB) {
        error(
            m_fnameA,
            is_anno ? ANNOTATION_PARAM_DELETED : FUNCTION_PARAM_DELETED,
            defA,
            Error_params(get_allocator())
                .add_signature(defA)
                .add(m_fnameB)
        );
        return;
    }

    for (int i = 0; i < n_paramsA; ++i) {
        IExpression const *initA = defA->get_default_param_initializer(i);

        if (initA != NULL) {
            IExpression const *initB = defB->get_default_param_initializer(i);

            if (initB == NULL) {
                ISymbol const *symA = NULL;
                IType const   *tpA = NULL;

                typeA->get_parameter(i, tpA, symA);

                error(
                    m_fnameA,
                    is_anno ? ANNOTATION_PARAM_DEF_ARG_DELETED : FUNCTION_PARAM_DEF_ARG_DELETED,
                    defA,
                    Error_params(get_allocator())
                        .add(symA)
                        .add_signature(defA)
                        .add(m_fnameB)
                );
            } else {
                if (!compare_expressions(initA, initB)) {
                    ISymbol const *symA = NULL;
                    IType const   *tpA = NULL;

                    typeA->get_parameter(i, tpA, symA);

                    error(
                        m_fnameA,
                        is_anno ? ANNOTATION_PARAM_DEF_ARG_CHANGED : FUNCTION_PARAM_DEF_ARG_CHANGED,
                        defA,
                        Error_params(get_allocator())
                        .add(symA)
                        .add_signature(defA)
                        .add(m_fnameB)
                    );

                }
            }
        }
    }
}

// Compare two values (from different modules) for equality.
bool Comparator::compare_values(
    IValue const *valueA,
    IValue const *valueB)
{
    IValue::Kind kind = valueA->get_kind();
    if (kind != valueB->get_kind())
        return false;

    switch (kind) {
    case IValue::VK_BAD:
        // there should be no bads here
        MDL_ASSERT(!"comparing BAD values");
        // .. but if are, all BADs are different
        return false;
    case IValue::VK_BOOL:
        return
            cast<IValue_bool>(valueA)->get_value() == cast<IValue_bool>(valueB)->get_value();
    case IValue::VK_INT:
        return
            cast<IValue_int>(valueA)->get_value() == cast<IValue_int>(valueB)->get_value();
    case IValue::VK_ENUM:
        return
            cast<IValue_enum>(valueA)->get_value() == cast<IValue_enum>(valueB)->get_value();
    case IValue::VK_FLOAT:
    case IValue::VK_DOUBLE:
        {
            IValue_FP const *fA = cast<IValue_FP>(valueA);
            IValue_FP const *fB = cast<IValue_FP>(valueB);

            // handle special values
            if (fA->get_fp_class() != fB->get_fp_class())
                return false;

            bool fin = fA->is_finite();
            if (fin != fB->is_finite())
                return false;

            if (fin) {
                if (kind == IValue_FP::VK_FLOAT) {
                    return
                        cast<IValue_float>(fA)->get_value() ==
                        cast<IValue_float>(fB)->get_value();
                } else {
                    MDL_ASSERT(kind == IValue_FP::VK_DOUBLE);
                    return
                        cast<IValue_double>(fA)->get_value() ==
                        cast<IValue_double>(fB)->get_value();
                }
            } else {
                // for non finite values we have already checked the class
                // we *could* check the NaN paylod here ...
                return true;
            }
        }
    case IValue::VK_STRING:
        return
            strcmp(
                cast<IValue_string>(valueA)->get_value(),
                cast<IValue_string>(valueB)->get_value()) == 0;
    case IValue::VK_VECTOR:
    case IValue::VK_MATRIX:
    case IValue::VK_ARRAY:
    case IValue::VK_RGB_COLOR:
    case IValue::VK_STRUCT:
        {
            IValue_compound const *compA = cast<IValue_compound>(valueA);
            IValue_compound const *compB = cast<IValue_compound>(valueB);

            int n_comp = compA->get_component_count();
            if (n_comp != compB->get_component_count())
                return false;

            for (int i = 0; i < n_comp; ++i) {
                IValue const *a = compA->get_value(i);
                IValue const *b = compB->get_value(i);

                if (!compare_values(a, b))
                    return false;
            }
            return true;
        }
    case IValue::VK_INVALID_REF:
        // all invalid refs of the save type are identical
        return compare_types(valueA->get_type(), valueB->get_type());

    case IValue::VK_TEXTURE:
    case IValue::VK_LIGHT_PROFILE:
    case IValue::VK_BSDF_MEASUREMENT:
        {
            IValue_resource const *resA = cast<IValue_resource>(valueA);
            IValue_resource const *resB = cast<IValue_resource>(valueB);

            if (resA->get_tag_value() != resB->get_tag_value())
                return false;
            if (resA->get_tag_version() != resB->get_tag_version())
                return false;
            return strcmp(resA->get_string_value(), resB->get_string_value()) == 0;
        }
    }
    MDL_ASSERT(!"unexpected value kind");
    return false;
}

// Compare two expressions (from different modules) for syntactical equality.
bool Comparator::compare_expressions(
    IExpression const *exprA,
    IExpression const *exprB)
{
    if (exprA == NULL)
        return exprB == NULL;
    if (exprB == NULL)
        return false;

    IExpression::Kind kind = exprA->get_kind();
    if (kind != exprB->get_kind())
        return false;

    switch (kind) {
    case IExpression::EK_INVALID:
        return true;
    case IExpression::EK_LITERAL:
        return compare_values(
            cast<IExpression_literal>(exprA)->get_value(),
            cast<IExpression_literal>(exprB)->get_value());
    case IExpression::EK_REFERENCE:
        {
            IExpression_reference const *refA = cast<IExpression_reference>(exprA);
            IExpression_reference const *refB = cast<IExpression_reference>(exprB);

            bool is_array_constr = refA->is_array_constructor();
            if (is_array_constr != refB->is_array_constructor())
                return false;

            if (is_array_constr) {
                return compare_type_names(refA->get_name(), refB->get_name());
            } else {
                return compare_definitions(
                    impl_cast<Definition>(refA->get_definition()),
                    impl_cast<Definition>(refB->get_definition()));
            }
        }
    case IExpression::EK_UNARY:
        {
            IExpression_unary const *unA = cast<IExpression_unary>(exprA);
            IExpression_unary const *unB = cast<IExpression_unary>(exprB);

            if (unA->get_operator() != unB->get_operator())
                return false;
            return compare_expressions(unA->get_argument(), unB->get_argument());
        }
    case IExpression::EK_BINARY:
        {
            IExpression_binary const *binA = cast<IExpression_binary>(exprA);
            IExpression_binary const *binB = cast<IExpression_binary>(exprB);

            if (binA->get_operator() != binB->get_operator())
                return false;
            if (!compare_expressions(binA->get_left_argument(), binB->get_left_argument()))
                return false;
            return compare_expressions(binA->get_right_argument(), binB->get_right_argument());
        }
    case IExpression::EK_CONDITIONAL:
        {
            IExpression_conditional const *condA = cast<IExpression_conditional>(exprA);
            IExpression_conditional const *condB = cast<IExpression_conditional>(exprB);

            if (!compare_expressions(condA->get_condition(), condB->get_condition()))
                return false;
            if (!compare_expressions(condA->get_true(), condB->get_true()))
                return false;
            return compare_expressions(condA->get_false(), condB->get_false());
        }
    case IExpression::EK_CALL:
        {
            IExpression_call const *callA = cast<IExpression_call>(exprA);
            IExpression_call const *callB = cast<IExpression_call>(exprB);

            int n_args = callA->get_argument_count();
            if (n_args != callB->get_argument_count())
                return false;

            if (!compare_expressions(callA->get_reference(), callB->get_reference()))
                return false;
            for (int i = 0; i < n_args; ++i) {
                if (!compare_arguments(callA->get_argument(i), callB->get_argument(i)))
                    return false;
            }
            return true;
        }
    case IExpression::EK_LET:
        // NYI
        return false;

    }
    MDL_ASSERT(!"unexpected expression kind");
    return false;
}

// Compare two type names (from different modules) for syntactical equality.
bool Comparator::compare_type_names(
    IType_name const *nameA,
    IType_name const *nameB)
{
    if (nameA->is_absolute() != nameB->is_absolute())
        return false;
    // NYI
    return false;
}

// Compare two definitions (from different modules) for syntactical equality.
bool Comparator::compare_definitions(
    Definition const *defA,
    Definition const *defB)
{
    IDefinition::Kind kind = defA->get_kind();
    if (kind != defB->get_kind())
        return false;

    if (strcmp(defA->get_sym()->get_name(), defB->get_sym()->get_name()) != 0)
        return false;

    switch (kind) {
    case IDefinition::DK_ERROR:
        return true;
    case IDefinition::DK_CONSTANT:
        return compare_values(defA->get_constant_value(), defB->get_constant_value());
    case IDefinition::DK_ENUM_VALUE:
    case IDefinition::DK_ANNOTATION:
    case IDefinition::DK_TYPE:
    case IDefinition::DK_FUNCTION:
    case IDefinition::DK_VARIABLE:
    case IDefinition::DK_MEMBER:
    case IDefinition::DK_CONSTRUCTOR:
    case IDefinition::DK_PARAMETER:
    case IDefinition::DK_ARRAY_SIZE:
    case IDefinition::DK_OPERATOR:
    case IDefinition::DK_NAMESPACE:
        // so far we check "syntactical", so this should be enough
        return true;
    }
    MDL_ASSERT(!"unsupported definition kind");
    return false;
}

// Compare call arguments (from different modules) for syntactical equality.
bool Comparator::compare_arguments(
    IArgument const *argA,
    IArgument const *argB)
{
    // we assume that the modules both are valid, hence we can ignore named arguments here
    return compare_expressions(argA->get_argument_expr(), argB->get_argument_expr());
}

// Compare two struct types.
void Comparator::compare_struct(
    Definition const   *defA,
    Definition const   *defB,
    Symbol_table const &symtabA,
    Symbol_table const &symtabB)
{
    Scope const *scopeA = defA->get_own_scope();
    Scope const *scopeB = defB->get_own_scope();

    typedef list<Definition const *>::Type                    Def_list;
    typedef std::pair<Definition const *, Definition const *> Def_pair;
    typedef list<Def_pair>::Type                              Def_pair_list;

    Def_pair_list changed_members(get_allocator());
    Def_list      missing_members(get_allocator());
    Def_pair_list normal_members(get_allocator());

    Definition const *elementalA = NULL;

    for (Definition const *memA = scopeA->get_first_definition_in_scope();
        memA != NULL;
        memA = memA->get_next_def_in_scope())
    {
        IDefinition::Kind kind = memA->get_kind();

        if (kind == IDefinition::DK_MEMBER) {
            ISymbol const *symA = memA->get_sym();
            ISymbol const *symB = symtabB.lookup_symbol(symA->get_name());

            // find this member in scope B
            Definition const *memB = NULL;
            if (symB != NULL) {
                memB = scopeB->find_definition_in_scope(symB);
                if (memB != NULL && memB->get_kind() != IDefinition::DK_MEMBER)
                    memB = NULL;
            }

            if (memB == NULL) {
                // we found a missing member
                missing_members.push_back(memA);
            } else if (!compare_types(memA->get_type(), memB->get_type())) {
                changed_members.push_back(Def_pair(memA, memB));
            } else {
                normal_members.push_back(Def_pair(memA, memB));
            }
        } else if (kind == IDefinition::DK_CONSTRUCTOR) {
            if (memA->get_semantics() == IDefinition::DS_ELEM_CONSTRUCTOR) {
                // found the elemental constructor
                elementalA = memA;
            }
        }
    }


    Def_list added_members(get_allocator());

    Definition const *elementalB = NULL;

    for (Definition const *memB = scopeB->get_first_definition_in_scope();
        memB != NULL;
        memB = memB->get_next_def_in_scope())
    {
        IDefinition::Kind kind = memB->get_kind();
        if (kind == IDefinition::DK_MEMBER) {
            ISymbol const *symB = memB->get_sym();
            ISymbol const *symA = symtabA.lookup_symbol(symB->get_name());

            // find this member in scope A
            Definition const *memA = NULL;
            if (symA != NULL) {
                memA = scopeA->find_definition_in_scope(symA);
                if (memA != NULL && memA->get_kind() != IDefinition::DK_MEMBER)
                    memA = NULL;
            }

            if (memA == NULL) {
                // we found a new member
                added_members.push_back(memB);
            }
        } else if (kind == IDefinition::DK_CONSTRUCTOR) {
            if (memB->get_semantics() == IDefinition::DS_ELEM_CONSTRUCTOR) {
                // found the elemental constructor
                elementalB = memB;
            }
        }
    }

    MDL_ASSERT(elementalA != NULL && elementalB != NULL);

    mi::base::Handle<Module const> moduleA(mi::base::make_handle_dup(m_modA));
    if (elementalA->has_flag(Definition::DEF_IS_IMPORTED)) {
        Module const *owner;
        elementalA = moduleA->get_original_definition(elementalA, owner);

        moduleA = mi::base::make_handle_dup(owner);
    }

    mi::base::Handle<Module const> moduleB(mi::base::make_handle_dup(m_modB));
    if (elementalB->has_flag(Definition::DEF_IS_IMPORTED)) {
        Module const *owner;
        elementalB = moduleB->get_original_definition(elementalB, owner);

        moduleB = mi::base::make_handle_dup(owner);
    }

    Def_pair_list init_members(get_allocator());

    for (Def_pair_list::const_iterator it(normal_members.begin()), end(normal_members.end());
        it != end;
        ++it)
    {
        Definition const *memA = it->first;
        Definition const *memB = it->second;

        int idxA = memA->get_field_index();
        int idxB = memB->get_field_index();

        IExpression const *initA = elementalA->get_default_param_initializer(idxA);
        IExpression const *initB = elementalB->get_default_param_initializer(idxB);

        if (initA == NULL && initB != NULL) {
            // it is allowed to add new defaults
            continue;
        }
        if (!compare_expressions(initA, initB))
            init_members.push_back(*it);
    }

    if (!missing_members.empty() || !changed_members.empty() ||
        !added_members.empty() || !init_members.empty()) {
        error(
            m_fnameA,
            INCOMPATIBLE_STRUCT,
            defA,
            Error_params(get_allocator()).add_signature(defA)
        );

        // note missing members
        for (Def_list::const_iterator it(missing_members.begin()), end(missing_members.end());
            it != end;
            ++it)
        {
            add_note(
                m_fnameA,
                MISSING_STRUCT_MEMBER,
                *it,
                Error_params(get_allocator()).add_signature(*it));
        }

        // node changed members
        for (Def_pair_list::const_iterator it(changed_members.begin()), end(changed_members.end());
            it != end;
            ++it)
        {
            Definition const *defA = it->first;
            Definition const *defB = it->second;

            add_note(
                m_fnameA,
                DIFFERENT_STRUCT_MEMBER_TYPE,
                defA,
                Error_params(get_allocator())
                    .add_signature(defA)
                    .add(defA->get_type())
                    .add(defB->get_type())
                    .add(m_fnameB));
        }

        // note added members
        for (Def_list::const_iterator it(added_members.begin()), end(added_members.end());
            it != end;
            ++it)
        {
            add_note(
                m_fnameB,
                ADDED_STRUCT_MEMBER,
                *it,
                Error_params(get_allocator()).add_signature(*it));
        }

        // note different default initializers
        for (Def_pair_list::const_iterator it(init_members.begin()), end(init_members.end());
            it != end;
            ++it)
        {
//          Definition const *old_def = it->first;
            Definition const *new_def = it->second;
            add_note(
                m_fnameB,
                DIFFERENT_DEFAULT_ARGUMENT,
                new_def,
                Error_params(get_allocator()).add_signature(new_def));
        }
    }
}

// Compare two enum types.
void Comparator::compare_enum(
    Definition const   *defA,
    Definition const   *defB,
    Symbol_table const &symtabA,
    Symbol_table const &symtabB)
{
    // enum values are in the defining scope, NOT in the enum scope
    Scope const *scopeA = defA->get_def_scope();
    Scope const *scopeB = defB->get_def_scope();

    IType_enum const *etpA = cast<IType_enum>(defA->get_type());
    IType_enum const *etpB = cast<IType_enum>(defB->get_type());

    typedef list<Definition const *>::Type Def_list;
    Def_list missing_values(get_allocator());

    typedef std::pair<Definition const *, Definition const *> Def_pair;
    typedef list<Def_pair>::Type                              Pair_list;
    Pair_list changed_values(get_allocator());

    for (Definition const *evA = scopeA->get_first_definition_in_scope();
        evA != NULL;
        evA = evA->get_next_def_in_scope())
    {
        if (evA->get_kind() == IDefinition::DK_ENUM_VALUE && evA->get_type() == etpA) {
            ISymbol const *symA = evA->get_sym();
            ISymbol const *symB = symtabB.lookup_symbol(symA->get_name());

            // find this enum value in scope B
            Definition const *evB = NULL;
            if (symB != NULL) {
                evB = scopeB->find_definition_in_scope(symB);
                if (evB != NULL && (evB->get_kind() != IDefinition::DK_ENUM_VALUE ||
                    evB->get_type() != etpB))
                    evB = NULL;
            }

            if (evB == NULL) {
                // we found a missing enum value
                missing_values.push_back(evA);
            } else if (!compare_values(evA->get_constant_value(), evB->get_constant_value())) {
                changed_values.push_back(Def_pair(evA, evB));
            }
        }
    }

    Def_list added_values(get_allocator());

    for (Definition const *evB = scopeB->get_first_definition_in_scope();
        evB != NULL;
        evB = evB->get_next_def_in_scope())
    {
        if (evB->get_kind() == IDefinition::DK_ENUM_VALUE && evB->get_type() == etpB) {
            ISymbol const *symB = evB->get_sym();
            ISymbol const *symA = symtabA.lookup_symbol(symB->get_name());

            // find this value in scope A
            Definition const *evA = NULL;
            if (symA != NULL) {
                evA = scopeA->find_definition_in_scope(symA);
                if (evA != NULL && (evA->get_kind() != IDefinition::DK_ENUM_VALUE ||
                    evA->get_type() != etpA))
                    evA = NULL;
            }

            if (evA == NULL) {
                // we found a new value
                added_values.push_back(evB);
            }
        }
    }

    if (!missing_values.empty() || !changed_values.empty() || !added_values.empty()) {
        error(
            m_fnameA,
            INCOMPATIBLE_ENUM,
            defA,
            Error_params(get_allocator()).add_signature(defA)
        );

        // note missing values
        for (Def_list::const_iterator it(missing_values.begin()), end(missing_values.end());
            it != end;
            ++it)
        {
            add_note(
                m_fnameA,
                MISSING_ENUM_VALUE,
                *it,
                Error_params(get_allocator()).add_signature(*it));
        }

        // node changed values
        for (Pair_list::const_iterator it(changed_values.begin()), end(changed_values.end());
            it != end;
            ++it)
        {
            Definition const *defA = it->first;
            Definition const *defB = it->second;

            add_note(
                m_fnameA,
                DIFFERENT_ENUM_VALUE,
                defA,
                Error_params(get_allocator())
                    .add_signature(defA)
                    .add(cast<IValue_enum>(defA->get_constant_value())->get_value())
                    .add(cast<IValue_enum>(defB->get_constant_value())->get_value())
                    .add(m_fnameB));
        }

        // note added values
        for (Def_list::const_iterator it(added_values.begin()), end(added_values.end());
            it != end;
            ++it)
        {
            add_note(
                m_fnameB,
                ADDED_ENUM_VALUE,
                *it,
                Error_params(get_allocator()).add_signature(*it));
        }
    }
}

// Prints a signature to the printer.
void Comparator::print_signature(Definition const *def)
{
    IDefinition::Kind kind = def->get_kind();

    if (kind == IDefinition::DK_MEMBER) {
        // print type_name::member_name
        Scope const *scope = def->get_def_scope();

        if (scope->get_owner_definition()->has_flag(Definition::DEF_IS_IMPORTED)) {
            // use full name for imported types
            m_printer->print(scope->get_scope_type());
        } else {
            // use the scope name (== type name without scope)
            m_printer->print(scope->get_scope_name());
        }
        m_printer->print("::");
        m_printer->print(def->get_symbol());
        return;
    } else if (kind != IDefinition::DK_FUNCTION &&
        kind != IDefinition::DK_CONSTRUCTOR &&
        kind != IDefinition::DK_ANNOTATION) {
        m_printer->print(def->get_symbol());
        return;
    }

    IType_function const *func_type = cast<IType_function>(def->get_type());

    bool is_material = false;
    if (kind == Definition::DK_FUNCTION) {
        IType const *ret_type = func_type->get_return_type();
        if (IType_struct const *s_type = as<IType_struct>(ret_type))
            if (s_type->get_predefined_id() == IType_struct::SID_MATERIAL)
                is_material = true;

        if (is_material) {
            m_printer->print(ret_type);
            m_printer->print(" ");
        }
    } else if (kind == Definition::DK_ANNOTATION)
        m_printer->print("annotation ");

    m_printer->print(def->get_symbol());

    if (!is_material) {
        m_printer->print("(");
        for (int i = 0, n = func_type->get_parameter_count(); i < n; ++i) {
            IType const   *p_type;
            ISymbol const *p_sym;
            func_type->get_parameter(i, p_type, p_sym);

            if (i > 0)
                m_printer->print(", ");
            m_printer->print(p_type);
        }
        m_printer->print(")");
    }
}

// Replace the archive prefix of the given file by another archive prefix given.
string Archive_comparator::replace_archive_prefix(
    char const   *f_name,
    string const &archive_fnameA,
    string const &archive_fnameB)
{
    string f(f_name, get_allocator());
    string res(get_allocator());

    size_t pos = f.find(".mdr:");
    if (pos == string::npos) {
        // bad, not inside an archive
        return res;
    }

    // add ".mdr"
    pos += 4;

    size_t l = archive_fnameA.length();

    if (l < pos) {
        MDL_ASSERT(f.substr(pos - l, l) == archive_fnameA);
        res = f.substr(0, pos - l);
    }

    res.append(archive_fnameB);
    res.append(f.substr(pos));

    return res;
}

// Compare the two archives.
void Archive_comparator::compare_archives()
{
    Allocator_builder builder(get_allocator());

    // Create an archive tool
    mi::base::Handle<Archive_tool> archive_tool(
        builder.create<Archive_tool>(get_allocator(), m_compiler.get()));

    mi::base::Handle<Manifest const> manifestA(archive_tool->get_manifest(m_fnameA.c_str()));
    if (!manifestA.is_valid_interface()) {
        // FIXME: error
        return;
    }

    mi::base::Handle<Manifest const> manifestB(archive_tool->get_manifest(m_fnameB.c_str()));
    if (!manifestB.is_valid_interface()) {
        // FIXME: error
        return;
    }

    // check the archive version number first
    Semantic_version const &verA = *manifestA->get_sema_version();
    Semantic_version const &verB = *manifestB->get_sema_version();

    if (!(verA <= verB)) {
        error(
            NULL,
            SEMA_VERSION_IS_HIGHER,
            zero,
            Error_params(get_allocator())
                .add(m_fnameA)
                .add(m_fnameB));
    }

    mi::base::Handle<IMDL_search_path> sp(m_compiler->get_search_path());

    for (size_t i = 0, n = manifestA->get_module_count(); i < n; ++i) {
        // Manifest contains the names WITHOUT leading '::'
        string module_name("::", get_allocator());
        module_name.append(manifestA->get_module_name(i));

        MDL_ASSERT(':' != manifestA->get_module_name(i)[0] && "unexpected '::' at manifest");

        if (m_cb != NULL) {
            m_cb->fire_event(IMDL_comparator_event::EV_COMPARING_MODULE, module_name.c_str());

            m_cb->percentage(i, n);
        }

        mi::base::Handle<Module const> moduleA(
            m_compiler->load_module(&m_ctx, module_name.c_str(), &m_cache_A));
        append_ctx_messages();

        if (!moduleA.is_valid_interface()) {
            // internal error: Manifest reports a module that cannot be opened
            continue;
        }
        cache_module(m_cache_A, moduleA.get());

        string file_name = replace_archive_prefix(moduleA->get_filename(), m_fnameA, m_fnameB);

        m_ctx.set_module_replacement_path(module_name.c_str(), file_name.c_str());

        // temporary replace the search path
        m_repl_sp->retain();
        m_compiler->install_search_path(m_repl_sp);

        mi::base::Handle<Module const> moduleB(
            m_compiler->load_module(&m_ctx, module_name.c_str(), &m_cache_B));
        append_ctx_messages();

        sp->retain();
        m_compiler->install_search_path(sp.get());

        if (!moduleB.is_valid_interface()) {
            // Module was removed
            error(
                m_fnameB.c_str(),
                ARCHIVE_DOES_NOT_CONTAIN_MODULE,
                zero,
                Error_params(get_allocator())
                    .add(module_name)
            );
            continue;
        }
        cache_module(m_cache_B, moduleB.get());

        // compare the modules
        Comparator comparator(
            get_allocator(),
            m_compiler.get(),
            moduleA.get(),
            moduleB.get(),
            m_ctx,
            m_cb);

        comparator.compare_modules();

        append_ctx_messages();
    }

    // finally copy the accumulated messages back to the context
    m_ctx.access_messages_impl().clear_messages();
    m_ctx.access_messages_impl().copy_messages(m_msgs);
}


namespace {

bool equal( const IValue* value_a, const IValue* value_b);
bool equal( const IExpression* expression_a, const IExpression* expression_b);
bool equal( const IAnnotation_block* block_a, const IAnnotation_block* block_b);
bool equal( const IStatement* stmt_a, const IStatement* stmt_b);
bool equal( const IDeclaration* decl_a, const IDeclaration* decl_b);

// Compares symbols.
bool equal( const ISymbol* symbol_a, const ISymbol* symbol_b)
{
    if( symbol_a->get_id() != symbol_b->get_id())
        return false;

    const char* a = symbol_a->get_name();
    const char* b = symbol_b->get_name();
    if( strcmp( a, b) != 0)
        return false;

    return true;
}

// Compares simple names. Supports NULL arguments.
bool equal( const ISimple_name* name_a, const ISimple_name* name_b)
{
    if( !name_a && !name_b)
        return true;
    if( !name_a || !name_b)
        return false;

    const ISymbol* symbol_a = name_a->get_symbol();
    const ISymbol* symbol_b = name_b->get_symbol();
    return equal( symbol_a, symbol_b);
}

// Compares qualified names. Supports NULL arguments.
bool equal( const IQualified_name* name_a, const IQualified_name* name_b)
{
    if( !name_a && !name_b)
        return true;
    if( !name_a || !name_b)
        return false;

    if( name_a->is_absolute() != name_b->is_absolute())
        return false;

    int count_a = name_a->get_component_count();
    int count_b = name_b->get_component_count();
    if( count_a != count_b)
        return false;

    for( int i = 0; i < count_a; ++i) {
        const ISimple_name* component_a = name_a->get_component( i);
        const ISimple_name* component_b = name_b->get_component( i);
        if( !equal( component_a, component_b))
            return false;
    }

    return true;
}

// Compares type names. Supports NULL arguments.
bool equal( const IType_name* name_a, const IType_name* name_b)
{
    if( !name_a && !name_b)
        return true;
    if( !name_a || !name_b)
        return false;

    if( name_a->is_absolute() != name_b->is_absolute())
        return false;

    if( name_a->get_qualifier() != name_b->get_qualifier())
        return false;

    const IQualified_name* qual_a = name_a->get_qualified_name();
    const IQualified_name* qual_b = name_b->get_qualified_name();
    if( !equal( qual_a, qual_b))
        return false;

    if( name_a->is_array() != name_b->is_array())
        return false;
    if( name_a->is_concrete_array() != name_b->is_concrete_array())
        return false;

    const IExpression* array_size_a = name_a->get_array_size();
    const IExpression* array_size_b = name_b->get_array_size();
    if( !equal( array_size_a, array_size_b))
        return false;

    const ISimple_name* size_name_a = name_a->get_size_name();
    const ISimple_name* size_name_b = name_b->get_size_name();
    if( !equal( size_name_a, size_name_b))
        return false;

    if( name_a->is_incomplete_array() != name_b->is_incomplete_array())
        return false;

    return true;
}

// Compares values.
bool equal( const IValue* value_a, const IValue* value_b)
{
    IValue::Kind kind_a = value_a->get_kind();
    IValue::Kind kind_b = value_b->get_kind();
    if( kind_a != kind_b)
        return false;

    switch( kind_a) {

        case IValue::VK_BAD:
            return true;

        case IValue::VK_BOOL: {
            auto bool_a = cast<IValue_bool>( value_a);
            auto bool_b = cast<IValue_bool>( value_b);
            return bool_a->get_value() == bool_b->get_value();
        }

        case IValue::VK_INT:
        case IValue::VK_ENUM: {
            auto int_valued_a = cast<IValue_int_valued>( value_a);
            auto int_valued_b = cast<IValue_int_valued>( value_b);
            return int_valued_a->get_value() == int_valued_b->get_value();
        }

        case IValue::VK_FLOAT: {
            auto float_a = cast<IValue_float>( value_a);
            auto float_b = cast<IValue_float>( value_b);
            IValue_FP::FP_class class_a = float_a->get_fp_class();
            IValue_FP::FP_class class_b = float_b->get_fp_class();
            if( class_a != class_b)
                return false;
            if( class_a != IValue_FP::FPC_NORMAL)
                return true;
            return float_a->get_value() == float_b->get_value();
        }

        case IValue::VK_DOUBLE: {
            auto double_a = cast<IValue_double>( value_a);
            auto double_b = cast<IValue_double>( value_b);
            IValue_FP::FP_class class_a = double_a->get_fp_class();
            IValue_FP::FP_class class_b = double_b->get_fp_class();
            if( class_a != class_b)
                return false;
            if( class_a != IValue_FP::FPC_NORMAL)
                return true;
            return double_a->get_value() == double_b->get_value();
        }

        case IValue::VK_STRING: {
            auto string_a = cast<IValue_string>( value_a);
            auto string_b = cast<IValue_string>( value_b);
            return strcmp( string_a->get_value(), string_b->get_value()) == 0;
        }

        case IValue::VK_VECTOR:
        case IValue::VK_MATRIX:
        case IValue::VK_ARRAY:
        case IValue::VK_RGB_COLOR:
        case IValue::VK_STRUCT: {
            auto compound_a = cast<IValue_compound>( value_a);
            auto compound_b = cast<IValue_compound>( value_b);
            int count_a = compound_a->get_component_count();
            int count_b = compound_b->get_component_count();
            if( count_a != count_b)
                return false;
            for( int i = 0; i < count_a; ++i) {
                const IValue* component_a = compound_a->get_value( i);
                const IValue* component_b = compound_b->get_value( i);
                if( !equal( component_a, component_b))
                    return false;
            }
            return true;
        }

        case IValue::VK_INVALID_REF:
            return true;

        case IValue::VK_TEXTURE: {
            auto texture_a = cast<IValue_texture>( value_a);
            auto texture_b = cast<IValue_texture>( value_b);
            if( texture_a->get_gamma_mode() != texture_b->get_gamma_mode())
                return false;
            if( texture_a->get_bsdf_data_kind() != texture_b->get_bsdf_data_kind())
                return false;
            return strcmp( texture_a->get_string_value(), texture_b->get_string_value()) == 0;
        }

        case IValue::VK_LIGHT_PROFILE:
        case IValue::VK_BSDF_MEASUREMENT: {
            auto resource_a = cast<IValue_resource>( value_a);
            auto resource_b = cast<IValue_resource>( value_b);
            return strcmp( resource_a->get_string_value(), resource_b->get_string_value()) == 0;
        }
    }

    MDL_ASSERT( !"unsupported type kind");
    return false;
}

// Compares literal expressions.
bool equal( const IExpression_literal* expr_a, const IExpression_literal* expr_b)
{
    const IValue* value_a = expr_a->get_value();
    const IValue* value_b = expr_b->get_value();
    if( !equal( value_a, value_b))
        return false;

    return true;
}

// Compares reference expressions.
bool equal( const IExpression_reference* expr_a, const IExpression_reference* expr_b)
{
    const IType_name* name_a = expr_a->get_name();
    const IType_name* name_b = expr_b->get_name();
    if( !equal( name_a, name_b))
        return false;

    if( expr_a->is_array_constructor() != expr_b->is_array_constructor())
        return false;

    return true;
}

// Compares unary expressions.
bool equal( const IExpression_unary* expr_a, const IExpression_unary* expr_b)
{
    IExpression_unary::Operator operator_a = expr_a->get_operator();
    IExpression_unary::Operator operator_b = expr_b->get_operator();
    if( operator_a != operator_b)
        return false;

    const IExpression* argument_a = expr_a->get_argument();
    const IExpression* argument_b = expr_b->get_argument();
    if( !equal( argument_a, argument_b))
        return false;

    const IType_name* type_name_a = expr_a->get_type_name();
    const IType_name* type_name_b = expr_b->get_type_name();
    if( !equal( type_name_a, type_name_b))
        return false;

    return true;
}

// Compares binary expressions.
bool equal( const IExpression_binary* expr_a, const IExpression_binary* expr_b)
{
    IExpression_binary::Operator operator_a = expr_a->get_operator();
    IExpression_binary::Operator operator_b = expr_b->get_operator();
    if( operator_a != operator_b)
        return false;

    const IExpression* left_argument_a = expr_a->get_left_argument();
    const IExpression* left_argument_b = expr_b->get_left_argument();
    if( !equal( left_argument_a, left_argument_b))
        return false;

    const IExpression* right_argument_a = expr_a->get_right_argument();
    const IExpression* right_argument_b = expr_b->get_right_argument();
    if( !equal( right_argument_a, right_argument_b))
        return false;

    return true;
}

// Compares conditional expressions.
bool equal( const IExpression_conditional* expr_a, const IExpression_conditional* expr_b)
{
    const IExpression* cond_argument_a = expr_a->get_condition();
    const IExpression* cond_argument_b = expr_b->get_condition();
    if( !equal( cond_argument_a, cond_argument_b))
        return false;

    const IExpression* true_argument_a = expr_a->get_true();
    const IExpression* true_argument_b = expr_b->get_true();
    if( !equal( true_argument_a, true_argument_b))
        return false;

    const IExpression* false_argument_a = expr_a->get_false();
    const IExpression* false_argument_b = expr_b->get_false();
    if( !equal( false_argument_a, false_argument_b))
        return false;

    return true;
}

// Compares let expressions.
bool equal( const IExpression_let* expr_a, const IExpression_let* expr_b)
{
    const IExpression* body_a = expr_a->get_expression();
    const IExpression* body_b = expr_b->get_expression();
    if( !equal( body_a, body_b))
        return false;

    int count_a = expr_a->get_declaration_count();
    int count_b = expr_b->get_declaration_count();
    if( count_a != count_b)
        return false;

    for( int i = 0; i < count_a; ++i) {
        const IDeclaration* decl_a = expr_a->get_declaration( i);
        const IDeclaration* decl_b = expr_b->get_declaration( i);
        if( !equal( decl_a, decl_b))
            return false;
    }

    return true;
}

// Compares arguments.
bool equal( const IArgument* arg_a, const IArgument* arg_b)
{
    IArgument::Kind kind_a = arg_a->get_kind();
    IArgument::Kind kind_b = arg_b->get_kind();
    if( kind_a != kind_b)
        return false;

    const IExpression* expr_a = arg_a->get_argument_expr();
    const IExpression* expr_b = arg_b->get_argument_expr();
    if( !equal( expr_a, expr_b))
        return false;

    if( kind_a == IArgument::AK_NAMED) {
        auto arg_named_a = cast<IArgument_named>( arg_a);
        auto arg_named_b = cast<IArgument_named>( arg_b);
        const ISimple_name* name_a = arg_named_a->get_parameter_name();
        const ISimple_name* name_b = arg_named_b->get_parameter_name();
        if( !equal( name_a, name_b))
            return false;
    }

    return true;
}

// Compares call expressions.
bool equal( const IExpression_call* expr_a, const IExpression_call* expr_b)
{
    const IExpression* ref_a = expr_a->get_reference();
    const IExpression* ref_b = expr_b->get_reference();
    if( !equal( ref_a, ref_b))
        return false;

    int count_a = expr_a->get_argument_count();
    int count_b = expr_b->get_argument_count();
    if( count_a != count_b)
        return false;

    for( int i = 0; i < count_a; ++i) {
        const IArgument* arg_a = expr_a->get_argument( i);
        const IArgument* arg_b = expr_b->get_argument( i);
        if( !equal( arg_a, arg_b))
            return false;
    }

    return true;
}

// Compares expressions. Supports NULL arguments.
bool equal( const IExpression* expr_a, const IExpression* expr_b)
{
    if( !expr_a && !expr_b)
        return true;
    if( !expr_a || !expr_b)
        return false;

    IExpression::Kind kind_a = expr_a->get_kind();
    IExpression::Kind kind_b = expr_b->get_kind();
    if( kind_a != kind_b)
        return false;

    if( expr_a->in_parenthesis() != expr_b->in_parenthesis())
        return false;

#define CASE( kind, type) \
        case IExpression::kind: { \
            auto type##_a = cast<IExpression_##type>( expr_a); \
            auto type##_b = cast<IExpression_##type>( expr_b); \
            return equal( type##_a, type##_b); \
        }

    switch( kind_a) {
        case IExpression::EK_INVALID:
            return true;
        CASE( EK_LITERAL, literal);
        CASE( EK_REFERENCE, reference);
        CASE( EK_UNARY, unary);
        CASE( EK_BINARY, binary);
        CASE( EK_CONDITIONAL, conditional);
        CASE( EK_CALL, call);
        CASE( EK_LET, let);
    }

#undef CASE

    MDL_ASSERT( !"unsupported type kind");
    return false;
}

// Compares annotations.
bool equal( const IAnnotation* anno_a, const IAnnotation* anno_b)
{
    IAnnotation::Kind kind_a = anno_a->get_kind();
    IAnnotation::Kind kind_b = anno_b->get_kind();
    if( kind_a != kind_b)
        return false;

    const IQualified_name* name_a = anno_a->get_name();
    const IQualified_name* name_b = anno_b->get_name();
    if( !equal( name_a, name_b))
        return false;

    int count_a = anno_a->get_argument_count();
    int count_b = anno_b->get_argument_count();
    if( count_a != count_b)
        return false;

    for( int i = 0; i < count_a; ++i) {
        const IArgument* arg_a = anno_a->get_argument( i);
        const IArgument* arg_b = anno_b->get_argument( i);
        if( !equal( arg_a, arg_b))
            return false;
    }

    if( kind_a == IAnnotation::AK_ENABLE_IF) {
        auto anno_enable_if_a = cast<IAnnotation_enable_if>( anno_a);
        auto anno_enable_if_b = cast<IAnnotation_enable_if>( anno_b);
        const IExpression* expr_a = anno_enable_if_a->get_expression();
        const IExpression* expr_b = anno_enable_if_b->get_expression();
        if( !equal( expr_a, expr_b))
            return false;
    }

    return true;
}

// Compares annotation blocks. Supports NULL arguments.
bool equal( const IAnnotation_block* block_a, const IAnnotation_block* block_b)
{
    if( !block_a && !block_b)
        return true;
    if( !block_a || !block_b)
        return false;

    int count_a = block_a->get_annotation_count();
    int count_b = block_b->get_annotation_count();
    if( count_a != count_b)
        return false;

    for( int i = 0; i < count_a; ++i) {
        const IAnnotation* anno_a = block_a->get_annotation( i);
        const IAnnotation* anno_b = block_b->get_annotation( i);
        if( !equal( anno_a, anno_b))
            return false;
    }

    return true;
}

// Compares compound statements.
bool equal( const IStatement_compound* stmt_a, const IStatement_compound* stmt_b)
{
    int count_a = stmt_a->get_statement_count();
    int count_b = stmt_b->get_statement_count();
    if( count_a != count_b)
        return false;

    for( int i = 0; i < count_a; ++i) {
        const IStatement* s_a = stmt_a->get_statement( i);
        const IStatement* s_b = stmt_b->get_statement( i);
        if( !equal( s_a, s_b))
           return false;
    }

    return true;
}

// Compares declaration statements.
bool equal( const IStatement_declaration* stmt_a, const IStatement_declaration* stmt_b)
{
    const IDeclaration* decl_a = stmt_a->get_declaration();
    const IDeclaration* decl_b = stmt_b->get_declaration();
    if( !equal( decl_a, decl_b))
       return false;

    return true;
}

// Compares expression statements.
bool equal( const IStatement_expression* stmt_a, const IStatement_expression* stmt_b)
{
    const IExpression* expr_a = stmt_a->get_expression();
    const IExpression* expr_b = stmt_b->get_expression();
    if( !equal( expr_a, expr_b))
       return false;

    return true;
}

// Compares if statements.
bool equal( const IStatement_if* stmt_a, const IStatement_if* stmt_b)
{
    const IExpression* expr_a = stmt_a->get_condition();
    const IExpression* expr_b = stmt_b->get_condition();
    if( !equal( expr_a, expr_b))
       return false;

    const IStatement* then_a = stmt_a->get_then_statement();
    const IStatement* then_b = stmt_b->get_then_statement();
    if( !equal( then_a, then_b))
       return false;

    const IStatement* else_a = stmt_a->get_else_statement();
    const IStatement* else_b = stmt_b->get_else_statement();
    if( !equal( else_a, else_b))
       return false;

    return true;
}

// Compares case statements.
bool equal( const IStatement_case* stmt_a, const IStatement_case* stmt_b)
{
    const IExpression* label_a = stmt_a->get_label();
    const IExpression* label_b = stmt_b->get_label();
    if( !equal( label_a, label_b))
       return false;

    const IStatement_compound* compound_a = stmt_a;
    const IStatement_compound* compound_b = stmt_b;
    return equal( compound_a, compound_b);
}

// Compares switch statements.
bool equal( const IStatement_switch* stmt_a, const IStatement_switch* stmt_b)
{
    const IExpression* cond_a = stmt_a->get_condition();
    const IExpression* cond_b = stmt_b->get_condition();
    if( !equal( cond_a, cond_b))
       return false;

    int count_a = stmt_a->get_case_count();
    int count_b = stmt_b->get_case_count();
    if( count_a != count_b)
        return false;

    for( int i = 0; i < count_a; ++i) {
        const IStatement* s_a = stmt_a->get_case( i);
        const IStatement* s_b = stmt_b->get_case( i);
        if( !equal( s_a, s_b))
           return false;
    }

    return true;
}

// Compares loop statements.
bool equal( const IStatement_loop* stmt_a, const IStatement_loop* stmt_b)
{
    const IExpression* cond_a = stmt_a->get_condition();
    const IExpression* cond_b = stmt_b->get_condition();
    if( !equal( cond_a, cond_b))
        return false;

    const IStatement* body_a = stmt_a->get_body();
    const IStatement* body_b = stmt_b->get_body();
    if( !equal( body_a, body_b))
        return false;

    return true;
}

// Compares for loop statements.
bool equal( const IStatement_for* stmt_a, const IStatement_for* stmt_b)
{
    const IStatement* init_a = stmt_a->get_init();
    const IStatement* init_b = stmt_b->get_init();
    if( !equal( init_a, init_b))
        return false;

    const IExpression* update_a = stmt_a->get_update();
    const IExpression* update_b = stmt_b->get_update();
    if( !equal( update_a, update_b))
        return false;

    const IStatement_loop* loop_a = stmt_a;
    const IStatement_loop* loop_b = stmt_b;
    return equal( loop_a, loop_b);
}

// Compares return statements.
bool equal( const IStatement_return* stmt_a, const IStatement_return* stmt_b)
{
    const IExpression* expr_a = stmt_a->get_expression();
    const IExpression* expr_b = stmt_b->get_expression();
    if( !equal( expr_a, expr_b))
        return false;

    return true;
}

// Compares statements. Supports NULL arguments.
bool equal( const IStatement* stmt_a, const IStatement* stmt_b)
{
    if( !stmt_a && !stmt_b)
        return true;
    if( !stmt_a || !stmt_b)
        return false;

    IStatement::Kind kind_a = stmt_a->get_kind();
    IStatement::Kind kind_b = stmt_b->get_kind();
    if( kind_a != kind_b)
        return false;

#define CASE( kind, type) \
        case IStatement::kind: { \
            auto type##_a = cast<IStatement_##type>( stmt_a); \
            auto type##_b = cast<IStatement_##type>( stmt_b); \
            return equal( type##_a, type##_b); \
        }

    switch( kind_a) {
        case IStatement::SK_INVALID:
            return true;
        CASE( SK_COMPOUND, compound);
        CASE( SK_DECLARATION, declaration);
        CASE( SK_EXPRESSION, expression);
        CASE( SK_IF, if);
        CASE( SK_CASE, case);
        CASE( SK_SWITCH, switch);
        CASE( SK_WHILE, while);
        CASE( SK_DO_WHILE, do_while);
        CASE( SK_FOR, for);
        case IStatement::SK_BREAK:
        case IStatement::SK_CONTINUE:
            return true;
        CASE( SK_RETURN, return);
    }

#undef CASE

    MDL_ASSERT( !"unsupported type kind");
    return false;
}

// Compares import declarations.
bool equal( const IDeclaration_import* decl_a, const IDeclaration_import* decl_b)
{
    const IQualified_name* module_name_a = decl_a->get_module_name();
    const IQualified_name* module_name_b = decl_b->get_module_name();
    if( !equal( module_name_a, module_name_b))
        return false;

    int count_a = decl_a->get_name_count();
    int count_b = decl_b->get_name_count();
    if( count_a != count_b)
        return false;

    for( int i = 0; i < count_a; ++i) {
        const IQualified_name* name_a = decl_a->get_name( i);
        const IQualified_name* name_b = decl_b->get_name( i);
        if( !equal( name_a, name_b))
            return false;
    }

    return true;
}

// Compares parameters.
bool equal( const IParameter* param_a, const IParameter* param_b)
{
    const IType_name* type_name_a = param_a->get_type_name();
    const IType_name* type_name_b = param_b->get_type_name();
    if( !equal( type_name_a, type_name_b))
        return false;

    const ISimple_name* name_a = param_a->get_name();
    const ISimple_name* name_b = param_b->get_name();
    if( !equal( name_a, name_b))
        return false;

    const IExpression* init_expr_a = param_a->get_init_expr();
    const IExpression* init_expr_b = param_b->get_init_expr();
    if( !equal( init_expr_a, init_expr_b))
        return false;

    const IAnnotation_block* block_a = param_a->get_annotations();
    const IAnnotation_block* block_b = param_b->get_annotations();
    if( !equal( block_a, block_b))
        return false;

    return true;
}

// Compares annotation declarations.
bool equal( const IDeclaration_annotation* decl_a, const IDeclaration_annotation* decl_b)
{
    const ISimple_name* name_a = decl_a->get_name();
    const ISimple_name* name_b = decl_b->get_name();
    if( !equal( name_a, name_b))
        return false;

    const IAnnotation_block* block_a = decl_a->get_annotations();
    const IAnnotation_block* block_b = decl_b->get_annotations();
    if( !equal( block_a, block_b))
        return false;

    int count_a = decl_a->get_parameter_count();
    int count_b = decl_b->get_parameter_count();
    if( count_a != count_b)
        return false;

    for( int i = 0; i < count_a; ++i) {
        const IParameter* parameter_a = decl_a->get_parameter( i);
        const IParameter* parameter_b = decl_b->get_parameter( i);
        if( !equal( parameter_a, parameter_b))
            return false;
    }

    return true;
}

// Compares constant declarations.
bool equal( const IDeclaration_constant* decl_a, const IDeclaration_constant* decl_b)
{
    const IType_name* type_name_a = decl_a->get_type_name();
    const IType_name* type_name_b = decl_b->get_type_name();
    if( !equal( type_name_a, type_name_b))
        return false;

    int count_a = decl_a->get_constant_count();
    int count_b = decl_b->get_constant_count();
    if( count_a != count_b)
        return false;

    for( int i = 0; i < count_a; ++i) {
        const ISimple_name* name_a = decl_a->get_constant_name( i);
        const ISimple_name* name_b = decl_b->get_constant_name( i);
        if( !equal( name_a, name_b))
            return false;
        const IExpression* expr_a = decl_a->get_constant_exp( i);
        const IExpression* expr_b = decl_b->get_constant_exp( i);
        if( !equal( expr_a, expr_b))
            return false;
        const IAnnotation_block* block_a = decl_a->get_annotations( i);
        const IAnnotation_block* block_b = decl_b->get_annotations( i);
        if( !equal( block_a, block_b))
            return false;
    }

    return true;
}

// Compares type alias declarations.
bool equal( const IDeclaration_type_alias* decl_a, const IDeclaration_type_alias* decl_b)
{
    const IType_name* type_name_a = decl_a->get_type_name();
    const IType_name* type_name_b = decl_b->get_type_name();
    if( !equal( type_name_a, type_name_b))
        return false;

    const ISimple_name* alias_name_a = decl_a->get_alias_name();
    const ISimple_name* alias_name_b = decl_b->get_alias_name();
    if( !equal( alias_name_a, alias_name_b))
        return false;

    return true;
}

// Compares type struct declarations.
bool equal( const IDeclaration_type_struct* decl_a, const IDeclaration_type_struct* decl_b)
{
    const ISimple_name* name_a = decl_a->get_name();
    const ISimple_name* name_b = decl_b->get_name();
    if( !equal( name_a, name_b))
        return false;

    const IAnnotation_block* block_a = decl_a->get_annotations();
    const IAnnotation_block* block_b = decl_b->get_annotations();
    if( !equal( block_a, block_b))
        return false;

    int count_a = decl_a->get_field_count();
    int count_b = decl_b->get_field_count();
    if( count_a != count_b)
        return false;

    for( int i = 0; i < count_a; ++i) {
        const IType_name* type_name_a = decl_a->get_field_type_name( i);
        const IType_name* type_name_b = decl_b->get_field_type_name( i);
        if( !equal( type_name_a, type_name_b))
            return false;
        const ISimple_name* name_a = decl_a->get_field_name( i);
        const ISimple_name* name_b = decl_b->get_field_name( i);
        if( !equal( name_a, name_b))
            return false;
        const IExpression* expr_a = decl_a->get_field_init( i);
        const IExpression* expr_b = decl_b->get_field_init( i);
        if( !equal( expr_a, expr_b))
            return false;
        const IAnnotation_block* block_a = decl_a->get_annotations( i);
        const IAnnotation_block* block_b = decl_b->get_annotations( i);
        if( !equal( block_a, block_b))
            return false;
    }

    return true;
}

// Compares type enum declarations.
bool equal( const IDeclaration_type_enum* decl_a, const IDeclaration_type_enum* decl_b)
{
    const ISimple_name* name_a = decl_a->get_name();
    const ISimple_name* name_b = decl_b->get_name();
    if( !equal( name_a, name_b))
        return false;

    const IAnnotation_block* block_a = decl_a->get_annotations();
    const IAnnotation_block* block_b = decl_b->get_annotations();
    if( !equal( block_a, block_b))
        return false;

    int count_a = decl_a->get_value_count();
    int count_b = decl_b->get_value_count();
    if( count_a != count_b)
        return false;

    for( int i = 0; i < count_a; ++i) {
        const ISimple_name* name_a = decl_a->get_value_name( i);
        const ISimple_name* name_b = decl_b->get_value_name( i);
        if( !equal( name_a, name_b))
            return false;
        const IExpression* expr_a = decl_a->get_value_init( i);
        const IExpression* expr_b = decl_b->get_value_init( i);
        if( !equal( expr_a, expr_b))
            return false;
        const IAnnotation_block* block_a = decl_a->get_annotations( i);
        const IAnnotation_block* block_b = decl_b->get_annotations( i);
        if( !equal( block_a, block_b))
            return false;
    }

    return true;
}

// Compares variable declarations.
bool equal( const IDeclaration_variable* decl_a, const IDeclaration_variable* decl_b)
{
    const IType_name* type_name_a = decl_a->get_type_name();
    const IType_name* type_name_b = decl_b->get_type_name();
    if( !equal( type_name_a, type_name_b))
        return false;

    int count_a = decl_a->get_variable_count();
    int count_b = decl_b->get_variable_count();
    if( count_a != count_b)
        return false;

    for( int i = 0; i < count_a; ++i) {
        const ISimple_name* name_a = decl_a->get_variable_name( i);
        const ISimple_name* name_b = decl_b->get_variable_name( i);
        if( !equal( name_a, name_b))
            return false;
        const IExpression* expr_a = decl_a->get_variable_init( i);
        const IExpression* expr_b = decl_b->get_variable_init( i);
        if( !equal( expr_a, expr_b))
            return false;
        const IAnnotation_block* block_a = decl_a->get_annotations( i);
        const IAnnotation_block* block_b = decl_b->get_annotations( i);
        if( !equal( block_a, block_b))
            return false;
    }

    return true;
}

// Compares function declarations.
bool equal( const IDeclaration_function* decl_a, const IDeclaration_function* decl_b)
{
    const IType_name* return_type_name_a = decl_a->get_return_type_name();
    const IType_name* return_type_name_b = decl_b->get_return_type_name();
    if( !equal( return_type_name_a, return_type_name_b))
        return false;

    const ISimple_name* name_a = decl_a->get_name();
    const ISimple_name* name_b = decl_b->get_name();
    if( !equal( name_a, name_b))
        return false;

    if( decl_a->is_preset() != decl_b->is_preset())
        return false;

    int count_a = decl_a->get_parameter_count();
    int count_b = decl_b->get_parameter_count();
    if( count_a != count_b)
        return false;

    for( int i = 0; i < count_a; ++i) {
        const IParameter* param_a = decl_a->get_parameter( i);
        const IParameter* param_b = decl_b->get_parameter( i);
        if( !equal( param_a, param_b))
            return false;
    }

    if( decl_a->get_qualifier() != decl_b->get_qualifier())
        return false;

    const IStatement* body_a = decl_a->get_body();
    const IStatement* body_b = decl_b->get_body();
    if( !equal( body_a, body_b))
        return false;

    const IAnnotation_block* block_a = decl_a->get_annotations();
    const IAnnotation_block* block_b = decl_b->get_annotations();
    if( !equal( block_a, block_b))
        return false;

    const IAnnotation_block* return_block_a = decl_a->get_return_annotations();
    const IAnnotation_block* return_block_b = decl_b->get_return_annotations();
    if( !equal( return_block_a, return_block_b))
        return false;

    return true;
}

// Compares module declarations.
bool equal( const IDeclaration_module* decl_a, const IDeclaration_module* decl_b)
{
    const IAnnotation_block* block_a = decl_a->get_annotations();
    const IAnnotation_block* block_b = decl_b->get_annotations();
    if( !equal( block_a, block_b))
        return false;

    return true;
}

// Compares namespace alias declarations.
bool equal( const IDeclaration_namespace_alias* decl_a, const IDeclaration_namespace_alias* decl_b)
{
    const ISimple_name* alias_a = decl_a->get_alias();
    const ISimple_name* alias_b = decl_b->get_alias();
    if( !equal( alias_a, alias_b))
        return false;

    const IQualified_name* namespace_a = decl_a->get_namespace();
    const IQualified_name* namespace_b = decl_b->get_namespace();
    if( !equal( namespace_a, namespace_b))
        return false;

    return true;
}

// Compares declarations.
bool equal( const IDeclaration* decl_a, const IDeclaration* decl_b)
{
    IDeclaration::Kind kind_a = decl_a->get_kind();
    IDeclaration::Kind kind_b = decl_b->get_kind();
    if( kind_a != kind_b)
        return false;

    if( decl_a->is_exported() != decl_b->is_exported())
        return false;

#define CASE( kind, type) \
        case IDeclaration::kind: { \
            auto type##_a = cast<IDeclaration_##type>( decl_a); \
            auto type##_b = cast<IDeclaration_##type>( decl_b); \
            return equal( type##_a, type##_b); \
        }

    switch( kind_a) {
        case IDeclaration::DK_INVALID:
            return true;
        CASE( DK_IMPORT, import);
        CASE( DK_ANNOTATION, annotation);
        CASE( DK_CONSTANT, constant);
        CASE( DK_TYPE_ALIAS, type_alias);
        CASE( DK_TYPE_STRUCT, type_struct);
        CASE( DK_TYPE_ENUM, type_enum);
        CASE( DK_VARIABLE, variable);
        CASE( DK_FUNCTION, function);
        CASE( DK_MODULE, module);
        CASE( DK_NAMESPACE_ALIAS, namespace_alias);
    }

#undef CASE

    MDL_ASSERT( !"unsupported type kind");
    return false;
}

} // namespace

bool equal( const IModule* module_a, const IModule* module_b)
{
    // compare module name
    const char* name_a = module_a->get_name();
    const char* name_b = module_b->get_name();
    if( strcmp( name_a, name_b) != 0)
        return false;

    // compare filename
    const char* filename_a = module_a->get_filename();
    const char* filename_b = module_b->get_filename();
    if( !!filename_a ^ !!filename_b)
        return false;
    if( filename_a && strcmp( filename_a, filename_b) != 0)
        return false;

    // compare version
    int major_a, minor_a, major_b, minor_b;
    module_a->get_version( major_a, minor_a);
    module_b->get_version( major_b, minor_b);
    if( major_a != major_b || minor_a != minor_b)
        return false;

    // compare analyzed/valid properties
    if( module_a->is_analyzed() != module_b->is_analyzed())
        return false;
    if( module_a->is_valid() != module_b->is_valid())
        return false;

    // compare stdlib/builtins/mdle properties
    if( module_a->is_stdlib() != module_b->is_stdlib())
        return false;
    if( module_a->is_builtins() != module_b->is_builtins())
        return false;
    if( module_a->is_mdle() != module_b->is_mdle())
        return false;

    // compare declaration count
    int n_decl_a = module_a->get_declaration_count();
    int n_decl_b = module_b->get_declaration_count();
    if( n_decl_a != n_decl_b)
        return false;

    // compare declarations
    for( int i = 0; i < n_decl_a; ++i) {
        const IDeclaration* decl_a = module_a->get_declaration( i);
        const IDeclaration* decl_b = module_b->get_declaration( i);
        if( !equal( decl_a, decl_b))
           return false;
    }

   return true;
}

}  // mdl
}  // mi
