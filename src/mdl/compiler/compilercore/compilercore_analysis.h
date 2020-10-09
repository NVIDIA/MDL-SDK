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

#ifndef MDL_COMPILERCORE_ANALYSIS_H
#define MDL_COMPILERCORE_ANALYSIS_H 1

#include <cstdarg>

#include <mi/base/handle.h>

#include <mi/mdl/mdl_mdl.h>
#include <mi/mdl/mdl_printers.h>
#include <mi/mdl/mdl_declarations.h>

#include "compilercore_cc_conf.h"

#include "compilercore_rawbitset.h"
#include "compilercore_errors.h"
#include "compilercore_type_cache.h"
#include "compilercore_visitor.h"
#include "compilercore_allocator.h"
#include "compilercore_streams.h"
#include "compilercore_def_table.h"
#include "compilercore_errors.h"
#include "compilercore_call_graph.h"
#include "compilercore_stmt_info.h"
#include "compilercore_stmt_info.h"

namespace mi {
namespace mdl {

class MDL;
class Module;
class Error_params;
class Symbol_table;
class ISymbol;
class IValue;
class Dependence_graph;
class Messages_impl;
class Err_location;
class Thread_context;

struct Resource_table_key;

///
/// A base class for all semantic analysis passes.
///
class Analysis : protected Module_visitor {
protected:
    typedef vector<Definition const *>::Type                        Definition_vec;
    typedef vector<IType const *>::Type                             IType_vec;
    typedef list<Definition const *>::Type                          Definition_list;
    typedef ptr_hash_map<ISymbol const, size_t>::Type               Name_index_map;
    typedef ptr_hash_map<Definition const, Position const *>::Type  Import_locations;

    ///
    /// Helper class to capture a constant fold exception.
    ///
    class Const_fold_expression : public IConst_fold_handler
    {
    public:
        /// Handle constant folding exceptions.
        void exception(
            Reason            r,
            IExpression const *expr,
            int               index = 0,
            int               length = 0) MDL_FINAL;

        /// Handle variable lookup.
        IValue const *lookup(
            IDefinition const *var) MDL_FINAL;

        /// Check whether evaluate_intrinsic_function should be called for an unhandled
        /// intrinsic functions with the given semantic.
        bool is_evaluate_intrinsic_function_enabled(
            IDefinition::Semantics semantic) const MDL_FINAL
        {
            return false;
        }

        /// Handle intrinsic call evaluation.
        IValue const *evaluate_intrinsic_function(
            IDefinition::Semantics semantic,
            const IValue *const arguments[],
            size_t n_arguments) MDL_FINAL;

        /// Constructor.
        explicit Const_fold_expression(Analysis &ana)
        : m_ana(ana), m_error_state(false)
        {
        }

        /// Clear the captures error state.
        void clear_error_state() { m_error_state = false; }

        /// Returns true if an error occurred since the last clear_error_state() call.
        bool has_error() const { return m_error_state; }

    private:
        /// The analysis pass.
        Analysis &m_ana;

        /// Set to true once an exception occurs.
        bool     m_error_state;
    };

public:
    // Every analysis produces compiler errors.
    static char const MESSAGE_CLASS = 'C';

    /// Retrieve the used allocator.
    IAllocator *get_allocator() const { return m_builder.get_allocator(); }

    /// Retrieve the current module.
    Module const &get_module() const { return m_module; }

    /// Enable all warnings.
    void enable_all_warning(bool flag) { m_all_warnings_are_off = !flag; }

    /// Creates a new error.
    ///
    /// \param code    the error code
    /// \param loc     location of the error
    /// \param params  error message parameter inserts
    void error(
        int                code,
        Err_location const &loc,
        Error_params const &params);

    /// Creates a new error in MDL 1.1+ and strict mode, a warning in MDL <1.1 relaxed mode.
    ///
    /// \param code    the error code
    /// \param loc     location of the error
    /// \param params  error message parameter inserts
    void error_mdl_11(
        int                code,
        Err_location const &loc,
        Error_params const &params);

    /// Creates a new error in MDL 1.3+ and strict mode, a warning in MDL <1.3 relaxed mode.
    ///
    /// \param code    the error code
    /// \param loc     location of the error
    /// \param params  error message parameter inserts
    void error_mdl_13(
        int                code,
        Err_location const &loc,
        Error_params const &params);

    /// Creates a new error in strict mode, a warning in relaxed mode.
    ///
    /// \param code    the error code
    /// \param loc     location of the error
    /// \param params  error message parameter inserts
    void error_strict(
        int                code,
        Err_location const &loc,
        Error_params const &params);

    /// Creates a new warning.
    ///
    /// \param code    the error code
    /// \param loc     location of the error
    /// \param params  warning message parameter inserts
    void warning(
        int                code,
        Err_location const &loc,
        Error_params const &params);


    /// Add a note to the last error/warning.
    ///
    /// \param code    the error code
    /// \param loc     location of the error
    /// \param params  note message parameter inserts
    void add_note(
        int                code,
        Err_location const &loc,
        Error_params const &params);

    /// Add an imported message to the last error
    ///
    /// \param fname_id  the file name id for the imported message
    /// \param msg       the message
    void add_imported_message(
        size_t         fname_id,
        IMessage const *msg);

    /// Get the definition of the base entity in a lvalue.
    ///
    /// \param expr  the expression
    ///
    /// \return the Definition if expr is an lvalue, NULL else
    static IDefinition const *get_lvalue_base(IExpression const *expr);

    /// Returns the return type of a function definition.
    ///
    /// \param def  the definition of a function
    static IType const *get_result_type(IDefinition const *def);

    /// If true, compile in strict mode.
    bool strict_mode() const { return m_strict_mode; }

    /// If true, experimental MDL features are enable.
    bool enable_experimental_features() const { return m_enable_experimental_features; }

    /// If true, resolve resources and generate errors if resources are missing.
    bool resolve_resources() const { return m_resolve_resources; }


protected:
    /// Enter an imported definition.
    ///
    /// \param imported           the definition that is imported
    /// \param owner_import_idx   the index of the original owner in the current import table
    /// \param loc                the location of the import (if any)
    Definition *import_definition(
        Definition const *imported,
        size_t           owner_import_idx,
        Position const   *loc);

    /// Get the import location of an imported definition.
    ///
    /// \param def  an imported definition
    Position const *get_import_location(Definition const *def) const;

    /// Add a compiler note for a previous definition.
    ///
    /// \param prev_def  the previous definition
    void add_prev_definition_note(Definition const *prev_def);

    /// Add a compiler note if the given expression references an let temporary.
    ///
    /// \param lhs  an expression used as an lvalue
    void add_let_temporary_note(IExpression const *lhs);

    /// Helper, gets the file_id for a module id.
    ///
    /// \param msgs     a message list
    /// \param imp_idx  a module import index
    ///
    /// \return the file id of this module in the message list
    size_t get_file_id(
        Messages_impl  &msgs,
        size_t         imp_idx);

    /// Helper, get the file id from an error location.
    ///
    /// \param msgs    a message list
    /// \param loc     the error location
    size_t get_file_id(
        Messages_impl      &msgs,
        Err_location const &params);

    /// Return a fully qualified name for the current scope.
    string get_current_scope_name() const;

    /// Issue an error for some previously defined entity.
    ///
    /// \param kind    the kind of the current entity
    /// \param def     the previous definition
    /// \param pos     position of the current definition
    /// \param err     the error to issue
    /// \param as_mat  current declaration is a material
    void err_redeclaration(
        Definition::Kind  kind,
        Definition const  *def,
        Position const    &pos,
        Compilation_error err,
        bool              as_mat = false);

    /// Issue a warning if some previously defined entity is shadowed.
    ///
    /// \param def     the previous definition
    /// \param pos     position of the current definition
    /// \param warn    the warning to issue
    void warn_shadow(
        Definition const  *def,
        Position const    &pos,
        Compilation_error warn);

    /// Resolve a scope operator and enter the scope.
    ///
    /// \param scope_node  the scope node: its children are n identifier
    /// \param had_error   set to true if the scope could not be resolved
    /// \param silent      set to true if no error messages should be reported
    ///
    /// \return a Scope if one was bound, NULL else
    Scope *resolve_scope(
        const IQualified_name *scope_node,
        bool &had_error,
        bool silent = false);

    /// Returns true if the given type is the base material.
    bool is_base_material(const IType *type) const;

    /// Possible restrictions for searching a best match.
    enum Match_restriction {
        MR_ANY,          ///< allow any match
        MR_MEMBER,       ///< members only
        MR_CALLABLE,     ///< functions and other callable objects
        MR_NON_CALLABLE, ///< non-callable objects
        MR_TYPE,         ///< only type names
        MR_ANNO          ///< annotations
    };

    /// Check if a given definition kind is allowed under the given match restriction.
    ///
    /// \param kind         a definition kind
    /// \param restriction  a match restriction
    static bool allow_definition(
        IDefinition::Kind kind,
        Match_restriction restriction);

    /// Try to find the best candidate for a search symbol using an edit distance metric.
    ///
    /// \param sym           the symbol that is searched (but not found).
    /// \param scope         if non-NULL, the scope where to search
    /// \param restriction  restrict possible matches to given kind
    ISymbol const *find_best_match(
        ISymbol const      *sym,
        Scope              *scope,
        Match_restriction  restrition) const;

    /// Try to find the best named subscope candidate for a search symbol using an
    /// edit distance metric.
    ///
    /// \param sym    the symbol that is searched (but not found).
    /// \param scope  the scope where to search
    ISymbol const *find_best_named_subscope_match(
        ISymbol const *sym,
        Scope         *scope) const;

    /// Find the definition for a qualified name.
    ///
    /// \param qual_name     the qualified name
    /// \param ignore_error  if true, do not issue an error message
    ///
    /// \return its definition (creating an error definition if needed)
    Definition *find_definition_for_qualified_name(
        IQualified_name const *name,
        bool                  ignore_error = false);

    /// Find the definition for an annotation name.
    ///
    /// \param qname  the qualified name of an annotation
    ///
    /// \return its definition (creating an error definition if needed)
    Definition *find_definition_for_annotation_name(IQualified_name const *qname);

    /// Check if a given qualified name name the material (type).
    ///
    /// \param qual_name  a qualified name
    bool is_material_qname(IQualified_name const *qual_name);

    /// Get the one and only "error definition" of the processed module.
    Definition *get_error_definition() const;

    /// Checks if the give expression is a constant one.
    ///
    /// \param expr         the expression
    /// \param is_invalid   set to true if expr or one of its sub expressions are invalid
    bool is_const_expression(IExpression const *expr, bool &is_invalid);

    /// Checks if the give expression is a constant array size.
    ///
    /// \param expr         the expression
    /// \param def          if its a abstract array size, set to its definition, else set to NULL
    /// \param is_invalid   set to true if expr or one of its sub expressions are invalid
    bool is_const_array_size(
        IExpression const *expr,
        IDefinition const *&def,
        bool              &is_invalid);

    /// Apply a type qualifier to a given type.
    ///
    /// \param qual  the qualifier to apply
    /// \param type  the unqualified type
    /// 
    /// \return the qualified type
    const IType *qualify_type(Qualifier qual, const IType *type);

    /// Return the full name of a definition.
    string full_name(const Definition *def) const;

    /// Return the full name of an entity.
    ///
    /// \param name        the qualified name
    /// \param only_scope  if set, return the scope only
    string full_name(const IQualified_name *name, bool scope_only);

    /// Create a reference expression for a given definition and put it into given position.
    ///
    /// \param def  the definition
    /// \param pos  the position
    IExpression_reference *create_reference(Definition const *def, Position const &pos);

    /// Get the default expression of a parameter of a function, constructor or annotation.
    ///
    /// \param def        the entity definition
    /// \param param_idx  the index of the parameter
    ///
    /// \note In contrast to Definition::get_default_param_initializer() this version
    ///       retrieves the parameter initializer from imported entities
    IExpression const *get_default_param_initializer(
        Definition const *def,
        int              param_idx) const;

    /// Constructor.
    ///
    /// \param compiler   the MDL compiler
    /// \param module     the module to analyze
    /// \param ctx        the current thread context
    Analysis(
        MDL            *compiler,
        Module         &module,
        Thread_context &ctx);

    /// Destructor.
    ~Analysis();

private:
    /// Format a message.
    string format_msg(int code, Error_params const &params);

    /// Parse warning options.
    void parse_warning_options();

protected:
    /// The builder.
    Allocator_builder m_builder;

    /// The MDL compiler.
    MDL *m_compiler;

    /// The current module.
    Module &m_module;

    /// The current thread context.
    Thread_context &m_ctx;

    /// The MDL version we are analyze (copied from the module).
    IMDL::MDL_version m_mdl_version;

    /// The symbol table, retrieved from the module.
    Symbol_table *m_st;

    /// The type cache, wrapping the type factory from the module.
    Type_cache m_tc;

    /// The definition table for entities.
    Definition_table *m_def_tab;

    /// set if we inside a select expression lookup
    IType const *m_in_select;

    /// The index of the last generated message, used to add a note.
    size_t m_last_msg_idx;

    /// The current match restriction.
    Match_restriction m_curr_restriction;

    /// A string buffer used for error messages.
    mi::base::Handle<Buffer_output_stream> m_string_buf;

    /// Printer for error messages.
    mi::base::Handle<IPrinter> m_printer;

    /// Exception handler the creates errors.
    Const_fold_expression m_exc_handler;

    /// Bitset containing disabled warnings.
    Raw_bitset<MAX_ERROR_NUM + 1> m_disabled_warnings;

    /// Bitset containing warnings treated as errors.
    Raw_bitset<MAX_ERROR_NUM + 1> m_warnings_are_errors;

    /// If non-NULL, messages are appended here.
    Messages_impl *m_compiler_msgs;

    /// If true, all warnings are errors.
    bool m_all_warnings_are_errors;

    /// If true, all warnings are off.
    bool m_all_warnings_are_off;

    /// If true, compile in strict mode.
    bool m_strict_mode;

    /// If true, compile with experimental MDL features.
    bool m_enable_experimental_features;

    /// If true, resolve resources and generate errors if resources are missing.
    bool m_resolve_resources;


    typedef map<size_t, size_t>::Type Module_2_file_id_map;

    /// The set of modules that produced a message.
    Module_2_file_id_map m_modid_2_fileid;

    /// A map that allows to track where a definition was imported.
    Import_locations m_import_locations;
};

/// An cache for operator lookup.
class Operator_lookup_cache {
private:
    /// An operator lookup cache key for unary and binary operators.
    struct Operator_signature {
        IExpression::Operator op;
        IType const           *type_left;
        IType const           *type_right;

        Operator_signature(
            IExpression::Operator op,
            IType const           *type_left,
            IType const           *type_right)
        : op(op), type_left(type_left), type_right(type_right)
        {
        }
    };

    /// Hash an Operator_cache_entry.
    struct Op_cache_hash {
        size_t operator()(Operator_signature const &p) const {
            Hash_ptr<IType const> hasher;
            return (hasher(p.type_right) << 12) + (hasher(p.type_left) << 6) + size_t(p.op);
        }
    };

    /// Compare an Operator_cache_entry.
    struct Op_cache_equal {
        unsigned operator()(Operator_signature const &a, Operator_signature const &b) const {
            return
                a.op == b.op &&
                a.type_left == b.type_left &&
                a.type_right == b.type_right;
        }
    };

    typedef hash_map<
        Operator_signature,
        Definition const *,
        Op_cache_hash,
        Op_cache_equal>::Type Op_cache;

public:
    /// Constructor.
    Operator_lookup_cache(IAllocator *alloc);

    /// Lookup an operator.
    ///
    /// \param op        the operator
    /// \param left_tp   the left argument type
    /// \param right_tp  the right argument type (or NULL for unary operators)
    ///
    /// \return the definition of this operator OR NULL if no operator is known
    ///         for the given signature
    Definition const *lookup(
        IExpression::Operator op,
        IType const           *left_tp,
        IType const           *right_tp);

    /// Store an operator definition into the cache
    ///
    /// \param op        the operator
    /// \param left_tp   the left argument type
    /// \param right_tp  the right argument type (or NULL for unary operators)
    /// \param def       the definition of the operator
    void insert(
        IExpression::Operator op,
        IType const           *left_tp,
        IType const           *right_tp,
        Definition const      *def);

private:
    /// the allocator, hold it to ensure its life time
    mi::base::Handle<IAllocator> m_alloc;

    // the cache itself
    Op_cache m_cache;
};

/// Helper class to manage auto imports.
class Auto_imports {
public:
    struct Entry {
        Definition const *def;       ///< The foreign definition that mustbe auto-imported.
        Definition const *imported;  ///< The imported definition of the foreign one.

        /// Default constructor.
        Entry()
            : def(NULL), imported(NULL)
        {}

        /// Constructor.
        Entry(Definition const *def, Definition const *imp)
            : def(def), imported(imp)
        {
        }
    };
public:
    /// Constructor.
    explicit Auto_imports(IAllocator *alloc);

    /// Get the number of entries.
    size_t size() const { return m_imports.size(); }

    /// Check if the map is empty.
    bool empty() const { return m_imports.empty(); }

    /// Get the entry at given index.
    Entry &operator[](size_t idx) { return m_imports[idx]; }

    /// Find the entry for a given definition.
    Entry const *find(Definition const *def) const;

    /// Insert a new foreign definition.
    ///
    /// \param def  the new foreign definition to add
    ///
    /// \return true  if def was added, false if the definition was already registered
    bool insert(Definition const *def);
private:

    typedef ptr_hash_map<Definition const, size_t>::Type Index_map;
    typedef vector<Entry>::Type                          Import_vec;

    Index_map  m_index_map;
    Import_vec m_imports;
};

///
/// The combined name and type analysis.
///
class NT_analysis : public Analysis, ICallgraph_scc_visitor {
    typedef Analysis Base;
    friend class Optimizer;
    friend class Default_initializer_modifier;

    /// RAII like function scope.
    class Enter_function {
    public:
        Enter_function(NT_analysis &ana, Definition *def)
        : m_ana(ana)
        {
            ana.push_function(def);
        }

        ~Enter_function() { (void)m_ana.pop_function(); };
    
    private:
        NT_analysis &m_ana;
    };

    /// A signature entry for function overload resolution.
    struct Signature_entry {
        IType const *const *signature;
        bool const *const  bounds;
        size_t             sig_length;
        Definition const   *def;

        Signature_entry(
            IType const *const *signature,
            bool const *const  bounds,
            size_t             sig_length,
            Definition const *def)
        : signature(signature), bounds(bounds), sig_length(sig_length), def(def) {}
    };

    typedef Arena_list<Signature_entry>::Type Signature_list;

    /// RAII like cleanup callback.
    class Cleanup_scope {
    public:
        Cleanup_scope(NT_analysis *ana, void (NT_analysis::*cleanup)(void))
        : m_ana(ana), m_cleanup(cleanup)
        {
        }

        ~Cleanup_scope() { (m_ana->*m_cleanup)(); }

    private:
        NT_analysis        *m_ana;
        void (NT_analysis::*m_cleanup)(void);
    };

    /// Helper class for resource entries.
    class Resource_entry {
        friend class NT_analysis;
    public:
        /// Constructor.
        Resource_entry(
            IAllocator           *alloc,
            char const           *url,
            char const           *filename,
            int                  resource_tag,
            IType_resource const *type,
            bool const            exists)
        : m_url(url, alloc)
        , m_filename(filename, alloc)
        , m_res_tag(resource_tag)
        , m_type(type)
        , m_exists(exists)
        {
        }

    private:
        string               m_url;
        string               m_filename;
        int                  m_res_tag;
        IType_resource const *m_type;
        bool                 m_exists;
    };

public:
    /// Returns the definition of a symbol at the at a given scope.
    /// \param sym    the symbol
    /// \param scope  the scope
    Definition *get_definition_at_scope(const ISymbol *sym, Scope *scope) const;

    /// Returns the definition of a symbol at the current scope only.
    /// \param sym  the symbol
    Definition *get_definition_at_scope(const ISymbol *sym) const;

    /// Get the call graph.
    ///
    /// \note The life time of the call graph is bound to the lite time of its owning analysis.
    Call_graph const &get_call_graph() const { return m_cg; }

    /// If the given definition is auto-imported, return its imported definition.
    ///
    /// \param def   the definition
    Definition const *get_auto_imported_definition(Definition const *def) const;

    /// Run the name and type analysis on this module.
    void run();

private:
    /// Push a Definition on the function stack.
    ///
    /// \param def  the function definition to push
    void push_function(Definition *def);

    /// Pop the function stack and return the TOS.
    Definition *pop_function();

    /// Return the current function on the stack.
    Definition *tos_function() const;

    /// Returns true if we are inside a function.
    bool inside_function() const { return m_func_stack_pos > 0; }

    /// Enter one predefined constant into the current scope.
    Definition *enter_builtin_constant(
        const ISymbol *sym,
        const IType   *type,
        const IValue  *val);

    /// Enter all predefined constants into the current scope.
    void enter_builtin_constants();

    /// Enter builtin annotations for stdlib modules.
    void enter_builtin_annotations();

    /// Enter builtin annotations for native modules.
    void enter_native_annotations();

    /// Create an exported constant declaration for a given value and add it to the current module.
    void create_exported_decl(ISymbol const *sym, IValue const *val, int line);

    /// Enter builtin constants for stdlib modules.
    void enter_stdlib_constants();

    /// Check if the given type has a const constructor.
    ///
    /// \param type  the type to check
    bool has_const_default_constructor(IType const *type) const;

    /// Create the default constructors and operators for a given struct type.
    ///
    /// \param s_type       the struct type
    /// \param sym          the type name
    /// \param def          the definition of the type
    /// \param struct_decl  the declaration of the struct type
    void create_default_members(
        IType_struct const             *s_type,
        ISymbol const                  *sym,
        Definition const               *def,
        IDeclaration_type_struct const *struct_decl);

    /// Create the default operators for a given enum type.
    ///
    /// \param e_type     the enum type
    /// \param sym        the type name
    /// \param def        the definition of the type
    /// \param enum_decl  the declaration of the enum type
    void create_default_operators(
        IType_enum const       *e_type,
        ISymbol const          *sym,
        Definition const       *def,
        IDeclaration_type_enum *enum_decl);

    /// Create the default constructors enum type.
    ///
    /// \param e_type     the enum type
    /// \param sym        the type name
    /// \param def        the definition of the type
    /// \param enum_decl  the declaration of the enum type
    void create_default_constructors(
        IType_enum const       *e_type,
        ISymbol const          *sym,
        Definition const       *def,
        IDeclaration_type_enum *enum_decl);

    /// Check archive dependencies on a newly imported module.
    ///
    /// \param imp_mod   the imported module
    /// \param pos       the position of the import for a warning
    void check_imported_module_dependencies(
        Module const   *imp_mod,
        Position const &pos);

    /// Find and load a module to if possible.
    ///
    /// \param rel_name       the (relative) name of the module
    /// \param ignore_last    if true, the last simple name of the rel_name
    ///                       is not part of the module name
    ///
    /// \return the loaded module or NULL
    Module const *load_module_to_import(
        IQualified_name const *rel_name,
        bool                  ignore_last);

    /// Check if the given imported definition is a re-export, if true, add its module
    /// to the current import table and return the import index of the owner module.
    ///
    /// \param imported  an imported definition from another module
    /// \param from      the module imported is from
    /// \param from_idx  the import index of the from modules inside the current one
    ///
    /// \returns the import index of the original owner
    size_t handle_reexported_entity(
        Definition const *imported,
        Module const     *from,
        size_t           from_idx);

    /// Enter the relative scope starting at the current scope
    /// of an imported module given by the module name and a prefix skip.
    ///
    /// \param ns           a namespace name
    /// \param prefix_skip  len of prefix to skip from name space
    /// \param ignore_last  if true, ignore the last component
    ///
    /// \note Creates the scope if it does not exists so far.
    Scope *enter_import_scope(
        IQualified_name const *ns,
        int                   prefix_skip,
        bool                  ignore_last);

    /// Import a qualified entity or a module.
    ///
    /// \param rel_name  the (relative) name of the entity given at the import declaration
    /// \param err_pos   position for error messages
    void import_qualified(
        IQualified_name const *rel_name,
        Position const        &err_pos);

    /// Import a definition from a module into the current scope.
    ///
    /// \param from         the module to import from
    /// \param from_idx     the import index of the from module
    /// \param def          the definition to import
    /// \param is_exported  if true, re-export the imported definition
    /// \param err_pos      position for error messages
    ///
    /// \return NULL if successfully imported, the definition that clashes
    ///              otherwise
    Definition const *import_definition(
        Module const     *from,
        size_t           from_idx,
        Definition const *def,
        bool             is_exported,
        Position const   &err_pos);

    /// Import a complete module.
    ///
    /// \param from         the module to import
    /// \param ns           the namespace under which the entities are imported
    /// \param prefix_skip  len of prefix to skip from namespace
    /// \param err_pos      position for error messages
    void import_all_definitions(
        Module const          *from,
        IQualified_name const *ns,
        int                   prefix_skip,
        Position const        &err_pos);

    /// Import a type scope.
    ///
    /// \param imported     the definition to import
    /// \param from         the modules imported belongs to
    /// \param from_idx     the import index of module from in the current module
    /// \param new_def      the already created new definition for imported
    /// \param is_exported  if true, re-export all entities
    /// \param err_pos      position for error messages
    void import_type_scope(
        Definition const *imported,
        Module const     *from,
        size_t           from_idx,
        Definition       *new_def,
        bool             is_exported,
        Position const   &err_pos);

    /// Import all entities of a scope (and possible sub-scopes) into the current scope.
    ///
    /// \param scope        the top scope to import
    /// \param from         the module scope belongs to
    /// \param from_idx     the import index of module from in the current module
    /// \param is_exported  if true, re-export all entities
    /// \param forced       if true, import ALL entities, not only exported once
    /// \param err_pos      position for error messages
    void import_scope_entities(
        Scope const    *scope,
        Module const   *from,
        size_t         from_idx,
        bool           is_exported,
        bool           forced,
        Position const &err_pos);

    /// Import an entity from a module (qualified and unqualified).
    ///
    /// \param rel_name     the (relative) name of the module given at the import declaration
    /// \param imp_mode     the module from which the entity is imported
    /// \param name         the name of the entity to import
    /// \param is_exported  if true, re-export the imported entities
    ///
    /// \return true on success
    bool import_entity(
        IQualified_name const *rel_name,
        Module const          *imp_mod,
        IQualified_name const *name,
        bool                  is_exported);

    /// Import a qualified entity from a module.
    ///
    /// \param name         the name of the entity to import
    /// \param imp_mod      the module from which the entity is imported
    /// \param prefix_skip  len of prefix to skip from name space
    ///
    /// \return true on success
    bool import_qualified_entity(
        IQualified_name const *name,
        Module const          *imp_mod,
        int                   prefix_skip);

    /// Import all exported entities from a module.
    ///
    /// \param rel_name     the (relative) name of the module given at the import declaration
    /// \param imp_mode     the module from which the entities are imported
    /// \param is_exported  if true, re-export the imported entities
    /// \param err_pos      position for error messages
    void import_all_entities(
        IQualified_name const *rel_name,
        Module const          *imp_mod,
        bool                  is_exported,
        Position const        &err_pos);

    /// Checks if the given type is allowed for material parameter types.
    ///
    /// \param type  the type to check
    ///
    /// \returns  the forbidden type or NULL if the type is ok
    IType const *has_forbidden_material_parameter_type(
        IType const *type);

    /// Checks if the given type is allowed for function return types.
    ///
    /// \param type        the type to check
    /// \param pos         position for error report
    /// \param func_name   the name of the function to check
    /// \param is_std_mod  a standard module is compiled (rules are relaxed)
    ///
    /// \return true if type is an allowed return type, false otherwise
    bool check_function_return_type(
        IType const    *type,
        Position const &pos,
        ISymbol const  *func_name,
        bool           is_std_mod);

    /// Check (and copy) default parameters from prototype.
    ///
    /// \param proto_def  the prototype definition
    /// \param curr_def   the current definition
    ///
    /// If a prototype is given, default parameters can only be defined at the prototype.
    void check_prototype_default_parameter(Definition const *proto_def, Definition *curr_def);

    /// Check if a function is redeclared or redefined.
    ///
    /// \param curr_def        the definition to check
    ///
    /// \return the first prototype of curr_def if any
    Definition const *check_function_redeclaration(
        Definition *curr_def);

    /// Check if function annotations are only present at the first declaration.
    ///
    /// \param curr_def        the definition to check
    /// \param proto_def       its first prototype if any
    void check_function_annotations(
        Definition const *curr_def,
        Definition const *proto_def);

    /// Check an expression against a range.
    ///
    /// \param anno   a range annotation
    /// \param expr   the expression to check
    /// \param code   the error code that should be generated if range check fails
    /// \param extra  if non-NULL, the fourth parameter for the error message
    void check_expression_range(
        IAnnotation const *anno,
        IExpression const *expr,
        Compilation_error code,
        ISymbol const     *extra);

    /// Check the parameter range for the given parameter if a hard_range was specified.
    ///
    /// \param param      the parameter to check
    /// \param init_expr  the initializer expression of this parameter
    void check_parameter_range(
        IParameter const  *param,
        IExpression const *init_expr);

    /// Check the field range for the given structure field if a hard_range was specified.
    ///
    /// \param block      the annotation block of the struct field to check
    /// \param init_expr  the initializer expression of this field
    void check_field_range(
        IAnnotation_block const *block,
        IExpression const       *init_expr);

    /// Check the field range for the given structure field value if a hard_range was specified.
    ///
    /// \param f_def  the field to check
    /// \param expr   the expression that is assigned to this field
    void check_field_assignment_range(
        IDefinition const *f_def,
        IExpression const *expr);

    /// Declare a new function.
    ///
    /// \param fkt_decl  the associated function declaration
    void declare_function(IDeclaration_function *fkt_decl);

    /// Declare a new function preset.
    ///
    /// \param fkt_decl  the associated function declaration
    void declare_function_preset(IDeclaration_function *fkt_decl);

    /// Check a parameter initializer.
    ///
    /// \param init_expr   the initializer expression
    /// \param p_type      the parameter type
    /// \param pt_name     the type name of the parameter
    /// \param param       the parameter that is initialized
    IExpression const *handle_parameter_initializer(
        IExpression const *init_expr,
        IType const       *p_type,
        IType_name const  *pt_name,
        IParameter const  *param);

    /// Check restrictions of material parameter type.
    ///
    /// \param p_type  the parameter type
    /// \param pos     the position of the parameter type declaration
    ///
    /// \returns true if the type is allowed, false otherwise
    bool check_material_parameter_type(
        IType const    *p_type,
        Position const &pos);

    /// Handle all enable_if annotations on the given parameter.
    ///
    /// \param  the parameter (function or material)
    void handle_enable_ifs(IParameter const *param);

    /// Declare a new material.
    ///
    /// \param fkt_decl  the associated material declaration
    void declare_material(IDeclaration_function *mat_decl);

    /// Collect the new default values for a preset.
    ///
    /// \param def           the definition of the preset instance
    /// \param call          the call expression containing new defaults
    /// \param new_defaults  array filled at return
    ///
    /// \returns true on success, false if errors were found
    bool collect_preset_defaults(
        Definition const         *def,
        IExpression_call const   *call,
        VLA<IExpression const *> &new_defaults);

    /// Declare a new material preset.
    ///
    /// \param fkt_decl  the associated material declaration
    void declare_material_preset(IDeclaration_function *mat_decl);

    /// Handle scope transitions for a select expression name lookup.
    ///
    /// \param sel_expr  the select expression
    void handle_select_scopes(IExpression_binary *sel_expr);

    // Resolve an overload.
    /// Resolve an overload.
    ///
    /// \param call_expr         a binary or unary expression
    /// \param def_list          the list of all possible overloads
    /// \param is_overloaded     true def_list has more than one element
    /// \param arg_types         array of argument types
    /// \param num_args          number of arguments
    /// \param arg_mask          mask of valid operand types
    ///
    /// \return a list of matching elements from def_list
    Definition_list resolve_operator_overload(
        IExpression const      *call_expr,
        Definition_list        &def_list,
        IType const            *arg_types[],
        size_t                 num_args,
        unsigned               &arg_mask);

    /// Find the definition of a binary or unary operator.
    ///
    /// \param op_call    a binary or unary expression
    /// \param op         the operator kind
    /// \param arg_types  argument types of the operator
    /// \param arg_count  argument count of the operator
    Definition const *find_operator(
        IExpression           *op_call,
        IExpression::Operator op,
        IType const           *arg_types[],
        size_t                arg_count);

    /// Check if there is an array assignment operator.
    ///
    /// \param op_call    a binary expression
    /// \param left_tp    left argument type of the operator
    /// \param right_tp   right argument type of the operator
    ///
    /// \return NULL if the arguments are not both arrays, else the definition
    ///        (might be the error definition)
    Definition const *find_array_assignment(
        IExpression_binary *op_call,
        IType const        *left_tp,
        IType const        *right_tp);

    /// Find the Definition of an conversion constructor or an conversion operator.
    ///
    /// \param from_tp        the source type of the conversion
    /// \param to_tp          the destination type of the conversion
    /// \param implicit_only  if true, search only for implicit conversions
    ///
    /// \return the Definition of the constructor/conversion operator or NULL
    ///         if a conversion is not possible.
    Definition *do_find_conversion(
        IType const *from_tp,
        IType const *to_tp,
        bool        implicit_only);

    /// Find the Definition of an implicit conversion constructor or an conversion
    /// operator.
    ///
    /// \param from_tp  the source type of the conversion
    /// \param to_tp    the destination type of the conversion
    ///
    /// \return the Definition of the constructor/conversion operator or NULL
    ///         if an implicit conversion is not possible.
    Definition *find_implicit_conversion(
        IType const *from_tp,
        IType const *to_tp);

    /// Find the Definition of an implicit or explicit conversion constructor or an conversion
    /// operator.
    ///
    /// \param from_tp  the source type of the conversion
    /// \param to_tp    the destination type of the conversion
    ///
    /// \return the Definition of the constructor/conversion operator or NULL
    ///         if a conversion is not possible.
    Definition *find_type_conversion(
        IType const *from_tp,
        IType const *to_tp);

    /// Convert a given expression to the destination type using a given conversion constructor.
    ///
    /// \param expr              the expression to convert
    /// \param def               the definition of a conversion constructor
    /// \param warn_if_explicit  if true, warn if def is explicit_warn marked
    ///
    /// \return a constructor call
    IExpression_call *create_type_conversion_call(
        IExpression const *expr,
        Definition const  *def,
        bool              warn_if_explicit);

    /// Convert a given expression to the destination type if an implicit
    /// constructor is available.
    ///
    /// \param expr      the expression to convert
    /// \param dst_type  the destination type
    ///
    /// \return a constructor call or NULL if no conversion is available
    IExpression *convert_to_type_implicit(
        IExpression const *expr,
        IType const       *dst_type);

    /// Convert a given expression to the destination type if an implicit or explicit
    /// constructor is available.
    ///
    /// \param expr      the expression to convert
    /// \param dst_type  the destination type
    ///
    /// \return a constructor call or NULL if no conversion is available
    IExpression *convert_to_type_explicit(
        IExpression const *expr,
        IType const       *dst_type);

    /// Convert a given const expression to the destination type if an implicit
    /// constructor is available.
    ///
    /// \param expr      the const expression to convert
    /// \param dst_type  the destination type
    ///
    /// \return a literal or NULL if no conversion is available
    IExpression *convert_to_type_and_fold(
        IExpression const *expr,
        IType const       *dst_type);

    /// Check if the given expression represents a boolean condition.
    ///
    /// \param cond              the expression to check
    /// \param has_error         set to true if an error was reported, otherwise unchanged
    /// \param inside_enable_if  if true, we are checking the condition of an enable_if
    ///
    /// \return a constructor expression if an implicit type cast was made explicit, NULL else.
    IExpression *check_bool_condition(
        IExpression const *cond,
        bool              &has_error,
        bool              inside_enable_if);

    /// Calculate the since and removed version for a given definition.
    ///
    /// \param def  the definition
    void calc_mdl_versions(Definition *def);

    /// Export the given definition.
    ///
    /// \param def  the definition to export
    void export_definition(Definition *def);

    /// Compare two signature entries representing functions for "specific-ness".
    ///
    /// \param a  first function entry
    /// \param b  second function entry
    ///
    /// \return true if a has some is more specific parameters than b
    bool is_more_specific(Signature_entry const &a, Signature_entry const &b);

    /// Given a list and a (call) signature, kill any less specific definition from the list.
    /// If new_sig is less specific then an entry in the list return false
    /// else true.
    ///
    /// \param list     the candidate list
    /// \param new_sig  a new candidate
    ///
    /// \return false : the candidate list contains an already more specific version,
    ///                 drop the new candidate
    ///         true  : add the new candidate
    bool kill_less_specific(Signature_list &list, Signature_entry const &new_sig);

    /// Check if a parameter type is already bound.
    ///
    /// \param abs_type  a deferred array size type
    bool is_bound_type(IType_array const *abs_type) const;

    /// Bind the given deferred sized array type to another array type.
    ///
    /// \param deferred_type  a deferred array size type
    /// \param type           the immediate or deferred sized array type that is bound
    void bind_array_type(IType_array const *deferred_type, IType_array const *type);

    /// Return the bound type for a deferred type.
    ///
    /// \param type  a type
    ///
    /// \return the bound array type or type itself it if was not bound
    IType const *get_bound_type(IType const *type);

    /// Clear all bindings of deferred sized array types.
    void clear_type_bindings();

    /// Check if it is possible to assign an argument type to the parameter
    /// type of a call.
    ///
    /// \param param_type          the type of a function parameter
    /// \param arg_type            the type of the corresponding call argument
    /// \param new_bound           set to true if a new type bound was executed
    /// \param need_explicit_conv  true, if an explicit conversion (EXPLICIT_WARN) is necessary
    bool can_assign_param(
        IType const *param_type,
        IType const *arg_type,
        bool        &new_bound,
        bool        &need_explicit_conv);

    /// Returns a short signature from a (function) definition.
    ///
    /// \param def  the function (or constructor) definition
    string get_short_signature(Definition const *def) const;

    /// Check if a parameter name exists in the parameters of a candidate.
    ///
    /// \param candidate  a candidate definition
    /// \param type       the function type of candidate
    /// \param name       the name of the parameter
    bool is_known_parameter(
        Definition const     *candidate,
        IType_function const *type,
        ISymbol const        *name) const;

    /// Resolve an overload.
    ///
    /// \param error_reported    set to true if an error was reported
    /// \param call_expr         a call expression
    /// \param def_list          the list of all possible overloads
    /// \param is_overloaded     true def_list has more than one element
    /// \param arg_types         array of argument types
    /// \param num_pos_args      number of positional arguments
    /// \param num_named_args    number of named arguments
    /// \param name_arg_indexes  maps names to argument indexes (into arg_types)
    /// \param arg_index         if >= 0, the index of the array constructor that is currently
    ///                          processed
    ///
    /// \return a list of matching elements from def_list
    Definition_list resolve_overload(
        bool                   &error_reported,
        IExpression_call const *call_expr,
        Definition_list        &def_list,
        bool                   is_overloaded,
        IType const            *arg_types[],
        size_t                 num_pos_args,
        size_t                 num_named_args,
        Name_index_map const   *name_arg_indexes,
        int                    arg_index = -1);

    /// Add a "candidates are" notes to the current error message.
    ///
    /// \param overloads  the overload set
    void add_candidates(
        Definition_list const &overloads);

    /// Resolve overloads for a call.
    ///
    /// \param def             the overload set definition
    /// \param call            the call expression
    /// \param bound_to_scope  if true, call is bound to def's scope
    Definition const *find_overload(
        Definition const *def,
        IExpression_call *call,
        bool             bound_to_scope);

    /// Resolve an annotation overload.
    ///
    /// \param error_reported    set to true if an error was reported
    /// \param anno              an annotation
    /// \param def_list          the list of all possible overloads
    /// \param is_overloaded     true def_list has more than one element
    /// \param arg_types         array of argument types
    /// \param num_pos_args      number of positional arguments
    /// \param num_named_args    number of named arguments
    /// \param name_arg_indexes  maps names to argument indexes (into arg_types)
    ///
    /// \return a list of matching elements from def_list
    Definition_list resolve_annotation_overload(
        bool                   &error_reported,
        IAnnotation const      *anno,
        Definition_list        &def_list,
        bool                   is_overloaded,
        IType const            *arg_types[],
        size_t                 num_pos_args,
        size_t                 num_named_args,
        Name_index_map const   *name_arg_indexes);

    /// Resolve overloads for an annotation.
    ///
    /// \param def             the overload set definition
    /// \param anno            the annotation
    /// \param bound_to_scope  if true, anno is bound to def's scope
    Definition const *find_annotation_overload(
        Definition const *def,
        IAnnotation      *anno,
        bool             bound_to_scope);

    /// Report an error type error.
    ///
    /// \param type  the (forbidden) array element type
    /// \param pos   the error position
    void report_array_type_error(IType const *type, Position const &pos);

    /// Check if the given call is an array copy constructor.
    ///
    /// \param call            the call expression
    bool handle_array_copy_constructor(
        IExpression_call const *call);

    /// Handle an array constructor.
    ///
    /// \param def             the overload set definition for the elementary constructor
    /// \param call            the call expression
    bool handle_array_constructor(
        Definition const *def,
        IExpression_call *call);

    /// Find constructor for an initializer.
    ///
    /// \param type       the type of the entity to initialize
    /// \param tname      the typename of the type
    /// \param init_expr  the initializer of the variable or NULL
    /// \param pos        the position of the expression that requires the constructor
    ///
    /// \return new initializer (constructor call)
    IExpression const *find_init_constructor(
        IType const        *type,
        IType_name const   *tname,
        IExpression const  *init_expr,
        Position const     &pos);

    typedef ptr_hash_map<IDefinition const, int>::Type Origin_map;

    /// Check if an expression has side effects.
    ///
    /// \param origins    helper map to store origins of possible side effects
    /// \param expr       the expression to check.
    /// \param index      original argument index of this expression
    /// \param dep_index  index of an argument that makes an assignment
    ///
    /// \return the definition of an entity that caused the dependence or NULL
    ///
    /// \note: In MDL, only assignments have a side effect, so an assigned
    ///        entity is returned.
    IDefinition const *has_side_effect(
        Origin_map        &origins,
        IExpression const *expr,
        int               index,
        int               &dep_index);

    /// Check the the arguments of a call are independent on evaluation order.
    ///
    /// \param call  the call to check
    void check_arguments_evaluation_order(
        IExpression_call const *call);

    /// Check argument range.
    ///
    /// \param decl  the declaration of the called entity
    /// \param expr  the call argument to check
    /// \param idx   the parameter index of the argument in the entity
    void check_argument_range(
        IDeclaration const *decl,
        IExpression const  *expr,
        int                idx);

    /// Replace default struct constructors by elemental constructors.
    ///
    /// \param callee_def        the definition of the called function
    /// \param call              the call, might be modified
    ///
    /// \return the new callee definition
    Definition const *reformat_default_struct_constructor(
        Definition const *callee_def,
        IExpression_call *call);

    /// Reformat and reorder the arguments of a call.
    ///
    /// - make all implicit type conversions explicit
    /// - make all default parameters explicit
    /// - for functions: turn all named parameters into positional
    /// - for material/df constructors: turn all positional parameters into named
    ///
    /// \param callee_def        the definition of the called function
    /// \param call              the call, might be modified
    /// \param pos_arg_types     the types of the original call's positional arguments
    /// \param pos_arg_count     number of positional arguments
    /// \param name_arg_indexes  maps names of named arguments to call's argument indexes
    /// \param arguments         original arguments of the call
    /// \param named_arg_count   number of named arguments
    void reformat_arguments(
        Definition const     *callee_def,
        IExpression_call     *call,
        IType const          *pos_arg_types[],
        size_t               pos_arg_count,
        Name_index_map const &name_arg_indexes,
        IArgument const      *arguments[],
        size_t               named_arg_count);

    /// Reformat and reorder the arguments of an annotation.
    ///
    /// - make all implicit type conversions explicit
    /// - make all default parameters explicit
    /// - turn all named parameters into positional
    ///
    /// \param anno_def          the definition of the annotation
    /// \param anno              the annotation, might be modified
    /// \param pos_arg_types     the types of the original anno's positional arguments
    /// \param pos_arg_count     number of positional arguments
    /// \param name_arg_indexes  maps names of named arguments to anno's argument indexes
    /// \param arguments         original arguments of the call
    /// \param named_arg_count   number of named arguments
    void reformat_annotation_arguments(
        Definition const     *anno_def,
        IAnnotation          *anno,
        IType const          *pos_arg_types[],
        size_t               pos_arg_count,
        Name_index_map const &name_arg_indexes,
        IArgument const      *arguments[],
        size_t               named_arg_count);

    /// Convert one argument of an array constructor call.
    ///
    /// \param call    the call, might be modified
    /// \param idx     the index of the argument to be converted
    /// \param e_type  the arrays element type
    ///
    /// \return false if conversion is not possible
    bool convert_array_cons_argument(
        IExpression_call     *call,
        int                  idx,
        IType const          *e_type);

    /// visit all default arguments of the material constructors.
    ///
    /// \param visitor  the visitor to be used
    void visit_material_default(Module_visitor &visitor);

    /// Update the call graph by a call to a function.
    ///
    /// \param callee  the definition of the callee
    void update_call_graph(Definition const *callee);

    /// Called for every called function in the semantic analysis.
    ///
    /// \param def  the definition of the called function/method
    void check_called_function_existance(Definition const *def);

    /// Called for every global definition.
    ///
    /// \param def  the global definition
    void check_used_function(Definition const *def);

    /// Report a recursion.
    ///
    /// \param scc  the strongly coupled component that form a recursion chain
    void process_scc(Call_node_vec const &scc) MDL_FINAL;

    /// Check that all called functions do exists.
    void check_called_functions();

    /// Dump a call graph to a file.
    ///
    /// \param alloc        the allocator
    /// \param module_name  the absolute name of the owner module
    /// \param cg           the call graph
    static void dump_call_graph(
        IAllocator       *alloc,
        char const       *module_name,
        Call_graph const &cg);

    /// Check that all exported entities exist.
    void check_exported_for_existance();

    /// Check restriction on the given file path.
    ///
    /// \param path  a file path
    /// \param pos   the position of the value inside the current module
    ///
    /// \return false if this is an invalid path
    bool check_file_path(char const *path, Position const &pos);

    /// Handle resource constructors.
    ///
    /// \param call_expr  a call representing an resource constructor call
    ///
    /// \return the associated resource value
    IExpression *handle_resource_constructor(IExpression_call *call_expr);

    /// Process a resource url.
    ///
    /// \param val       a string or resource value representing the URL
    /// \param lit       the owning literal expression of the value
    /// \param res_type  the resource type of the resource
    void handle_resource_url(
        IValue const         *val,
        IExpression_literal  *lit,
        IType_resource const *res_type);

    /// Add a new resource entry.
    ///
    /// \param url           an absolute MDL url
    /// \param file_name     the OS specific file name of the url
    /// \param resource_tag  if non-NULL, the TAG of a in-memory resource
    /// \param type          the type of the resource
    /// \param exists        if FALSE, the resource does not exists
    void add_resource_entry(
        char const           *url,
        char const           *file_name,
        int                  resource_tag,
        IType_resource const *type,
        bool                 exists);

    /// Copy the given resolver messages to the current module.
    ///
    /// \param messages     the resolver messages
    /// \param is_resource  true, if the messages come from resolving a resource
    void copy_resolver_messages_to_module(
        Messages_impl const &messages,
        bool is_resource);

    /// Try to import a symbol from a given module.
    ///
    /// \param imp_mod  the original module of this symbol
    /// \param sym      the symbol to import
    ///
    /// \return true of success, false if importing failed
    bool auto_import(
        Module const  *imp_mod,
        ISymbol const *sym);

    /// Add an import declaration to the current module.
    ///
    /// \param imp_mod  the module from which to import
    /// \param sym      the symbol to import
    void add_auto_import_declaration(
        Module const  *imp_mod,
        ISymbol const *sym);

    /// Insert all auto imports from default initializers.
    void fix_auto_imports();

    /// Return the Type from a IType_name, handling errors if the IType_name
    /// does not name a type.
    ///
    /// \param type_name  an type name
    ///
    /// \return the type, if type_name does not name a type returns the error type
    ///         and enters it into type_name
    IType const *as_type(IType_name const *type_name);

    /// Get a array size from a definition.
    ///
    /// \param def       the definition
    /// \param abs_name  if non-NULL, create a new size under this name
    IType_array_size const *get_array_size(IDefinition const *def, ISymbol const *abs_name);

    /// Collect all builtin entities and put them into the current module.
    void collect_builtin_entities();

    /// Calculate, which function uses the state.
    void calc_state_usage();


    /// Enter a new semantic version for given module.
    ///
    /// \param module      the module that will be modified
    /// \param major       the major version
    /// \param minor       the minor version
    /// \param patch       the patch version
    /// \param prerelease  the pre-release string
    /// \param pos         the position of the annotation
    ///
    /// \return the position of a previous set semantic number or NULL
    Position const *set_module_sem_version(
        Module         &module,
        int            major,
        int            minor,
        int            patch,
        char const     *prelease,
        Position const &pos);

    /// Enter a new semantic dependency for the current module.
    ///
    /// \param anno  the dependency annotation
    ///
    /// \return true if the dependency is sound, false otherwise
    bool check_module_dependency(
        IAnnotation const *anno);

    /// Given a range annotation definition find the overload for the given type.
    ///
    /// \param def   a valid range overload
    /// \param type  the parameter type for the annotation overload to find
    Definition const *select_range_overload(
        Definition const *def,
        IType const      *type);

    /// Handle known annotations.
    ///
    /// \param def   the definition of the known annotation
    /// \param anno  the annotation itself
    ///
    /// \returns the error definition if the known definition has errors, def else
    Definition const *handle_known_annotation(
        Definition const *def,
        IAnnotation      *anno);

    /// Check whether the condition expression of an enable_if annotation conforms to the
    /// MDL specification, i.e. uses the specified subset of MDL expressions only
    /// and does not reference the annotated parameter.
    ///
    /// \param expr        the condition expression to check
    /// \param param_sym   the parameter symbol which is annotated
    void check_enable_if_condition(
        IExpression const *expr,
        ISymbol const     *param_sym);

    /// Check the given type for any possible performance restrictions and warn if necessary.
    ///
    /// \param type  the type to check
    /// \param pos   position to issue the warning
    ///
    /// \return \c type
    IType const *check_performance_restriction(
        IType const    *type,
        Position const &pos);

    /// Check if the given source type can be casted into the destination type.
    ///
    /// \param pos                the source code position of the cast
    /// \param src_type           the source type
    /// \param dst_type           the destination type
    /// \param dst_is_incomplete  true, if dst_type is the element type of an incomplete array
    /// \param report_error       if true, report errors, else only return value
    ///
    /// \return NULL if cannot be casted, the result type otherwise
    IType const *check_cast_conversion(
        Position const &pos,
        IType const    *src_type,
        IType const    *dst_type,
        bool           dst_is_incomplete,
        bool           report_error);

    /// Handle a cast expression.
    ///
    /// \param cast_expr  the cast expression
    void handle_cast_expression(IExpression_unary *cast_expr);

    bool pre_visit(IDeclaration_import *import_decl) MDL_OVERRIDE;

    bool pre_visit(IDeclaration_constant *con_decl) MDL_OVERRIDE;

    bool pre_visit(IDeclaration_type_enum *type_enum) MDL_OVERRIDE;

    bool pre_visit(IDeclaration_type_struct *type_struct) MDL_OVERRIDE;

    bool pre_visit(IDeclaration_annotation *anno_decl) MDL_OVERRIDE;

    bool pre_visit(IDeclaration_type_alias *alias_decl) MDL_OVERRIDE;

    bool pre_visit(IParameter *param) MDL_OVERRIDE;

    bool pre_visit(IDeclaration_function *fun_decl) MDL_OVERRIDE;

    bool pre_visit(IDeclaration_module *mod_decl) MDL_OVERRIDE;

    bool pre_visit(IDeclaration_namespace_alias *alias) MDL_OVERRIDE;

    bool pre_visit(IDeclaration_variable *var_decl) MDL_OVERRIDE;

    bool pre_visit(IStatement_compound *block) MDL_OVERRIDE;
    void post_visit(IStatement_compound *block) MDL_OVERRIDE;

    void post_visit(IStatement_if *if_stmt) MDL_OVERRIDE;

    bool pre_visit(IStatement_switch *switch_stmt) MDL_OVERRIDE;

    void post_visit(IStatement_while *while_stmt) MDL_OVERRIDE;

    void post_visit(IStatement_do_while *do_while_stmt) MDL_OVERRIDE;

    bool pre_visit(IStatement_for *for_stmt) MDL_OVERRIDE;
    void post_visit(IStatement_for *for_stmt) MDL_OVERRIDE;

    void post_visit(IStatement_return *ret_stmt) MDL_OVERRIDE;

    IExpression *post_visit(IExpression_invalid *inv_expr) MDL_OVERRIDE;

    IExpression *post_visit(IExpression_literal *lit) MDL_OVERRIDE;

    IExpression *post_visit(IExpression_reference *ref) MDL_OVERRIDE;

    IExpression *post_visit(IExpression_unary *un_expr) MDL_OVERRIDE;

    bool pre_visit(IExpression_binary *bin_expr) MDL_OVERRIDE;
    IExpression *post_visit(IExpression_binary *bin_expr) MDL_OVERRIDE;

    IExpression *post_visit(IExpression_conditional *cond_expr) MDL_OVERRIDE;

    bool pre_visit(IExpression_let *let_expr) MDL_OVERRIDE;

    bool pre_visit(IExpression_call *call_expr) MDL_OVERRIDE;
    IExpression *post_visit(IExpression_call *call_expr) MDL_OVERRIDE;

    void post_visit(IQualified_name *qual_name) MDL_OVERRIDE;

    bool pre_visit(IType_name *type_name) MDL_OVERRIDE;

    bool pre_visit(IAnnotation *anno) MDL_OVERRIDE;

    void post_visit(IAnnotation_block *anno) MDL_OVERRIDE;

public:
    /// Constructor.
    explicit NT_analysis(
        MDL              *compiler,
        Module           &module,
        Thread_context   &ctx,
        IModule_cache    *cache);

private:
    /// The currently processed "preset-overload".
    IExpression_call const *m_preset_overload;

    /// Set if we analyze a standard module;
    bool m_is_stdlib;

    /// Set while we are in an expected constant expression.
    bool m_in_expected_const_expr;

    /// If true, do not issue an error for undefined array size.
    bool m_ignore_deferred_size;

    /// Set while we are visiting an array size.
    bool m_in_array_size;

    /// If set to true, definitions for parameters will be marked as used.
    bool m_params_are_used;

    /// If set to true, we are inside a material constructor.
    bool m_inside_material_constr;

    /// If set to true, let expressions are allowed.
    bool m_allow_let_expression;

    /// If set to true, we are inside a let_expression declarations.
    bool m_inside_let_decls;

    /// If set to true, we are inside the material defaults visitor.
    bool m_inside_material_defaults;

    /// If set, we are inside a perameter initializer context;
    bool m_inside_param_initializer;

    /// If set, the current function can throw a bounds exception.
    bool m_can_throw_bounds;

    /// If set, the current function can throw a division by zero exception.
    bool m_can_throw_divzero;

    /// If set, the call graph will be dumped.
    bool m_opt_dump_cg;

    /// If set, resource file paths are kept as is (and are not converted to absolute file paths).
    bool m_opt_keep_original_resource_file_paths;

    /// If set, we are visiting an annotation on the current module.
    bool m_is_module_annotation;

    /// If set, we are visiting an annotation on a return type.
    bool m_is_return_annotation;

    /// If true, the current module has the array assignment operator.
    bool m_has_array_assignment;


    /// The current module cache.
    IModule_cache *m_module_cache;

    /// The current annotated definition.
    Definition *m_annotated_def;

    /// Index of the next created parameter.
    int m_next_param_idx;

    /// Maximum depth of the function stack.
    size_t m_func_stack_depth;

    /// Current TOS of the function stack.
    size_t m_func_stack_pos;

    /// The function stack.
    vector<Definition *>::Type m_func_stack;

    /// A cache for speeding up the operator call resolver.
    Operator_lookup_cache m_op_cache;

    /// The call graph.
    Call_graph m_cg;

    typedef ptr_hash_map<IType const, IType const *>::Type                       Bind_type_map;
    typedef ptr_hash_map<IType_array_size const, int>::Type                      Bind_size_map;
    typedef ptr_hash_map<IType_array_size const, IType_array_size const *>::Type Bind_symbol_map;

    /// Type bindings for overload resolution.
    Bind_type_map   m_type_bindings;
    Bind_size_map   m_size_bindings;
    Bind_symbol_map m_sym_bindings;

    typedef ptr_hash_map<IDefinition const, IType_array_size const *>::Type Array_size_map;

    /// All known abstract array sizes of this module.
    Array_size_map m_array_size_map;

    /// All exported declarations only, to be checked later.
    vector<Definition *>::Type m_exported_decls_only;

    /// The auto-import map.
    Auto_imports m_auto_imports;

    /// List of definitions whose initializer must go through auto-import fix.
    vector<Definition *>::Type m_initializers_must_be_fixed;

    /// If the current analyzed module had a semantic version, its Position.
    Position const *m_sema_version_pos;

    typedef map<Resource_table_key, Resource_entry>::Type Resource_table;

    /// Resource entries of the current module.
    Resource_table m_resource_entries;
};

struct Resource_table_key
{
    const string key_string;
    const int key_tag;

    Resource_table_key(string key_string, int key_tag)
        : key_string(key_string)
        , key_tag(key_tag)
    {
    }

    bool operator==(const Resource_table_key& other) const 
    {
        if (key_tag != 0 && key_tag == other.key_tag)
            return true;

        return key_string == other.key_string;
    }

    bool operator<(const Resource_table_key& other) const
    {
        if (key_tag != 0 && key_tag != other.key_tag)
            return key_tag < other.key_tag;

        return key_string < other.key_string;
    }
};

///
/// Extra semantic analysis pass.
///
/// This pass calculates and reports:
/// - dead code
/// - checks switch cases for completeness
/// - checks break and continue contexts
/// - check for missing return statements
///
class Sema_analysis : public Analysis
{
public:
    /// Run the semantic analysis on a module.
    void run();

    /// Retrieve the statement analysis data.
    Stmt_info_data const &get_statement_info_data() const { return m_stmt_info_data; }

    /// Constructor.
    explicit Sema_analysis(
        MDL            *compiler,
        Module         &module,
        Thread_context &ctx);

private:
    /// Report unused entities.
    void report_unused_entities();

    typedef ptr_hash_set<IType const>::Type Bad_type_set;

    /// Check that the given type is exported.
    ///
    /// \param user       an exported entity that uses the type
    /// \param type       the type that must be exported
    /// \param bad_types  the set of bad types already reported
    ///
    /// \return false if the given type is not exported, true otherwise
    bool check_exported_type(
        Definition const *user,
        IType const      *type,
        Bad_type_set     &bad_types);

    /// Check that all referenced entities of an exported function are exported.
    ///
    /// \param func_def   the definition of the exported function
    /// \param func_decl  the exported function declaration
    void check_exported_function_completeness(
        Definition const            *func_def,
        IDeclaration_function const *func_decl);

    /// Check that all referenced entities of an exported struct type are exported.
    ///
    /// \param strict_def   the definition of the exported struct type
    /// \param struct_decl  the exported struct declaration
    void check_exported_struct_completeness(
        Definition const               *struct_def,
        IDeclaration_type_struct const *struct_decl);

    /// Check that all referenced entities of an exported annotation are exported.
    ///
    /// \param anno_def   the definition of the exported annotation
    /// \param anno_decl  the exported annotation declaration
    void check_exported_annotation_completeness(
        Definition const              *anno_def,
        IDeclaration_annotation const *anno_decl);

    /// Check that all referenced entities of an exported constant are exported.
    ///
    /// \param const_def   the definition of the exported constant
    /// \param const_decl  the exported constant declaration
    void check_exported_constant_completeness(
        Definition const            *const_def,
        IDeclaration_constant const *const_decl);

    /// Check that all referenced types and all referenced entities
    /// (in a default initializer) of an exported entity are exported too.
    void check_exported_completeness();

    /// Finds a parameter for the given array size if any.
    ///
    /// \param arr_size   the array size
    Definition const *find_parameter_for_array_size(Definition const *arr_size);

    /// Mark the given entity as used and check for deprecation.
    ///
    /// \param def  the definition of the entity
    /// \param pos  the position of the access that caused the mark
    void mark_used(Definition const *def, Position const &pos);

private:
    /// Get the analysis info for a statement.
    ///
    /// \param stmt  the statement
    Stmt_info &get_stmt_info(IStatement const *stmt);

    // Enter a statement and put it into the context stack.
    ///
    /// \param stmt  the statement
    void enter_statement(IStatement const *stmt);

    /// Leave a statement and remove it from the context stack.
    ///
    /// \param stmt  the statement
    void leave_statement(IStatement const *stmt);

    /// Enter a loop and put it onto the context stack.
    ///
    /// \param loop  the loop statement
    void enter_loop(IStatement_loop const *loop);

    /// Leave a loop and remove it from the context stack.
    ///
    /// \param loop  the loop statement
    void leave_loop(IStatement_loop const *loop);

    /// Enter a switch and put it onto the context stack.
    ///
    /// \param stmt  the switch statement
    void enter_switch(IStatement_switch const *stmt);

    /// Leave a switch and remove it from the context stack.
    ///
    /// \param stmt  the switch statement
    void leave_switch(IStatement_switch const *stmt);

    /// Enter a case and put it onto the context stack.
    ///
    /// \param stmt  the case statement
    void enter_case(IStatement_case const *stmt);

    /// Leave a case and remove it from the context stack.
    ///
    /// \param stmt  the case statement
    void leave_case(IStatement_case const *stmt);

    /// Check if we are inside a loop.
    bool inside_loop() const { return m_loop_depth > 0; }

    /// Check if we are inside a switch.
    bool inside_switch() const { return m_switch_depth > 0; }

    /// Return the top context stack statement.
    ///
    /// \param depth  the depth of the context, 0 for the current parent
    IStatement const *get_context_stmt(size_t depth) const;

    /// Check if a switch statement has a implicit default (that exists the switch).
    ///
    /// \param stmt       the switch statement
    /// \param has_error  set to true if the statement is erroneous
    ///
    /// \return true if there is an implicit default
    bool has_switch_implicit_default(
        IStatement_switch const *stmt,
        bool                    &has_error);

    /// Return true if a switch has one case that can left the switch.
    ///
    /// \param switch_stmt  the switch statement to check
    /// \param has_error    set to true if an error occurred during processing
    ///
    /// \return true   if a case (or an implicit default) were found that can exist
    ///                the switch
    ///         false  otherwise
    bool has_one_case_reachable_exit(
        IStatement_switch const *switch_stmt,
        bool                    &has_error);

    /// Returns true if the parent statement is an if (i.e. we are inside a the or an else)
    bool inside_if() const;

    /// Check an if-cascade for identical conditions.
    ///
    /// \param stmt  top statement of an if cascade
    void check_if_cascade(IStatement_if const *stmt);

    /// Begin of a statement.
    ///
    /// \param stmt  the statement
    Stmt_info &start_statement(IStatement const *stmt);

    /// End of a statement.
    ///
    /// \param stmt  the statement
    void end_statement(IStatement const *stmt);

    /// End of a statement.
    ///
    /// \param stmt_info  the statement info
    void end_statement(Stmt_info &stmt_info);

    /// Start of an expression.
    void start_expression();

    /// End of an expression.
    ///
    /// \param expr  the expression
    void end_expression(IExpression const *expr);

    /// Return true if we are at nested expression level.
    bool inside_nested_expression() const { return m_expr_depth > 1; }

    /// Check if two expressions are syntactically identical.
    ///
    /// \param lhs  an expression
    /// \param rhs  an expression
    bool identical_expressions(
        IExpression const *lhs,
        IExpression const *rhs) const;

    /// Check if two statements are syntactically identical.
    ///
    /// \param lhs  a statement
    /// \param rhs  a statement
    bool identical_statements(
        IStatement const *lhs,
        IStatement const *rhs) const;

    /// Check if two declarations are syntactically identical.
    ///
    /// \param lhs  a declaration
    /// \param rhs  a declaration
    bool identical_declarations(
        IDeclaration const *lhs,
        IDeclaration const *rhs) const;

    /// Check if two type names are syntactically identical.
    ///
    /// \param lhs  a type name
    /// \param rhs  a type_name
    bool identical_type_names(
        IType_name const *lhs,
        IType_name const *rhs) const;

    /// Check if two annotation blocks are syntactically identical.
    ///
    /// \param lhs  an annotation block
    /// \param rhs  an annotation block
    bool identical_anno_blocks(
        IAnnotation_block const *lhs,
        IAnnotation_block const *rhs) const;

    /// Check if two qualified names are syntactically identical.
    ///
    /// \param lhs  a qualified name
    /// \param rhs  a qualified name
    bool identical_qnames(
        IQualified_name const *lhs,
        IQualified_name const *rhs) const;

    /// Returns true if the current statement is unconditional inside a loop.
    bool is_loop_unconditional() const;

    /// Replace the third and fourth parameter of assert by the
    /// current module name and this expression's line.
    ///
    /// \param expr  an debug::assert call
    void insert_assert_params(IExpression_call *expr);

    bool pre_visit(IStatement *stmt) MDL_FINAL;
    void post_visit(IStatement *stmt) MDL_FINAL;

    bool pre_visit(IStatement_compound *stmt) MDL_FINAL;
    void post_visit(IStatement_compound *stmt) MDL_FINAL;
    bool pre_visit(IStatement_while *stmt) MDL_FINAL;
    bool pre_visit(IStatement_do_while *stmt) MDL_FINAL;
    bool pre_visit(IStatement_for *stmt) MDL_FINAL;
    bool pre_visit(IStatement_switch *stmt) MDL_FINAL;
    bool pre_visit(IStatement_case *stmt) MDL_FINAL;
    bool pre_visit(IStatement_if *stmt) MDL_FINAL;
    bool pre_visit(IStatement_expression *stmt) MDL_FINAL;

    bool pre_visit(IDeclaration_function *decl) MDL_FINAL;

    void post_visit(IStatement_while *stmt) MDL_FINAL;
    void post_visit(IStatement_for *stmt) MDL_FINAL;
    void post_visit(IStatement_do_while *stmt) MDL_FINAL;
    void post_visit(IStatement_switch *stmt) MDL_FINAL;
    void post_visit(IStatement_case *stmt) MDL_FINAL;
    void post_visit(IStatement_break *stmt) MDL_FINAL;
    void post_visit(IStatement_continue *stmt) MDL_FINAL;
    void post_visit(IStatement_return *stmt) MDL_FINAL;
    void post_visit(IStatement_expression *stmt) MDL_FINAL;

    void post_visit(IDeclaration_function *decl) MDL_FINAL;

    bool pre_visit(IExpression *expr) MDL_FINAL;
    IExpression *post_visit(IExpression *expr) MDL_FINAL;
    bool pre_visit(IExpression_binary *expr) MDL_FINAL;
    bool pre_visit(IExpression_unary *expr) MDL_FINAL;
    IExpression *post_visit(IExpression_call *expr) MDL_FINAL;
    IExpression *post_visit(IExpression_reference *expr) MDL_FINAL;
    IExpression *post_visit(IExpression_conditional *expr) MDL_FINAL;

    /// Set the current function name from a (valid) function definition.
    ///
    /// \param def           a function definition
    /// \param is_material   true, if this is a material definition
    void set_curr_funcname(
        IDefinition const *def,
        bool              is_material);

private:
    /// Set if the next statement in a sequence is reachable.
    bool m_next_stmt_is_reachable;
    bool m_last_stmt_is_reachable;

    /// Set if an expression with a side effect was executed.
    bool m_has_side_effect;

    /// Set if an expression with a call was executed.
    bool m_has_call;

    /// Set if we are inside a material body or single expression function.
    bool m_inside_single_expr_body;

    /// If set, the current assigned definition.
    Definition const *m_curr_assigned_def;

    /// If set, we are inside the body of a function/material declaration.
    IDeclaration_function const *m_curr_entity_decl;

    /// If set, we are inside a function/material declaration.
    Definition const *m_curr_entity_def;

    /// The context stack.
    vector<IStatement const *>::Type m_context_stack;

    /// Current depth of the context stack.
    size_t m_context_depth;

    /// Current loop depth: 0 outside loop, > 0 in loop of nested depth n.
    size_t m_loop_depth;

    /// Current switch depth: 0 outside switch, > 0, in switch of nested depth n.
    size_t m_switch_depth;

    /// Current expression depth: 0 outside expression, 1, top-level, > 1 nested expression.
    size_t m_expr_depth;

    /// To store the analysis info for statements.
    Stmt_info_data m_stmt_info_data;

    /// Current function/material name for assert.
    string m_curr_funcname;
};

///
/// The auto typing analysis.
///
class AT_analysis : public Analysis
{
    typedef Analysis Base;

    typedef stack<size_t>::Type Dependency_stack;

    /// Helper stack class that allows to "stop" at given depth.
    class Assignment_stack {
    public:
        /// Constructor.
        /// 
        /// \param alloc  the allocator
        explicit Assignment_stack(IAllocator *alloc)
        : m_stack(Dependency_stack::container_type(alloc))
        , m_depth(0)
        , m_stop_depth(0)
        {
        }

        /// Push an ID.
        void push(size_t id) { m_stack.push(id); ++m_depth; }

        /// Pop an ID.
        void pop() { m_stack.pop(); --m_depth; }

        /// Check if empty.
        bool empty() const { return m_depth <= m_stop_depth; }

        /// Return the top element.
        size_t top() const { return m_stack.top(); }

        /// Set a new stop depth.
        size_t set_stop_depth(size_t stop_depth) 
        {
            size_t old = m_stop_depth;
            m_stop_depth = stop_depth;
            return old;
        }

        /// Set a new stop depth at current depth.
        size_t set_stop_depth() 
        {
            size_t old = m_stop_depth;
            m_stop_depth = m_depth;
            return old;
        }

    private:
        // The stack itself.
        Dependency_stack m_stack;

        /// Current depth of the stack.
        size_t m_depth;

        /// Current stop depth.
        size_t m_stop_depth;
    };

public:
    /// Run the auto typing analysis on this module.
    ///
    /// \param compiler  the current compiler
    /// \param module    the module to analyze
    /// \param ctx       the thread context
    /// \param cg        the call graph
    static void run(
        MDL              *compiler,
        Module           &module,
        Thread_context   &ctx,
        Call_graph const &cg);

private:
    /// Predefined node Id's..
    enum {
        return_value_id  = 0,   ///< ID of the return value node
        varying_call_id  = 1,   ///< ID of the varying call node
        first_free_id    = 2,   ///< first dynamic ID
    };

    /// Constructor.
    ///
    /// \param compiler  the current compiler
    /// \param module    the module to analyze
    /// \param ctx       the thread context
    /// \param cg        the call graph
    explicit AT_analysis(
        MDL              *compiler,
        Module           &module,
        Thread_context   &ctx,
        Call_graph const &cg);

    /// Add control dependence to the given node if one exists.
    ///
    /// \param node_id  the ID of the node
    void add_control_dependence(size_t node_id);

    /// Set the control dependence uplink to the given node if one exists.
    ///
    /// \param node_id  the ID of the control node
    void set_control_dependence_uplink(size_t node_id);

    /// Use the calculated auto-types to check the AST.
    ///
    /// \param decl  the function declaration that is currently processed
    void check_auto_types(IDeclaration_function *decl);

    /// Dump the dependency graph.
    ///
    /// \param suffix  name suffix for dumped file name
    void dump_dg(char const *suffix);

    /// Process a function given by its definition.
    ///
    /// \param def  a function definition
    void process_function(Definition const *def);

    bool pre_visit(IDeclaration_function *decl) MDL_FINAL;
    void post_visit(IDeclaration_function *decl) MDL_FINAL;

    bool pre_visit(IDeclaration_variable *decl) MDL_FINAL;

    bool pre_visit(IStatement_return *stmt) MDL_FINAL;

    bool pre_visit(IStatement_if *stmt) MDL_FINAL;

    bool pre_visit(IStatement_while *stmt) MDL_FINAL;

    bool pre_visit(IStatement_do_while *stmt) MDL_FINAL;

    bool pre_visit(IStatement_for *stmt) MDL_FINAL;

    bool pre_visit(IStatement_switch *stmt) MDL_FINAL;

    void post_visit(IStatement_break *stmt) MDL_FINAL;

    void post_visit(IStatement_continue *stmt) MDL_FINAL;

    bool pre_visit(IExpression_binary *expr) MDL_FINAL;

    bool pre_visit(IExpression_unary *expr) MDL_FINAL;

    bool pre_visit(IExpression_call *expr) MDL_FINAL;

    IExpression *post_visit(IExpression_reference *ref) MDL_FINAL;

    bool pre_visit(IParameter *param) MDL_FINAL;

private:
    /// A memory arena used for all temporary objects.
    Memory_arena m_arena;

    /// The builder for the temporary memory arena.
    Arena_builder m_builder;

    /// The call graph.
    Call_graph const &m_cg;

    /// The dependence graph of the currently analyzed function.
    Dependence_graph *m_dg;

    /// The assignment stack.
    Assignment_stack m_assignment_stack;

    /// The control stack.
    Dependency_stack m_control_stack;

    /// The context stack: contains loops AND switches.
    Dependency_stack m_context_stack;

    /// The loop stack: contains loops only.
    Dependency_stack m_loop_stack;

    /// If set, the analysis detects a varying call inside a function.
    bool m_has_varying_call;

    /// if set, we are inside a default argument.
    bool m_inside_def_arg;

    /// If set, the current function has the uniform modifier
    bool m_curr_func_is_uniform;

    /// If set, the dependence graph will be dumped.
    bool m_opt_dump_dg;

    ///
    /// Helper class to run the auto typing analysis on the
    /// call graph.
    ///
    class AT_visitor : public ICallgraph_visitor {
    public:
        /// Visit a node of the call graph.
        void visit_cg_node(Call_node *node, ICallgraph_visitor::Order order) MDL_FINAL;

        /// Constructor.
        explicit AT_visitor(AT_analysis &ana);

    private:
        /// The analysis
        AT_analysis &m_ana;

        /// The ID of the current module.
        size_t m_curr_mod_id;
    };

    friend class AT_visitor;
};

/// Checks for generated AST errors, especially if the AST is a tree, not a DAG.
class AST_checker : protected Module_visitor
{
    typedef Module_visitor Base;

public:
    /// Checker, asserts on failure.
    static bool check_module(IModule const *module);

private:
    void post_visit(IStatement *stmt) MDL_FINAL;
    IExpression *post_visit(IExpression *expr) MDL_FINAL;
    void post_visit(IDeclaration *decl) MDL_FINAL;

private:
    /// Constructor.
    explicit AST_checker(IAllocator *alloc);

private:
    typedef ptr_hash_set<IStatement const>::Type    Stmt_set;
    typedef ptr_hash_set<IExpression const>::Type   Expr_set;
    typedef ptr_hash_set<IDeclaration const>::Type  Decl_set;

    /// Set of all visited statements.
    Stmt_set m_stmts;

    /// Set of all visited expressions.
    Expr_set m_exprs;

    /// Set of all visited declarations.
    Decl_set m_decls;

    /// Set to true, if errors was found.
    bool m_errors;
};

/// Returns true for error definitions.
///
/// \param def  the definition to check
extern inline bool is_error(IDefinition const *def)
{
    IType const *type = def->get_type();
    if (is<IType_error>(type))
        return true;
    return def->get_symbol()->get_id() == ISymbol::SYM_ERROR;
}

/// Returns true for error expressions.
///
/// \param expr  the expression to check
extern inline bool is_error(IExpression const *expr)
{
    IType const *type = expr->get_type();
    return is<IType_error>(type);
}

/// Returns true for error names.
extern inline bool is_error(ISimple_name const *name)
{
    return name->get_symbol()->get_id() == ISymbol::SYM_ERROR;
}

/// Returns true for error names.
bool is_error(IQualified_name const *name);

/// Debug helper: Dump the AST of a compilation unit.
///
/// \param module  the module to dump
void dump_ast(IModule const *module);

/// Debug helper: Dump the definition table of a compilation unit.
///
/// \param module  the module to dump
void dump_def_tab(IModule const *module);

/// Debug helper: Dump the AST of a declaration.
///
/// \param decl  the declaration to dump
void dump_ast(IDeclaration const *decl);

/// Debug helper: Dump the AST of a compilation unit.
///
/// \param expr  the expression to dump
void dump_expr(IExpression const *expr);

/// Debug helper: Dump a definition.
///
/// \param def  the definition too dump
void dump_def(IDefinition const *def);

}  // mdl
}  // mi

#endif

