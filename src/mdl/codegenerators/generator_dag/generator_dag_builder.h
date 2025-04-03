/******************************************************************************
 * Copyright (c) 2015-2025, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDL_GENERATOR_DAG_BUILDER_H
#define MDL_GENERATOR_DAG_BUILDER_H 1

#include <mi/mdl/mdl_expressions.h>
#include <mi/mdl/mdl_generated_dag.h>
#include "mdl/compiler/compilercore/compilercore_allocator.h"
#include "mdl/compiler/compilercore/compilercore_cstring_hash.h"

namespace mi {
namespace mdl {

class DAG_mangler;
class DAG_node_factory_impl;
class Name_printer;
class IAnnotation;
class IDeclaration_constant;
class IDeclaration_function;
class IDeclaration_variable;
class IModule;
class IParameter;
class IStatement;
class ISymbol;
class IType_struct;
class IType_vector;
class IValue_resource;
class Symbol_table;
class Type_factory;
class Value_factory;

/// Helper struct to map (local) AST file IDs to (global) DAG file IDs.
struct File_id_table {
    size_t module_id;   ///< the ID of the module that "produces" this table
    size_t len;         ///< number of entries in this table
    size_t map[1];      ///< the table itself

public:
    /// Constructor.
    File_id_table(
        size_t module_id,
        size_t len)
        : module_id(module_id)
        , len(len)
    {
        memset(map, 0, sizeof(map[0]) * len);
    }

    /// Get DAG id for an AST id.
    size_t get_dag_id(size_t ast_id) const {
        if (ast_id < len) {
            return map[ast_id];
        }
        MDL_ASSERT(!"cannot map unknown file ID");
        return 0;
    }

    /// Set a DAG_id for an AST_id.
    void set_dag_id(size_t ast_id, size_t dag_id)
    {
        if (ast_id < len) {
            map[ast_id] = dag_id;
            return;
        } else {
            MDL_ASSERT(!"cannot map unknown file ID");
        }
    }
};

/// Builder from AST-nodes to DAG nodes.
class DAG_builder {
    friend class Module_scope;

public:
    /// The type of vectors of reference expressions.
    typedef vector<IExpression_reference const *>::Type Ref_vector;

    /// The type of maps from definitions to temporary values (DAG-IR nodes).
    typedef ptr_hash_map<IDefinition const, DAG_node const *>::Type Definition_temporary_map;

private:
    /// The type of vectors of definitions.
    typedef vector<IDefinition const *>::Type Definition_vector;

    /// RAII-like helper class to handle accessible parameters inside inlining.
    class Inline_scope {
    public:
        /// Constructor.
        Inline_scope(DAG_builder &builder, bool in_same_function = false)
        : m_builder(builder)
        , m_old_accesible_parameters(builder.get_allocator())
        , m_old_tmp_value_map(
            0,
            Definition_temporary_map::hasher(),
            Definition_temporary_map::key_equal(),
            builder.get_allocator())
        , m_in_same_function(in_same_function)
        {
            if (!m_in_same_function) {
                builder.m_accesible_parameters.swap(m_old_accesible_parameters);
            }

            // remember current state of the environment map
            m_old_tmp_value_map = builder.m_tmp_value_map;
        }

        /// Destructor.
        ~Inline_scope()
        {
            if (!m_in_same_function) {
                m_builder.m_accesible_parameters.swap(m_old_accesible_parameters);
            } else {
                // update the environment map
                for (Definition_temporary_map::iterator it = m_old_tmp_value_map.begin(),
                    end = m_old_tmp_value_map.end(); it != end; ++it)
                {
                    it->second = m_builder.m_tmp_value_map[it->first];
                }
            }

            // restore the environment map
            m_builder.m_tmp_value_map = m_old_tmp_value_map;
        }

    private:
        DAG_builder              &m_builder;
        Definition_vector        m_old_accesible_parameters;
        Definition_temporary_map m_old_tmp_value_map;
        bool                     m_in_same_function;
    };

public:
    /// Constructor.
    ///
    /// \param alloc               the allocator
    /// \param node_factory        the IR node factory used to create DAG nodes
    /// \param mangler             the DAG name mangler
    DAG_builder(
        IAllocator            *alloc,
        DAG_node_factory_impl &node_factory,
        DAG_mangler           &mangler);

    /// Retrieve the allocator.
    IAllocator *get_allocator() {return m_alloc; }

    /// Retrieve the node factory.
    DAG_node_factory_impl &get_node_factory() { return m_node_factory; }

    /// Retrieve the node factory.
    DAG_node_factory_impl const &get_node_factory() const { return m_node_factory; }

    /// Retrieve the type factory.
    Type_factory &get_type_factory() { return m_type_factory; }

    /// Set a resource modifier.
    void set_resource_modifier(IResource_modifier *modifier);

    /// Enable/disable local function calls.
    ///
    /// \param flag  True if local function calls are forbidden, False otherwise
    ///
    /// \return The old setting.
    bool forbid_local_function_calls(bool flag);

    /// Enable/disable the target material model compilation mode.
    ///
    /// \param flag  True if target material compilation mode should be enabled
    ///
    /// \return The old setting.
    bool enable_target_material_compilation_mode(bool flag);


    /// Convert a definition to a name.
    ///
    /// \param def                     The definition to convert.
    /// \param module_name             The name of the owner module of this definition.
    /// \param with_signature_suffix   Indicates whether to include the signature suffix
    /// \returns                       The name for the definition.
    string def_to_name(
        IDefinition const *def, const char *module_name, bool with_signature_suffix = true) const;

    /// Convert a definition to a name.
    ///
    /// \param def                     The definition to convert.
    /// \param module                  The (virtual) owner module of this definition.
    /// \param with_signature_suffix   Indicates whether to include the signature suffix
    /// \returns                       The name for the definition.
    string def_to_name(
        IDefinition const *def, IModule const *module, bool with_signature_suffix = true) const;

    /// Convert a definition to a name.
    ///
    /// \param def                     The definition to convert.
    /// \param with_signature_suffix   Indicates whether to include the signature suffix
    /// \returns                       The name for the definition (using the original owner
    ///                                module).
    string def_to_name(IDefinition const *def, bool with_signature_suffix = true) const;

    /// Convert a type to a name (not using the mangler).
    ///
    /// \param type  The type to convert.
    /// \returns     The name for the type.
    string type_to_name(IType const *type) const;

    /// Convert a parameter type to a name (using the mangler).
    ///
    /// \param type  The type to convert.
    /// \returns     The name for the type.
    string parameter_type_to_name(IType const *type) const;

    /// Clear temporary data to restart code generation.
    void reset();

    /// Make the given function/material parameter accessible.
    ///
    /// \param p_def  the parameter definition
    ///
    /// Marks this parameter as accessible from expressions.
    void make_accessible(mi::mdl::IDefinition const *p_def);

    /// Make the given function/material parameter accessible.
    ///
    /// \param param  the parameter
    ///
    /// Marks this parameter as accessible from expressions.
    void make_accessible(mi::mdl::IParameter const *param);

    /// Gets the error list.
    Ref_vector const &get_errors() const { return m_error_calls; }

    /// Convert an MDL let temporary declaration to a DAG IR node.
    ///
    /// \param var_decl     The MDL variable declaration to convert.
    /// \returns            The DAG IR node representing the MDL statement.
    ///
    DAG_node const *var_decl_to_dag(
        IDeclaration_variable const *var_decl);

    /// Convert an MDL constant declaration to a DAG IR node.
    ///
    /// \param const_decl   The MDL constant declaration to convert.
    /// \returns            The DAG IR node representing the MDL statement.
    ///
    DAG_node const *const_decl_to_dag(
        IDeclaration_constant const *const_decl);

    /// Convert an MDL statement to a DAG IR node.
    ///
    /// \param stmt         The MDL statement to convert.
    /// \returns            The DAG IR node representing the MDL statement.
    ///
    DAG_node const *stmt_to_dag(
        IStatement const  *stmt);

    /// Insert a decl_cast to the destination type.
    ///
    /// \param dst_type     The type to cast to.
    /// \param expr         Expression to cast.
    /// \returns            The DAG IR node representing the cast operation.
    DAG_node const *insert_decl_cast(
        IType const *dst_type,
        DAG_node const *expr);

    /// Insert a decl_cast to the destination type, if needed. For non-decl-struct
    /// expressions, this is the identity.
    ///
    /// \param dst_type     The type to cast to.
    /// \param expr         Expression to cast.
    /// \returns            The DAG IR node representing the cast operation.
    DAG_node const *maybe_insert_decl_cast(
        IType const    *dst_type,
        DAG_node const *expr);

    /// Convert an MDL expression to a DAG IR node.
    ///
    /// \param expr         The MDL expression to convert.
    /// \returns            The DAG IR node representing the MDL expression.
    ///
    DAG_node const *expr_to_dag(
        IExpression const *expr);

    /// Convert an MDL expression to a DAG IR node and decl_cast it to dst_type if necessary.
    ///
    /// \param expr         The MDL expression to convert.
    /// \returns            The DAG IR node representing the MDL expression.
    ///
    DAG_node const *expr_to_dag(
        IType const       *dst_type,
        IExpression const *expr);

    /// Convert an MDL annotation to a DAG IR node.
    ///
    /// \param exp          The MDL annotation to convert.
    /// \returns            The DAG IR node.
    ///
    DAG_node const *annotation_to_dag(
        IAnnotation const *annotation);

    /// Convert an MDL preset call expression to a DAG IR node.
    ///
    /// \param orig_mat_def   The definition of original material.
    /// \returns              The DAG IR node.
    ///
    DAG_node const *preset_to_dag(IDefinition const *orig_mat_def);

    /// Creates an anno::hidden() annotation.
    ///
    /// \returns  The DAG IR node for the annotation or NULL.
    ///           if inside the builtin module.
    DAG_node const *create_hidden_annotation();

    /// Return the top-of-stack module, not retained.
    ///
    /// \note Does NOT increase the reference count of the returned
    ///       module, do NOT decrease it just because of this call.
    Module const *tos_module() const;

    /// Get the unique file ID for the given file name.
    ///
    /// \param fname  a file name
    ///
    /// \return  the unique ID for this file name inside the current builder
    size_t get_unique_file_id(char const *fname);

    /// Get the unique file ID for the given module.
    ///
    /// \param mod  an MDL module
    ///
    /// \return  the unique ID for this file name inside the current builder
    size_t get_unique_file_id(Module const *mod);

    /// Get the file ID mapping table for a given module.
    ///
    /// \param mod  the module
    ///
    /// \return the File ID table for this module, creating one if it does not exists
    File_id_table const *get_file_id_table(Module const *mod);

    /// Try to inline the given call.
    ///
    /// \param owner_dag    The owner code DAG of the definition to inline.
    /// \param def          The definition of the function to inline.
    /// \param call         The DAG call expression to convert.
    /// \returns            The DAG expression if the function call could be inlined,
    ///                     NULL otherwise.
    DAG_node const *try_inline(
        IGenerated_code_dag const     *owner_dag,
        IDefinition const             *def,
        DAG_call::Call_argument const *args,
        size_t                        n_args);

    /// Returns the name of a binary operator.
    ///
    /// \param expr                    The MDL expression.
    /// \param with_signature_suffix   Indicates whether to include the signature suffix
    string get_binary_name(IExpression_binary const *expr, bool with_signature_suffix = true) const;

    /// Returns the name of an unary operator.
    ///
    /// \param expr                    The MDL expression.
    /// \param with_signature_suffix   Indicates whether to include the signature suffix
    string get_unary_name(IExpression_unary const *expr, bool with_signature_suffix = true) const;

    /// Convert an unary MDL operator to a name.
    ///
    /// \param op           The unary MDL operator to convert.
    /// \returns            The name.
    ///
    static char const *unary_op_to_name(IExpression_unary::Operator op);

    /// Return the name of a binary MDL operator.
    ///
    /// \param op           The binary MDL operator.
    /// \returns            The name.
    ///
    static char const *binary_op_to_name(IExpression_binary::Operator op);

    /// Returns true if a conditional operator was created.
    bool cond_created() const { return m_conditional_created; }

    /// Returns true if a conditional operator on *df types was created.
    bool cond_df_created() const { return m_conditional_df_created; }

    /// Push a module on the module stack and process debug info of the module.
    ///
    /// \param  the module to push
    ///
    /// \Note: Should not be called directly, use \c Module_scope
    void push_module(Module const *mod);

    /// Pop a module from the module stack.
    ///
    /// \Note: Should not be called directly, use \c Module_scope
    void pop_module();

    /// Returns true if errors were detected (and clear the flag).
    bool error_state()
    {
        bool res = m_error_detected;
        m_error_detected = false;
        return res;
    }

    /// Convert a constant into an elemental constructor call representing
    /// the same value.
    /// 
    /// \param cnst  The constant DAG node to convert.
    /// 
    /// \returns     The equivalent call expression.
    DAG_call const *constant_to_call(
        DAG_constant const *cnst);

    /// Create a struct field select.
    ///
    /// \param compound  the compound value
    /// \param field     the name of the field
    /// \param f_type    the type of the field
    /// \param dbg_info  the debug info for this field select if any
    DAG_node const *create_field_select(
        DAG_node const *compound,
        char const     *field,
        IType const    *f_type,
        DAG_DbgInfo    dbg_info);

private:
    /// Get the allocator.
    IAllocator *get_allocator() const { return m_alloc; }

    /// Get the definition of the parameter of a function/material at index.
    ///
    /// \param decl         The declaration of the function/material.
    /// \param index        The index of the parameter.
    /// \returns            The definition of the parameter.
    ///
    /// \note Does not work for clones, those must be resolved in advance.
    static IDefinition const *get_parameter_definition(
        IDeclaration_function const *decl,
        int                         index);

    /// Convert an AST position into a DAG debug info.
    ///
    /// \param pos  the AST position if any
    DAG_DbgInfo get_dbg_info(Position const *pos);

    /// Convert an AST position into a DAG debug info.
    ///
    /// \param ast  an AST entity
    template<typename AST>
    DAG_DbgInfo get_dbg_info(AST ast) { return this->get_dbg_info(&ast->access_position()); }

    /// Set array size variable for size-deferred arrays, if necessary.
    ///
    /// \param decl         The declaration of the function/material.
    /// \param index        The index of the parameter.
    /// \param arg_exp      The argument for the parameter, potentially being an array.
    void set_parameter_array_size_var(
        IDeclaration_function const *decl,
        int                         index,
        DAG_node const *            arg_exp);

    /// Run local optimizations for a binary expression.
    ///
    /// \param op        The opcode of the binary expression.
    /// \param l         The left argument.
    /// \param r         The right argument.
    /// \param type      The type of the binary expression.
    /// \param dbg_info  The debug info for this expression if any.
    ///
    /// \returns   The DAG IR node representing the requested expression.
    DAG_node const *optimize_binary_operator(
        IExpression_binary::Operator op,
        DAG_node const               *l,
        DAG_node const               *r,
        IType const                  *type,
        DAG_DbgInfo                  dbg_info);

    /// Convert an MDL literal expression to a DAG constant.
    ///
    /// \param lit          The MDL literal expression to convert.
    /// \returns            The DAG IR node.
    ///
    DAG_constant const *lit_to_dag(
        mi::mdl::IExpression_literal const *lit);

    /// Convert an MDL reference expression to a DAG IR node.
    ///
    /// \param ref          The MDL reference expression to convert.
    /// \returns            The DAG IR node.
    ///
    DAG_node const *ref_to_dag(
        IExpression_reference const *ref);

    /// Convert an MDL unary expression to a DAG IR node.
    ///
    /// \param unary        The MDL unary expression to convert.
    /// \returns            The DAG IR node.
    ///
    DAG_node const *unary_to_dag(
        IExpression_unary const *unary);

    /// Creates a insert pseudo-instruction on a struct value.
    ///
    /// \param s_type  the type of the struct
    /// \param index   the index of the element that will be inserted
    /// \param c_node  the node representing the compound value
    /// \param e_node  the node representing the element value that will be inserted at index
    /// \param pos     the source position for this instruction if any
    DAG_node const *create_struct_insert(
        IType_struct const *s_type,
        int                index,
        DAG_node const     *c_val,
        DAG_node const     *e_val,
        Position const     *pos);

    /// Creates a insert pseudo-instruction on a vector value.
    ///
    /// \param v_type  the type of the vector
    /// \param index   the index of the element that will be inserted
    /// \param c_node  the node representing the compound value
    /// \param e_node  the node representing the element value that will be inserted at index
    /// \param pos     the source position for this instruction if any
    DAG_node const *create_vector_insert(
        IType_vector const *v_type,
        int                index,
        DAG_node const     *c_val,
        DAG_node const     *e_val,
        Position const     *pos);

    /// Convert a node to a destination type.
    ///
    /// \param dst_type  the destination type
    /// \param n         the DAG node
    DAG_node const *convert_to_type(
        IType const    *dst_type,
        DAG_node const *n);

    /// Convert an MDL binary expression to a DAG IR node.
    ///
    /// \param binary       The MDL binary expression to convert.
    /// \returns            The DAG IR node.
    ///
    DAG_node const *binary_to_dag(
        IExpression_binary const *binary);

    /// Convert an MDL conditional expression to a DAG IR node.
    ///
    /// \param cond         The MDL conditional expression to convert.
    /// \returns            The DAG IR node.
    ///
    DAG_node const *cond_to_dag(
        IExpression_conditional const *cond);

    /// Try to inline the given call.
    ///
    /// \param call         The MDL call expression to convert.
    /// \returns            The DAG expression if the function call could be inlined,
    ///                     NULL otherwise.
    DAG_node const *try_inline(
        IExpression_call const *call);

    /// Convert an MDL call expression to a DAG IR node.
    ///
    /// \param call         The MDL call expression to convert.
    /// \returns            The DAG IR node.
    ///
    DAG_node const *call_to_dag(
        IExpression_call const *call);

    /// Convert an MDL let expression to a DAG IR node.
    ///
    /// \param let          The MDL let expression to convert.
    /// \returns            The DAG IR node.
    ///
    DAG_node const *let_to_dag(
        IExpression_let const *let);

    /// Find a parameter for a given array size symbol.
    ///
    /// \param sym   the array size symbol
    IDefinition const *find_parameter_for_size(ISymbol const *sym) const;

    /// Report an error due to a call to a local function.
    ///
    /// \param ref  the reference of the function that is called
    void error_local_call(IExpression_reference const *ref);

    /// Creates a default initializer for the given type.
    ///
    /// \param type  the type
    DAG_constant const *default_initializer(IType const *type);

    /// Creates a default initializer for the given type.
    ///
    /// \param type  the type
    IValue const *default_initializer_value(IType const *type);

    // no copy constructor
    DAG_builder(DAG_builder const &) MDL_DELETED_FUNCTION;

    // no assignment operator
    DAG_builder const &operator=(DAG_builder const &) MDL_DELETED_FUNCTION;

private:
    /// The type of vectors of values.
    typedef vector<IValue const *>::Type Value_vector;

    /// The allocator.
    IAllocator *m_alloc;

    /// The IR node factory.
    DAG_node_factory_impl &m_node_factory;

    /// The value factory.
    Value_factory &m_value_factory;

    /// The type factory.
    Type_factory &m_type_factory;

    /// The symbol table;
    Symbol_table &m_sym_tab;

    /// The definition mangler.
    DAG_mangler &m_mangler;

    /// The printer for names.
    Name_printer &m_printer;

    /// The used Resource modifier.
    IResource_modifier *m_resource_modifier;

    /// The map from definitions to temporary indices.
    Definition_temporary_map m_tmp_value_map;

    /// The type of the module stack.
    typedef vector<mi::base::Handle<Module const> >::Type Module_stack;

    /// The module stack.
    Module_stack m_module_stack;

    /// Module stack tos index;
    size_t m_module_stack_tos;

    /// Helper: Accessible parameters when translating an expression.
    Definition_vector m_accesible_parameters;

    /// The current file ID table.
    File_id_table const *m_curr_file_id_table;

    enum Inline_skip_flag
    {
        INL_NO_SKIP = 0,
        INL_SKIP_BREAK = 1,
        INL_SKIP_RETURN = 2
    };

    /// Specifies whether and how instructions should be skipped.
    Inline_skip_flag m_skip_flags;

    /// Node returned by a return statement during inlining.
    DAG_node const *m_inline_return_node;

    /// If non-empty, the list of all called forbidden functions.
    Ref_vector m_error_calls;

    /// The Arena we allocate file tables on.
    Memory_arena m_file_table_arena;

    typedef ptr_hash_map<Module const, File_id_table const *>::Type  File_ID_table_map;

    /// Map modules to its file ID table.
    File_ID_table_map m_module_2_file_id_map;

    typedef hash_map<char const *, size_t, cstring_hash, cstring_equal_to>::Type  File_2_ID_map;

    /// Map source file names to file ID's.
    File_2_ID_map m_file_2_id_map;

    /// If true, local calls are forbidden in materials.
    bool m_forbid_local_calls;

    /// If true, a conditional operator was created.
    bool m_conditional_created;

    /// If true, a conditional operator on *df types was created.
    bool m_conditional_df_created;

    /// if true we are in target model mode.
    bool m_target_material_model_mode;

    /// Set to true if an error during build occurred.
    bool m_error_detected;
};

/// RAII helper class to handle the module stack.
class Module_scope {
public:
    /// Constructor.
    ///
    /// \param builder  the DAG_builder
    /// \param mod  the module to push
    Module_scope(DAG_builder &builder, Module const *mod)
    : m_builder(builder)
    {
        builder.push_module(mod);
    }

    /// Destructor.
    ~Module_scope() { m_builder.pop_module(); }

private:
    DAG_builder &m_builder;
};

/// RAII Helper class to handle the "forbid local function call" flag.
class Forbid_local_functions_scope {
public:
    /// Constructor.
    ///
    /// \param builder  the DAG builder to manipulate
    /// \param flag     True if local function calls are forbidden, False otherwise
    Forbid_local_functions_scope(
        DAG_builder &builder,
        bool       flag)
    : m_builder(builder)
    {
        m_old = builder.forbid_local_function_calls(flag);
    }

    /// Destructor.
    ~Forbid_local_functions_scope() { m_builder.forbid_local_function_calls(m_old); }

private:
    /// The builder to manipulate.
    DAG_builder &m_builder;

    /// The old value of the flag.
    bool        m_old;
};

/// RAII Helper class to handle the "target material model mode" flag.
class Target_material_model_mode_scope {
public:
    /// Constructor.
    ///
    /// \param builder  the DAG builder to manipulate
    /// \param flag     True target material model compilation mode is used, False otherwise
    Target_material_model_mode_scope(
        DAG_builder &builder,
        bool        flag)
    : m_builder(builder)
    {
        m_old = builder.enable_target_material_compilation_mode(flag);
    }

    /// Destructor.
    ~Target_material_model_mode_scope()
    {
        m_builder.enable_target_material_compilation_mode(m_old);
    }

private:
    /// The builder to manipulate.
    DAG_builder &m_builder;

    /// The old value of the flag.
    bool        m_old;
};

} // mdl
} // mi

#endif // MDL_GENERATOR_DAG_BUILDER_H

