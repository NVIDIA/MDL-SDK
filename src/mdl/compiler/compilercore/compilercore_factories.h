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

#ifndef MDL_COMPILERCORE_FACTORIES_H
#define MDL_COMPILERCORE_FACTORIES_H 1

#include <mi/base/handle.h>

#include <mi/mdl/mdl_iowned.h>
#include <mi/mdl/mdl_declarations.h>
#include <mi/mdl/mdl_expressions.h>
#include <mi/mdl/mdl_statements.h>
#include <mi/mdl/mdl_types.h>
#include <mi/mdl/mdl_annotations.h>

#include "compilercore_cc_conf.h"
#include "compilercore_allocator.h"
#include "compilercore_memory_arena.h"
#include "compilercore_symbols.h"
#include "compilercore_cstring_hash.h"
#include "compilercore_assert.h"

namespace mi {
namespace mdl {

class IMDL;
class Type_material;
class Type_struct;
class Type_array_size;
class Type_enum;
class Module_serializer;
class Module_deserializer;

/// Implementation of the Declaration factory.
class Declaration_factory : public IDeclaration_factory
{
    typedef IDeclaration_factory Base;
public:

    /// Create a new invalid declaration.
    IDeclaration_invalid *create_invalid(
        bool exported = false,
        int  start_line = 0,
        int  start_column = 0,
        int  end_line = 0,
        int  end_column = 0) MDL_FINAL;

    /// Create a new import declaration.
    IDeclaration_import *create_import(
        IQualified_name const *module_name = NULL,
        bool                  exported = false,
        int                   start_line = 0,
        int                   start_column = 0,
        int                   end_line = 0,
        int                   end_column = 0) MDL_FINAL;

    /// Create a new parameter.
    IParameter const *create_parameter(
        IType_name const        *type_name,
        ISimple_name const      *name,
        IExpression const       *init,
        IAnnotation_block const *annotations,
        int                     start_line = 0,
        int                     start_column = 0,
        int                     end_line = 0,
        int                     end_column = 0) MDL_FINAL;

    /// Create a new annotation declaration.
    IDeclaration_annotation *create_annotation(
        ISimple_name const      *name = NULL,
        IAnnotation_block const *annotations = NULL,
        bool                    exported = false,
        int                     start_line = 0,
        int                     start_column = 0,
        int                     end_line = 0,
        int                     end_column = 0) MDL_FINAL;

    /// Create a new constant declaration.
    IDeclaration_constant *create_constant(
        IType_name const *type_name = NULL,
        bool             exported = false,
        int              start_line = 0,
        int              start_column = 0,
        int              end_line = 0,
        int              end_column = 0) MDL_FINAL;

    /// Create a new type alias declaration.
    IDeclaration_type_alias *create_alias(
        IType_name const   *type_name = NULL,
        ISimple_name const *alias_name = NULL,
        bool               exported = false,
        int                start_line = 0,
        int                start_column = 0,
        int                end_line = 0,
        int                end_column = 0) MDL_FINAL;

    /// Create a new struct type declaration.
    IDeclaration_type_struct *create_struct(
        ISimple_name const      *struct_name = NULL,
        IAnnotation_block const *annotations = NULL,
        bool                    exported = false,
        int                     start_line = 0,
        int                     start_column = 0,
        int                     end_line = 0,
        int                     end_column = 0) MDL_FINAL;

    /// Create a new enum type declaration.
    IDeclaration_type_enum *create_enum(
        ISimple_name const      *enum_name = NULL,
        IAnnotation_block const *annotations = NULL,
        bool                    exported = false,
        bool                    is_enum_class = false,
        int                     start_line = 0,
        int                     start_column = 0,
        int                     end_line = 0,
        int                     end_column = 0) MDL_FINAL;

    /// Create a new variable declaration.
    IDeclaration_variable *create_variable(
        IType_name const *type_name = NULL,
        bool             exported = false,
        int              start_line = 0,
        int              start_column = 0,
        int              end_line = 0,
        int              end_column = 0) MDL_FINAL;

    /// Create a new function declaration.
    IDeclaration_function *create_function(
        IType_name const        *return_type_name = NULL,
        IAnnotation_block const *ret_annotations = NULL,
        ISimple_name const      *function_name = NULL,
        bool                    is_preset = false,
        IStatement const        *body = NULL,
        IAnnotation_block const *annotations = NULL,
        bool                    is_exported = false,
        int                     start_line = 0,
        int                     start_column = 0,
        int                     end_line = 0,
        int                     end_column = 0) MDL_FINAL;

    /// Create a new module declaration.
    IDeclaration_module *create_module(
        IAnnotation_block const *annotations = NULL,
        int                     start_line = 0,
        int                     start_column = 0,
        int                     end_line = 0,
        int                     end_column = 0) MDL_FINAL;

    /// Create a new namespace alias.
    IDeclaration_namespace_alias *create_namespace_alias(
        ISimple_name const    *alias,
        IQualified_name const *ns,
        int                   start_line = 0,
        int                   start_column = 0,
        int                   end_line = 0,
        int                   end_column = 0) MDL_FINAL;

public:
    /// Constructor.
    ///
    /// \param arena  memory arena to allocate objects
    explicit Declaration_factory(Memory_arena &arena);

private:
    // non copyable
    Declaration_factory(Declaration_factory const &) MDL_DELETED_FUNCTION;
    Declaration_factory &operator=(Declaration_factory const &) MDL_DELETED_FUNCTION;

private:
    /// The builder for declarations.
    Arena_builder m_builder;
};

/// Implementation of the Expression factory.
class Expression_factory : public IExpression_factory
{
    typedef IExpression_factory Base;
public:

    /// Create a new invalid expression.
    IExpression_invalid *create_invalid(
        int start_line = 0,
        int start_column = 0,
        int end_line = 0,
        int end_column = 0) MDL_FINAL;

    /// Create a new literal expression.
    ///
    /// \param value The value of the literal.
    IExpression_literal *create_literal(
        IValue const *value = NULL,
        int          start_line = 0,
        int          start_column = 0,
        int          end_line = 0,
        int          end_column = 0) MDL_FINAL;

    /// Create a new reference expression.
    ///
    /// \param name The qualified name of the reference.
    IExpression_reference *create_reference(
        IType_name const *name = NULL,
        int              start_line = 0,
        int              start_column = 0,
        int              end_line = 0,
        int              end_column = 0) MDL_FINAL;

    /// Create a new unary expression.
    ///
    /// \param op       The operator.
    /// \param argument The argument of the operator.
    IExpression_unary *create_unary(
        IExpression_unary::Operator const op,
        IExpression const                *argument = NULL,
        int                              start_line = 0,
        int                              start_column = 0,
        int                              end_line = 0,
        int                              end_column = 0) MDL_FINAL;

    /// Create a new binary expression.
    ///
    /// \param op       The operator.
    /// \param left     The left argument of the operator.
    /// \param right    The right argument of the operator.
    IExpression_binary *create_binary(
        IExpression_binary::Operator const op,
        IExpression const *left = NULL,
        IExpression const *right = NULL,
        int start_line = 0,
        int start_column = 0,
        int end_line = 0,
        int end_column = 0) MDL_FINAL;

    /// Create a new conditional expression.
    ///
    /// \param condition    The condition.
    /// \param true_exp     The true expression.
    /// \param false_exp    The false expression.
    IExpression_conditional *create_conditional(
        IExpression const *condition = NULL,
        IExpression const *true_expr = NULL,
        IExpression const *false_expr = NULL,
        int               start_line = 0,
        int               start_column = 0,
        int               end_line = 0,
        int               end_column = 0) MDL_FINAL;

    /// Create a new positional argument.
    IArgument_positional const *create_positional_argument(
        IExpression const *expr,
        int               start_line = 0,
        int               start_column = 0,
        int               end_line = 0,
        int               end_column = 0) MDL_FINAL;

    /// Create a new named argument.
    IArgument_named const *create_named_argument(
        ISimple_name const *parameter_name,
        IExpression const  *expr,
        int                start_line = 0,
        int                start_column = 0,
        int                end_line = 0,
        int                end_column = 0) MDL_FINAL;

    /// Create a new call expression.
    /// \param name     The reference to the constructor or function called.
    IExpression_call *create_call(
        IExpression const *name = NULL,
        int               start_line = 0,
        int               start_column = 0,
        int               end_line = 0,
        int               end_column = 0) MDL_FINAL;

    /// Create a new let expression.
    /// \param exp  The expression in the let.
    IExpression_let *create_let(
        IExpression const *expr = NULL,
        int               start_line = 0,
        int               start_column = 0,
        int               end_line = 0,
        int               end_column = 0) MDL_FINAL;

public:
    /// Constructor.
    ///
    /// \param arena  memory arena to allocate objects
    explicit Expression_factory(Memory_arena &arena);

private:
    // non copyable
    Expression_factory(Expression_factory const &) MDL_DELETED_FUNCTION;
    Expression_factory &operator=(Expression_factory const &) MDL_DELETED_FUNCTION;

private:
    /// The builder for expressions.
    Arena_builder m_builder;
};

///Implementation of the statement factory.
class Statement_factory : public IStatement_factory
{
    typedef IStatement_factory Base;
public:

    /// Create a new invalid statement.
    IStatement_invalid *create_invalid(
        int start_line = 0,
        int start_column = 0,
        int end_line = 0,
        int end_column = 0) MDL_FINAL;

    /// Create a new compound statement.
    IStatement_compound *create_compound(
        int start_line = 0,
        int start_column = 0,
        int end_line = 0,
        int end_column = 0) MDL_FINAL;

    /// Create a new declaration statement.
    IStatement_declaration *create_declaration(
        IDeclaration const *decl = NULL,
        int                start_line = 0,
        int                start_column = 0,
        int                end_line = 0,
        int                end_column = 0) MDL_FINAL;

    /// Create a new expression statement.
    IStatement_expression *create_expression(
        IExpression const *expr = NULL,
        int               start_line = 0,
        int               start_column = 0,
        int               end_line = 0,
        int               end_column = 0) MDL_FINAL;

    /// Create a new conditional statement.
    IStatement_if *create_if(
        IExpression const *cond = NULL,
        IStatement const  *then_stmnt = NULL,
        IStatement const  *else_stmnt = NULL,
        int               start_line = 0,
        int               start_column = 0,
        int               end_line = 0,
        int               end_column = 0) MDL_FINAL;

    /// Create a new switch case.
    IStatement_case *create_switch_case(
        IExpression const *label = NULL,
        int               start_line = 0,
        int               start_column = 0,
        int               end_line = 0,
        int               end_column = 0) MDL_FINAL;

    /// Create a new switch statement.
    IStatement_switch *create_switch(
        IExpression const *cond = NULL,
        int               start_line = 0,
        int               start_column = 0,
        int               end_line = 0,
        int               end_column = 0) MDL_FINAL;

    /// Create a new while loop.
    IStatement_while *create_while(
        IExpression const *cond = NULL,
        IStatement const  *body = NULL,
        int               start_line = 0,
        int               start_column = 0,
        int               end_line = 0,
        int               end_column = 0) MDL_FINAL;

    /// Create a new do-while loop.
    IStatement_do_while *create_do_while(
        IExpression const *cond = NULL,
        IStatement const  *body = NULL,
        int               start_line = 0,
        int               start_column = 0,
        int               end_line = 0,
        int               end_column = 0) MDL_FINAL;

    /// Create a new for loop with an initializing expression.
    IStatement_for *create_for(
        IStatement const  *init = NULL,
        IExpression const *cond = NULL,
        IExpression const *update = NULL,
        IStatement const  *body = 0,
        int               start_line = 0,
        int               start_column = 0,
        int               end_line = 0,
        int               end_column = 0) MDL_FINAL;

    /// Create a new break statement.
    IStatement_break *create_break(
        int start_line = 0,
        int start_column = 0,
        int end_line = 0,
        int end_column = 0) MDL_FINAL;

    /// Create a new continue statement.
    IStatement_continue *create_continue(
        int start_line = 0,
        int start_column = 0,
        int end_line = 0,
        int end_column = 0) MDL_FINAL;

    /// Create a new return statement.
    IStatement_return *create_return(
        IExpression const *expr = NULL,
        int               start_line = 0,
        int               start_column = 0,
        int               end_line = 0,
        int               end_column = 0) MDL_FINAL;

public:
    /// Constructor.
    ///
    /// \param arena  memory arena to allocate objects
    explicit Statement_factory(Memory_arena &arena);

private:
    // non copyable
    Statement_factory(Statement_factory const &) MDL_DELETED_FUNCTION;
    Statement_factory &operator=(Statement_factory const &) MDL_DELETED_FUNCTION;

private:
    /// The builder for statements.
    Arena_builder m_builder;
};

/// Implementation of the Type factory.
class Type_factory : public IType_factory
{
    typedef IType_factory Base;
    friend class Type_cache;

    /// A type cache key.
    struct Type_cache_key {
        enum Kind {
            KEY_FUNC_TYPE,      ///< a function type itself
            KEY_FUNC_KEY,       ///< a function type search key
            KEY_ALIAS,          ///< an alias search key
            KEY_SIZED_ARRAY,    ///< a sized array search key
            KEY_ABSTRACT_ARRAY, ///< an abstract array search key
        };
        Kind kind;

        struct Function_type {
            IType_factory::Function_parameter const * params;
            size_t                                    n_params;
        };

        IType const *type;

        union {
            // empty for KEY_FUNC_TYPE

            // for KEY_FUNC_KEY
            Function_type func;

            // for KEY_ALIAS
            struct {
                ISymbol const         *sym;
                IType::Modifiers      mod;
            } alias;

            // for KEY_SIZED_ARRAY
            struct {
                int                   size;
            } fixed_array;
            struct {
                Type_array_size const *size;
            } abstract_array;
        } u;  // PVS: -V730_NOINIT

        /// Create a key for a function type.
        /*implicit*/ Type_cache_key(IType_function const *func)
        : kind(KEY_FUNC_TYPE), type(func)
        {
        }

        /// Create a key for a function type.
        Type_cache_key(
            IType const                                     *ret,
            IType_factory::Function_parameter const * const params,
            size_t                                          n)
        : kind(KEY_FUNC_KEY), type(ret)
        {
            u.func.params   = params;
            u.func.n_params = n;
        }

        /// Create a key for an alias type.
        Type_cache_key(IType const *t, ISymbol const *sym, IType::Modifiers m)
        : kind(KEY_ALIAS), type(t)
        {
            u.alias.sym = sym;
            u.alias.mod = m;
        }

        /// Create a key for a sized array type.
        Type_cache_key(int size, IType const *t)
        : kind(KEY_SIZED_ARRAY), type(t)
        {
            u.fixed_array.size = size;
        }

        /// Create a key for an abstract array type.
        Type_cache_key(IType const *t, Type_array_size const *abs_size)
        : kind(KEY_ABSTRACT_ARRAY), type(t)
        {
            u.abstract_array.size = abs_size;
        }

        /// Functor to hash a type cache keys.
        struct Hash {
            size_t operator()(Type_cache_key const &key) const
            {
                switch (key.kind) {
                case KEY_FUNC_TYPE:
                    {
                        IType_function const *ft = static_cast<IType_function const *>(key.type);
                        IType const *ret_type = ft->get_return_type();
                        size_t n_params = size_t(ft->get_parameter_count());

                        size_t t = ret_type - (IType const *)0;
                        t = ((t) >> 3) ^ (t >> 16) ^                     //-V2007
                            size_t(KEY_FUNC_TYPE) ^ n_params;

                        for (size_t i = 0; i < n_params; ++i) {
                            IType const *p_type;
                            ISymbol const *p_sym;

                            ft->get_parameter(i, p_type, p_sym);

                            t *= 3;
                            t ^=
                                ((char *)p_type - (char *)0) ^
                                ((char *)p_sym  - (char *)0);
                        }
                        return t;
                    }
                case KEY_FUNC_KEY:
                    {
                        size_t t = key.type - (IType const *)0;
                        t = ((t) >> 3) ^ (t >> 16) ^                     //-V2007
                            size_t(KEY_FUNC_TYPE) ^ key.u.func.n_params;

                        IType_factory::Function_parameter const *p = key.u.func.params;
                        for (size_t i = 0; i < key.u.func.n_params; ++i) {
                            t *= 3;
                            t ^=
                                ((char *)p[i].p_type - (char *)0) ^
                                ((char *)p[i].p_sym  - (char *)0);
                        }
                        return t;
                    }
                case KEY_ALIAS:
                    {
                        size_t t = key.type - (IType const *)0;
                        return ((t) >> 3) ^ (t >> 16) ^
                            size_t(key.kind) ^
                            ((char *)key.u.alias.sym - (char *)0) ^
                            size_t(key.u.alias.mod);
                    }
                case KEY_SIZED_ARRAY:
                    {
                        size_t t = key.type - (IType const *)0;
                        return ((t) >> 3) ^ (t >> 16) ^
                            size_t(key.kind) ^
                            size_t(key.u.fixed_array.size);
                    }
                case KEY_ABSTRACT_ARRAY:
                    {
                        size_t t = key.type - (IType const *)0;
                        return ((t) >> 3) ^ (t >> 16) ^
                            size_t(key.kind) ^
                            ((char *)key.u.abstract_array.size - (char *)0);
                    }
                default:
                    return 0;
                }
            }
        };

        /// Functor to compare two type cache keys.
        struct Equal {
            bool operator() (Type_cache_key const &a, Type_cache_key const &b) const
            {
                if (a.kind != b.kind) {
                    IType_function const *ft = NULL;
                    IType const          *rt;
                    Function_type const  *sk;

                    // compare a function type and a function search key
                    if (a.kind == KEY_FUNC_TYPE && b.kind == KEY_FUNC_KEY) {
                        ft = static_cast<IType_function const *>(a.type);
                        sk = &b.u.func;
                        rt = b.type;
                    } else if (a.kind == KEY_FUNC_KEY && b.kind == KEY_FUNC_TYPE) {
                        ft = static_cast<IType_function const *>(b.type);
                        sk = &a.u.func;
                        rt = a.type;
                    }

                    if (ft != NULL) {
                        if (rt != ft->get_return_type())
                            return false;
                        if (size_t(ft->get_parameter_count()) != sk->n_params)
                            return false;

                        for (size_t i = 0; i < sk->n_params; ++i) {
                            IType const   *p_type;
                            ISymbol const *p_sym;

                            ft->get_parameter(i, p_type, p_sym);

                            if (p_type != sk->params[i].p_type ||
                                p_sym  != sk->params[i].p_sym)
                                return false;
                        }
                        return true;
                    }
                    return false;
                }
                switch (a.kind) {
                case KEY_FUNC_TYPE:
                    return
                        a.type == b.type;
                case KEY_FUNC_KEY:
                    // should be NEVER inside the type hash
                    MDL_ASSERT(!"function search key in type cache detected");
                    return false;
                case KEY_ALIAS:
                    return
                        a.type == b.type &&
                        a.u.alias.sym == b.u.alias.sym &&
                        a.u.alias.mod == b.u.alias.mod;
                case KEY_SIZED_ARRAY:
                    return
                        a.type == b.type &&
                        a.u.fixed_array.size == b.u.fixed_array.size;
                case KEY_ABSTRACT_ARRAY:
                    return
                        a.type == b.type &&
                        a.u.abstract_array.size == b.u.abstract_array.size;
                default:
                    return false;
                }
            }
        };
    };

public:

    /// Create a new type alias instance.
    ///
    /// \param type       The aliased type.
    /// \param name       The alias name, may be NULL.
    /// \param modifiers  The type modifiers.
    IType const *create_alias(
        IType const *type,
        ISymbol const *name,
        IType::Modifiers modifiers) MDL_FINAL;

    /// Create a new type error instance.
    IType_error const *create_error() MDL_FINAL;

    /// Create a new type incomplete instance.
    IType_incomplete const *create_incomplete() MDL_FINAL;

    /// Create a new type bool instance.
    IType_bool const *create_bool() MDL_FINAL;

    /// Create a new type int instance.
    IType_int const *create_int() MDL_FINAL;

    /// Create a new type enum instance.
    ///
    /// \param name The name of the enum.
    IType_enum *create_enum(ISymbol const *name) MDL_FINAL;

    /// Lookup an enum type.
    /// \param name The name of the enum.
    /// \returns the type enum instance or NULL if it does not exist.
    IType_enum const *lookup_enum(char const *name) const MDL_FINAL;

    /// Create a new type float instance.
    IType_float const *create_float() MDL_FINAL;

    /// Create a new type double instance.
    IType_double const *create_double() MDL_FINAL;

    /// Create a new type string instance.
    IType_string const *create_string() MDL_FINAL;

    /// Create a new type bsdf instance.
    IType_bsdf const *create_bsdf() MDL_FINAL;

    /// Create a new type hair_bsdf instance.
    IType_hair_bsdf const *create_hair_bsdf() MDL_FINAL;

    /// Create a new type edf instance.
    IType_edf const *create_edf() MDL_FINAL;

    /// Create a new type vdf instance.
    IType_vdf const *create_vdf() MDL_FINAL;

    /// Create a new type light profile instance.
    IType_light_profile const *create_light_profile() MDL_FINAL;

    /// Create a new type vector instance.
    ///
    /// \param element_type The type of the vector elements.
    /// \param size         The size of the vector.
    IType_vector const *create_vector(
        IType_atomic const *element_type,
        int size) MDL_FINAL;

    /// Create a new type matrix instance.
    ///
    /// \param element_type The type of the matrix elements.
    /// \param columns      The number of columns.
    /// \param rows         The number of rows.
    IType_matrix const *create_matrix(
        IType_vector const *element_type,
        int columns) MDL_FINAL;

    /// Create a new type abstract array instance.
    ///
    /// \param element_type   The element type of the array.
    /// \param abs_name       The absolute name of the array size.
    /// \param sym            The symbol of the abstract array size.
    ///
    /// \return IType_error if element_type was of IType_error, an IType_array instance else.
    IType const *create_array(
        IType const   *element_type,
        ISymbol const *abs_name,
        ISymbol const *sym) MDL_FINAL;

    /// Create a new type sized array instance.
    ///
    /// \param element_type The element type of the array.
    /// \param size         The size of the array.
    ///
    /// \return IType_error if element_type was of IType_error, an IType_array instance else.
    IType const *create_array(
        IType const *element_type,
        size_t      size) MDL_FINAL;

    /// Create a new type color instance.
    IType_color const *create_color() MDL_FINAL;

    /// Create a new type function type instance.
    ///
    /// \param return_type   The return type of the function.
    /// \param parameters    The parameters of the function.
    /// \param n_parameters  The number of parameters.
    IType_function const *create_function(
        IType const                      *return_type,
        Function_parameter const * const parameters,
        size_t                           n_parameters) MDL_FINAL;

    /// Create a new type struct instance.
    ///
    /// \param name  The name of the struct.
    IType_struct *create_struct(ISymbol const *name) MDL_FINAL;

    /// Lookup a struct type.
    /// \param name The name of the struct.
    /// \returns the type struct instance or NULL if it does not exist.
    IType_struct const *lookup_struct(char const *name) const MDL_FINAL;

    /// Create a new type texture sampler instance.
    /// \param texture_type The texture type.
    IType_texture const *create_texture(
        IType_texture::Shape shape) MDL_FINAL;

    /// Create a new type bsdf_measurement instance.
    IType_bsdf_measurement const *create_bsdf_measurement() MDL_FINAL;

    /// Import a type from another type factory.
    ///
    /// \param type  the type to import
    IType const *import(IType const *type) MDL_FINAL;

    /// Return a predefined struct.
    ///
    /// \param part  the ID of the predefined struct
    IType_struct *get_predefined_struct(IType_struct::Predefined_id part) MDL_FINAL;

    /// Return a predefined enum.
    ///
    /// \param part  the ID of the predefined enum
    IType_enum *get_predefined_enum(IType_enum::Predefined_id part) MDL_FINAL;

    /// Return the symbol table of this type factory.
    Symbol_table *get_symbol_table() MDL_FINAL;

    // Non interface methods
public:

    /// Get the equivalent type for a given type in our type factory or return NULL if
    /// type is not imported.
    ///
    /// \param type  the type to import
    ///
    /// \note Similar to import(), but does not create new types and does not support
    ///       function and abstract array types.
    IType const *get_equal(IType const *type) const;

    /// Get the array size for a given absolute abstract array length name.
    ///
    /// \param abs_name       The absolute name of the array size.
    /// \param sym            The symbol of the abstract array size.
    IType_array_size const *get_array_size(
        ISymbol const *abs_name,
        ISymbol const *sym);

    /// Create a new type abstract array instance.
    ///
    /// \param element_type   The element type of the array.
    /// \param array_size     The array size.
    ///
    /// \return IType_error if element_type was of IType_error, an IType_array instance else.
    IType const *create_array(
        IType const            *element_type,
        IType_array_size const *array_size);

    /// Find an sized array type instance.
    ///
    /// \param element_type The element type of the array.
    /// \param size         The size of the array.
    /// \return NULL if this array was not found
    IType const *find_array(IType const *element_type, int size) const;

    /// Find any deferred sized array type with the same base type.
    ///
    /// \param element_type The element type of the array.
    /// \return NULL if no deferred array of this element type was not found
    IType const *find_any_deferred_array(IType const *element_type) const;

    /// Serialize the type table.
    ///
    /// \param serializer  the factory serializer
    void serialize(Factory_serializer &serializer) const;

    /// Deserialize the type table.
    ///
    /// \param deserializer  the factory deserializer
    void deserialize(Factory_deserializer &deserializer);

    /// Check if the given type is owned by this type factory.
    ///
    /// \param owner  the owner id of a type factory
    /// \param type   the type to check
    static bool is_owned(
        size_t      owner,
        IType const *type);

    /// Get the owner id of a type.
    ///
    /// \param type  the type
    ///
    /// \returns the owner id of a type or 0 if the type is not a user defined type
    static size_t get_owner_id(IType const *type);

    /// Checks if this type factory owns the given type
    bool is_owner(IType const *type) const;

    /// Constructs a new type factory.
    ///
    /// \param arena         the memory arena used to allocate new types
    /// \param compiler      the compiler
    /// \param sym_tab       the symbol table for symbols inside types
    explicit Type_factory(
        Memory_arena  &arena,
        IMDL          *compiler,
        Symbol_table  *sym_tab);

private:
    // non copyable
    Type_factory(Type_factory const &) MDL_DELETED_FUNCTION;
    Type_factory &operator=(Type_factory const &) MDL_DELETED_FUNCTION;

private:
    /// Register builtin-types.
    ///
    /// \param serializer  the factory serializer
    void register_builtins(Factory_serializer &serializer) const;

    /// Register builtin-types.
    ///
    /// \param deserializer  the factory deserializer
    void register_builtins(Factory_deserializer &deserializer);

private:
    /// The builder for types.
    Arena_builder m_builder;

    /// The id of this factory, for debugging.
    size_t const m_id;

    /// The type factory of the compiler itself or NULL.
    IType_factory * const m_compiler_factory;

    /// The symbol table used to create new symbols for types.
    Symbol_table * const m_symtab;

    /// Predefined structs (material parts).
    Type_struct *m_predefined_structs[IType_struct::SID_LAST + 1];

    /// Predefined enums (used in constructors of default types).
    Type_enum *m_predefined_enums[IType_enum::EID_LAST + 1];

    /// Hashtable of cached types.
    typedef Arena_hash_map<
        Type_cache_key,
        IType const *,
        Type_cache_key::Hash,
        Type_cache_key::Equal>::Type Type_cache;

    /// Cache of composed immutable types that could be reused (alias, array, function types).
    Type_cache m_type_cache;

    /// Hashtable of cached abstract array sizes.
    typedef Arena_hash_map<
        ISymbol const *,
        Type_array_size const *,
        Hash_ptr<ISymbol const>,
        Equal_ptr<ISymbol const> >::Type Array_size_cache;

    /// Cache of array sizes that could be reused.
    Array_size_cache m_array_size_cache;

    /// Hashtable of imported user types.
    typedef Arena_hash_map<
        char const *,
        IType const *,
        cstring_hash,
        cstring_equal_to>::Type Type_import_map;

    // Cache of imported user types (struct and enums).
    Type_import_map m_imported_types_cache;
};

/// Implementation of the Annotation factory.
class Annotation_factory : public IAnnotation_factory
{
    typedef IAnnotation_factory Base;
public:

    /// Create a new annotation.
    ///
    /// \param name          name of the annotation
    /// \param start_line    start line of the annotation
    /// \param start_column  start column of the annotation
    /// \param end_line      end line of the annotation
    /// \param end_column    end column of the annotation
    IAnnotation *create_annotation(
        IQualified_name const *name,
        int                   start_line = 0,
        int                   start_column = 0,
        int                   end_line = 0,
        int                   end_column = 0) MDL_FINAL;

    /// Create a new annotation block.
    ///
    /// \param start_line    start line of the annotation block
    /// \param start_column  start column of the annotation block
    /// \param end_line      end line of the annotation block
    /// \param end_column    end column of the annotation block
    IAnnotation_block *create_annotation_block(
        int start_line = 0,
        int start_column = 0,
        int end_line = 0,
        int end_column = 0) MDL_FINAL;

public:
    /// Constructor.
    ///
    /// \param arena  memory arena to allocate objects
    explicit Annotation_factory(Memory_arena &arena);

private:
    // non copyable
    Annotation_factory(Annotation_factory const &) MDL_DELETED_FUNCTION;
    Annotation_factory &operator=(Annotation_factory const &) MDL_DELETED_FUNCTION;

private:
    /// The arena builder for annotations.
    Arena_builder m_builder;
};

/// Implementation of the Value factory.
class Value_factory : public IValue_factory
{
    typedef IValue_factory Base;

    struct IValue_hash {
        size_t operator()(IValue const *value) const;
    };

    struct IValue_equal {
        bool operator()(IValue const *a, IValue const *b) const;
    };

    typedef hash_set<IValue const *, IValue_hash, IValue_equal>::Type Value_table;
public:

    /// Create the bad value.
    IValue_bad const *create_bad() MDL_FINAL;

    /// Create a new value of type boolean.
    ///
    /// \param value The value of the boolean.
    IValue_bool const *create_bool(bool value) MDL_FINAL;

    /// Create a new value of type integer.
    ///
    /// \param value The value of the integer.
    IValue_int const *create_int(int value) MDL_FINAL;

    /// Create a new value of type enum.
    ///
    /// \param type     The type of the enum.
    /// \param index    The index of the enum constant inside the type.
    IValue_enum const *create_enum(IType_enum const *type, size_t index) MDL_FINAL;

    /// Create a new value of type float.
    ///
    /// \param value The value of the float.
    IValue_float const *create_float(float value) MDL_FINAL;

    /// Create a new value of type double.
    ///
    /// \param value The value of the double.
    IValue_double const *create_double(double value) MDL_FINAL;

    /// Create a new value of type string.
    ///
    /// \param value The value of the string, NULL is not allowed.
    IValue_string const *create_string(char const *value) MDL_FINAL;

    /// Create a new value of type vector.
    ///
    /// \param type    The type of the vector.
    /// \param values  The values for the elements of the matrix columns.
    /// \param size    The number of values, must match the vector size.
    IValue_vector const *create_vector(
        IType_vector const   *type,
        IValue const * const values[],
        size_t               size) MDL_FINAL;

    /// Create a new value of type matrix.
    ///
    /// \param type    The type of the matrix.
    /// \param values  The values for the elements of the matrix columns.
    /// \param size    The number of values, must match the column size.
    IValue_matrix const *create_matrix(
        IType_matrix const   *type,
        IValue const * const values[],
        size_t               size) MDL_FINAL;

    /// Create a new value of type array.
    ///
    /// \param type    The type of the array.
    /// \param values  The values for the elements of the array.
    /// \param size    The number of values, must match the array size.
    IValue_array const *create_array(
        IType_array const    *type,
        IValue const * const values[],
        size_t               size) MDL_FINAL;

    /// Create a new RGB value of type color.
    ///
    /// \param value_r  The (float) value for the red channel.
    /// \param value_g  The (float) value for the green channel.
    /// \param value_b  The (float) value for the blue channel.
    IValue_rgb_color const *create_rgb_color(
        IValue_float const *value_r,
        IValue_float const *value_g,
        IValue_float const *value_b) MDL_FINAL;

    /// Create a new value of type color from a spectrum.
    ///
    /// \param wavelengths  The (array of float) values for the wavelengths.
    /// \param amplitudes  The (array of float) values for the amplitudes.
    ///
    /// \return IValue_bad if the arrays are not of same size or not float arrays.
    IValue const *create_spectrum_color(
        IValue_array const *wavelengths,
        IValue_array const *amplitudes) MDL_FINAL;

    /// Create a new value of type struct.
    /// 
    /// \param type    The type of the struct.
    /// \param values  The values for the fields of the struct.
    /// \param size    The number of values, must match the number of fields.
    IValue_struct const *create_struct(
        IType_struct const   *type,
        IValue const * const values[],
        size_t               size) MDL_FINAL;

    /// Create a new texture value.
    ///
    /// \param type            The type of the texture.
    /// \param value           The string value of the texture, NULL is not allowed.
    /// \param gamma           The gamma override value of the texture.
    /// \param tag_value       The tag value of the texture.
    /// \param tag_version     The version of the tag value.
    IValue_texture const *create_texture(
        IType_texture const            *type,
        char const                     *value,
        IValue_texture::gamma_mode     gamma,
        int                            tag_value,
        unsigned                       tag_version) MDL_FINAL;

    /// Create a new bsdf_data texture value.
    ///
    /// \param bsdf_data_kind  The BSDF data kind.
    /// \param tag_value       The tag value of the texture.
    /// \param tag_version     The version of the tag value.
    IValue_texture const *create_bsdf_data_texture(
        IValue_texture::Bsdf_data_kind bsdf_data_kind,
        int                            tag_value,
        unsigned                       tag_version) MDL_FINAL;

    /// Create a new string light profile value.
    ///
    /// \param type         The type of the light profile.
    /// \param value        The string value of the light profile, NULL is not allowed.
    /// \param tag_value    The tag value of the light profile.
    /// \param tag_version  The version of the tag value.
    IValue_light_profile const *create_light_profile(
        IType_light_profile const *type,
        char const                *value,
        int                       tag_value,
        unsigned                  tag_version) MDL_FINAL;

    /// Create a new string bsdf measurement value.
    ///
    /// \param type         The type of the light profile.
    /// \param value        The string value of the bsdf measurement, NULL is not allowed.
    /// \param tag_value    The tag value of the bsdf measurement.
    /// \param tag_version  The version of the tag value.
    IValue_bsdf_measurement const *create_bsdf_measurement(
        IType_bsdf_measurement const *type,
        char const                   *value,
        int                          tag_value,
        unsigned                     tag_version) MDL_FINAL;

    /// Create a new invalid reference.
    ///
    /// \param type     The type of the reference.
    IValue_invalid_ref const *create_invalid_ref(
        IType_reference const *type) MDL_FINAL;

    /// Create a new compound value.
    ///
    /// \param type    The compound type.
    /// \param values  The values for the compound.
    /// \param size    The number of values, must match the number of compound elements.
    IValue_compound const *create_compound(
        IType_compound const *type,
        IValue const * const values[],
        size_t               size) MDL_FINAL;

    /// Create a additive neutral zero if supported for the given type.
    ///
    /// \param type   The type of the constant to create
    /// \return A zero constant or IValue_bad if type does not support addition.
    IValue const *create_zero(IType const *type) MDL_FINAL;

    /// Return the type factory of this value factory.
    ///
    /// Note: the returned type factory can create built-in types only.
    Type_factory *get_type_factory() MDL_FINAL;

    /// Import a value from another value factory.
    ///
    /// \param value  the value to import
    IValue const *import(IValue const *value) MDL_FINAL;

    // Non interface methods
public:
    /// Dump all values owned by this Value table.
    void dump() const;

    typedef Value_table::const_iterator const_value_iterator;

    /// Get the begin value of this Value factory.
    ///
    /// \return an iterator of the first owned value of this value factory
    const_value_iterator values_begin() const;

    /// Get the end value of this Value factory.
    ///
    /// \return an iterator of the end owned value of this value factory
    const_value_iterator values_end() const;

    /// Serialize the type table.
    ///
    /// \param serializer  the factory serializer
    void serialize(Factory_serializer &serializer) const;

    /// Deserialize the type table.
    ///
    /// \param deserializer  the factory deserializer
    void deserialize(Factory_deserializer &deserializer);

    /// Check if this value factory is the owner of the given value.
    bool is_owner(IValue const *value) const;

    /// Get the allocator.
    IAllocator *get_allocator() const { return m_builder.get_arena()->get_allocator(); }

    /// Constructor.
    ///
    /// \param arena   the memory arena used to allocate values
    /// \param tf      the type factory for types of the values
    explicit Value_factory(Memory_arena &arena, Type_factory &tf);

private:
    // non copyable
    Value_factory(Value_factory const &) MDL_DELETED_FUNCTION;
    Value_factory &operator=(Value_factory const &) MDL_DELETED_FUNCTION;

private:
    /// The builder for values.
    Arena_builder m_builder;

    /// A type factory, use to get the atomic types.
    Type_factory & m_tf;

    /// The value table.
    Value_table m_vt;

    /// The bad value.
    IValue_bad const *const m_bad_value;

    /// The true value.
    IValue_bool const *const m_true_value;

    /// The false value.
    IValue_bool const *const m_false_value;
};

/// Skip all presets returning the original function declaration.
///
/// \param[in]    func_decl  a function declaration
/// \param[inout] owner_mod  the owner module of a function declaration
///
/// \return func_decl itself if this is not a preset, the original definition otherwise
IDeclaration_function const *skip_presets(
    IDeclaration_function const     *func_decl,
    mi::base::Handle<IModule const> &owner_mod);

/// Skip all presets returning the original function definition.
///
/// \param[in]    func_def   a function definition
/// \param[inout] owner_mod  the owner module of a function definition
///
/// \return func_decl itself if this is not a preset, the original definition otherwise
IDefinition const *skip_presets(
    IDefinition const               *func_def,
    mi::base::Handle<IModule const> &owner_mod);

}  // mdl
}  // mi

#endif // MDL_COMPILERCORE_FACTORIES_H
