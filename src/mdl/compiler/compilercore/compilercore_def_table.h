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

#ifndef MDL_COMPILERCORE_DEF_TABLE_H
#define MDL_COMPILERCORE_DEF_TABLE_H 1

#include <mi/mdl/mdl_iowned.h>
#include <mi/mdl/mdl_types.h>
#include <mi/mdl/mdl_definitions.h>
#include <mi/mdl/mdl_expressions.h>

#include "compilercore_cc_conf.h"
#include "compilercore_hash_ptr.h"
#include "compilercore_rawbitset.h"
#include "compilercore_allocator.h"
#include "compilercore_memory_arena.h"
#include "compilercore_assert.h"
#include "compilercore_dynamic_memory.h"
#include "compilercore_serializer.h"

namespace mi {
namespace mdl {

/// Forward declaration of the interface to declarations.
class IDeclaration;
class IDeclaration_namespace_alias;
class Position;
class IType;
class IType_error;
class ISymbol;
class Scope;
class Definition_table;
class Module;
class Printer;

/// Implementation of a definition.
class Definition : public IDefinition
{
    typedef IDefinition Base;
    friend class Scope;
    friend class Definition_table;
    friend class Module;
    friend class Arena_builder;

    /// A helper struct containing all initializer expressions.
    struct Initializers {
        size_t count;                   ///< number of initializer expressions
        IExpression const *exprs[1];    ///< the initializer expressions itself
    };
public:
    enum Flag {
        DEF_IS_PREDEFINED,          ///< This is a predefined definition.
        DEF_IS_DECL_ONLY,           ///< This is a declaration only, NOT a definition.
        DEF_IS_INCOMPLETE,          ///< This definition is not completed yet.
        DEF_IS_EXPLICIT,            ///< This is a explicit constructor.
        DEF_IS_EXPLICIT_WARN,       ///< This is a explicit constructor, but we allow implicit use.
        DEF_IS_COMPILER_GEN,        ///< This definition is compiler generated.
        DEF_IS_EXPORTED,            ///< This definition is exported.
        DEF_IS_IMPORTED,            ///< This definition is imported.
        DEF_OP_LVALUE,              ///< The first argument of this operator must be an lvalue.
        DEF_IGNORE_OVERLOAD,        ///< This definition is ignored in overload resolution.
        DEF_IS_CONST_CONSTRUCTOR,   ///< This definition constructs constant values.
        DEF_IS_REACHABLE,           ///< This definition is reachable from an exported entity.
        DEF_IS_STDLIB,              ///< This is a stdlib function.
        DEF_IS_USED,                ///< This definition is used.
        DEF_IS_UNUSED,              ///< This definition is explicitly marked as unused.
        DEF_IS_DEPRECATED,          ///< This definition is explicitly marked as deprecated.
        DEF_USED_INCOMPLETE,        ///< This definition was used incomplete.
        DEF_IS_UNIFORM,             ///< This definition has the uniform modifier.
        DEF_IS_VARYING,             ///< This definition has the varying modifier.
        DEF_IS_WRITTEN,             ///< This definition's entity is written to.
        DEF_IS_LET_TEMPORARY,       ///< This definition is a let temporary.
        DEF_NO_INLINE,              ///< This definition should never be inlined.
        DEF_USES_STATE,             ///< This function uses the state.
        DEF_USES_TEXTURES,          ///< This function uses textures.
        DEF_USES_SCENE_DATA,        ///< This function uses scene data.
        DEF_CAN_THROW_BOUNDS,       ///< This function can throw a bounds exception.
        DEF_CAN_THROW_DIVZERO,      ///< This function can throw a division by zero exception.
        DEF_REF_BY_DEFAULT_INIT,    ///< This function is referenced by a default initializer.
        DEF_READ_TEXTURE_ATTR,      ///< This function reads texture attributes (width, etc).
        DEF_READ_LP_ATTR,           ///< This function reads light_profile attributes (power, etc).
        DEF_USES_VARYING_STATE,     ///< This function uses the varying state.
        DEF_USES_DEBUG_CALLS,       ///< This function uses calls to ::debug module functions.
        DEF_USES_OBJECT_ID,         ///< This function uses calls to state::object_id().
        DEF_USES_TRANSFORM,         ///< This function uses calls to state::transform*().
        DEF_USES_NORMAL,            ///< This function uses calls to state::normal().
        DEF_IS_NATIVE,              ///< This function is declared native.
        DEF_IS_CONST_EXPR,          ///< This function is declared const_expr.
        DEF_USES_DERIVATIVES,       ///< This function uses derivatives.
        DEF_IS_DERIVABLE,           ///< This parameter or return type is derivable.
        DEF_LITERAL_PARAM,          ///< The argument of the first parameter must be a literal.
        DEF_LAST
    };

    /// RAII like helper class that temporary set/reset a flag on a definition.
    class Scope_flag {
    public:
        Scope_flag(Definition &def, Flag flag) : m_def(def), m_flag(flag) {
            if (!is<IType_error>(m_def.get_type()))
                m_def.set_flag(m_flag);
        }

        ~Scope_flag() {
            if (!is<IType_error>(m_def.get_type()))
                m_def.clear_flag(m_flag);
        }
    private:
        Definition  &m_def;
        Flag        m_flag;
    };

    /// Returns the kind of this definition.
    Kind get_kind() const MDL_FINAL;

    /// Get the symbol of the definition.
    ///
    /// \note: inside the compiler use \link get_sym() instead.
    ISymbol const *get_symbol() const MDL_FINAL;

    /// Get the type of the definition.
    IType const *get_type() const MDL_FINAL;

    /// Get the declaration of the definition.
    IDeclaration const *get_declaration() const MDL_FINAL;

    /// Get the default expression of a parameter of a function, constructor or annotation.
    ///
    /// \param param_idx  the index of the parameter
    IExpression const *get_default_param_initializer(int param_idx) const MDL_FINAL;

    /// Return the value of an enum constant or a global constant.
    IValue const *get_constant_value() const MDL_FINAL;

    /// Return the field index of a field member.
    int get_field_index() const MDL_FINAL;

    /// Return the semantics of a function/constructor.
    Semantics get_semantics() const MDL_FINAL;

    /// Return the parameter index of a parameter.
    int get_parameter_index() const MDL_FINAL;

    /// Return the namespace of a namespace alias.
    ISymbol const *get_namespace() const MDL_FINAL;

    /// Get the prototype declaration of the definition if any.
    IDeclaration const *get_prototype_declaration() const MDL_FINAL;

    /// Get a boolean property of this definition.
    ///
    /// \param prop  the requested property
    bool get_property(Property prop) const MDL_FINAL;

    /// Return the position of this definition if any.
    Position const *get_position() const MDL_FINAL;

    /// Set the position of this definition if any.
    ///
    /// \param pos  the new position
    void set_position(Position const *pos) MDL_FINAL;

    /// Return the mask specifying which parameters of a function are derivable.
    ///
    /// For example, if bit 0 is set, a backend supporting derivatives may provide derivative
    /// values as the first parameter of the function.
    virtual unsigned get_parameter_derivable_mask() const MDL_FINAL;

    // Non interface member

    /// Change the type of the definition.
    ///
    /// \param type  the new type
    ///
    /// \note Only allowed for variables and auto-typing.
    void set_type(IType const *type);

    /// Set the (syntactical) declaration of the definition.
    ///
    /// \param decl  the declaration of this definition
    void set_declaration(IDeclaration const *decl);

    /// Set the (syntactical) prototype declaration of the definition.
    ///
    /// \param decl  the prototype declaration of this definition
    void set_prototype_declaration(IDeclaration const *decl);

    /// Returns the associated symbol for this definition.
    ISymbol const *get_sym() const { return m_sym; }

    /// Set the default expression of a parameter of a function, constructor or annotation.
    ///
    /// \param param_idx  the index of the parameter
    /// \param expr       the initializer expression
    void set_default_param_initializer(size_t param_idx, IExpression const *expr);

    /// Return the definition scope of this definition.
    Scope *get_def_scope() const { return m_def_scope; }

    /// Return scope that this definition creates.
    Scope *get_own_scope() const { return m_own_scope; }

    /// Set the scope that this definition creates.
    void set_own_scope(Scope *scope);

    /// Link a previous definition for the same symbol.
    ///
    /// Note that the symbol table is updated, and shows a list of same
    /// definitions that occurred before the current one.
    ///
    void link_same_def(Definition *prev_def) {
        m_same_prev           = prev_def;
        prev_def->m_same_next = this;
    }

    /// Returns the next Definition in the parent scope.
    Definition *get_next_def_in_scope() const { return m_next; }

    /// Return the previous definition for the same symbol.
    Definition *get_prev_def() const { return m_same_prev; }

    /// Return the next definition for the same symbol.
    Definition *get_next_def() const { return m_same_next; }

    /// Return the definition for this symbol in the outer scope.
    Definition *get_outer_def() const { return m_outer_def; }

    /// Set the definition for this symbol in the outer scope.
    ///
    /// \param outer  the outer definition
    void set_outer_def(Definition *outer) { m_outer_def = outer; }

    /// Check if this definition has the given flag.
    ///
    /// \param flag  the flag to check
    bool has_flag(Flag flag) const { return m_flags.test_bit(flag); }

    /// Set the given flag.
    ///
    /// \param flag  the flag to set
    void set_flag(Flag flag) { m_flags.set_bit(flag); }

    /// Clear the given flag.
    ///
    /// param flag  the flag to clear
    void clear_flag(Flag flag) { m_flags.clear_bit(flag); }

    /// Get the version flags of this definition.
    unsigned get_version_flags() const { return m_version_flags; }

    /// Set the version flags of this definition.
    ///
    /// \param flags  the version flags
    void set_version_flags(unsigned flags) { m_version_flags = flags;}

    /// Set the mask specifying which parameters are derivable.
    ///
    /// \param mask  the bit mask
    void set_parameter_derivable_mask(unsigned mask) { m_parameter_deriv_mask = mask; }

    /// Return the definite definition for this definition (which represents a
    /// declaration in the semantic sense).
    Definition *get_definite_definition() {
        if (has_flag(DEF_IS_DECL_ONLY))
            return m_definite_def;
        return this;
    }

    /// Return the definite definition for this definition (which represents a
    /// declaration in the semantic sense).
    Definition const *get_definite_definition() const {
        if (has_flag(DEF_IS_DECL_ONLY))
            return m_definite_def;
        return this;
    }

    /// Set the definite definition for this definition.
    ///
    /// \param definite_def  the definite definition
    void set_definite_definition(Definition *definite_def) {
        MDL_ASSERT(has_flag(DEF_IS_DECL_ONLY));
        m_definite_def = definite_def;
    }

    /// Return the unique ID of this definition in this definition table.
    size_t get_unique_id() const { return m_unique_id; }

    /// Return the unique ID of the owner module of this definition.
    size_t get_owner_module_id() const { return m_owner_module_id; }

    /// Return the original unique ID of this definition.
    size_t get_original_unique_id() const { return m_original_unique_id; }

    /// Return the import index of the original owner module or 0 if not imported.
    size_t get_original_import_idx() const { return m_original_module_import_idx; }

    /// Copy the default initializers from one definition to another.
    ///
    /// \param module    the owner module of this definition
    /// \param prev_def  the definition from which the init8ializers should be copied
    ///
    /// \note: both definitions must be in the same module
    void copy_initializers(Module *module, Definition const *prev_def);

    /// Set the value of an enum or global constant.
    ///
    /// \param value  the value of the constant
    void set_constant_value(IValue const *value);

    /// Set the field index of a member field.
    ///
    /// \param index  the index of this member field
    void set_field_index(int index);

    /// Set the semantic of a function or constructor.
    ///
    /// \param sema  the semantic
    void set_semantic(Semantics sema);

    /// Set the parameter index of a parameter.
    ///
    /// \param index  the index of this parameter
    void set_parameter_index(int index);

    /// Set the namespace of a namespace alias.
    ///
    /// \param name_space  the namespace
    void set_namespace(
        ISymbol const *name_space);

private:
    /// Constructor.
    explicit Definition(
        Kind           kind,
        size_t         module_id,
        ISymbol const  *sym,
        IType const    *type,
        Position const *pos,
        Scope          *parent_scope,
        Definition     *outer,
        size_t         id);

    /// Creates a new definition from an imported one.
    explicit Definition(
        Definition const &other,
        ISymbol const    *imp_sym,
        IType const      *imp_type,
        Position const   *imp_pos,
        Scope            *parent_scope,
        Definition       *outer,
        size_t           module_id,
        size_t           id,
        size_t           owner_import_idx);

private:
    /// The kind of the definition.
    Kind const m_kind;

    /// The unique id of this definition.
    size_t const m_unique_id;

    /// The unique id of the owner module of this definition.
    size_t const m_owner_module_id;

    /// If this Definition is imported, its original id, else equal to m_unique_id.
    size_t m_original_unique_id;

    /// If the original module that defined this definition is imported, its import id, else 0.
    size_t m_original_module_import_idx;

    /// The symbol of this definition.
    ISymbol const * const m_sym;

    /// The type of this definition.
    IType const * m_type;

    /// The parameter default initializer of this definition if any.
    Initializers *m_parameter_inits;

    /// The scope where this definition was defined.
    Scope * const m_def_scope;

    /// The scope that this definition owns.
    Scope * m_own_scope;

    /// The position of this definition if any.
    Position const *m_pos;

    /// The declaration of this definition once set.
    IDeclaration const *m_decl;

    /// The prototype declaration of this definition once set.
    IDeclaration const *m_proto_decl;

    /// Points to the next definition in this scope.
    Definition *m_next;

    /// Points to the previous definition of the same symbol in this scope.
    Definition *m_same_prev;

    /// Points to the next definition of the same symbol in this scope.
    Definition *m_same_next;

    /// Points to the definition of the same symbol in outer scope.
    Definition *m_outer_def;

    /// If this definition is a declaration only, this is the definite definition.
    Definition *m_definite_def;

    /// A value coupled with the definition.
    IValue const *m_value;

    union {
        unsigned  code;                 ///< Biggest type for bitwise copy.
        int       field_index;          ///< For DK_MEMBER, the field index.
        int       param_index;          ///< For DK_PARAMETER, the parameter index.
        Semantics sema_code;            ///< For DK_FUNCTION, DK_OPERATOR, DK_CONSTRUCTOR, and
                                        ///  DK_ANNOTATION the semantics.
        ISymbol const *name_space;      ///< For DK_NAMESPACE, the namespace.
    } m_u;

    /// Version flags of this definition.
    unsigned m_version_flags;

    /// Flags of this definition.
    Raw_bitset<DEF_LAST> m_flags;

    /// Mask specifying which parameters of a function may expect derivable values.
    unsigned m_parameter_deriv_mask;
};

/// An interface for visiting definitions
class IDefinition_visitor {
public:
    /// Called for every visited definition.
    ///
    /// \param def  the definition
    virtual void visit(Definition const *def) const = 0;
};

///
/// An environment scope.
///
/// Holds all entities declared or imported inside a scope.
class Scope : public Interface_owned
{
    typedef Interface_owned Base;
    friend class Arena_builder;
    friend class Definition_table;
    friend class Module;

public:
    typedef list<Definition const *>::Type Definition_list;

    /// Return the unique id of this scope.
    inline size_t get_unique_id() const {
        return m_id;
    }

    /// Return the parent scope.
    inline Scope *get_parent() const {
        return m_parent;
    }

    /// Return the owner definition of this scope.
    inline Definition const *get_owner_definition() const {
        return m_owner_definition;
    }

    /// Set the owner definition of this scope.
    ///
    /// \param owner_def  the owner definition of this scope
    void set_owner_definition(Definition const *owner_def);

    /// Return the associated type of this scope.
    inline IType const *get_scope_type() const {
        return m_scope_type;
    }

    /// Return the associated name of this scope.
    inline ISymbol const *get_scope_name() const {
        return m_scope_name;
    }

    /// Add a definition to this scope.
    ///
    /// \param def  the definition to add
    inline void add_definition(Definition *def) {
        def->m_next   = m_definitions;
        m_definitions = def;
    }

    /// Returns the first definition in this scope.
    inline Definition *get_first_definition_in_scope() const {
        return m_definitions;
    }

    /// Re-enter the definitions for all defined symbols in the scope.
    ///
    /// \param owner_table  the definition table that owns this scope
    void enter_definitions(Definition_table *owner_table) const;

    /// Restore the definitions for all defined symbols in the scope.
    ///
    /// \param owner_table  the definition table that owns this scope
    void restore_definitions(Definition_table *owner_table);

    /// Find a definition inside this scope only.
    ///
    /// \param sym  the name of the entity to lookup
    Definition *find_definition_in_scope(ISymbol const *sym) const;

    /// Find the definition of the given ID in this scope only.
    ///
    /// \param ID  the ID of the definition
    Definition *find_definition_in_scope(size_t id) const;

    /// Find a definition inside this scope or parent scopes.
    ///
    /// \param sym  the name of the entity to lookup
    Definition *find_def_in_scope_or_parent(ISymbol const *sym) const;

    /// Find a named sub-scope.
    ///
    /// \param name  the name of the sub-scope
    Scope *find_named_subscope(ISymbol const *name) const;

    /// Get the first named sub-scope.
    Scope *get_first_named_subscope() const;

    /// Get the next named sub-scope.
    Scope *get_next_named_subscope() const;

    /// Returns the scope depth.
    inline size_t get_depth() const { return m_depth; }

    /// Walk over all definition inside this scope and all its sub-scopes.
    ///
    /// \param visitor  the visitor
    void walk(IDefinition_visitor const *visitor) const;

    /// Returns true if this is an empty scope that can be thrown away.
    bool is_empty() const;

    /// Collect the definitions of all enum values of a given type.
    ///
    /// \param e_type  the enum type
    /// \param values  all found values will be added to this list
    ///
    /// \note Because enum values are not inside the type scope in MDL, we
    ///       need a way to collecting them.
    void collect_enum_values(IType_enum const *e_type, Definition_list &values);

    /// Get the first sub scope of a scope.
    Scope const *get_first_subscope() const { return m_sub_scopes; }

    /// Get the next (sibling) sub-scope.
    Scope const *get_next_subscope() const { return m_next_subscope; }

private:
    /// Creates a new environmental scope.
    ///
    /// \param parent     the parent scope or NULL
    /// \param id         an unique id for identifying this scope
    /// \param owner_def  the owner definition of this scope or NULL
    /// \param type       the type creating the scope or NULL
    /// \param name       the type name of the scope or NULL
    explicit Scope(
        Scope            *parent,
        size_t           id,
        Definition const *owner_def,
        IType const      *type,
        ISymbol const    *name);

    /// Creates a new named scope.
    ///
    /// \param parent   the parent scope or NULL
    /// \param name     the name of the scope to create
    /// \param id       an unique id for identifying this scope
    explicit Scope(
        Scope         *parent,
        ISymbol const *name,
        size_t        id);

    /// Remove this scope from its parent sub-scopes.
    void remove_from_parent();

    /// Serialize this scope.
    ///
    /// \param dt          the owning definition table
    /// \param serializer  the module serializer
    void serialize(
        Definition_table const &dt,
        Module_serializer      &serializer) const;

    /// Deserialize this scope.
    ///
    /// \param dt            the owning definition table
    /// \param deserializer  the module deserializer
    void deserialize(
        Definition_table    &dt,
        Module_deserializer &deserializer);

    /// Debug helper: Print this scope to the given printer.
    ///
    /// \param alloc     an allocator for temporary space
    /// \param printer   the printer
    /// \param indent    the indentation depth
    /// \param is_owned  true is printed by its owner
    void dump(
        IAllocator *alloc,
        Printer    *printer,
        size_t     indent,
        bool       is_owned) const;

private:
    /// List of all definitions in this scope.
    Definition *m_definitions;

    /// The owner definition of this scope if any.
    Definition const *m_owner_definition;

    /// Points to the parent scope.
    Scope *m_parent;

    /// Points to the head of all sub-scopes.
    Scope *m_sub_scopes;

    /// Points to the last sub-scope of the parent.
    Scope *m_last_sub_scope;

    /// Points to the next sub-scope of the parent.
    Scope *m_next_subscope;

    /// Points to the previous sub-scope of the parent.
    Scope *m_prev_subscope;

    /// If this scope represents a type, the associated one, else NULL.
    IType const *m_scope_type;

    /// If this scope represents a named scope or a type, the associated name, else NULL.
    ISymbol const *m_scope_name;

    /// An unique id of this scope.
    size_t m_id;

    /// The depth of this scope.
    size_t m_depth;
};

/// The definition table.
class Definition_table {
    friend class Module;
public:
    /// Enter a new scope empty scope for deserialization.
    ///
    /// \param deserializer  the Module deserializer
    Scope *enter_scope(Module_deserializer &deserializer);

    /// Enter a new scope.
    ///
    /// \param def  the definition that owns this scope or NULL
    Scope *enter_scope(Definition const *def);

    /// Enter a new scope created by a type declaration.
    ///
    /// \param type      the type defined by this scope
    /// \param type_def  the definition of the type
    Scope *enter_scope(IType const *type, Definition const *type_def);

    /// Enter a named scope (module or package import).
    ///
    /// \param name  the scope name
    Scope *enter_named_scope(ISymbol const *name);

    /// Leave the current scope.
    ///
    /// This restores the definitions that were current before this scope was entered.
    ///
    /// The current scope will be set to the parent scope of the leaved one.
    void leave_scope();

    /// Reopen an already leaved scope.
    ///
    /// \param scope  the scope to open
    ///
    /// Node that scope must be a sub-scope of the current scope.
    void reopen_scope(Scope *scope);

    /// Do a transition to the given scope.
    ///
    /// \param scope  this scope will be the new current one, can be NULL
    ///
    /// Leave and enter scopes until the given scope is reached.
    void transition_to_scope(Scope *scope);

    /// Enter a new (entity) definition.
    ///
    /// \param kind    the kind of the definition to enter
    /// \param symbol  the symbol of the definition
    /// \param type    the type of the entity
    /// \param pos     the position of the symbol
    Definition *enter_definition(
        Definition::Kind kind,
        ISymbol const    *symbol,
        IType const      *type,
        Position const   *pos);

    /// Enter a new operator definition.
    ///
    /// \param kind    the operator kind of the definition to enter
    /// \param symbol  the symbol of the definition
    /// \param type    the type of the entity
    Definition *enter_operator_definition(
        IExpression::Operator kind,
        ISymbol const         *symbol,
        IType const           *type);

    /// Enter an error definition for the given symbol.
    ///
    /// \param symbol    this symbol will be defined in the current scope as an error
    /// \param err_type  the error type
    Definition *enter_error(ISymbol const *symbol, const IType_error *err_type);

    /// Return the current scope of the compilation unit.
    Scope *get_curr_scope() const { return m_curr_scope; }

    /// Return the outer scope of the compilation unit.
    /// This scope contains the predefined definitions.
    Scope *get_predef_scope() const { return m_predef_scope; }

    /// Return the global scope of the compilation unit.
    Scope *get_global_scope() const { return m_global_scope; }

    /// Return a scope by its type.
    ///
    /// \param type  the type to lookup
    Scope *get_type_scope(IType const *type) const;

    /// Return the current definition for a symbol in this definition table.
    ///
    /// \param sym  the symbol
    Definition *get_definition(ISymbol const *sym) const;

    /// Set the current definition for a symbol in this definition table.
    ///
    /// \param sym  the symbol
    /// \param def  the current definition for this symbol
    void set_definition(ISymbol const *sym, Definition *def);

    /// Restore the current definition for a symbol in this definition table.
    ///
    /// \param sym  the symbol
    /// \param def  the restored definition for this symbol
    void restore_definition(ISymbol const *sym, Definition *def);

    /// Return the definition for an operator in this definition table.
    ///
    /// \param op  the operator kind
    Definition *get_operator_definition(IExpression::Operator op) const;

    /// Set the definition for an operator in this definition table.
    ///
    /// \param op   the operator kind
    /// \param def  the current definition for this operator
    void set_operator_definition(IExpression::Operator op, Definition *def);

    /// Enter an imported definition.
    ///
    /// \param imported           the definition that is imported
    /// \param owner_import_idx   the index of the original owner in the current import table
    Definition *import_definition(
        Definition const *imported,
        size_t           owner_import_idx);

    /// Get a namespace alias.
    ///
    /// \param alias  the symbol of the alias
    ///
    /// \return the namespace definition if this alias is known, NULL otherwise
    Definition const *get_namespace_alias(
        ISymbol const *alias);

    /// Enter a new namespace alias.
    ///
    /// \param alias  the alias
    /// \param ns     the namespace
    /// \param decl   the namespace alias declaration
    Definition *enter_namespace_alias(
        ISymbol const                      *alias,
        ISymbol const                      *ns,
        IDeclaration_namespace_alias const *decl);

    /// Walk over all definitions of this definition table.
    ///
    /// \param visitor  the visitor
    void walk(IDefinition_visitor const *visitor) const;

    /// Clear the definition table.
    void clear();

    /// Returns the amount of used memory by this definition table.
    size_t get_memory_size() const;

    /// Remove an empty scope from the scope tree.
    ///
    /// \param scope  An empty scope (i.e. scope->is_empty() == true).
    void remove_empty_scope(Scope *scope);

    /// Serialize this definition table.
    ///
    /// \param serializer  the module serializer
    void serialize(Module_serializer &serializer) const;

    /// Deserialize this definition table.
    ///
    /// \param deserializer  the module deserializer
    void deserialize(Module_deserializer &deserializer);

    /// Serialize a definition.
    ///
    /// \param def         the definition
    /// \param serializer  the module serializer
    void serialize_def(Definition const *def, Module_serializer &serializer) const;

    /// Deserialize a definition.
    ///
    /// \param deserializer  the module deserializer
    Definition *deserialize_def(Module_deserializer &deserializer);

    /// Associate a scope and a type in the deserialization.
    ///
    /// \param deserializer  the module deserializer
    /// \param scope         the scope
    /// \param type          the type associated with this scope
    void associate_scope_type(
        Module_deserializer &deserializer,
        Scope               *scope,
        IType const         *type);

    /// Iterate over ALL visible definitions.
    ///
    /// \param[inout] index  current index, start with 0
    ///
    /// \return the current visible definition (and increases index) or NULL
    ///         if index is above the visible count
    Definition const *get_visible_definition(size_t &index) const;

    /// Debug helper: Prints the definition table to the given printer.
    ///
    /// \param printer  the printer
    /// \param name     the module name
    void dump(Printer *printer, char const *name) const;

private:
    /// Create a new definition table.
    ///
    /// \param owner  the owner module of this definition table
    explicit Definition_table(Module &owner);

    // non copyable
    Definition_table(Definition_table const &) MDL_DELETED_FUNCTION;
    Definition_table &operator=(Definition_table const &) MDL_DELETED_FUNCTION;

private:
    /// Create a new scope.
    ///
    /// \param parent     the parent scope or NULL
    /// \param id         an unique id for identifying this scope
    /// \param owner_def  the owner definition of this scope or NULL
    /// \param type       the type creating the scope or NULL
    /// \param name       the type name of the scope or NULL
    Scope *create_scope(
        Scope            *parent,
        size_t           id,
        Definition const *owner_def = NULL,
        IType const      *type = NULL,
        ISymbol const    *name = NULL);

    /// Creates a new named scope.
    ///
    /// \param parent   the parent scope or NULL
    /// \param name     the name of the scope to create
    /// \param id       an unique id for identifying this scope
    Scope *create_scope(
        Scope         *parent,
        ISymbol const *name,
        size_t        id);

    /// Register all definition in a predefined scope.
    ///
    /// \param serializer  the module serializer
    /// \param scope         a predefined scope
    void register_predefined_entities(
        Module_serializer &serializer,
        Scope const       *scope) const;

    /// Register all definition in a predefined scope.
    ///
    /// \param deserializer  the module deserializer
    /// \param scope         a predefined scope
    /// \param tag           next free tag
    ///
    /// \return next free tag
    Tag_t register_predefined_entities(
        Module_deserializer &deserializer,
        Scope const         *scope,
        Tag_t               tag);

private:
    /// The owner module of this definition table
    Module &m_owner;

    /// Points to the top of the scope stack.
    Scope *m_curr_scope;

    /// The outer scope of the module. Contains all predefined entities.
    Scope *m_predef_scope;

    /// The global scope of the module. Contains all definitions of this module.
    Scope *m_global_scope;

    /// The list of free scopes.
    Scope *m_free_scopes;

    /// The next id for a definition.
    size_t m_next_definition_id;

    /// Memory arena for all sub objects.
    Memory_arena m_arena;

    /// Builder for sub objects.
    Arena_builder m_builder;

    typedef ptr_hash_map<IType const, Scope *>::Type Type_scope_map;

    /// Associate types to scopes.
    Type_scope_map m_type_scopes;

    typedef ptr_map<ISymbol const, Definition const *>::Type Namespace_aliases_map;

    /// The namespace aliases.
    Namespace_aliases_map m_namespace_aliases;

    typedef vector<Definition *>::Type Def_vector;

    /// Vector of entity Definitions in this table for lookup, index by its symbol ids.
    Def_vector m_definitions;

    /// The definitions of all operators.
    Definition *m_operator_definitions[IExpression::OK_LAST];

public:
    /// Helper class for scope transitions using RAII.
    class Scope_transition {
    public:
        /// Remember the current scope and transition to another scope.
        ///
        /// \param def_table   the definition table
        /// \param scope       the destination scope
        Scope_transition(Definition_table &def_tab, Scope *scope)
            : m_deftab(def_tab), m_curr_scope(def_tab.get_curr_scope())
        {
            def_tab.transition_to_scope(scope);
        }

        /// Return to the previous scope.
        ~Scope_transition() {
            m_deftab.transition_to_scope(m_curr_scope);
        }

    private:
        /// The definition table.
        Definition_table &m_deftab;
        /// The current scope before the transition.
        Scope            *m_curr_scope;
    };

    /// Helper class for scope entering using RAII.
    class Scope_enter {
    public:
        /// Enter a new scope.
        ///
        /// \param def_table   the definition table
        /// \param def         the (function) definition that "owns" the newly created scope
        Scope_enter(Definition_table &def_tab, Definition *def)
        : m_def_tab(def_tab)
        , m_scope(NULL)
        {
            Scope *scope = m_def_tab.enter_scope(def);
            if (def->get_kind() != IDefinition::DK_ERROR) {
                def->set_own_scope(scope);
            }
        }

        /// Reopen given scope.
        ///
        /// \param def_table   the definition table
        /// \param scope       the scope that will be reopened
        Scope_enter(Definition_table &def_tab, Scope *scope)
        : m_def_tab(def_tab)
        , m_scope(NULL)
        {
            m_def_tab.reopen_scope(scope);
        }

        /// Enter a new type scope.
        ///
        /// \param def_table   the definition table
        /// \param type        the type that "owns" the new scope
        /// \param type_def    the definition of the type
        Scope_enter(
            Definition_table &def_tab,
            IType const      *type,
            Definition       *type_def)
        : m_def_tab(def_tab)
        , m_scope(NULL)
        {
            Scope *scope = m_def_tab.enter_scope(type, type_def);
            if (type_def->get_kind() != IDefinition::DK_ERROR)
                type_def->set_own_scope(scope);
        }

        /// Enter a new (compound statement) scope.
        ///
        /// \param def_table   the definition table
        Scope_enter(Definition_table &def_tab)
        : m_def_tab(def_tab)
        , m_scope(def_tab.enter_scope(/*def=*/NULL))
        {
        }

        /// Leave current scope.
        ~Scope_enter()
        {
            m_def_tab.leave_scope();
            if (m_scope != NULL && m_scope->is_empty()) {
                // drop it to save some space
                m_def_tab.remove_empty_scope(m_scope);
            }
        }

    private:
        /// The definition table.
        Definition_table &m_def_tab;

        /// If non-NULL, remove this scope if empty.
        Scope *m_scope;
    };

    /// Helper class for scope entering using RAII.
    class Scope_enter_conditional {
    public:
        /// Reopen given scope.
        ///
        /// \param def_table   the definition table
        /// \param scope       the scope that will be reopened
        Scope_enter_conditional(Definition_table &def_tab, Scope *scope)
        : m_def_tab(def_tab)
        , m_valid_scope(scope != NULL)
        {
            if (m_valid_scope)
                m_def_tab.reopen_scope(scope);
        }

        /// Leave current scope.
        ~Scope_enter_conditional() {
            if (m_valid_scope)
                m_def_tab.leave_scope();
        }

    private:
        /// The definition table.
        Definition_table &m_def_tab;

        /// True, if the scope is valid.
        bool m_valid_scope;
    };
};

/// Decode the since version.
///
/// \param flags  version flags of a definition
unsigned mdl_since_version(unsigned flags);

/// Decode the removed version.
///
/// \param flags  version flags of a definition
unsigned mdl_removed_version(unsigned flags);

/// Check if a entity is available in the given MDL language level of a module.
///
/// \param module_version  the MDL language level of the owner module
/// \param version_flags   entity version flags
bool is_available_in_mdl(
    unsigned module_version,
    unsigned version_flags);

}  // mdl
}  // mi

#endif
