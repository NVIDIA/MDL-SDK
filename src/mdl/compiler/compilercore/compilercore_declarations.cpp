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

#include <mi/mdl/mdl_declarations.h>
#include <mi/mdl/mdl_iowned.h>
#include <mi/base/iallocator.h>

#include "compilercore_cc_conf.h"
#include "compilercore_memory_arena.h"
#include "compilercore_factories.h"
#include "compilercore_positions.h"
#include "compilercore_tools.h"

namespace mi {
namespace mdl {

/// A mixin for all base declaration methods.
template <typename Interface>
class Decl_base : public Interface
{
    typedef Interface Base;
public:

    /// Get the kind of declaration.
    typename Interface::Kind get_kind() const MDL_FINAL { return Interface::s_kind; }

    /// Test if the declaration is exported.
    bool is_exported() const MDL_FINAL { return m_is_exported; }

    /// Set the export status of the declaration.
    void set_export(bool exp) MDL_FINAL { m_is_exported = exp; }

    /// Access the position.
    Position &access_position() MDL_FINAL { return m_pos; }

    /// Access the position.
    Position const &access_position() const MDL_FINAL { return m_pos; }

protected:
    explicit Decl_base(bool is_exported)
    : Base()
    , m_is_exported(is_exported)
    , m_pos(0, 0, 0, 0)
    {
    }

private:
    // non copyable
    Decl_base(Decl_base const &) MDL_DELETED_FUNCTION;
    Decl_base &operator=(Decl_base const &) MDL_DELETED_FUNCTION;

protected:
    /// Set if this declaration is exported.
    bool m_is_exported;

    /// The position of this declaration.
    Position_impl m_pos;
};

/// An declaration base mixin for declarations with variadic number of arguments.
template <typename Interface, typename ArgIf>
class Decl_base_variadic : public Decl_base<Interface>
{
    typedef Decl_base<Interface> Base;
public:

protected:
    explicit Decl_base_variadic(Memory_arena *arena, bool is_exported)
    : Base(is_exported)
    , m_args(arena)
    {
    }

    /// Return the number of variadic arguments.
    size_t argument_count() const { return m_args.size(); }

    /// Add a new argument.
    void add_argument(ArgIf arg) { m_args.push_back(arg); }

    /// Get the argument at given position.
    ArgIf const &argument_at(size_t pos) const { return m_args.at(pos); }

    /// Get the argument at given position.
    ArgIf &argument_at(size_t pos) { return m_args.at(pos); }

    /// The arguments.
    typename Arena_vector<ArgIf>::Type m_args;
};

/// A mixin for single entity definitions.
template <typename Interface>
class Single_decl : public Decl_base<Interface>
{
    typedef Decl_base<Interface> Base;
public:
    /// Get the definition of this declaration.
    IDefinition const *get_definition() const MDL_FINAL { return m_def; }

    /// Set the definition of this declaration.
    void set_definition(IDefinition const *def) MDL_FINAL { m_def = def; }

protected:

    explicit Single_decl(bool is_exported)
    : Base(is_exported)
    , m_def(NULL)
    {
    }

    /// The Definition of this declaration (if any).
    IDefinition const *m_def;
};

/// A mixin for single entity definitions.
template <typename Interface, typename ArgIf>
class Single_decl_variadic : public Decl_base_variadic<Interface, ArgIf>
{
    typedef Decl_base_variadic<Interface, ArgIf> Base;
public:
    /// Get the definition of this declaration.
    IDefinition const *get_definition() const MDL_FINAL { return m_def; }

    /// Set the definition of this declaration.
    void set_definition(IDefinition const *def) MDL_FINAL { m_def = def; }

protected:

    explicit Single_decl_variadic(Memory_arena *arena, bool is_exported)
    : Base(arena, is_exported)
    , m_def(NULL)
    {
    }

    /// The Definition of this declaration (if any).
    IDefinition const *m_def;
};

/// A mixin for base parameter methods.
template <typename Interface>
class Parameter_base : public Interface
{
    typedef Interface Base;
public:

    /// Get the definition of this declaration.
    IDefinition const *get_definition() const MDL_FINAL { return m_def; }

    /// Set the definition of this declaration.
    void set_definition(IDefinition const *def) MDL_FINAL { m_def = def; }

    /// Access the position.
    Position &access_position() MDL_FINAL { return m_pos; }

    /// Access the position.
    Position const &access_position() const MDL_FINAL { return m_pos; }

protected:
    explicit Parameter_base()
    : Base()
    , m_def(NULL)
    , m_pos(0, 0, 0, 0)
    {
    }

private:
    // non copyable
    Parameter_base(Parameter_base const &) MDL_DELETED_FUNCTION;
    Parameter_base &operator=(Parameter_base const &) MDL_DELETED_FUNCTION;

protected:
    /// The Definition of this declaration (if any).
    IDefinition const *m_def;

    /// The position of this declaration.
    Position_impl m_pos;
};

/// An invalid declaration.
class Declaration_invalid: public Decl_base<IDeclaration_invalid>
{
    typedef Decl_base<IDeclaration_invalid> Base;
public:
    explicit Declaration_invalid(bool is_exported)
    : Base(is_exported)
    {
    }
};

/// An import declaration.
class Declaration_import : public Decl_base_variadic<IDeclaration_import, IQualified_name const *>
{
    typedef Decl_base_variadic<IDeclaration_import, IQualified_name const *> Base;
public:
    /// Get the name of the imported module.
    IQualified_name const *get_module_name() const MDL_FINAL { return m_module_name; }

    /// Set the name of the imported module.
    void set_module_name(IQualified_name const *name) MDL_FINAL { m_module_name = name; }

    /// Get the count of imported names.
    int get_name_count() const MDL_FINAL { return Base::argument_count(); }

    /// Get the imported name at index.
    IQualified_name const *get_name(int index) const MDL_FINAL {
        return Base::argument_at(index);
    }

    /// Add a name to the list of imported names.
    void add_name(IQualified_name const *name) MDL_FINAL { Base::add_argument(name); }

    explicit Declaration_import(
        Memory_arena          *arena,
        IQualified_name const *name,
        bool                  is_exported)
    : Base(arena, is_exported)
    , m_module_name(name)
    {
    }

private:
    /// The qualified name of the imported module.
    IQualified_name const *m_module_name;
};

/// A declared Parameter.
class Parameter : public Parameter_base<IParameter>
{
    typedef Parameter_base<IParameter> Base;
public:

    /// Get the type name of this parameter.
    IType_name const *get_type_name() const { return m_type_name; }

    /// Get the name of this parameter.
    ISimple_name const *get_name() const { return m_name; }

    /// Get the init expression of this parameter.
    IExpression const *get_init_expr() const { return m_init_expr; }

    /// Set the init expression of this parameter.
    void set_init_expr(IExpression const *expr) { m_init_expr = expr; }

    /// Get the annotations of this parameter or NULL.
    IAnnotation_block const *get_annotations() const { return m_annotations; }

    /// Constructor.
    ///
    /// \param type_name    the type name of this parameter
    /// \param name         the name of the parameter itself
    /// \param init_expr    the initializer expression if any
    /// \param annotations  the annotation block of this parameter if any
    explicit Parameter(
        IType_name const        *type_name,
        ISimple_name const      *name,
        IExpression const       *init_expr,
        IAnnotation_block const *annotations)
    : Base()
    , m_type_name(type_name)
    , m_name(name)
    , m_init_expr(init_expr)
    , m_annotations(annotations)
    {
    }

private:
    /// The type name of the parameter.
    IType_name const *m_type_name;

    /// The name of the parameter.
    ISimple_name const *m_name;

    /// The initializer of the parameter (if any).
    IExpression const *m_init_expr;

    /// The annotations of this parameter (if any).
    IAnnotation_block const *m_annotations;
};

/// The type of vectors of parameters.
typedef Arena_vector<IParameter const *>::Type Parameter_vector;

/// An annotation declaration.
class Declaration_annotation : public Single_decl<IDeclaration_annotation>
{
    typedef Single_decl<IDeclaration_annotation> Base;
public:
    /// Get the name of the annotation.
    ISimple_name const *get_name() const MDL_FINAL { return m_name; }

    /// Set the name of the annotation.
    void set_name(ISimple_name const *name) MDL_FINAL { m_name = name; }

    /// Get the annotation block of this annotation declaration if any.
    IAnnotation_block const *get_annotations() const MDL_FINAL { return m_annotations; }

    /// Set the annotation block of this annotation declaration.
    void set_annotations(IAnnotation_block const *annos) MDL_FINAL { m_annotations = annos; }

    /// Get the number of parameters.
    int get_parameter_count() const MDL_FINAL { return int(m_parameters.size()); }

    /// Get the parameter at index.
    IParameter const *get_parameter(int index) const MDL_FINAL {
        return m_parameters.at(index);
    }

    /// Add a parameter.
    void add_parameter(IParameter const *parameter) MDL_FINAL
    {
        m_parameters.push_back(parameter);
    }

    explicit Declaration_annotation(
        Memory_arena            *arena,
        ISimple_name const      *name,
        IAnnotation_block const *annotations,
        bool                    is_exported)
    : Base(is_exported)
    , m_name(name)
    , m_annotations(annotations)
    , m_parameters(arena)
    {
    }

private:
    /// The name of this annotation.
    ISimple_name const *m_name;

    /// The annotations of this entity (if any).
    IAnnotation_block const *m_annotations;

    /// The vector of parameters.
    Parameter_vector m_parameters;
};

/// A declared entity (variable or constant).
class Entity {
public:
    Entity(
        ISimple_name const      *name,
        IExpression const       *init_expr,
        IAnnotation_block const *annotations
    )
    : m_name(name)
    , m_init_expr(init_expr)
    , m_annotations(annotations)
    {
    }

    /// Return the name of this entity.
    ISimple_name const *get_name() const { return m_name; }

    /// Return the init expression of this entity.
    IExpression const *get_init_expr() const { return m_init_expr; }

    /// Set the init expression of this entity.
    void set_init_expr(IExpression const *init) { m_init_expr = init; }

    /// Return the annotations of this entity.
    IAnnotation_block const *get_annotations() const { return m_annotations; }

private:
    /// The name of the entity.
    ISimple_name const *m_name;

    /// The initializer of this entity.
    IExpression const *m_init_expr;

    /// The annotations of this entity (if any).
    IAnnotation_block const *m_annotations;
};

/// A constant declaration.
class Declaration_constant : public Decl_base_variadic<IDeclaration_constant, Entity>
{
    typedef Decl_base_variadic<IDeclaration_constant, Entity> Base;
public:

    /// Get the type name of the constant.
    IType_name const *get_type_name() const MDL_FINAL { return m_type_name; }

    /// Set the type name of the constant.
    void set_type_name(IType_name const *type) MDL_FINAL { m_type_name = type; }

    /// Get the number of constants defined.
    int get_constant_count() const MDL_FINAL { return Base::argument_count(); }

    /// Get the name of the constant at index.
    ISimple_name const *get_constant_name(int index) const MDL_FINAL {
        Entity const &constant = Base::argument_at(index);
        return constant.get_name();
    }

    /// Get the expression of the constant at index.
    IExpression const *get_constant_exp(int index) const MDL_FINAL {
        Entity const &constant = Base::argument_at(index);
        return constant.get_init_expr();
    }

    /// Get the annotations of the constant at index.
    IAnnotation_block const *get_annotations(int index) const MDL_FINAL {
        Entity const &constant = Base::argument_at(index);
        return constant.get_annotations();
    }

    /// Add a constant.
    void add_constant(
        ISimple_name const      *name,
        IExpression const       *expr,
        IAnnotation_block const *annotations = 0) MDL_FINAL
    {
        Base::add_argument(Entity(name, expr, annotations));
    }

    /// Set the initializer expression of the variable at index.
    void set_variable_init(int index, IExpression const *init_expr) MDL_FINAL
    {
        Entity &constant = Base::argument_at(index);
        constant.set_init_expr(init_expr);
    }

    explicit Declaration_constant(
        Memory_arena     *arena,
        IType_name const *type_name,
        bool             is_exported)
    : Base(arena, is_exported)
    , m_type_name(type_name)
    {
    }

private:
    /// The name of the type of all constants.
    IType_name const *m_type_name;
};

/// A type alias declaration.
class Declaration_type_alias : public Single_decl<IDeclaration_type_alias>
{
    typedef Single_decl<IDeclaration_type_alias> Base;
public:

    /// Get the name of the aliased type.
    IType_name const *get_type_name() const MDL_FINAL { return m_type_name; }

    /// Set name of the the aliased type.
    void set_type_name(IType_name const *type) MDL_FINAL { m_type_name = type; }

    /// Get the alias name.
    ISimple_name const *get_alias_name() const MDL_FINAL { return m_alias_name; }

    /// Set the alias name.
    void set_alias_name(ISimple_name const *name) MDL_FINAL { m_alias_name = name; }

    explicit Declaration_type_alias(
        IType_name const   *type_name,
        ISimple_name const *alias_name,
        bool               is_exported)
    : Base(is_exported)
    , m_type_name(type_name)
    , m_alias_name(alias_name)
    {
    }

private:
    /// The name of the aliased type.
    IType_name const *m_type_name;

    /// The alias name.
    ISimple_name const *m_alias_name;
};

/// A declared structure field.
class Structure_field {
public:
    explicit Structure_field(
        IType_name const        *type_name,
        ISimple_name const      *name,
        IExpression const       *init_expr,
        IAnnotation_block const *annotations)
    : m_type_name(type_name)
    , m_name(name)
    , m_init_expr(init_expr)
    , m_annotations(annotations)
    {
    }

    /// Get the type name of this field.
    IType_name const *get_type_name() const { return m_type_name; }

    /// Get the name of this field.
    ISimple_name const *get_name() const { return m_name; }

    /// Get he initialization expression of this field or NULL.
    IExpression const *get_init_expr() const { return m_init_expr; }

    /// Set the initialization expression of this field.
    void set_init_expr(IExpression const *init) { m_init_expr = init; }

    /// Get the annotations of the field.
    IAnnotation_block const *get_annotations() const { return m_annotations; }

private:

    /// The type name of this field.
    IType_name const *m_type_name;

    /// The name of this field.
    ISimple_name const *m_name;

    /// The initialization expression of this field (if any).
    IExpression const *m_init_expr;

    /// The annotations of the field.
    IAnnotation_block const *m_annotations;
};

/// A struct type declaration.
class Declaration_type_struct :
    public Single_decl_variadic<IDeclaration_type_struct, Structure_field>
{
    typedef Single_decl_variadic<IDeclaration_type_struct, Structure_field> Base;
public:

    /// Get the name of the struct type.
    ISimple_name const *get_name() const MDL_FINAL { return m_name; }

    /// Set the name of the struct type.
    void set_name(ISimple_name const *name) MDL_FINAL { m_name = name; }

    /// Get the annotations of the struct if any.
    IAnnotation_block const *get_annotations() const MDL_FINAL
    {
        return m_annotations;
    }

    /// Get the number of fields.
    int get_field_count() const MDL_FINAL { return Base::argument_count(); }

    /// Get the type name of the field at index.
    IType_name const *get_field_type_name(int index) const MDL_FINAL {
        Structure_field const &field = Base::argument_at(index);
        return field.get_type_name();
    }

    /// Get the name of the field at index.
    ISimple_name const *get_field_name(int index) const MDL_FINAL {
        Structure_field const &field = Base::argument_at(index);
        return field.get_name();
    }

    /// Get the initializer of the field at index.
    IExpression const *get_field_init(int index) const MDL_FINAL {
        Structure_field const &field = Base::argument_at(index);
        return field.get_init_expr();
    }

    /// Get the annotations of the field at index.
    IAnnotation_block const *get_annotations(int index) const MDL_FINAL {
        Structure_field const &field = Base::argument_at(index);
        return field.get_annotations();
    }

    /// Add a field.
    void add_field(
        IType_name const        *type_name,
        ISimple_name const      *field_name,
        IExpression const       *init = NULL,
        IAnnotation_block const *annotations = NULL) MDL_FINAL
    {
        Base::add_argument(Structure_field(type_name, field_name, init, annotations));
    }


    /// Set the initializer of the field at index.
    void set_field_init(int index, IExpression const *init) MDL_FINAL
    {
        Structure_field &field = Base::argument_at(index);
        field.set_init_expr(init);
    }

    explicit Declaration_type_struct(
        Memory_arena            *arena,
        ISimple_name const      *name,
        IAnnotation_block const *annotations,
        bool                    is_exported)
    : Base(arena, is_exported)
    , m_name(name)
    , m_annotations(annotations)
    {
    }

private:
    /// The name of this struct type.
    ISimple_name const *m_name;

    /// The annotations of the struct.
    IAnnotation_block const *m_annotations;
};

/// An enum value.
class Enum_value {
public:
    explicit Enum_value(
        ISimple_name const      *name,
        IExpression const       *init_expr,
        IAnnotation_block const *annotations)
    : m_name(name)
    , m_init_expr(init_expr)
    , m_annotations(annotations)
    {
    }

    /// Get the name of this enum value.
    ISimple_name const *get_name() const { return m_name; };

    /// Get the initializer expression of this value or NULL.
    IExpression const *get_init_expr() const { return m_init_expr; }

    /// Set the initializer expression of this value.
    void set_init_expr(IExpression const *init) { m_init_expr = init; }

    /// Get the annotations of this value or NULL.
    IAnnotation_block const *get_annotations() const { return m_annotations; }

private:
    /// The name of this enum value.
    ISimple_name const * m_name;

    /// The initializer expression of this value if any.
    IExpression const *m_init_expr;

    /// The annotations of this value if any.
    IAnnotation_block const *m_annotations;
};

/// An enum type declaration.
class Declaration_type_enum : public Single_decl_variadic<IDeclaration_type_enum, Enum_value>
{
    typedef Single_decl_variadic<IDeclaration_type_enum, Enum_value> Base;
public:

    /// Get the name of the enum type.
    ISimple_name const *get_name() const MDL_FINAL { return m_name; };

    /// Set the name of the enum type.
    void set_name(ISimple_name const *name) MDL_FINAL { m_name = name; };

    /// Get the annotations of the enum if any.
    IAnnotation_block const *get_annotations() const MDL_FINAL
    {
        return m_annotations;
    }

    /// Get the number of enum values.
    int get_value_count() const MDL_FINAL { return Base::argument_count(); };

    /// Get the name of the value at index.
    ISimple_name const *get_value_name(int index) const MDL_FINAL {
        Enum_value const &value = Base::argument_at(index);
        return value.get_name();
    }

    /// Get the initializer of the value at index.
    IExpression const *get_value_init(int index) const {
        Enum_value const &value = Base::argument_at(index);
        return value.get_init_expr();
    }

    /// Set the initializer of the value at index.
    void set_value_init(int index, IExpression const *init) MDL_FINAL {
        Enum_value &value = Base::argument_at(index);
        value.set_init_expr(init);
    }

    /// Get the annotations of the value at index.
    IAnnotation_block const *get_annotations(int index) const MDL_FINAL {
        Enum_value const &value = Base::argument_at(index);
        return value.get_annotations();
    }

    /// Add a value.
    void add_value(
        ISimple_name const      *name,
        IExpression const       *init = NULL,
        IAnnotation_block const *annotations = NULL) MDL_FINAL
    {
        Base::add_argument(Enum_value(name, init, annotations));
    }

    /// Check is this enum declaration is an enum class.
    bool is_enum_class() const MDL_FINAL
    {
        return m_is_enum_class;
    }

    explicit Declaration_type_enum(
        Memory_arena            *arena,
        ISimple_name const      *name,
        IAnnotation_block const *annotations,
        bool                    is_exported,
        bool                    is_enum_class)
    : Base(arena, is_exported)
    , m_name(name)
    , m_annotations(annotations)
    , m_is_enum_class(is_enum_class)
    {
    }

private:
    /// The name of this enum.
    ISimple_name const *m_name;

    /// The annotations of the struct.
    IAnnotation_block const *m_annotations;

    /// True, if this is a class enum.
    bool m_is_enum_class;
};

/// A variable declaration.
class Declaration_variable : public Decl_base_variadic<IDeclaration_variable, Entity>
{
    typedef Decl_base_variadic<IDeclaration_variable, Entity> Base;
public:

    /// Get the type name of the variable.
    IType_name const *get_type_name() const MDL_FINAL { return m_type_name; }

    /// Set the type name of the variable.
    void set_type_name(IType_name const *type) MDL_FINAL { m_type_name = type; }

    /// Get the number of variables defined.
    int get_variable_count() const MDL_FINAL { return Base::argument_count(); }

    /// Get the name of the variable at index.
    ISimple_name const *get_variable_name(int index) const MDL_FINAL {
        Entity const &variable = Base::argument_at(index);
        return variable.get_name();
    }

    /// Get the initializer expression of the variable at index.
    IExpression const *get_variable_init(int index) const MDL_FINAL {
        Entity const &variable = Base::argument_at(index);
        return variable.get_init_expr();
    }

    /// Get the annotations of the variable at index.
    IAnnotation_block const *get_annotations(int index) const MDL_FINAL {
        Entity const &variable = Base::argument_at(index);
        return variable.get_annotations();
    }

    /// Add a variable.
    void add_variable(
        ISimple_name const      *name,
        IExpression const       *init = NULL,
        IAnnotation_block const *annotations = NULL) MDL_FINAL
    {
        Base::add_argument(Entity(name, init, annotations));
    }

    /// Set the initializer expression of the variable at index.
    void set_variable_init(int index, IExpression const *init_expr) MDL_FINAL
    {
        Entity &variable = Base::argument_at(index);
        variable.set_init_expr(init_expr);
    }

    explicit Declaration_variable(
        Memory_arena     *arena,
        IType_name const *type_name,
        bool             is_exported)
    : Base(arena, is_exported)
    , m_type_name(type_name)
    {
    }

private:
    /// The name of the type of all variables.
    IType_name const *m_type_name;
};

/// A function declaration.
class Declaration_function : public Single_decl<IDeclaration_function>
{
    typedef Single_decl<IDeclaration_function> Base;
public:

    /// Get the name of the return type.
    IType_name const *get_return_type_name() const MDL_FINAL { return m_return_type_name; }

    /// Set the name of the return type.
    void set_return_type_name(IType_name const *type_name) MDL_FINAL {
        m_return_type_name = type_name;
    }

    /// Get the name of the function.
    ISimple_name const *get_name() const MDL_FINAL { return m_name; }

    /// Set the name of the function.
    void set_name(ISimple_name const *name) MDL_FINAL { m_name = name; }

    /// Check if this is a clone.
    bool is_preset() const MDL_FINAL { return m_is_preset; }

    /// Set the preset flag.
    void set_preset(bool preset) MDL_FINAL { m_is_preset = preset; }

    /// Get the number of parameters.
    int get_parameter_count() const MDL_FINAL { return m_parameters.size(); }

    /// Get the parameter at index.
    IParameter const *get_parameter(int index) const MDL_FINAL {
        return m_parameters.at(index);
    }

    /// Add a parameter.
    void add_parameter(IParameter const *parameter) MDL_FINAL {
        m_parameters.push_back(parameter);
    }

    /// Get the qualifier.
    Qualifier get_qualifier() const MDL_FINAL {
        return m_qualifier;
    }

    /// Set the qualifier.
    void set_qualifier(Qualifier qualifier) MDL_FINAL {
        m_qualifier = qualifier;
    }

    /// Get the function body.
    IStatement const *get_body() const MDL_FINAL { return m_body; }

    /// Set the function body.
    void set_body(IStatement const *body) MDL_FINAL { m_body = body; }

    /// Get the annotations of the function.
    IAnnotation_block const *get_annotations() const MDL_FINAL { return m_annotations; }

    /// Set the annotations of the function.
    void set_annotation(IAnnotation_block const *annotations) MDL_FINAL {
        m_annotations = annotations;
    }

    /// Get the annotations of the functions return type if any.
    IAnnotation_block const *get_return_annotations() const MDL_FINAL {
        return m_return_annotations;
    }

    /// Set the annotations of the functions return type.
    void set_return_annotation(IAnnotation_block const *annotations) MDL_FINAL {
        m_return_annotations = annotations;
    }

    explicit Declaration_function(
        Memory_arena *arena,
        IType_name const        *type_name,
        IAnnotation_block const *ret_annotations,
        ISimple_name const      *name,
        bool                    preset,
        IStatement const        *body,
        IAnnotation_block const *annotations,
        bool is_exported)
    : Base(is_exported)
    , m_return_type_name(type_name)
    , m_name(name)
    , m_is_preset(preset)
    , m_parameters(arena)
    , m_qualifier(FQ_NONE)
    , m_body(body)
    , m_annotations(annotations)
    , m_return_annotations(ret_annotations)
    {
    }

private:
    /// The name of the return type.
    IType_name const *m_return_type_name;

    /// The name of this function.
    ISimple_name const *m_name;

    /// The preset flag.
    bool m_is_preset;

    /// The vector of parameters.
    Parameter_vector m_parameters;

    /// The qualifier.
    Qualifier m_qualifier;

    /// The body of this function.
    IStatement const *m_body;

    /// The annotations of this function (if any).
    IAnnotation_block const *m_annotations;

    /// The annotations of this functions return type (if any).
    IAnnotation_block const *m_return_annotations;
};

/// A module declaration.
class Declaration_module : public IDeclaration_module
{
public:
    typedef IDeclaration_module Base;
public:

    /// Get the kind of declaration.
    Kind get_kind() const MDL_FINAL { return IDeclaration_module::s_kind; }

    /// Test if the declaration is exported.
    bool is_exported() const MDL_FINAL { return false; }

    /// Set the export status of the declaration.
    void set_export(bool exp) MDL_FINAL {
        MDL_ASSERT(!exp && "module declarations cannot be exported");
    }

    /// Access the position.
    Position &access_position() MDL_FINAL { return m_pos; }

    /// Access the position.
    Position const &access_position() const MDL_FINAL { return m_pos; }

    /// Get the annotations of the module if any.
    IAnnotation_block const *get_annotations() const MDL_FINAL { return m_annotations; }

    /// Set the annotations of the module.
    void set_annotation(IAnnotation_block const *annotations) MDL_FINAL {
        m_annotations = annotations;
    }

    explicit Declaration_module(IAnnotation_block const *annos)
    : Base()
    , m_pos(0, 0, 0, 0)
    , m_annotations(annos)
    {
    }

private:
    // non copyable
    Declaration_module(Declaration_module const &) MDL_DELETED_FUNCTION;
    Declaration_module &operator=(Declaration_module const &) MDL_DELETED_FUNCTION;

protected:
    /// The position of this declaration.
    Position_impl m_pos;

    /// The annotations of this function (if any).
    IAnnotation_block const *m_annotations;
};

/// A namespace alias declaration.
class Declaration_namespace_alias : public IDeclaration_namespace_alias
{
public:
    typedef IDeclaration_namespace_alias Base;
public:

    /// Get the kind of declaration.
    Kind get_kind() const MDL_FINAL { return Base::s_kind; }

    /// Test if the declaration is exported.
    bool is_exported() const MDL_FINAL { return false; }

    /// Set the export status of the declaration.
    void set_export(bool exp) MDL_FINAL {
        MDL_ASSERT(!exp && "namespace alias cannot be exported");
    }

    /// Access the position.
    Position &access_position() MDL_FINAL { return m_pos; }

    /// Access the position.
    Position const &access_position() const MDL_FINAL { return m_pos; }

    /// Get the alias name of the namespace.
    ISimple_name const *get_alias() const MDL_FINAL { return m_alias; }

    /// Get the namespace.
    IQualified_name const *get_namespace() const MDL_FINAL { return m_namespace; }

    /// Set the alias name of the namespace.
    ///
    /// \param name  the new namespace alias name
    void set_name(ISimple_name const *name) MDL_FINAL { m_alias = name; }

    /// Set the namespace.
    ///
    /// \param ns  the new namespace
    void set_namespace(IQualified_name const *ns) MDL_FINAL { m_namespace = ns; }

    /// Constructor.
    explicit Declaration_namespace_alias(
        ISimple_name const    *alias,
        IQualified_name const *ns)
    : Base()
    , m_pos(0, 0, 0, 0)
    , m_alias(alias)
    , m_namespace(ns)
    {
    }

private:
    // non copyable
    Declaration_namespace_alias(Declaration_namespace_alias const &) MDL_DELETED_FUNCTION;
    Declaration_namespace_alias &operator=(Declaration_namespace_alias const &) MDL_DELETED_FUNCTION;

protected:
    /// The position of this declaration.
    Position_impl m_pos;

    /// The alias name of the namespace.
    ISimple_name const *m_alias;

    /// The namespace itself.
    IQualified_name const *m_namespace;
};

// -------------------------------------- declaration factory --------------------------------------

Declaration_factory::Declaration_factory(Memory_arena &arena)
: Base()
, m_builder(arena)
{
}

/// Set position on a declaration.
static void set_position(
    IDeclaration *decl,
    int          start_line,
    int          start_column,
    int          end_line,
    int          end_column)
{
    Position &pos = decl->access_position();
    pos.set_start_line(start_line);
    pos.set_start_column(start_column);
    pos.set_end_line(end_line);
    pos.set_end_column(end_column);
}

/// Set position on a parameter.
static void set_position(
    IParameter *param,
    int        start_line,
    int        start_column,
    int        end_line,
    int        end_column)
{
    Position &pos = param->access_position();
    pos.set_start_line(start_line);
    pos.set_start_column(start_column);
    pos.set_end_line(end_line);
    pos.set_end_column(end_column);
}

/// Create a new invalid declaration.
IDeclaration_invalid *Declaration_factory::create_invalid(
    bool exported,
    int  start_line,
    int  start_column,
    int  end_line,
    int  end_column)
{
    IDeclaration_invalid *result = m_builder.create<Declaration_invalid>(exported);
    set_position(result,start_line, start_column, end_line, end_column);
    return result;
}

/// Create a new import declaration.
IDeclaration_import *Declaration_factory::create_import(
    IQualified_name const *module_name,
    bool                  exported,
    int                   start_line,
    int                   start_column,
    int                   end_line,
    int                   end_column)
{
    IDeclaration_import *result =
        m_builder.create<Declaration_import>(m_builder.get_arena(), module_name, exported);
    set_position(result,start_line, start_column, end_line, end_column);
    return result;
}

/// Create a new parameter.
IParameter const *Declaration_factory::create_parameter(
    IType_name const        *type_name,
    ISimple_name const      *name,
    IExpression const       *init,
    IAnnotation_block const *annotations,
    int                     start_line,
    int                     start_column,
    int                     end_line,
    int                     end_column)
{
    IParameter *result = m_builder.create<Parameter>(type_name, name, init, annotations);
    set_position(result,start_line, start_column, end_line, end_column);
    return result;
}

/// Create a new annotation declaration.
IDeclaration_annotation *Declaration_factory::create_annotation(
    ISimple_name const      *name,
    IAnnotation_block const *annotations,
    bool                    exported,
    int                     start_line,
    int                     start_column,
    int                     end_line,
    int                     end_column)
{
    IDeclaration_annotation *result = m_builder.create<Declaration_annotation>(
            m_builder.get_arena(), name, annotations, exported);
    set_position(result,start_line, start_column, end_line, end_column);
    return result;
}

/// Create a new constant declaration.
IDeclaration_constant *Declaration_factory::create_constant(
    IType_name const *type_name,
    bool             exported,
    int              start_line,
    int              start_column,
    int              end_line,
    int              end_column)
{
    IDeclaration_constant *result =
        m_builder.create<Declaration_constant>(m_builder.get_arena(), type_name, exported);
    set_position(result,start_line, start_column, end_line, end_column);
    return result;
}

/// Create a new type alias declaration.
IDeclaration_type_alias *Declaration_factory::create_alias(
    IType_name const   *type_name,
    ISimple_name const *alias_name,
    bool               exported,
    int                start_line,
    int                start_column,
    int                end_line,
    int                end_column)
{
    IDeclaration_type_alias *result =
        m_builder.create<Declaration_type_alias>(type_name, alias_name, exported);
    set_position(result,start_line, start_column, end_line, end_column);
    return result;
}

/// Create a new struct type declaration.
IDeclaration_type_struct *Declaration_factory::create_struct(
    ISimple_name const      *struct_name,
    IAnnotation_block const *annotations,
    bool                    exported,
    int                     start_line,
    int                     start_column,
    int                     end_line,
    int                     end_column)
{
    IDeclaration_type_struct *result =
        m_builder.create<Declaration_type_struct>(
            m_builder.get_arena(),
            struct_name,
            annotations,
            exported);
    set_position(result,start_line, start_column, end_line, end_column);
    return result;
}

/// Create a new enum type declaration.
IDeclaration_type_enum *Declaration_factory::create_enum(
    ISimple_name const      *enum_name,
    IAnnotation_block const *annotations,
    bool                    exported,
    bool                    is_enum_class,
    int                     start_line,
    int                     start_column,
    int                     end_line,
    int                     end_column)
{
    IDeclaration_type_enum *result = m_builder.create<Declaration_type_enum>(
        m_builder.get_arena(),
        enum_name,
        annotations,
        exported,
        is_enum_class);
    set_position(result,start_line, start_column, end_line, end_column);
    return result;
}

/// Create a new variable declaration.
IDeclaration_variable *Declaration_factory::create_variable(
    IType_name const *type_name,
    bool             exported,
    int              start_line,
    int              start_column,
    int              end_line,
    int              end_column)
{
    IDeclaration_variable *result =
        m_builder.create<Declaration_variable>(m_builder.get_arena(), type_name, exported);
    set_position(result,start_line, start_column, end_line, end_column);
    return result;
}

/// Create a new function declaration.
IDeclaration_function *Declaration_factory::create_function(
    IType_name const        *return_type_name,
    IAnnotation_block const *ret_annotations,
    ISimple_name const      *function_name,
    bool                    is_preset,
    IStatement const        *body,
    IAnnotation_block const *annotations,
    bool                    is_exported,
    int                     start_line,
    int                     start_column,
    int                     end_line,
    int                     end_column)
{
    IDeclaration_function *result =
        m_builder.create<Declaration_function>(
            m_builder.get_arena(), return_type_name, ret_annotations,
            function_name, is_preset, body, annotations, is_exported);
    set_position(result,start_line, start_column, end_line, end_column);
    return result;
}

// Create a new module declaration.
IDeclaration_module *Declaration_factory::create_module(
    IAnnotation_block const *annotations,
    int                     start_line,
    int                     start_column,
    int                     end_line,
    int                     end_column)
{
    IDeclaration_module *result = m_builder.create<Declaration_module>(annotations);
    set_position(result, start_line, start_column, end_line, end_column);
    return result;
}

// Create a new namespace alias.
IDeclaration_namespace_alias *Declaration_factory::create_namespace_alias(
    ISimple_name const    *alias,
    IQualified_name const *ns,
    int                   start_line,
    int                   start_column,
    int                   end_line,
    int                   end_column)
{
    IDeclaration_namespace_alias *result = m_builder.create<Declaration_namespace_alias>(alias, ns);
    set_position(result, start_line, start_column, end_line, end_column);
    return result;
}

// Skip all presets returning the original definition.
IDeclaration_function const *skip_presets(
    IDeclaration_function const     *func_decl,
    mi::base::Handle<IModule const> &owner_mod)
{
    if (func_decl->is_preset()) {
        // A preset, retrieve its original definition.
        // Beware, might be a preset of a preset, so find out the original (non-preset)
        // declaration by iteration.
        do {
            IStatement_expression const *preset_body
                = cast<IStatement_expression>(func_decl->get_body());
            IExpression const           *expr = preset_body->get_expression();

            // skip let expressions
            while (IExpression_let const *let = as<IExpression_let>(expr)) {
                expr = let->get_expression();
            }
            IExpression_call const      *inst = cast<IExpression_call>(expr);
            IExpression_reference const *ref  =
                cast<IExpression_reference>(inst->get_reference());

            IDefinition const *orig_def = ref->get_definition();

            mi::base::Handle<IModule const> next(owner_mod->get_owner_module(orig_def));

            orig_def = owner_mod->get_original_definition(orig_def);
            owner_mod = next;

            // get the prototype of the preset if any, else its definition
            func_decl = cast<IDeclaration_function>(orig_def->get_declaration());
        } while (func_decl->is_preset());
    }
    return func_decl;
}

// Skip all presets returning the original function definition.
IDefinition const *skip_presets(
    IDefinition const               *func_def,
    mi::base::Handle<IModule const> &owner_mod)
{
    MDL_ASSERT(
        !func_def->get_property(IDefinition::DP_IS_IMPORTED) &&
        "skip_presets() called on imported entity");
    if (func_def->get_kind() != IDefinition::DK_FUNCTION)
        return func_def;

    IDeclaration const *decl = func_def->get_declaration();
    if (decl == NULL)
        return func_def;

    IDeclaration_function const *f_decl = cast<IDeclaration_function>(decl);
    f_decl = skip_presets(f_decl, owner_mod);

    return f_decl->get_definition();
}

}  // mdl
}  // mi
