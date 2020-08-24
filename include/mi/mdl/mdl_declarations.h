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
/// \file mi/mdl/mdl_declarations.h
/// \brief Interfaces for declarations inside the MDL AST
#ifndef MDL_DECLARATIONS_H
#define MDL_DECLARATIONS_H 1

#include <mi/mdl/mdl_iowned.h>
#include <mi/mdl/mdl_types.h>
#include <mi/mdl/mdl_values.h>
#include <mi/mdl/mdl_definitions.h>
#include <mi/mdl/mdl_positions.h>
#include <mi/mdl/mdl_names.h>
#include <mi/mdl/mdl_expressions.h>
#include <mi/mdl/mdl_statements.h>
#include <mi/mdl/mdl_annotations.h>

namespace mi {
namespace mdl {

/// The base interface of a declaration inside the MDL AST.
class IDeclaration : public Interface_owned
{
public:
    /// The possible kinds of declarations.
    enum Kind {
        DK_INVALID,         ///< An invalid declaration.
        DK_IMPORT,          ///< An import declaration.
        DK_ANNOTATION,      ///< An annotation declaration.
        DK_CONSTANT,        ///< A constant declaration.
        DK_TYPE_ALIAS,      ///< A type alias declaration (typedef).
        DK_TYPE_STRUCT,     ///< A struct type declaration.
        DK_TYPE_ENUM,       ///< An enum type declaration.
        DK_VARIABLE,        ///< A variable declaration.
        DK_FUNCTION,        ///< A function declaration.
        DK_MODULE,          ///< A module declaration.
        DK_NAMESPACE_ALIAS  ///< A namespace alias declaration.
    };

    /// Get the kind of declaration.
    virtual Kind get_kind() const = 0;

    /// Test if the declaration is exported.
    virtual bool is_exported() const = 0;

    /// Set the export status of the declaration.
    virtual void set_export(bool exp) = 0;

    /// Access the position.
    virtual Position &access_position() = 0;

    /// Access the position.
    virtual Position const &access_position() const = 0;
};

/// An invalid declaration, created by the parser in case of a syntax error.
class IDeclaration_invalid : public IDeclaration
{
public:
    /// The kind of this subclass.
    static Kind const s_kind = DK_INVALID;
};

/// An import declaration.
class IDeclaration_import : public IDeclaration
{
public:
    /// The kind of this subclass.
    static Kind const s_kind = DK_IMPORT;

    /// Get the name of the imported module.
    ///
    /// If this returns NULL, the declaration is of the form "import ...",
    /// otherwise it is of the form "using <module> import ..."
    virtual IQualified_name const *get_module_name() const = 0;

    /// Set the name of the imported module.
    ///
    /// \param name  the qualified name of the module from which entities are imported
    virtual void set_module_name(IQualified_name const *name) = 0;

    /// Get the count of imported names.
    virtual int get_name_count() const = 0;

    /// Get the imported name at index.
    ///
    /// \param index  the index
    virtual IQualified_name const *get_name(int index) const = 0;

    /// Add a name to the list of imported names.
    ///
    /// \param name  the qualified name of the entity (or module) to import
    virtual void add_name(IQualified_name const *name) = 0;
};

/// The interface of a function, annotation, or material parameter inside the MDL AST.
class IParameter : public Interface_owned
{
public:
    /// Get the type name.
    virtual IType_name const *get_type_name() const = 0;

    /// Get the parameter name.
    virtual ISimple_name const *get_name() const = 0;

    /// Get the initializing expression if any.
    virtual IExpression const *get_init_expr() const = 0;

    /// Set the initializing expression.
    ///
    /// \param expr  the new initializing expression, might be NULL
    virtual void set_init_expr(IExpression const *expr) = 0;

    /// Get the annotation block of this parameter if any.
    virtual IAnnotation_block const *get_annotations() const = 0;

    /// Get the definition of this declaration.
    virtual IDefinition const *get_definition() const = 0;

    /// Set the definition of this declaration.
    virtual void set_definition(IDefinition const *def) = 0;

    /// Access the position.
    virtual Position &access_position() = 0;

    /// Access the position.
    virtual Position const &access_position() const = 0;
};

/// An annotation declaration inside the MDL AST.
class IDeclaration_annotation : public IDeclaration
{
public:
    /// The kind of this subclass.
    static Kind const s_kind = DK_ANNOTATION;

    /// Get the name of the annotation.
    virtual ISimple_name const *get_name() const = 0;

    /// Set the name of the annotation.
    ///
    /// \param name  the name of this annotation
    virtual void set_name(ISimple_name const *name) = 0;

    /// Get the annotation block of this annotation declaration if any.
    virtual IAnnotation_block const *get_annotations() const = 0;

    /// Set the annotation block of this annotation declaration.
    virtual void set_annotations(IAnnotation_block const *annos) = 0;

    /// Get the definition of this declaration if already set.
    virtual IDefinition const *get_definition() const = 0;

    /// Set the definition of this declaration.
    ///
    /// \param def  the definition of this annotation
    virtual void set_definition(IDefinition const *def) = 0;

    /// Get the number of parameters.
    virtual int get_parameter_count() const = 0;

    /// Get the parameter at index.
    ///
    /// \param index  the index of the requested parameter
    virtual IParameter const *get_parameter(int index) const = 0;

    /// Add a parameter (at the end of the parameter list).
    ///
    /// \param parameter  the parameter to add
    virtual void add_parameter(IParameter const *parameter) = 0;
};

/// A constant declaration in the MDL AST.
class IDeclaration_constant : public IDeclaration
{
public:
    /// The kind of this subclass.
    static Kind const s_kind = DK_CONSTANT;

    /// Get the type name of the constant.
    virtual IType_name const *get_type_name() const = 0;

    /// Set the type name of the constant.
    ///
    /// \param type  the new typename of the constant
    virtual void set_type_name(IType_name const *type) = 0;

    /// Get the number of constants defined.
    virtual int get_constant_count() const = 0;

    /// Get the name of the constant at index.
    ///
    /// \param index  the index of the requested constant
    virtual ISimple_name const *get_constant_name(int index) const = 0;

    /// Get the expression of the constant at index.
    ///
    /// \param index  the index of the requested constant
    virtual IExpression const *get_constant_exp(int index) const = 0;

    /// Get the annotations of the constant at index if any.
    ///
    /// \param index  the index of the requested constant
    virtual const IAnnotation_block *get_annotations(int index) const = 0;

    /// Add a constant (at the end of the list).
    ///
    /// \param name         the name of the constant
    /// \param expr         the expression of the constant
    /// \param annotations  the annotations of the constant if any
    virtual void add_constant(
        ISimple_name const      *name,
        IExpression const       *expr,
        IAnnotation_block const *annotations = NULL) = 0;

    /// Set the initializer expression of the variable at index.
    virtual void set_variable_init(int index, const IExpression *init_expr) = 0;
};

/// A type declaration.
class IDeclaration_type : public IDeclaration
{
public:
    /// Get the definition of this declaration.
    virtual IDefinition const *get_definition() const = 0;

    /// Set the definition of this declaration.
    virtual void set_definition(IDefinition const *def) = 0;
};

/// A type alias declaration inside the MDL AST.
class IDeclaration_type_alias : public IDeclaration_type
{
public:
    /// The kind of this subclass.
    static Kind const s_kind = DK_TYPE_ALIAS;

    /// Get the type name of the aliased type.
    virtual IType_name const *get_type_name() const = 0;

    /// Set the type name of the the aliased type.
    ///
    /// \param type  the new type name
    virtual void set_type_name(IType_name const *type) = 0;

    /// Get the alias name.
    virtual ISimple_name const *get_alias_name() const = 0;

    /// Set the alias name.
    ///
    /// \param name  the new alias name
    virtual void set_alias_name(ISimple_name const *name) = 0;
};

/// A struct type declaration inside the MDL AST.
class IDeclaration_type_struct : public IDeclaration_type
{
public:
    /// The kind of this subclass.
    static Kind const s_kind = DK_TYPE_STRUCT;

    /// Get the name of the struct type.
    virtual ISimple_name const *get_name() const = 0;

    /// Set the name of the struct type.
    ///
    /// \param name  the new name of this struct declaration
    virtual void set_name(ISimple_name const *name) = 0;

    /// Get the annotations of the struct declaration if any.
    virtual IAnnotation_block const *get_annotations() const = 0;

    /// Get the number of fields of this struct declaration.
    virtual int get_field_count() const = 0;

    /// Get the type name of the field at index.
    ///
    /// \param index  the index of the requested field
    virtual IType_name const *get_field_type_name(int index) const = 0;

    /// Get the name of the field at index.
    ///
    /// \param index  the index of the requested field
    virtual ISimple_name const *get_field_name(int index) const = 0;

    /// Get the initializer of the field at index if any.
    ///
    /// \param index  the index of the requested field
    virtual IExpression const *get_field_init(int index) const = 0;

    /// Get the annotations of the field if any.
    ///
    /// \param index  the index of the requested field
    virtual IAnnotation_block const *get_annotations(int index) const = 0;

    /// Add a field (at the end of this struct declaration).
    ///
    /// \param type_name    the type name of the new field
    /// \param field_name   the name of the new field
    /// \param init         the initializer expression of the new field or NULL
    /// \param annotations  the annotations of the new field or NULL
    virtual void add_field(
        IType_name const        *type_name,
        ISimple_name const      *field_name,
        IExpression const       *init = NULL,
        IAnnotation_block const *annotations = NULL) = 0;

    /// Set the initializer expression of the field at index.
    ///
    /// \param index  the index of the requested field
    /// \param init   the new initializer expression, might be NULL
    virtual void set_field_init(
        int               index,
        IExpression const *init) = 0;
};

/// An enum type declaration inside the MDL AST.
class IDeclaration_type_enum : public IDeclaration_type
{
public:
    /// The kind of this subclass.
    static Kind const s_kind = DK_TYPE_ENUM;

    /// Get the name of the enum type.
    virtual ISimple_name const *get_name() const = 0;

    /// Set the name of the enum type.
    ///
    /// \param name  the new name of the enum type
    virtual void set_name(ISimple_name const *name) = 0;

    /// Get the annotations of the enum if any.
    virtual IAnnotation_block const *get_annotations() const = 0;

    /// Get the number of enum values.
    virtual int get_value_count() const = 0;

    /// Get the name of the value at index.
    ///
    /// \param index  the index of the requested enum value
    virtual ISimple_name const *get_value_name(int index) const = 0;

    /// Get the initializer of the value at index if any.
    ///
    /// \param index  the index of the requested enum value
    virtual IExpression const *get_value_init(int index) const = 0;

    /// Set the initializer of the value at index.
    ///
    /// \param index  the index of the requested enum value
    /// \param init   the new initializer expression
    virtual void set_value_init(
        int         index,
        IExpression const *init) = 0;

    /// Get the annotations of the value at index if any.
    ///
    /// \param index  the index of the requested enum value
    virtual IAnnotation_block const *get_annotations(int index) const = 0;

    /// Add a value (at the end of the enum type declaration).
    ///
    /// \param name         the name of the new enum value
    /// \param init         the initializer expression of the new enum value or NULL
    /// \param annotations  the annotations of the new enum value or NULL
    virtual void add_value(
        ISimple_name const      *name,
        IExpression const       *init = NULL,
        IAnnotation_block const *annotations = NULL) = 0;

    /// Check is this enum declaration is an enum class.
    virtual bool is_enum_class() const = 0;
};

/// A variable declaration inside the MDL AST.
class IDeclaration_variable : public IDeclaration
{
public:
    /// The kind of this subclass.
    static Kind const s_kind = DK_VARIABLE;

    /// Get the type name of the variable declaration.
    virtual IType_name const *get_type_name() const = 0;

    /// Set the type name of the variable declaration.
    ///
    /// \param type  the new type name
    virtual void set_type_name(IType_name const *type) = 0;

    /// Get the number of variables defined.
    virtual int get_variable_count() const = 0;

    /// Get the name of the variable at index.
    ///
    /// \param index  the index of the requested variable
    virtual ISimple_name const *get_variable_name(int index) const = 0;

    /// Get the initializer expression of the variable at index if any.
    ///
    /// \param index  the index of the requested variable
    virtual IExpression const *get_variable_init(int index) const = 0;

    /// Get the annotations of the variable at index if any.
    ///
    /// \param index  the index of the requested variable
    virtual IAnnotation_block const *get_annotations(int index) const = 0;

    /// Add a variable (at the end of the variable declaration).
    ///
    /// \param name         the name of the new variable
    /// \param init         the initializer expression of the new variable or NULL
    /// \param annotations  the annotations of the new variable or NULL
    virtual void add_variable(
        ISimple_name const      *name,
        IExpression const       *init = NULL,
        IAnnotation_block const *annotations = NULL) = 0;

    /// Set the initializer expression of the variable at index.
    ///
    /// \param index      the index of the requested variable
    /// \param init_expr  the new initializer expression or NULL
    virtual void set_variable_init(
        int index,
        IExpression const *init_expr) = 0;
};

/// A function declaration inside the MDL AST.
///
/// \note that function declaration are also used to express materials
class IDeclaration_function : public IDeclaration
{
public:
    /// The kind of this subclass.
    static Kind const s_kind = DK_FUNCTION;

    /// Get the type name of the function's return type.
    virtual IType_name const *get_return_type_name() const = 0;

    /// Set the type name of the function's return type.
    ///
    /// \param type_name  the new type name
    virtual void set_return_type_name(IType_name const *type_name) = 0;

    /// Get the name of the function.
    virtual ISimple_name const *get_name() const = 0;

    /// Set the name of the function.
    ///
    /// \param name  the new function name
    virtual void set_name(ISimple_name const *name) = 0;

    /// Get the definition of this declaration if any.
    virtual IDefinition const *get_definition() const = 0;

    /// Set the definition of this declaration.
    ///
    /// \param def  the definition
    virtual void set_definition(IDefinition const *def) = 0;

    /// Check if this is a preset.
    virtual bool is_preset() const = 0;

    /// Set the preset flag.
    ///
    /// \param preset  set to true if this declaration is a preset, false else
    virtual void set_preset(bool preset) = 0;

    /// Get the number of parameters.
    virtual int get_parameter_count() const = 0;

    /// Get the parameter at index.
    ///
    /// \param index  the index of the requested parameter
    virtual IParameter const *get_parameter(int index) const = 0;

    /// Add a parameter (at the end of the parameter list).
    ///
    /// \param parameter  the parameter to add
    virtual void add_parameter(IParameter const *parameter) = 0;

    /// Get the qualifier.
    virtual Qualifier get_qualifier() const = 0;

    /// Set the qualifier.
    ///
    /// \param qualifier  the qualifier
    virtual void set_qualifier(Qualifier qualifier) = 0;

    /// Get the function body if any.
    ///
    /// \note For a error free MDL function this should be a IStatement_compound,
    ///       for a error  free material a IStatement_expression.
    virtual IStatement const *get_body() const = 0;

    /// Set the function body.
    ///
    /// \param body  the new boby of the funtion declaration
    ///
    /// \note For a error free MDL function this should be a IStatement_compound,
    ///       for a error  free material a IStatement_expression.
    virtual void set_body(IStatement const *body) = 0;

    /// Get the annotations of the function if any.
    virtual IAnnotation_block const *get_annotations() const = 0;

    /// Set the annotations of the function.
    ///
    /// \param annotations  new annotations for the function declaration
    virtual void set_annotation(IAnnotation_block const *annotations) = 0;

    /// Get the annotations of the functions return type if any.
    virtual IAnnotation_block const *get_return_annotations() const = 0;

    /// Set the annotations of the functions return type.
    ///
    /// \param annotations  new annotations for the function's return type
    virtual void set_return_annotation(IAnnotation_block const *annotations) = 0;
};

/// A module declaration inside the MDL AST.
class IDeclaration_module : public IDeclaration
{
public:
    /// The kind of this subclass.
    static Kind const s_kind = DK_MODULE;

    /// Get the annotations of the module if any.
    virtual IAnnotation_block const *get_annotations() const = 0;

    /// Set the annotations of the module.
    ///
    /// \param annotations  new module annotations
    virtual void set_annotation(IAnnotation_block const *annotations) = 0;
};

/// A namespace alias declaration inside the MDL AST.
class IDeclaration_namespace_alias : public IDeclaration
{
public:
    /// The kind of this subclass.
    static Kind const s_kind = DK_NAMESPACE_ALIAS;

    /// Get the alias name of the namespace.
    virtual ISimple_name const *get_alias() const = 0;

    /// Get the namespace.
    virtual IQualified_name const *get_namespace() const = 0;

    /// Set the alias name of the namespace.
    ///
    /// \param name  the new namespace alias name
    virtual void set_name(ISimple_name const *name) = 0;

    /// Set the namespace.
    ///
    /// \param ns  the new namespace
    virtual void set_namespace(IQualified_name const *ns) = 0;
};

/// Cast to subtype or return null if types do not match.
template<typename T>
T *as(IDeclaration *decl) {
    if(!decl)
        return 0;
    return (decl->get_kind() == T::s_kind) ? static_cast<T *>(decl) : 0;
}

/// Cast to subtype or return NULL if types do not match.
template<typename T>
T const *as(IDeclaration const *decl) {
    if (decl == NULL)
        return NULL;
    return (decl->get_kind() == T::s_kind) ? static_cast<T const *>(decl) : 0;
}

/// Cast to IDeclaration_type or return null if types do not match.
template<>
inline IDeclaration_type *as<IDeclaration_type>(IDeclaration *decl) {
    if (decl == NULL)
        return NULL;
    switch (decl->get_kind()) {
    case IDeclaration::DK_TYPE_ALIAS:
    case IDeclaration::DK_TYPE_STRUCT:
    case IDeclaration::DK_TYPE_ENUM:
        return static_cast<IDeclaration_type *>(decl);
    default:
        return NULL;
    }
}

/// Cast to IDeclaration_type or return NULL if types do not match.
template<>
inline IDeclaration_type const *as<IDeclaration_type>(IDeclaration const *decl) {
    if (decl == NULL)
        return NULL;
    return const_cast<IDeclaration_type const *>
            (as<IDeclaration_type>(const_cast<IDeclaration *>(decl)));
}

/// Check if a value is of a certain type.
template<typename T>
bool is(IDeclaration const *decl) {
    return as<T>(decl) != NULL;
}

/// The interface for creating MDL AST declarations.
/// An IDeclaration_factory interface can be obtained by calling
/// the method create_declaration_factory() on the interface IModule.
class IDeclaration_factory : public Interface_owned
{
public:
    /// Create a new invalid declaration.
    ///
    /// \param exported         Flag to indicate if this declaration is exported.
    /// \param start_line       The line on which the declaration begins.
    /// \param start_column     The column on which the declaration begins.
    /// \param end_line         The line on which the declaration ends.
    /// \param end_column       The column on which the declaration ends.
    /// \returns                The created declaration.
    virtual IDeclaration_invalid *create_invalid(
        bool exported = false,
        int  start_line = 0,
        int  start_column = 0,
        int  end_line = 0,
        int  end_column = 0) = 0;

    /// Create a new import declaration.
    ///
    /// \param module_name      The name of the module from which names are imported unqualified,
    ///                         or null if the names are imported qualified.
    /// \param exported         Flag to indicate if this declaration is exported.
    /// \param start_line       The line on which the declaration begins.
    /// \param start_column     The column on which the declaration begins.
    /// \param end_line         The line on which the declaration ends.
    /// \param end_column       The column on which the declaration ends.
    /// \returns                The created declaration.
    virtual IDeclaration_import *create_import(
        IQualified_name const *module_name = NULL,
        bool                  exported = false,
        int                   start_line = 0,
        int                   start_column = 0,
        int                   end_line = 0,
        int                   end_column = 0) = 0;

    /// Create a new parameter.
    ///
    /// \param type_name        The name of the parameters type.
    /// \param name             The name of the parameter.
    /// \param init             The default initializing expression of the parameter.
    /// \param annotations      The annotation block of the parameter.
    /// \param start_line       The line on which the declaration begins.
    /// \param start_column     The column on which the declaration begins.
    /// \param end_line         The line on which the declaration ends.
    /// \param end_column       The column on which the declaration ends.
    /// \returns                The created declaration.
    virtual IParameter const *create_parameter(
        IType_name const        *type_name,
        ISimple_name const      *name,
        IExpression const       *init,
        IAnnotation_block const *annotations,
        int                     start_line = 0,
        int                     start_column = 0,
        int                     end_line = 0,
        int                     end_column = 0) = 0;

    /// Create a new annotation declaration.
    ///
    /// \param name             The name of the annotation.
    /// \param annotations      The annotations of this declaration.
    /// \param exported         Flag to indicate if this declaration is exported.
    /// \param start_line       The line on which the declaration begins.
    /// \param start_column     The column on which the declaration begins.
    /// \param end_line         The line on which the declaration ends.
    /// \param end_column       The column on which the declaration ends.
    /// \returns                The created declaration.
    virtual IDeclaration_annotation *create_annotation(
        ISimple_name const      *name = NULL,
        IAnnotation_block const *annotations = NULL,
        bool                    exported = false,
        int                     start_line = 0,
        int                     start_column = 0,
        int                     end_line = 0,
        int                     end_column = 0) = 0;

    /// Create a new constant declaration.
    ///
    /// \param type_name        The name of the constants type.
    /// \param exported         Flag to indicate if this declaration is exported.
    /// \param start_line       The line on which the declaration begins.
    /// \param start_column     The column on which the declaration begins.
    /// \param end_line         The line on which the declaration ends.
    /// \param end_column       The column on which the declaration ends.
    /// \returns                The created declaration.
    virtual IDeclaration_constant *create_constant(
        IType_name const *type_name = NULL,
        bool             exported = false,
        int              start_line = 0,
        int              start_column = 0,
        int              end_line = 0,
        int              end_column = 0) = 0;

    /// Create a new type alias declaration.
    ///
    /// \param type_name        The name of the aliased type.
    /// \param alias_name       The name of the alias type.
    /// \param exported         Flag to indicate if this declaration is exported.
    /// \param start_line       The line on which the declaration begins.
    /// \param start_column     The column on which the declaration begins.
    /// \param end_line         The line on which the declaration ends.
    /// \param end_column       The column on which the declaration ends.
    /// \returns                The created declaration.
    virtual IDeclaration_type_alias *create_alias(
        IType_name const   *type_name = NULL,
        ISimple_name const *alias_name = NULL,
        bool               exported = false,
        int                start_line = 0,
        int                start_column = 0,
        int                end_line = 0,
        int                end_column = 0) = 0;

    /// Create a new struct type declaration.
    ///
    /// \param struct_name      The name of the struct type.
    /// \param annotations      The annotations of the struct type.
    /// \param exported         Flag to indicate if this declaration is exported.
    /// \param start_line       The line on which the declaration begins.
    /// \param start_column     The column on which the declaration begins.
    /// \param end_line         The line on which the declaration ends.
    /// \param end_column       The column on which the declaration ends.
    /// \returns                The created declaration.
    virtual IDeclaration_type_struct *create_struct(
        ISimple_name const      *struct_name = NULL,
        IAnnotation_block const *annotations = NULL,
        bool                    exported = false,
        int                     start_line = 0,
        int                     start_column = 0,
        int                     end_line = 0,
        int                     end_column = 0) = 0;

    /// Create a new enum type declaration.
    ///
    /// \param enum_name        The name of the enum type.
    /// \param annotations      The annotations of the enum type.
    /// \param exported         Flag to indicate if this declaration is exported.
    /// \param is_enum_class    True if this is a enum class, false otherwise.
    /// \param start_line       The line on which the declaration begins.
    /// \param start_column     The column on which the declaration begins.
    /// \param end_line         The line on which the declaration ends.
    /// \param end_column       The column on which the declaration ends.
    /// \returns                The created declaration.
    virtual IDeclaration_type_enum *create_enum(
        ISimple_name const      *enum_name = NULL,
        IAnnotation_block const *annotations = NULL,
        bool                    exported = false,
        bool                    is_enum_class = false,
        int                     start_line = 0,
        int                     start_column = 0,
        int                     end_line = 0,
        int                     end_column = 0) = 0;

    /// Create a new variable declaration.
    ///
    /// \param type_name        The name of the variables type.
    /// \param exported         Flag to indicate if this declaration is exported.
    /// \param start_line       The line on which the declaration begins.
    /// \param start_column     The column on which the declaration begins.
    /// \param end_line         The line on which the declaration ends.
    /// \param end_column       The column on which the declaration ends.
    /// \returns                The created declaration.
    virtual IDeclaration_variable *create_variable(
        IType_name const *type_name = NULL,
        bool             exported = false,
        int              start_line = 0,
        int              start_column = 0,
        int              end_line = 0,
        int              end_column = 0) = 0;

    /// Create a new function declaration.
    ///
    /// \param return_type_name The name of the functions return type.
    /// \param ret_annotations  The annotation block of the functions return type.
    /// \param function_name    The name of the function.
    /// \param is_clone         Flag to indicate if this function is a clone.
    /// \param body             The body of the function.
    /// \param annotations      The annotation block of the function.
    /// \param is_exported      Flag to indicate if this declaration is exported.
    /// \param start_line       The line on which the declaration begins.
    /// \param start_column     The column on which the declaration begins.
    /// \param end_line         The line on which the declaration ends.
    /// \param end_column       The column on which the declaration ends.
    /// \returns                The created declaration.
    virtual IDeclaration_function *create_function(
        IType_name const        *return_type_name = NULL,
        IAnnotation_block const *ret_annotations = NULL,
        ISimple_name const      *function_name = NULL,
        bool                    is_clone = false,
        IStatement const        *body = NULL,
        IAnnotation_block const *annotations = NULL,
        bool                    is_exported = false,
        int                     start_line = 0,
        int                     start_column = 0,
        int                     end_line = 0,
        int                     end_column = 0) = 0;

    /// Create a new module declaration.
    ///
    /// \param annotations      The annotation block of the module.
    /// \param start_line       The line on which the declaration begins.
    /// \param start_column     The column on which the declaration begins.
    /// \param end_line         The line on which the declaration ends.
    /// \param end_column       The column on which the declaration ends.
    /// \returns                The created declaration.
    virtual IDeclaration_module *create_module(
        IAnnotation_block const *annotations = NULL,
        int                     start_line = 0,
        int                     start_column = 0,
        int                     end_line = 0,
        int                     end_column = 0) = 0;

    /// Create a new namespace alias.
    ///
    /// \param alias            The alias name of the namespace.
    /// \param ns               The namespace.
    /// \param start_line       The line on which the declaration begins.
    /// \param start_column     The column on which the declaration begins.
    /// \param end_line         The line on which the declaration ends.
    /// \param end_column       The column on which the declaration ends.
    /// \returns                The created declaration.
    virtual IDeclaration_namespace_alias *create_namespace_alias(
        ISimple_name const    *alias,
        IQualified_name const *ns,
        int                   start_line = 0,
        int                   start_column = 0,
        int                   end_line = 0,
        int                   end_column = 0) = 0;
};

}  // mdl
}  // mi

#endif
