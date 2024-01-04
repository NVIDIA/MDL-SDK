/******************************************************************************
 * Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDLTLC_TYPES_H
#define MDLTLC_TYPES_H 1

#include <mi/mdl/mdl_iowned.h>
#include <mi/mdl/mdl_definitions.h>
#include <mi/mdl/mdl_mdl.h>

#include <mdl/compiler/compilercore/compilercore_memory_arena.h>

#define MI_MDLTLC_NODE_TYPES
#include <mi/mdl/mdl_distiller_node_types.h>

#include "mdltlc_symbols.h"

#include "mdltlc_pprint.h"

class Type_factory;


typedef mi::mdl::vector<mi::mdl::IType const*>::Type Type_ptr_list;

/// Entry for the map of builtin (e.g. loaded from standard library
/// modules or Distiller-specific) DFs and functions.
class Builtin_entry {
    Symbol *m_fq_symbol;
    Type_ptr_list m_type_list;
    mi::mdl::IDefinition::Semantics m_semantics;
    char const *m_selector;

  public:

  Builtin_entry(Symbol *fq_symbol,
                Type_ptr_list type_list,
                mi::mdl::IDefinition::Semantics semantics)
      : m_fq_symbol(fq_symbol)
        , m_type_list(type_list)
        , m_semantics(semantics)
        , m_selector("::NONE::")
        {
        }

  Builtin_entry(Symbol *fq_symbol,
                Type_ptr_list type_list,
                mi::mdl::IDefinition::Semantics semantics,
                char const *selector)
      : m_fq_symbol(fq_symbol)
        , m_type_list(type_list)
        , m_semantics(semantics)
        , m_selector(selector)
    {
    }

    /// Return the fully qualified name of the entry.
    Symbol *get_fq_symbol() { return m_fq_symbol; }

    /// Return the list of overloaded types for the entry.
    Type_ptr_list &get_type_list() { return m_type_list; }

    /// Return the semantics of the entry.
    mi::mdl::IDefinition::Semantics get_semantics() { return m_semantics; };

    /// Return the semantics of the (read-only) entry.
    mi::mdl::IDefinition::Semantics get_semantics() const { return m_semantics; };

    /// Return the selector of the entry. The selector is the string
    /// to be used for cases in in generated switch statements.
    char const *get_selector() { return m_selector; };

    /// Return the selector of the (read-only) entry.
    char const *get_selector() const { return m_selector; };
};

typedef mi::mdl::ptr_hash_map<Symbol, Builtin_entry >::Type Builtin_type_map;

/// The mdltl type.
///
/// These are the types in the mdltl type system.
class Type {
public:
    enum Kind {
        TK_BOOL,              ///< The bool type.
        TK_INT,               ///< The int type.
        TK_ENUM,              ///< The enum type.
        TK_FLOAT,             ///< The float type.
        TK_DOUBLE,            ///< The double type.
        TK_STRING,            ///< The string type.
        TK_LIGHT_PROFILE,     ///< The string type.
        TK_BSDF,              ///< The bsdf type.
        TK_HAIR_BSDF,         ///< The hair bsdf type.
        TK_EDF,               ///< The edf type.
        TK_VDF,               ///< The vdf type.
        TK_VECTOR,            ///< The vector type.
        TK_MATRIX,            ///< The matrix type.
        TK_ARRAY,             ///< The array type.
        TK_COLOR,             ///< The color type.
        TK_FUNCTION,          ///< The function type.
        TK_STRUCT,            ///< The struct type.
        TK_TEXTURE,           ///< The texture type.
        TK_BSDF_MEASUREMENT,  ///< The bsdf_measurement type.
        TK_MATERIAL_EMISSION, ///< The material_emission type.
        TK_MATERIAL_SURFACE,  ///< The material_surface type.
        TK_MATERIAL_VOLUME,   ///< The material_volume type.
        TK_MATERIAL_GEOMETRY, ///< The material_geometry type.
        TK_MATERIAL,          ///< The material type.

        TK_VAR,               ///< A type variable.
        TK_ERROR,             ///< The error type.
    };

public:
    /// Get the type kind.
    virtual Kind get_kind() const = 0;

    /// Pretty-print the type using the given pretty-printer.
    virtual void pp(pp::Pretty_print &p) const = 0;

protected:
    /// Constructor.
    explicit Type();

private:
};

/// The error type represents a type error in the syntax tree.
class Type_error : public Type
{
    typedef Type Base;
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_ERROR;

    /// Get the type kind.
    Kind get_kind() const;

    /// Pretty-print the type using the given pretty-printer.
    void pp(pp::Pretty_print &p) const;

public:
    /// Constructor.
    explicit Type_error();
};

/// A type variable represents an unknown type.
class Type_var : public Type
{
    typedef Type Base;
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_VAR;

    /// Get the type kind.
    Kind get_kind() const;

    /// Return the type variable's index, which is unique for each
    /// distinct type variable.
    unsigned get_index() const;

    void assign_type(Type *type, Type_factory &tf);

    /// Return the type this type variable is bound to, or NULL if
    /// unbound.
    Type *get_type() const;
    bool is_bound() const;

    /// Pretty-print the type using the given pretty-printer.
    void pp(pp::Pretty_print &p) const;

public:
    /// Constructor.
    explicit Type_var(unsigned index);

private:
    unsigned m_index;
    Type *m_type;
};

/// An atomic type.
class Type_atomic : public Type
{
    typedef Type Base;
protected:
    /// Constructor.
    explicit Type_atomic();
};

/// The mdltl bool type.
class Type_bool : public Type_atomic
{
    typedef Type_atomic Base;
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_BOOL;

    /// Get the type kind.
    Kind get_kind() const;

    /// Pretty-print the type using the given pretty-printer.
    void pp(pp::Pretty_print &p) const;

public:
    /// Constructor.
    explicit Type_bool();
};

/// The mdltl integer type.
class Type_int : public Type_atomic
{
    typedef Type_atomic Base;
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_INT;

    /// Get the type kind.
    Kind get_kind() const;

    /// Pretty-print the type using the given pretty-printer.
    void pp(pp::Pretty_print &p) const;

public:
    /// Constructor.
    explicit Type_int();
};

/// Used for enum fields.
class Enum_variant_list_elem : public mi::mdl::Ast_list_element<Enum_variant_list_elem>
{
    typedef mi::mdl::Ast_list_element<Enum_variant_list_elem> Base;

    friend class mi::mdl::Arena_builder;

public:

    Symbol *get_name();
    Symbol const *get_name() const;

    int get_code();
    int get_code() const;
protected:
    /// Constructor.
    explicit Enum_variant_list_elem(Symbol *name, int code);

private:
    // non copyable
    Enum_variant_list_elem(Enum_variant_list_elem const &) = delete;
    Enum_variant_list_elem &operator=(Enum_variant_list_elem const &) = delete;

protected:
    Symbol *m_name;
    int m_code;
};

typedef mi::mdl::Ast_list<Enum_variant_list_elem> Enum_variant_list;

/// The mdltl enum type.
class Type_enum : public Type_atomic
{
    typedef Type_atomic Base;
  public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_ENUM;

    /// Get the type kind.
    Kind get_kind() const;

    Symbol *get_name();
    Symbol const *get_name() const;

    /// Pretty-print the type using the given pretty-printer.
    void pp(pp::Pretty_print &p) const;

    void add_variant(Enum_variant_list_elem *elem);
    int lookup_variant(Symbol const *name) const;

  public:
    /// Constructor.
    explicit Type_enum(Symbol *name);
  private:
    Symbol *m_name;
    Enum_variant_list m_variants;
};

/// The mdltl float type.
class Type_float : public Type_atomic
{
    typedef Type_atomic Base;
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_FLOAT;

    /// Get the type kind.
    Kind get_kind() const;

    /// Pretty-print the type using the given pretty-printer.
    void pp(pp::Pretty_print &p) const;

public:
    /// Constructor.
    explicit Type_float();
};

/// The mdltl double type.
class Type_double : public Type_atomic
{
    typedef Type_atomic Base;
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_DOUBLE;

    /// Get the type kind.
    Kind get_kind() const;

    /// Pretty-print the type using the given pretty-printer.
    void pp(pp::Pretty_print &p) const;

public:
    /// Constructor.
    explicit Type_double();
};

/// The mdltl string type.
class Type_string : public Type_atomic
{
    typedef Type_atomic Base;
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_STRING;

    /// Get the type kind.
    Kind get_kind() const;

    /// Pretty-print the type using the given pretty-printer.
    void pp(pp::Pretty_print &p) const;

public:
    /// Constructor.
    explicit Type_string();
};

/// The mdltl light profile type.
class Type_light_profile : public Type
{
    typedef Type Base;
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_LIGHT_PROFILE;

    /// Get the type kind.
    Kind get_kind() const;

    /// Pretty-print the type using the given pretty-printer.
    void pp(pp::Pretty_print &p) const;

public:
    /// Constructor.
    explicit Type_light_profile();
};

/// The mdltl bsdf type.
class Type_bsdf : public Type
{
    typedef Type Base;
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_BSDF;

    /// Get the type kind.
    Kind get_kind() const;

    /// Pretty-print the type using the given pretty-printer.
    void pp(pp::Pretty_print &p) const;

public:
    /// Constructor.
    explicit Type_bsdf();
};

/// The mdltl hair_bsdf type.
class Type_hair_bsdf : public Type
{
    typedef Type Base;
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_HAIR_BSDF;

    /// Get the type kind.
    Kind get_kind() const;

    /// Pretty-print the type using the given pretty-printer.
    void pp(pp::Pretty_print &p) const;

public:
    /// Constructor.
    explicit Type_hair_bsdf();
};

/// The mdltl edf type.
class Type_edf : public Type
{
    typedef Type Base;
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_EDF;

    /// Get the type kind.
    Kind get_kind() const;

    /// Pretty-print the type using the given pretty-printer.
    void pp(pp::Pretty_print &p) const;

public:
    /// Constructor.
    explicit Type_edf();
};

/// The mdltl vdf type.
class Type_vdf : public Type
{
    typedef Type Base;
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_VDF;

    /// Get the type kind.
    Kind get_kind() const;

    /// Pretty-print the type using the given pretty-printer.
    void pp(pp::Pretty_print &p) const;

public:
    /// Constructor.
    explicit Type_vdf();
};

/// The mdltl vector type.
class Type_vector : public Type
{
    typedef Type Base;
  public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_VECTOR;

    /// Get the type kind.
    Kind get_kind() const;

    /// Pretty-print the type using the given pretty-printer.
    void pp(pp::Pretty_print &p) const;

    Type *get_element_type() const;
    unsigned get_size() const;

  public:
    /// Constructor.
    explicit Type_vector(unsigned size, Type *element_type);
  private:
    unsigned m_size;
    Type *m_element_type;
};

/// The mdltl matrix type.
class Type_matrix : public Type
{
    typedef Type Base;
  public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_MATRIX;

    /// Get the type kind.
    Kind get_kind() const;

    /// Pretty-print the type using the given pretty-printer.
    void pp(pp::Pretty_print &p) const;

    Type *get_element_type() const;
    unsigned get_column_count() const;

  public:
    /// Constructor.
    explicit Type_matrix(unsigned column_count, Type *element_type);

  private:
    unsigned m_column_count;
    Type *m_element_type;
};

/// The mdltl array type.
class Type_array : public Type
{
    typedef Type Base;
  public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_ARRAY;

    /// Get the type kind.
    Kind get_kind() const;

    /// Pretty-print the type using the given pretty-printer.
    void pp(pp::Pretty_print &p) const;

    Type *get_element_type() const;

  public:
    /// Constructor.
    explicit Type_array(Type *element_type);
  private:
    Type *m_element_type;
};

/// The mdltl color type.
class Type_color : public Type
{
    typedef Type Base;
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_COLOR;

    /// Get the type kind.
    Kind get_kind() const;

    /// Pretty-print the type using the given pretty-printer.
    void pp(pp::Pretty_print &p) const;

public:
    /// Constructor.
    explicit Type_color();
};

/// Usef for function parameters.
class Type_list_elem : public mi::mdl::Ast_list_element<Type_list_elem>
{
    typedef mi::mdl::Ast_list_element<Type_list_elem> Base;

    friend class mi::mdl::Arena_builder;

public:

    Type *get_type();
    Type const *get_type() const;

    /// Pretty-print the type using the given pretty-printer.
    void pp(pp::Pretty_print &p) const;

protected:
    /// Constructor.
    explicit Type_list_elem(Type *type);

private:
    // non copyable
    Type_list_elem(Type_list_elem const &) = delete;
    Type_list_elem &operator=(Type_list_elem const &) = delete;

protected:
    Type *m_type;
};

typedef mi::mdl::Ast_list<Type_list_elem> Type_list;


/// Used for struct fields.
class Named_type_list_elem : public mi::mdl::Ast_list_element<Named_type_list_elem>
{
    typedef mi::mdl::Ast_list_element<Named_type_list_elem> Base;

    friend class mi::mdl::Arena_builder;

public:

    Symbol *get_name();
    Symbol const *get_name() const;

    Type *get_type();
    Type const *get_type() const;

    /// Pretty-print the type using the given pretty-printer.
    void pp(pp::Pretty_print &p) const;

protected:
    /// Constructor.
    explicit Named_type_list_elem(Symbol *name, Type *type);

private:
    // non copyable
    Named_type_list_elem(Named_type_list_elem const &) = delete;
    Named_type_list_elem &operator=(Named_type_list_elem const &) = delete;

protected:
    Symbol *m_name;
    Type *m_type;
};

typedef mi::mdl::Ast_list<Named_type_list_elem> Named_type_list;

/// The mdltl struct type.
class Type_struct : public Type
{
    typedef Type Base;
  public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_STRUCT;

    /// Get the type kind.
    Kind get_kind() const;
    Symbol *get_name();
    Symbol const *get_name() const;

    /// Pretty-print the type using the given pretty-printer.
    void pp(pp::Pretty_print &p) const;

    void add_field(Named_type_list_elem *field_type);
    Type *get_field_type(Symbol const*field_name);

  public:
    /// Constructor.
    explicit Type_struct(Symbol *name);

  private:
    Symbol *m_name;
    Named_type_list m_fields;
};

/// The mdltl texture type.
class Type_texture : public Type
{
    typedef Type Base;
  public:
    enum Texture_kind {
        TK_2D,
        TK_3D,
        TK_CUBE,
        TK_PTEX,
        TK_BSDF_DATA
    };

    /// The kind of this subclass.
    static Kind const s_kind = TK_TEXTURE;

    /// Get the type kind.
    Kind get_kind() const;

    Texture_kind get_texture_kind();

    /// Pretty-print the type using the given pretty-printer.
    void pp(pp::Pretty_print &p) const;

  public:
    /// Constructor.
    explicit Type_texture(Texture_kind texture_kind);

  private:
    Texture_kind m_texture_kind;
};

/// The mdltl bsdf_measurement type.
class Type_bsdf_measurement : public Type
{
    typedef Type Base;
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_BSDF_MEASUREMENT;

    /// Get the type kind.
    Kind get_kind() const;

    /// Pretty-print the type using the given pretty-printer.
    void pp(pp::Pretty_print &p) const;

public:
    /// Constructor.
    explicit Type_bsdf_measurement();
};

/// The mdltl material_emission type.
class Type_material_emission : public Type
{
    typedef Type Base;
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_MATERIAL_EMISSION;

    /// Get the type kind.
    Kind get_kind() const;

    /// Pretty-print the type using the given pretty-printer.
    void pp(pp::Pretty_print &p) const;

public:
    /// Constructor.
    explicit Type_material_emission();
};

/// The mdltl material_surface type.
class Type_material_surface : public Type
{
    typedef Type Base;
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_MATERIAL_SURFACE;

    /// Get the type kind.
    Kind get_kind() const;

    /// Pretty-print the type using the given pretty-printer.
    void pp(pp::Pretty_print &p) const;

public:
    /// Constructor.
    explicit Type_material_surface();
};

/// The mdltl material_volume type.
class Type_material_volume : public Type
{
    typedef Type Base;
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_MATERIAL_VOLUME;

    /// Get the type kind.
    Kind get_kind() const;

    /// Pretty-print the type using the given pretty-printer.
    void pp(pp::Pretty_print &p) const;

public:
    /// Constructor.
    explicit Type_material_volume();
};

/// The mdltl material_geometry type.
class Type_material_geometry : public Type
{
    typedef Type Base;
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_MATERIAL_GEOMETRY;

    /// Get the type kind.
    Kind get_kind() const;

    /// Pretty-print the type using the given pretty-printer.
    void pp(pp::Pretty_print &p) const;

public:
    /// Constructor.
    explicit Type_material_geometry();
};

/// The mdltl material type.
class Type_material : public Type
{
    typedef Type Base;
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_MATERIAL;

    /// Get the type kind.
    Kind get_kind() const;

    /// Pretty-print the type using the given pretty-printer.
    void pp(pp::Pretty_print &p) const;

public:
    /// Constructor.
    explicit Type_material();
};


/// The mdltl function type.
class Type_function : public Type
{
    typedef Type Base;
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_FUNCTION;

    /// Get the type kind.
    Kind get_kind() const;

    /// Pretty-print the type using the given pretty-printer.
    void pp(pp::Pretty_print &p) const;

    Type *get_return_type();
    Type const *get_return_type() const;
    int get_parameter_count();
    Type *get_parameter_type(int index);

    void set_semantics(mi::mdl::IDefinition::Semantics semantics);
    mi::mdl::IDefinition::Semantics get_semantics();
    mi::mdl::IDefinition::Semantics get_semantics() const;

    void set_selector(char const *selector);
    char const *get_selector();
    char const *get_selector() const;

    void set_node_type(mi::mdl::Node_type const *nt);
    mi::mdl::Node_type const *get_node_type() const;
    mi::mdl::Node_type const *get_node_type();

public:
    /// Constructor.
    explicit Type_function(Type *return_type);
    void add_parameter(Type_list_elem *param_type);
private:
    Type *m_return_type;
    Type_list m_parameters;
    int m_parameter_count;
    mi::mdl::IDefinition::Semantics m_semantics;
    char const *m_selector;
    mi::mdl::Node_type const *m_node_type;
};

/// Check if a type is of a certain type.
template<typename T>
bool is(Type *type) {
    return type->get_kind() == T::s_kind;
}

/// Check if a type is of a certain type.
template<typename T>
bool is(Type const *type) {
    return type->get_kind() == T::s_kind;
}

/// Check if a type is of type Type_atomic.
template<>
inline bool is<Type_atomic>(Type *type) {
    switch (type->get_kind()) {
    case Type::TK_BOOL:
    case Type::TK_INT:
    case Type::TK_ENUM:
    case Type::TK_FLOAT:
    case Type::TK_DOUBLE:
    case Type::TK_STRING:
        return true;
    default:
        return false;
    }
}

/// Cast to subtype or return NULL if types do not match.
template<typename T>
T *as(Type *type) {
    return is<T>(type) ? static_cast<T *>(type) : NULL;
}

/// Cast to subtype or return NULL if types do not match.
template<typename T>
T const *as(Type const *type) {
    return is<T>(type) ? static_cast<T const *>(type) : NULL;
}

/// A static_cast with check in debug mode
template <typename T>
inline T *cast(Type *arg) {
    MDL_ASSERT(arg == NULL || is<T>(arg));
    return static_cast<T *>(arg);
}

/// The interface for creating types.
/// A Type_factory interface can be obtained by calling
/// the method get_type_factory() on the interfaces Compilation_unit and Value_factory.
class Type_factory : public mi::mdl::Interface_owned
{
public:
    /// Get the (singleton) error type instance.
    Type_error *get_error();

    /// Get the (singleton) bool type instance.
    Type_bool *get_bool();

    /// Get the (singleton) int type instance.
    Type_int *get_int();

    /// Get the (singleton for now) enum type instance.
    Type_enum *create_enum(Symbol *name);

    /// Get the (singleton) float type instance.
    Type_float *get_float();

    /// Get the (singleton) double type instance.
    Type_double *get_double();

    /// Get the (singleton) string type instance.
    Type_string *get_string();

    /// Get the (singleton for now) light_profile type instance.
    Type_light_profile *get_light_profile();

    /// Get the (singleton for now) bsdf type instance.
    Type_bsdf *get_bsdf();

    /// Get the (singleton for now) hair_bsdf type instance.
    Type_hair_bsdf *get_hair_bsdf();

    /// Get the (singleton for now) edf type instance.
    Type_edf *get_edf();

    /// Get the (singleton for now) vdf type instance.
    Type_vdf *get_vdf();

    /// Get the (singleton for now) vector type instance.
    Type_vector *get_vector(unsigned size, Type *element_type);

    /// Get the (singleton for now) matrix type instance.
    Type_matrix *get_matrix(unsigned column_count, Type *element_type);

    /// Get the (singleton for now) array type instance.
    Type_array *get_array(Type *element_type);

    /// Get the (singleton for now) color type instance.
    Type_color *get_color();

    /// Get a function type with the given return type.
    Type_function *create_function(Type *return_type);

    /// Create a fresh parameter wrapper for the given type.
    Type_list_elem *create_type_list_elem(Type *type);

    /// Create a fresh struct field wrapper for the given name and type.
    Named_type_list_elem *create_named_type_list_elem(Symbol *name, Type *type);

    /// Get the (singleton for now) struct type instance.
    Type_struct *create_struct(Symbol *name);

    /// Get the (singleton for now) texture type instance.
    Type_texture *get_texture(Type_texture::Texture_kind texture_kind);

    /// Get the (singleton for now) bsdf_measurement type instance.
    Type_bsdf_measurement *get_bsdf_measurement();

    /// Get the (singleton for now) material_emission type instance.
    Type_material_emission *get_material_emission();

    /// Get the (singleton for now) material_surface type instance.
    Type_material_surface *get_material_surface();

    /// Get the (singleton for now) material_volume type instance.
    Type_material_volume *get_material_volume();

    /// Get the (singleton for now) material_geometry type instance.
    Type_material_geometry *get_material_geometry();

    /// Get the (singleton for now) material type instance.
    Type_material *get_material();

    /* /// Get the (singleton for now) tex_gamma_mode type instance. */
    /* Type_tex_gamma_mode *get_tex_gamma_mode(); */

    /* /// Get the (singleton for now) intensity_mode type instance. */
    /* Type_intensity_mode *get_intensity_mode(); */

    /* /// Get the (singleton for now) scatter_mode type instance. */
    /* Type_scatter_mode *get_scatter_mode(); */

    /// Get a fresh type variable.
    Type_var *create_type_variable();

    /// Create a new type that matches the given MDL type.
    Type *import_type(mi::mdl::IType const *mdl_type);

    /// Return true if both types are representing the same type.
    bool types_equal(Type *type1, Type *type2);

    /// Return true if the first type (as an actual parameter) matches
    /// the second (as a formal parameter).
    bool types_match(Type *type1, Type *type2);

    /// Return the symbol table of this type factory.
    Symbol_table &get_symbol_table() { return m_symtab; }

    /// Constructs a new type factory.
    ///
    /// \param arena         the memory arena used to allocate new types
    /// \param sym_tab       the symbol table for symbols inside types
    explicit Type_factory(
        mi::mdl::Memory_arena  &arena,
        Symbol_table  &sym_tab);

private:
    mi::mdl::Memory_arena &m_arena;

    /// The builder for types.
    mi::mdl::Arena_builder m_builder;

    /// The symbol table used to create new symbols for types.
    Symbol_table &m_symtab;

    unsigned m_next_var_index;

    Type_list m_structs;
    Type_list m_enums;
};

bool is_scalar(Type *type);
bool is_color(Type *type);
bool is_vector(Type *type);
bool is_matrix(Type *type);
bool is_array(Type *type);
Type *promoted_type(Type *type1, Type *type2);
Type *deref(Type *type);

#endif // MDLTLC_TYPES_H
