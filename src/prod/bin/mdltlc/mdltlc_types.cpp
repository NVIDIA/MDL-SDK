/******************************************************************************
 * Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
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

#include <mdl/compiler/compilercore/compilercore_memory_arena.h>
#include <mi/mdl/mdl_symbols.h>

#include "mdltlc_types.h"

#include "mdltl_enum_names.i"  // get_semantics_name()

static const bool SHOW_TYPE_VARS = false;

namespace {

// create the builtin types
#define BUILTIN_TYPE(type, name, args) type name args;

BUILTIN_TYPE(Type_error,             mdltlc_error_type, )
BUILTIN_TYPE(Type_bool,              mdltlc_bool_type, )
BUILTIN_TYPE(Type_int,               mdltlc_int_type, )
BUILTIN_TYPE(Type_float,             mdltlc_float_type, )
BUILTIN_TYPE(Type_double,            mdltlc_double_type, )
BUILTIN_TYPE(Type_string,            mdltlc_string_type, )
BUILTIN_TYPE(Type_light_profile,     mdltlc_light_profile_type, )
BUILTIN_TYPE(Type_bsdf,              mdltlc_bsdf_type, )
BUILTIN_TYPE(Type_hair_bsdf,         mdltlc_hair_bsdf_type, )
BUILTIN_TYPE(Type_edf,               mdltlc_edf_type, )
BUILTIN_TYPE(Type_vdf,               mdltlc_vdf_type, )
BUILTIN_TYPE(Type_color,             mdltlc_color_type, )
BUILTIN_TYPE(Type_bsdf_measurement,  mdltlc_bsdf_measurement_type, )
BUILTIN_TYPE(Type_material_emission, mdltlc_material_emission_type, )
BUILTIN_TYPE(Type_material_surface,  mdltlc_material_surface_type, )
BUILTIN_TYPE(Type_material_volume,   mdltlc_material_volume_type, )
BUILTIN_TYPE(Type_material_geometry, mdltlc_material_geometry_type, )
BUILTIN_TYPE(Type_material,          mdltlc_material_type, )
}  // anonymous


/// Return true of the given type is a scalar numeric type.
bool is_scalar(Type *type) {
    return is<Type_int>(type) || is<Type_float>(type) || is<Type_double>(type);
}

/// Return true of the given type is the color type.
bool is_color(Type *type) {
    return is<Type_color>(type);
}

/// Return true of the given type is a vector type.
bool is_vector(Type *type) {
    return is<Type_vector>(type);
}

/// Return true of the given type is a matrix type.
bool is_matrix(Type *type) {
    return is<Type_matrix>(type);
}

/// Return true of the given type is an array type.
bool is_array(Type *type) {
    return is<Type_array>(type);
}

/// Return the type to which the given types can be promoted. `type1`
/// and `type2` must be scalar numeric types.
Type *promoted_type(Type *type1, Type *type2) {
    if (is<Type_double>(type1))
        return type1;
    if (is<Type_double>(type2))
        return type2;
    if (is<Type_float>(type1))
        return type1;
    if (is<Type_float>(type2))
        return type2;
    return type1;
}

/// Dereference bound type variables to the type they represent.
Type *deref(Type *type) {
    if (Type_var *tv = as<Type_var>(type)) {
        if (tv->is_bound()) {
            return tv->get_type();
        }
    }
    return type;
}

// ---------------------------- Base type ----------------------------

// Constructor.
Type::Type()
{
}


// ---------------------------- Error type ----------------------------

// Constructor.
Type_error::Type_error()
: Base()
{
}

// Get the type kind.
Type::Kind Type_error::get_kind() const
{
    return s_kind;
}

void Type_error::pp(pp::Pretty_print &p) const {
    p.string("error");
}

// ---------------------------- Atomic type ----------------------------

// Constructor.
Type_atomic::Type_atomic()
    : Base()
{
}

// ---------------------------- type variable--------------------------

// Constructor.
Type_var::Type_var(unsigned index)
    : Base()
    , m_index(index)
    , m_type()
{
}

// Get the type kind.
Type::Kind Type_var::get_kind() const
{
    return s_kind;
}

// Get the variable index.
unsigned Type_var::get_index() const {
    return m_index;
}

Type *Type_var::get_type() const {
    return m_type;
}

void Type_var::assign_type(Type *type, Type_factory &tf) {
    if (m_type) {
        if (tf.types_equal(m_type, type))
            return;
        printf("[error] type variable assigned to multiple types\n");
        return;
    } else {
        m_type = type;
    }
}

bool Type_var::is_bound() const {
    return m_type;
}

void Type_var::pp(pp::Pretty_print &p) const {
    if (SHOW_TYPE_VARS) {
        p.string("v");
        if (is_bound()) {
            p.string("v");
            p.integer(m_index);
            p.string("<");
            get_type()->pp(p);
            p.string(">");
        }
    } else {
        if (is_bound()) {
            get_type()->pp(p);
        } else {
            p.string("v");
            p.integer(m_index);
        }
    }
}

// ----------------------------- bool type ----------------------------

// Constructor.
Type_bool::Type_bool()
    : Base()
{
}

// Get the type kind.
Type::Kind Type_bool::get_kind() const
{
    return s_kind;
}

void Type_bool::pp(pp::Pretty_print &p) const {
    p.string("bool");
}

// ----------------------------- int type -----------------------------

// Constructor.
Type_int::Type_int()
  : Base()
{
}

// Get the type kind.
Type::Kind Type_int::get_kind() const
{
    return s_kind;
}

void Type_int::pp(pp::Pretty_print &p) const {
    p.string("int");
}

// ----------------------------- enum type -----------------------------

// Constructor.
Type_enum::Type_enum(Symbol *name)
    : Base()
    , m_name(name)
    , m_variants()
{
}

// Get the type kind.
Type::Kind Type_enum::get_kind() const
{
    return s_kind;
}

// Get the enum name.
Symbol *Type_enum::get_name()
{
    return m_name;
}

// Get the enum name.
Symbol const *Type_enum::get_name() const
{
    return m_name;
}

void Type_enum::add_variant(Enum_variant_list_elem *elem) {
    m_variants.push(elem);
}

int Type_enum::lookup_variant(Symbol const *name) const {
    for (Enum_variant_list::const_iterator it(m_variants.begin()), end(m_variants.end());
         it != end; ++it) {
        if (it->get_name() == name)
            return it->get_code();
    }
    return -1;
}

void Type_enum::pp(pp::Pretty_print &p) const {
    p.string(m_name->get_name());
}

// -------------------------- enum variant element -------------------------

Enum_variant_list_elem::Enum_variant_list_elem(Symbol *name, int code)
    : Base()
    , m_name(name)
    , m_code(code)
{
}

Symbol *Enum_variant_list_elem::get_name() {
    return m_name;
}

Symbol const *Enum_variant_list_elem::get_name() const {
    return m_name;
}

int Enum_variant_list_elem::get_code() {
    return m_code;
}

int Enum_variant_list_elem::get_code() const {
    return m_code;
}

// ---------------------------- float type ----------------------------

// Constructor.
Type_float::Type_float()
    : Base()
{
}

// Get the type kind.
Type::Kind Type_float::get_kind() const
{
    return s_kind;
}

void Type_float::pp(pp::Pretty_print &p) const {
    p.string("float");
}

// ---------------------------- double type ----------------------------

// Constructor.
Type_double::Type_double()
    : Base()
{
}

// Get the type kind.
Type::Kind Type_double::get_kind() const
{
    return s_kind;
}

void Type_double::pp(pp::Pretty_print &p) const {
    p.string("double");
}

// ---------------------------- string type ----------------------------

// Constructor.
Type_string::Type_string()
    : Base()
{
}

// Get the type kind.
Type::Kind Type_string::get_kind() const
{
    return s_kind;
}

void Type_string::pp(pp::Pretty_print &p) const {
    p.string("string");
}


// ----------------------------- light_profile type -----------------------------

// Constructor.
Type_light_profile::Type_light_profile()
    : Base()
{
}

// Get the type kind.
Type::Kind Type_light_profile::get_kind() const
{
    return s_kind;
}

void Type_light_profile::pp(pp::Pretty_print &p) const {
    p.string("light_profile");
}

// ----------------------------- bsdf type -----------------------------

// Constructor.
Type_bsdf::Type_bsdf()
  : Base()
{
}

// Get the type kind.
Type::Kind Type_bsdf::get_kind() const
{
    return s_kind;
}

void Type_bsdf::pp(pp::Pretty_print &p) const {
    p.string("bsdf");
}

// ----------------------------- hair_bsdf type -----------------------------

// Constructor.
Type_hair_bsdf::Type_hair_bsdf()
  : Base()
{
}

// Get the type kind.
Type::Kind Type_hair_bsdf::get_kind() const
{
    return s_kind;
}

void Type_hair_bsdf::pp(pp::Pretty_print &p) const {
    p.string("hair_bsdf");
}

// ----------------------------- edf type -----------------------------

// Constructor.
Type_edf::Type_edf()
: Base()
{
}

// Get the type kind.
Type::Kind Type_edf::get_kind() const
{
    return s_kind;
}

void Type_edf::pp(pp::Pretty_print &p) const {
    p.string("edf");
}

// ----------------------------- vdf type -----------------------------

// Constructor.
Type_vdf::Type_vdf()
: Base()
{
}

// Get the type kind.
Type::Kind Type_vdf::get_kind() const
{
    return s_kind;
}

void Type_vdf::pp(pp::Pretty_print &p) const {
    p.string("vdf");
}

// ----------------------------- vector type -----------------------------

// Constructor.
Type_vector::Type_vector(unsigned size, Type *element_type)
: Base()
, m_size(size)
, m_element_type(element_type)
{
}

// Get the type kind.
Type::Kind Type_vector::get_kind() const
{
    return s_kind;
}

void Type_vector::pp(pp::Pretty_print &p) const {
    m_element_type->pp(p);
    p.integer(m_size);
}

unsigned Type_vector::get_size() const {
    return m_size;
}

Type * Type_vector::get_element_type() const {
    return m_element_type;
}

// ----------------------------- matrix type -----------------------------

// Constructor.
Type_matrix::Type_matrix(unsigned column_count, Type *element_type)
: Base()
, m_column_count(column_count)
, m_element_type(element_type)
{
}

// Get the type kind.
Type::Kind Type_matrix::get_kind() const
{
    return s_kind;
}

void Type_matrix::pp(pp::Pretty_print &p) const {
    m_element_type->pp(p);
    p.string("x");
    p.integer(m_column_count);
}

Type *Type_matrix::get_element_type() const {
    return m_element_type;
}

unsigned Type_matrix::get_column_count() const {
    return m_column_count;
}

// ----------------------------- array type -----------------------------

// Constructor.
Type_array::Type_array(Type *element_type)
: Base()
, m_element_type(element_type)
{
}

// Get the type kind.
Type::Kind Type_array::get_kind() const
{
    return s_kind;
}

// Get the type kind.
Type *Type_array::get_element_type() const
{
    return m_element_type;
}

void Type_array::pp(pp::Pretty_print &p) const {
    m_element_type->pp(p);
    p.string("[]");
}

// ----------------------------- color type -----------------------------

// Constructor.
Type_color::Type_color()
: Base()
{
}

// Get the type kind.
Type::Kind Type_color::get_kind() const
{
    return s_kind;
}

void Type_color::pp(pp::Pretty_print &p) const {
    p.string("color");
}

// ----------------------------- struct type -----------------------------

// Constructor.
Type_struct::Type_struct(Symbol *name)
  : Base()
  , m_name(name)
  , m_fields()
{
}

// Get the type kind.
Type::Kind Type_struct::get_kind() const
{
    return s_kind;
}

// Get the struct name.
Symbol *Type_struct::get_name()
{
    return m_name;
}

// Get the struct name.
Symbol const *Type_struct::get_name() const

{
    return m_name;
}

void Type_struct::add_field(Named_type_list_elem *field_type) {
    m_fields.push(field_type);
}

Type *Type_struct::get_field_type(Symbol const *field_name) {
    for (Named_type_list::iterator it(m_fields.begin()), end(m_fields.end());
         it != end; ++it) {
        if (it->get_name() == field_name)
            return it->get_type();
    }
    return nullptr;
}

void Type_struct::pp(pp::Pretty_print &p) const {
    p.string("struct");
    p.space();
    p.string(m_name->get_name());
}

// ----------------------------- texture type -----------------------------

// Constructor.
Type_texture::Type_texture(Type_texture::Texture_kind texture_kind)
  : Base()
  , m_texture_kind(texture_kind)
{
}

// Get the type kind.
Type::Kind Type_texture::get_kind() const
{
    return s_kind;
}

// Get the texture kind.
Type_texture::Texture_kind Type_texture::get_texture_kind()
{
    return m_texture_kind;
}

void Type_texture::pp(pp::Pretty_print &p) const {
    switch (m_texture_kind) {
    case Texture_kind::TK_2D: {
        p.string("texture_2d");
        break;
    }
    case Texture_kind::TK_3D: {
        p.string("texture_3d");
        break;
    }
    case Texture_kind::TK_CUBE: {
        p.string("cube");
        break;
    }
    case Texture_kind::TK_PTEX: {
        p.string("ptex");
        break;
    }
    case Texture_kind::TK_BSDF_DATA: {
        p.string("bsdf_data");
        break;
    }
    }
}

// ----------------------------- bsdf_measurement type -----------------------------

// Constructor.
Type_bsdf_measurement::Type_bsdf_measurement()
: Base()
{
}

// Get the type kind.
Type::Kind Type_bsdf_measurement::get_kind() const
{
    return s_kind;
}

void Type_bsdf_measurement::pp(pp::Pretty_print &p) const {
    p.string("bsdf_measurement");
}

// ----------------------------- material_emission type -----------------------------

// Constructor.
Type_material_emission::Type_material_emission()
: Base()
{
}

// Get the type kind.
Type::Kind Type_material_emission::get_kind() const
{
    return s_kind;
}

void Type_material_emission::pp(pp::Pretty_print &p) const {
    p.string("material_emission");
}

// ----------------------------- material_surface type -----------------------------

// Constructor.
Type_material_surface::Type_material_surface()
: Base()
{
}

// Get the type kind.
Type::Kind Type_material_surface::get_kind() const
{
    return s_kind;
}

void Type_material_surface::pp(pp::Pretty_print &p) const {
    p.string("material_surface");
}

// ----------------------------- material_geometry type -----------------------------

// Constructor.
Type_material_geometry::Type_material_geometry()
: Base()
{
}

// Get the type kind.
Type::Kind Type_material_geometry::get_kind() const
{
    return s_kind;
}

void Type_material_geometry::pp(pp::Pretty_print &p) const {
    p.string("material_geometry");
}

// ----------------------------- material_volume type -----------------------------

// Constructor.
Type_material_volume::Type_material_volume()
: Base()
{
}

// Get the type kind.
Type::Kind Type_material_volume::get_kind() const
{
    return s_kind;
}

void Type_material_volume::pp(pp::Pretty_print &p) const {
    p.string("material_volume");
}

// ----------------------------- material type -----------------------------

// Constructor.
Type_material::Type_material()
: Base()
{
}

// Get the type kind.
Type::Kind Type_material::get_kind() const
{
    return s_kind;
}

void Type_material::pp(pp::Pretty_print &p) const {
    p.string("material");
}

Type_list_elem::Type_list_elem(Type *type)
    : Base()
    , m_type(type)
{
}

Type *Type_list_elem::get_type() {
    return m_type;
}

Type const *Type_list_elem::get_type() const {
    return m_type;
}

void Type_list_elem::pp(pp::Pretty_print &p) const {
    m_type->pp(p);
}

// -------------------------- type list element -------------------------

Named_type_list_elem::Named_type_list_elem(Symbol *name, Type *type)
    : Base()
    , m_name(name)
    , m_type(type)
{
}

Symbol *Named_type_list_elem::get_name() {
    return m_name;
}

Symbol const *Named_type_list_elem::get_name() const {
    return m_name;
}

Type *Named_type_list_elem::get_type() {
    return m_type;
}

Type const *Named_type_list_elem::get_type() const {
    return m_type;
}

void Named_type_list_elem::pp(pp::Pretty_print &p) const {
    m_type->pp(p);
}

// ---------------------------- function type ----------------------------

// Constructor.
Type_function::Type_function(Type *return_type)
    : m_return_type(return_type)
    , m_parameters()
    , m_parameter_count(0)
    , m_semantics(mi::mdl::IDefinition::Semantics::DS_UNKNOWN)
    , m_selector(nullptr)
    , m_node_type(nullptr)
{
}

void Type_function::add_parameter(Type_list_elem *param_type) {
    m_parameters.push(param_type);
    m_parameter_count++;
}

// Get the type kind.
Type::Kind Type_function::get_kind() const
{
    return s_kind;
}

// Get the return type.
Type *Type_function::get_return_type()
{
    return m_return_type;
}

// Get the return type.
Type const *Type_function::get_return_type() const
{
    return m_return_type;
}

// Get the builtin semantics for this function type.
mi::mdl::IDefinition::Semantics Type_function::get_semantics()
{
    return m_semantics;
}

// Get the builtin semantics for this function type.
mi::mdl::IDefinition::Semantics Type_function::get_semantics() const
{
    return m_semantics;
}

// Set the builtin semantics for this function type.
void Type_function::set_semantics(mi::mdl::IDefinition::Semantics semantics)
{
    m_semantics = semantics;
}

void Type_function::set_selector(char const *selector) {
    m_selector = selector;
}

char const *Type_function::get_selector() {
    return m_selector;
}

char const *Type_function::get_selector() const {
    return m_selector;
}

void Type_function::set_node_type(mi::mdl::Node_type const *nt) {
    m_node_type = nt;
}

mi::mdl::Node_type const *Type_function::get_node_type() const{
    return m_node_type;
}

mi::mdl::Node_type const *Type_function::get_node_type() {
    return m_node_type;
}

void Type_function::pp(pp::Pretty_print &p) const {

    m_return_type->pp(p);
    p.string("(*)");

    bool first = true;
    p.lparen();

    for (mi::mdl::Ast_list<Type_list_elem>::const_iterator it(m_parameters.begin()), end(m_parameters.end());
         it != end;
         ++it) {
        if (first)
            first = false;
        else {
            p.comma();
            p.space();
        }
        it->get_type()->pp(p);
    }
    p.rparen();
}

int Type_function::get_parameter_count() {
    return m_parameter_count;
}

Type *Type_function::get_parameter_type(int index) {
    int i = 0;
    for (mi::mdl::Ast_list<Type_list_elem>::iterator it(m_parameters.begin()), end(m_parameters.end());
         it != end ;
         ++it, i++) {
        if (i == index)
            return it->get_type();
    }
    return nullptr;
}

// ---------------------------- type factory ----------------------------

// Constructor.
Type_factory::Type_factory(
    mi::mdl::Memory_arena  &arena,
    Symbol_table  &sym_tab)
    : m_arena(arena)
    , m_builder(m_arena)
    , m_symtab(sym_tab)
    , m_next_var_index(0)
    , m_structs()
    , m_enums()
{
}

// Get the (singleton) error type instance.
Type_error *Type_factory::get_error()
{
    return &mdltlc_error_type;
}

// Get the (singleton) bool type instance.
Type_bool *Type_factory::get_bool()
{
    return &mdltlc_bool_type;
}

// Get the (singleton) int type instance.
Type_int *Type_factory::get_int()
{
    return &mdltlc_int_type;
}

// Get the (singleton) enum type instance.
Type_enum *Type_factory::create_enum(Symbol *name)
{
    for (Type_list::iterator it(m_enums.begin()), end(m_enums.end());
         it != end; ++it) {
        if (Type_enum *te = as<Type_enum>(it->get_type())) {
            if (te->get_name() == name) {
                return te;
            }
        }

    }
    Type_enum *te = m_builder.create<Type_enum>(name);
    Type_list_elem *tle = m_builder.create<Type_list_elem>(te);
    m_enums.push(tle);
    return te;
}

// Get the (singleton) float type instance.
Type_float *Type_factory::get_float()
{
    return &mdltlc_float_type;
}

// Get the (singleton) double type instance.
Type_double *Type_factory::get_double()
{
    return &mdltlc_double_type;
}

// Get the (singleton) bsdf type instance.
Type_bsdf *Type_factory::get_bsdf()
{
    return &mdltlc_bsdf_type;
}

// Get the (singleton) hair_bsdf type instance.
Type_hair_bsdf *Type_factory::get_hair_bsdf()
{
    return &mdltlc_hair_bsdf_type;
}

// Get the (singleton) edf type instance.
Type_edf *Type_factory::get_edf()
{
    return &mdltlc_edf_type;
}

// Get the (singleton) vdf type instance.
Type_vdf *Type_factory::get_vdf()
{
    return &mdltlc_vdf_type;
}

// Get the (singleton) string type instance.
Type_string *Type_factory::get_string()
{
    return &mdltlc_string_type;
}

// Get the (singleton) light_profile type instance.
Type_light_profile *Type_factory::get_light_profile()
{
    return &mdltlc_light_profile_type;
}

// Get the (singleton) vector type instance.
Type_vector *Type_factory::get_vector(unsigned size, Type *element_type)
{
    return m_builder.create<Type_vector>(size, element_type);
}

// Get the (singleton) matrix type instance.
Type_matrix *Type_factory::get_matrix(unsigned column_count, Type *element_type)
{
    return m_builder.create<Type_matrix>(column_count, element_type);
}

// Get the (singleton) array type instance.
Type_array *Type_factory::get_array(Type *element_type)
{
    return m_builder.create<Type_array>(element_type);
}

// Get the (singleton) color type instance.
Type_color *Type_factory::get_color()
{
    return &mdltlc_color_type;
}

// Get the (singleton) struct type instance.
Type_struct *Type_factory::create_struct(Symbol *name)
{
    for (Type_list::iterator it(m_structs.begin()), end(m_structs.end());
         it != end; ++it) {
        if (Type_struct *ts = as<Type_struct>(it->get_type())) {
            if (ts->get_name() == name) {
                return ts;
            }
        }

    }
    Type_struct *ts = m_builder.create<Type_struct>(name);
    Type_list_elem *tle = m_builder.create<Type_list_elem>(ts);
    m_structs.push(tle);
    return ts;
}

// Get the (singleton) texture type instance.
Type_texture *Type_factory::get_texture(Type_texture::Texture_kind texture_kind)
{
    return m_builder.create<Type_texture>(texture_kind);
}

// Get the (singleton) bsdf_measurement type instance.
Type_bsdf_measurement *Type_factory::get_bsdf_measurement()
{
    return &mdltlc_bsdf_measurement_type;
}

// Get the (singleton) material_emission type instance.
Type_material_emission *Type_factory::get_material_emission()
{
    return &mdltlc_material_emission_type;
}

// Get the (singleton) material_surface type instance.
Type_material_surface *Type_factory::get_material_surface()
{
    return &mdltlc_material_surface_type;
}

// Get the (singleton) material_geometry type instance.
Type_material_geometry *Type_factory::get_material_geometry()
{
    return &mdltlc_material_geometry_type;
}

// Get the (singleton) material_volume type instance.
Type_material_volume *Type_factory::get_material_volume()
{
    return &mdltlc_material_volume_type;
}

// Get the (singleton) material type instance.
Type_material *Type_factory::get_material()
{
    return &mdltlc_material_type;
}

// Get the a fresh type variable.
Type_var *Type_factory::create_type_variable()
{
    m_next_var_index++;
    return m_builder.create<Type_var>(m_next_var_index);
}

/// Get a fresh function type.
Type_function *Type_factory::create_function(Type *return_type) {
    return m_builder.create<Type_function>(return_type);
}

/// Create a fresh parameter wrapper for the given type.
Type_list_elem *Type_factory::create_type_list_elem(Type *type) {
    return m_builder.create<Type_list_elem>(type);
}

/// Create a fresh struct field wrapper for the given name and type.
Named_type_list_elem *Type_factory::create_named_type_list_elem(Symbol *name, Type *type) {
    return m_builder.create<Named_type_list_elem>(name, type);
}

/// Create a new type that matches the given MDL type.
Type *Type_factory::import_type(mi::mdl::IType const *mdl_type) {
    switch (mdl_type->get_kind()) {
    case mi::mdl::IType::Kind::TK_ALIAS:
        return import_type(mdl_type->skip_type_alias());

    case mi::mdl::IType::Kind::TK_BOOL:
        return get_bool();

    case mi::mdl::IType::Kind::TK_INT:
        return get_int();

    case mi::mdl::IType::Kind::TK_ENUM:
    {
        mi::mdl::IType_enum const *te = as<mi::mdl::IType_enum>(mdl_type);
        Symbol *name = m_symtab.get_symbol(te->get_symbol()->get_name());

        Type_enum *enum_t = create_enum(name);
        for (size_t i = 0; i < te->get_value_count(); i++) {
            mi::mdl::IType_enum::Value const *e_value = te->get_value(i);
            mi::mdl::ISymbol const *param_name = e_value->get_symbol();
            int param_code = e_value->get_code();

            Symbol *n = m_symtab.get_symbol(param_name->get_name());
            Enum_variant_list_elem *field = m_builder.create<Enum_variant_list_elem>(n, param_code);
            enum_t->add_variant(field);

        }
        return enum_t;
    }

    case mi::mdl::IType::Kind::TK_FLOAT:
        return get_float();

    case mi::mdl::IType::Kind::TK_DOUBLE:
        return get_double();

    case mi::mdl::IType::Kind::TK_STRING:
        return get_string();

    case mi::mdl::IType::Kind::TK_LIGHT_PROFILE:
        return get_light_profile();

    case mi::mdl::IType::Kind::TK_BSDF:
        return get_bsdf();

    case mi::mdl::IType::Kind::TK_HAIR_BSDF:
        return get_hair_bsdf();

    case mi::mdl::IType::Kind::TK_EDF:
        return get_edf();

    case mi::mdl::IType::Kind::TK_VDF:
        return get_vdf();

    case mi::mdl::IType::Kind::TK_VECTOR:
    {
        mi::mdl::IType_vector const *tf = as<mi::mdl::IType_vector>(mdl_type);
        Type *element_type = import_type(tf->get_element_type());
        return get_vector(tf->get_size(), element_type);
    }

    case mi::mdl::IType::Kind::TK_MATRIX:
    {
        mi::mdl::IType_matrix const *tf = as<mi::mdl::IType_matrix>(mdl_type);
        Type *element_type = import_type(tf->get_element_type());
        return get_matrix(unsigned(tf->get_columns()), element_type);
    }

    // TODO: Handle deferred-size arrays.
    case mi::mdl::IType::Kind::TK_ARRAY:
    {
        mi::mdl::IType_array const *tf = as<mi::mdl::IType_array>(mdl_type);
        Type *element_type = import_type(tf->get_element_type());
        return get_array(element_type);
    }

    case mi::mdl::IType::Kind::TK_COLOR:
        return get_color();

    case mi::mdl::IType::Kind::TK_FUNCTION: {
        mi::mdl::IType_function const *tf = as<mi::mdl::IType_function>(mdl_type);
        Type *return_type = import_type(tf->get_return_type());
        Type_function *function_type = create_function(return_type);
        for (int i = 0; i < tf->get_parameter_count(); i++) {
            mi::mdl::IType const *param_type;
            mi::mdl::ISymbol const *param_name;

            tf->get_parameter(i, param_type, param_name);

            Type *t = import_type(param_type);
            Type_list_elem *parameter = m_builder.create<Type_list_elem>(t);
            function_type->add_parameter(parameter);
        }
        return function_type;
    }

    case mi::mdl::IType::Kind::TK_STRUCT:
    {
        mi::mdl::IType_struct const *ts = as<mi::mdl::IType_struct>(mdl_type);
        Symbol *name = m_symtab.get_symbol(ts->get_symbol()->get_name());

        // Special case to make the builtin "material" type and
        // "struct material" the same in the mdltl compiler.
        //
        if (name == m_symtab.get_symbol("material"))
            return get_material();

        Type_struct *struct_t = create_struct(name);
        for (size_t i = 0; i < ts->get_field_count(); i++) {
            mi::mdl::IType_struct::Field const *s_field = ts->get_field(i);
            mi::mdl::IType const *param_type = s_field->get_type();
            mi::mdl::ISymbol const *param_name = s_field->get_symbol();

            Symbol *n = m_symtab.get_symbol(param_name->get_name());
            Type *t = import_type(param_type);
            Named_type_list_elem *field = m_builder.create<Named_type_list_elem>(n, t);
            struct_t->add_field(field);

        }
        return struct_t;
    }

    case mi::mdl::IType::Kind::TK_TEXTURE:
    {
        mi::mdl::IType_texture const *tf = as<mi::mdl::IType_texture>(mdl_type);
        mi::mdl::IType_texture::Shape s = tf->get_shape();
        Type_texture::Texture_kind tk = Type_texture::Texture_kind::TK_2D;
        switch (s) {
        case mi::mdl::IType_texture::Shape::TS_2D:
            tk = Type_texture::Texture_kind::TK_2D;
            break;
        case mi::mdl::IType_texture::Shape::TS_3D:
            tk = Type_texture::Texture_kind::TK_3D;
            break;
        case mi::mdl::IType_texture::Shape::TS_CUBE:
            tk = Type_texture::Texture_kind::TK_CUBE;
            break;
        case mi::mdl::IType_texture::Shape::TS_PTEX:
            tk = Type_texture::Texture_kind::TK_PTEX;
            break;
        case mi::mdl::IType_texture::Shape::TS_BSDF_DATA:
            tk = Type_texture::Texture_kind::TK_BSDF_DATA;
            break;
        }
        return get_texture(tk);
    }

    case mi::mdl::IType::Kind::TK_BSDF_MEASUREMENT:
        return get_bsdf_measurement();

    case mi::mdl::IType::Kind::TK_ERROR:
        return get_error();

    default:
        printf("[error] unhandled MDL type: %d\n", mdl_type->get_kind());
        return get_error();
    }
}

/// Return true if both types are representing the same type.
bool Type_factory::types_equal(Type *type1, Type *type2) {
    if (type1 == type2)
        return true;

    if (type1->get_kind() != type2->get_kind())
        return false;

    switch (type1->get_kind()) {
    case Type::Kind::TK_BOOL:
    case Type::Kind::TK_INT:
    case Type::Kind::TK_FLOAT:
    case Type::Kind::TK_DOUBLE:
    case Type::Kind::TK_STRING:
    case Type::Kind::TK_LIGHT_PROFILE:
    case Type::Kind::TK_BSDF:
    case Type::Kind::TK_HAIR_BSDF:
    case Type::Kind::TK_EDF:
    case Type::Kind::TK_VDF:
    case Type::Kind::TK_VECTOR:
    case Type::Kind::TK_MATRIX:
    case Type::Kind::TK_ARRAY:
    case Type::Kind::TK_COLOR:
    case Type::Kind::TK_TEXTURE:
    case Type::Kind::TK_BSDF_MEASUREMENT:
    case Type::Kind::TK_MATERIAL_EMISSION:
    case Type::Kind::TK_MATERIAL_SURFACE:
    case Type::Kind::TK_MATERIAL_GEOMETRY:
    case Type::Kind::TK_MATERIAL_VOLUME:
    case Type::Kind::TK_MATERIAL:
        return true;

    case Type::Kind::TK_ENUM:
    {
        // FIXME: Handle qualified names.
        Type_enum *te1 = cast<Type_enum>(type1);
        Type_enum *te2 = cast<Type_enum>(type2);
        return te1->get_name() == te2->get_name();
    }

    case Type::Kind::TK_STRUCT:
    {
        // FIXME: Handle qualified names.
        Type_struct *te1 = cast<Type_struct>(type1);
        Type_struct *te2 = cast<Type_struct>(type2);
        return te1->get_name() == te2->get_name();
    }

    case Type::Kind::TK_FUNCTION:
    {
        Type_function *tf1 = cast<Type_function>(type1);
        Type_function *tf2 = cast<Type_function>(type2);

        if (tf1->get_parameter_count() != tf2->get_parameter_count())
            return false;

        for (int i = 0; i < tf1->get_parameter_count(); i++) {
            Type *p1 = tf1->get_parameter_type(i);
            Type *p2 = tf2->get_parameter_type(i);
            if (!types_equal(p1, p2))
                return false;
        }
        return true;
    }
    case Type::Kind::TK_VAR:
    {
        Type_var *tv1 = cast<Type_var>(type1);
        Type_var *tv2 = cast<Type_var>(type2);
        return tv1->get_index() == tv2->get_index();
    }
    case Type::Kind::TK_ERROR:
        return false;
    }
    return false;
}

/// Return true if the first type (as an actual parameter) matches
/// the second (as a formal parameter).
bool Type_factory::types_match(Type *type1, Type *type2) {
    type1 = deref(type1);
    type2 = deref(type2);

    // Bound variables are skipped by above calls to deref, so all
    // remaining vars arriving here are unbound.
    if (is<Type_var>(type1) || is<Type_var>(type2))
        return true;

    return types_equal(type1, type2);
}
