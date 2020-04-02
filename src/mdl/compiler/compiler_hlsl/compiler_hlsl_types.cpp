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

#include <cstring>
#include <cstdio>

#include "mdl/compiler/compilercore/compilercore_memory_arena.h"
#include "compiler_hlsl_symbols.h"
#include "compiler_hlsl_types.h"

#if defined(BIT64) || defined(MACOSX)
#ifdef WIN_NT
#  define FMT_SIZE_T        "llu"
#else
#  define FMT_SIZE_T        "zu"
#endif
#else
#  define FMT_SIZE_T        "u"
#endif

namespace mi {
namespace mdl {
namespace hlsl {

namespace {

// Create a copy of sub entities.
template<typename T>
T *copy_sub_entities(
    Memory_arena       &arena,
    Array_ref<T> const &entities)
{
    size_t size = sizeof(T) * entities.size();
    T      *res = reinterpret_cast<T *>(arena.allocate(size));

    for (size_t i = 0, n = entities.size(); i < n; ++i) {
        res[i] = entities[i];
    }
    return res;
}

/// Find the predefined symbol ID for an texture shape.
///
/// \param shape  the texture shape
Symbol::Predefined_id get_texture_id(Texture_shape shape)
{
    switch (shape) {
    case mi::mdl::hlsl::SHAPE_UNKNOWN:
        return Symbol::SYM_TYPE_TEXTURE;
    case mi::mdl::hlsl::SHAPE_1D:
        return Symbol::SYM_TYPE_TEXTURE1D;
    case mi::mdl::hlsl::SHAPE_2D:
        return Symbol::SYM_TYPE_TEXTURE2D;
    case mi::mdl::hlsl::SHAPE_3D:
        return Symbol::SYM_TYPE_TEXTURE3D;
    case mi::mdl::hlsl::SHAPE_CUBE:
        return Symbol::SYM_TYPE_TEXTURECUBE;

    case mi::mdl::hlsl::SHAPE_1D_ARRAY:
        return Symbol::SYM_TYPE_TEXTURE1DARRAY;
    case mi::mdl::hlsl::SHAPE_2D_ARRAY:
        return Symbol::SYM_TYPE_TEXTURE2DARRAY;
    }
    // unsupported if we are here
    return Symbol::SYM_ERROR;
}

// create the builtin types
#define BUILTIN_TYPE(type, name, args) type name args;

#include "compiler_hlsl_builtin_types.h"

}  // anonymous


// ---------------------------- Base type ----------------------------

// Constructor.
Type::Type(Symbol *name)
: m_sym(name)
{
}

// Access to the base type.
Type *Type::skip_type_alias()
{ 
    return this;
}

// Get the type modifiers of a type
Type::Modifiers Type::get_type_modifiers()
{
    return MK_NONE;
}

// ---------------------------- Alias type ----------------------------

// Constructor.
Type_alias::Type_alias(
    Type       *aliased,
    Modifiers  modifiers,
    Symbol     *sym)
: Base(sym)
, m_aliased_type(aliased)
, m_modifiers(modifiers)
{
}

// Get the type kind.
Type::Kind Type_alias::get_kind()
{
    return s_kind;
}

// Access to the base type.
Type *Type_alias::skip_type_alias()
{
    return m_aliased_type->skip_type_alias();
}

// Get the modifier set of this type.
Type::Modifiers Type_alias::get_type_modifiers()
{
    return m_modifiers;
}

// ---------------------------- Error type ----------------------------

// Constructor.
Type_error::Type_error()
: Base(Symbol_table::get_error_symbol())
{
}

// Get the type kind.
Type::Kind Type_error::get_kind()
{
    return s_kind;
}

// ---------------------------- Atomic type ----------------------------

// Constructor.
Type_scalar::Type_scalar(Symbol *sym)
: Base(sym)
{
}

// ---------------------------- void type ----------------------------

// Constructor.
Type_void::Type_void()
: Base(Symbol_table::get_predefined_symbol(Symbol::SYM_TYPE_VOID))
{
}

// Get the type kind.
Type::Kind Type_void::get_kind()
{
    return s_kind;
}

// ---------------------------- bool type ----------------------------

// Constructor.
Type_bool::Type_bool()
: Base(Symbol_table::get_predefined_symbol(Symbol::SYM_TYPE_BOOL))
{
}

// Get the type kind.
Type::Kind Type_bool::get_kind()
{
    return s_kind;
}

// ---------------------------- two complement type ----------------------------

// Constructor.
Type_two_complement::Type_two_complement(
    Symbol *sym,
    size_t bitsize)
: Base(sym)
, m_bitsize(bitsize)
{
}

// ---------------------------- signed two complement type ----------------------------

// Constructor.
Type_signed_int::Type_signed_int(
    Symbol *sym,
    size_t bitsize)
: Base(sym, bitsize)
{
}

// ---------------------------- unsigned two complement type ----------------------------

// Constructor.
Type_unsigned_int::Type_unsigned_int(
    Symbol *sym,
    size_t bitsize)
: Base(sym, bitsize)
{
}

// ---------------------------- int type ----------------------------

// Constructor.
Type_int::Type_int()
: Base(Symbol_table::get_predefined_symbol(Symbol::SYM_TYPE_INT), 32)
{
}

// Get the type kind.
Type::Kind Type_int::get_kind()
{
    return s_kind;
}

// ---------------------------- uint type ----------------------------

// Constructor.
Type_uint::Type_uint()
: Base(Symbol_table::get_predefined_symbol(Symbol::SYM_TYPE_UINT), 32)
{
}

// Get the type kind.
Type::Kind Type_uint::get_kind()
{
    return s_kind;
}

// ---------------------------- Type_min12int type ----------------------------

// Constructor.
Type_min12int::Type_min12int()
: Base(Symbol_table::get_predefined_symbol(Symbol::SYM_TYPE_MIN12INT), 12)
{
}

// Get the type kind.
Type::Kind Type_min12int::get_kind()
{
    return s_kind;
}

// ---------------------------- Type_min16int type ----------------------------

// Constructor.
Type_min16int::Type_min16int()
: Base(Symbol_table::get_predefined_symbol(Symbol::SYM_TYPE_MIN16INT), 16)
{
}

// Get the type kind.
Type::Kind Type_min16int::get_kind()
{
    return s_kind;
}

// ---------------------------- Type_min16uint type ----------------------------

// Constructor.
Type_min16uint::Type_min16uint()
: Base(Symbol_table::get_predefined_symbol(Symbol::SYM_TYPE_MIN16UINT), 16)
{
}

// Get the type kind.
Type::Kind Type_min16uint::get_kind()
{
    return s_kind;
}

// ---------------------------- half type ----------------------------

// Constructor.
Type_half::Type_half()
: Base(Symbol_table::get_predefined_symbol(Symbol::SYM_TYPE_HALF))
{
}

// Get the type kind.
Type::Kind Type_half::get_kind()
{
    return s_kind;
}

// ---------------------------- float type ----------------------------

// Constructor.
Type_float::Type_float()
: Base(Symbol_table::get_predefined_symbol(Symbol::SYM_TYPE_FLOAT))
{
}

// Get the type kind.
Type::Kind Type_float::get_kind()
{
    return s_kind;
}

// ---------------------------- double type ----------------------------

// Constructor.
Type_double::Type_double()
: Base(Symbol_table::get_predefined_symbol(Symbol::SYM_TYPE_DOUBLE))
{
}

// Get the type kind.
Type::Kind Type_double::get_kind()
{
    return s_kind;
}

// ---------------------------- min10float type ----------------------------

// Constructor.
Type_min10float::Type_min10float()
: Base(Symbol_table::get_predefined_symbol(Symbol::SYM_TYPE_MIN10FLOAT))
{
}

// Get the type kind.
Type::Kind Type_min10float::get_kind()
{
    return s_kind;
}

// ---------------------------- min16float type ----------------------------

// Constructor.
Type_min16float::Type_min16float()
: Base(Symbol_table::get_predefined_symbol(Symbol::SYM_TYPE_MIN16FLOAT))
{
}

// Get the type kind.
Type::Kind Type_min16float::get_kind()
{
    return s_kind;
}

// ---------------------------- compound type ----------------------------

// Constructor.
Type_compound::Type_compound(Symbol *sym)
: Base(sym)
{
}

// ---------------------------- vector type ----------------------------

// Constructor.
Type_vector::Type_vector(Type_scalar *base, size_t size)
: Base(get_vector_sym(base, size))
, m_base(base)
, m_size(size)
{
    HLSL_ASSERT(1 <= size && size <= 4);
}

// Get the symbol for given type.
Symbol *Type_vector::get_vector_sym(
    Type_scalar *base,
    size_t      size)
{
    Symbol::Predefined_id id = Symbol::SYM_ERROR;

    switch (base->get_kind()) {
    case Type::TK_BOOL:
        id = Symbol::SYM_TYPE_BOOL;
        break;
    case Type::TK_INT:
        id = Symbol::SYM_TYPE_INT;
        break;
    case Type::TK_UINT:
        id = Symbol::SYM_TYPE_UINT;
        break;
    case Type::TK_HALF:
        id = Symbol::SYM_TYPE_HALF;
        break;
    case Type::TK_FLOAT:
        id = Symbol::SYM_TYPE_FLOAT;
        break;
    case Type::TK_DOUBLE:
        id = Symbol::SYM_TYPE_DOUBLE;
        break;
    case Type::TK_MIN12INT:
        id = Symbol::SYM_TYPE_MIN12INT;
        break;
    case Type::TK_MIN16INT:
        id = Symbol::SYM_TYPE_MIN16INT;
        break;
    case Type::TK_MIN16UINT:
        id = Symbol::SYM_TYPE_MIN16UINT;
        break;
    case Type::TK_MIN10FLOAT:
        id = Symbol::SYM_TYPE_MIN10FLOAT;
        break;
    case Type::TK_MIN16FLOAT:
        id = Symbol::SYM_TYPE_MIN16FLOAT;
        break;
    default:
        break;
    }

    if (id != Symbol::SYM_ERROR)
        id = Symbol::Predefined_id(id + size);

    return Symbol_table::get_predefined_symbol(id);
}

// Get the type kind.
Type::Kind Type_vector::get_kind()
{
    return s_kind;
}

// Get the compound type at index.
Type *Type_vector::get_compound_type(size_t index)
{
    return index == 0 ? m_base : NULL;
}

// Get the number of compound elements.
size_t Type_vector::get_compound_size()
{
    return 1;
}

// ---------------------------- matrix type ----------------------------

// Constructor.
Type_matrix::Type_matrix(
    Type_vector *column_type,
    size_t      n_columns)
: Base(get_matrix_sym(column_type, n_columns))
, m_column_type(column_type)
, m_columns(n_columns)
{
    HLSL_ASSERT(1 <= n_columns && n_columns <= 4);
}

// Get the symbol for given type.
Symbol *Type_matrix::get_matrix_sym(
    Type_vector *column_type,
    size_t      columns)
{
    Type_scalar *a_type = column_type->get_element_type();
    size_t      rows    = column_type->get_size();

    Symbol::Predefined_id id = Symbol::SYM_ERROR;
    switch (a_type->get_kind()) {
    case Type::TK_BOOL:
        id = Symbol::SYM_TYPE_BOOL1X1;
        break;
    case Type::TK_INT:
        id = Symbol::SYM_TYPE_INT1X1;
        break;
    case Type::TK_UINT:
        id = Symbol::SYM_TYPE_UINT1X1;
        break;
    case Type::TK_HALF:
        id = Symbol::SYM_TYPE_HALF1X1;
        break;
    case Type::TK_FLOAT:
        id = Symbol::SYM_TYPE_FLOAT1X1;
        break;
    case Type::TK_DOUBLE:
        id = Symbol::SYM_TYPE_DOUBLE1X1;
        break;
    case Type::TK_MIN12INT:
        id = Symbol::SYM_TYPE_MIN12INT1X1;
        break;
    case Type::TK_MIN16INT:
        id = Symbol::SYM_TYPE_MIN16INT1X1;
        break;
    case Type::TK_MIN16UINT:
        id = Symbol::SYM_TYPE_MIN16UINT1X1;
        break;
    case Type::TK_MIN10FLOAT:
        id = Symbol::SYM_TYPE_MIN10FLOAT1X1;
        break;
    case Type::TK_MIN16FLOAT:
        id = Symbol::SYM_TYPE_MIN16FLOAT1X1;
        break;
    default:
        break;
    }

    if (id != Symbol::SYM_ERROR)
        id = Symbol::Predefined_id(id + (columns - 1) * 4 + (rows - 1));
    return Symbol_table::get_predefined_symbol(id);
}

// Get the type kind.
Type::Kind Type_matrix::get_kind()
{
    return s_kind;
}

// Get the compound type at index.
Type *Type_matrix::get_compound_type(size_t index)
{
    return index == 0 ? m_column_type : NULL;
}

// Get the number of compound elements.
size_t Type_matrix::get_compound_size()
{
    return 1;
}

// ---------------------------- array type ----------------------------

// Constructor.
Type_array::Type_array(
    Type   *element_type,
    size_t size,
    Symbol *sym)
: Base(sym)
, m_element_type(element_type)
, m_size(size)
{
}

// Get the type kind.
Type::Kind Type_array::get_kind()
{
    return s_kind;
}

// Get the compound type at index.
Type *Type_array::get_compound_type(size_t index)
{
    return index == 0 ? m_element_type : NULL;
}

// Get the number of compound elements.
size_t Type_array::get_compound_size()
{
    return 1;
}

// Get the modifier set of array type.
Type::Modifiers Type_array::get_type_modifiers()
{
    // always return the type modifiers of the element type
    return m_element_type->get_type_modifiers();
}

// ---------------------------- struct type ----------------------------

// Constructor.
Type_struct::Type_struct(
    Memory_arena           *arena,
    Array_ref<Field> const &fields,
    Symbol                 *sym)
: Base(sym)
, m_fields(copy_sub_entities(*arena, fields))
, m_n_fields(fields.size())
{
}

// Get the type kind.
Type::Kind Type_struct::get_kind()
{
    return s_kind;
}

// Get the compound type at index.
Type *Type_struct::get_compound_type(size_t index)
{
    return index < m_n_fields ? m_fields[index].get_type() : NULL;
}

// Get the number of compound elements.
size_t Type_struct::get_compound_size()
{
    return m_n_fields;
}

// Get a field.
Type_struct::Field *Type_struct::get_field(size_t index)
{
    if (index < m_n_fields)
        return &m_fields[index];
    HLSL_ASSERT(!"index out of range");
    return NULL;
}

// ---------------------------- function type ----------------------------

// Constructor.
Type_function::Type_function(
    Memory_arena               *arena,
    Type                       *ret_type,
    Array_ref<Parameter> const &params)
: Base(NULL)  // FIXME: all functions types share the same name '*'
, m_ret_type(ret_type)
, m_params(copy_sub_entities(*arena, params))
, m_n_params(params.size())
{
    HLSL_ASSERT(ret_type != NULL);
}

// Get the type kind.
Type::Kind Type_function::get_kind()
{
    return s_kind;
}

// Get a parameter of the function type.
Type_function::Parameter *Type_function::get_parameter(size_t index)
{
    if (index < m_n_params)
        return &m_params[index];
    HLSL_ASSERT(!"index out of range");
    return NULL;
}

// ---------------------------- texture type ----------------------------

// Constructor.
Type_texture::Type_texture(
    Texture_shape shape)
: Base(Symbol_table::get_predefined_symbol(get_texture_id(shape)))
, m_shape(shape)
{
}

// Get the type kind.
Type::Kind Type_texture::get_kind()
{
    return s_kind;
}

// ---------------------------- type factory ----------------------------

// Constructor.
Type_factory::Type_factory(
    Memory_arena  &arena,
    Symbol_table  &symtab)
: m_builder(arena)
, m_symtab(symtab)
, m_type_cache(0, Type_cache::hasher(), Type_cache::key_equal(), &arena)
{
}

// Get a new type alias instance.
Type *Type_factory::get_alias(
    Type            *type,
    Type::Modifiers modifiers)
{
    // an alias of the error type is still the error type
    if (is<Type_error>(type))
        return type;

    Type::Modifiers old_modifiers = type->get_type_modifiers();

    if ((old_modifiers & modifiers) == modifiers) {
        // original type has already all requested modifiers)
        return type;
    }

    modifiers |= old_modifiers;

    // cannot set modifiers on a array type, set them on its element type
    if (Type_array *arr_type = as<Type_array>(type)) {
        Type *elem_type = arr_type->get_element_type();

        elem_type = get_alias(elem_type, modifiers);

        return get_array(elem_type, arr_type->get_size());
    }

    type = type->skip_type_alias();

    Type_cache_key key(type, modifiers);

    Type_cache::const_iterator it = m_type_cache.find(key);
    if (it == m_type_cache.end()) {
        string new_name(type->get_sym()->get_name(), m_builder.get_arena()->get_allocator());

        Symbol *sym = m_symtab.get_user_type_symbol(new_name.c_str());

        Type *alias_type = m_builder.create<Type_alias>(type, modifiers, sym);

        it = m_type_cache.insert(Type_cache::value_type(key, alias_type)).first;
    }
    return it->second;
}

// Get the (singleton) error type instance.
Type_error *Type_factory::get_error()
{
    return &hlsl_error_type;
}

// Get the (singleton) void type instance.
Type_void *Type_factory::get_void()
{
    return &hlsl_void_type;
}

// Get the (singleton) bool type instance.
Type_bool *Type_factory::get_bool()
{
    return &hlsl_bool_type;
}

// Get the (singleton) int type instance.
Type_int *Type_factory::get_int()
{
    return &hlsl_int_type;
}

// Get the (singleton) uint type instance.
Type_uint *Type_factory::get_uint()
{
    return &hlsl_uint_type;
}

// Get the (singleton) half type instance.
Type_half *Type_factory::get_half()
{
    return &hlsl_half_type;
}

// Get the (singleton) float type instance.
Type_float *Type_factory::get_float()
{
    return &hlsl_float_type;
}

// Get the (singleton) double type instance.
Type_double *Type_factory::get_double()
{
    return &hlsl_double_type;
}

// Get the (singleton) min12int type instance.
Type_min12int *Type_factory::get_min12int()
{
    return &hlsl_min12int_type;
}

// Get the (singleton) min16int type instance.
Type_min16int *Type_factory::get_min16int()
{
    return &hlsl_min16int_type;
}

// Get the (singleton) min16uint type instance.
Type_min16uint *Type_factory::get_min16uint()
{
    return &hlsl_min16uint_type;
}

// Get the (singleton) min10float type instance.
Type_min10float *Type_factory::get_min10float()
{
    return &hlsl_min10float_type;
}

// Get the (singleton) min16float type instance.
Type_min16float *Type_factory::get_min16float()
{
    return &hlsl_min16float_type;
}

// Get a vector type 1instance.
Type_vector *Type_factory::get_vector(
    Type_scalar *element_type,
    size_t      size)
{
#define CASE(N) \
    case N: \
        switch (element_type->get_kind()) { \
        case Type::TK_BOOL: \
            return &hlsl_bool ## N ## _type; \
        case Type::TK_INT: \
            return &hlsl_int ## N ## _type; \
        case Type::TK_UINT: \
            return &hlsl_uint ## N ## _type; \
        case Type::TK_HALF: \
            return &hlsl_half ## N ## _type; \
        case Type::TK_FLOAT: \
            return &hlsl_float ## N ## _type; \
        case Type::TK_DOUBLE: \
            return &hlsl_double ## N ## _type; \
        case Type::TK_MIN12INT: \
            return &hlsl_min12int ## N ## _type; \
        case Type::TK_MIN16INT: \
            return &hlsl_min16int ## N ## _type; \
        case Type::TK_MIN16UINT: \
            return &hlsl_min16uint ## N ## _type; \
        case Type::TK_MIN10FLOAT: \
            return &hlsl_min10float ## N ## _type; \
        case Type::TK_MIN16FLOAT: \
            return &hlsl_min16float ## N ## _type; \
        default: \
            break; \
        } \
        break


    switch (size) {
    CASE(1);
    CASE(2);
    CASE(3);
    CASE(4);
    }
    // unsupported
    return NULL;

#undef CASE
}

// Get a matrix type instance.
Type_matrix *Type_factory::get_matrix(
    Type_vector *element_type,
    size_t      columns)
{
    Type::Kind kind = element_type->get_element_type()->get_kind();
    size_t     rows = element_type->get_size();

#define CASE(COL, ROW) \
    case ROW: \
        switch (kind) { \
            case Type::TK_BOOL: \
                return &hlsl_bool ## ROW ## x ## COL ## _type; \
            case Type::TK_INT: \
                return &hlsl_int ## ROW ## x ## COL ## _type; \
            case Type::TK_UINT: \
                return &hlsl_uint ## ROW ## x ## COL ## _type; \
            case Type::TK_HALF: \
                return &hlsl_half ## ROW ## x ## COL ## _type; \
            case Type::TK_FLOAT: \
                return &hlsl_float ## ROW ## x ## COL ## _type; \
            case Type::TK_DOUBLE: \
                return &hlsl_double ## ROW ## x ## COL ## _type; \
            case Type::TK_MIN12INT: \
                return &hlsl_min12int ## ROW ## x ## COL ## _type; \
            case Type::TK_MIN16INT: \
                return &hlsl_min16int ## ROW ## x ## COL ## _type; \
            case Type::TK_MIN16UINT: \
                return &hlsl_min16uint ## ROW ## x ## COL ## _type; \
            case Type::TK_MIN10FLOAT: \
                return &hlsl_min10float ## ROW ## x ## COL ## _type; \
            case Type::TK_MIN16FLOAT: \
                return &hlsl_min16float ## ROW ## x ## COL ## _type; \
            default: \
                break; \
            } \
        break

    switch (columns) {
    case 1:
        switch (rows) {
        CASE(1, 1);
        CASE(1, 2);
        CASE(1, 3);
        CASE(1, 4);
        }
        break;
    case 2:
        switch (rows) {
        CASE(2, 1);
        CASE(2, 2);
        CASE(2, 3);
        CASE(2, 4);
        }
        break;
    case 3:
        switch (rows) {
        CASE(3, 1);
        CASE(3, 2);
        CASE(3, 3);
        CASE(3, 4);
        }
        break;
    case 4:
        switch (rows) {
        CASE(4, 1);
        CASE(4, 2);
        CASE(4, 3);
        CASE(4, 4);
        }
        break;
    }
    // unsupported
    HLSL_ASSERT(!"cannot retrieve unsupported matrix type");
    return NULL;

#undef CASE
}

// Get an array type instance.
Type *Type_factory::get_array(
    Type   *element_type,
    size_t size)
{
    if (is<Type_error>(element_type)) {
        // cannot create an array of error type
        return element_type;
    }

    HLSL_ASSERT(size > 0 && "array size must be > 0");

    Type_cache_key key(size, element_type);

    Type_cache::const_iterator it = m_type_cache.find(key);
    if (it == m_type_cache.end()) {
        IAllocator *alloc = m_builder.get_arena()->get_allocator();
        string name(element_type->get_sym()->get_name(), alloc);
        
        name += '[';
        char buffer[16];
        snprintf(buffer, sizeof(buffer), "%" FMT_SIZE_T, size);
        buffer[sizeof(buffer) - 1] = '\0';
        name += buffer;
        name += ']';

        Symbol *sym = m_symtab.get_user_type_symbol(name.c_str());
        Type *array_type = m_builder.create<Type_array>(element_type, size, sym);

        it = m_type_cache.insert(Type_cache::value_type(key, array_type)).first;
    }
    return it->second;
}

// Get an unsized array type instance.
Type *Type_factory::get_unsized_array(
    Type *element_type)
{
    if (is<Type_error>(element_type)) {
        // cannot create an array of error type
        return element_type;
    }

    Type_cache_key key(0, element_type);

    Type_cache::const_iterator it = m_type_cache.find(key);
    if (it == m_type_cache.end()) {
        IAllocator *alloc = m_builder.get_arena()->get_allocator();
        string name(element_type->get_sym()->get_name(), alloc);

        name += '[';
        name += ']';

        Symbol *sym = m_symtab.get_symbol(name.c_str());
        Type *array_type = m_builder.create<Type_array>(element_type, 0, sym);

        it = m_type_cache.insert(Type_cache::value_type(key, array_type)).first;
    }
    return it->second;
}

// Get a function type instance.
Type_function *Type_factory::get_function(
    Type                                      *return_type,
    Array_ref<Type_function::Parameter> const &parameters)
{
    Type_cache_key key(return_type, parameters.data(), parameters.size());

    Type_cache::const_iterator it = m_type_cache.find(key);
    if (it == m_type_cache.end()) {
        Type_function *fun_type = m_builder.create<Type_function>(
            m_builder.get_arena(),
            return_type,
            parameters);

        it = m_type_cache.insert(Type_cache::value_type(fun_type, fun_type)).first;
    }
    return cast<Type_function>(it->second);
}

// Get a struct type instance.
Type_struct *Type_factory::get_struct(
    Array_ref<Type_struct::Field> const &fields,
    Symbol                              *sym)
{
    // we assume here that the name is unique so it is not necessary to
    // ensure the uniqueness by caching for the type
    return m_builder.create<Type_struct>(m_builder.get_arena(), fields, sym);
}

// Get an texture type
Type_texture *Type_factory::get_texture(
    Texture_shape shape)
{
    switch (shape) {
    case SHAPE_UNKNOWN:
        return &hlsl_texture_type;
    case SHAPE_1D:
        return &hlsl_texture1D_type;
    case SHAPE_2D:
        return &hlsl_texture2D_type;
    case SHAPE_3D:
        return &hlsl_texture3D_type;
    case SHAPE_CUBE:
        return &hlsl_textureCube_type;
    case SHAPE_1D_ARRAY:
        return &hlsl_texture1DArray_type;
    case SHAPE_2D_ARRAY:
        return &hlsl_texture2DArray_type;
    }
    HLSL_ASSERT(!"unsupported texture shape");
    return NULL;
}


}  // hlsl
}  // mdl
}  // mi
