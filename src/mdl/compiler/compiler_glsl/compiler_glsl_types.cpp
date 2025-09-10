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

#include "pch.h"

#include <cstring>
#include <cstdio>

#include "mdl/compiler/compilercore/compilercore_memory_arena.h"
#include "compiler_glsl_symbols.h"
#include "compiler_glsl_types.h"

#ifdef WIN_NT
#  define FMT_SIZE_T        "llu"
#else
#  define FMT_SIZE_T        "zu"
#endif

namespace mi {
namespace mdl {
namespace glsl {

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

/// Find the predefined symbol ID for a sampler type.
///
/// \param base   the base type
/// \param shape  the texture shape
Symbol::Predefined_id get_sampler_id(Type_scalar *base, Texture_shape shape)
{
    switch (base->get_kind()) {
    case Type::TK_FLOAT:
        switch (shape) {
        case mi::mdl::glsl::SHAPE_1D:
            return Symbol::SYM_TYPE_SAMPLER1D;
        case mi::mdl::glsl::SHAPE_2D:
            return Symbol::SYM_TYPE_SAMPLER2D;
        case mi::mdl::glsl::SHAPE_3D:
            return Symbol::SYM_TYPE_SAMPLER3D;
        case mi::mdl::glsl::SHAPE_CUBE:
            return Symbol::SYM_TYPE_SAMPLERCUBE;
        case mi::mdl::glsl::SHAPE_1D_SHADOW:
            return Symbol::SYM_TYPE_SAMPLER1DSHADOW;
        case mi::mdl::glsl::SHAPE_2D_SHADOW:
            return Symbol::SYM_TYPE_SAMPLER2DSHADOW;
        case mi::mdl::glsl::SHAPE_CUBE_SHADOW:
            return Symbol::SYM_TYPE_SAMPLERCUBESHADOW;
        case mi::mdl::glsl::SHAPE_1D_ARRAY:
            return Symbol::SYM_TYPE_SAMPLER1DARRAY;
        case mi::mdl::glsl::SHAPE_2D_ARRAY:
            return Symbol::SYM_TYPE_SAMPLER2DARRAY;
        case mi::mdl::glsl::SHAPE_1D_ARRAY_SHADOW:
            return Symbol::SYM_TYPE_SAMPLER1DARRAYSHADOW;
        case mi::mdl::glsl::SHAPE_2D_ARRAY_SHADOW:
            return Symbol::SYM_TYPE_SAMPLER2DARRAYSHADOW;
        case mi::mdl::glsl::SHAPE_2D_RECT:
            return Symbol::SYM_TYPE_SAMPLER2DRECT;
        case mi::mdl::glsl::SHAPE_2D_RECT_SHADOW:
            return Symbol::SYM_TYPE_SAMPLER2DRECTSHADOW;
        case mi::mdl::glsl::SHAPE_BUFFER:
            return Symbol::SYM_TYPE_SAMPLERBUFFER;
        case mi::mdl::glsl::SHAPE_2DMS:
            return Symbol::SYM_TYPE_SAMPLER2DMS;
        case mi::mdl::glsl::SHAPE_2DMS_ARRAY:
            return Symbol::SYM_TYPE_SAMPLER2DMSARRAY;
        case mi::mdl::glsl::SHAPE_CUBE_ARRAY:
            return Symbol::SYM_TYPE_SAMPLERCUBEARRAY;
        case mi::mdl::glsl::SHAPE_CUBE_ARRAY_SHADOW:
            return Symbol::SYM_TYPE_SAMPLERCUBEARRAYSHADOW;

        case mi::mdl::glsl::SHAPE_EXTERNAL_OES:
            // GL_OES_EGL_image_external extension
            return Symbol::SYM_TYPE_SAMPLEREXTERNALOES;
            break;
        }
        break;

    case Type::TK_INT:
        switch (shape) {
        case mi::mdl::glsl::SHAPE_1D:
            return Symbol::SYM_TYPE_ISAMPLER1D;
        case mi::mdl::glsl::SHAPE_2D:
            return Symbol::SYM_TYPE_ISAMPLER2D;
        case mi::mdl::glsl::SHAPE_3D:
            return Symbol::SYM_TYPE_ISAMPLER3D;
        case mi::mdl::glsl::SHAPE_CUBE:
            return Symbol::SYM_TYPE_ISAMPLERCUBE;

        case mi::mdl::glsl::SHAPE_1D_SHADOW:
        case mi::mdl::glsl::SHAPE_2D_SHADOW:
        case mi::mdl::glsl::SHAPE_CUBE_SHADOW:
        case mi::mdl::glsl::SHAPE_1D_ARRAY_SHADOW:
        case mi::mdl::glsl::SHAPE_2D_ARRAY_SHADOW:
        case mi::mdl::glsl::SHAPE_2D_RECT_SHADOW:
        case mi::mdl::glsl::SHAPE_CUBE_ARRAY_SHADOW:
            // no shadow sampler on int
            break;

        case mi::mdl::glsl::SHAPE_1D_ARRAY:
            return Symbol::SYM_TYPE_ISAMPLER1DARRAY;
        case mi::mdl::glsl::SHAPE_2D_ARRAY:
            return Symbol::SYM_TYPE_ISAMPLER2DARRAY;
        case mi::mdl::glsl::SHAPE_2D_RECT:
            return Symbol::SYM_TYPE_ISAMPLER2DRECT;
        case mi::mdl::glsl::SHAPE_BUFFER:
            return Symbol::SYM_TYPE_ISAMPLERBUFFER;
        case mi::mdl::glsl::SHAPE_2DMS:
            return Symbol::SYM_TYPE_ISAMPLER2DMS;
        case mi::mdl::glsl::SHAPE_2DMS_ARRAY:
            return Symbol::SYM_TYPE_ISAMPLER2DMSARRAY;
        case mi::mdl::glsl::SHAPE_CUBE_ARRAY:
            return Symbol::SYM_TYPE_ISAMPLERCUBEARRAY;

        case mi::mdl::glsl::SHAPE_EXTERNAL_OES:
            // no external texture sampler on int
            break;
        }
        break;
    case Type::TK_UINT:
        switch (shape) {
        case mi::mdl::glsl::SHAPE_1D:
            return Symbol::SYM_TYPE_USAMPLER1D;
        case mi::mdl::glsl::SHAPE_2D:
            return Symbol::SYM_TYPE_USAMPLER2D;
        case mi::mdl::glsl::SHAPE_3D:
            return Symbol::SYM_TYPE_USAMPLER3D;
        case mi::mdl::glsl::SHAPE_CUBE:
            return Symbol::SYM_TYPE_USAMPLERCUBE;

        case mi::mdl::glsl::SHAPE_1D_SHADOW:
        case mi::mdl::glsl::SHAPE_2D_SHADOW:
        case mi::mdl::glsl::SHAPE_CUBE_SHADOW:
        case mi::mdl::glsl::SHAPE_1D_ARRAY_SHADOW:
        case mi::mdl::glsl::SHAPE_2D_ARRAY_SHADOW:
        case mi::mdl::glsl::SHAPE_2D_RECT_SHADOW:
        case mi::mdl::glsl::SHAPE_CUBE_ARRAY_SHADOW:
            // no shadow sampler on uint
            break;

        case mi::mdl::glsl::SHAPE_1D_ARRAY:
            return Symbol::SYM_TYPE_USAMPLER1DARRAY;
        case mi::mdl::glsl::SHAPE_2D_ARRAY:
            return Symbol::SYM_TYPE_USAMPLER2DARRAY;
        case mi::mdl::glsl::SHAPE_2D_RECT:
            return Symbol::SYM_TYPE_USAMPLER2DRECT;
        case mi::mdl::glsl::SHAPE_BUFFER:
            return Symbol::SYM_TYPE_USAMPLERBUFFER;
        case mi::mdl::glsl::SHAPE_2DMS:
            return Symbol::SYM_TYPE_USAMPLER2DMS;
        case mi::mdl::glsl::SHAPE_2DMS_ARRAY:
            return Symbol::SYM_TYPE_USAMPLER2DMSARRAY;
        case mi::mdl::glsl::SHAPE_CUBE_ARRAY:
            return Symbol::SYM_TYPE_USAMPLERCUBEARRAY;

        case mi::mdl::glsl::SHAPE_EXTERNAL_OES:
            // no external texture sampler on uint
            break;
        }
        break;
    default:
        break;
    }
    // unsupported if we are here
    return Symbol::SYM_ERROR;
}

/// Find the predefined symbol ID for an image type.
///
/// \param base   the base type
/// \param shape  the texture shape
Symbol::Predefined_id get_image_id(Type_scalar *base, Texture_shape shape)
{
    switch (base->get_kind()) {
    case Type::TK_FLOAT:
        switch (shape) {
        case mi::mdl::glsl::SHAPE_1D:
            return Symbol::SYM_TYPE_IMAGE1D;
        case mi::mdl::glsl::SHAPE_2D:
            return Symbol::SYM_TYPE_IMAGE2D;
        case mi::mdl::glsl::SHAPE_3D:
            return Symbol::SYM_TYPE_IMAGE3D;
        case mi::mdl::glsl::SHAPE_CUBE:
            return Symbol::SYM_TYPE_IMAGECUBE;

        case mi::mdl::glsl::SHAPE_1D_SHADOW:
        case mi::mdl::glsl::SHAPE_2D_SHADOW:
        case mi::mdl::glsl::SHAPE_CUBE_SHADOW:
        case mi::mdl::glsl::SHAPE_1D_ARRAY_SHADOW:
        case mi::mdl::glsl::SHAPE_2D_ARRAY_SHADOW:
        case mi::mdl::glsl::SHAPE_2D_RECT_SHADOW:
        case mi::mdl::glsl::SHAPE_CUBE_ARRAY_SHADOW:
            // no shadow image
            break;

        case mi::mdl::glsl::SHAPE_1D_ARRAY:
            return Symbol::SYM_TYPE_IMAGE1DARRAY;
        case mi::mdl::glsl::SHAPE_2D_ARRAY:
            return Symbol::SYM_TYPE_IMAGE2DARRAY;
        case mi::mdl::glsl::SHAPE_2D_RECT:
            return Symbol::SYM_TYPE_IMAGE2DRECT;
        case mi::mdl::glsl::SHAPE_BUFFER:
            return Symbol::SYM_TYPE_IMAGEBUFFER;
        case mi::mdl::glsl::SHAPE_2DMS:
            return Symbol::SYM_TYPE_IMAGE2DMS;
        case mi::mdl::glsl::SHAPE_2DMS_ARRAY:
            return Symbol::SYM_TYPE_IMAGE2DMSARRAY;
        case mi::mdl::glsl::SHAPE_CUBE_ARRAY:
            return Symbol::SYM_TYPE_IMAGECUBEARRAY;

        case mi::mdl::glsl::SHAPE_EXTERNAL_OES:
            // only used for samplers
            break;
        }
        break;

    case Type::TK_INT:
        switch (shape) {
        case mi::mdl::glsl::SHAPE_1D:
            return Symbol::SYM_TYPE_IIMAGE1D;
        case mi::mdl::glsl::SHAPE_2D:
            return Symbol::SYM_TYPE_IIMAGE2D;
        case mi::mdl::glsl::SHAPE_3D:
            return Symbol::SYM_TYPE_IIMAGE3D;
        case mi::mdl::glsl::SHAPE_CUBE:
            return Symbol::SYM_TYPE_IIMAGECUBE;

        case mi::mdl::glsl::SHAPE_1D_SHADOW:
        case mi::mdl::glsl::SHAPE_2D_SHADOW:
        case mi::mdl::glsl::SHAPE_CUBE_SHADOW:
        case mi::mdl::glsl::SHAPE_1D_ARRAY_SHADOW:
        case mi::mdl::glsl::SHAPE_2D_ARRAY_SHADOW:
        case mi::mdl::glsl::SHAPE_2D_RECT_SHADOW:
        case mi::mdl::glsl::SHAPE_CUBE_ARRAY_SHADOW:
            // no shadow image
            break;

        case mi::mdl::glsl::SHAPE_1D_ARRAY:
            return Symbol::SYM_TYPE_IIMAGE1DARRAY;
        case mi::mdl::glsl::SHAPE_2D_ARRAY:
            return Symbol::SYM_TYPE_IIMAGE2DARRAY;
        case mi::mdl::glsl::SHAPE_2D_RECT:
            return Symbol::SYM_TYPE_IIMAGE2DRECT;
        case mi::mdl::glsl::SHAPE_BUFFER:
            return Symbol::SYM_TYPE_IIMAGEBUFFER;
        case mi::mdl::glsl::SHAPE_2DMS:
            return Symbol::SYM_TYPE_IIMAGE2DMS;
        case mi::mdl::glsl::SHAPE_2DMS_ARRAY:
            return Symbol::SYM_TYPE_IIMAGE2DMSARRAY;
        case mi::mdl::glsl::SHAPE_CUBE_ARRAY:
            return Symbol::SYM_TYPE_IIMAGECUBEARRAY;

        case mi::mdl::glsl::SHAPE_EXTERNAL_OES:
            // only used for samplers
            break;
        }
        break;
    case Type::TK_UINT:
        switch (shape) {
        case mi::mdl::glsl::SHAPE_1D:
            return Symbol::SYM_TYPE_UIMAGE1D;
        case mi::mdl::glsl::SHAPE_2D:
            return Symbol::SYM_TYPE_UIMAGE2D;
        case mi::mdl::glsl::SHAPE_3D:
            return Symbol::SYM_TYPE_UIMAGE3D;
        case mi::mdl::glsl::SHAPE_CUBE:
            return Symbol::SYM_TYPE_UIMAGECUBE;

        case mi::mdl::glsl::SHAPE_1D_SHADOW:
        case mi::mdl::glsl::SHAPE_2D_SHADOW:
        case mi::mdl::glsl::SHAPE_CUBE_SHADOW:
        case mi::mdl::glsl::SHAPE_1D_ARRAY_SHADOW:
        case mi::mdl::glsl::SHAPE_2D_ARRAY_SHADOW:
        case mi::mdl::glsl::SHAPE_2D_RECT_SHADOW:
        case mi::mdl::glsl::SHAPE_CUBE_ARRAY_SHADOW:
            // no shadow image
            break;

        case mi::mdl::glsl::SHAPE_1D_ARRAY:
            return Symbol::SYM_TYPE_UIMAGE1DARRAY;
        case mi::mdl::glsl::SHAPE_2D_ARRAY:
            return Symbol::SYM_TYPE_UIMAGE2DARRAY;
        case mi::mdl::glsl::SHAPE_2D_RECT:
            return Symbol::SYM_TYPE_UIMAGE2DRECT;
        case mi::mdl::glsl::SHAPE_BUFFER:
            return Symbol::SYM_TYPE_UIMAGEBUFFER;
        case mi::mdl::glsl::SHAPE_2DMS:
            return Symbol::SYM_TYPE_UIMAGE2DMS;
        case mi::mdl::glsl::SHAPE_2DMS_ARRAY:
            return Symbol::SYM_TYPE_UIMAGE2DMSARRAY;
        case mi::mdl::glsl::SHAPE_CUBE_ARRAY:
            return Symbol::SYM_TYPE_UIMAGECUBEARRAY;

        case mi::mdl::glsl::SHAPE_EXTERNAL_OES:
            // only used for samplers
            break;
        }
        break;
    default:
        break;
    }
    // unsupported if we are here
    return Symbol::SYM_ERROR;
}

// create the builtin types
#define BUILTIN_TYPE(type, name, args) type name args;

#include "compiler_glsl_builtin_types.h"

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

// Get the type class of this type.
Type::Type_class Type_alias::get_type_class()
{
    return m_aliased_type->get_type_class();
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

// Get the type class of this type.
Type::Type_class Type_error::get_type_class()
{
    return TC_other;
}

// ---------------------------- Scalar type ----------------------------

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

// Get the type class of this type.
Type::Type_class Type_void::get_type_class()
{
    return TC_other;
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

// Get the type class of this type.
Type::Type_class Type_bool::get_type_class()
{
    return TC_genBType;
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

// Get the type class of this type.
Type::Type_class Type_signed_int::get_type_class()
{
    return TC_genIType;
}

// ---------------------------- unsigned two complement type ----------------------------

// Constructor.
Type_unsigned_int::Type_unsigned_int(
    Symbol *sym,
    size_t bitsize)
: Base(sym, bitsize)
{
}

// Get the type class of this type.
Type::Type_class Type_unsigned_int::get_type_class()
{
    return TC_genUType;
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

// ---------------------------- int8_t type ----------------------------

// Constructor.
Type_int8_t::Type_int8_t()
: Base(Symbol_table::get_predefined_symbol(Symbol::SYM_TYPE_INT8_T), 8)
{
}

// Get the type kind.
Type::Kind Type_int8_t::get_kind()
{
    return s_kind;
}

// ---------------------------- uint8_t type ----------------------------

// Constructor.
Type_uint8_t::Type_uint8_t()
: Base(Symbol_table::get_predefined_symbol(Symbol::SYM_TYPE_UINT8_T), 8)
{
}

// Get the type kind.
Type::Kind Type_uint8_t::get_kind()
{
    return s_kind;
}

// ---------------------------- int16_t type ----------------------------

// Constructor.
Type_int16_t::Type_int16_t()
: Base(Symbol_table::get_predefined_symbol(Symbol::SYM_TYPE_INT16_T), 16)
{
}

// Get the type kind.
Type::Kind Type_int16_t::get_kind()
{
    return s_kind;
}

// ---------------------------- uint16_t type ----------------------------

// Constructor.
Type_uint16_t::Type_uint16_t()
: Base(Symbol_table::get_predefined_symbol(Symbol::SYM_TYPE_UINT16_T), 16)
{
}

// Get the type kind.
Type::Kind Type_uint16_t::get_kind()
{
    return s_kind;
}

// ---------------------------- int64_t type ----------------------------

// Constructor.
Type_int64_t::Type_int64_t()
: Base(Symbol_table::get_predefined_symbol(Symbol::SYM_TYPE_INT64_T), 64)
{
}

// Get the type kind.
Type::Kind Type_int64_t::get_kind()
{
    return s_kind;
}

// ---------------------------- uint64_t type ----------------------------

// Constructor.
Type_uint64_t::Type_uint64_t()
: Base(Symbol_table::get_predefined_symbol(Symbol::SYM_TYPE_UINT64_T), 64)
{
}

// Get the type kind.
Type::Kind Type_uint64_t::get_kind()
{
    return s_kind;
}

// ---------------------------- float half ----------------------------

// Constructor.
Type_half::Type_half()
: Base(Symbol_table::get_predefined_symbol(Symbol::SYM_TYPE_FLOAT16_T))
{
}

// Get the type kind.
Type::Kind Type_half::get_kind()
{
    return s_kind;
}

// Get the type class of this type.
Type::Type_class Type_half::get_type_class()
{
    return TC_other;
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

// Get the type class of this type.
Type::Type_class Type_float::get_type_class()
{
    return TC_genType;
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

// Get the type class of this type.
Type::Type_class Type_double::get_type_class()
{
    return TC_genDType;
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
    GLSL_ASSERT(2 <= size && size <= 4);
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
    case Type::TK_INT8_T:
        id = Symbol::SYM_TYPE_INT8_T;
        break;
    case Type::TK_UINT8_T:
        id = Symbol::SYM_TYPE_UINT8_T;
        break;
    case Type::TK_INT16_T:
        id = Symbol::SYM_TYPE_INT16_T;
        break;
    case Type::TK_UINT16_T:
        id = Symbol::SYM_TYPE_UINT16_T;
        break;
    case Type::TK_INT64_T:
        id = Symbol::SYM_TYPE_INT64_T;
        break;
    case Type::TK_UINT64_T:
        id = Symbol::SYM_TYPE_UINT64_T;
        break;
    case Type::TK_FLOAT:
        id = Symbol::SYM_TYPE_FLOAT;
        break;
    case Type::TK_DOUBLE:
        id = Symbol::SYM_TYPE_DOUBLE;
        break;
    default:
        break;
    }

    if (id != Symbol::SYM_ERROR)
        id = Symbol::Predefined_id(id + size - 1);

    return Symbol_table::get_predefined_symbol(id);
}

// Get the type kind.
Type::Kind Type_vector::get_kind()
{
    return s_kind;
}

// Get the type class of this type.
Type::Type_class Type_vector::get_type_class()
{
    return get_element_type()->get_type_class();
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
    GLSL_ASSERT(2 <= n_columns && n_columns <= 4);
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
    case Type::TK_FLOAT:
        id = Symbol::SYM_TYPE_MAT2;
        break;
    case Type::TK_DOUBLE:
        id = Symbol::SYM_TYPE_DMAT2;
        break;
    default:
        break;
    }

    if (id != Symbol::SYM_ERROR)
        id = Symbol::Predefined_id(id + (columns - 2) * 3 + (rows - 2));
    return Symbol_table::get_predefined_symbol(id);
}

// Get the type kind.
Type::Kind Type_matrix::get_kind()
{
    return s_kind;
}

// Get the type class of this type.
Type::Type_class Type_matrix::get_type_class()
{
    Type_class tc = get_element_type()->get_type_class();

    if (tc == TC_genType) {
        return TC_mat;
    } else {
        GLSL_ASSERT(tc == TC_genDType);
        return TC_dmat;
    }
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

// Get the type class of this type.
Type::Type_class Type_array::get_type_class()
{
    return TC_other;
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

// Get the type class of this type.
Type::Type_class Type_struct::get_type_class()
{
    return TC_other;
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
    GLSL_ASSERT(!"index out of range");
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
    GLSL_ASSERT(ret_type != NULL);
}

// Get the type kind.
Type::Kind Type_function::get_kind()
{
    return s_kind;
}

// Get the type class of this type.
Type::Type_class Type_function::get_type_class()
{
    return TC_other;
}

// Get a parameter of the function type.
Type_function::Parameter *Type_function::get_parameter(size_t index)
{
    if (index < m_n_params)
        return &m_params[index];
    GLSL_ASSERT(!"index out of range");
    return NULL;
}

// ---------------------------- Pointer type ----------------------------

// Constructor.
Type_pointer::Type_pointer(
    Symbol *sym,
    Type   *points_to_type)
: Base(sym)
, m_points_to_type(points_to_type)
{
    GLSL_ASSERT(points_to_type != NULL);
}

// Get the type kind.
Type::Kind Type_pointer::get_kind()
{
    return s_kind;
}

// Get the type class of this type.
Type::Type_class Type_pointer::get_type_class()
{
    return TC_other;
}

// ----------------------------opaque types ----------------------------

// Constructor.
Type_opaque::Type_opaque(
    Symbol      *sym,
    Type_scalar *base)
: Base(sym)
, m_base_tp(base)
{
}

// Get the type class of this type.
Type::Type_class Type_opaque::get_type_class()
{
    // all opaque types are of class other
    return TC_other;
}

// ---------------------------- atomic uint type ----------------------------

// Constructor.
Type_atomic_uint::Type_atomic_uint()
: Base(Symbol_table::get_predefined_symbol(Symbol::SYM_TYPE_ATOMIC_UINT), &glsl_uint_type)
{
}

// Get the type kind.
Type::Kind Type_atomic_uint::get_kind()
{
    return s_kind;
}

// ---------------------------- sample type ----------------------------

// Constructor.
Type_sampler::Type_sampler(
    Type_scalar   *base_tp,
    Texture_shape shape)
: Base(Symbol_table::get_predefined_symbol(get_sampler_id(base_tp, shape)), base_tp)
, m_shape(shape)
{
}

// Get the type kind.
Type::Kind Type_sampler::get_kind()
{
    return s_kind;
}

// ---------------------------- image type ----------------------------

// Constructor.
Type_image::Type_image(
    Type_scalar   *base_tp,
    Texture_shape shape)
: Base(Symbol_table::get_predefined_symbol(get_image_id(base_tp, shape)), base_tp)
, m_shape(shape)
{
}

// Get the type kind.
Type::Kind Type_image::get_kind()
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
    type = type->skip_type_alias();

    Type_cache_key key(type, modifiers);

    Type_cache::const_iterator it = m_type_cache.find(key);
    if (it == m_type_cache.end()) {
        string new_name(type->get_sym()->get_name(), m_builder.get_arena()->get_allocator());

        if (modifiers & Type::MK_CONST) {
            new_name = "const " + new_name;
        }
        if (modifiers & Type::MK_UNIFORM) {
            new_name = "uniform " + new_name;
        }
        if (modifiers & Type::MK_VARYING) {
            new_name = "uniform " + new_name;
        }
        if (modifiers & Type::MK_LOWP) {
            new_name = "lowp " + new_name;
        }
        if (modifiers & Type::MK_MEDIUMP) {
            new_name = "mediump " + new_name;
        }
        if (modifiers & Type::MK_HIGHP) {
            new_name = "highp " + new_name;
        }
        Symbol *sym = m_symtab.get_user_type_symbol(new_name.c_str());

        Type *alias_type = m_builder.create<Type_alias>(type, modifiers, sym);

        it = m_type_cache.insert(Type_cache::value_type(key, alias_type)).first;
    }
    return it->second;
}

// Get the (singleton) error type instance.
Type_error *Type_factory::get_error()
{
    return &glsl_error_type;
}

// Get the (singleton) void type instance.
Type_void *Type_factory::get_void()
{
    return &glsl_void_type;
}

// Get the (singleton) bool type instance.
Type_bool *Type_factory::get_bool()
{
    return &glsl_bool_type;
}

// Get the (singleton) int type instance.
Type_int *Type_factory::get_int()
{
    return &glsl_int_type;
}

// Get the (singleton) uint type instance.
Type_uint *Type_factory::get_uint()
{
    return &glsl_uint_type;
}

// Get the (singleton) int8_t type instance.
Type_int8_t *Type_factory::get_int8_t()
{
    return &glsl_int8_t_type;
}

// Get the (singleton) uint8_t type instance.
Type_uint8_t *Type_factory::get_uint8_t()
{
    return &glsl_uint8_t_type;
}

// Get the (singleton) int16_t type instance.
Type_int16_t *Type_factory::get_int16_t()
{
    return &glsl_int16_t_type;
}

// Get the (singleton) uint16_t type instance.
Type_uint16_t *Type_factory::get_uint16_t()
{
    return &glsl_uint16_t_type;
}

// Get the (singleton) int32_t type instance, same as int.
Type_int32_t *Type_factory::get_int32_t()
{
    return get_int();
}

// Get the (singleton) uint32_t type instance, same as uint.
Type_uint32_t *Type_factory::get_uint32_t()
{
    return get_uint();
}

// Get the (singleton) int64_t type instance.
Type_int64_t *Type_factory::get_int64_t()
{
    return &glsl_int64_t_type;
}

// Get the (singleton) uint64_t type instance.
Type_uint64_t *Type_factory::get_uint64_t()
{
    return &glsl_uint64_t_type;
}

// Get the (singleton) half type instance.
Type_half *Type_factory::get_half()
{
    return &glsl_half_type;
}

// Get the (singleton) float type instance.
Type_float *Type_factory::get_float()
{
    return &glsl_float_type;
}

// Get the (singleton) double type instance.
Type_double *Type_factory::get_double()
{
    return &glsl_double_type;
}

// Get a vector type 1instance.
Type_vector *Type_factory::get_vector(
    Type_scalar *element_type,
    size_t      size)
{
    switch (size) {
    case 2:
        switch (element_type->get_kind()) {
        case Type::TK_BOOL:
            return &glsl_bvec2_type;
        case Type::TK_INT:
            return &glsl_ivec2_type;
        case Type::TK_UINT:
            return &glsl_uvec2_type;
        case Type::TK_INT8_T:
            return &glsl_i8vec2_type;
        case Type::TK_UINT8_T:
            return &glsl_u8vec2_type;
        case Type::TK_INT16_T:
            return &glsl_i16vec2_type;
        case Type::TK_UINT16_T:
            return &glsl_u16vec2_type;
        case Type::TK_INT64_T:
            return &glsl_i64vec2_type;
        case Type::TK_UINT64_T:
            return &glsl_u64vec2_type;
        case Type::TK_FLOAT:
            return &glsl_vec2_type;
        case Type::TK_DOUBLE:
            return &glsl_dvec2_type;
        default:
            break;
        }
        break;
    case 3:
        switch (element_type->get_kind()) {
        case Type::TK_BOOL:
            return &glsl_bvec3_type;
        case Type::TK_INT:
            return &glsl_ivec3_type;
        case Type::TK_UINT:
            return &glsl_uvec3_type;
        case Type::TK_INT8_T:
            return &glsl_i8vec3_type;
        case Type::TK_UINT8_T:
            return &glsl_u8vec3_type;
        case Type::TK_INT16_T:
            return &glsl_i16vec3_type;
        case Type::TK_UINT16_T:
            return &glsl_u16vec3_type;
        case Type::TK_INT64_T:
            return &glsl_i64vec3_type;
        case Type::TK_UINT64_T:
            return &glsl_u64vec3_type;
        case Type::TK_FLOAT:
            return &glsl_vec3_type;
        case Type::TK_DOUBLE:
            return &glsl_dvec3_type;
        default:
            break;
        }
        break;
    case 4:
        switch (element_type->get_kind()) {
        case Type::TK_BOOL:
            return &glsl_bvec4_type;
        case Type::TK_INT:
            return &glsl_ivec4_type;
        case Type::TK_UINT:
            return &glsl_uvec4_type;
        case Type::TK_INT8_T:
            return &glsl_i8vec4_type;
        case Type::TK_UINT8_T:
            return &glsl_u8vec4_type;
        case Type::TK_INT16_T:
            return &glsl_i16vec4_type;
        case Type::TK_UINT16_T:
            return &glsl_u16vec4_type;
        case Type::TK_INT64_T:
            return &glsl_i64vec4_type;
        case Type::TK_UINT64_T:
            return &glsl_u64vec4_type;
        case Type::TK_FLOAT:
            return &glsl_vec4_type;
        case Type::TK_DOUBLE:
            return &glsl_dvec4_type;
        default:
            break;
        }
        break;
    }
    // unsupported
    return NULL;
}

// Get a matrix type instance.
Type_matrix *Type_factory::get_matrix(
    Type_vector *element_type,
    size_t      columns)
{
    Type::Kind kind = element_type->get_element_type()->get_kind();

    switch (columns) {
    case 2:
        switch (element_type->get_size()) {
        case 2:
            if (kind == Type::TK_FLOAT)
                return &glsl_mat2_type;
            else if (kind == Type::TK_DOUBLE)
                return &glsl_dmat2_type;
            break;
        case 3:
            if (kind == Type::TK_FLOAT)
                return &glsl_mat3x2_type;
            else if (kind == Type::TK_DOUBLE)
                return &glsl_dmat3x2_type;
            break;
        case 4:
            if (kind == Type::TK_FLOAT)
                return &glsl_mat4x2_type;
            else if (kind == Type::TK_DOUBLE)
                return &glsl_dmat4x2_type;
            break;
        }
        break;
    case 3:
        switch (element_type->get_size()) {
        case 2:
            if (kind == Type::TK_FLOAT)
                return &glsl_mat2x3_type;
            else if (kind == Type::TK_DOUBLE)
                return &glsl_dmat2x3_type;
            break;
        case 3:
            if (kind == Type::TK_FLOAT)
                return &glsl_mat3_type;
            else if (kind == Type::TK_DOUBLE)
                return &glsl_dmat3_type;
            break;
        case 4:
            if (kind == Type::TK_FLOAT)
                return &glsl_mat4x3_type;
            else if (kind == Type::TK_DOUBLE)
                return &glsl_dmat4x3_type;
            break;
        }
        break;
    case 4:
        switch (element_type->get_size()) {
        case 2:
            if (kind == Type::TK_FLOAT)
                return &glsl_mat2x4_type;
            else if (kind == Type::TK_DOUBLE)
                return &glsl_dmat2x4_type;
            break;
        case 3:
            if (kind == Type::TK_FLOAT)
                return &glsl_mat3x4_type;
            else if (kind == Type::TK_DOUBLE)
                return &glsl_dmat3x4_type;
            break;
        case 4:
            if (kind == Type::TK_FLOAT)
                return &glsl_mat4_type;
            else if (kind == Type::TK_DOUBLE)
                return &glsl_dmat4_type;
            break;
        }
        break;
    }
    // unsupported
    GLSL_ASSERT(!"cannot retrieve unsupported matrix type");
    return NULL;
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

    GLSL_ASSERT(size > 0 && "array size mustbe > 0");

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

        Symbol *sym = m_symtab.get_symbol(name.c_str());
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

// Get a pointer type instance.
Type *Type_factory::get_pointer(
    Type *points_to_type)
{
    if (is<Type_error>(points_to_type)) {
        // cannot create a pointer to the error type
        return points_to_type;
    }

    Type_cache_key key(points_to_type);

    Type_cache::const_iterator it = m_type_cache.find(key);
    if (it == m_type_cache.end()) {
        IAllocator *alloc = m_builder.get_arena()->get_allocator();
        string name(points_to_type->get_sym()->get_name(), alloc);

        name += " *";
        Symbol *sym = m_symtab.get_user_type_symbol(name.c_str());
        Type_pointer *pointer_type = m_builder.create<Type_pointer>(sym, points_to_type);

        it = m_type_cache.insert(Type_cache::value_type(key, pointer_type)).first;
    }
    return it->second;
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

// Get the (singleton) atomic_uint type instance.
Type_atomic_uint *Type_factory::get_atomic_uint()
{
    return &glsl_atomic_uint_type;
}

// Get an opaque sampler type
Type_sampler *Type_factory::get_sampler(
    Type_scalar   *base_tp,
    Texture_shape shape)
{
    GLSL_ASSERT(base_tp != NULL);
    Symbol::Predefined_id id = get_sampler_id(base_tp, shape);
    GLSL_ASSERT(id != Symbol::SYM_ERROR && "No such sampler type in GLSL");

    switch (id) {
    case Symbol::SYM_TYPE_SAMPLER1D:
        return &glsl_sampler1D_type;
    case Symbol::SYM_TYPE_SAMPLER2D:
        return &glsl_sampler2D_type;
    case Symbol::SYM_TYPE_SAMPLER3D:
        return &glsl_sampler3D_type;
    case Symbol::SYM_TYPE_SAMPLERCUBE:
        return &glsl_samplerCube_type;
    case Symbol::SYM_TYPE_SAMPLER1DSHADOW:
        return &glsl_sampler1DShadow_type;
    case Symbol::SYM_TYPE_SAMPLER2DSHADOW:
        return &glsl_sampler2DShadow_type;
    case Symbol::SYM_TYPE_SAMPLERCUBESHADOW:
        return &glsl_samplerCubeShadow_type;
    case Symbol::SYM_TYPE_SAMPLER1DARRAY:
        return &glsl_sampler1DArray_type;
    case Symbol::SYM_TYPE_SAMPLER2DARRAY:
        return &glsl_sampler2DArray_type;
    case Symbol::SYM_TYPE_SAMPLER1DARRAYSHADOW:
        return &glsl_sampler1DArrayShadow_type;
    case Symbol::SYM_TYPE_SAMPLER2DARRAYSHADOW:
        return &glsl_sampler2DArrayShadow_type;
    case Symbol::SYM_TYPE_SAMPLER2DRECT:
        return &glsl_sampler2DRect_type;
    case Symbol::SYM_TYPE_SAMPLER2DRECTSHADOW:
        return &glsl_sampler2DRectShadow_type;
    case Symbol::SYM_TYPE_SAMPLERBUFFER:
        return &glsl_samplerBuffer_type;
    case Symbol::SYM_TYPE_SAMPLER2DMS:
        return &glsl_sampler2DMS_type;
    case Symbol::SYM_TYPE_SAMPLER2DMSARRAY:
        return &glsl_sampler2DMSArray_type;
    case Symbol::SYM_TYPE_SAMPLERCUBEARRAY:
        return &glsl_samplerCubeArray_type;
    case Symbol::SYM_TYPE_SAMPLERCUBEARRAYSHADOW:
        return &glsl_samplerCubeArrayShadow_type;
    case Symbol::SYM_TYPE_ISAMPLER1D:
        return &glsl_sampler1D_type;
    case Symbol::SYM_TYPE_ISAMPLER2D:
        return &glsl_isampler2D_type;
    case Symbol::SYM_TYPE_ISAMPLER3D:
        return &glsl_isampler3D_type;
    case Symbol::SYM_TYPE_ISAMPLERCUBE:
        return &glsl_isamplerCube_type;
    case Symbol::SYM_TYPE_ISAMPLER1DARRAY:
        return &glsl_isampler1DArray_type;
    case Symbol::SYM_TYPE_ISAMPLER2DARRAY:
        return &glsl_isampler2DArray_type;
    case Symbol::SYM_TYPE_ISAMPLER2DRECT:
        return &glsl_isampler2DRect_type;
    case Symbol::SYM_TYPE_ISAMPLERBUFFER:
        return &glsl_isamplerBuffer_type;
    case Symbol::SYM_TYPE_ISAMPLER2DMS:
        return &glsl_isampler2DMS_type;
    case Symbol::SYM_TYPE_ISAMPLER2DMSARRAY:
        return &glsl_isampler2DMSArray_type;
    case Symbol::SYM_TYPE_ISAMPLERCUBEARRAY:
        return &glsl_isamplerCubeArray_type;
    case Symbol::SYM_TYPE_USAMPLER1D:
        return &glsl_usampler1D_type;
    case Symbol::SYM_TYPE_USAMPLER2D:
        return &glsl_usampler2D_type;
    case Symbol::SYM_TYPE_USAMPLER3D:
        return &glsl_usampler3D_type;
    case Symbol::SYM_TYPE_USAMPLERCUBE:
        return &glsl_usamplerCube_type;
    case Symbol::SYM_TYPE_USAMPLER1DARRAY:
        return &glsl_usampler1DArray_type;
    case Symbol::SYM_TYPE_USAMPLER2DARRAY:
        return &glsl_usampler2DArray_type;
    case Symbol::SYM_TYPE_USAMPLER2DRECT:
        return &glsl_usampler2DRect_type;
    case Symbol::SYM_TYPE_USAMPLERBUFFER:
        return &glsl_usamplerBuffer_type;
    case Symbol::SYM_TYPE_USAMPLER2DMS:
        return &glsl_usampler2DMS_type;
    case Symbol::SYM_TYPE_USAMPLER2DMSARRAY:
        return &glsl_usampler2DMSArray_type;
    case Symbol::SYM_TYPE_USAMPLERCUBEARRAY:
        return &glsl_usamplerCubeArray_type;

    // OES_EGL_image_external
    case Symbol::SYM_TYPE_SAMPLEREXTERNALOES:
        return &glsl_samplerExternalOES_type;

    default:
        return NULL;
    }
}

// Get an opaque image type
Type_image *Type_factory::get_image(
    Type_scalar   *base_tp,
    Texture_shape shape)
{
    GLSL_ASSERT(base_tp != NULL);
    Symbol::Predefined_id id = get_image_id(base_tp, shape);
    GLSL_ASSERT(id != Symbol::SYM_ERROR && "No such image type in GLSL");

    switch (id) {
    case Symbol::SYM_TYPE_IMAGE1D:
        return &glsl_image1D_type;
    case Symbol::SYM_TYPE_IMAGE2D:
        return &glsl_image2D_type;
    case Symbol::SYM_TYPE_IMAGE3D:
        return &glsl_image3D_type;
    case Symbol::SYM_TYPE_IMAGECUBE:
        return &glsl_imageCube_type;
    case Symbol::SYM_TYPE_IMAGE1DARRAY:
        return &glsl_image1DArray_type;
    case Symbol::SYM_TYPE_IMAGE2DARRAY:
        return &glsl_image2DArray_type;
    case Symbol::SYM_TYPE_IMAGE2DRECT:
        return &glsl_image2DRect_type;
    case Symbol::SYM_TYPE_IMAGEBUFFER:
        return &glsl_imageBuffer_type;
    case Symbol::SYM_TYPE_IMAGE2DMS:
        return &glsl_image2DMS_type;
    case Symbol::SYM_TYPE_IMAGE2DMSARRAY:
        return &glsl_image2DMSArray_type;
    case Symbol::SYM_TYPE_IMAGECUBEARRAY:
        return &glsl_imageCubeArray_type;
    case Symbol::SYM_TYPE_IIMAGE1D:
        return &glsl_image1D_type;
    case Symbol::SYM_TYPE_IIMAGE2D:
        return &glsl_iimage2D_type;
    case Symbol::SYM_TYPE_IIMAGE3D:
        return &glsl_iimage3D_type;
    case Symbol::SYM_TYPE_IIMAGECUBE:
        return &glsl_iimageCube_type;
    case Symbol::SYM_TYPE_IIMAGE1DARRAY:
        return &glsl_iimage1DArray_type;
    case Symbol::SYM_TYPE_IIMAGE2DARRAY:
        return &glsl_iimage2DArray_type;
    case Symbol::SYM_TYPE_IIMAGE2DRECT:
        return &glsl_iimage2DRect_type;
    case Symbol::SYM_TYPE_IIMAGEBUFFER:
        return &glsl_iimageBuffer_type;
    case Symbol::SYM_TYPE_IIMAGE2DMS:
        return &glsl_iimage2DMS_type;
    case Symbol::SYM_TYPE_IIMAGE2DMSARRAY:
        return &glsl_iimage2DMSArray_type;
    case Symbol::SYM_TYPE_IIMAGECUBEARRAY:
        return &glsl_iimageCubeArray_type;
    case Symbol::SYM_TYPE_UIMAGE1D:
        return &glsl_uimage1D_type;
    case Symbol::SYM_TYPE_UIMAGE2D:
        return &glsl_uimage2D_type;
    case Symbol::SYM_TYPE_UIMAGE3D:
        return &glsl_uimage3D_type;
    case Symbol::SYM_TYPE_UIMAGECUBE:
        return &glsl_uimageCube_type;
    case Symbol::SYM_TYPE_UIMAGE1DARRAY:
        return &glsl_uimage1DArray_type;
    case Symbol::SYM_TYPE_UIMAGE2DARRAY:
        return &glsl_uimage2DArray_type;
    case Symbol::SYM_TYPE_UIMAGE2DRECT:
        return &glsl_uimage2DRect_type;
    case Symbol::SYM_TYPE_UIMAGEBUFFER:
        return &glsl_uimageBuffer_type;
    case Symbol::SYM_TYPE_UIMAGE2DMS:
        return &glsl_uimage2DMS_type;
    case Symbol::SYM_TYPE_UIMAGE2DMSARRAY:
        return &glsl_uimage2DMSArray_type;
    case Symbol::SYM_TYPE_UIMAGECUBEARRAY:
        return &glsl_uimageCubeArray_type;
    default:
        return NULL;
    }
}

// If a given type has an unsigned variant, return it.
Type *Type_factory::to_unsigned_type(Type *type)
{
    switch (type->get_kind()) {
    case Type::TK_ALIAS:
        return to_unsigned_type(type->skip_type_alias());

    case Type::TK_VOID:
    case Type::TK_BOOL:
    case Type::TK_HALF:
    case Type::TK_FLOAT:
    case Type::TK_DOUBLE:
    case Type::TK_ARRAY:
    case Type::TK_STRUCT:
    case Type::TK_FUNCTION:
    case Type::TK_POINTER:
    case Type::TK_IMAGE:
    case Type::TK_SAMPLER:
    case Type::TK_ATOMIC_UINT:
    case Type::TK_ERROR:
        return NULL;

    case Type::TK_INT8_T:
        return get_uint8_t();

    case Type::TK_UINT8_T:
        return type;

    case Type::TK_INT16_T:
        return get_uint16_t();

    case Type::TK_UINT16_T:
        return type;

    case Type::TK_INT:
        return get_uint();

    case Type::TK_UINT:
        return type;

    case Type::TK_INT64_T:
        return get_uint64_t();

    case Type::TK_UINT64_T:
        return type;

    case Type::TK_VECTOR:
        {
            Type_vector *v_type = cast<Type_vector>(type);
            Type_scalar *e_type = v_type->get_element_type();
            Type_scalar *u_type = cast<Type_scalar>(to_unsigned_type(e_type));

            if (u_type != NULL) {
                if (u_type != e_type) {
                    return get_vector(u_type, v_type->get_size());
                }
                return type;
            }
        }
        return NULL;

    case Type::TK_MATRIX:
        {
            Type_matrix *m_type = cast<Type_matrix>(type);
            Type_vector *e_type = m_type->get_element_type();
            Type_vector *u_type = cast<Type_vector>(to_unsigned_type(e_type));

            if (u_type != NULL) {
                if (u_type != e_type) {
                    return get_matrix(u_type, m_type->get_columns());
                }
                return type;
            }
        }
        return NULL;
    }
    GLSL_ASSERT("!unexpected type kind");
    return NULL;
}

// Get the size of a GLSL type in bytes.
size_t Type_factory::get_type_size(Type *type)
{
    switch (type->get_kind()) {
    case Type::TK_ALIAS:
        return get_type_size(type->skip_type_alias());

    case Type::TK_VOID:
        return 0;
    case Type::TK_BOOL:
        return 1;
    case Type::TK_HALF:
        // 16bit in GLSL
        return 2;
    case Type::TK_INT8_T:
    case Type::TK_UINT8_T:
        return 1;
    case Type::TK_INT16_T:
    case Type::TK_UINT16_T:
        return 2;
    case Type::TK_INT:
    case Type::TK_UINT:
        // 32bit in GLSL
        return 4;
    case Type::TK_INT64_T:
    case Type::TK_UINT64_T:
        return 8;
    case Type::TK_FLOAT:
    case Type::TK_ATOMIC_UINT:
        // 32bit in GLSL
        return 4;
    case Type::TK_DOUBLE:
        // 64bit in GLSL
        return 8;

    case Type::TK_VECTOR:
        {
            Type_vector *v_tp = cast<Type_vector>(type);
            return v_tp->get_size() * get_type_size(v_tp->get_element_type());
        }
    case Type::TK_MATRIX:
        {
            Type_matrix *m_tp = cast<Type_matrix>(type);
            return m_tp->get_columns() * get_type_size(m_tp->get_element_type());
        }

    case Type::TK_ARRAY:
        {
            Type_array *a_tp = cast<Type_array>(type);
            GLSL_ASSERT(!a_tp->is_unsized());
            return a_tp->get_size() * get_type_size(a_tp->get_element_type());
        }

    case Type::TK_STRUCT:
        {
            Type_struct *s_tp = cast<Type_struct>(type);

            size_t s         = 0;
            size_t max_align = 0;
            for (size_t i = 0, n = s_tp->get_compound_size(); i < n; ++i) {
                Type   *e_tp = s_tp->get_compound_type(i);
                size_t align = get_type_alignment(e_tp);

                if (align > max_align)
                    max_align = align;
                s = (s + align - 1) & ~(align - 1);
                s += get_type_size(e_tp);
            }
            return (s + max_align - 1) & ~(max_align - 1);
        }

    case Type::TK_FUNCTION:
        GLSL_ASSERT(!"size of function type");
        return 0;

    case Type::TK_POINTER:
        // FIXME: either 32bit or 64bit
        return 8;

    case Type::TK_IMAGE:
    case Type::TK_SAMPLER:
        // FIXME: unsupported yet
        GLSL_ASSERT(!"size of image/sample NYI");
        return 8;

    case Type::TK_ERROR:
        GLSL_ASSERT(!"size of error type");
        return 0;
    }
    GLSL_ASSERT(!"unsupported type kind");
    return 0;
}

// Get the alignment of a GLSL type in bytes.
size_t Type_factory::get_type_alignment(Type *type)
{
    switch (type->get_kind()) {
    case Type::TK_ALIAS:
        return get_type_alignment(type->skip_type_alias());

    case Type::TK_VOID:
        return 1;
    case Type::TK_BOOL:
        return 1;
    case Type::TK_HALF:
        // 16bit in GLSL
        return 2;
    case Type::TK_INT8_T:
    case Type::TK_UINT8_T:
        return 1;
    case Type::TK_INT16_T:
    case Type::TK_UINT16_T:
        return 2;
    case Type::TK_INT:
    case Type::TK_UINT:
    case Type::TK_FLOAT:
    case Type::TK_ATOMIC_UINT:
        // 32bit in GLSL
        return 4;
    case Type::TK_INT64_T:
    case Type::TK_UINT64_T:
    case Type::TK_DOUBLE:
        // 64bit in GLSL
        return 8;

    case Type::TK_VECTOR:
        {
            Type_vector *v_tp = cast<Type_vector>(type);
            return get_type_alignment(v_tp->get_element_type());
        }
    case Type::TK_MATRIX:
        {
            Type_matrix *m_tp = cast<Type_matrix>(type);
            return get_type_alignment(m_tp->get_element_type());
        }

    case Type::TK_ARRAY:
        {
            Type_array *a_tp = cast<Type_array>(type);
            return get_type_alignment(a_tp->get_element_type());
        }

    case Type::TK_STRUCT:
        {
            Type_struct *s_tp = cast<Type_struct>(type);

            size_t max_align = 0;
            for (size_t i = 0, n = s_tp->get_compound_size(); i < n; ++i) {
                Type   *e_tp = s_tp->get_compound_type(i);
                size_t align = get_type_alignment(e_tp);

                if (align > max_align)
                    max_align = align;
            }
            return max_align;
        }

    case Type::TK_FUNCTION:
        GLSL_ASSERT(!"size of function type");
        return 1;

    case Type::TK_POINTER:
        // FIXME: either 32bit or 64bit
        return 8;

    case Type::TK_IMAGE:
    case Type::TK_SAMPLER:
        // FIXME: unsupported yet
        GLSL_ASSERT(!"size of image/sample NYI");
        return 8;

    case Type::TK_ERROR:
        GLSL_ASSERT(!"size of error type");
        return 1;
    }
    GLSL_ASSERT(!"unsupported type kind");
    return 1;
}

}  // glsl
}  // mdl
}  // mi
