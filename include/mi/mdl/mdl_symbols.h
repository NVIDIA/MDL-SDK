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
/// \file mi/mdl/mdl_symbols.h
/// \brief interfaces for the MDL symbol table and symbols
#ifndef MDL_SYMBOLS_H
#define MDL_SYMBOLS_H 1

#include <cstddef>
#include <mi/mdl/mdl_iowned.h>

namespace mi {
namespace mdl {

/// A symbol.
///
/// A symbol is used to represent "an itentifiwe" in an MDL program.
/// For fast lookup, every symbol is unique and has an Id (a number) associated.
/// Two symbols of the same symbol table are equal if and only if their addresses are
/// equal and different otherwise.
/// 
/// \note Note that equal symbols from _different_ symbol tables are represented by different
///       ISymbol instance and cannot be compared that way.
///       These symbols must be imported into the symbol table.
///
class ISymbol : public Interface_owned
{
public:
    /// Symbol Id's of predefined symbols.
    ///
    /// \note Note that operator and (fully qualified) user type name symbols are not
    /// used for symbol lookup in the definition table, hence they share one Id.
    ///
    enum Predefined_id {
        SYM_ERROR,                  ///< special error symbol
        SYM_OPERATOR,               ///< the Id of ALL operators
        SYM_STAR,                   ///< the Id of the star ('*') in import names
        SYM_DOT,                    ///< the Id of the dot ('.') in import names
        SYM_DOTDOT,                 ///< the Id of the dotdot ("..") in import names

        // types
        SYM_TYPE_FIRST,
        SYM_TYPE_BOOL = SYM_TYPE_FIRST,
        SYM_TYPE_BOOL2,
        SYM_TYPE_BOOL3,
        SYM_TYPE_BOOL4,
        SYM_TYPE_INT,
        SYM_TYPE_INT2,
        SYM_TYPE_INT3,
        SYM_TYPE_INT4,
        SYM_TYPE_FLOAT,
        SYM_TYPE_FLOAT2,
        SYM_TYPE_FLOAT3,
        SYM_TYPE_FLOAT4,
        SYM_TYPE_DOUBLE,
        SYM_TYPE_DOUBLE2,
        SYM_TYPE_DOUBLE3,
        SYM_TYPE_DOUBLE4,
        SYM_TYPE_FLOAT2X2,
        SYM_TYPE_FLOAT2X3,
        SYM_TYPE_FLOAT3X2,
        SYM_TYPE_FLOAT3X3,
        SYM_TYPE_FLOAT3X4,
        SYM_TYPE_FLOAT4X3,
        SYM_TYPE_FLOAT4X4,
        SYM_TYPE_FLOAT4X2,
        SYM_TYPE_FLOAT2X4,
        SYM_TYPE_DOUBLE2X2,
        SYM_TYPE_DOUBLE2X3,
        SYM_TYPE_DOUBLE3X2,
        SYM_TYPE_DOUBLE3X3,
        SYM_TYPE_DOUBLE3X4,
        SYM_TYPE_DOUBLE4X3,
        SYM_TYPE_DOUBLE4X4,
        SYM_TYPE_DOUBLE4X2,
        SYM_TYPE_DOUBLE2X4,
        SYM_TYPE_TEXTURE_2D,
        SYM_TYPE_TEXTURE_3D,
        SYM_TYPE_TEXTURE_CUBE,
        SYM_TYPE_TEXTURE_PTEX,
        SYM_TYPE_STRING,
        SYM_TYPE_COLOR,
        SYM_TYPE_LIGHT_PROFILE,
        SYM_TYPE_BSDF_MEASUREMENT,
        SYM_TYPE_BSDF,
        SYM_TYPE_VDF,
        SYM_TYPE_EDF,
        SYM_TYPE_MATERIAL_EMISSION,
        SYM_TYPE_MATERIAL_SURFACE,
        SYM_TYPE_MATERIAL_VOLUME,
        SYM_TYPE_MATERIAL_GEOMETRY,
        SYM_TYPE_MATERIAL,
        SYM_TYPE_TEX_GAMMA_MODE,
        SYM_TYPE_INTENSITY_MODE,
        SYM_TYPE_HAIR_BSDF,
        SYM_TYPE_LAST = SYM_TYPE_HAIR_BSDF,

        // enums
        SYM_ENUM_FIRST,
        SYM_ENUM_GAMMA_DEFAULT = SYM_ENUM_FIRST,
        SYM_ENUM_GAMMA_LINEAR,
        SYM_ENUM_GAMMA_SRGB,
        SYM_ENUM_INTENSITY_RADIANT_EXITANCE,
        SYM_ENUM_INTENSITY_POWER,
        SYM_ENUM_LAST = SYM_ENUM_INTENSITY_POWER,

        // fields
        SYM_FIELD_FIRST,
        SYM_FIELD_ABSORPTION_COEFFICIENT = SYM_FIELD_FIRST,
        SYM_FIELD_BACKFACE,
        SYM_FIELD_CUTOUT_OPACITY,
        SYM_FIELD_DISPLACEMENT,
        SYM_FIELD_EMISSION,
        SYM_FIELD_GEOMETRY,
        SYM_FIELD_INTENSITY,
        SYM_FIELD_IOR,
        SYM_FIELD_NORMAL,
        SYM_FIELD_ROUNDED_EDGES_ACROSS_MATERIALS,
        SYM_FIELD_ROUNDED_EDGES_RADIUS,
        SYM_FIELD_SCATTERING,
        SYM_FIELD_SCATTERING_COEFFICIENT,
        SYM_FIELD_SURFACE,
        SYM_FIELD_THIN_WALLED,
        SYM_FIELD_VOLUME,
        SYM_FIELD_MODE,
        SYM_FIELD_HAIR,
        SYM_FIELD_LAST = SYM_FIELD_HAIR,

        // constants
        SYM_CNST_FIRST,
        SYM_CNST_TRUE = SYM_CNST_FIRST,
        SYM_CNST_FALSE,
        SYM_CNST_STATE,
        SYM_CNST_TEX,
        SYM_CNST_LAST = SYM_CNST_TEX,

        // parameters of constructors
        SYM_PARAM_FIRST,
        SYM_PARAM_X = SYM_PARAM_FIRST,
        SYM_PARAM_Y,
        SYM_PARAM_Z,
        SYM_PARAM_W,
        SYM_PARAM_R,
        SYM_PARAM_G,
        SYM_PARAM_B,
        SYM_PARAM_VALUE,
        SYM_PARAM_RGB,
        SYM_PARAM_NAME,
        SYM_PARAM_WAVELENGTHS,
        SYM_PARAM_AMPLITUDES,
        SYM_PARAM_M00,
        SYM_PARAM_M01,
        SYM_PARAM_M02,
        SYM_PARAM_M03,
        SYM_PARAM_M10,
        SYM_PARAM_M11,
        SYM_PARAM_M12,
        SYM_PARAM_M13,
        SYM_PARAM_M20,
        SYM_PARAM_M21,
        SYM_PARAM_M22,
        SYM_PARAM_M23,
        SYM_PARAM_M30,
        SYM_PARAM_M31,
        SYM_PARAM_M32,
        SYM_PARAM_M33,
        SYM_PARAM_COL0,
        SYM_PARAM_COL1,
        SYM_PARAM_COL2,
        SYM_PARAM_COL3,
        SYM_PARAM_GAMMA,
        SYM_PARAM_BIG_N,
        SYM_PARAM_LAST = SYM_PARAM_BIG_N,

        // These two MUST be last in the given order.
        SYM_SHARED_NAME,   ///< the Id of ALL shared symbols (including fully qualified type names)
        SYM_FREE           ///< First free id.
    };

    /// Get the name of the symbol.
    virtual char const *get_name() const = 0;

    /// Get the id of the symbol.
    virtual size_t get_id() const = 0;
};

/// The interface of the symbol table.
///
/// In MDL Core we use the classic terminology, a symbol table is _just_
/// the table of all symbols in a programm, the association from symbols
/// to definitions is handled by teh definition table.
///
/// An ISymbol_table interface can be obtained by calling
/// the method get_symbol_table() on the interfaces IModule and IType_factory.
class ISymbol_table : public Interface_owned
{
public:
    /// Create a symbol or return the existing one.
    ///
    /// \param  name        The name of the symbol.
    ///
    /// \returns            The symbol.
    virtual ISymbol const *create_symbol(char const *name) = 0;

    /// Create a symbol for the given user type name or return the existing one.
    ///
    /// \param name         The user type name.
    ///
    /// \returns            The symbol.
    ///
    /// \note Use this symbols only for the name of user types!
    ///       Currently this method is exactly the same as \c create_shared_symbol()
    virtual ISymbol const *create_user_type_symbol(char const *name) = 0;

    /// Create a shared symbol (no unique ID).
    ///
    /// \param name         The shared symbol name.
    ///
    /// \returns            The symbol.
    virtual ISymbol const *create_shared_symbol(char const *name) = 0;
};

}  // mdl
}  // mi

#endif
