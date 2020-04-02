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
/// \file mi/mdl/mdl_values.h
/// \brief Interfaces for MDL values in the AST and in the DAG IR
#ifndef MDL_VALUES_H
#define MDL_VALUES_H 1

#include <cstddef>

#include <mi/mdl/mdl_iowned.h>
#include <mi/mdl/mdl_types.h>

namespace mi {
namespace mdl {

class ISymbol;
class IValue_factory;

/// The base interface of a value.
///
/// Values are unique and immutable in MDL Core. Hence, most values are equal in MDL Core,
/// if they have the same pointer, and different otherwise. The only exception from this
/// rule are floating point values +0, -0, and NaN. Floating point values cannot be
/// compared using their pointers.
class IValue : public Interface_owned
{
public:
    /// The possible kinds of values.
    enum Kind {
        VK_BAD,                 ///< A bad value.
        VK_BOOL,                ///< A boolean value.
        VK_INT,                 ///< An integer value.
        VK_ENUM,                ///< An enum value.
        VK_FLOAT,               ///< A float value.
        VK_DOUBLE,              ///< A double value.
        VK_STRING,              ///< A string value.
        VK_VECTOR,              ///< A vector value.
        VK_MATRIX,              ///< A matrix value.
        VK_ARRAY,               ///< An array value.
        VK_RGB_COLOR,           ///< A color value.
        VK_STRUCT,              ///< A struct value.
        VK_INVALID_REF,         ///< An invalid reference value.
        VK_TEXTURE,             ///< A texture value.
        VK_LIGHT_PROFILE,       ///< A light profile value.
        VK_BSDF_MEASUREMENT,    ///< A bsdf_measurement value.
    };

    /// Get the kind of value.
    virtual Kind get_kind() const = 0;

    /// Get the type of this value.
    ///
    /// \note Also values are const objects, they always return the unqualified type.
    virtual IType const *get_type() const = 0;

    /// Negate this value.
    ///
    /// \param factory  a factory to create new values.
    ///
    /// \return IValue_bad if not supported
    virtual IValue const *minus(IValue_factory *factory) const = 0;

    /// Bitwise negate this value.
    ///
    /// \param factory  a factory to create new values.
    ///
    /// \return IValue_bad if not supported
    virtual IValue const *bitwise_not(IValue_factory *factory) const = 0;

    /// Logically negate this value.
    ///
    /// \param factory  a factory to create new values.
    ///
    /// \return IValue_bad if not supported
    virtual IValue const *logical_not(IValue_factory *factory) const = 0;

    /// Extract a sub value from a compound value.
    ///
    /// \param factory  a factory to create new values.
    /// \param index    the compound index
    ///
    /// \return IValue_bad if not supported
    virtual IValue const *extract(IValue_factory *factory, int index) const = 0;

    /// Multiply.
    ///
    /// \param factory  a factory to create new values.
    /// \param rhs      the right hand operand
    ///
    /// \return IValue_bad if not supported
    virtual IValue const *multiply(IValue_factory *factory, IValue const *rhs) const = 0;

    /// Divide.
    ///
    /// \param factory  a factory to create new values.
    /// \param rhs      the right hand operand
    ///
    /// \return IValue_bad if not supported
    virtual IValue const *divide(IValue_factory *factory, IValue const *rhs) const = 0;

    /// Modulo.
    ///
    /// \param factory  a factory to create new values.
    /// \param rhs      the right hand operand
    ///
    /// \return IValue_bad if not supported
    virtual IValue const *modulo(IValue_factory *factory, IValue const *rhs) const = 0;

    /// Add.
    ///
    /// \param factory  a factory to create new values.
    /// \param rhs      the right hand operand
    ///
    /// \return IValue_bad if not supported
    virtual IValue const *add(IValue_factory *factory, IValue const *rhs) const = 0;

    /// Subtract.
    ///
    /// \param factory  a factory to create new values.
    /// \param rhs      the right hand operand
    ///
    /// \return IValue_bad if not supported
    virtual IValue const *sub(IValue_factory *factory, IValue const *rhs) const = 0;

    /// Shift left.
    ///
    /// \param factory  a factory to create new values.
    /// \param rhs      the right hand operand
    ///
    /// \return IValue_bad if not supported
    virtual IValue const *shl(IValue_factory *factory, IValue const *rhs) const = 0;

    /// Arithmetic shift right.
    ///
    /// \param factory  a factory to create new values.
    /// \param rhs      the right hand operand
    ///
    /// \return IValue_bad if not supported
    virtual IValue const *asr(IValue_factory *factory, IValue const *rhs) const = 0;

    /// Logical shift right.
    ///
    /// \param factory  a factory to create new values.
    /// \param rhs      the right hand operand
    ///
    /// \return IValue_bad if not supported
    virtual IValue const *lsr(IValue_factory *factory, IValue const *rhs) const = 0;

    /// Xor.
    ///
    /// \param factory  a factory to create new values.
    /// \param rhs      the right hand operand
    ///
    /// \return IValue_bad if not supported
    virtual IValue const *bitwise_xor(IValue_factory *factory, IValue const *rhs) const = 0;

    /// Bitwise Or.
    ///
    /// \param factory  a factory to create new values.
    /// \param rhs      the right hand operand
    ///
    /// \return IValue_bad if not supported
    virtual IValue const *bitwise_or(IValue_factory *factory, IValue const *rhs) const = 0;

    /// BitWise And.
    ///
    /// \param factory  a factory to create new values.
    /// \param rhs      the right hand operand
    ///
    /// \return IValue_bad if not supported
    virtual IValue const *bitwise_and(IValue_factory *factory, IValue const *rhs) const = 0;

    /// Logical Or.
    ///
    /// \param factory  a factory to create new values.
    /// \param rhs      the right hand operand
    ///
    /// \return IValue_bad if not supported
    virtual IValue const *logical_or(IValue_factory *factory, IValue const *rhs) const = 0;

    /// Logical And.
    ///
    /// \param factory  a factory to create new values.
    /// \param rhs      the right hand operand
    ///
    /// \return IValue_bad if not supported
    virtual IValue const *logical_and(IValue_factory *factory, IValue const *rhs) const = 0;

    /// Possible compare results (relations).
    enum Compare_results {
        CR_EQ  = 0x01,                  ///< equal
        CR_LT  = 0x02,                  ///< less than
        CR_LE  = CR_LT | CR_EQ,         ///< less or equal
        CR_GT  = 0x04,                  ///< greater than
        CR_GE  = CR_GT | CR_EQ,         ///< greater or equal
        CR_NE  = CR_GT | CR_LT,         ///< not equal
        CR_O   = CR_LT | CR_EQ | CR_GT, ///< ordered
        CR_UO  = 0x08,                  ///< unordered
        CR_UEQ = CR_UO | CR_EQ,         ///< unordered or equal
        CR_ULT = CR_UO | CR_LT,         ///< unordered or less than
        CR_ULE = CR_UO | CR_LE,         ///< unordered or less or equal
        CR_UGT = CR_UO | CR_GT,         ///< unordered or greater than
        CR_UGE = CR_UO | CR_GE,         ///< unordered or greater or equal
        CR_UNE = CR_UO | CR_NE,         ///< unordered or not equal
        CR_BAD = 0x10                   ///< unknown
    };

    /// Get the inverse compare result, i.e. R^-1.
    ///
    /// \param r  the compare result
    static inline Compare_results inverse(Compare_results r) {
        unsigned x = r & CR_NE;
        if (x == CR_LT || x == CR_GT) {
            // contains either < or >, turn around
            return Compare_results(r ^ CR_NE);
        }
        return r;
    }

    /// Compare.
    ///
    /// \param rhs      the right hand operand
    ///
    /// \return compare results
    ///
    /// \note This method always returns the relation with the best precision, i.e.
    ///       never combined relations like '<=' but either '==' or '<'.
    virtual Compare_results compare(IValue const *rhs) const = 0;

    /// Convert a value into another type.
    ///
    /// \param factory  a factory to create new values.
    /// \param dst_tp   the destination type
    ///
    /// \return IValue_bad if conversion is not supported
    virtual IValue const *convert(IValue_factory *factory, IType const *dst_tp) const = 0;

    /// Returns true if the value is the ZERO (i.e. additive neutral).
    virtual bool is_zero() const = 0;

    /// Returns true if the value is the ONE (i.e. multiplicative neutral).
    ///
    /// \note Returns NEVER true for matrix types, because there are two neutrals.
    virtual bool is_one() const = 0;

    /// Returns true if all components of this value are ONE.
    virtual bool is_all_one() const = 0;

    /// Returns true if this value is finite (i.e. neither Inf nor NaN).
    virtual bool is_finite() const = 0;
};

/// The singleton value of type error.
class IValue_bad : public IValue
{
public:
    /// The kind of this subclass.
    static Kind const s_kind = VK_BAD;

    /// Get the type of this value (always IType_error).
    IType_error const *get_type() const = 0;
};

/// An invalid reference value.
///
/// An invalid reference value describes despite its name "unset" references.
/// Note that these are not \c NULL values, but the result of the default constructor of
/// the given type.
/// So, an \c IValue_invalid_ref of the IType_bsdf references the MDL default bsdf
/// constructor \c bsdf().
class IValue_invalid_ref : public IValue
{
public:
    /// The kind of this subclass.
    static Kind const s_kind = VK_INVALID_REF;

    /// Get the type of this value.
    IType_reference const *get_type() const = 0;
};

/// An atomic value.
class IValue_atomic : public IValue
{
    /// Get the type of this value.
    IType_atomic const *get_type() const = 0;
};

/// A value of type boolean.
class IValue_bool : public IValue_atomic
{
public:
    /// The kind of this subclass.
    static Kind const s_kind = VK_BOOL;

    /// Get the type of this value.
    IType_bool const *get_type() const = 0;

    /// Get the value.
    virtual bool get_value() const = 0;
};

/// An integer based value, base class for IValue_int and IValue_enum.
class IValue_int_valued : public IValue_atomic
{
public:
    /// Get the integer value.
    virtual int get_value() const = 0;
};

/// A value of type integer.
class IValue_int : public IValue_int_valued
{
public:
    /// The kind of this subclass.
    static Kind const s_kind = VK_INT;

    /// Get the type of this value.
    IType_int const *get_type() const = 0;

    /// Get the value.
    virtual int get_value() const = 0;
};

/// A value of type enum.
class IValue_enum : public IValue_int_valued
{
public:
    /// The kind of this subclass.
    static Kind const s_kind = VK_ENUM;

    /// Get the type of this value.
    virtual IType_enum const *get_type() const = 0;

    /// Get the (integer) value of this enum value.
    virtual int get_value() const = 0;

    /// Get the index of this enum value.
    virtual size_t get_index() const = 0;
};

/// A floating-point value.
class IValue_FP : public IValue_atomic
{
public:
    /// Additional classes to differentiate between special FP values.
    enum FP_class {
        FPC_PLUS_ZERO,       ///< The +0.0.
        FPC_MINUS_ZERO,      ///< The -0.0.
        FPC_PLUS_INF,        ///< The +inf.
        FPC_MINUS_INF,       ///< The -inf.
        FPC_NAN,             ///< Is a NaN.
        FPC_NORMAL,          ///< Any other Fp value.
    };

public:
    /// Return the FP class of this value.
    virtual FP_class get_fp_class() const = 0;
};


/// A value of type float.
class IValue_float : public IValue_FP
{
public:
    /// The kind of this subclass.
    static Kind const s_kind = VK_FLOAT;

    /// Get the type of this value.
    IType_float const *get_type() const = 0;

    /// Get the value.
    virtual float get_value() const = 0;
};

/// A value of type double.
class IValue_double : public IValue_FP
{
public:
    /// The kind of this subclass.
    static Kind const s_kind = VK_DOUBLE;

    /// Get the type of this value.
    IType_double const *get_type() const = 0;

    /// Get the value.
    virtual double get_value() const = 0;
};

/// A value of type string.
class IValue_string : public IValue_atomic
{
public:
    /// The kind of this subclass.
    static Kind const s_kind = VK_STRING;

    /// Get the type of this value.
    IType_string const *get_type() const = 0;

    /// Get the value.
    virtual const char *get_value() const = 0;
};

/// A compound value.
///
/// Compound values contain several sub values.
class IValue_compound : public IValue
{
public:
    /// Get the type of this value.
    virtual IType_compound const *get_type() const = 0;

    /// Get the number of components in this compound value.
    virtual int get_component_count() const = 0;

    /// Get the sub value at index.
    ///
    /// \param index  the index
    virtual IValue const *get_value(int index) const = 0;

    /// Get the value by its MDL (field) name.
    ///
    /// \param name  the (field) name
    ///
    /// \return NULL if name does not exists, else return the corresponding value
    virtual IValue const *get_value(char const *name) const = 0;

    /// Return the array of (all compound) values.
    virtual IValue const * const * get_values() const = 0;
};

/// A value of type vector.
class IValue_vector : public IValue_compound
{
public:
    /// The kind of this subclass.
    static Kind const s_kind = VK_VECTOR;

    /// Get the type of this value.
    virtual IType_vector const *get_type() const = 0;
};

/// A value of type matrix.
class IValue_matrix : public IValue_compound
{
public:
    /// The kind of this subclass.
    static Kind const s_kind = VK_MATRIX;

    /// Get the type of this value.
    virtual IType_matrix const *get_type() const = 0;
};

/// A value of type array.
class IValue_array : public IValue_compound
{
public:
    /// The kind of this subclass.
    static Kind const s_kind = VK_ARRAY;

    /// Get the type of this value.
    virtual IType_array const *get_type() const = 0;
};

/// A value of type color represented as RGB values.
class IValue_rgb_color : public IValue_compound
{
public:
    /// The kind of this subclass.
    static Kind const s_kind = VK_RGB_COLOR;

    /// Get the type of this value.
    virtual IType_color const *get_type() const = 0;

    /// Get the value at index.
    ///
    /// \param index  the index
    virtual IValue_float const *get_value(int index) const = 0;
};

/// A value of type struct.
class IValue_struct : public IValue_compound
{
public:
    /// The kind of this subclass.
    static Kind const s_kind = VK_STRUCT;

    /// Get the type of this value.
    virtual IType_struct const *get_type() const = 0;

    /// Get the value of a field.
    ///
    /// \param name     The name of the field.
    /// \returns        The value of the field.
    virtual IValue const *get_field(ISymbol const *name) const = 0;

    /// Get the value of a field.
    ///
    /// \param name     The name of the field.
    /// \returns        The value of the field.
    virtual IValue const *get_field(char const *name) const = 0;
};

/// A string or tag valued resource base class.
///
/// MDL Core supports two kinds of resource values.
/// From MDL source, string-based resources are created. The string values of such resources
/// are MDL URLs.
/// Additionally, tag-based resources are supported. In this kind of resources, the user
/// application must maintain a (tag, version) pair to identify the resource.
/// MDL Core does never change the tag or version, these values are just transparently
/// transported.
class IValue_resource : public IValue
{
public:
    /// Get the type of this value.
    virtual IType_resource const *get_type() const = 0;

    /// Get the string value of this resource, typically an MDL URL.
    virtual char const *get_string_value() const = 0;

    /// Get the tag value.
    virtual int get_tag_value() const = 0;

    /// Get the tag version.
    virtual unsigned get_tag_version() const = 0;
};

/// A texture value.
class IValue_texture : public IValue_resource
{
public:
    /// Possible gamma modes of a texture.
    enum gamma_mode {
        gamma_default,
        gamma_linear,
        gamma_srgb
    };

    /// The kind of BSDF data in case of BSDF data textures (otherwise BDK_NONE).
    /// For BSDF data textures, the string value is an empty string.
    enum Bsdf_data_kind {
        BDK_NONE,

        BDK_SIMPLE_GLOSSY_MULTISCATTER,
        BDK_BACKSCATTERING_GLOSSY_MULTISCATTER,
        BDK_BECKMANN_SMITH_MULTISCATTER,
        BDK_GGX_SMITH_MULTISCATTER,
        BDK_BECKMANN_VC_MULTISCATTER,
        BDK_GGX_VC_MULTISCATTER,
        BDK_WARD_GEISLER_MORODER_MULTISCATTER,
        BDK_SHEEN_MULTISCATTER,

        BDK_LAST_KIND = BDK_SHEEN_MULTISCATTER
    };

    /// The kind of this subclass.
    static Kind const s_kind = VK_TEXTURE;

    /// Get the type of this value.
    virtual IType_texture const *get_type() const = 0;

    /// Get the gamma mode.
    virtual gamma_mode get_gamma_mode() const = 0;

    /// Get the BSDF data kind for BSDF data textures.
    virtual Bsdf_data_kind get_bsdf_data_kind() const = 0;
};

/// A light profile value.
class IValue_light_profile : public IValue_resource
{
public:
    /// The kind of this subclass.
    static Kind const s_kind = VK_LIGHT_PROFILE;

    /// Get the type of this value.
    virtual IType_light_profile const *get_type() const = 0;
};

/// A bsdf measurement value.
class IValue_bsdf_measurement : public IValue_resource
{
public:
    /// The kind of this subclass.
    static Kind const s_kind = VK_BSDF_MEASUREMENT;

    /// Get the type of this value.
    virtual IType_bsdf_measurement const *get_type() const = 0;
};

/// Cast to subtype or return NULL if types do not match.
template<typename T>
T *as(IValue *value) {
    return (value->get_kind() == T::s_kind) ? static_cast<T *>(value) : NULL;
}

/// Cast to subtype or return NULL if types do not match.
template<typename T>
T const *as(IValue const *value) {
    return (value->get_kind() == T::s_kind) ? static_cast<T const *>(value) : NULL;
}

/// Cast to IValue_atomic or return NULL if types do not match.
template<>
inline IValue_atomic *as<IValue_atomic>(IValue *value) {
    switch (value->get_kind()) {
    case IValue::VK_BOOL:
    case IValue::VK_INT:
    case IValue::VK_ENUM:
    case IValue::VK_FLOAT:
    case IValue::VK_DOUBLE:
    case IValue::VK_STRING:
        return static_cast<IValue_atomic *>(value);
    default:
        return NULL;
    }
}

/// Cast to IValue_atomic or return NULL if types do not match.
template<>
inline IValue_atomic const *as<IValue_atomic>(IValue const *value) { //-V659 PVS
    return const_cast<IValue_atomic const *>(as<IValue_atomic>(const_cast<IValue *>(value)));
}

/// Cast to IValue_FP or return NULL if types do not match.
template<>
inline IValue_FP *as<IValue_FP>(IValue *value) {
    switch (value->get_kind()) {
    case IValue::VK_FLOAT:
    case IValue::VK_DOUBLE:
        return static_cast<IValue_FP *>(value);
    default:
        return NULL;
    }
}

/// Cast to IValue_FP or return NULL if types do not match.
template<>
inline IValue_FP const *as<IValue_FP>(IValue const *value) { //-V659 PVS
    return const_cast<IValue_FP const *>(as<IValue_FP>(const_cast<IValue *>(value)));
}

/// Cast to IValue_int_valued or return NULL if types do not match.
template<>
inline IValue_int_valued *as<IValue_int_valued>(IValue *value) {
    switch (value->get_kind()) {
    case IValue::VK_INT:
    case IValue::VK_ENUM:
        return static_cast<IValue_int_valued *>(value);
    default:
        return NULL;
    }
}

/// Cast to IValue_atomic or return NULL if types do not match.
template<>
inline IValue_int_valued const *as<IValue_int_valued>(IValue const *value) { //-V659 PVS
    return
        const_cast<IValue_int_valued const *>(as<IValue_int_valued>(const_cast<IValue *>(value)));
}

/// Cast to IValue_compound or return NULL if types do not match.
template<>
inline IValue_compound *as<IValue_compound>(IValue *value) {
    switch (value->get_kind()) {
    case IValue::VK_VECTOR:
    case IValue::VK_MATRIX:
    case IValue::VK_ARRAY:
    case IValue::VK_RGB_COLOR:
    case IValue::VK_STRUCT:
        return static_cast<IValue_compound *>(value);
    default:
        return NULL;
    }
}

/// Cast to IValue_compound or return null if types do not match.
template<>
inline IValue_compound const *as<IValue_compound>(IValue const *value) { //-V659 PVS
    return const_cast<IValue_compound const *>(as<IValue_compound>(const_cast<IValue *>(value)));
}

/// Cast to IValue_resource or return NULL if types do not match.
template<>
inline IValue_resource *as<IValue_resource>(IValue *value) {
    switch (value->get_kind()) {
    case IValue::VK_TEXTURE:
    case IValue::VK_LIGHT_PROFILE:
    case IValue::VK_BSDF_MEASUREMENT:
        return static_cast<IValue_resource *>(value);
    default:
        return NULL;
    }
}

/// Cast to IValue_resource or return NULL if types do not match.
template<>
inline IValue_resource const *as<IValue_resource>(IValue const *value) { //-V659 PVS
    return as<IValue_resource>(const_cast<IValue *>(value));
}

/// Check if a value is of a certain type.
template<typename T>
bool is(const IValue *value) {
    return as<T>(value) != NULL;
}

/// The interface of a Value factory.
///
/// Every value factory owns all created values. All values are destroyed if
/// the value factory is destroyed.
///
/// An IValue_factory interface can be obtained by calling
/// the method create_value_factory() on the interface IMDL.
class IValue_factory : public Interface_owned
{
public:
    /// Create the bad value.
    ///
    /// \return the singleton \c IValue_bad value
    virtual IValue_bad const *create_bad() = 0;

    /// Create a new value of type boolean.
    ///
    /// \param value  The value of the boolean.
    virtual IValue_bool const *create_bool(bool value) = 0;

    /// Create a new value of type integer.
    ///
    /// \param value  The value of the integer.
    virtual IValue_int const *create_int(int value) = 0;

    /// Create a new value of type enum.
    ///
    /// \param type     The type of the enum.
    /// \param index    The index of the enum constant inside the type.
    virtual IValue_enum const *create_enum(
        IType_enum const *type,
        size_t           index) = 0;

    /// Create a new value of type float.
    ///
    /// \param value  The value of the float.
    virtual IValue_float const *create_float(float value) = 0;

    /// Create a new value of type double.
    ///
    /// \param value  The value of the double.
    virtual IValue_double const *create_double(double value) = 0;

    /// Create a new value of type string.
    ///
    /// \param value  The value of the string, NULL is not allowed.
    virtual IValue_string const *create_string(char const *value) = 0;

    /// Create a new value of type vector.
    ///
    /// \param type    The type of the vector.
    /// \param values  The values for the elements of the vector.
    /// \param size    The number of values, must match the vector size.
    virtual IValue_vector const *create_vector(
        IType_vector const   *type,
        IValue const * const values[],
        size_t               size) = 0;

    /// Create a new value of type matrix.
    ///
    /// \param type    The type of the matrix.
    /// \param values  The values for the elements of the matrix columns.
    /// \param size    The number of values, must match the column size.
    virtual IValue_matrix const *create_matrix(
        IType_matrix const   *type,
        IValue const * const values[],
        size_t               size) = 0;

    /// Create a new value of type array.
    ///
    /// \param type    The type of the array.
    /// \param values  The values for the elements of the array.
    /// \param size    The number of values, must match the array size.
    virtual IValue_array const *create_array(
        IType_array const    *type,
        IValue const * const values[],
        size_t               size) = 0;

    /// Create a new RGB value of type color.
    ///
    /// \param value_r  The (float) value for the red channel.
    /// \param value_g  The (float) value for the green channel.
    /// \param value_b  The (float) value for the blue channel.
    virtual IValue_rgb_color const *create_rgb_color(
        IValue_float const *value_r,
        IValue_float const *value_g,
        IValue_float const *value_b) = 0;

    /// Create a new spectrum value of type color.
    ///
    /// \param wavelengths  The (array of float) values for the wavelengths.
    /// \param amplitudes   The (array of float) values for the amplitudes.
    ///
    /// \return IValue_bad if both arrays have a different size or one is
    ///         is not of type array of floats.
    virtual IValue const *create_spectrum_color(
        IValue_array const *wavelengths,
        IValue_array const *amplitudes) = 0;

    /// Create a new value of type struct.
    ///
    /// \param type    The type of the struct.
    /// \param values  The values for the fields of the struct.
    /// \param size    The number of values, must match the number of fields.
    virtual IValue_struct const *create_struct(
        IType_struct const   *type,
        IValue const * const values[],
        size_t               size) = 0;

    /// Create a new texture value.
    ///
    /// \param type            The type of the texture.
    /// \param value           The string value of the texture, NULL is not allowed.
    /// \param gamma           The gamma override value of the texture.
    /// \param tag_value       The tag value of the texture.
    /// \param tag_version     The version of the tag value.
    virtual IValue_texture const *create_texture(
        IType_texture const            *type,
        const char                     *value,
        IValue_texture::gamma_mode     gamma,
        int                            tag_value,
        unsigned                       tag_version) = 0;

    /// Create a new bsdf_data texture value.
    ///
    /// \param bsdf_data_kind  The BSDF data kind.
    /// \param tag_value       The tag value of the texture.
    /// \param tag_version     The version of the tag value.
    virtual IValue_texture const *create_bsdf_data_texture(
        IValue_texture::Bsdf_data_kind bsdf_data_kind,
        int                            tag_value,
        unsigned                       tag_version) = 0;

    /// Create a new string light profile value.
    ///
    /// \param type         The type of the light profile.
    /// \param value        The string value of the light profile, NULL is not allowed.
    /// \param tag_value    The tag value of the light profile.
    /// \param tag_version  The version of the tag value.
    virtual IValue_light_profile const *create_light_profile(
        IType_light_profile const *type,
        char const                *value,
        int                       tag_value,
        unsigned                  tag_version) = 0;

    /// Create a new string bsdf measurement value.
    ///
    /// \param type         The type of the light profile.
    /// \param value        The string value of the bsdf measurement, NULL is not allowed.
    /// \param tag_value    The tag value of the bsdf measurement.
    /// \param tag_version  The version of the tag value.
    virtual IValue_bsdf_measurement const *create_bsdf_measurement(
        IType_bsdf_measurement const *type,
        char const                   *value,
        int                          tag_value,
        unsigned                     tag_version) = 0;

    /// Create a new invalid reference.
    ///
    /// \param type     The type of the reference.
    virtual IValue_invalid_ref const *create_invalid_ref(
        IType_reference const *type) = 0;

    /// Create a new compound value.
    ///
    /// \param type    The compound type.
    /// \param values  The values for the compound.
    /// \param size    The number of values, must match the number of compound elements.
    virtual IValue_compound const *create_compound(
        IType_compound const *type,
        IValue const * const values[],
        size_t               size) = 0;

    /// Create a additive neutral zero if supported for the given type.
    ///
    /// \param type   The type of the constant to create
    ///
    /// \return A zero constant or IValue_bad if type does not support addition.
    virtual IValue const *create_zero(IType const *type) = 0;

    /// Return the type factory of this value factory.
    virtual IType_factory *get_type_factory() = 0;

    /// Import a value from another value factory.
    ///
    /// \param value  the value to import
    ///
    /// \return a copy of the value owned by this factory
    virtual IValue const *import(IValue const *value) = 0;
};

}  // mdl
}  // mi

#endif
