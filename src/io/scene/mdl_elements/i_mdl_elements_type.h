/***************************************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
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
 **************************************************************************************************/
/// \file
/// \brief      Types of the MDL type system
///
///             For documentation, see the counterparts in the public API
///             in <mi/neuraylib/itype.h>.

#ifndef IO_SCENE_MDL_ELEMENTS_I_MDL_ELEMENTS_TYPE_H
#define IO_SCENE_MDL_ELEMENTS_I_MDL_ELEMENTS_TYPE_H

#include <mi/base/handle.h>
#include <mi/base/interface_declare.h>

#include <string>
#include <utility>
#include <vector>

namespace mi { class IString; }

namespace MI {

namespace SERIAL { class Deserializer; class Serializer; }

namespace MDL {

class IAnnotation_block;

class IType : public
    mi::base::Interface_declare<0xa0949f39,0xbec9,0x458a,0x80,0x04,0xc6,0xcc,0x58,0x79,0x61,0x48>
{
public:
    enum Kind {
        TK_ALIAS,
        TK_BOOL,
        TK_INT,
        TK_ENUM,
        TK_FLOAT,
        TK_DOUBLE,
        TK_STRING,
        TK_VECTOR,
        TK_MATRIX,
        TK_COLOR,
        TK_ARRAY,
        TK_STRUCT,
        TK_TEXTURE,
        TK_LIGHT_PROFILE,
        TK_BSDF_MEASUREMENT,
        TK_BSDF,
        TK_HAIR_BSDF,
        TK_EDF,
        TK_VDF,
        TK_FORCE_32_BIT = 0xffffffffU
    };

    enum Modifier {
        MK_NONE        = 0,
        MK_UNIFORM     = 2,
        MK_VARYING     = 4,
        MK_FORCE_32_BIT
    };

    virtual Kind get_kind() const = 0;

    virtual mi::Uint32 get_all_type_modifiers() const = 0;

    virtual const IType* skip_all_type_aliases() const = 0;

    virtual mi::Size get_memory_consumption() const = 0;
};

mi_static_assert( sizeof( IType::Kind) == sizeof( mi::Uint32));
mi_static_assert( sizeof( IType::Modifier) == sizeof( mi::Uint32));

class IType_alias : public
    mi::base::Interface_declare<0xc110f1ed,0x8385,0x4a61,0xb6,0x1e,0x00,0xe9,0xb8,0x3c,0xf4,0xb1,
                                IType>
{
public:
    static const Kind s_kind = TK_ALIAS;

    virtual const IType* get_aliased_type() const = 0;

    template <class T>
    const T* get_aliased_type() const
    {
        mi::base::Handle<const IType> ptr_type( get_aliased_type());
        if( !ptr_type)
            return nullptr;
        return static_cast<const T*>( ptr_type->get_interface( typename T::IID()));
    }

    virtual mi::Uint32 get_type_modifiers() const = 0;

    virtual const char* get_symbol() const = 0;
};

class IType_atomic : public
    mi::base::Interface_declare<0x5b52eef5,0xe81e,0x43d0,0xb7,0x5e,0xdb,0x05,0x75,0xba,0x85,0x5e,
                                IType>
{
};

class IType_bool : public
    mi::base::Interface_declare<0x49de3825,0xb045,0x431d,0x96,0x13,0x1d,0x88,0xf4,0xcc,0x43,0x17,
                                IType_atomic>
{
public:
    static const Kind s_kind = TK_BOOL;
};

class IType_int : public
    mi::base::Interface_declare<0xf44caa7b,0xeb38,0x4b2e,0x87,0x2b,0x3e,0xf2,0xda,0x23,0x9a,0xda,
                                IType_atomic>
{
public:
    static const Kind s_kind = TK_INT;
};

class IType_enum : public
    mi::base::Interface_declare<0x5f1b6222,0xa0c9,0x44ca,0x89,0x6f,0x92,0x13,0x45,0x4c,0x6d,0x43,
                                IType_atomic>
{
public:
    /// Datastructure used by #IType_factory to register enum types.
    typedef std::vector<std::pair<std::string, mi::Sint32> > Values;
    /// Datastructure used by #IType_factory to register enum types.
    typedef std::vector<mi::base::Handle<const IAnnotation_block> > Value_annotations;

    enum Predefined_id {
        EID_USER           = -1,
        EID_TEX_GAMMA_MODE =  0,
        EID_INTENSITY_MODE =  1,
        EID_FORCE_32_BIT   =  0x7fffffff,
    };

    static const Kind s_kind = TK_ENUM;

    virtual const char* get_symbol() const = 0;

    virtual mi::Size get_size() const = 0;

    virtual const char* get_value_name( mi::Size index) const = 0;

    virtual mi::Sint32 get_value_code( mi::Size index, mi::Sint32* errors = nullptr) const = 0;

    virtual mi::Size find_value( const char* name) const = 0;

    virtual mi::Size find_value( mi::Sint32 code) const = 0;

    virtual Predefined_id get_predefined_id() const = 0;

    virtual const IAnnotation_block* get_annotations() const = 0;

    virtual const IAnnotation_block* get_value_annotations( mi::Size index) const = 0;
};

mi_static_assert( sizeof( IType_enum::Predefined_id) == sizeof( mi::Uint32));

class IType_float : public
    mi::base::Interface_declare<0xc57b325d,0x05eb,0x4dd5,0x81,0xde,0x2e,0xbc,0x1f,0xd1,0x76,0xdc,
                                IType_atomic>
{
public:
    static const Kind s_kind = TK_FLOAT;
};

class IType_double : public
    mi::base::Interface_declare<0x7834017d,0xb3c7,0x4571,0xb3,0x33,0xf5,0x7a,0xb0,0x67,0x2e,0x42,
                                IType_atomic>
{
public:
    static const Kind s_kind = TK_DOUBLE;
};

class IType_string : public
    mi::base::Interface_declare<0xf3d94211,0x6c5f,0x45a1,0xa8,0xef,0x56,0x45,0x20,0x82,0xf6,0x61,
                                IType_atomic>
{
public:
    static const Kind s_kind = TK_STRING;
};

class IType_compound : public
    mi::base::Interface_declare<0xd81be521,0x2508,0x4895,0xa7,0x69,0x34,0xce,0x99,0x91,0x77,0xd9,
                                IType>
{
public:
    virtual const IType* get_component_type( mi::Size index) const = 0;

    template <class T>
    const T* get_component_type( mi::Size index) const
    {
        mi::base::Handle<const IType> ptr_type( get_component_type( index));
        if( !ptr_type)
            return nullptr;
        return static_cast<const T*>( ptr_type->get_interface( typename T::IID()));
    }

    virtual mi::Size get_size() const = 0;
};

class IType_vector : public
    mi::base::Interface_declare<0x68db57ee,0xe5ee,0x4afc,0x96,0x30,0x04,0x7a,0x95,0xb4,0x88,0x9f,
                                IType_compound>
{
public:
    static const Kind s_kind = TK_VECTOR;

    virtual const IType_atomic* get_element_type() const = 0;

    template <class T>
    const T* get_element_type() const
    {
        mi::base::Handle<const IType_atomic> ptr_type( get_element_type());
        if( !ptr_type)
            return nullptr;
        return static_cast<const T*>( ptr_type->get_interface( typename T::IID()));
    }
};

class IType_matrix : public
    mi::base::Interface_declare<0x1aff489c,0xf2d8,0x4b37,0xbb,0x05,0xd1,0x8c,0x7b,0x12,0x28,0x15,
                                IType_compound>
{
public:
    static const Kind s_kind = TK_MATRIX;

    virtual const IType_vector* get_element_type() const = 0;

    template <class T>
    const T* get_element_type() const
    {
        mi::base::Handle<const IType_vector> ptr_type( get_element_type());
        if( !ptr_type)
            return nullptr;
        return static_cast<const T*>( ptr_type->get_interface( typename T::IID()));
    }
};

class IType_color : public
    mi::base::Interface_declare<0x5cd4ff64,0x36f4,0x4d8b,0x98,0xd3,0x91,0x49,0x89,0x63,0xfd,0xec,
                                IType_compound>
{
public:
    static const Kind s_kind = TK_COLOR;
};

class IType_array : public
    mi::base::Interface_declare<0xf5032c18,0xdf05,0x45f8,0xbf,0x1e,0x57,0x42,0x9d,0x4c,0xb8,0xe5,
                                IType_compound>
{
public:
    static const Kind s_kind = TK_ARRAY;

    virtual const IType* get_element_type() const = 0;

    template <class T>
    const T* get_element_type() const
    {
        mi::base::Handle<const IType> ptr_type( get_element_type());
        if( !ptr_type)
            return nullptr;
        return static_cast<const T*>( ptr_type->get_interface( typename T::IID()));
    }

    virtual bool is_immediate_sized() const = 0;

    virtual const char* get_deferred_size() const = 0;
};

class IType_struct : public
    mi::base::Interface_declare<0x3f9c7bab,0x2731,0x49cc,0xa4,0x4e,0xf3,0x96,0x15,0xa7,0x64,0x71,
                                IType_compound>
{
public:
    /// Datastructure used by #IType_factory to register struct types.
    typedef std::vector<
        std::pair<mi::base::Handle<const IType>, std::string> > Fields;
    /// Datastructure used by #IType_factory to register struct types.
    typedef std::vector<mi::base::Handle<const IAnnotation_block> > Field_annotations;

    enum Predefined_id {
        SID_USER               = -1,
        SID_MATERIAL_EMISSION  =  0,
        SID_MATERIAL_SURFACE   =  1,
        SID_MATERIAL_VOLUME    =  2,
        SID_MATERIAL_GEOMETRY  =  3,
        SID_MATERIAL           =  4,
        SID_FORCE_32_BIT       =  0x7fffffff,
    };

    static const Kind s_kind = TK_STRUCT;

    virtual const char* get_symbol() const = 0;

    virtual const IType* get_field_type( mi::Size index) const = 0;

    template <class T>
    const T* get_field_type( mi::Size index) const
    {
        mi::base::Handle<const IType> ptr_type( get_field_type( index));
        if( !ptr_type)
            return nullptr;
        return static_cast<const T*>( ptr_type->get_interface( typename T::IID()));
    }

    virtual const char* get_field_name( mi::Size index) const = 0;

    virtual mi::Size find_field( const char* name) const = 0;

    virtual Predefined_id get_predefined_id() const = 0;

    virtual const IAnnotation_block* get_annotations() const = 0;

    virtual const IAnnotation_block* get_field_annotations( mi::Size index) const = 0;
};

mi_static_assert( sizeof( IType_struct::Predefined_id) == sizeof( mi::Uint32));

class IType_reference : public
    mi::base::Interface_declare<0x2d682296,0x5789,0x42fd,0xa8,0x2a,0x70,0xd9,0x38,0x0f,0x3a,0x8d,
                                IType>
{
};

class IType_resource : public
    mi::base::Interface_declare<0x07644a43,0xc59a,0x4935,0x81,0x2f,0xba,0xbf,0xc0,0x84,0x05,0x0b,
                                IType_reference>
{
};

class IType_texture : public
    mi::base::Interface_declare<0xd5e972c9,0xa40f,0x4a5f,0x99,0xae,0x8a,0x07,0x0c,0x1e,0x72,0xa0,
                                IType_resource>
{
public:
    static const Kind s_kind = TK_TEXTURE;

    enum Shape {
        TS_2D           = 0,
        TS_3D           = 1,
        TS_CUBE         = 2,
        TS_PTEX         = 3,
        TS_BSDF_DATA    = 4,
        TS_FORCE_32_BIT = 0xffffffffU
    };

    virtual Shape get_shape() const = 0;
};

mi_static_assert( sizeof( IType_texture::Shape) == sizeof( mi::Uint32));

class IType_light_profile : public
    mi::base::Interface_declare<0x6bac728e,0x19e3,0x49a3,0xb0,0xbd,0xbc,0x8a,0xbc,0xe2,0x46,0x79,
                                IType_resource>
{
public:
    static const Kind s_kind = TK_LIGHT_PROFILE;
};

class IType_bsdf_measurement : public
    mi::base::Interface_declare<0x80d8946c,0x4c90,0x4cbd,0x9a,0xd4,0x63,0x84,0x26,0xb0,0xb2,0xc7,
                                IType_resource>
{
public:
    static Kind const s_kind = TK_BSDF_MEASUREMENT;
};

class IType_df : public
    mi::base::Interface_declare<0x15a372df,0x2d9b,0x4a0f,0xa8,0x64,0x86,0xb0,0x81,0x9d,0x41,0x10,
                                IType_reference>
{
};

class IType_bsdf : public
    mi::base::Interface_declare<0x05c903b8,0xf274,0x4245,0xaa,0x5f,0xa7,0x46,0xa3,0xd3,0x1e,0xed,
                                IType_df>
{
public:
    static const Kind s_kind = TK_BSDF;
};

class IType_hair_bsdf : public
    mi::base::Interface_declare<0xeb5cd213,0x1484,0x4db4,0xaf,0xdb,0xd6,0xf2,0x91,0x9b,0xd1,0xff,
                                IType_df>
{
public:
    static const Kind s_kind = TK_HAIR_BSDF;
};

class IType_edf : public
    mi::base::Interface_declare<0x13671528,0xe45b,0x4186,0xb0,0x89,0x4a,0x26,0xef,0x9f,0x45,0x79,
                                IType_df>
{
public:
    static const Kind s_kind = TK_EDF;
};

class IType_vdf : public
    mi::base::Interface_declare<0xee69b2f9,0x2fc9,0x40f0,0xbc,0x8a,0xc0,0x98,0xb0,0x30,0xbf,0x97,
                                IType_df>
{
public:
    static const Kind s_kind = TK_VDF;
};

class IType_list : public
    mi::base::Interface_declare<0x3fc44420,0xf157,0x4901,0xa2,0x76,0xec,0xff,0x4e,0xe4,0x96,0x57>
{
public:
    virtual mi::Size get_size() const = 0;

    virtual mi::Size get_index( const char* name) const = 0;

    virtual const char* get_name( mi::Size index) const = 0;

    virtual const IType* get_type( mi::Size index) const = 0;

    template <class T>
    const T* get_type( mi::Size index) const
    {
        mi::base::Handle<const IType> ptr_type( get_type( index));
        if( !ptr_type)
            return nullptr;
        return static_cast<const T*>( ptr_type->get_interface( typename T::IID()));
    }

    virtual const IType* get_type( const char* name) const = 0;

    template <class T>
    const T* get_type( const char* name) const
    {
        mi::base::Handle<const IType> ptr_type( get_type( name));
        if( !ptr_type)
            return nullptr;
        return static_cast<const T*>( ptr_type->get_interface( typename T::IID()));
    }

    virtual mi::Sint32 set_type( mi::Size index, const IType* type) = 0;

    virtual mi::Sint32 set_type( const char* name, const IType* type) = 0;

    virtual mi::Sint32 add_type( const char* name, const IType* type) = 0;

    virtual mi::Size get_memory_consumption() const = 0;
};

class IType_factory : public
    mi::base::Interface_declare<0xfd74477a,0xa6a6,0x4ca4,0x91,0x1b,0x82,0xcb,0x09,0x0c,0x09,0x92>
{
public:
    /// \name Interface of #mi::neuraylib::IType_factory
    //@{

    virtual const IType_alias* create_alias(
        const IType* type, mi::Uint32 modifiers, const char* symbol) const = 0;

    virtual const IType_bool* create_bool() const = 0;

    virtual const IType_int* create_int() const = 0;

    virtual const IType_enum* create_enum( const char* symbol) const = 0;

    virtual const IType_float* create_float() const = 0;

    virtual const IType_double* create_double() const = 0;

    virtual const IType_string* create_string() const = 0;

    virtual const IType_vector* create_vector(
        const IType_atomic* element_type, mi::Size size) const = 0;

    virtual const IType_matrix* create_matrix(
        const IType_vector* column_type, mi::Size columns) const = 0;

    virtual const IType_color* create_color() const = 0;

    virtual const IType_array* create_immediate_sized_array(
        const IType* element_type, mi::Size size) const = 0;

    virtual const IType_array* create_deferred_sized_array(
        const IType* element_type, const char* size) const = 0;

    virtual const IType_struct* create_struct( const char* symbol) const = 0;

    virtual const IType_texture* create_texture( IType_texture::Shape shape) const = 0;

    virtual const IType_light_profile* create_light_profile() const = 0;

    virtual const IType_bsdf_measurement* create_bsdf_measurement() const = 0;

    virtual const IType_bsdf* create_bsdf() const = 0;

    virtual const IType_hair_bsdf* create_hair_bsdf() const = 0;

    virtual const IType_edf* create_edf() const = 0;

    virtual const IType_vdf* create_vdf() const = 0;

    virtual IType_list* create_type_list() const = 0;

    virtual const IType_enum* get_predefined_enum( IType_enum::Predefined_id id) const = 0;

    virtual const IType_struct* get_predefined_struct( IType_struct::Predefined_id id) const = 0;

    virtual mi::Sint32 compare( const IType* lhs, const IType* rhs) const = 0;

    virtual mi::Sint32 compare( const IType_list* lhs, const IType_list* rhs) const = 0;

    virtual mi::Sint32 is_compatible(const IType* src, const IType* dst) const = 0;

    virtual const mi::IString* dump( const IType* type, mi::Size depth = 0) const = 0;

    virtual const mi::IString* dump( const IType_list* list, mi::Size depth = 0) const = 0;

    //@}
    /// \name Type registration
    //@{

    /// Returns an instance of a given enum type.
    ///
    /// If the symbol is already registered, its definition is compared with \p id and \p values. In
    /// case of a match the registered type is returned, otherwise \c NULL. If the symbol is not yet
    /// registered, it will be registered with a type constructed from \p id, \p values,
    /// \p annotations, and \p value_annotations and the just registered type is returned.
    ///
    /// \return
    ///           -  0: Success.
    ///           - -1: Invalid parameters.
    ///           - -2: The given symbol is not a valid MDL identifier.
    ///           - -3: The given symbol is already registered and of a different type (enum vs
    ///                 struct).
    ///           - -4: The given symbol is already registered and of a different type (ID and/or
    ///                 values do not match).
    virtual const IType_enum* create_enum(
        const char* symbol,
        IType_enum::Predefined_id id,
        const IType_enum::Values& values,
        mi::base::Handle<const IAnnotation_block>& annotations,
        const IType_enum::Value_annotations& value_annotations,
        mi::Sint32* errors) = 0;

    ///  Returns an instance of a given struct type.
    ///
    /// If the symbol is already registered, its definition is compared with \p id and \p fields. In
    /// case of a match the registered type is returned, otherwise \c NULL. If the symbol is not yet
    /// registered, it will be registered with a type constructed from \p id, \p fields,
    /// \p annotations, and \p field_annotations and the just registered type is returned.
    ///
    /// \return
    ///           -  0: Success.
    ///           - -1: Invalid parameters.
    ///           - -2: The given symbol is not a valid MDL identifier.
    ///           - -3: The given symbol is already registered and of a different type (enum vs
    ///                 struct).
    ///           - -4: The given symbol is already registered and of a different type (ID and/or
    ///                 fields do not match).
    virtual const IType_struct* create_struct(
        const char* symbol,
        IType_struct::Predefined_id id,
        const IType_struct::Fields& fields,
        mi::base::Handle<const IAnnotation_block>& annotations,
        const IType_struct::Field_annotations& field_annotations,
        mi::Sint32* errors) = 0;

    //@}
    /// \name Serialization
    //@{

    /// Serializes a type to \p serializer.
    virtual void serialize( SERIAL::Serializer* serializer, const IType* type) const = 0;

    /// Deserializes a type from \p deserializer.
    virtual const IType* deserialize( SERIAL::Deserializer* deserializer) = 0;

    /// Deserializes a type from \p deserializer.
    template <class T>
    const T* deserialize( SERIAL::Deserializer* deserializer)
    {
        mi::base::Handle<const IType> ptr_type( deserialize( deserializer));
        if( !ptr_type)
            return nullptr;
        return static_cast<const T*>( ptr_type->get_interface( typename T::IID()));
    }

    /// Serializes a type list to \p serializer.
    virtual void serialize_list(
        SERIAL::Serializer* serializer, const IType_list* type_list) const = 0;

    /// Deserializes a type list from \p deserializer.
    virtual IType_list* deserialize_list( SERIAL::Deserializer* deserializer) = 0;

    //@}
};

/// Returns the global type factory.
IType_factory* get_type_factory();

// See base/lib/mem/i_mem_consumption.h
inline bool has_dynamic_memory_consumption( const IType*) { return true; }
inline size_t dynamic_memory_consumption( const IType* t)
{ return t->get_memory_consumption(); }

inline bool has_dynamic_memory_consumption( const IType_list*) { return true; }
inline size_t dynamic_memory_consumption( const IType_list* l)
{ return l->get_memory_consumption(); }

} // namespace MDL

} // namespace MI

#endif // IO_SCENE_MDL_ELEMENTS_I_MDL_ELEMENTS_TYPE_H
