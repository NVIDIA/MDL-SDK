/***************************************************************************************************
 * Copyright (c) 2010-2020, NVIDIA CORPORATION. All rights reserved.
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

/** \file
 ** \brief Source for the Type_utilities implementation.
 **/

#include "pch.h"

#include "neuray_class_factory.h"
#include "neuray_type_utilities.h"

#include <mi/base/handle.h>
#include <mi/neuraylib/ienum_decl.h>
#include <mi/neuraylib/istructure_decl.h>
#include <mi/neuraylib/itransaction.h>

#include <base/util/string_utils/i_string_lexicographic_cast.h>
#include <base/lib/log/i_log_logger.h>

namespace MI {

namespace NEURAY {


bool Type_utilities::s_initialized = false;

Type_utilities::Map_name_code Type_utilities::s_map_name_code;

Type_utilities::Map_code_name Type_utilities::s_map_code_name;

mi::base::Lock Type_utilities::s_lock;

bool Type_utilities::is_valid_attribute_type( const std::string& type_name)
{
    return is_valid_simple_attribute_type( type_name)
        || is_valid_enum_attribute_type( type_name)
        || is_valid_array_attribute_type( type_name)
        || is_valid_structure_attribute_type( type_name);
}

bool Type_utilities::is_valid_simple_attribute_type( const std::string& type_name)
{
    return convert_type_name_to_type_code( type_name) != ATTR::TYPE_UNDEF;
}

bool Type_utilities::is_valid_array_attribute_type( const std::string& type_name)
{
    // extract delimiters
    mi::Size left_bracket = type_name.rfind( '[');
    if( left_bracket == std::string::npos)
        return false;
    mi::Size right_bracket = type_name.rfind( ']');
    if( right_bracket != type_name.length() - 1)
        return false;

    // extract length
    std::string length_str = type_name.substr( left_bracket+1, right_bracket-left_bracket-1);
    if( !length_str.empty()) {
        STLEXT::Likely<mi::Size> length_likely
            = STRING::lexicographic_cast_s<mi::Size>( length_str);
        if( !length_likely.get_status())
            return false;
        if( *length_likely.get_ptr() == 0) //-V522 PVS
            return false;
    }

    // extract element type name
    std::string element_type_name = type_name.substr( 0, left_bracket);

    // nested arrays are not supported
    return is_valid_simple_attribute_type( element_type_name)
        || is_valid_enum_attribute_type( element_type_name)
        || is_valid_structure_attribute_type( element_type_name);
}

bool Type_utilities::is_valid_structure_attribute_type( const std::string& type_name)
{
    mi::base::Handle<const mi::IStructure_decl> decl(
        s_class_factory->get_structure_decl( type_name.c_str()));
    if( !decl.is_valid_interface())
        return false;

    mi::Size n = decl->get_length();
    for( mi::Size i = 0; i < n; ++i) {
        const char* type_name = decl->get_member_type_name( i);
        if( !is_valid_attribute_type( type_name))
            return false;
    }

    return true;
}

bool Type_utilities::is_valid_enum_attribute_type( const std::string& type_name)
{
    mi::base::Handle<const mi::IEnum_decl> decl(
        s_class_factory->get_enum_decl( type_name.c_str()));
    return decl.is_valid_interface();
}

ATTR::Type_code Type_utilities::convert_attribute_type_name_to_type_code(
    const std::string& type_name)
{
    if( is_valid_enum_attribute_type( type_name))
        return ATTR::TYPE_ENUM;
    if( is_valid_array_attribute_type( type_name))
        return ATTR::TYPE_ARRAY;
    if( is_valid_structure_attribute_type( type_name))
        return ATTR::TYPE_STRUCT;
    if( !is_valid_simple_attribute_type( type_name))
        return ATTR::TYPE_UNDEF;
    ATTR::Type_code type_code = convert_type_name_to_type_code( type_name);
    ASSERT( M_NEURAY_API, type_code != ATTR::TYPE_UNDEF);
    return type_code;
}

const char* Type_utilities::convert_type_code_to_attribute_type_name( ATTR::Type_code type_code)
{
    ASSERT( M_NEURAY_API, type_code != ATTR::TYPE_UNDEF);
    ASSERT( M_NEURAY_API, type_code != ATTR::TYPE_ENUM);
    ASSERT( M_NEURAY_API, type_code != ATTR::TYPE_ARRAY);
    ASSERT( M_NEURAY_API, type_code != ATTR::TYPE_STRUCT);

    const char* type_name = convert_type_code_to_type_name( type_code);
    if( !type_name || !is_valid_attribute_type( type_name))
        return nullptr;
    return type_name;
}

std::string Type_utilities::get_attribute_array_element_type_name( const std::string& type_name)
{
    ASSERT( M_NEURAY_API, is_valid_array_attribute_type( type_name));
    mi::Size left_bracket = type_name.rfind( '[');
    ASSERT( M_NEURAY_API, left_bracket != std::string::npos);
    return type_name.substr( 0, left_bracket);
}

mi::Size Type_utilities::get_attribute_array_length( const std::string& type_name)
{
    ASSERT( M_NEURAY_API, is_valid_array_attribute_type( type_name));
    if( type_name.substr( type_name.length()-2, 2) == "[]")
        return 0;
    mi::Size left_bracket = type_name.rfind( '[');
    ASSERT( M_NEURAY_API, left_bracket != std::string::npos);
    mi::Size right_bracket = type_name.rfind( ']');
    ASSERT( M_NEURAY_API, right_bracket != std::string::npos);
    std::string length = type_name.substr( left_bracket+1, right_bracket-left_bracket-1);
    STLEXT::Likely<mi::Size> length_likely = STRING::lexicographic_cast_s<mi::Size>( length);
    ASSERT( M_NEURAY_API, length_likely.get_status());
    ASSERT( M_NEURAY_API, *length_likely.get_ptr() > 0); //-V522 PVS
    return *length_likely.get_ptr();
}


std::string Type_utilities::strip_array( const std::string& type_name, mi::Size& length)
{
    // extract delimiters
    mi::Size left_bracket = type_name.rfind( '[');
    if( left_bracket == std::string::npos)
        return "";
    mi::Size right_bracket = type_name.rfind( ']');
    if( right_bracket != type_name.length() - 1)
        return "";

    // extract length
    std::string length_str = type_name.substr( left_bracket+1, right_bracket-left_bracket-1);
    if( !length_str.empty()) {
        STLEXT::Likely<mi::Size> length_likely
            = STRING::lexicographic_cast_s<mi::Size>( length_str);
        if( !length_likely.get_status())
            return "";
        if( *length_likely.get_ptr() == 0) //-V522 PVS
            return "";
        length = *length_likely.get_ptr();
    } else
        length = 0;

    // extract element type name
    return type_name.substr( 0, left_bracket);
}

std::string Type_utilities::strip_map( const std::string& type_name)
{
    if( type_name.substr( 0, 4) != "Map<")
        return "";
    if( type_name[type_name.size()-1] != '>')
        return "";
    return type_name.substr( 4, type_name.size() - 5);
}

std::string Type_utilities::strip_pointer( const std::string& type_name)
{
    if( type_name.substr( 0, 8) != "Pointer<")
        return "";
    if( type_name[type_name.size()-1] != '>')
        return "";
    return type_name.substr( 8, type_name.size() - 9);
}

std::string Type_utilities::strip_const_pointer( const std::string& type_name)
{
    if( type_name.substr( 0, 14) != "Const_pointer<")
        return "";
    if( type_name[type_name.size()-1] != '>')
        return "";
    return type_name.substr( 14, type_name.size() - 15);
}

bool Type_utilities::compatible_types(
    const std::string& lhs, const std::string& rhs, bool relaxed_array_check)
{
    if( lhs == rhs)
        return true;

    // Typically, these method is use for structure types, so check this case first.

    // compare structures
    mi::base::Handle<const mi::IStructure_decl> lhs_structure_decl(
        s_class_factory->get_structure_decl( lhs.c_str()));
    if( lhs_structure_decl.is_valid_interface()) {
        mi::base::Handle<const mi::IStructure_decl> rhs_structure_decl(
            s_class_factory->get_structure_decl( rhs.c_str()));
        return rhs_structure_decl.is_valid_interface()
            && compatible_structure_types(
                lhs_structure_decl.get(), rhs_structure_decl.get(), relaxed_array_check);
    }

    // compare enums
    mi::base::Handle<const mi::IEnum_decl> lhs_enum_decl(
        s_class_factory->get_enum_decl( lhs.c_str()));
    if( lhs_enum_decl.is_valid_interface()) {
        mi::base::Handle<const mi::IEnum_decl> rhs_enum_decl(
            s_class_factory->get_enum_decl( rhs.c_str()));
        return rhs_enum_decl.is_valid_interface()
            && compatible_enum_types(
                lhs_enum_decl.get(), rhs_enum_decl.get(), relaxed_array_check);
    }

    // compare arrays
    mi::Size lhs_length;
    const std::string& lhs_array_element = strip_array( lhs, lhs_length);
    if( !lhs_array_element.empty()) {
        mi::Size rhs_length;
        const std::string& rhs_array_element = strip_array( rhs, rhs_length);
        if( lhs_length != rhs_length && (!relaxed_array_check || rhs_length > 0))
            return false;
        return compatible_types( lhs_array_element, rhs_array_element, relaxed_array_check);
    }

    // compare maps
    const std::string& lhs_map_value = strip_map( lhs);
    if( !lhs_map_value.empty()) {
        const std::string& rhs_map_value = strip_map( rhs);
        return compatible_types( lhs_map_value, rhs_map_value, relaxed_array_check);
    }

    // compare pointers
    const std::string& lhs_pointer_nested = strip_pointer( lhs);
    if( !lhs_pointer_nested.empty()) {
        const std::string& rhs_pointer_nested = strip_pointer( rhs);
        return compatible_types( lhs_pointer_nested, rhs_pointer_nested, relaxed_array_check);
    }

    // compare const pointers
    const std::string& lhs_const_pointer_nested = strip_const_pointer( lhs);
    if( !lhs_const_pointer_nested.empty()) {
        const std::string& rhs_const_pointer_nested = strip_const_pointer( rhs);
        return compatible_types(
            lhs_const_pointer_nested, rhs_const_pointer_nested, relaxed_array_check);
    }

    return false;
}

#ifdef ENABLE_ASSERT
void Type_utilities::check_type_name(
    mi::neuraylib::ITransaction* transaction, const char* type_name)
{
    if( !type_name)
        return;

    mi::base::Handle<mi::IData> tmp( transaction->create<mi::IData>( type_name));
    ASSERT( M_NEURAY_API, tmp.is_valid_interface());
}
#else // ENABLE_ASSERT
// see .h file
#endif // ENABLE_ASSERT


ATTR::Type_code Type_utilities::convert_type_name_to_type_code( const std::string& type_name)
{
    mi::base::Lock::Block block( &s_lock);
    if( !s_initialized)
        init();

    if(    type_name == "Ref<Texture>"
        || type_name == "Ref<Lightprofile>"
        || type_name == "Ref<Bsdf_measurement>") {
        LOG::mod_log->error( M_NEURAY_API, LOG::Mod_log::C_DATABASE,
            "Using attributes of type \"%s\" is no longer supported. Use type \"Ref\" instead.",
            type_name.c_str());
        return ATTR::TYPE_UNDEF;
    }

    Map_name_code::const_iterator it = s_map_name_code.find( type_name);
    if( it == s_map_name_code.end())
        return ATTR::TYPE_UNDEF;
    return it->second;
}

ATTR::Type_code Type_utilities::convert_type_name_to_type_code( const char* type_name)
{
    if( !type_name)
        return ATTR::TYPE_UNDEF;
    return convert_type_name_to_type_code( std::string( type_name));
}

const char* Type_utilities::convert_type_code_to_type_name( ATTR::Type_code type_code)
{
    mi::base::Lock::Block block( &s_lock);
    if( !s_initialized)
        init();
    Map_code_name::const_iterator it = s_map_code_name.find( type_code);
    if( it == s_map_code_name.end())
        return nullptr;

    if(    type_code == ATTR::TYPE_TEXTURE
        || type_code == ATTR::TYPE_LIGHTPROFILE
        || type_code == ATTR::TYPE_BSDF_MEASUREMENT) {
        LOG::mod_log->error( M_NEURAY_API, LOG::Mod_log::C_DATABASE,
            "Using attributes of type \"%s\" is deprecated. Use type \"Ref\" instead.",
            it->second.c_str());
        return nullptr;
    }

    return it->second.c_str();
}


bool Type_utilities::compatible_structure_types(
    const mi::IStructure_decl* lhs, const mi::IStructure_decl* rhs, bool relaxed_array_check)
{
    ASSERT( M_NEURAY_API, lhs);
    ASSERT( M_NEURAY_API, rhs);

    mi::Size n = lhs->get_length();
    if( rhs->get_length() != n)
        return false;

    for( mi::Size i = 0; i < n; ++i) {
        const char* lhs_member_name = lhs->get_member_name( i);
        const char* rhs_member_name = rhs->get_member_name( i);
        if( strcmp( lhs_member_name, rhs_member_name) != 0)
            return false;
        const char* lhs_member_type_name = lhs->get_member_type_name( i);
        const char* rhs_member_type_name = rhs->get_member_type_name( i);
        // The next check is included in the second check, but we want to avoid recursion for
        // this very common case.
        if( strcmp( lhs_member_type_name, rhs_member_type_name) == 0)
            continue;
        if( compatible_types( lhs_member_type_name, rhs_member_type_name, relaxed_array_check))
            continue;
        return false;
    }

    return true;
}

bool Type_utilities::compatible_enum_types(
    const mi::IEnum_decl* lhs, const mi::IEnum_decl* rhs, bool relaxed_array_check)
{
    ASSERT( M_NEURAY_API, lhs);
    ASSERT( M_NEURAY_API, rhs);

    mi::Size n = lhs->get_length();
    if( rhs->get_length() != n)
        return false;

    for( mi::Size i = 0; i < n; ++i) {
        const char* lhs_name = lhs->get_name( i);
        const char* rhs_name = rhs->get_name( i);
        if( strcmp( lhs_name, rhs_name) != 0)
            return false;
        mi::Sint32 lhs_value = lhs->get_value( i);
        mi::Sint32 rhs_value = rhs->get_value( i);
        if( lhs_value != rhs_value)
            return false;
    }

    return true;
}


void Type_utilities::register_mapping( const std::string& type_name, ATTR::Type_code type_code)
{
    ASSERT( M_NEURAY_API, s_map_name_code.find( type_name) == s_map_name_code.end());
    s_map_name_code[type_name] = type_code;
    ASSERT( M_NEURAY_API, s_map_code_name.find( type_code) == s_map_code_name.end());
    s_map_code_name[type_code] = type_name;
}

void Type_utilities::init()
{
    if( s_initialized)
        return;

    // Types with a one-to-one mapping.
    register_mapping( "Boolean",               ATTR::TYPE_BOOLEAN);
    register_mapping( "Sint8",                 ATTR::TYPE_INT8);
    register_mapping( "Sint16",                ATTR::TYPE_INT16);
    register_mapping( "Sint32",                ATTR::TYPE_INT32);
    register_mapping( "Sint64",                ATTR::TYPE_INT64);
    register_mapping( "Float32",               ATTR::TYPE_SCALAR);
    register_mapping( "Float64",               ATTR::TYPE_DSCALAR);
    register_mapping( "String",                ATTR::TYPE_STRING);
    register_mapping( "Ref",                   ATTR::TYPE_TAG);
    register_mapping( "Float32<2>",            ATTR::TYPE_VECTOR2);
    register_mapping( "Float32<3>",            ATTR::TYPE_VECTOR3);
    register_mapping( "Float32<4>",            ATTR::TYPE_VECTOR4);
    register_mapping( "Float64<2>",            ATTR::TYPE_DVECTOR2);
    register_mapping( "Float64<3>",            ATTR::TYPE_DVECTOR3);
    register_mapping( "Float64<4>",            ATTR::TYPE_DVECTOR4);
    register_mapping( "Boolean<2>",            ATTR::TYPE_VECTOR2B);
    register_mapping( "Boolean<3>",            ATTR::TYPE_VECTOR3B);
    register_mapping( "Boolean<4>",            ATTR::TYPE_VECTOR4B);
    register_mapping( "Sint32<2>",             ATTR::TYPE_VECTOR2I);
    register_mapping( "Sint32<3>",             ATTR::TYPE_VECTOR3I);
    register_mapping( "Sint32<4>",             ATTR::TYPE_VECTOR4I);
    register_mapping( "Float32<2,2>",          ATTR::TYPE_MATRIX2X2);
    register_mapping( "Float32<2,3>",          ATTR::TYPE_MATRIX2X3);
    register_mapping( "Float32<2,4>",          ATTR::TYPE_MATRIX2X4);
    register_mapping( "Float32<3,2>",          ATTR::TYPE_MATRIX3X2);
    register_mapping( "Float32<3,3>",          ATTR::TYPE_MATRIX3X3);
    register_mapping( "Float32<3,4>",          ATTR::TYPE_MATRIX3X4);
    register_mapping( "Float32<4,2>",          ATTR::TYPE_MATRIX4X2);
    register_mapping( "Float32<4,3>",          ATTR::TYPE_MATRIX4X3);
    register_mapping( "Float32<4,4>",          ATTR::TYPE_MATRIX);
    register_mapping( "Float64<2,2>",          ATTR::TYPE_DMATRIX2X2);
    register_mapping( "Float64<2,3>",          ATTR::TYPE_DMATRIX2X3);
    register_mapping( "Float64<2,4>",          ATTR::TYPE_DMATRIX2X4);
    register_mapping( "Float64<3,2>",          ATTR::TYPE_DMATRIX3X2);
    register_mapping( "Float64<3,3>",          ATTR::TYPE_DMATRIX3X3);
    register_mapping( "Float64<3,4>",          ATTR::TYPE_DMATRIX3X4);
    register_mapping( "Float64<4,2>",          ATTR::TYPE_DMATRIX4X2);
    register_mapping( "Float64<4,3>",          ATTR::TYPE_DMATRIX4X3);
    register_mapping( "Float64<4,4>",          ATTR::TYPE_DMATRIX);
    register_mapping( "Color",                 ATTR::TYPE_COLOR);
    register_mapping( "Spectrum",              ATTR::TYPE_SPECTRUM);
    register_mapping( "Color3",                ATTR::TYPE_RGB_FP);

    ASSERT( M_NEURAY_API, !s_initialized);
    s_initialized = true;
}


} // namespace NEURAY

} // namespace MI

