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
 ** \brief Header for the Type_utilities implementation.
 **/

#ifndef API_API_NEURAY_NEURAY_TYPE_UTILITIES_H
#define API_API_NEURAY_NEURAY_TYPE_UTILITIES_H

#include <boost/unordered_map.hpp>
#include <string>

#include <mi/base/lock.h>

#include <boost/core/noncopyable.hpp>
#include <base/lib/log/i_log_assert.h>
#include <base/data/attr/i_attr_types.h>

namespace mi { class IEnum_decl; class IStructure_decl; }

namespace mi { namespace neuraylib { class ITransaction; } }

namespace MI {

namespace NEURAY {

class Transaction;

/// Misc utility functions related to types.
///
/// Most of the functions are related to attribute types and are used to convert between API type
/// names and ATTR type codes. The higher level functions working on ATTR types are in the class
/// Attribute_set_impl_helper.
///
/// Some functions also operate on types (or rather type names) in general and are not specific to
/// attributes.
///
/// \note The term "simple" attribute type does not coincide with the #mi::IData_simple interface,
///       see (1a) in README.attributes for details.
class Type_utilities : public boost::noncopyable
{
public:

    // public API methods

    // (none)

    // internal methods

    /// \name Attribute types
    //@{

    /// Returns whether an API type name indicates a valid attribute type.
    ///
    /// \note Since enum and structure declarations can be unregistered, an attribute type name can
    ///       suddenly become invalid.
    ///
    /// \param type_name   A valid API type name.
    /// \return            \true if \p type_name is a supported attribute type, \c false otherwise.
    static bool is_valid_attribute_type( const std::string& type_name);

    /// Returns whether an API type name indicates a valid simple attribute type.
    ///
    /// See (1a) in README.attributes for the definition of simple attribute type.
    ///
    /// \param type_name   A valid API type name.
    /// \return            \true if \p type_name is a supported simple attribute type,
    ///                    \c false otherwise.
    static bool is_valid_simple_attribute_type( const std::string& type_name);

    /// Returns whether an API type name indicates a valid enum attribute type.
    ///
    /// A type name is a valid enum attribute type name if there is an enum declaration registered
    /// for that type name.
    ///
    /// \note Since enum declarations can be unregistered, an attribute type name can suddenly
    ///       become invalid.
    ///
    /// \param type_name   A valid API type name.
    /// \return            \true if \p type_name is a supported simple enum type,
    ///                    \c false otherwise.
    static bool is_valid_enum_attribute_type( const std::string& type_name);

    /// Returns whether an API type name indicates a valid array attribute type.
    ///
    /// A valid array attribute type name is "T[N]" or "T[]" where N is a positive integer and T is
    /// a valid attribute type name (but no array type).
    ///
    /// \note Since structure and enum declarations can be unregistered, an attribute type name can
    ///       suddenly become invalid.
    ///
    /// \param type_name   A valid API type name.
    /// \return            \true if \p type_name is a supported array attribute type,
    ///                    \c false otherwise.
    static bool is_valid_array_attribute_type( const std::string& type_name);

    /// Returns whether an API type name indicates a valid structure attribute type.
    ///
    /// A type name is a valid structure attribute type name if there is a structure declaration
    /// registered for that type name, and the type names of all members are valid attribute type
    /// names.
    ///
    /// \note Since structure and enum declarations can be unregistered, an attribute type name can
    ///       suddenly become invalid.
    ///
    /// \param type_name   A valid API type name.
    /// \return            \true if \p type_name is a supported structure attribute type,
    ///                    \c false otherwise.
    static bool is_valid_structure_attribute_type( const std::string& type_name);

    /// Converts a API attribute type name into an ATTR type code.
    ///
    /// \param type_name   A valid API attribute type name.
    /// \return            The corresponding ATTR type code, or ATTR::TYPE_UNDEF in case of errors.
    ///                    For arrays, ATTR::TYPE_ARRAY is returned; for structures,
    ///                    ATTR::TYPE_STRUCT is returned.
    static ATTR::Type_code convert_attribute_type_name_to_type_code(
        const std::string& type_name);

    /// Converts an ATTR type code into an API attribute type name.
    ///
    /// ATTR::TYPE_ENUM, ATTR::TYPE_ARRAY, and ATTR::TYPE_STRUCT are not handled (because they do
    /// not contain enough information to convert it into an API attribute type name).
    ///
    /// \param type_code   An ATTR type code.
    /// \return            The corresponding API attribute type name, or \c NULL in case of errors.
    static const char* convert_type_code_to_attribute_type_name( ATTR::Type_code type_code);

    /// Returns the element type name of an attribute array type name.
    ///
    /// The element type name is not checked for validity.
    ///
    /// \pre is_valid_array_attribute_type() returns \c true
    static std::string get_attribute_array_element_type_name( const std::string& type_name);

    /// Returns the length of an attribute array type name.
    ///
    /// The element type name is not checked for validity. Returns 0 for dynamic arrays.
    ///
    /// \pre is_valid_array_attribute_type() returns \c true
    static mi::Size get_attribute_array_length( const std::string& type_name);

    //@}
    /// \name General types
    //@{

    /// Returns the element type name of an array type name, or the empty string in case of failure.
    ///
    /// The length of the array is stored in \p length (only in case of success, zero for dynamic
    /// arrays).
    static std::string strip_array( const std::string& type_name, mi::Size& length);

    /// Returns the value type name of a map type name, or the empty string in case of failure.
    static std::string strip_map( const std::string& type_name);

    /// Returns the nested type name of a pointer type name, or the empty string in case of failure.
    static std::string strip_pointer( const std::string& type_name);

    /// Returns the nested type name of a const pointer type name, or the empty string in case of
    /// failure.
    static std::string strip_const_pointer( const std::string& type_name);

    /// Checks whether two types are compatible.
    ///
    /// Two types are called compatible if they have the same memory layout, that is, the type names
    /// are exchangeable. Therefore, with the exception of structure types, two types are compatible
    /// if their type names are identical (since there is exactly one type name per type).
    ///
    /// \note For simplicity, the method does not check that the type names themselves are valid.
    ///
    /// \param lhs                    Type name of the actual type.
    /// \param rhs                    Type name of the expected type.
    /// \param relaxed_array_check    If \c true, a static array \p lhs is compatible with a dynamic
    ///                               array \p rhs if the element types are compatible (even though
    ///                               the memory layout is different).
    /// \return                       \c true if both types are compatible.
    static bool compatible_types(
        const std::string& lhs, const std::string& rhs, bool relaxed_array_check);

    /// Checks that the passed type is a valid API type name (if assertions are enabled).
    ///
    /// \param transaction   Transaction used to create an instance of the type name.
    /// \param type_name     The type name to check. In this context \c NULL is considered as a
    ///                      valid type name.
    static void check_type_name( mi::neuraylib::ITransaction* transaction, const char* type_name)
#ifdef ENABLE_ASSERT
        ; // see .cpp file
#else // ENABLE_ASSERT
        { }
#endif // ENABLE_ASSERT

    //@}

private:

    /// \name Attribute types
    //@{

    /// Converts a API type name into an ATTR type code.
    ///
    /// \param type_name   A valid API type name.
    /// \return            The corresponding ATTR type code, or ATTR::TYPE_UNDEF in case of errors.
    static ATTR::Type_code convert_type_name_to_type_code( const std::string& type_name);

    /// Converts a API type name into an ATTR type code.
    ///
    /// \param type_name   A valid API type name.
    /// \return            The corresponding ATTR type code, or ATTR::TYPE_UNDEF in case of errors.
    static ATTR::Type_code convert_type_name_to_type_code( const char* type_name);

    /// Converts an ATTR type code into an API type name.
    ///
    /// \param type_code   An ATTR type code.
    /// \return            The corresponding API type name, or \c NULL in case of errors.
    static const char* convert_type_code_to_type_name( ATTR::Type_code type_code);

    //@}
    /// \name General types
    //@{

public:
    /// Checks whether two structure types are compatible.
    ///
    /// \see #compatible_types()
    ///
    /// \param lhs                    Type declaration of the actual structure type.
    /// \param rhs                    Type declaration of the expected structure type.
    /// \param relaxed_array_check    See #compatible_types().
    /// \return                       \c true if both types are compatible.
    static bool compatible_structure_types(
        const mi::IStructure_decl* lhs, const mi::IStructure_decl* rhs, bool relaxed_array_check);

private:
    /// Checks whether two enum types are compatible.
    ///
    /// \see #compatible_types()
    ///
    /// \param lhs                    Type declaration of the actual enum type.
    /// \param rhs                    Type declaration of the expected enum type.
    /// \param relaxed_array_check    See #compatible_types().
    /// \return                       \c true if both types are compatible.
    static bool compatible_enum_types(
        const mi::IEnum_decl* lhs, const mi::IEnum_decl* rhs, bool relaxed_array_check);

    //@}
    /// \name Initialization
    //@{

    /// Initializes the data structures, kind of a lazy constructor for the static members.
    ///
    /// Must only be called if s_initialized is \c false. The caller needs to hold the lock.
    static void init();

    /// Registers a mapping from type name to type code and vice versa.
    ///
    /// Not supposed to be called from any method other that init().
    static void register_mapping( const std::string& type_name, ATTR::Type_code type_code);

    //@}

    /// Indicates whether the data structures have been initialized.
    static bool s_initialized;

    /// A hash function to support enums in std::hash_map.
    template <typename T>
    struct hash {
        int operator()(T type) const {
            return static_cast<int>( type);
        }
    };

    /// The type of s_map_name_code.
    typedef boost::unordered_map<std::string, ATTR::Type_code> Map_name_code;

    /// The type of s_map_name_code.
    typedef boost::unordered_map<
        ATTR::Type_code, std::string, hash<ATTR::Type_code> > Map_code_name;

    /// The map for the mapping from type name to type code.
    static Map_name_code s_map_name_code;

    /// The map for the mapping from type code to type name.
    static Map_code_name s_map_code_name;

    /// Lock to protect map accesses
    static mi::base::Lock s_lock;
};

} // namespace NEURAY

} // namespace MI

#endif // API_API_NEURAY_NEURAY_TYPE_UTILITIES_H

