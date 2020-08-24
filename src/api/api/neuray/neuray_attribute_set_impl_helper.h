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
 ** \brief Header for the Attribute_set_impl_helper implementation.
 **/

#ifndef API_API_NEURAY_NEURAY_ATTRIBUTE_SET_IMPL_HELPER_H
#define API_API_NEURAY_NEURAY_ATTRIBUTE_SET_IMPL_HELPER_H

#include <mi/base/iinterface.h>
#include <mi/base/lock.h>
#include <mi/neuraylib/iattribute_set.h>

#include <string>
#include <base/data/attr/i_attr_types.h>
#include <base/data/db/i_db_journal_type.h>

namespace mi { class IData; class IEnum_decl; class IStructure_decl; }

namespace mi { namespace neuraylib { class ITransaction; } }

namespace MI {

namespace ATTR { class Attribute_set; class Attribute; class Type; }

namespace NEURAY {

class IAttribute_context;
class IDb_element;

/// The static methods of this class contain the actual implementation of IAttribute_set.
///
/// The methods in Attribute_set_impl just forward all calls to the corresponding static method
/// in this class. This split allows to do the actual implementation as a non-template which
/// (1) avoids the dependencies of all scene elements on the implementation of the IAttribute_set
/// methods and (2) significantly reduces the size of the object code ( ~10% less for
/// api/api/neuray/libneuray.so).
///
/// \note The term "simple" attribute type does not coincide with the #mi::IData_simple interface,
///       see (1a) in README.attributes for details.
class Attribute_set_impl_helper
{
public:

    // public API methods of mi::neuraylib::IAttribute_set

    static mi::IData* create_attribute(
        ATTR::Attribute_set* attribute_set,
        IDb_element* db_element,
        const char* name, const char* type_name,
        bool skip_type_check);

    static bool destroy_attribute(
        ATTR::Attribute_set* attribute_set,
        IDb_element* db_element,
        const char* name);

    static const mi::IData* access_attribute(
        const ATTR::Attribute_set* attribute_set,
        const IDb_element* db_element,
        const char* name);

    static mi::IData* edit_attribute(
        ATTR::Attribute_set* attribute_set,
        IDb_element* db_element,
        const char* name);

    static bool is_attribute(
        const ATTR::Attribute_set* attribute_set,
        const IDb_element* db_element,
        const char* name);

    /// Returns \c "" in case of errors.
    static std::string get_attribute_type_name(
        const ATTR::Attribute_set* attribute_set,
        const IDb_element* db_element,
        const char* name);

    static mi::Sint32 set_attribute_propagation(
        ATTR::Attribute_set* attribute_set,
        IDb_element* db_element,
        const char* name,
        mi::neuraylib::Propagation_type value);

    static mi::neuraylib::Propagation_type get_attribute_propagation(
        const ATTR::Attribute_set* attribute_set,
        const IDb_element* db_element,
        const char* name);

    static const char* enumerate_attributes(
        const ATTR::Attribute_set* attribute_set,
        const IDb_element* db_element,
        mi::Sint32 index);

    // internal methods

    /// Returns the attribute (or part of the attribute) identified by a name.
    ///
    /// \param owner            The attribute context references the DB element and the ATTR
    ///                         attribute to use.
    /// \param attribute_name   The name of the attribute, or the name that identifies a part
    ///                         of the attribute.
    /// \return                 The attribute, or \c NULL in case of failure.
    static mi::IData* get_attribute(
        const IAttribute_context* owner, const std::string& attribute_name);

    /// The mill variant re-uses get_attribute(), get_attribute_type(), and
    /// get_attribute_type_name() from this class.
    friend class Mill_attribute_set_impl_helper;

private:

    /// Returns the attribute (or part of the attribute) identified by a name.
    ///
    /// This method contains the actual implementation. It uses low-level parameters because it is
    /// is re-used by the mill variant of this class.
    ///
    /// \param owner            The attribute context references the DB element and the ATTR
    ///                         attribute to use.
    /// \param attribute_name   The name of the attribute, or the name that identifies a part
    ///                         of the attribute.
    /// \param attribute_type   The type of the attribute, or the type of the part of the
    ///                         attribute.
    /// \param pointer          The address of the actual data.
    /// \return                 The attribute, or \c NULL in case of failure.
    static mi::IData* get_attribute(
        mi::neuraylib::ITransaction* transaction,
        const mi::base::IInterface* owner,
        const std::string& attribute_name,
        const ATTR::Type* attribute_type,
        void* pointer);

    /// Returns the ATTR type for an API type name.
    ///
    /// \param type_name           An API type name.
    /// \param name                The name of the attribute. Used as the name of the top-level
    ///                            node (note: special case handling for arrays required).
    /// \return                    The corresponding ATTR type name, or one with ATTR::TYPE_UNDEF as
    ///                            ATTR type code of the top-level node.
    static ATTR::Type get_attribute_type(
        const std::string& type_name, const std::string& name);

    /// Returns the API type name of an ATTR type.
    ///
    /// If nodes of type TYPE_STRUCT come with a struct type name, this name is used if there is a
    /// registered structure declaration that matches. Otherwise, an artificial struct name using
    /// C-like notation will be made up.
    ///
    /// \param type                An ATTR type.
    /// \param ignore_array_size   Indicates whether the value of get_array_size() of the root node
    ///                            shall get ignored. This is important for recursive calls of this
    ///                            method. The parameter should never be set by external callers.
    /// \return                    The corresponding API type name, or \c "" in case of errors.
    static std::string get_attribute_type_name(
        const ATTR::Type& type, bool ignore_array_size = false);

    /// Returns a registered structure declaration for an ATTR type.
    ///
    /// \param type                An ATTR type which top-level node is of type TYPE_STRUCT. The
    ///                            array size in the top-level node is ignored.
    /// \return                    The corresponding registered structure declaration, or \c NULL
    ///                            if none is registered.
    static const mi::IStructure_decl* get_structure_decl( const ATTR::Type& type);

    /// Returns a registered enum declaration for an ATTR type.
    ///
    /// \param type                An ATTR type which top-level node is of type TYPE_ENUM. The
    ///                            array size in the top-level node is ignored.
    /// \return                    The corresponding registered enum declaration, or \c NULL
    ///                            if none is registered.
    static const mi::IEnum_decl* get_enum_decl( const ATTR::Type& type);

    /// Creates a structure declaration for an ATTR type.
    ///
    /// This method assumes that declarations for nested structures have already been registered.
    ///
    /// \param type                An ATTR type which top-level node is of type TYPE_STRUCT. The
    ///                            array size in the top-level node is ignored.
    /// \return                    The corresponding structure declaration, or \c NULL in case of
    ///                            failure.
    static const mi::IStructure_decl* create_structure_decl( const ATTR::Type& type);

    /// Creates an enum declaration for an ATTR type.
    ///
    /// This method assumes that declarations for nested enums have already been registered.
    ///
    /// \param type                An ATTR type which top-level node is of type TYPE_ENUM. The
    ///                            array size in the top-level node is ignored.
    /// \return                    The corresponding enum declaration, or \c NULL in case of
    ///                            failure.
    static const mi::IEnum_decl* create_enum_decl( const ATTR::Type& type);

    /// Registers all structure and enum declarations for an ATTR type.
    ///
    /// Locks #m_register_decls_lock and calls #register_decls_locked().
    static void register_decls( const ATTR::Type& type);

    /// Registers all structure and enum declarations for an ATTR type.
    ///
    /// The method traverses the ATTR type and registers a structure or enum declaration for each
    /// TYPE_STRUCT and TYPE_ENUM node (unless there is already one registered). For structures and
    /// enums, the type name stored in ATTR::Type is used if it does not contradict an already
    /// registered structure or enum declaration and is fully qualified, i.e., it starts with "::"
    /// (to prevent name clashes or confusingly similar names, like "Float32" vs. "float").
    ///
    /// \param type                An ATTR type.
    static void register_decls_locked( const ATTR::Type& type);

    /// Checks whether an ATTR type matches a structure declaration.
    ///
    /// In an ideal world this method would not be needed, as the API enforces that type names are
    /// unique. However, ATTR does not, and internal code might reuse type names for different
    /// types, or use type names in a different way as the API.
    ///
    /// \param type                An ATTR type which top-level node is of type TYPE_STRUCT. The
    ///                            array size in the top-level node is ignored.
    /// \param decl                The structure declaration to compare against.
    /// \return                    \c true if the ATTR type and the structure declaration match,
    ///                            \c false otherwise.
    static bool type_matches_structure_decl(
        const ATTR::Type& type, const mi::IStructure_decl* decl);

    /// Checks whether an ATTR type matches an enum declaration.
    ///
    /// In an ideal world this method would not be needed, as the API enforces that type names are
    /// unique. However, ATTR does not, and internal code might reuse type names for different
    /// types, or use type names in a different way as the API.
    ///
    /// \param type                An ATTR type which top-level node is of type TYPE_ENUM. The
    ///                            array size in the top-level node is ignored.
    /// \param decl                The enum declaration to compare against.
    /// \return                    \c true if the ATTR type and the enum declaration match,
    ///                            \c false otherwise.
    static bool type_matches_enum_decl(
        const ATTR::Type& type, const mi::IEnum_decl* decl);

    /// Indicates whether the type of the attribute can be represented by the API.
    ///
    /// \param attribute   An ATTR attribute.
    /// \return            \c true if the attribute has a valid API type,
    ///                    \c false otherwise.
    static bool has_valid_api_type( const ATTR::Attribute* attribute);

    /// Checks whether the type for certain attributes is correct.
    ///
    /// It is correct if there is a structure declaration registered for \p type name, and the
    /// declarations for \p type_name and "Approx" have the same layout. Strictly speaking, the
    /// same members (names and types) would be sufficient, but the internal code in
    /// TRAVERSE::collect_object_attributes() expects a certain layout without checking.
    static bool is_correct_type_for_attribute(
        const std::string& attribute_name,
        ATTR::Attribute_id attribute_id,
        const std::string& type_name);

    /// Returns the attribute name for a name that identifies a part of an attribute (or the name
    /// itself if it already identifies the top-level attribute).
    ///
    /// Strips off the first '.' or '[' and everything that follows.
    static std::string get_top_level_name( const char* name);

    /// Compute journal flags for the given attribute and DB element.
    ///
    /// If the given attribute has an attribute specification, use that to obtain the journal
    /// flags.
    static DB::Journal_type compute_journal_flags( const ATTR::Attribute* attr, ATTR::Attribute_id attribute_id);

    /// Lock to synchronize calls to #register_decls().
    static mi::base::Lock s_register_decls_lock;
};

} // namespace NEURAY

} // namespace MI

#endif // API_API_NEURAY_NEURAY_ATTRIBUTE_SET_IMPL_HELPER_H
