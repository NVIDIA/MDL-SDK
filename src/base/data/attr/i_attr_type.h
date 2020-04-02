/***************************************************************************************************
 * Copyright (c) 2008-2020, NVIDIA CORPORATION. All rights reserved.
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
/// \brief The definition of the ATTR::Type

#ifndef BASE_DATA_ATTR_I_ATTR_TYPE_H
#define BASE_DATA_ATTR_I_ATTR_TYPE_H

#include "i_attr_types.h"

#include <base/data/serial/i_serial_serializable.h>
#include <base/system/main/i_module_id.h>
#include <base/system/main/types.h>

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#ifndef DEBUG_NON_INLINE
#define MI_INLINE inline
#else
#define MI_INLINE
#endif

namespace MI {
namespace ATTR {

//==================================================================================================

/// A \c Type is a description of a value and the corresponding memory layout.
///
/// \c Types are used to describe the exact memory layout of the \c Attribute's value.
/// \c Types are also used for other things such as \c IMAGE::Image pixel types. Each
/// \c Type may have a name, like "ambient".
///
/// A \c Type might be connected with or contain other \c Types and hence describe
/// complex values. Internally a \c Type is a normal syntax tree. It can describe
/// a simple type, or a list of simple types (like a struct), or arrays with
/// a fixed or dynamic number of values.
/// Eg
/// \code
///	struct { Scalar x[2]; struct {int a; bool b;}; Vector3[] t; }
/// \endcode
///
/// would be
/// \code
///     type         size_one size_all  type_asize     name     next   child
///     TYPE_SCALAR      4        8         2           "x"     yes     no
///     TYPE_STRUCT      8        8         1            0      yes -+  yes -+
///        TYPE_INT32    4        4         1           "a"     yes  |  no  <+
///        TYPE_BOOLEAN  1        1         1           "b"     no   |  no
///	TYPE_VECTOR3    12        4         0           "t"     no <-+  no
/// \endcode
/// Note how the \c TYPE_STRUCT size is 8, not 5, due to C++ padding. The indented
/// lines indicate that they are children of the struct. The dynamic array
/// has, on 32-bit hosts, size
/// \code
/// 4 == sizeof(Type *);
/// \endcode
/// which would be 8 on 64-bit hosts with 64bit pointers, while the size of one element is 12.
class Type : public SERIAL::Serializable
{
  public:
    /// Constructor.
    /// \param type the actual type, e.g. bool, int, ...
    /// \param name the name of the type, can be 0
    /// \param array_size number of elements, 0=dynamic
    explicit Type(
        Type_code type = TYPE_UNDEF,
        const char* name = 0,
        Uint array_size = 1);
    /// Copy constructor.
    /// \param type the input
    Type(
        const Type& type);
    /// Special copy constructor. This one is added \b temporarily to replace a set_arraysize()
    /// call in impmi_function_decl.cpp, Struct_stack::create_type_from_structure().
    /// \param type the input
    /// \param array_size the size of the array
    Type(
        const Type& type,
        Uint array_size);
    /// Assignment operator for type tree. Makes a deep copy of the type.
    /// \param other the input
    /// \return the reference to this
    Type& operator=(
        const Type& other);

    /// The destructor.
    ~Type();

    /// Compare of equality.
    /// \param other the other \c Type
    /// \return true if the two types incl. subtypes are exactly equivalent, false else. Note that
    /// the names are not compared, though.
    MI_INLINE bool operator==(
        const Type& other) const;
    /// Compare of inequality.
    /// \param other the other \c Type
    /// \return true if the two types incl. subtypes are exactly equivalent, false else. Note that
    /// the names are not compared, though.
    MI_INLINE bool operator!=(
        const Type &other) const;

    /// @name Type's data retrieval
    /// The following functions offer read-access to the \c Type's data.
    //@{
    /// Retrieve the name of the \c Type.
    /// \return the name, could be 0
    MI_INLINE const char* get_name() const;
    /// Retrieve the actual type of the \c Type.
    /// \return the corresponding \c Type_code
    MI_INLINE Type_code get_typecode() const;
    /// Retrieve whether the \c Type is const or not.
    /// \return true, when \c Type is const, false else
    MI_INLINE bool get_const() const;
    /// Retrieve the current array size.
    /// \return array size, 0 means dynamic array
    MI_INLINE Uint get_arraysize() const;
    /// Retrieve the next \c Type in the type hierarchy.
    /// \return const pointer to next \c Type, could be 0
    MI_INLINE const Type* get_next() const;
    /// Retrieve the child \c Type in the type hierarchy.
    /// \return const pointer to child \c Type, could be 0
    MI_INLINE const Type* get_child() const;
    /// Retrieve the next \c Type in the type hierarchy.
    /// \return pointer to next \c Type, could be 0
    MI_INLINE Type* get_next();
    /// Retrieve the child \c Type in the type hierarchy.
    /// \return const pointer to child \c Type, could be 0
    MI_INLINE Type* get_child();
    /// Retrieve the pointer to the enum collection.
    /// \return pointer to enum collection
    MI_INLINE std::vector<std::pair<int, std::string> >** set_enum();
    /// Retrieve the pointer to the enum collection.
    /// \return pointer to enum collection, might be 0
    MI_INLINE std::vector<std::pair<int, std::string> >* get_enum() const;
    //@}

    /// @name Type's data setters
    /// The following functions offer write-access to the \c Type's data. Note that several
    /// functions working on \c TYPE_ARRAY \c Types have special behaviour, e.g. \c set_child()
    /// will cause that both the array size and the name of the array \c Type will be overridden
    /// by their child \c Type counterparts.
    //@{
    /// Set the name.
    /// \param name the new name
    void set_name(
        const char* name);
    /// Set the new type.
    /// \param typecode the new type
    MI_INLINE void set_typecode(
        Type_code typecode);
    /// Set the new constness state.
    /// \param isconst state of constness. \c true means no changed, can be baked into shader
    MI_INLINE void set_const(
        bool isconst=true);
    /// Set the new array size. Note that this works only on array \c Types and will
    /// be ignored otherwise. Dynamic arrays are currently not supported using this member.
    /// \param count number of elements
    void set_arraysize(
        Uint count);
    /// Create new branches in the Type tree by making a deep copy. Note that due to the
    /// deep copy (i.e. no sharing) later changes to next go undetected!
    /// \param next new successor
    void set_next(
        const Type& next);
    /// Create new branches in the Type tree by making a deep copy. Note that due to the
    /// deep copy (i.e. no sharing) later changes to child go undetected!
    /// \param child new substruct
    void set_child(
        const Type& child);
    //@}

    /// get more info about values of this type. The size of this Type includes
    /// the static array members; the size of a Type_code does not. Alignments
    /// specify that a value of this type can be stored only at an address that
    /// is a multiple of the alignment. Eg, doubles have an alignment of 8.

    /// @name Object Type information retrieval
    /// The following functions allow the retrieval of some current \c Type's object data
    /// while the retrieval of the corresponding static data retrievals for a given
    /// \c Type_code does not consider array size, etc.
    //@{
    /// Retrieve alignment of this \c Type without considering its child and next \c Types.
    /// \return alignment excluding child and next types
    MI_INLINE size_t align_one() const;
    /// Retrieve alignment for this \c Type considering its child and next \c Types.
    /// \return alignment including child and next types
    MI_INLINE size_t align_all() const;
    /// Retrieve size of the \c Type object.
    /// \return byte size including static array
    MI_INLINE size_t sizeof_one() const;
    /// Retrieve the overall size.
    /// \return size of object including child and next types
    MI_INLINE size_t sizeof_all() const;

    /// Retrieve the size of one element in the array.
    /// \return size of one element in the array
    MI_INLINE size_t sizeof_elem() const;
    /// Retrieve the number of components. E.g. a \c Type of TYPE_COLOR is made up of 4 scalar
    /// components. Note that in the case of arrays the value of the contained type is returned.
    /// \return number of components
    MI_INLINE Uint component_count() const;
    /// Retrieve its component's type. E.g. a \c Type of TYPE_COLOR is made up of 4 scalar
    /// components. Note that in the case of arrays the value of the contained type is returned.
    /// \return the component's type
    MI_INLINE Type_code component_type() const; 
    /// Retrieve a human-readable name for the \c Type.
    /// \return the \c Type's name, e.g. "vector3"
    virtual const char* type_name() const;
    //@}

    /// @name Static characteristics
    /// The following functions allow access to the Type's class properties.
    //@{
    /// Retrieve the size of one member of \p type. Note that a compound \c Type has
    /// a size of 0 and that arrays are not considered here.
    /// \param type \c Type_code of the \c Type in question
    /// \return size in bytes of one member of \c type
    static size_t sizeof_one(
        Type_code type);
    /// Retrieve the name of a \c Type \p type.
    /// \param type \c Type_code of the \c Type in question
    /// \return \c Type's name, e.g. "vector3"
    static const char* type_name(
        Type_code type);
    /// Retrieve the number of components in \c Type \p type.
    /// \param type \c Type_code of the \c Type in question
    /// \return the number of components
    static Uint component_count(	// number of components in the type
        Type_code type);
    /// Retrieve the base type of the \c Type's components. E.g.
    /// \code
    ///   Type::component_type(TYPE_VECTOR4) == TYPE_SCALAR;
    /// \endcode
    /// \param type \c Type_code of the \c Type in question
    /// \return type of the components
    static Type_code component_type(
        Type_code type);
    /// Retrieve the name of the \c Type.
    /// \param type \c Type_code of the \c Type in question
    /// \return printable name of the type
    static const char *component_name(
        Type_code type);
    //@}

    /// Looking up a type by given a complete name., return the Type of the subtree, and an offset
    /// into a value structure where that Type stores its data. The name must
    /// be a complete path, such as a[2].b if a is a struct array containing b.
    /// Dynamic arrays are not handled because their value is int+ptr. Slow.
    ///
    /// Note that for array elements the returned type is not correct. The method returns
    /// a type tree where the top-level element has the array size of the array itself (and
    /// not 1 as one would expect for a non-nested array). This is due to the fact that the
    /// method returns a pointer to a subtree of the type tree of the attribute itself.
    ///
    /// \param[in]  name the name to look up
    /// \param[out] offs the offset into the value struct
    /// \param[in]  begin add this to the returned offset \p offs
    /// \return the named \c Type of the subtree
    const Type* lookup(
        const char* name,
        Uint* offs = 0,
        Uint begin = 0) const;

    /// Looking up a type by given a complete name., return the Type of the subtree, and an offset
    /// into a value structure where that Type stores its data. The name must
    /// be a complete path, such as a[2].b if a is a struct array containing b.
    /// Slow.
    ///
    /// In contrast to method above, this one also supports dynamic arrays. To make that work
    /// it does not return an offset, but the real address (and for all other types it needs the
    /// base address, since it cannot figure that out).
    ///
    /// For dynamic arrays the method returns the address of the Dynamic_array struct, not the
    /// address of the data this struct points to (otherwise you are not able to find out the
    /// current length or to resize the buffer).
    ///
    /// Note that for array elements the returned type is not correct. The method returns
    /// a type tree where the top-level element has the array size of the array itself (and
    /// not 1 as one would expect for a non-nested array). This is due to the fact that the
    /// method returns a pointer to a subtree of the type tree of the attribute itself.
    ///
    /// \param[in]  name         the name to look up
    /// \param[in]  base_address the base address of the attribute (from Attribute::set_values())
    /// \param[out] ret_address  the real address of the data for the named attribute element
    /// \param[in]  offset       internally used, don't pass anything else except 0 as caller
    /// \return                  the named \c Type of the subtree
    const Type* lookup(
        const char* name,
        const char* base_address,
        const char** ret_address,
        Uint offset = 0) const;

    /// As above, but for mutable value structures.
    const Type* lookup(
        const char* name,
        char* base_address,
        char** ret_address,
        Uint offset = 0) const;

    /// Am I the given subclass? (poor man's dynamic_cast support)
    /// \param id the requested subclass' id
    /// \return whether \c Type is a subclass of \p id
    virtual bool is_type_subclass(
        SERIAL::Class_id id) const;

    /// Create a recursive type description.
    /// Useful for type mismatch messages etc.
    /// \return a string containing a recursive type description
    std::string print() const;


    /// @name Serialization utilities.
    /// The following functions implement the required \c Serialization functionality.
    //@{
    /// Return the approximate size in bytes of the element including all its
    /// substructures. This is used to make decisions about garbage collection.
    /// \return approximate size in bytes of the element
    MI_INLINE size_t get_size() const;
    /// Unique class ID so that the receiving host knows which class to create.
    /// \return unique class ID
    MI_INLINE SERIAL::Class_id get_class_id() const;
    /// Serialize the object to the given serializer including all sub elements.
    /// \param serializer the \c Serializer to use
    /// \return a pointer behind itself (e.g. this + 1) to handle arrays
    const SERIAL::Serializable* serialize(
        SERIAL::Serializer* serializer) const;
    /// Deserialize the object and all sub-objects from the given deserializer.
    /// \param deser the \c Deserializer to use
    /// \return a pointer behind itself (e.g. this + 1) to handle arrays
    SERIAL::Serializable* deserialize(
        SERIAL::Deserializer* deser);	// useful functions for byte streams
    /// Serialize the given data assuming that the data is described by this type.
    /// \param serializer the \c Serializer to use
    /// \param values the actual data to be serialized
    void serialize_data(
        SERIAL::Serializer* serializer,
        char* values) const;
    /// Deserialize data from the given stream assuming that the data is
    /// described by this type. Use the given pointer to write data to.
    /// \param deser the \c Deserializer to use
    /// \param values the actual data to be serialized
    void deserialize_data(
        SERIAL::Deserializer* deser,
        char* values);
    //@}
    /// @name Serialization utilities.
    /// The following function is required for implementing the \c Serialization functionality.
    //@{
    /// Factory function used for deserialization.
    /// \return a newly created \c Type object
    static SERIAL::Serializable* factory();
    //@}
    /// @name Serialization utilities.
    /// The following member is required for implementing the \c Serialization functionality.
    //@{
    /// The \c Type's class id.
    static const SERIAL::Class_id id = ID_TYPE;
    //@}

    /// Dump attribute to info messages.
    /// For debugging only.
    void dump() const;

    /// @name New type string addition.
    //@
    /// Set the identifying type-describing string.
    /// \param str the identifying type-describing string
    void set_type_name(
        const std::string& str);
    /// Retrieve the identifying type-describing string.
    /// \return the identifying type-describing string
    const std::string& get_type_name() const { return m_type_name; }
    //@}

  private:
    /// Shared back-end for the two serializers above.
    /// \param serializer \c Serializer to be used
    /// \param fast using short version w/o names?
    /// \return the pointer to this + 1
    const SERIAL::Serializable* do_serialize(
        SERIAL::Serializer* serializer) const;

    /// The private type info type.
    struct Typeinfo
    {
        const char*	name;				///< the name
        Uint		comp;				///< number of components
        Type_code	base;				///< type of base
        size_t		size;				///< size
    };
    /// The table of all supported types with their type infos.
    static const Typeinfo m_typeinfo[];

    std::string m_name;				///< field name, must be defined
    Uint8 m_typecode;					///< primitive Type_code: bool, int, ...
    bool m_const;					///< immutable value, can hardwire in shd
    bool m_spare;					///< not used
    Uint m_arraysize;					///< number of elements, 0=dynamic
    Type* m_next;					///< if part of struct, next member
    union {
    Type* m_child;					///< if TYPE_STRUCT, list of members
    std::vector<std::pair<int, std::string> >* m_enum;
    };
    std::string m_type_name;                          ///< the unique name of the type

    /// Implementation of the deep copy.
    /// \param other the other \c Type
    void do_the_deep_copy(
        const Type& other);
    /// Helper to copy the enum values.
    void create_enum(
        std::vector<std::pair<int, std::string> >* enum_values);
};

/// Utility helper. Comparing two enum collections.
bool compare_enum_collections(
    const std::vector<std::pair<int, std::string> >& one,
    const std::vector<std::pair<int, std::string> >& other);

//==================================================================================================

/// Find out whether the given \c Type contains data of the given \c Type_code or not. The only
/// reason for this function is that it checks in arrays, too. Currently it does not consider
/// nested arrays, since those are not supported yet - but enhancing it should be easy.
/// \param type the given \c Type
/// \param expected the expected \c Type_code
/// \return does \p type contains data of the given \c Type_code?
bool contains_expected_type(
    const Type& type,
    Type_code expected);


#undef MI_INLINE

}
}

#ifndef DEBUG_NON_INLINE
#define MI_INLINE inline
#include "attr_inline_type.h"
#undef MI_INLINE
#endif

#endif
