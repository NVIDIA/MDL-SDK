/***************************************************************************************************
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
 **************************************************************************************************/
 
/// \file
/// \brief Attribute class definition.

#ifndef BASE_DATA_ATTR_I_ATTR_ATTRIBUTE_H
#define BASE_DATA_ATTR_I_ATTR_ATTRIBUTE_H

#include <base/data/db/i_db_tag.h>
#include <base/data/serial/i_serial_serializable.h>
#include <base/lib/cont/i_cont_array.h>
#include <base/system/main/types.h>
#include <mi/math/color.h>

#include "i_attr_type.h"
#include "i_attr_types.h"
#include "i_attr_type_code_traits.h"

namespace MI {
namespace ATTR {

/// Attachments are stored in Attributes to define shaders that provide values for the attribute.
/// They have the form "a" = <shadertag>."b", where "a" is the name of the attribute member field,
/// <shadertag> is the shader to call, and "b" is the name of a field in the return struct of the
/// shader. If the whole attribute value is assigned and not a struct subfield, "a" is a null
/// pointer. If the whole shader return is assigned, "b" is a null pointer. The attribute can also
/// be attached to a phenomenon interface, in which case "b" is the name of a phenomenon interface
/// field, and m_is_interface is true. "a" and "b" may be structured too, as in
/// "tex[5].substruct.field"; in such structures it is possible to chain more than one attachment.
///
/// \todo For now, the safe and basic approach is copying Attachments around all of the time. This
/// will result in a performance hit and has to be fixed as soon as the ownership/object lifetime
/// issues are clear to optimize Attachment handling appropriately.
///
/// An attachment can be made to the entire attribute or to a struct member or list element. When
/// the attachment is made to an element of an attribute, the m_member_name field describes the
/// member being attached.
///
/// For example if the attribute is a list attribute and the attachment is made to a list element,
/// m_member_name will begin with "[n]" where 'n' is the index of the list element.
///
/// If the attribute is a struct and the attachment is made to a struct member, m_member_name
/// will begin with the member name.
///
/// Combinations can occur with lists and/or nested structs. The following are
/// all valid example values for m_member_name:
/// "[3].diffuse"      -- Refers to the 'diffuse' struct member of the 3rd list
///                       element
/// "texture.map"     --  Refers to a sub-struct member named map. 'texture' is
///                       also a struct member. The original declaration for the
///                       parameter might have looked something like:
///
/// \code
///                       struct "my_parameter" {
///                         struct "texture" {
///                           texture map
///                           scalar weight
///                         }
///                         Color some_other_member
///                       }
/// \endcode
///                       In this example "my_parameter" is the attribute.
///
/// The m_target Tag refers to the shader that is the target of the attachment. When the attachment
/// is made to a phenomenon interface parameter m_is_interface is true. Otherwise the target of
/// the attachment is the output of a shader.
///
/// The m_target_name string identifies a sub-element of the target if the the attachment is not
/// being made to the entire target. The syntax for this string is the same as described for the
/// m_member_name string.
///
/// Attachments can be chained to describe multiple attachments to struct members or list elements.
/// Each Attachment overrides the literal value provided in the attribute's m_values field.
struct Attachment
{
    Attachment();

    std::string m_member_name;			///< ""=whole attr, else target field
    DB::Tag m_target;					///< src shader that provides values
    std::string m_target_name;			///< ""=whole return,else result field
    bool m_is_interface;				///< target is a phen interface param
};


/// An Attribute represents a value and can be attached to an Element.
///
/// The stored value can be interpreted with the accompanying Type, which describes the exact
/// layout. An attribute name may be freely chosen, such as "material". It is stored in the
/// (root) Type. That name corresponds to a key id (which is a hash of the name) for faster
/// manipulation.
///
/// The type of the Attribute can be any Type tree, including array types. In addition the
/// Attribute may be a list of such Type elements; for example, a motion path Attribute has
/// elements of type Vector3[15] (for path length 15), with a list of one such element per vertex.
class Attribute : public SERIAL::Serializable
{
  public:
    /// 3D vector type
    typedef mi::math::Vector<Scalar,3> Vector3;

    /// 4D vector type
    typedef mi::math::Vector<Scalar,4> Vector4;

    /// Create and destroy an attribute. Attributes are constructed with zero-filled values arrays
    /// that can later be accessed with get/set_value*. For on-demand attributes the list_size
    /// (and the values pointer) is initially 0 until the 'execute' function fills it.
    /// This constructor assumes that the root of type contains the attribute name
    /// and will compute the attribute id from that.
    /// \param type data type, may be tree but not list
    /// \param override inheritance: parent overrides child
    explicit Attribute(
        const Type& type,
        Attribute_propagation override=PROPAGATION_STANDARD);

    /// Create and destroy an attribute. Attributes are constructed with zero-filled values arrays
    /// that can later be accessed with get/set_value*. For on-demand attributes the list_size
    /// (and the values pointer) is initially 0 until the 'execute' function fills it.
    /// \param id identifies attribute for lookups
    /// \param type data type, may be tree but not list
    /// \param override inheritance: parent overrides child
    explicit Attribute(
        Attribute_id id,
        const Type& type,
        Attribute_propagation override=PROPAGATION_STANDARD);

    /// A convenience constructor like the preceding one, except it takes care of simple,
    /// non-structured types automatically. Also creates an attribute ID from the attribute name.
    /// \param type primitive type: bool, int, ...
    /// \param name name of atribute
    /// \param type_asize number of elements > 0
    /// \param override inheritance: parent overrides child
    /// \param global not inheritance, nailed to element
    /// \param is_const is value immutable?
    explicit Attribute(
        Type_code	type,
        const char	*name,
        Uint		type_asize = 1,
        Attribute_propagation override=PROPAGATION_STANDARD,
        bool		global=false,
        bool		is_const=false);

    /// Another convenience constructor for the derived Attribute_object.
    /// \param id identifies attribute for lookups
    /// \param type primitive type: bool, int, ...
    /// \param type_asize number of elements > 0
    /// \param override inheritance: parent overrides child
    /// \param global not inheritance, nailed to element
    /// \param is_const is value immutable?
    explicit Attribute(
        Attribute_id	id,
        Type_code	type,
        Uint		type_asize = 1,
        Attribute_propagation override=PROPAGATION_STANDARD,
        bool		global=false,
        bool		is_const=false);

    // Destructor.
    ~Attribute();

    /// Make a copy. Used when GAP splits objects into regions.
    /// \param other object to copy from
    Attribute(
        const Attribute& other);

    /// \name access functions.
    /// There are no set members because of interdependencies.
    /// Note that get_values returns what set_values has stored, and in the
    /// case of string attributes that's a *pointer* to chars, not a char array.
    // @{
    /// identifies attribute for lookups
    Attribute_id get_id()	const;
    /// root of type tree
    const Type& get_type()	const;
    /// attribute name (toplevel type name)
    const char* get_name()	const;
    /// @}

    /// Retrieve the global state.
    bool get_global() const;
    /// Set the global state.
    /// \param global participates in inheritance?
    void set_global(
        bool global=true);

    /// Inheritance: parent overrides child.
    Attribute_propagation get_override() const;
    /// Change the override status for inheritance: parent overrides child.
    void	 set_override(
        Attribute_propagation ov);
    /// Beginning of value byte block.
    const char* get_values() const;
    /// Beginning of value byte block.
    char* set_values();

    /// Typed read-only value access for convenience (could use get_values).
    /// \param n if array, get n-th value
    /// \return boolean value
    bool get_value_bool(
        Uint n=0) const;

    /// \param v new value to set
    /// \param n if array, set n-th value
    void set_value_bool(
        bool v,
        Uint n=0);

    /// Typed read-only value access for convenience (could use get_values).
    /// \param n if array, get n-th value
    /// \return int value
    int get_value_int(
        Uint n=0) const;

    /// \param v new value to set
    /// \param n if array, set n-th value
    void set_value_int(
        int v,
        Uint n=0);

    /// Typed read-only value access for convenience (could use get_values).
    /// \param n if array, get n-th value
    /// \return scalar value
    Scalar get_value_scalar(
        Uint n=0) const;

    /// Typed read-only value access for convenience (could use get_values).
    /// \param n if array, get n-th value
    /// \return scalar value
    Dscalar get_value_dscalar(
        Uint n=0) const;

    /// Typed read-only value access for convenience (could use get_values).
    /// \param n if array, get n-th value
    /// \return Vector3 value
    Vector3 get_value_vector3(
        Uint n=0) const;

    /// Typed read-only value access for convenience (could use get_values).
    /// \param n if array, get n-th value
    /// \return Color value
    mi::math::Color get_value_color(
        Uint n=0) const;

    /// \param v new value to set
    /// \param n if array, set n-th value
    void set_value_scalar(
        Scalar v,
        Uint n=0);

    /// \param v new value to set
    /// \param n if array, set n-th value
    void set_value_dscalar(
        Dscalar v,
        Uint n=0);

    /// \param v new value to set (string will be copied)
    /// \param n if array, set n-th value
    void set_value_string(
        const char* v,
        Uint n=0);

    /// \param v new value to set
    /// \param n if array, set n-th value
    void set_value_vector3(
        const Vector3& v,
        Uint n=0);

    /// Typed read-only value access for convenience (could use get_values).
    /// \param n if array, set n-th value
    const char* get_value_string(
        Uint n=0) const;

    /// Templatized get, to hide casts.
    /// \note Since a const-reference is returned, the type of the attribute needs to match
    /// exactly. No implicit conversions will be performed.
    /// \param n if array, get n-th value
    /// \return value
    template <typename T> const T& get_value_ref(
        Uint n=0) const;

    /// Templatized get, to hide casts.
    /// \param n if array, get n-th value
    /// \return value
    template <typename T> T get_value(
        Uint n=0) const;

    // Templatized set, to hide casts.
    /// \param v new value to set
    /// \param n if array, set n-th value
    template <typename T> void set_value(
        const T& v,
        Uint n=0);

    /// Add an attachment to the internal list.
    /// \param attachment the attachment
    void add_attachment(
        const Attachment &attachment);

    /// Remove an attachment from the internal list.
    /// \param member_name null=whole attr, else target field
    void remove_attachment(
        const char *member_name);

    /// Return the list of attachments.
    const CONT::Array<Attachment>& get_attachments() const;

    /// User-defined attributes are named, but the attribute system deals only
    /// with integer IDs. Create a new ID for a name. 0 is reserved and is never returned.
    /// \param name new name to register
    /// \return the attribute id for this name
    static Attribute_id id_create(
        const char* name);

    /// Lookup an id.
    /// \param name name to look up
    static Attribute_id id_lookup(
        const char* name);

    /// Fast exchange of two Attributes.
    /// \param other the attribute to swap with
    void swap(
        Attribute& other);

    /// Return a copy of this attribute.
    /// This is needed by the Attribute_set copy constructor.
    virtual Attribute* copy() const;

    /// Return the approximate size in bytes of the element including all its
    /// substructures. This is used to make decisions about garbage collection.
    virtual size_t get_size() const;

    /// Return a set of all tags in the attribute value.
    /// \param[out] result return all referenced tags
    void get_references(
        DB::Tag_set* result) const;

    /// Unique class ID so that the receiving host knows which class to create.
    SERIAL::Class_id get_class_id() const;

    /// Check, if this attribute is of the given type. This is true, if either
    /// the class id of this attribute equals the given class id, or the class
    /// is derived from another class which has the given class id.
    /// \param id the class id to check
    /// \returnt true or false
    virtual bool is_type_of(
        SERIAL::Class_id id) const;

    /// Serialize the object to the given serializer including all sub elements.
    /// it must return a pointer behind itself (e.g. this + 1) to handle arrays.
    /// \param serializer useful for byte streams
    const SERIAL::Serializable* serialize(
        SERIAL::Serializer* serializer) const;

    /// Special version of serialize that writes no values.
    const SERIAL::Serializable* serialize_no_values(
        SERIAL::Serializer* serializer) const;

    /// Deserialize the object and all sub-objects from the given deserializer.
    /// it must return a pointer behind itself (e.g. this + 1) to handle arrays.
    /// \param deser useful functions for byte streams
    SERIAL::Serializable* deserialize(
        SERIAL::Deserializer* deser);

    /// Dump attribute to info messages, for debugging only.
    void dump() const;

    /// Factory function used for deserialization.
    static SERIAL::Serializable* factory();

    static const SERIAL::Class_id id = ID_ATTRIBUTE; ///< for serialization

    /// \name Attribute value memory handling.
    /// Storage for Attribute values should be allocated using these functions.
    //@{
    /// Allocate storage for given number of bytes.
    static void* allocate_storage(size_t, bool zero_init=true);
    /// Deallocate storage.
    static void deallocate_storage(void*);

    /// Copy given string to attribute value memory. The given pointer
    /// is assumed to reference string storage for the Attribute class.
    static void set_string(char* & storage, const char* str);
    static void set_string(char* & storage, std::string const& str)
        {set_string( storage, str.c_str()); }
    //@}

    /// This default constructor must be used by derived classes during deserialization.
    Attribute();

    /// A clone method that may optionally change the name.
    Attribute* clone(const char *name = 0) const;

    /// The assignment operator.
    Attribute& operator=(const Attribute&);

  protected:
    /// Flush the attribute's value array, return the amount of memory flushed
    /// in bytes
    /// \return amount of memory flushed
    virtual size_t flush();

  private:
    Attribute_id m_id;				///< identifies attribute for lookups
    Attribute_propagation m_override;		///< inheritance: parent overrides child
  protected:
    Type m_type;				///< data type, toplevel member type provides name
    char* m_values;				///< binary data block described by type tree
    CONT::Array<Attachment> m_attachments;	///< a list of attachments, may be empty

    /// Constructors for Attribute_object.
    //@{
    /// Create and destroy an attribute. Attributes are constructed with zero-
    /// filled values arrays that can later be accessed with get/set_value*.
    /// For on-demand attributes the  list_size (and the values pointer) is
    /// initially 0 until the 'execute' function fills it.
    /// \param id identifies attribute for lookups
    /// \param type data type, may be tree but not list
    /// \param list_size if attribute list, list size > 1
    /// \param override inheritance: parent overrides child
    explicit Attribute(
        Attribute_id	id,
        const Type	&type,
        Uint		list_size,
        Attribute_propagation override);

    /// A convenience constructor like the preceding one, except it takes care
    /// of simple, non-structured types automatically. Also creates an attribute
    /// ID from the attribute name.
    /// \param type primitive type: bool, int, ...
    /// \param name name of atribute
    /// \param type_asize number of elements > 0
    /// \param list_size if attribute list, list size > 1
    /// \param override inheritance: parent overrides child
    /// \param global not inheritance, nailed to element
    /// \param is_const is value immutable?
    explicit Attribute(
        Type_code	type,
        const char	*name,
        Uint		type_asize,
        Uint		list_size,
        Attribute_propagation override,
        bool		global,
        bool		is_const);

    /// Another convenience constructor for the derived Attribute_object.
    /// \param id identifies attribute for lookups
    /// \param type primitive type: bool, int, ...
    /// \param type_asize number of elements > 0
    /// \param list_size if attribute list, list size > 1
    /// \param override inheritance: parent overrides child
    /// \param global not inheritance, nailed to element
    /// \param is_const is value immutable?
    explicit Attribute(
        Attribute_id	id,
        Type_code	type,
        Uint		type_asize,
        Uint		list_size,
        Attribute_propagation override,
        bool		global,
        bool		is_const);
    //@}

    /// Internal list-aware functions to access the binary value block for Attribute_objects.
    //@{

    /// Retrieve beginning of value byte block. Internal function.
    /// \param i if attribute list, list index
    const char* get_values_i(
        Uint i)	const;
    /// Retrieve beginning of value byte block. Internal function.
    /// \param i if attribute list, list index
    char* set_values_i(
        Uint i);
    /// Typed read-only value access for convenience (could use get_values).
    /// \param i if attribute list, list index
    /// \param n if array, get n-th value
    /// \return boolean value
    bool get_value_bool_i(
        Uint		i,
        Uint		n = 0) const;

    /// \param v new value to set
    /// \param i if attribute list, list index
    /// \param n if array, set n-th value
    void set_value_bool_i(
        bool		v,
        Uint		i,
        Uint		n = 0);

    /// Typed read-only value access for convenience (could use get_values).
    /// \param i if attribute list, list index
    /// \param n if array, get n-th value
    /// \return int value
    int get_value_int_i(
        Uint		i,
        Uint		n = 0) const;

    /// \param v new value to set
    /// \param i if attribute list, list index
    /// \param n if array, set n-th value
    void set_value_int_i(
        int		v,
        Uint		i,
        Uint		n = 0);

    /// Typed read-only value access for convenience (could use get_values).
    /// \param i if attribute list, list index
    /// \param n if array, get n-th value
    /// \return scalar value
    Scalar get_value_scalar_i(
        Uint		i,
        Uint		n = 0) const;

    /// Typed read-only value access for convenience (could use get_values).
    /// \param i if attribute list, list index
    /// \param n if array, get n-th value
    /// \return scalar value
    Dscalar get_value_dscalar_i(
        Uint		i,
        Uint		n = 0) const;

    /// Typed read-only value access for convenience (could use get_values).
    /// \param i if attribute list, list index
    /// \param n if array, get n-th value
    /// \return Vector3 value
    Vector3 get_value_vector3_i(
        Uint		i,
        Uint		n = 0) const;

    /// Typed read-only value access for convenience (could use get_values).
    /// \param i if attribute list, list index
    /// \param n if array, get n-th value
    /// \return Color value
    mi::math::Color get_value_color_i(
        Uint		i,
        Uint		n = 0) const;

    /// \param v new value to set
    /// \param i if attribute list, list index
    /// \param n if array, set n-th value
    void set_value_scalar_i(
        Scalar		v,
        Uint		i,
        Uint		n = 0);

    /// \param v new value to set
    /// \param i if attribute list, list index
    /// \param n if array, set n-th value
    void set_value_dscalar_i(
        Dscalar		v,
        Uint		i,
        Uint		n = 0);

    /// \param v new value to set (string will be copied)
    /// \param i if attribute list, list index
    /// \param n if array, set n-th value
    void set_value_string_i(
        const char      *v,
        Uint            i,
        Uint            n = 0);

    /// \param v new value to set
    /// \param i if attribute list, list index
    /// \param n if array, set n-th value
    void set_value_vector3_i(
        const Vector3&	v,
        Uint		i,
        Uint		n = 0);

    /// Typed read-only value access for convenience (could use get_values).
    /// \param i if attribute list, list index
    /// \param n if array, set n-th value
    const char *get_value_string_i(
        Uint            i,
        Uint            n = 0) const;
    
    /// Templatized get, to hide casts.
    /// \note Since a const-reference is returned, the type of the attribute needs to match
    /// exactly. No implicit conversions will be performed.
    /// \param i if attribute list, list index
    /// \param n if array, get n-th value
    /// \return value
    template <typename T> inline const T& get_value_ref_i(
        Uint		i = 0,
        Uint		n = 0) const;

    /// Templatized get, to hide casts.
    /// \param i if attribute list, list index
    /// \param n if array, get n-th value
    /// \return value
    template <typename T> inline T get_value_i(
        Uint		i = 0,
        Uint		n = 0) const;

    // Templatized set, to hide casts.
    /// \param v new value to set
    /// \param i if attribute list, list index
    /// \param n if array, set n-th value
    template <typename T> inline void set_value_i(
        const T		&v,
        Uint		i = 0,
        Uint		n = 0);
    //}

    /// Retrieve the listsize. Since this base class has no list, we can simply return 1.
    virtual Uint get_listsize() const { return 1; }

  private:
    bool m_global;				///< participates in inheritance

    /// Test code needs to be able to test internal variables
    friend void test_attribute();
protected:
    /// Common constructor code.
    void init(
        Attribute_id id, Attribute_propagation override,
        Uint list_size, bool is_const, bool is_global);
private:
    friend void get_references(const Attribute&, DB::Tag_set&, Compare);

    /// The Attribute_set uses the listsize retrieval. See above comments - needs to be reviewed.
    friend class Attribute_set;

};


/// Overload of the standard \c swap() for swapping two \c Attribute.
void swap(
    ATTR::Attribute& one,
    ATTR::Attribute& other);

}
}

#include "attr_inline_attr.h"

#endif
