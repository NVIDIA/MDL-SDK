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
/// \brief Attribute_list class definition.
 
#ifndef BASE_DATA_ATTR_I_ATTR_ATTRIBUTE_LIST_H
#define BASE_DATA_ATTR_I_ATTR_ATTRIBUTE_LIST_H

#include "i_attr_attribute.h"

namespace MI {
namespace ATTR {

/// The base attribute class for list-based attributes. This allows to name those without
/// including headers from SCENE.
class Attribute_list : public Attribute
{
    /// Internally used typedef.
    typedef Attribute Parent_type;
  public:
    /// 3D vector type
    typedef mi::math::Vector<Scalar,3> Vector3;

    /// Constructor.
    Attribute_list();
    /// Destructor.
    ~Attribute_list();
    /// Copy constructor.
    Attribute_list(
        const Attribute_list& other);

    /// Forwarding constructors list.
    //@{
    explicit Attribute_list(
        Attribute_id id,
        const Type& type,
        Uint list_size,
        Attribute_propagation override)
      : Attribute(id, type, list_size, override),
        m_listsize(list_size),
        m_listalloc(list_size)
    { init(id, override, list_size, false, false); }
    explicit Attribute_list(
        Type_code	type,
        const char	*name,
        Uint		type_asize,
        Uint		list_size,
        Attribute_propagation override,
        bool		global,
        bool		is_const)
      : Attribute(type, name, type_asize, list_size, override, global, is_const),
        m_listsize(list_size),
        m_listalloc(list_size)
    { init(Attribute::id_create(name), override, list_size, is_const, global); }
    explicit Attribute_list(
        Attribute_id	id,
        Type_code	type,
        Uint		type_asize,
        Uint		list_size,
        Attribute_propagation override,
        bool		global,
        bool		is_const)
      : Attribute(id, type, type_asize, list_size, override, global, is_const),
        m_listsize(list_size),
        m_listalloc(list_size)
    { init(id, override, list_size, is_const, global); }
    //@}

    /// Attribute virtual functions. Return the approximate size in bytes of the element
    /// including all its substructures. This is used to make decisions about garbage collection.
    /// \return the approximate size in bytes
    //size_t get_size() const;

    /// \name Scene_element_networking
    /// Networking functions. See description header in attr.h.
    //@{
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
    /// Deserialize the object and all sub-objects from the given deserializer.
    /// it must return a pointer behind itself (e.g. this + 1) to handle arrays.
    /// \param deser useful functions for byte streams
    SERIAL::Serializable* deserialize(
        SERIAL::Deserializer* deser);
    /// Factory function used for deserialization.
    static SERIAL::Serializable* factory();

    static const SERIAL::Class_id id = ID_ATTRIBUTE_LIST; ///< for serialization
    //@}


    /// formerly ATTR::Attribute members
    //@{
    /// Beginning of value byte block.
    /// \param i if attribute list, list index
    const char* get_list_values(
        Uint i)	const;
    /// Beginning of value byte block.
    /// \param i if attribute list, list index
    char* set_list_values(
        Uint i);
    /// Typed read-only value access for convenience (could use get_values).
    /// \param i if attribute list, list index
    /// \param n if array, get n-th value
    /// \return boolean value
    bool get_list_value_bool(
        Uint i,
        Uint n=0) const;
    /// \param v new value to set
    /// \param i if attribute list, list index
    /// \param n if array, set n-th value
    void set_list_value_bool(
        bool v,
        Uint i,
        Uint n=0);

    /// Typed read-only value access for convenience (could use get_values).
    /// \param i if attribute list, list index
    /// \param n if array, get n-th value
    /// \return int value
    int get_list_value_int(
        Uint i,
        Uint n=0) const;
    /// \param v new value to set
    /// \param i if attribute list, list index
    /// \param n if array, set n-th value
    void set_list_value_int(
        int v,
        Uint i,
        Uint n=0);

    /// Typed read-only value access for convenience (could use get_values).
    /// \param i if attribute list, list index
    /// \param n if array, get n-th value
    /// \return scalar value
    Scalar get_list_value_scalar(
        Uint i,
        Uint n=0) const;
    /// Typed read-only value access for convenience (could use get_values).
    /// \param i if attribute list, list index
    /// \param n if array, get n-th value
    /// \return Vector3 value
    Vector3 get_list_value_vector3(
        Uint i,
        Uint n=0) const;

    /// Typed read-only value access for convenience (could use get_values).
    /// \param i if attribute list, list index
    /// \param n if array, get n-th value
    /// \return Color value
    mi::math::Color get_list_value_color(
        Uint i,
        Uint n=0) const;
    /// \param v new value to set
    /// \param i if attribute list, list index
    /// \param n if array, set n-th value
    void set_list_value_scalar(
        Scalar v,
        Uint i,
        Uint n=0);

    /// \param v new value to set (string will be copied)
    /// \param i if attribute list, list index
    /// \param n if array, set n-th value
    void set_list_value_string(
        const char* v,
        Uint i,
        Uint n=0);

    /// \param v new value to set
    /// \param i if attribute list, list index
    /// \param n if array, set n-th value
    void set_list_value_vector3(
        const Vector3& v,
        Uint i,
        Uint n=0);

    /// Typed read-only value access for convenience (could use get_values).
    /// \param i if attribute list, list index
    /// \param n if array, set n-th value
    const char* get_list_value_string(
        Uint i,
        Uint n=0) const;

    /// Templatized get, to hide casts.
    /// \note Since a const-reference is returned, the type of the attribute needs to match
    /// exactly. No implicit conversions will be performed.
    /// \param i if attribute list, list index
    /// \param n if array, get n-th value
    /// \return value
    template <typename T> inline const T& get_list_value_ref(
        Uint		i = 0,
        Uint		n = 0) const;

    /// Templatized get, to hide casts.
    /// \param i if attribute list, list index
    /// \param n if array, get n-th value
    /// \return value
    template <typename T> inline const T get_list_value(
        Uint		i = 0,
        Uint		n = 0) const;

    // Templatized set, to hide casts.
    /// \param v new value to set
    /// \param i if attribute list, list index
    /// \param n if array, set n-th value
    template <typename T> inline void set_list_value(
        const T		&v,
        Uint		i = 0,
        Uint		n = 0);

    /// If the attribute is a list, such as per-vertex motion paths, resize it.
    /// A non-list attribute is an attribute with list size 1.
    /// \param list_size new list size (number of data elems)
    /// \param force use the exact size, don't optimize
    void set_listsize(
        Uint list_size,
        bool force=false);
    /// Reserve space for a given number of attribute elements
    /// \param list_capacity new list capacity
    void list_reserve(
        Uint list_capacity );
    /// list size (number of data elements)
    Uint get_listsize()	const;
    /// Explicitly shrink capacity to given size
    /// \param listsize new size
    void list_shrink(Uint listsize);
    /// list capacity (# data elements)
    Uint get_listcapacity( ) const;
    //@}

    /// Fast exchange of two Attribute_list
    /// \param other the other attribute
    void swap(
        Attribute_list& other);

  protected:
    /// Flush the attribute's value array, return the amount of memory flushed
    /// in bytes
    /// \return amount of memory flushed
    virtual size_t flush();

  private:
    Uint m_listsize;				///< the list size
    /// if greater 1: \c m_values is a consecutive array of \c m_listsize elements of type \c m_type
    Uint m_listalloc;				///< actually allocated list size
};


/// Overload of the default swap() for Attribute_list.
/// \param one the one object
/// \param other the other object
void swap(
    Attribute_list& one,
    Attribute_list& other);

}
}

#include "attr_attribute_list_inline.h"

#endif
