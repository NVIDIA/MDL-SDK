/******************************************************************************
 * Copyright (c) 2004-2020, NVIDIA CORPORATION. All rights reserved.
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

/// \file
/// \brief inlined Attribute_set functions

namespace MI {
namespace ATTR {

// return the number of attributes in the attribute_set
inline size_t Attribute_set::size() const
{
    return m_attrs.size();
}


// value access for the boolean attributes. The functions are similar to the
// access functions of Attributes, except that we have only non-array booleans.
// Note that this function returns false even when the flag attribute was not
// attached at all to the Attribute_set. Using Likely<bool> or exceptions would
// be more correct.
inline bool get_bool_attrib(
    const Attribute_set& attr_set,
    Attribute_id id)	// which flag?
{
    const Attribute* attr = attr_set.lookup(id);
    if (attr)
        return *(bool*)attr->get_values();
    else
        return false;
}


// every built-in flag (and every Attribute) comes with an override flag
// that during inheritance causes parent values to override child values.
inline Attribute_propagation get_override(
    const Attribute_set& attr_set,
    Attribute_id id)	// which flag?
{
    const Attribute* attr = attr_set.lookup(id);
    // you shouldn't ask this for non-existing attributes
    ASSERT(M_ATTR, attr);
    if (attr)
        return attr->get_override();
    else
        return PROPAGATION_UNDEF;
}


// convenience method to retrieve values of single-value attributes.
// if the attribute is found and type-checking succeeded the value
// is written to the second argument and the method returns true.
template <typename T>
bool get_attribute_value(
    const Attribute_set& attr_set,
    const char* name,
    T& value)
{
    const ATTR::Attribute* attr = attr_set.lookup(name);
    if (!attr)
        return false;
    // we do not accept Attribute_object and derivatives here!!
    ASSERT(M_ATTR, attr->get_class_id() == ID_ATTRIBUTE);

    if (contains_expected_type(attr->get_type(), Type_code_traits<T>::type_code)) {
        value = attr->get_value<T>(0);
        return true;
    }
    return false;
}


//
// unique class ID so that the receiving host knows which class to create
//

inline SERIAL::Class_id Attribute_set::get_class_id() const
{
    return id;
}


//
// for iterating over attached attributes, in undefined order.
//


//
// see Attribute_set::swap().
//

inline void swap(
    Attribute_set &one,		// the one
    Attribute_set &other) 	// the other
{
    one.swap(other);
}

} // namespace ATTR
} // namespace MI
