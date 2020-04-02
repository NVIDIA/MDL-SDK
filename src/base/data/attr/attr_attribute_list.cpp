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
/// \brief Attribute_list class implementation.
 
#include "pch.h"

#include "i_attr_attribute_list.h"
#include "attr.h"

#include <base/data/db/i_db_transaction.h>
#include <base/data/serial/i_serializer.h>

namespace MI {
namespace ATTR {

extern void copy_constructor(Type_iterator&, Type_iterator&);

Attribute_list::Attribute_list()
  : m_listsize(0),
    m_listalloc(0)
{}


Attribute_list::Attribute_list(
    const Attribute_list& other)
  : Attribute(other), m_listsize(other.m_listsize), m_listalloc(other.m_listsize)
{
    // (2) now copy the actual data. But this is only required for list sizes > 1,
    // for the first "block" Attribute's copy constructor had already taken care of it.
    for (Uint l=1; l < this->get_listsize(); ++l) {
        Type_iterator in_iter(&other.get_type(), const_cast<char*>(other.get_values_i(l)));
        Type_iterator out_iter(&get_type(), set_values_i(l));
        copy_constructor(in_iter, out_iter);
    }
}


Attribute_list::~Attribute_list()
{}


SERIAL::Class_id Attribute_list::get_class_id() const
{
    return Attribute_list::id;
}


bool Attribute_list::is_type_of(
    SERIAL::Class_id id) const
{
    return id == Attribute_list::id ? true : Parent_type::is_type_of(id);
}


// factory function used for deserialization
SERIAL::Serializable* Attribute_list::factory()
{
    return new Attribute_list;
}


const SERIAL::Serializable* Attribute_list::serialize(
    SERIAL::Serializer* serializer) const // useful functions for byte streams
{
    // NOTE Setting the listsize *before* calling the base class' serialize() allows to use
    // there the virtual function get_listsize() and do the correct size compuations!!
    serializer->write(m_listsize);
    Parent_type::serialize(serializer);
    return this+1;
}


SERIAL::Serializable* Attribute_list::deserialize(
    SERIAL::Deserializer* deserializer) // functions for byte streams
{
    // See note about the importance of this order in the serialize() member.
    deserializer->read((Uint *)&m_listsize);
    m_listalloc = m_listsize;
    Parent_type::deserialize(deserializer);
    return this+1;
}


// if the attribute is a list, such as per-vertex motion paths, resize it.
// A non-list attribute is an attribute with list size 1. 0 means no change.
// Reallocate if the new list size is less than 1/1.5 of the old size; and
// if the new size is greater than the old one in which case grow by 1.5.
void Attribute_list::set_listsize(
    Uint		list_size,	// new size, 0 means no change
    bool		force)		// use the exact size, don't optimize
{
    if (list_size == m_listsize)	// same as before: do nothing
        return;

    if (force) {			// force: use requested size
        if ( list_size < m_listsize ) {
            list_shrink(list_size);
        } else {
            list_reserve(list_size);
        }
    }
    else {					// otherwise optimize:
        if (!list_size)                         // - shrink only if set to 0
            list_shrink(list_size);
        if (list_size > m_listalloc) {		// - grow to more than alloc:
            Uint newcapacity = m_listalloc*3/2; //   grow by 50% but
            if (list_size > newcapacity)	//   at least as much as asked
                newcapacity = list_size;
            list_reserve(newcapacity);
        }
    }
    m_listsize = list_size;
}


// Reserve space for a given number of attribute elements
void Attribute_list::list_reserve(
    Uint        list_capacity ) // new list capacity
{
    // never shrink
    if ( list_capacity <= m_listalloc ) {
        return;
    }

    const size_t ts = get_type().sizeof_all();
    const size_t nc = ts * list_capacity;
    const size_t os = ts * m_listsize;

    char* newvalues = static_cast<char*>(ATTR::Attribute::allocate_storage(nc));

    if (nc > os) {                          // now larger: copy all and pad
        if (m_values)
            memcpy(newvalues, m_values, os);
        memset(newvalues + os, 0, nc - os);
    }

    char* junk = m_values;                  // replace memory pointer
    m_values = newvalues;
    ATTR::Attribute::deallocate_storage(junk);
    m_listalloc = list_capacity;
}


// Explicitly shrink list to given size. This works by doing a copy of the appropriate listsize.
void Attribute_list::list_shrink(
    Uint          listsize )
{
    if (listsize >= m_listsize)
        return;

    size_t ts = get_type().sizeof_all();
    size_t ns = ts * listsize;

    char* newvalues = static_cast<char*>(Attribute::allocate_storage(ns));

    // now copy the actual data into the newvalues
    for (Uint l=0; l<listsize; ++l) {
        ATTR::Type_iterator in_iter(&get_type(), const_cast<char*>(get_list_values(l)));
        ATTR::Type_iterator out_iter(&get_type(), newvalues + l*ts);
        ATTR::copy_constructor(in_iter, out_iter);
    }

    // store under m_values and free (now old) newvalues
    std::swap(m_values, newvalues);
    ATTR::Attribute::deallocate_storage(newvalues);

    m_listalloc = listsize;
    m_listsize = listsize;
}


size_t Attribute_list::flush()
{
    size_t res = Parent_type::flush() * m_listalloc;
    m_listalloc = 0;
    m_listsize = 0;
    return res;
}

}
}
