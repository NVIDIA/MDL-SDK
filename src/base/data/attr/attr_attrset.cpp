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
/// \brief Attribute_set functions

#include "pch.h"
#include "attr.h"
#include "i_attr_type_value_iterator.h"
#include "i_attr_utilities.h"

#include <base/system/main/types.h>
#include <base/data/db/i_db_transaction.h>
#include <base/data/serial/i_serializer.h>
#include <base/lib/cont/i_cont_rle_array.h>
#include <base/lib/log/i_log_logger.h>
#include <boost/shared_ptr.hpp>

#include <limits>
#include <utility>

namespace {
// Dump the given Attribute.
void print_attribute(
    const MI::ATTR::Attribute *attr,	// the attribute
    MI::DB::Transaction *transaction);	// transaction
// Given a raw pointer to some data and a type, print out the correct value.
void print_type_value(
    const MI::ATTR::Type &type,		// type of value
    const char *val_ptr,		// raw data of value
    MI::DB::Transaction *transaction);	// transaction
}

namespace MI {
namespace ATTR {

using namespace MI::LOG;


//
// clear the attribute set, i.e. detach all attributes
//

void Attribute_set::clear()
{
    m_attrs.clear();
}


//
// constructor and destructor
//

Attribute_set::Attribute_set()
{}


Attribute_set::~Attribute_set()
{}


//
// for const iterating over attached attributes, in undefined order.
//

// Attribute_set::Const_iterator Attribute_set::const_iterator() const
// {
//     return Attribute_set::Const_iterator(&m_attr);
// }


//
// copy attribute set: copy all attributes from attrset to this
//

Attribute_set &Attribute_set::operator=(
    const Attribute_set &attrset)	// attribute set to copy
{
    if (&attrset == this)
        return *this;
    // clean current set of Attributes first
    m_attrs.clear();
    // now copy the given on
    deep_copy(attrset);

    return *this;
}


//
// copy attribute set.
//

Attribute_set::Attribute_set(
    const Attribute_set &attrset)	// attribute set to copy
{
    *this = attrset;
}


//
// swap this Attribute_set with another Attribute_set.
// This function swaps two attribute_sets by exchanging the sets and the flags,
// which is done in constant time. Note that the global swap() functionality
// falls back to this function due to its template specialization.
//

void Attribute_set::swap(
    Attribute_set	&other)		// the other
{
    using std::swap;
    swap(m_attrs, other.m_attrs);
}


//
// Full-blown Attribute objects even for flags. The flag_present value is
// not required anymore, but both the flag_value and the flag_override are.
//

// Set the value appropriately - create Attribute, if it's not there.
void set_bool_attrib(
    Attribute_set& attr_set,
    Attribute_id id,			// which attribute
    bool v,				// new value
    bool create)			// create when missing?
{
    Attribute* flag = attr_set.lookup(id);

    if (!flag && create) {
        const char* name = 0;
        Uint array_size = 1;
        // m_flag_value
        Type type(TYPE_BOOLEAN, name, array_size);
        boost::shared_ptr<Attribute> attr_ptr(new Attribute(id, type));
        attr_ptr->set_global(true);
        attr_set.attach(attr_ptr);
        flag = attr_set.lookup(id);
    }
    if (flag) {
        bool* values = (bool*)flag->set_values();
        *values = v;
    }
}


//
// look up an attached attribute by ID or by name. Return 0 on failure.
//

Attribute *Attribute_set::lookup(
    Attribute_id		    id)	// ID of attribute
{
    Iter it = m_attrs.find(id);
    if (it == m_attrs.end())
        return 0;
    else
        return (*it).second.get();
}

const Attribute *Attribute_set::lookup(
    Attribute_id		    id) const       // ID of attribute
{
    Const_iter it = m_attrs.find(id);
    if (it == m_attrs.end())
        return 0;
    else
        return (*it).second.get();
}

const Attribute *Attribute_set::lookup(
    const char		*name) const	// name of attribute to look up
{
    return lookup(Attribute::id_lookup(name));
}

Attribute *Attribute_set::lookup(
    const char          *name)          // name of attribute to look up
{
    return lookup(Attribute::id_lookup(name));
}


boost::shared_ptr<Attribute> Attribute_set::lookup_shared_ptr(
    Attribute_id	id) const	// ID of attribute to look up
{
    Const_iter it = m_attrs.find(id);
    if (it == m_attrs.end())
        return boost::shared_ptr<Attribute>();
    else
        return (*it).second;
}


//
// attach an attribute. It's an error if the attribute is already there.
// It may be a good idea to move the check into #ifdef DEBUG because it's
// expensive and expands to a lot of code. If we decide we always need this,
// the attribute set should know its parent tag so the messages can refer
// to the Element the attribute set is attached to by name.
//

bool Attribute_set::attach(
    const boost::shared_ptr<Attribute> &attr)// attribute to attach
{
    // error checking
    const Attribute *other = lookup(attr->get_id());
    if (other) {
        const char *aname = attr->get_name();
        const char *oname = other->get_name();
        if (aname && oname && strcmp(aname, oname) != 0)
            mod_log->error(M_ATTR, LOG::Mod_log::C_DATABASE, 1,
                "name hash collision between \"%s\" and \"%s\"", aname, oname);
        else if (aname)
            mod_log->error(M_ATTR, LOG::Mod_log::C_DATABASE, 2,
                "duplicate attribute \"%s\", already attached", aname);
        else
            mod_log->error(M_ATTR, LOG::Mod_log::C_DATABASE, 3,
                "duplicate unnamed attribute %u, already attached",
                                                        attr->get_id());
    }

    // the actual attachment happens here (won't accept const here)
    bool result = m_attrs.insert(
        std::make_pair(attr->get_id(), attr)).second;
    ASSERT(M_ATTR, result || other);

    // return success
    return result;
}


//
// detach an attribute, but don't destroy it. The boost::shared_ptr return value
// takes care of destroying the attribute in the end.
//

boost::shared_ptr<Attribute> Attribute_set::detach(
    Attribute_id	id)		// ID of attribute to detach
{
    boost::shared_ptr<Attribute> attr;
    Iter it = m_attrs.find(id);
    if (it != m_attrs.end()) {
        attr = (*it).second;
        m_attrs.erase(it);
    }

    return attr;
}

boost::shared_ptr<Attribute> Attribute_set::detach(
    const char          *name)          // name of attribute to detach
{
    return detach(Attribute::id_lookup(name));
}


//
// deep copy of all Attributes
//

void Attribute_set::deep_copy(
    const Attribute_set &other)		// the other
{
    ASSERT(M_ATTR, m_attrs.empty());

    Const_iter it, end = other.m_attrs.end();
    for (it=other.m_attrs.begin(); it != end; ++it) {
        const Attribute *attr = (*it).second.get();
        if (attr) {
            boost::shared_ptr<Attribute> new_attr(attr->copy());

            // I do not use attach() here to avoid the useless error checking.
            // It is useless here since we start with an empty m_attrs.
            m_attrs.insert(std::make_pair(new_attr->get_id(), new_attr));
        }
    }
}


//
// return set of attributes
//

const Attributes &Attribute_set::get_attributes() const
{
    return m_attrs;
}


Attributes &Attribute_set::get_attributes()
{
    return m_attrs;
}


// Return the approximate size in bytes of the element including all its
// substructures. This is used to make decisions about garbage collection.
size_t Attribute_set::get_size() const
{
    size_t res = sizeof(*this);

    Attributes::const_iterator it, end=get_attributes().end();
    for (it=get_attributes().begin(); it != end; ++it) {
        res += it->second->get_size();
    }
    return res;
}


void collect_tags_rec(Type_iterator& iter, DB::Tag_set& result)
{
    // iterate over all elements of the type and deal with them type-accordingly
    // Please note that the client's code takes care of the listsize.
    for (; !iter.at_end(); iter.to_next()) {
        char* value = iter.get_value();
        int arraysize = iter->get_arraysize();

        // a dynamic array - hence create out_array and reset the value pointers accordingly
        if (!arraysize && iter->get_typecode() != TYPE_ARRAY) {
            Dynamic_array* array = (Dynamic_array*)value;
            arraysize = array->m_count;

            // no alignment required here
            value = array->m_value;
        }

        // arrays can be handled recursively without any changes to the value pointers
        if (iter->get_typecode() == TYPE_ARRAY) {
            Type_iterator sub(&iter, value);
            size_t size = iter->sizeof_elem() * iter->get_arraysize();
            collect_tags_rec(sub, result);

            // increase offset by the array element type's size since an array has no size itself
            value += size;
        }
        // rle arrays reset the value pointers accordingly
        else if (iter->get_typecode() == TYPE_RLE_UINT_PTR) {
            // nothing to do here since we are looking for Tag-typed data
#if 0
            Rle_array<Uint>* array = *(Rle_array<Uint>**)value;
            Rle_chunk_iterator<Uint> iterator = array->begin_chunk();
            size_t size = array->get_index_size();
            for (size_t i=0; i < size; ++i) {
                out_array->push_back(iterator.data(), iterator.count());
                ++iterator;
            }
#endif
            value += sizeof(CONT::Rle_array<Uint>*);
        }
        // structs can be handled recursively
        else if (iter->get_typecode() == TYPE_STRUCT || iter->get_typecode() == TYPE_ATTACHABLE
            || iter->get_typecode() == TYPE_CALL)
        {
            size_t size = iter->sizeof_elem();
            for (int a=0; a < arraysize; ++a) {
                // in case of TYPE_CALL, the value points right to the function's tag
                if (iter->get_typecode() == TYPE_CALL) {
                    DB::Tag tag(*reinterpret_cast<DB::Tag*>(value));
                    if (tag.is_valid())
                        result.insert(tag);
                    // need to reset the value pointer such that following Type_iterator sub is
                    // correct
                    value += Type::sizeof_one(TYPE_STRING);
                    value += Type::sizeof_one(TYPE_STRING);
                }
                Type_iterator sub(&iter, value);
                collect_tags_rec(sub, result);
                value += size;
                if (iter->get_typecode() == TYPE_CALL)
                    value -= 2*Type::sizeof_one(TYPE_STRING);
            }
        }
        else {
            int type, count, size;
            Type_code type_code = iter->get_typecode();
            eval_typecode(type_code, &type, &count, &size);
            // only interested in Tag-typed data
            if (type == 't') {
                ASSERT(M_ATTR, count == 1);
                for (int a=0; a < arraysize; ++a) {
                    DB::Tag tag (*reinterpret_cast<DB::Tag*>(value));
                    if (tag.is_valid())
                        result.insert(tag);
                    value += size;
                }
            }
        }
    }
}


// Return complete list of referenced tags. Unlike bundle(), this must be
// exact because reference counting is used to prevent deletion of scene
// elements that are still referenced somewhere.
void Attribute_set::get_references(
    DB::Tag_set		*result) const	// return all referenced tags
{
    if (!result)
        return;

    Const_iter it, end = m_attrs.end();

    // iterating the Attribute_set's attributes
    for (it=m_attrs.begin(); it != end; ++it) {
        const boost::shared_ptr<Attribute>& attr = it->second;

        // iterate over all elements and collect all is_tag()-typed ones
        for (Uint i=0; i < attr->get_listsize(); ++i) {
            Type_iterator iter(&attr->get_type(), attr->set_values_i(i));
            collect_tags_rec(iter, *result);
        }
    }
}


// Retrieve all tags from the given Attribute.
void get_references(
    const Attribute& attr,
    DB::Tag_set& result,
    Compare type_comparison)
{
    // iterate over all elements and collect all is_tag()-typed ones
    for (Uint i=0; i < attr.get_listsize(); ++i) {
        Type_iterator iter(&attr.get_type(), (const_cast<Attribute&>(attr)).set_values_i(i));
        collect_tags_rec(iter, result);
    }
}


// serialize the object to the given serializer including all sub elements.
// It must return a pointer behind itself (e.g. this + 1) to handle arrays.
const SERIAL::Serializable *Attribute_set::serialize(
    SERIAL::Serializer	*serializer) const	// functions for byte streams
{
    ASSERT(M_ATTR, m_attrs.size() <= std::numeric_limits<Uint32>::max());
    Uint32 size = Uint32(m_attrs.size());
    serializer->write(size);

    Const_iter it, end = m_attrs.end();
    for (it=m_attrs.begin(); it != end; ++it) {
        ASSERT(M_ATTR, (*it).second);		// no NULL pointers supported
        serializer->serialize(const_cast<Attribute*>((*it).second.get()));
    }

    return this+1;
}


//
// deserialize the object and all sub-objects from the given deserializer.
// It must return a pointer behind itself (e.g. this + 1) to handle arrays.
//

SERIAL::Serializable *Attribute_set::deserialize(
    SERIAL::Deserializer *deser)	// useful functions for byte streams
{
    Uint32 size;
    deser->read(&size);
    for (Uint32 i=0; i < size; ++i) {
        Attribute *attr = (Attribute *)deser->deserialize();
        m_attrs.insert(
            std::make_pair(
                attr->get_id(),boost::shared_ptr<Attribute>(attr)));
    }
    return this+1;
}


//
// dump attribute set to debug messages, for debugging only.
//
#if 0
const char* get(bool v)
{
    if (v)
        return "true";
    else
        return "false";
}
#endif

void Attribute_set::dump() const
{
    Const_iter it, end = m_attrs.end();
    for (it=m_attrs.begin(); it != end; ++it)
        (*it).second->dump();
}


//
// factory function used for deserialization
//

SERIAL::Serializable *Attribute_set::factory()
{
    return new Attribute_set;
}


//
// print the given attribute including its values.
//

void Attribute_set::print(
    DB::Transaction *transaction) const	// transaction to use for tag retrieval
{
    mod_log->debug(M_ATTR, LOG::Mod_log::C_DATABASE,
        "---Print current Attribute_set");

    Const_iter it, end = m_attrs.end();
    for (it=m_attrs.begin(); it != end; ++it) {
        boost::shared_ptr<Attribute> attr = (*it).second;
        print_attribute(attr.get(), transaction);
    }
}


//=============================================================================
//
// Constructor.
//

Attribute_set_type_iterator::Attribute_set_type_iterator()
    : m_attr_iter(), m_type_iter(0), m_toplevel_only(false)
{}


//
// Constructor.
//

Attribute_set_type_iterator::Attribute_set_type_iterator(
    Const_iter	begin,			// iterator over the Attributes
    const	Attributes &attrs,	// corresponding Attribute_set
    bool	top)			// traverse only top level
    : m_attr_iter(begin),
      m_type_iter((begin==attrs.end()? 0 : &(*begin).second->get_type()),top),
      m_toplevel_only(top)
{}


//
// Copy constructor.
//

Attribute_set_type_iterator::Attribute_set_type_iterator(
    const Attribute_set_type_iterator &iter)	// the other one
    : m_attr_iter(iter.m_attr_iter),
      m_type_iter(iter.m_type_iter),
      m_toplevel_only(iter.m_toplevel_only)
{}


//
// Assignment operator.
//

Attribute_set_type_iterator &Attribute_set_type_iterator::operator=(
    const Attribute_set_type_iterator &other)	// the other one
{
    m_attr_iter     = other.m_attr_iter;
    m_type_iter     = other.m_type_iter;
    m_toplevel_only = other.m_toplevel_only;
    return *this;
}


//
// Incrementing the iterator.
//

Attribute_set_type_iterator &Attribute_set_type_iterator::operator++()
{
    // iterate over top-level only
    if (m_toplevel_only) {
        ++m_attr_iter;
        m_type_iter = Type_iterator_rec(0, m_toplevel_only);
    } else {
        // iterate the current type upto the end
        if (m_type_iter.get_type())
            ++m_type_iter;
        // check if we have reached it - if yes, iterate to next attribute
        if (!m_type_iter.get_type()) {
            ++m_attr_iter;
            m_type_iter = Type_iterator_rec(0, m_toplevel_only);
        }
    }
    return *this;
}


//
// Incrementing the iterator - the slow version.
//

Attribute_set_type_iterator Attribute_set_type_iterator::operator++(
    int)				// dummy
{
    Attribute_set_type_iterator tmp(*this);
    ++(*this);
    return tmp;
}


//
// Dereferencing the iterator.
//

const ATTR::Type &Attribute_set_type_iterator::operator*() const
{
    if (!m_type_iter.get_type())
        m_type_iter = Type_iterator_rec(&(*m_attr_iter).second->get_type(),
                                        m_toplevel_only);
    return *m_type_iter;
}


//
// Dereferencing the iterator.
//

const Type *Attribute_set_type_iterator::operator->() const
{
    return &(operator*());
}


//
// Comparing for equality.
//

bool Attribute_set_type_iterator::operator==(
    const Attribute_set_type_iterator &iter) const	// the other iterator
{
    if (m_attr_iter != iter.m_attr_iter)
        return false;

    // TO DO: check for end!
    return m_type_iter == iter.m_type_iter;
}


//
// Comparing for inequality.
//

bool Attribute_set_type_iterator::operator!=(
    const Attribute_set_type_iterator &iter) const	// the other iterator
{
    return !(*this == iter);
}


//
// Is iteration over top-level types only?
//

bool Attribute_set_type_iterator::is_toplevel_only() const
{
    return m_toplevel_only;
}


}} // namespace MI::ATTR


namespace {
using namespace MI;
using namespace MI::ATTR;

// Dump the given Attribute.
void print_attribute(
    const Attribute *attr,				// the attribute
    DB::Transaction *transaction)			// transaction
{
    if (!attr)
        return;
    attr->dump();
    mod_log->debug(M_ATTR, LOG::Mod_log::C_DATABASE,
        "  of type %s",
        Type::type_name(attr->get_type().get_typecode()));
    print_type_value(attr->get_type(), attr->get_values(), transaction);

}

// Given a raw pointer to some data and a type, print out the correct value.
void print_type_value(
    const Type &typ,					// type of value
    const char *val_ptr,				// raw data of value
    DB::Transaction *transaction)			// transaction
{
    Type_code code = typ.get_typecode();
    size_t array_size = typ.get_arraysize();

    switch (code) {
      case TYPE_RGB: {
        Uint8 *ptr = (Uint8*)val_ptr;
        mod_log->debug(M_ATTR, Mod_log::C_DATABASE,
            "  (%d, %d, %d), arraysize %" FMT_SIZE_T ", name %s",
            (int)ptr[0], (int)ptr[1], (int)ptr[2],
            array_size, typ.get_name()?typ.get_name():"<nil>");
        break;
      }
      case TYPE_RGBA: {
        Uint8 *ptr = (Uint8*)val_ptr;
        mod_log->debug(M_ATTR, Mod_log::C_DATABASE,
            "  (%d, %d, %d, %d), arraysize %" FMT_SIZE_T ", name %s",
            (int)ptr[0], (int)ptr[1], (int)ptr[2], (int)ptr[3],
            array_size, typ.get_name());
        break;
      }
      case TYPE_COLOR: {
        Scalar *ptr = (Scalar*)val_ptr;
        mod_log->debug(M_ATTR, Mod_log::C_DATABASE,
            "  %s (%f, %f, %f) with arraysize %" FMT_SIZE_T,
            typ.get_name(), ptr[0], ptr[1], ptr[2], array_size);
        break;
      }
      case TYPE_BOOLEAN: {
        bool *ptr = (bool*)val_ptr;
        mod_log->debug(M_ATTR, Mod_log::C_DATABASE,
            "  %s, arraysize %" FMT_SIZE_T ", name %s",
            (*ptr? "true" : "false"), array_size, typ.get_name());
        break;
      }
      case TYPE_SCALAR: {
        if (array_size == 1) {
            Scalar *ptr = (Scalar*)val_ptr;
            mod_log->debug(M_ATTR, Mod_log::C_DATABASE, "  %f, name %s",
                *ptr, typ.get_name());
        }
        else
          //print_array<Scalar>(array_size, val_ptr);
          mod_log->debug(M_ATTR, Mod_log::C_DATABASE,
            "arrays currently not supported");
        break;
      }
      case TYPE_VECTOR2: {
        Scalar *ptr = (Scalar*)val_ptr;
        mod_log->debug(M_ATTR, Mod_log::C_DATABASE,
            "  (%f, %f), arraysize %" FMT_SIZE_T ", name %s",
            ptr[0], ptr[1], array_size, typ.get_name());
        break;
      }
      case TYPE_VECTOR3: {
        Scalar *ptr = (Scalar*)val_ptr;
        mod_log->debug(M_ATTR, Mod_log::C_DATABASE,
            "  (%f, %f, %f), arraysize %" FMT_SIZE_T ", name %s",
            ptr[0], ptr[1], ptr[2], array_size, typ.get_name());
        break;
      }
      case TYPE_VECTOR4: {
        Scalar *ptr = (Scalar*)val_ptr;
        mod_log->debug(M_ATTR, Mod_log::C_DATABASE,
            "  (%f, %f, %f, %f) with arraysize %" FMT_SIZE_T ", name %s",
            ptr[0], ptr[1], ptr[2], ptr[3], array_size, typ.get_name());
        break;
      }
      case TYPE_DSCALAR: {
        Dscalar *ptr = (Dscalar*)val_ptr;
        mod_log->debug(M_ATTR, Mod_log::C_DATABASE,
            "  %f, arraysize %" FMT_SIZE_T ", name %s",
            *ptr, array_size, typ.get_name());
        break;
      }
      case TYPE_DVECTOR2: {
        Dscalar *ptr = (Dscalar*)val_ptr;
        mod_log->debug(M_ATTR, Mod_log::C_DATABASE,
            "  (%f, %f), arraysize %" FMT_SIZE_T ", name %s",
            ptr[0], ptr[1], array_size, typ.get_name());
        break;
      }
      case TYPE_DVECTOR3: {
        Dscalar *ptr = (Dscalar*)val_ptr;
        mod_log->debug(M_ATTR, Mod_log::C_DATABASE,
            "  (%f, %f, %f), arraysize %" FMT_SIZE_T ", name %s",
            ptr[0], ptr[1], ptr[2], array_size, typ.get_name());
        break;
      }
      case TYPE_DVECTOR4: {
        Dscalar *ptr = (Dscalar*)val_ptr;
        mod_log->debug(M_ATTR, Mod_log::C_DATABASE,
            "  (%f, %f, %f, %f), arraysize %" FMT_SIZE_T ", name %s",
            ptr[0], ptr[1], ptr[2], ptr[3], array_size, typ.get_name());
        break;
      }
      case TYPE_MATRIX: {
//	Scalar *ptr = (Scalar*)val_ptr;
        break;
      }
      case TYPE_DMATRIX: {
//	Dscalar *ptr = (Dscalar*)val_ptr;
        break;
      }
      case TYPE_TAG: {
        Uint32 *ptr = (Uint32*)val_ptr;
        DB::Tag tag(*ptr);
        mod_log->debug(M_ATTR, Mod_log::C_DATABASE,
            "  arraysize %" FMT_SIZE_T ", reference to %s, name %s",
            array_size,
            (tag.is_valid()? transaction->tag_to_name(tag) : "<invalid>"),
            typ.get_name());
        for (size_t i=0; i<array_size; ++i) {
            DB::Tag tag(*ptr);
            std::string name;
            if (tag)
                name = transaction->tag_to_name(tag);
            else
                name = "<invalid>";
            mod_log->debug(M_ATTR, Mod_log::C_DATABASE,
                "  elem[%" FMT_SIZE_T "] = %u, reference to %s",
                i, *ptr++, name.c_str());
        }
        break;
      }
      case TYPE_ID:
      case TYPE_PARAMETER:
      case TYPE_TEMPORARY:
      case TYPE_LIGHT:
      case TYPE_INT32: {
        Uint32 *ptr = (Uint32*)val_ptr;
        mod_log->debug(M_ATTR, Mod_log::C_DATABASE,
            "  %u with arraysize %" FMT_SIZE_T ", name %s",
            *ptr, array_size, typ.get_name());
        break;
      }
      case TYPE_STRUCT: {
        mod_log->debug(M_ATTR, Mod_log::C_DATABASE,
            "\tnow diving into struct \"%s\"...", typ.get_name());
        break;
      }
      case TYPE_ATTACHABLE: {
        mod_log->debug(M_ATTR, Mod_log::C_DATABASE,
            "\tnow diving into ref proxy \"%s\"...", typ.get_name());
        break;
      }
      case TYPE_CALL: {
        mod_log->debug(M_ATTR, Mod_log::C_DATABASE,
            "\tnow diving into call \"%s\"...", typ.get_name());
        break;
      }
      case TYPE_RLE_UINT_PTR: {
        CONT::Rle_array<Uint> **val = (CONT::Rle_array<Uint>**)val_ptr;
        CONT::Rle_array<Uint> *ptr = *val;
        mod_log->debug(M_ATTR, Mod_log::C_DATABASE,
            "  %" FMT_SIZE_T " elements, name %s",
            ptr->size(), typ.get_name());
        for (size_t i=0; i<ptr->size(); ++i)
            mod_log->debug(M_ATTR, Mod_log::C_DATABASE,
                "  elem[%" FMT_SIZE_T "] = %u",
                i, (*ptr)[i]);
        break;
      }
      case TYPE_ARRAY: {
        // not handled yet
        ASSERT(M_ATTR, 0);
        break;
      }
      default:
        mod_log->debug(M_ATTR, Mod_log::C_DATABASE,
            "  value of type %s currently not handled",
            ((code <= TYPE_NUM)? Type::type_name(code) : "undefined"));
        break;
    }

}


}
