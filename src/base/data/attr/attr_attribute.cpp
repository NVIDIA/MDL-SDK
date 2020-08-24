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
/// \brief Attribute class implementation.
 
 
#include "pch.h"

#include "i_attr_attribute.h"
#include "attr.h"
#include "attr_module.h"
#include "i_attr_utilities.h"

#include <base/data/db/i_db_transaction.h>
#include <base/data/serial/i_serializer.h>
#include <base/lib/cont/i_cont_rle_array.h>
#include <base/lib/log/i_log_logger.h>
#include <base/lib/zlib/i_zlib.h>
#include <base/system/main/access_module.h>

namespace MI {
namespace ATTR {

using namespace std;
using namespace LOG;
using namespace CONT;

//==================================================================================================

// Default constructor.
Attachment::Attachment()
  : m_is_interface(false)
{}


//==================================================================================================

// This default constructor should only be used during deserialization.
Attribute::Attribute()
  : m_id(0),
    m_override(PROPAGATION_STANDARD),
    m_values(0),
    m_global(false)
{}


// create and destroy an attribute. Attributes are constructed with zero-filled
// values arrays that can later be accessed with get/set_value*. Attributes
// have a type, including structured types, but dynamic arrays are not allowed
// because it's unclear how the pointers get allocated and deleted.
Attribute::Attribute(
    const Type		&type,		// data type, may be list or tree
    Attribute_propagation override)	// inheritance: parent overrides child
  : m_type(type)
{
    init(id_create(m_type.get_name()), override, 1, false, false);
#if ENABLE_ASSERTIONS
    if (m_type.get_typecode() == TYPE_ATTACHABLE) {
        const Type* ref = type.get_child();
        ASSERT(M_ATTR, ref->get_typecode == TYPE_REF);
        ASSERT(M_ATTR, ref->get_name() = 0);
        const Type* value = ref->get_next();
        ASSERT(M_ATTR, ref->get_typecode != TYPE_ATTACHABLE);
        ASSERT(M_ATTR, value->get_name() != 0);
        ASSERT(M_ATTR, value->get_next() == 0);
    }
#endif
    // just to avoid that some strangely configured enum collection enters the system
    ASSERT(M_ATTR, m_type.get_typecode() != TYPE_ENUM
        || (m_type.get_enum() && !m_type.get_enum()->empty()));
}

// create and destroy an attribute. Attributes are constructed with zero-filled
// values arrays that can later be accessed with get/set_value*. Attributes
// have a type, including structured types, but dynamic arrays are not allowed
// because it's unclear how the pointers get allocated and deleted.
Attribute::Attribute(
    Attribute_id	id,		// identifies attribute for lookups
    const Type		&type,		// data type, may be list or tree
    Attribute_propagation override)	// inheritance: parent overrides child
  : m_type(type)
{
    if (!m_type.get_name()) {
        SYSTEM::Access_module<ATTR::Attr_module> attr_module(false);
        m_type.set_name(attr_module->get_reserved_attr(id));
    }

    init(id, override, 1, false, false);

    // just to avoid that some strangely configured enum collection enters the system
    ASSERT(M_ATTR, m_type.get_typecode() != TYPE_ENUM
        || (m_type.get_enum() && !m_type.get_enum()->empty()));
}


// a convenience constructor like the preceding one, except it takes care
// of simple, non-structured types automatically. Also creates an attribute
// ID from the attribute name.
Attribute::Attribute(
    Type_code		type,		// primitive type: bool, int, ...
    const char		*name,		// name of attribute (stored in type)
    Uint		type_asize,	// number of elements, > 0
    Attribute_propagation override,	// inheritance: parent overrides child
    bool		global,		// not inheritable, nailed to element
    bool		is_const)	// is value immutable?
  : m_type(type, name, type_asize)
{
    ASSERT(M_ATTR,                      // that's most probably an error
        type != TYPE_STRUCT && type != TYPE_ARRAY && type != TYPE_ATTACHABLE && type != TYPE_CALL);
    ASSERT(M_ATTR, name && *name);

    init(id_create(name), override, 1, is_const, global);

    // just to avoid that some strangely configured enum collection enters the system
    ASSERT(M_ATTR, m_type.get_typecode() != TYPE_ENUM
        || (m_type.get_enum() && !m_type.get_enum()->empty()));
}


// another convenience constructor for the derived Attribute_object.
Attribute::Attribute(
    Attribute_id	id,		// identifies attribute for lookups
    Type_code		type,		// primitive type: bool, int, ...
    Uint		type_asize,	// number of elements > 0
    Attribute_propagation override,	// inheritance: parent overrides child
    bool		global,		// not inheritable, nailed to element
    bool		is_const)	// is value immutable?
  : m_type(type, 0, type_asize)
{
    SYSTEM::Access_module<ATTR::Attr_module> attr_module(false);
    m_type.set_name(attr_module->get_reserved_attr(id));

    ASSERT(M_ATTR,                      // that's most probably an error
        type != TYPE_STRUCT && type != TYPE_ARRAY && type != TYPE_ATTACHABLE && type != TYPE_CALL);

    init(id, override, 1, is_const, global);

    // just to avoid that some strangely configured enum collection enters the system
    ASSERT(M_ATTR, m_type.get_typecode() != TYPE_ENUM
        || (m_type.get_enum() && !m_type.get_enum()->empty()));
}


// TODO
/*namespace {*/ void copy_constructor(Type_iterator&, Type_iterator&); /*}*/

Attribute::Attribute(
    const Attribute& other)
{
    // (1) do the work of Attribute::Attribute() constructor
    m_type = other.m_type;
    init(
        other.m_id,
        other.m_override,
        other.get_listsize(),
        other.m_type.get_const(),
        other.m_global);

    //const size_t size = other.m_type.sizeof_all();
    if (!m_type.get_name()) {
        SYSTEM::Access_module<ATTR::Attr_module> attr_module(false);
        m_type.set_name(attr_module->get_reserved_attr(m_id));
    }

    // (2) now copy the actual data - here at most the first list entry only
    if (other.get_listsize()) {
        Type_iterator in_iter(&other.get_type(), const_cast<char*>(other.get_values()));
        Type_iterator out_iter(&get_type(), set_values());
        copy_constructor(in_iter, out_iter);
    }

    // (3) finally
    m_attachments = other.m_attachments;

    // just to avoid that some strangely configured enum collection enters the system
    ASSERT(M_ATTR, m_type.get_typecode() != TYPE_ENUM
        || (m_type.get_enum() && !m_type.get_enum()->empty()));
}


namespace { void destructor(Type_iterator&); }

// destructor. The type tree lives on the heap, delete all attached Type nodes.
Attribute::~Attribute()
{
    for (Uint l=0; l<this->get_listsize(); ++l) {
        Type_iterator iter(&get_type(), set_values_i(l));
        destructor(iter);
    }
    deallocate_storage(m_values);
}


void* Attribute::allocate_storage(size_t n, bool zero_init)
{
    // enforce minimum alignment of word size for Dynamic_array struct
    size_t s = sizeof(size_t);
    size_t k = (n + s-1) / s;
    void* p = new size_t[k];
    if (zero_init && p && n > 0) {                      //-V668 PVS-Studio
        memset(p, 0, n);
    }
    return p;
}

void Attribute::deallocate_storage(void* p)
{
    delete [] (size_t*) p;
}


// Duplicate the given string, using Attributes allocator.
static char* attr_string_dup(const char* str)
{
    if (!str)
        return 0;
    const size_t len = strlen(str) + 1;
    ASSERT(M_ATTR, len <= numeric_limits<Uint32>::max());
    char* ret = static_cast<char*>(Attribute::allocate_storage(len));
    memcpy(ret, str, len);
    return ret;
}

void Attribute::set_string(char* & storage, const char* str)
{
    if (storage) {
        Attribute::deallocate_storage(storage);
    }
    storage = attr_string_dup(str);
}


// not inlined, needs string_dup
void Attribute::set_value_string(
    const char      *v,
    Uint            n)
{
    ASSERT(M_ATTR, contains_expected_type(m_type, TYPE_STRING));
    ASSERT(M_ATTR, n < m_type.get_arraysize());
    if (n < m_type.get_arraysize()) {
        char*& p = ((char**)m_values)[n];
        set_string(p, v);
    }
}

// not inlined, needs string_dup
void Attribute::set_value_string_i(
    const char      *v,
    Uint            i,
    Uint            n)
{
    ASSERT(M_ATTR, contains_expected_type(m_type, TYPE_STRING));
    ASSERT(M_ATTR, i < get_listsize());
    ASSERT(M_ATTR, n < m_type.get_arraysize());
    if (n < m_type.get_arraysize()) {
        char*& p = ((char**)m_values)[i * m_type.get_arraysize() + n];
        set_string(p, v);
    }
}


// flush the attribute's value array, return the amount of memory flushed in bytes
size_t Attribute::flush()
{
    size_t size = get_type().sizeof_all();

    for (Uint l=0; l<this->get_listsize(); ++l) {
        Type_iterator iter(&get_type(), set_values_i(l));
        destructor(iter);
    }
    char* junk = m_values;			// set memory pointer to 0
    m_values = 0;
    deallocate_storage(junk);

    return size;
}


// user-defined attributes are named, but the attribute system deals only
// with integer IDs. Create a new ID for a name. ID 0 is reserved and cannot
// be registered. This creates an ID from the name using a hash. Should use
// MD5 here as soon as we have this in base/lib! Never hash to 0..reserved_ids,
// they are reserved for the fixed boolean attributes in Attribute_set and
// fast predefined IDs in SCENE.
Attribute_id Attribute::id_create(
    const char		*name)		// new name to register
{
    return id_lookup(name);		// just hashing, no net symtab
}


Attribute_id Attribute::id_lookup(
    const char		*name)		// name to look up
{
    if (!name)
        return null_index;

    Uint attr_id = Attr_module_impl::s_attr_module->m_registry.get_id(name);
    if (attr_id == null_index) {
        attr_id = ZLIB::crc32(name, strlen(name));
        if (attr_id < reserved_ids)
            attr_id = ~attr_id;
    }

    return attr_id;
}


// Add an attachment to the internal list.
void Attribute::add_attachment(
    const Attachment &attachment)	// the attachment
{
    m_attachments.append(attachment);
}


// Remove an attachment from the internal list.
void Attribute::remove_attachment(
    const char *name)       // the member name
{
    const string member_name = name ? name : "";

    int removal_index = -1;
    for(size_t count = 0; ((count < m_attachments.size()) && (-1 == removal_index)); ++count)
        if(m_attachments[count].m_member_name == member_name)
            removal_index = count;

    if(-1 != removal_index)
        m_attachments.remove(removal_index);
}


// serialize the object to the given serializer including all sub elements.
// It must return a pointer behind itself (e.g. this + 1) to handle arrays.
const SERIAL::Serializable* Attribute::serialize(
    SERIAL::Serializer* serializer) const	// useful functions for byte streams
{
    serializer->write(m_id);
    const Uint32 override_as_uint = m_override;
    serializer->write(override_as_uint);

    m_type.serialize(serializer);

    char* values = m_values;
    const size_t size = m_type.sizeof_all();

    for (Uint i = 0; i < get_listsize(); i++) {
    m_type.serialize_data(serializer, values);
    values += size;
    }

    serializer->write((Uint32)m_attachments.size());
    for (Uint i = 0; i < m_attachments.size(); ++i) {
        const Attachment& attachment = m_attachments[i];
        serializer->write(attachment.m_member_name);
        serializer->write(attachment.m_target);
        serializer->write(attachment.m_target_name);
        serializer->write(attachment.m_is_interface);
    }

    serializer->write(m_global);

    return this+1;
}


const SERIAL::Serializable * Attribute::serialize_no_values(
    SERIAL::Serializer*serializer ) const
{
    serializer->write(m_id);
    const Uint32 override_as_uint = m_override;
    serializer->write(override_as_uint);

    m_type.serialize(serializer);

    serializer->write((Uint32)m_attachments.size());
    for (Uint i = 0; i < m_attachments.size(); ++i) {
        const Attachment& attachment = m_attachments[i];
        serializer->write(attachment.m_member_name);
        serializer->write(attachment.m_target);
        serializer->write(attachment.m_target_name);
        serializer->write(attachment.m_is_interface);
    }

    serializer->write(m_global);

    return this+1;
}





// deserialize the object and all sub-objects from the given deserializer.
// It must return a pointer behind itself (e.g. this + 1) to handle arrays.
SERIAL::Serializable* Attribute::deserialize(
    SERIAL::Deserializer* deser)		// useful functions for byte streams
{
    deser->read(&m_id);
    Uint32 override_as_uint = PROPAGATION_STANDARD;
    deser->read(&override_as_uint);
    m_override = Attribute_propagation(override_as_uint);
//    deser->read(&m_listsize);
    m_type.deserialize(deser);

    m_values = get_listsize() > 0 ?
        static_cast<char*>(allocate_storage(m_type.sizeof_all() * get_listsize())) : 0;
    char* values = m_values;
    const size_t size = m_type.sizeof_all();

    for (Uint i = 0; i < get_listsize(); ++i) {
        m_type.deserialize_data(deser, values);
        values += size;
    }

    Uint nr_of_attachments;
    deser->read(&nr_of_attachments);
    for (Uint i = 0; i < nr_of_attachments; ++i) {
        Attachment attachment;
        deser->read(&attachment.m_member_name);
        deser->read(&attachment.m_target);
        deser->read(&attachment.m_target_name);
        deser->read(&attachment.m_is_interface);
        m_attachments.append(attachment);
    }

    deser->read(&m_global);

    return this+1;
}


// dump attribute to debug messages, for debugging only.
void Attribute::dump() const
{
    std::string head("attribute id ");
    head += std::to_string(m_id);
    if(m_global)
        head += " global";
    if(m_override == PROPAGATION_OVERRIDE)
        head += " override";
    mod_log->debug(M_ATTR, LOG::Mod_log::C_DATABASE, "%s", head.c_str());
    dump_attr_values(get_type(),get_name(),get_values(),1);
    for (Uint i=0; i < m_attachments.size(); i++) {
        const Attachment *at = &m_attachments[i];
        mod_log->debug(M_ATTR, LOG::Mod_log::C_DATABASE,
                        "\tattachment: %s = shadertag_" FMT_TAG " %s%s",
                        at->m_member_name.empty() ? "<all>" : at->m_member_name.c_str(),
                        at->m_target.get_uint(),
                        at->m_is_interface ? "interface " : "",
                        at->m_target_name.empty() ? "<all>" : at->m_target_name.c_str());
    }
}

// factory function used for deserialization
SERIAL::Serializable *Attribute::factory()
{
    return new Attribute;
}


const char* Attribute::get_name() const
{
    const char* name = Attr_module_impl::s_attr_module->get_reserved_attr(m_id);
    if (!name)
        name = m_type.get_name();
    return name;
}


/// A clone method that may optionally change the name.
Attribute* Attribute::clone(const char *name) const
{
    Attribute* result = new Attribute(*this);
    if(name) {
        result->m_type.set_name(name);
        result->m_id = id_create(name);
    }
    return result;
}

/// The assignment operator.
Attribute& Attribute::operator=(const Attribute& other)
{
    if (&other == this)
        return *this;

    // cleaning up first
    Type_iterator iter(&get_type(), set_values());
    destructor(iter);
    deallocate_storage(m_values);

    // (1) do the work of Attribute::Attribute() constructor
    m_type = other.m_type;
    init(
        other.m_id,
        other.m_override,
        other.get_listsize(),
        other.m_type.get_const(),
        other.m_global);

    //const size_t size = other.m_type.sizeof_all();

    if (!m_type.get_name()) {
        SYSTEM::Access_module<ATTR::Attr_module> attr_module(false);
        m_type.set_name(attr_module->get_reserved_attr(m_id));
    }

    // (2) now copy the actual data
    Type_iterator in_iter(&other.get_type(), const_cast<char*>(other.get_values()));
    Type_iterator out_iter(&get_type(), set_values());
    copy_constructor(in_iter, out_iter);

    // (3) finally
    m_attachments = other.m_attachments;

    return *this;
}

// TODO Clean-up this hack where inside scene_object_attr.cpp the following function
// copy_constructor() is used, too.
//namespace {

/// Local function to copy the data of the in iterator into the data fields of the out iterator.
/// It requires that the out iterator points to freshly allocated (or probably cleaned up)
/// memory and is used from inside the copy constructor. Hence its very name.
/// Note that this function is not aware of listsize, ie the client has to take care of that!
void copy_constructor(
    Type_iterator& in_it,		// in iterator
    Type_iterator& out_it)		// out iterator
{
    // iterate over all elements of the type and copy them type-accordingly
    // Please note that the client's code takes care of the listsize.
    for (; !in_it.at_end(); in_it.to_next(), out_it.to_next()) {
        char* in_value = in_it.get_value();
        char* out_value = out_it.get_value();
        int arraysize = in_it->get_arraysize();

        // a dynamic array - hence create out_array and reset the value pointers accordingly
        if (!arraysize && in_it->get_typecode() != TYPE_ARRAY) {
            Dynamic_array* in_array = (Dynamic_array*)in_value;
            Dynamic_array* out_array = (Dynamic_array*)out_value;
            arraysize = in_array->m_count;
            out_array->m_count = in_array->m_count;
            out_array->m_value =
                static_cast<char*>(Attribute::allocate_storage(
                    out_it->sizeof_elem() * arraysize));

            // no alignment required here
            in_value = in_array->m_value;
            out_value = out_array->m_value;
        }

        // arrays can be handled recursively without any changes to the value pointers
        if (in_it->get_typecode() == TYPE_ARRAY) {
            Type_iterator in_sub(&in_it, in_value);
            Type_iterator out_sub(&out_it, out_value);
            size_t size = in_it->sizeof_elem() * in_it->get_arraysize();
            copy_constructor(in_sub, out_sub);

            // increase offset by the array element type's size since an array has no size itself
            in_value += size;
            out_value += size;
        }
        // rle arrays reset the value pointers accordingly
        else if (in_it->get_typecode() == TYPE_RLE_UINT_PTR) {
            Rle_array<Uint>* in_array = *(Rle_array<Uint>**)in_value;
            Rle_array<Uint>* out_array = new Rle_array<Uint>;
            Rle_chunk_iterator<Uint> iterator = in_array->begin_chunk();
            size_t size = in_array->get_index_size();
            for (size_t i=0; i < size; ++i) {
                out_array->push_back(iterator.data(), iterator.count());
                ++iterator;
            }
            *(Rle_array<Uint>**)out_value = out_array;

            in_value += sizeof(Rle_array<Uint>*);
            out_value += sizeof(Rle_array<Uint>*);
        }
        // structs can be handled recursively
        else if (in_it->get_typecode() == TYPE_STRUCT || in_it->get_typecode() == TYPE_ATTACHABLE
            || in_it->get_typecode() == TYPE_CALL)
        {
            size_t size = in_it->sizeof_elem();
            for (int a=0; a < arraysize; ++a) {
                if (in_it->get_typecode() == TYPE_CALL) {
                    // name - which now is actually a tag, but still occupies space of a pointer
                    memcpy(out_value, in_value, 4);
                    in_value += Type::sizeof_one(TYPE_STRING);
                    out_value += Type::sizeof_one(TYPE_STRING);
                    // parameters
                    Attribute::set_string(*(char**)out_value, *(const char**)in_value);
                    in_value += Type::sizeof_one(TYPE_STRING);
                    out_value += Type::sizeof_one(TYPE_STRING);
                }
                Type_iterator in_sub(&in_it, in_value);
                Type_iterator out_sub(&out_it, out_value);
                copy_constructor(in_sub, out_sub);
                out_value += size;
                in_value += size;
                if (in_it->get_typecode() == TYPE_CALL) {
                    out_value -= 2*Type::sizeof_one(TYPE_STRING);
                    in_value -= 2*Type::sizeof_one(TYPE_STRING);
                }
            }
        }
        else {
            int type, count, size;
            Type_code type_code = in_it->get_typecode();
            eval_typecode(type_code, &type, &count, &size);
            for (int a=0; a < arraysize; ++a) {
                for (int i=0; i < count; ++i) {

                    switch(type) {
                      case '*': {
                          if (*(char **)in_value) {
                              Attribute::set_string(
                                  *(char**)out_value,
                                  *(const char**)in_value);
                          }
                          break; }
                      case 'c':
                      case 's':
                      case 'i':
                      case 'q':
                      case 'f':
                      case 'd':
                      case 't':
                          memcpy(out_value, in_value, size);
                          break;
                      default:  ASSERT(M_ATTR, 0); break;
                    }
                    out_value += size;
                    in_value += size;
                }
            }
        }
    }
}

// TODO This can go as soon as the above function gets into the anonymous namespace again.
namespace {

/// Local function to free up all dynamic memory within the data fields of the iter.
/// This function is used from iside the destructor. Hence its very name.
/// Note that this function is not aware of listsize, ie the client has to take care of that!
void destructor(Type_iterator& iter)
{
    // iterate over all elements of the type
    for (; !iter.at_end(); iter.to_next()) {
        char* values = iter.get_value();
        int arraysize = iter->get_arraysize();

        if (!arraysize && iter->get_typecode() != TYPE_ARRAY) {
            Dynamic_array* array = (Dynamic_array*)values;
            values = array->m_value;
            arraysize = array->m_count;
        }

        if (iter->get_typecode() == TYPE_ARRAY) {
            size_t size = iter->sizeof_elem() * iter->get_arraysize();
            Type_iterator sub(&iter, values);
            destructor(sub);

            // increase offset by the array element type's size since an array has no size itself
            values += size;
        }
        else if (iter->get_typecode() == TYPE_RLE_UINT_PTR) {
            Rle_array<Uint>* array = *(Rle_array<Uint>**)values;
            delete array;
            values += sizeof(Rle_array<Uint>*);
        }
        else if (iter->get_typecode() == TYPE_STRUCT || iter->get_typecode() == TYPE_ATTACHABLE
            || iter->get_typecode() == TYPE_CALL)
        {
            size_t size = iter->sizeof_elem();
            for (int a=0; a < arraysize; ++a) {
                if (iter->get_typecode() == TYPE_CALL) {
                    // clean up initial string coming next to the tag
                    // nothing to do for the initial tag - but it occupies space of a pointer
                    values += Type::sizeof_one(TYPE_STRING);
                    Attribute::deallocate_storage(*(char **)values);
                    values += Type::sizeof_one(TYPE_STRING);
                }
                Type_iterator sub(&iter, values);
                destructor(sub);
                values += size;
                if (iter->get_typecode() == TYPE_CALL)
                    values -= 2*Type::sizeof_one(TYPE_STRING);
            }
        }
        else {
            int type, count, size;
            Type_code type_code = iter->get_typecode();
            eval_typecode(type_code, &type, &count, &size);
            for (int a=0; a < arraysize; ++a) {
                for (int i=0; i < count; ++i) {
                    switch(type) {
                      case '*':
                          Attribute::deallocate_storage(*(char **)values);
                          break;
                      case 'c':
                      case 's':
                      case 'i':
                      case 'q':
                      case 'f':
                      case 'd':
                      case 't':
                          break;
                      default:  ASSERT(M_ATTR, 0); break;
                    }
                    values += size;
                }
            }
        }
        if (!iter->get_arraysize() && iter->get_typecode() != TYPE_ARRAY) {
            // this is a dynamic array
            Dynamic_array* array = (Dynamic_array*)iter.get_value();
            Attribute::deallocate_storage(array->m_value);
        }
    }
}
}

//==================================================================================================
//
// Set of protected contructors to create appropriate m_values memory

void Attribute::init(
    Attribute_id id,
    Attribute_propagation override,
    Uint list_size,
    bool is_const,
    bool is_global)
{
    ASSERT(M_ATTR, m_type.get_typecode() != TYPE_UNDEF);
    m_type.set_const(is_const);

    m_id = id;
    m_override = override;
    m_global = is_global;

    m_values = 0;
    const size_t storage_size = m_type.sizeof_all() * list_size;
    if (storage_size > 0) {
        m_values = static_cast<char*>(allocate_storage(storage_size));
    }
}


// create and destroy an attribute. Attributes are constructed with zero-filled
// values arrays that can later be accessed with get/set_value*. Attributes
// have a type, including structured types, but dynamic arrays are not allowed
// because it's unclear how the pointers get allocated and deleted.
//
Attribute::Attribute(
    Attribute_id	id,		// identifies attribute for lookups
    const Type		&type,		// data type, may be list or tree
    Uint		list_size,	// if attribute list, list size > 1
    Attribute_propagation override)	// inheritance: parent overrides child
: m_id(~0u)
, m_override(PROPAGATION_UNDEF)
, m_type(type)
, m_values(nullptr)
, m_global(false)
{
    if (!m_type.get_name()) {
        SYSTEM::Access_module<ATTR::Attr_module> attr_module(false);
        m_type.set_name(attr_module->get_reserved_attr(id));
    }
    // just to avoid that some strangely configured enum collection enters the system
    ASSERT(M_ATTR, m_type.get_typecode() != TYPE_ENUM
        || (m_type.get_enum() && !m_type.get_enum()->empty()));
}


Attribute::Attribute(
    Type_code		type,		// primitive type: bool, int, ...
    const char		*name,		// name of attribute (stored in type)
    Uint		type_asize,	// number of elements, > 0
    Uint		list_size,	// if attribute list, list size > 1
    Attribute_propagation override,	// inheritance: parent overrides child
    bool		global,		// not inheritable, nailed to element
    bool		is_const)	// is value immutable?
: m_id(~0u)
, m_override(PROPAGATION_UNDEF)
, m_type(type, name, type_asize)
, m_values(nullptr)
, m_global(false)
{
    ASSERT(M_ATTR, // that's most probably an error
        type != TYPE_STRUCT && type != TYPE_ARRAY && type != TYPE_ATTACHABLE && type != TYPE_CALL);
    ASSERT(M_ATTR, name && *name);
    // just to avoid that some strangely configured enum collection enters the system
    ASSERT(M_ATTR, m_type.get_typecode() != TYPE_ENUM
        || (m_type.get_enum() && !m_type.get_enum()->empty()));
}


//
// another convenience constructor for the derived Attribute_object.
//

Attribute::Attribute(
    Attribute_id	id,		// identifies attribute for lookups
    Type_code		type,		// primitive type: bool, int, ...
    Uint		type_asize,	// number of elements > 0
    Uint		list_size,	// if attribute list, list size > 1
    Attribute_propagation override,	// inheritance: parent overrides child
    bool		global,		// not inheritable, nailed to element
    bool		is_const)	// is value immutable?
: m_id(~0u)
, m_override(PROPAGATION_UNDEF)
, m_type(type, 0, type_asize)
, m_values(nullptr)
, m_global(false)
{
    SYSTEM::Access_module<ATTR::Attr_module> attr_module(false);
    m_type.set_name(attr_module->get_reserved_attr(id));

    ASSERT(M_ATTR,                      // that's most probably an error
        type != TYPE_STRUCT && type != TYPE_ARRAY && type != TYPE_ATTACHABLE && type != TYPE_CALL);

    // just to avoid that some strangely configured enum collection enters the system
    ASSERT(M_ATTR, m_type.get_typecode() != TYPE_ENUM
        || (m_type.get_enum() && !m_type.get_enum()->empty()));
}

namespace {
size_t calculate_size(Type_iterator& iter)
{
    size_t res = 0;

    // iterate over all elements of the type
    for (; !iter.at_end(); iter.to_next()) {
        char* values = iter.get_value();
        int arraysize = iter->get_arraysize();

        if (!arraysize && iter->get_typecode() != TYPE_ARRAY) {
            Dynamic_array* array = (Dynamic_array*)values;
            res += sizeof(Dynamic_array);

            // update offset
            values = array->m_value;
            arraysize = array->m_count;
        }

        if (iter->get_typecode() == TYPE_ARRAY) {
            size_t size = iter->sizeof_elem() * iter->get_arraysize();
            Type_iterator sub(&iter, values);
            res += calculate_size(sub);

            // increase offset by the array element type's size since an array has no size itself
            values += size;
        }
        else if (iter->get_typecode() == TYPE_RLE_UINT_PTR) {
            Rle_array<Uint>* array = *(Rle_array<Uint>**)values;
            res += sizeof(Rle_array<Uint>)
                + array->get_byte_size()
                // this term is based on assumption that indices are of type size_t
                + array->get_index_size() * sizeof(size_t);

            // increase offset
            values += sizeof(Rle_array<Uint>*);
        }
        else if (iter->get_typecode() == TYPE_STRUCT || iter->get_typecode() == TYPE_ATTACHABLE
            || iter->get_typecode() == TYPE_CALL)
        {
            if (iter->get_typecode() == TYPE_CALL)
                values += 2*Type::sizeof_one(TYPE_STRING);
            size_t size = iter->sizeof_elem();
            for (int a=0; a < arraysize; ++a) {
                Type_iterator sub(&iter, values);
                res += calculate_size(sub);
                values += size;
            }
        }
        else {
            int type, count, size;
            Type_code type_code = iter->get_typecode();
            eval_typecode(type_code, &type, &count, &size);
            if (type != '*')
                res += size*static_cast<size_t>(count)*arraysize;
            else {
                ASSERT(M_ATTR, count == 1);
                res += arraysize * sizeof(char**);
                for (int a=0; a < arraysize; ++a) {
                    if (const char* str = *(char **)values)
                        res += strlen(str) + 1;
                    values += size;
                }
            }
        }
    }
    return res;
}
}

// Return the approximate size in bytes of the element including all its
// substructures. This is used to make decisions about garbage collection.
size_t Attribute::get_size() const
{
    size_t res = sizeof(*this);
    res += m_type.get_size() - sizeof(Type);
    // get real size of the Type-specific m_values block
    for (Uint l=0; l < get_listsize(); ++l) {
        Type_iterator iter(&get_type(), const_cast<char*>(get_values_i(l)));
        res += calculate_size(iter);
    }
    res += m_attachments.size() * sizeof(Attachment);
    return res;
}


}
}
