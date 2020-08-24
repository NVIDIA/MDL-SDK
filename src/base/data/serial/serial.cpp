/***************************************************************************************************
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
 **************************************************************************************************/
/// \file
/// \brief This module provides an implementation for the serializer and deserializer framework. 
///
/// Currently it does not do anything.

#include "pch.h"

#include "serial.h"

#include <set>

#include <mi/base/lock.h>
#include <mi/math/color.h>
#include <base/lib/mem/i_mem_allocatable.h>
#include <base/lib/cont/i_cont_bitvector.h>
#include <limits>
#include <base/lib/log/log.h>

using namespace std;

namespace MI
{

namespace SERIAL
{

// This class is used to store the factory function for a given class id.
class Deserialization_class : public MI::MEM::Allocatable
{
  public:
    // constructor
    Deserialization_class(
        Class_id class_identifier,		// class id to be registered
        Factory_function* factory);		// factory for this class id

    // constructor
    Deserialization_class(
        Class_id class_identifier,		// class id to be registered
        IDeserialization_factory* factory_class); // factory class for this class id

    // constructor mainly used for searching
    Deserialization_class(
        Class_id class_identifier);

    Class_id m_class_identifier;		// class id to be registered
    Factory_function* m_factory;		// factory for this class id
    IDeserialization_factory* m_factory_class;	// factory class this class id
                                                // mutually exclusive with factory
};


// This class is used to compare Deserialization classes by id
class Deserialization_class_compare
{
  public:
    bool operator() (const Deserialization_class &c1, const Deserialization_class &c2) const
        { return (c1.m_class_identifier < c2.m_class_identifier); }

};


// The deserialization manager is needed to provide an instance of a class
// identified by the class id found in the deserialization stream. This class
// will then provide the deserialization function.
class Deserialization_manager_impl : public Deserialization_manager
{
  public:
    // destructor
    ~Deserialization_manager_impl();

    // Any module may register a factory function for a given Class_id. When
    // the deserializer finds this Class_id it will call the given factory
    // function to create an appropriate object. This object then provides the
    // applicable deserialize function.
    void register_class(
        Class_id class_identifier,		// class id to be registered
        Factory_function* factory);		// factory for this class id

    // Any module may register a factory function for a given Class_id. When the deserializer finds
    // this Class_id it will call use the given factory class to create an appropriate object. This
    // object then provides the applicable deserialize function.
    void register_class(
        Class_id class_identifier,			// class id to be registered
        IDeserialization_factory* factory);		// factory class for this class id

    // Call the appropriate factory function for the given class id and return
    // it.
    Serializable* construct(
        Class_id class_identifier);		// class id to be constructed

    // Check, if a class has been registered.
    bool is_registered(
        Class_id class_identifier);		// class id to be checked

  private:
      mi::base::Lock m_lock;			// lock for protecting the set

    // Container for keeping deserialization classes
    typedef std::set<Deserialization_class,  Deserialization_class_compare> Class_set;
    Class_set	 m_classes; 			// map of registered class.
};

void Serializer_impl::serialize(const Serializable* serializable, bool shared)
{
    // Cast pointer type to an unsigned integer.
    Uint64 ptr( mi::base::binary_cast<Pointer_as_uint>(serializable) );

    // This is the id used when the shared flag is on. If off then this only needs to
    // be 0 for null serializables, and for instance 1 otherwise.
    Uint64 id = (serializable == 0 ? 0 : 1);

    // If serializable was null, write 0 and then we are done
    if (id == 0)
    {
        write(id);
        return;
    }

    // If shared flag is on we need to look up/create a unique id for this pointer within
    // this serialization context. (the pointer can't be used directly since this would mean
    // that the same object type with the same data would serialize to two different binary
    // chunks in different runs, which is bad for the cloud-bridge project)
    if (shared)
    {
        Id_map::iterator iter;
        iter = m_shared_id_map.find(ptr);
        if (iter == m_shared_id_map.end())
        {
            // Not written before, write and mark in map
            id = m_id_counter++;
            m_shared_id_map[ptr]=id;

            write(id);
            write(serializable->get_class_id());
            m_helper.serialize_with_end_marker(this, serializable);
        }
        else
        {
            // Written before, just write the id already assigned
            write(iter->second);
        }
    }
    else
    {
        // No sharing, just write the non-null marker and the data
        write(id);
        write(serializable->get_class_id());
        m_helper.serialize_with_end_marker(this, serializable);

    }
}

void Serializer_impl::write_size_t(size_t value)
{
    write((Uint64)value);
}

void Serializer_impl::write(const DB::Tag& value)
{
    write(value.get_uint());
}

void Serializer_impl::write(const char* value)
{
    if (value == NULL)
    {
        write_size_t(static_cast<size_t>(0));
        return;
    }

    size_t size = strlen(value);
    // serialize length+1 to distinguish NULL from '\0'
    write_size_t(size + 1u);
    write(value, size);
}

void Serializer_impl::write(const std::string& value)
{
    write_size_t(value.size() + 1u);
    write(value.c_str(), value.size());
}

void Serializer_impl::write(const mi::base::Uuid& value)
{
    write(value.m_id1);
    write(value.m_id2);
    write(value.m_id3);
    write(value.m_id4);
}

void Serializer_impl::write(const mi::math::Color& value)
{
    write(value.r);
    write(value.g);
    write(value.b);
    write(value.a);
}

void Serializer_impl::write(const CONT::Bitvector& value)
{
    const size_t nb8  = value.get_binary_size();
    const Uint8* const data = value.get_binary_data();
    write_size_t(value.size());
    write_size_t(nb8);
    write_range(*this, data, data + nb8);
}

void Serializer_impl::write(const CONT::Dictionary& value)
{
    ASSERT(M_SERIAL, !"should not be called");
}

void Serializer_impl::clear_shared_objects()
{
    m_shared_id_map.clear();
    m_id_counter = 1;
}

void Serializer_impl::start_extension()
{
    m_helper.set_extension_marker(this);
}

void Serializer_impl::write_direct(Serializer* serializer, const char* buffer, size_t size)
{
    Serializer_impl* si(reinterpret_cast<Serializer_impl*>(serializer));
    ASSERT(M_DB, si);
    si->write_impl(buffer, size);
}

class Deserializer_default_error_handler:
        public IDeserializer_error_handler<Serializable>
{
public:

    void handle(Marker_status status, const Serializable* serializable)
    {
        ASSERT(M_DB, status != MARKER_FOUND);

        Class_id class_id = serializable->get_class_id();

        if (status == MARKER_BAD_CHECKSUM)
            LOG::mod_log->fatal(M_SERIAL, LOG::Mod_log::C_MISC, 1,
                                "Deserialization error for class id 0x%x: bad checksum",
                                class_id);
        else if (status == MARKER_NOT_FOUND)
            LOG::mod_log->fatal(M_SERIAL, LOG::Mod_log::C_MISC, 1,
                                "Deserialization error for class id 0x%x: end marker not found.",
                                class_id);
        else
            LOG::mod_log->fatal(M_SERIAL, LOG::Mod_log::C_MISC, 1,
                                "Deserialization error for class id 0x%x: unknown, status=%d.",
                                class_id, status);
    }
};

Deserializer_impl::Deserializer_impl(
    Deserialization_manager* deserialization_manager)
    : m_deserialization_manager(deserialization_manager)
    , m_error_handler(new Deserializer_default_error_handler())
{
}

Serializable* Deserializer_impl::deserialize(bool shared)
{
    Uint64 id;
    read(&id);
    if (id == 0)
    {
        // Id 0 is reserved and means that a 0 pointer was encountered.
        return NULL;
    }

    if (!shared)
    {
        Class_id class_id;
        Serializable* serializable;

        read(&class_id);
        ASSERT(M_DB, m_deserialization_manager != NULL);
        serializable = m_deserialization_manager->construct(class_id);
        ASSERT(M_DB, serializable != NULL);

        Marker_status status = m_helper.deserialize_with_end_marker(this, serializable);
        if (status != MARKER_FOUND)
            m_error_handler->handle(status, serializable);

        return serializable;
    }

    Reference_map::iterator i = m_objects.find(id);
    if (i == m_objects.end())
    {
        Class_id class_id;
        Serializable* serializable;

        read(&class_id);
        ASSERT(M_DB, m_deserialization_manager != NULL);
        serializable = m_deserialization_manager->construct(class_id);
        ASSERT(M_DB, serializable != NULL);

        // This object has not been deserialized. So do it now. Make sure it's
        // in the map already (with a wrong pointer), so that it is not
        // deserialized multiple times, if the same pointer is encountered
        // while deserializing this one.
        m_objects.insert(make_pair(id, serializable));

        Marker_status status = m_helper.deserialize_with_end_marker(this, serializable);
        if (status != MARKER_FOUND)
            m_error_handler->handle(status, serializable);

        return serializable;
    }

    return i->second;
}

void Deserializer_impl::read_size_t(size_t* value)
{
    Uint64 value64;
    read(&value64);
#ifndef BIT64
    if (value64 > std::numeric_limits<size_t>::max())
    {
        LOG::mod_log->fatal(M_SERIAL, LOG::Mod_log::C_MISC, 1,
            "Received memory block size which cannot be handled"
            " by a 32 bit machine. Remove 32 bit machines from the cluster and try again.");
    }
#endif
    *value = (size_t)value64;
}

void Deserializer_impl::read(DB::Tag* value_pointer)
{
    Uint32 tag;
    read(&tag);
    *value_pointer = DB::Tag(tag);
}

void Deserializer_impl::read(char** value_pointer)
{
    size_t size;
    read_size_t(&size);

    if (size == 0)
    {
        *value_pointer = NULL;
        return;
    }

    // length+1 was serialized to distinguish NULL from '\0'
    *value_pointer = new char[size-1 + 1];
    read(*value_pointer, size-1);
    (*value_pointer)[size-1] = '\0';
}

void Deserializer_impl::read(std::string* value_pointer)
{
    size_t size;
    read_size_t(&size);

    if (size == 0) {
        // This should not happen unless someone serializes a NULL const char* and
        // deserializes it as std::string.
        value_pointer->clear();
        return;
    }

    // length+1 was serialized to distinguish NULL from '\0'
    size -= 1;
    value_pointer->resize(size);
    if (size > 0)
        read(&((*value_pointer)[0]), size);
}

void Deserializer_impl::read(mi::base::Uuid* value_pointer)
{
    read(&value_pointer->m_id1);
    read(&value_pointer->m_id2);
    read(&value_pointer->m_id3);
    read(&value_pointer->m_id4);
}

void Deserializer_impl::read(mi::math::Color* value_pointer)
{
    read(&value_pointer->r);
    read(&value_pointer->g);
    read(&value_pointer->b);
    read(&value_pointer->a);
}

void Deserializer_impl::read(CONT::Bitvector* value_pointer)
{
    size_t size;
    size_t nb8;
    read_size_t(&size);
    read_size_t(&nb8);
    Uint8 *data = MEM::new_array<Uint8>(nb8);
    read_range(*this, data, data + nb8);
    value_pointer->set_binary_data(size, data);
    MEM::delete_array<Uint8>(data);
}

void Deserializer_impl::read(CONT::Dictionary *value_pointer)
{
    ASSERT(M_SERIAL, !"should not be called");
}

void Deserializer_impl::release(const char *str)
{
    delete[] str;
}

void Deserializer_impl::clear_shared_objects()
{
    m_objects.clear();
}

// constructor
Deserialization_class::Deserialization_class(
    Class_id class_identifier,			// class id to be registered
    Factory_function* factory)			// factory for this class id
{
    m_class_identifier = class_identifier;
    m_factory = factory;
    m_factory_class = NULL;
}

// constructor
Deserialization_class::Deserialization_class(
    Class_id class_identifier,			// class id to be registered
    IDeserialization_factory* factory_class)	// factory class for this class id
{
    m_class_identifier = class_identifier;
    m_factory_class = factory_class;
    m_factory = NULL;
}

// constructor mainly used for searching
Deserialization_class::Deserialization_class(
    Class_id class_identifier)			// class id of searched class
: m_class_identifier(class_identifier)
, m_factory(0)
, m_factory_class(0)
{
}

// destructor
Deserialization_manager::~Deserialization_manager()
{
}

// Create an instance of a class implementing the interface
Deserialization_manager* Deserialization_manager::create()
{
    return new Deserialization_manager_impl();
}

// Release an instance of a class implementing the interface.
void Deserialization_manager::release(
    Deserialization_manager* mgr)
{
    delete mgr;
}

// destructor
Deserialization_manager_impl::~Deserialization_manager_impl()
{
}

// Any module may register a factory function for a given Class_id. When
// the deserializer finds this Class_id it will call the given factory
// function to create an appropriate object. This object then provides the
// applicable deserialize function.
void Deserialization_manager_impl::register_class(
    Class_id class_identifier,			// class id to be registered
    Factory_function* factory)			// factory for this class id
{
    ASSERT(M_DB, class_identifier || !"Registering serialization class with null ID");
    mi::base::Lock::Block block(&m_lock);
    Deserialization_class pattern(class_identifier);
    ASSERT(M_DB, m_classes.find(pattern) == m_classes.end()
             || !"Duplicate registration of serialization class");

    m_classes.insert(Deserialization_class(class_identifier, factory));
}

// Any module may register a factory function for a given Class_id. When the deserializer finds
// this Class_id it will call use the given factory class to create an appropriate object. This
// object then provides the applicable deserialize function.
void Deserialization_manager_impl::register_class(
    Class_id class_identifier,			// class id to be registered
    IDeserialization_factory* factory)		// factory class for this class id
{
    ASSERT(M_DB, class_identifier || !"Registering serialization class with null ID");
    mi::base::Lock::Block block(&m_lock);
    Deserialization_class pattern(class_identifier);
    ASSERT(M_DB, m_classes.find(pattern) == m_classes.end()
             || !"Duplicate registration of serialization class");

    m_classes.insert(Deserialization_class(class_identifier, factory));
}

// Call the appropriate factory function for the given class id and return
// it.
Serializable* Deserialization_manager_impl::construct(
    Class_id class_identifier)			// class id to be constructed
{
    mi::base::Lock::Block block(&m_lock);
    Deserialization_class pattern(class_identifier);
    Class_set::const_iterator it  = m_classes.find(pattern);
    ASSERT(M_DB, it != m_classes.end());
    if (it == m_classes.end())
        return NULL;
    Deserialization_class deserialization_class = *it;	// copy outside of the lock
    block.release();

    if (deserialization_class.m_factory != NULL)
        return deserialization_class.m_factory();
    return deserialization_class.m_factory_class->serializable_factory(class_identifier);
}

// Check if the class with given id is registered
bool Deserialization_manager_impl::is_registered(
    Class_id class_identifier)			// class id to be checked
{
    mi::base::Lock::Block block(&m_lock);
    Deserialization_class pattern(class_identifier);
    Class_set::const_iterator it  = m_classes.find(pattern);
    return it != m_classes.end();
}

bool Deserializer_impl::check_extension()
{
    return m_helper.read_extension_marker(this) == MARKER_FOUND;
}

void Deserializer_impl::set_error_handler(IDeserializer_error_handler<>* handler)
{
    m_error_handler = mi::base::make_handle_dup< IDeserializer_error_handler<> >(handler);
}

} // namespace DB

} // namespace MI

