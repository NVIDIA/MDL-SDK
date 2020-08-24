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
/// \brief Declarations for the serialization functionality

#ifndef BASE_DATA_SERIAL_SERIAL_H
#define BASE_DATA_SERIAL_SERIAL_H

#include <mi/base/handle.h>

#include "i_serializer.h"

#include "serial_marker_helpers.h"

namespace MI {

namespace SERIAL {

/// The Serializer_impl will abstract from the concrete serialization target and will free the
/// Serializable classes from having to write out class id etc.
class Serializer_impl : public Serializer, public MI::MEM::Allocatable
{
public:
    Serializer_impl() : m_id_counter(1), m_helper() { }

    virtual ~Serializer_impl() { };

    bool is_remote() { return false; }

    virtual void serialize(const Serializable* serializable, bool shared = false);

    using Serializer::write;

    // Default implementation for basic types.
    virtual void write(bool value) { do_write(&value, 1); }
    virtual void write(Uint8 value) { do_write(&value, 1); }
    virtual void write(Uint16 value) { do_write(&value, 1); }
    virtual void write(Uint32 value) { do_write(&value, 1); }
    virtual void write(Uint64 value) { do_write(&value, 1); }
    virtual void write(Sint8 value) { do_write(&value, 1); }
    virtual void write(Sint16 value) { do_write(&value, 1); }
    virtual void write(Sint32 value) { do_write(&value, 1); }
    virtual void write(Sint64 value) { do_write(&value, 1); }
    virtual void write(float value) { do_write(&value, 1); }
    virtual void write(double value) { do_write(&value, 1); }


    /// Default implementations for serializing an array of value types. These
    /// default implementations can/should be overridden by the derived classes.
    virtual void write(const char* values, size_t count) { do_write(values, count); }
    virtual void write(const bool* values, Size count) { do_write(values, count); }
    virtual void write(const Uint8* values, Size count) { do_write(values, count); }
    virtual void write(const Uint16* values, Size count) { do_write(values, count); }
    virtual void write(const Uint32* values, Size count) { do_write(values, count); }
    virtual void write(const Uint64* values, Size count) { do_write(values, count); }
    virtual void write(const Sint8* values, Size count) { do_write(values, count); }
    virtual void write(const Sint16* values, Size count) { do_write(values, count); }
    virtual void write(const Sint32* values, Size count) { do_write(values, count); }
    virtual void write(const Sint64* values, Size count) { do_write(values, count); }
    virtual void write(const float* values, Size count) { do_write(values, count); }
    virtual void write(const double* values, Size count) { do_write(values, count); }
    void write(const DB::Tag& value);
    void write(const char* value);
    void write(const std::string& value);
    void write(const mi::base::Uuid& value);
    void write(const mi::math::Color& value);
    void write(const CONT::Bitvector& value);
    void write(const CONT::Dictionary& value);

    void write_size_t(size_t value);

    void write(const Serializable& object) { object.serialize(this); }

    void reserve(size_t size) { };

    void flush() { }

    void start_extension();

    static void write_direct(Serializer* serializer, const char* buffer, size_t size);

protected:

    /// This is meant to be called from the database or similar modules only! It will flush out all
    /// shared objects collected so far.
    void clear_shared_objects();

    /// All specializations should implement this. This is where the write actually happens.
    virtual void write_impl(const char* value, size_t count) = 0;

private:
    typedef std::map<Uint64,Uint64> Id_map;
    Id_map m_shared_id_map;
    Uint64 m_id_counter;
    Serializer_marker_helper m_helper;

    /// Template function for writing out an array of basic types and computing the
    /// checksum of the data.
    template <class T>
    void do_write(T* values, Size count)
    {
        const char *buffer = reinterpret_cast<const char *>(values);
        size_t size = count * sizeof(T);
        write_impl(buffer, size);
        m_helper.update_checksum(buffer, size);
    }
};

class Deserialization_manager;

/// The Deserializer_impl will abstract from the concrete deserialization source.
///
/// In case of deserialization error (end marker not found, etc) it will call the
/// error handler. The default error handler logs a fatal error.
class Deserializer_impl : public Deserializer, public MI::MEM::Allocatable
{
public:
    Deserializer_impl(Deserialization_manager* deserialization_manager);

    virtual ~Deserializer_impl() { };

    virtual bool is_remote() { return false; }

    virtual Serializable* deserialize(bool shared = false);

    using Deserializer::read;

    /// Default implementations for deserializing an array of value types. These
    /// default implementations can/should be overridden by the derived classes.
    virtual void read(bool* value_pointer) { do_read(value_pointer, 1); }
    virtual void read(Uint8* value_pointer) { do_read(value_pointer, 1); }
    virtual void read(Uint16* value_pointer) { do_read(value_pointer, 1); }
    virtual void read(Uint32* value_pointer) { do_read(value_pointer, 1); }
    virtual void read(Uint64* value_pointer) { do_read(value_pointer, 1); }
    virtual void read(Sint8* value_pointer) { do_read(value_pointer, 1); }
    virtual void read(Sint16* value_pointer) { do_read(value_pointer, 1); }
    virtual void read(Sint32* value_pointer) { do_read(value_pointer, 1); }
    virtual void read(Sint64* value_pointer) { do_read(value_pointer, 1); }
    virtual void read(float* value_pointer) { do_read(value_pointer, 1); }
    virtual void read(double* value_pointer) { do_read(value_pointer, 1); }

    /// Default implementations for deserializing an array of value types. These
    /// default implementations can/should be overridden by the derived classes.
    virtual void read(char* value_pointer, size_t count) { do_read(value_pointer, count); }
    virtual void read(bool* value_pointer, Size count) { do_read(value_pointer, count); }
    virtual void read(Uint8* value_pointer, Size count) { do_read(value_pointer, count); }
    virtual void read(Uint16* value_pointer, Size count) { do_read(value_pointer, count); }
    virtual void read(Uint32* value_pointer, Size count) { do_read(value_pointer, count); }
    virtual void read(Uint64* value_pointer, Size count) { do_read(value_pointer, count); }
    virtual void read(Sint8* value_pointer, Size count) { do_read(value_pointer, count); }
    virtual void read(Sint16* value_pointer, Size count) { do_read(value_pointer, count); }
    virtual void read(Sint32* value_pointer, Size count) { do_read(value_pointer, count); }
    virtual void read(Sint64* value_pointer, Size count) { do_read(value_pointer, count); }
    virtual void read(float* value_pointer, Size count) { do_read(value_pointer, count); }
    virtual void read(double* value_pointer, Size count) { do_read(value_pointer, count); }

    void read(DB::Tag* value_pointer);
    void read(char** value_pointer);
    void read(std::string* value_pointer);
    void read(mi::base::Uuid* value_pointer);
    void read(mi::math::Color* value_pointer);
    void read(CONT::Bitvector* value_type);
    void read(CONT::Dictionary* value_pointer);

    void read_size_t(size_t* value);

    void release(const char *str);

    void read(Serializable* object) { object->deserialize(this); }

    bool check_extension();

    void set_error_handler(IDeserializer_error_handler<>* handler);

protected:
    // This is meant to be called from the database or similar modules only! It will clear the map
    // of shared objects.
    void clear_shared_objects();

    /// Where the read actually happens. Must be implemented by all specializations.
    virtual void read_impl(char* buffer, size_t size) = 0;

    friend class Deserializer_marker_helper;

private:
    typedef std::map<Uint64, Serializable*> Reference_map;
    Reference_map m_objects;				// map of all known objects
    Deserialization_manager* m_deserialization_manager;	// for deserialization
    Deserializer_marker_helper m_helper;

    /// What to do in case of error.
    mi::base::Handle< IDeserializer_error_handler<> > m_error_handler;

    /// Template function for reading an array of basic types and computing the checksum
    /// of the data.
    template <class T>
    void do_read(T* value_pointer, Size count)
    {
        char *buffer = reinterpret_cast<char *>(value_pointer);
        size_t size = count * sizeof(T);
        read_impl(buffer, size);
        m_helper.update_checksum(buffer, size);
    }
};

// The deserialization manager is needed to provide an instance of a class identified by the class
// id found in the deserialization stream. This class will then provide the deserialization
// function.
class Deserialization_manager : public MI::MEM::Allocatable
{
public:
    // Create an instance of a class implementing the interface.
    static Deserialization_manager* create();
    // Release an instance of a class implementing the interface. Note that the passed-in \c mgr
    // cannot be used anymore after that call!
    static void release(
    Deserialization_manager* mgr);

    // destructor
    virtual ~Deserialization_manager();

    // Any module may register a factory function for a given Class_id. When the deserializer finds
    // this Class_id it will call the given factory function to create an appropriate object. This
    // object then provides the applicable deserialize function.
    virtual void register_class(
    Class_id class_identifier,			// class id to be registered
    Factory_function* factory) = 0;			// factory for this class id

    // Trivial wrapper that allows convenient registration of classes that derive from Element<T>.
    template <class T> void register_class()	{ register_class(T::id, &factory<T>); }

    // Any module may register a factory function for a given Class_id. When the deserializer finds
    // this Class_id it will call use the given factory class to create an appropriate object. This
    // object then provides the applicable deserialize function.
    virtual void register_class(
    Class_id class_identifier,			// class id to be registered
    IDeserialization_factory* factory) = 0;		// factory class for this class id

    // Call the appropriate factory function for the given class id and return it.
    virtual Serializable* construct(
    Class_id class_identifier) = 0;			// class id to be constructed

    // Check if an appropriate factory function for the given class id is registered.
    virtual bool is_registered(
    Class_id class_identifier) = 0;			// class id to be checked
};

} // namespace SERIAL

} // namespace MI

#endif // BASE_DATA_SERIAL_SERIAL_H
