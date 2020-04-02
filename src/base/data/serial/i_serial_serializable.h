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
/// \brief The Serializable definition.

#ifndef BASE_DATA_SERIAL_I_SERIAL_SERIALIZABLE_H
#define BASE_DATA_SERIAL_I_SERIAL_SERIALIZABLE_H

#include "i_serial_classid.h"
#include <base/lib/mem/i_mem_allocatable.h>

namespace MI
{
namespace SERIAL
{
// Forward declarations
class Serializer;
class Deserializer;

/// All serializable objects have to be derived from this.
class Serializable : public MI::MEM::Allocatable
{
  public:
    /// Constructor.
    Serializable() {}

    /// Virtual destructor for being able to delete a \c Serializable.
    virtual ~Serializable() {}

    /// This will return the class id of the given object. This is needed by the serializer when it
    /// wants to write the class id in the stream. It is public because the smart pointers need to
    /// access it.
    /// \return class id of the given object
    virtual Class_id get_class_id() const = 0;

    /// This will serialize the object to the given serializer including all sub elements pointed to
    /// but serialized together with this object. The function must return a pointer behind itself
    /// (e.g. this + 1) to be able to serialize arrays.
    /// \param serializer serialize to this serializer
    /// \return pointer behind itself
    virtual const Serializable* serialize(
	Serializer* serializer) const = 0;

    /// This will deserialize the object from the given deserializer including all sub elements
    /// pointed to but serialized together with this object. The function must return a pointer
    /// behind itself (e.g. this + 1) to be able to serialize arrays.
    /// \param deserializer deserialize from here
    /// \return pointer behind itself
    virtual Serializable* deserialize(
	Deserializer* deserializer) = 0;

    /// Optional function to dump the contents of the element for debugging purposes to stdout.
    virtual void dump() const {}

};

/// The deserialization factory function returns an initialized object of a certain type which is
/// derived from Serializable.
typedef Serializable* Factory_function();

/// A generic factory method for objects that allow trivial construction.
/// \return the newly created \c Serializable
template <class T> inline Serializable * factory() { return new T; }

/// This is a base class for factories to create serializables.
class IDeserialization_factory
{
public:
    /// The deserialization factory function returns an initialized object of a certain type which 
    /// is derived from Serializable.
    /// \param class_id Class identifier of the class to be created
    virtual Serializable* serializable_factory(
	Class_id class_id) = 0;
};

}
}
#endif
