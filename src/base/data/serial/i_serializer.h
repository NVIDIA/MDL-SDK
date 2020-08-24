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
///
/// This header file defines the framework for serialization and deserialization for elements in the
/// database. Each top level data element has to be derived from Serializable so that the database
/// is able to serialize/deserialize it. Serialization is always done to a Serializer class which
/// provides functions for streaming out primitive data types. When deserializing the Deserializer
/// provides the complementary functionality. The framework is designed to ensure that each single
/// serialization call does very few work itself to keep the latency small. Instead of directly
/// serializing a huge array of objects a serialization function may instead instruct the Serializer
/// to do the work after the function returned. The Serializer is then able to do the work in small
/// chunks (e.g. per object) thus keeping the latency small.
/// Notes: The framework currently uses many virtual functions. E.g. each write and read function of
/// the Serializer is virtual. This is done to hide the Serialization implementation as much as
/// possible. It might be changed without impacts on other modules using the interface, if it
/// becomes a (performance) problem.
///
/// Below a pattern is given which can be used to implement a serializable class. <CLASS NAME> and
/// <ASSIGNED CLASS ID> must be replaced by the chosen values. "..." means that code has to be added
/// here.
///
///  // The class <CLASS NAME> is a new type for top level elements
///  class <CLASS NAME> : public DB::Serializable
///  {
///    private:
///      // Member variables belonging to the top level element
///      ...
///
///    public:
///      // Member functions
///       ...
///
///      //
///      // The code below is needed for serialization/deserialization.
///      //
///
///      // The class id which must be globally unique for all serializable
///      // classes
///      static const Class_id id = <ASSIGNED CLASS ID>;
///
///      // A factory function which returns a pointer to an object of this
///      // class.
///      static Serializable* factory()
///      {
///          return new <CLASS NAME>;
///      }
///
///    protected:
///      // The serializer function using a serializer to write out the members.
///      Serializable* serialize(Serializer* serializer)
///      {
///          // Serialize the data here
///          serializer->write(...);
///          ...
///
///          // Must always return a pointer behind the object
///          return this + 1;
///      }
///
///      // The deserializer function using a deserializer to fill the members.
///      Serializable* deserialize(Deserializer* deserializer)
///      {
///          // Deserialize the data here
///          deserializer->read(&...);
///          ...
///
///           // Must always return a pointer behind the object
///          return this + 1;
///      }
///
///      // Return the unique id of the class
///      Class_id get_class_id()
///      {
///          return id;
///      }
///  };

#ifndef BASE_DATA_SERIAL_I_SERIALIZER_H
#define BASE_DATA_SERIAL_I_SERIALIZER_H

#include <mi/math/vector.h>
#include <mi/math/matrix.h>
#include <base/lib/mem/i_mem_allocatable.h>
#include <base/lib/cont/i_cont_array.h>
#include <cstddef>
#include <base/data/db/i_db_transaction_id.h>
#include <base/data/db/i_db_tag.h>

#include <base/system/stlext/i_stlext_safe_cast.h>
#include <map>
#include <set>
#include <list>
#include <string>
#include <vector>
#include <utility>

#include <mi/base/interface_implement.h>
#include <mi/base/iinterface.h>

namespace mi {
namespace math {
    class Color;
} // namespace math
} // namespace mi

#include "i_serial_serializable.h"

namespace MI {

namespace CONT { class Bitvector; class Dictionary; }

namespace SERIAL {

class Serializer;
class Deserializer;

/// Serialization function for arbitrary data. This is useful e.g. for long arrays of triangles
/// which should not have an embedded vtable pointer. This function should cast the <source> to a
/// class/struct of the expected type and use the given Serializer to stream out the data. It has
/// to assume that the given class/struct is part of an array and return a pointer to the next
/// element of the array.
typedef void* Serialization_function(
    void* source,                                       /// read data from here
    Serializer* serializer);                            /// write data here

/// Deserialization function for arbitrary data. This is useful e.g. for long arrays of triangles
/// which should not have an embedded vtable pointer. This function should cast the <destination> to
/// a class/struct of the expected type and use the given Deserializer to stream in the data. It
/// has to assume that the given class/struct is part of an array and return a pointer to the next
/// element of the array.
typedef void* Deserialization_function(
    void* destination,                                  /// write to this destination
    Deserializer* deserializer);                        /// read data from here


/// The Serializer will abstract from the concrete serialization target and will free the
/// Serializable classes from having to write out class id etc.
class Serializer
{
public:
    /// Override to return true if the data is intended to be sent to a remote cluster. Some
    /// db elements might need to serialize differently when the deserialization will take place
    /// in a remote cluster where for instance the file system will be different.
    virtual bool is_remote() = 0;

    /// Directly serialize the given Serializable. This will write the class id into the
    /// serialization stream, automatically. On the deserialization side the class id is taken to
    /// construct an object of the given type. After that the deserialization function of the
    /// constructed object will be used to read the member variables.
    ///
    /// \param serializable   serialize this serializable
    /// \param shared         serialize same pointers only once?
    virtual void serialize(const Serializable* serializable, bool shared = false) = 0;

    /// Write out various value types
    ///
    /// \param buffer         read data from here
    /// \param size           write this amount of data
    virtual void write(const char* buffer, size_t size) = 0;

    /// Write out various value types
    virtual void write(bool value) = 0;
    virtual void write(Uint8 value) = 0;
    virtual void write(Uint16 value) = 0;
    virtual void write(Uint32 value) = 0;
    virtual void write(Uint64 value) = 0;
    virtual void write(Sint8 value) = 0;
    virtual void write(Sint16 value) = 0;
    virtual void write(Sint32 value) = 0;
    virtual void write(Sint64 value) = 0;
    virtual void write(float value) = 0;
    virtual void write(double value) = 0;

    /// Write out arrays of various value types
    virtual void write(const bool* values, Size count) = 0;
    virtual void write(const Uint8* values, Size count) = 0;
    virtual void write(const Uint16* values, Size count) = 0;
    virtual void write(const Uint32* values, Size count) = 0;
    virtual void write(const Uint64* values, Size count) = 0;
    virtual void write(const Sint8* values, Size count) = 0;
    virtual void write(const Sint16* values, Size count) = 0;
    virtual void write(const Sint32* values, Size count) = 0;
    virtual void write(const Sint64* values, Size count) = 0;
    virtual void write(const float* values, Size count) = 0;
    virtual void write(const double* values, Size count) = 0;

    /// Write out back more complex types (typically implemented by using the previous types).
    virtual void write(const DB::Tag& value) = 0;
    virtual void write(const char* value) = 0;
    virtual void write(const std::string& value) = 0;
    virtual void write(const mi::base::Uuid& value) = 0;
    virtual void write(const mi::math::Color& value) = 0;
    virtual void write(const CONT::Bitvector& value) = 0;
    virtual void write(const CONT::Dictionary& value) = 0;

    /// Write out a size_t. This function is special because a size_t has a different size on a 32
    /// bit machine than on a 64 bit machine. The policy is as follows: A size_t will always be
    /// serialized as a Uint64. On a 32 bit machines this can lead to overflow if actually more
    /// than 32 bits are used. In that case the RealityServer terminates itself with a fatal.
    /// This has the consequence that pure 64 bit machine clusters can use the complete available
    /// address space. Still 32 bit and 64 bit machines can cooperate, but only as long as the
    /// used address space is restricted so that 32 bit machines can actually handle it.
    /// Otherwise the user gets a clear notification what went wrong.
    virtual void write_size_t(size_t value) = 0;

    /// Type-safe wrapper for the tag type.
    template <typename T> void write(const DB::Typed_tag<T>& value);

    /// These functions write out containers. Note that they assume that T is a Serializable object
    /// or a primitive type. For arrays of T*, T must be a Serializable. The class id of each
    /// element is serialized. This is needed if the elements are instances of subclasses of T.
    template <typename T> void write(const CONT::Array<T>& array);
    template <typename T> void write(const CONT::Array<T*>& array);
    template <typename T, typename A1, typename A2>
    void write(const std::vector< std::vector<T, A1>, A2>& array);

    template <typename T1, typename T2> void write(const std::pair<T1, T2>& pair);
    template <typename T, typename SWO> void write(const std::set<T, SWO>& set);

    /// Write a serializable object to the stream.
    virtual void write(const Serializable& object) = 0;

    /// Give a hint to the serializer that the given number of bytes
    /// are written to the serializer soon.
    virtual void reserve(size_t size) = 0;

    /// Flushes the so-far serialized data.
    ///
    /// The meaning of \em flushing depends on the context in which the serializer is used. If
    /// flushing is not supported in some context, nothing happens. If flushing is supported in some
    /// context, it typically means to process the already serialized data in the same way as the
    /// entire data at the end of the serialization would have been processed. For example, large
    /// buffers whose content is produced slowly over time can be subdivided into smaller chunks
    /// which can then be processed earlier than the entire buffer.
    virtual void flush() = 0;

    /// Add extension marker to this serialized object.
    ///
    /// Extension markers are used to extend serialized objects. They act like optional
    /// data that the deserializer can choose to deserialize or not (skipping to the
    /// end marker).
    ///
    /// Serializable objects should set this marker and the extension markers will be
    /// explicitly read during deserialization.
    virtual void start_extension() = 0;
};

enum Marker_status
{
    MARKER_FOUND = 0,
    MARKER_NOT_FOUND = -1,
    MARKER_BAD_CHECKSUM = -2,
};

template <typename T = Serializable>
class IDeserializer_error_handler: public mi::base::Interface_implement<mi::base::IInterface>
{
public:
    /// Called when deserialization of an object failed
    ///
    /// \param marker_status    Type of error encountered.
    /// \param serializable     Pointer to serializable object that failed deserializing.
    ///                         Only its class id is known to be defined.
    virtual void handle(Marker_status status, const T* serializable) = 0;
};

/// The Deserializer will abstract from the concrete deserialization source.
class Deserializer
{
public:
    /// Override to return true if the data is received from a remote cluster. Some
    /// db elements might need to deserialize differently when the serialization took place
    /// in a remote cluster where for instance the file system will be different.
    virtual bool is_remote() = 0;

    /// Directly deserialize an object. This will assume that the next data to be read is a class
    /// id. The class id will be used to construct the object using a factory function registered
    /// with the deserialization manager. After that the deserialization function of that object
    /// will be called which may deserialize the member variables of the class.
    ///
    /// \param shared    were same pointers serialized only once?
    virtual Serializable* deserialize(bool shared = false) = 0;

    /// Read back various value types
    ///
    /// \param buffer   destination for writing data
    /// \param size     number of bytes to read
    virtual void read(char* buffer, size_t size) = 0;

    /// Read back various value types
    virtual void read(bool* value_pointer) = 0;
    virtual void read(Uint8* value_pointer) = 0;
    virtual void read(Uint16* value_pointer) = 0;
    virtual void read(Uint32* value_pointer) = 0;
    virtual void read(Uint64* value_pointer) = 0;
    virtual void read(Sint8* value_pointer) = 0;
    virtual void read(Sint16* value_pointer) = 0;
    virtual void read(Sint32* value_pointer) = 0;
    virtual void read(Sint64* value_pointer) = 0;
    virtual void read(float* value_pointer) = 0;
    virtual void read(double* value_pointer) = 0;

    /// Read an array of various value types
    virtual void read(bool* value_pointer, Size count) = 0;
    virtual void read(Uint8* value_pointer, Size count) = 0;
    virtual void read(Uint16* value_pointer, Size count) = 0;
    virtual void read(Uint32* value_pointer, Size count) = 0;
    virtual void read(Uint64* value_pointer, Size count) = 0;
    virtual void read(Sint8* value_pointer, Size count) = 0;
    virtual void read(Sint16* value_pointer, Size count) = 0;
    virtual void read(Sint32* value_pointer, Size count) = 0;
    virtual void read(Sint64* value_pointer, Size count) = 0;
    virtual void read(float* value_pointer, Size count) = 0;
    virtual void read(double* value_pointer, Size count) = 0;

    /// Read back more complex types (typically implemented by using the previous types).
    virtual void read(DB::Tag* value_pointer) = 0;
    virtual void read(char** value_pointer) = 0; ///< Use release() to free the memory.
    virtual void read(std::string* value_pointer) = 0;
    virtual void read( mi::base::Uuid* value_pointer) = 0;
    virtual void read(mi::math::Color* value_pointer) = 0;
    virtual void read(CONT::Bitvector* value_type) = 0;
    virtual void read(CONT::Dictionary* value_pointer) = 0;

    /// Read in a size_t. This function is special because a size_t has a different size on a 32
    /// bit machine than on a 64 bit machine. The policy is as follows: A size_t will always be
    /// serialized as a Uint64. On a 32 bit machines this can lead to overflow if actually more
    /// than 32 bits are used. In that case the RealityServer terminates itself with a fatal.
    /// This has the consequence that pure 64 bit machine clusters can use the complete available
    /// address space. Still 32 bit and 64 bit machines can cooperate, but only as long as the
    /// used address space is restricted so that 32 bit machines can actually handle it.
    /// Otherwise the user gets a clear notification what went wrong.
    virtual void read_size_t(size_t* value) = 0;

    /// This method should be called for releasing of deserialized strings.
    virtual void release(const char *str) = 0;

    /// Type-safe wrapper for the tag type.
    template <typename T> void read(DB::Typed_tag<T>* value_pointer);

    /// These functions read in containers. Note that they assume that T is a Serializable object or
    /// a primitive, for T* only Serializable.
    template <typename T> void read(CONT::Array<T>* array);
    template <typename T> void read(CONT::Array<T*>* array);
    template <typename T, typename A1, typename A2>
    void read(std::vector< std::vector<T, A1>, A2>* array);
    template <typename T, typename SWO> void read(std::set<T, SWO>* set);
    template <typename T1, typename T2> void read(std::pair<T1, T2>* pair);

    /// Read back a serializable object from the stream.
    virtual void read(Serializable* object) = 0;

    /// Checks if extension marker exists.
    ///
    /// This is used during deserialization to check if an extension marker comes next.
    /// When an extension marker is found, the deserializer can read more values.
    /// If an extension marker is not found, it MUST be an end marker.
    ///
    /// \return True if bytes read correspond to end marker. False otherwise.
    ///
    virtual bool check_extension() = 0;

    /// Check if we can keep on reading from this deserializer.
    virtual bool is_valid() const = 0;

    /// Install handler for deserialization error.
    virtual void set_error_handler(IDeserializer_error_handler<>* handler) = 0;
};

/// Write out various value types
void write(Serializer* serial, bool value);
void write(Serializer* serial, Uint8 value);
void write(Serializer* serial, Uint16 value);
void write(Serializer* serial, Uint32 value);
void write(Serializer* serial, Uint64 value);
void write(Serializer* serial, Sint8 value);
void write(Serializer* serial, Sint16 value);
void write(Serializer* serial, Sint32 value);
void write(Serializer* serial, Sint64 value);
void write(Serializer* serial, float value);
void write(Serializer* serial, double value);

/// These functions offer a default implementation build atop the above functions.
void write(Serializer* serial, const DB::Tag& value);
template <class T> void write(Serializer* serial, const DB::Typed_tag<T>& value);
void write(Serializer* serial, const DB::Tag_version& value);
template <typename T, Size R, Size C> void write(Serializer* serial,const mi::math::Matrix<T,R,C>&);
void write(Serializer* serial, const char* value);
void write(Serializer* serial, const std::string& value);
void write(Serializer* serial, const mi::base::Uuid& value);
void write(Serializer* serial, const mi::math::Color& value);
template <typename T, Size DIM> void write(Serializer* serial,const mi::math::Vector<T,DIM>& value);
void write(Serializer* serial, const CONT::Bitvector& value);
void write(Serializer* serial, const CONT::Dictionary& value);
void write(Serializer* serial, const DB::Transaction_id& value);
void write(Serializer* serial, const Serializable& object);

/// Read back various value types
void read(Deserializer* deser, bool* value_pointer);
void read(Deserializer* deser, Uint8* value_pointer);
void read(Deserializer* deser, Uint16* value_pointer);
void read(Deserializer* deser, Uint32* value_pointer);
void read(Deserializer* deser, Uint64* value_pointer);
void read(Deserializer* deser, Sint8* value_pointer);
void read(Deserializer* deser, Sint16* value_pointer);
void read(Deserializer* deser, Sint32* value_pointer);
void read(Deserializer* deser, Sint64* value_pointer);
void read(Deserializer* deser, float* value_pointer);
void read(Deserializer* deser, double* value_pointer);

/// These functions offer a default implementation build atop the above functions.
void read(Deserializer* deser, DB::Tag* value_pointer);
template <class T> void read(Deserializer* deser, DB::Typed_tag<T>* value_pointer);
void read(Deserializer* deser, DB::Tag_version* value_pointer);
template <typename T, Size R, Size C> void read(Deserializer* deser, mi::math::Matrix<T,R,C>*);
void read(Deserializer* deser, char** value_pointer);
void read(Deserializer* deser, std::string* value_pointer);
void read(Deserializer* deser, const mi::base::Uuid* value_pointer);
void read(Deserializer* deser, mi::math::Color* value_pointer);
template <typename T, Size DIM> void read(Deserializer* deser, mi::math::Vector<T,DIM>* value_type);
void read(Deserializer* deser, CONT::Bitvector* value_type);
void read(Deserializer* deser, CONT::Dictionary* value_pointer);
void read(Deserializer* deser, DB::Transaction_id* value_pointer);
void read(Deserializer* deser, Serializable* object);

/// A small helper function for de-serializing ranges of values. The function iterates over the
/// range "[begin, end)" and reads a value into every position using the given serializer. As an
/// example, consider the following code snippet:
///
///     char    alphabet[] = "abcdefghijklmnopqrstuvwxyz";
///     char *  begin = &alphabet[0];
///     char *  end   = &alphabet[26];
///
///     write_range(serializer, begin, end);
///     [...]
///     memset(alphabet, 0, 26);
///     read_range(deserializer, begin, end);
///
/// \param deserializer   de-serializer to read from
/// \param begin          first spot of the range
/// \param end            one past the last spot of the range
template <class Iterator>
inline void read_range(Deserializer& deserializer, Iterator begin, Iterator end);

/// A small helper function for de-serializing arrays of values.
///
/// \param deserializer   de-serializer to read from
/// \param arr            the array to deserialize
template <typename T, size_t N>
inline void read_range(Deserializer& deserializer, T (&arr)[N]);

/// A small helper function for serializing ranges of values. write_range() can serialize ranges of
/// any serializable type from any given container.
///
/// \param serializer     serializer to write to
/// \param begin          first spot of the range
/// \param end            one past the last spot of the range
template <class Iterator>
inline void write_range(Serializer& serializer, Iterator begin, Iterator end);

/// A small helper function for serializing ranges of values. write_range() can serialize arrays.
///
/// \param serializer     serializer to write to
/// \param arr            the array to serialize
template <typename T, size_t N>
inline void write_range(Serializer& serializer, const T (&arr)[N]);

/// Serialize a vector.
template <typename T>
void write(Serializer* serializer, const std::vector<T>& array);
template <typename T,typename A>
void write(Serializer* serializer, const std::vector<T,A>& array);
template <typename T>
void write(Serializer* serializer, const std::vector<T*>& array);
template <typename T,typename A>
void write(Serializer* serializer, const std::vector<T*,A>& array);

/// Deserialize a vector.
template <typename T>
void read(Deserializer* deserializer, std::vector<T>* array);
inline void read(Deserializer* deserializer, const std::vector<bool>* array);
template <typename T,typename A>
void read(Deserializer* deserializer, std::vector<T,A>* array);
template <typename T>
void read(Deserializer* deserializer, std::vector<T*>* array);
template <typename T,typename A>
void read(Deserializer* deserializer, std::vector<T*,A>* array);

/// Serialize a list.
template <typename T>
void write(Serializer* serializer, const std::list<T>& list);

/// Deserialize a list.
template <typename T>
void read(Deserializer* deserializer, std::list<T>* list);

/// Serialize a pair.
template <typename T, typename U>
void write(Serializer* serializer, const std::pair<T, U>& pair);

/// Deserialize a pair.
template <typename T, typename U>
void read(Deserializer* deserializer, std::pair<T, U>* pair);

/// Serialize a set.
template <typename T, typename SWO>
void write(Serializer* serializer, const std::set<T,SWO>& set);

/// Deserialize a set.
template <typename T, typename SWO>
void read(Deserializer* deserializer, std::set<T,SWO>* set);

/// Serialize a map.
template <typename T, typename U, typename SWO>
void write(Serializer* serializer, const std::map<T,U,SWO>& map);

/// Deserialize a map.
template <typename T, typename U, typename SWO>
void read(Deserializer* deserializer, std::map<T,U,SWO>* map);


template<class K, class V, class C, class A>
void write(Serializer* serializer, const std::multimap<K,V,C,A>&);

template<class K, class V, class C, class A>
void read(Deserializer* deserializer, std::multimap<K,V,C,A>*);

template <typename Enum_type>
void write_enum(Serializer* serializer, Enum_type enum_value );

template <typename Enum_type>
void read_enum(Deserializer* deserializer, Enum_type* enum_value );

} /// namespace SERIAL

} /// namespace MI

#include "serial_inline.h"

#endif // BASE_DATA_SERIAL_I_SERIALIZER_H

