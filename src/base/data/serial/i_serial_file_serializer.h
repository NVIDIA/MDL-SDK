/******************************************************************************
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
 ******************************************************************************/
/// \file
/// \brief file serializer
/// Provide a serializer/deserializer pair which serializes to and
/// deserializes from file. This can be used for different purposes
/// including implementing importer/exporter. Based on
/// base/data/serial/i_serial_buffer_serializer.h. The following
/// example explains how to use it:
/// \code
///     Some_serializable serializable(...);
///     File_serializer serializer;
///     serializable.serialize(serializer);
///     File_deserializer deserializer;
///     Some_serializable deserialized;
///     deserializer.deserialize(&deserialized, serializer->get_buffer(),
///         serializer->get_buffer_size());
/// \endcode
///
/// TODO: Current implementation do not care about endian.
/// If endian is a problem, please implement each read/write

#ifndef BASE_DATA_SERIAL_I_SERIAL_FILE_SERIALIZER_H
#define BASE_DATA_SERIAL_I_SERIAL_FILE_SERIALIZER_H

#include "serial.h"

#include <base/data/db/i_db_tag.h>

namespace MI {
  namespace DISK {
    class IFile;
  }
}

namespace MI
{
namespace SERIAL
{
/// File serializer. serialize (Serializable) objects to a file.
///
/// A common use case:
/// \code
/// Setializable_type serializable_object;
/// DISK::IFile file;
/// file.open("serialized_dest.data", DISK::IFile::M_READWRITE);
/// File_serializer fserializer;
/// fserializer.set_output_file(&file);
/// fserializer.serialize(&serializable_object);
/// \endcode
///
/// \see test_file_serializer.cpp
class File_serializer : public Serializer_impl
{
public:
    /// Constructor
    File_serializer();

    /// Destructor
    virtual ~File_serializer();

    /// set output file
    /// \param p_file file object. This should be opened and should be
    /// good status. (is_opened() && !eof())
    void set_output_file(
        DISK::IFile * p_file
        );

    /// peek current file
    /// \return pointer to the current file object.
    DISK::IFile * peek_output_file() const;

    /// Is this serializer valid?
    /// \return the last serialization status. true when there was no
    /// error.
    bool is_valid() const;

    using Serializer_impl::write;

protected:

    /// Writer specialization implementation.
    ///
    /// \param buffer read data from here
    /// \param size   write this amount of data
    void write_impl(const char* buffer, size_t size);

    /// set serializer status.
    /// Usually this should not be used. Only use when you know the
    /// serializer status (e.g., use from derivecd class.)
    /// \param  the last serialization status. true when there was no
    /// error.
    void set_valid(
        bool is_valid
        );

    /// check the file validity. Is it write ready?
    /// \param p_file file object to check.
    /// \return true when the file object is write ready.
    bool is_file_valid(
        DISK::IFile * p_file
        ) const ;

private:
    /// a reference to the file where to write data to
    DISK::IFile * m_p_file;
    /// the status of last write operation.
    bool m_is_valid;
};

//======================================================================

/// File deserializer. Deserialize (Serializable) objects from a file.
///
/// A common use case:
/// \code
/// DISK::File file;
/// file.open("serialized_dest.data", DISK::IFile::M_READWRITE); // already serialized
/// File_deserializer fdeserializer(DATA::mod_data->get_deserialization_manager());
/// fdeserializer.set_input_file(&file);
/// Serializable * p_new_obj = fdeserializer.deserialize_file();
/// CHECK(p_new_obj->get_class_id() == Serializable_type's class id);
/// Serializable_type dst1 = *(dynamic _ cast< Serializable_type *>(p_new_obj)); // copy
/// delete p_new_obj; // or you can of course keep it as a Serializable obejct.
/// \endcode
///
/// You may find oddness at the line,
/// \code
/// Serializable * p_new_obj = fdeserializer.deserialize_file();
/// \endcode
/// Why this is not
/// \code
/// Serializable * p_new_obj = fdeserializer.deserialize();
/// \endcode
/// ?
///
/// Actually this still work, but I do not recommend that since this
/// one has less safety check. I could not write
/// File_deserializer::deserialize() method, because base class has
/// Deserializer_impl::deserialize(bool shard = false). This default
/// argument makes File_deserializer::deserialize() ambiguous and
/// casts a compile error.  Also it is a bad idea to overload this by
/// File_deserializer::deserialize(T *), since any pointer can be
/// implicitly converted to bool.
///
/// I think File_deserializer has not yet proven to useful except
/// importer/exporter in neuray. I might regret this, but keep this
/// pitfall here not to impact the current neuray implementation.
///
/// \see test_file_serializer.cpp
class File_deserializer : public Deserializer_impl
{
public:
    using Deserializer_impl::deserialize;

    /// This deserializer's read() and write() functions work without
    /// having a Deserialization_manager, but the deserialize() method
    /// does not because it needs to look up the class's constructor
    /// function. The global manager instance can be obtained from
    /// Mod_data::get_deserialization_manager().
    ///
    /// \param p_manager the set of registered classes
    explicit File_deserializer(
        Deserialization_manager* p_manager = 0); // the set of registered classes

    /// Destructor
    virtual ~File_deserializer();

    /// set input file
    /// \param p_file file object. This should be opened and should be
    /// good status. (is_opened() && !eof())
    void set_input_file(
        DISK::IFile * p_file
        );

    /// peek current file
    /// \return pointer to the current file object.
    DISK::IFile * peek_input_file() const;

    /// Is this deserializer valid?
    /// \return the last deserialization status. true when there was
    /// no error.
    bool is_valid() const;

    /// Deserialize from a file
    /// \return newed deserialized object.
    ///
    /// Note: Cannot write deserialize() which seems more suitable.
    /// Because base class Deserializer has a method deserialize(bool
    /// shared = false). This makes ambiguous. Also any pointer
    /// argument method has a similar problem since any pointer can be
    /// implicitly converted to a bool.
    ///
    /// One problem is deserialize() still works but without check.
    Serializable* deserialize_file();

    using Deserializer_impl::read;

protected:

    /// Reader specialization implementation.
    /// \param buffer destination for writing data
    /// \param size   number of bytes to read
    virtual void read_impl(
        char* buffer,           // destination for writing data
        size_t size);           // number of bytes to read


    /// set serializer status.
    /// Usually this should not be used. Only use when you know the
    /// serializer status (e.g., use from derivecd class.)
    /// \param  the last deserialization status. true when there was no
    /// error.
    void set_valid(
        bool is_valid
        );

    /// check the file validity. Is it write ready?
    /// \param p_file file object to check.
    /// \return true when the file object is write ready.
    bool is_file_valid(
        DISK::IFile * p_file
        ) const ;

private:
    /// a reference to the file from which we read the data.
    DISK::IFile * m_p_file;
    /// the deserializer status.
    bool m_is_valid;
};

} // namespace SERIAL
} // namespace MI
#endif // #ifndef BASE_DATA_SERIAL_I_SERIAL_FILE_SERIALIZER_H
