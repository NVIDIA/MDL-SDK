/******************************************************************************
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDL_COMPILERCORE_ZIP_UTILS_H
#define MDL_COMPILERCORE_ZIP_UTILS_H 1

#include "compilercore_allocator.h"
#include <base/lib/libzip/zip.h>
#include <mi/mdl/mdl_entity_resolver.h>
#include <mi/mdl/mdl_streams.h>

namespace mi {
namespace mdl {

class File_handle;
class MDL_zip_container;
class MDL_zip_container_file;

enum Extra_attributes {
    MDLE_EXTRA_FIELD_ID_MD = 0x444d  // MD
};

struct MDL_zip_container_header
{
    char prefix[4];             ///< leader marker, i.e., MDR, MDLE, or ...
    uint16_t major_version_min; ///< minimum supported major version number.
    uint16_t minor_version_min; ///< minimum supported minor version number.
    uint16_t major_version_max; ///< maximum supported major version number.
    uint16_t minor_version_max; ///< maximum supported minor version number.
    unsigned char hash[16];     ///< maximum supported minor version number.

    /// Constructor.
    ///
    /// \param prefix           the header marker, i.e., MDR, MDLE, or ...
    /// \param prefix_size      number of prefix characters (max 4)
    /// \param major            the major version number, sets both min and max
    /// \param minor            the minor version number, sets both min and max
    explicit MDL_zip_container_header(
        char const *prefix,
        uint8_t    prefix_size,
        uint16_t   major,
        uint16_t   minor);

    /// Constructor.
    ///
    /// \param prefix           the header marker, i.e., MDR, MDLE, or ...
    /// \param prefix_size      number of prefix characters (max 4)
    /// \param major_min        the minimum supported major version number
    /// \param minor_min        the minimum supported minor version number
    /// \param major_max        the maximum supported major version number
    /// \param minor_max        the maximum supported minor version number
    explicit MDL_zip_container_header(
        char const *prefix,
        uint8_t    prefix_size,
        uint16_t   major_min,
        uint16_t   minor_min,
        uint16_t   major_max,
        uint16_t   minor_max);

    /// Copy constructor.
    MDL_zip_container_header(MDL_zip_container_header const &to_copy);
    MDL_zip_container_header &operator=(MDL_zip_container_header const &);
};


/// Helper class to implement a layered callback for libzip's source streams.
class Layered_zip_source {
public:
    /// Constructor.
    ///
    /// param base  the base source layer, takes ownership
    explicit Layered_zip_source(zip_source_t *base, const MDL_zip_container_header &header);

    /// Destructor.
    virtual ~Layered_zip_source();

    /// Open the layered stream
    ///
    /// param ze if this function fails, ze contains the lipzip error
    zip_source_t *open(zip_error_t &ze);

private:
    /// The source callback function, delegates all requests to the base layer but adjusts
    /// the offsets.
    static zip_int64_t callback(
        void             *env,
        void             *data,
        zip_uint64_t     len,
        zip_source_cmd_t cmd);

    /// the base layer
    zip_source_t *m_base;

    // the header that is written at the beginning of the file
    MDL_zip_container_header m_header;

    /// The error
    zip_error_t m_ze;
};


/// interface to create a read zip_source without seek support from an IMDL_resource_reader.
class Resource_zip_source
{
    friend class Allocator_builder;

public:
    virtual ~Resource_zip_source() {}

    /// Open the layered stream
    ///
    /// param ze if this function fails, ze contains the lipzip error
    zip_source_t *open(zip_error_t &ze);

private:
    /// The source callback function invoked by libzip
    static zip_int64_t callback(
        void             *env,
        void             *data,
        zip_uint64_t     len,
        zip_source_cmd_t cmd);

    explicit Resource_zip_source(
        IMDL_resource_reader *reader);

    // non copyable
    Resource_zip_source(Resource_zip_source const &) MDL_DELETED_FUNCTION;
    Resource_zip_source &operator=(Resource_zip_source const &) MDL_DELETED_FUNCTION;

    /// mdl resource
    mi::base::Handle<IMDL_resource_reader> m_reader;
    zip_error_t m_ze;
};

// --------------------------------------------------------------------------

/// Implementation of a resource reader from a file.
class MDL_zip_resource_reader : public Allocator_interface_implement<IMDL_resource_reader>
{
    typedef Allocator_interface_implement<IMDL_resource_reader> Base;
public:
    /// Read a memory block from the resource.
    ///
    /// \param ptr   Pointer to a block of memory with a size of size bytes
    /// \param size  Number of bytes to read
    ///
    /// \returns    The total number of bytes successfully read.
    Uint64 read(void *ptr, Uint64 size) MDL_FINAL;

    /// Get the current position.
    Uint64 tell() MDL_FINAL;

    /// Reposition stream position indicator.
    ///
    /// \param offset  Number of bytes to offset from origin
    /// \param origin  Position used as reference for the offset
    ///
    /// \return true on success
    bool seek(Sint64 offset, Position origin) MDL_FINAL;

    /// Get the UTF8 encoded name of the resource on which this reader operates.
    /// \returns    The name of the resource or NULL.
    char const *get_filename() const MDL_FINAL;

    /// Get the UTF8 encoded absolute MDL resource name on which this reader operates.
    /// \returns    The absolute MDL url of the resource or NULL.
    char const *get_mdl_url() const MDL_FINAL;

    /// Returns the associated hash of this resource.
    ///
    /// \param[out]  get the hash value (16 bytes)
    ///
    /// \return true if this resource has an associated hash value, false otherwise
    bool get_resource_hash(unsigned char hash[16]) MDL_FINAL;

    /// Constructor.
    ///
    /// \param alloc             the allocator
    /// \param f                 the file handle
    /// \param filename          the file name
    /// \param mdl_url           the MDL url
    explicit MDL_zip_resource_reader(
        IAllocator  *alloc,
        File_handle *f,
        char const  *filename,
        char const  *mdl_url);

private:
    /// Destructor
    ~MDL_zip_resource_reader() MDL_FINAL;

    // non copyable
    MDL_zip_resource_reader(MDL_zip_resource_reader const &) MDL_DELETED_FUNCTION;
    MDL_zip_resource_reader &operator=(MDL_zip_resource_reader const &) MDL_DELETED_FUNCTION;

    /// The File handle.
    File_handle *m_file;

    /// The filename.
    string m_file_name;

    /// The absolute MDL url.
    string m_mdl_url;
};

/// Error codes possible with some container operations.
enum MDL_zip_container_error_code
{
    EC_OK,                      ///< No error.
    EC_CONTAINER_NOT_EXIST,     ///< A container does not exists.
    EC_CONTAINER_OPEN_FAILED,   ///< A container could not be opened.
    EC_FILE_OPEN_FAILED,        ///< A file could not be opened.
    EC_INVALID_CONTAINER,       ///< A container is invalid.
    EC_NOT_FOUND,               ///< A file was not found (inside an archive).
    EC_IO_ERROR,                ///< Some I/O error occurred.
    EC_CRC_ERROR,               ///< A CRC error occurred.
    EC_INVALID_PASSWORD,        ///< Invalid password.
    EC_MEMORY_ALLOCATION,       ///< Memory allocation failed.
    EC_RENAME_ERROR,            ///< Rename operation failed.
    EC_INVALID_HEADER,          ///< The zip header does not start MDR, MDLE, or ...
    EC_INVALID_HEADER_VERSION,  ///< The zip header prefix is correct but the version is not.
    EC_PRE_RELEASE_VERSION,     ///< Container has a pre-release version number.
    EC_INTERNAL_ERROR,          ///< Internal archiver error.
};

/// Helper class for archives and MDLe.
class MDL_zip_container
{
    friend class Allocator_builder;

public:
    /// Close an MDL archive.
    void close();

    /// Get the number of files inside an archive. 
    int get_num_entries();

    /// Get the i'th file name inside an archive.
    char const *get_entry_name(int i);

    /// Check if the given file name exists in the archive.
    bool contains(char const *file_name) const;

    /// Check if the given file mask exists in the archive.
    bool contains_mask(char const *file_mask) const;

    /// Open a read_only file from an archive.
    MDL_zip_container_file *file_open(char const *name) const;

    /// Compute the MD5 hash for a file inside a, archive.
    ///
    /// \param[in]  name  the file name inside the archive
    /// \param[out] md5   the computed hash
    ///
    /// \return true on success, false if the file was not found inside the container
    bool compute_file_hash(char const *name, unsigned char md5[16]) const;

    /// Get the version number of an opened container.
    ///
    /// \param[out] major_version   Major version number found in the container header
    /// \param[out] minor_version   Minor version number found in the container header
    void get_version(uint16_t &major_version, uint16_t &minor_version) const
    {
        major_version = m_header.major_version_min; // min and max are equal for loaded files
        minor_version = m_header.minor_version_min; // min and max are equal for loaded files
    }

    /// Get the allocator.
    IAllocator *get_allocator() const { return m_alloc; }

    /// Get the zip archive name.
    char const *get_container_name() const { return m_path.c_str(); }

    /// Returns true if this container supports resource hashes.
    bool has_resource_hashes() const { return m_has_resource_hashes; }

protected:
    /// Constructor.
    explicit MDL_zip_container(
        IAllocator *alloc,
        char const *path,
        zip_t      *za,
        bool       supports_resource_hashes);

    /// Destructor
    virtual ~MDL_zip_container();

protected:
    /// Open a container file.
    ///
    /// \param[in]  alloc                   the allocator
    /// \param[in]  path                    the UTF8 encoded archive path
    /// \param[out] err                     error code
    /// \param[in]  header_info             file header to check, will contain the file hash
    ///                                     in case it was set while writing
    static zip_t *open(
        IAllocator                      *alloc,
        char const                      *path,
        MDL_zip_container_error_code    &err,
        MDL_zip_container_header        &header_info);

private:
    // Get the length of a file from the file pointer.
    static size_t file_length(FILE *fp);

    // non copyable
    MDL_zip_container(MDL_zip_container const &) MDL_DELETED_FUNCTION;
    MDL_zip_container &operator=(MDL_zip_container const &) MDL_DELETED_FUNCTION;

protected:
    /// The allocator.
    IAllocator *m_alloc;

    /// The name of the archive.
    string m_path;

    /// The zip archive handle.
    zip_t *m_za;

    /// The header of the zip file.
    MDL_zip_container_header m_header;

    /// True, if this container supports resource hashes.
    bool m_has_resource_hashes;
};

/// Helper class for file from an archive.
class MDL_zip_container_file
{
    friend class Allocator_builder;
    friend class MDL_zip_container;

public:
    /// Close a file inside an archive.
    void close();

    /// Read from a file inside an archive.
    ///
    /// \param buffer  destination buffer
    /// \param len     number of bytes to read
    ///
    /// \return number of bytes read
    zip_int64_t read(void *buffer, zip_uint64_t len);

    /// Seek inside a file inside an archive.
    ///
    /// \param offset   seek offset
    /// \param origin   the origin for this seek operation, SEEK_CUR, SEEK_SET, or SEEK_END
    zip_int64_t seek(zip_int64_t offset, int origin);

    /// Get the current file position.
    zip_int64_t tell();

    /// Get the value of an extra field that belongs to this file.
    ///
    /// \param extra_field_id   the filed id to read.
    /// \param length           number of bytes of the returned data.
    ///
    /// \return                 content of the extra field. Memory is managed by the zip archive.
    unsigned char const *get_extra_field(zip_uint16_t extra_field_id, size_t &length);

private:
    /// Opens a file inside a container.
    ///
    /// \param alloc  the allocator
    /// \param za     the zip archive handle
    /// \param name   the name inside the container (full path using '/' as separator)
    static MDL_zip_container_file *open(
        IAllocator *alloc,
        zip_t      *za,
        char const *name);

    /// Constructor.
    ///
    /// \param alloc    the allocator
    /// \param za       the zip archive handle
    /// \param f        the zip file handle
    /// \param index    the associated index of the file inside the zip archive
    /// \param no_seek  if true, seek operation is not possible
    explicit MDL_zip_container_file(
        IAllocator  *alloc,
        zip_t       * za,
        zip_file_t  *f,
        zip_uint64_t index,
        zip_uint64_t file_len,
        bool         no_seek);

    /// Destructor.
    virtual ~MDL_zip_container_file();

private:
    /// The allocator to be used.
    IAllocator   *m_alloc;

    /// The archive handle.
    zip_t        *m_za;

    /// The file handle.
    zip_file_t   *m_f;

    /// The index of the file inside the archive.
    zip_uint64_t m_index;

    /// Current file offset.
    zip_uint64_t m_ofs;

    /// Length of the file.
    zip_uint64_t m_file_len;

    /// True, if the file is stored uncompressed.
    bool         m_have_seek_tell;

    /// Buffer for simulated seek.
    static char g_trash[1024];
};

// ------------------------------------------------------------------------------------------------


/// Implementation of the IInput_stream interface wrapping a IMDL_resource_reader.
class Resource_Input_stream : public Allocator_interface_implement<IInput_stream>
{
    typedef Allocator_interface_implement<IInput_stream> Base;
public:
    /// Acquires a const interface.
    mi::base::IInterface const *get_interface(
        mi::base::Uuid const &interface_id) const MDL_FINAL;

    /// Acquires a non-const interface.
    mi::base::IInterface *get_interface(
        mi::base::Uuid const &interface_id) MDL_FINAL;

    /// Read a character from the input stream.
    /// \returns    The code of the character read, or -1 on the end of the stream.
    int read_char() MDL_OVERRIDE;

    /// Get the name of the file on which this input stream operates.
    /// \returns    The name of the file or null if the stream does not operate on a file.
    char const *get_filename() MDL_FINAL;

    /// Construct an input stream from a character buffer.
    /// Does NOT copy the buffer, so it must stay until the lifetime of the
    /// Input stream object!
    ///
    /// \param alloc     the allocator
    /// \param buffer    the character buffer
    /// \param length    the length of the buffer
    /// \param filename  the name of the buffer or NULL
    explicit Resource_Input_stream(
        IAllocator           *alloc,
        IMDL_resource_reader *reader);

private:
    // non copyable
    Resource_Input_stream(Resource_Input_stream const &) MDL_DELETED_FUNCTION;
    Resource_Input_stream &operator=(Resource_Input_stream const &) MDL_DELETED_FUNCTION;

protected:
    virtual ~Resource_Input_stream();

private:
    /// Current position.
    mi::base::Handle<IMDL_resource_reader> m_reader;
};


}  // mdl
}  // mi

#endif
