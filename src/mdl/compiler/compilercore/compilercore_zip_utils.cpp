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

#include "pch.h"

#include <mi/mdl/mdl_entity_resolver.h>

#include "compilercore_file_utils.h"
#include "compilercore_file_resolution.h"
#include "compilercore_hash.h"
#include "compilercore_zip_utils.h"

// defined in zipint.h
extern "C" int zip_source_remove(zip_source_t *);
extern "C" zip_int64_t zip_source_supports(zip_source_t *src);

namespace mi {
namespace mdl {

Layered_zip_source::Layered_zip_source(
    zip_source_t                   *base,
    MDL_zip_container_header const &header)
: m_base(base)
, m_header(header)
{
    zip_error_init(&m_ze);
}

Layered_zip_source::~Layered_zip_source()
{
}

/// Open the layered stream
zip_source_t *Layered_zip_source::open(zip_error_t &ze)
{
    return zip_source_function_create(callback, this, &ze);
}

/// The source callback function, delegates all requests to the base layer but adjusts the offsets.
zip_int64_t Layered_zip_source::callback(
    void             *env,
    void             *data,
    zip_uint64_t     len,
    zip_source_cmd_t cmd)
{
    Layered_zip_source *self = reinterpret_cast<Layered_zip_source *>(env);
    zip_source_t *src = self->m_base;
    zip_error_t  &ze  = self->m_ze;

    switch (cmd) {
    case ZIP_SOURCE_BEGIN_WRITE:
        {
            zip_int64_t res = zip_source_begin_write(src);
            ze = *zip_source_error(src);
            if (res < 0)
                return res;

            char version[4];
            version[0] = static_cast<char>(self->m_header.major_version_max >> 8);
            version[1] = static_cast<char>(self->m_header.major_version_max % (1 << 8));
            version[2] = static_cast<char>(self->m_header.minor_version_max >> 8);
            version[3] = static_cast<char>(self->m_header.minor_version_max % (1 << 8));

            res = zip_source_write(src, &self->m_header.prefix, 4);
            res = zip_source_write(src, version, 4);
            res = zip_source_write(src, self->m_header.hash, 16); // zeros at this point
            ze = *zip_source_error(src);
            return res;
        }

    case ZIP_SOURCE_COMMIT_WRITE:
        {
            int res = zip_source_commit_write(src);
            ze = *zip_source_error(src);
            return res;
        }

    case ZIP_SOURCE_CLOSE:
        {
            int res = zip_source_close(src);
            ze = *zip_source_error(src);
            return res;
        }

    case ZIP_SOURCE_ERROR:
        return zip_error_to_data(&ze, data, len);

    case ZIP_SOURCE_FREE:
        zip_source_free(src);
        return 0;

    case ZIP_SOURCE_OPEN:
        {
            int res =zip_source_open(src);
            ze = *zip_source_error(src);
            return res;
        }

    case ZIP_SOURCE_READ:
        {
            zip_int64_t res = zip_source_read(src, data, len);
            ze = *zip_source_error(src);
            return res;
        }

    case ZIP_SOURCE_REMOVE:
        {
            int res = zip_source_remove(src);
            ze = *zip_source_error(src);
            return res;
        }

    case ZIP_SOURCE_ROLLBACK_WRITE:
        {
            zip_source_rollback_write(src);
            ze = *zip_source_error(src);
            return 0;
        }

    case ZIP_SOURCE_SEEK:
        {
            zip_source_args_seek_t *args =
                ZIP_SOURCE_GET_ARGS(zip_source_args_seek_t, data, len, &ze);

            if (args == NULL)
                return -1;

            zip_int64_t offset = args->offset;
            int         whence = args->whence;

            switch (whence) {
            case SEEK_SET:
                // adjust position
                offset += 8;
                break;
            case SEEK_END:
            case SEEK_CUR:
                break;
            }

            int res = zip_source_seek(src, offset, whence);
            ze = *zip_source_error(src);
            return res;
        }

    case ZIP_SOURCE_SEEK_WRITE:
        {
            zip_source_args_seek_t *args;

            args = ZIP_SOURCE_GET_ARGS(zip_source_args_seek_t, data, len, &ze);
            if (args == NULL) {
                return -1;
            }

            zip_int64_t offset = args->offset;
            int         whence = args->whence;

            switch (whence) {
            case SEEK_SET:
                // adjust position
                offset += 8;
                break;
            case SEEK_END:
            case SEEK_CUR:
                break;
            }

            int res = zip_source_seek_write(src, offset, whence);
            ze = *zip_source_error(src);
            return res;
        }

    case ZIP_SOURCE_STAT:
        {
            zip_stat_t *st = (zip_stat_t *)data;
            int res = zip_source_stat(src, st);
            ze = *zip_source_error(src);

            if (res < 0)
                return res;

            zip_uint64_t header_offset = 8;
            if (st->valid & ZIP_STAT_SIZE) {
                if (st->size >= header_offset)
                    st->size -= header_offset;
            }
            return res;
        }

    case ZIP_SOURCE_SUPPORTS:
        return zip_source_supports(src);

    case ZIP_SOURCE_TELL:
        {
            zip_int64_t pos = zip_source_tell(src);
            zip_int64_t header_offset = static_cast<zip_int64_t>(8);
            ze = *zip_source_error(src);
            if (pos >= header_offset)
                pos -= header_offset;
            return pos;
        }

    case ZIP_SOURCE_TELL_WRITE:
        {
            zip_int64_t pos = zip_source_tell_write(src);
            zip_int64_t header_offset = static_cast<zip_int64_t>(8);
            ze = *zip_source_error(src);
            if (pos >= header_offset)
                pos -= header_offset;
            return pos;
        }

    case ZIP_SOURCE_WRITE:
        {
            zip_int64_t res = zip_source_write(src, data, len);
            ze = *zip_source_error(src);
            return res;
        }

    default:
        zip_error_set(&ze, ZIP_ER_OPNOTSUPP, 0);
        return -1;
    }
}

// ------------------------------------------------------------------------------------------------

Resource_zip_source::Resource_zip_source(IMDL_resource_reader *reader)
: m_reader(mi::base::make_handle_dup(reader))
{
    zip_error_init(&m_ze);
}

// creates the actual zip source
zip_source_t *Resource_zip_source::open(zip_error_t &ze)
{
    // create user data object
    return zip_source_function_create(callback, this, &ze);
}

// The source callback function invoked by libzip
zip_int64_t Resource_zip_source::callback(
    void             *env,
    void             *data,
    zip_uint64_t     len,
    zip_source_cmd_t cmd)
{
    Resource_zip_source *self = reinterpret_cast<Resource_zip_source*>(env);

    switch (cmd)
    {
    case ZIP_SOURCE_CLOSE:
        return 0;

    case ZIP_SOURCE_ERROR:
        return zip_error_to_data(&self->m_ze, data, len);

    case ZIP_SOURCE_OPEN:
        {
            if (!self->m_reader)
                return -1;
            return 0;
        }

    case ZIP_SOURCE_READ:
        {
            zip_int64_t res = self->m_reader->read(data, len);
            return res;
        }

    case ZIP_SOURCE_STAT:
        {
            zip_stat_t *st = (zip_stat_t *) data;
            zip_stat_init(st);
            return sizeof(struct zip_stat);
        }

    case ZIP_SOURCE_FREE:
    {
        // nothing todo here
    }

    default:
        zip_error_set(&self->m_ze, ZIP_ER_OPNOTSUPP, 0);
        return -1;
    }
}

// ------------------------------------------------------------------------------------------------


// Read a memory block from the resource.
Uint64 MDL_zip_resource_reader::read(void *ptr, Uint64 size)
{
    return m_file->get_container_file()->read(ptr, size);
}

// Get the current position.
Uint64 MDL_zip_resource_reader::tell()
{
    return m_file->get_container_file()->tell();
}

// Reposition stream position indicator.
bool MDL_zip_resource_reader::seek(Sint64 offset, Position origin)
{
    return m_file->get_container_file()->seek(offset, origin) != 0;
}

// Get the UTF8 encoded name of the resource on which this reader operates.
char const *MDL_zip_resource_reader::get_filename() const
{
    return m_file_name.empty() ? NULL : m_file_name.c_str();
}

// Get the UTF8 encoded absolute MDL resource name on which this reader operates.
char const *MDL_zip_resource_reader::get_mdl_url() const
{
    return m_mdl_url.empty() ? NULL : m_mdl_url.c_str();
}

// Returns the associated hash of this resource.
bool MDL_zip_resource_reader::get_resource_hash(unsigned char hash[16])
{
    if (MDL_zip_container_file *z_f = m_file->get_container_file()) {
        size_t length = 0;
        unsigned char const *stored_hash =
            z_f->get_extra_field(MDLE_EXTRA_FIELD_ID_MD, length);

        if (stored_hash != NULL && length == 16) {
            memcpy(hash, stored_hash, 16);
            return true;
        }
    }
    return false;
}

// Constructor.
MDL_zip_resource_reader::MDL_zip_resource_reader(
    IAllocator  *alloc,
    File_handle *f,
    char const  *filename,
    char const  *mdl_url)
: Base(alloc)
, m_file(f)
, m_file_name(filename, alloc)
, m_mdl_url(mdl_url, alloc)
{
}

// Destructor
MDL_zip_resource_reader::~MDL_zip_resource_reader()
{
    File_handle::close(m_file);
}


// ------------------------------------------------------------------------------------------------

static MDL_zip_container_error_code translate_zip_error(zip_error_t const &ze)
{
    switch (ze.zip_err)
    {
    case ZIP_ER_MULTIDISK:
        /* N Multi-disk zip archives not supported */
        return EC_INVALID_CONTAINER;

    case ZIP_ER_RENAME:
        /* S Renaming temporary file failed */
        return EC_RENAME_ERROR;

    case ZIP_ER_CLOSE:
        /* S Closing zip archive failed */
    case ZIP_ER_SEEK:
        /* S Seek error */
    case ZIP_ER_READ:
        /* S Read error */
    case ZIP_ER_WRITE:
        /* S Write error */
        return EC_IO_ERROR;

    case ZIP_ER_CRC:
        /* N CRC error */
        return EC_CRC_ERROR;

    case ZIP_ER_ZIPCLOSED:
        /* N Containing zip archive was closed */
        return EC_INTERNAL_ERROR;

    case ZIP_ER_NOENT:
        /* N No such file */
        return EC_CONTAINER_NOT_EXIST;

    case ZIP_ER_EXISTS:
        /* N File already exists */
        return EC_INTERNAL_ERROR; // no write operation here

    case ZIP_ER_OPEN:
        /* S Can't open file */
        return EC_CONTAINER_OPEN_FAILED;

    case ZIP_ER_TMPOPEN:
        /* S Failure to create temporary file */
        return EC_INTERNAL_ERROR; // no write operation here

    case ZIP_ER_ZLIB:
        /* Z Zlib error */
        return EC_INVALID_CONTAINER;

    case ZIP_ER_MEMORY:
        /* N Malloc failure */
        return EC_MEMORY_ALLOCATION;

    case ZIP_ER_COMPNOTSUPP:
        /* N Compression method not supported */
        return EC_INVALID_CONTAINER;

    case ZIP_ER_EOF:
        /* N Premature end of file */
        return EC_INVALID_CONTAINER;

    case ZIP_ER_NOZIP:
        /* N Not a zip archive */
        return EC_INVALID_CONTAINER;

    case ZIP_ER_INCONS:
        /* N Zip archive inconsistent */
        return EC_INVALID_CONTAINER;

    case ZIP_ER_REMOVE:
        /* S Can't remove file */
        return EC_INTERNAL_ERROR; // no write operation here

    case ZIP_ER_ENCRNOTSUPP:
        /* N Encryption method not supported */
        return EC_INVALID_CONTAINER;

    case ZIP_ER_RDONLY:
        /* N Read-only archive */
        return EC_INTERNAL_ERROR; // no write operation here

    case ZIP_ER_NOPASSWD:
        /* N No password provided */
    case ZIP_ER_WRONGPASSWD:
        /* N Wrong password provided */
        return EC_INVALID_PASSWORD;

    case ZIP_ER_CHANGED:
        /* N Entry has been changed */
    case ZIP_ER_INTERNAL:
        /* N Internal error */
    case ZIP_ER_INVAL:
        /* N Invalid argument */
    case ZIP_ER_DELETED:
        /* N Entry has been deleted */
    case ZIP_ER_OPNOTSUPP:
        /* N Operation not supported */
    case ZIP_ER_INUSE:
        /* N Resource still in use */
    case ZIP_ER_TELL:
        /* S Tell error */
        return EC_INTERNAL_ERROR;

    default:
        return EC_INTERNAL_ERROR;
    }
}

// Constructor.
MDL_zip_container_header::MDL_zip_container_header(
    char const *prefix,
    uint8_t prefix_size,
    uint16_t major,
    uint16_t minor)
: MDL_zip_container_header(prefix, prefix_size, major, minor, major, minor)
{
}

// Constructor.
MDL_zip_container_header::MDL_zip_container_header(
    char const *prefix,
    uint8_t prefix_size,
    uint16_t major_min,
    uint16_t minor_min,
    uint16_t major_max,
    uint16_t minor_max)
: prefix()
, major_version_min(major_min)
, minor_version_min(minor_min)
, major_version_max(major_max)
, minor_version_max(minor_max)
{
    memset((void*) (&this->prefix[0]), 0, 4);
    MDL_ASSERT(prefix_size <= 4);
    memcpy((void*) (&this->prefix[0]), prefix, prefix_size);
    memset((void*) (&this->hash[0]), 0, 16);
}

/// Copy constructor.
MDL_zip_container_header::MDL_zip_container_header(const MDL_zip_container_header &to_copy)
    : prefix()
    , major_version_min(to_copy.major_version_min)
    , minor_version_min(to_copy.minor_version_min)
    , major_version_max(to_copy.major_version_max)
    , minor_version_max(to_copy.minor_version_max)
{
    memcpy((void*) &prefix[0], to_copy.prefix, 4);
    memcpy((void*) &hash[0], to_copy.hash, 16);
}

MDL_zip_container_header& MDL_zip_container_header::operator=(
    const MDL_zip_container_header &to_copy)
{
    // check for self-assignment
    if (&to_copy == this)
        return *this;

    major_version_min = to_copy.major_version_min;
    minor_version_min = to_copy.minor_version_min;
    major_version_max = to_copy.major_version_max;
    minor_version_max = to_copy.minor_version_max;
    memcpy((void*) &prefix[0], to_copy.prefix, 4);
    memcpy((void*) &hash[0], to_copy.hash, 16);

    return *this;
}

// Constructor.
MDL_zip_container::MDL_zip_container(
    IAllocator *alloc,
    char const *path,
    zip_t      *za,
    bool       supports_resource_hashes)
: m_alloc(alloc)
, m_path(path, alloc)
, m_za(za)
, m_header("\0\0\0\0", 4, 0, 0)
, m_has_resource_hashes(supports_resource_hashes)
{
}

// Destructor
MDL_zip_container::~MDL_zip_container()
{
}

// Open a container file.
zip_t *MDL_zip_container::open(
    IAllocator                      *alloc,
    char const                      *path,
    MDL_zip_container_error_code    &err,
    MDL_zip_container_header        &header_info)
{
    err = EC_OK;
    FILE *fp = fopen_utf8(alloc, path, "rb");
    if (fp == NULL) {
        err = (errno == ENOENT) ? EC_CONTAINER_NOT_EXIST : EC_CONTAINER_OPEN_FAILED;
        return NULL;
    }

    size_t len = file_length(fp);

    if (len < 8) {
        err = EC_INVALID_CONTAINER;
        fclose(fp);
        return NULL;
    }

    vector<unsigned char>::Type header(alloc);
    header.resize(8);

    if (fread(header.data(), 1, 8, fp) != 8) {
        err = EC_IO_ERROR;
        fclose(fp);
        return NULL;
    }

    // check header prefix
    for (size_t i = 0; i < 4; ++i) {
        if (header[i] != header_info.prefix[i]) {
            // not valid MDL archive header
            err = EC_INVALID_HEADER;
            fclose(fp);
            return NULL;
        }
    }

    // check header version
    uint16_t major = (header[4] << 8) + header[5];
    uint16_t minor = (header[6] << 8) + header[7];
    uint32_t mask = (major << 16) + minor;
    uint32_t mask_min =( header_info.major_version_min << 16)
                      + header_info.minor_version_min;
    uint32_t mask_max = (header_info.major_version_max << 16)
                      + header_info.minor_version_max;

    if (mask < mask_min || mask > mask_max) {
        err = EC_INVALID_HEADER_VERSION;
        fclose(fp);
        return NULL;
    }

    // write the read version number
    header_info.major_version_max = major;
    header_info.major_version_min = major;
    header_info.minor_version_max = minor;
    header_info.minor_version_min = minor;

    // MDLE format version
    if (strncmp(header_info.prefix, "MDLE", 4) == 0) {
        // since MDLE version 0.2 there is a 16 byte MD5 hash in the header
        if (mask >= 2) {
            if (fread(header_info.hash, 1, 16, fp) != 16) {
                err = EC_IO_ERROR;
                fclose(fp);
                return NULL;
            }
        }
        // version below 1.0
        if (major < 1) {
            err = EC_PRE_RELEASE_VERSION;
        }
    }

    zip_error_t error;
    zip_error_init(&error);

    zip_source_t *zs = zip_source_filep_create(
        fp, 8, len - 8, &error);
    if (zs == NULL) {
        err = translate_zip_error(error);
        return NULL;
    }

    zip_t *za = zip_open_from_source(zs, ZIP_RDONLYNOLASTMOD, &error);
    if (za == NULL) {
        err = translate_zip_error(error);
        return NULL;
    }

    return za;
}

// Close an MDL container.
void MDL_zip_container::close()
{
    zip_close(m_za);

    Allocator_builder builder(m_alloc);
    return builder.destroy(this);
}

// Get the number of files inside an container.
int MDL_zip_container::get_num_entries()
{
    return zip_get_num_files(m_za);
}

// Get the i'th file name inside an container.
char const *MDL_zip_container::get_entry_name(int i)
{
    return zip_get_name(m_za, i, ZIP_FL_ENC_STRICT);
}

// Check if the given file name exists in the container.
bool MDL_zip_container::contains(
    char const *file_name) const
{
    // ZIP uses '/'
    string forward(file_name, m_alloc);
    forward = convert_os_separators_to_slashes(forward);
    return zip_name_locate(m_za, forward.c_str(), ZIP_FL_ENC_STRICT) != -1;
}

// Check if the given file mask exists in the container.
bool MDL_zip_container::contains_mask(
    char const *file_mask) const
{
    // ZIP uses '/'
    string forward(file_mask, m_alloc);
    forward = convert_os_separators_to_slashes(forward);
    for (int i = 0, n = zip_get_num_files(m_za); i < n; ++i) {
        char const *file_name = zip_get_name(m_za, i, ZIP_FL_ENC_STRICT);

        if (utf8_match(forward.c_str(), file_name))
            return true;
    }
    return false;
}

// Open a read_only file from an container
MDL_zip_container_file *MDL_zip_container::file_open(char const *name) const
{
    // ZIP uses '/'
    string zip_name(name, m_alloc);
    zip_name = convert_os_separators_to_slashes(zip_name);
    return MDL_zip_container_file::open(m_alloc, m_za, zip_name.c_str());
}

// Compute the MD5 hash for a file inside a container.
bool MDL_zip_container::compute_file_hash(char const *name, unsigned char md5[16]) const
{
    MDL_zip_container_file *file = file_open(name);
    if (file == NULL)
        return false;

    MD5_hasher hasher;
    unsigned char buffer[1024];
    while (zip_int64_t count = file->read(buffer, 1024)) {
        if (count < 0) {
            file->close();
            return false;
        }
        hasher.update(buffer, count);
    }

    hasher.final(md5);
    file->close();
    return true;
}



// Get the length of a file from the file pointer.
size_t MDL_zip_container::file_length(FILE *fp)
{
#if defined(MI_PLATFORM_LINUX) || defined(MI_PLATFORM_MACOSX)
    size_t curr = ftello(fp);
    fseeko(fp, 0, SEEK_END);
    size_t len = ftello(fp);
    fseeko(fp, curr, SEEK_SET);
    return len;
#elif defined(MI_PLATFORM_WINDOWS)
    size_t curr = _ftelli64(fp);
    _fseeki64(fp, 0, SEEK_END);
    size_t len = _ftelli64(fp);
    _fseeki64(fp, curr, SEEK_SET);
    return len;
#else
    size_t curr = ftell(fp);
    fseek(fp, 0, SEEK_END);
    size_t len = ftell(fp);
    fseek(fp, curr, SEEK_SET);
    return len;
#endif
}

// ------------------------------------------------------------------------------------------------

// Trash buffer.
char MDL_zip_container_file::g_trash[1024];

// Constructor.
MDL_zip_container_file::MDL_zip_container_file(
    IAllocator   *alloc,
    zip_t        *za,
    zip_file_t   *f,
    zip_uint64_t index,
    zip_uint64_t file_len,
    bool         no_seek)
: m_alloc(alloc)
, m_za(za)
, m_f(f)
, m_index(index)
, m_ofs(0)
, m_file_len(file_len)
, m_have_seek_tell(!no_seek)
{
}

// Destructor.
MDL_zip_container_file::~MDL_zip_container_file()
{
    if (m_f != NULL)
        zip_fclose(m_f);
}

// Close a file inside an archive.
void MDL_zip_container_file::close()
{
    Allocator_builder builder(m_alloc);
    builder.destroy(this);
}

// Read from a file inside an archive.
zip_int64_t MDL_zip_container_file::read(void *buffer, zip_uint64_t len)
{
    if (m_f == NULL) {
        // happens, if reopen failed
        return -1;
    }

    zip_int64_t res = zip_fread(m_f, buffer, len);

    if (res > 0)
        m_ofs += res;
    return res;
}

// Seek inside a file inside an archive.
zip_int64_t MDL_zip_container_file::seek(zip_int64_t offset, int origin)
{
    if (m_have_seek_tell) {
        return zip_fseek(m_f, offset, origin);
    }
    if (m_f == NULL) {
        // happens, if reopen failed
        return -1;
    }

    zip_uint64_t nofs = 0;

    switch (origin)
    {
    case SEEK_CUR:
        if (offset < 0 && zip_uint64_t(-offset) > m_ofs)
            nofs = 0;
        else
            nofs = m_ofs + offset;
        break;
    case SEEK_SET:
        if (offset < 0)
            nofs = 0;
        else
            nofs = offset;
        break;
    case SEEK_END:
        if (offset < 0 && zip_uint64_t(-offset) > m_file_len)
            nofs = 0;
        else
            nofs = m_file_len + offset;
        break;
    }

    if (nofs > m_file_len)
        nofs = m_file_len;

    if (nofs < m_file_len) {
        // seek backwards, reopen
        zip_fclose(m_f);

        m_f = zip_fopen_index(m_za, m_index, 0);
        m_ofs = 0;

        if (m_f == NULL)
            return -1;
    }

    while (m_ofs < nofs) {
        zip_uint64_t n = nofs - m_ofs;

        if (n > sizeof(g_trash))
            n = sizeof(g_trash);

        if (read(g_trash, n) <= 0) {
            // prevent endless loop
            return -1;
        }
    }
    return 0;
}

// Get the current file position.
zip_int64_t MDL_zip_container_file::tell()
{
    if (m_have_seek_tell) {
        return zip_ftell(m_f);
    }
    if (m_f == NULL) {
        // happens, if reopen failed
        return -1;
    }

    return m_ofs;
}

// Get the value of an extra field that belongs to this file.
unsigned char const *MDL_zip_container_file::get_extra_field(
    zip_uint16_t extra_field_id,
    size_t       &length)
{
    zip_uint16_t lenp = 0;
    zip_uint8_t const *data = zip_file_extra_field_get_by_id(
        m_za,
        m_index,
        extra_field_id,
        0,
        &lenp,
        ZIP_FL_LOCAL);

    if (data != NULL) {
        length = size_t(lenp);
        return reinterpret_cast<unsigned char const*>(data);
    }
    length = 0;
    return NULL;
}

// Opens a file inside a container.
MDL_zip_container_file *MDL_zip_container_file::open(
    IAllocator *alloc,
    zip_t      *za,
    char const *name)
{
    zip_int64_t index = zip_name_locate(za, name, 0);
    if (index < 0) {
        return NULL;
    }
    zip_file_t *f = zip_fopen_index(za, index, 0);
    if (f == NULL) {
        return NULL;
    }

    zip_uint64_t file_len = 0;
    zip_stat_t st;
    bool forbid_seek = false;
    if (zip_stat_index(za, index, 0, &st) == 0) {
        if (st.valid & ZIP_STAT_SIZE)
            file_len = st.size;
        if (st.valid & ZIP_STAT_COMP_METHOD) {
            if (st.comp_method != ZIP_CM_STORE)
                forbid_seek = true;
        } else {
            // unknown compression
            forbid_seek = true;
        }
        if (st.valid & ZIP_STAT_ENCRYPTION_METHOD) {
            if (st.encryption_method != ZIP_EM_NONE)
                forbid_seek = true;
        }
    } else {
        forbid_seek = true;
    }

    Allocator_builder builder(alloc);

    return builder.create<MDL_zip_container_file>(alloc, za, f, index, file_len, forbid_seek);
}

//-------------------------------------------------------------------------------------------------

// Implementation of the IInput_stream interface wrapping a IMDL_resource_reader.

// Acquires a const interface.
mi::base::IInterface const *Resource_Input_stream::get_interface(
    mi::base::Uuid const &interface_id) const
{
    if (interface_id == IMDL_resource_reader::IID()) {
        m_reader->retain();
        return m_reader.get();
    }
    return Base::get_interface(interface_id);
}

// Acquires a non-const interface.
mi::base::IInterface *Resource_Input_stream::get_interface(
    mi::base::Uuid const &interface_id)
{
    if (interface_id == IMDL_resource_reader::IID()) {
        m_reader->retain();
        return m_reader.get();
    }
    return Base::get_interface(interface_id);
}

// Read a character from the input stream.
int Resource_Input_stream::read_char()
{
    char c;

    if (m_reader->read(&c, 1) == 1)
        return static_cast<unsigned char>(c);
    return -1;
}

// Get the name of the file on which this input stream operates.
char const *Resource_Input_stream::get_filename()
{
    return m_reader->get_filename();
}

// Construct an input stream from a character buffer.
Resource_Input_stream::Resource_Input_stream(
    IAllocator           *alloc,
    IMDL_resource_reader *reader)
: Base(alloc)
, m_reader(reader, mi::base::DUP_INTERFACE)
{
}

// Destructor
 Resource_Input_stream::~Resource_Input_stream()
{
}

}  // mdl
}  // mi
