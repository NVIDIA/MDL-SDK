/******************************************************************************
 * Copyright (c) 2016-2018, NVIDIA CORPORATION. All rights reserved.
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

#include <cstdio>

#include <mi/base/interface_implement.h>

#include "base/lib/libzip/zip.h"

#include "compilercore_file_resolution.h"
#include "compilercore_assert.h"
#include "compilercore_mdl.h"
#include "compilercore_messages.h"
#include "compilercore_errors.h"
#include "compilercore_wchar_support.h"
#include "compilercore_allocator.h"
#include "compilercore_streams.h"
#include "compilercore_printers.h"
#include "compilercore_tools.h"
#include "compilercore_file_utils.h"
#include "compilercore_manifest.h"
#include "compilercore_archiver.h"

namespace mi {
namespace mdl {

class MDL_archive_file;

typedef Store<Position const *> Position_store;

/// Helper class for file from an archive.
class MDL_archive_file {
    friend class Allocator_builder;
    friend class MDL_archive;

public:
    /// Close a file inside an archive.
    void close()
    {
        Allocator_builder builder(m_alloc);
        builder.destroy(this);
    }

    /// Read from a file inside an archive.
    ///
    /// \param buffer  destination buffer
    /// \param len     number of bytes to read
    ///
    /// \return number of bytes read
    zip_int64_t read(void *buffer, zip_uint64_t len)
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

    /// Seek inside a file inside an archive.
    ///
    /// \param offset   seek offset
    /// \param origin   the origin for this seek operation, SEEK_CUR, SEEK_SET, or SEEK_END
    zip_int64_t seek(zip_int64_t offset, int origin)
    {
        if (m_have_seek_tell) {
            return zip_fseek(m_f, offset, origin);
        }
        if (m_f == NULL) {
            // happens, if reopen failed
            return -1;
        }

        zip_uint64_t nofs = 0;

        switch (origin) {
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

            m_f   = zip_fopen_index(m_za, m_index, 0);
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

    /// Get the current file position.
    zip_int64_t tell()
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

private:
    /// Opens a file inside an archive.
    ///
    /// \param alloc  the allocator
    /// \param za     the archiv handle
    /// \param name   the name inside the archive (full path using '/' as separator)
    static MDL_archive_file *open(
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

        return builder.create<MDL_archive_file>(alloc, za, f, index, file_len, forbid_seek);
    }

    /// Constructor.
    ///
    /// \param alloc    the allocator
    /// \param za       the archiv handle
    /// \param f        the zip file handle
    /// \param index    the associated index of the file inside the archive
    /// \param no_seek  if true, seek operation is not possible
    MDL_archive_file(
        IAllocator  *alloc,
        zip_t       * za,
        zip_file_t  *f,
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

    /// Destructor.
    ~MDL_archive_file()
    {
        if (m_f != NULL)
            zip_fclose(m_f);
    }

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

// ------------------------------------------------------------------------

// Trash buffer.
char MDL_archive_file::g_trash[1024];

/// Translate libzip errors into archiv error codes.
///
/// \param ze  a libzip error code
static Archiv_error_code translate_zip_error(zip_error_t const &ze)
{
    switch (ze.zip_err) {
    case ZIP_ER_MULTIDISK:
        /* N Multi-disk zip archives not supported */
        return EC_INVALID_ARCHIVE;

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
        return EC_ARC_NOT_EXIST;

    case ZIP_ER_EXISTS:
        /* N File already exists */
        return EC_INTERNAL_ERROR; // no write operation here

    case ZIP_ER_OPEN:
        /* S Can't open file */
        return EC_ARC_OPEN_FAILED;

    case ZIP_ER_TMPOPEN:
        /* S Failure to create temporary file */
        return EC_INTERNAL_ERROR; // no write operation here

    case ZIP_ER_ZLIB:
        /* Z Zlib error */
        return EC_INVALID_ARCHIVE;

    case ZIP_ER_MEMORY:
        /* N Malloc failure */
        return EC_MEMORY_ALLOCATION;

    case ZIP_ER_COMPNOTSUPP:
        /* N Compression method not supported */
        return EC_INVALID_ARCHIVE;

    case ZIP_ER_EOF:
        /* N Premature end of file */
        return EC_INVALID_ARCHIVE;

    case ZIP_ER_NOZIP:
        /* N Not a zip archive */
        return EC_INVALID_ARCHIVE;

    case ZIP_ER_INCONS:
        /* N Zip archive inconsistent */
        return EC_INVALID_ARCHIVE;

    case ZIP_ER_REMOVE:
        /* S Can't remove file */
        return EC_INTERNAL_ERROR; // no write operation here

    case ZIP_ER_ENCRNOTSUPP:
        /* N Encryption method not supported */
        return EC_INVALID_ARCHIVE;

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


/// Helper class for an archive.
class MDL_archive {
    friend class Allocator_builder;

public:
    /// Open an MDL archive.
    ///
    /// \param[in]  alloc          the allocator
    /// \param[in]  path           the UTF8 encoded archive path
    /// \param[out] err            error code
    /// \param[in]  with_manifest  load and check the manifest
    static MDL_archive *open(
        IAllocator        *alloc,
        char const        *path,
        Archiv_error_code &err,
        bool               with_manifest = true)
    {
        err = EC_OK;
        FILE *fp = fopen_utf8(alloc, path, "rb");
        if (fp == NULL) {
            err = (errno == ENOENT) ? EC_ARC_NOT_EXIST : EC_ARC_OPEN_FAILED;
            return NULL;
        }

        size_t len = file_length(fp);

        if (len < 8) {
            err = EC_INVALID_ARCHIVE;
            fclose(fp);
            return NULL;
        }

        unsigned char header[8];

        if (fread(header, 8, 1, fp) != 1) {
            err = EC_IO_ERROR;
            fclose(fp);
            return NULL;
        }

        if (header[0] != 0x4D ||
            header[1] != 0x44 ||
            header[2] != 0x52 ||
            header[3] != 0x00 ||
            header[4] != 0x00 ||
            header[5] != 0x01 ||
            header[6] != 0x00 ||
            header[7] != 0x00)
        {
            // not valid MDL archive header
            err = EC_INVALID_ARCHIVE;
            fclose(fp);
            return NULL;
        }

        zip_error_t error;
        zip_error_init(&error);

        zip_source_t *zs = zip_source_filep_create(fp, 8, len - 8, &error);
        if (zs == NULL) {
            err = translate_zip_error(error);
            return NULL;
        }

        zip_t *za = zip_open_from_source(zs, ZIP_RDONLY, &error);
        if (za == NULL) {
            err = translate_zip_error(error);
            return NULL;
        }

        zip_int64_t manifest_idx = zip_name_locate(za, "MANIFEST", ZIP_FL_ENC_UTF_8);
        if (manifest_idx != 0) {
            // MANIFEST must be the first entry in an archive
            zip_close(za);
            err = EC_INVALID_ARCHIVE;
            return NULL;
        }

        zip_stat_t st;
        if (zip_stat_index(za, manifest_idx, ZIP_FL_ENC_UTF_8, &st) < 0) {
            zip_close(za);
            err = EC_INVALID_ARCHIVE;
            return NULL;
        }
        if ((st.valid & ZIP_STAT_COMP_METHOD) == 0 || st.comp_method != ZIP_CM_STORE) {
            // MANIFEST is not stored uncompressed
            zip_close(za);
            err = EC_INVALID_ARCHIVE;
            return NULL;
        }

        Allocator_builder builder(alloc);
        MDL_archive *archiv = builder.create<MDL_archive>(alloc, path, za, with_manifest);

        if (with_manifest) {
            mi::base::Handle<Manifest const> m(archiv->get_manifest());
            if (!m.is_valid_interface()) {
                // MANIFEST missing or parse error occurred
                builder.destroy(archiv);
                err = EC_INVALID_ARCHIVE;
                return NULL;
            }
        }
        return archiv;
    }

    /// Close an MDL archive.
    void close()
    {
        zip_close(m_za);

        Allocator_builder builder(m_alloc);
        return builder.destroy(this);
    }

    /// Get the number of files inside an archive.
    int get_num_entries()
    {
        return zip_get_num_files(m_za);
    }

    /// Get the i'th file name inside an archive.
    char const *get_entry_name(int i)
    {
        return zip_get_name(m_za, i, ZIP_FL_ENC_STRICT);
    }

    /// Check if the given file name exists in the archive.
    bool contains(
        char const *file_name) const
    {
        // ZIP uses '/'
        string forward(file_name, m_alloc);
        forward = convert_os_separators_to_slashes(forward);
        return zip_name_locate(m_za, forward.c_str(), ZIP_FL_ENC_STRICT) != -1;
    }

    /// Check if the given file mask exists in the archive.
    bool contains_mask(
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

    /// Open a read_only file from an archive
    MDL_archive_file *file_open(char const *name) const
    {
        // ZIP uses '/'
        string zip_name(name, m_alloc);
        zip_name = convert_os_separators_to_slashes(zip_name);
        return MDL_archive_file::open(m_alloc, m_za, zip_name.c_str());
    }

    /// Get the archive name.
    char const *get_archive_name() const { return m_archive_name.c_str(); }

    /// Get the allocator.
    IAllocator *get_allocator() const { return m_alloc; }

    /// Get the manifest of this archive.
    /// If the manifest was not loaded with open, already, it will be loaded now.
    Manifest const *get_manifest()
    {
        Manifest const *m = m_manifest.get();
        if (m == NULL) {
            m_manifest = parse_manifest();
            m = m_manifest.get();
        }
        if (m != NULL)
            m->retain();
        return m;
    }

private:
    // Get the length of a file from the file pointer.
    static size_t file_length(FILE *fp)
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

    /// Get the manifest.
    Manifest *parse_manifest();

private:
    /// Constructor.
    MDL_archive(
        IAllocator *alloc,
        char const *archive_name,
        zip_t      *za,
        bool        with_manifest)
    : m_alloc(alloc)
    , m_archive_name(archive_name, alloc)
    , m_za(za)
    , m_manifest(with_manifest ? parse_manifest() : NULL)
    {
    }

    // non copyable
    MDL_archive(MDL_archive const &) MDL_DELETED_FUNCTION;
    MDL_archive &operator=(MDL_archive const &) MDL_DELETED_FUNCTION;

private:
    /// The allocator.
    IAllocator *m_alloc;

    /// The name of the archive.
    string m_archive_name;

    /// The zip archive handle.
    zip_t *m_za;

    /// The manifest of this archive.
    mi::base::Handle<Manifest const> m_manifest;
};

/// Helper class to transparently handle files on file system and inside archives.
class File_handle {
    friend class Allocator_builder;
public:
    /// Returns true if this handle represents an archive.
    bool is_archive() const { return m_archive != NULL; }

    /// Get the FILE handle if this object represents an ordinary file.
    FILE *get_file() { return u.fp; }

    /// Get the archive if this object represents a file inside a MDL archive.
    MDL_archive *get_archive() { return m_archive; }

    /// Get the compressed file handle if this object represents a file inside a MDL archive.
    MDL_archive_file *get_archive_file() { return u.z_fp; }

    /// Open a file handle.
    ///
    /// \param[in]  alloc  the allocator
    /// \param[in]  name   a file name (might describe a file inside an archive)
    /// \param[out] err    error code
    static File_handle *open(
        IAllocator        *alloc,
        char const        *name,
        Archiv_error_code &err)
    {
        Allocator_builder builder(alloc);

        const char *p = name;

        err = EC_OK;
        p = strstr(p, ".mdr:");
        if (p == NULL) {
            // must be a file
            if (FILE *fp = fopen_utf8(alloc, name, "rb")) {
                return builder.create<File_handle>(alloc, fp);
            }
            err = EC_FILE_OPEN_FAILED;
            return NULL;
        }

        string root_name(name, p + 4, alloc);
        p += 5;

        // check if the root is an archive itself
        if (MDL_archive *archive = MDL_archive::open(alloc, root_name.c_str(), err)) {
            if (MDL_archive_file *fp = archive->file_open(p)) {
                return builder.create<File_handle>(alloc, archive, /*owns_archive=*/true, fp);
            }
            archive->close();

            // not in the archive
            err = EC_NOT_FOUND;
            return NULL;
        }
        return NULL;
    }

    /// Close a file handle.
    static void close(File_handle *h)
    {
        if (h != NULL) {
            Allocator_builder builder(h->m_alloc);

            builder.destroy(h);
        }
    }

    /// Get the manifest.
    Manifest const *get_manifest()
    {
        if (!is_archive())
            return NULL;
        return m_archive->get_manifest();
    }

private:
    /// Constructor from a FILE handle.
    ///
    /// \param alloc  the allocator
    /// \param fp     a FILE pointer, takes ownership
    explicit File_handle(
        IAllocator *alloc,
        FILE       *fp)
    : m_alloc(alloc)
    , m_archive(NULL)
    , m_owns_archive(false)
    {
        u.fp = fp;
    }

    /// Constructor from archive file.
    ///
    /// \param alloc        the allocator
    /// \param archiv       a MDL archive pointer, takes ownership
    /// \param owns_archive true if this File_handle owns the archive (will be closed then)
    /// \param fp           a MDL compressed file pointer, takes ownership
    File_handle(
        IAllocator       *alloc,
        MDL_archive      *archive,
        bool             owns_archive,
        MDL_archive_file *fp)
    : m_alloc(alloc)
    , m_archive(archive)
    , m_owns_archive(owns_archive)
    {
        u.z_fp = fp;
    }

    /// Constructor from another File_handle archive.
    ///
    /// \param alloc   the allocator
    /// \param fh      another archive file handle
    /// \param fp      a MDL compressed file pointer, takes ownership
    File_handle(
        File_handle      *fh,
        MDL_archive_file *fp)
    : m_alloc(fh->m_alloc)
    , m_archive(fh->m_archive)
    , m_owns_archive(false)
    {
        MDL_ASSERT(fh->is_archive());
        u.z_fp = fp;
    }

    /// Destructor.
    ~File_handle()
    {
        if (is_archive()) {
            u.z_fp->close();
            if (m_owns_archive)
                m_archive->close();
        } else {
            fclose(u.fp);
        }
    }

private:
    /// Current allocator.
    IAllocator *m_alloc;

    /// If non-null, this is an archive.
    MDL_archive *m_archive;

    union {
        FILE             *fp;
        MDL_archive_file *z_fp;
    } u;

    /// If true, this file handle has ownership on the MDL archiv.
    bool m_owns_archive;
};

// ------------------------------------------------------------------------

/// Implementation of a resource reader from a file.
class File_resource_reader : public Allocator_interface_implement<IMDL_resource_reader>
{
    typedef Allocator_interface_implement<IMDL_resource_reader> Base;
public:
    /// Read a memory block from the resource.
    ///
    /// \param ptr   Pointer to a block of memory with a size of size bytes
    /// \param size  Number of bytes to read
    ///
    /// \returns    The total number of bytes successfully read.
    Uint64 read(void *ptr, Uint64 size) MDL_FINAL
    {
        return fread(ptr, 1, size, m_file->get_file());
    }

    /// Get the current position.
    Uint64 tell() MDL_FINAL
    {
#if defined(MI_PLATFORM_LINUX) || defined(MI_PLATFORM_MACOSX)
        return ftello(m_file->get_file());
#elif defined(MI_PLATFORM_WINDOWS)
        return _ftelli64(m_file->get_file());
#else
        return ftell(m_file->get_file());
#endif
    }

    /// Reposition stream position indicator.
    ///
    /// \param offset  Number of bytes to offset from origin
    /// \param origin  Position used as reference for the offset
    ///
    /// \return true on success
    bool seek(Sint64 offset, Position origin) MDL_FINAL
    {
#if defined(MI_PLATFORM_LINUX) || defined(MI_PLATFORM_MACOSX)
        return fseeko(m_file->get_file(), off_t(offset), origin) == 0;
#elif defined(MI_PLATFORM_WINDOWS)
        return _fseeki64(m_file->get_file(), offset, origin) == 0;
#else
        return fseek(m_file->get_file(), long(offset), origin) == 0;
#endif
    }

    /// Get the UTF8 encoded name of the resource on which this reader operates.
    /// \returns    The name of the resource or NULL.
    char const *get_filename() const MDL_FINAL
    {
        return m_file_name.empty() ? NULL : m_file_name.c_str();
    }

    /// Get the UTF8 encoded absolute MDL resource name on which this reader operates.
    /// \returns    The absolute MDL url of the resource or NULL.
    char const *get_mdl_url() const MDL_FINAL
    {
        return m_mdl_url.empty() ? NULL : m_mdl_url.c_str();
    }

    /// Constructor.
    ///
    /// \param alloc             the allocator
    /// \param f                 the file handle
    /// \param filename          the file name
    /// \param mdl_url           the MDL url
    ///                          if this object is destroyed
    explicit File_resource_reader(
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

private:
    // non copyable
    File_resource_reader(File_resource_reader const &) MDL_DELETED_FUNCTION;
    File_resource_reader &operator=(File_resource_reader const &) MDL_DELETED_FUNCTION;

private:
    ~File_resource_reader() MDL_FINAL
    {
        File_handle::close(m_file);
    }

private:
    /// The file handle.
    File_handle *m_file;

    /// The filename.
    string m_file_name;

    /// The absolute MDl url.
    string m_mdl_url;
};

namespace {

/// Implementation of the IInput_stream interface using FILE I/O.
class Simple_input_stream : public Allocator_interface_implement<IInput_stream>
{
    typedef Allocator_interface_implement<IInput_stream> Base;
public:
    /// Constructor.
    explicit Simple_input_stream(
        IAllocator  *alloc,
        File_handle *f,
        char const *filename)
    : Base(alloc)
    , m_file(f)
    , m_filename(filename, alloc)
    {}

    /// Destructor.
    ~Simple_input_stream() MDL_FINAL
    {
        File_handle::close(m_file);
    }

    /// Read a character from the input stream.
    /// \returns    The code of the character read, or -1 on the end of the stream.
    int read_char() MDL_FINAL
    {
        return fgetc(m_file->get_file());
    }

    /// Get the name of the file on which this input stream operates.
    /// \returns    The name of the file or null if the stream does not operate on a file.
    char const *get_filename() MDL_FINAL
    {
        return m_filename.empty() ? 0 : m_filename.c_str();
    }

private:
    /// The file handle.
    File_handle *m_file;

    /// The filename.
    string m_filename;
};

/// Implementation of the IArchive_input_stream interface using archive I/O.
class Archive_input_stream : public Allocator_interface_implement<IArchive_input_stream>
{
    typedef Allocator_interface_implement<IArchive_input_stream> Base;
public:
    /// Constructor.
    explicit Archive_input_stream(
        IAllocator     *alloc,
        File_handle    *f,
        char const     *filename,
        Manifest const *manifest)
    : Base(alloc)
    , m_file(f)
    , m_filename(filename, alloc)
    , m_manifest(manifest, mi::base::DUP_INTERFACE)
    {
    }

protected:
    /// Destructor.
    ~Archive_input_stream() MDL_FINAL
    {
        File_handle::close(m_file);
    }

public:
    /// Read a character from the input stream.
    /// \returns    The code of the character read, or -1 on the end of the stream.
    int read_char() MDL_FINAL
    {
        unsigned char buf;
        if (m_file->get_archive_file()->read(&buf, 1) != 1)
            return -1;
        return int(buf);
    }

    /// Get the name of the file on which this input stream operates.
    /// \returns    The name of the file or null if the stream does not operate on a file.
    char const *get_filename() MDL_FINAL
    {
        return m_filename.empty() ? 0 : m_filename.c_str();
    }

    /// Get the manifest of the owning archive.
    IArchive_manifest const *get_manifest() const MDL_FINAL
    {
        if (m_manifest.is_valid_interface()) {
            m_manifest->retain();
            return m_manifest.get();
        }
        return NULL;
    }

private:
    /// The file handle.
    File_handle *m_file;

    /// The filename.
    string m_filename;

    /// The archive manifest.
    mi::base::Handle<Manifest const> m_manifest;
};

} // anonymous

// ------------------------------------------------------------------------

typedef struct zip      zip_t;
typedef struct zip_file zip_file_t;


static char const *get_builtin_prefix()
{
#ifndef MI_PLATFORM_WINDOWS
    return "/<builtin>/";
#else // MI_PLATFORM_WINDOWS
    return "C:\\<builtin>\\";
#endif // MI_PLATFORM_WINDOWS
}

static char const *get_string_based_prefix()
{
#ifndef MI_PLATFORM_WINDOWS
    return "/<string>/";
#else // MI_PLATFORM_WINDOWS
    return "C:\\<string>\\";
#endif // MI_PLATFORM_WINDOWS
}

// Constructor.
File_resolver::File_resolver(
    MDL const                                &mdl,
    IModule_cache                            *module_cache,
    mi::base::Handle<IMDL_search_path> const &search_path,
    mi::base::Lock                           &sp_lock,
    Messages_impl                            &msgs,
    char const                               *front_path)
: m_alloc(mdl.get_allocator())
, m_mdl(mdl)
, m_module_cache(module_cache)
, m_msgs(msgs)
, m_pos(NULL)
, m_search_path(search_path)
, m_resolver_lock(sp_lock)
, m_paths(m_alloc)
, m_resource_paths(m_alloc)
, m_killed_packages(String_set::key_compare(), m_alloc)
, m_front_path(front_path)
, m_resolve_entity(NULL)
, m_last_msg_idx(0)
, m_pathes_read(false)
, m_resolving_resource(false)
{
}

// Creates a new error.
void File_resolver::error(
    int                code,
    Err_location const &loc,
    Error_params const &params)
{
    string msg(m_msgs.format_msg(code, MESSAGE_CLASS, params));
    m_last_msg_idx = m_msgs.add_error_message(
        code, MESSAGE_CLASS, /*file_id=*/0, loc.get_position(), msg.c_str());
}

// Add a note to the last error/warning.
void File_resolver::add_note(
    int                code,
    Err_location const &loc,
    Error_params const &params)
{
    string msg(m_msgs.format_msg(code, MESSAGE_CLASS, params));
    m_msgs.add_note(
        m_last_msg_idx,
        IMessage::MS_INFO,
        code,
        MESSAGE_CLASS,
        /*file_id=*/0,
        loc.get_position(),
        msg.c_str());
}

// Creates a new warning.
void File_resolver::warning(
    int                code,
    Err_location const &loc,
    Error_params const &params)
{
    string msg(m_msgs.format_msg(code, MESSAGE_CLASS, params));
    m_msgs.add_warning_message(code, MESSAGE_CLASS, /*file_id=*/0, loc.get_position(), msg.c_str());
}

// Convert a name which might be either an url (separator '/') or an module name (separator '::')
// into an url.
string File_resolver::to_url(
    char const *input_name) const
{
    string input_url(m_alloc);
    for (;;) {
        char const *p = strstr(input_name, "::");

        if (p == NULL) {
            input_url.append(input_name);
            return input_url;
        }
        input_url.append(input_name, p - input_name);
        input_url.append('/');
        input_name = p + 2;
    }
}

// Convert an url (separator '/') into  a module name (separator '::')
string File_resolver::to_module_name(
    char const *input_url) const
{
    string input_name(m_alloc);
    for (;;) {
        char const *p = strchr(input_url, '/');

#ifdef MI_PLATFORM_WINDOWS
        char const *q = strchr(input_url, '\\');

        if (q != NULL)
            if (p == NULL || q < p)
                p = q;
#endif

        if (p == NULL) {
            input_name.append(input_url);
            return input_name;
        }
        input_name.append(input_url, p - input_url);
        input_name.append("::");
        input_url = p + 1;
    }
}

// Convert an url (separator '/') into an archive name (separator '.')
string File_resolver::to_archive(
    char const *input_url) const
{
    string input_name(m_alloc);
    for (;;) {
        char const *p = strchr(input_url, '/');

#ifdef MI_PLATFORM_WINDOWS
        char const *q = strchr(input_url, '\\');

        if (q != NULL)
            if (p == NULL || q < p)
                p = q;
#endif

        if (p == NULL) {
            input_name.append(input_url);
            return input_name;
        }
        input_name.append(input_url, p - input_url);
        input_name.append('.');
        input_url = p + 1;
    }
}

// Check if an archive name is a prefix of a archive path.
bool File_resolver::is_archive_prefix(
    string const &archive_name,
    string const &archive_path) const
{
    char const *prefix = archive_name.c_str();
    size_t l = archive_name.length() - 4; // strip ".mdr"

    if (l >= archive_path.size())
        return false;

    if (archive_path[l] != '.') {
        // fast check: prefix must be followed by '.'
        return false;
    }

    return strncmp(prefix, archive_path.c_str(), l) == 0;
}

// Check whether the given archive contains the requested file.
bool File_resolver::archive_contains(
    char const *archive_name,
    char const *file_name)
{
    bool res = false;

    Archiv_error_code err;
    if (MDL_archive *archive = MDL_archive::open(m_alloc, archive_name, err, false)) {
        res = archive->contains(file_name);
        archive->close();
    } else {
        if (err == EC_INVALID_ARCHIVE) {
            warning(
                INVALID_MDL_ARCHIVE_DETECTED,
                *m_pos,
                Error_params(m_alloc).add(archive_name));
        }
    }
    return res;
}

// Check whether the given archive contains a file matching a mask.
bool File_resolver::archive_contains_mask(
    char const *archive_name,
    char const *file_mask)
{
    bool res = false;

    Archiv_error_code err;
    if (MDL_archive *archive = MDL_archive::open(m_alloc, archive_name, err, false)) {
        res = archive->contains_mask(file_mask);
        archive->close();
    } else {
        if (err == EC_INVALID_ARCHIVE) {
            warning(
                INVALID_MDL_ARCHIVE_DETECTED,
                *m_pos,
                Error_params(m_alloc).add(archive_name));
        }
    }
    return res;
}

// Returns the nesting level of a module, i.e., the number of "::" substrings in the
// fully-qualified module name minus 1.
size_t File_resolver::get_module_nesting_level(char const *module_name)
{
    MDL_ASSERT(module_name[0] == ':' && module_name[1] == ':');

    char const *p    = module_name;
    size_t     level = 0;
    do {
        ++level;
        p = strstr(p + 2, "::");
    } while(p != NULL);

    MDL_ASSERT(level > 0);
    return level - 1;
}

// Splits a file path into a directory path and file name.
void File_resolver::split_file_path(
    string const &input_url,
    string       &directory_path,
    string       &file_path)
{
    string::size_type pos = input_url.rfind('/');
    if (pos != string::npos) {
        directory_path.assign(input_url.substr(0, pos + 1));
        file_path.assign(input_url.substr(pos + 1));
    } else {
        directory_path.clear();
        file_path.assign(input_url);
    }
}

// Splits a module file system path into current working directory, current search path, and
// current module path.
void File_resolver::split_module_file_system_path(
    string const &module_file_system_path,
    string const &module_name,
    size_t       module_nesting_level,
    string       &current_working_directory,
    bool         &cwd_is_archive,
    string       &current_search_path,
    bool         &csp_is_archive,
    string       &current_module_path)
{
    cwd_is_archive = false;
    csp_is_archive = false;
    // special case for string-based modules (not in spec)
    if (module_file_system_path.empty()) {
        current_working_directory.clear();
        current_search_path.clear();

        // remove last "::" and remainder, replace "::" by "/"
        size_t pos = module_name.rfind( "::");
        current_module_path = to_url(module_name.substr(0, pos).c_str());
        MDL_ASSERT(current_module_path.empty() || current_module_path[0] == '/');
        return;
    }

    // regular case for file-based modules (as in spec)
    char sep = os_separator();

    size_t archive_pos = module_file_system_path.find(".mdr:");
    if (archive_pos != string::npos) {
        // inside an archive
        archive_pos += 4; // add ".mdr"
        string simple_path = module_file_system_path.substr(archive_pos + 1);

        size_t last_sep = simple_path.find_last_of(sep);
        if (last_sep != string::npos) {
            current_working_directory =
                module_file_system_path.substr(0, archive_pos + 1) +
                simple_path.substr(0, last_sep);
        } else {
            // only the archive
            current_working_directory = module_file_system_path.substr(0, archive_pos);
            cwd_is_archive = true;
        }

        // points now to ':'
        ++archive_pos;
    } else {
        size_t last_sep = module_file_system_path.find_last_of(sep);
        MDL_ASSERT(last_sep != string::npos);
        current_working_directory = module_file_system_path.substr(0, last_sep);

        archive_pos = 0;
    }

    current_search_path = current_working_directory;
    size_t strip_dotdot = 0;
    while (module_nesting_level-- > 0) {
        size_t last_sep = current_search_path.find_last_of(sep);
        MDL_ASSERT(last_sep != MISTD::string::npos);
        if (last_sep < archive_pos) {
            // do NOT remove the archive name, thread its ':' like '/'
            last_sep = archive_pos - 1;
            // should never try to go out!
            MDL_ASSERT(module_nesting_level == 0);
            archive_pos = 0;
            strip_dotdot = 1;
        } else {
            strip_dotdot = 0;
        }
        current_search_path = current_search_path.substr(0, last_sep);
    }

    csp_is_archive = strip_dotdot != 0;
    current_module_path = convert_os_separators_to_slashes(
        current_working_directory.substr(current_search_path.size() + strip_dotdot));
    if (strip_dotdot)
        current_module_path = '/' + current_module_path;
    MDL_ASSERT(current_module_path.empty() || current_module_path[0] == '/');
}

// Checks that \p file_path contains no "." or ".." directory names.
bool File_resolver::check_no_dots(
    char const *s)
{
    bool absolute = s[0] == '/';
    if (absolute)
        ++s;

    bool start = true;
    for (; s[0] != '\0'; ++s) {
        if (start) {
            if (s[0] == '.') {
                ++s;
                char const *p= ".";
                if (s[0] == '.') {
                    ++s;
                    p = "..";
                }
                if (s[0] == '/' || s[0] == '\0') {
                    error(
                        INVALID_DIRECTORY_NAME,
                        *m_pos,
                        absolute ?
                            Error_params(m_alloc).add_absolute_path_prefix().add(p) :
                            Error_params(m_alloc).add_weak_relative_path_prefix().add(p));
                    return false;
                }
            }
            start = false;
        }
        if (s[0] == '/')
            start = true;
    }
    return true;
}

// Checks that \p file_path contains at most one leading "." directory name, at most
// nesting_level leading ".." directory names, and no such non-leading directory names.
bool File_resolver::check_no_dots_strict_relative(
    char const *s,
    size_t     nesting_level)
{
    MDL_ASSERT(s[0] == '.');
    char const *b = s;

    bool leading  = true;
    bool start    = true;
    bool too_many = false;

    char const *err = NULL;
    for (; s[0] != '\0'; ++s) {
        if (start) {
            if (s[0] == '.') {
                if (s[1] == '/' || s[1] == '\0') {
                    if (s != b) {
                        err = ".";
                        break;
                    } else {
                        leading = false;
                    }
                } else if (s[1] == '.' && (s[2] == '/' || s[2] == '\0')) {
                    if (!leading) {
                        err = "..";
                        break;
                    } else if (nesting_level == 0) {
                        too_many = true;
                        break;
                    } else {
                        --nesting_level;
                    }
                } else
                    leading = false;
            } else
                leading = false;
            start = false;
        }
        if (s[0] == '/')
            start = true;
    }

    if (err != NULL) {
        error(
            INVALID_DIRECTORY_NAME,
            *m_pos,
            Error_params(m_alloc).add_strict_relative_path_prefix().add(err));
        return false;
    }
    if (too_many) {
        error(
            INVALID_DIRECTORY_NAME_NESTING,
            *m_pos,
            Error_params(m_alloc).add_strict_relative_path_prefix());
        return false;
    }

    return true;
}

// Normalizes a file path given by its directory path and file name.
string File_resolver::normalize_file_path(
    string const &file_path,
    string const &file_mask,
    string const &directory_path,
    string const &file_name,
    size_t       nesting_level,
    string const &module_file_system_path,
    string const &current_working_directory,
    bool         cwd_is_archive,
    string const &current_module_path)
{
    // strict relative file paths
    if (file_path[0] == '.' &&
        (file_path[1] == '/' || (file_path[1] == '.' && file_path[2] == '/')))
    {
        // special case (not in spec)
        if (module_file_system_path.empty()) {
            error(
                STRICT_RELATIVE_PATH_IN_STRING_MODULE,
                *m_pos,
                Error_params(m_alloc).add(file_path.c_str()));
            return string(m_alloc);
        }

        // reject invalid strict relative file paths
        if( !check_no_dots_strict_relative(file_path.c_str(), nesting_level))
            return string(m_alloc);

        // reject if file does not exist w.r.t. current working directory
        string file_mask_os = convert_slashes_to_os_separators(file_mask);

        string file = cwd_is_archive ?
            current_working_directory + ':' + simplify_path(file_mask_os, os_separator()) :
            simplify_path(join_path(current_working_directory, file_mask_os), os_separator());
        if (!file_exists(file.c_str())) {
            // FIXME: do we need an error here
            return string(m_alloc);
        }

        // canonical path is the file path resolved w.r.t. the current module path
        return simplify_path(current_module_path + "/" + file_path, '/');
    }

    // absolute file paths
    if (file_path[0] == '/') {
        // reject invalid absolute paths
        if (!check_no_dots(file_path.c_str()))
            return string(m_alloc);

        // canonical path is the same as the file path
        return file_path;
    }


    // weak relative file paths

    // reject invalid weak relative paths
    if (!check_no_dots(file_path.c_str()))
        return string(m_alloc);

    // special case (not in spec)
    if (module_file_system_path.empty())
        return "/" + file_path;

    // if file does not exist w.r.t. current working directory: canonical path is file path
    // prepended with a slash
    string file_mask_os = convert_slashes_to_os_separators(file_mask);
    string file = join_path(current_working_directory, file_mask_os);

    // if the searched file does not exists locally, assume the weak relative path is absolute
    if (!file_exists(file.c_str()))
        return "/" + file_path;

    // otherwise canonical path is the file path resolved w.r.t. the current module path
    return current_module_path + "/" + file_path;
}

bool File_resolver::is_builtin(string const &canonical_file_path) const
{
    string module_name = to_module_name(canonical_file_path.c_str());
    MDL_ASSERT(module_name.substr(module_name.size() - 4) == ".mdl");
    module_name = module_name.substr(0, module_name.size() - 4);
    return m_mdl.is_builtin_module(module_name.c_str());
}

// Check if the given canonical_file_path names a string based module.
bool File_resolver::is_string_based(string const &canonical_file_path) const
{
    if (!m_module_cache)
        return false;
    string module_name = to_module_name(canonical_file_path.c_str());
    MDL_ASSERT(module_name.substr(module_name.size() - 4) == ".mdl");
    module_name = module_name.substr(0, module_name.size() - 4);

    // string modules are never built-in
    if (m_mdl.is_builtin_module(module_name.c_str()))
        return false;

    // check the module cache
    mi::base::Handle<const mi::mdl::IModule> module(m_module_cache->lookup(module_name.c_str()));
    if (!module.is_valid_interface())
        return false;

    // string modules should have no file name
    char const *filename = module->get_filename();
    if (filename != NULL && filename[0] != '\0')
        return false;
    return true;
}

// Loops over the search paths to resolve \p canonical_file_path.
string File_resolver::consider_search_paths(
    string const &canonical_file_mask,
    bool         is_resource,
    char const   *file_path,
    UDIM_mode    udim_mode)
{
    MDL_ASSERT(canonical_file_mask[0] == '/');

    if (!is_resource && is_builtin(canonical_file_mask))
        return get_builtin_prefix() + canonical_file_mask.substr(1);

    string canonical_file_mask_os = convert_slashes_to_os_separators(canonical_file_mask);

    string resolved_filename      = search_mdl_path(
        canonical_file_mask_os.c_str() + 1,
        /*in_extra_resource_path=*/false,
        m_front_path,
        udim_mode);

    // For backward compatibility we also consider the resource search paths for resources.
    if (is_resource && resolved_filename.empty()) {
        resolved_filename      = search_mdl_path(
            canonical_file_mask_os.c_str() + 1,
            /*in_extra_resource_path=*/true,
            m_front_path,
            udim_mode);
        if (!resolved_filename.empty()) {
            warning(
                DEPRECATED_RESOURCE_PATH_USED,
                *m_pos,
                Error_params(m_alloc).add(file_path));
        }
    }

    if (!is_resource && resolved_filename.empty() && is_string_based(canonical_file_mask)) {
        MDL_ASSERT(udim_mode == NO_UDIM && "non-resource files should not be UDIM");
        return get_string_based_prefix() + canonical_file_mask.substr(1);
    }

    // Make resolved filename absolute.
    if (!resolved_filename.empty() && !is_path_absolute(resolved_filename.c_str())) {
        if (resolved_filename.size() >= 2) {
            if (resolved_filename[0] == '.' && resolved_filename[1] == os_separator()) {
                resolved_filename = resolved_filename.substr(2);
            }
        }
        resolved_filename = join_path(get_cwd(m_alloc), resolved_filename);
        resolved_filename = simplify_path(resolved_filename, os_separator());
    }

    return resolved_filename;
}

// Checks whether the resolved file system location passes the consistency checks in the MDL spec.
bool File_resolver::check_consistency(
    string const &resolved_file_system_location,
    string const &canonical_file_path,
    string const &file_path,
    string const &current_working_directory,
    string const &current_search_path,
    bool         is_resource,
    bool         csp_is_archive,
    bool         is_string_module)
{
    // strict relative file paths
    if (file_path.substr(0, 2) == "./" || file_path.substr(0, 3) == "../") {
        // should have already been rejected in the normalization step for string-based modules
        MDL_ASSERT(!is_string_module);

        // check whether resolved file system location is in current search path
        size_t len = current_search_path.size();
        string resolved_search_path = resolved_file_system_location.substr(0, len);
        if (resolved_search_path == current_search_path &&
            (resolved_file_system_location[len] == os_separator() ||
                (len > 4 &&
                    resolved_file_system_location[len]     == ':' &&
                    resolved_file_system_location[len - 1] == 'r' &&
                    resolved_file_system_location[len - 2] == 'd' &&
                    resolved_file_system_location[len - 3] == 'm' &&
                    resolved_file_system_location[len - 4] == '.')
            )
        ) {
            return true;
        } else {
            error(
                FILE_PATH_CONSISTENCY,
                *m_pos,
                Error_params(m_alloc)
                    .add(file_path.c_str())
                    .add(resolved_file_system_location.c_str())
                    .add_current_search_path()
                    .add(current_search_path.c_str()));
            return false;
        }
    }

    // absolute or weak relative file paths

    // skip check for string-based modules (not in spec)
    if (is_string_module)
        return true;

    // check precondition whether canonical file path exists w.r.t. current search path
    string canonical_file_path_os = convert_slashes_to_os_separators(canonical_file_path);
    string file(m_alloc);
    if (csp_is_archive) {
        // construct an "archive path"
        MDL_ASSERT(canonical_file_path_os[0] == os_separator());
        file = current_search_path + ':' + canonical_file_path_os.substr(1);
        if (!file_exists(file.c_str()))
            return true;
    } else {
        file = current_search_path + canonical_file_path_os;
        if (!file_exists(file.c_str()))
            return true;
    }

    // check precondition whether local file is in current working directory (and not below)
    size_t len = current_working_directory.size();
    string directory = file.substr(0, len + 1);
    if (directory != current_working_directory + os_separator())
        return true;
    if (file.substr(len + 1).find(os_separator()) != string::npos)
        return true;

    if (!is_resource && is_builtin(canonical_file_path)) {
        // the following check will fail for builtin modules
        return true;
    }

    // check whether resolved file system location is different from file in current working
    // directory
    if (resolved_file_system_location != file) {
        error(
            FILE_PATH_CONSISTENCY,
            *m_pos,
            Error_params(m_alloc)
                .add(file_path.c_str())
                .add(resolved_file_system_location.c_str())
                .add_current_directory()
                .add(current_working_directory.c_str()));
        return false;
    }
    return true;
}

// Simplifies a file path.
string File_resolver::simplify_path(
    string const &file_path,
    char         sep)
{
    return mi::mdl::simplify_path(m_alloc, file_path, sep);
}

// Check if a given archive is killed by another archive OR a existing directory
bool File_resolver::is_killed(
    char const   *path,
    string const &archive_name,
    String_map   &archives)
{
    string package(m_alloc);

    for (size_t i = 0, n = archive_name.length() - 4; i < n; ++i) {
        char c = archive_name[i];

        if (c == '.')
            package.append(os_separator());
        else
            package.append(c);
    }

    string dir_path(path, m_alloc);

    dir_path.append(os_separator());
    dir_path.append(package);

    if (is_directory_utf8(m_alloc, dir_path.c_str())) {
        error(
            ARCHIVE_CONFLICT,
            *m_pos,
            Error_params(m_alloc)
                .add(path)
                .add(archive_name.c_str())
                .add_package_or_archive(/*is_package=*/true)
                .add(package.c_str())
        );

        m_killed_packages.insert(MDL_MOVE(package));
        return true;
    }

    // ugly: linear search leading to quadratic runtime
    size_t l = archive_name.length() - 4;
    string t = archive_name.substr(0, l);
    for (String_map::iterator it(archives.begin()), end(archives.end()); it != end; ++it) {
        string const &o = it->first;

        if (t == o) {
            // NOT a conflict, this one
            continue;
        }

        size_t ol = o.length();
        if (l < ol) {
            if (o[l] == '.' && t == o.substr(0, l)) {
                // this archive is a prefix of the other => conflict
                if (archives[t]) {
                    // report the error only archive is not bad already
                    error(
                        ARCHIVE_CONFLICT,
                        *m_pos,
                        Error_params(m_alloc)
                            .add(path)
                            .add(archive_name.c_str())
                            .add_package_or_archive(/*is_package=*/false)
                            .add((o + ".mdr").c_str())
                    );
                }

                // mark both as bad
                it->second = false;
                archives[t] = false;
                return true;
            }
        } else {
            if (t[ol] == '.' && t.substr(0, ol) == o) {
                // other is a prefix of this => conflict
                if (archives[t]) {
                    // report the error only archive is not bad already
                    error(
                        ARCHIVE_CONFLICT,
                        *m_pos,
                        Error_params(m_alloc)
                            .add(path)
                            .add(archive_name.c_str())
                            .add_package_or_archive(/*is_package=*/false)
                            .add((o + ".mdr").c_str())
                    );
                }
                // mark both as bad
                it->second = false;
                archives[t] = false;
                return true;
            }
        }
    }

    return false;
}

// Check if a directory path is killed due to a conflicting archive.
bool File_resolver::is_killed(
    char const *file_mask)
{
    if (m_killed_packages.empty()) {
        // speed up the most common case
        return false;
    }

    for (char const *s = file_mask;;) {
        char const *e = strchr(s, os_separator());

        if (e == NULL) {
            // file part found, stop here
            break;
        }

        string dir(file_mask, e, m_alloc);
        if (m_killed_packages.find(dir) != m_killed_packages.end()) {
            // prefix hit
            return true;
        }

        if (e != NULL)
            s = e + 1;
        else
            break;
    }

    return false;
}

// Search the given path in all MDL search paths and return the complete path if found
string File_resolver::search_mdl_path(
    char const *file_mask,
    bool       in_resource_path,
    char const *front_path,
    UDIM_mode  udim_mode)
{
    // Calls to IMDL_search_path are not thread-safe.
    // Ensure that at least this compiler will serialize it.
    // Note that this is still unsafe, if more than one compiler share the same module
    // resolver ...

    // Moreover, try to minimize the lock time by copying the data
    if (!m_pathes_read) {
        mi::base::Lock::Block block(&m_resolver_lock);

        // check again, another thread might catch it!
        if (!m_pathes_read) {
            size_t n = m_search_path->get_search_path_count(IMDL_search_path::MDL_SEARCH_PATH);
            m_paths.reserve(n + (front_path != NULL ? 1 : 0));
            if (front_path != NULL)
                m_paths.push_back(
                convert_slashes_to_os_separators(string(front_path, m_alloc)));
            for (size_t i = 0; i < n; ++i) {
                char const *path = m_search_path->get_search_path(
                    IMDL_search_path::MDL_SEARCH_PATH, i);
                if (path != NULL) {
                    m_paths.push_back(
                        convert_slashes_to_os_separators(string(path, m_alloc)));
                }
            }

            n = m_search_path->get_search_path_count(IMDL_search_path::MDL_RESOURCE_PATH);
            m_resource_paths.reserve(n);
            for (size_t i = 0; i < n; ++i) {
                char const *path = m_search_path->get_search_path(
                    IMDL_search_path::MDL_RESOURCE_PATH, i);
                if (path != NULL) {
                    m_resource_paths.push_back(
                        convert_slashes_to_os_separators(string(path, m_alloc)));
                }
            }
        }
        m_pathes_read = true;
    }

    // the archive name ('.' separators)
    string archive_path = to_archive(file_mask);

    // names inside archive using only '/' as a separator
    string file_name_fs = convert_os_separators_to_slashes(string(file_mask, m_alloc));

    typedef list<string>::Type Place_list;

    Place_list places(m_alloc);
    size_t n_places = 0;

    String_vec const &paths = in_resource_path ? m_resource_paths : m_paths;

    for (String_vec::const_iterator it(paths.begin()), end(paths.end()); it != end; ++it) {
        char const *path = it->c_str();

        // kills happen for ONE search root only
        m_killed_packages.clear();

        if (!in_resource_path) {
            Directory dir(m_alloc);
            if (!dir.open(path)) {
                // directory does not exist
                continue;
            }

            String_map archives(String_map::key_compare(), get_allocator());

            // collect all archives first for the KILL test
            for (char const *entry = dir.read(); entry != NULL; entry = dir.read()) {
                string e(entry, m_alloc);
                size_t l = e.size();

                if (l < 5)
                    continue;
                if (e[l - 4] != '.' || e[l - 3] != 'm' || e[l - 2] != 'd' || e[l - 1] != 'r')
                    continue;

                // remove .mdr
                archives.insert(String_map::value_type(e.substr(0, l - 4), true));
            }

            // search for archives
            for (String_map::const_iterator it(archives.begin()), end(archives.end());
                 it != end;
                ++it)
            {
                string e = it->first;

                // append the extension again
                e.append(".mdr");

                if (!is_archive_prefix(e, archive_path)) {
                    continue;
                }

                string mdr_path = join_path(string(path, m_alloc), e);
                if (udim_mode == NO_UDIM) {
                    // found a possibly matching archive, check if it contains the searched file
                    if (archive_contains(mdr_path.c_str(), file_name_fs.c_str())) {
                        if (!is_killed(path, e, archives)) {
                            // use ':' to separate archive name from the content
                            string joined_file_name = mdr_path + ':' + file_mask;

                            // get one
                            places.push_back(convert_slashes_to_os_separators(joined_file_name));
                            ++n_places;
                        }
                    }
                } else {
                    // found a possibly matching archive, check if it contains the searched mask

                    // names inside archive using only '/' as a separator
                    string mask_fs = convert_os_separators_to_slashes(string(file_mask, m_alloc));

                    if (archive_contains_mask(mdr_path.c_str(), file_mask)) {
                        if (!is_killed(path, e, archives)) {
                            // use ':' to separate archive name from the content
                            string joined_file_mask = mdr_path + ':' + file_mask;

                            // get one
                            places.push_back(convert_slashes_to_os_separators(joined_file_mask));
                            ++n_places;
                        }
                    }
                }
            }
            dir.close();
        }

        // no archives

        if (udim_mode == NO_UDIM) {
            string joined_file_name = join_path(string(path, m_alloc), string(file_mask, m_alloc));
            if (!is_killed(file_mask)) {
                if (is_file_utf8(m_alloc, joined_file_name.c_str())) {
                    places.push_back(convert_slashes_to_os_separators(joined_file_name));
                    ++n_places;
                }
            }
        } else {
            if (!is_killed(file_mask)) {
                if (has_file_utf8(m_alloc, path, file_mask)) {
                    string joined_file_mask = join_path(
                        string(path, m_alloc), string(file_mask, m_alloc));

                    places.push_back(convert_slashes_to_os_separators(joined_file_mask));
                    ++n_places;
                }
            }
        }

        if (n_places == 1) {
            return places.front();
        } else if (n_places > 1) {
            error(
                AMBIGUOUS_FILE_LOCATION,
                *m_pos,
                Error_params(m_alloc)
                .add_entity_kind(
                    m_resolving_resource ? Error_params::EK_RESOURCE : Error_params::EK_MODULE)
                .add(m_resolve_entity)
            );

            while (!places.empty()) {
                string place = places.front();

                size_t pos = place.find(".mdr:");
                if (pos != string::npos) {
                    place = place.substr(0, pos + 4);
                }

                add_note(
                    LOCATED_AT,
                    *m_pos,
                    Error_params(m_alloc).add(place.c_str()));

                places.pop_front();
                --n_places;
            }
        }
    }

    return string(m_alloc);
}

// Check if the given file name (UTF8 encoded) names a file on the file system or inside
// an archive.
bool File_resolver::file_exists(
    char const *fname) const
{
    char const *p = strstr(fname, ".mdr:");
    if (p == NULL) {
        // not inside an archive
        string dname(m_alloc);

        char const *p = strrchr(fname, os_separator());
        if (p != NULL) {
            dname = string(fname, p - fname, m_alloc);
            fname = p + 1;
        }

        return has_file_utf8(m_alloc, dname.c_str(), fname);
    }
    string archive_name(fname, p + 4, m_alloc);
    char const *a_fname = p + 5;

    Archiv_error_code err;
    if (MDL_archive *archiv = MDL_archive::open(m_alloc, archive_name.c_str(), err, false)) {
        bool res = archiv->contains_mask(a_fname);
        archiv->close();
        return res;
    }
    return false;
}

// Resolve a MDL file name.
string File_resolver::resolve_filename(
    string         &abs_file_name,
    char const     *file_path,
    bool           is_resource,
    char const     *module_file_system_path,
    char const     *module_name,
    Position const *pos,
    UDIM_mode      &udim_mode)
{
    abs_file_name.clear();

    Position_store store(m_pos, pos);

    string url(file_path, m_alloc);

    size_t nesting_level = module_name != NULL ? get_module_nesting_level(module_name) : 0;

    bool owner_is_string_module = module_file_system_path == NULL;
    string module_file_system_path_str(
        owner_is_string_module ? "" : module_file_system_path, m_alloc);

    // Step 0: compute terms defined in MDL spec
    string directory_path(m_alloc);
    string file_name(m_alloc);
    split_file_path( url, directory_path, file_name);

    string current_working_directory(m_alloc);
    bool cwd_is_archive = false;
    string current_search_path(m_alloc);
    bool csp_is_archive = false;
    string current_module_path(m_alloc);
    split_module_file_system_path(
        module_file_system_path_str,
        string(module_name != NULL ? module_name : "", m_alloc),
        nesting_level,
        current_working_directory,
        cwd_is_archive,
        current_search_path,
        csp_is_archive,
        current_module_path);

    // handle UDIM
    string url_mask(m_alloc);
    if (is_resource) {
        // check if one of the supported markers are inside the resource path, if yes compute
        // its mode and mask

        char const *p;
        char const *file_name = url.c_str();

        if ((p = strstr(file_name, UDIM_MARI_MARKER)) != NULL) {
            udim_mode = UM_MARI;
            url_mask.append(file_name, p - file_name);
            url_mask.append("[0-9][0-9][0-9][0-9]");
            url_mask.append(p + strlen(UDIM_MARI_MARKER));
        } else if ((p = strstr(file_name, UDIM_ZBRUSH_MARKER)) != NULL) {
            udim_mode = UM_ZBRUSH;
            url_mask.append(file_name, p - file_name);
            url_mask.append("_u-?[0-9]+_v-?[0-9]+");
            url_mask.append(p + strlen(UDIM_ZBRUSH_MARKER));
        } else if ((p = strstr(file_name, UDIM_MUDBOX_MARKER)) != NULL) {
            udim_mode = UM_MUDBOX;
            url_mask.append(file_name, p - file_name);
            url_mask.append("_u-?[0-9]+_v-?[0-9]+");
            url_mask.append(p + strlen(UDIM_MUDBOX_MARKER));
        } else {
            udim_mode = NO_UDIM;
            url_mask = url;
        }
    } else {
        udim_mode = NO_UDIM;
        url_mask = url;
    }

    // Step 1: normalize file path
    string canonical_file_path = normalize_file_path(
        url,
        url_mask,
        directory_path,
        file_name,
        nesting_level,
        module_file_system_path_str,
        current_working_directory,
        cwd_is_archive,
        current_module_path);
    if (canonical_file_path.empty()) {
        if (is_resource) {
            string module_for_error_msg(m_alloc);

            if (m_pos->get_start_line() == 0) {
                if (!module_file_system_path_str.empty())
                    module_for_error_msg = module_file_system_path_str;
                else if (module_name != NULL)
                    module_for_error_msg = module_name;
            }

            error(
                UNABLE_TO_RESOLVE,
                *m_pos,
                Error_params(m_alloc)
                .add(file_path)
                .add_module_origin(module_for_error_msg.c_str()));
        }

        return string(m_alloc);
    }

    string canonical_file_mask(m_alloc);
    {
        // check if one of the supported markers are inside the resource path, if yes compute
        // its mode and mask

        char const *p;
        char const *file_name = canonical_file_path.c_str();

        switch (udim_mode) {
        case UM_MARI:
            p = strstr(file_name, UDIM_MARI_MARKER);
            canonical_file_mask.append(file_name, p - file_name);
            canonical_file_mask.append("[0-9][0-9][0-9][0-9]");
            canonical_file_mask.append(p + strlen(UDIM_MARI_MARKER));
            break;
        case UM_ZBRUSH:
            p = strstr(file_name, UDIM_ZBRUSH_MARKER);
            canonical_file_mask.append(file_name, p - file_name);
            canonical_file_mask.append("_u-?[0-9]+_v-?[0-9]+");
            canonical_file_mask.append(p + strlen(UDIM_ZBRUSH_MARKER));
            break;
        case UM_MUDBOX:
            p = strstr(file_name, UDIM_MUDBOX_MARKER);
            canonical_file_mask.append(file_name, p - file_name);
            canonical_file_mask.append("_u-?[0-9]+_v-?[0-9]+");
            canonical_file_mask.append(p + strlen(UDIM_MUDBOX_MARKER));
            break;
        case NO_UDIM:
            canonical_file_mask = canonical_file_path;
            break;
        }
    }

    // Step 2: consider search paths
    string resolved_file_system_location = consider_search_paths(
        canonical_file_mask, is_resource, file_path, udim_mode);
    if (resolved_file_system_location.empty()) {
        if (is_resource) {
            string module_for_error_msg(m_alloc);

            if (m_pos->get_start_line() == 0) {
                if (!module_file_system_path_str.empty())
                    module_for_error_msg = module_file_system_path_str;
                else if (module_name != NULL)
                    module_for_error_msg = module_name;
            }

            error(
                UNABLE_TO_RESOLVE,
                *m_pos,
                Error_params(m_alloc)
                    .add(file_path)
                    .add_module_origin(module_for_error_msg.c_str()));
        }
        return string(m_alloc);
    }

    // Step 3: consistency checks
    if (!check_consistency(
        resolved_file_system_location,
        canonical_file_path,
        url,
        current_working_directory,
        current_search_path,
        is_resource,
        csp_is_archive,
        owner_is_string_module))
        return string(m_alloc);

    abs_file_name = resolved_file_system_location;
    return canonical_file_path;
}

// The resolver resolves a module.
void File_resolver::mark_module_search(char const *module_name)
{
    m_resolve_entity     = module_name;
    m_resolving_resource = false;
}

// The resolver resolves a resource.
void File_resolver::mark_resource_search(char const *file_name)
{
    m_resolve_entity     = file_name;
    m_resolving_resource = true;
}

// Map a file error.
void File_resolver::handle_file_error(Archiv_error_code err)
{
    // FIXME
    switch (err) {
    case EC_OK:
        return;
    case EC_ARC_NOT_EXIST:
        return;
    case EC_ARC_OPEN_FAILED:
        return;
    case EC_FILE_OPEN_FAILED:
        return;
    case EC_INVALID_ARCHIVE:
        return;
    case EC_NOT_FOUND:
        return;
    case EC_IO_ERROR:
        return;
    case EC_CRC_ERROR:
        return;
    case EC_INVALID_PASSWORD:
        return;
    case EC_MEMORY_ALLOCATION:
        return;
    case EC_RENAME_ERROR:
        return;
    case EC_INTERNAL_ERROR:
        return;
    }
}


// Resolve an import (module) name to the corresponding absolute module name.
string File_resolver::resolve_import(
    Position const         &pos,
    char const             *import_name,
    char const             *owner_name,
    char const             *owner_filename)
{
    mark_module_search(import_name);

    string import_file(to_url(import_name) + ".mdl");

    if (owner_name != NULL && owner_name[0] == '\0')
        owner_name = NULL;
    if (owner_filename != NULL && owner_filename[0] == '\0')
        owner_filename = NULL;

    UDIM_mode udim_mode = NO_UDIM;
    string os_file_name(m_alloc);
    string canonical_file_name = resolve_filename(
        os_file_name,
        import_file.c_str(),
        /*is_resource=*/false,
        owner_filename,
        owner_name,
        &pos,
        udim_mode);
    MDL_ASSERT(udim_mode == NO_UDIM && "only resources should return a file mask");
    if (canonical_file_name.empty()) {
        return string(m_alloc);
    }

    string absolute_name = to_module_name(canonical_file_name.c_str());
    MDL_ASSERT(absolute_name.substr(absolute_name.size() - 4) == ".mdl");
    absolute_name = absolute_name.substr(0, absolute_name.size() - 4);

    return absolute_name;
}

// Resolve a resource (file) name to the corresponding absolute file path.
string File_resolver::resolve_resource(
    string         &abs_file_name,
    Position const &pos,
    char const     *import_file,
    char const     *owner_name,
    char const     *owner_filename,
    UDIM_mode      &udim_mode)
{
    mark_resource_search(import_file);

    if (owner_name != NULL && owner_name[0] == '\0')
        owner_name = NULL;
    if (owner_filename != NULL && owner_filename[0] == '\0')
        owner_filename = NULL;

    // Make owner file system path absolute
    string owner_filename_str(m_alloc);
    if (owner_filename == NULL)
        owner_filename_str = "";
    else {
        MDL_ASSERT(is_path_absolute(owner_filename));
        owner_filename_str = owner_filename;
    }

    udim_mode = NO_UDIM;

    string os_file_name(m_alloc);
    abs_file_name = resolve_filename(
        os_file_name, import_file, /*is_resource=*/true,
        owner_filename_str.c_str(), owner_name, &pos, udim_mode);
    return os_file_name;
}


// Checks whether the given module source exists.
IInput_stream *File_resolver::open(
    char const *module_name)
{
    MDL_ASSERT(module_name != NULL && module_name[0] == ':' && module_name[1] == ':');

    string canonical_file_path = to_url(module_name);
    canonical_file_path += ".mdl";

    UDIM_mode udim_mode = NO_UDIM;
    string resolved_file_path = consider_search_paths(
        canonical_file_path, /*is_resource=*/false, module_name, udim_mode);
    MDL_ASSERT(udim_mode == NO_UDIM && "resolved modules should not be file masks");
    if (resolved_file_path.empty()) {
        MDL_ASSERT(!"open called on nonexisting module");
        return NULL;
    }

    Archiv_error_code err;
    File_handle *file = File_handle::open(m_alloc, resolved_file_path.c_str(), err);
    if (file == NULL) {
        handle_file_error(err);
        return NULL;
    }

    Allocator_builder builder(m_alloc);

    if (file->is_archive()) {
        mi::base::Handle<Manifest const> manifest(file->get_manifest());
        return builder.create<Archive_input_stream>(
            m_alloc, file, resolved_file_path.c_str(), manifest.get());
    } else {
        return builder.create<Simple_input_stream>(
            m_alloc, file, resolved_file_path.c_str());
    }
}

/// Checks whether the given module source exists.
bool File_resolver::exists(
    char const *module_name)
{
    MDL_ASSERT(module_name != NULL && module_name[0] == ':' && module_name[1] == ':');

    string canonical_file_path(to_url(module_name));
    canonical_file_path += ".mdl";

    UDIM_mode udim_mode = NO_UDIM;
    string resolved_file_path = consider_search_paths(
        canonical_file_path, /*is_resource=*/false, module_name, udim_mode);
    return !resolved_file_path.empty();
}

// --------------------------------------------------------------------------

/// Implementation of a resource reader from a file.
class Archive_resource_reader : public Allocator_interface_implement<IMDL_resource_reader>
{
    typedef Allocator_interface_implement<IMDL_resource_reader> Base;
public:
    /// Read a memory block from the resource.
    ///
    /// \param ptr   Pointer to a block of memory with a size of size bytes
    /// \param size  Number of bytes to read
    ///
    /// \returns    The total number of bytes successfully read.
    Uint64 read(void *ptr, Uint64 size) MDL_FINAL
    {
        return m_file->get_archive_file()->read(ptr, size);
    }

    /// Get the current position.
    Uint64 tell() MDL_FINAL
    {
        return m_file->get_archive_file()->tell();
    }

    /// Reposition stream position indicator.
    ///
    /// \param offset  Number of bytes to offset from origin
    /// \param origin  Position used as reference for the offset
    ///
    /// \return true on success
    bool seek(Sint64 offset, Position origin) MDL_FINAL
    {
        return m_file->get_archive_file()->seek(offset, origin) != 0;
    }

    /// Get the UTF8 encoded name of the resource on which this reader operates.
    /// \returns    The name of the resource or NULL.
    char const *get_filename() const MDL_FINAL
    {
        return m_file_name.empty() ? NULL : m_file_name.c_str();
    }

    /// Get the UTF8 encoded absolute MDL resource name on which this reader operates.
    /// \returns    The absolute MDL url of the resource or NULL.
    char const *get_mdl_url() const MDL_FINAL
    {
        return m_mdl_url.empty() ? NULL : m_mdl_url.c_str();
    }

    /// Constructor.
    ///
    /// \param alloc             the allocator
    /// \param f                 the file handle
    /// \param filename          the file name
    /// \param mdl_url           the MDL url
    explicit Archive_resource_reader(
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

private:
    // non copyable
    Archive_resource_reader(Archive_resource_reader const &) MDL_DELETED_FUNCTION;
    Archive_resource_reader &operator=(Archive_resource_reader const &) MDL_DELETED_FUNCTION;

private:
    ~Archive_resource_reader() MDL_FINAL
    {
        File_handle::close(m_file);
    }

private:
    /// The File handle.
    File_handle *m_file;

    /// The filename.
    string m_file_name;

    /// The absolute MDL url.
    string m_mdl_url;
};

/// Implementation of a resource reader from a file.
class Buffered_archive_resource_reader : public Allocator_interface_implement<IMDL_resource_reader>
{
    typedef Allocator_interface_implement<IMDL_resource_reader> Base;
public:
    /// Read a memory block from the resource.
    ///
    /// \param ptr   Pointer to a block of memory with a size of size bytes
    /// \param size  Number of bytes to read
    ///
    /// \returns    The total number of bytes successfully read.
    Uint64 read(void *ptr, Uint64 size) MDL_FINAL
    {
        // first, empty the buffer
        size_t prefix_size = m_buf_size - m_curr_pos;
        if (prefix_size > 0) {
            // buffer is not empty
            if (size_t(size) < prefix_size) {
                prefix_size = size_t(size);
            }
            memcpy(ptr, m_buffer + m_curr_pos, prefix_size);
            m_curr_pos += prefix_size;
        }
        size -= prefix_size;

        if (size == 0)
            return prefix_size;

        // can we read a big chunk?
        if (size > sizeof(m_buffer)) {
            return m_file->get_archive_file()->read((char *)ptr + prefix_size, size) + prefix_size;
        }

        // fill the buffer
        m_curr_pos = 0;
        zip_int64_t read_bytes = m_file->get_archive_file()->read(m_buffer, sizeof(m_buffer));
        if (read_bytes < 0) {
            return prefix_size > 0 ? prefix_size : read_bytes;
        }
        m_buf_size = size_t(read_bytes);

        size_t suffix_size = m_buf_size - m_curr_pos;
        if (suffix_size > 0) {
            // buffer is not empty
            if (size_t(size) < suffix_size) {
                suffix_size = size_t(size);
            }
            memcpy((char *)ptr + prefix_size, m_buffer, suffix_size);
            m_curr_pos += suffix_size;
        }
        return prefix_size + suffix_size;
    }

    /// Get the current position.
    Uint64 tell() MDL_FINAL
    {
        return m_file->get_archive_file()->tell()- (m_buf_size - m_curr_pos);
    }

    /// Reposition stream position indicator.
    ///
    /// \param offset  Number of bytes to offset from origin
    /// \param origin  Position used as reference for the offset
    ///
    /// \return true on success
    bool seek(Sint64 offset, Position origin) MDL_FINAL
    {
        // drop buffer
        m_curr_pos = m_buf_size = 0;

        // now seek
        return m_file->get_archive_file()->seek(offset, origin) != 0;
    }

    /// Get the UTF8 encoded name of the resource on which this reader operates.
    /// \returns    The name of the resource or NULL.
    char const *get_filename() const MDL_FINAL
    {
        return m_file_name.empty() ? NULL : m_file_name.c_str();
    }

    /// Get the UTF8 encoded absolute MDL resource name on which this reader operates.
    /// \returns    The absolute MDL url of the resource or NULL.
    char const *get_mdl_url() const MDL_FINAL
    {
        return m_mdl_url.empty() ? NULL : m_mdl_url.c_str();
    }

    /// Constructor.
    ///
    /// \param alloc             the allocator
    /// \param f                 the file handle
    /// \param filename          the file name
    /// \param mdl_url           the MDL url
    explicit Buffered_archive_resource_reader(
        IAllocator  *alloc,
        File_handle *f,
        char const  *filename,
        char const  *mdl_url)
    : Base(alloc)
    , m_file(f)
    , m_file_name(filename, alloc)
    , m_mdl_url(mdl_url, alloc)
    , m_curr_pos(0u)
    , m_buf_size(0u)
    {
    }

private:
    // non copyable
    Buffered_archive_resource_reader(
        Buffered_archive_resource_reader const &) MDL_DELETED_FUNCTION;
    Buffered_archive_resource_reader &operator=(
        Buffered_archive_resource_reader const &) MDL_DELETED_FUNCTION;

private:
    ~Buffered_archive_resource_reader() MDL_FINAL
    {
        File_handle::close(m_file);
    }

private:
    /// Buffer for buffered input.
    unsigned char m_buffer[1024];

    /// The File handle.
    File_handle *m_file;

    /// The filename.
    string m_file_name;

    /// The absolute MDL url.
    string m_mdl_url;

    /// Current position inside the buffer if used.
    size_t m_curr_pos;

    /// Size of the current buffer if used.
    size_t m_buf_size;
};

// --------------------------------------------------------------------------

// Constructor from one file name/url pair (typical case).
MDL_resource_set::MDL_resource_set(
    IAllocator *alloc,
    char const *url,
    char const *filename)
: Base(alloc)
, m_arena(alloc)
, m_entries(
    1,
    Resource_entry(Arena_strdup(m_arena, url), Arena_strdup(m_arena, filename), 0, 0),
    &m_arena)
, m_is_udim_set(false)
{
}

// Empty Constructor.
MDL_resource_set::MDL_resource_set(
    IAllocator *alloc)
: Base(alloc)
, m_arena(alloc)
, m_entries(&m_arena)
, m_is_udim_set(true)
{
}

// Create a resource set from a file mask.
MDL_resource_set *MDL_resource_set::from_mask(
    IAllocator               *alloc,
    char const               *url,
    char const               *file_mask,
    File_resolver::UDIM_mode udim_mode)
{
    char const *p = strstr(file_mask, ".mdr:");
    if (p == NULL) {
        return from_mask_file(alloc, url, file_mask, udim_mode);
    }

    string arc_name(file_mask, p + 4, alloc);
    p += 5;

    return from_mask_archive(alloc, url, arc_name.c_str(), p, udim_mode);
}

// Create a resource set from a file mask describing files on disk.
MDL_resource_set *MDL_resource_set::from_mask_file(
    IAllocator               *alloc,
    char const               *url,
    char const               *file_mask,
    File_resolver::UDIM_mode udim_mode)
{
    Directory dir(alloc);
    string dname(alloc);

    char sep = os_separator();
    char const *p = strrchr(file_mask, sep);

    // p should never be NULL here, because the mask is absolute, but handle it gracefully if not
    if (p != NULL) {
        dname = string(file_mask, p - file_mask, alloc);

        if (!dir.open(dname.c_str())) {
            // directory does not exists
            return NULL;
        }
        file_mask = p + 1;
    } else {
        dname = ".";
        if (!dir.open(".")) {
            // directory does not exists
            return NULL;
        }
    }

    p = NULL;

    char const *q = NULL;

    switch (udim_mode) {
    case File_resolver::NO_UDIM:
        MDL_ASSERT(!"UDIM mode not set");
        return NULL;
    case File_resolver::UM_MARI:
        // UDIM (Mari), expands to four digits calculated as 1000+(u+1+v*10)
        p = strstr(file_mask, "[0-9][0-9][0-9][0-9]");
        q = strstr(url, UDIM_MARI_MARKER);
        break;
    case File_resolver::UM_ZBRUSH:
        // 0-based (Zbrush), expands to "_u0_v0" for the first tile
        p = strstr(file_mask, "_u-?[0-9]+_v-?[0-9]+");
        q = strstr(url, UDIM_ZBRUSH_MARKER);
        break;
    case File_resolver::UM_MUDBOX:
        // 1-based (Mudbox), expands to "_u1_v1" for the first tile
        p = strstr(file_mask, "_u-?[0-9]+_v-?[0-9]+");
        q = strstr(url, UDIM_MUDBOX_MARKER);
        break;
    }

    if (p == NULL) {
        MDL_ASSERT(!"Could not find udim regexp");
        return NULL;
    }

    size_t ofs = p - file_mask;

    Allocator_builder builder(alloc);

    MDL_resource_set *s = builder.create<MDL_resource_set>(alloc);

    for (char const *entry = dir.read(); entry != NULL; entry = dir.read()) {
        if (utf8_match(file_mask, entry)) {
            string purl(url, alloc);

            if (q != NULL) {
                // also patch the URL if possible
                purl = string(url, q - url, alloc);
                purl += entry + ofs;
            }

            parse_u_v(s, entry, ofs, purl.c_str(), dname, sep, udim_mode);
        }
    }
    return s;
}

// Create a resource set from a file mask describing files on an archive.
MDL_resource_set *MDL_resource_set::from_mask_archive(
    IAllocator               *alloc,
    char const               *url,
    char const               *arc_name,
    char const               *file_mask,
    File_resolver::UDIM_mode udim_mode)
{
    Archiv_error_code err;
    if (MDL_archive *archiv = MDL_archive::open(alloc, arc_name, err)) {
        char const *p = NULL;
        char const *q = NULL;

        switch (udim_mode) {
        case File_resolver::NO_UDIM:
            MDL_ASSERT(!"UDIM mode not set");
            return NULL;
        case File_resolver::UM_MARI:
            // UDIM (Mari), expands to four digits calculated as 1000+(u+1+v*10)
            p = strstr(file_mask, "[0-9][0-9][0-9][0-9]");
            q = strstr(url, UDIM_MARI_MARKER);
            break;
        case File_resolver::UM_ZBRUSH:
            // 0-based (Zbrush), expands to "_u0_v0" for the first tile
            p = strstr(file_mask, "_u-?[0-9]+_v-?[0-9]+");
            q = strstr(url, UDIM_ZBRUSH_MARKER);
            break;
        case File_resolver::UM_MUDBOX:
            // 1-based (Mudbox), expands to "_u1_v1" for the first tile
            p = strstr(file_mask, "_u-?[0-9]+_v-?[0-9]+");
            q = strstr(url, UDIM_MUDBOX_MARKER);
            break;
        }

        if (p == NULL) {
            MDL_ASSERT(!"Could not find udim regexp");
            return NULL;
        }

        size_t ofs = p - file_mask;

        Allocator_builder builder(alloc);

        MDL_resource_set *s = builder.create<MDL_resource_set>(alloc);

        // ZIP uses '/'
        string forward(file_mask, alloc);
        forward = convert_os_separators_to_slashes(forward);

        string archive_name(arc_name, alloc);

        for (int i = 0, n = archiv->get_num_entries(); i < n; ++i) {
            char const *file_name = archiv->get_entry_name(i);

            if (utf8_match(forward.c_str(), file_name)) {
                string purl(url, alloc);

                if (q != NULL) {
                    // also patch the URL if possible
                    purl = string(url, q - url, alloc);
                    purl += file_name + ofs;
                }

                // convert back to OS notation
                string fname(file_name, alloc);
                fname = convert_slashes_to_os_separators(fname);

                parse_u_v(s, fname.c_str(), ofs, purl.c_str(), archive_name, ':', udim_mode);
            }
        }

        archiv->close();
        return s;
    }
    return NULL;
}

// Parse a file name and enter it into a resource set.
void MDL_resource_set::parse_u_v(
    MDL_resource_set         *s,
    char const               *name,
    size_t                   ofs,
    char const               *url,
    string const             &prefix,
    char                     sep,
    File_resolver::UDIM_mode udim_mode)
{
    int u = 0, v = 0, sign = 1;
    switch (udim_mode) {
    case File_resolver::NO_UDIM:
        break;
    case File_resolver::UM_MARI:
        // UDIM (Mari), expands to four digits calculated as 1000+(u+1+v*10)
        {
            char const *n = name + ofs;
            unsigned num =
                1000 * (n[0] - '0') +
                100 * (n[1] - '0') +
                10 * (n[2] - '0') +
                1 * (n[3] - '0') - 1001;

            // assume u, v [0..9]
            u = num % 10;
            v = num / 10;

            s->m_entries.push_back(
                Resource_entry(
                    Arena_strdup(s->m_arena, url),
                    Arena_strdup(s->m_arena, (prefix + sep + name).c_str()),
                    u,
                    v
                )
            );
        }
        break;
    case File_resolver::UM_ZBRUSH:
        // 0-based (Zbrush), expands to "_u0_v0" for the first tile
    case File_resolver::UM_MUDBOX:
        // 1-based (Mudbox), expands to "_u1_v1" for the first tile
        {
            char const *n = name + ofs + 2;

            if (*n == '-') {
                sign = -1;
                ++n;
            } else {
                sign = +1;
            }
            while (isdigit(*n)) {
                u = u * 10 + *n - '0';
                ++n;
            }
            u *= sign;

            if (*n == '_')
                ++n;
            if (*n == 'v')
                ++n;

            if (*n == '-') {
                sign = -1;
                ++n;
            } else {
                sign = +1;
            }
            while (isdigit(*n)) {
                v = v * 10 + *n - '0';
                ++n;
            }
            v *= sign;

            if (udim_mode == File_resolver::UM_MUDBOX) {
                u -= 1;
                v -= 1;
            }

            s->m_entries.push_back(
                Resource_entry(
                    Arena_strdup(s->m_arena, url),
                    Arena_strdup(s->m_arena, (prefix + sep + name).c_str()),
                    u,
                    v
                )
            );
        }
        break;
    }
}

// Get the number of resolved file names.
size_t MDL_resource_set::get_count() const
{
    return m_entries.size();
}

// Get the i'th file name of the ordered set.
char const *MDL_resource_set::get_filename(size_t i) const
{
    if (i < m_entries.size())
        return m_entries[i].filename;
    return NULL;
}

// Get the i'th MDL url of the ordered set.
char const *MDL_resource_set::get_mdl_url(size_t i) const
{
    if (i < m_entries.size())
        return m_entries[i].url;
    return NULL;
}

// If the ordered set represents an UDIM mapping, returns it, otherwise NULL.
bool MDL_resource_set::get_udim_mapping(size_t i, int &u, int &v) const
{
    u = v = 0;
    if (m_is_udim_set) {
        if (i < m_entries.size()) {
            u = m_entries[i].u;
            v = m_entries[i].v;
            return true;
        }
    }
    return false;
}

// Opens a reader for the i'th entry.
IMDL_resource_reader *MDL_resource_set::open_reader(size_t i) const
{
    if (i < m_entries.size()) {
        Archiv_error_code err;
        IMDL_resource_reader *res = open_resource_file(
            get_allocator(),
            m_entries[i].url,
            m_entries[i].filename,
            err);

        // FIXME: handle error?
        return res;
    }
    return NULL;
}

// --------------------------------------------------------------------------

// Constructor.
Entity_resolver::Entity_resolver(
    IAllocator *alloc,
    MDL const                                *compiler,
    IModule_cache                            *module_cache,
    mi::base::Handle<IMDL_search_path> const &search_path)
: Base(alloc)
, m_msg_list(alloc, /*owner_fname=*/"")
, m_resolver(
    *compiler,
    module_cache,
    search_path,
    compiler->get_search_path_lock(),
    m_msg_list,
    /*front_path=*/NULL)
, m_search_path(search_path)
, m_tmp(alloc)
{
}

// Resolve a resource file name.
MDL_resource_set *Entity_resolver::resolve_resource_file_name(
    char const     *file_path,
    char const     *owner_file_path,
    char const     *owner_name,
    Position const *pos)
{
    Position_impl zero(0, 0, 0, 0);

    if (pos == NULL)
        pos = &zero;

    m_msg_list.clear();

    File_resolver::UDIM_mode udim_mode = File_resolver::NO_UDIM;

    string abs_file_name(get_allocator());
    string resolved_file_path = m_resolver.resolve_resource(
        abs_file_name,
        *pos,
        file_path,
        owner_name,
        owner_file_path,
        udim_mode);

    if (resolved_file_path.empty())
        return NULL;

    if (udim_mode != File_resolver::NO_UDIM) {
        // lookup ALL files for the given mask
        return MDL_resource_set::from_mask(
            get_allocator(),
            abs_file_name.c_str(),
            resolved_file_path.c_str(),
            udim_mode);
    } else {
        // single return
        Allocator_builder builder(get_allocator());

        return builder.create<MDL_resource_set>(
            get_allocator(),
            abs_file_name.c_str(),
            resolved_file_path.c_str());
    }
}

// Opens a resource.
IMDL_resource_reader *Entity_resolver::open_resource(
    char const     *file_path,
    char const     *owner_file_path,
    char const     *owner_name,
    Position const *pos)
{
    Position_impl zero(0, 0, 0, 0);

    if (pos == NULL)
        pos = &zero;

    m_msg_list.clear();

    File_resolver::UDIM_mode udim_mode = File_resolver::NO_UDIM;

    string abs_file_name(get_allocator());
    string resolved_file_path = m_resolver.resolve_resource(
        abs_file_name,
        *pos,
        file_path,
        owner_name,
        owner_file_path,
        udim_mode);

    MDL_ASSERT(
        udim_mode == File_resolver::NO_UDIM &&
        "open_resource() cannot be used to open a UDIM mask");

    if (udim_mode != File_resolver::NO_UDIM || resolved_file_path.empty())
        return NULL;

    Archiv_error_code err;
    IMDL_resource_reader *res = open_resource_file(
        get_allocator(),
        abs_file_name.c_str(),
        resolved_file_path.c_str(),
        err);
    // FIXME: handle error?
    return res;
}

// Resolve a module name.
char const *Entity_resolver::resolve_module_name(
    char const *name)
{
    Position_impl zero(0, 0, 0, 0);

    m_msg_list.clear();

    m_tmp = m_resolver.resolve_import(zero, name, /*owner_name=*/NULL, /*owner_filename=*/NULL);
    return m_tmp.c_str();
}

// Access messages of last resolver operation.
Messages const &Entity_resolver::access_messages() const
{
    return m_msg_list;
}

// Open a resource file read-only.
IMDL_resource_reader *open_resource_file(
    IAllocator        *alloc,
    const char        *abs_mdl_path,
    char const        *resource_path,
    Archiv_error_code &err)
{
    File_handle *file = File_handle::open(alloc, resource_path, err);
    if (file == NULL) {
        return NULL;
    }

    Allocator_builder builder(alloc);

    if (file->is_archive()) {
        return builder.create<Archive_resource_reader>(
            alloc, file, resource_path, abs_mdl_path);
    } else {
        return builder.create<File_resource_reader>(
            alloc, file, resource_path, abs_mdl_path);
    }
}

// Get the manifest.
Manifest *MDL_archive::parse_manifest()
{
    if (MDL_archive_file *fp = file_open("MANIFEST")) {
        Allocator_builder builder(m_alloc);

        File_handle *manifest_fp =
            builder.create<File_handle>(get_allocator(), this, /*owns_archive=*/false, fp);

        mi::base::Handle<Buffered_archive_resource_reader> reader(
            builder.create<Buffered_archive_resource_reader>(
            m_alloc,
            manifest_fp,
            "MANIFEST",
            /*mdl_url=*/""));

        Manifest *manifest = Archive_tool::parse_manifest(m_alloc, reader.get());
        if (manifest != NULL) {
            string arc_name(get_archive_name(), m_alloc);
            size_t pos = arc_name.rfind(os_separator());
            if (pos == string::npos)
                pos = 0;
            else
                pos += 1;
            size_t e = arc_name.length() - 4; // skip ".mdr"
            arc_name = arc_name.substr(pos, e - pos);
            manifest->set_archive_name(arc_name.c_str());
        }
        return manifest;
    }
    return NULL;
}

}  // mdl
}  // mi
