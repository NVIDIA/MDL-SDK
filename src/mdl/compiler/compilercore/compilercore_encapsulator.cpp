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
#include "compilercore_encapsulator.h"

#include "compilercore_allocator.h"
#include "compilercore_file_resolution.h"
#include "compilercore_file_utils.h"
#include "compilercore_mdl.h"
#include "compilercore_messages.h"
#include "compilercore_options.h"
#include "compilercore_streams.h"
#include "compilercore_zip_utils.h"
#include "compilercore_hash.h"

#include <mi/mdl/mdl_entity_resolver.h>

namespace mi {
namespace mdl {

namespace{

    static const MDL_zip_container_header header_supported_read_version = MDL_zip_container_header(
        "MDLE", 4,  // prefix
        0, 1,       // major.minor version min
        1, 0);      // major.minor version max

    static const MDL_zip_container_header header_write_version = MDL_zip_container_header(
        "MDLE", 4,  // prefix
        1,          // major version
        0);         // minor version
} // anonymous

// Open a container file.
MDL_zip_container_mdle *MDL_zip_container_mdle::open(
    IAllocator                   *alloc,
    char const                   *path,
    MDL_zip_container_error_code &err)
{
    MDL_zip_container_header header_info = header_supported_read_version;
    zip_t* za = MDL_zip_container::open(alloc, path, err, header_info);

    // load a pre-released version (0.2) will probably get an error at some point in time
    if (err != EC_OK && err != EC_PRE_RELEASE_VERSION)
        return NULL;

    Allocator_builder builder(alloc);
    MDL_zip_container_mdle *mdle = builder.create<MDL_zip_container_mdle>(alloc, path, za);
    if (mdle != NULL)
        mdle->m_header = header_info;
    return mdle;
}

// Destructor
MDL_zip_container_mdle::~MDL_zip_container_mdle()
{
}

// Constructor.
MDL_zip_container_mdle::MDL_zip_container_mdle(
    IAllocator *alloc,
    char const *path,
    zip_t      *za)
: MDL_zip_container(alloc, path, za, /*supports_resource_hashes=*/true)
{
}

// Get the stored top level MD5 hash that allows to compare MDLE files without iterating
// over the entire content.
bool MDL_zip_container_mdle::get_hash(unsigned char hash[16]) const
{
    if (m_header.major_version_min == 0 && m_header.minor_version_min == 1)
        return false;

    memcpy(hash, m_header.hash, 16);
    return true;
}

//-------------------------------------------------------------------------------------------------

// Constructor.
Encapsulate_tool::Encapsulate_tool(
    IAllocator *alloc,
    MDL        *compiler)
: Base(alloc)
, m_compiler(compiler)
, m_msg_list(alloc, /*owner_fname=*/"")
, m_last_msg_idx(0)
, m_options(alloc)
{
    m_options.add_option(
        MDL_ENCAPS_OPTION_OVERWRITE,
        "false",
        "Overwrite existing files");
}


// Translate zip errors.
void Encapsulate_tool::translate_zip_error(char const *mdle_name, int zip_error)
{
    switch (zip_error) {
    case ZIP_ER_MULTIDISK:
        /* N Multi-disk zip archives not supported */
        error(MDLE_INVALID_MDLE, Error_params(get_allocator()).add(mdle_name));
        break;

    case ZIP_ER_RENAME:
        /* S Renaming temporary file failed */
        error(MDLE_RENAME_FAILED, Error_params(get_allocator()));
        break;

    case ZIP_ER_CLOSE:
        /* S Closing zip archive failed */
    case ZIP_ER_SEEK:
        /* S Seek error */
    case ZIP_ER_READ:
        /* S Read error */
    case ZIP_ER_WRITE:
        /* S Write error */
        error(MDLE_IO_ERROR, Error_params(get_allocator()).add(mdle_name));
        break;

    case ZIP_ER_CRC:
        /* N CRC error */
        error(MDLE_CRC_ERROR, Error_params(get_allocator()).add(mdle_name));
        break;

    case ZIP_ER_ZIPCLOSED:
        /* N Containing zip archive was closed */
        error(MDLE_INTERNAL_ERROR, Error_params(get_allocator()));
        break;

    case ZIP_ER_NOENT:
        /* N No such file */
        error(MDLE_FILE_DOES_NOT_EXIST, Error_params(get_allocator()).add(mdle_name));
        break;

    case ZIP_ER_EXISTS:
        /* N File already exists */
        error(MDLE_FILE_ALREADY_EXIST, Error_params(get_allocator()).add(mdle_name));
        break;

    case ZIP_ER_OPEN:
        /* S Can't open file */
        error(MDLE_CANT_OPEN_FILE, Error_params(get_allocator()).add(mdle_name));
        break;

    case ZIP_ER_TMPOPEN:
        /* S Failure to create temporary file */
        error(MDLE_FAILED_TO_OPEN_TEMPFILE, Error_params(get_allocator()));
        break;

    case ZIP_ER_ZLIB:
        /* Z Zlib error */
        error(MDLE_INVALID_MDLE, Error_params(get_allocator()).add(mdle_name));
        break;

    case ZIP_ER_MEMORY:
        /* N Malloc failure */
        error(MDLE_MEMORY_ALLOCATION, Error_params(get_allocator()));
        break;

    case ZIP_ER_COMPNOTSUPP:
        /* N Compression method not supported */
        error(MDLE_INVALID_MDLE, Error_params(get_allocator()).add(mdle_name));
        break;

    case ZIP_ER_EOF:
        /* N Premature end of file */
        error(MDLE_INVALID_MDLE, Error_params(get_allocator()).add(mdle_name));
        break;

    case ZIP_ER_NOZIP:
        /* N Not a zip archive */
        error(MDLE_INVALID_MDLE, Error_params(get_allocator()).add(mdle_name));
        break;

    case ZIP_ER_INCONS:
        /* N Zip archive inconsistent */
        error(MDLE_INVALID_MDLE, Error_params(get_allocator()).add(mdle_name));
        break;

    case ZIP_ER_REMOVE:
        /* S Can't remove file */
        error(FAILED_TO_REMOVE, Error_params(get_allocator()));
        break;

    case ZIP_ER_ENCRNOTSUPP:
        /* N Encryption method not supported */
        error(MDLE_INVALID_MDLE, Error_params(get_allocator()).add(mdle_name));
        break;

    case ZIP_ER_RDONLY:
        /* N Read-only archive */
        error(MDLE_INVALID_MDLE, Error_params(get_allocator()));
        break;

    case ZIP_ER_NOPASSWD:
        /* N No password provided */
    case ZIP_ER_WRONGPASSWD:
        /* N Wrong password provided */
        error(INVALID_PASSWORD, Error_params(get_allocator()));
        break;

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
        error(MDLE_INTERNAL_ERROR, Error_params(get_allocator()));
        break;

    default:
        break;
    }
}

// Translate container errors.
void Encapsulate_tool::translate_container_error(
    MDL_zip_container_error_code err,
    char const                   *mdle_name,
    char const                   *file_name)
{
    switch (err) {
    case EC_OK:
        return;
    case EC_CONTAINER_NOT_EXIST:
        error(MDLE_FILE_DOES_NOT_EXIST, Error_params(get_allocator()).add(mdle_name));
        return;
    case EC_CONTAINER_OPEN_FAILED:
        error(MDLE_CANT_OPEN_FILE, Error_params(get_allocator()).add(mdle_name));
        return;
    case EC_FILE_OPEN_FAILED:
        if (strcmp("MANIFEST", file_name) != 0) {
            error(
                MDLE_DOES_NOT_CONTAIN_ENTRY,
                Error_params(get_allocator())
                .add(mdle_name)
                .add(file_name));
            return;
        }
        // fall-through
    case EC_INVALID_CONTAINER:
        error(
            MDLE_INVALID_MDLE,
            Error_params(get_allocator()).add(mdle_name));
        return;
    case EC_NOT_FOUND:
        error(
            MDLE_DOES_NOT_CONTAIN_ENTRY,
            Error_params(get_allocator())
            .add(mdle_name)
            .add(file_name));
        return;
    case EC_IO_ERROR:
        error(MDLE_IO_ERROR, Error_params(get_allocator()).add(mdle_name));
        return;
    case EC_CRC_ERROR:
        error(MDLE_CRC_ERROR, Error_params(get_allocator()).add(mdle_name));
        return;
    case EC_INVALID_PASSWORD:
        error(MDLE_INVALID_PASSWORD, Error_params(get_allocator()).add(mdle_name));
        return;
    case EC_MEMORY_ALLOCATION:
        error(MDLE_MEMORY_ALLOCATION, Error_params(get_allocator()).add(mdle_name));
        return;
    case EC_RENAME_ERROR:
        error(MDLE_RENAME_FAILED, Error_params(get_allocator()).add(mdle_name));
        return;
    case EC_INVALID_HEADER:
        error(MDLE_INVALID_HEADER, Error_params(get_allocator()).add(mdle_name));
        return;
    case EC_INVALID_HEADER_VERSION:
        error(MDLE_INVALID_HEADER_VERSION, Error_params(get_allocator()).add(mdle_name));
        return;

    case EC_PRE_RELEASE_VERSION:
        warning(MDLE_PRE_RELEASE_VERSION, Error_params(get_allocator()).add(mdle_name));
        return;

    case EC_INTERNAL_ERROR:
        error(MDLE_INTERNAL_ERROR, Error_params(get_allocator()).add(mdle_name));
        return;
    }
}

// Translate zip errors.
void Encapsulate_tool::translate_zip_error(
    char const *mdle_name,
    zip_t      *za)
{
    translate_zip_error(mdle_name, zip_get_error(za)->zip_err);
}

// Translate zip errors
void Encapsulate_tool::translate_zip_error(
    char const        *mdle_name,
    zip_error_t const &ze)
{
    translate_zip_error(mdle_name, ze.zip_err);
}

// Translate zip errors.
void Encapsulate_tool::translate_zip_error(
    char const *mdle_name,
    zip_file_t *src)
{
    translate_zip_error(mdle_name, *zip_file_get_error(src));
}

// Adds a file uncompressed to the given ZIP archive.
zip_int64_t Encapsulate_tool::add_file_uncompressed(
    zip_t        *za,
    char const   *name,
    zip_source_t *src)
{
    // remove leading "./"
    if (name[0] == '.' && name[1] == '/')
        name += 2;

    // add the file
    zip_int64_t index = zip_file_add(za, name, src, ZIP_FL_ENC_UTF_8);
    if (index < 0)
        return -1;

    // turn off compression
    if (zip_set_file_compression(za, index, ZIP_CM_STORE, 0) != 0)
        return -1;

    return index;
}

namespace
{
    struct MD5_hash {
        unsigned char data[16];
    };

    struct memcmp_string_less {
        bool operator()(char const *a, char const *b) const
        {
            size_t la = strlen(a);
            size_t lb = strlen(b);

            int cmp = memcmp(a, b, std::min(la, lb));
            if (cmp == 0)
                return la < lb;

            return cmp < 0;
        }
    };

    typedef map<char const *, MD5_hash, memcmp_string_less>::Type MD5_file_map;
} // anonymous

bool Encapsulate_tool::add_file_uncompressed(
    zip_t                              *za,
    mi::mdl::IMDL_resource_reader      *reader,
    char const                         *target_name,
    char const                         *mdle_name,
    vector<Resource_zip_source*>::Type &add_sources,
    unsigned char                      hash[16])
{
    // compute MD5 hash
    // it would be nice to do this on the fly when the zip is reading the file
    // but this happens later
    MD5_hasher hasher;
    unsigned char buffer[1024];
    while (size_t count = reader->read(buffer, 1024))
        hasher.update(buffer, count);

    hasher.final(hash);
    reader->seek(0, mi::mdl::IMDL_resource_reader::MDL_SEEK_SET);

    // wrap reader into zip source
    Allocator_builder builder(get_allocator());
    add_sources.push_back(builder.create<Resource_zip_source>(reader));

    zip_error_t ze;
    zip_source_t *source = add_sources.back()->open(ze);
    if (source == NULL) {
        translate_zip_error(mdle_name, ze.zip_err);
        return false;
    }

    // add the file to the resource
    zip_int64_t index = add_file_uncompressed(za, target_name, source);
    if (index < 0) {
        translate_zip_error(mdle_name, za);
        return false;
    }

    // add checksum for fast duplicate detection
    if (0 != zip_file_extra_field_set(za, index, MDLE_EXTRA_FIELD_ID_MD, ZIP_EXTRA_FIELD_NEW,
        reinterpret_cast<zip_uint8_t const *>(hash), 16, ZIP_FL_LOCAL))
    {
        translate_zip_error(mdle_name, za);
        return false;
    }
    return true;
}

// Create a new encapsulated mdl file from a given module.
bool Encapsulate_tool::create_encapsulated_module(
    IModule const                 *module,
    char const                    *mdle_name,
    char const                    *dest_path,
    Mdle_export_description const &desc)
{
    string file_name = string(mdle_name, get_allocator());
    file_name.append(".mdle");

    // map to keep track of MD5 hashes to eventually compute the MDLE top level hash
    MD5_file_map sorted_md5_map(get_allocator());

    // create output folder if not present
    if (dest_path != NULL && strcmp(dest_path, "") != 0 &&
        !is_directory_utf8(get_allocator(), dest_path))
    {
        if (!mkdir_utf8(get_allocator(), dest_path)) {
            translate_zip_error(mdle_name, ZIP_ER_WRITE);
            return false;
        }
    }

    // create the writable stream
    zip_error_t ze;
    string file_path = join_path(string(dest_path, get_allocator()), file_name);
    zip_source_t *src = zip_source_file_create(file_path.c_str(), 0, -1, &ze);
    if (src == NULL) {
        translate_zip_error(mdle_name, ze);
        return false;
    }

    // wrap it by the extra layer that writes our MDR header
    Layered_zip_source layer(src, header_write_version);
    zip_source_t *lsrc = layer.open(ze);
    if (lsrc == 0) {
        zip_source_free(src);
        translate_zip_error(mdle_name, ze);
        return false;
    }

    // parse options
    bool override_file = m_options.get_bool_option(MDL_ENCAPS_OPTION_OVERWRITE);

    // create the archive
    zip_t *za = zip_open_from_source(
        lsrc,
        ZIP_CREATE | (override_file ? ZIP_TRUNCATE : ZIP_EXCL),
        &ze);
    if (za == NULL) {
        zip_source_free(lsrc);
        translate_zip_error(mdle_name, ze);
        return false;
    }
    bool has_error = false;

    // ensure the life time of the output buffer lasts until zip_close() where the data is written
    Allocator_builder builder(get_allocator());

    // first, write the module
    mi::base::Handle<Buffer_output_stream> os(builder.create<Buffer_output_stream>(get_allocator()));
    mi::base::Handle<mi::mdl::IMDL_exporter> exporter(m_compiler->create_exporter());

    exporter->export_module(os.get(), module, desc.resource_callback);

    zip_source_t *module_src = zip_source_buffer(
        za, os->get_data(), os->get_data_size(), /*freep=*/0);

    zip_int64_t index = add_file_uncompressed(za, "main.mdl", module_src);
    if (index < 0) {
        translate_zip_error(mdle_name, za);
        has_error = true;
    }
    // make sure the module is the "first" file in the zip
    MDL_ASSERT(index == 0);

    // add checksum for fast duplicate detection
    MD5_hasher hasher;
    hasher.update(reinterpret_cast<unsigned char*>(os->get_data()), os->get_data_size());
    MD5_hash hash;
    hasher.final(hash.data);
    sorted_md5_map["main.mdl"] = hash;

    if (0 != zip_file_extra_field_set(
        za,
        index,
        MDLE_EXTRA_FIELD_ID_MD,
        ZIP_EXTRA_FIELD_NEW,
        reinterpret_cast<zip_uint8_t const *>(hash.data),
        16,
        ZIP_FL_LOCAL))
    {
        translate_zip_error(mdle_name, za);
        has_error = true;
    }

    // keep track of zip sources
    vector<Resource_zip_source*>::Type added_zip_sources(get_allocator());

    // write resources
    for (size_t f = 0, f_n = desc.resource_collector->get_resource_count(); f < f_n; ++f) {
        // source
        mi::base::Handle<mi::mdl::IMDL_resource_reader> reader(
            desc.resource_collector->get_resource_reader(f));
        if (!reader) {
            error(MDLE_INVALID_RESOURCE, Error_params(get_allocator()).add(mdle_name));
            has_error = true;
            break;
        }

        // target
        char const *target_name = desc.resource_collector->get_mlde_resource_path(f);

        // remove leading "./"
        if (target_name[0] == '.' && target_name[1] == '/')
            target_name += 2;

        if (!add_file_uncompressed(
                za, reader.get(), target_name, mdle_name, added_zip_sources, hash.data))
        {
            has_error = true;
            break;
        }

        // store hash to eventually compute the MDLE top level hash
        sorted_md5_map[target_name] = hash;
    }

    // write user files specified by file name
    for (size_t i = 0, n = desc.additional_file_count; i < n; ++i) {
        // source
        mi::base::Handle<mi::mdl::IMDL_resource_reader> reader(
            desc.resource_collector->get_additional_data_reader(
                desc.additional_file_source_paths[i]));
        if (!reader) {
            error(MDLE_INVALID_USER_FILE, Error_params(
                get_allocator()).add(mdle_name).add(desc.additional_file_source_paths[i]));
            has_error = true;
            break;
        }

        // target
        char const *target_name = desc.additional_file_target_paths[i];

        if (!add_file_uncompressed(
            za, reader.get(), target_name, mdle_name, added_zip_sources, hash.data)) 
        {
            has_error = true;
            break;
        }

        // store hash to eventually compute the MDLE top level hash
        sorted_md5_map[target_name] = hash;
    }

    // add zip file comments
    string author("written by: ", get_allocator());
    author.append(desc.authoring_tool_name_and_version);

    if (zip_set_archive_comment(za, author.c_str(), author.size()) != 0) {
        warning(
            MDLE_FAILED_TO_ADD_ZIP_COMMENT,
            Error_params(get_allocator())
            .add(mdle_name));
    }

    // close the file, even when errors occurred
    if (zip_close(za) != 0) {
        translate_zip_error(mdle_name, za);
        has_error = true;
    }

    // free zip sources
    for (vector<Resource_zip_source *>::Type::iterator it(added_zip_sources.begin()),
        end(added_zip_sources.end());
        it != end;
        ++it)
    {
        builder.destroy(*it);
    }

    // compute the MDLE top level hash and
    // add checksum for fast duplicate detection
    hasher.restart();
    for (MD5_file_map::const_iterator it(sorted_md5_map.begin()), end(sorted_md5_map.end());
        it != end;
        ++it)
    {
        hasher.update(reinterpret_cast<unsigned char const *>(it->first), strlen(it->first));
        hasher.update(it->second.data, 16);
    }
    hasher.final(hash.data);

    FILE *fp = fopen_utf8(get_allocator(), file_path.c_str(), "rb+");
    if (fp == NULL ||
        fseek(fp, 8, SEEK_SET) != 0 ||
        fwrite(hash.data, 1, 16, fp) != 16 ||
        fclose(fp) != 0)
    {
        translate_zip_error(mdle_name, ZIP_ER_WRITE);
        has_error = true;
    }

    if (has_error) {
        return false;
    }
    return true;
}

// Get the content of a file into a memory buffer.
IMDL_resource_reader *Encapsulate_tool::get_content_buffer(
    char const *archive_name,
    char const *file_name)
{
    if (archive_name == NULL) {
        error(MDLE_INVALID_NAME, Error_params(get_allocator()).add("<NULL>"));
        return NULL;
    }

    size_t l = strlen(archive_name);
    if (l < 5 || strcmp(&archive_name[l - 5], ".mdle") != 0) {
        error(MDLE_INVALID_NAME, Error_params(get_allocator()).add(archive_name));
        return NULL;
    }

    string full_path(archive_name, get_allocator());
    full_path.append(':');
    full_path.append(file_name);

    MDL_zip_container_error_code err = EC_OK;
    mi::base::Handle<IMDL_resource_reader> res(open_resource_file(
        get_allocator(),
        /*mdl_url=*/"",
        full_path.c_str(),
        err));

    translate_container_error(err, archive_name, file_name);

    if (res.is_valid_interface()) {
        res->retain();
        return res.get();
    }
    return NULL;
}

// Get the content from any file out of an archive on the file system.
IInput_stream *Encapsulate_tool::get_file_content(
    char const *mdle_path,
    char const *file_name)
{
    if (file_name == NULL) {
        error(
            MDLE_DOES_NOT_CONTAIN_ENTRY,
            Error_params(get_allocator())
            .add(mdle_path)
            .add("<NULL>"));
        return NULL;
    }

    string afn(convert_os_separators_to_slashes(string(file_name, get_allocator())));
    mi::base::Handle<IMDL_resource_reader> reader(get_content_buffer(mdle_path, afn.c_str()));

    if (!reader.is_valid_interface()) {
        // failed
        return NULL;
    }

    Allocator_builder builder(get_allocator());
    return builder.create<Resource_Input_stream>(get_allocator(), reader.get());
}

// Opens an MDLE zip container for access to its file list and the MD5 hashes.
MDL_zip_container_mdle* Encapsulate_tool::open_encapsulated_module(
    char const *mdle_path)
{
    // open the mdle
    MDL_zip_container_error_code err;
    MDL_zip_container_mdle *capsule = MDL_zip_container_mdle::open(get_allocator(), mdle_path, err);
    if (capsule == NULL) {
        error(
            MDLE_FILE_DOES_NOT_EXIST,
            Error_params(get_allocator())
            .add(mdle_path));
        return NULL;
    }
    return capsule;
}

// Checks the MD5 hashes of all files in the MDLE to identify changes from outside
bool Encapsulate_tool::check_integrity(
    char const *mdle_path)
{
    // open the mdle
    MDL_zip_container_error_code err;
    MDL_zip_container_mdle *capsule = MDL_zip_container_mdle::open(get_allocator(), mdle_path, err);
    if (capsule == NULL) {
        error(
            MDLE_FILE_DOES_NOT_EXIST,
            Error_params(get_allocator())
            .add(mdle_path));
        return false;
    }


    // map to keep track of MD5 hashes to eventually compute the MDLE top level hash
    MD5_file_map sorted_md5_map(get_allocator());

    // check all files
    MD5_hash hash;
    bool success = true;
    for (int i = 0, n = capsule->get_num_entries(); i < n && success; ++i) {
        char const *file_name = capsule->get_entry_name(i);

        success = check_file_content_integrity(capsule, file_name, hash.data);

        if (!success)
            break;

        // store the hash to compute the top level hash
        sorted_md5_map[file_name] = hash;
    }

    uint16_t major_version, minor_version;
    capsule->get_version(major_version, minor_version);

    // compute the MDLE top level hash
    if (success && (major_version > 0 || minor_version > 1)) {
        MD5_hasher hasher;
        for (MD5_file_map::const_iterator it(sorted_md5_map.begin()), end(sorted_md5_map.end());
            it != end;
            ++it)
        {
            hasher.update(reinterpret_cast<unsigned char const *>(it->first), strlen(it->first));
            hasher.update(it->second.data, 16);
        }
        hasher.final(hash.data);

        MD5_hash stored_hash;
        capsule->get_hash(stored_hash.data);

        // compare the hashes
        for (size_t i = 0; i < 16; ++i) {
            if (hash.data[i] != stored_hash.data[i]) {
                error(
                    MDLE_CONTENT_FILE_INTEGRITY_FAIL,
                    Error_params(get_allocator())
                    .add(mdle_path)
                    .add("top level hash"));

                success = false;
                break;
            }
        }
    }

    capsule->close();
    return success;
}

// For each resource, an MD5 hash is stored in the extra fields of the archive files.
// This method checks if the content matches the hash to identify changes from outside.
bool Encapsulate_tool::check_file_content_integrity(
    char const    *mdle_path,
    char const    *file_name,
    unsigned char out_hash[16])
{
    if (file_name == NULL) {
        error(
            MDLE_DOES_NOT_CONTAIN_ENTRY,
            Error_params(get_allocator())
            .add(mdle_path)
            .add("<NULL>"));
        return false;
    }

    // open the mdle
    MDL_zip_container_mdle *capsule = open_encapsulated_module(mdle_path);
    if (capsule == NULL)
        return false;

    bool success = true;
    success = success && check_file_content_integrity(capsule, file_name, out_hash);
    capsule->close();
    return success;
}

// This method checks if the content matches the hash to identify changes from outside.
bool Encapsulate_tool::check_file_content_integrity(
    MDL_zip_container_mdle *mdle,
    char const             *file_name,
    unsigned char          out_hash[16])
{
    // read the stored hash
    unsigned char stored_hash[16];
    if (!get_file_content_hash(mdle, file_name, stored_hash)) {
        error(
            MDLE_CONTENT_FILE_INTEGRITY_FAIL,
            Error_params(get_allocator())
            .add(mdle->get_container_name())
            .add(file_name));

        memset(out_hash, '\0', 16);
        return false;
    }

    // compute the hash of the file content
    unsigned char computed_hash[16];
    if (!mdle->compute_file_hash(file_name, computed_hash)) {
        error(
            MDLE_DOES_NOT_CONTAIN_ENTRY,
            Error_params(get_allocator())
            .add(mdle->get_container_name())
            .add(file_name));

        memset(out_hash, 0, 16);
        return false;
    }

    // compare the hashes
    for (size_t i = 0; i < 16; ++i) {
        if (computed_hash[i] != stored_hash[i]) {
            error(
                MDLE_CONTENT_FILE_INTEGRITY_FAIL,
                Error_params(get_allocator())
                .add(mdle->get_container_name())
                .add(file_name));

            memset(out_hash, '\0', 16);
            return false;
        }
    }

    memcpy(out_hash, stored_hash, 16);
    return true;
}

// For each resource, an MD5 hash is stored in the extra fields of the archive files.
// This method reads this hash for a selected file.
bool Encapsulate_tool::get_file_content_hash(
    MDL_zip_container_mdle *mdle,
    char const             *file_name,
    unsigned char          out_hash[16])
{
    // read the stored hash
    MDL_zip_container_file *file = mdle->file_open(file_name);

    if (file != NULL) {
        size_t length = 0;
        unsigned char const *stored_hash =
            file->get_extra_field(MDLE_EXTRA_FIELD_ID_MD, length);

        bool res = stored_hash != NULL && length == 16;
        if (res)
            memcpy(out_hash, stored_hash, 16);

        file->close();
        return res;
    }

    return false;
}

// Access messages of last operation.
Messages const &Encapsulate_tool::access_messages() const
{
    return m_msg_list;
}

// Creates a new error.
void Encapsulate_tool::error(int code, Error_params const &params)
{
    Position_impl zero(0, 0, 0, 0);

    string msg(m_msg_list.format_msg(code, MESSAGE_CLASS, params));
    m_last_msg_idx = m_msg_list.add_error_message(
        code, MESSAGE_CLASS, 0, &zero, msg.c_str());
}

// Creates a new warning.
void Encapsulate_tool::warning(int code, Error_params const &params)
{
    Position_impl zero(0, 0, 0, 0);

    string msg(m_msg_list.format_msg(code, MESSAGE_CLASS, params));
    m_last_msg_idx = m_msg_list.add_warning_message(
        code, MESSAGE_CLASS, 0, &zero, msg.c_str());
}

// Adds a new note to the previous message.
void Encapsulate_tool::add_note(int code, Error_params const &params)
{
    Position_impl zero(0, 0, 0, 0);

    string msg(m_msg_list.format_msg(code, MESSAGE_CLASS, params));
    m_msg_list.add_note(
        m_last_msg_idx,
        IMessage::MS_INFO,
        code,
        MESSAGE_CLASS,
        0,
        &zero,
        msg.c_str());
}

// Access options.
Options &Encapsulate_tool::access_options()
{
    return m_options;
}

}  // mdl
}  // mi
