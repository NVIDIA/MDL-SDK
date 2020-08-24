/******************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
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
#include "compilercore_file_utils.h"
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
#include "compilercore_encapsulator.h"

namespace mi {
namespace mdl {

typedef Store<Position const *> Position_store;

// Get the FILE handle if this object represents an ordinary file.
FILE *File_handle::get_file() { return u.fp; }

// Get the archive if this object represents a file inside a MDL archive.
MDL_zip_container *File_handle::get_container()
{
    return m_container;
}

// Get the compressed file handle if this object represents a file inside a MDL archive.
MDL_zip_container_file *File_handle::get_container_file()
{
    if (m_kind == FH_FILE)
        return NULL;
    return u.z_fp;
}

// Open a file handle.
File_handle *File_handle::open(
    IAllocator                     *alloc,
    char const                     *name,
    MDL_zip_container_error_code  &err)
{
    Allocator_builder builder(alloc);
    char const *p = NULL;
    err = EC_OK;

    // handle archives
    p = strstr(name, ".mdr:");
    if (p != NULL) {
        string root_name(name, p + 4, alloc);
        p += 5;

        // check if the root is an archive itself
        if (MDL_zip_container_archive *archive = MDL_zip_container_archive::open(
            alloc,
            root_name.c_str(),
            err))
        {
            if (MDL_zip_container_file *fp = archive->file_open(p)) {
                return builder.create<File_handle>(
                    alloc, File_handle::FH_ARCHIVE, archive, /*owns_archive=*/true, fp);
            }
            archive->close();

            // not in the archive
            err = EC_NOT_FOUND;
            return NULL;
        }
        return NULL;
    }

    // handle MDLe
    p = strstr(name, ".mdle:");
    if (p != NULL) {
        string root_name(name, p + 5, alloc);
        p += 6;

        // check if the root is an capsule itself
        if (MDL_zip_container_mdle *capsule = MDL_zip_container_mdle::open(
            alloc,
            root_name.c_str(),
            (MDL_zip_container_error_code&) err))
        {
            if (MDL_zip_container_file *fp = capsule->file_open(p)) {
                return builder.create<File_handle>(
                    alloc, File_handle::FH_MDLE, capsule, /*owns_archive=*/true, fp);
            }
            capsule->close();

            // not in the EMDL
            err = EC_NOT_FOUND;
            return NULL;
        }
        return NULL;
    }

    // must be a file then
    if (FILE *fp = fopen_utf8(alloc, name, "rb")) {
        return builder.create<File_handle>(alloc, fp);
    }
    err = EC_FILE_OPEN_FAILED;
    return NULL;
}

// Close a file handle.
void File_handle::close(File_handle *h)
{
    if (h != NULL) {
        Allocator_builder builder(h->m_alloc);

        builder.destroy(h);
    }
}

// Constructor from a FILE handle.
File_handle::File_handle(
    IAllocator *alloc,
    FILE       *fp)
: m_alloc(alloc)
, m_container(NULL)
, m_kind(FH_FILE)
, m_owns_container(false)
{
    u.fp = fp;
}

// Constructor from archive file.
File_handle::File_handle(
    IAllocator             *alloc,
    Kind                    kind,
    MDL_zip_container      *archive,
    bool                    owns_archive,
    MDL_zip_container_file *fp)
: m_alloc(alloc)
, m_container(archive)
, m_kind(kind)
, m_owns_container(owns_archive)
{
    u.z_fp = fp;
}

// Constructor from another File_handle archive.
File_handle::File_handle(
    File_handle            *fh,
    Kind                    kind,
    MDL_zip_container_file *fp)
: m_alloc(fh->m_alloc)
, m_container(fh->m_container)
, m_kind(kind)
, m_owns_container(false)
{
    MDL_ASSERT(fh->get_kind() != FH_FILE);
    u.z_fp = fp;
}

// Destructor.
File_handle::~File_handle()
{
    if (m_kind != FH_FILE) {
        u.z_fp->close();
        if (m_owns_container)
            m_container->close();
    } else {
        fclose(u.fp);
    }
}

// ------------------------------------------------------------------------

/// Implementation of a resource reader from a file.

// Read a memory block from the resource.
Uint64 File_resource_reader::read(void *ptr, Uint64 size)
{
    if (m_file->get_kind() == File_handle::FH_FILE) {
        return fread(ptr, 1, size, m_file->get_file());
    } else {
        return m_file->get_container_file()->read(ptr, size);
    }
}

// Get the current position.
Uint64 File_resource_reader::tell()
{
    if (m_file->get_kind() == File_handle::FH_FILE) {
#if defined(MI_PLATFORM_LINUX) || defined(MI_PLATFORM_MACOSX)
        return ftello(m_file->get_file());
#elif defined(MI_PLATFORM_WINDOWS)
        return _ftelli64(m_file->get_file());
#else
        return ftell(m_file->get_file());
#endif
    } else {
        return m_file->get_container_file()->tell();
    }
}

// Reposition stream position indicator.
bool File_resource_reader::seek(Sint64 offset, Position origin)
{
    if (m_file->get_kind() == File_handle::FH_FILE) {
#if defined(MI_PLATFORM_LINUX) || defined(MI_PLATFORM_MACOSX)
        return fseeko(m_file->get_file(), off_t(offset), origin) == 0;
#elif defined(MI_PLATFORM_WINDOWS)
        return _fseeki64(m_file->get_file(), offset, origin) == 0;
#else
        return fseek(m_file->get_file(), long(offset), origin) == 0;
#endif
    } else {
        switch (origin) {
        case MDL_SEEK_CUR:
            return m_file->get_container_file()->seek(offset, SEEK_CUR) == 0;
        case MDL_SEEK_END:
            return m_file->get_container_file()->seek(offset, SEEK_END) == 0;
        case MDL_SEEK_SET:
            return m_file->get_container_file()->seek(offset, SEEK_SET) == 0;
        }
    }
    assert(false);
    return false;
}

// Get the UTF8 encoded name of the resource on which this reader operates.
char const *File_resource_reader::get_filename() const
{
    return m_file_name.empty() ? NULL : m_file_name.c_str();
}

// Get the UTF8 encoded absolute MDL resource name on which this reader operates.
char const *File_resource_reader::get_mdl_url() const
{
    return m_mdl_url.empty() ? NULL : m_mdl_url.c_str();
}

// Returns the associated hash of this resource.
bool File_resource_reader::get_resource_hash(unsigned char hash[16])
{
    // not supported on ordinary files
    return false;
}

// Constructor.
File_resource_reader::File_resource_reader(
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

// Destructor.
File_resource_reader::~File_resource_reader()
{
    File_handle::close(m_file);
}

namespace {

/// Implementation of the IInput_stream interface using FILE I/O.
class Simple_file_input_stream : public Allocator_interface_implement<IInput_stream>
{
    typedef Allocator_interface_implement<IInput_stream> Base;
public:
    /// Constructor.
    ///
    /// \param alloc     the allocator
    /// \param f         the FILE handle, takes ownership
    /// \param filename  the filename of the handle
    explicit Simple_file_input_stream(
        IAllocator  *alloc,
        File_handle *f,
        char const  *filename)
    : Base(alloc)
    , m_file(f)
    , m_filename(filename, alloc)
    {}

    /// Destructor.
    ///
    /// \note Closes the file handle.
    ~Simple_file_input_stream() MDL_FINAL
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
        if (m_file->get_container_file()->read(&buf, 1) != 1)
            return -1;
        return int(buf);
    }

    /// Get the name of the file on which this input stream operates.
    /// \returns    The name of the file or null if the stream does not operate on a file.
    char const *get_filename() MDL_FINAL
    {
        return m_filename.empty() ? NULL : m_filename.c_str();
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


/// Implementation of the IMdle_input_stream interface using mdle I/O.
class Mdle_input_stream : public Allocator_interface_implement<IMdle_input_stream>
{
    typedef Allocator_interface_implement<IMdle_input_stream> Base;
public:
    /// Constructor.
    explicit Mdle_input_stream(
        IAllocator     *alloc,
        File_handle    *f,
        char const     *filename)
    : Base(alloc)
    , m_file(f)
    , m_filename(filename, alloc)
    {
    }

protected:
    /// Destructor.
    ~Mdle_input_stream() MDL_FINAL
    {
        File_handle::close(m_file);
    }

public:
    /// Read a character from the input stream.
    /// \returns    The code of the character read, or -1 on the end of the stream.
    int read_char() MDL_FINAL
    {
        unsigned char buf;
        if (m_file->get_container_file()->read(&buf, 1) != 1)
            return -1;
        return int(buf);
    }

    /// Get the name of the file on which this input stream operates.
    /// \returns    The name of the file or null if the stream does not operate on a file.
    char const *get_filename() MDL_FINAL
    {
        return m_filename.empty() ? NULL : m_filename.c_str();
    }

private:
    /// The file handle.
    File_handle *m_file;

    /// The filename.
    string m_filename;
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
    mi::base::Handle<IEntity_resolver> const &external_resolver,
    mi::base::Handle<IMDL_search_path> const &search_path,
    mi::base::Lock                           &sp_lock,
    Messages_impl                            &msgs,
    char const                               *front_path)
: m_alloc(mdl.get_allocator())
, m_mdl(mdl)
, m_module_cache(module_cache)
, m_msgs(msgs)
, m_pos(NULL)
, m_external_resolver(external_resolver)
, m_search_path(search_path)
, m_resolver_lock(sp_lock)
, m_paths(m_alloc)
, m_resource_paths(m_alloc)
, m_killed_packages(String_set::key_compare(), m_alloc)
, m_front_path(front_path)
, m_resolve_entity(NULL)
, m_repl_module_name(m_alloc)
, m_repl_file_name(m_alloc)
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
    // handle MDLe
    string input_url_str(input_url, m_alloc);
    size_t l = input_url_str.size();
    if (l > 5 &&
        input_url_str[l - 5] == '.' &&
        input_url_str[l - 4] == 'm' &&
        input_url_str[l - 3] == 'd' &&
        input_url_str[l - 2] == 'l' &&
        input_url_str[l - 1] == 'e') {

        string input_name("::", m_alloc);
        input_name.append(convert_os_separators_to_slashes(input_url_str));
        return input_name;
    }

    // handle MDL
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

    MDL_zip_container_error_code err = EC_OK;
    if (MDL_zip_container_archive *archive = MDL_zip_container_archive::open(
        m_alloc, archive_name, err, false))
    {
        res = archive->contains(file_name);
        archive->close();
    } else {
        if (err == EC_OK) {
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

    MDL_zip_container_error_code err = EC_OK;
    if (MDL_zip_container_archive *archive = MDL_zip_container_archive::open(
        m_alloc, archive_name, err, false))
    {
        res = archive->contains_mask(file_mask);
        archive->close();
    } else {
        if (err == EC_OK) {
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
    bool         &cwd_is_container,
    string       &current_search_path,
    bool         &csp_is_container,
    string       &current_module_path)
{
    cwd_is_container = false;
    csp_is_container = false;
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

    size_t container_pos = module_file_system_path.find(".mdr:");
    File_handle::Kind container_kind = File_handle::FH_ARCHIVE;

    if (container_pos == string::npos) {
        container_pos = module_file_system_path.find(".mdle:");
        container_kind = File_handle::FH_MDLE;
    }

    if (container_pos != string::npos) {

        // inside an archive
        if( container_kind == File_handle::FH_ARCHIVE)
            container_pos += 4; // add ".mdr"
                // inside an archive
        else if (container_kind == File_handle::FH_MDLE)
            container_pos += 5; // add ".mdle"

        string simple_path = module_file_system_path.substr(container_pos + 1);

        size_t last_sep = simple_path.find_last_of(sep);
        if (last_sep != string::npos) {
            current_working_directory =
                module_file_system_path.substr(0, container_pos + 1) +
                simple_path.substr(0, last_sep);
        } else {
            // only the archive
            current_working_directory = module_file_system_path.substr(0, container_pos);
            cwd_is_container = true;
        }

        // points now to ':'
        ++container_pos;
    } else {
        size_t last_sep = module_file_system_path.find_last_of(sep);
        MDL_ASSERT(last_sep != string::npos);
        current_working_directory = module_file_system_path.substr(0, last_sep);

        container_pos = 0;
    }

    current_search_path = current_working_directory;
    size_t strip_dotdot = 0;
    while (module_nesting_level-- > 0) {
        size_t last_sep = current_search_path.find_last_of(sep);
        if (last_sep == string::npos) {
            current_module_path = "";
            csp_is_container = false;
            return;
        }
        if (last_sep < container_pos) {
            // do NOT remove the archive name, thread its ':' like '/'
            last_sep = container_pos - 1;
            // should never try to go out!
            MDL_ASSERT(module_nesting_level == 0);
            container_pos = 0;
            strip_dotdot = 1;
        } else {
            strip_dotdot = 0;
        }
        current_search_path = current_search_path.substr(0, last_sep);
    }

    csp_is_container = strip_dotdot != 0;
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
    bool absolute = is_path_absolute(s);
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
    bool         file_mask_is_regex,
    string const &directory_path,
    string const &file_name,
    size_t       nesting_level,
    string const &module_file_system_path,
    string const &current_working_directory,
    bool         cwd_is_archive,
    string const &current_module_path)
{
    if (file_path.rfind(".mdle:") != std::string::npos) {
        // Assume, this MDL URL is prefixed by an MDLe name, forming an absolute URL
        return file_path;
    }

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
        if (!check_no_dots_strict_relative(file_path.c_str(), nesting_level))
            return string(m_alloc);

        // reject if file does not exist w.r.t. current working directory
        string file_mask_os = convert_slashes_to_os_separators(file_mask);

        string file = cwd_is_archive ?
            current_working_directory + ':' + simplify_path(file_mask_os, os_separator()) :
            simplify_path(join_path(current_working_directory, file_mask_os), os_separator());
        if (!file_exists(file.c_str(), file_mask_is_regex)) {
            // FIXME: do we need an error here
            return string(m_alloc);
        }

        // canonical path is the file path resolved w.r.t. the current module path
        return simplify_path(current_module_path + "/" + file_path, '/');
    }

    // reject invalid weak relative paths
    if (!check_no_dots(file_path.c_str()))
        return string(m_alloc);

    // absolute file paths (note: this is an URL, not a file on the OS file system)
    if (!file_path.empty() && file_path[0] == '/') {
        // canonical path is the same as the file path
        return file_path;
    }

    // weak relative file paths

    // special case (not in spec)
    if (module_file_system_path.empty())
        return "/" + file_path;

    // if file does not exist w.r.t. current working directory: canonical path is file path
    // prepended with a slash
    string file_mask_os = convert_slashes_to_os_separators(file_mask);
    string file = join_path(current_working_directory, file_mask_os);

    // if the searched file does not exists locally, assume the weak relative path is absolute
    if (!file_exists(file.c_str(), file_mask_is_regex))
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
    mi::base::Handle<const mi::mdl::IModule> module(
        m_module_cache->lookup(module_name.c_str(), NULL));
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
    bool         is_regex,
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
                    resolved_file_system_location[len - 4] == '.') ||
                (len > 5 &&
                    resolved_file_system_location[len]     == ':' &&
                    resolved_file_system_location[len - 1] == 'e' &&
                    resolved_file_system_location[len - 2] == 'l' &&
                    resolved_file_system_location[len - 3] == 'd' &&
                    resolved_file_system_location[len - 4] == 'm' &&
                    resolved_file_system_location[len - 5] == '.')

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
        if (!file_exists(file.c_str(), is_regex))
            return true;
    } else {
        file = current_search_path + canonical_file_path_os;
        if (!file_exists(file.c_str(), is_regex))
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

        s = e + 1;
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
            if (!dir.open(path, "*.mdr")) {
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
    char const *fname,
    bool       is_regex) const
{
    char const *p_archive = strstr(fname, ".mdr:");
    char const *p_mdle = (p_archive == NULL) ? strstr(fname, ".mdle:") : NULL;

    if (p_archive == NULL && p_mdle == NULL) {
        // not inside an archive
        if (is_regex) {
            string dname(m_alloc);

            char const *p = strrchr(fname, os_separator());
            if (p != NULL) {
                dname = string(fname, p - fname, m_alloc);
                fname = p + 1;
            }

            return has_file_utf8(m_alloc, dname.c_str(), fname);
        }
        return is_file_utf8(m_alloc, fname);
    }

    // open container, mdr or mdle
    char const *container_file_name = NULL;
    MDL_zip_container *container = NULL;
    MDL_zip_container_error_code err = EC_OK;

    if (p_archive != NULL) {
        string container_name = string(fname, p_archive + 4, m_alloc);
        container_file_name = p_archive + 5;
        container = MDL_zip_container_archive::open(m_alloc, container_name.c_str(), err, false);
    }

    if (p_mdle != NULL) {
        string container_name = string(fname, p_mdle + 5, m_alloc);
        container_file_name = p_mdle + 6;
        container = MDL_zip_container_mdle::open(m_alloc, container_name.c_str(), err);
    }

    // check if there is a corresponding file
    if (container != NULL) {
        bool res = is_regex ?
            container->contains_mask(container_file_name) :
            container->contains(container_file_name);
        container->close();
        return res;
    }
    return false;
}

#ifdef MI_PLATFORM_WINDOWS

static bool is_drive_letter(char c)
{
    return ((c >= 'A') && (c <= 'Z')) || ((c >= 'a') && (c <= 'z'));
}

#endif

/// Check if a given MDL url is absolute.
static bool is_url_absolute(char const *url)
{
    if (url == NULL)
        return false;

    if (url[0] == '/')
        return true;

#ifdef MI_PLATFORM_WINDOWS
    if (strlen(url) > 1) {
        // classic drive letter
        if (is_drive_letter(url[0]) && url[1] == ':') {
            return true;
        }
        // UNC
        if (url[0] == os_separator() && url[1] == os_separator()) {
            return true;
        }
    }
#endif

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
    udim_mode = NO_UDIM;

    Position_store store(m_pos, pos);

    string url(file_path, m_alloc);

    size_t nesting_level = module_name != NULL ? get_module_nesting_level(module_name) : 0;

    bool owner_is_string_module = module_file_system_path == NULL;
    string module_file_system_path_str(
        owner_is_string_module ? "" : module_file_system_path, m_alloc);

    // check if this is an mdle or a resource inside one
    bool is_mdle_module = false;
    if (is_resource && module_name != NULL) {
        size_t l = strlen(module_name);
        if (l > 5 &&
            module_name[l - 5] == '.' &&
            module_name[l - 4] == 'm' &&
            module_name[l - 3] == 'd' &&
            module_name[l - 2] == 'l' &&
            module_name[l - 1] == 'e') {

           is_mdle_module = true;
        }
    } else {
        size_t l = url.size();
        if (l > 5 &&
            url[l - 5] == '.' &&
            url[l - 4] == 'm' &&
            url[l - 3] == 'd' &&
            url[l - 2] == 'l' &&
            url[l - 1] == 'e') {

            is_mdle_module = true;
        }
    }

    // Step 0: compute terms defined in MDL spec
    string directory_path(m_alloc);
    string file_name(m_alloc);
    split_file_path(url, directory_path, file_name);

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
        udim_mode != NO_UDIM,
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

    // If this is an absolute module name AND we have a cache, see if the module exists.
    // If yes, use it.
    if (m_module_cache && !is_resource && is_url_absolute(file_path)) {
        string module_name(to_module_name(file_path));

        // remove .mdl
        size_t l = module_name.size();
        if (l > 4) {
            if (module_name[l - 4] == '.' &&
                module_name[l - 3] == 'm' &&
                module_name[l - 2] == 'd' &&
                module_name[l - 1] == 'l')
            {
                module_name = module_name.substr(0, l - 4);
            }
        }

        mi::base::Handle<IModule const> mod(m_module_cache->lookup(module_name.c_str(), NULL));

        if (mod.is_valid_interface()) {
            abs_file_name = string(mod->get_filename(), m_alloc);
            return string(file_path, m_alloc);
        }
    }


    string resolved_file_system_location(m_alloc);
    // Step 2.1 check for MDLe existence
    if (is_mdle_module) {
        string file(m_alloc);

        if (is_resource) {
            file = convert_slashes_to_os_separators(url_mask);
            file = simplify_path(file, os_separator());
            if (!is_url_absolute(file.c_str())) {

                // skip first slash
                // TODO fix earlier
                if (file[0] == os_separator())
                    file = file.substr(1);

                // for MDLE, the current working directory is the MDLE file
                file = current_working_directory + ':' + file;
            }
            file = convert_slashes_to_os_separators(file);
        } else {
            file = convert_slashes_to_os_separators(url_mask);
        }

        if (!file_exists(file.c_str(), udim_mode != NO_UDIM)) {
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
                .add(file.c_str())
                .add_module_origin(module_for_error_msg.c_str()));
            return string(m_alloc);
        }

        resolved_file_system_location = file.c_str();
    } else {
        // Step 2: consider search paths
        resolved_file_system_location = consider_search_paths(
            canonical_file_mask, is_resource, file_path, udim_mode);

        // the referenced resource is part of an MDLE
        // Note, this is invalid for mdl modules in the search paths!
        if (resolved_file_system_location.empty() && strstr(file_path, ".mdle:") != NULL) {
            if (file_exists(file_path, udim_mode != NO_UDIM))
                resolved_file_system_location = file_path;
        }

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
    }
    // Step 3: consistency checks
    if (!check_consistency(
        resolved_file_system_location,
        canonical_file_path,
        udim_mode != NO_UDIM,
        url,
        current_working_directory,
        current_search_path,
        is_resource,
        csp_is_archive,
        owner_is_string_module))
    {
        return string(m_alloc);
    }

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
void File_resolver::handle_file_error(MDL_zip_container_error_code err)
{
    // FIXME
    switch (err) {
    case EC_OK:
        return;
    case EC_CONTAINER_NOT_EXIST:
        return;
    case EC_CONTAINER_OPEN_FAILED:
        return;
    case EC_FILE_OPEN_FAILED:
        return;
    case EC_INVALID_CONTAINER:
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
    case EC_INVALID_HEADER:
        return;
    case EC_INVALID_HEADER_VERSION:
        return;
    case EC_PRE_RELEASE_VERSION:
        return;
    case EC_INTERNAL_ERROR:
        return;
    }
}


// Resolve an import (module) name to the corresponding absolute module name.
IMDL_import_result *File_resolver::resolve_import(
    Position const &pos,
    char const     *import_name,
    char const     *owner_name,
    char const     *owner_filename)
{
    mark_module_search(import_name);

    // detect MDLe
    bool is_mdle = false;
    string import_file = to_url(import_name);
    size_t l = import_file.length();
    if (l > 5 &&
        import_file[l - 5] == '.' &&
        import_file[l - 4] == 'm' &&
        import_file[l - 3] == 'd' &&
        import_file[l - 2] == 'l' &&
        import_file[l - 1] == 'e') {

        // undo 'to_url' and remove the leading 'module ::' when handling MDLE
        if (import_name[0] == ':' && import_name[1] == ':')
            import_file = import_name + 2;

        // use forward slashes to detect absolute filenames correctly
        std::replace(import_file.begin(), import_file.end(), '\\', '/');

        is_mdle = true;
    } else {
        // no MDLe
        import_file.append(".mdl");
    }

    if (owner_name != NULL && owner_name[0] == '\0')
        owner_name = NULL;
    if (owner_filename != NULL && owner_filename[0] == '\0')
        owner_filename = NULL;

    if (m_external_resolver.is_valid_interface()) {
        IMDL_import_result *result = m_external_resolver->resolve_module(
                import_name,
                owner_filename,
                owner_name,
                &pos);
        m_msgs.copy_messages(m_external_resolver->access_messages());
        return result;
    }

    string os_file_name(m_alloc);
    UDIM_mode udim_mode = NO_UDIM;
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
        return NULL;
    }

    Allocator_builder builder(m_alloc);

    if (is_mdle) {
        string absolute_name = to_module_name(canonical_file_name.c_str());

        MDL_ASSERT(absolute_name.substr(absolute_name.size() - 5) == ".mdle");
        return builder.create<MDL_import_result>(m_alloc, absolute_name, os_file_name);
    }

    string absolute_name = to_module_name(canonical_file_name.c_str());
    MDL_ASSERT(absolute_name.substr(absolute_name.size() - 4) == ".mdl");
    absolute_name = absolute_name.substr(0, absolute_name.size() - 4);

    if (absolute_name == m_repl_module_name) {
        // do the replacement
        os_file_name = m_repl_file_name;
    }

    return builder.create<MDL_import_result>(
        m_alloc,
        absolute_name,
        os_file_name);
}

// Resolve a resource (file) name to the corresponding absolute file path.
IMDL_resource_set *File_resolver::resolve_resource(
    Position const &pos,
    char const     *import_file,
    char const     *owner_name,
    char const     *owner_filename)
{
    mark_resource_search(import_file);

    if (owner_name != NULL && owner_name[0] == '\0')
        owner_name = NULL;
    if (owner_filename != NULL && owner_filename[0] == '\0')
        owner_filename = NULL;

    if (m_external_resolver.is_valid_interface()) {
        IMDL_resource_set *result = m_external_resolver->resolve_resource_file_name(
            import_file,
            owner_filename,
            owner_name,
            &pos);
        m_msgs.copy_messages(m_external_resolver->access_messages());
        return result;
    }

    // Make owner file system path absolute
    string owner_filename_str(m_alloc);
    if (owner_filename == NULL) {
        owner_filename_str = "";
    } else {
        MDL_ASSERT(is_path_absolute(owner_filename));
        owner_filename_str = owner_filename;
    }

    UDIM_mode udim_mode = NO_UDIM;
    string os_file_name(m_alloc);
    string abs_file_name = resolve_filename(
        os_file_name, import_file, /*is_resource=*/true,
        owner_filename_str.c_str(), owner_name, &pos, udim_mode);

    if (abs_file_name.empty())
        return NULL;

    if (udim_mode != NO_UDIM) {
        // lookup ALL files for the given mask
        return MDL_resource_set::from_mask(
            m_alloc,
            abs_file_name.c_str(),
            os_file_name.c_str(),
            udim_mode);
    } else {
        // single return
        Allocator_builder builder(m_alloc);

        bool has_hash = false;
        unsigned char hash[16];

        char const *p = strstr(os_file_name.c_str(), ".mdle:");
        if (p != NULL) {
            string container_name(os_file_name.c_str(), p + 5, m_alloc);
            p += 6;

            MDL_zip_container_error_code err;
            MDL_zip_container *container =
                MDL_zip_container_mdle::open(m_alloc, container_name.c_str(), err);

            if (container != NULL) {
                if (container->has_resource_hashes()) {
                    if (MDL_zip_container_file *z_f = container->file_open(p)) {
                        size_t length = 0;
                        unsigned char const *stored_hash =
                            z_f->get_extra_field(MDLE_EXTRA_FIELD_ID_MD, length);

                        if (stored_hash != NULL && length == 16) {
                            memcpy(hash, stored_hash, 16);
                            has_hash = true;
                        }
                        z_f->close();
                    }
                }
                container->close();
            }
        }

        return builder.create<MDL_resource_set>(
            m_alloc,
            abs_file_name.c_str(),
            os_file_name.c_str(),
            has_hash ? hash : NULL);
    }
}

// Checks whether the given module source exists.
IInput_stream *File_resolver::open(
    char const *module_name)
{
    MDL_ASSERT(module_name != NULL && module_name[0] == ':' && module_name[1] == ':');

    string canonical_file_path = to_url(module_name);
    string resolved_file_path(m_alloc);

    if (canonical_file_path.size() > 5 &&
            canonical_file_path.substr(canonical_file_path.size() - 5) == ".mdle") {
        resolved_file_path.append(":main.mdl");
    } else {
        canonical_file_path += ".mdl";
        UDIM_mode udim_mode = NO_UDIM;
        resolved_file_path = consider_search_paths(
            canonical_file_path, /*is_resource=*/false, module_name, udim_mode);
        MDL_ASSERT(udim_mode == NO_UDIM && "resolved modules should not be file masks");
        if (resolved_file_path.empty()) {
            MDL_ASSERT(!"open called on non-existing module");
            return NULL;
        }
    }

    if (strcmp(module_name, m_repl_module_name.c_str()) == 0) {
        // do the replacement
        resolved_file_path = m_repl_file_name;
    }

    MDL_zip_container_error_code err = EC_OK;
    File_handle *file = File_handle::open(m_alloc, resolved_file_path.c_str(), err);
    if (file == NULL) {
        handle_file_error(err);
        return NULL;
    }

    Allocator_builder builder(m_alloc);

    switch (file->get_kind()) {
    case File_handle::FH_FILE:
        return builder.create<Simple_file_input_stream>(
            m_alloc, file, resolved_file_path.c_str());

    case File_handle::FH_ARCHIVE:
        {
            MDL_zip_container_archive *archive = static_cast<MDL_zip_container_archive *>(
                file->get_container());
            mi::base::Handle<Manifest const> manifest(archive->get_manifest());
            return builder.create<Archive_input_stream>(
                m_alloc, file, resolved_file_path.c_str(), manifest.get());
        }

    case File_handle::FH_MDLE:
        return builder.create<Mdle_input_stream>(
            m_alloc, file, resolved_file_path.c_str());

    default:
        return NULL;
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

// Read a memory block from the resource.
Uint64 Buffered_archive_resource_reader::read(void *ptr, Uint64 size)
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
        return m_file->get_container_file()->read((char *)ptr + prefix_size, size) + prefix_size;
    }

    // fill the buffer
    m_curr_pos = 0;
    zip_int64_t read_bytes = m_file->get_container_file()->read(m_buffer, sizeof(m_buffer));
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

// Get the current position.
Uint64 Buffered_archive_resource_reader::tell()
{
    return m_file->get_container_file()->tell()- (m_buf_size - m_curr_pos);
}

// Reposition stream position indicator.
bool Buffered_archive_resource_reader::seek(Sint64 offset, Position origin)
{
    // drop buffer
    m_curr_pos = m_buf_size = 0;

    // now seek
    return m_file->get_container_file()->seek(offset, origin) != 0;
}

// Get the UTF8 encoded name of the resource on which this reader operates.
char const *Buffered_archive_resource_reader::get_filename() const
{
    return m_file_name.empty() ? NULL : m_file_name.c_str();
}

// Get the UTF8 encoded absolute MDL resource name on which this reader operates.
char const *Buffered_archive_resource_reader::get_mdl_url() const
{
    return m_mdl_url.empty() ? NULL : m_mdl_url.c_str();
}

// Returns the associated hash of this resource.
bool Buffered_archive_resource_reader::get_resource_hash(unsigned char hash[16])
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
Buffered_archive_resource_reader::Buffered_archive_resource_reader(
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

// Destructor.
Buffered_archive_resource_reader::~Buffered_archive_resource_reader()
{
    File_handle::close(m_file);
}

// --------------------------------------------------------------------------

// Constructor from one file name/url pair (typical case).
MDL_resource_set::MDL_resource_set(
    IAllocator *alloc,
    char const *url,
    char const *filename,
    unsigned char const hash[16])
: Base(alloc)
, m_arena(alloc)
, m_entries(
    1,
    Resource_entry(Arena_strdup(m_arena, url), Arena_strdup(m_arena, filename), 0, 0, hash),
    &m_arena)
, m_udim_mode(NO_UDIM)
, m_url_mask(url, alloc)
, m_filename_mask(filename, alloc)
{
}

// Empty Constructor from masks.
MDL_resource_set::MDL_resource_set(
    IAllocator *alloc,
    UDIM_mode  udim_mode,
    char const *url_mask,
    char const *filename_mask)
: Base(alloc)
, m_arena(alloc)
, m_entries(&m_arena)
, m_udim_mode(udim_mode)
, m_url_mask(url_mask, alloc)
, m_filename_mask(filename_mask, alloc)
{
}

// Create a resource set from a file mask.
MDL_resource_set *MDL_resource_set::from_mask(
    IAllocator *alloc,
    char const *url,
    char const *file_mask,
    UDIM_mode  udim_mode)
{
    char const *p = strstr(file_mask, ".mdr:");
    if (p != NULL) {
        string container_name(file_mask, p + 4, alloc);
        p += 5;
        return from_mask_container(
            alloc, url, container_name.c_str(), File_handle::FH_ARCHIVE, p, udim_mode);
    }

    p = strstr(file_mask, ".mdle:");
    if (p != NULL) {
        string container_name(file_mask, p + 5, alloc);
        p += 6;
        return from_mask_container(
            alloc, url, container_name.c_str(), File_handle::FH_MDLE, p, udim_mode);
    }

    return from_mask_file(alloc, url, file_mask, udim_mode);
}

// Create a resource set from a file mask describing files on disk.
MDL_resource_set *MDL_resource_set::from_mask_file(
    IAllocator *alloc,
    char const *url,
    char const *file_mask,
    UDIM_mode  udim_mode)
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
    case NO_UDIM:
        MDL_ASSERT(!"UDIM mode not set");
        return NULL;
    case UM_MARI:
        // UDIM (Mari), expands to four digits calculated as 1000+(u+1+v*10)
        p = strstr(file_mask, "[0-9][0-9][0-9][0-9]");
        q = strstr(url, UDIM_MARI_MARKER);
        break;
    case UM_ZBRUSH:
        // 0-based (Zbrush), expands to "_u0_v0" for the first tile
        p = strstr(file_mask, "_u-?[0-9]+_v-?[0-9]+");
        q = strstr(url, UDIM_ZBRUSH_MARKER);
        break;
    case UM_MUDBOX:
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

    MDL_resource_set *s = builder.create<MDL_resource_set>(alloc, udim_mode, url, file_mask);

    for (char const *entry = dir.read(); entry != NULL; entry = dir.read()) {
        if (utf8_match(file_mask, entry)) {
            string purl(url, alloc);

            if (q != NULL) {
                // also patch the URL if possible
                purl = string(url, q - url, alloc);
                purl += entry + ofs;
            }

            // so far no hashes for files
            parse_u_v(s, entry, ofs, purl.c_str(), dname, sep, udim_mode, NULL);
        }
    }
    return s;
}

// Create a resource set from a file mask describing files on an archive.
MDL_resource_set *MDL_resource_set::from_mask_container(
    IAllocator        *alloc,
    char const        *url,
    char const        *container_name,
    File_handle::Kind container_kind,
    char const        *file_mask,
    UDIM_mode         udim_mode)
{
    MDL_zip_container_error_code err = EC_OK;
    MDL_zip_container *container = NULL;
    switch (container_kind) {
    case File_handle::FH_ARCHIVE:
        container = MDL_zip_container_archive::open(alloc, container_name, err);
        break;
    case File_handle::FH_MDLE:
        container = MDL_zip_container_mdle::open(alloc, container_name, err);
        break;
    default:
        break;
    }

    if (container != NULL) {
        char const *p = NULL;
        char const *q = NULL;

        switch (udim_mode) {
        case NO_UDIM:
            MDL_ASSERT(!"UDIM mode not set");
            return NULL;
        case UM_MARI:
            // UDIM (Mari), expands to four digits calculated as 1000+(u+1+v*10)
            p = strstr(file_mask, "[0-9][0-9][0-9][0-9]");
            q = strstr(url, UDIM_MARI_MARKER);
            break;
        case UM_ZBRUSH:
            // 0-based (Zbrush), expands to "_u0_v0" for the first tile
            p = strstr(file_mask, "_u-?[0-9]+_v-?[0-9]+");
            q = strstr(url, UDIM_ZBRUSH_MARKER);
            break;
        case UM_MUDBOX:
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

        MDL_resource_set *s = builder.create<MDL_resource_set>(alloc, udim_mode, url, file_mask);

        // ZIP uses '/'
        string forward(file_mask, alloc);
        forward = convert_os_separators_to_slashes(forward);

        string container_name_str(container_name, alloc);

        for (int i = 0, n = container->get_num_entries(); i < n; ++i) {
            char const *file_name = container->get_entry_name(i);

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

                bool has_hash = false;
                unsigned char hash[16];

                if (container->has_resource_hashes()) {
                    if (MDL_zip_container_file *z_f = container->file_open(file_name)) {
                        size_t length = 0;
                        unsigned char const *stored_hash =
                            z_f->get_extra_field(MDLE_EXTRA_FIELD_ID_MD, length);

                        if (stored_hash != NULL && length == 16) {
                            memcpy(hash, stored_hash, 16);
                            has_hash = true;
                        }
                        z_f->close();
                    }
                }

                parse_u_v(
                    s, fname.c_str(), ofs, purl.c_str(), container_name_str, ':', udim_mode,
                    has_hash ? hash : NULL);
            }
        }

        container->close();
        return s;
    }
    return NULL;
}

// Parse a file name and enter it into a resource set.
void MDL_resource_set::parse_u_v(
    MDL_resource_set *s,
    char const       *name,
    size_t           ofs,
    char const       *url,
    string const     &prefix,
    char             sep,
    UDIM_mode        udim_mode,
    unsigned char    hash[16])
{
    int u = 0, v = 0, sign = 1;
    switch (udim_mode) {
    case NO_UDIM:
        break;
    case UM_MARI:
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
                    v,
                    hash
                )
            );
        }
        break;
    case UM_ZBRUSH:
        // 0-based (Zbrush), expands to "_u0_v0" for the first tile
    case UM_MUDBOX:
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

            if (udim_mode == UM_MUDBOX) {
                u -= 1;
                v -= 1;
            }

            s->m_entries.push_back(
                Resource_entry(
                    Arena_strdup(s->m_arena, url),
                    Arena_strdup(s->m_arena, (prefix + sep + name).c_str()),
                    u,
                    v,
                    hash
                )
            );
        }
        break;
    }
}

// Get the MDL URL mask of the ordered set.
char const *MDL_resource_set::get_mdl_url_mask() const
{
    return m_url_mask.c_str();
}

// Get the file name mask of the ordered set.
char const *MDL_resource_set::get_filename_mask() const
{
    return m_filename_mask.c_str();
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
    if (m_udim_mode != NO_UDIM) {
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
        MDL_zip_container_error_code err = EC_OK;
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

// Get the UDIM mode for this set.
UDIM_mode MDL_resource_set::get_udim_mode() const
{
    return m_udim_mode;
}

// Get the resource hash value for the i'th file in the set if any.
bool MDL_resource_set::get_resource_hash(
    size_t i,
    unsigned char hash[16]) const
{
    if (i < m_entries.size()) {
        if (m_entries[i].has_hash) {
            memcpy(hash, m_entries[i].hash, sizeof(m_entries[i].hash));
            return true;
        }
    }
    return false;
}

// --------------------------------------------------------------------------

// Constructor.
MDL_import_result::MDL_import_result(
    IAllocator   *alloc,
    string const &abs_name,
    string const &os_file_name)
: Base(alloc)
, m_abs_name(abs_name)
, m_os_file_name(os_file_name)
{
}

// Return the absolute MDL name of the found entity, or NULL, if the entity could not be resolved.
char const *MDL_import_result::get_absolute_name() const
{
    return m_abs_name.empty() ? NULL : m_abs_name.c_str();
}

// Return the OS-dependent file name of the found entity, or NULL, if the entity could not
// be resolved.
char const *MDL_import_result::get_file_name() const
{
    return m_os_file_name.empty() ? NULL : m_os_file_name.c_str();
}

// Return an input stream to the given entity if found, NULL otherwise.
IInput_stream *MDL_import_result::open(IThread_context *context) const
{
    Thread_context *ctx = impl_cast<Thread_context>(context);

    if (m_os_file_name.empty())
        return NULL;

    IAllocator *alloc = get_allocator();

    string resolved(m_os_file_name);

    size_t l = m_os_file_name.size();
    if (l > 5 &&
            m_os_file_name[l - 1] == 'e' &&
            m_os_file_name[l - 2] == 'l' &&
            m_os_file_name[l - 3] == 'd' &&
            m_os_file_name[l - 4] == 'm' &&
            m_os_file_name[l - 5] == '.') {
        resolved.append(":main.mdl");
    }

    MDL_zip_container_error_code  err;
    File_handle *file = File_handle::open(alloc, resolved.c_str(), err);

    Position_impl zero(0, 0, 0, 0);
    switch (err) {
        case EC_OK:
            break;

        // load a pre-released version (0.2) will probably get an error at some point in time
        case EC_PRE_RELEASE_VERSION:
        {
            string msg(ctx->access_messages_impl().format_msg(
                MDLE_PRE_RELEASE_VERSION, 'E', Error_params(alloc).add(m_os_file_name)));
            ctx->access_messages_impl().add_warning_message(
                MDLE_PRE_RELEASE_VERSION, 'E', 0, &zero, msg.c_str());
            break;
        }

        case EC_CRC_ERROR:
        {
            string msg(ctx->access_messages_impl().format_msg(
                MDLE_CRC_ERROR, 'E', Error_params(alloc).add(m_os_file_name)));
            ctx->access_messages_impl().add_error_message(
                MDLE_CRC_ERROR, 'E', 0, &zero, msg.c_str());
            break;
        }

        default:
        {
            string msg(ctx->access_messages_impl().format_msg(
                MDLE_IO_ERROR, 'E', Error_params(alloc).add(m_os_file_name)));
            ctx->access_messages_impl().add_error_message(
                MDLE_IO_ERROR, 'E', 0, &zero, msg.c_str());
            break;
        }
    }

    if (file == NULL) {
        return NULL;
    }

    Allocator_builder builder(alloc);

    switch (file->get_kind()) {
    case File_handle::FH_FILE:
        return builder.create<Simple_file_input_stream>(
            alloc, file, resolved.c_str());

    case File_handle::FH_ARCHIVE:
        {
            MDL_zip_container_archive *archive = static_cast<MDL_zip_container_archive *>(
                file->get_container());
            mi::base::Handle<Manifest const> manifest(archive->get_manifest());
            return builder.create<Archive_input_stream>(
                alloc, file, resolved.c_str(), manifest.get());
        }

    case File_handle::FH_MDLE:
        return builder.create<Mdle_input_stream>(
            alloc, file, resolved.c_str());

    default:
        return NULL;
    }
}

// --------------------------------------------------------------------------

// Constructor.
Entity_resolver::Entity_resolver(
    IAllocator *alloc,
    MDL const                                *compiler,
    IModule_cache                            *module_cache,
    mi::base::Handle<IEntity_resolver> const &external_resolver,
    mi::base::Handle<IMDL_search_path> const &search_path)
: Base(alloc)
, m_msg_list(alloc, /*owner_fname=*/"")
, m_resolver(
    *compiler,
    module_cache,
    external_resolver,
    search_path,
    compiler->get_search_path_lock(),
    m_msg_list,
    /*front_path=*/NULL)
{
}

// Resolve a resource file name.
IMDL_resource_set *Entity_resolver::resolve_resource_file_name(
    char const     *file_path,
    char const     *owner_file_path,
    char const     *owner_name,
    Position const *pos)
{
    if (!file_path)
        return NULL;

    Position_impl zero(0, 0, 0, 0);

    if (pos == NULL)
        pos = &zero;

    m_msg_list.clear();

    return m_resolver.resolve_resource(
        *pos,
        file_path,
        owner_name,
        owner_file_path);
}

// Resolve a module name.
IMDL_import_result *Entity_resolver::resolve_module(
    char const     *mdl_name,
    char const     *owner_file_path,
    char const     *owner_name,
    Position const *pos)
{
    Position_impl zero(0, 0, 0, 0);

    m_msg_list.clear();

    Allocator_builder builder(get_allocator());

    return m_resolver.resolve_import(
        pos == NULL ? zero : *pos,
        mdl_name,
        owner_name,
        owner_file_path);
}

// Access messages of last resolver operation.
Messages const &Entity_resolver::access_messages() const
{
    return m_msg_list;
}

// Open a resource file read-only.
IMDL_resource_reader *open_resource_file(
    IAllocator                    *alloc,
    char const                    *abs_mdl_path,
    char const                    *resource_path,
    MDL_zip_container_error_code  &err)
{
    File_handle *file = File_handle::open(alloc, resource_path, err);
    if (file == NULL) {
        return NULL;
    }

    Allocator_builder builder(alloc);

    if (file->get_kind() == File_handle::FH_FILE) {
        return builder.create<File_resource_reader>(
            alloc, file, resource_path, abs_mdl_path);
    } else {
        return builder.create<MDL_zip_resource_reader>(
            alloc, file, resource_path, abs_mdl_path);
    }
}

}  // mdl
}  // mi
