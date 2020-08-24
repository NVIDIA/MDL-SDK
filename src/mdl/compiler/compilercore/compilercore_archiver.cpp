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

#include "compilercore_file_utils.h"
#include "compilercore_mdl.h"
#include "compilercore_errors.h"
#include "compilercore_streams.h"
#include "compilercore_tools.h"
#include "compilercore_archiver.h"
#include "compilercore_manifest.h"
#include "compilercore_file_resolution.h"
#include "compilercore_wchar_support.h"

namespace mi {
namespace mdl {

namespace {

/// Check if the given string ends with a suffix.
bool has_suffix(string const &str, char const *suffix)
{
    size_t l = strlen(suffix);
    return str.size() >= l && str.compare(str.size() - l, l, suffix) == 0;
}

static const MDL_zip_container_header header_supported_read_version = MDL_zip_container_header(
    "MDR", 3,   // prefix
    1,          // major version
    0);         // minor version

static const MDL_zip_container_header header_write_version = MDL_zip_container_header(
    "MDR", 3,   // prefix
    1,          // major version
    0);         // minor version


/// Base class for Archive operators.
class Archive_helper {
protected:
    /// Constructor.
    ///
    /// \param alloc      the allocator
    /// \param arc_tool   the archive tool to append messages
    /// \param cb         the event callback if any
    Archive_helper(
        IAllocator          *alloc,
        Archive_tool        &arc_tool,
        IArchive_tool_event *cb)
    : m_alloc(alloc)
    , m_arc_tool(arc_tool)
    , m_cb(cb)
    , m_archive_name(alloc)
    , m_has_error(false)
    {
    }

public:
    /// Get the allocator.
    IAllocator *get_allocator() const { return m_alloc; }

protected:
    /// Copy all messages from the given message list.
    void copy_messages(Messages const &src) {
        m_arc_tool.copy_messages(src);
    }

    /// Translate zip errors.
    void translate_zip_error(int zip_error);

    /// Translate zip errors
    void translate_zip_error(zip_t *za);

    /// Translate zip errors
    void translate_zip_error(zip_error_t const &ze);

    /*
    // Translate zip errors.
    void translate_zip_error(zip_source_t *src);
    */

    // Translate zip errors.
    void translate_zip_error(zip_file_t *src);

    /// Fire an event.
    void fire_event(IArchive_tool_event::Event ev, char const *name) {
        if (m_cb != NULL)
            m_cb->fire_event(ev, name);
    }

    /// Creates a new error.
    ///
    /// \param code    the error code
    /// \param params  additional parameters
    void error(int code, Error_params const &params) { m_arc_tool.error(code, params); }

    /// Creates a new warning.
    ///
    /// \param code    the error code
    /// \param params  additional parameters
    void warning(int code, Error_params const &params) { m_arc_tool.warning(code, params); }

    /// Adds a note.
    ///
    /// \param code    the error code
    /// \param params  additional parameters
    void add_note(int code, Error_params const &params) { m_arc_tool.add_note(code, params); }

    /// Set the archive name (for error reporting).
    void set_archive_name(string const &name) { m_archive_name = name; }

    /// Get the name of the current archive.
    string const &get_archive_name() const { return m_archive_name; }

protected:
    /// The allocator.
    IAllocator *m_alloc;

    /// The archive tool to append messages on.
    Archive_tool &m_arc_tool;

    /// The event callback if any.
    IArchive_tool_event *m_cb;

    /// The archive name once set.
    string m_archive_name;

    /// Set to true if a zip operation failed.
    bool m_has_error;
};

typedef hash_set<string, string_hash<string> >::Type String_set;

/// Helper class for the archive builder.
class Archive_builder : public Archive_helper {
    typedef Archive_helper Base;

    typedef hash_map<string, Semantic_version, string_hash<string> >::Type Dependency_map;

public:
    /// Constructor.
    ///
    /// \param compiler           the MDL core compiler
    /// \param arc_tool           the archive tool to append messages
    /// \param dest_path          the destination path
    /// \param overwrite          true, if files can be overwritten
    /// \param allow_extra_files  true, if extra files in the source directories are allowed
    /// \param manifest           the Manifest
    /// \param cb                 the event callback if any
    Archive_builder(
        MDL                 *compiler,
        Archive_tool        &arc_tool,
        char const          *dest_path,
        bool                overwrite,
        bool                allow_extra_files,
        Manifest            *manifest,
        IArchive_tool_event *cb);

    /// Add a compressed resource suffix.
    ///
    /// \param suffix  the suffix WITHOUT leading '.'
    void add_compressed_resource_suffix(char const *suffix);

    /// Collect all files.
    ///
    /// \param root_path     The path to the root of all files to store into the archive.
    /// \param root_package  The root package of the archive
    bool collect(
        char const *root_path,
        char const *root_package);

    /// Compile all collected modules.
    bool compile_modules();

    /// Get the name of the archive.
    string const &get_archive_name() const { return m_archive_name; }

    /// Create the archive.
    bool create_zip_archive();

private:
    /// Check the content of a directory.
    bool check_content(
        Directory    &dir,
        string const &package,
        bool         package_mdl_allowed);

    /// Collect files from a directory.
    ///
    /// \param dir                the directory
    /// \param path               the path so far
    /// \param valid_mdl_package  true if this directory is a valid MDL package
    /// \param top_package        if non-empty, the name of the top level package
    void collect_from_dir(
        Directory    &dir,
        string const &path,
        bool         valid_mdl_package,
        string const &top_package);

    /// Convert the given name with '/' separators to a module name
    string convert_to_module_name(
        string const &name);

    /// Lower suffix.
    static void lower_suffix(string &suffix);

    /// Check if a resource name has one on the known file extensions that should be compressed.
    bool should_be_compressed(string const &fname) const;

    /// Update the current manifest from the given module.
    ///
    /// \param[in]  mod       current module
    /// \param[in]  mod_name  the absolute module name
    /// \param[out] dep_map   the dependency map to be updated
    void update_manifest(
        Module const    *mod,
        string const    &mod_name,
        Dependency_map  &dep_map);

private:
    /// The compiler.
    MDL *m_compiler;

    /// The name of the archive that will be created.
    string m_archive_name;

    /// The directory prefix to every file in the archive.
    string m_arch_prefix;

    /// The root_path to root package.
    string m_root_path;

    /// The destination path.
    string m_dest_path;

    typedef list<string>::Type String_list;

    /// List of directories to include.
    String_list m_directory_list;

    /// List of modules to include.
    String_list m_module_list;

    /// List of resources to include.
    String_list m_resource_list;

    /// The current Manifest.
    mi::base::Handle<Manifest> m_manifest;

    typedef hash_set<string, string_hash<string> >::Type Suffix_set;

    /// The set of always uncompressed resource suffixes.
    Suffix_set m_uncompressed_suffix_set;

    /// The set of compressed resource suffixes.
    Suffix_set m_compressed_suffix_set;

    struct Ignored_file {
        Ignored_file(string const &d, string const &f)
        : directory(d), filename(f)
        {
        }

        string directory;
        string filename;
    };

    typedef list<Ignored_file>::Type Ignored_list;

    /// The list of ignored files.
    Ignored_list m_ignored_files;

    /// Set to true if files can be overwritten.
    bool const m_overwrite;

    /// Set to true if extra files in the source directory are allowed (and ignored).
    bool const m_allow_extra_files;

    /// Set if a package or module name was found that requires at least MDL 1.6.
    bool m_mdl_16_names;
};

/// Helper class for extracting an archive.
class Archive_extractor : public Archive_helper {
    typedef Archive_helper Base;
public:
    /// Constructor.
    ///
    /// \param alloc      the allocator
    /// \param arc_tool   the archive tool to append messages
    /// \param dest_path  the destination path
    /// \param overwrite  true, if files can be overwritten
    /// \param cb         the event callback if any
    Archive_extractor(
        IAllocator          *alloc,
        Archive_tool        &arc_tool,
        char const          *dest_path,
        bool                overwrite,
        IArchive_tool_event *cb);

    /// Extract an archive.
    ///
    /// \param archive_name  the file name of the MDR archive
    void extract(
        char const *archive_name);

    /// Get the content of a file into a memory buffer.
    ///
    /// \param archive_name  the file name of the MDR archive
    /// \param file_name     the name of the file inside the archive
    IMDL_resource_reader *get_content_buffer(
        char const *archive_name,
        char const *file_name);

private:
    /// Make directories.
    void mkdir(string const &path);

    /// Copy data.
    void copy_data(FILE *dst, zip_file_t *src);

    /// Normalize separators to '/'.
    void normalize_separators(string &path);

    /// Map a file error.
    void handle_file_error(
        MDL_zip_container_error_code   err,
        char const                     *archive_name,
        char const                     *file_name);

private:
    /// Set to true if files can be overwritten.
    bool const m_overwrite;

    /// The destination path.
    string m_dest_path;
};

/// Helper class to build a manifest.
class Manifest_builder {
public:
    /// Constructor.
    ///
    /// \param alloc     the allocator
    /// \param manifest  the manifest to fill up
    Manifest_builder(
        IAllocator *alloc,
        Manifest   &manifest);

    /// Add a Key, value pair.
    ///
    /// \param key    the key
    /// \param value  the value
    void add_pair(u32string const &key, u32string const &value);

    /// Check existence of mandatory fields.
    void check_mandatory_fields_existence();

    /// Return number of errors.
    size_t get_error_count() const;

private:
    /// Parse an export.
    ///
    /// \param kind  the export kind
    /// \param exp   the export
    void parse_export(IArchive_manifest::Export_kind kind, string const &exp);

    /// Parse a module export.
    ///
    /// \param exp   the export string
    void parse_module(string const &exp);

    /// Parse a dependency.
    ///
    /// \param dep  the dependency string
    void parse_dependency(string const &dep);

    /// Parse a semantic version.
    Semantic_version parse_sema_version(string const &s);

    /// Check the time format.
    void check_time_format(string const &s);

    enum Error_code {
        EC_OK = 0,
        EC_MDL_FIELD_MISSING,          ///< The "mdl" field must exists.
        EC_MULTIPLE_MDL_FIELD,         ///< The "mdl" field must be used only once.
        EC_UNSUPPORTED_MDL_VERSION,    ///< Unsupported MDL version.
        EC_VERSION_FIELD_MISSING,      ///< The "version" field must exists.
        EC_MULTIPLE_VERSION_FIELD,     ///< The "version" field must be used only once.
        EC_MULTIPLE_COPYRIGHT_FIELD,   ///< The "copyright_notice" field must be used only once.
        EC_MULTIPLE_CREATED_FIELD,     ///< The "created" field must be used only once.
        EC_MULTIPLE_MODIFIED_FIELD,    ///< The "modified" field must be used only once.
        EC_MULTIPLE_DESCRIPTION_FIELD, ///< The "description" field must be used only once.
        EC_INVALID_EXPORT,             ///< A qualified export name is invalid.
        EC_INVALID_DEPENDENCY,         ///< A dependency is invalid.
        EC_INVALID_TIME,               ///< A time entry is invalid.
    };

    /// Creates an error.
    void error(Error_code code);

private:
    /// The allocator.
    IAllocator *m_alloc;

    /// The manifest.
    Manifest &m_manifest;

    typedef hash_map<string, size_t, string_hash<string> >::Type Module_hash;

    /// Map module names to indexes.
    Module_hash m_modules;

    enum Mandatory_fields {
        MF_MDL              = (1 << 0),
        MF_VERSION          = (1 << 1),
        MF_MODULE           = (1 << 2),
        MF_DEPENDENCY       = (1 << 3),
        MF_COPYRIGHT_NOTICE = (1 << 4),
        MF_CREATED          = (1 << 5),
        MF_MODIFIED         = (1 << 6),
        MF_DESCRIPTION      = (1 << 7),
    };

    /// Bitset of mandatory fields seen.
    unsigned m_seen_fields;

    /// Current error.
    Error_code m_error;

    /// Temporary buffer for version prerelease strings.
    string m_tmp;
};

enum Token_kind {
    TK_ID,      ///< Identifier.
    TK_DOT,     ///< '.'
    TK_EQUAL,   ///< '='
    TK_LIT,     ///< String literal.
    TK_ERROR,   ///< Unsupported character.
    TK_EOF,     ///< End of file.
};

/// Scanner class for the archive manifest.
class Manifest_scanner {
public:
    /// Constructor.
    ///
    /// \param alloc   the allocator
    /// \param is      the input stream to read from
    Manifest_scanner(
        IAllocator                      *alloc,
        mi::base::Handle<IInput_stream> &is)
    : m_is(is)
    , m_eof(false)
    , m_r_pos(0)
    , m_w_pos(0)
    , m_column(0)
    , m_curr_column(0)
    , m_line(0)
    , m_curr_line(0)
    , m_text(alloc)
    {
        m_c = get_unicode_char();

        // get the first token
        next();
    }

    /// Get the current token.
    Token_kind current_token() { return m_token; }

    /// Move to next token.
    Token_kind next() { m_token = next_token(); return m_token; }

    /// Get the token text.
    u32string const &token_value() { return m_text; }

    /// Get the column of the token.
    unsigned token_column() { return m_column; }

    /// Get the line of the token.
    unsigned token_line() { return m_line; }

private:
    /// Lookup of the next four bytes from the input stream.
    ///
    /// \param ofs  current offset, 0-3 possible
    unsigned char look_byte(unsigned ofs);

    /// Get the next unicode char from the input stream.
    unsigned get_unicode_char();

    /// Get the size of the ring buffer.
    unsigned size() const { return (m_w_pos - m_r_pos) & 7u;}

    /// Throw bytes from the ring buffer.
    void throw_bytes(unsigned count);

    /// Get the next token.
    Token_kind next_token();

private:
    /// The input stream.
    mi::base::Handle<IInput_stream> &m_is;

    /// The current token.
    Token_kind m_token;

    /// Current token text.

    /// Set if eof was reached.
    bool m_eof;

    /// Lookup ring buffer for the utf8 decoder.
    unsigned char m_buffer[8];

    /// Ring buffer read position.
    unsigned m_r_pos;

    /// Ring buffer write position.
    unsigned m_w_pos;

    /// Column of the current token.
    unsigned m_column;

    /// Column of the current UTF-8 character.
    unsigned m_curr_column;

    /// Line of the current token.
    unsigned m_line;

    /// Line of the current UTF-8 character.
    unsigned m_curr_line;

    /// The text of the current token.
    u32string m_text;

    /// Current character read.
    unsigned m_c;
};

/// Parser class for the archive manifest.
class Manifest_parser {
public:
    /// Constructor.
    ///
    /// \param alloc     the allocator
    /// \param scanner   the scanner
    /// \param manifest  the manifest to fill
    Manifest_parser(
        IAllocator       *alloc,
        Manifest_scanner &scanner,
        Manifest_builder &manifest)
    : m_scanner(scanner)
    , m_manifest_builder(manifest)
    , m_full_key(alloc)
    , m_full_value(alloc)
    , m_syntax_errors(false)
    {
    }

    /// Parse.
    bool parse();

private:
    /// Parse a key value pair.
    void pair();

    /// Parse a key.
    void key();

    /// generate an error.
    void error();

private:
    /// The scanner.
    Manifest_scanner &m_scanner;

    /// The manifest builder.
    Manifest_builder &m_manifest_builder;

    /// The combined key.
    u32string m_full_key;

    /// The combined value.
    u32string m_full_value;

    /// True if syntax errors occurred.
    bool m_syntax_errors;
};

// ------------------------------------ helper ------------------------------------

// Translate zip errors.
void Archive_helper::translate_zip_error(int zip_error)
{
    switch (zip_error) {
    case ZIP_ER_MULTIDISK:
        /* N Multi-disk zip archives not supported */
        error(
            INVALID_MDL_ARCHIVE,
            Error_params(m_alloc).add(get_archive_name().c_str()));
        break;

    case ZIP_ER_RENAME:
        /* S Renaming temporary file failed */
        error(RENAME_FAILED, Error_params(m_alloc));
        break;

    case ZIP_ER_CLOSE:
        /* S Closing zip archive failed */
    case ZIP_ER_SEEK:
        /* S Seek error */
    case ZIP_ER_READ:
        /* S Read error */
    case ZIP_ER_WRITE:
        /* S Write error */
        error(IO_ERROR, Error_params(m_alloc).add(get_archive_name().c_str()));
        break;

    case ZIP_ER_CRC:
        /* N CRC error */
        error(CRC_ERROR, Error_params(m_alloc).add(get_archive_name().c_str()));
        break;

    case ZIP_ER_ZIPCLOSED:
        /* N Containing zip archive was closed */
        error(INTERNAL_ARCHIVER_ERROR, Error_params(m_alloc));
        break;

    case ZIP_ER_NOENT:
        /* N No such file */
        error(
            ARCHIVE_DOES_NOT_EXIST,
            Error_params(m_alloc).add(get_archive_name().c_str()));
        break;

    case ZIP_ER_EXISTS:
        /* N File already exists */
        error(
            ARCHIVE_ALREADY_EXIST,
            Error_params(m_alloc).add(get_archive_name().c_str()));
        break;

    case ZIP_ER_OPEN:
        /* S Can't open file */
        error(
            CANT_OPEN_ARCHIVE,
            Error_params(m_alloc).add(get_archive_name().c_str()));
        break;

    case ZIP_ER_TMPOPEN:
        /* S Failure to create temporary file */
        error(FAILED_TO_OPEN_TEMPFILE, Error_params(m_alloc));
        break;

    case ZIP_ER_ZLIB:
        /* Z Zlib error */
        error(
            INVALID_MDL_ARCHIVE,
            Error_params(m_alloc).add(get_archive_name().c_str()));
        break;

    case ZIP_ER_MEMORY:
        /* N Malloc failure */
        error(MEMORY_ALLOCATION, Error_params(m_alloc));
        break;

    case ZIP_ER_COMPNOTSUPP:
        /* N Compression method not supported */
        error(
            INVALID_MDL_ARCHIVE,
            Error_params(m_alloc).add(get_archive_name().c_str()));
        break;

    case ZIP_ER_EOF:
        /* N Premature end of file */
        error(
            INVALID_MDL_ARCHIVE,
            Error_params(m_alloc).add(get_archive_name().c_str()));
        break;

    case ZIP_ER_NOZIP:
        /* N Not a zip archive */
        error(
            INVALID_MDL_ARCHIVE,
            Error_params(m_alloc).add(get_archive_name().c_str()));
        break;

    case ZIP_ER_INCONS:
        /* N Zip archive inconsistent */
        error(
            INVALID_MDL_ARCHIVE,
            Error_params(m_alloc).add(get_archive_name().c_str()));
        break;

    case ZIP_ER_REMOVE:
        /* S Can't remove file */
        error(FAILED_TO_REMOVE, Error_params(m_alloc));
        break;

    case ZIP_ER_ENCRNOTSUPP:
        /* N Encryption method not supported */
        error(
            INVALID_MDL_ARCHIVE,
            Error_params(m_alloc).add(get_archive_name().c_str()));
        break;

    case ZIP_ER_RDONLY:
        /* N Read-only archive */
        error(READ_ONLY_ARCHIVE, Error_params(m_alloc));
        break;

    case ZIP_ER_NOPASSWD:
        /* N No password provided */
    case ZIP_ER_WRONGPASSWD:
        /* N Wrong password provided */
        error(INVALID_PASSWORD, Error_params(m_alloc));
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
        error(INTERNAL_ARCHIVER_ERROR, Error_params(m_alloc));
        break;

    default:
        break;
    }
    m_has_error = true;
}

// Translate zip errors.
void Archive_helper::translate_zip_error(zip_t *za)
{
    translate_zip_error(zip_get_error(za)->zip_err);
}

// Translate zip errors
void Archive_helper::translate_zip_error(zip_error_t const &ze)
{
    translate_zip_error(ze.zip_err);
}

/*
// Translate zip errors.
void Archive_helper::translate_zip_error(zip_source_t *src)
{
    translate_zip_error(*zip_source_error(src));
}
*/

// Translate zip errors.
void Archive_helper::translate_zip_error(zip_file_t *src)
{
    translate_zip_error(*zip_file_get_error(src));
}

// ------------------------------------ builder ------------------------------------

// Constructor.
Archive_builder::Archive_builder(
    MDL                 *compiler,
    Archive_tool        &arc_tool,
    char const          *dest_path,
    bool                overwrite,
    bool                allow_extra_files,
    Manifest            *manifest,
    IArchive_tool_event *cb)
: Base(compiler->get_allocator(), arc_tool, cb)
, m_compiler(compiler)
, m_archive_name(m_alloc)
, m_arch_prefix(m_alloc)
, m_root_path(m_alloc)
, m_dest_path(dest_path, m_alloc)
, m_directory_list(m_alloc)
, m_module_list(m_alloc)
, m_resource_list(m_alloc)
, m_manifest(manifest, mi::base::DUP_INTERFACE)
, m_uncompressed_suffix_set(0, Suffix_set::hasher(), Suffix_set::key_equal(), m_alloc)
, m_compressed_suffix_set(0, Suffix_set::hasher(), Suffix_set::key_equal(), m_alloc)
, m_ignored_files(m_alloc)
, m_overwrite(overwrite)
, m_allow_extra_files(allow_extra_files)
, m_mdl_16_names(false)
{
    // Fill the uncompressed suffix set with known suffixes from the MDL spec that
    // should NEVER be compressed
    m_uncompressed_suffix_set.insert(string("png", m_alloc));
    m_uncompressed_suffix_set.insert(string("exr", m_alloc));
    m_uncompressed_suffix_set.insert(string("jpg", m_alloc));
    m_uncompressed_suffix_set.insert(string("jpeg", m_alloc));
    m_uncompressed_suffix_set.insert(string("ptx", m_alloc));
}

template<typename T>
static void print_list(T const &list, char const *n)
{
    printf("%s:\n", n);
    for (typename T::const_iterator it(list.begin()), end(list.end()); it != end; ++it) {
        string const &e = *it;

        printf("%s\n", e.c_str());
    }
}

// Add a compressed resource suffix.
void Archive_builder::add_compressed_resource_suffix(
    char const *suffix)
{
    if (suffix[0] == '.')
        ++suffix;

    string s(suffix, m_alloc);
    lower_suffix(s);

    m_compressed_suffix_set.insert(s);
}

// Collect all files.
bool Archive_builder::collect(
    char const *root_path,
    char const *root_package)
{
    Directory root_dir(m_alloc);

    m_root_path = root_path;
    if (!is_path_absolute(m_root_path.c_str())) {
        if (m_root_path.size() >= 2) {
            if (m_root_path[0] == '.' && m_root_path[1] == os_separator()) {
                m_root_path = m_root_path.substr(2);
            }
        }
        m_root_path = join_path(get_cwd(m_alloc), m_root_path);
    }

    // compute a canonical path
    m_root_path = simplify_path(
        m_alloc, convert_slashes_to_os_separators(m_root_path), os_separator());

    if (!root_dir.open(m_root_path.c_str())) {
        error(
            DIRECTORY_MISSING,
            Error_params(m_alloc).add(root_path));
        return false;
    }

    string top_path(m_root_path);
    string package_name(m_alloc);

    for (;;) {
        char const *scope = strstr(root_package, "::");
        if (scope == NULL) {
            package_name = root_package;
        } else {
            package_name = string(root_package, scope, m_alloc);
            root_package = scope + 2;

            if (!m_arch_prefix.empty())
                m_arch_prefix.append('/');
            m_arch_prefix.append(package_name);
        }

        if (!m_compiler->is_valid_mdl_identifier(package_name.c_str())) {
            // need at least MDL 1.6 because of non valid MDL identifier
            m_mdl_16_names = true;
        }

        if (!m_archive_name.empty())
            m_archive_name.append('.');
        m_archive_name.append(package_name);

        if (scope != NULL) {
            if (!check_content(root_dir, package_name, /*package_mdl_allowed=*/false)) {
                // error already reported
                return false;
            }
            root_dir.close();
            top_path.append(os_separator());
            top_path.append(package_name);

            if (!root_dir.open(top_path.c_str())) {
                error(
                    DIRECTORY_MISSING,
                    Error_params(m_alloc).add(top_path.c_str()));
                return false;
            }
        } else {
            break;
        }
    }

    m_archive_name.append(".mdr");

    if (!check_content(root_dir, package_name, /*package_mdl_allowed=*/true)) {
        // error already reported
        return false;
    }

    // rewind the directory, we have already read it for the checks
    root_dir.rewind();

    collect_from_dir(root_dir, m_arch_prefix, /*valid_mdl_package=*/true, package_name);

    if (m_resource_list.empty() && m_module_list.empty()) {
        // do not create empty archives.
        error(
            EMPTY_ARCHIVE_CONTENT,
            Error_params(m_alloc).add(package_name.c_str())
        );

        // report which files was ignored
        for (Ignored_list::const_iterator it(m_ignored_files.begin()), end(m_ignored_files.end());
            it != end;
            ++it)
        {
            Ignored_file const &e = *it;
            add_note(
                EXTRA_FILES_IGNORED,
                Error_params(m_alloc).add(e.directory.c_str()).add(e.filename.c_str()));
        }
        return false;
    }

    return true;
}

// Check the content of a directory.
bool Archive_builder::check_content(
    Directory    &dir,
    string const &package,
    bool         package_mdl_allowed)
{
    bool res = true;
    for (char const *entry = dir.read(); entry != NULL; entry = dir.read()) {
        string e(entry, m_alloc);

        if (e == "." || e == "..") {
            // skip special entries
            continue;
        }

        if (e == package)
            continue;
        if (package_mdl_allowed && e == package + ".mdl")
            continue;
        else {
            if (!m_allow_extra_files) {
                // extra files found
                error(
                    EXTRA_FILES_FOUND,
                    Error_params(m_alloc).add(dir.get_curr_dir().c_str()).add(e.c_str()));
                res = false;
            } else {
                m_ignored_files.push_back(Ignored_file(dir.get_curr_dir(), e));
            }
        }
    }
    return res;
}

// Collect files from a directory.
void Archive_builder::collect_from_dir(
    Directory    &dir,
    string const &path,
    bool         valid_mdl_package,
    string const &top_package)
{
    Directory sub(m_alloc);

    for (char const *entry = dir.read(); entry != NULL; entry = dir.read()) {
        string e(entry, m_alloc);

        if (e == "." || e == "..") {
            // skip special entries
            continue;
        }

        string file = path.empty() ? e : path + '/' + e;

        if (!top_package.empty()) {
            if (e != top_package && e != top_package + ".mdl") {
                // ignore extra content
                fire_event(IArchive_tool_event::EV_IGNORED, file.c_str());
                continue;
            }
        }

        string subdir_name = m_root_path + '/' + file;
        if (sub.open(subdir_name.c_str())) {
            // found a subdirectory
            m_directory_list.push_back(file);

            if (!m_compiler->is_valid_mdl_identifier(e.c_str())) {
                m_mdl_16_names = true;
            }

            collect_from_dir(sub, file, valid_mdl_package, string(m_alloc));
            sub.close();
        } else {
            fire_event(IArchive_tool_event::EV_DISCOVERED, file.c_str());
            if (has_suffix(e, ".mdl")) {
                if (!valid_mdl_package) {
                    error(
                        INACCESSIBLE_MDL_FILE,
                        Error_params(m_alloc)
                            .add(path.empty() ? "." : path.c_str())
                            .add(e.c_str()));
                } else {
                    string base_name(e.substr(0, e.length() - 4));

                    if (!m_compiler->is_valid_mdl_identifier(base_name.c_str())) {
                        m_mdl_16_names = true;
                    }
                    // found valid file
                    m_module_list.push_back(file);
                }
            } else {
                m_resource_list.push_back(file);
            }
        }
    }
}

// Convert the given name with '/' separators to a module name
string Archive_builder::convert_to_module_name(
    string const &name)
{
    string res(m_alloc);
    res.reserve(name.length() + 16);

    // strip .mdr and start with "::"
    res = "::";
    for (size_t i = 0, n = name.size() - 4; i < n; ++i) {
        char c = name[i];

        if (c == '/')
            res.append("::");
        else
            res.append(c);
    }
    return res;
}

namespace {

/// Helper class to handle resource restrictions.
class Archive_resource_restrictions : public IResource_restriction_handler
{
public:
    /// Process a referenced resource.
    ///
    /// \param owner  the owner module of the resource
    /// \param url    the URL of a resource
    Resource_restriction process(
        IModule const *owner,
        char const    *url) MDL_OVERRIDE
    {
        if (url == NULL || url[0] == '\0')
            return IResource_restriction_handler::RR_NOT_EXISTANT;

        IAllocator *alloc = m_builder.get_allocator();
        string res(url, alloc);

        if (m_resources.find(res) == m_resources.end()) {
            // this resource was not discovered in the current archive
            return IResource_restriction_handler::RR_OUTSIDE_ARCHIVE;
        }

        return IResource_restriction_handler::RR_OK;
    }

public:
    /// Constructor.
    Archive_resource_restrictions(
        Archive_builder  &builder,
        String_set const &resources)
    : m_builder(builder)
    , m_resources(resources)
    {
    }

private:
    /// The archive builder.
    Archive_builder &m_builder;

    /// The set of all archive resources.
    String_set const &m_resources;
};

}  // anonymous

// Compile all collected modules.
bool Archive_builder::compile_modules()
{
    mi::base::Handle<mi::mdl::Thread_context> ctx(m_compiler->create_thread_context());

    // archives are always build in STRICT mode
    Options &opt = ctx->access_options();
    opt.set_option(MDL::option_strict, "true");

    // add the root path in front, to ensure that modules are first searched here
    ctx->set_front_path(m_root_path.c_str());

    Dependency_map dep_map(0, Dependency_map::hasher(), Dependency_map::key_equal(), m_alloc);

    // prepare the set of all resources
    String_set resources(0, String_set::hasher(), String_set::key_equal(), m_alloc);

    for (String_list::const_iterator it(m_resource_list.begin()), end(m_resource_list.end());
        it != end;
        ++it)
    {
        string res = *it;

        if (!is_path_absolute(res.c_str())) {
            if (res.size() >= 2) {
                if (res[0] == '.' && res[1] == os_separator()) {
                    res = res.substr(2);
                }
            }
            res = join_path(m_root_path, res);
        }

        // compute a canonical path
        res = simplify_path(
            m_alloc, convert_slashes_to_os_separators(res), os_separator());

        resources.insert(res);
    }

    // set the resource restriction handler
    Archive_resource_restrictions rrh(*this, resources);
    ctx->set_resource_restriction_handler(&rrh);

    bool res = true;
    for (String_list::const_iterator it(m_module_list.begin()), end(m_module_list.end());
        it != end;
        ++it)
    {
        string const &entry = *it;

        string mod_name = convert_to_module_name(entry);

        fire_event(IArchive_tool_event::EV_COMPILING, mod_name.c_str());

        mi::base::Handle<mi::mdl::Module const> mod(
            m_compiler->load_module(ctx.get(), mod_name.c_str(), /*module_cache=*/NULL));

        mi::mdl::Messages const &msgs = ctx->access_messages();
        copy_messages(msgs);

        if (msgs.get_error_message_count() > 0) {
            res = false;
        } else {
            update_manifest(mod.get(), mod_name, dep_map);
        }
    }

    // update the dependencies
    for (Dependency_map::const_iterator it(dep_map.begin()), end(dep_map.end()); it != end; ++it) {
        string const           &name = it->first;
        Semantic_version const &ver  = it->second;

        m_manifest->add_dependency(name.c_str(), ver);
    }

    if (m_mdl_16_names) {
        // requires at least MDL 1.6
        m_manifest->add_mdl_version(IMDL::MDL_VERSION_1_6);
    }
    return res;
}

// Lower suffix.
void Archive_builder::lower_suffix(string &suffix)
{
    // lower ASCII-7bit subset: Do not use "tolower" which depends on the current locale AND
    // is NOT utf8 aware
    // So far, we lower the ASCII 7bit subset only, assuming that no-one will use extended suffixes
    for (size_t i = 0, n = suffix.length(); i < n; ++i) {
        unsigned char c = suffix[i];

        if ('A' <= c && c <= 'Z') {
            c = c - 'A' + 'a';
            suffix[i] = c;
        }
    }
}

// Check if a resource name has one on the known file extensions that should be compressed.
bool Archive_builder::should_be_compressed(string const &fname) const
{
    size_t suffix_pos = fname.rfind('.');
    if (suffix_pos == string::npos)
        return false;
    string suffix = fname.substr(suffix_pos + 1);
    lower_suffix(suffix);

    if (m_uncompressed_suffix_set.find(suffix) != m_uncompressed_suffix_set.end()) {
        // never compress
        return false;
    }

    return m_compressed_suffix_set.find(suffix) != m_compressed_suffix_set.end();
}

// Create the archive.
bool Archive_builder::create_zip_archive()
{
    string arc_name = join_path(m_dest_path, get_archive_name());

    set_archive_name(arc_name);

    // create the writable stream
    zip_error_t ze;
    zip_source_t *src = zip_source_file_create(arc_name.c_str(), 0, -1, &ze);
    if (src == NULL) {
        translate_zip_error(ze);
        return false;
    }

    // wrap it by the extra layer that writes our MDR header
    Layered_zip_source layer(src, header_write_version);
    zip_source_t *lsrc = layer.open(ze);
    if (lsrc == 0) {
        zip_source_free(src);
        translate_zip_error(ze);
        return false;
    }

    // create the archive
    zip_t *za = zip_open_from_source(
        lsrc,
        ZIP_CREATE | (m_overwrite ? ZIP_TRUNCATE : ZIP_EXCL),
        &ze);
    if (za == NULL) {
        zip_source_free(lsrc);
        translate_zip_error(ze);
        return false;
    }

    // ensure the life time of the output buffer lasts until zip_close() where the data is written
    Allocator_builder builder(m_alloc);
    mi::base::Handle<Buffer_output_stream> os(builder.create<Buffer_output_stream>(m_alloc));

    // first write the MANIFEST
    if (!m_has_error) {
        mi::base::Handle<Printer> printer(m_compiler->create_printer(os.get()));

        printer->print(m_manifest.get());

        size_t       manifest_len      = os->get_data_size();
        char const   *manifest_content = os->get_data();
        zip_source_t *manifest_src     =
            zip_source_buffer(za, manifest_content, manifest_len, /*freep=*/0);

        fire_event(IArchive_tool_event::EV_STORING, "MANIFEST");

        zip_int64_t index = zip_file_add(za, "MANIFEST", manifest_src, ZIP_FL_ENC_UTF_8);
        if (index < 0) {
            translate_zip_error(za);
        }
        MDL_ASSERT(index == 0);

        if (!m_has_error) {
            // do not compress the MANIFEST
            if (zip_set_file_compression(za, index, ZIP_CM_STORE, 0) != 0) {
                translate_zip_error(zip_get_error(za)->zip_err);
            }
        }
    }

    // add directories
    if (!m_has_error) {
        for (String_list::const_iterator it(m_directory_list.begin()), end(m_directory_list.end());
            it != end;
            ++it)
        {
            string const &entry = *it;

            zip_int64_t index = zip_dir_add(za, entry.c_str(), ZIP_FL_ENC_UTF_8);
            if (index < 0) {
                translate_zip_error(za);
                break;
            }
        }
    }

    // add modules
    if (!m_has_error) {
        for (String_list::const_iterator it(m_module_list.begin()), end(m_module_list.end());
            it != end;
            ++it)
        {
            string const &entry = *it;

            string fname = join_path(m_root_path, entry);

            zip_error_t err;
            zip_source_t *source = zip_source_file_create(fname.c_str(), 0, -1, &err);
            if (source == NULL) {
                translate_zip_error(err);
                break;
            }

            fire_event(IArchive_tool_event::EV_COMPRESSING, entry.c_str());

            zip_int64_t index = zip_file_add(za, entry.c_str(), source, ZIP_FL_ENC_UTF_8);
            if (index < 0) {
                translate_zip_error(za);
                break;
            }
            if (zip_set_file_compression(za, index, ZIP_CM_DEFLATE, 0) != 0) {
                translate_zip_error(zip_get_error(za)->zip_err);
                break;
            }
        }
    }

    // add resources
    if (!m_has_error) {
        for (String_list::const_iterator it(m_resource_list.begin()), end(m_resource_list.end());
            it != end;
            ++it)
        {
            string const &entry = *it;

            string fname = join_path(m_root_path, entry);

            zip_error_t err;
            zip_source_t *source = zip_source_file_create(fname.c_str(), 0, -1, &err);
            if (source == NULL) {
                translate_zip_error(err.zip_err);
                break;
            }

            // do not compress resources by default
            zip_int32_t comp_method = ZIP_CM_STORE;

            if (should_be_compressed(fname))
                comp_method = ZIP_CM_DEFAULT;

            fire_event(
                comp_method == ZIP_CM_STORE ?
                    IArchive_tool_event::EV_STORING :
                    IArchive_tool_event::EV_COMPRESSING,
                entry.c_str());

            zip_int64_t index = zip_file_add(za, entry.c_str(), source, ZIP_FL_ENC_UTF_8);
            if (index < 0) {
                translate_zip_error(zip_get_error(za)->zip_err);
                break;
            }

            if (zip_set_file_compression(za, index, comp_method, 0) != 0) {
                translate_zip_error(zip_get_error(za)->zip_err);
                break;
            }
        }
    }

    if (zip_close(za) != 0)
        translate_zip_error(za);

    if (m_has_error) {
        return false;
    }
    return true;
}

// Update the current manifest from the given module.
void Archive_builder::update_manifest(
    Module const    *mod,
    string const    &mod_name,
    Dependency_map  &dep_map)
{
    m_manifest->add_mdl_version(mod->get_mdl_version());

    size_t id = m_manifest->add_module(mod_name.c_str());

    // check if a module export comes from an archive
    for (int i = 0, n = mod->get_import_count(); i < n; ++i) {
        mi::base::Handle<Module const> imp_mod(mod->get_import(i));

        if (Module::Archive_version const *a_ver = imp_mod->get_owner_archive_version()) {
            string                 name(a_ver->get_name(), m_alloc);
            Semantic_version const &ver = a_ver->get_version();

            dep_map.insert(Dependency_map::value_type(name, ver));
        }
    }

    // Collect exports.
    // Note: we have only one name space in MDL, hence it is not necessary to differentiate
    // between the kinds of the exported entities to figure out which name was already
    // used
    typedef ptr_hash_set<ISymbol const>::Type Symbol_set;
    Symbol_set exported_entities(0, Symbol_set::hasher(), Symbol_set::key_equal(), m_alloc);

    for (int i = 0, n = mod->get_exported_definition_count(); i < n; ++i) {
        Definition const *def = mod->get_exported_definition(i);
        ISymbol const    *sym = def->get_sym();

        if (!exported_entities.insert(sym).second) {
            // already seen
            continue;
        }

        switch (def->get_kind()) {
        case IDefinition::DK_FUNCTION:
            {
                IType_function const *f_tp = cast<IType_function>(def->get_type());

                if (is_material_type(f_tp->get_return_type())) {
                    // exported material
                    m_manifest->add_export(Manifest::EK_MATERIAL, id, sym->get_name());
                } else {
                    // exported function
                    m_manifest->add_export(Manifest::EK_FUNCTION, id, sym->get_name());
                }
            }
            break;

        case IDefinition::DK_TYPE:
            {
                IType const *tp = def->get_type();

                IType::Kind tp_kind = tp->get_kind();
                if (tp_kind == IType::TK_STRUCT) {
                    // found exported struct
                    m_manifest->add_export(Manifest::EK_STRUCT, id, sym->get_name());
                } else if (tp_kind == IType::TK_ENUM) {
                    // found exported enum
                    m_manifest->add_export(Manifest::EK_ENUM, id, sym->get_name());
                }
            }
            break;

        case IDefinition::DK_CONSTANT:
            m_manifest->add_export(Manifest::EK_CONST, id, sym->get_name());
            break;

        case IDefinition::DK_ANNOTATION:
            m_manifest->add_export(Manifest::EK_ANNOTATION, id, sym->get_name());
            break;

        case IDefinition::DK_ERROR:
            // should not occur at this point
            MDL_ASSERT(!"error definition detected in valid module");
            break;
        case IDefinition::DK_ENUM_VALUE:
            // ignored so far
            break;
        case IDefinition::DK_VARIABLE:
        case IDefinition::DK_MEMBER:
        case IDefinition::DK_PARAMETER:
            // cannot be exported
            MDL_ASSERT(!"unexpected definition kind in export list");
            break;
        case IDefinition::DK_CONSTRUCTOR:
        case IDefinition::DK_ARRAY_SIZE:
        case IDefinition::DK_OPERATOR:
        case IDefinition::DK_NAMESPACE:
            // ignored so far
            break;
        }
    }
}

// ------------------------------------ extractor ------------------------------------

// Constructor.
Archive_extractor::Archive_extractor(
    IAllocator          *alloc,
    Archive_tool        &arc_tool,
    char const          *dest_path,
    bool                overwrite,
    IArchive_tool_event *cb)
: Base(alloc, arc_tool, cb)
, m_overwrite(overwrite)
, m_dest_path(dest_path, alloc)
{
    normalize_separators(m_dest_path);
}

// Extract an archive.
void Archive_extractor::extract(
    char const *archive_name)
{
    if (archive_name == NULL) {
        error(
            INVALID_MDL_ARCHIVE_NAME,
            Error_params(m_alloc).add("<NULL>"));
        return;
    }

    size_t l = strlen(archive_name);
    if (l < 4 || strcmp(&archive_name[l - 4], ".mdr") != 0) {
        error(
            INVALID_MDL_ARCHIVE_NAME,
            Error_params(m_alloc).add(archive_name));
        return;
    }

    string arc_name(archive_name, m_alloc);

    set_archive_name(arc_name);

    // create the the stream
    zip_error_t ze;
    zip_error_init(&ze);

    zip_source_t *src = zip_source_file_create(arc_name.c_str(), 0, -1, &ze);
    if (src == NULL) {
        translate_zip_error(ze);
        return;
    }

    // wrap it by the extra layer that writes our MDR header
    Layered_zip_source layer(src, header_write_version);
    zip_source_t *lsrc = layer.open(ze);
    if (lsrc == 0) {
        zip_source_free(src);
        translate_zip_error(ze);
        return;
    }

    // create the archive
    zip_t *za = zip_open_from_source(
        lsrc,
        ZIP_RDONLYNOLASTMOD,
        &ze);
    if (za == NULL) {
        zip_source_free(lsrc);
        translate_zip_error(ze);
        return;
    }

    if (!m_has_error) {
        if (!m_dest_path.empty())
            mkdir(m_dest_path);

        zip_int64_t n = zip_get_num_entries(za, ZIP_FL_UNCHANGED);

        if (n == 0 || strcmp(zip_get_name(za, 0, ZIP_FL_ENC_UTF_8), "MANIFEST") != 0) {
            error(
                INVALID_MDL_ARCHIVE, Error_params(m_alloc).add(archive_name));
            add_note(
                MANIFEST_MISSING, Error_params(m_alloc));
            return;
        }

        // ignore MANIFEST
        for (zip_int64_t i = 1; i < n; ++i) {
            char const *name = zip_get_name(za, i, ZIP_FL_ENC_UTF_8);
            if (name == NULL) {
                // should not happen
                translate_zip_error(za);
                continue;
            }

            string path(name, get_allocator());
            size_t l = path.size();

            bool is_directory = false;
            if (l > 0 && path[l - 1] == '/') {
                // is a directory
                is_directory = true;
                path = path.substr(0, l - 1);
            }

            size_t pos = path.rfind('/');
            if (pos != string::npos) {
                string full_dir(join_path(m_dest_path, path.substr(0, pos)));
                normalize_separators(full_dir);
                mkdir(full_dir);
            }

            if (is_directory)
                continue;

            zip_file_t *zf = zip_fopen_index(za, i, ZIP_FL_UNCHANGED);
            if (zf == NULL) {
                translate_zip_error(za);
                continue;
            }

            string full_path(join_path(m_dest_path, path));
            normalize_separators(full_path);

            FILE *f = fopen(full_path.c_str(), "wb");
            if (f == NULL) {
                zip_fclose(zf);
                error(
                    CREATE_FILE_FAILED,
                    Error_params(get_allocator()).add(full_path.c_str()));
                continue;
            }

            copy_data(f, zf);

            fclose(f);
            zip_fclose(zf);

            fire_event(IArchive_tool_event::EV_EXTRACTED, path.c_str());
        }
    }
    zip_close(za);
}

// Get the content of a file into a memory buffer.
IMDL_resource_reader *Archive_extractor::get_content_buffer(
    char const *archive_name,
    char const *file_name)
{
    if (archive_name == NULL) {
        error(
            INVALID_MDL_ARCHIVE_NAME,
            Error_params(m_alloc).add("<NULL>"));
        return NULL;
    }

    size_t l = strlen(archive_name);
    if (l < 4 || strcmp(&archive_name[l - 4], ".mdr") != 0) {
        error(
            INVALID_MDL_ARCHIVE_NAME,
            Error_params(m_alloc).add(archive_name));
        return NULL;
    }

    string full_path(archive_name, m_alloc);
    full_path.append(':');
    full_path.append(file_name);

    MDL_zip_container_error_code err= EC_OK;
    IMDL_resource_reader *res = open_resource_file(
        m_alloc,
        /*mdl_url=*/"",
        full_path.c_str(),
        err);
    handle_file_error(err, archive_name, file_name);
    return res;
}

// Make directories.
void Archive_extractor::mkdir(string const &path)
{
    size_t pos = 0;
    string dir(get_allocator());
    for (;;) {
        size_t next = path.find('/', pos);
        dir = path.substr(0, next);

        if (!is_directory_utf8(get_allocator(), dir.c_str())) {
            mkdir_utf8(get_allocator(), dir.c_str());
        }

        if (next != string::npos)
            pos = next + 1;
        else
            break;
    }
}

// Copy data.
void Archive_extractor::copy_data(FILE *dst, zip_file_t *src)
{
    char buf[1024];

    for (;;) {
        zip_int64_t l = zip_fread(src, buf, 1024);
        if (l == 0)
            break;
        if (l < 0) {
            translate_zip_error(src);
            break;
        }
        size_t w = fwrite(buf, 1, size_t(l), dst);

        if (w != size_t(l)) {
            error(
                IO_ERROR,
                Error_params(get_allocator()).add(m_archive_name.c_str())
            );
            break;
        }
    }
}

// Normalize separators to '/'.
void Archive_extractor::normalize_separators(string &path)
{
    for (size_t i = 0, n = path.size(); i < n; ++i)
        if (path[i] == '\\')
            path[i] = '/';
}

// Map a file error.
void Archive_extractor::handle_file_error(
    MDL_zip_container_error_code   err,
    char const                     *archive_name,
    char const                     *file_name)
{
    switch (err) {
    case EC_OK:
        return;
    case EC_CONTAINER_NOT_EXIST:
        error(ARCHIVE_DOES_NOT_EXIST, Error_params(m_alloc).add(archive_name));
        return;
    case EC_CONTAINER_OPEN_FAILED:
        error(CANT_OPEN_ARCHIVE, Error_params(m_alloc).add(archive_name));
        return;
    case EC_FILE_OPEN_FAILED:
        if (strcmp("MANIFEST", file_name) != 0) {
            error(
                ARCHIVE_DOES_NOT_CONTAIN_ENTRY,
                Error_params(m_alloc)
                    .add(archive_name)
                    .add(file_name));
            return;
        }
        // fall-through
    case EC_INVALID_CONTAINER:
        error(
            INVALID_MDL_ARCHIVE,
            Error_params(m_alloc).add(archive_name));
        return;
    case EC_NOT_FOUND:
        error(
            ARCHIVE_DOES_NOT_CONTAIN_ENTRY,
            Error_params(m_alloc)
            .add(archive_name)
            .add(file_name));
        return;
    case EC_IO_ERROR:
        error(IO_ERROR, Error_params(m_alloc).add(archive_name));
        return;
    case EC_CRC_ERROR:
        error(CRC_ERROR, Error_params(m_alloc).add(archive_name));
        return;
    case EC_INVALID_PASSWORD:
        error(INVALID_PASSWORD, Error_params(m_alloc).add(archive_name));
        return;
    case EC_MEMORY_ALLOCATION:
        error(MEMORY_ALLOCATION, Error_params(m_alloc).add(archive_name));
        return;
    case EC_RENAME_ERROR:
        error(RENAME_FAILED, Error_params(m_alloc).add(archive_name));
        return;
    case EC_INVALID_HEADER:
        error(MDR_INVALID_HEADER, Error_params(m_alloc).add(archive_name));
        return;
    case EC_INVALID_HEADER_VERSION:
        error(MDR_INVALID_HEADER_VERSION, Error_params(m_alloc).add(archive_name));
        return;
    case EC_PRE_RELEASE_VERSION:
        error(MDR_PRE_RELEASE_VERSION, Error_params(m_alloc).add(archive_name));
        return;
    case EC_INTERNAL_ERROR:
        error(INTERNAL_ARCHIVER_ERROR, Error_params(m_alloc).add(archive_name));
        return;
    }
}

// ------------------------------------ scanner ------------------------------------

// Lookup of the next four bytes from the input stream.
unsigned char Manifest_scanner::look_byte(unsigned ofs)
{
    MDL_ASSERT(ofs < 3);
    if (ofs >= 3)
        ofs = 0;

    while (size() <= ofs) {
        int c = m_is->read_char();
        if (c < 0) {
            m_eof = true;
            return 0;
        }
        m_buffer[m_w_pos] = static_cast<unsigned char>(c);
        m_w_pos = (m_w_pos + 1) & 7u;
    }

    return m_buffer[(m_r_pos + ofs) & 7u];
}

// Throw bytes from the ring buffer.
void Manifest_scanner::throw_bytes(unsigned count)
{
    if (m_eof && count > size())
        return;
    MDL_ASSERT(count <= size());
    m_r_pos = (m_r_pos + count) & 7u;
}

// Get the next unicode char from the input stream.
unsigned Manifest_scanner::get_unicode_char()
{
    bool error = false;
    unsigned res = 0;

    unsigned char ch = look_byte(0);

    // find start code: either 0xxxxxxx or 11xxxxxx
    while ((ch >= 0x80) && ((ch & 0xC0) != 0xC0) && !m_eof) {
        throw_bytes(1);
        ch = look_byte(0);
    }

    if (ch <= 0x7F) {
        // 0xxxxxxx
        res = ch;
        throw_bytes(1);
    } else if ((ch & 0xF8) == 0xF0) {
        // 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
        unsigned c1 = ch & 0x07; ch = look_byte(1); error |= (ch & 0xC0) != 0x80;
        unsigned c2 = ch & 0x3F; ch = look_byte(2); error |= (ch & 0xC0) != 0x80;
        unsigned c3 = ch & 0x3F; ch = look_byte(3); error |= (ch & 0xC0) != 0x80;
        unsigned c4 = ch & 0x3F;
        res = (c1 << 18) | (c2 << 12) | (c3 << 6) | c4;

        // must be U+10000 .. U+10FFFF
        error |= (res < 0x1000) || (res > 0x10FFFF);
        if (!error) {
            throw_bytes(4);
        } else {
            res = 0xFFFD;  // replacement character
            throw_bytes(1);
        }
    } else if ((ch & 0xF0) == 0xE0) {
        // 1110xxxx 10xxxxxx 10xxxxxx
        unsigned c1 = ch & 0x0F; ch = look_byte(1); error |= (ch & 0xC0) != 0x80;
        unsigned c2 = ch & 0x3F; ch = look_byte(2); error |= (ch & 0xC0) != 0x80;
        unsigned c3 = ch & 0x3F;
        res = (c1 << 12) | (c2 << 6) | c3;

        // must be U+0800 .. U+FFFF
        error |= res < 0x800;

        if (!error) {
            throw_bytes(3);
        } else {
            res = 0xFFFD;  // replacement character
            throw_bytes(1);
        }
    } else if ((ch & 0xE0) == 0xC0) {
        // 110xxxxx 10xxxxxx
        unsigned c1 = ch & 0x1F; ch = look_byte(1); error |= (ch & 0xC0) != 0x80;
        unsigned c2 = ch & 0x3F;
        res = (c1 << 6) | c2;

        // must be U+0080 .. U+07FF
        error |= res < 0x80;

        if (!error) {
            throw_bytes(2);
        } else {
            res = 0xFFFD;  // replacement character
            throw_bytes(1);
        }
    } else {
        // error
        res = 0xFFFD;  // replacement character
        throw_bytes(1);
    }
    ++m_curr_column;
    if (res == '\n' || res == '\r') {
        m_curr_column = 0;
        ++m_curr_line;
    }

    if (m_eof)
        return ~0;

    return res;
}

// Get the next token.
Token_kind Manifest_scanner::next_token()
{
restart:
    // skip white space
    while (m_c == ' ' || m_c == '\t' || m_c == '\r' || m_c == '\n') {
        m_c = get_unicode_char();
    }

    // skip comment
    if (m_c == '#') {
        bool at_first_column = m_curr_column == 1;
        do {
            m_c = get_unicode_char();
        } while (m_c != ~0 && m_c != '\n' && m_c != '\r');

        if (!at_first_column) {
            // enforce at first column
            return TK_ERROR;
        }

        goto restart;
    }

    m_text.clear();

    if (m_c == ~0)
        return TK_EOF;

    // token start
    m_text.append(m_c);
    m_column = m_curr_column;
    m_line   = m_curr_line;

    if (('a' <= m_c && m_c <= 'z') || ('A' <= m_c && m_c <= 'Z')) {
        // start of an identifier
        m_c = get_unicode_char();
        while (
            ('a' <= m_c && m_c <= 'z') || ('A' <= m_c && m_c <= 'Z') ||
            ('0' <= m_c && m_c <= '9') || m_c == '_')
        {
            m_text.append(m_c);
            m_c = get_unicode_char();
        }
        return TK_ID;
    } else if (m_c == '.') {
        m_c = get_unicode_char();
        return TK_DOT;
    } else if (m_c == '=') {
        m_c = get_unicode_char();
        return TK_EQUAL;
    } else if (m_c == '"') {
        m_text.clear();

        for (;;) {
            m_c = get_unicode_char();

            if (m_c == '"' || m_c == ~0)
                break;
            if (m_c == '\\') {
                m_c = get_unicode_char();
                if (m_c != '\\' && m_c != '"') {
                    // not an escape
                    m_text.append('\\');
                }
            }
            m_text.append(m_c);
        }

        if (m_c == '"') {
            // found end, good
            m_c = get_unicode_char();
        } else {
            // missing '"' at eof
        }
        return TK_LIT;
    } else {
        m_c = get_unicode_char();
        return TK_ERROR;
    }
}

// ---------------------------- Parser ----------------------------

bool Manifest_parser::parse()
{
    while (m_scanner.current_token() != TK_EOF) {
        pair();
    }

    m_manifest_builder.check_mandatory_fields_existence();

    return !m_syntax_errors && m_manifest_builder.get_error_count() == 0;
}

// Parse a key value pair.
void Manifest_parser::pair()
{
    unsigned value_line = 0;

    key();
    if (m_scanner.current_token() == TK_EQUAL) {
        m_scanner.next();
    } else {
        error();
    }
    if (m_scanner.current_token() == TK_LIT) {
        value_line   = m_scanner.token_line();
        m_full_value = m_scanner.token_value();
        m_scanner.next();
    } else {
        return error();
    }

    while (m_scanner.current_token() == TK_LIT) {
        unsigned this_line = m_scanner.token_line();
        if (m_scanner.token_column() == 1 || value_line == this_line) {
            // continuation is not allowed at column 1 nor at same line
            error();
        }
        value_line = this_line;
        m_full_value.append(m_scanner.token_value());
        m_scanner.next();
    }

    m_manifest_builder.add_pair(m_full_key, m_full_value);
}

// Parse a key.
void Manifest_parser::key()
{
    if (m_scanner.current_token() != TK_ID) {
        // key is not an identifier
        return error();
    }
    if (m_scanner.token_column() != 1) {
        // key does not start at first column
        error();
    }
    m_full_key = m_scanner.token_value();
    while (m_scanner.next() == TK_DOT) {
        m_full_key.append('.');
        if (m_scanner.next() == TK_ID) {
            m_full_key.append(m_scanner.token_value());
        } else {
            return error();
        }
    }
}

// generate an error.
void Manifest_parser::error()
{
    m_syntax_errors = true;

    // forward to the next token starting at column 1
    for (;;) {
        m_scanner.next();
        if (m_scanner.current_token() == TK_EOF || m_scanner.token_column() == 1)
            break;
    }
}

}  // anonymous

// ---------------------------- Archive ZIP Container ---------------------------------------------

// Open a container file.
MDL_zip_container_archive *MDL_zip_container_archive::open(
    IAllocator                     *alloc,
    char const                     *path,
    MDL_zip_container_error_code  &err,
    bool                            with_manifest)
{
    MDL_zip_container_header header_info = header_supported_read_version;
    zip_t* za = MDL_zip_container::open(alloc, path, err, header_info);

    if (err != EC_OK)
        return NULL;

    zip_int64_t manifest_idx = zip_name_locate(za, "MANIFEST", ZIP_FL_ENC_UTF_8);
    if (manifest_idx != 0) {
        // MANIFEST must be the first entry in an archive
        zip_close(za);
        err = EC_INVALID_CONTAINER;
        return NULL;
    }

    zip_stat_t st;
    if (zip_stat_index(za, manifest_idx, ZIP_FL_ENC_UTF_8, &st) < 0) {
        zip_close(za);
        err = EC_INVALID_CONTAINER;
        return NULL;
    }

    if ((st.valid & ZIP_STAT_COMP_METHOD) == 0 || st.comp_method != ZIP_CM_STORE) {
        // MANIFEST is not stored uncompressed
        zip_close(za);
        err = EC_INVALID_CONTAINER;
        return NULL;
    }

    Allocator_builder builder(alloc);
    MDL_zip_container_archive *archiv = builder.create<MDL_zip_container_archive>(
        alloc, path, za, with_manifest);

    if (with_manifest) {
        mi::base::Handle<Manifest const> m(archiv->get_manifest());
        if (!m.is_valid_interface()) {
            // MANIFEST missing or parse error occurred
            builder.destroy(archiv);
            err = EC_INVALID_CONTAINER;
            return NULL;
        }
    }

    archiv->m_header = header_info;
    return archiv;
}


// Get the manifest of this archive.
Manifest const *MDL_zip_container_archive::get_manifest()
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

// Destructor
MDL_zip_container_archive::~MDL_zip_container_archive()
{
}

// Constructor.
MDL_zip_container_archive::MDL_zip_container_archive(
    IAllocator  *alloc,
    char const  *path,
    zip_t       *za,
    bool         with_manifest)
: MDL_zip_container(alloc, path, za, /*supports_resource_hashes=*/false)
, m_manifest(with_manifest ? parse_manifest() : NULL)
{
}

// Get the manifest.
Manifest *MDL_zip_container_archive::parse_manifest()
{
    if (MDL_zip_container_file *fp = file_open("MANIFEST")) {
        Allocator_builder builder(m_alloc);

        File_handle *manifest_fp =
            builder.create<File_handle>(
                get_allocator(),
                File_handle::FH_ARCHIVE,
                this,
                /*owns_archive=*/false,
                fp);

        mi::base::Handle<Buffered_archive_resource_reader> reader(
            builder.create<Buffered_archive_resource_reader>(
            m_alloc,
            manifest_fp,
            "MANIFEST",
            /*mdl_url=*/""));

        Manifest *manifest = Archive_tool::parse_manifest(m_alloc, reader.get());
        if (manifest != NULL) {
            string arc_name(get_container_name(), m_alloc);
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

// ---------------------------- Archive -----------------------------------------------------------

// Constructor.
Archive::Archive(
    MDL *compiler)
: Base(compiler->get_allocator())
, m_archive_name(get_allocator())
, m_manifest()
{
    Allocator_builder builder(get_allocator());

    m_manifest = builder.create<Manifest>(get_allocator());
}

// Get the archive name.
char const *Archive::get_archive_name() const
{
    return m_archive_name.c_str();
}

// Get the MANIFEST of an archive.
Manifest const *Archive::get_manifest() const
{
    m_manifest->retain();
    return m_manifest.get();
}

// Set the archive name.
void Archive::set_archive_name(char const *name)
{
    m_archive_name = name != NULL ? name : "";
    m_manifest->set_archive_name(m_archive_name.c_str());
}

// Get the MANIFEST of an archive.
Manifest *Archive::get_manifest() {
    m_manifest->retain();
    return m_manifest.get();
}

// ---------------------------- Manifest_builder  ----------------------------

// Constructor.
Manifest_builder::Manifest_builder(
    IAllocator *alloc,
    Manifest   &manifest)
: m_alloc(alloc)
, m_manifest(manifest)
, m_modules(0, Module_hash::hasher(), Module_hash::key_equal(), alloc)
, m_seen_fields(0)
, m_error(EC_OK)
, m_tmp(alloc)
{
    // prefill the hash table
    for (size_t i = 0, n = manifest.get_module_count(); i < n; ++i) {
        string m(manifest.get_module_name(i), alloc);

        Module_hash::const_iterator it(m_modules.find(m));
        if (it == m_modules.end()) {
            size_t mod_id = m_manifest.add_module(m.c_str());
            m_modules.insert(Module_hash::value_type(m, mod_id));
        }
    }
}

// Add a Key, value pair.
void Manifest_builder::add_pair(u32string const &key, u32string const &value)
{
    string k(m_alloc);
    string v(m_alloc);

    k.reserve(key.size());
    v.reserve(value.size());

    utf32_to_utf8(k, key.c_str());
    utf32_to_utf8(v, value.c_str());

    if (k == "mdl") {
        if (m_seen_fields & MF_MDL) {
            // multiple "mdl" fields
            error(EC_MULTIPLE_MDL_FIELD);
        }

        IMDL::MDL_version ver = IMDL::MDL_VERSION_1_0;
        if (v == "1.0")
            ver = IMDL::MDL_VERSION_1_0;
        else if (v == "1.1")
            ver = IMDL::MDL_VERSION_1_1;
        else if (v == "1.2")
            ver = IMDL::MDL_VERSION_1_2;
        else if (v == "1.3")
            ver = IMDL::MDL_VERSION_1_3;
        else if (v == "1.4")
            ver = IMDL::MDL_VERSION_1_4;
        else if (v == "1.5")
            ver = IMDL::MDL_VERSION_1_5;
        else if (v == "1.6")
            ver = IMDL::MDL_VERSION_1_6;
        else if (v == "1.7")
            ver = IMDL::MDL_VERSION_1_7;
        else {
            error(EC_UNSUPPORTED_MDL_VERSION);
        }

        m_manifest.set_mdl_version(ver);
        m_seen_fields |= MF_MDL;
    } else if (k == "version") {
        if (m_seen_fields & MF_VERSION) {
            error(EC_MULTIPLE_VERSION_FIELD);
        }

        // just parsed for error detection
        Semantic_version ver(parse_sema_version(v));

        m_manifest.set_sema_version(v.c_str());
        m_seen_fields |= MF_VERSION;
    } else if (k == "module") {
        parse_module(v);
        m_seen_fields |= MF_MODULE;
    } else if (k == "dependency") {
        parse_dependency(v);
        m_seen_fields |= MF_DEPENDENCY;
    } else if (k == "exports.function") {
        parse_export(IArchive_manifest::EK_FUNCTION, v);
    } else if (k == "exports.material") {
        parse_export(IArchive_manifest::EK_MATERIAL, v);
    } else if (k == "exports.struct") {
        parse_export(IArchive_manifest::EK_STRUCT, v);
    } else if (k == "exports.enum") {
        parse_export(IArchive_manifest::EK_ENUM, v);
    } else if (k == "exports.const") {
        parse_export(IArchive_manifest::EK_CONST, v);
    } else if (k == "exports.annotation") {
        parse_export(IArchive_manifest::EK_ANNOTATION, v);
    } else if (k == "author") {
        m_manifest.add_opt_author(v.c_str());
    } else if (k == "contributor") {
        m_manifest.add_opt_contributor(v.c_str());
    } else if (k == "copyright_notice") {
        if (m_seen_fields & MF_COPYRIGHT_NOTICE) {
            error(EC_MULTIPLE_COPYRIGHT_FIELD);
        }
        m_manifest.set_opt_copyright_notice(v.c_str());
        m_seen_fields |= MF_COPYRIGHT_NOTICE;
    } else if (k == "description") {
        if (m_seen_fields & MF_DESCRIPTION) {
            error(EC_MULTIPLE_DESCRIPTION_FIELD);
        }
        m_manifest.set_opt_description(v.c_str());
        m_seen_fields |= MF_DESCRIPTION;
    } else if (k == "created") {
        if (m_seen_fields & MF_CREATED) {
            error(EC_MULTIPLE_CREATED_FIELD);
        }
        check_time_format(v);
        m_manifest.set_opt_created(v.c_str());
        m_seen_fields |= MF_CREATED;
    } else if (k == "modified") {
        if (m_seen_fields & MF_MODIFIED) {
            error(EC_MULTIPLE_MODIFIED_FIELD);
        }
        check_time_format(v);
        m_manifest.set_opt_modified(v.c_str());
        m_seen_fields |= MF_MODIFIED;
    } else {
        // user key
        m_manifest.add_user_pair(k.c_str(), v.c_str());
    }
}

// Check existence of mandatory fields.
void Manifest_builder::check_mandatory_fields_existence()
{
    if ((m_seen_fields & MF_MDL) == 0)
        error(EC_MDL_FIELD_MISSING);
    if ((m_seen_fields & MF_VERSION) == 0)
        error(EC_VERSION_FIELD_MISSING);
}

// Return number of errors.
size_t Manifest_builder::get_error_count() const
{
    return m_error == EC_OK ? 0 : 1;
}

// Parse an export.
void Manifest_builder::parse_export(
    IArchive_manifest::Export_kind kind,
    string const                   &full)
{
    size_t ofs = 0;
    if (full.length() >= 2 && full[0] == ':' && full[1] == ':')
        ofs = 2;

    size_t pos = full.rfind("::");
    if (pos == string::npos || pos <= ofs) {
        error(EC_INVALID_EXPORT);
        return;
    }

    string m(full.substr(ofs, pos - ofs));
    string e(full.substr(pos + 2));

    if (m.empty() || e.empty()) {
        error(EC_INVALID_EXPORT);
        return;
    }

    size_t mod_id = 0;

    Module_hash::const_iterator it(m_modules.find(m));
    if (it != m_modules.end()) {
        mod_id = it->second;
    } else {
        mod_id = m_manifest.add_module(m.c_str());
        m_modules.insert(Module_hash::value_type(m, mod_id));
    }
    m_manifest.add_export(kind, mod_id, e.c_str());
}

// Parse a module export.
void Manifest_builder::parse_module(string const &full)
{
    size_t ofs = 0;
    if (full.length() >= 2 && full[0] == ':' && full[1] == ':')
        ofs = 2;

    string m(full.substr(ofs));

    Module_hash::const_iterator it(m_modules.find(m));
    if (it == m_modules.end()) {
        size_t mod_id = m_manifest.add_module(m.c_str());
        m_modules.insert(Module_hash::value_type(m, mod_id));
    }
}

// Parse a dependency.
void Manifest_builder::parse_dependency(string const &full)
{
    size_t l   = full.length();
    size_t s = 0, e = 0;

    if (full.length() >= 2 && full[0] == ':' && full[1] == ':')
        s = 2;

    for (e = s; e < l; ++e) {
        if (full[e] == ' ')
            break;
    }

    string m(full.substr(s, e - s));
    string d(full.substr(e + 1));

    Semantic_version ver = parse_sema_version(d);

    m_manifest.add_dependency(m.c_str(), ver);
}

// Parse a semantic version.
Semantic_version Manifest_builder::parse_sema_version(string const &s)
{
#define DIGIT(c) ('0' <= (c) && (c) <= '9')

    Semantic_version v(0, 0, 0, "");

    size_t l = s.length();
    size_t i = 0;

    if (i >= l) {
        error(EC_INVALID_DEPENDENCY);
        return v;
    }

    if (!DIGIT(s[i])) {
        error(EC_INVALID_DEPENDENCY);
        return v;
    }

    int major = 0;
    do {
        major = major * 10 + s[i] - '0';
        ++i;
    } while (i < l && DIGIT(s[i]));

    if (i >= l) {
        error(EC_INVALID_DEPENDENCY);
        return v;
    }
    if (s[i] != '.') {
        error(EC_INVALID_DEPENDENCY);
        return v;
    }
    ++i;
    if (i >= l) {
        error(EC_INVALID_DEPENDENCY);
        return v;
    }

    if (!DIGIT(s[i])) {
        error(EC_INVALID_DEPENDENCY);
        return v;
    }

    int minor = 0;
    do {
        minor = minor * 10 + s[i] - '0';
        ++i;
    } while (i < l && DIGIT(s[i]));

    if (i >= l) {
        error(EC_INVALID_DEPENDENCY);
        return v;
    }
    if (s[i] != '.') {
        error(EC_INVALID_DEPENDENCY);
        return v;
    }
    ++i;
    if (i >= l) {
        error(EC_INVALID_DEPENDENCY);
        return v;
    }

    if (!DIGIT(s[i])) {
        error(EC_INVALID_DEPENDENCY);
        return v;
    }

    int patch = 0;
    do {
        patch = patch * 10 + s[i] - '0';
        ++i;
    } while (i < l && DIGIT(s[i]));

    m_tmp.clear();

    if (i < l) {
        if (s[i] != '-') {
            error(EC_INVALID_DEPENDENCY);
            return v;
        }
        m_tmp = s.substr(i + 1);
        if (m_tmp.empty()) {
            error(EC_INVALID_DEPENDENCY);
            return v;
        }
    }
    return Semantic_version(major, minor, patch, m_tmp.c_str());
#undef DIGIT
}

// Check the time format.
void Manifest_builder::check_time_format(string const &s)
{
    size_t l = s.length();
    if (l < 10) {
        return error(EC_INVALID_TIME);
    }

#define DIGIT(c) ('0' <= (c) && (c) <= '9')

    if (!DIGIT(s[0]) || !DIGIT(s[1]) || !DIGIT(s[2]) || !DIGIT(s[3]) || s[4] != '-' ||
        !DIGIT(s[5]) || !DIGIT(s[6]) || s[7] != '-' ||
        !DIGIT(s[8]) || !DIGIT(s[9]))
    {
        return error(EC_INVALID_TIME);
    }

    if (l > 10) {
        if (s[10] != ',') {
            return error(EC_INVALID_TIME);
        }
    }
#undef DIGIT
}

// Creates an error.
void Manifest_builder::error(Error_code code)
{
    // for now, just the first error
    if (m_error == EC_OK)
        m_error = code;
}

// ---------------------------- Archiver ----------------------------

// Constructor.
Archive_tool::Archive_tool(
    IAllocator *alloc,
    MDL        *compiler)
: Base(alloc)
, m_compiler(compiler, mi::base::DUP_INTERFACE)
, m_msg_list(alloc, /*owner_fname=*/"")
, m_options(alloc)
, m_cb(NULL)
, m_last_msg_idx(0)
{
    m_options.add_option(
        MDL_ARC_OPTION_ONLY_REFERENCED,
        "false",
        "Include only referenced resources");
    m_options.add_option(
        MDL_ARC_OPTION_OVERWRITE,
        "false",
        "Overwrite existing files");
    m_options.add_option(
        MDL_ARC_OPTION_IGNORE_EXTRA_FILES,
        "false",
        "Ignore extra files in source directory");
    // Fill the compressed suffix set with known suffixes from the MDL spec that
    // should be compressed
    m_options.add_option(
        MDL_ARC_OPTION_COMPRESS_SUFFIXES,
        ".ies,.mbsdf,.txt,.html",
        "Comma separated list of resource suffixes that should be stored compressed");
}

// Create a new archive.
IArchive const *Archive_tool::create_archive(
    char const            *root_path,
    char const            *root_package,
    char const            *dest_path,
    Key_value_entry const manifest_entries[],
    size_t                me_cnt)
{
    IAllocator *alloc = get_allocator();
    Allocator_builder builder(alloc);

    mi::base::Handle<Archive> the_archive(
        builder.create<Archive>(m_compiler.get()));
    mi::base::Handle<Manifest> the_manifest(the_archive->get_manifest());

    Archive_builder arc_builder(
        m_compiler.get(),
        *this,
        dest_path,
        m_options.get_bool_option(MDL_ARC_OPTION_OVERWRITE),
        m_options.get_bool_option(MDL_ARC_OPTION_IGNORE_EXTRA_FILES),
        the_manifest.get(),
        m_cb);

    // set compressed resource suffixes
    char const *suffixes = m_options.get_string_option(MDL_ARC_OPTION_COMPRESS_SUFFIXES);
    if (suffixes != NULL) {
        for (char const *p = NULL, *s = suffixes; s != NULL; s = p) {
            string suffix(get_allocator());

            if (s[0] == '.')
                ++s;
            p = strchr(s, ',');
            if (p != NULL) {
                suffix = string(s, p, get_allocator());
                ++p;
            } else {
                suffix = s;
            }
            arc_builder.add_compressed_resource_suffix(suffix.c_str());
        }
    }

    if (root_package[0] == ':' && root_package[1] == ':') {
        // skip first '::'
        root_package += 2;
    }

    if (!arc_builder.collect(root_path, root_package)) {
        // failed, errors reported
        return NULL;
    }

    if (!arc_builder.compile_modules()) {
        // messages copied
        return NULL;
    }

    for (size_t i = 0; i < me_cnt; ++i) {
        Key_value_entry const &e     = manifest_entries[i];
        char const            *key   = e.key;
        char const            *value = e.value;

        IArchive_manifest::Error_code code = the_manifest->add_key_value(key, value);
        switch (code) {
        case IArchive_manifest::ERR_OK:
            break;
        case IArchive_manifest::ERR_NULL_ARG:
            error(KEY_NULL_PARAMETERS, Error_params(alloc));
            break;
        case IArchive_manifest::ERR_TIME_FORMAT:
            error(VALUE_MUST_BE_IN_TIME_FORMAT, Error_params(alloc).add(key));
            break;
        case IArchive_manifest::ERR_VERSION_FORMAT:
            error(VALUE_MUST_BE_IN_SEMA_VERSION_FORMAT, Error_params(alloc).add(key));
            break;
        case IArchive_manifest::ERR_FORBIDDEN:
            error(FORBIDDEN_KEY, Error_params(alloc).add(key));
            break;
        case IArchive_manifest::ERR_SINGLE:
            error(SINGLE_VALUED_KEY, Error_params(alloc).add(key));
            break;
        }
    }

    if (!arc_builder.create_zip_archive()) {
        // errors reported
        return NULL;
    }

    // all went fine
    string arc_name = arc_builder.get_archive_name();
    the_archive->set_archive_name(arc_name.c_str());

    the_archive->retain();
    return the_archive.get();
}

// Create an archive MANIFEST template.
IArchive_manifest const *Archive_tool::create_manifest_template(
    char const *root_path,
    char const *root_package)
{
    Allocator_builder builder(get_allocator());

    mi::base::Handle<Manifest> the_manifest(
        builder.create<Manifest>(get_allocator()));

    Archive_builder arc_builder(
        m_compiler.get(),
        *this,
        /*dest_path=*/"",
        /*overwrite=*/false,
        /*ignore_extra_files=*/true,
        the_manifest.get(),
        m_cb);

    if (root_package[0] == ':' && root_package[1] == ':') {
        // skip first '::'
        root_package += 2;
    }

    if (!arc_builder.collect(root_path, root_package)) {
        // failed, errors reported
        return NULL;
    }

    if (!arc_builder.compile_modules()) {
        // messages copied
        return NULL;
    }

    // all went fine
    the_manifest->retain();
    return the_manifest.get();
}

// Extract an archive to the file system.
void Archive_tool::extract_archive(
    char const *archive_path,
    char const *dest_path)
{
    Archive_extractor arc_extractor(
        get_allocator(),
        *this,
        dest_path,
        m_options.get_bool_option(MDL_ARC_OPTION_OVERWRITE),
        m_cb);

    arc_extractor.extract(archive_path);
}

// Get the MANIFEST from an archive to the file system.
Manifest const *Archive_tool::get_manifest(
    char const *archive_path)
{
    mi::base::Handle<IInput_stream> is(get_manifest_content(archive_path));

    if (!is.is_valid_interface()) {
        return NULL;
    }

    IAllocator *alloc = get_allocator();

    Allocator_builder builder(alloc);
    mi::base::Handle<Manifest> manifest(builder.create<Manifest>(alloc));

    Manifest_builder helper(alloc, *manifest.get());

    Manifest_scanner scanner(alloc, is);
    Manifest_parser  parser(alloc, scanner, helper);

    if (!parser.parse()) {
        error(MANIFEST_BROKEN, Error_params(alloc));
        return NULL;
    }

    manifest->retain();
    return manifest.get();
}

// Get the MANIFEST from resource reader.
Manifest *Archive_tool::parse_manifest(
    IAllocator           *alloc,
    IMDL_resource_reader *reader)
{
    if (reader == NULL) {
        return NULL;
    }

    Allocator_builder builder(alloc);
    mi::base::Handle<IInput_stream> is(
        builder.create<Resource_Input_stream>(alloc, reader));

    mi::base::Handle<Manifest> manifest(builder.create<Manifest>(alloc));

    Manifest_builder helper(alloc, *manifest.get());
    Manifest_scanner scanner(alloc, is);
    Manifest_parser  parser(alloc, scanner, helper);

    if (!parser.parse()) {
        // no error, just failure
        return NULL;
    }

    manifest->retain();
    return manifest.get();
}


// Get the MANIFEST content from an archive to the file system.
IInput_stream *Archive_tool::get_manifest_content(
    char const *archive_path)
{
    Archive_extractor arc_extractor(
        get_allocator(),
        *this,
        /*dest_path*/"",
        /*overwrite=*/false,
        /*callback=*/NULL);

    mi::base::Handle<IMDL_resource_reader> reader(
        arc_extractor.get_content_buffer(archive_path, "MANIFEST"));

    if (!reader.is_valid_interface()) {
        // failed
        return NULL;
    }

    Allocator_builder builder(get_allocator());
    return builder.create<Resource_Input_stream>(get_allocator(), reader.get());
}

// Get the content from any file out of an archive on the file system.
IInput_stream *Archive_tool::get_file_content(
    char const *archive_path,
    char const *file_name)
{
    if (file_name == NULL) {
        error(
            ARCHIVE_DOES_NOT_CONTAIN_ENTRY,
            Error_params(get_allocator())
            .add(archive_path)
            .add("<NULL>"));
        return NULL;
    }

    Archive_extractor arc_extractor(
        get_allocator(),
        *this,
        /*dest_path*/"",
        /*overwrite=*/false,
        /*callback=*/NULL);

    string afn(convert_os_separators_to_slashes(string(file_name, get_allocator())));

    mi::base::Handle<IMDL_resource_reader> reader(
        arc_extractor.get_content_buffer(archive_path, afn.c_str()));

    if (!reader.is_valid_interface()) {
        // failed
        return NULL;
    }

    if (strcmp("MANIFEST", file_name) == 0) {
        // do not allow to read MANIFEST
        error(
            ARCHIVE_DOES_NOT_CONTAIN_ENTRY,
            Error_params(get_allocator())
            .add(archive_path)
            .add(file_name));
        return NULL;
    }

    Allocator_builder builder(get_allocator());
    return builder.create<Resource_Input_stream>(get_allocator(), reader.get());
}

// Access archiver messages of last archive operation.
Messages const &Archive_tool::access_messages() const
{
    return m_msg_list;
}

// Access options.
Options &Archive_tool::access_options()
{
    return m_options;
}

// Set an event callback.
void Archive_tool::set_event_cb(IArchive_tool_event *cb)
{
    m_cb = cb;
}

// Creates a new error.
void Archive_tool::error(int code, Error_params const &params)
{
    Position_impl zero(0, 0, 0, 0);

    string msg(m_msg_list.format_msg(code, MESSAGE_CLASS, params));
    m_last_msg_idx = m_msg_list.add_error_message(
        code, MESSAGE_CLASS, 0, &zero, msg.c_str());
}

// Creates a new warning.
void Archive_tool::warning(int code, Error_params const &params)
{
    Position_impl zero(0, 0, 0, 0);

    string msg(m_msg_list.format_msg(code, MESSAGE_CLASS, params));
    m_last_msg_idx = m_msg_list.add_warning_message(
        code, MESSAGE_CLASS, 0, &zero, msg.c_str());
}

// Adds a new note to the previous message.
void Archive_tool::add_note(int code, Error_params const &params)
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

}  // mdl
}  // mi
