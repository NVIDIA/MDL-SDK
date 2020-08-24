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

#ifndef MDL_COMPILERCORE_FILE_RESOLUTION_H
#define MDL_COMPILERCORE_FILE_RESOLUTION_H 1

#include <mi/mdl/mdl_mdl.h>
#include <mi/mdl/mdl_entity_resolver.h>
#include <mi/base/handle.h>
#include <mi/base/lock.h>

#include "compilercore_allocator.h"
#include "compilercore_messages.h"
#include "compilercore_zip_utils.h"

namespace mi {
namespace mdl {

class Manifest;
class MDL;
class Messages_impl;
class Module;
class Err_location;
class Error_params;
class Position;
class IModule_cache;
class IInput_stream;
class IValue_resource;
class Error_params;
class Type_factory;
class Value_factory;

#define UDIM_MARI_MARKER    "<UDIM>"
#define UDIM_ZBRUSH_MARKER  "<UVTILE0>"
#define UDIM_MUDBOX_MARKER  "<UVTILE1>"

/// Implements file resolution.
class File_resolver {
    typedef set<string>::Type       String_set;
    typedef map<string, bool>::Type String_map;

public:
    // The file resolver produces compiler messages.
    static char const MESSAGE_CLASS = 'C';

    /// Resolve an import (module) name to the corresponding absolute module name.
    ///
    /// \param pos                     The position of the referencing MDL statement.
    /// \param import_name             The relative module name to import.
    /// \param owner_name              The absolute name of the owner MDL module.
    ///                                This may be NULL or "" for the top-level import.
    /// \param owner_filename          Absolute filename of the owner MDL module.
    ///                                This may be NULL or "" for the top-level import.
    /// \returns                       An import result interface or NULL
    ///                                if the module does not exist.
    IMDL_import_result *resolve_import(
        Position const &pos,
        char const     *import_name,
        char const     *owner_name,
        char const     *owner_filename);

    /// Resolve a resource (file) name to the corresponding absolute file path.
    ///
    /// \param[in]  pos             The position of the referencing MDL statement.
    /// \param[in]  file_name       The (possible relative) resource name.
    /// \param[in]  owner_name      The absolute name of the owner MDL module.
    /// \param[in]  owner_filename  Absolute filename of the owner MDL module.
    ///
    /// \returns                    The resource set or NULL if the resource does not exist.
    IMDL_resource_set *resolve_resource(
        Position const &pos,
        char const     *import_name,
        char const     *owner_name,
        char const     *owner_filename);

    /// Open an input stream to the modules source.
    /// \param  module_name             The absolute module name.
    /// \returns                        An interface to an input stream to the module source,
    ///                                 or NULL if the module could not be opened.
    IInput_stream *open(
        char const *module_name);

    /// Checks whether the given module source exists.
    /// \param  module_name             The absolute module name.
    /// \returns                        True if the module exists. open() should not fail in
    ///                                 that case!
    bool exists(
        char const *module_name);

    /// Set a replacement path for a given (absolute) module name.
    ///
    /// \param module_name  the absolute module name
    /// \param file_name    the replacement file name for this module
    void set_module_replacement(
        char const *module_name,
        char const *file_name)
    {
        m_repl_module_name = module_name;
        m_repl_file_name   = file_name;
    }

    /// Get the allocator.
    IAllocator *get_allocator() const { return m_alloc; }

    /// Get the compiler messages.
    Messages_impl &get_messages_impl() { return m_msgs; }

public:
    /// Constructor.
    ///
    /// \param compiler           the MDL compiler
    /// \param module_cache       the module cache if any (used only for is-string-module checks)
    /// \param external_resolver  if valid, the external entity resolver to be used
    /// \param search_path        the search path helper to use
    /// \param sp_lock            the lock for search path access
    /// \param msgs               compiler messages to append
    /// \param front_path         if non-NULL, search this MDL path first
    File_resolver(
        MDL const                                &compiler,
        IModule_cache                            *module_cache,
        mi::base::Handle<IEntity_resolver> const &external_resolver,
        mi::base::Handle<IMDL_search_path> const &search_path,
        mi::base::Lock                           &sp_lock,
        Messages_impl                            &msgs,
        char const                               *front_path);

private:
    /// Creates a new error.
    ///
    /// \param code    the error code
    /// \param loc     location of the error
    /// \param params  error message parameter inserts
    void error(
        int                code,
        Err_location const &loc,
        Error_params const &params);

    /// Add a note to the last error/warning.
    ///
    /// \param code    the error code
    /// \param loc     location of the error
    /// \param params  note message parameter inserts
    void add_note(
        int                code,
        Err_location const &loc,
        Error_params const &params);

    /// Creates a new warning.
    ///
    /// \param code    the error code
    /// \param loc     location of the error
    /// \param params  error message parameter inserts
    void warning(
        int                code,
        Err_location const &loc,
        Error_params const &params);

    /// Convert a name which might be either an url (separator '/') or an module name
    /// (separator '::') into an url.
    string to_url(
        char const *input_name) const;

    /// Convert an url (separator '/') into  a module name (separator '::').
    string to_module_name(
        char const *input_url) const;

    /// Convert an url (separator '/') into an archive name (separator '.').
    string to_archive(
        char const *file_name) const;

    /// Check if an archive name is a prefix of a archive path.
    bool is_archive_prefix(
        string const &archive_name,
        string const &archive_path) const;

    /// Check whether the given archive contains the requested file.
    ///
    /// \param archive_name   the name of the archive
    /// \param file_name      the (full) name of the file
    bool archive_contains(
        char const *archive_name,
        char const *file_name);

    /// Check whether the given archive contains a file matching a mask.
    ///
    /// \param archive_name   the name of the archive
    /// \param file_mask      the mask
    bool archive_contains_mask(
        char const *archive_name,
        char const *file_mask);

    /// Returns the nesting level of a module, i.e., the number of "::" substrings in the
    /// fully-qualified module name minus 1.
    static size_t get_module_nesting_level(char const *module_name);

    /// Checks that \p file_path contains no "." or ".." directory names.
    ///
    /// \param file_path  The file path to be checked (SLASH).
    ///
    /// \return           \c true in case of success, \c false otherwise.
    bool check_no_dots(
        char const *file_path);

    /// Checks that \p file_path contains at most one leading "." directory name, at most
    /// \p nesting_level leading ".." directory names, and no such non-leading directory names.
    ///
    /// \param file_path      The file path to be checked (SLASH).
    /// \param nesting_level  The nesting level of the importing MDL module (\see check_no_dots()).
    ///
    /// \return               \c true in case of success, \c false otherwise.
    bool check_no_dots_strict_relative(
        char const *file_path,
        size_t     nesting_level);

    /// Splits a file path into a directory path and file name.
    ///
    /// See MDL spec, section 2.2 for details.
    ///
    /// \param file_path             The file path to split (SLASH).
    /// \param[out] directory_path   The computed directory path. Empty or ends with a slash.
    /// \param[out] file_name        The compute file name.
    void split_file_path(
        string const &input_url,
        string       &directory_path,
        string       &file_path);

    /// Splits a module file system path into current working directory, current search path, and
    /// current module path.
    ///
    /// See MDL spec, section 2.2 for details.
    ///
    /// \param module_file_system_path          The module file system path to split (OS).
    /// \param module_name                      The fully-qualified MDL module name (only used for
    ///                                         string-based modules).
    /// \param module_nesting_level             The nesting level of the importing MDL module.
    /// \param[out] current_working_directory   The computed current working directory (OS).
    /// \param[out] cwd_is_container            True, if the \c current_working_directory names a
    ///                                         container.
    /// \param[out] current_search_path         The computed search path (OS).
    /// \param[out] csp_is_container            True, if the \c current_search_path names a
    ///                                         container.
    /// \param[out] current_module_path         The computed module path (SLASH).
    ///                                         Either empty or begins with a slash.
    void split_module_file_system_path(
        string const &module_file_system_path,
        string const &module_name,
        size_t       module_nesting_level,
        string       &current_working_directory,
        bool         &cwd_is_container,
        string       &current_search_path,
        bool         &csp_is_container,
        string       &current_module_path);

    /// Normalizes a file path given by its directory path and file name.
    ///
    /// See MDL spec, section 2.2 for details.
    ///
    /// \param file_path                   The file path to normalize (SLASH).
    /// \param file_mask                   The file mask to normalize (SLASH).
    /// \param file_mask_is_regex          True, if file_mask is a regular expression.
    /// \param directory_path              The directory path split from the file path (SLASH).
    /// \param file_name                   The file name split from the file path (SLASH).
    /// \param nesting_level               The nesting level of the importing MDL module.
    /// \param module_file_system_path     The module file system path (OS).
    /// \param current_working_directory   The current working directory split from the module file
    ///                                    system path (OS).
    /// \param cwd_is_container            True, if the \c current_working_directory is a container.
    /// \param current_module_path         The current module path split from the module file system
    ///                                    path (SLASH).
    /// \return                            The normalized (canonical) file path, or \c "" in case of
    ///                                    failure.
    string normalize_file_path(
        string const &file_path,
        string const &file_mask,
        bool         file_mask_is_regex,
        string const &directory_path,
        string const &file_name,
        size_t       nesting_level,
        string const &module_file_system_path,
        string const &current_working_directory,
        bool         cwd_is_container,
        string const &current_module_path);

    bool is_builtin(string const &canonical_file_path) const;

    /// Check if the given \p canonical_file_path names a string based module.
    ///
    /// \param canonical_file_path       The canonical file path to resolve (SLASH).
    bool is_string_based(string const &canonical_file_path) const;

    /// Loops over the search paths to resolve \p canonical_file_path.
    ///
    /// \param canonical_file_mask  The canonical file path (maybe regex) to resolve (SLASH).
    /// \param is_resource          True if the file path describes a resource.
    /// \param file_path            Used to report the filepath of error messages
    /// \param udim_mode            If != NO_UDIM the returned file path is a mask.
    ///
    /// \return                     The resolved file path, or \c "" in case of failures.
    string consider_search_paths(
        string const &canonical_file_mask,
        bool         is_resource,
        char const   *file_path,
        UDIM_mode    udim_mode);

    /// Checks whether the resolved file system location passes the consistency checks in the
    /// MDL spec.
    ///
    /// See MDL spec, section 2.2 for details.
    ///
    /// \param resolved_file_system_location  The resolved file system location to be checked (OS).
    /// \param canonical_file_path            The canonical file path (SLASH).
    /// \param is_regex                       True, if the canonical file path is a
    ///                                       regular expression.
    /// \param file_path                      The (original) file path (to select the type of check
    ///                                       and error messages, SLASH).
    /// \param current_working_directory      The current working directory (OS).
    /// \param current_search_path            The current search path (OS).
    /// \param is_resource                    True if the location belongs to a resource.
    /// \param csp_is_archive                 True if \c current_search_path names an archive.
    /// \param is_string_module               True if the owner is a string module.
    /// \return                               \c true in case of success, \c false otherwise.
    bool check_consistency(
        string const &resolved_file_system_location,
        string const &canonical_file_path,
        bool         is_regex,
        string const &file_path,
        string const &current_working_directory,
        string const &current_search_path,
        bool         is_resource,
        bool         csp_is_archive,
        bool         is_string_module);

    /// Simplifies a file path by removing directory names "." and pairs of directory names like
    /// ("foo", ".."). Slashes are used as separators. Leading and trailing slashes in the input are
    /// preserved.
    ///
    /// \param file_path  the file path to simplify
    /// \param sep        the separator (as string)
    /// The input must be valid w.r.t. to the number of directory names "..".
    string simplify_path(
        string const &file_path,
        char         sep);

    /// Check if a given archive is killed by another archive OR a existing directory.
    ///
    /// \param path          current search path
    /// \param archive_name  the name of the archive to check
    /// \param archives      the set of all archives in this search path (without .mdr)
    ///
    /// \returns true if an conflict was detected
    bool is_killed(
        char const    *path,
        string const  &archive_name,
        String_map    &archives);

    /// Check if a directory path is killed due to a conflicting archive.
    ///
    /// \param file_mask   a file mask that could point to a MDL module or resource
    ///
    /// \returns true if an conflict was detected
    bool is_killed(
        char const *file_mask);

    /// Search the given path in all MDL search paths and return the absolute path if found.
    ///
    /// \param file_mask    the path to search (maybe a regex)
    /// \param is_resource  true if search in extra resource path
    /// \param front_path   if non-NULL, search this MDL path first
    /// \param udim_mode    if != NO_UDIM the returned value is a file mask
    ///
    /// \return the absolute path or mask if found
    string search_mdl_path(
        char const *file_mask,
        bool       in_resource_path,
        char const *front_path,
        UDIM_mode  udim_mode);

    /// Check if the given file name (UTF8 encoded) names a file on the file system or inside
    /// an archive.
    ///
    /// \param fname     a file name
    /// \param is_regex  if true, threat fname as a regular expression
    bool file_exists(
        char const *fname,
        bool       is_regex) const;

    /// Resolve a MDL file name
    ///
    /// \param[out] abs_file_name    the resolved absolute file name (on file system)
    /// \param[in]  file_path        the MDL file path to resolve
    /// \param[in]  is_resource      true if file_path names a resource
    /// \param[in]  owner_name       the absolute name of the owner
    /// \param[in]  owner_file_path  the file path of the owner
    /// \param[in]  pos              the position of the import statement
    /// \param[out] udim_mode        if != NO_UDIM the returned absolute file name is a file mask
    string resolve_filename(
        string         &abs_file_name,
        char const     *file_path,
        bool           is_resource,
        char const     *module_file_system_path,
        char const     *module_name,
        Position const *pos,
        UDIM_mode      &udim_mode);

    /// The resolver resolves a module.
    void mark_module_search(char const *module_name);

    /// The resolver resolves a resource.
    void mark_resource_search(char const *file_name);

    /// Map a file error.
    void handle_file_error(MDL_zip_container_error_code err);

private:
    /// The memory allocator.
    IAllocator *m_alloc;

    /// The current compiler.
    MDL const &m_mdl;

    /// The module cache if any.
    IModule_cache *m_module_cache;

    /// Compiler messages to be used.
    Messages_impl &m_msgs;

    /// Position of the import statement.
    Position const *m_pos;

    /// The external entity resolver if any.
    mi::base::Handle<IEntity_resolver> m_external_resolver;

    /// The search path helper.
    mi::base::Handle<IMDL_search_path> m_search_path;

    /// Lock for the resolver.
    mi::base::Lock &m_resolver_lock;

    typedef vector<string>::Type String_vec;

    /// Cache for the MDL search paths.
    String_vec m_paths;

    /// Cache for the MDL resource paths.
    String_vec m_resource_paths;

    /// The set of "killed" packages.
    String_set m_killed_packages;

    /// If non-NULL, search this MDL path first.
    char const *m_front_path;

    /// The entity we are trying to resolve.
    char const *m_resolve_entity;

    /// If non-empty the name of the replacement module.
    string m_repl_module_name;

    /// If non-empty the name of the replacement module file name.
    string m_repl_file_name;

    // Index of the last generated error message.
    int m_last_msg_idx;

    /// Set if the caches were filled.
    volatile bool m_pathes_read;

    /// True if we are resolving a resource, false if we are resolving a module.
    bool m_resolving_resource;
};


//-------------------------------------------------------------------------------------------------

/// Helper class to transparently handle files on file system and inside archives.
class File_handle
{
    friend class Allocator_builder;
public:

    enum Kind {
        FH_FILE,
        FH_ARCHIVE,
        FH_MDLE,
    };

    /// Check if this is a handle to normal file or if it is in an archive or mdle.
    Kind get_kind() const { return m_kind; }

    /// Get the FILE handle if this object represents an ordinary file.
    FILE *get_file();

    /// Get the container if this object represents a file inside a MDL container.
    MDL_zip_container *get_container();

    /// Get the compressed file handle if this object represents a file inside a MDL container.
    MDL_zip_container_file *get_container_file();

    /// Open a file handle.
    ///
    /// \param[in]  alloc  the allocator
    /// \param[in]  name   a file name (might describe a file inside an container)
    /// \param[out] err    error code
    static File_handle *open(
        IAllocator                     *alloc,
        char const                     *name,
        MDL_zip_container_error_code   &err);

    /// Close a file handle.
    static void close(File_handle *h);

private:
    /// Constructor from a FILE handle.
    ///
    /// \param alloc        the allocator
    /// \param fp           a FILE pointer, takes ownership
    explicit File_handle(
        IAllocator *alloc,
        FILE       *fp);

    /// Constructor from container file.
    ///
    /// \param alloc            the allocator
    /// \param kind             kind of this file
    /// \param container        a MDL container pointer, takes ownership
    /// \param owns_container   true if this File_handle owns the container (will be closed then)
    /// \param fp               a MDL container file pointer, takes ownership
    File_handle(
        IAllocator             *alloc,
        Kind                    kind,
        MDL_zip_container      *container,
        bool                    owns_container,
        MDL_zip_container_file *fp);


    /// Constructor from another File_handle container.
    ///
    /// \param alloc        the allocator
    /// \param kind         kind of this file
    /// \param fh           another container file handle
    /// \param fp           a MDL compressed file pointer, takes ownership
    File_handle(
        File_handle            *fh,
        Kind                    kind,
        MDL_zip_container_file *fp);

    /// Destructor.
    ~File_handle();

private:
    /// Current allocator.
    IAllocator *m_alloc;

    /// If non-null, this is an container.
    MDL_zip_container *m_container;

    union {
        FILE                   *fp;
        MDL_zip_container_file *z_fp;
    } u;

    /// The kind this file.
    Kind m_kind;

    /// If true, this file handle has ownership on the MDL container.
    bool m_owns_container;
};

/// Implementation of the IMDL_resource_set interface.
class MDL_resource_set : public Allocator_interface_implement<IMDL_resource_set>
{
    typedef Allocator_interface_implement<IMDL_resource_set> Base;
    friend class Allocator_builder;
public:
    /// Get the MDL URL mask of the ordered set.
    ///
    /// \returns the MDL URL mask of the set
    char const *get_mdl_url_mask() const MDL_FINAL;

    /// Get the file name mask of the ordered set.
    ///
    /// \param i  the index
    ///
    /// \returns the file name mask of the set
    ///
    /// \note If this resource is inside an MDL archive, the returned name
    ///       uses the format 'MDL_ARCHIVE_FILENAME:RESOURCE_FILENAME'.
    char const *get_filename_mask() const MDL_FINAL;

    /// Get the number of resolved file names.
    size_t get_count() const MDL_FINAL;

    /// Get the i'th MDL url of the ordered set.
    ///
    /// \param i  the index
    ///
    /// \returns the i'th MDL url of the set or NULL if the index is out of range.
    char const *get_mdl_url(size_t i) const MDL_FINAL;

    /// Get the i'th file name of the ordered set.
    ///
    /// \param i  the index
    ///
    /// \returns the i'th file name of the set or NULL if the index is out of range.
    char const *get_filename(size_t i) const MDL_FINAL;

    /// If the ordered set represents an UDIM mapping, returns it, otherwise NULL.
    ///
    /// \param[in]  i  the index
    /// \param[out] u  the u coordinate
    /// \param[out] v  the v coordinate
    ///
    /// \returns true if a mapping is available, false otherwise
    bool get_udim_mapping(size_t i, int &u, int &v) const MDL_FINAL;

    /// Opens a reader for the i'th entry.
    ///
    /// \param i  the index
    ///
    /// \returns an reader for the i'th entry of the set or NULL if the index is out of range.
    IMDL_resource_reader *open_reader(size_t i) const MDL_FINAL;

    /// Get the UDIM mode for this set.
    UDIM_mode get_udim_mode() const MDL_FINAL;

    /// Get the resource hash value for the i'th file in the set if any.
    ///
    /// \param[in]  i     the index
    /// \param[out] hash  the hash value if exists
    ///
    /// \return true if this entry has a hash, false otherwise
    bool get_resource_hash(
        size_t i,
        unsigned char hash[16]) const MDL_FINAL;

    /// Create a resource set from a file mask.
    ///
    /// \param alloc      the allocator
    /// \param url        the absolute MDL url
    /// \param filename   the file name
    /// \param udim_mode  the UDIM mode
    static MDL_resource_set *from_mask(
        IAllocator *alloc,
        char const *url,
        char const *file_mask,
        UDIM_mode  udim_mode);

private:
    /// Create a resource set from a file mask describing files on disk.
    ///
    /// \param alloc      the allocator
    /// \param url        the absolute MDL url
    /// \param filename   the file name
    /// \param udim_mode  the UDIM mode
    static MDL_resource_set *from_mask_file(
        IAllocator *alloc,
        char const *url,
        char const *file_mask,
        UDIM_mode  udim_mode);

    /// Parse a file name and enter it into a resource set.
    ///
    /// \param s          the resource set
    /// \param name       the file name of one UDIM part
    /// \param ofs        offset inside name where the u,v info is found
    /// \param url        the already (already updated) url name of the entry
    /// \param prefix     the (directory or archive name) prefix
    /// \param sep        the separator to be used between prefix and file name
    /// \param udim_mode  the UDIM mode
    /// \param hash       if non-NULL, the hash value of the resource file
    static void parse_u_v(
        MDL_resource_set *s,
        char const       *name,
        size_t           ofs,
        char const       *url,
        string const     &prefix,
        char             sep,
        UDIM_mode        udim_mode,
        unsigned char    hash[16]);

    /// Create a resource set from a file mask describing files on a container.
    ///
    /// \param alloc            the allocator
    /// \param url              the absolute MDL url
    /// \param arc_name         the container name
    /// \param container_kind   the kind of container
    /// \param filename         the file name
    /// \param udim_mode        the UDIM mode
    static MDL_resource_set *from_mask_container(
        IAllocator        *alloc,
        char const        *url,
        char const        *container_name,
        File_handle::Kind container_kind,
        char const        *file_mask,
        UDIM_mode         udim_mode);

private:
    /// Constructor from one file name/url pair (typical case).
    ///
    /// \param alloc     the allocator
    /// \param url       the absolute MDL url
    /// \param filename  the file name
    /// \param hash      the resource hash value if any
    MDL_resource_set(
        IAllocator          *alloc,
        char const          *url,
        char const          *filename,
        unsigned char const hash[16]);

    /// Empty constructor from masks.
    ///
    /// \param alloc      the allocator
    /// \param udim_mode  the UDIM mode
    MDL_resource_set(
        IAllocator *alloc,
        UDIM_mode  udim_mode,
        char const *url_mask,
        char const *filename_mask);

private:
    /// The arena for allocation data.
    Memory_arena m_arena;

    /// An entry inside the resource file set;
    struct Resource_entry {
        /// Constructor.
        Resource_entry(
            char const          *url,
            char const          *filename,
            int                 u,
            int                 v,
            unsigned char const hash[16])
        : url(url), filename(filename), u(u), v(v), has_hash(false)
        {
            if (hash != NULL) {
                memcpy(this->hash, hash, sizeof(this->hash));
                has_hash = true;
            } else {
                memset(this->hash, 0, sizeof(this->hash));
            }
        }

        unsigned char hash[16];
        char const *url;
        char const *filename;
        int        u;
        int        v;
        bool       has_hash;
    };

    typedef Arena_vector<Resource_entry>::Type Entry_vec;

    /// The file name list.
    Entry_vec m_entries;

    /// The UDIM mode.
    UDIM_mode m_udim_mode;

    /// The url mask.
    string m_url_mask;

    /// The filename mask.
    string m_filename_mask;
};

// ------------------------------------------------------------------------

/// Implementation of the IMDL_import_result interface.
class MDL_import_result : public Allocator_interface_implement<IMDL_import_result>
{
    typedef Allocator_interface_implement<IMDL_import_result> Base;
    friend class Allocator_builder;

public:
    /// Return the absolute MDL name of the found entity, or NULL, if the entity could not
    /// be resolved.
    char const *get_absolute_name() const MDL_FINAL;

    /// Return the OS-dependent file name of the found entity, or NULL, if the entity could not
    /// be resolved.
    char const *get_file_name() const MDL_FINAL;

    /// Return an input stream to the given entity if found, NULL otherwise.
    IInput_stream *open(IThread_context *ctx) const MDL_FINAL;

private:
    /// Constructor.
    ///
    /// \param alloc         the allocator
    /// \param abs_name      the absolute name of the resolved import
    /// \param os_file_name  the OS dependent file name of the resolved module.
    MDL_import_result(
        IAllocator   *alloc,
        string const &abs_name,
        string const &os_file_name);

private:
    /// The absolute MDL name of the import.
    string const m_abs_name;

    /// The OS-dependent file name of the resolved module.
    string const m_os_file_name;
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
    explicit File_resource_reader(
        IAllocator  *alloc,
        File_handle *f,
        char const  *filename,
        char const  *mdl_url);
private:
    // non copyable
    File_resource_reader(File_resource_reader const &) MDL_DELETED_FUNCTION;
    File_resource_reader &operator=(File_resource_reader const &) MDL_DELETED_FUNCTION;

private:
    ~File_resource_reader() MDL_FINAL;

private:
    /// The file handle.
    File_handle *m_file;

    /// The filename.
    string m_file_name;

    /// The absolute MDl url.
    string m_mdl_url;
};

//-------------------------------------------------------------------------------------------------

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
    explicit Buffered_archive_resource_reader(
        IAllocator  *alloc,
        File_handle *f,
        char const  *filename,
        char const  *mdl_url);

private:
    // non copyable
    Buffered_archive_resource_reader(
        Buffered_archive_resource_reader const &) MDL_DELETED_FUNCTION;
    Buffered_archive_resource_reader &operator=(
        Buffered_archive_resource_reader const &) MDL_DELETED_FUNCTION;

private:
    ~Buffered_archive_resource_reader() MDL_FINAL;

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

//-------------------------------------------------------------------------------------------------

/// Implementation of the IEntity_resolver interface.
class Entity_resolver : public Allocator_interface_implement<IEntity_resolver>
{
    typedef Allocator_interface_implement<IEntity_resolver> Base;
    friend class Allocator_builder;
public:
    /// Resolve a resource file name.
    /// If \p owner_name and \p owner_file_path are not provided, no relative paths can be resolved.
    ///
    /// \param file_path         The MDL file path of the resource to resolve. In addition, for
    ///                          resources from MDLE files, it is also possible to provide the
    ///                          absolute OS file system path to the MDLE file (with slashes instead
    ///                          of backslashes on Windows), followed by a colon, followed by the
    ///                          relative path inside the MDLE container.
    /// \param owner_file_path   if non-NULL, the file path of the owner
    /// \param owner_name        if non-NULL, the absolute name of the owner
    /// \param pos               if non-NULL, the position of the import statement for error
    ///                          messages
    ///
    /// \return the set of resolved resources or NULL if this name could not be resolved
    IMDL_resource_set *resolve_resource_file_name(
        char const     *file_path,
        char const     *owner_file_path,
        char const     *owner_name,
        Position const *pos) MDL_FINAL;

    /// Resolve a module name.
    ///
    /// \param mdl               the (weak-)relative or absolute MDL module name to resolve
    /// \param owner_file_path   if non-NULL, the file path of the owner
    /// \param owner_name        if non-NULL, the absolute name of the owner
    /// \param pos               if non-NULL, the position of the import statement for error
    ///                          messages
    ///
    /// \return the absolute module name or NULL if this name could not be resolved
    IMDL_import_result *resolve_module(
        char const     *mdl_name,
        char const     *owner_file_path,
        char const     *owner_name,
        Position const *pos) MDL_FINAL;

    /// Access messages of last resolver operation.
    Messages const &access_messages() const MDL_FINAL;

    /// Get the allocator.
    IAllocator *get_allocator() { return m_resolver.get_allocator(); }

private:
    /// Constructor.
    ///
    /// \param alloc              the allocator
    /// \param compiler           the MDL compiler interface
    /// \param module_cache       the module cache if any
    /// \param external_resolver  the external entity resolver if any
    /// \param search_path        the search path to use
    Entity_resolver(
        IAllocator                               *alloc,
        MDL const                                *compiler,
        IModule_cache                            *module_cache,
        mi::base::Handle<IEntity_resolver> const &external_resolver,
        mi::base::Handle<IMDL_search_path> const &search_path);

private:
    /// Messages if any.
    Messages_impl m_msg_list;

    /// The used file resolver.
    File_resolver m_resolver;
};

/// Open a resource file read-only.
///
/// \param[in]  alloc          the allocator
/// \param[in]  abs_mdl_path   absolute MDL path of the resource
/// \param[in]  resource_path  the resource file path on file system
/// \param[out] err            error code
IMDL_resource_reader *open_resource_file(
    IAllocator                     *alloc,
    const char                     *abs_mdl_path,
    char const                     *resource_path,
    MDL_zip_container_error_code  &err);

/// Retarget a (relative path) resource url from one package to be accessible from another
/// package.
///
/// \param r         the resource
/// \param pos       the source code position of the resource for error messages
/// \param alloc     the allocator to be used
/// \param tf        the type factory for the new owner
/// \param vf        the value factory for the new owner
/// \param src       the source (owner) module of r
/// \param resolver  the file resolver to be used
///
/// \return a new resource owned by dst if the URL was rewritten or r itself
///         if no rewrite is necessary (because the URL is absolute)
IValue_resource const *retarget_resource_url(
    IValue_resource const *r,
    Position const        &pos,
    IAllocator            *alloc,
    Type_factory          &tf,
    Value_factory         &vf,
    IModule const          *src,
    File_resolver         &resolver);

/// Retarget a (relative path) resource url from one package to be accessible from another
/// package.
///
/// \param r         the resource
/// \param pos       the source code position of the resource for error messages
/// \param dst       the destination module
/// \param src       the source (owner) module of r
/// \param resolver  the file resolver to be used
///
/// \return a new resource owned by dst if the URL was rewritten or r itself
///         if no rewrite is necessary (because the URL is absolute)
IValue_resource const *retarget_resource_url(
    IValue_resource const *r,
    Position const        &pos,
    Module                *dst,
    Module const         *src,
    File_resolver         &resolver);

}  // mdl
}  // mi

#endif
