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

#ifndef MDL_COMPILERCORE_MANIFEST_H
#define MDL_COMPILERCORE_MANIFEST_H 1

#include <mi/base/handle.h>
#include <mi/base/iinterface.h>
#include <mi/mdl/mdl_mdl.h>
#include <mi/mdl/mdl_archiver.h>

#include "compilercore_allocator.h"
#include "compilercore_memory_arena.h"
#include "compilercore_cstring_hash.h"
#include "compilercore_streams.h"
#include "compilercore_printers.h"
#include "compilercore_modules.h"

namespace mi {
namespace mdl {

class MDL;
class MDL_zip_container_archive;

/// An export entry.
class Export_entry : public IArchive_manifest_export {
public:
    /// Constructor.
    Export_entry(
        char const *name)
    : m_name(name)
    , m_next(NULL)
    {
    }

    /// Get the name.
    char const *get_export_name() const MDL_FINAL { return m_name; }

    /// Get the next entry;
    Export_entry const *get_next() const MDL_FINAL { return m_next; }

    /// Get the next entry;
    Export_entry *get_next() { return m_next; }

    /// Set the next entry.
    void set_next(Export_entry *next) { m_next = next; }

private:
    char const * const m_name;  ///< The name.
    Export_entry       *m_next; ///< Points to the next entry.
};

/// A value entry.
class Value_entry : public IArchive_manifest_value {
public:
    /// Constructor.
    Value_entry(
        char const *value)
    : m_value(value)
    , m_next(NULL)
    {
    }

    /// Get the value.
    char const *get_value() const MDL_FINAL { return m_value; }

    /// Get the next entry;
    Value_entry const *get_next() const MDL_FINAL { return m_next; }

    /// Get the next entry;
    Value_entry *get_next() { return m_next; }

    /// Set the next entry.
    void set_next(Value_entry *next) { m_next = next; }

private:
    char const * const m_value;  ///< The value.
    Value_entry        *m_next;  ///< Points to the next entry.
};

// An dependency entry.
class Dependency_entry : public IArchive_manifest_dependency {
public:
    /// Get the archive name this archive depends on.
    char const *get_archive_name() const MDL_FINAL { return m_name; }

    /// Get the version this archive depends on.
    ISemantic_version const *get_version() const MDL_FINAL { return &m_ver; }

    /// Get the next value of the same key or NULL if this was the last one
    /// from the given key.
    Dependency_entry const *get_next() const MDL_FINAL { return m_next; }

    /// Get the next entry;
    Dependency_entry *get_next() { return m_next; }

    /// Set the next entry.
    void set_next(Dependency_entry *next) { m_next = next; }

public:
    /// Constructor.
    ///
    /// \param name  the archive name
    /// \param ver   the version of the archive
    Dependency_entry(
        char const             *name,
        Semantic_version const &ver)
    : m_name(name)
    , m_ver(ver)
    , m_next(NULL)
    {
    }

private:
    char const * const m_name;   ///< The archive name.
    Semantic_version   m_ver;    ///< The version.
    Dependency_entry   *m_next;  ///< Points to the next entry.
};

/// An Module entry.
class Module_entry {
public:
    typedef IArchive_manifest::Export_kind Export_kind;

public:
    /// Constructor.
    Module_entry(
    char const *module_name)
    : m_module_name(module_name)
    {
        for (size_t i = 0; i <= IArchive_manifest::EK_LAST; ++i) {
            m_exports[i] = NULL;
        }
    }

    /// Get the module absolute name.
    char const *get_module_name() const { return m_module_name; }

    /// Get the export list.
    Export_entry const *get_export(Export_kind kind) const { return m_exports[kind]; }

    /// Get the export list.
    Export_entry *get_export(Export_kind kind) { return m_exports[kind]; }

    /// Enter a new export list entry.
    void enter_export(Export_kind kind, Export_entry *e) {
        e->set_next(m_exports[kind]);
        m_exports[kind] = e;
    }

private:
    /// the fully qualified module name
    char const * const m_module_name;

    /// the export lists
    Export_entry       *m_exports[IArchive_manifest::EK_LAST + 1];
};

/// The Manifest is a container for meta-information.
class Manifest : public Allocator_interface_implement<IArchive_manifest> {
    typedef Allocator_interface_implement<IArchive_manifest> Base;
    friend class Archive;
    friend class File_handle;
    friend class MDL_zip_container_archive;
public:
    /// Acquires a const interface.
    ///
    /// If this interface is derived from or is the interface with the passed
    /// \p interface_id, then return a non-\c NULL \c const #mi::base::IInterface* that
    /// can be casted via \c static_cast to an interface pointer of the interface type
    /// corresponding to the passed \p interface_id. Otherwise return \c NULL.
    ///
    /// In the case of a non-\c NULL return value, the caller receives ownership of the
    /// new interface pointer, whose reference count has been retained once. The caller
    /// must release the returned interface pointer at the end to prevent a memory leak.
    mi::base::IInterface const *get_interface(
        mi::base::Uuid const &interface_id) const MDL_FINAL;

    /// Get the name of the archive this manifest belongs too.
    char const *get_archive_name() const MDL_FINAL;

    /// Get the MDL version of the archive.
    IMDL::MDL_version get_mdl_version() const MDL_FINAL;

    /// Get the semantic version of the archive.
    Semantic_version const *get_sema_version() const MDL_FINAL;

    /// Get the first dependency if any.
    Dependency_entry const *get_first_dependency() const MDL_FINAL;

    /// Get the authors list of the archive if any.
    Value_entry const *get_opt_author() const MDL_FINAL;

    /// Get the contributors list of the archive if any.
    Value_entry const *get_opt_contributor() const MDL_FINAL;

    /// Get the copyright notice of the archive if any.
    char const *get_opt_copyrigth_notice() const MDL_FINAL;

    /// Get the description of the archive if any.
    char const *get_opt_description() const MDL_FINAL;

    /// Get the created date of the archive if any.
    char const *get_opt_created() const MDL_FINAL;

    /// Get the modified date of the archive if any.
    char const *get_opt_modified() const MDL_FINAL;

    /// Get the number of modules inside the archive.
    size_t get_module_count() const MDL_FINAL;

    /// Get the i'th module name inside the archive.
    ///
    /// \param i  the module index
    ///
    /// \return the absolute module name without the leasing '::'
    char const *get_module_name(size_t i) const MDL_FINAL;

    /// Get the first export of the given kind from the given module.
    ///
    /// \param i     the module index
    /// \param kind  the export kind
    ///
    /// \return the first export entry of the given kind or NULL
    Export_entry const *get_first_export(
        size_t      i,
        Export_kind kind) const MDL_FINAL;

    /// Get the number of all (predefined and user supplied) keys.
    size_t get_key_count() const MDL_FINAL;

    /// Get the i'th key.
    ///
    /// \param i  the key index
    ///
    /// \return the key
    ///
    /// \note keys with index < PK_FIRST_USER_ID are the predefined keys
    char const *get_key(size_t i) const MDL_FINAL;

    /// Get the first value from the given key index.
    ///
    /// \param i     the key index
    ///
    /// \return the first value entry or NULL if this key has no value set
    ///
    /// \note if i < PK_FIRST_USER_ID, this is a predefined key. Use the specific
    ///       access functions, get_first_value() will return NULL here
    Value_entry const *get_first_value(size_t i) const MDL_FINAL;

    /// Add a key, value pair. Works for predefined and user keys.
    ///
    /// \param key    the key to set
    /// \param value  the value
    Error_code add_key_value(
        char const *key,
        char const *value) MDL_FINAL;

    // ----------------------- non-interface -------------------

    /// Add a MDL version.
    void add_mdl_version(IMDL::MDL_version version);

    /// Set a MDL version.
    void set_mdl_version(IMDL::MDL_version version);

    /// Set the semantic version of the archive.
    void set_sema_version(char const *version);

    /// Add a dependency of the archive.
    void add_dependency(
        char const             *dependency,
        Semantic_version const &ver);

    /// Add an author of the archive.
    void add_opt_author(char const *author);

    /// Add an contributor of the archive.
    void add_opt_contributor(char const *contributor);

    /// Set the copyright notice of the archive.
    void set_opt_copyright_notice(char const *copyright_notice);

    /// Set the description of the archive.
    void set_opt_description(char const *description);

    /// Set the created date of the archive.
    void set_opt_created(char const *created);

    /// Set the modified date of the archive.
    void set_opt_modified(char const *modified);

    /// Get the i'th module.
    ///
    /// \param i     the module index
    Module_entry const *get_module(size_t i) const;

    /// Add a module.
    ///
    /// \param abs_name  the absolute module name
    ///
    /// \return the index of this module
    size_t add_module(char const *abs_name);

    /// Add an export.
    ///
    /// \param kind   kind of this export
    /// \param id     the owner module ID
    /// \param ident  the unqualified name of this export
    void add_export(Export_kind kind, size_t mod_id, char const *ident);

    /// Add a user supplied key, value pair.
    ///
    /// \param key           the key
    /// \param value         the value to add
    void add_user_pair(
        char const *key,
        char const *value);

public:
    /// Constructor.
    ///
    /// \param alloc  the allocator
    explicit Manifest(
        IAllocator *alloc);

private:
    /// Set the archive name.
    void set_archive_name(char const *archive_name);

    /// Add an key, value pair.
    ///
    /// \param key           the key
    /// \param is_user_key   true, if this is a user supplied key
    /// \param value         the value to add
    void add_pair(
        char const *key,
        bool       is_user_key,
        char const *value);

    /// Check time format.
    bool check_time_format(char const *s);

    /// Check version format.
    bool check_version_format(char const *s);

private:
    /// The arena for all allocated strings.
    Memory_arena m_arena;

    /// The name of the archive.
    char const *m_archive_name;

    /// The MDL version of the archive, mandatory.
    IMDL::MDL_version m_mdl_version;

    /// The semantic version of the archive, mandatory.
    Semantic_version m_sema_version;

    /// The first dependency if any.
    Dependency_entry *m_dependecies;

    /// The copyright notice, optional.
    string m_copyright_notice;

    /// The description, optional.
    string m_description;

    /// The created date, optional.
    string m_created;

    /// The modified date, optional.
    string m_modified;

    typedef vector<Module_entry *>::Type Module_vec;

    /// All modules inside the archive.
    Module_vec m_modules;

    typedef hash_map<
        char const *,
        Value_entry *,
        cstring_hash,
        cstring_equal_to>::Type Key_map;

    /// The key value map;
    Key_map m_key_map;

    typedef vector<char const *>::Type Key_vec;

    /// All user supplied keys.
    Key_vec m_user_keys;
};

///
/// A printer for archive manifests.
///
class Manifest_printer : public Allocator_interface_implement<IPrinter_interface> {
    typedef Allocator_interface_implement<IPrinter_interface> Base;
    friend class Allocator_builder;
public:
    /// Print the interface to the given printer.
    void print(Printer *printer, mi::base::IInterface const *code) const MDL_FINAL;

private:
    /// Prints a list.
    void print_list(
        Printer                       *printer,
        char const                    *key,
        IArchive_manifest_value const *value) const;

protected:
    /// Constructor.
    ///
    /// \param alloc  the allocator
    Manifest_printer(IAllocator *alloc);
};

}  // mdl
}  // mi

#endif
