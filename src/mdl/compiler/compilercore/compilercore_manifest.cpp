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

#include "compilercore_mdl.h"
#include "compilercore_manifest.h"

namespace mi {
namespace mdl {

// Constructor.
Manifest::Manifest(
    IAllocator *alloc)
: Base(alloc)
, m_arena(alloc)
, m_archive_name(NULL)
, m_mdl_version(IMDL::MDL_VERSION_1_0)
, m_sema_version(1, 0, 0, "")
, m_dependecies(NULL)
, m_copyright_notice(alloc)
, m_description(alloc)
, m_created(alloc)
, m_modified(alloc)
, m_modules(alloc)
, m_key_map(0, Key_map::hasher(), Key_map::key_equal(), alloc)
, m_user_keys(alloc)
{
}

// Acquires a const interface.
mi::base::IInterface const *Manifest::get_interface(
    mi::base::Uuid const &interface_id) const
{
    if (interface_id == IPrinter_interface::IID()) {
        Allocator_builder builder(get_allocator());
        return builder.create<Manifest_printer>(get_allocator());
    }
    return Base::get_interface(interface_id);
}

// Get the name of the archive this manifest belongs too.
char const *Manifest::get_archive_name() const
{
    return m_archive_name;
}

// Get the MDL version of the archive.
IMDL::MDL_version Manifest::get_mdl_version() const
{
    return m_mdl_version;
}

// Get the semantic version of the archive.
Semantic_version const *Manifest::get_sema_version() const
{
    return &m_sema_version;
}

// Get the first dependency if any.
Dependency_entry const *Manifest::get_first_dependency() const
{
    return m_dependecies;
}

// Get the author of the archive if any
Value_entry const *Manifest::get_opt_author() const
{
    Key_map::const_iterator it(m_key_map.find(get_key(PK_AUTHOR)));
    if (it != m_key_map.end())
        return it->second;
    return NULL;
}

// Get the author of the archive if any.
Value_entry const *Manifest::get_opt_contributor() const
{
    Key_map::const_iterator it(m_key_map.find(get_key(PK_CONTRIBUTOR)));
    if (it != m_key_map.end())
        return it->second;
    return NULL;
}

// Get the copyright notice of the archive if any.
char const *Manifest::get_opt_copyrigth_notice() const
{
    if (m_copyright_notice.empty())
        return NULL;
    return m_copyright_notice.c_str();
}

// Get the description of the archive if any.
char const *Manifest::get_opt_description() const
{
    if (m_description.empty())
        return NULL;
    return m_description.c_str();
}

// Get the created date of the archive if any.
char const *Manifest::get_opt_created() const
{
    if (m_created.empty())
        return NULL;
    return m_created.c_str();
}

// Get the modified date of the archive if any.
char const *Manifest::get_opt_modified() const
{
    if (m_modified.empty())
        return NULL;
    return m_modified.c_str();
}

// Get the number of modules inside the archive.
size_t Manifest::get_module_count() const
{
    return m_modules.size();
}

// Get the i'th module name inside the archive.
char const *Manifest::get_module_name(size_t i) const
{
    Module_entry const *e = get_module(i);
    return e != NULL ? e->get_module_name() : NULL;
}

// Get the first export of the given kind from the given module.
Export_entry const *Manifest::get_first_export(
    size_t      i,
    Export_kind kind) const
{
    Module_entry const *e = get_module(i);

    return e != NULL ? e->get_export(kind) : NULL;
}

// Get the number of all (predefined and user supplied) keys.
size_t Manifest::get_key_count() const
{
    return PK_FIRST_USER_ID + m_user_keys.size();
}

// Get the i'th key.
char const *Manifest::get_key(size_t i) const
{
    if (i < PK_FIRST_USER_ID) {
        switch (Predefined_key(i)) {
        case PK_MDL:              return "mdl";
        case PK_VERSION:          return "version";
        case PK_DEPENDENCY:       return "dependency";
        case PK_MODULE:           return "module";
        case PK_EX_FUNCTION:      return "exports.function";
        case PK_EX_MATERIAL:      return "exports.material";
        case PK_EX_STRUCT:        return "exports.struct";
        case PK_EX_ENUM:          return "exports.enum";
        case PK_EX_CONST:         return "exports.const";
        case PK_EX_ANNOTATION:    return "exports.annotation";
        case PK_AUTHOR:           return "author";
        case PK_CONTRIBUTOR:      return "contributor";
        case PK_COPYRIGHT_NOTICE: return "copyright_notice";
        case PK_DESCRIPTION:      return "description";
        case PK_CREATED:          return "created";
        case PK_MODIFIED:         return "modified";
        case PK_FIRST_USER_ID:
            break;
        }
    }
    i -= PK_FIRST_USER_ID;
    if (i < m_user_keys.size())
        return m_user_keys[i];
    return NULL;
}

// Add a key, value pair. Works for predefined and user keys.
Value_entry const *Manifest::get_first_value(size_t i) const
{
    if (i < PK_FIRST_USER_ID) {
        // use the specific accessors
        return NULL;
    }
    if (i - PK_FIRST_USER_ID < m_user_keys.size()) {
        Key_map::const_iterator it(m_key_map.find(get_key(i)));

        if (it != m_key_map.end())
            return it->second;
    }
    return NULL;
}

// Add a key, value pair.
Manifest::Error_code Manifest::add_key_value(
    char const *key,
    char const *value)
{
    if (key == NULL || value == NULL)
        return ERR_NULL_ARG;

    switch (key[0]) {
    case 'a':
        if (strcmp("author", key) == 0) {
            add_opt_author(value);
            return ERR_OK;
        }
        break;
    case 'c':
        if (strcmp("contributor", key) == 0) {
            add_opt_contributor(value);
            return ERR_OK;
        }
        if (strcmp("copyright_notice", key) == 0) {
            // FIXME: check set
            set_opt_copyright_notice(value);
            return ERR_OK;
        }
        if (strcmp("created", key) == 0) {
            if (!check_time_format(value))
                return ERR_TIME_FORMAT;
            set_opt_created(value);
            return ERR_OK;
        }
        break;
    case 'd':
        if (strcmp("dependency", key) == 0) {
            return ERR_FORBIDDEN;
        }
        if (strcmp("description", key) == 0) {
            // FIXME: check set
            set_opt_description(value);
            return ERR_OK;
        }
        break;
    case 'e':
        if (strcmp("exports.function", key) == 0)
            return ERR_FORBIDDEN;
        if (strcmp("exports.material", key) == 0)
            return ERR_FORBIDDEN;
        if (strcmp("exports.struct", key) == 0)
            return ERR_FORBIDDEN;
        if (strcmp("exports.enum", key) == 0)
            return ERR_FORBIDDEN;
        if (strcmp("exports.const", key) == 0)
            return ERR_FORBIDDEN;
        if (strcmp("exports.annotation", key) == 0)
            return ERR_FORBIDDEN;
        break;
    case 'm':
        if (strcmp("mdl", key) == 0)
            return ERR_FORBIDDEN;
        if (strcmp("module", key) == 0)
            return ERR_FORBIDDEN;
        if (strcmp("modified", key) == 0) {
            if (!check_time_format(value))
                return ERR_TIME_FORMAT;
            set_opt_modified(value);
            return ERR_OK;
        }
        break;
    case 'v':
        if (strcmp("version", key) == 0) {
            if (!check_version_format(value))
                return ERR_VERSION_FORMAT;
            set_sema_version(value);
            return ERR_OK;
        }
        break;
    }

    // user key
    add_user_pair(key, value);
    return ERR_OK;
}

// Add a MDL version
void Manifest::add_mdl_version(IMDL::MDL_version version)
{
    if (version > m_mdl_version)
        m_mdl_version = version;
}

// Set a MDL version.
void Manifest::set_mdl_version(IMDL::MDL_version version)
{
    m_mdl_version = version;
}

// Set the semantic version of the archive.
void Manifest::set_sema_version(char const *version)
{
#define DIGIT(c)  ('0' <= (c) && (c) <= '9')

    char const *s = version;
    int major = 0, minor = 0, patch = 0;

    while (DIGIT(*s)) {
        major = major * 10 + *s - '0';
        ++s;
    }

    if (*s == '.') ++s;

    while (DIGIT(*s)) {
        minor = minor * 10 + *s - '0';
        ++s;
    }

    if (*s == '.') ++s;

    while (DIGIT(*s)) {
        patch = patch * 10 + *s - '0';
        ++s;
    }

    char const *prerelease = "";
    if (*s == '-') {
        ++s;
        prerelease = Arena_strdup(m_arena, s);
    } else if (*s != '\0') {
        MDL_ASSERT(!"wrong semantic version format");
    }

    m_sema_version = Semantic_version(major, minor, patch, prerelease);
}

// Add a dependency of the archive.
void Manifest::add_dependency(
    char const             *dependency,
    Semantic_version const &ver)
{
    Arena_builder builder(m_arena);

    // We must copy the prerelease string here
    Semantic_version v(
        ver.get_major(),
        ver.get_minor(),
        ver.get_patch(),
        Arena_strdup(m_arena, ver.get_prerelease()));

    Dependency_entry *entry =
        builder.create<Dependency_entry>(Arena_strdup(m_arena, dependency), v);
    entry->set_next(m_dependecies);
    m_dependecies = entry;
}

// Add an author of the archive.
void Manifest::add_opt_author(char const *author)
{
    return add_pair(get_key(PK_AUTHOR), /*need_copy=*/false, author);
}

// Add an contributor of the archive.
void Manifest::add_opt_contributor(char const *contributor)
{
    return add_pair(get_key(PK_CONTRIBUTOR), /*need_copy=*/false, contributor);
}

// Set the copyright notice of the archive.
void Manifest::set_opt_copyright_notice(char const *copyright_notice)
{
    m_copyright_notice = copyright_notice == NULL ? "" : copyright_notice;
}

// Set the description of the archive.
void Manifest::set_opt_description(char const *description)
{
    m_description = description == NULL ? "" : description;
}

// Set the created date of the archive.
void Manifest::set_opt_created(char const *created)
{
    m_created = created == NULL ? "" : created;
}

// Set the modified date of the archive.
void Manifest::set_opt_modified(char const *modified)
{
    m_modified = modified== NULL ? "" : modified;
}

// Add a module.
size_t Manifest::add_module(char const *abs_name)
{
    if (abs_name[0] == ':' && abs_name[1] == ':')
        abs_name += 2;

    abs_name = Arena_strdup(m_arena, abs_name);

    Arena_builder builder(m_arena);

    Module_entry *e = builder.create<Module_entry>(abs_name);
    size_t res = m_modules.size();
    m_modules.push_back(e);
    return res;
}

// Get the i'th module.
Module_entry const *Manifest::get_module(size_t i) const
{
    if (i < m_modules.size())
        return m_modules[i];
    return NULL;
}

// Add an export.
void Manifest::add_export(Export_kind kind, size_t mod_id, char const *ident)
{
    Arena_builder builder(m_arena);

    ident = Arena_strdup(m_arena, ident);
    Export_entry *e = builder.create<Export_entry>(ident);

    m_modules[mod_id]->enter_export(kind, e);
}

// Add an key, value pair.
void Manifest::add_user_pair(char const *key, char const *value)
{
    return add_pair(key, /*need_copy=*/true, value);
}

// Set the archive name.
void Manifest::set_archive_name(char const *archive_name)
{
    m_archive_name = Arena_strdup(m_arena, archive_name);
}

// Add an key, value pair.
void Manifest::add_pair(char const *key, bool user_key, char const *value)
{
    Key_map::iterator it(m_key_map.find(key));

    Value_entry *p = NULL;
    if (it != m_key_map.end()) {
        p = it->second;
        for (;;) {
            Value_entry *q = p->get_next();

            if (q == NULL)
                break;
            p = q;
        }
    }

    Arena_builder builder(m_arena);

    value = Arena_strdup(m_arena, value);
    Value_entry *e = builder.create<Value_entry>(value);

    if (p != NULL)
        p->set_next(e);
    else {
        if (user_key) {
            key = Arena_strdup(m_arena, key);
            m_user_keys.push_back(key);
        }
        m_key_map.insert(Key_map::value_type(key, e));
    }
}

// Check time format.
bool Manifest::check_time_format(char const *s)
{
#define DIGIT(c)  ('0' <= (c) && (c) <= '9')

    if (!DIGIT(s[0]))
        return false;
    if (!DIGIT(s[1]))
        return false;
    if (!DIGIT(s[2]))
        return false;
    if (!DIGIT(s[3]))
        return false;
    if (s[4] != '-')
        return false;
    if (!DIGIT(s[5]))
        return false;
    if (!DIGIT(s[6]))
        return false;
    if (s[7] != '-')
        return false;
    if (!DIGIT(s[8]))
        return false;
    if (!DIGIT(s[9]))
        return false;
    if (s[10] != '\0' && s[10] != ',')
        return false;
    return true;
#undef DIGIT
}

// Check time format.
bool Manifest::check_version_format(char const *s)
{
#define DIGIT(c)  ('0' <= (c) && (c) <= '9')

    // major
    if (!DIGIT(*s))
        return false;
    do {
        ++s;
    } while(DIGIT(*s));

    if (*s != '.')
        return false;
    ++s;

    // minor
    if (!DIGIT(*s))
        return false;
    do {
        ++s;
    } while(DIGIT(*s));

    if (*s != '.')
        return false;
    ++s;

    // patch
    if (!DIGIT(*s))
        return false;
    do {
        ++s;
    } while(DIGIT(*s));

    // prerelease
    if (*s != '-' && *s != '\0')
        return false;

    return true;
#undef DIGIT
}
// -------------------------------- printer --------------------------------

// Constructor.
Manifest_printer::Manifest_printer(IAllocator *alloc)
: Base(alloc)
{
}

// Prints a list.
void Manifest_printer::print_list(
    Printer                       *printer,
    char const                    *key,
    IArchive_manifest_value const *value) const
{
    for (IArchive_manifest_value const *p = value; p != NULL; p = p->get_next()) {
        printer->printf("%s = \"%s\"\n", key, p->get_value());
    }
}

// Print the interface to the given printer.
void Manifest_printer::print(Printer *printer, mi::base::IInterface const *iface) const
{
    mi::base::Handle<IArchive_manifest const> manifest(
        iface->get_interface<IArchive_manifest>());

    if (!manifest.is_valid_interface())
        return;

    // mandatory fields

    char const *s = "1.0";
    switch (manifest->get_mdl_version()) {
    case IMDL::MDL_VERSION_1_0: s = "1.0"; break;
    case IMDL::MDL_VERSION_1_1: s = "1.1"; break;
    case IMDL::MDL_VERSION_1_2: s = "1.2"; break;
    case IMDL::MDL_VERSION_1_3: s = "1.3"; break;
    case IMDL::MDL_VERSION_1_4: s = "1.4"; break;
    case IMDL::MDL_VERSION_1_5: s = "1.5"; break;
    case IMDL::MDL_VERSION_1_6: s = "1.6"; break;
    case IMDL::MDL_VERSION_1_7: s = "1.7"; break;
    }
    printer->printf("%s = \"%s\"\n", manifest->get_key(IArchive_manifest::PK_MDL), s);

    ISemantic_version const *ver = manifest->get_sema_version();
    int major = ver->get_major();
    int minor = ver->get_minor();
    int patch = ver->get_patch();
    char const *prerelease = ver->get_prerelease();

    printer->printf("%s = \"%d.%d.%d%s%s\"\n",
        manifest->get_key(IArchive_manifest::PK_VERSION),
        major,
        minor,
        patch,
        prerelease != NULL && prerelease[0] != '\0' ? "-" : "",
        prerelease);

    for (IArchive_manifest_dependency const *p = manifest->get_first_dependency();
        p != NULL;
        p = p->get_next())
    {
        ISemantic_version const *ver        = p->get_version();
        char const              *prerelease = ver->get_prerelease();
        printer->printf(
            "%s = \"%s %d.%d.%d%s%s\"\n",
            manifest->get_key(IArchive_manifest::PK_DEPENDENCY),
            p->get_archive_name(),
            ver->get_major(),
            ver->get_minor(),
            ver->get_patch(),
            prerelease != NULL && prerelease[0] != '\0' ? "-" : "",
            prerelease);
    }

    // modules
    for (size_t index = 0, n = manifest->get_module_count(); index < n; ++index) {
        string name("::", get_allocator());

        name.append(manifest->get_module_name(index));
        printer->printf(
            "%s = \"%s\"\n", manifest->get_key(IArchive_manifest::PK_MODULE), name.c_str());
    }

    // generate exports
    for (int j = 0; j <= IArchive_manifest::EK_LAST; ++j) {
        IArchive_manifest::Export_kind export_index = IArchive_manifest::Export_kind(j);

        char const *key = "???";
        switch (export_index) {
        case IArchive_manifest::EK_FUNCTION:
            key = "exports.function";
            break;
        case IArchive_manifest::EK_MATERIAL:
            key = "exports.material";
            break;
        case IArchive_manifest::EK_STRUCT:
            key = "exports.struct";
            break;
        case IArchive_manifest::EK_ENUM:
            key = "exports.enum";
            break;
        case IArchive_manifest::EK_CONST:
            key = "exports.const";
            break;
        case IArchive_manifest::EK_ANNOTATION:
            key = "exports.annotation";
            break;
        }

        for (size_t index = 0, n = manifest->get_module_count(); index < n; ++index) {
            string prefix("::", get_allocator());

            prefix.append(manifest->get_module_name(index));
            if (prefix.length() > 2)
                prefix.append("::");

            for (IArchive_manifest_export const *e =
                    manifest->get_first_export(index, export_index);
                 e != NULL;
                 e = e->get_next())
            {
                string name(prefix);
                name.append(e->get_export_name());

                printer->printf("%s = \"%s\"\n", key, name.c_str());
            }
        }
    }

    // optionally fields
    if (IArchive_manifest_value const *authors = manifest->get_opt_author())
        print_list(printer, manifest->get_key(IArchive_manifest::PK_AUTHOR), authors);
    if (IArchive_manifest_value const *contributors = manifest->get_opt_contributor())
        print_list(printer, manifest->get_key(IArchive_manifest::PK_CONTRIBUTOR), contributors);
    if (char const *copyright_notice = manifest->get_opt_copyrigth_notice())
        printer->printf("copyright_notice = \"%s\"\n", copyright_notice);
    if (char const *description = manifest->get_opt_description())
        printer->printf("description = \"%s\"\n", description);
    if (char const *created = manifest->get_opt_created())
        printer->printf("created = \"%s\"\n", created);
    if (char const *modified = manifest->get_opt_modified())
        printer->printf("modified = \"%s\"\n", modified);

    // user defined
    for (size_t i = IArchive_manifest::PK_FIRST_USER_ID, n = manifest->get_key_count();
         i < n;
         ++i)
    {
        char const *key = manifest->get_key(i);
        if (IArchive_manifest_value const *value = manifest->get_first_value(i))
            print_list(printer, key, value);
    }
}

}  // mdl
}  // mi
