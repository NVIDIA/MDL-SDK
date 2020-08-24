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
#include "command.h"
#include "archive.h"
#include "util.h"
#include "options.h"
#include "errors.h"
#include "version.h"
#include "search_path.h"
#include <base/util/string_utils/i_string_utils.h>
#include <base/hal/hal/i_hal_ospath.h>
#include <base/hal/disk/disk.h>
#include <map>
#include <iostream>
using namespace mdlm;
using std::vector;
using std::string;
using std::cout;
using std::endl;
using std::find;
using std::map;
using mi::base::Handle;
using mi::neuraylib::IMdl_info;
using mi::neuraylib::IMdl_impexp_api;

// In the GNU C Library, "minor" and "major" are defined
#ifdef minor
    #undef minor
#endif

#ifdef major
    #undef major
#endif

namespace mdlm
{
    extern mi::neuraylib::INeuray * neuray(); //Application::theApp().neuray(
    extern void report(const std::string & msg); //Application::theApp().report(
    extern bool freeimage_available(); //Application::theApp().freeimage_available(
}

int Command::execute()
{
    return 0;
}

class Compatibility_api_helper
{
    mi::base::Handle<mi::neuraylib::IMdl_compatibility_api> m_compatibility_api;

private:
    void handle_message(const mi::neuraylib::IMessage * message) const
    {
        mi::base::Message_severity severity(message->get_severity());
        const char* str = message->get_string();
        //mi::Sint32 code(message->get_code());
        if (str)
        {
            Util::log(str, severity);
        }
        for (mi::Size i = 0; i < message->get_notes_count(); i++)
        {
            mi::base::Handle<const mi::neuraylib::IMessage> note(message->get_note(i));
            handle_message(note.get());
        }
    }
    void handle_return_code(const mi::Sint32 & code, bool ctx_empty) const
    {
        if (code == -1)
        {
            Util::log_error("Comparison failed: Invalid parameters");
        }
        else if (code == -2)
        {
            if (ctx_empty) {
                // error code -2 implies additional errors reported through the context
                Util::log_error("An error occurred during module comparison");
            } else {
                Util::log_error("Comparison failed:");
            }
        }
    }
    void handle_return_code_and_messages(
        const mi::Sint32 & code, const mi::neuraylib::IMdl_execution_context * context) const
    {
        mi::Size msg_n = context->get_messages_count();
        mi::Size err_n = context->get_error_messages_count();

        handle_return_code(code, msg_n == 0 && err_n == 0);

        for (mi::Size i = 0; i < msg_n; ++i)
        {
            mi::base::Handle<const mi::neuraylib::IMessage> msg(context->get_message(i));
            check_success(msg != NULL);
            handle_message(msg.get());
        }
        for (mi::Size i = 0; i < err_n; ++i)
        {
            mi::base::Handle<const mi::neuraylib::IMessage> msg(context->get_error_message(i));
            check_success(msg != NULL);
            handle_message(msg.get());
        }
    }

public:
    Compatibility_api_helper() 
    {
        mi::neuraylib::INeuray * neuray(mdlm::neuray());
        mi::base::Handle<mi::neuraylib::IMdl_compatibility_api> compatibility_api
        (
            neuray->get_api_component<mi::neuraylib::IMdl_compatibility_api>()
        );
        check_success(compatibility_api != NULL);
        m_compatibility_api = mi::base::make_handle_dup(compatibility_api.get());
    }
public:
    static bool Test()
    {
        return true;
    }
public:
    mi::Sint32 compare_modules(std::string & m1, std::string & m2)
    {
        mi::neuraylib::INeuray * neuray(mdlm::neuray());
        mi::base::Handle<mi::neuraylib::IMdl_factory> mdl_factory
        (
            neuray->get_api_component<mi::neuraylib::IMdl_factory>()
        );
        mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
            mdl_factory->create_execution_context());

        mi::Sint32 rtn = m_compatibility_api->compare_modules(
            m1.c_str(), m2.c_str(), NULL /*search_paths*/, context.get());

        handle_return_code_and_messages(rtn, context.get());

        return rtn;
    }

    mi::Sint32 compare_archives(std::string & a1, std::string & a2)
    {
        mi::neuraylib::INeuray * neuray(mdlm::neuray());
        mi::base::Handle<mi::neuraylib::IMdl_factory> mdl_factory
        (
            neuray->get_api_component<mi::neuraylib::IMdl_factory>()
        );
        mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
            mdl_factory->create_execution_context());

        mi::Sint32 rtn = m_compatibility_api->compare_archives(
            a1.c_str(), a2.c_str(), NULL /*search_paths*/, context.get());

        handle_return_code_and_messages(rtn, context.get());

        return rtn;
    }
};

Compatibility::Compatibility(const string & old_archive, const string & new_archive)
    : m_old_archive(Archive::with_extension(old_archive))
    , m_new_archive(Archive::with_extension(new_archive))
    , m_compatible(UNDEFINED)
    , m_test_version_only(false)
{
}

int Compatibility::execute()
{
    m_compatible = UNDEFINED;

    std::string old_archive_path(m_old_archive);
    std::string new_archive_path(m_new_archive);
    if (Util::equivalent(old_archive_path, new_archive_path))
    {
        Util::log_report("compatibility: Same files");
        m_compatible = COMPATIBLE;
        report_compatibility_result();
        return COMPATIBLE;
    }

    Archive oldarch(m_old_archive);
    if (!oldarch.is_valid())
    {
        Util::log_error("Invalid archive: " + Util::normalize(oldarch.full_name()));
        return INVALID_ARCHIVE;
    }
    Util::log_verbose("Archive 1 set: " + Util::normalize(oldarch.full_name()));

    Archive newarch(m_new_archive);
    if (!newarch.is_valid())
    {
        Util::log_error("Invalid archive: " + Util::normalize(newarch.full_name()));
        return INVALID_ARCHIVE;
    }
    Util::log_verbose("Archive 2 set: " + Util::normalize(newarch.full_name()));

    Version oldVersion;
    Version newVersion;
    check_success(0 == oldarch.get_version(oldVersion));
    check_success(0 == newarch.get_version(newVersion));

    Util::log_debug("Test Archive Compatibility");
    Util::log_debug("\t Archive 1 (" + to_string(oldVersion) + "): " + Util::normalize(oldarch.full_name()));
    Util::log_debug("\t Archive 2 (" + to_string(newVersion) + "): " + Util::normalize(newarch.full_name()));

    if (oldarch.base_name() != newarch.base_name())
    {
        Util::log_fatal("Archives are incompatible: different names");
        m_compatible = NOT_COMPATIBLE;
        report_compatibility_result();
        return ARCHIVE_DIFFERENT_NAME;
    }

    Util::log_verbose("Archive 1 version: " + to_string(oldVersion));
    Util::log_verbose("Archive 2 version: " + to_string(newVersion));

    if (oldVersion == newVersion)
    {
        Util::log_debug("Compatible archives: Archives are of the same version: "
            + to_string(newVersion));
        // Note: Even if the version are the same it makes sense to
        // look into each of the archives to check all the modules
        // Therefore continue...
        m_compatible = COMPATIBLE_SAME_VERSION;
    }
    else
    {
        if (newVersion < oldVersion)
        {
            Util::log_debug("Incompatible archives: Archive 1 is newer than archive 2");
            m_compatible = NOT_COMPATIBLE;
            report_compatibility_result();
            return NOT_COMPATIBLE;
        }
        else if (newVersion.major() != oldVersion.major())
        {
            Util::log_debug("Incompatible archives: Major version changed");
            m_compatible = NOT_COMPATIBLE;
            report_compatibility_result();
            return NOT_COMPATIBLE;
        }
        else
        {
            m_compatible = COMPATIBLE;
        }
    }
    if (m_test_version_only)
    {
        // stop here
        report_compatibility_result();
        if (m_compatible == COMPATIBLE || m_compatible == COMPATIBLE_SAME_VERSION)
        {
            return COMPATIBLE;
        }
        return NOT_COMPATIBLE;
    }

    Compatibility_api_helper helper;
    mi::Sint32 rtn = helper.compare_archives(m_old_archive, m_new_archive);

    if (rtn == 0)
    {
        m_compatible = COMPATIBLE;
    }
    else if (rtn == -2)
    {
        m_compatible = NOT_COMPATIBLE;
    }
    else
    {
        m_compatible = UNDEFINED;
    }

    report_compatibility_result();

    if (m_compatible == COMPATIBLE || m_compatible == COMPATIBLE_SAME_VERSION)
    {
        return COMPATIBLE;
    }
    return NOT_COMPATIBLE;
}

void Compatibility::report_compatibility_result() const
{
    // Report the result of the compatibility check
    // This is logged as log info
    Archive oldarch(m_old_archive);
    Archive newarch(m_new_archive);
    string str;
    str = "Compatibility test ";
    str += (m_test_version_only ? "(version only)" : "(full test)");
    Util::log_info(str);
    Util::log_info("New archive: " + Util::normalize(newarch.full_name()));
    if (m_compatible == COMPATIBLE)
    {
        str = "is compatible with old archive: ";
    }
    else if (m_compatible == COMPATIBLE_SAME_VERSION)
    {
        str = "is compatible (same version) with archive: ";
    }
    else if (m_compatible == NOT_COMPATIBLE)
    {
        str = "is not compatible with archive: ";
    }
    else
    {
        str = "Undefined result for compatibility test: ";
    }
    str += Util::normalize(oldarch.full_name());
    Util::log_info(str);
}

Create_archive::Create_archive(const std::string & mdl_directory, const std::string & archive)
    : m_mdl_directory(mdl_directory)
    , m_archive(Archive::with_extension(archive))
{
    // Convert SYSTEM, USER to real directory locations
    Util::File::convert_symbolic_directory(m_mdl_directory);
}

int Create_archive::execute()
{
    mi::base::Handle<mi::neuraylib::IMdl_archive_api> archive_api
    (
        mdlm::neuray()->get_api_component<mi::neuraylib::IMdl_archive_api>()
    );

    /// \param manifest_fields   A static or dynamic array of structs of type \c "Manifest_field"
    ///                          which holds fields with optional or user-defined keys to be added
    ///                          to the manifest. The struct has two members, \c "key" and
    ///                          \c "value", both of type \c "String". \c NULL is treated like an
    ///                          empty array.
    mi::base::Handle<mi::neuraylib::IDatabase> database
    (
        mdlm::neuray()->get_api_component<mi::neuraylib::IDatabase>()
    );
    mi::base::Handle<mi::neuraylib::IScope> scope(database->get_global_scope());
    mi::base::Handle<mi::neuraylib::ITransaction> transaction(scope->create_transaction());
    mi::base::Handle<mi::IDynamic_array> manifest_fields
    (
        transaction->create<mi::IDynamic_array>("Manifest_field[]")
    );
    manifest_fields->set_length(m_keys.size());

    vector<KeyValuePair>::iterator it;
    mi::Size i;
    for (it = m_keys.begin(), i = 0; it != m_keys.end(); it++, i++)
    {
        mi::base::Handle<mi::IStructure> field(manifest_fields->get_value<mi::IStructure>(i));
        mi::base::Handle<mi::IString> key(field->get_value<mi::IString>("key"));
        key->set_c_str(it->first.c_str());
        mi::base::Handle<mi::IString> value(field->get_value<mi::IString>("value"));
        value->set_c_str(it->second.c_str());
    }

    mi::Sint32 rtn = archive_api->create_archive(
          m_mdl_directory.c_str()
        , m_archive.c_str()
        , manifest_fields.get()
    );

    transaction->commit();

    std::map<int, string> errors = 
    { 
          { -1, "Invalid parameters" }
        , { -2, "archive does not end with \""+ Archive::extension +"\"" }
        , { -3, string("An array element of manifest_fields or a struct member ") +
                       "of one of the array elements has an incorrect type." }
        ,{ -4, "Failed to create the archive" }
    };
    if (rtn != 0)
    {
        if (!errors[rtn].empty())
        {
            Util::log_error(errors[rtn]);
        }
    }
    else
    {
        Util::log_info("Archive successfully created: " + m_archive);
    }
    return rtn == 0 ? SUCCESS : UNSPECIFIED_FAILURE;
}

int Create_archive::add_key_value(const std::string & key_value)
{
    size_t pos = key_value.find('=');
    if (pos != std::string::npos)
    {
        KeyValuePair keyvalue(key_value.substr(0, pos), key_value.substr(pos + 1));
        m_keys.push_back(keyvalue);
    }
    else
    {
        Util::log_error(
            "Invalid key/value pair will be ignored (should be key=value): " + key_value);
        return UNSPECIFIED_FAILURE;
    }
    return SUCCESS;
}

Install::Install(const std::string & archive, const std::string & mdl_path)
    : m_archive(Archive::with_extension(archive))
    , m_mdl_directory(mdl_path)
    , m_force(false)
    , m_dryrun(false)
{
    Util::File::convert_symbolic_directory(m_mdl_directory);
}

Compatibility::COMPATIBILITY Install::test_compatibility(const std::string & directory) const
{
    Archive new_archive(m_archive);

    std::string old_archive = Util::path_appends(directory, new_archive.base_name());
    if (Util::file_is_readable(old_archive))
    {
        // Archive already installed in the given directory
        // Invoke compatibility command
        const std::string old_archive_name(old_archive);
        const std::string new_archive_name(new_archive.full_name());
        Compatibility test_compatibilty(old_archive_name, new_archive_name);
        // During installation, we only rely on the archive versions
        test_compatibilty.set_test_version_only(true);
        check_success(test_compatibilty.execute() >= 0);
        Compatibility::COMPATIBILITY compatibility = test_compatibilty.compatibility();

        return compatibility;
    }

    return Compatibility::UNDEFINED;
}

class Compatibility_result
{
public:
    Compatibility_result(const string & archive_file)
        : m_archive(archive_file)
    {
    }
    Archive m_archive;
    Compatibility::COMPATIBILITY m_compatibility;
    typedef enum { higher, lower, same } LEVEL;
    LEVEL m_level;
    string level() const
    {
        return
            (m_level == higher)
            ? ("higher level")
            : ((m_level == lower) ? "lower level" : "same level");
    }
    void report() const
    {
        if (m_compatibility == Compatibility::NOT_COMPATIBLE)
        {
            Util::log_info(
                "Found incompatible archive at " + level() + " : " + Util::normalize(m_archive.full_name()));
        }
        else if (m_compatibility == Compatibility::COMPATIBLE)
        {
            Util::log_info(
                "Found compatible archive at " + level() + " : " + Util::normalize(m_archive.full_name()));
        }
        else if (m_compatibility == Compatibility::COMPATIBLE_SAME_VERSION)
        {
            Util::log_info(
                "Found same version of the archive at " + level() + " : " + Util::normalize(m_archive.full_name()));
        }
        else
        {
            Util::log_info(
                "Did not find archive at " + level() + " : " + Util::normalize(m_archive.full_name()));
        }
    }
};

class Compatibility_result_vector : public vector<Compatibility_result>
{
    bool m_can_install = true;
public:
    void analyze_results()
    {
        // TODO: Replace with a map of:
        // typedef std::pair<Compatibility_result::LEVEL, Compatibility::COMPATIBILITY> KEY;
        // std::map<KEY, bool> comp;
        // comp[KEY(HIGHER, COMPATIBLE)] = false;
        // ...

        // Assume we can install the archive, e.g. the list can be empty
        m_can_install = true;
        const_iterator it;
        for (it = begin(); it != end() && m_can_install; it++)
        {
            it->report();

            if (it->m_level == Compatibility_result::higher)
            {
                // Higher priority path
                // --------------------
                // - Compatible versions: 
                //      "Earlier version of the archive is installed at a higher priority path
                //      No install / error"	
                //  - Identical versions
                //      "This version of the archive is installed at a higher priority path
                //      No install / warning"
                // - Incompatible versions
                //      "More recent version of the archive is installed at a higher priority path
                //      No install / warning"	
                // - No archive installed
                //      Install
                //
                switch (it->m_compatibility)
                {
                case Compatibility::COMPATIBLE:
                    m_can_install = false;
                    Util::log_error(
                        "Earlier version of the archive is installed at a higher priority path:");
                    Util::log_error(Util::normalize(it->m_archive.full_name()));
                    break;
                case Compatibility::COMPATIBLE_SAME_VERSION:
                    m_can_install = false;
                    Util::log_warning(
                        "This version of the archive is installed at a higher priority path:");
                    Util::log_warning(Util::normalize(it->m_archive.full_name()));
                    break;
                case Compatibility::NOT_COMPATIBLE:
                    m_can_install = false;
                    Util::log_warning(
                        string("More recent version of the archive is installed at a ") +
                        "higher priority path:");
                    Util::log_warning(Util::normalize(it->m_archive.full_name()));
                    break;
                case Compatibility::UNDEFINED:
                    m_can_install &= true;
                    break;
                default:
                    break;
                }
            }
            else if (it->m_level == Compatibility_result::same)
            {
                // Destination path
                // --------------------
                // - Compatible versions: 
                //      "Earlier version of the archive is installed in the destination directory
                //      Install / fine"	
                //  - Identical versions
                //      "This version of the archive is installed in the destination directory
                //      No install / warning"
                // - Incompatible versions
                //      "More recent version of the archive is installed in the destination
                //       directory
                //      No install / warning"	
                // - No archive installed
                //      Install
                switch (it->m_compatibility)
                {
                case Compatibility::COMPATIBLE:
                    m_can_install &= true;
                    Util::log_info(
                        "Earlier version of the archive is installed "
                        "in the destination directory:");
                    Util::log_info(Util::normalize(it->m_archive.full_name()));
                    break;
                case Compatibility::COMPATIBLE_SAME_VERSION:
                    m_can_install = false;
                    Util::log_warning(
                        "This version of the archive is installed in the destination directory:");
                    Util::log_warning(Util::normalize(it->m_archive.full_name()));
                    break;
                case Compatibility::NOT_COMPATIBLE:
                    m_can_install = false;
                    Util::log_warning(
                        string("Incompatible version of the archive is installed ") +
                        "in the destination directory:");
                    Util::log_warning(Util::normalize(it->m_archive.full_name()));
                    break;
                case Compatibility::UNDEFINED:
                    m_can_install &= true;
                    break;
                default:
                    break;
                }
            }
            else if (it->m_level == Compatibility_result::lower)
            {
                // Lower priority path
                // --------------------
                // - Compatible versions: 
                //      "Earlier version of the archive is installed at lower priority path.
                //      This will be shadowed by the new archive
                //      No install / warning"
                //  - Identical versions
                //      "This version of the archive is installed at a lower priority path
                //      No install / warning"
                // - Incompatible versions
                //      "More recent version of the archive is installed at a lower priority path
                //      No install / warning"
                // - No archive installed
                //      Install
                switch (it->m_compatibility)
                {
                case Compatibility::COMPATIBLE:
                    m_can_install &= false;
                    Util::log_warning(
                        "Earlier version of the archive is installed at lower priority path:");
                    Util::log_warning(Util::normalize(it->m_archive.full_name()));
                    Util::log_warning("This will be shadowed by the new archive.");
                    break;
                case Compatibility::COMPATIBLE_SAME_VERSION:
                    m_can_install &= false;
                    Util::log_warning(
                        "This version of the archive is installed at a lower priority path:");
                    Util::log_warning(Util::normalize(it->m_archive.full_name()));
                    break;
                case Compatibility::NOT_COMPATIBLE:
                    m_can_install = false;
                    Util::log_warning(
                        string("More recent version of the archive is installed ") +
                        "at a lower priority path:");
                    Util::log_warning(Util::normalize(it->m_archive.full_name()));
                    break;
                case Compatibility::UNDEFINED:
                    m_can_install &= true;
                    break;
                default:
                    break;
                }
            }
        }
    }
    bool can_install()
    {
        return m_can_install;
    }
};

int Install::execute()
{
    Util::log_info("Install Archive");
    Util::log_info("\tArchive: " + Util::normalize(m_archive));
    Util::log_info("\tMDL Path: " + Util::normalize(m_mdl_directory));

    // Input archive
    Archive new_archive(m_archive);
    {
        string src(m_archive);
        string dest(Util::path_appends(m_mdl_directory, new_archive.base_name()));
        // Ensure src and target are not the same
        if (Util::equivalent(src, dest))
        {
            Util::log_error("Archive not installed, same source and target");
            Util::log_error("\tArchive source: " + Util::normalize(src));
            Util::log_error("\tArchive target: " + Util::normalize(dest));
            return SUCCESS;
        }
    }
    {
        // Test input archive
        if (!new_archive.is_valid())
        {
            Util::log_error("Invalid archive: " + m_archive);
            return INVALID_ARCHIVE;
        }
        Util::log_info("Begin instalation of archive: " + Util::normalize(m_archive));
        Util::log_info("Destination directory: " + Util::normalize(m_mdl_directory));
    }
    {
        // Test destination directory
        if (!Util::directory_is_writable(m_mdl_directory))
        {
            Util::log_error("Invalid installation directory: " + Util::normalize(m_mdl_directory));
            return INVALID_MDL_PATH;
        }
    }
    bool do_install = true;

    // Check possible conflicts with installed packages, modules, archives
    // In all MDL search paths, including target folder
    if (do_install)
    {
        // Build list of MDL folders
        Search_path sp(mdlm::neuray());
        sp.snapshot();
        if (!sp.find_module_path(m_mdl_directory))
        {
            sp.add_module_path(m_mdl_directory);
            sp.snapshot();
        }
        const std::vector<std::string> & paths(sp.paths());
        for(auto & p : paths)
        { 
            if (new_archive.conflict(p))
            {
                Util::log_warning("Archive conflict detected");
                Util::log_warning("\tArchive: " + Util::normalize(m_archive));
                Util::log_warning("\tInstall path: " + Util::normalize(m_mdl_directory));
                do_install = false;

                mdlm::report("Conflict found in directory: " + p);
                break;
            }
            else
            {
                Util::log_info("Archive conflict test passed against directory: " + p);
            }
        }
        sp.restore_snapshot();
    }

    // Check conflicts with same installed archive in any MDL search root
    if (do_install)
    {
        // Add the target MDL path to the MDL roots if necessary
        Search_path sp(mdlm::neuray());
        sp.snapshot();
        if (!sp.find_module_path(m_mdl_directory))
        {
            sp.add_module_path(m_mdl_directory);
            sp.snapshot();
        }

        // Test whether an archive with same name is already installed:
        //      - in the target search root?
        //      - at a higher prioritized search root?
        //      - at a lower prioritized search root?
        std::string install_path(m_mdl_directory);
        vector<string>::const_iterator it;
        Compatibility_result::LEVEL level = Compatibility_result::higher;
        Compatibility_result_vector compatibility_list;
        for (it = sp.paths().begin(); it != sp.paths().end(); it++)
        {
            std::string current(*it);

            // Test compatibility
            const Compatibility::COMPATIBILITY return_code =
                test_compatibility(current);

            std::string archive_dest = Util::path_appends(current, new_archive.base_name());
            Compatibility_result comp(archive_dest);
            comp.m_compatibility = return_code;
            if (Util::equivalent(current, install_path))
            {
                comp.m_level = Compatibility_result::same;
                level = Compatibility_result::lower;// switch to lower level
            }
            else
            {
                comp.m_level = level;
            }

            if (return_code != Compatibility::UNDEFINED)
            {
                compatibility_list.push_back(comp);
            }
        }

        // Analyze the results
        compatibility_list.analyze_results();

        bool can_install = compatibility_list.can_install();

        if (!can_install)
        {
            mdlm::report("Conflict archive found in MDL search path");
        }

        do_install &= can_install;
    }

    // Check dependencies
    if (do_install)
    {
        do_install &= new_archive.all_dependencies_are_installed();
    }

    if (!do_install)
    {
        if (m_force)
        {
            mdlm::report(
                "Installation proceed due to -f|--force");
            do_install = true;
        }
        else
        {
            mdlm::report(
                "Installation canceled (use -f|--force to force)");
        }
    }
    if (do_install)
    {
        std::string archive_src = new_archive.full_name();
        std::string archive_dest = Util::path_appends(m_mdl_directory, new_archive.base_name());
        Util::log_info("Archive ready to be installed");
        Util::log_info("\tArchive: " + Util::normalize(m_archive));
        Util::log_info("\tMDL Path: " + Util::normalize(m_mdl_directory));
        if (m_dryrun)
        {
            mdlm::report("Archive not installed due to --dry-run option");
        }
        else
        {
            if (Util::copy_file(archive_src, archive_dest))
            {
                mdlm::report("Archive " + new_archive.base_name()
                    + " successfully installed in directory: " + Util::normalize(m_mdl_directory));
            }
            else
            {
                mdlm::report("Archive not installed due to unknown error");
            }
        }
    }

    return SUCCESS;
}

Show_archive::Show_archive(const std::string & archive)
    : m_archive(Archive::with_extension(archive))
{}

int Show_archive::execute()
{
    Archive archive(m_archive);
    if (!archive.is_valid())
    {
        Util::log_error("Invalid archive");
        return INVALID_ARCHIVE;
    }
    Util::log_verbose("Show archive: " + Util::normalize(archive.full_name()));
    Util::log_verbose("Archives set: " + Util::normalize(archive.full_name()));

    mi::base::Handle<mi::neuraylib::IMdl_archive_api>
        archive_api(mdlm::neuray()->get_api_component<mi::neuraylib::IMdl_archive_api>());

    mi::base::Handle<const mi::neuraylib::IManifest> manifest(
        archive_api->get_manifest(m_archive.c_str()));

    check_success(manifest.is_valid_interface());

    for (mi::Size i = 0; i < manifest->get_number_of_fields(); i++)
    {
        const char* key = manifest->get_key(i);
        if (list_field(key))
        {
            const char* value = manifest->get_value(i);
            if (m_report)
            {
                mdlm::report(key + string(" = \"") + value + string("\""));
            }
            m_manifest.insert(std::pair<string, string>(key, value));
        }
    }

    return SUCCESS;
}

const std::string List_dependencies::dependency_keyword = "dependency";

List_dependencies::List_dependencies(const std::string & archive)
    : Show_archive(Archive::with_extension(archive))
{
    add_filter_field(dependency_keyword);
}

void List_dependencies::get_dependencies(std::multimap<Archive::NAME, Version> & dependencies) const
{
    std::pair<std::multimap<std::string, std::string>::const_iterator, std::multimap<std::string, std::string>::const_iterator> ret;
    ret = m_manifest.equal_range(dependency_keyword);
    for (std::multimap<std::string, std::string>::const_iterator it = ret.first; it != ret.second; ++it)
    {
        string archiveAndVersion(it->second);
        vector<string> tmp(Util::split(archiveAndVersion, ' '));
        if (tmp.size() == 2)
        {
            std::string archiveName(tmp[0]);
            std::string archiveVersion(tmp[1]);
            dependencies.insert(std::pair<Archive::NAME, Version>(archiveName, Version(archiveVersion)));
        }
    }
}

Extract::Extract(const std::string & archive, const std::string & path)
    : m_archive(Archive::with_extension(archive))
    , m_directory(path)
    , m_force(false)
{}

int Extract::execute()
{
    Archive archive(m_archive);
    if (!archive.is_valid())
    {
        Util::log_error("Invalid archive");
        return INVALID_ARCHIVE;
    }

    bool proceed(true);
    Util::File folder(m_directory);
    if (folder.is_directory())
    {
        // Folder exists
        if (! folder.is_empty())
        {
            Util::log_warning("Directory is not empty");
            proceed = false;
        }
    }
    else
    {
        // create folder
        if (!Util::create_directory(m_directory))
        {
            Util::log_error("Can not create directory: " + m_directory);
            return CAN_NOT_CREATE_PATH;
        }
    }
    if (m_force)
    {
        proceed = true;
    }
    if (proceed)
    {
        if (archive.extract_to_directory(m_directory) == 0)
        {
            Util::log_report("Archive extracted");
        }
    }
    return SUCCESS;
}

Create_mdle::Create_mdle(const std::string & prototype, const std::string & mdle)
    : m_prototype(prototype)
    , m_mdle(mdle)
{
}

int Create_mdle::add_user_file(const std::string& source_path, const std::string& target_path)
{
    m_user_files.push_back({source_path, target_path});
    return SUCCESS;
}

int Create_mdle::execute()
{
    bool is_mdle = m_prototype.rfind(".mdle") == (m_prototype.size() - 5);
    std::string module_name, function_name;
    if (is_mdle)
    {
        if (!Util::file_is_readable(m_prototype))
        {
            Util::log_error("Source MDLE file '" + m_prototype + "' does not exist.");
            return -1;
        }

        // make module_name the absolute MDLE file path and use forward slashes
        module_name = m_prototype;

        // material name is always main for MDLE
        function_name = "main";
    }
    else
    {
        // parse module and material name
        size_t p = m_prototype.find('(');
        std::string tmp = m_prototype.substr(0, p);
        p = tmp.rfind("::");
        if (p == 0 || p == std::string::npos || m_prototype[0] != ':' || m_prototype[1] != ':')
        {
            Util::log_error("Invalid material name '" + m_prototype + "' "
                            "A full qualified name with leading '::' is expected.");
            return -1;
        }

        module_name = m_prototype.substr(0, p);
        function_name = m_prototype.substr(p + 2);
    }

    // check if target file name was set
    if (m_mdle.empty())
    {
        Util::log_error("Target MDLE file not specified.");
        return -1;
    }

    // some user (debug) feedback
    Util::log_report("Creating MDLE: " + m_mdle + "\n"
                     "         from: " + m_prototype);

    if (m_user_files.size() > 0)
    {
        for(size_t i = 0; i < m_user_files.size(); ++i)
            Util::log_report("    user file: " + m_user_files[i].first + " -> " + m_user_files[i].second);
    }

    mi::base::Handle<mi::neuraylib::IMdle_api> mdle_api(
        mdlm::neuray()->get_api_component<mi::neuraylib::IMdle_api>());


    // Load the FreeImage plugin.
    if(!mdlm::freeimage_available())
    {
        Util::log_error("failed to load nv_freeimage plugin.");
        return -1;
    }

    mi::base::Handle<mi::neuraylib::IDatabase> database(
        mdlm::neuray()->get_api_component<mi::neuraylib::IDatabase>());

    mi::base::Handle<mi::neuraylib::IScope> scope(database->get_global_scope());
    mi::base::Handle<mi::neuraylib::ITransaction> transaction(scope->create_transaction());

    int return_value = UNSPECIFIED_FAILURE;
    for(;;)
    {
        mi::base::Handle<mi::neuraylib::IMdl_factory> mdl_factory(
            mdlm::neuray()->get_api_component<mi::neuraylib::IMdl_factory>());

        mi::base::Handle < mi::neuraylib::IMdl_execution_context> context(
            mdl_factory->create_execution_context());

        // force experimental to true for now
        context->set_option("experimental", true);

        // load module
        Handle<IMdl_impexp_api> mdl_impexp_api(
            mdlm::neuray()->get_api_component<IMdl_impexp_api>());
        mdl_impexp_api->load_module(transaction.get(), module_name.c_str(), context.get());
        if (context->get_error_messages_count() > 0)
            break;

        // get the resulting db name
        mi::base::Handle<const mi::IString> module_db_name(
            mdl_factory->get_db_module_name(module_name.c_str()));

        // parameter is no a valid qualified module name
        if (!module_db_name)
        {
            Util::log_warning("module name to load '%s' is invalid" + module_name);
            return -1;
        }

        std::string db_name(module_db_name->get_c_str());

        // since "main" could be a function the signature is required
        // therefore, get the module and iterate over the functions that are called main (only one)
        mi::base::Handle<const mi::neuraylib::IModule> mdl_module(
            transaction->access<mi::neuraylib::IModule>(db_name.c_str()));

        // db_name is now the name of the function/material
        db_name += "::" + function_name;
        mi::base::Handle<const mi::IArray> func_list(mdl_module->get_function_overloads(db_name.c_str()));

        bool unknown_signature = false;
        std::string potential_msg = "";
        switch (func_list->get_length())
        {
            case 0: // material names are unique
            {
                break;
            }

            case 1: // if there is only one function with the selected name it is okay too
            {
                mi::base::Handle<const mi::IString> func_sign(func_list->get_element<mi::IString>(0));
                db_name = func_sign->get_c_str();
                break;
            }

            default: // if there are overloads, check if the selected one exists
            {
                unknown_signature = true;
                potential_msg = 
                    "multiple functions with name: '" + function_name + "' found in module: '" + module_name + "'.\n"
                    "Valid function names are:\n";

                for (mi::Size i = 0, n = func_list->get_length(); i < n; ++i)
                {
                    mi::base::Handle<const mi::IString> func_sign(func_list->get_element<mi::IString>(i));
                    potential_msg += "-   " + std::string(func_sign->get_c_str() + 3 /* skip 'mdl'*/) + "\n";

                    if (strcmp(db_name.c_str(), func_sign->get_c_str()) == 0)
                    {
                        unknown_signature = false;
                        break;
                    }
                }
                break;
            }
        }
        if (unknown_signature)
        {
            Util::log_error(potential_msg);
            break;
        }

        // check if the material/function is available
        mi::base::Handle<const mi::neuraylib::IMaterial_definition> material_definition(
            transaction->access<mi::neuraylib::IMaterial_definition>(db_name.c_str()));
        mi::base::Handle<const mi::neuraylib::IFunction_definition> function_definition(
            transaction->access<mi::neuraylib::IFunction_definition>(db_name.c_str()));
        if (!material_definition && !function_definition)
        {
            Util::log_error("failed to find the selected material/function: '" + function_name + "' "
                            "in module: '" + module_name + "'.");
            break;
        }

        // setup the export to mdle
        mi::base::Handle<mi::IStructure> data(transaction->create<mi::IStructure>("Mdle_data"));

        mi::base::Handle<mi::IString> prototype(data->get_value<mi::IString>("prototype_name"));
        prototype->set_c_str(db_name.c_str());

        // keep the thumbnail (since defaults can't be change here)
        std::string thumbnail_path("");
        mi::Size parameter_count = 0;
        mi::Size defaults_count = 0;

        if(material_definition) 
        {
            const char* p = material_definition->get_thumbnail();
            parameter_count = material_definition->get_parameter_count();
            mi::base::Handle<const mi::neuraylib::IExpression_list> defaults(
                material_definition->get_defaults());
            defaults_count = defaults->get_size();
            thumbnail_path = p ? p : "";
        } 
        else if (function_definition) 
        {
            const char* p = function_definition->get_thumbnail();
            parameter_count = function_definition->get_parameter_count();
            mi::base::Handle<const mi::neuraylib::IExpression_list> defaults(
                function_definition->get_defaults());
            defaults_count = defaults->get_size();
            thumbnail_path = p ? p : "";
        }

        // check if all parameters have defaults
        if (parameter_count != defaults_count)
        {
            Util::log_error("failed to create MDLE for '" + module_name + "::" + function_name +
                            "' because at least one parameter is unspecified. "
                            "For MDLEs all function/material parameters have to be defined. "
                            "MDLM does not support the export of such functions/materials.");
            break;
        }

        if (!thumbnail_path.empty()) {
            mi::base::Handle<mi::IString> thumbnail(data->get_value<mi::IString>("thumbnail_path"));
            thumbnail->set_c_str(thumbnail_path.c_str());
        }

        // add user files
        if (!m_user_files.empty())
        {
            size_t n = m_user_files.size();
            std::string array_type = MI::STRING::formatted_string("Mdle_user_file[%d]", n);
            mi::base::Handle<mi::IArray> user_file_array(transaction->create<mi::IArray>(array_type.c_str()));
            for (mi::Size i = 0; i < n; ++i)
            {
                mi::base::Handle<mi::IStructure> user_file(transaction->create<mi::IStructure>("Mdle_user_file"));
                mi::base::Handle<mi::IString> source_path(user_file->get_value<mi::IString>("source_path"));
                source_path->set_c_str(m_user_files[i].first.c_str());
                mi::base::Handle<mi::IString> target_path(user_file->get_value<mi::IString>("target_path"));
                target_path->set_c_str(m_user_files[i].second.c_str());
                user_file_array->set_element(i, user_file.get());
            }
            data->set_value("user_files", user_file_array.get());
        }

        // create the mdle
        mdle_api->export_mdle(transaction.get(), m_mdle.c_str(), data.get(), context.get());
        if (context->get_error_messages_count() > 0)
            break;

        // check if the created mdle is valid
        mdle_api->validate_mdle(m_mdle.c_str(), context.get());
        if (context->get_error_messages_count() > 0)
            break;

        return_value = SUCCESS;
        break;
    }
    transaction->commit();
    return return_value;
}

Check_mdle::Check_mdle(const std::string& mdle)
    : m_mdle(mdle)
{
}

int Check_mdle::execute()
{
    // check if target file name was set
    if (m_mdle.empty())
    {
        Util::log_error("Target MDLE file not specified.");
        return -1;
    }

    mi::base::Handle<mi::neuraylib::IMdl_factory> mdl_factory(
        mdlm::neuray()->get_api_component<mi::neuraylib::IMdl_factory>());

    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
        mdl_factory->create_execution_context());

    mi::base::Handle<mi::neuraylib::IMdle_api> mdle_api(
        mdlm::neuray()->get_api_component<mi::neuraylib::IMdle_api>());

    return mdle_api->validate_mdle(m_mdle.c_str(), context.get());
}

int Help::execute()
{
    MDLM_option_parser parser;
    parser.output_usage(cout);

    Option_set options = parser.get_known_options().find_options_from_name(m_command);
    if (options.empty())
    {
        Util::log_error("Invalid command: " + m_command);
        return -1;
    }

    Option_set::iterator it;
    for (it = options.begin(); it != options.end(); it++)
    {
        it->output_usage(cout);
    }
    return 0;
}

List_cmd::List_cmd(const std::string & archive)
    : m_archive_name(Archive::with_extension(archive))
{}

int List_cmd::execute()
{
    // Initialize
    m_result = List_result();
    Search_path sp(mdlm::neuray());
    sp.snapshot();

    // Look for all archives
    if (m_archive_name.empty())
    {
        // List all archives installed
        for (auto& directory : sp.paths())
        {
            MI::DISK::Directory d;
            if (d.open(directory.c_str()))
            {
                std::string testFile;
                while (!(testFile = d.read()).empty())
                {
                    Archive testArchive(Util::path_appends(directory, testFile));
                    if (testArchive.is_valid())
                    {
                        m_result.m_archives.push_back(testArchive);
                        Util::log_report("Found archive: " + Util::normalize(testArchive.full_name()));
                    }
                }
            }
        }
        return SUCCESS;
    }

    // Look for specific archive names
    bool found(false);
    for (auto& directory : sp.paths())
    {
        Archive testArchive(Util::path_appends(directory, m_archive_name));
        if (testArchive.is_valid())
        {
            if (found == true)
            {
                // Already found one installed archive report a warning
                Util::log_warning("Found shadowed archive (inconsistent installation): " + 
                    Util::normalize(testArchive.full_name()));
            }
            found = true;
            m_result.m_archives.push_back(testArchive);
            Util::log_report("Found archive: " + Util::normalize(testArchive.full_name()));
        }
    }
    return SUCCESS;
}

Remove_cmd::Remove_cmd(const std::string & archive)
    : m_archive_name(Archive::with_extension(archive))
{}

 int Remove_cmd::find_archive(Archive & archive)
{
    archive = Archive(m_archive_name);
    if (archive.is_valid())
    {
        return SUCCESS;
    }
    List_cmd list(m_archive_name);
    if (list.execute() != List_cmd::SUCCESS)
    {
        Util::log_error("Unable to list archive: " + m_archive_name);
        return UNSPECIFIED_FAILURE;
    }
    List_cmd::List_result archives(list.get_result());
    if (archives.m_archives.empty())
    {
        Util::log_error("Can not find archive: " + m_archive_name);
        return ARCHIVE_NOT_FOUND;
    }

    if (archives.m_archives.size() > 1)
    {
        Util::log_warning("Found multiple installations of archive: " + m_archive_name);
        Util::log_warning("Considering: " + archives.m_archives[0].full_name());
    }

    archive = archives.m_archives[0];
    return SUCCESS;
}

int Remove_cmd::Remove_cmd::execute()
{
    Archive toRemove("");
    int rtn;
    if ((rtn = find_archive(toRemove)) != SUCCESS)
    {
        return rtn;
    }
    
    // Look for dependencies on the archive being uninstalled
    Util::log_verbose("Looking for dependencies on archive: " + toRemove.full_name());
    List_cmd listAll("");
    listAll.execute();
    List_cmd::List_result allArchives(listAll.get_result());
    for (auto & a : allArchives.m_archives)
    {
        if (a.full_name() == toRemove.full_name())
        {
            // Do not test self
            continue;
        }
        List_dependencies dependsCmd(a.full_name());
        dependsCmd.set_report(false);
        if (dependsCmd.execute() != List_dependencies::SUCCESS)
        {
            Util::log_error("Unable to list archive dependencies: " + m_archive_name);
            return UNSPECIFIED_FAILURE;
        }

        std::multimap<Archive::NAME, Version> depends;
        dependsCmd.get_dependencies(depends);

        std::pair<std::multimap<Archive::NAME, Version>::const_iterator, std::multimap<Archive::NAME, Version>::const_iterator> ret;
        ret = depends.equal_range(toRemove.stem()/* remove the .mdr extension*/);

        for (std::multimap<Archive::NAME, Version>::const_iterator it = ret.first; it != ret.second; ++it)
        {
            if (toRemove.get_version() == it->second)
            {
                string archiveWithDependencies(it->first);
                Util::log_warning(a.stem() + " depends on archive " + toRemove.stem() + ". Can not remove.");
                return SUCCESS;
            }
        }
    }

    // All tests passed, we can remove the archive
    Util::log_report("Removing archive: " + Util::normalize(toRemove.full_name()));

    Util::delete_file_or_directory(Util::normalize(toRemove.full_name()));
         
    return SUCCESS;
}

Command * Command_factory::build_command(const Option_set & option)
{
    Option_set_type::const_iterator it;
    for (it = option.begin(); it != option.end(); it++)
    {
        if (it->is_valid() && it->get_is_command())
        {
            // Check the number of arguments for the command
            if (it->get_number_of_parameters() != it->value().size())
            {
                break;
            }

            if (it->id() == MDLM_option_parser::COMPATIBILITY)
            {
                return new Compatibility(it->value()[0], it->value()[1]);
            }
            else if (it->id() == MDLM_option_parser::HELP_CMD)
            {
                return new Help(it->value()[0]);
            }
            else if (it->id() == MDLM_option_parser::LIST)
            {
                return new List_cmd(it->value()[0]);
            }
            else if (it->id() == MDLM_option_parser::LIST_ALL)
            {
                return new List_cmd("");
            }
            else if (it->id() == MDLM_option_parser::INSTALL_ARCHIVE)
            {
                string archive = it->value()[0];
                string directory = it->value()[1];
                Install * cmd = new Install(archive, directory);
                Option_parser * command_options = it->get_options();
                if (command_options)
                {
                    Option_set dummy;
                    if (command_options->is_set(MDLM_option_parser::FORCE, dummy))
                    {
                        cmd->set_force_installation(true);
                    }
                    if (command_options->is_set(MDLM_option_parser::DRY_RUN, dummy))
                    {
                        cmd->set_dryrun(true);
                    }
                }
                return cmd;
            }
            else if (it->id() == MDLM_option_parser::CREATE_ARCHIVE)
            {
                Create_archive * create = new Create_archive(it->value()[0], it->value()[1]);
                Option_parser * command_options = it->get_options();
                if (command_options)
                {
                    Option_set setkey;
                    if (command_options->is_set(MDLM_option_parser::SET_KEY, setkey))
                    {
                        Option opt;
                        Option_set_type::iterator it;
                        for (it = setkey.begin(); it < setkey.end(); it++)
                        {
                            opt = (*it);
                            if (!it->value().empty())
                            {
                                create->add_key_value(it->value()[0]);
                            }
                        }
                    }
                }
                return create;
            }
            else if (it->id() == MDLM_option_parser::SHOW_ARCHIVE)
            {
                return new Show_archive(it->value()[0]);
            }
            else if (it->id() == MDLM_option_parser::DEPENDS)
            {
                return new List_dependencies(it->value()[0]);
            }
            else if (it->id() == MDLM_option_parser::EXTRACT)
            {
                string archive = it->value()[0];
                string directory = it->value()[1];
                Extract * cmd = new Extract(archive, directory);
                Option_parser * command_options = it->get_options();
                if (command_options)
                {
                    Option_set dummy;
                    if (command_options->is_set(MDLM_option_parser::FORCE, dummy))
                    {
                        cmd->set_force_extract(true);
                    }
                }
                return cmd;

            }
            else if (it->id() == MDLM_option_parser::REMOVE)
            {
                return new Remove_cmd(it->value()[0]);
            }
            else if (it->id() == MDLM_option_parser::CREATE_MDLE)
            {
                Create_mdle* create = new Create_mdle(it->value()[0], it->value()[1]);
                Option_parser* command_options = it->get_options();
                if (command_options)
                {
                    Option_set add_user_file;
                    if (command_options->is_set(MDLM_option_parser::CREATE_MDLE_ADD_USER_FILE, add_user_file))
                    {
                        Option opt;
                        Option_set_type::iterator it;
                        for (it = add_user_file.begin(); it < add_user_file.end(); it++)
                        {
                            opt = (*it);
                            if (it->value().size() == 2)
                            {
                                create->add_user_file(it->value()[0], it->value()[1]);
                            }
                        }
                    }
                }
                return create;
            }

            else if (it->id() == MDLM_option_parser::CHECK_MDLE)
            {
                return new Check_mdle(it->value()[0]);
            }
        }
    }

    return NULL;
}
