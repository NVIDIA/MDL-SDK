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
#include "archive.h"
#include "util.h"
#include "application.h"
#include "search_path.h"
#include "errors.h"
#include "version.h"
#include "command.h"
#include <boost/algorithm/string/replace.hpp>
#include <base/hal/disk/disk.h>
#include <base/hal/hal/i_hal_ospath.h>
#include <base/lib/path/i_path.h>
using namespace mdlm;
using std::string;
using std::vector;
using std::pair;

// In the GNU C Library, "minor" and "major" are defined
#ifdef minor
    #undef minor
#endif

#ifdef major
    #undef major
#endif

bool validate_archive(const std::string& archive, const std::string& path);

bool Archive::Test()
{
    {
        Archive a("");
        check_success(!a.is_valid_archive_name());
    }
    {
        Archive a("foo.bar.mdr");
        check_success(a.is_valid_archive_name());
    }
    return true;
}

const string Archive::extension = ".mdr";

string Archive::with_extension(const string & input)
{
    // If empty input
    string output(input);
    if (!output.empty()) // do not modify empty input
    {
        static const size_t len = Archive::extension.size();
        const size_t len_output = output.size();
        bool add_extension = len_output < len; // to small string, add 
        if (!add_extension)
        {
            // string longer than extension, test if extension is present
            add_extension = (output.substr(len_output - len, len) != Archive::extension);
        }
        if (add_extension)
        {
            output += Archive::extension;
            Util::log_verbose("Appending : " + Archive::extension + " to the name : "
                + input + " = " + output);
        }
    }
    return output;
}

Archive::Archive(const string & archive_file)
    : m_archive_file(archive_file)
{
}

Archive::~Archive()
{
}

bool Archive::is_valid() const
{
    // Archive name validity
    if (!is_valid_archive_name())
    {
        Util::log_debug("Invalid archive name");
        return false;
    }

    // Archive file exist and is a valid MDL archive
    if ( ! Util::has_ending(m_archive_file, Archive::extension) )
    {
        Util::log_debug(m_archive_file + ": wrong extension, should be '"
            + Archive::extension + "'");
        return false;
    }

    if (!Util::File(m_archive_file).is_file())
    {
        Util::log_debug(m_archive_file + ": is not a file");
        return false;
    }

    if ( ! Util::file_is_readable(m_archive_file) )
    {
        Util::log_debug(m_archive_file + ": is not readable");
        return false;
    }

    // Try to get version information from the manifest
    Version version;
    const mi::Sint32 rtn = get_version(version);

    return (rtn == 0);
}

mi::Sint32 Archive::get_version(Version & version) const
{
    string versionString;
    mi::Sint32 rtn = get_value("version", versionString);
    if (0 == rtn)
    {
        version = Version(versionString);
        return 0;
    }
    return -1;
}

Version Archive::get_version() const
{
    Version version;
    check_success(0 == Archive::get_version(version));
    return version;
}

mi::Sint32 Archive::get_values(const string & key, vector<string> & values, bool all_values) const
{
    mi::base::Handle<mi::neuraylib::IMdl_archive_api>
        archive_api(Application
            ::theApp().neuray()->get_api_component<mi::neuraylib::IMdl_archive_api>());
    mi::base::Handle<const mi::neuraylib::IManifest>
        manifest(archive_api->get_manifest(m_archive_file.c_str()));
    if (!manifest)
    {
        Util::log_error("Failed to retrieve manifest from archive: " + m_archive_file);
        return -1;
    }

    mi::Size i;
    for (i = 0; i < manifest->get_number_of_fields(); i++)
    {
        const char* manifestKey = manifest->get_key(i);
        if (manifestKey && key == manifestKey)
        {
            const char* manifestValue = manifest->get_value(i);
            if (manifestValue)
            {
                values.push_back(manifestValue);
                Util::log_verbose(
                    "Found MANIFEST key/value pair (" +
                    m_archive_file +
                    "): " +
                    key + " = \"" + manifestValue + "\""
                );
                if (!all_values)
                {
                    return 0;
                }
            }
        }
    }
    return 0;
}

string Archive::base_name() const
{
    return Util::basename(full_name());
}

string Archive::full_name() const
{
    return m_archive_file;
}

string Archive::stem() const
{
    return Util::stem(m_archive_file);
}

mi::Sint32 Archive::extract_to_directory(const string & directory) const
{
    mi::base::Handle<mi::neuraylib::IMdl_archive_api> 
        archive_api(
            Application::theApp().neuray()->get_api_component<mi::neuraylib::IMdl_archive_api>());
    mi::Sint32 result = archive_api->extract_archive(m_archive_file.c_str(), directory.c_str());
    if (result < 0)
    {
        Util::log_error("Archive : " 
            + full_name() 
            + " Extraction failed with error code: " 
            + to_string(result));
    }
    else
    {
        Util::log_info("Archive : " + full_name() + " extracted to directory: " + directory);
    }
    return result;
}

bool Archive::is_valid_archive_name() const
{
    if (stem().empty())
    {
        return false;
    }
    std::vector<std::string> vec = Util::split(stem(), '.');
    for (const std::string & elem : vec)
    {
        if (!Util::is_valid_archive_name(elem))
        {
            return false;
        }
    }
    return true;
}

bool Archive::all_dependencies_are_installed() const
{
    // List all dependencies for this archive
    vector<pair<string, Version>> depends = dependencies();

    // Check all dependencies are valid
    bool all_valid(true);
    for (auto& elem : depends)
    {
        string archive_name(elem.first);
        Version required_version(elem.second);

        List_cmd command(archive_name);
        const int rtn = command.execute();
        check_success(rtn == 0);
        List_cmd::List_result list = command.get_result();

        if (list.m_archives.empty())
        {
            Util::log_warning("Missing archive dependency: "
                + archive_name + " " + mdlm::to_string(required_version));
            all_valid = false;
        }
        else
        {
            // Test only the first archive of the list of found archives
            // for (auto& archive : list.m_archives)
            auto& archive = list.m_archives[0];
            {
                if (archive.get_version() < required_version
                    || archive.get_version().major() != required_version.major())
                {
                    Util::log_warning("Missing archive dependency: " 
                        + archive_name + " " + mdlm::to_string(required_version)
                        + " ("
                        + archive.stem() + " " + mdlm::to_string(archive.get_version())
                        + " is installed)"
                    );
                    all_valid = false;
                }
            }
        }
    }

    return all_valid;
}

vector<pair<string, Version>> Archive::dependencies() const
{
    // Get all keywords "dependency" from the manifest
    vector<string> depends;
    check_success2( get_values("dependency", depends) == 0, Errors::ERR_ARCHIVE_FAILURE );

    vector<pair<string, Version>> rtn;
    // For each dependency, list the installed archive
    for (auto& elem : depends)
    {
        // Use the list command to list installed archive details
        std::stringstream ss(elem);
        string archive;
        ss >> archive;
        string version;
        ss >> version;
        rtn.push_back(pair<string, Version>(archive, Version(version)));
    }
    return rtn;
}

bool Archive::conflict(const std::string & directory) const
{
    // Beware to input the mdl archive file without its ".mdr" extension
    return ! validate_archive(stem(), directory);
}

/////////////////////////////////////////////////////////////////
//// Copy validate_archive() from Mdl_discovery_api_impl
/////////////////////////////////////////////////////////////////
std::string dot_to_slash(std::string val)
{
#ifdef MI_PLATFORM_WINDOWS
    boost::replace_all(val, ".", "\\");
#else
    boost::replace_all(val, ".", "/");
#endif
    return val;
}

bool validate_archive(
    std::pair<const std::string, bool>& archive,
    std::map<std::string, bool>& archives,
    std::vector<std::string>& invalid_directories,
    const std::string& path)
{
    const std::string& a = archive.first;

    // Check file system
    std::string resolved_path = MI::HAL::Ospath::join(path, dot_to_slash(a));
    if (MI::DISK::is_directory(resolved_path.c_str()))
    {
        invalid_directories.push_back(resolved_path);
        archive.second = false;
    }
    else
    {
        std::string mdl = resolved_path.append(".mdl");
        if (MI::DISK::is_file(mdl.c_str()))
        {
            invalid_directories.push_back(resolved_path);
            archive.second = false;
        }
    }

    for (auto& other_archive : archives)
    {
        if (other_archive.second == false)
            continue;

        const std::string& o = other_archive.first;
        if (a == o)
            continue;

        auto l = a.size();
        auto ol = o.size();

        if (l < ol)
        {
            if (o.substr(0, l) == a)
            {
                other_archive.second = false;
                archive.second = false;
            }
        }
        else if (ol < l)
        {
            if (a.substr(0, ol) == o)
            {
                other_archive.second = false;
                archive.second = false;
            }
        }
    }
    return archive.second;
}

bool validate_archive(
    // archive.first set to archive name to test, BEWARE should be without .mdr extension
    // archive.second init to true set to false if the archive is not valid
    const std::string& archive,

    // Input search path root
    const std::string& path)
{
    // Find all archive files in the given directory
    std::map<std::string, bool> archives;
    MI::DISK::Directory dir;
    if (!dir.open(path.c_str()))
    {
        return false;
    }
    while (true)
    {
        std::string fn = dir.read(true/*nodot*/);
        if (fn.empty())
        {
            break;
        }
        Util::File file(Util::path_appends(path, fn));
        if (file.is_file())
        {
            if (Util::extension(fn) == Archive::extension)
            {
                // BEWARE archive should be without .mdr extension
                archives[Util::stem(fn)] = true;
            }
        }
    }

    std::pair<const std::string, bool> archivePair(archive, true);

    std::vector<std::string> invalid_directories;

    return validate_archive(archivePair, archives, invalid_directories, path);
}
