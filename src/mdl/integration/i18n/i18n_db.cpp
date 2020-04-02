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
#include "i18n_db.h"

#include <vector>
#include <clocale>
#include <map>
#include <set>
#include <memory>
#include <base/lib/tinyxml2/tinyxml2.h>
#include <base/system/main/access_module.h>
#include <base/lib/path/i_path.h>
#include <base/hal/hal/i_hal_ospath.h>
#include <base/hal/disk/disk.h>
#include <base/lib/log/i_log_assert.h>
#include <base/util/string_utils/i_string_utils.h>
#include <base/lib/log/i_log_logger.h>
#include <base/system/main/module_registration.h>
#include <mdl/integration/mdlnr/i_mdlnr.h>
#include <mi/mdl/mdl_mdl.h>
#include <mi/mdl/mdl_archiver.h>
#include <mi/mdl/mdl_streams.h>

using std::string;
using std::vector;
using std::map;
using std::set;
using tinyxml2::XMLElement;
using tinyxml2::XMLDocument;

typedef string Qualified_name;
typedef string Folder;

bool has_ending(const string & full_string, const string & ending)
{
    if (full_string.length() >= ending.length())
    {
        return (0 == full_string.compare(
            full_string.length() - ending.length(), ending.length(), ending));
    }
    else
    {
        return false;
    }
}

void replace_all(string & str, const string& from, const string& to)
{
    size_t start_pos = 0;
    while ((start_pos = str.find(from, start_pos)) != string::npos)
    {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length(); // Handles case where 'to' is a substring of 'from'
    }
}

bool has_beginning(const string & full_string, const string & beginning)
{
    if (full_string.length() >= beginning.length())
    {
        return (0 == full_string.compare(
            0, beginning.length(), beginning));
    }
    else
    {
        return false;
    }
}

class File_discovery
{
public:
    /// Search files with ending (e.g. "_fr.xlf") in the given folder 
    void discover(
        const string & suffix
        , const string & folder
        , vector<string> & filenames
        , bool recursive = false
        , string prefix = "" // File name must match some of this prefix
    ) const
    {
        string path = MI::HAL::Ospath::normpath_v2(folder);

        if (!MI::DISK::access(path.c_str(), false))
            return;

        // Collect all archives
        MI::DISK::Directory dir;
        if (!dir.open(path.c_str()))
            return;

        string entry = dir.read();
        while (!entry.empty())
        {
            string entry_path(MI::HAL::Ospath::join(path, entry));
            if (MI::DISK::is_file(entry_path.c_str()))
            {
                // Search for pattern in the filename
                if (has_ending(entry_path, suffix))
                {
                    // Text prefix
                    bool ok(true);
                    if (!prefix.empty())
                    {
                        string test(entry.substr(0, entry.length() - suffix.length()));
                        size_t idx = prefix.find(test);
                        if (idx != 0)
                        {
                            ok = false;
                        }
                    }
                    if (ok)
                    {
                        filenames.push_back(entry_path);
                    }
                }
            }
            else if (recursive && MI::DISK::is_directory(entry_path.c_str()))
            {
                discover(suffix, entry_path, filenames, recursive);
            }

            entry = dir.read();
        }
    }
    /// Search the list of folders
    void discover(
        const string & pattern
        , const vector<string> & folders
        , vector<string> & filenames
        , bool recursive = false
    ) const
    {
        for (const string & folder : folders)
        {
            discover(pattern, folder, filenames, recursive);
        }
    }
    void discover_archives(
        const string & folder // Folder to look into
        , const string & qualified_name
        , vector<string> & filenames
    ) const
    {
        string prefix(qualified_name);
        // Replace "::" with "."
        replace_all(prefix, "::", ".");
        // Remove leading "."
        if (prefix.find_first_of(".") == 0)
        {
            prefix = prefix.substr(1);
        }
        discover(".mdr", folder, filenames, false /*recursive*/, prefix);
    }
};

// Basic data structure types
class Mdl_search_path
{
    typedef bool Root_flag;

    /// MDL directory cache
    /// For each directory, the value is true if the directory is MDL directroy root
    map<Folder, Root_flag> m_directory_cache;

    /// Map directory name to its qualified name
    map<Folder, Qualified_name> m_qualified_name_cache;

public:
    static Mdl_search_path & get()
    {
        static Mdl_search_path singleton;
        return singleton;
    }

public:
    void cleanup()
    {
        m_directory_cache.clear();
        m_qualified_name_cache.clear();
    }
public:
    /// Return true is MDL path changed compared to what is cached
    bool has_changed()
    {
        // Disable this code.
        // It costs 40% more time to start MDL browser with translation
        // compared to without translation
        return false;

        //bool changed = false;
        //const vector<string> & current_paths = get_paths();
        //// First compare the number of elements
        //if (current_paths.size() != m_roots.size())
        //{
        //    changed = true;
        //}
        //else
        //{
        //    for (size_t i = 0; i < current_paths.size(); i++)
        //    {
        //        if (!fs::equivalent(current_paths[i], m_roots[i]))
        //        {
        //            changed = true;
        //            break;
        //        }
        //    }
        //}
        //if (changed)
        //{
        //    cleanup();
        //}
        //return changed;
    }
    const vector<string> & get_paths() const
    {
        MI::SYSTEM::Access_module<MI::PATH::Path_module> path_module(false);
        return path_module->get_search_path(MI::PATH::MDL);
    }
    bool is_root(const string & folder)
    {
        map<string, bool>::const_iterator it = m_directory_cache.find(folder);
        if (it != m_directory_cache.end())
        {
            return it->second;
        }
        // Get MDL path roots
        const vector<string> & roots = get_paths();

        string test = MI::HAL::Ospath::normpath_v2(folder);
        bool is_root(false);
        for (const string & f : roots)
        {
            if (f == test)
            {
                is_root = true;
                break;
            }
        }
        m_directory_cache[folder] = is_root;
        return is_root;
    }
    string folder_to_qualified_name(const string & folder)
    {
        if (m_qualified_name_cache.find(folder) != m_qualified_name_cache.end())
        {
            return m_qualified_name_cache[folder];
        }

        string qname("::");
        if (!is_root(folder))
        {
            string test_path = MI::HAL::Ospath::normpath_v2(folder);
            const vector<string> & folders = get_paths();
            for (const string & f : folders)
            {
                string sp_path = MI::HAL::Ospath::normpath_v2(f);

                string package(test_path);
                string MDL_root(sp_path);

                if (package.length() >= MDL_root.length())
                {
                    if (package.substr(0, MDL_root.length()) == MDL_root)
                    {
                        string sub(package.substr(MDL_root.length()));

                        vector<string> tokens;
                        MI::STRING::split(sub, MI::HAL::Ospath::sep(), tokens);

                        qname = "";
                        for (const string & t : tokens)
                        {
                            if (!t.empty())
                            {
                                qname += "::" + t;
                            }
                        }
                        break;
                    }
                }
            }
        }

        m_qualified_name_cache[folder] = qname;
        return(qname);
    }
    string qualified_name_to_folder(const Qualified_name & name)
    {
        string folder(name);
        replace_all(folder, "::", MI::HAL::Ospath::sep());
        return folder;
    }
};

class Dictionary : public map<string, string>
{
public:
    bool translate(const string & source, string & target) const;
};

class Context_dictionaries : public map<Qualified_name, Dictionary>
{
public:
    bool translate(
        const string & context
        , const string & source
        , string & target
    ) const;
};

namespace helper
{
class File
{
    string m_filename;
    string m_archive_filename;
    bool m_is_archive;
public:
    File()
    {}
    File(const string & filename)
        : m_filename(filename)
        , m_is_archive(false)
    {}
    File(const string & filename, const string & archive_filename)
        : m_filename(filename)
        , m_archive_filename(archive_filename)
        , m_is_archive(true)
    {}
    bool is_archive() const
    {
        return m_is_archive;
    }
    string get_filename() const
    {
        return m_filename;
    }
    string get_archive_filename() const
    {
        return m_archive_filename;
    }
    bool exist() const
    {
        if (is_archive())
        {
            MI::SYSTEM::Access_module<MI::MDLC::Mdlc_module> mdlc_module;
            mdlc_module.set();
            mi::base::Handle<mi::mdl::IMDL> mdl(mdlc_module->get_mdl());

            mi::base::Handle<mi::mdl::IArchive_tool> archive_tool(mdl->create_archive_tool());

            mi::base::Handle<mi::mdl::IInput_stream> file(archive_tool->get_file_content(
                m_archive_filename.c_str(), m_filename.c_str()));

            mdlc_module.reset();

            return file.is_valid_interface();
        }
        else
        {
            return MI::DISK::is_file(m_filename.c_str());
        }
        return false;
    }
};

class Qualified_name_object : public string
{
public:
    Qualified_name_object()
        : string("INVALID")
    {}
    Qualified_name_object(const string & qualified_name)
        : string(qualified_name)
    {}
    const Qualified_name & get_name() const
    {
        return *this;
    }
    bool is_valid() const
    {
        return *this != "INVALID";
    }
    string get_parent_name()
    {
        size_t index = rfind("::");
        if (string::npos != index)
        {
            return substr(0, index);
        }
        return "";
    }
    bool is_root() const
    {
        return empty();
    }
};

class Package : public Qualified_name_object
{
public:
    Package(const string & qualified_name)
        : Qualified_name_object(qualified_name)
    {}
    Package get_parent()
    {
        return Package(get_parent_name());
    }
};

class Module : public Qualified_name_object
{
public:
    Module(const string & qualified_name)
        : Qualified_name_object(qualified_name)
    {}
    Package get_package()
    {
        return Package(get_parent_name());
    }
};

class Context : public Qualified_name_object
{
public:
    Context(const string & qualified_name)
        : Qualified_name_object(qualified_name)
    {}
};

} //namespace helper

class File_iterator
{
protected:
    Qualified_name m_qualified_name;
    string m_locale;
    const vector<string> & m_paths;
    size_t m_index;

    // Archives
    class Archives : public vector<string>
    {
        size_t m_index;
    public:
        Archives()
            : m_index(0)
        {}
        bool get_next_archive(string & archive)
        {
            if (m_index < size())
            {
                archive = this->at(m_index);
                m_index++;
                return true;
            }
            return false;
        }
    };
    map<string, Archives> m_archives;

protected:
    Archives & discover_archives(const string & path)
    {
        map<string, Archives>::iterator it = m_archives.find(path);
        if (it == m_archives.end())
        {
            Archives candidates;
            File_discovery helper;
            helper.discover_archives(path, m_qualified_name, candidates);

            m_archives[path] = candidates;
        }
        return m_archives[path];
    }
public:
    File_iterator(const Qualified_name & qualified_name, const string & locale)
        : m_qualified_name(qualified_name)
        , m_locale(locale)
        , m_paths(Mdl_search_path::get().get_paths())
        , m_index(0)
    {}
    virtual bool get_next_file(helper::File & file) = 0;
};

class File_package_iterator : public File_iterator
{
    string m_folder_name;

public:
    File_package_iterator(const helper::Package & package, const string & locale)
        : File_iterator(package, locale)
    {
        m_folder_name = Mdl_search_path::get().qualified_name_to_folder(m_qualified_name);
    }
    bool get_next_file(helper::File & file)
    {
        bool rtn(false);
        if (m_index < m_paths.size())
        {
            const string & path(m_paths[m_index]);

            // discover archives
            Archives & archives = discover_archives(path);

            string archive_file;
            if (archives.get_next_archive(archive_file))
            {
                string filename = MI::HAL::Ospath::join(m_folder_name, m_locale + ".xlf");

                if (filename.find(MI::HAL::Ospath::sep()) == 0)
                {
                    // remove leading '/'
                    filename = filename.substr(MI::HAL::Ospath::sep().size());
                }

                file = helper::File(filename, archive_file);
                rtn = true;
            }
            else
            {
                // look for folders
                string folder(MI::HAL::Ospath::join(path, m_folder_name));
                file = MI::HAL::Ospath::join(folder, m_locale + ".xlf");
                rtn = true;
                m_index++;
            }
        }
        return rtn;
    }
};

class File_module_iterator : public File_iterator
{
    string m_module_prefix;
public:
    File_module_iterator(const helper::Module & module, const string & locale)
        : File_iterator(module, locale)
    {
        m_module_prefix = Mdl_search_path::get().qualified_name_to_folder(m_qualified_name);
    }
    bool get_next_file(helper::File & file)
    {
        bool rtn(false);
        if (m_index < m_paths.size())
        {
            const string & path(m_paths[m_index]);

            // discover archives
            Archives & archives = discover_archives(path);

            string archive_file;
            if (archives.get_next_archive(archive_file))
            {
                string filename(m_module_prefix + "_" + m_locale + ".xlf");

                if (filename.find(MI::HAL::Ospath::sep()) == 0)
                {
                    // remove leading '/'
                    filename = filename.substr(MI::HAL::Ospath::sep().size());
                }

                file = helper::File(filename, archive_file);
                rtn = true;
            }
            else
            {
                // look for folders
                string prefix(MI::HAL::Ospath::join(path, m_module_prefix));
                file = prefix + "_" + m_locale + ".xlf";
                rtn = true;
            }
        }
        m_index++;
        return rtn;
    }
};

class XLIFF_loader
{
protected:
    string m_filename;
public:
    virtual ~XLIFF_loader()
    {}
    bool load_file(const helper::File & file);

    virtual void add_trans_unit(const string & source, const string & target) = 0;
    virtual void add_trans_unit(
        const string & context, const string & source, const string & target) = 0;
private:
    bool load_file(const string & filename);
    bool load_file_from_archive(const string & filename, const string & archive_file);
    bool parse_document(const XMLDocument & document);
};

bool XLIFF_loader::load_file(const helper::File & file)
{
    if (file.is_archive())
    {
        return load_file_from_archive(file.get_filename(), file.get_archive_filename());
    }
    else
    {
        return load_file(file.get_filename());
    }
}

bool XLIFF_loader::load_file(const string & filename)
{
    m_filename = filename;

    // TODO: More checks for NULL ptrs and invalid data...
    XMLDocument document;

    if (document.LoadFile(filename.c_str()) != tinyxml2::XML_SUCCESS)
    {
        ::MI::LOG::mod_log->error(
            MI::M_I18N
            , MI::LOG::ILogger::C_PLUGIN
            , "Failed to load file: %s"
            , filename.c_str()
        );
        return false;
    }
    ::MI::LOG::mod_log->info(
        MI::M_I18N
        , MI::LOG::ILogger::C_PLUGIN
        , "Successfully loaded file: %s"
        , filename.c_str()
    );
    bool rtn = parse_document(document);
    return rtn;
}

bool XLIFF_loader::load_file_from_archive(const string & filename, const string & archive_file)
{
    m_filename = filename;

    MI::SYSTEM::Access_module<MI::MDLC::Mdlc_module> mdlc_module;
    mdlc_module.set();
    mi::base::Handle<mi::mdl::IMDL> mdl(mdlc_module->get_mdl());

    mi::base::Handle<mi::mdl::IArchive_tool> archive_tool(mdl->create_archive_tool());

    mi::base::Handle<mi::mdl::IInput_stream> file(archive_tool->get_file_content(
        archive_file.c_str(), filename.c_str()));

    mdlc_module.reset();

    bool rtn = false;
    if (file.is_valid_interface())
    {
        string buffer;
        int c;
        while ((c = file->read_char()) != -1)
        {
            buffer += c;
        }

        XMLDocument document;
        if (document.Parse(buffer.c_str()) != tinyxml2::XML_SUCCESS)
        {
            return false;
        }
        rtn = parse_document(document);
    }
    return rtn;
}

bool XLIFF_loader::parse_document(const XMLDocument & document)
{
    const XMLElement * root =
        document.RootElement()->FirstChildElement("file")->FirstChildElement("body");
    // process child elements
    for (const XMLElement * child = root->FirstChildElement();
         child != nullptr;
         child = child->NextSiblingElement())
    {
        const char * child_name = child->Value();

        if (child_name && string(child_name) == "trans-unit")
        {
            // Global XMLElement
            const XMLElement * source_elt = child->FirstChildElement("source");
            const char * source = source_elt->GetText();
            const XMLElement* target_elt = source_elt->NextSiblingElement();
            const char * target = target_elt->GetText();
            if (source && target)
            {
                add_trans_unit(source, target);
            }
        }
        else if (child_name && string(child_name) == "group")
        {
            const char * context = child->Attribute("resname");
            if (context)
            {
                for (const XMLElement * child2 = child->FirstChildElement();
                     child2 != nullptr;
                     child2 = child2->NextSiblingElement())
                {
                    const char * child_name = child2->Value();

                    if (child_name && string(child_name) == "trans-unit")
                    {
                        // Global dictionary
                        const XMLElement * source_elt = child2->FirstChildElement("source");
                        const char * source = source_elt->GetText();
                        const XMLElement * target_elt = source_elt->NextSiblingElement();
                        const char * target = target_elt->GetText();
                        if (source && target)
                        {
                            add_trans_unit(context, source, target);
                        }
                    }
                }
            }
        }
    }
    return true;
}

class MI::MDL::I18N::Database_impl
{
public:
    virtual ~Database_impl()
    {}
    virtual Sint32 translate(Mdl_translator_module::Translation_unit & sentence) = 0;
    virtual Sint32 cleanup() = 0;
};

class Flexible_database_impl : public MI::MDL::I18N::Database_impl
{
    class Translation_db : public XLIFF_loader
    {
    private:
        Dictionary m_global_dictionary;
        Context_dictionaries m_context_dictionaries;
        Qualified_name m_qualified_name;
    public:
        Translation_db()
        {}
        Translation_db(const Qualified_name & qualified_name)
            : m_qualified_name(qualified_name)
        {}
        // From XLIFF_loader
        void add_trans_unit(const string & source, const string & target) override
        {
            m_global_dictionary[source] = target;
        }
        // From XLIFF_loader
        void add_trans_unit(
            const string & context, const string & source, const string & target) override
        {
            // Prepend the qualified name to the relative context
            string context_key = m_qualified_name + "::" + context;

            // Since we prepend the context, no need to do the test
            //if(has_beginning(context, m_qualified_name))
            if(1)
            {
                m_context_dictionaries[context_key][source] = target;
            }
            else
            {
                if (!m_filename.empty())
                {
                    ::MI::LOG::mod_log->error(
                        MI::M_I18N
                        , MI::LOG::ILogger::C_PLUGIN
                        , "Error in file: %s"
                        , m_filename.c_str()
                    );
                }
                ::MI::LOG::mod_log->error(
                    MI::M_I18N
                    , MI::LOG::ILogger::C_PLUGIN
                    , "Ignoring translation of: %s. Context does not match package: %s"
                    , context.c_str()
                    , m_qualified_name.c_str()
                );
            }
        }

        bool translate(MI::MDL::I18N::Mdl_translator_module::Translation_unit & sentence) const
        {
            // Try with context
            string translation;
            bool translated = m_context_dictionaries.translate(
                sentence.get_context(), sentence.get_source(), translation);
            if (!translated)
            {
                // Try without context
                translated = m_global_dictionary.translate(sentence.get_source(), translation);
            }
            if (translated)
            {
                sentence.set_target(translation);
            }
            return translated;
        }
    };

    typedef map<helper::Module, Translation_db> Module_map;
    typedef map<helper::Package, Translation_db> Package_map;
    Module_map m_module_dictionaries;
    Package_map m_package_dictionaries;
    set<helper::Module> m_initialized_modules;
    set<helper::Package> m_initialized_packages;

private:
    void init_db(const helper::Module & module, const string & locale)
    {
        // if not init
        // Look for XLIFF files corresponding to the given module and locale
        if (m_initialized_modules.find(module) == m_initialized_modules.end())
        {
            File_module_iterator it(module, locale);

            helper::File file;
            while (it.get_next_file(file))
            {
                if (file.exist())
                {
                    m_module_dictionaries[module] = Translation_db(module);
                    m_module_dictionaries[module].load_file(file);
                    break;
                }
            }

            m_initialized_modules.insert(module);
        }
    }

    void init_db(const helper::Package & package, const string & locale)
    {
        // if not init
        // Look for XLIFF files corresponding to the given package and locale
        if (m_initialized_packages.find(package) == m_initialized_packages.end())
        {
            File_package_iterator it(package, locale);

            helper::File file;
            while (it.get_next_file(file))
            {
                if (file.exist())
                {
                    m_package_dictionaries[package] = Translation_db(package);
                    m_package_dictionaries[package].load_file(file);
                    break;
                }
            }
            m_initialized_packages.insert(package);
        }
    }

    bool do_translate(
        const helper::Module & module, MI::MDL::I18N::Mdl_translator_module::Translation_unit & sentence)
    {
        // Find DB for this module and attempt to translate
        string locale(sentence.get_locale());
        init_db(module, locale);

        Module_map::const_iterator it(m_module_dictionaries.find(module));
        if (it != m_module_dictionaries.end())
        {
            return it->second.translate(sentence);
        }
        return false;
    }

    bool do_translate(
        const helper::Package & package, MI::MDL::I18N::Mdl_translator_module::Translation_unit & sentence)
    {
        // Find DB for this package and attempt to translate
        string locale(sentence.get_locale());
        init_db(package, locale);

        Package_map::const_iterator it(m_package_dictionaries.find(package));
        if (it != m_package_dictionaries.end())
        {
            return it->second.translate(sentence);
        }
        return false;
    }

public:
    MI::Sint32 translate(MI::MDL::I18N::Mdl_translator_module::Translation_unit & sentence) override
    {
        if (!sentence.get_module_name())
        {
            // Try to find out module from context qualified name
            helper::Qualified_name_object qname(sentence.get_context());
            string m(qname.get_parent_name());
            if (!m.empty())
            {
                sentence.set_module_name(m.c_str());
            }
            else
            {
                return -3; //Invalid sentence
            }
        }
        bool translated(false);

        // Attempt to translate at the module level
        helper::Module module(sentence.get_module_name());
        translated = do_translate(module, sentence);

        if (!translated)
        {
            // Attempt to translate at the package level
            helper::Package package(module.get_package());
            translated = do_translate(package, sentence);

            while (!translated && !package.is_root())
            {
                // Attempt to translate at the parent package levels
                package = package.get_parent();
                translated = do_translate(package, sentence);
            }
        }
        return translated ? 0 : -1;
    }

    MI::Sint32 cleanup() override
    {
        m_module_dictionaries.clear();
        m_package_dictionaries.clear();
        m_initialized_modules.clear();
        m_initialized_packages.clear();
        return 0;
    }
};

bool Dictionary::translate(const string & source, string & target) const
{
    Dictionary::const_iterator it = find(source);
    if (it != end())
    {
        target = it->second;
        return true;
    }
    return false;
}

bool Context_dictionaries::translate(
    const string & context
    , const string & source
    , string & target
) const
{
    bool translated = false;
    // Find dictionary
    Context_dictionaries::const_iterator it = find(context);
    if (it != end())
    {
        // Use context dictionary to translate
        translated = it->second.translate(source, target);
    }
    return translated;
}

MI::MDL::I18N::Database::Database()
{
    m_translation_db = new Flexible_database_impl();
}

MI::MDL::I18N::Database::~Database()
{
    delete m_translation_db;
}

MI::Sint32 MI::MDL::I18N::Database::translate(MI::MDL::I18N::Mdl_translator_module::Translation_unit & sentence) const
{
    return m_translation_db->translate(sentence);
}

MI::Sint32 MI::MDL::I18N::Database::cleanup()
{
    Mdl_search_path::get().cleanup();
    return m_translation_db->cleanup();
}
