/***************************************************************************************************
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
 **************************************************************************************************/

/** \file
 ** \brief Source for the IMdl_discovery_api implementation.
 **/

#include "pch.h"

#include "neuray_mdl_discovery_api_impl.h"
#include "neuray_string_impl.h"

#include <mi/mdl_sdk.h>
#include <mi/mdl/mdl_mdl.h>
#include <mi/mdl/mdl_modules.h>

#include <base/hal/disk/disk.h>
#include <base/hal/hal/i_hal_ospath.h>
#include <base/lib/path/i_path.h>
#include <base/system/main/i_module_id.h>
#include <mdl/integration/mdlnr/i_mdlnr.h>
#include <io/scene/mdl_elements/i_mdl_elements_module.h>

#include <algorithm>
#include <string>
#include <boost/algorithm/string/replace.hpp>


// helper function
std::string add_slash(std::string val)
{
#ifdef _WIN32
    val.append("\\");
#else
    val.append("/");
#endif
    return val;
}

std::string dot_to_slash(std::string val)
{
#ifdef MI_PLATFORM_WINDOWS
    boost::replace_all(val, ".", "\\");
#else
    boost::replace_all(val, ".", "/");
#endif
    return val;
}

std::string colon_to_slash(std::string val)
{
#ifdef MI_PLATFORM_WINDOWS
    boost::replace_all(val, "::", "\\");
#else
    boost::replace_all(val, "::", "/");
#endif
    return val;
}

std::string slash_to_colon(std::string val)
{
#ifdef MI_PLATFORM_WINDOWS
    boost::replace_all(val, "\\", "::");
#else
    boost::replace_all(val, "/", "::");
#endif
    return val;
}

namespace MI {

namespace NEURAY {

namespace
{
    template<class T>
    // Comparison function for alphabetical sort
    bool compare_simple_names(const mi::base::Handle<const T> &hp1,
        const mi::base::Handle<const T> &hp2)
    {
        return 0 > strcmp(hp1->get_simple_name(), hp2->get_simple_name());
    }
}

Mdl_module_info_impl::Mdl_module_info_impl(const char* name,
    const char* q_name, const char* r_path, const char* s_path, mi::Size s_index)
    : m_resolved_path(r_path)
{
    m_qualified_name = std::string(q_name);
    m_search_index = s_index;
    m_search_path = std::string(s_path);
    m_simple_name = std::string(name);
    m_in_archive = false;
}

Mdl_module_info_impl::~Mdl_module_info_impl()
{
}

void Mdl_module_info_impl::add_shadow(Mdl_module_info_impl* shadow)
{
    m_shadows.push_back(make_handle_dup(shadow));
}

const mi::IString* Mdl_module_info_impl::get_resolved_path() const
{
    mi::IString* istring = new String_impl();
    istring->set_c_str(m_resolved_path.c_str());
    return istring;
}

mi::Size Mdl_module_info_impl::get_search_path_index() const
{
    return m_search_index;
}

const char* Mdl_module_info_impl::get_search_path() const
{
    return m_search_path.c_str();
}

mi::Size Mdl_module_info_impl::get_shadows_count() const
{
    return m_shadows.size();
}

const Mdl_module_info_impl* Mdl_module_info_impl::get_shadow(mi::Size index) const
{
    if ( m_shadows.size() <= index )
        return nullptr;

    const mi::base::Handle<const Mdl_module_info_impl>& shadow = m_shadows.at(index);
    shadow->retain();
    return shadow.get();
}

bool Mdl_module_info_impl::in_archive() const
{
    return m_in_archive;
}

void Mdl_module_info_impl::set_archive(bool val)
{
    m_in_archive = val;
}

// Sorts shadow list alphabetically
void Mdl_module_info_impl::sort_shadows()
{
    std::sort(m_shadows.begin(), m_shadows.end(), compare_simple_names<Mdl_module_info_impl>);
}

Mdl_package_info_impl::Mdl_package_info_impl(const char* name, 
    const char* s_path, const char* r_path, mi::Sint32 p_idx, const char* q_name)
{
    m_qualified_name = q_name;
    m_simple_name = name;
   
    // The root of the graph does not inherit search paths
    if (p_idx >= 0)
    {
        m_paths.push_back(s_path);
        m_path_idx.push_back(p_idx);
        m_resolved_paths.push_back(r_path);
        m_in_archive.push_back(false);
    }
}

Mdl_package_info_impl::~Mdl_package_info_impl()
{
}

void Mdl_package_info_impl::add_module(Mdl_module_info_impl* module)
{
    m_modules.push_back(make_handle_dup(module));
}

void Mdl_package_info_impl::add_package(Mdl_package_info_impl* child)
{
    m_packages.push_back(make_handle_dup(child));
}

// Checks if a package already exists in graph and returns 
// the index of the existing child
// -1 if it does not exist
mi::Sint32 Mdl_package_info_impl::check_package(Mdl_package_info_impl* child)
{
    for (mi::Uint32 c = 0; c < m_packages.size(); c++)
    {
        if (strcmp(m_packages[c]->get_qualified_name(), child->get_qualified_name()) == 0)
            return Sint32(c);
    }
    return -1;
}

// Sorts modules and child packages list alphabetically
void Mdl_package_info_impl::sort_children()
{
    std::sort(m_packages.begin(), m_packages.end(), compare_simple_names<Mdl_package_info_impl>);

    for (mi::Uint32 m = 0; m < m_packages.size(); m++)
        m_packages[m]->sort_children();

    for (mi::Uint32 m = 0; m < m_modules.size(); m++)
    {
        Mdl_module_info_impl* mod = m_modules[m].get();
        mod->sort_shadows();
    }
    std::sort(m_modules.begin(), m_modules.end(), compare_simple_names<Mdl_module_info_impl>);
}

// Checks if one ore more modules already exists in a certain package
// - 0 if the module has been shadowed (shadow module has been added)
// - 1 if no shadowing has happened
mi::Sint32 Mdl_package_info_impl::shadow_module(Mdl_module_info_impl* new_module)
{
    bool known = false;
    mi::Uint32 idx = 0;
    while (!known)
    {
        if (m_modules.size() <= idx)
            return -1;

        const Mdl_module_info_impl* nm = m_modules[idx].get();
        if (strcmp(nm->get_qualified_name(), new_module->get_qualified_name()) == 0)
        {
            known = true;
            mi::base::Handle< Mdl_module_info_impl>sh_module(new Mdl_module_info_impl(*nm));
            sh_module->add_shadow(new_module);
            m_modules[idx] = sh_module;
            return 0;
        }
        idx++;
    }
    return -1;
}

const Mdl_package_info_impl* Mdl_package_info_impl::merge_packages(
    const Mdl_package_info_impl* old_node,
    const Mdl_package_info_impl* new_node)
{
    Mdl_package_info_impl* merge_node = new Mdl_package_info_impl(*old_node);
    if(new_node->get_search_path_index_count() > 0) // iterate over all paths to merge?
    {
        merge_node->add_path(new_node->get_search_path(0));
        merge_node->add_path_index(new_node->get_search_path_index(0));
        merge_node->add_resolved_path(new_node->get_resolved_path(0)->get_c_str());
        merge_node->add_in_archive(new_node->in_archive(0));
    }

    
    return merge_node;
}

void Mdl_package_info_impl::add_resolved_path(const char* p)
{
    if (p != nullptr)
        m_resolved_paths.push_back(p);
}

void Mdl_package_info_impl::add_path(const char* p)
{
    if( p != nullptr)
        m_paths.push_back(p);
}

void Mdl_package_info_impl::add_path_index(mi::Size s)
{
    m_path_idx.push_back(s);
}

void Mdl_package_info_impl::add_in_archive(bool value)
{
    m_in_archive.push_back(value);
}

mi::Size Mdl_package_info_impl::get_module_count() const
{
    return m_modules.size();
}

const Mdl_module_info_impl* Mdl_package_info_impl::get_module(mi::Size index) const
{
    if (m_modules.size() <= index)
        return nullptr;

    const mi::base::Handle<const Mdl_module_info_impl>& child = m_modules.at(index);
    child->retain();
    return child.get();
}

mi::Size Mdl_package_info_impl::get_package_count() const
{
    return m_packages.size();
}

const Mdl_package_info_impl* Mdl_package_info_impl::get_package(mi::Size index) const
{
    if (m_packages.size() <= index)
        return nullptr;

    const mi::base::Handle<const Mdl_package_info_impl>& child = m_packages.at(index);
    child->retain();
    return child.get();
}

mi::Size Mdl_package_info_impl::get_child_count() const
{
    return m_packages.size() + m_modules.size();
}

const mi::neuraylib::IMdl_info* Mdl_package_info_impl::get_child(mi::Size index) const
{
    const mi::Size package_count = m_packages.size();
    if (index < package_count)  
        return get_package(index);

    index -= package_count;
    if (index < m_modules.size())  
        return get_module(index);

    return nullptr; 
}

const mi::IString* Mdl_package_info_impl::get_resolved_path(mi::Size idx) const
{
    if (m_resolved_paths.size() <= idx)
        return nullptr;

    mi::IString* istring = new String_impl();
    std::string rp(m_resolved_paths[idx]);
    istring->set_c_str(rp.c_str());
    return istring;
}

const char* Mdl_package_info_impl::get_search_path(mi::Size idx) const
{
    if (m_paths.size() <= idx)
        return nullptr;

    return m_paths[idx].c_str();
}

mi::Size Mdl_package_info_impl::get_search_path_index(mi::Size idx) const
{
    if (m_path_idx.size() <= idx)
        return -1;

    mi::Size s = m_path_idx[idx];
    return s;
}

bool Mdl_package_info_impl::in_archive(Size index) const
{
    if (m_in_archive.size() <= index)
        return false;

    return m_in_archive[index];
}

void Mdl_package_info_impl::set_archive(Size index, bool val)
{
    if (m_in_archive.size() <= index)
        return;

    m_in_archive[index] = val;
}


mi::Size Mdl_package_info_impl::get_search_path_index_count() const
{
    return m_paths.size();
}

void Mdl_package_info_impl::reset_package(Mdl_package_info_impl* child, int index)
{
    if( child != nullptr)
        m_packages[index] = make_handle_dup(child);
}

Mdl_discovery_api_impl::Mdl_discovery_api_impl(mi::neuraylib::INeuray* neuray)
    : m_neuray(neuray)
    , m_mdlc_module(true)
    , m_path_module(true)
{
}

Mdl_discovery_api_impl::~Mdl_discovery_api_impl()
{
    m_neuray = 0;
}

bool Mdl_discovery_api_impl::check_ident_validity(const char* identifier) const
{
    mi::base::Handle<mi::mdl::IMDL> mdl(m_mdlc_module->get_mdl());
    return mdl->is_valid_mdl_identifier(identifier);
}


bool Mdl_discovery_api_impl::discover_recursive(mi::base::Handle<Mdl_package_info_impl> parent,
    const char* search_path, mi::Size s_idx, const char* path, 
    const std::vector<std::string>& invalid_dirs) const
{
    DISK::Directory dir;
    if (!dir.open(path))
        return false;

    std::string current_path(path);
    std::string package_path = slash_to_colon(current_path.substr(strlen(search_path))) + "::";

    std::string entry = dir.read();
    while (!entry.empty())
    {
        std::string resolved_path = HAL::Ospath::join(current_path, entry);
        if (DISK::is_directory(resolved_path.c_str()))
        {
            if (std::find(invalid_dirs.begin(), invalid_dirs.end(), 
                resolved_path) != invalid_dirs.end())
            {
                // todo: add error message
                entry = dir.read();
                continue;
            }

            mi::base::Handle< Mdl_package_info_impl>
                child_package(new Mdl_package_info_impl(entry.c_str(),
                    search_path,
                    resolved_path.c_str(),
                    mi::Uint32(s_idx),
                    (package_path + entry).c_str()));

            mi::Sint32 idx = parent->check_package(child_package.get());
            if (idx >= 0)
            {
                // Package exists already -> merge nodes 
                const Mdl_package_info_impl* mg(parent->get_package(idx));
                mg = parent->merge_packages(mg, child_package.get());

                mi::base::Handle< Mdl_package_info_impl>
                    merge_package(new Mdl_package_info_impl(*mg));

                // Continue recursion with merged node
                discover_recursive(merge_package,
                    search_path, s_idx, resolved_path.c_str(), invalid_dirs);
                parent->reset_package(merge_package.get(), idx);
            }
            else
            {
                // No merge has happened -> continue with new node
                discover_recursive(child_package,
                    search_path, s_idx, resolved_path.c_str(), invalid_dirs);
                parent->add_package(child_package.get());
            }
        }
        else
        {
            std::size_t found_mdl = resolved_path.rfind(".mdl");
            // Filter sym-links and non-mdl files
            if (!DISK::is_file(resolved_path.c_str()) || (found_mdl != resolved_path.length() - 4))
            {
                entry = dir.read();
                continue;
            }

            if (std::find(invalid_dirs.begin(), invalid_dirs.end(), 
                resolved_path.substr(0, resolved_path.length() - 4)) != invalid_dirs.end())
            {
                // todo: add error message
                entry = dir.read();
                continue;
            }

            // Handle module from file
            if (found_mdl != std::string::npos)
            {
                entry = entry.substr(0, entry.size() - 4);

                // Check if file name is valid for MDL
                if (!check_ident_validity(entry.c_str()))
                {
                    entry = dir.read();
                    continue; //ToDo: Add error message 
                }
                mi::base::Handle< Mdl_module_info_impl>
                    module(new Mdl_module_info_impl(entry.c_str(),
                        (package_path + entry).c_str(),
                        resolved_path.c_str(),
                        search_path,
                        s_idx));

                mi::Sint32 res = parent->shadow_module(module.get());
                if (res < 0)
                    parent->add_module(module.get());
            }
        }
        entry = dir.read();
    }
    dir.close();

    return true;
}

namespace {

bool validate_archive(
    std::pair<const std::string, bool>& archive,
    std::map<std::string, bool>& archives,
    std::vector<std::string>& invalid_directories, 
    const std::string& path)
{
    const std::string& a = archive.first;

    // Check file system
    std::string resolved_path = HAL::Ospath::join(path, dot_to_slash(a));
    if (DISK::is_directory(resolved_path.c_str()))
    {
        invalid_directories.push_back(resolved_path);
        archive.second = false;
    }
    else {
        std::string mdl = resolved_path.append(".mdl");
        if (DISK::is_file(mdl.c_str()))
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

} // end namespace

const mi::neuraylib::IMdl_discovery_result* Mdl_discovery_api_impl::discover() const
{
    SYSTEM::Access_module<PATH::Path_module> m_path_module(false);
    const std::vector<std::string>& search_paths = m_path_module->get_search_path(PATH::MDL);

    mi::base::Handle<Mdl_package_info_impl> root_package(
        new Mdl_package_info_impl("", "", "", -1, ""));

    for (mi::Size i = 0; i < search_paths.size(); ++i)
    {
        std::string path = search_paths[i]; 
        if(!DISK::access(path.c_str(), false))
            continue;

        if (!DISK::is_path_absolute(path))
            path = HAL::Ospath::join(DISK::get_cwd(), path);
        path = HAL::Ospath::normpath_v2(path);

        // Collect all archives
        DISK::Directory dir;
        if (!dir.open(path.c_str()))
            continue;

        std::string entry = dir.read();
        std::map<std::string, bool> archives;
        while (!entry.empty())
        {
            if(DISK::is_file(HAL::Ospath::join(path, entry).c_str()))
            {
                std::size_t found_mdr = entry.rfind(".mdr");
                if (found_mdr != std::string::npos && found_mdr == entry.size() - 4)
                    archives.insert(std::make_pair(entry.substr(0, found_mdr), true));  
            }
            entry = dir.read();
        }

        // Process archives
        std::vector<std::string> invalid_directies;
        for (auto& archive : archives)
        {
            if (validate_archive(archive, archives, invalid_directies, path))
            {
                // Discover archive in search path (root folder only)
                std::string resolved_path = HAL::Ospath::join(path, archive.first);
                resolved_path += ".mdr";
                discover_archive(root_package, path.c_str(), i, resolved_path.c_str());
            }
        }

        // Discover file system
        discover_recursive(root_package, path.c_str(), i, path.c_str(), invalid_directies);
    }
    // Sort graph nodes alphabetically
    root_package->sort_children();
    
    mi::base::Handle<Mdl_discovery_result_impl>
        disc_res(new Mdl_discovery_result_impl(root_package.get(), search_paths));
    disc_res->retain();
    return disc_res.get();
}

bool Mdl_discovery_api_impl::create_archive_graph_recursive(
    mi::base::Handle<Mdl_package_info_impl> parent, 
    const char* previous_module,
    const char* full_q_path, 
    const char* search_path,
    const char* resolved_path,
    mi::Size s_idx,
    mi::Size level) const
{
    std::string quali = std::string(full_q_path);
    size_t p = quali.find_first_of("::");
    bool is_package = true;

    // Package found
    std::string entry = quali.substr(0, p);
    std::string qualified_path("");
    std::string strip_path(full_q_path);

    for (mi::Size l = 0; l <= level; l++)
    {
        // strip entry and qualified path from full path
        size_t e = strip_path.find_first_of("::");

        if (e == std::string::npos)
        {
            // Check if file name is valid for MDL
            if (!check_ident_validity(entry.c_str()))
                return false; //ToDo: Add error message 

            // Convert to correct path convention
            std::string qpath_form(qualified_path + strip_path);
            std::string rpath_form(resolved_path);
            rpath_form += ":" + colon_to_slash(qpath_form) + ".mdl";
            qpath_form = "::" + qpath_form;

            // Add module
            mi::base::Handle< Mdl_module_info_impl>
                module(new Mdl_module_info_impl(strip_path.c_str(),
                        qpath_form.c_str(),
                        rpath_form.c_str(),
                        search_path,
                        s_idx));
            module->set_archive(true);

            mi::Sint32 res = parent->shadow_module(module.get());
            if (res < 0)
                parent->add_module(module.get());

            // Terminate
            is_package = false;
        }
        qualified_path.append(strip_path.substr(0, e + 2));
        entry = strip_path.substr(0, e);
        strip_path = strip_path.substr(e + 2, strip_path.size());
    }

    if (is_package)
    {
        // Convert to correct path convention
        std::string qpath_form(qualified_path.substr(0, qualified_path.size() - 2));
        std::string rpath_form(resolved_path);
        rpath_form += ":" + colon_to_slash(qpath_form);
        qpath_form = "::" + qpath_form;

        mi::base::Handle< Mdl_package_info_impl>new_package(new Mdl_package_info_impl(
            entry.c_str(),
            search_path,
            rpath_form.c_str(),
            mi::Uint32(s_idx),
            qpath_form.c_str()));
        new_package->set_archive(0, true);
          
        // Check if package already exists
        size_t f = std::string(previous_module).find(qualified_path);
        if ((f == std::string::npos) && (s_idx == 0))
        {
            // Add package to first graph
            parent->add_package(new_package.get());
            create_archive_graph_recursive(new_package, previous_module,
                full_q_path, search_path, resolved_path, s_idx, level + 1);               
        }
        else
        {
            mi::Sint32 idx = parent->check_package(new_package.get());
            if (idx >= 0)
            {
                // Reuse package
                const Mdl_package_info_impl* mg(parent->get_package(idx));
                    
                mi::base::Handle< Mdl_package_info_impl>
                    reuse_package(new Mdl_package_info_impl(*mg));

                // Add new search path if it does not exist yet
                bool found = false;
                mi::Size i = 0;
                while ((i < mg->get_search_path_index_count()) && (found == false))
                {
                    if (strcmp(mg->get_search_path(i), search_path) == 0)
                        found = true;
                    i++;
                }
                if (!found)
                {
                    reuse_package->add_path(search_path);
                    reuse_package->add_path_index(s_idx);
                    reuse_package->add_resolved_path(rpath_form.c_str());
                    reuse_package->add_in_archive(true);
                }

                // Continue recursion with reused node
                parent->reset_package(reuse_package.get(), idx);
                create_archive_graph_recursive(reuse_package, previous_module,
                    full_q_path, search_path, resolved_path, s_idx, level + 1);
            }
            else
            {
                // Add new package
                parent->add_package(new_package.get());
                create_archive_graph_recursive(new_package, previous_module,
                    full_q_path, search_path, resolved_path, s_idx, level + 1);
            }
        }
    }
    return true;
}

bool Mdl_discovery_api_impl::create_archive_graph(
    mi::base::Handle<const mi::neuraylib::IManifest> m,
    mi::base::Handle<Mdl_package_info_impl> parent,
    const char* search_path, 
    mi::Size s_idx,
    const char* res_path) const
{
    // Read modules from archive
    std::vector<std::string> module_list;
    for (mi::Size i = 0; i < m->get_number_of_fields(); ++i)
    {
        const char* manifest_value = m->get_value(i);
        if (manifest_value)
        {
            const char* manifest_key = m->get_key(i);
            if (strcmp(manifest_key, "module") == 0)
            {
               std::string manifest_val(m->get_value(i));
               if (manifest_val.size() > 0)
                   module_list.push_back(manifest_val);
            }
        }
    }

    for (mi::Size x=0; x < module_list.size(); ++x)
    {
        mi::Size p = 0;
        while (module_list[x][p] == ':')
            p++;

        std::string module_path = module_list[x].substr(p);

        create_archive_graph_recursive(parent,
            x == 0 ? "" : module_list[x - 1].c_str(), // previous module
            module_path.c_str(),
            search_path,
            res_path,
            s_idx,
            0);
    }
    return true;
}

bool Mdl_discovery_api_impl::discover_archive(mi::base::Handle<Mdl_package_info_impl> parent,
    const char* search_path, mi::Size s_idx, const char* res_path) const
{
    mi::base::Handle<mi::neuraylib::IMdl_archive_api>archive_api(
        m_neuray->get_api_component<mi::neuraylib::IMdl_archive_api>());
    mi::base::Handle<const mi::neuraylib::IManifest>
        manifest(archive_api->get_manifest(res_path));
    if (!manifest)
        return false;
    create_archive_graph(manifest, parent, search_path, s_idx, res_path);
    return true;
}

mi::Sint32 Mdl_discovery_api_impl::start()
{
    m_path_module.set();
    m_mdlc_module.set();
    return 0;
}

mi::Sint32 Mdl_discovery_api_impl::shutdown()
{
    m_path_module.reset();
    m_mdlc_module.reset();
    return 0;
}

Mdl_discovery_result_impl::Mdl_discovery_result_impl(const Mdl_package_info_impl* graph,
    const std::vector<std::string>paths)
    : m_graph(make_handle_dup(graph))
{
    for (mi::Size i=0; i < paths.size(); ++i)
        m_search_paths.push_back(paths[i]);
}

const mi::neuraylib::IMdl_package_info* Mdl_discovery_result_impl::get_graph() const
{
    m_graph->retain();
    return m_graph.get();
}

const char* Mdl_discovery_result_impl::get_search_path(mi::Size index) const
{
    if (index >= m_search_paths.size())
        return nullptr;
    return m_search_paths.at(index).c_str();
}

mi::Size Mdl_discovery_result_impl::get_search_paths_count() const
{
    return m_search_paths.size();
}

} // namespace NEURAY

} // namespace MI
