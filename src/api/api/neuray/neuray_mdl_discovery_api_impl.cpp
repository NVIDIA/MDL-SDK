/***************************************************************************************************
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
 **************************************************************************************************/

/** \file
 ** \brief Source for the IMdl_discovery_api implementation.
 **/

#include "pch.h"

#include <mdl/compiler/compilercore/compilercore_archiver.h>
#include <mdl/compiler/compilercore/compilercore_allocator.h>
#include <mdl/compiler/compilercore/compilercore_zip_utils.h>

#include "neuray_mdl_discovery_api_impl.h"
#include "neuray_string_impl.h"

#include <mi/mdl/mdl_archiver.h>
#include <mi/mdl/mdl_entity_resolver.h>
#include <mi/mdl/mdl_mdl.h>
#include <mi/mdl/mdl_modules.h>
#include <mi/mdl/mdl_streams.h>

#include <base/hal/disk/disk.h>
#include <base/hal/hal/i_hal_ospath.h>
#include <base/lib/path/i_path.h>
#include <base/system/main/i_module_id.h>
#include <base/util/string_utils/i_string_utils.h>
#include <mdl/integration/mdlnr/i_mdlnr.h>
#include <io/scene/mdl_elements/i_mdl_elements_module.h>
#include <io/scene/mdl_elements/i_mdl_elements_utilities.h>

#include <string>

namespace MI {

namespace NEURAY {

namespace
{
    template<class T>
    bool compare_simple_names(const mi::base::Handle<const T> &hp1,
        const mi::base::Handle<const T> &hp2)
    {
        return 0 > strcmp(hp1->get_simple_name(), hp2->get_simple_name());
    }
}

void Mdl_module_info_impl::add_shadow(Mdl_module_info_impl* shadow)
{
    m_shadows.push_back(make_handle_dup(shadow));
}

mi::neuraylib::IMdl_info::Kind Mdl_module_info_impl::get_kind() const
{
    return mi::neuraylib::IMdl_info::Kind::DK_MODULE;
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
    if (m_shadows.size() <= index)
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

void Mdl_module_info_impl::sort_shadows()
{
    std::sort(
        m_shadows.begin(), 
        m_shadows.end(), 
        compare_simple_names<Mdl_module_info_impl>);
}

const char* Mdl_xliff_info_impl::get_extension() const
{
    return m_extension.c_str();
}

mi::neuraylib::IMdl_info::Kind Mdl_xliff_info_impl::get_kind() const
{
    return mi::neuraylib::IMdl_info::Kind::DK_XLIFF;
}

const char* Mdl_xliff_info_impl::get_resolved_path() const
{
    return m_resolved_path.c_str();
}

mi::Size Mdl_xliff_info_impl::get_search_path_index() const
{
    return m_search_index;
}

const char* Mdl_xliff_info_impl::get_search_path() const
{
    return m_search_path.c_str();
}

bool Mdl_xliff_info_impl::in_archive() const
{
    return m_in_archive;
}

void Mdl_xliff_info_impl::set_archive(bool val)
{
    m_in_archive = val;
}

mi::neuraylib::IMdl_info::Kind Mdl_texture_info_impl::get_kind() const
{
    return mi::neuraylib::IMdl_info::Kind::DK_TEXTURE;
}

mi::neuraylib::IMdl_info::Kind Mdl_lightprofile_info_impl::get_kind() const
{
    return mi::neuraylib::IMdl_info::Kind::DK_LIGHTPROFILE;
}

mi::neuraylib::IMdl_info::Kind Mdl_measured_bsdf_info_impl::get_kind() const
{
    return mi::neuraylib::IMdl_info::Kind::DK_MEASURED_BSDF;
}

Mdl_package_info_impl::Mdl_package_info_impl(
    const std::string& simple_name,
    const std::string& search_path,
    const std::string& resolved_path,
    mi::Sint32 p_idx,
    const std::string& qualified_name)
{
    m_kind = mi::neuraylib::IMdl_info::Kind::DK_PACKAGE;
    m_qualified_name = qualified_name;
    m_simple_name = simple_name;

    if (p_idx >= 0) {
        m_paths.push_back(search_path);
        m_path_idx.push_back(p_idx);
        m_resolved_paths.push_back(resolved_path);
        m_in_archive.push_back(false);
    }
}

mi::neuraylib::IMdl_info::Kind Mdl_package_info_impl::get_kind() const
{
    return m_kind;
}

void Mdl_package_info_impl::add_module(Mdl_module_info_impl* module)
{
    m_modules.push_back(make_handle_dup(module));
}

void Mdl_package_info_impl::add_package(Mdl_package_info_impl* child)
{
    m_packages.push_back(make_handle_dup(child));
}

void Mdl_package_info_impl::add_xliff(Mdl_xliff_info_impl* xliff)
{
    m_xliffs.push_back(make_handle_dup(xliff));
}

void Mdl_package_info_impl::add_texture(Mdl_texture_info_impl* texture)
{
    m_textures.push_back(make_handle_dup(texture));
}

void Mdl_package_info_impl::add_lightprofile(Mdl_lightprofile_info_impl* lightprofile)
{
    m_lightprofiles.push_back(make_handle_dup(lightprofile));
}

void Mdl_package_info_impl::add_measured_bsdf(Mdl_measured_bsdf_info_impl* bsdf)
{
    m_measured_bsdfs.push_back(make_handle_dup(bsdf));
}

mi::Sint32 Mdl_package_info_impl::check_package(Mdl_package_info_impl* child)
{
    for (mi::Uint32 c = 0; c < m_packages.size(); c++)
        if (strcmp(m_packages[c]->get_qualified_name(), child->get_qualified_name()) == 0)
            return Sint32(c);
    return -1;
}

void Mdl_package_info_impl::sort_children()
{
    std::sort(m_packages.begin(), m_packages.end(), compare_simple_names<Mdl_package_info_impl>);
    for (mi::Uint32 m = 0; m < m_packages.size(); m++)
        m_packages[m]->sort_children();

    for (mi::Uint32 m = 0; m < m_modules.size(); m++) {
        Mdl_module_info_impl* mod = m_modules[m].get();
        mod->sort_shadows();
    }

    for (mi::Uint32 m = 0; m < m_textures.size(); m++) {
        Mdl_texture_info_impl* tex = m_textures[m].get();
        tex->sort_shadows();
    }

    for (mi::Uint32 m = 0; m < m_lightprofiles.size(); m++) {
        Mdl_lightprofile_info_impl* lp = m_lightprofiles[m].get();
        lp->sort_shadows();
    }

    for (mi::Uint32 m = 0; m < m_measured_bsdfs.size(); m++) {
        Mdl_measured_bsdf_info_impl* mb = m_measured_bsdfs[m].get();
        mb->sort_shadows();
    }

    std::sort(m_modules.begin(), m_modules.end(), compare_simple_names<Mdl_module_info_impl>);
    std::sort(m_textures.begin(), m_textures.end(), compare_simple_names<Mdl_texture_info_impl>);
    std::sort(m_lightprofiles.begin(), m_lightprofiles.end(), compare_simple_names<Mdl_lightprofile_info_impl>);
    std::sort(m_measured_bsdfs.begin(), m_measured_bsdfs.end(), compare_simple_names<Mdl_measured_bsdf_info_impl>);
}

mi::Sint32 Mdl_package_info_impl::shadow_module(Mdl_module_info_impl* new_module)
{
    for (mi::Uint32 idx = 0; idx < m_modules.size(); idx++) {
        const Mdl_module_info_impl* nm = m_modules[idx].get();
        if (strcmp(nm->get_qualified_name(), new_module->get_qualified_name()) == 0) {
            mi::base::Handle<Mdl_module_info_impl>sh_module(new Mdl_module_info_impl(*nm));
            sh_module->add_shadow(new_module);
            m_modules[idx] = sh_module;
            return 0;
        }
    }
    return -1;
}

mi::Sint32 Mdl_package_info_impl::shadow_texture(Mdl_texture_info_impl* new_texture)
{
    for (mi::Uint32 idx = 0; idx < m_textures.size(); idx++) {
        const Mdl_texture_info_impl* nm = m_textures[idx].get();
        if (strcmp(nm->get_resolved_path(), new_texture->get_resolved_path()) == 0) {
            mi::base::Handle<Mdl_texture_info_impl>sh_texture(new Mdl_texture_info_impl(*nm));
            sh_texture->add_shadow(new_texture);
            m_textures[idx] = sh_texture;
            return 0;
        }
    }
    return -1;
}

mi::Sint32 Mdl_package_info_impl::shadow_lightprofile(
    Mdl_lightprofile_info_impl* new_lightprofile)
{
    for (mi::Uint32 idx = 0; idx < m_lightprofiles.size(); idx++) {
        const Mdl_lightprofile_info_impl* nm = m_lightprofiles[idx].get();
        if (strcmp(nm->get_resolved_path(), new_lightprofile->get_resolved_path()) == 0) {
            mi::base::Handle<Mdl_lightprofile_info_impl>sh_lightprofile(
                new Mdl_lightprofile_info_impl(*nm));
            sh_lightprofile->add_shadow(new_lightprofile);
            m_lightprofiles[idx] = sh_lightprofile;
            return 0;
        }
    }
    return -1;
}

mi::Sint32 Mdl_package_info_impl::shadow_measured_bsdf(
    Mdl_measured_bsdf_info_impl* new_measured_bsdf)
{
    for (mi::Uint32 idx = 0; idx < m_measured_bsdfs.size(); idx++) {
        const Mdl_measured_bsdf_info_impl* nm = m_measured_bsdfs[idx].get();
        if (strcmp(nm->get_resolved_path(), new_measured_bsdf->get_resolved_path()) == 0) {
            mi::base::Handle<Mdl_measured_bsdf_info_impl>sh_measured_bsdf(
                new Mdl_measured_bsdf_info_impl(*nm));
            sh_measured_bsdf->add_shadow(new_measured_bsdf);
            m_measured_bsdfs[idx] = sh_measured_bsdf;
            return 0;
        }
    }
    return -1;
}

const Mdl_package_info_impl* Mdl_package_info_impl::merge_packages(
    const Mdl_package_info_impl* old_node, 
    const Mdl_package_info_impl* new_node)
{
    Mdl_package_info_impl* merge_node = new Mdl_package_info_impl(*old_node);
    if (new_node->get_search_path_index_count() > 0) {
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
    if (p != nullptr)
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

mi::Size Mdl_package_info_impl::get_xliff_count() const
{
    return m_xliffs.size();
}

const Mdl_xliff_info_impl* Mdl_package_info_impl::get_xliff(mi::Size index) const
{
    if (m_xliffs.size() <= index)
        return nullptr;

    const mi::base::Handle<const Mdl_xliff_info_impl>& child = m_xliffs.at(index);
    child->retain();
    return child.get();
}


mi::Size Mdl_package_info_impl::get_texture_count() const
{
    return m_textures.size();
}

const Mdl_texture_info_impl* Mdl_package_info_impl::get_texture(mi::Size index) const
{
    if (m_textures.size() <= index)
        return nullptr;

    const mi::base::Handle<const Mdl_texture_info_impl>& child = m_textures.at(index);
    child->retain();
    return child.get();
}


mi::Size Mdl_package_info_impl::get_lightprofile_count() const
{
    return m_lightprofiles.size();
}

const Mdl_lightprofile_info_impl* Mdl_package_info_impl::get_lightprofile(mi::Size index) const
{
    if (m_lightprofiles.size() <= index)
        return nullptr;

    const mi::base::Handle<const Mdl_lightprofile_info_impl>& child = m_lightprofiles.at(index);
    child->retain();
    return child.get();
}

mi::Size Mdl_package_info_impl::get_measured_bsdf_count() const
{
    return m_measured_bsdfs.size();
}

const Mdl_measured_bsdf_info_impl* Mdl_package_info_impl::get_measured_bsdf(mi::Size index) const
{
    if (m_measured_bsdfs.size() <= index)
        return nullptr;

    const mi::base::Handle<const Mdl_measured_bsdf_info_impl>& child = m_measured_bsdfs.at(index);
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
    mi::Size children_count = m_packages.size() + m_modules.size() + m_xliffs.size();
    children_count += (m_textures.size() + m_lightprofiles.size() + m_measured_bsdfs.size());
    return children_count;
}

const mi::neuraylib::IMdl_info* Mdl_package_info_impl::get_child(mi::Size index) const
{
    const mi::Size package_count = m_packages.size();
    if (index < package_count)  
        return get_package(index);

    const mi::Size modules_count = m_modules.size();
    index -= package_count;
    if (index < m_modules.size())
        return get_module(index);

    const mi::Size xliff_count = m_xliffs.size();
    index -= modules_count;
    if (index < m_xliffs.size())
        return get_xliff(index);

    const mi::Size texture_count = m_textures.size();
    index -= xliff_count;
    if (index < m_textures.size())
        return get_texture(index);

    const mi::Size lightprofile_count = m_lightprofiles.size();
    index -= texture_count;
    if (index < m_lightprofiles.size())
        return get_lightprofile(index);

    index -= lightprofile_count;
    if (index < m_measured_bsdfs.size())
        return get_measured_bsdf(index);

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

void Mdl_package_info_impl::set_kind(mi::neuraylib::IMdl_info::Kind k)
{
    m_kind = k;
}

mi::Size Mdl_package_info_impl::get_search_path_index_count() const
{
    return m_paths.size();
}

void Mdl_package_info_impl::reset_package(Mdl_package_info_impl* child, int index)
{
    if (child != nullptr)
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
    m_neuray = nullptr;
}

bool Mdl_discovery_api_impl::is_valid_node_name(const char* identifier) const
{
    if (!identifier)
        return false;

    std::string s = identifier;
    s = "::" + s;
    return MDL::is_valid_module_name(s);
}

bool Mdl_discovery_api_impl::is_known_search_path(
    const char* path,
    mi::base::Handle<Mdl_package_info_impl>package) const
{
    for (mi::Size i = 0; i < package->get_search_path_index_count(); ++i) {
        if (strcmp(package->get_search_path(i), path) == 0)
            return true;
    }
    return false;
}

namespace {

    void replace_expression(
        const std::string& input,
        const std::string& old,
        const std::string& with,
        std::string& output)
    {
        if (input.size() == 0)
            output = input;

        std::string sentance(input);
        size_t offset(0);
        size_t pos(0);
        while ((pos = sentance.find(old, offset)) != std::string::npos) {
            sentance.replace(pos, old.length(), with);
            offset = pos + with.length();
        }
        output = sentance;
    }

    bool validate_archive(
        std::pair<const std::string, bool>& archive,
        std::map<std::string, bool>& archives,
        std::vector<std::string>& invalid_directories,
        const std::string& path)
    {
        const std::string& a = archive.first;
        std::string resolved_path = HAL::Ospath::join(path, a);
        std::string res;
#ifdef MI_PLATFORM_WINDOWS      
        replace_expression(resolved_path, ".", "\\", res);
#else
        replace_expression(resolved_path, ".", "/", res);
#endif
        resolved_path = res;
        if (DISK::is_directory(resolved_path.c_str())) {
            invalid_directories.push_back(resolved_path);
            archive.second = false;
        }
        else {
            std::string mdl = resolved_path.append(".mdl");
            if (DISK::is_file(mdl.c_str())) {
                invalid_directories.push_back(resolved_path);
                archive.second = false;
            }
        }

        for (auto& other_archive : archives) {
            if (other_archive.second == false)
                continue;

            const std::string& o = other_archive.first;
            if (a == o)
                continue;

            auto l = a.size();
            auto ol = o.size();

            if (l < ol) {
                if (o.substr(0, l) == a) {
                    other_archive.second = false;
                    archive.second = false;
                }
            }
            else if (ol < l) {
                if (a.substr(0, ol) == o) {
                    other_archive.second = false;
                    archive.second = false;
                }
            }
        }
        return archive.second;
    }

    void qualified_path_to_resolved_path(
        std::string val,
        std::string& res)
    {
        if (val.size() > 0) {
#ifdef MI_PLATFORM_WINDOWS
            replace_expression(val, "::", "\\", res);
#else
            replace_expression(val, "::", "/", res);
#endif
        }
    }

    void resolved_path_to_qualified_path(
        std::string val,
        std::string& res)
    {
        if (val.size() > 0) {
#ifdef MI_PLATFORM_WINDOWS
            replace_expression(val, "\\", "::", res);
#else
            replace_expression(val, "/", "::", res);
#endif
        }
    }

    bool is_valid_path(
        const std::vector<std::string>& invalid_dirs,
        std::string& res_path)
    {
        if (std::find(
            invalid_dirs.begin(),
            invalid_dirs.end(),
            res_path) == invalid_dirs.end())
            return true;
        return false;
    }

    bool is_mdl_file(const std::string& extension)
    {
        STRING::to_lower(extension);
        if (extension == ".mdl")
            return true;
        return false;
    }

    bool is_xlf_file(const std::string& extension)
    {
        STRING::to_lower(extension);
        if (extension == ".xlf")
            return true;
        return false;
    }

    bool is_texture_file(const std::string& extension)
    {
        STRING::to_lower(extension);
        return (
            extension == ".png" ||
            extension == ".exr" ||
            extension == ".jpg" ||
            extension == ".jpeg" ||
            extension == ".ptx");
    };

    bool is_lightprofile_file(const std::string& extension)
    {
        STRING::to_lower(extension);
        if (extension == ".ies")
            return true;
        return false;
    };

    bool is_bsdf_file(const std::string& extension)
    {
        STRING::to_lower(extension);
        if (extension == ".mbsdf")
            return true;
        return false;
    };

    void get_resource_qualified_path(
        const std::string& resolved_path, 
        const std::string& search_path,
        std::string& output ) {
        std::string resource_q_path = resolved_path.substr(search_path.size());
#ifdef MI_PLATFORM_WINDOWS
        replace_expression(resource_q_path, "\\", "/", output);
#endif
    }

} // end namespace

bool Mdl_discovery_api_impl::discover_filesystem_recursive(
    mi::base::Handle<Mdl_package_info_impl> parent,
    const char* search_path, 
    mi::Size s_idx, 
    const char* path, 
    const std::vector<std::string>& invalid_dirs,
    mi::Uint32 filter) const
{
    DISK::Directory dir;
    if (!dir.open(path))
        return false;

    std::string current_path(path);
    std::string package_path;
    resolved_path_to_qualified_path(
        current_path.substr(strlen(search_path)), 
        package_path);
    package_path += "::";

    std::string entry = dir.read();
    while (!entry.empty()) {
        std::string resolved_path = HAL::Ospath::join(current_path, entry);
        if (DISK::is_directory(resolved_path.c_str())) {
            if (!is_valid_path( 
                invalid_dirs, 
                resolved_path)) {
                entry = dir.read();
                continue;
            }
           
            mi::base::Handle<Mdl_package_info_impl> child_package(
                new Mdl_package_info_impl(
                    entry.c_str(), 
                    search_path,
                    resolved_path.c_str(), 
                    mi::Uint32(s_idx), 
                    (package_path + entry).c_str()));

            if (!is_valid_node_name(entry.c_str()) ||
                (parent->get_kind() == mi::neuraylib::IMdl_info::Kind::DK_DIRECTORY)) {
                if (filter & mi::neuraylib::IMdl_info::Kind::DK_DIRECTORY) {
                    child_package->set_kind(mi::neuraylib::IMdl_info::Kind::DK_DIRECTORY);
                }
                else {
                    entry = dir.read();
                    continue;
                }
            }

            mi::Sint32 idx = parent->check_package(child_package.get());
            if (idx >= 0) { 
                const Mdl_package_info_impl* mg(parent->get_package(idx));
                mg = parent->merge_packages(mg, child_package.get());
                mi::base::Handle< Mdl_package_info_impl> merge_package(
                    new Mdl_package_info_impl(*mg));

                // Continue recursion with a merged node
                discover_filesystem_recursive(
                    merge_package,
                    search_path, 
                    s_idx, 
                    resolved_path.c_str(), 
                    invalid_dirs,
                    filter);
                parent->reset_package(merge_package.get(), idx);
            }
            else{
                // Continue recursion with a new node
                discover_filesystem_recursive(
                    child_package,
                    search_path, 
                    s_idx, 
                    resolved_path.c_str(), 
                    invalid_dirs,
                    filter);
                parent->add_package(child_package.get());
            }
        }
        else {
            size_t pos_e = entry.find_last_of('.');
            if (pos_e == std::string::npos) {
                entry = dir.read();
                continue;
            }
            else {
                entry = entry.substr(0, pos_e);
                if (!is_valid_node_name(entry.c_str())) {
                    entry = dir.read();
                    continue;
                }
            }
            
            std::string res_qualified_path;
            get_resource_qualified_path(resolved_path, search_path, res_qualified_path);

            size_t pos_rp = resolved_path.find_last_of('.');
            std::string short_path(resolved_path.substr(0, pos_rp));
            if (DISK::is_file(resolved_path.c_str()) &&
                (is_valid_path(invalid_dirs, short_path)) &&
                (pos_rp != std::string::npos)) {
                std::string ext = resolved_path.substr(
                    pos_rp,
                    resolved_path.size() - 1);
                if ((filter & mi::neuraylib::IMdl_info::Kind::DK_MODULE) &&
                        (is_mdl_file(ext)) &&
                        (parent->get_kind() == mi::neuraylib::IMdl_info::Kind::DK_PACKAGE)) {
                        mi::base::Handle<Mdl_module_info_impl> module(
                            new Mdl_module_info_impl(
                                entry.c_str(),
                                (package_path + entry).c_str(),
                                resolved_path.c_str(),
                                search_path,
                                s_idx,
                                false));
                        if (parent->shadow_module(module.get()) < 0)
                            parent->add_module(module.get());
                }
                else if ((filter & mi::neuraylib::IMdl_info::Kind::DK_XLIFF) &&
                        (is_xlf_file(ext))) {
                        mi::base::Handle<Mdl_xliff_info_impl>xliff(
                            new Mdl_xliff_info_impl(
                                entry.c_str(),
                                res_qualified_path.c_str(),
                                resolved_path.c_str(),
                                ext.c_str(),
                                search_path,
                                s_idx,
                                false));
                            parent->add_xliff(xliff.get());
                }
                else if ((filter & mi::neuraylib::IMdl_info::Kind::DK_TEXTURE) &&
                        (is_texture_file(ext))) {
                        mi::base::Handle<Mdl_texture_info_impl>texture(
                            new Mdl_texture_info_impl(
                                entry.c_str(),
                                res_qualified_path.c_str(),
                                resolved_path.c_str(),
                                ext.c_str(),
                                search_path,
                                s_idx,
                                false));
                        if (parent->shadow_texture(texture.get()) < 0)
                            parent->add_texture(texture.get());
                }
                else if ((filter & mi::neuraylib::IMdl_info::Kind::DK_LIGHTPROFILE) &&
                        (is_lightprofile_file(ext))) {
                        mi::base::Handle<Mdl_lightprofile_info_impl>lightprofile(
                            new Mdl_lightprofile_info_impl(
                                entry.c_str(),
                                res_qualified_path.c_str(),
                                resolved_path.c_str(),
                                ext.c_str(),
                                search_path,
                                s_idx,
                                false));
                        if (parent->shadow_lightprofile(lightprofile.get()) < 0)
                            parent->add_lightprofile(lightprofile.get());
                }
                else if ((filter & mi::neuraylib::IMdl_info::Kind::DK_MEASURED_BSDF) &&
                        (is_bsdf_file(ext))) {
                        mi::base::Handle<Mdl_measured_bsdf_info_impl>measured_bsdf(
                            new Mdl_measured_bsdf_info_impl(
                                entry.c_str(),
                                res_qualified_path.c_str(),
                                resolved_path.c_str(),
                                ext.c_str(),
                                search_path,
                                s_idx,
                                false));
                        if (parent->shadow_measured_bsdf(measured_bsdf.get()) < 0)
                            parent->add_measured_bsdf(measured_bsdf.get());
                }
            }
        }
        entry = dir.read();
    }
    dir.close();
    return true;
}

const mi::neuraylib::IMdl_discovery_result* Mdl_discovery_api_impl::discover(
    mi::Uint32 filter) const
{
    const std::vector<std::string>& search_paths = m_path_module->get_search_path(PATH::MDL);

    mi::base::Handle<Mdl_package_info_impl> root_package(
        new Mdl_package_info_impl("", "", "", -1, ""));

    for (mi::Size i = 0; i < search_paths.size(); ++i) {
        std::string path = search_paths[i]; 
        if (!DISK::access(path.c_str(), false))
            continue;

        if (!DISK::is_path_absolute(path))
            path = HAL::Ospath::join(DISK::get_cwd(), path);
        path = HAL::Ospath::normpath_v2(path);

        DISK::Directory dir;
        if (!dir.open(path.c_str()))
            continue;

        std::string entry = dir.read();
        std::map<std::string, bool> archives;
        while (!entry.empty()) {
            if (DISK::is_file(HAL::Ospath::join(path, entry).c_str())) {
                std::size_t found_mdr = entry.rfind(".mdr");
                if (found_mdr != std::string::npos && found_mdr == entry.size() - 4)
                    archives.insert(
                        std::make_pair(entry.substr(0, found_mdr), 
                        true));  
            }
            entry = dir.read();
        }

        // Discover archives
        std::vector<std::string> invalid_directies;
        for (auto& archive : archives) {
            if (validate_archive(
                archive, 
                archives, 
                invalid_directies, 
                path)) {
                    std::string resolved_path = HAL::Ospath::join(path, archive.first);
                    resolved_path += ".mdr";
                    discover_archive(
                        root_package, 
                        path.c_str(), 
                        i, 
                        resolved_path.c_str(),
                        filter);
            }
        }

        // Discover file system
        discover_filesystem_recursive(
            root_package, 
            path.c_str(), 
            i, 
            path.c_str(), 
            invalid_directies,
            filter);
    }
    root_package->sort_children();
    
    mi::base::Handle<Mdl_discovery_result_impl>
        disc_res(new Mdl_discovery_result_impl(
            root_package.get(), 
            search_paths));
    disc_res->retain();
    return disc_res.get();
}

bool Mdl_discovery_api_impl::add_archive_entries(
    mi::base::Handle<Mdl_package_info_impl> parent,  
    mi::Size s_idx, 
    const char* sp, 
    const char* fqp, 
    const char* rp, 
    std::string& qualified_path,
    const char* extension, 
    std::string& entry, 
    mi::Size level) const
{
    std::string quali = std::string(fqp);
    entry = quali.substr(0, quali.find_first_of("::"));
    std::string skip_path(fqp);

    bool entry_added = false;

    for (mi::Size l = 0; l <= level; l++) {
        size_t e = skip_path.find_first_of("::");

        if (e == std::string::npos) {
            if (!is_valid_node_name(entry.c_str()))
                return false;
            // Convert paths to compiler convention
            std::string rpath_form(rp);
            std::string qpath_form(qualified_path + skip_path);
            std::string conv;
            qualified_path_to_resolved_path(
                qpath_form,
                conv);
            rpath_form += ":" + conv + std::string(extension);
            qpath_form = "::" + qpath_form;

            std::string resource_qualified_path;
            get_resource_qualified_path(rpath_form, sp, resource_qualified_path);

            if (is_mdl_file(extension)) {
                // Add module
                mi::base::Handle<Mdl_module_info_impl> module(
                    new Mdl_module_info_impl(
                        skip_path.c_str(),
                        qpath_form.c_str(),
                        rpath_form.c_str(),
                        sp,
                        s_idx,
                        true));

                if (parent->shadow_module(module.get()) < 0)
                    parent->add_module(module.get());
                entry_added = true;
            }
            else if (is_xlf_file(extension)) {
                mi::base::Handle<Mdl_xliff_info_impl>xliff(
                    new Mdl_xliff_info_impl(
                        skip_path.c_str(),
                        resource_qualified_path.c_str(),
                        rpath_form.c_str(),
                        extension,
                        sp,
                        s_idx,
                        true));
                parent->add_xliff(xliff.get());
                entry_added = true;
            }
            else if (is_texture_file(extension)) {
                mi::base::Handle<Mdl_texture_info_impl>texture(
                    new Mdl_texture_info_impl(
                        skip_path.c_str(),
                        resource_qualified_path.c_str(),
                        rpath_form.c_str(),
                        extension,
                        sp,
                        s_idx,
                        true));
                if (parent->shadow_texture(texture.get()) < 0) {
                    parent->add_texture(texture.get());
                    entry_added = true;
                }
            }
            else if (is_lightprofile_file(extension)) {
                mi::base::Handle<Mdl_lightprofile_info_impl>lightprofile(
                    new Mdl_lightprofile_info_impl(
                        skip_path.c_str(),
                        resource_qualified_path.c_str(),
                        rpath_form.c_str(),
                        extension,
                        sp,
                        s_idx,
                        true));
                lightprofile->set_archive(true);
                if (parent->shadow_lightprofile(lightprofile.get()) < 0) {
                    parent->add_lightprofile(lightprofile.get());
                    entry_added = true;
                }   
            }
            else if (is_bsdf_file(extension)) {
                mi::base::Handle<Mdl_measured_bsdf_info_impl>measured_bsdf(
                    new Mdl_measured_bsdf_info_impl(
                        skip_path.c_str(),
                        resource_qualified_path.c_str(),
                        rpath_form.c_str(),
                        extension,
                        sp,
                        s_idx,
                        true));
                measured_bsdf->set_archive(true);
                if (parent->shadow_measured_bsdf(measured_bsdf.get()) < 0) {
                    parent->add_measured_bsdf(measured_bsdf.get());
                    entry_added = true;
                }
            }
        }

        if ((e == std::string::npos) && (std::string(extension).size() == 0)){
            // Last package in archive entry path
            qualified_path.append(skip_path);
            entry = skip_path;
        }
        else {
            qualified_path.append(skip_path.substr(0, e + 2));
            entry = skip_path.substr(0, e);
            skip_path = skip_path.substr(e + 2, skip_path.size());
        }
    }
    return entry_added;
}

bool Mdl_discovery_api_impl::discover_archive_recursive(
    mi::base::Handle<Mdl_package_info_impl> parent,
    const char* previous_module,
    const char* full_q_path,
    const char* search_path,
    const char* resolved_path,
    const char* extension,
    mi::Size s_idx,
    mi::Size level) const
{
    std::string entry("");
    std::string qualified_path("");

    bool is_file = add_archive_entries(
        parent,
        s_idx,
        search_path,
        full_q_path,
        resolved_path,
        qualified_path,
        extension,
        entry,
        level);

        if ((std::string(extension).size() == 0) || (!is_file)) {

            // Convert paths to compiler convention
            std::string rpath_formatted(resolved_path);
            std::string qpath_formatted(qualified_path);
            std::string q_end(qualified_path.substr(qualified_path.size() - 2));
            if (strcmp(q_end.c_str(), "::") == 0)
                qpath_formatted = qualified_path.substr(0, qualified_path.size() - 2);
            std::string conv;
            qualified_path_to_resolved_path(qpath_formatted, conv);
            rpath_formatted += ":" + conv;
            std::string resolved_no_delimiter(qpath_formatted);
            qpath_formatted = "::" + qpath_formatted;

            // Termination criteria for package case: end of qualified path reached
            bool terminate = false;
            std::string archive_entry = std::string(full_q_path);
            std::size_t found = archive_entry.rfind(entry);
            mi::Uint32 p(found + entry.size());
            if ((archive_entry.size() <= p) && (std::string(extension).size() == 0))
                terminate = true;

            // Create new package
            mi::base::Handle<Mdl_package_info_impl> new_package(
                new Mdl_package_info_impl(
                    entry.c_str(),
                    search_path,
                    rpath_formatted.c_str(),
                    mi::Uint32(s_idx), 
                    qpath_formatted.c_str()));
            new_package->set_archive(0, true);
          
            // Check if the qualified path has changed in last recursion step
            std::size_t f = std::string(previous_module).find(resolved_no_delimiter);
            if ((f != std::string::npos) || (s_idx != 0)) {
                mi::Sint32 idx = parent->check_package(new_package.get());
                if (idx >= 0) {
                    // Reuse package
                    const Mdl_package_info_impl* mg(parent->get_package(idx));
                    mi::base::Handle<Mdl_package_info_impl> reuse_pkg(new Mdl_package_info_impl(*mg));
                    if (!is_known_search_path(search_path, reuse_pkg)) {
                        reuse_pkg->add_path(search_path);
                        reuse_pkg->add_path_index(s_idx);
                        reuse_pkg->add_resolved_path(rpath_formatted.c_str());
                        reuse_pkg->add_in_archive(true);
                    }
                    parent->reset_package(reuse_pkg.get(), idx);
                    if (!terminate)
                        discover_archive_recursive(
                            reuse_pkg,
                            previous_module,
                            full_q_path,
                            search_path,
                            resolved_path,
                            extension,
                            s_idx,
                            level + 1);
                }
            }
            else{
                // Add new package
                parent->add_package(new_package.get());
                if (!terminate)
                    discover_archive_recursive(
                        new_package, 
                        previous_module,
                        full_q_path, 
                        search_path, 
                        resolved_path, 
                        extension,
                        s_idx, 
                        level + 1);       
            }
    }
    return true;
}

bool Mdl_discovery_api_impl::read_archive(
    const char* res_path, 
    std::vector<std::string>& e_list,
    mi::Uint32 filter) const
{
    std::string full_path(res_path);
#ifdef MI_PLATFORM_WINDOWS
    std::string::size_type pos = full_path.find_last_of('\\');
#else
    std::string::size_type pos = full_path.find_last_of('/');
#endif
    std::string file = full_path.substr(pos + 1);
    if( file.find(".mdr") == std::string::npos)
        return false;

    mi::base::Handle<mi::mdl::IMDL> mdl(m_mdlc_module->get_mdl());
    mi::mdl::MDL_zip_container_error_code err = mi::mdl::MDL_zip_container_error_code::EC_OK;
    mi::mdl::MDL_zip_container_archive* zip_archive =
        mi::mdl::MDL_zip_container_archive::open(
            mdl->get_mdl_allocator(),
            full_path.c_str(), 
            err);
    if (!zip_archive)
        return false;

    std::vector<std::string>unhandled_packages;
    std::string ext;
    for (int i = 0; i < zip_archive->get_num_entries(); ++i) {
        std::string e = zip_archive->get_entry_name(i);

        size_t e_pos = e.find_last_of('.');
        bool valid_entry = false;
        bool is_filtered = true;

        if (e_pos != std::string::npos) {
            ext = e.substr(e_pos, e.size());
            if (is_mdl_file(ext)) {
                valid_entry = true;
                if (filter & mi::neuraylib::IMdl_info::Kind::DK_MODULE)
                    is_filtered = false;
            }
            else if (is_xlf_file(ext)) {
                valid_entry = true;
                if (filter & mi::neuraylib::IMdl_info::Kind::DK_XLIFF)
                    is_filtered = false;
            }
            else if (is_texture_file(ext)) {
                valid_entry = true;
                if (filter & mi::neuraylib::IMdl_info::Kind::DK_TEXTURE)
                    is_filtered = false;
            }
            else if (is_lightprofile_file(ext)) {
                valid_entry = true;
                if (filter & mi::neuraylib::IMdl_info::Kind::DK_LIGHTPROFILE)
                    is_filtered = false;
            }
            else if (is_bsdf_file(ext)) {
                valid_entry = true;
                if (filter & mi::neuraylib::IMdl_info::Kind::DK_MEASURED_BSDF)
                    is_filtered = false;
            }

            // Collect filtered paths only filter DK_PACKAGE is set 
            if (filter == mi::neuraylib::IMdl_info::Kind::DK_PACKAGE) {
#ifndef MI_PLATFORM_WINDOWS
                size_t s_pos = e.find_last_of('\\');
#else
                size_t s_pos = e.find_last_of('/');
#endif
                std::vector<std::string>::iterator it;
                if (s_pos != std::string::npos) {
                    std::string package_path = e.substr(0, s_pos);
                    it = std::find(unhandled_packages.begin(), unhandled_packages.end(), package_path);
                    if (it == unhandled_packages.end()) {
                        // Add an unhandled path 
                        if (is_filtered)
                            unhandled_packages.push_back(package_path);
                    }
                }
            }
            if ((valid_entry) && (!is_filtered)) {
                e_list.push_back(std::string(zip_archive->get_entry_name(i)));
                std::string res;
                replace_expression(
                    e_list[e_list.size() - 1],
                    "/",
                    "::",
                    res);
                e_list[e_list.size() - 1] = res;
            }
        }
    }

    // Special case when omly filter DK_PACKAGE is set 
    for ( auto p : unhandled_packages) {
        // Add unhandled_paths
        e_list.push_back(std::string(p));
        std::string res;
        replace_expression(
                e_list[e_list.size() - 1],
                "/",
                "::",
                res);
        e_list[e_list.size() - 1] = res;
    }

    zip_archive->close();
    return true;
}

bool Mdl_discovery_api_impl::discover_archive(
    mi::base::Handle<Mdl_package_info_impl> parent,
    const char* search_path, 
    mi::Size s_idx, 
    const char* res_path,
    mi::Uint32 filter) const
{
    std::vector<std::string> entry_list;
    if (!read_archive(res_path, entry_list, filter))
        return false;
    
    for (mi::Size x = 0; x < entry_list.size(); ++x) {
        mi::Size p = 0;
        while (entry_list[x][p] == ':')
            p++;
        std::string fqp = entry_list[x].substr(p);
        std::size_t dot = fqp.find_last_of('.');

        std::string extension("");
        if (dot != std::string::npos) {
            extension = (fqp.substr(fqp.find_last_of('.'), fqp.size()));
            fqp = fqp.substr(0, fqp.find_last_of('.'));
        }

        discover_archive_recursive(
            parent,
            x == 0 ? "" : entry_list[x - 1].c_str(), // previous module
            fqp.c_str(),
            search_path,
            res_path,
            extension.c_str(),
            s_idx,
            0);
    }
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

Mdl_discovery_result_impl::Mdl_discovery_result_impl(
    const Mdl_package_info_impl* graph,
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
