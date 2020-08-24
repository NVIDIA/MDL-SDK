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
 ** \brief Header for the IMdl_discovery_definition implementation.
 **/

#ifndef API_API_NEURAY_MDL_DISCOVERY_IMPL_H
#define API_API_NEURAY_MDL_DISCOVERY_IMPL_H

#include <mi/neuraylib/imdl_discovery_api.h>
#include <mi/neuraylib/imdl_archive_api.h>
#include <mi/neuraylib/ineuray.h>
#include <mi/neuraylib/istring.h>
#include <mi/base/handle.h>
#include <mi/base/interface_implement.h>
#include <base/system/main/access_module.h>

#include <string>
#include <vector>
#include <unordered_map>
#include <boost/core/noncopyable.hpp>


namespace mi { namespace neuraylib { class INeuray; } }

namespace MI {

    namespace PATH { class Path_module; }
    namespace MDLC { class Mdlc_module; }
    namespace MDL { class Mdl_module; }

namespace NEURAY {

    class Mdl_discovery_result_impl;

/// This is the base class of all mdl discovery info data structures.
template<class T_info>
class Mdl_info_impl : public mi::base::Interface_implement<T_info>
{
    public:

        Mdl_info_impl() {};

        Mdl_info_impl(
            const std::string& simple_name,
            const std::string& qualified_name)
        : m_simple_name(simple_name)
        , m_qualified_name(qualified_name) {};

        virtual  ~Mdl_info_impl() {};

        /// Returns the simple_name.
        const char* get_simple_name() const final { return m_simple_name.c_str(); }

        /// Returns the qualified name.
        const char* get_qualified_name() const final { return m_qualified_name.c_str(); }

    protected:
        mi::base::Handle<const Mdl_discovery_result_impl> m_result;
        std::string m_simple_name;
        std::string m_qualified_name;
    };

/// This class implements the MDL data structure IMdl_Module_Info.
class Mdl_module_info_impl
    : public Mdl_info_impl<mi::neuraylib::IMdl_module_info>
{
    public:

        Mdl_module_info_impl(
            const std::string& simple_name,
            const std::string& qualified_name,
            const std::string& resolved_path,
            const std::string& search_path,
            mi::Size s_index,
            bool is_archive)
            : Mdl_info_impl(simple_name, qualified_name)
            , m_resolved_path(resolved_path)
            , m_search_path(search_path)
            , m_search_index(s_index)
            , m_in_archive(is_archive)
            {};

        ~Mdl_module_info_impl() {};

        void add_shadow(Mdl_module_info_impl* shadow);

        mi::neuraylib::IMdl_info::Kind get_kind() const final;

        const mi::IString* get_resolved_path() const final;

        mi::Size get_search_path_index() const final;

        const char* get_search_path() const final;

        mi::Size get_shadows_count() const final;

        const Mdl_module_info_impl* get_shadow(mi::Size index) const final;

        bool in_archive() const final;

        void set_archive(bool val);

        void sort_shadows();

    private:
        std::string             m_resolved_path;
        std::string             m_search_path;
        mi::Size                m_search_index;
        bool                    m_in_archive;
        std::vector<mi::base::Handle<Mdl_module_info_impl>>m_shadows;
};


/// This class implements the MDL data structure IMdl_xliff_info_impl.  
class Mdl_xliff_info_impl
    : public Mdl_info_impl<mi::neuraylib::IMdl_xliff_info>
{
public:

    Mdl_xliff_info_impl(
        const std::string& simple_name,
        const std::string& qualified_name,
        const std::string& resolved_path,
        const std::string& extension,
        const std::string& search_path,
        mi::Size s_index,
        bool is_archive)
        : Mdl_info_impl(simple_name, qualified_name)
        , m_resolved_path(resolved_path)
        , m_search_path(search_path)
        , m_extension(extension)
        , m_search_index(s_index)
        , m_in_archive(is_archive)
    {};

    ~Mdl_xliff_info_impl() {};

    const char* get_extension() const;

    mi::neuraylib::IMdl_info::Kind get_kind() const final;

    const char* get_resolved_path() const final;

    mi::Size get_search_path_index() const final;

    const char* get_search_path() const final;

    bool in_archive() const final;

    void set_archive(bool val);

private:
    std::string             m_resolved_path;
    std::string             m_search_path;
    std::string             m_extension;
    mi::Size                m_search_index;
    bool                    m_in_archive;
};


namespace
{
    template<class R>
    bool compare_resource_simple_names(const mi::base::Handle<const R> &hp1,
        const mi::base::Handle<const R> &hp2)
    {
        return 0 > strcmp(hp1->get_simple_name(), hp2->get_simple_name());
    }
};

template<class R_info>

/// This is the base class of all mdl discovery resource data structures.
class Mdl_resource_info_impl : public Mdl_info_impl<R_info>
{
public:

    Mdl_resource_info_impl(
        const std::string& simple_name,
        const std::string& qualified_name,
        const std::string& resolved_path,
        const std::string& extension,
        const std::string& search_path,
        mi::Size s_index,
        bool is_archive)
        : Mdl_info_impl<R_info>(simple_name, qualified_name)
        , m_resolved_path(resolved_path)
        , m_extension(extension)
        , m_search_path(search_path)
        , m_search_index(s_index)
        , m_in_archive(is_archive)
    {};

    virtual ~Mdl_resource_info_impl() {};

    void add_shadow(Mdl_resource_info_impl* shadow) {
        m_shadows.push_back(make_handle_dup(shadow));
    }

    const char* get_extension() const { return m_extension.c_str(); };

    const char* get_resolved_path() const final { return m_resolved_path.c_str(); }

    const char* get_search_path() const final { return m_search_path.c_str(); }

    mi::Size get_search_path_index() const final { return m_search_index; }

    mi::Size get_shadows_count() const final { return m_shadows.size(); }

    const Mdl_resource_info_impl* get_shadow(mi::Size index) const final {
        if (m_shadows.size() <= index)
            return nullptr;

        const mi::base::Handle<const Mdl_resource_info_impl>& shadow = m_shadows.at(index);
        shadow->retain();
        return shadow.get();
    }

    bool in_archive() const final { return m_in_archive; }

    void set_archive(bool val) { m_in_archive = val; }

    void sort_shadows() {
        std::sort(
            m_shadows.begin(),
            m_shadows.end(),
            compare_resource_simple_names<Mdl_resource_info_impl>);
    };

private:

    std::string             m_resolved_path;
    std::string             m_extension;
    std::string             m_search_path;
    mi::Size                m_search_index;
    bool                    m_in_archive;

    std::vector<mi::base::Handle<Mdl_resource_info_impl>>m_shadows;
};


/// This class implements the MDL data structure IMdl_texture_info.
class Mdl_texture_info_impl
    : public Mdl_resource_info_impl<mi::neuraylib::IMdl_texture_info>
{
public:

    Mdl_texture_info_impl(
        const std::string& simple_name,
        const std::string& qualified_name,
        const std::string& resolved_path,
        const std::string& extension,
        const std::string& search_path,
        mi::Size s_index,
        bool is_archive)
        : Mdl_resource_info_impl(
            simple_name,
            qualified_name,
            resolved_path,
            extension,
            search_path,
            s_index,
            is_archive) {};

    virtual ~Mdl_texture_info_impl() {};

    mi::neuraylib::IMdl_info::Kind get_kind() const final;
};

/// This class implements the MDL data structure IMdl_lightprofile_info.
class Mdl_lightprofile_info_impl
    : public Mdl_resource_info_impl<mi::neuraylib::IMdl_lightprofile_info>
{
public:

    Mdl_lightprofile_info_impl(
        const std::string& simple_name,
        const std::string& qualified_name,
        const std::string& resolved_path,
        const std::string& extension,
        const std::string& search_path,
        mi::Size s_index,
        bool is_archive)
        : Mdl_resource_info_impl(
            simple_name,
            qualified_name,
            resolved_path,
            extension,
            search_path,
            s_index,
            is_archive) {};

    virtual ~Mdl_lightprofile_info_impl() {};

    mi::neuraylib::IMdl_info::Kind get_kind() const final;
};

/// This class implements the MDL data structure IMdl_bsdf_info.
class Mdl_measured_bsdf_info_impl
    : public Mdl_resource_info_impl<mi::neuraylib::IMdl_measured_bsdf_info>
{
public:

    Mdl_measured_bsdf_info_impl(
        const std::string& simple_name,
        const std::string& qualified_name,
        const std::string& resolved_path,
        const std::string& extension,
        const std::string& search_path,
        mi::Size s_index,
        bool is_archive)
        : Mdl_resource_info_impl(
            simple_name,
            qualified_name,
            resolved_path,
            extension,
            search_path,
            s_index,
            is_archive) {};

        virtual ~Mdl_measured_bsdf_info_impl() {};

        mi::neuraylib::IMdl_info::Kind get_kind() const final;
};


/// This class implements the MDL graph data structure IMdl_package_info.
class Mdl_package_info_impl
    : public Mdl_info_impl<mi::neuraylib::IMdl_package_info>
{
    public:
        mi::neuraylib::IMdl_info::Kind get_kind() const final;

        Mdl_package_info_impl(
            const std::string& simple_name,
            const std::string& search_path,
            const std::string& resolved_path,
            mi::Sint32 p_idx,
            const std::string& qualified_name);

        ~Mdl_package_info_impl() {};

        void add_module(Mdl_module_info_impl* module);

        void add_resolved_path(const char* p);

        void add_xliff(Mdl_xliff_info_impl* resource);

        void add_texture(Mdl_texture_info_impl* resource);

        void add_lightprofile(Mdl_lightprofile_info_impl* resource);

        void add_measured_bsdf(Mdl_measured_bsdf_info_impl* resource);

        void add_package(Mdl_package_info_impl* child);

        void add_path(const char* p);

        void add_path_index(mi::Size s);

        void add_in_archive(bool value);

        // Checks if a package already exists in graph   
        // returns the index of the existing child or -1 if it does not exist
        mi::Sint32 check_package(Mdl_package_info_impl* child);

        const Mdl_package_info_impl* merge_packages(
            const Mdl_package_info_impl* p1,
            const Mdl_package_info_impl* p2);

        void reset_package(Mdl_package_info_impl* merge_child, int index);

        // Checks if one ore more modules already exist in a certain package.
        // - 0 if the module has been shadowed and added to the shadow list
        // - 1 if no shadowing has happened
        mi::Sint32 shadow_module(Mdl_module_info_impl* new_module);

        // Checks if one ore more textures already exist in a certain package.
        // - 0 if the texture has been shadowed and added to the shadow list
        // - 1 if no shadowing has happened
        mi::Sint32 shadow_texture(Mdl_texture_info_impl* new_texture);

        // Checks if one ore more lightprofiles already exist in a certain package.
        // - 0 if the lightprofiles has been shadowed and added to the shadow list
        // - 1 if no shadowing has happened
        mi::Sint32 shadow_lightprofile(Mdl_lightprofile_info_impl* new_lightprofile);

        // Checks if one ore more measured bsdfs already exist in a certain package.
        // - 0 if the measured bsdf has been shadowed and added to the shadow list
        // - 1 if no shadowing has happened
        mi::Sint32 shadow_measured_bsdf(Mdl_measured_bsdf_info_impl* new_measured_bsdf);

        void set_archive(Size index, bool val);

        void set_kind(mi::neuraylib::IMdl_info::Kind k);

        // Sorts modules, child packages and resource list alphabetically.
        void sort_children();

        // public API methods

        /// Returns the number of modules inherited.
        mi::Size get_module_count() const;

        /// Returns the qualified name of a module referred by index.
        const Mdl_module_info_impl* get_module(mi::Size index) const;

        /// Returns the number of xliff files contained.
        mi::Size get_xliff_count() const;

        /// Returns the resolved path of a xliff files referred by index.
        const Mdl_xliff_info_impl* get_xliff(mi::Size index) const;

        /// Returns the number of textures files contained.
        mi::Size get_texture_count() const;

        /// Returns the resolved path of a texture files referred by index.
        const Mdl_texture_info_impl* get_texture(mi::Size index) const;

        /// Returns the number of lightprofile files contained.
        mi::Size get_lightprofile_count() const;

        /// Returns the resolved path of a lightprofile files referred by index.
        const Mdl_lightprofile_info_impl* get_lightprofile(mi::Size index) const;

        /// Returns the number of measured bsdf files contained.
        mi::Size get_measured_bsdf_count() const;

        /// Returns the resolved path of a measured bsdf files referred by index.
        const Mdl_measured_bsdf_info_impl* get_measured_bsdf(mi::Size index) const;

        /// Returns the number of deriving graph nodes.
         mi::Size get_package_count() const;

        /// Returns a deriving package referenced by index.
        const Mdl_package_info_impl* get_package(mi::Size index) const;

        /// Returns the number of modules and sub-packages contained in this package.
        mi::Size get_child_count() const final;

        /// Returns a child of this package.
        /// index \in [0, max(0, get_child_count()-1)]
        const IMdl_info* get_child(mi::Size index) const final;

        /// Returns an absolute path to the package in the file system.
        const mi::IString* get_resolved_path(mi::Size index) const final;

        /// Returns a search path referenced by index.
        const char* get_search_path(mi::Size index) const final;

        /// Returns the number of search paths.
        mi::Size get_search_path_index_count() const final;

        /// Returns the index of search paths.
        mi::Size get_search_path_index(mi::Size idx) const final;

        /// Returns true if the package has been discovered inside of an archive, false if not.
        bool in_archive(Size index) const final;

    private:
        std::vector<std::string> m_paths;
        std::vector<std::string> m_resolved_paths;
        std::vector<bool> m_in_archive;
        mi::neuraylib::IMdl_info::Kind m_kind;
        std::vector<mi::Size> m_path_idx;
        std::vector<mi::base::Handle<Mdl_module_info_impl>> m_modules;
        std::vector<mi::base::Handle<Mdl_package_info_impl>> m_packages;
        std::vector<mi::base::Handle<Mdl_xliff_info_impl>> m_xliffs;
        std::vector<mi::base::Handle<Mdl_texture_info_impl>> m_textures;
        std::vector<mi::base::Handle<Mdl_lightprofile_info_impl>> m_lightprofiles;
        std::vector<mi::base::Handle<Mdl_measured_bsdf_info_impl>> m_measured_bsdfs;
};


/// This class implements features to discover MDL content.
class Mdl_discovery_api_impl
    : public mi::base::Interface_implement< mi::neuraylib::IMdl_discovery_api>,
    public boost::noncopyable
{
    public:
        Mdl_discovery_api_impl(mi::neuraylib::INeuray* neuray);

        ~Mdl_discovery_api_impl();

        const mi::neuraylib::IMdl_discovery_result* discover(mi::Uint32 filter) const final;

        mi::Sint32 start();

        mi::Sint32 shutdown();

    private:

        // Checks if a graph item name is a valid MDL module or package name.
        bool is_valid_node_name(const char* identifier) const;

        // Checks if a search path exists already.
        bool is_known_search_path(
            const char* path,
            mi::base::Handle<Mdl_package_info_impl> package) const;

        // Assigns modules from the current archive recursion level.
        bool add_archive_entries(
            mi::base::Handle<Mdl_package_info_impl> parent, 
            mi::Size s_idx,
            const char* sp,
            const char* fqp,
            const char* rp,
            std::string& qualified_path,
            const char* extension,
            std::string& entry,
            mi::Size level) const;

        // Reads an archive and adds all modules and resource entries to e_list.
        bool read_archive(
            const char* res_path, 
            std::vector<std::string>& e_list,
            mi::Uint32 filter) const;

        // Creates a graph structure out of an mdl archive file.
        bool discover_archive(
            mi::base::Handle<Mdl_package_info_impl> parent,
            const char* search_path,
            mi::Size search_idx,
            const char* dir,
            mi::Uint32 filter) const;

        // Direct recursion to create a graph out of an archive entry.
        bool discover_archive_recursive(
            mi::base::Handle<Mdl_package_info_impl> parent,
            const char* previous_module,
            const char* q_path,
            const char* search_path,
            const char* res_path,
            const char* extension,
            mi::Size s_idx,
            mi::Size level) const;

        // Direct recursion to create a graph out of a folder from a file system.
        bool discover_filesystem_recursive(
            mi::base::Handle<Mdl_package_info_impl> parent,
            const char* search_path,
            mi::Size search_idx,
            const char* dir,
            const std::vector<std::string>& invalid_dirs,
            mi::Uint32 filter) const;

        mi::neuraylib::INeuray*                          m_neuray;
        MI::SYSTEM::Access_module<MI::MDLC::Mdlc_module> m_mdlc_module;
        MI::SYSTEM::Access_module<PATH::Path_module> m_path_module;
};

/// This class implements the discover result.
class Mdl_discovery_result_impl
    : public mi::base::Interface_implement< mi::neuraylib::IMdl_discovery_result>
{
    public:
        Mdl_discovery_result_impl(
            const Mdl_package_info_impl* graph,
            const std::vector<std::string>);
        ~Mdl_discovery_result_impl() {};

        /// Returns a pointer to the root of mdl graph.
        const mi::neuraylib::IMdl_package_info* get_graph() const final;

        /// Returns a search path referenced by index.
        const char* get_search_path(mi::Size index) const final;

        /// Returns the number of search paths.
        mi::Size get_search_paths_count() const final;

    private:
        mi::base::Handle<const mi::neuraylib::IMdl_package_info> m_graph;
        std::vector<std::string> m_search_paths;
};

} // namespace NEURAY

} // namespace MI

#endif // API_API_NEURAY_MDL_DISCOVERY_IMPL_H
