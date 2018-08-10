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
        
        Mdl_info_impl(const Mdl_info_impl &obj) : 
            mi::base::Interface_implement<T_info>(obj)
        {
            m_simple_name = std::string(obj.get_simple_name());
            m_qualified_name = std::string(obj.get_qualified_name());
            m_result = NULL;
        }

        ~Mdl_info_impl() {};

        /// Returns the simple_name.
        const char* get_simple_name() const { return m_simple_name.c_str(); }

        /// Returns the qualified name.
        const char* get_qualified_name() const { return m_qualified_name.c_str(); }

    protected:
        mi::base::Handle<const Mdl_discovery_result_impl> m_result;
        std::string m_qualified_name;
        std::string m_simple_name;

    };

/// This class implements the MDL data structure IMdl_Module_Info.  
class Mdl_module_info_impl
    : public Mdl_info_impl<mi::neuraylib::IMdl_module_info>
{
    public:

        mi::neuraylib::IMdl_info::Kind get_kind() const 
                { return mi::neuraylib::IMdl_info::Kind::DK_MODULE; }

        Mdl_module_info_impl(const char* name,
            const char* q_name,
            const char* r_path,
            const char* s_path,
            mi::Size s_index);

        ~Mdl_module_info_impl();

        void add_shadow(Mdl_module_info_impl* shadow);

        const mi::IString* get_resolved_path() const;

        mi::Size get_search_path_index() const;

        const char* get_search_path() const;

        mi::Size get_shadows_count() const;

        const Mdl_module_info_impl* get_shadow(mi::Size index) const;

        bool in_archive() const;

        void set_archive(bool val);

        void sort_shadows();

    private:

        bool successfully_constructed() { return m_successfully_constructed; }

        bool                    m_in_archive;
        std::string             m_resolved_path;
        mi::Size                m_search_index;
        std::string             m_search_path;
        std::vector<mi::base::Handle<Mdl_module_info_impl>>m_shadows;

        /// Indicates whether the constructor successfully constructed the instance.
        /// \see #successfully_constructed()
        bool m_successfully_constructed;
};

/// This class implements the MDL graph data structure IMdl_package_info 
class Mdl_package_info_impl
    : public Mdl_info_impl<mi::neuraylib::IMdl_package_info>
{
    public:
        mi::neuraylib::IMdl_info::Kind get_kind() const override
            { return mi::neuraylib::IMdl_info::Kind::DK_PACKAGE; }

        Mdl_package_info_impl(const char* name, 
            const char* s_path,
            const char* r_path,
            mi::Sint32 p_idx,
            const char* q_name);

        ~Mdl_package_info_impl();

        void add_module(Mdl_module_info_impl* module);

        void add_resolved_path(const char* p);

        void add_package(Mdl_package_info_impl* child);

        void add_path(const char* p);

        void add_path_index(mi::Size s);

        void add_in_archive(bool value);

        mi::Sint32 check_package(Mdl_package_info_impl* child);

        const Mdl_package_info_impl* merge_packages(const Mdl_package_info_impl* p1,
            const Mdl_package_info_impl* p2);

        void reset_package(Mdl_package_info_impl* merge_child, int index);

        mi::Sint32 shadow_module(Mdl_module_info_impl* new_module);

        void set_archive(Size index, bool val);

        void sort_children();

        // public API methods

        /// Returns the number of modules inherited.
        mi::Size get_module_count() const;

        /// Returns the qualified name of a module referred by index.
        const Mdl_module_info_impl* get_module(mi::Size index) const;

        /// Returns the number of deriving graph nodes.
         mi::Size get_package_count() const;

        /// Returns a deriving package referenced by index.
        const Mdl_package_info_impl* get_package(mi::Size index) const;

        /// Returns the number of modules and sub-packages contained in this package.
        mi::Size get_child_count() const override;

        /// Returns a child of this package.
        /// index \in [0, max(0, get_child_count()-1)]
        const IMdl_info* get_child(mi::Size index) const override;

        /// Returns an absolute path to the package in the file system
        const mi::IString* get_resolved_path(mi::Size index) const override;

        /// Returns a search path referenced by index.
        const char* get_search_path(mi::Size index) const override;

        /// Returns the number of search paths.
        mi::Size get_search_path_index_count() const override;

        /// Returns the index of search paths.
        mi::Size get_search_path_index(mi::Size idx) const override;

        /// Returns true if the package has been discovered inside of an archive, false if not.
        bool in_archive(Size index) const override;

    private:

        bool successfully_constructed() { return m_successfully_constructed; }

        std::vector<std::string> m_paths;
        std::vector<std::string> m_resolved_paths;
        std::vector<bool> m_in_archive;
        std::vector<mi::Size> m_path_idx;
        std::vector<mi::base::Handle<Mdl_package_info_impl>> m_packages;
        std::vector<mi::base::Handle<Mdl_module_info_impl>> m_modules;

        /// Indicates whether the constructor successfully constructed the instance.
        /// \see #successfully_constructed()
        bool m_successfully_constructed;
};


/// This class implements features to discover MDL content.
class Mdl_discovery_api_impl
    : public mi::base::Interface_implement< mi::neuraylib::IMdl_discovery_api>,
    public boost::noncopyable
{
    public:

    /// \param neuray      The neuray instance which contains this Mdl_discovery_api_impl
    Mdl_discovery_api_impl(mi::neuraylib::INeuray* neuray); 

    /// Destructor of Mdl_discovery_api_impl
    ~Mdl_discovery_api_impl();

    /// Returns a pointer to the discovery result.
    const mi::neuraylib::IMdl_discovery_result* discover() const;

    /// Starts this API component.
    ///
    /// \return            0, in case of success, -1 in case of failure.
    mi::Sint32 start();

    /// Shuts down this API component.
    ///
    /// \return           0, in case of success, -1 in case of failure
    mi::Sint32 shutdown();

    private:

        /// Checks if a graph item name is a valid MDL identifier.
        bool check_ident_validity(const char* identifier) const;

        bool create_archive_graph(mi::base::Handle<const mi::neuraylib::IManifest> m,
            mi::base::Handle<Mdl_package_info_impl> parent,
            const char* search_path,
            mi::Size s_idx,
            const char* res_path) const;

        bool create_archive_graph_recursive(mi::base::Handle<Mdl_package_info_impl> parent, 
            const char* previous_module,
            const char* q_path, 
            const char* search_path,
            const char* res_path,
            mi::Size s_idx,
            mi::Size level) const;

        bool discover_archive(mi::base::Handle<Mdl_package_info_impl> parent,
            const char* search_path,
            mi::Size search_idx,
            const char* dir) const;

        bool discover_recursive(mi::base::Handle<Mdl_package_info_impl> parent, 
            const char* search_path,
            mi::Size search_idx,
            const char* dir,
            const std::vector<std::string>& invalid_dirs) const;

        mi::neuraylib::INeuray*                          m_neuray;
        MI::SYSTEM::Access_module<MI::MDLC::Mdlc_module> m_mdlc_module;
        MI::SYSTEM::Access_module<MI::PATH::Path_module> m_path_module;
};

/// This class implements the discover result.  
class Mdl_discovery_result_impl
    : public mi::base::Interface_implement< mi::neuraylib::IMdl_discovery_result>
{
    public:
        Mdl_discovery_result_impl(const Mdl_package_info_impl* graph,
            const std::vector<std::string>);
        ~Mdl_discovery_result_impl() {};

        /// Returns a pointer to the root of mdl graph
        const mi::neuraylib::IMdl_package_info* get_graph() const;

        /// Returns a search path referenced by index.
        const char* get_search_path(mi::Size index) const;

        /// Returns the number of search paths.
        mi::Size get_search_paths_count() const;

    private:
        mi::base::Handle<const mi::neuraylib::IMdl_package_info> m_graph;
        std::vector<std::string> m_search_paths;
};

} // namespace NEURAY

} // namespace MI

#endif // API_API_NEURAY_MDL_DISCOVERY_IMPL_H
