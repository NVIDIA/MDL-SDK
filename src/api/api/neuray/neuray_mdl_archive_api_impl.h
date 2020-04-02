/***************************************************************************************************
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
 **************************************************************************************************/

/** \file
 ** \brief Header for the IMdl_archive_api implementation.
 **/

#ifndef API_API_NEURAY_MDL_ARCHIVE_API_IMPL_H
#define API_API_NEURAY_MDL_ARCHIVE_API_IMPL_H

#include <mi/base/interface_implement.h>
#include <mi/mdl/mdl_archiver.h>
#include <mi/neuraylib/imdl_archive_api.h>

#include <map>
#include <string>
#include <utility>
#include <vector>
#include <boost/core/noncopyable.hpp>
#include <base/system/main/access_module.h>

namespace mi { namespace neuraylib { class INeuray; } }

namespace MI {

namespace MDLC { class Mdlc_module; }

namespace NEURAY {

class Mdl_archive_api_impl final
  : public mi::base::Interface_implement<mi::neuraylib::IMdl_archive_api>,
    public boost::noncopyable
{
public:
    /// Constructor of Mdl_archive_api_impl
    ///
    /// \param neuray      The neuray instance which contains this Mdl_archive_api_impl
    Mdl_archive_api_impl( mi::neuraylib::INeuray* neuray);

    /// Destructor of Library_authentication_impl
    ~Mdl_archive_api_impl();

    // public API methods

    mi::Sint32 create_archive(
        const char* directory,
        const char* archive,
        const mi::IArray* manifest_fields) final;

    mi::Sint32 extract_archive(
        const char* archive,
        const char* directory)  final;

    const mi::neuraylib::IManifest* get_manifest(
        const char* archive) final;

    mi::neuraylib::IReader* get_file(
        const char* archive,
        const char* filename) final;

    mi::neuraylib::IReader* get_file(
        const char* filename) final;

    mi::Sint32 set_extensions_for_compression(
        const char* extensions) final;

    const char* get_extensions_for_compression() const final;

    // internal methods

    /// Starts this API component.
    ///
    /// The implementation of INeuray::start() calls the #start() method of each API component.
    /// This method performs the API component's specific part of the library start.
    ///
    /// \return            0, in case of success, -1 in case of failure.
    mi::Sint32 start();

    /// Shuts down this API component.
    ///
    /// The implementation of INeuray::shutdown() calls the #shutdown() method of each API
    /// component. This method performs the API component's specific part of the library shutdown.
    ///
    /// \return           0, in case of success, -1 in case of failure
    mi::Sint32 shutdown();

private:
    mi::neuraylib::INeuray* m_neuray;

    /// Access to the MDLC module
    SYSTEM::Access_module<MDLC::Mdlc_module> m_mdlc_module;

    /// Comma-separated list of file name extensions.
    std::string m_compression_extensions;
};

class Manifest_impl
  : public mi::base::Interface_implement<mi::neuraylib::IManifest>,
    public boost::noncopyable
{
public:
    /// Constructor.
    ///
    /// Populates #m_fields and #m_index_count from \p manifest.
    Manifest_impl( const mi::mdl::IArchive_manifest* manifest);

    // methods of mi::neuraylib::IManifest

    mi::Size get_number_of_fields() const final;

    const char* get_key( mi::Size index) const final;

    const char* get_value( mi::Size index) const final;

    mi::Size get_number_of_fields( const char* key) const final;

    const char* get_value( const char* key, mi::Size index) const final;

private:

    /// Populates #m_fields and #m_index_count from \p manifest for all fields with key \p key.
    void convert_exports(
        const mi::mdl::IArchive_manifest* manifest, mi::mdl::IArchive_manifest::Predefined_key key);

    /// Returns a string representation of mi::mdl::IMDL::MDL_version.
    static const char* convert_mdl_version( mi::mdl::IMDL::MDL_version version);

    /// Returns a string representation of mi::mdl::ISemantic_version.
    static std::string convert_sema_version( const mi::mdl::ISemantic_version* version);

    /// All fields of the manifest as list of (key, value) pairs. Some keys may occur multiple
    /// times. In such a case they appear in a continuous section in this list.
    std::vector<std::pair<std::string, std::string> > m_fields;

    typedef std::map<std::string, std::pair<size_t, size_t> > Index_count_map;

    /// Stores for each key the (first) index into m_fields and the number of such keys.
    Index_count_map m_index_count;
};

} // namespace NEURAY

} // namespace MI

#endif // API_API_NEURAY_MDL_ARCHIVE_API_IMPL_H
