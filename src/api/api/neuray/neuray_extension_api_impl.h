/***************************************************************************************************
 * Copyright (c) 2010-2026, NVIDIA CORPORATION. All rights reserved.
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
 ** \brief Header for the IExtension_api implementation.
 **/

#ifndef API_API_NEURAY_NEURAY_EXTENSION_API_IMPL_H
#define API_API_NEURAY_NEURAY_EXTENSION_API_IMPL_H

#include <mi/base/interface_implement.h>
#include <mi/neuraylib/iextension_api.h>

#include <boost/core/noncopyable.hpp>

namespace mi { namespace neuraylib { class INeuray; } }

namespace MI {

namespace NEURAY {

class Class_factory;

class Extension_api_impl
  : public mi::base::Interface_implement<mi::neuraylib::IExtension_api>,
    public boost::noncopyable
{
public:
    /// Constructor of Extension_api_impl
    Extension_api_impl( mi::neuraylib::INeuray* neuray, Class_factory* class_factory);

    /// Destructor of Extension_api_impl
    ~Extension_api_impl();

    // public API methods

    mi::Sint32 register_structure_decl(
        const char* structure_name, const mi::IStructure_decl* decl);

    mi::Sint32 unregister_structure_decl( const char* structure_name);

    mi::Sint32 register_enum_decl(
        const char* enum_name, const mi::IEnum_decl* decl);

    mi::Sint32 unregister_enum_decl( const char* enum_name);

    mi::Sint32 register_class(
        const char* class_name, mi::base::Uuid uuid, mi::neuraylib::IUser_class_factory* factory);

    mi::Sint32 unregister_class( const char* class_name, mi::base::Uuid uuid);

    mi::Sint32 register_importer( mi::neuraylib::IImporter* importer);

    mi::Sint32 register_exporter( mi::neuraylib::IExporter* exporter);

    mi::Sint32 unregister_importer( mi::neuraylib::IImporter* importer);

    mi::Sint32 unregister_exporter( mi::neuraylib::IExporter* exporter);

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
    mi::neuraylib::INeuray* m_neuray;               ///< Owning neuray instance.
    Class_factory* m_class_factory;                 ///< Class factory.
};

} // namespace NEURAY

} // namespace MI

#endif // API_API_NEURAY_NEURAY_EXTENSION_API_IMPL_H
