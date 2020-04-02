/***************************************************************************************************
 * Copyright (c) 2012-2020, NVIDIA CORPORATION. All rights reserved.
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

/// \file
/// \brief Factory for the MDL SDK library.

#include "pch.h"

#include <mi/neuraylib/factory.h>

#include <api/api/mdl/mdl_neuray_impl.h>
#include <api/api/neuray/neuray_version_impl.h>

extern "C"
MI_DLL_EXPORT
mi::neuraylib::INeuray* mi_neuray_factory_deprecated(
    mi::neuraylib::IAllocator* allocator, mi::Uint32 version)
{
    if( version != MI_NEURAYLIB_API_VERSION)
        return 0;

    // Reject user-supplied allocators which are not supported.
    if( allocator)
        return 0;

    if( ++MI::MDL::Neuray_impl::s_instance_count != 1) {
        --MI::MDL::Neuray_impl::s_instance_count;
         return 0;
    }

    return new MI::MDL::Neuray_impl();
}

extern "C"
MI_DLL_EXPORT
mi::base::IInterface* mi_factory(
    const mi::base::Uuid& iid)
{
    switch ( uuid_hash32( iid))
    {
    case mi::neuraylib::INeuray::IID::hash32:
        if( ++MI::MDL::Neuray_impl::s_instance_count != 1) {
            --MI::MDL::Neuray_impl::s_instance_count;
            return 0;
        }
        return new MI::MDL::Neuray_impl();
    case mi::neuraylib::IVersion::IID::hash32:
        return new MI::NEURAY::Version_impl();
    default:
        break;
    }
    return 0;
}
