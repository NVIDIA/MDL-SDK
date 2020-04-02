/******************************************************************************
 * Copyright (c) 2013-2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDL_GENERATOR_JIT_RES_MANAGER_H
#define MDL_GENERATOR_JIT_RES_MANAGER_H 1

#include <mi/mdl/mdl_types.h>
#include <mi/mdl/mdl_values.h>

namespace mi {
namespace mdl {

class IValue_resource;

/// An Interface to handle resources in JIT compiled code.
class IResource_manager {
public:
    /// Returns the resource index for the given resource usable by the target code resource
    /// handler for the corresponding resource type.
    ///
    /// \param kind        the resource kind
    /// \param url         the resource url (might be NULL)
    /// \param tag         the resource tag (if assigned)
    /// \param shape       if the resource is a texture: its shape
    /// \param gamma_mode  if the resource is a texture: its gamma mode
    ///
    /// \returns a resource index or 0 if no resource index can be returned
    virtual size_t get_resource_index(
        Resource_tag_tuple::Kind   kind,
        char const                 *url,
        int                        tag,
        IType_texture::Shape       shape = IType_texture::TS_2D,
        IValue_texture::gamma_mode gamma_mode = IValue_texture::gamma_default) = 0;

    /// Register a string constant and return its 1 based index in the string table.
    ///
    /// \param string  the MDL string value to register
    virtual size_t get_string_index(IValue_string const *string) = 0;
};

}  // mdl
}  // mi

#endif // MDL_GENERATOR_JIT_RES_MANAGER_H
