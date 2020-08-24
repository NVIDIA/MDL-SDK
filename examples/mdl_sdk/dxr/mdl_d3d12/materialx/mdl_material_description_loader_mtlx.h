/******************************************************************************
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

 // examples/mdl_sdk/dxr/mdl_d3d12/materialx/mdl_material_description_loader_mtlx.h

#ifndef MATERIALX_MDL_MATERIAL_DESCRIPTION_LOADER_MTLX_H
#define MATERIALX_MDL_MATERIAL_DESCRIPTION_LOADER_MTLX_H

#include "../common.h"
#include "../mdl_material_description.h"

namespace mi {namespace examples { namespace mdl_d3d12 { namespace materialx
{
    /// loader to generate MDL from MaterialX
    /// see IMdl_material_description_loader for documentation
    class Mdl_material_description_loader_mtlx : public IMdl_material_description_loader
    {
    public:
        bool match_gltf_name(const std::string& gltf_name) const final;

        std::string generate_mdl_source_code(
            const std::string& gltf_name,
            const std::string& scene_directory) const final;

        std::string get_scene_name_prefix() const final { return "[MTLX]"; }

        bool supports_reload() const final { return true; }

        size_t get_file_type_count() const final;
        std::string get_file_type_extension(size_t index) const final;
        std::string get_file_type_description(size_t index) const final;
    };

}}}}
#endif
