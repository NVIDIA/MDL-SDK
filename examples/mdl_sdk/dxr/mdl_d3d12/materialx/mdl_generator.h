/******************************************************************************
 * Copyright (c) 2020-2025, NVIDIA CORPORATION. All rights reserved.
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

 // examples/mdl_sdk/dxr/mdl_d3d12/materialx/mdl_generator.h

#ifndef MATERIALX_MDL_GENERATOR_H
#define MATERIALX_MDL_GENERATOR_H

#include "../common.h"

namespace mi {namespace examples { namespace mdl_d3d12 {

class Mdl_sdk;

namespace materialx
{
    class Mdl_generator_result
    {
    public:
        std::string materialx_file_name;
        std::string generated_mdl_code;
        std::string materialx_material_name;
        std::string generated_mdl_name;
    };

    // --------------------------------------------------------------------------------------------

    /// Basic MDL code-gen from MaterialX.
    /// A more sophisticated supports will be needed for full function support.
    class Mdl_generator
    {
    public:
        /// Constructor.
        explicit Mdl_generator();

        /// Destructor.
        ~Mdl_generator() = default;

        /// Specify an additional absolute search path location (e.g. '/projects/MaterialX').
        /// This path will be queried when locating standard data libraries,
        /// XInclude references, and referenced images.
        void add_path(const std::string& mtlx_path);

        /// Specify an additional relative path to a custom data library folder
        /// (e.g. 'libraries/custom'). MaterialX files at the root of this folder will be included
        /// in all content documents.
        void add_library(const std::string& mtlx_library);

        /// Specify the MDL language version the code generator should produce.
        void set_mdl_version(const std::string& mdl_version);

        /// The MaterialXTest applications needs some special setup to align texture coordinates.
        /// Do NOT use this mode in any other environment since this would indicate inconsistent inputs.
        void set_materialxtest_mode(bool enabled);

        /// set the main mtlx file of the material to generate code from.
        bool set_source(const std::string& mtlx_material, const std::string& material_name);

        /// generate mdl code
        ///       - handle relative resources
        bool generate(Mdl_sdk& sdk, Mdl_generator_result& inout_result) const;

    private:
        std::vector<std::string> m_mtlx_search_paths;
        std::vector<std::string> m_mtlx_relative_library_paths;
        std::string m_mtlx_source;
        std::string m_mtlx_material_name;
        std::string m_mdl_version;
        bool m_materialxtest_mode;
    };

}}}}
#endif
