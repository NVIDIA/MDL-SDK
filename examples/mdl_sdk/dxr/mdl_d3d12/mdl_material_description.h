/******************************************************************************
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

 // examples/mdl_sdk/dxr/mdl_d3d12/mdl_material_description.h

#ifndef MDL_D3D12_MDL_MATERIAL_DESCRIPTION_H
#define MDL_D3D12_MDL_MATERIAL_DESCRIPTION_H

#include "common.h"
#include "scene.h"
#include <mi/base.h>

namespace mi
{
    namespace neuraylib
    {
        class IExpression_list;
        class IMdl_execution_context;
    }
}

namespace mdl_d3d12
{
    class Base_application;

    /// Description of a mdl material instance to create.
    class Mdl_material_description final
    {
    public:
        /// Constructor.
        explicit Mdl_material_description(
            Base_application* app,
            IScene_loader::Material description);

        /// Destructor.
        ~Mdl_material_description() = default;

        /// Get the mdl parameter list for this material instance. This can contain complex graphs.
        /// When this function is called by the material library, this materials parenting module
        /// is already loaded, so it is valid to use exported types and types from imported modules.
        /// All other types present in the database are also valid.
        const mi::neuraylib::IExpression_list* get_parameters() const;

        /// Get the (qualified) module name. 
        /// This is expected to be of the form: [::<package>]::<module>
        const std::string& get_qualified_module_name() const { return m_qualified_module_name; }

        /// Get the (simple) material name. 
        /// This is only <material>-part of a qualified material name 
        /// of the form [::<package>]::<module>::<material>
        const std::string& get_material_name() const { return m_material_name; }

        // get material flags e.g. for optimization
        IMaterial::Flags get_flags() const { return m_flags; }

    private:
        // helper function to create mdl parameters from the material description data
        const mi::neuraylib::IExpression_list* parameterize_support_material() const;

        Base_application* m_app;
        IScene_loader::Material m_description;
        std::string m_qualified_module_name;
        std::string m_material_name;
        IMaterial::Flags m_flags;
    };
}

#endif
