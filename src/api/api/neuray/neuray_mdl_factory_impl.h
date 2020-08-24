/***************************************************************************************************
 * Copyright (c) 2014-2020, NVIDIA CORPORATION. All rights reserved.
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
 ** \brief Header for the IMdl_factory implementation.
 **/

#ifndef API_API_NEURAY_MDL_FACTORY_IMPL_H
#define API_API_NEURAY_MDL_FACTORY_IMPL_H

#include <mi/base/interface_implement.h>
#include <mi/neuraylib/imdl_factory.h>

#include <boost/core/noncopyable.hpp>

namespace mi { namespace neuraylib { class INeuray; class IMdl_execution_context;} }

namespace MI {

namespace NEURAY {

class Class_factory;

class Mdl_factory_impl
  : public mi::base::Interface_implement<mi::neuraylib::IMdl_factory>,
    public boost::noncopyable
{
public:
    /// Constructor of Mdl_factory_impl
    ///
    /// \param neuray      The neuray instance which contains this Mdl_factory_impl
    Mdl_factory_impl( mi::neuraylib::INeuray* neuray, const Class_factory* class_factory);

    /// Destructor of Library_authentication_impl
    ~Mdl_factory_impl();


    // public API methods

    mi::neuraylib::IType_factory* create_type_factory(
        mi::neuraylib::ITransaction* transaction) final;

    mi::neuraylib::IValue_factory* create_value_factory(
        mi::neuraylib::ITransaction* transaction) final;

    mi::neuraylib::IExpression_factory* create_expression_factory(
        mi::neuraylib::ITransaction* transaction) final;

    mi::Sint32 create_variants(
        mi::neuraylib::ITransaction* transaction,
        const char* module_name,
        const mi::IArray* variant_data) final;

    mi::Sint32 create_materials(
        mi::neuraylib::ITransaction* transaction,
        const char* module_name,
        const mi::IArray* material_data) final;

    mi::Sint32 create_materials(
        mi::neuraylib::ITransaction* transaction,
        const char* module_name,
        const mi::IArray* mdl_data,
        mi::neuraylib::IMdl_execution_context *context) final;

    mi::neuraylib::IValue_texture* create_texture(
        mi::neuraylib::ITransaction* transaction,
        const char* file_path,
        mi::neuraylib::IType_texture::Shape shape,
        mi::Float32 gamma,
        bool shared,
        mi::Sint32* errors) final;

    mi::neuraylib::IValue_light_profile* create_light_profile(
        mi::neuraylib::ITransaction* transaction,
        const char* file_path,
        bool shared,
        mi::Sint32* errors) final;

    mi::neuraylib::IValue_bsdf_measurement* create_bsdf_measurement(
        mi::neuraylib::ITransaction* transaction,
        const char* file_path,
        bool shared,
        mi::Sint32* errors) final;

    mi::neuraylib::IMdl_execution_context* create_execution_context() final;
    
    mi::neuraylib::IMdl_module_transformer* create_module_transformer(
        mi::neuraylib::ITransaction* transaction,
        const char* module_name,
        mi::neuraylib::IMdl_execution_context* context) final;
    
    const mi::IString* get_db_module_name( const char* mdl_name) final;

    const mi::IString* get_db_definition_name( const char* mdl_name) final;

    // internal methods

private:
    mi::neuraylib::INeuray* m_neuray;
    const Class_factory* m_class_factory;
};

} // namespace NEURAY

} // namespace MI

#endif // API_API_NEURAY_MDL_FACTORY_IMPL_H
