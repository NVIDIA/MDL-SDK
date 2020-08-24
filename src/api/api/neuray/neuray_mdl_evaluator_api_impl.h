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
 ** \brief Header for the IMdl_evaluator_api implementation.
 **/

#ifndef API_API_NEURAY_MDL_EVALUATOR_API_IMPL_H
#define API_API_NEURAY_MDL_EVALUATOR_API_IMPL_H

#include <mi/base/interface_implement.h>
#include <mi/neuraylib/imdl_evaluator_api.h>

#include <base/system/main/access_module.h>

namespace mi {
namespace neuraylib {
class INeuray;
class IFunction_call;
class IMaterial_instance;
class ITransaction;
class IValue_bool;
class IValue_factory;
}  // neuraylib
}  // mi

namespace MI {

namespace MDLC { class Mdlc_module; }

namespace NEURAY {

class Mdl_evaluator_api_impl final
  : public mi::base::Interface_implement<mi::neuraylib::IMdl_evaluator_api>
{
public:
    /// Constructor of Mdl_evaluator_api_impl.
    ///
    /// \param neuray      The neuray instance which contains this Mdl_evaluator_api_impl
    Mdl_evaluator_api_impl(mi::neuraylib::INeuray* neuray);

    /// Destructor of Mdl_evaluator_api_impl.
    virtual ~Mdl_evaluator_api_impl();

public:
    // public API methods

    /// Evaluates if a material instance parameter is enabled (i.e. the enable_if condition
    /// computes true).
    ///
    /// \param[in]  trans  the transaction
    /// \param[in]  fact   the expression factory to create the result value
    /// \param[in]  inst   the material instance
    /// \param[in]  index  the index of the material instance parameter
    /// \param[out] error  if non-NULL gets an error code if the evaluation fails
    ///
    /// \return NULL if the condition was to complex to evaluate, else true or false
    mi::neuraylib::IValue_bool const *is_material_parameter_enabled(
        mi::neuraylib::ITransaction             *trans,
        mi::neuraylib::IValue_factory           *fact,
        mi::neuraylib::IMaterial_instance const *inst,
        mi::Size                                 index,
        mi::Sint32                               *error) const final;

    /// Evaluates if a function call parameter is enabled (i.e. the enable_if condition
    /// computes true).
    ///
    /// \param[in]  trans  the transaction
    /// \param[in]  fact   the expression factory to create the result value
    /// \param[in]  call   the function call
    /// \param[in]  index  the index of the material instance parameter
    /// \param[out] error  if non-NULL gets an error code if the evaluation fails
    ///
    /// \return NULL if the condition was to complex to evaluate, else true or false
    mi::neuraylib::IValue_bool const *is_function_parameter_enabled(
        mi::neuraylib::ITransaction             *trans,
        mi::neuraylib::IValue_factory           *fact,
        mi::neuraylib::IFunction_call const     *call,
        mi::Size                                index,
        mi::Sint32                              *error) const final;

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
    // non copyable
    Mdl_evaluator_api_impl(Mdl_evaluator_api_impl const &) = delete;
    Mdl_evaluator_api_impl &operator=(Mdl_evaluator_api_impl const &) = delete;

private:
    mi::neuraylib::INeuray *m_neuray;

    SYSTEM::Access_module<MDLC::Mdlc_module> m_mdlc_module;
};

} // namespace NEURAY
} // namespace MI

#endif // API_API_NEURAY_MDL_EVALUATOR_API_IMPL_H
