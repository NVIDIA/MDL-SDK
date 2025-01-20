/***************************************************************************************************
 * Copyright (c) 2018-2025, NVIDIA CORPORATION. All rights reserved.
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
/// \brief      API component that gives access to the MDL evaluator.

#ifndef MI_NEURAYLIB_IMDL_EVALUATOR_H
#define MI_NEURAYLIB_IMDL_EVALUATOR_H

#include <mi/base/interface_declare.h>

namespace mi {
namespace neuraylib {

class IFunction_call;
class ITransaction;
class IValue_bool;
class IValue_factory;

/** \addtogroup mi_neuray_mdl_misc
@{
*/

/// Provides access to various functions for the evaluation of MDL expressions.
class IMdl_evaluator_api : public
    mi::base::Interface_declare<0x1dc8e8c2,0xa19e,0x4dc9,0xa3,0x0f,0xeb,0xb4,0x0a,0xf1,0x08,0x58>
{
public:
    /// Evaluates if a function call parameter is enabled, i.e., the \c enable_if condition
    /// evaluates to \c true).
    ///
    /// \param[in]  trans  the transaction
    /// \param[in]  fact   the expression factory to create the result value
    /// \param[in]  call   the function call
    /// \param[in]  index  the index of the material instance parameter
    /// \param[out] errors An optional pointer to an #mi::Sint32 to which an error code will be
    ///                    written. The error codes have the following meaning:
    ///                    -  0: Success.
    ///                    - -1: An input parameter is \c nullptr.
    ///                    - -2: The parameter index is out of bounds.
    ///                    - -3: A malformed expression (contains temporaries).
    ///                    - -4: An unsupported expression occurred.
    ///                    - -5: The evaluation was aborted, too complex to evaluate.
    ///
    /// \return \c nullptr if the condition was to complex to evaluate, else \c true or \c false.
    virtual const IValue_bool* is_function_parameter_enabled(
        ITransaction* trans,
        IValue_factory* fact,
        const IFunction_call* call,
        Size index,
        Sint32* errors) const = 0;
};

/**@}*/ // end group mi_neuray_mdl_misc

} // namespace neuraylib
} // namespace mi

#endif // MI_NEURAYLIB_IMDL_EVALUATOR_H
