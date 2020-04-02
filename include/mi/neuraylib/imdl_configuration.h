/***************************************************************************************************
 * Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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
/// \brief API component for MDL related settings.

#ifndef MI_NEURAYLIB_IMDL_CONFIGURATION_H
#define MI_NEURAYLIB_IMDL_CONFIGURATION_H

#include <mi/base/interface_declare.h>

namespace mi {

namespace neuraylib {

/** \addtogroup mi_neuray_configuration
@{
*/

/// This interface can be used to query and change the MDL configuration.
class IMdl_configuration : public
    mi::base::Interface_declare<0x2657ec0b,0x8a40,0x46c5,0xa8,0x3f,0x2b,0xb5,0x72,0xa0,0x8b,0x9c>
{
public:

    /// Defines whether a cast operator is automatically inserted for compatible argument types.
    ///
    /// If set to \c true, an appropriate cast operator is automatically inserted if arguments for
    /// instances of #mi::neuraylib::IFunction_call or #mi::neuraylib::IMaterial_instance have a
    /// different but compatible type. If set to \c false, such an assignment fails and it is
    /// necessary to insert the cast operator explicitly. Default: \c true.
    ///
    /// \see #mi::neuraylib::IExpression_factory::create_cast().
    ///
    /// \param value    \c True to enable the feature, \c false otherwise.
    /// \return
    ///                -  0: Success.
    ///                - -1: The method cannot be called at this point of time.
    virtual Sint32 set_implicit_cast_enabled( bool value) = 0;

    /// Indicates whether the SDK is supposed to automatically insert the cast operator for
    /// compatible types.
    ///
    /// \see #set_implicit_cast_enabled()
    virtual bool get_implicit_cast_enabled() const = 0;
    
    /// Defines whether an attempt is made to expose names of let expressions.
    ///
    /// If set to \c true, the MDL compiler attempts to represent let expressions as temporaries,
    /// and makes the name of let expressions available as names of such temporaries. In order to
    /// do so, certain optimizations are disabled, in particular, constant folding. These names are
    /// only available on material and functions definitions, not on compiled materials, which are
    /// always highly optimized. Default: \c false.
    ///
    /// \note Since some optimizations are essential for inner workings of the MDL compiler, there
    //        is no guarantee that the name of a particular let expression is exposed.
    ///
    /// \see #mi::neuraylib::IFunction_definition::get_temporary_name(),
    ///      #mi::neuraylib::IMaterial_definition::get_temporary_name()
    virtual Sint32 set_expose_names_of_let_expressions( bool value) = 0;

    /// Indicates whether an attempt is made to expose names of let expressions.
    ///
    /// \see #set_expose_names_of_let_expressions()
    virtual bool get_expose_names_of_let_expressions() const = 0;
};

/*@}*/ // end group mi_neuray_configuration

} // namespace neuraylib

} // namespace mi

#endif // MI_NEURAYLIB_IMDL_CONFIGURATION_H
