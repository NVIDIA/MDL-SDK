/***************************************************************************************************
 * Copyright (c) 2012-2023, NVIDIA CORPORATION. All rights reserved.
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
/// \brief API component representing the MDL compiler

#ifndef MI_NEURAYLIB_IMDL_COMPILER_H
#define MI_NEURAYLIB_IMDL_COMPILER_H

#include <mi/base/interface_declare.h>

namespace mi {

namespace neuraylib {

/** \addtogroup mi_neuray_mdl_compiler
@{
*/

/// The MDL compiler allows to register builtin modules.
class IMdl_compiler : public
    mi::base::Interface_declare<0x8fff0a2d,0x7df7,0x4552,0x92,0xf7,0x36,0x1d,0x31,0xc6,0x30,0x08>
{
public:

    /// Adds a builtin MDL module.
    ///
    /// Builtin modules allow to use the \c native() annotation which is not possible for regular
    /// modules. Builtin modules can only be added before the first regular module has been loaded.
    ///
    /// \note After adding a builtin module it is still necessary to load it using
    ///       #mi::neuraylib::IMdl_impexp_api::load_module() before it can actually be used.
    ///
    /// \param module_name     The MDL name of the module.
    /// \param module_source   The MDL source code of the module.
    /// \return
    ///                        -  0: Success.
    ///                        - -1: Possible failure reasons: invalid parameters (\c NULL pointer),
    ///                              \p module_name is not a valid module name, failure to compile
    ///                              the module, or a regular module has already been loaded.
    virtual Sint32 add_builtin_module( const char* module_name, const char* module_source) = 0;
};

/**@}*/ // end group mi_neuray_mdl_compiler

} // namespace neuraylib

} // namespace mi

#endif // MI_NEURAYLIB_IMDL_COMPILER_H
