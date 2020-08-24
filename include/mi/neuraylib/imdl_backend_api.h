/***************************************************************************************************
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
 **************************************************************************************************/
/// \file
/// \brief API component for MDL backend related operations.

#ifndef MI_NEURAYLIB_IMDL_BACKEND_API_H
#define MI_NEURAYLIB_IMDL_BACKEND_API_H

#include <mi/base/interface_declare.h>

namespace mi {

namespace neuraylib {

class IMdl_backend;

/** \defgroup mi_neuray_mdl_compiler MDL compiler
    \ingroup mi_neuray

    This module contains the \neurayAdjectiveName API components representing the MDL compiler,
    its backends and the generated target code.

    \if MDL_SDK_API
      The MDL compiler can be obtained from #mi::neuraylib::INeuray::get_api_component().
    \endif
    The backends can be obtained via the MDL backend API from
    #mi::neuraylib::IMdl_backend_api::get_backend().
*/

/** \addtogroup mi_neuray_mdl_types
@{
*/


/// Possible kinds of distribution function data.
enum Df_data_kind {
    DFK_NONE,
    DFK_INVALID,
    DFK_SIMPLE_GLOSSY_MULTISCATTER,
    DFK_BACKSCATTERING_GLOSSY_MULTISCATTER,
    DFK_BECKMANN_SMITH_MULTISCATTER,
    DFK_BECKMANN_VC_MULTISCATTER,
    DFK_GGX_SMITH_MULTISCATTER,
    DFK_GGX_VC_MULTISCATTER,
    DFK_WARD_GEISLER_MORODER_MULTISCATTER,
    DFK_SHEEN_MULTISCATTER,
    DFK_FORCE_32_BIT = 0xffffffffU
};

mi_static_assert(sizeof(Df_data_kind) == sizeof(Uint32));

/// This interface can be used to obtain the MDL backends.
class IMdl_backend_api : public
    mi::base::Interface_declare<0x425559dd,0xbf91,0x459a,0xaa,0xaf,0xc3,0x14,0x8a,0x5a,0x2f,0x99>
{
public:

    /// Currently available MDL backends.
    enum Mdl_backend_kind {
        MB_CUDA_PTX,          ///< Generate CUDA PTX code.
        MB_LLVM_IR,           ///< Generate LLVM IR (LLVM 7.0 compatible).
        MB_GLSL,              ///< \if MDL_SOURCE_RELEASE Reserved \else Generate GLSL code \endif.
        MB_NATIVE,            ///< Generate native code.
        MB_HLSL,              ///< \if MDL_SOURCE_RELEASE Reserved \else Generate HLSL code \endif.
        MB_FORCE_32_BIT = 0xffffffffU //   Undocumented, for alignment only
    };

    /// Returns an MDL backend generator.
    ///
    /// \param kind              The desired backend generator.
    /// \return                  The backend generator, or \c NULL if the requested backend is not
    ///                          available.
    virtual IMdl_backend* get_backend(Mdl_backend_kind kind) = 0;

    //@}

    /// Returns the distribution function data of the texture identified by \p kind.
    ///
    /// \param kind       The kind of the distribution function data texture.
    /// \param [out] rx   The resolution of the texture in x.
    /// \param [out] ry   The resolution of the texture in y.
    /// \param [out] rz   The resolution of the texture in z.
    /// \return           A pointer to the texture data or \c NULL, if \p kind does not
    ///                   correspond to a distribution function data texture.
    virtual const Float32* get_df_data_texture(
        Df_data_kind kind,
        Size &rx,
        Size &ry,
        Size &rz) const = 0;
};

/*@}*/ // end group mi_neuray_mdl_types

} // namespace neuraylib

} // namespace mi

#endif // MI_NEURAYLIB_IMDL_BACKEND_API_H

