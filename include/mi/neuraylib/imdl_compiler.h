/***************************************************************************************************
 * Copyright (c) 2012-2020, NVIDIA CORPORATION. All rights reserved.
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
#include <mi/neuraylib/imdl_backend.h>
#include <mi/neuraylib/version.h>

namespace mi {

namespace base { class ILogger; }

class IString;

namespace neuraylib {

class IBsdf_isotropic_data;
class ICanvas;
class ILightprofile;
class IMdl_backend;
class IMdl_execution_context;
class ITransaction;

/** \addtogroup mi_neuray_mdl_compiler
@{
*/

/// The MDL compiler allows to register builtin modules.
class IMdl_compiler : public
    mi::base::Interface_declare<0x8fff0a2d,0x7df7,0x4552,0x92,0xf7,0x36,0x1d,0x31,0xc6,0x30,0x08>
{
public:

    virtual void MI_NEURAYLIB_DEPRECATED_METHOD_11_1(set_logger)( base::ILogger* logger) = 0;

    virtual base::ILogger* MI_NEURAYLIB_DEPRECATED_METHOD_11_1(get_logger)() = 0;

    virtual Sint32 MI_NEURAYLIB_DEPRECATED_METHOD_11_1(add_module_path)( const char* path) = 0;

    virtual Sint32 MI_NEURAYLIB_DEPRECATED_METHOD_11_1(remove_module_path)( const char* path) = 0;


    virtual void MI_NEURAYLIB_DEPRECATED_METHOD_11_1(clear_module_paths)() = 0;

    virtual Size MI_NEURAYLIB_DEPRECATED_METHOD_11_1(get_module_paths_length)() const = 0;

    virtual const IString* MI_NEURAYLIB_DEPRECATED_METHOD_11_1(get_module_path)( Size index) const = 0;


    virtual Sint32 MI_NEURAYLIB_DEPRECATED_METHOD_11_1(add_resource_path)( const char* path) = 0;

    virtual Sint32 MI_NEURAYLIB_DEPRECATED_METHOD_11_1(remove_resource_path)( const char* path) = 0;

    virtual void MI_NEURAYLIB_DEPRECATED_METHOD_11_1(clear_resource_paths)() = 0;

    virtual Size MI_NEURAYLIB_DEPRECATED_METHOD_11_1(get_resource_paths_length)() const = 0;

    virtual const IString* MI_NEURAYLIB_DEPRECATED_METHOD_11_1(get_resource_path)( Size index) const = 0;

    virtual Sint32 MI_NEURAYLIB_DEPRECATED_METHOD_11_1(load_plugin_library)( const char* path) = 0;

    virtual Sint32 MI_NEURAYLIB_DEPRECATED_METHOD_11_1(load_module)(
        ITransaction* transaction, const char* module_name, IMdl_execution_context* context = 0) = 0;

    virtual const char* MI_NEURAYLIB_DEPRECATED_METHOD_11_1(get_module_db_name)(
        ITransaction* transaction, const char* module_name, IMdl_execution_context* context = 0) = 0;

    virtual Sint32 MI_NEURAYLIB_DEPRECATED_METHOD_11_1(load_module_from_string)(
        ITransaction* transaction,
        const char* module_name,
        const char* module_source,
        IMdl_execution_context* context = 0) = 0;

    /// Adds a builtin MDL module.
    ///
    /// Builtin modules allow to use the \c native() annotation which is not possible for regular
    /// modules. Builtin modules can only be added before the first regular module has been loaded.
    ///
    /// \note After adding a builtin module it is still necessary to load it using
    ///       #mi::neuraylib::IMdl_impexp_api::load_module() before it can actually be used.
    ///
    /// \param module_name     The fully-qualified MDL name of the MDL module (including package
    ///                        names, starting with "::").
    /// \param module_source   The MDL source code of the module.
    /// \return
    ///                        -  0: Success.
    ///                        - -1: Possible failure reasons: invalid parameters (\c NULL pointer),
    ///                              \p module_name is not a valid module name, failure to compile
    ///                              the module, or a regular module has already been loaded.
    virtual Sint32 add_builtin_module( const char* module_name, const char* module_source) = 0;

    virtual Sint32 MI_NEURAYLIB_DEPRECATED_METHOD_11_1(export_module)(
        ITransaction* transaction,
        const char* module_name,
        const char* filename,
        IMdl_execution_context* context = 0) = 0;

    virtual Sint32 MI_NEURAYLIB_DEPRECATED_METHOD_11_1(export_module_to_string)(
        ITransaction* transaction,
        const char* module_name,
        IString* exported_module,
        IMdl_execution_context* context = 0) = 0;

    virtual Sint32 MI_NEURAYLIB_DEPRECATED_METHOD_11_1(export_canvas)(
        const char* filename, const ICanvas* canvas, Uint32 quality = 100) const = 0;

    virtual Sint32 MI_NEURAYLIB_DEPRECATED_METHOD_11_1(export_lightprofile)(
        const char* filename, const ILightprofile* lightprofile) const = 0;

    virtual Sint32 MI_NEURAYLIB_DEPRECATED_METHOD_11_1(export_bsdf_data)(
        const char* filename,
        const IBsdf_isotropic_data* reflection,
        const IBsdf_isotropic_data* transmission) const = 0;

    virtual const IString* MI_NEURAYLIB_DEPRECATED_METHOD_11_1(uvtile_marker_to_string)(
        const char* marker,
        Sint32 u,
        Sint32 v) const = 0;

    virtual const IString*  MI_NEURAYLIB_DEPRECATED_METHOD_11_1(uvtile_string_to_marker)(
        const char* str, const char* marker) const = 0;

    enum Mdl_backend_kind {
        MB_CUDA_PTX,
        MB_LLVM_IR,
        MB_GLSL,
        MB_NATIVE,
        MB_HLSL,
        MB_FORCE_32_BIT = 0xffffffffU //   Undocumented, for alignment only
    };

    virtual IMdl_backend* MI_NEURAYLIB_DEPRECATED_METHOD_11_1(get_backend)( Mdl_backend_kind kind) = 0;

    virtual const Float32* MI_NEURAYLIB_DEPRECATED_METHOD_11_1(get_df_data_texture)(
        Df_data_kind kind,
        Size &rx,
        Size &ry,
        Size &rz) const = 0;
};

mi_static_assert( sizeof( IMdl_compiler::Mdl_backend_kind)== sizeof( Uint32));

/*@}*/ // end group mi_neuray_mdl_compiler

} // namespace neuraylib

} // namespace mi

#endif // MI_NEURAYLIB_IMDL_COMPILER_H
