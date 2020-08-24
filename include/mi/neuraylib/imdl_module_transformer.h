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
/// \brief      The MDL module transformer

#ifndef MI_NEURAYLIB_IMDL_MODULE_TRANSFORMER_H
#define MI_NEURAYLIB_IMDL_MODULE_TRANSFORMER_H

#include <mi/base/interface_declare.h>
#include <mi/neuraylib/imodule.h>

namespace mi {

class IString;

namespace neuraylib {

class IMdl_execution_context;

/** \addtogroup mi_neuray_mdl_misc
@{
*/

/// The module transformer allows to apply certain transformations on an MDL module.
///
/// \note Beware of the following implicit changes to MDL modules when using the module transformer,
///       in particular from the point of view of MDL source code:
///       - All comments are removed.
///       - Defaults are inserted for parameters without arguments.
///       - Conversion between named and positional arguments.
///       - Reformatting of the MDL source code.
///       - Possibly restructering of alias declarations (including introduction of new ones).
class IMdl_module_transformer : public
    mi::base::Interface_declare<0x3501f2ef,0xe7c0,0x492e,0xb2,0xd5,0x73,0xe2,0x33,0xa0,0x77,0x36>
{
public:
    /// Upgrades the MDL version.
    ///
    /// The MDL version can only be increased, not decreased. The new MDL needs to be at least MDL
    /// version 1.3.
    virtual Sint32 upgrade_mdl_version( Mdl_version version, IMdl_execution_context* context) = 0;

    /// Changes import declarations to absolute style.
    ///
    /// Only import declaration that match the include filter, but not the exclude filter, are
    /// changed.
    ///
    /// \param include_filter  An extended regular expression [\ref OGBS7] of absolute MDL module
    ///                        names to include, or \c NULL which matches any module.
    /// \param exclude_filter  An extended regular expression [\ref OGBS7] of absolute MDL module
    ///                        names to exclude, or \c NULL which matches no module.
    /// \param context         The execution context can be used to obtain messages like errors or
    ///                        warnings are stored in the context. Can be \c NULL.
    /// \return                A return code. The return codes have the following meaning:
    ///                        -  0: Success.
    ///                        - -1: An error occurred. Details are provided in the execution
    ///                              context.
    virtual Sint32 use_absolute_import_declarations(
        const char* include_filter,
        const char* exclude_filter,
        IMdl_execution_context* context) = 0;

    /// Changes import declarations to strict relative style.
    ///
    /// Only import declaration that match the include filter, but not the exclude filter, are
    /// changed. Import declarations for builtin modules or modules in different search paths are
    /// never changed.
    ///
    /// This transformation requires MDL version >= 1.3.
    ///
    /// \param include_filter  An extended regular expression [\ref OGBS7] of absolute MDL module
    ///                        names to include, or \c NULL which matches any module.
    /// \param exclude_filter  An extended regular expression [\ref OGBS7] of absolute MDL module
    ///                        names to exclude, or \c NULL which matches no module.
    /// \param context         The execution context can be used to obtain messages like errors or
    ///                        warnings are stored in the context. Can be \c NULL.
    /// \return                A return code. The return codes have the following meaning:
    ///                        -  0: Success.
    ///                        - -1: An error occurred. Details are provided in the execution
    ///                              context.
    virtual Sint32 use_relative_import_declarations(
        const char* include_filter,
        const char* exclude_filter,
        IMdl_execution_context* context) = 0;

    /// Changes resource file paths to absolute style.
    ///
    /// Only resource file paths that match the include filter, but not the exclude filter, are
    /// changed. Memory-based resources do not have any file paths. They are not affected by this
    /// transformation. Later, they are exported next to the module with relative paths.
    ///
    /// \param include_filter  An extended regular expression [\ref OGBS7] of absolute file paths
    ///                        to include, or \c NULL which matches any resource.
    /// \param exclude_filter  An extended regular expression [\ref OGBS7] of absolute file paths
    ///                        to exclude, or \c NULL which matches no resource.
    /// \param context         The execution context can be used to obtain messages like errors or
    ///                        warnings are stored in the context. Can be \c NULL.
    /// \return                A return code. The return codes have the following meaning:
    ///                        -  0: Success.
    ///                        - -1: An error occurred. Details are provided in the execution
    ///                              context.
    virtual Sint32 use_absolute_resource_file_paths(
        const char* include_filter,
        const char* exclude_filter,
        IMdl_execution_context* context) = 0;

    /// Changes resource file paths to relative style.
    ///
    /// Only resource file paths that match the include filter, but not the exclude filter, are
    /// changed. Resource file paths for resources in different search paths are never changed.
    ///
    /// Memory-based resources do not have any file paths. They are not affected by this
    /// transformation. Later, they are exported next to the module with relative paths.
    ///
    /// This transformation requires MDL version >= 1.3.
    ///
    /// \param include_filter  An extended regular expression [\ref OGBS7] of absolute file paths
    ///                        to include, or \c NULL which matches any resource.
    /// \param exclude_filter  An extended regular expression [\ref OGBS7] of absolute file paths
    ///                        to exclude, or \c NULL which matches no resource.
    /// \param context         The execution context can be used to obtain messages like errors or
    ///                        warnings are stored in the context. Can be \c NULL.
    /// \return                A return code. The return codes have the following meaning:
    ///                        -  0: Success.
    ///                        - -1: An error occurred. Details are provided in the execution
    ///                              context.
    virtual Sint32 use_relative_resource_file_paths(
        const char* include_filter,
        const char* exclude_filter,
        IMdl_execution_context* context) = 0;

    /// Inline imported modules.
    ///
    /// Only modules that match the include filter, but not the exclude filter, are inlined.
    /// Builtin modules are never inlined.
    ///
    /// \param include_filter    An extended regular expression [\ref OGBS7] of absolute MDL module
    ///                          names to include, or \c NULL which matches any module.
    /// \param exclude_filter    An extended regular expression [\ref OGBS7] of absolute MDL module
    ///                          names to exclude, or \c NULL which matches no module.
    /// \param omit_anno_origin  The \c anno::origin annotation causes an MDL version of at least
    ///                          1.5 for the new module. Omitting the annotation allows to create
    ///                          MDL modules of older versions (depending on the other features
    ///                          used).
    /// \param context           The execution context can be used to obtain messages like errors or
    ///                          warnings are stored in the context. Can be \c NULL.
    /// \return                  A return code. The return codes have the following meaning:
    ///                          -  0: Success.
    ///                          - -1: An error occurred. Details are provided in the execution
    ///                              context.
    virtual Sint32 inline_imported_modules(
        const char* include_filter,
        const char* exclude_filter,
        bool omit_anno_origin,
        IMdl_execution_context* context) = 0;

    /// Exports the transformed MDL module to disk.
    ///
    /// The following options are supported:
    /// - \c "bundle_resources" of type bool: If \c true, referenced resources are exported
    ///   into the same directory as the module, even if they can be found via the module search
    ///   path. Default: \c false.
    ///
    /// \param filename          The name of the file to be used for the export.
    /// \param context           The execution context can be used to obtain messages like errors
    ///                          or warnings are stored in the context. Can be \c NULL.
    /// \return
    ///                          -     0: Success.
    ///                          -    -1: Invalid parameters (\c NULL pointer).
    ///                          -    -2: Failed to open \p filename for write operations.
    ///                          - -6002: There is no MDL module in the database of the given name.
    ///                          - -6003: The export failed for unknown reasons.
    ///                          - -6004: The MDL module can not be exported since it is a builtin
    ///                                   module.
    ///                          - -6005: The MDL module can not be exported since \p filename does
    ///                                   not result in a valid MDL identifier.
    ///                          - -6010: Incorrect type for a referenced resource.
    ///                          - -6013: The export of a file-based resource failed.
    ///                          - -6014: The export of a memory-based resource failed.
    ///                          - -6016: The export of an archive-based resource failed.
    virtual Sint32 export_module( const char* filename, IMdl_execution_context* context) = 0;

    /// Exports the transformed MDL module to string.
    ///
    /// \param exported_module   The exported module source code is written to this string.
    /// \param context           The execution context can be used to obtain messages like errors
    ///                          or warnings are stored in the context. Can be \c NULL.
    /// \return
    ///                          -     0: Success.
    ///                          -    -1: Invalid parameters (\c NULL pointer).
    ///                          - -6002: There is no MDL module in the database of the given name.
    ///                          - -6003: The export failed for unknown reasons.
    ///                          - -6004: The MDL module can not be exported since it is a builtin
    ///                                   module.
    ///                          - -6006: The option \c bundle_resources is not supported for
    ///                                   string-based exports.
    ///                          - -6010: Incorrect type for a referenced resource.
    ///                          - -6011: The export of file-based resources is not supported for
    ///                                   string-based exports.
    ///                          - -6012: The export of memory-based resources is not supported for
    ///                                   string-based exports.
    ///                          - -6013: The export of a file-based resource failed.
    ///                          - -6014: The export of a memory-based resource failed.
    ///                          - -6015: The export of archive-based resources is not supported for
    ///                                   string-based exports.
    ///                          - -6016: The export of an archive-based resource failed.
    virtual Sint32 export_module_to_string(
        IString* exported_module, IMdl_execution_context* context) = 0;
};

/*@}*/ // end group mi_neuray_mdl_misc

} // namespace neuraylib

} // namespace mi

#endif // MI_NEURAYLIB_IMDL_MODULE_TRANSFORMER_H
