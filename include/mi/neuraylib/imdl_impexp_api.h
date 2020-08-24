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
/// \brief API component for MDL related import and export operations.

#ifndef MI_NEURAYLIB_IMDL_IMPEXP_API_H
#define MI_NEURAYLIB_IMDL_IMPEXP_API_H

#include <mi/base/interface_declare.h>

namespace mi {

class IString;

namespace neuraylib {

class IBsdf_isotropic_data;
class ICanvas;
class ILightprofile;
class IMdl_execution_context;
class ITransaction;

/** \addtogroup mi_neuray_mdl_types
@{
*/

/// API component for MDL related import and export operations.
class IMdl_impexp_api : public
    mi::base::Interface_declare<0xd8584ade,0xa400,0x486b,0xab,0x29,0x39,0xcd,0x87,0x55,0x14,0x5d>
{
public:

    /// Loads an MDL module from disk (or a builtin module) into the database.
    ///
    /// The module is located on disk according to the module search paths
    /// (see #mi::neuraylib::IMdl_configuration::add_mdl_path()), loaded, and compiled.
    /// If successful, the method creates DB elements for the module and all
    /// its imported modules, as well as for all material and function definitions contained in
    /// these modules.
    ///
    /// The method can also be used for builtin modules for which the first step, locating the
    /// module on disk, is skipped.
    ///
    /// \param transaction   The transaction to be used.
    /// \param module_name   The fully-qualified MDL name of the MDL module (including package
    ///                      names, starting with "::") or an MDLE file path (absolute or relative
    ///                      to the current working directory).
    /// \param context       The execution context can be used to pass options to control the
    ///                      behavior of the MDL compiler. The following options are supported
    ///                      by this operation:
    ///                      - string "internal_space" = "coordinate_object"|"coordinate_world"
    ///                        (default = "coordinate_world")
    ///                      .
    ///                      During module loading, compiler messages
    ///                      like errors or warnings are stored in the context. Can be \c NULL.
    /// \return
    ///                      -  1: Success (module exists already, loading from file was skipped).
    ///                      -  0: Success (module was actually loaded from file or is a builtin
    ///                            module).
    ///                      - -1: The module name \p module_name is invalid or a \c NULL pointer.
    ///                      - -2: Failed to find or to compile the module \p module_name.
    ///                      - -3: The DB name for an imported module is already in use but is not
    ///                            an MDL module, or the DB name for a definition in this module is
    ///                            already in use.
    ///                      - -4: Initialization of an imported module failed.
    virtual Sint32 load_module(
        ITransaction* transaction, const char* module_name, IMdl_execution_context* context = 0) = 0;

    /// Loads an MDL module from memory into the database.
    ///
    /// The provided module source is compiled. If successful, the method creates DB elements for
    /// the module and all its imported modules, as well as for all material and function
    /// definitions contained in these modules.
    ///
    /// \param transaction   The transaction to be used.
    /// \param module_name   The fully-qualified MDL name of the MDL module (including package
    ///                      names, starting with "::").
    /// \param module_source The MDL source code of the module.
    /// \param context       The execution context can be used to pass options to control the
    ///                      behavior of the MDL compiler. The following options are supported
    ///                      by this operation:
    ///                      - string "internal_space" = "coordinate_object"|"coordinate_world"
    ///                        (default = "coordinate_world")
    ///                      .
    ///                      During module loading, compiler messages
    ///                      like errors or warnings are stored in the context. Can be \c NULL.
    /// \return
    ///                      -  1: Success (module exists already, creating from \p module_source
    ///                            was skipped).
    ///                      -  0: Success (module was actually created from \p module_source).
    ///                      - -1: The module name \p module_name is invalid, or \p module_name or
    ///                            \p module_source is a \c NULL pointer.
    ///                      - -2: Shadows a file-based module or failed to compile the module \p
    ///                            module_name.
    ///                      - -3: The DB name for an imported module is already in use but is not
    ///                            an MDL module, or the DB name for a definition in this module is
    ///                            already in use.
    ///                      - -4: Initialization of an imported module failed.
    virtual Sint32 load_module_from_string(
        ITransaction* transaction,
        const char* module_name,
        const char* module_source,
        IMdl_execution_context* context = 0) = 0;

    /// Exports an MDL module from the database to disk.
    ///
    /// The following options are supported:
    /// - \c "bundle_resources" of type bool: If \c true, referenced resources are exported
    ///   into the same directory as the module, even if they can be found via the module search
    ///   path. Default: \c false.
    ///
    /// \param transaction       The transaction to be used.
    /// \param module_name       The DB name of the MDL module to export.
    /// \param filename          The name of the file to be used for the export.
    /// \param context           The execution context can be used to pass options to control the
    ///                          behavior of the MDL compiler. During module loading, compiler
    ///                          messages like errors or warnings are stored in the context.
    ///                          Can be \c NULL.
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
    virtual Sint32 export_module(
        ITransaction* transaction,
        const char* module_name,
        const char* filename,
        IMdl_execution_context* context = 0) = 0;

    /// Exports an MDL module from the database to string.
    ///
    /// \param transaction       The transaction to be used.
    /// \param module_name       The DB name of the MDL module to export.
    /// \param exported_module   The exported module source code is written to this string.
    /// \param context           The execution context can be used to pass options to control the
    ///                          behavior of the MDL compiler. During module loading, compiler
    ///                          messages like errors or warnings are stored in the context.
    ///                          Can be \c NULL.
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
        ITransaction* transaction,
        const char* module_name,
        IString* exported_module,
        IMdl_execution_context* context = 0) = 0;

    /// Exports a canvas to a file on disk.
    ///
    /// If the image plugin that is selected for the export based on the \p filename parameter is
    /// not capable of handling the pixel type of \p canvas, the canvas is internally converted into
    /// one of the pixel types supported by that image plugin for export. If the image plugin
    /// supports multiple pixel types for export, the "best" of them (w.r.t. the pixel type of the
    /// canvas) is chosen.
    ///
    /// The "best" pixel type is determined by attempting to apply the following conversions in the
    /// given order to the pixel type of the canvas:
    /// - use an equivalent pixel type (\c "Color" instead of \c "Float32<4>" and vice versa,
    ///   similar for \c "Rgb_fp" / \c "Float32<3>" and \c "Rgba" / \c "Sint32"),
    /// - add an alpha channel (if not already present),
    /// - increase bits per channel (smaller increase preferred),
    /// - add additional channels (if possible),
    /// - decrease bits per channel (smaller decrease preferred), and
    /// - drop one or more channels.
    ///
    /// \param filename          The file name of the resource to export the canvas to. The ending
    ///                          of the file name determines the image format, e.g., \c ".jpg". Note
    ///                          that support for a given image format requires an image plugin
    ///                          capable of handling that format.
    /// \param canvas            The canvas to export.
    /// \param quality           The compression quality is an integer in the range from 0 to 100,
    ///                          where 0 is the lowest quality, and 100 is the highest quality.
    /// \return
    ///                          -  0: Success.
    ///                          - -1: Invalid URI.
    ///                          - -2: Invalid canvas.
    ///                          - -3: Invalid quality.
    ///                          - -4: Unspecified failure.
    virtual Sint32 export_canvas(
        const char* filename, const ICanvas* canvas, Uint32 quality = 100) const = 0;

    /// Exports a light profile to disk.
    ///
    /// \param filename          The file name of the resource to export the light profile to.
    /// \param lightprofile      The light profile to export.
    /// \return
    ///                          -  0: Success.
    ///                          - -1: Invalid file name.
    ///                          - -2: Invalid light profile.
    ///                          - -4: Unspecified failure.
    virtual Sint32 export_lightprofile(
        const char* filename, const ILightprofile* lightprofile) const = 0;

    /// Exports BSDF data to a file on disk.
    ///
    /// \param filename          The file name of the resource to export the BSDF measurement to.
    /// \param reflection        The BSDF data for reflection to export. Can be \p NULL.
    /// \param transmission      The BSDF data for transmission to export. Can be \p NULL.
    /// \return
    ///                          -  0: Success.
    ///                          - -1: Invalid file name.
    ///                          - -4: Unspecified failure.
    virtual Sint32 export_bsdf_data(
        const char* filename,
        const IBsdf_isotropic_data* reflection,
        const IBsdf_isotropic_data* transmission) const = 0;

    //@}
    /// \name Convenience
    //@{

    /// Resolves a string containing a UDIM/uv-tile marker and a corresponding u,v pair
    /// to a pattern as used in the filename of a UDIM/uv-tile sequence.
    ///
    /// \param marker string containing a valid MDL UDIM/uv-tile marker.
    /// \param u      uv-tile position in u-direction
    /// \param v      uv-tile position in v-direction
    /// \return       a string containing the resolved pattern.
    virtual const IString* uvtile_marker_to_string(
        const char* marker,
        Sint32 u,
        Sint32 v) const = 0;

    /// Replaces the pattern describing the tile index of a UDIM/uv-tile image sequence
    /// by the given marker, if the pattern exists in the string.
    ///
    /// \param str      string containing the pattern, e.g. _u1_v1
    /// \param marker   the marker to replace the pattern with
    /// \return string with marker or \p NULL, if a corresponding pattern could not be found.
    virtual const IString*  uvtile_string_to_marker(
        const char* str, const char* marker) const = 0;
};

/*@}*/ // end group mi_neuray_mdl_types

} // namespace neuraylib

} // namespace mi

#endif // MI_NEURAYLIB_IMDL_IMPEXP_API_H
