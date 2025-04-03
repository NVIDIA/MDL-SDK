/***************************************************************************************************
 * Copyright (c) 2020-2025, NVIDIA CORPORATION. All rights reserved.
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
#include <mi/neuraylib/version.h>

namespace mi {

class IMap;
class IString;

namespace neuraylib {

class IBuffer;
class IBsdf_isotropic_data;
class ICanvas;
class IDeserialized_function_name;
class IDeserialized_module_name;
class ILightprofile;
class IMdl_execution_context;
class IMdle_deserialization_callback;
class IMdle_serialization_callback;
class IReader;
class ISerialized_function_name;
class ITransaction;
class IType;
class IType_list;
class IWriter;

/** \addtogroup mi_neuray_mdl_misc
@{
*/

/// API component for MDL related import and export operations.
class IMdl_impexp_api : public
    mi::base::Interface_declare<0xd8584ade,0xa400,0x486b,0xab,0x29,0x39,0xcd,0x87,0x55,0x14,0x5d>
{
public:

    /// \name Import
    //@{

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
    /// \param argument      The MDL name of the module (for non-MDLE modules), or an MDLE file
    ///                      path (absolute or relative to the current working directory).
    /// \param context       The execution context can be used to pass options to control the
    ///                      behavior of the MDL compiler. The following options are supported
    ///                      by this operation:
    ///                      - string "internal_space" = "coordinate_object"|"coordinate_world"
    ///                        (default = "coordinate_world")
    ///                      .
    ///                      During module loading, compiler messages
    ///                      like errors or warnings are stored in the context. Can be \c nullptr.
    /// \return
    ///                      -  1: Success (module exists already, loading from file was skipped).
    ///                      -  0: Success (module was actually loaded from file or is a builtin
    ///                            module).
    ///                      - -1: The MDL module name/MDLE file path \p argument is
    ///                            invalid or a \c nullptr.
    ///                      - -2: Failed to find or to compile the module \p argument.
    ///                      - -3: The DB name for an imported module is already in use but is not
    ///                            an MDL module, or the DB name for a definition in this module is
    ///                            already in use.
    ///                      - -4: Initialization of an imported module failed.
    ///
    /// \see #mi::neuraylib::IMdl_impexp_api::get_mdl_module_name()
    virtual Sint32 load_module(
        ITransaction* transaction,
        const char* argument,
        IMdl_execution_context* context = nullptr) = 0;

    /// Loads an MDL module from memory into the database.
    ///
    /// The provided module source is compiled. If successful, the method creates DB elements for
    /// the module and all its imported modules, as well as for all material and function
    /// definitions contained in these modules.
    ///
    /// \note String-based module have limitations compared to regular modules loaded from disk:
    /// - no support for resources, and
    /// - string-based modules referenced in an import statement need to be loaded explicitly
    ///   upfront (no automatic recursive loading as for file-based modules).
    ///
    /// \param transaction   The transaction to be used.
    /// \param module_name   The MDL name of the module.
    /// \param module_source The MDL source code of the module.
    /// \param context       The execution context can be used to pass options to control the
    ///                      behavior of the MDL compiler. The following options are supported
    ///                      by this operation:
    ///                      - string "internal_space" = "coordinate_object"|"coordinate_world"
    ///                        (default = "coordinate_world")
    ///                      .
    ///                      During module loading, compiler messages
    ///                      like errors or warnings are stored in the context. Can be \c nullptr.
    /// \return
    ///                      -  1: Success (module exists already, creating from \p module_source
    ///                            was skipped).
    ///                      -  0: Success (module was actually created from \p module_source).
    ///                      - -1: The module name \p module_name is invalid, or \p module_name or
    ///                            \p module_source is a \c nullptr.
    ///                      - -2: Shadows a file-based module or failed to compile the module \p
    ///                            module_name.
    ///                      - -3: The DB name for an imported module is already in use but is not
    ///                            an MDL module, or the DB name for a definition in this module is
    ///                            already in use.
    ///                      - -4: Initialization of an imported module failed.
    ///
    /// \see #mi::neuraylib::IMdl_impexp_api::get_mdl_module_name()
    virtual Sint32 load_module_from_string(
        ITransaction* transaction,
        const char* module_name,
        const char* module_source,
        IMdl_execution_context* context = nullptr) = 0;

    //@}
    /// \name Export
    //@{

    /// Exports an MDL module from the database to disk.
    ///
    /// The following options are supported:
    /// - \c bool "bundle_resources": If \c true, referenced resources are exported
    ///   into the same directory as the module, even if they can be found via the module search
    ///   path. Default: \c false.
    /// - \c bool "export_resources_with_module_prefix": If \c true, the name of the exported
    ///   resources start with the module name as prefix. Default: \c true.
    /// - \c std::string "handle_filename_conflicts": Controls what to do in case of filename
    ///   conflicts for resources during export. Possible values:
    ///   - \c "generate_unique": Always generates a unique filename that does not conflict with an
    ///     existing resource file (adding a counter suffix if necessary).
    ///   - \c "fail_if_existing": The export fails if an existing resource file would be
    ///     overwritten by the export operation.
    ///   - \c "overwrite_existing": The export operation silently overwrites existing resource
    ///     files. Note that using this setting might destroy other modules. Setting the option
    ///     "export_resources_with_module_prefix" (see above) to \c true reduces that risk (but does
    ///     not eliminate it).
    ///   Default: \c "generate_unique".
    ///
    /// \param transaction       The transaction to be used.
    /// \param module_name       The DB name of the MDL module to export.
    /// \param filename          The name of the file to be used for the export. Note that the
    ///                          context option "handle_filename_conflicts" affects only resources,
    ///                          not modules: if this file exists already, it will be overwritten.
    /// \param context           The execution context can be used to pass options to control the
    ///                          behavior of the export operation. Messages like errors or warnings
    ///                          are stored in the context.
    ///                          Can be \c nullptr.
    /// \return
    ///                          -     0: Success.
    ///                          -    -1: Invalid parameters (\c nullptr).
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
    ///                          - -6016: The export of an container-based resource failed.
    virtual Sint32 export_module(
        ITransaction* transaction,
        const char* module_name,
        const char* filename,
        IMdl_execution_context* context = nullptr) = 0;

    /// Exports an MDL module from the database to string.
    ///
    /// \note See #load_module_from_string() for limitations of string-based modules.
    ///
    /// \param transaction       The transaction to be used.
    /// \param module_name       The DB name of the MDL module to export.
    /// \param exported_module   The exported module source code is written to this string.
    /// \param context           The execution context can be used to pass options to control the
    ///                          behavior of the export operation. Messages like errors or warnings
    ///                          are stored in the context.
    ///                          Can be \c nullptr.
    /// \return
    ///                          -     0: Success.
    ///                          -    -1: Invalid parameters (\c nullptr).
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
    ///                          - -6015: The export of container-based resources is not supported for
    ///                                   string-based exports.
    ///                          - -6016: The export of an container-based resource failed.
    virtual Sint32 export_module_to_string(
        ITransaction* transaction,
        const char* module_name,
        IString* exported_module,
        IMdl_execution_context* context = nullptr) = 0;

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
    /// \param filename              The file name of the resource to export the canvas to. The
    ///                              ending of the file name determines the image format, e.g.,
    ///                              \c ".jpg". Note that support for a given image format requires
    ///                              an image plugin capable of handling that format.
    /// \param canvas                The canvas to export.
    /// \param export_options        See \ref mi_image_export_options for supported options.
    /// \return
    ///                             -  0: Success.
    ///                             - -1: Invalid file name.
    ///                             - -2: Invalid canvas.
    ///                             - -4: Unspecified failure.
    virtual Sint32 export_canvas(
        const char* filename,
        const ICanvas* canvas,
        const IMap* export_options = nullptr) const = 0;

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
    /// \param reflection        The BSDF data for reflection to export. Can be \c nullptr.
    /// \param transmission      The BSDF data for transmission to export. Can be \c nullptr.
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

    /// Controls the behavior of #mi::neuraylib::IMdl_impexp_api::get_mdl_module_name().
    enum Search_option
    {
        /// Derive module name from the first search path that matches.
        SEARCH_OPTION_USE_FIRST    = 0,
        /// Derive module name from the shortest search path that matches.
        SEARCH_OPTION_USE_SHORTEST = 1,
        /// Derive module name from the longest search path that matches.
        SEARCH_OPTION_USE_LONGEST  = 2,
        //  Undocumented, for alignment only.
        SEARCH_OPTION_FORCE_32_BIT = 0xffffffffU
    };

    /// Returns the MDL name for an MDL module identified by its filename.
    ///
    /// The return value can be passed to #mi::neuraylib::IMdl_impexp_api::load_module() or
    /// #mi::neuraylib::IMdl_factory::get_db_module_name().
    ///
    /// \note This method does not support MDLE modules. This is also not necessary, since in case
    ///       of MDLEs the filename can be directly passed to
    ///       #mi::neuraylib::IMdl_impexp_api::load_module() or
    ///       #mi::neuraylib::IMdl_factory::get_db_module_name().
    ///
    /// \param filename   The filename of an MDL module (excluding MDLE modules).
    /// \param option     Controls the algorithm's behavior if several overlapping search paths
    ///                   contain the given filename.
    /// \return           The MDL name of the given module, or \c nullptr in case of failures.
    virtual const IString* get_mdl_module_name(
        const char* filename, Search_option option = SEARCH_OPTION_USE_FIRST) const = 0;

    /// Replaces a frame and/or uv-tile marker by coordinates of a given uv-tile.
    ///
    /// \param marker   String containing a valid frame and/or uv-tile marker.
    /// \param f        The frame number of the uv-tile.
    /// \param u        The u-coordinate of the uv-tile.
    /// \param v        The v-coordinate of the uv-tile.
    /// \return         String with the frame and/or uv-tile marker replaced by the coordinates of
    ///                 the uv-tile, or \c nullptr in case of errors.
    virtual const IString* frame_uvtile_marker_to_string(
        const char* marker, Size f, Sint32 u, Sint32 v) const = 0;

    //@}
    /// \name Serialized names
    //@{

    /// Serializes the name of a function or material definition.
    ///
    /// \see \ref mi_mdl_serialized_names
    ///
    /// \param definition_name   The DB name of the function or material definition.
    /// \param argument_types    The arguments of the corresponding function call or material
    ///                          instance. Required for template-like functions, ignored (can be
    ///                          \c nullptr) in all other cases.
    /// \param return_type       The arguments of the corresponding function call or material
    ///                          instance. Required for the cast operator, ignored (can be
    ///                          \c nullptr) in all other cases.
    /// \param mdle_callback     A callback to map the filename of MDLE modules. Ignored for
    ///                          non-MDLE modules. Can be \c nullptr (which is treated like a
    ///                          callback implementing the identity transformation).
    /// \param context           The execution context can be used to pass options and to retrieve
    ///                          error and/or warning messages. Can be \c nullptr.
    /// \return                  The serialized function name, or \c nullptr in case of errors.
    virtual const ISerialized_function_name* serialize_function_name(
        const char* definition_name,
        const IType_list* argument_types,
        const IType* return_type,
        IMdle_serialization_callback* mdle_callback,
        IMdl_execution_context* context) const = 0;

    /// Deserializes the serialized name of a function or material definition (first overload)
    ///
    /// \see \ref mi_mdl_serialized_names
    ///
    /// \param transaction       The transaction to be used.
    /// \param function_name     The serialized name of a function or material definition.
    /// \param mdle_callback     A callback to map the filename of MDLE modules. Ignored for
    ///                          non-MDLE modules. Can be \c nullptr (which is treated like a
    ///                          callback implementing the identity transformation).
    /// \param context           The execution context can be used to pass options and to retrieve
    ///                          error and/or warning messages. Can be \c nullptr.
    /// \return                  The deserialized function name, or \c nullptr in case of errors.
    virtual const IDeserialized_function_name* deserialize_function_name(
        ITransaction* transaction,
        const char* function_name,
        IMdle_deserialization_callback* mdle_callback,
        IMdl_execution_context* context) const = 0;

    /// Deserializes the serialized name of a function or material definition (second overload).
    ///
    /// If the corresponding module has not been loaded, it will be loaded as a side effect. The
    /// method also performs an overload resolution on the deserialized function or material
    /// definition (as in
    /// #mi::neuraylib::IModule::get_function_overloads(const char*,const IArray*)const).
    ///
    /// \see \ref mi_mdl_serialized_names
    ///
    /// \param transaction       The transaction to be used.
    /// \param module_name       The serialized name of a module.
    /// \param function_name_without_module_name    The serialized name of a function or material
    ///                                             definition without the module name and \c "::"
    ///                                             (as returned by
    ///         #mi::neuraylib::ISerialized_function_name::get_function_name_without_module_name()).
    /// \param mdle_callback     A callback to map the filename of MDLE modules. Ignored for
    ///                          non-MDLE modules. Can be \c nullptr (which is treated like a
    ///                          callback implementing the identity transformation).
    /// \param context           The execution context can be used to pass options and to retrieve
    ///                          error and/or warning messages. Can be \c nullptr.
    /// \return                  The deserialized function name, or \c nullptr in case of errors.
    virtual const IDeserialized_function_name* deserialize_function_name(
        ITransaction* transaction,
        const char* module_name,
        const char* function_name_without_module_name,
        IMdle_deserialization_callback* mdle_callback,
        IMdl_execution_context* context) const = 0;

    /// Deserializes the serialized name of a module.
    ///
    /// If the corresponding module has not been loaded, it will be loaded as a side effect. The
    /// method also performs an overload resolution on the deserialized function or material
    /// definition (as in
    /// #mi::neuraylib::IModule::get_function_overloads(const char*,const IArray*)const).
    ///
    /// \see \ref mi_mdl_serialized_names
    ///
    /// \param module_name       The serialized name of a module.
    /// \param mdle_callback     A callback to map the filename of MDLE modules. Ignored for
    ///                          non-MDLE modules. Can be \c nullptr (which is treated like a
    ///                          callback implementing the identity transformation).
    /// \param context           The execution context can be used to pass options and to retrieve
    ///                          error and/or warning messages. Can be \c nullptr.
    /// \return                  The deserialized module name, or \c nullptr in case of errors.
    virtual const IDeserialized_module_name* deserialize_module_name(
        const char* module_name,
        IMdle_deserialization_callback* mdle_callback,
        IMdl_execution_context* context) const = 0;

    //@}
    /// \name Generic reader/writer support
    //@{

    /// Creates a random-access reader for a given buffer.
    virtual IReader* create_reader( const IBuffer* buffer) const = 0;

    /// Returns a random-access reader for a given file.
    ///
    /// \param filename   The filename of the file to get the reader for.
    /// \return           A reader that can be used to read the file, or \c nullptr in case of
    ///                   failures (e.g., there is no such file).
    virtual IReader* create_reader( const char* filename) const = 0;

    /// Returns a random-access writer for a given file.
    ///
    /// \param filename   The filename of the file to get the writer for.
    /// \return           A writer that can be used to write to that file, or \c nullptr in case of
    ///                   failures (e.g., insufficient permissions).
    virtual IWriter* create_writer( const char* filename) const = 0;

    //@}

    virtual Sint32 deprecated_export_canvas(
        const char* filename,
        const ICanvas* canvas,
        Uint32 quality,
        bool force_default_gamma) const = 0;

#ifdef MI_NEURAYLIB_DEPRECATED_15_0
    inline Sint32 export_canvas(
        const char* filename,
        const ICanvas* canvas,
        Uint32 quality,
        bool force_default_gamma = false) const
    { return deprecated_export_canvas( filename, canvas, quality, force_default_gamma); }
#endif
};

mi_static_assert( sizeof( IMdl_impexp_api::Search_option) == sizeof( Uint32));

/// Represents a serialized function name.
///
/// \see #mi::neuraylib::IMdl_impexp_api::serialize_function_name()
class ISerialized_function_name : public
    mi::base::Interface_declare<0x1b22f27d,0xf815,0x495f,0x96,0x71,0x64,0x08,0x5a,0xcc,0x8c,0x0d>
{
public:
    /// Returns the serialized function name.
    ///
    /// Useful for serialization schemes that support only a single string entity.
    ///
    /// Pass to the first overload of
    /// #mi::neuraylib::IMdl_impexp_api::deserialize_function_name() during deserialization.
    virtual const char* get_function_name() const = 0;

    /// Returns the serialized module name.
    ///
    /// Useful for serialization schemes that support two string entities.
    ///
    /// Pass to #mi::neuraylib::IMdl_impexp_api::deserialize_module_name() or the second overload
    /// of #mi::neuraylib::IMdl_impexp_api::deserialize_function_name() during deserialization.
    virtual const char* get_module_name() const = 0;

    /// Returns the serialized function name (without the module name).
    ///
    /// Useful for serialization schemes that support two string entities.
    ///
    /// Pass to the second overload
    /// of #mi::neuraylib::IMdl_impexp_api::deserialize_function_name() during deserialization.
    virtual const char* get_function_name_without_module_name() const = 0;
};

/// Represents a deserialized function name.
///
/// \see #mi::neuraylib::IMdl_impexp_api::deserialize_function_name() (two overloads)
class IDeserialized_function_name : public
    mi::base::Interface_declare<0x2bb03f26,0x3a73,0x499d,0x90,0x64,0x19,0x79,0xea,0x40,0xc1,0x49>
{
public:
    /// Returns the DB name of the function of material definition.
    virtual const char* get_db_name() const = 0;

    /// Returns the argument types of the serialized function call or material instance.
    ///
    /// The argument types are identical to the parameter types of the corresponding definition,
    /// unless it is one of the \ref mi_neuray_mdl_template_like_function_definitions. The remarks
    /// about the expression list for creating calls to the \ref mi_neuray_mdl_cast_operator apply
    /// correspondingly.
    virtual const IType_list* get_argument_types() const = 0;
};

/// Represents a deserialized module name.
///
/// \see #mi::neuraylib::IMdl_impexp_api::deserialize_module_name()
class IDeserialized_module_name : public
    mi::base::Interface_declare<0xe2136899,0x0011,0x45d1,0xb0,0x45,0xa7,0xbb,0x03,0xa7,0xf4,0x0c>
{
public:
    /// Returns the DB name of the module.
    virtual const char* get_db_name() const = 0;

    /// Returns a string suitable for #mi::neuraylib::IMdl_impexp_api::load_module().
    virtual const char* get_load_module_argument() const = 0;
};

/// Callback to map references to MDLE modules during serialization.
///
/// The name of an MDLE module is the absolute filename of the module. This makes it difficult to
/// move such modules around since that requires adjusting all references. This callback allows
/// to rewrite such references during serialization. The reverse mapping should be implemented in
/// an corresponding instance of #mi::neuraylib::IMdle_deserialization_callback.
///
/// One possible solution is to choose a filename that is relative to a known reference point that
/// will be moved together with the MDLE file, e.g., the main scene file.
///
/// \see #mi::neuraylib::IMdle_deserialization_callback,
///      #mi::neuraylib::IMdl_impexp_api::serialize_function_name()
class IMdle_serialization_callback : public
    mi::base::Interface_declare<0x5888652a,0x79d6,0x49ba,0x87,0x90,0x0c,0x1b,0x32,0x83,0xf8,0x63>
{
public:
    /// Returns a serialized filename for the given MDLE filename.
    ///
    /// The implemented mapping should be reversible, otherwise you will run into problems
    /// implementing the corresponding instance of #mi::neuraylib::IMdle_deserialization_callback.
    ///
    /// The callback might get involved several times with the same argument. You might want to
    /// cache results if the computation is expensive.
    ///
    /// \param filename   The current filename of an MDLE module.
    /// \return           The "serialized filename" of that MDLE module. Technically, this can be
    ///                   any string with \c ".mdle" suffix, it does \em not need to refer to an
    ///                   existing MDLE file on disk.
    virtual const IString* get_serialized_filename( const char* filename) const = 0;
};

/// Callback to map references to MDLE modules during deserialization.
///
/// The name of an MDLE module is the absolute filename of the module. This makes it difficult to
/// to move such modules around since that requires adjusting all references. This callback allows
/// to rewrite such references during deserialization. The reverse mapping should be implemented in
/// an corresponding instance of #mi::neuraylib::IMdle_serialization_callback.
///
/// One possible solution is to choose a filename that is relative to a known reference point that
/// will be moved together with the MDLE file, e.g., the main scene file.
///
/// \see #mi::neuraylib::IMdle_serialization_callback,
///      #mi::neuraylib::IMdl_impexp_api::deserialize_function_name() (two overloads),
///      #mi::neuraylib::IMdl_impexp_api::deserialize_module_name()
class IMdle_deserialization_callback : public
    mi::base::Interface_declare<0xe7f636eb,0x8d04,0x4e3b,0x97,0x67,0x3a,0x93,0x1c,0x90,0xc9,0x7e>
{
public:
    /// Returns a the filename of an MDLE module given its serialized filename.
    ///
    /// The callback might get involved several times with the same argument. You might want to
    /// cache results if the computation is expensive.
    ///
    /// \param serialized_filename   The "serialized filename" of an MDLE module. This is the string
    ///                              that has been returned by
    ///                     #mi::neuraylib::IMdle_serialization_callback::get_serialized_filename().
    /// \return                      The actual filename of that MDLE module.
    virtual const IString* get_deserialized_filename( const char* serialized_filename) const = 0;
};

/**@}*/ // end group mi_neuray_mdl_misc

} // namespace neuraylib

} // namespace mi

#endif // MI_NEURAYLIB_IMDL_IMPEXP_API_H
