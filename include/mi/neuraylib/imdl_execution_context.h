/***************************************************************************************************
 * Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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
/// \brief      API component that gives access to the MDL compatibility API.

#ifndef MI_NEURAYLIB_IMDL_EXECUTION_CONTEXT_H
#define MI_NEURAYLIB_IMDL_EXECUTION_CONTEXT_H

#include <mi/base/interface_declare.h>
#include <mi/base/enums.h>

namespace mi {

namespace neuraylib {


/** \addtogroup mi_neuray_mdl_types
@{
*/

/// Message interface.
class IMessage: public
    base::Interface_declare<0x51965a01,0xcd3f,0x41fc,0xb1,0x8b,0x8,0x1c,0x7b,0x4b,0xba,0xb2>
{
public:

    /// The possible kinds of messages.
    /// A message can be uniquely identified by the message code and kind, 
    /// except for uncategorized messages.
    enum Kind {

        /// MDL Core compiler message.
        MSG_COMILER_CORE,
        /// MDL Core compiler backend message.
        MSG_COMILER_BACKEND,
        /// MDL Core DAG generator message.
        MSG_COMPILER_DAG,
        /// MDL Core archive tool message.
        MSG_COMPILER_ARCHIVE_TOOL,
        /// MDL import/exporter message.
        MSG_IMP_EXP,
        /// MDL integration message.
        MSG_INTEGRATION,
        /// Uncategorized messages do not have a code.
        MSG_UNCATEGORIZED,
        //  Undocumented, for alignment only.
        MSG_FORCE_32_BIT = 0xffffffffU
    };

    /// Returns the kind of message.
    virtual Kind get_kind() const = 0;

    /// Returns the severity of the message.
    virtual base::Message_severity get_severity() const = 0;
    
    /// Returns the message string.
    virtual const char* get_string() const = 0;
    
    /// Returns a unique identifier for the message.
    virtual Sint32 get_code() const = 0;
    
    /// Returns the number of notes associated with the message
    ///
    /// Notes can be used to describe an error message further or add additional details.
    virtual Size get_notes_count() const = 0;
    
    /// Returns the note at index or NULL, if no such index exists.
    virtual const IMessage* get_note(Size index) const = 0;
};

/// The execution context can be used to query status information like error
/// and warning messages concerning the operation it was passed into.
///
/// The context supports the following options:
///
/// Options for module loading
/// - "internal_space": Set the internal space of the backend. Possible values: "coordinate_world",
///   "coordinate_object". Default: "coordinate_world".
/// - "experimental": If \c true, enables undocumented experimental MDL features. Default: false.
///
/// Options for MDL export
/// - "bundle_resources": If \c true, referenced resources are exported into the same directory as
///   the module, even if they can be found via the module search path. Default: false.
///
/// Options for material compilation
/// - "meters_per_scene_unit": The conversion ratio between meters and scene units for this
///   material. Default: 1.0f.
/// - "wavelength_min": The smallest supported wavelength. Default: 380.0f.
/// - "wavelength_max": The largest supported wavelength. Default: 780.0f.
///
/// Options for code generation
/// - "meters_per_scene_unit": The conversion ratio between meters and scene units for this
///   material. Default: 1.0f.
/// - "wavelength_min": The smallest supported wavelength. Default: 380.0f.
/// - "wavelength_max": The largest supported wavelength. Default: 780.0f.
/// - "include_geometry_normal": If true, the \c "geometry.normal" field will be applied to the
///   MDL state prior to evaluation of the given DF. Default: true.

class IMdl_execution_context: public
    base::Interface_declare<0x28eb1f99,0x138f,0x4fa2,0xb5,0x39,0x17,0xb4,0xae,0xfb,0x1b,0xca>
{
public:

    /// Returns the number of messages.
    virtual Size get_messages_count() const = 0;

    /// Returns the number of error messages.
    virtual Size get_error_messages_count() const = 0;

    /// Returns the message at index or NULL, if no such index exists.
    virtual const IMessage* get_message(Size index) const = 0;

    /// Returns the error message at index or NULL, if no such index exists.
    virtual const IMessage* get_error_message(Size index) const = 0;


    /// Returns the number of supported options.
    virtual Size get_option_count() const = 0;

    /// Returns the option name at index.
    virtual const char* get_option_name(Size index) const = 0;

    /// Returns the option type name at index.
    virtual const char* get_option_type(const char* name) const = 0;

    /// Get a string option.
    /// \param name     option name
    /// \param value    pointer the option value is written to.
    /// \return
    ///                          -  0: Success.
    ///                          - -1: Invalid option name.
    ///                          - -2: The option type does not match the value type.
    virtual Sint32 get_option(const char* name, const char*& value) const = 0;

    /// Get a float option.
    /// \param name     option name
    /// \param value    pointer the option value is written to.
    /// \return
    ///                          -  0: Success.
    ///                          - -1: Invalid option name.
    ///                          - -2: The option type does not match the value type.
    virtual Sint32 get_option(const char* name, Float32& value) const = 0;

    /// Get a bool option.
    /// \param name     option name
    /// \param value    pointer the option value is written to.
    /// \return
    ///                          -  0: Success.
    ///                          - -1: Invalid option name.
    ///                          - -2: The option type does not match the value type.
    virtual Sint32 get_option(const char* name, bool& value) const = 0;

    /// Set a string option.
    /// \param name     option name
    /// \param value    option value.
    /// \return
    ///                          -  0: Success.
    ///                          - -1: Invalid option name.
    ///                          - -2: The option type does not match the value type.
    virtual Sint32 set_option(const char* name, const char* value) = 0;

    /// Set a float option.
    /// \param name     option name
    /// \param value    option value.
    /// \return
    ///                          -  0: Success.
    ///                          - -1: Invalid option name.
    ///                          - -2: The option type does not match the value type.
    virtual Sint32 set_option(const char* name, Float32 value) = 0;

    /// Set a bool option.
    /// \param name     option name
    /// \param value    option value.
    /// \return
    ///                          -  0: Success.
    ///                          - -1: Invalid option name.
    ///                          - -2: The option type does not match the value type.
    ///                          - -3: The value is invalid in the context of the option.
    virtual Sint32 set_option(const char* name, bool value) = 0;
};


/*@}*/ // end group mi_neuray_mdl_types

} // namespace neuraylib
} // namespace mi

#endif // MI_NEURAYLIB_IMDL_EXECUTION_CONTEXT_H
