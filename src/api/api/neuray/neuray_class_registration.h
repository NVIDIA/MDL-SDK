/***************************************************************************************************
 * Copyright (c) 2010-2020, NVIDIA CORPORATION. All rights reserved.
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
 ** \brief Header for the class registration.
 **/

#ifndef API_API_NEURAY_NEURAY_CLASS_REGISTRATION_H
#define API_API_NEURAY_NEURAY_CLASS_REGISTRATION_H

#include <boost/core/noncopyable.hpp>

#include <mi/base/handle.h>


namespace mi { namespace neuraylib { class IExtension_api; } }

namespace MI {

namespace NEURAY {

class Class_factory;

/// Performs the registration of all built-in classes.
///
/// This functionality is in a translation unit of its own due to the large amount of dependencies.
class Class_registration : public boost::noncopyable
{
public:

    /// Registers all built-in classes, part 1.
    ///
    /// Part 1 encompasses all classes derived from mi::IData. They do not need the database
    /// or the deserialization manager and can be used before neuray has been started.
    static void register_classes_part1( Class_factory* factory);

    /// Registers all built-in classes, part 2.
    ///
    /// Part 2 encompasses the API wrappers for DB elements and related classes. They need the
    /// database or the deserialization manager and can only be user after neuray has been started.
    static void register_classes_part2( Class_factory* factory);

    /// Registers all built-in structure declarations
    static void register_structure_declarations( Class_factory* factory);

    /// Unregisters all built-in structure declarations
    static void unregister_structure_declarations( Class_factory* factory);

    /// Indicates whether \p name is the name of a predefined structure declaration
    static bool is_predefined_structure_declaration( const char* name);

    /// Registers all built-in importers
    static void register_importers( mi::neuraylib::IExtension_api* extension_api);

    /// Unregisters all built-in importers
    static void unregister_importers( mi::neuraylib::IExtension_api* extension_api);

    /// Registers all built-in exporters
    static void register_exporters( mi::neuraylib::IExtension_api* extension_api);

    /// Unregisters all built-in exporters
    static void unregister_exporters( mi::neuraylib::IExtension_api* extension_api);

private:

};

} // namespace NEURAY

} // namespace MI

#endif // API_API_NEURAY_NEURAY_CLASS_REGISTRATION_H

