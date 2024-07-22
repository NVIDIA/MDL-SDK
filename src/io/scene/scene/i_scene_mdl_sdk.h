/***************************************************************************************************
 * Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
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
/// \brief Stubs for the MDL SDK library.

#ifndef IO_SCENE_SCENE_I_SCENE_MDL_SDK_H
#define IO_SCENE_SCENE_I_SCENE_MDL_SDK_H

#include <base/system/main/i_module.h>

namespace MI {

namespace DB { class Database; }
namespace SYSTEM { class Module_registration_entry; }

namespace SCENE {

class Scene_module : public SYSTEM::IModule
{
public:
    // methods of SYSTEM::IModule

    bool init() { return true; }

    void exit() { }

    static const char* get_name() { return "SCENE"; }

    static SYSTEM::Module_registration_entry* get_instance();

    // own methods

    void register_db_elements( DB::Database* db);
};

} // namespace SCENE

} // namespace MI

#endif // IO_SCENE_SCENE_I_SCENE_MDL_SDK_H
