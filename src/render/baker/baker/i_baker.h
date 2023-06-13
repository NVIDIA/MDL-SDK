//*****************************************************************************
// Copyright 2023 NVIDIA Corporation. All rights reserved.
//*****************************************************************************
/// \file i_baker.h
/// \brief The definition of the interface of the Baker_module
//*****************************************************************************

#ifndef RENDER_BAKER_BAKER_I_BAKER_H
#define RENDER_BAKER_BAKER_I_BAKER_H

#include <mi/neuraylib/imdl_distiller_api.h>
#include <mi/neuraylib/typedefs.h>

#include <base/system/main/i_module.h>
#include <string>

namespace mi { namespace neuraylib { class ITarget_code; } }

namespace MI {

namespace DB { class Transaction; }
namespace MDL { class Mdl_compiled_material; class Mdl_function_call; }
namespace SYSTEM { class Module_registration_entry; }

namespace BAKER {

class IBaker_code : public
    mi::base::Interface_declare<0xc3880444,0x1c95,0x4ed0,0xbb,0xa3,0x1d,0x72,0x49,0xdd,0xe2,0x3e,
                                mi::base::IInterface>
{
public:
    virtual mi::Uint32 get_used_gpu_device_id() const = 0;

    virtual const mi::neuraylib::ITarget_code* get_gpu_target_code() const = 0;

    virtual const mi::neuraylib::ITarget_code* get_cpu_target_code() const = 0;
};

enum Baker_state_flags {
    BAKER_STATE_POSITION_DIRECTION = 1u << 0  // state::position() is direction for spherical environment map
};

class Baker_module: public SYSTEM::IModule
{
public:

    /// Result of a bake_constant() call.
    union Constant_result {
        bool                    b;
        mi::Float32             f;
        mi::Float32_3::Pod_type v;
        mi::Spectrum_struct     s;
    };

    static const char* get_name() { return "BAKER"; }

    static SYSTEM::Module_registration_entry* get_instance();

    virtual const IBaker_code* create_baker_code(
        DB::Transaction* transaction,
        const MDL::Mdl_compiled_material* compiled_material,
        const char* path,
        mi::neuraylib::Baker_resource resource,
        mi::Uint32 gpu_device_id,
        std::string& pixel_type,
        bool& is_uniform) const = 0;

    virtual const IBaker_code* create_environment_baker_code(
        DB::Transaction* transaction,
        const MDL::Mdl_function_call* environment_function,
        mi::neuraylib::Baker_resource resource,
        mi::Uint32 gpu_device_id,
        bool &is_uniform) const = 0;

    /// Bake a texture.
    ///
    /// \return  0  on success
    /// \return -1  execution error
    virtual mi::Sint32 bake_texture(
        DB::Transaction* transaction,
        const IBaker_code* baker_code,
        mi::neuraylib::ICanvas* texture,
        mi::Uint32 samples,
        mi::Uint32 state_flags = 0) const = 0;

    /// Bake a constant (aka constant texture).
    ///
    /// \return  0  on success
    /// \return -1  execution error
    virtual mi::Sint32 bake_constant(
        DB::Transaction* transaction,
        const IBaker_code* baker_code,
        Constant_result& constant,
        mi::Uint32 samples,
        const char *pixel_type) const = 0;
};

} // namespace BAKER

} // namespace MI

#endif // RENDER_BAKER_BAKER_I_BAKER_H
