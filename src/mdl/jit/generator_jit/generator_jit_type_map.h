/******************************************************************************
 * Copyright (c) 2013-2020, NVIDIA CORPORATION. All rights reserved.
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
 *****************************************************************************/

#ifndef MDL_GENERATOR_JIT_TYPE_MAP_H
#define MDL_GENERATOR_JIT_TYPE_MAP_H 1

#include <cstring>

#include <mi/mdl/mdl_types.h>

#include <mdl/compiler/compilercore/compilercore_allocator.h>
#include <mdl/compiler/compilercore/compilercore_hash_ptr.h>
#include <mdl/compiler/compilercore/compilercore_cstring_hash.h>

#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DerivedTypes.h>

namespace llvm {
    class DIBuilder;
    class DIFile;
    class DIScope;
    class DISubroutineType;
    class DIType;
    class LLVMContext;
}

namespace mi {
namespace mdl {

class IType;
class IType_function;

///
/// A Mapper class to map MDL types to LLVM types.
///
class Type_mapper {
public:
    /// The Tag type, a 32bit (unsigned) type
    typedef int Tag;

    /// State mappings.
    enum State_mapping {
        SM_USE_BITANGENT          = 1 << 0,  ///< Use bitangent instead of tangent_u and tangent_v.
        SM_INCLUDE_UNIFORM_STATE  = 1 << 1,  ///< Include the uniform state.
        SM_USE_DERIVATIVES        = 1 << 2,  ///< Use derivative types.
        SM_INCLUDE_ARG_BLOCK_OFFS = 1 << 3,  ///< Include an argument block offset.
    };

    /// Supported state subsets.
    enum State_subset_mode {
        SSM_NO_STATE     = 0,      ///< Empty state subset for constant functions.
        SSM_ENVIRONMENT  = 1 << 0, ///< The subset for environment functions.
        SSM_CORE         = 1 << 1, ///< The subset for core functions.
        SSM_FULL_SET     = 3,      ///< The full state set (debugging only).
    };

    /// Supported type mapping modes.
    enum Type_mapping_mode {
        TM_ALL_SCALAR        =  0,    ///< Do not use vector types at all, MDL vectors and matrices
                                      ///  are translated to arrays of scalar type.
        TM_SMALL_VECTORS     =  1,    ///< Use LLVM vector types only for MDL vector types, MDL
                                      ///  matrices are translated into arrays of vectors.
        TM_BIG_VECTORS       =  2,    ///< Use LLVM vector types for MDL vector and matrix types,
                                      ///  creating vectors of up to 16 elements.

        TM_VECTOR_MASK       =  3,    ///< The mask for vector modes.

        TM_BOOL1_SUPPORTED   =  4,    ///< If set, backend supports automatic conversions from i1

        TM_BIG_VECTOR_RETURN =  8,    ///< If set, backend supports return of vectors > 4

        TM_STRINGS_ARE_IDS   = 16,    ///< Map strings to IDs

        TM_NO_REFERENCE      = 32,    ///< never pass-by-reference/reference return

        TM_NO_POINTER_SUPPORT = 64,   ///< The target does not support pointers

        /// The mode for native x86 compilation.
        TM_NATIVE_X86 = TM_BIG_VECTORS | TM_BIG_VECTOR_RETURN | TM_BOOL1_SUPPORTED,

        /// The mode for PTX compilation.
        TM_PTX        = TM_BIG_VECTORS,

        /// The mode for HLSL compilation.
        TM_HLSL       = TM_SMALL_VECTORS | TM_NO_REFERENCE | TM_NO_POINTER_SUPPORT |
                        TM_BOOL1_SUPPORTED
    };

    /// array_desc<T> access indexes.
    enum Array_desc_indexes {
        ARRAY_DESC_BASE = 0,
        ARRAY_DESC_SIZE = 1
    };

    /// State fields.
    enum State_field {
        // Environment context
        STATE_ENV_DIRECTION,                ///< Result of state::direction().
        STATE_ENV_RO_DATA_SEG,              ///< Pointer to the read only data segment.

        // Core context
        STATE_CORE_NORMAL,                  ///< Result of state::normal().
        STATE_CORE_GEOMETRY_NORMAL,         ///< Result of state::geometry_normal().
        STATE_CORE_POSITION,                ///< Result of state::position().
        STATE_CORE_ANIMATION_TIME,          ///< Result of state::animation_time().
        STATE_CORE_TEXTURE_COORDINATE,      ///< Array of texture_coordinates.
        STATE_CORE_TANGENT_U,               ///< Array to tangent_u.
        STATE_CORE_TANGENT_V,               ///< Array to tangent_v.
        STATE_CORE_BITANGENTS,              ///< Array to compute tangents from.
        STATE_CORE_TEXT_RESULTS,            ///< texture results lookup table.
        STATE_CORE_RO_DATA_SEG,             ///< Pointer to the read only data segment.

        // the following fields are available if the uniform state is included only
        STATE_CORE_W2O_TRANSFORM,           ///< The world-to-object transform matrix.
        STATE_CORE_O2W_TRANSFORM,           ///< The object-to-world transform matrix.
        STATE_CORE_OBJECT_ID,               ///< Result of state::object_id().
        STATE_CORE_METERS_PER_SCENE_UNIT,   ///< Result of state::meters_per_scene_unit()
                                            ///< and reciprocal of state::scene_units_per_meter().

        // only available for HLSL
        STATE_CORE_ARG_BLOCK_OFFSET,        ///< The argument block offset.
    };

    /// Exception state access index.
    enum Exc_state_indexes {
        EXC_STATE_HANDLER = 0,  ///< The exception handler.
        EXC_STATE_ABORT   = 1   ///< Points to the abort location.
    };

    /// Texture handler access index.
    enum Tex_handler_indexes {
        TH_VTABLE = 0,   ///< The vtable.
    };

    // Res_data_pair access index.
    enum Res_data_pair_indexes {
        RDP_SHARED_DATA = 0,  ///< The shared resource data.
        RDP_THREAD_DATA = 1,  ///< The thread resource data.
    };

    /// Texture handler vtable access index.
    enum Tex_handler_vtable_index {
        THV_tex_lookup_float4_2d,           ///< tex_lookup_float4_2d()
        THV_tex_lookup_float3_2d,           ///< tex_lookup_float3_2d()
        THV_tex_texel_float4_2d,            ///< tex_texel_float4_2d()
        THV_tex_lookup_float4_3d,           ///< tex_lookup_float4_3d()
        THV_tex_lookup_float3_3d,           ///< tex_lookup_float3_3d()
        THV_tex_texel_float4_3d,            ///< tex_texel_float4_3d()
        THV_tex_lookup_float4_cube,         ///< tex_lookup_float4_cube()
        THV_tex_lookup_float3_cube,         ///< tex_lookup_float3_cube()
        THV_tex_resolution_2d,              ///< tex_resolution_2d()
        THV_tex_resolution_3d,              ///< tex_resolution_3d()
        THV_tex_texture_isvalid,            ///< tex_texture_isvalid()
        THV_light_profile_power,            ///< df_light_profile_power()
        THV_light_profile_maximum,          ///< df_light_profile_maximum()
        THV_light_profile_isvalid,          ///< df_light_profile_isvalid()
        THV_light_profile_evaluate,         ///< df_light_profile_evaluate()
        THV_light_profile_sample,           ///< df_light_profile_sample()
        THV_light_profile_pdf,              ///< df_light_profile_pdf()
        THV_bsdf_measurement_isvalid,       ///< df_bsdf_measurement_isvalid()
        THV_bsdf_measurement_resolution,    ///< df_bsdf_measurement_resolution()
        THV_bsdf_measurement_evaluate,      ///< df_bsdf_measurement_evaluate()
        THV_bsdf_measurement_sample,        ///< df_bsdf_measurement_sample()
        THV_bsdf_measurement_pdf,           ///< df_bsdf_measurement_pdf()
        THV_bsdf_measurement_albedos,       ///< df_bsdf_measurement_albedos()
        THV_scene_data_isvalid,             ///< scene_data_isvalid()
        THV_scene_data_lookup_float,        ///< scene_data_lookup_float()
        THV_scene_data_lookup_float2,       ///< scene_data_lookup_float2()
        THV_scene_data_lookup_float3,       ///< scene_data_lookup_float3()
        THV_scene_data_lookup_float4,       ///< scene_data_lookup_float4()
        THV_scene_data_lookup_int,          ///< scene_data_lookup_int()
        THV_scene_data_lookup_int2,         ///< scene_data_lookup_int2()
        THV_scene_data_lookup_int3,         ///< scene_data_lookup_int3()
        THV_scene_data_lookup_int4,         ///< scene_data_lookup_int4()
        THV_scene_data_lookup_color,        ///< scene_data_lookup_color()
        THV_scene_data_lookup_deriv_float,  ///< scene_data_lookup_deriv_float()
        THV_scene_data_lookup_deriv_float2, ///< scene_data_lookup_deriv_float2()
        THV_scene_data_lookup_deriv_float3, ///< scene_data_lookup_deriv_float3()
        THV_scene_data_lookup_deriv_float4, ///< scene_data_lookup_deriv_float4()
        THV_scene_data_lookup_deriv_color,  ///< scene_data_lookup_deriv_color()
        THV_LAST
    };

    /// Texture_attritube_entry access index.
    enum Texture_attribute_entry_index {
        TAE_ISVALID  = 0,
        TAE_WIDTH    = 1,
        TAE_HEIGHT   = 2,
        TAE_DEPTH    = 3
    };

    /// Light_profile_attritube_entry access index.
    enum Light_profile_attribute_entry_index {
        LAE_ISVALID  = 0,
        LAE_POWER    = 1,
        LAE_MAXIMUM  = 2
    };

    /// Bsdf_measuremente_attritube_entry access index.
    enum Bsdf_measurement_attribute_entry_index {
        BAE_ISVALID  = 0
    };

public:
    /// Constructor.
    ///
    /// \param alloc                the allocator
    /// \param context              the LLVM context to be used inside type construction
    /// \param data_layout          LLVM data layout info for the JIT mode
    /// \param state_mapping        how to map the MDL state
    /// \param tm_mode              the type mapping mode
    /// \param num_texture_spaces   the number of supported texture spaces
    /// \param num_texture_results  the number of texture result entries
    Type_mapper(
        IAllocator             *alloc,
        llvm::LLVMContext      &context,
        llvm::DataLayout const *data_layout,
        unsigned               state_mapping,
        Type_mapping_mode      tm_mode,
        unsigned               num_texture_spaces,
        unsigned               num_texture_results);

    /// Get the LLVM context of this type mapper
    llvm::LLVMContext &get_llvm_context() const { return m_context; }

    /// Returns true if strings are mapped to IDs.
    bool strings_mapped_to_ids() const {
        return (m_tm_mode & TM_STRINGS_ARE_IDS) != 0;
    }

    /// Returns true if the API structs don't contain any pointers.
    bool target_supports_pointers() const {
        return (m_tm_mode & TM_NO_POINTER_SUPPORT) == 0;
    }

    /// Returns true if the functions may use sret parameters.
    bool may_use_sret() const {
        return (m_tm_mode & TM_NO_REFERENCE) == 0;
    }

    /// Checks if bitangents are used.
    bool use_bitangents() const {
        return (m_state_mapping & SM_USE_BITANGENT) != 0;
    }

    /// Checks if the render state includes the uniform state.
    bool state_includes_uniform_state() const {
        return (m_state_mapping & SM_INCLUDE_UNIFORM_STATE) != 0;
    }

    /// Returns true if derivatives are used.
    bool use_derivatives() const {
        return (m_state_mapping & SM_USE_DERIVATIVES) != 0;
    }

    /// Returns true if the state includes an argument block offset.
    bool state_includes_arg_block_offset() const {
        return (m_state_mapping & SM_INCLUDE_ARG_BLOCK_OFFS) != 0;
    }

    /// Get the index of a state field in the current state struct.
    ///
    /// \param state_field  the requested state field
    ///
    /// \return the index of the state field inside the state struct depending on state options
    int get_state_index(State_field state_field);

    /// Get an LLVM type for an MDL type.
    ///
    /// \param context      the LLVM context to be used inside type construction
    /// \param type         the MDL type to lookup
    /// \param arr_size     if >= 0, the instantiated array size of type
    llvm::Type *lookup_type(
        llvm::LLVMContext &context,
        mdl::IType const  *type,
        int               arr_size = -1) const;

    /// Get an LLVM type with derivatives for an MDL type.
    /// This is always a struct type with the original type as first element and the
    /// two more elements of this type for the x and y derivative, respectively.
    ///
    /// \param type         the MDL type to lookup
    /// \param arr_size     if >= 0, the instantiated array size of type
    llvm::StructType *lookup_deriv_type(
        mdl::IType const  *type,
        int               arr_size = -1) const;

    /// Get an LLVM type with derivatives for an LLVM type.
    /// This is always a struct type with the original type as first element and the
    /// two more elements of this type for the x and y derivative, respectively.
    ///
    /// \param type         the MDL type to lookup
    llvm::StructType *lookup_deriv_type(llvm::Type *type) const;

    // TODO: Maybe this is not a reliable check, as other unidentified struct types might also match
    /// Checks if the given LLVM type is a derivative type.
    ///
    /// \param type  the type to check
    bool is_deriv_type(llvm::Type *type) const;

    /// Checks if the given MDL type is a derivative type.
    ///
    /// \param type  the type to check
    bool is_deriv_type(mi::mdl::IType const *type) const;

    /// Checks if the given MDL type is based on a floating point type.
    ///
    /// \param type  the type to check
    bool is_floating_point_based_type(IType const *type) const;

    /// Get the base value type of a derivative type.
    ///
    /// \param type  the MDL type
    ///
    /// \returns the base value type of the derivative type or NULL if it is not a derivative type
    mi::mdl::IType const *get_deriv_base_type(mi::mdl::IType const *type) const;

    /// Get the base value LLVM type of a derivative LLVM type.
    ///
    /// \param type  the LLVM type
    ///
    /// \returns the base value type of the derivative type or NULL if it is not a derivative type
    llvm::Type *get_deriv_base_type(llvm::Type *type) const;

    /// Skip to the base value type of a derivative type or just return the type itself for
    /// non-derivative types.
    ///
    /// \param type  the MDL type
    mi::mdl::IType const *skip_deriv_type(mi::mdl::IType const *type) const;

    /// Skip to the base value LLVM type of a derivative LLVM type or just return the type itself
    /// for non-derivative types.
    ///
    /// \param type  the LLVM type
    llvm::Type *skip_deriv_type(llvm::Type *type) const;

    /// Checks if a given MDL type needs struct return calling convention.
    ///
    /// \param type  the type to check
    bool need_reference_return(mi::mdl::IType const *type) const;

    /// Check if the given parameter type must be passed by reference.
    ///
    /// \param type   the type of the parameter
    bool is_passed_by_reference(mi::mdl::IType const *type) const;

    /// Get a pointer type from a base type.
    ///
    /// \param type  the base type
    static llvm::PointerType *get_ptr(llvm::Type *type);

    /// Get the LLVM void type.
    llvm::Type *get_void_type() const { return m_type_void; }

    /// Get the LLVM bool type.
    llvm::IntegerType *get_bool_type() const { return m_type_bool; }

    // map the i1 result to the bool type representation
    llvm::IntegerType *get_predicate_type() const { return m_type_predicate; };

    /// Get the LLVM char type.
    llvm::IntegerType *get_char_type() const { return m_type_char; }

    /// Get the LLVM char * type.
    llvm::PointerType *get_char_ptr_type() const { return get_ptr(m_type_char); }

    /// Get the LLVM int type.
    llvm::IntegerType *get_int_type() const { return m_type_int; }

    /// Get the LLVM size_t type.
    llvm::IntegerType *get_size_t_type() const { return m_type_size_t; }

    /// Get the LLVM float type.
    llvm::Type *get_float_type() const { return m_type_float; }

    /// Get the LLVM float * type.
    llvm::PointerType *get_float_ptr_type() const { return get_ptr(m_type_float); }

    /// Get the LLVM double type.
    llvm::Type *get_double_type() const { return m_type_double; }

    /// Get the LLVM color type.
    llvm::Type *get_color_type() const { return m_type_color; }

    /// Get the LLVM tag type.
    llvm::IntegerType *get_tag_type() const { return m_type_tag; }

    /// Get the LLVM C-string type.
    llvm::PointerType *get_cstring_type() const { return m_type_cstring; }

    /// Get the LLVM string type.
    llvm::Type *get_string_type() const {
        return strings_mapped_to_ids() ? (llvm::Type *)m_type_tag : (llvm::Type *)m_type_cstring;
    }

    /// Get the LLVM void * type.
    llvm::PointerType *get_void_ptr_type() const { return m_type_void_ptr; }

    /// Get the LLVM execution context type.
    llvm::StructType *get_exec_ctx_type() const { return m_type_exec_ctx; }

    /// Get the LLVM execution context * type.
    llvm::PointerType *get_exec_ctx_ptr_type() const { return m_type_exec_ctx_ptr; }


    /// Get the LLVM State * type for the given state context.
    llvm::PointerType *get_state_ptr_type(State_subset_mode mode);

    /// Get the LLVM Exception_state * type.
    llvm::PointerType *get_exc_state_ptr_type() const { return m_type_exc_state_ptr; }

    /// Get the Tex_handler * type.
    llvm::PointerType *get_tex_handler_ptr_type() const { return m_type_core_tex_handler_ptr; }

    /// Get the LLVM int[2] type.
    llvm::ArrayType *get_arr_int_2_type() const { return m_type_arr_int_2; }

    /// Get the LLVM int[2] * type.
    llvm::PointerType *get_arr_int_2_ptr_type() const { return get_ptr(m_type_arr_int_2); }

    /// Get the LLVM int[3] type.
    llvm::ArrayType *get_arr_int_3_type() const { return m_type_arr_int_3; }

    /// Get the LLVM int[3] * type.
    llvm::PointerType *get_arr_int_3_ptr_type() const { return get_ptr(m_type_arr_int_3); }

    /// Get the LLVM int[4] type.
    llvm::ArrayType *get_arr_int_4_type() const { return m_type_arr_int_4; }

    /// Get the LLVM int[4] * type.
    llvm::PointerType *get_arr_int_4_ptr_type() const { return get_ptr(m_type_arr_int_4); }

    /// Get the LLVM float[2] type.
    llvm::ArrayType *get_arr_float_2_type() const { return m_type_arr_float_2; }

    /// Get the LLVM (float[2]) * type.
    llvm::PointerType *get_arr_float_2_ptr_type() const { return get_ptr(m_type_arr_float_2); }

    /// Get the LLVM float[3] type.
    llvm::ArrayType *get_arr_float_3_type() const { return m_type_arr_float_3; }

    /// Get the LLVM (float[3]) * type.
    llvm::PointerType *get_arr_float_3_ptr_type() const { return get_ptr(m_type_arr_float_3); }

    /// Get the LLVM float[4] type.
    llvm::ArrayType *get_arr_float_4_type() const { return m_type_arr_float_4; }

    /// Get the LLVM (float[4]) * type.
    llvm::PointerType *get_arr_float_4_ptr_type() const { return get_ptr(m_type_arr_float_4); }

    /// Get the LLVM derivative type for float.
    llvm::Type *get_deriv_float_type() const { return m_type_deriv_float; }

    /// Get the LLVM derivative type for float2.
    llvm::StructType *get_deriv_float2_type() const { return m_type_deriv_float2; }

    /// Get the LLVM derivative type for float[2].
    llvm::StructType *get_deriv_arr_float_2_type() const { return m_type_deriv_arr_float_2; }

    /// Get the LLVM derivative type for float3.
    llvm::StructType *get_deriv_float3_type() const { return m_type_deriv_float3; }

    /// Get the LLVM derivative type for float[3].
    llvm::StructType *get_deriv_arr_float_3_type() const { return m_type_deriv_arr_float_3; }

    /// Get the LLVM derivative type for float[4].
    llvm::StructType *get_deriv_arr_float_4_type() const { return m_type_deriv_arr_float_4; }

    /// Get the LLVM float3x3 type.
    llvm::Type *get_float3x3_type() const { return m_type_float3x3; }

    /// Get the LLVM float4x4 type.
    llvm::Type *get_float4x4_type() const { return m_type_float4x4; }

    /// Get the LLVM float2 type.
    llvm::Type *get_float2_type() const { return m_type_float2; }

    /// Get the LLVM float2 * type.
    llvm::PointerType *get_float2_ptr_type() const { return get_ptr(m_type_float2); }

    /// Get the LLVM float3 type.
    llvm::Type *get_float3_type() const { return m_type_float3; }

    /// Get the LLVM float3 * type.
    llvm::PointerType *get_float3_ptr_type() const { return get_ptr(m_type_float3); }

    /// Get the LLVM float4 type.
    llvm::Type *get_float4_type() const { return m_type_float4; }

    /// Get the LLVM float4 * type.
    llvm::PointerType *get_float4_ptr_type() const { return get_ptr(m_type_float4); }

    /// Get the LLVM int2 type.
    llvm::Type *get_int2_type() const { return m_type_int2; }

    /// Get the LLVM int3 type.
    llvm::Type *get_int3_type() const { return m_type_int3; }

    /// Get the LLVM Res_data_pair type.
    llvm::StructType *get_res_data_pair_type() const { return m_type_res_data_pair; }

    /// Get the LLVM Res_data_pair * type.
    llvm::PointerType *get_res_data_pair_ptr_type() const { return m_type_res_data_pair_ptr; }

    /// Get the LLVM core texture handler type.
    llvm::StructType *get_core_tex_handler_type() const { return m_type_core_tex_handler; }

    /// Get the LLVM core texture handler * type.
    llvm::PointerType *get_core_tex_handler_ptr_type() const { return m_type_core_tex_handler_ptr; }

    /// Get the texture attribute entry type.
    llvm::StructType *get_texture_attribute_entry_type() const {
        return m_type_texture_attribute_entry;
    }

    /// Get the texture attribute entry * type.
    llvm::PointerType *get_texture_attribute_entry_ptr_type() const {
        return m_type_texture_attribute_entry_ptr;
    }

    /// Get the light profile attribute entry type.
    llvm::StructType *get_light_profile_attribute_entry_type() const {
        return m_type_light_profile_attribute_entry;
    }

    /// Get the light profile attribute entry * type.
    llvm::PointerType *get_light_profile_attribute_entry_ptr_type() const {
        return m_type_light_profile_attribute_entry_ptr;
    }

    /// Get the bsdf measurement attribute entry type.
    llvm::StructType *get_bsdf_measurement_attribute_entry_type() const {
        return m_type_bsdf_measurement_attribute_entry;
    }

    /// Get the bsdf measurement attribute entry * type.
    llvm::PointerType *get_bsdf_measurement_attribute_entry_ptr_type() const {
        return m_type_bsdf_measurement_attribute_entry_ptr;
    }

    /// Get the Optix rti_internal_typeinfo::rti_typeinfo type.
    llvm::StructType *get_optix_typeinfo_type();

    /// Get the debug info type for an MDL type.
    ///
    /// \param diBuilder   the debug info builder
    /// \param file        the file of this type
    /// \param scope       the scope for this type
    /// \param type        the MDL type
    llvm::DIType *get_debug_info_type(
        llvm::DIBuilder      *diBuilder,
        llvm::DIFile         *file,
        llvm::DIScope        *scope,
        mi::mdl::IType const *type) const;

    /// Get the debug info type for an MDL function type type.
    ///
    /// \param diBuilder   the debug info builder
    /// \param file        the file of this function type
    /// \param type        the MDL function type
    llvm::DISubroutineType *get_debug_info_type(
        llvm::DIBuilder               *diBuilder,
        llvm::DIFile                  *file,
        mi::mdl::IType_function const *type) const;

private:
    /// Construct the State type for the environment context.
    ///
    /// \param context        the LLVM context this type is build belongs to
    /// \param float3_type    the LLVM float3 type
    llvm::StructType *construct_state_environment_type(
        llvm::LLVMContext      &context,
        llvm::Type             *float3_type);

    /// Construct the State type for the iray core context.
    ///
    /// \param context           the LLVM context this type is build belongs to
    /// \param int_type          the LLVM integer type
    /// \param float3_type       the LLVM float3 type
    /// \param float4_type       the LLVM float4 type
    /// \param float_type        the LLVM float type
    /// \param byte_ptr_type     the LLVM byte * type
    /// \param deriv_float3_type the LLVM derivative float3 type
    /// \param float4x4_type     the LLVM float4x4 type
    /// \param num_texture_spaces   the number of supported texture spaces
    /// \param num_texture_results  the number of texture result entries
    llvm::StructType *construct_state_core_type(
        llvm::LLVMContext      &context,
        llvm::Type             *int_type,
        llvm::Type             *float3_type,
        llvm::Type             *float4_type,
        llvm::Type             *float_type,
        llvm::Type             *byte_ptr_type,
        llvm::Type             *deriv_float3_type,
        llvm::Type             *float4x4_type,
        unsigned               num_texture_spaces,
        unsigned               num_texture_results);

    /// Construct the exception state type.
    ///
    /// \param context        the LLVM context this type is build belongs to
    /// \param void_ptr_type  the LLVM void * type
    /// \param atom32_type    the LLVM mi::base::Atom32 type
    static llvm::StructType *construct_exception_state_type(
        llvm::LLVMContext &context,
        llvm::Type        *void_ptr_type,
        llvm::Type        *atom32_type);

    /// Construct the Res_data_pair type.
    ///
    /// \param context        the LLVM context this type is build belongs to
    llvm::StructType *construct_res_data_pair_type(
        llvm::LLVMContext      &context);

    /// Construct the exec_ctx type.
    static llvm::StructType *construct_exec_ctx_type(
        llvm::LLVMContext      &context,
        llvm::Type             *state_core_ptr_type,
        llvm::Type             *res_data_pair_ptr_type,
        llvm::Type             *exc_state_ptr_type,
        llvm::Type             *void_ptr_type);


    /// Construct the core texture handler type.
    ///
    /// \param context        the LLVM context this type is build belongs to
    llvm::StructType *construct_core_texture_handler_type(
        llvm::LLVMContext &context);

    /// Construct the texture attribute entry type.
    ///
    /// \param context        the LLVM context this type is build belongs to
    llvm::StructType *construct_texture_attribute_entry_type(
        llvm::LLVMContext &context);

    /// Construct the light profile attribute entry type.
    ///
    /// \param context        the LLVM context this type is build belongs to
    llvm::StructType *construct_light_profile_attribuute_entry_type(
        llvm::LLVMContext &context);

    /// Construct the bsdf measurement attribute entry type.
    ///
    /// \param context        the LLVM context this type is build belongs to
    llvm::StructType *construct_bsdf_measurement_attribuute_entry_type(
        llvm::LLVMContext &context);

private:
    /// The state mapping.
    unsigned m_state_mapping;

    /// The context all types belong too.
    llvm::LLVMContext &m_context;

    /// The used data layout.
    llvm::DataLayout m_data_layout;

    /// The type mapping mode.
    Type_mapping_mode m_tm_mode;

    // basic types
    llvm::IntegerType *m_type_size_t;
    llvm::Type        *m_type_void;
    llvm::PointerType *m_type_void_ptr;
    llvm::IntegerType *m_type_predicate;
    llvm::IntegerType *m_type_bool;
    llvm::Type        *m_type_bool2;
    llvm::Type        *m_type_bool3;
    llvm::Type        *m_type_bool4;
    llvm::IntegerType *m_type_int;
    llvm::Type        *m_type_int2;
    llvm::Type        *m_type_int3;
    llvm::Type        *m_type_int4;
    llvm::Type        *m_type_float;
    llvm::Type        *m_type_float2;
    llvm::Type        *m_type_float3;
    llvm::Type        *m_type_float4;
    llvm::Type        *m_type_double;
    llvm::Type        *m_type_double2;
    llvm::Type        *m_type_double3;
    llvm::Type        *m_type_double4;
    llvm::Type        *m_type_color;

    llvm::ArrayType   *m_type_arr_int_2;
    llvm::ArrayType   *m_type_arr_int_3;
    llvm::ArrayType   *m_type_arr_int_4;

    llvm::ArrayType   *m_type_arr_float_2;
    llvm::ArrayType   *m_type_arr_float_3;
    llvm::ArrayType   *m_type_arr_float_4;

    llvm::StructType  *m_type_deriv_float;
    llvm::StructType  *m_type_deriv_float2;
    llvm::StructType  *m_type_deriv_float3;
    llvm::StructType  *m_type_deriv_arr_float_2;
    llvm::StructType  *m_type_deriv_arr_float_3;
    llvm::StructType  *m_type_deriv_arr_float_4;

    llvm::Type        *m_type_float2x2;
    llvm::Type        *m_type_float3x2;
    llvm::Type        *m_type_float4x2;
    llvm::Type        *m_type_float2x3;
    llvm::Type        *m_type_float3x3;
    llvm::Type        *m_type_float4x3;
    llvm::Type        *m_type_float2x4;
    llvm::Type        *m_type_float3x4;
    llvm::Type        *m_type_float4x4;

    llvm::Type        *m_type_double2x2;
    llvm::Type        *m_type_double3x2;
    llvm::Type        *m_type_double4x2;
    llvm::Type        *m_type_double2x3;
    llvm::Type        *m_type_double3x3;
    llvm::Type        *m_type_double4x3;
    llvm::Type        *m_type_double2x4;
    llvm::Type        *m_type_double3x4;
    llvm::Type        *m_type_double4x4;

    llvm::IntegerType *m_type_char;
    llvm::PointerType *m_type_cstring;
    llvm::IntegerType *m_type_tag;

    llvm::StructType  *m_type_state_environment;
    llvm::PointerType *m_type_state_environment_ptr;

    llvm::StructType  *m_type_state_core;
    llvm::PointerType *m_type_state_core_ptr;

    llvm::StructType  *m_type_exc_state;
    llvm::PointerType *m_type_exc_state_ptr;

    llvm::StructType  *m_type_res_data_pair;
    llvm::PointerType *m_type_res_data_pair_ptr;

    llvm::StructType  *m_type_exec_ctx;
    llvm::PointerType *m_type_exec_ctx_ptr;


    llvm::StructType  *m_type_core_tex_handler;
    llvm::PointerType *m_type_core_tex_handler_ptr;

    llvm::StructType  *m_type_texture_attribute_entry;
    llvm::PointerType *m_type_texture_attribute_entry_ptr;

    llvm::StructType  *m_type_light_profile_attribute_entry;
    llvm::PointerType *m_type_light_profile_attribute_entry_ptr;

    llvm::StructType  *m_type_bsdf_measurement_attribute_entry;
    llvm::PointerType *m_type_bsdf_measurement_attribute_entry_ptr;

    llvm::Type        *m_float_matrix[9];
    llvm::Type        *m_double_matrix[9];

    llvm::StructType  *m_type_optix_type_info;

    typedef ptr_hash_map<
        const char,
        llvm::Type *,
        cstring_hash,
        cstring_equal_to>::Type Type_struct_map;

    /// The type map cache: Stores mappings from MDL struct types to LLVM ones.
    mutable Type_struct_map m_type_struct_cache;

    typedef ptr_hash_map<llvm::Type, llvm::StructType *>::Type Deriv_type_map;

    /// The derivative type cache: Stores mappings from LLVM types to LLVM derivative types.
    mutable Deriv_type_map m_deriv_type_cache;

    typedef ptr_hash_set<llvm::Type>::Type Deriv_type_set;

    /// Stores all derivatives types to allow checking whether the type is a derivative type.
    /// TODO: Probably, this approach does not work for unidentified types.
    ///    Maybe switching to identified types isn't a problem though...
    mutable Deriv_type_set m_deriv_type_set;

    /// An array type cache key.
    struct Array_type_cache_key {
        /// Constructor.
        Array_type_cache_key(
            mdl::IType const *type = NULL,
            int              arr_size = -1)
        : type(type), arr_size(arr_size)
        {
        }

        mdl::IType const *type;
        int              arr_size;
    };

    /// Hash an Type_cache_key.
    struct Array_type_cache_hash {
        size_t operator()(Array_type_cache_key const &p) const {
            Hash_ptr<IType const>      hash_type;
            return (hash_type(p.type) << 6) + 7u * p.arr_size;
        }
    };

    /// Compare an Type_cache_key.
    struct Array_type_cache_equal {
        unsigned operator()(Array_type_cache_key const &a, Array_type_cache_key const &b) const {
            return a.type == b.type && a.arr_size == b.arr_size;
        }
    };

    typedef hash_map<
        Array_type_cache_key,
        llvm::Type *,
        Array_type_cache_hash,
        Array_type_cache_equal
    >::Type Type_array_map;

    /// The array type map cache: Stores mappings from MDL array types to LLVM ones.
    mutable Type_array_map m_type_arr_cache;
};

}  // mdl
}  // mi

#endif

