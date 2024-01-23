/******************************************************************************
 * Copyright (c) 2015-2024, NVIDIA CORPORATION. All rights reserved.
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

#include "pch.h"

#include "compiler_glsl_version.h"
#include "compiler_glsl_assert.h"

namespace mi {
namespace mdl {
namespace glsl {

/// Map extension name to enum value.
///
/// \param name  the extension name
///
/// \return -1 if extension is unknown
static int glsl_extension(char const *name)
{
    if (strncmp(name, "GL_", 3) != 0) {
        return -1;
    }
    name += 3;

    if (strncmp(name, "OES_", 4) == 0) {
        name += 4;

        if (strcmp(name, "texture_3D") == 0) {
            return GL_OES_texture_3D;
        }
        if (strcmp(name, "standard_derivatives") == 0) {
            return GL_OES_standard_derivatives;
        }
        if (strcmp(name, "EGL_image_external") == 0) {
            return GL_OES_EGL_image_external;
        }

        return -1;
    } else if (strncmp(name, "EXT_", 4) == 0) {
        name += 4;

        if (strcmp(name, "frag_depth") == 0) {
            return GL_EXT_frag_depth;
        }
        if (strcmp(name, "shader_texture_lod") == 0) {
            return GL_EXT_shader_texture_lod;
        }
        if (strcmp(name, "shader_implicit_conversions") == 0) {
            return GL_EXT_shader_implicit_conversions;
        }

        return -1;
    } else if (strncmp(name, "3DL_", 4) == 0) {
        name += 4;

        if (strcmp(name, "array_objects") == 0) {
            return GL_3DL_array_objects;
        }

        return -1;
    } else if (strncmp(name, "ARB_", 4) == 0) {
        name += 4;

        switch (name[0]) {
        case 'b':
            if (strcmp(name, "bindless_texture") == 0) {
                return GL_ARB_bindless_texture;
            }
            break;
        case 'c':
            if (strcmp(name, "cull_distance") == 0) {
                return GL_ARB_cull_distance;
            }
            break;
        case 'd':
            if (strcmp(name, "derivative_control") == 0) {
                return GL_ARB_derivative_control;
            }
            break;
        case 'e':
            if (strcmp(name, "enhanced_layouts") == 0) {
                return GL_ARB_enhanced_layouts;
            }
            if (strcmp(name, "explicit_attrib_location") == 0) {
                return GL_ARB_explicit_attrib_location;
            }
            break;
        case 'g':
            if (strcmp(name, "gpu_shader5") == 0) {
                return GL_ARB_gpu_shader5;
            }
            if (strcmp(name, "gpu_shader_fp64") == 0) {
                return GL_ARB_gpu_shader_fp64;
            }
            if (strcmp(name, "gpu_shader_int64") == 0) {
                return GL_ARB_gpu_shader_int64;
            }
            if (strcmp(name, "GL_ARB_shader_bit_encoding") == 0) {
                return GL_ARB_shader_bit_encoding;
            }
            break;
        case 's':
            if (strcmp(name, "shading_language_420pack") == 0) {
                return GL_ARB_shading_language_420pack;
            }
            if (strcmp(name, "separate_shader_objects") == 0) {
                return GL_ARB_separate_shader_objects;
            }
            if (strcmp(name, "shader_texture_lod") == 0) {
                return GL_ARB_shader_texture_lod;
            }
            if (strcmp(name, "shader_image_load_store") == 0) {
                return GL_ARB_shader_image_load_store;
            }
            if (strcmp(name, "shader_atomic_counters") == 0) {
                return GL_ARB_shader_atomic_counters;
            }
            if (strcmp(name, "shader_texture_image_samples") == 0) {
                return GL_ARB_shader_texture_image_samples;
            }
            if (strcmp(name, "shader_subroutine") == 0) {
                return GL_ARB_shader_subroutine;
            }
            if (strcmp(name, "arrays_of_arrays") == 0) {
                return GL_ARB_arrays_of_arrays;
            }
            if (strcmp(name, "shader_storage_buffer_object") == 0) {
                return GL_ARB_shader_storage_buffer_object;
            }
            break;
        case 't':
            if (strcmp(name, "texture_rectangle") == 0) {
                return GL_ARB_texture_rectangle;
            }
            if (strcmp(name, "texture_gather") == 0) {
                return GL_ARB_texture_gather;
            }
            if (strcmp(name, "tessellation_shader") == 0) {
                return GL_ARB_tessellation_shader;
            }
            if (strcmp(name, "texture_cube_map_array") == 0) {
                return GL_ARB_texture_cube_map_array;
            }
            break;
        case 'v':
            if (strcmp(name, "viewport_array") == 0) {
                return GL_ARB_viewport_array;
            }
            break;
        }

        return -1;
    } else if (strncmp(name, "KHR_", 4) == 0) {
        name += 4;
        if (strcmp(name, "vulkan_glsl") == 0) {
            return GL_KHR_vulkan_glsl;
        }
    } else if (strncmp(name, "NV_", 3) == 0) {
        name += 3;
        switch (name[0]) {
        case 'g':
            if (strcmp(name, "gpu_shader5") == 0) {
                return GL_NV_gpu_shader5;
            }
            break;
        case 's':
            if (strcmp(name, "shader_buffer_load") == 0) {
                return GL_NV_shader_buffer_load;
            }
            break;
        }

        return -1;
    } else if (strncmp(name, "AMD_", 4) == 0) {
        name += 4;
        switch (name[0]) {
        case 'g':
            if (strcmp(name, "gpu_shader_half_float") == 0) {
                return GL_AMD_gpu_shader_half_float;
            }
            break;
        }

        return -1;
    } else if (strncmp(name, "GOOGLE_", 7) == 0) {
        name += 7;

        switch (name[0]) {
        case 'c':
            if (strcmp(name, "cpp_style_line_directive") == 0) {
                return GL_GOOGLE_cpp_style_line_directive;
            }
            break;
        case 'i':
            if (strcmp(name, "include_directive") == 0) {
                return GL_GOOGLE_include_directive;
            }
            break;
        }

        return -1;
    }
    return -1;
}

// Constructor.
GLSLang_context::GLSLang_context(
    GLSL_language lang)
: m_lang(lang)
, m_version(GLSL_VERSION_1_10)
, m_profile(GLSL_PROFILE_CORE)
, m_extensions()
, m_strict(false)
{
    m_extension_behavior[GL_OES_texture_3D                  ] = EB_DISABLE;
    m_extension_behavior[GL_OES_standard_derivatives        ] = EB_DISABLE;
    m_extension_behavior[GL_OES_EGL_image_external          ] = EB_DISABLE;
    m_extension_behavior[GL_EXT_frag_depth                  ] = EB_DISABLE;
    m_extension_behavior[GL_EXT_shader_texture_lod          ] = EB_DISABLE;
    m_extension_behavior[GL_EXT_shader_implicit_conversions ] = EB_DISABLE;
    m_extension_behavior[GL_ARB_texture_rectangle           ] = EB_DISABLE;
    m_extension_behavior[GL_ARB_shading_language_420pack    ] = EB_DISABLE;
    m_extension_behavior[GL_ARB_texture_gather              ] = EB_DISABLE;
    m_extension_behavior[GL_ARB_gpu_shader5                 ] = EB_DISABLE;
    m_extension_behavior[GL_ARB_separate_shader_objects     ] = EB_DISABLE;
    m_extension_behavior[GL_ARB_tessellation_shader         ] = EB_DISABLE;
    m_extension_behavior[GL_ARB_enhanced_layouts            ] = EB_DISABLE;
    m_extension_behavior[GL_ARB_texture_cube_map_array      ] = EB_DISABLE;
    m_extension_behavior[GL_ARB_shader_texture_lod          ] = EB_DISABLE;
    m_extension_behavior[GL_ARB_explicit_attrib_location    ] = EB_DISABLE;
    m_extension_behavior[GL_ARB_shader_image_load_store     ] = EB_DISABLE;
    m_extension_behavior[GL_ARB_shader_atomic_counters      ] = EB_DISABLE;
    m_extension_behavior[GL_ARB_derivative_control          ] = EB_DISABLE;
    m_extension_behavior[GL_ARB_shader_texture_image_samples] = EB_DISABLE;
    m_extension_behavior[GL_ARB_viewport_array              ] = EB_DISABLE;
    m_extension_behavior[GL_ARB_cull_distance               ] = EB_DISABLE;
    m_extension_behavior[GL_ARB_gpu_shader_fp64             ] = EB_DISABLE;
    m_extension_behavior[GL_ARB_shader_subroutine           ] = EB_DISABLE;
    m_extension_behavior[GL_ARB_arrays_of_arrays            ] = EB_DISABLE;
    m_extension_behavior[GL_ARB_shader_storage_buffer_object] = EB_DISABLE;
    m_extension_behavior[GL_ARB_bindless_texture            ] = EB_DISABLE;
    m_extension_behavior[GL_ARB_gpu_shader_int64            ] = EB_DISABLE;
    m_extension_behavior[GL_ARB_shader_bit_encoding         ] = EB_DISABLE;
    m_extension_behavior[GL_3DL_array_objects               ] = EB_DISABLE;
    m_extension_behavior[GL_KHR_vulkan_glsl                 ] = EB_DISABLE;
    m_extension_behavior[GL_NV_shader_buffer_load           ] = EB_DISABLE;
    m_extension_behavior[GL_NV_gpu_shader5                  ] = EB_DISABLE;
    m_extension_behavior[GL_AMD_gpu_shader_half_float       ] = EB_DISABLE;
    m_extension_behavior[GL_GOOGLE_cpp_style_line_directive ] = EB_DISABLE;
    m_extension_behavior[GL_GOOGLE_include_directive        ] = EB_DISABLE;

    // ensure consistency
    for (size_t i = 0; i <= LAST_EXTENSION; ++i) {
        Extension_behavior eb = m_extension_behavior[i];

        enable_extension_no_update(GLSL_extensions(i), eb != EB_DISABLE);
    }
}

// Check, if we are in the ES profile.
bool GLSLang_context::es() const
{
    return (m_profile & GLSL_PROFILE_ES) != 0;
}

// Check, if we are in the ES profile and the version is lesser than a given version.
bool GLSLang_context::es_LT(unsigned version) const
{
    return es() && m_version < version;
}

// Check, if we are in the ES profile and the version is greater or equal
// than a given version.
bool GLSLang_context::es_GE(unsigned version) const
{
    return es() && m_version >= version;
}

// Check, if we are NOT in the ES profile and the version is lesser than given version.
bool GLSLang_context::non_es_LT(unsigned version) const
{
    return !es() && m_version < version;
}

// Check, if we are NOT in the ES profile and the version is greater or equal
// than given version.
bool GLSLang_context::non_es_GE(unsigned version) const
{
    return !es() && m_version >= version;
}

// Check, if the given extension is necessary, or if it is already included in the current
// GLSLang version
bool GLSLang_context::necessary_extension(GLSL_extensions ext) const
{
    switch (ext) {
    case mi::mdl::glsl::GL_ARB_gpu_shader_fp64:
        // double is supported in GL from 4.00
        return !non_es_GE(400);
    case mi::mdl::glsl::GL_ARB_arrays_of_arrays:
    case mi::mdl::glsl::GL_ARB_shader_storage_buffer_object:
        // builtin from ES 310, core 430
        return !(es_GE(310) || non_es_GE(430));
    default:
        return true;
    }
}

// Set the GLSL language version and profile.
bool GLSLang_context::set_version(unsigned version, unsigned profile)
{
    if (profile == GLSL_PROFILE_ES) {
        // for ES we support only three versions
        if (version != GLSL_VERSION_ES_1_00 &&
            version != GLSL_VERSION_ES_3_00 &&
            version != GLSL_VERSION_ES_3_10) {
            return false;
        }
    } else {
        switch (version) {
        case GLSL_VERSION_1_10:
        case GLSL_VERSION_1_20:
        case GLSL_VERSION_1_30:
        case GLSL_VERSION_1_40:
        case GLSL_VERSION_1_50:
        case GLSL_VERSION_3_30:
        case GLSL_VERSION_4_00:
        case GLSL_VERSION_4_10:
        case GLSL_VERSION_4_20:
        case GLSL_VERSION_4_30:
        case GLSL_VERSION_4_40:
        case GLSL_VERSION_4_50:
        case GLSL_VERSION_4_60:
            break;
        default:
            // unsupported
            return false;
        }
    }


    m_version = version;
    m_profile = profile;

    return true;
}

// Enable/Disable a GLSL extension but do not update the scanner..
void GLSLang_context::enable_extension_no_update(GLSL_extensions ext, bool enable)
{
    if (enable) {
        m_extensions.set_bit(ext);
    } else {
        m_extensions.clear_bit(ext);
    }
}

// Enable/Disable a GLSL extension and update the scanner if any was registered
void GLSLang_context::enable_extension(GLSL_extensions ext, bool enable)
{
    Static_bitset<LAST_EXTENSION + 1> old(m_extensions);

    enable_extension_no_update(ext, enable);

}

// Update an extension.
GLSLang_context::Ext_error_code GLSLang_context::update_extension(
    char const *ext_name,
    char const *behavior)
{
    Extension_behavior eb;

    if (strcmp("require", behavior) == 0) {
        eb = EB_REQUIRE;
    } else if (strcmp("enable", behavior) == 0) {
        eb = EB_ENABLE;
    } else if (strcmp("disable", behavior) == 0) {
        eb = EB_DISABLE;
    } else if (strcmp("warn", behavior) == 0) {
        eb = EB_WARN;
    } else {
        return EC_UNKNOWN_BEHAVIOR;
    }
    return update_extension(ext_name, eb);
}

// Update an extension.
GLSLang_context::Ext_error_code GLSLang_context::update_extension(
    char const         *ext_name,
    Extension_behavior eb)
{
    if (strcmp(ext_name, "all") == 0) {
        // the 'all' extension; apply it to every extension present
        if (eb == EB_REQUIRE || eb == EB_ENABLE) {
            return EC_ALL_BH_RESTRICTED;
        } else {
            // set the behavior to ALL extensions
            bool enabled = eb != EB_DISABLE;
            for (size_t i = 0; i < LAST_EXTENSION; ++i) {
                m_extension_behavior[i] = eb;
                enable_extension_no_update(GLSL_extensions(i), enabled);
            }
        }

        return EC_OK;
    }
    int i = glsl_extension(ext_name);
    if (i < 0) {
        return eb == EB_REQUIRE ? EC_UNSUPPORTED_EXTENSION_REQUIRED: EC_UNSUPPORTED_EXTENSION;
    }
    GLSL_extensions ext = GLSL_extensions(i);

    m_extension_behavior[ext] = eb;

    enable_extension(ext, eb != EB_DISABLE);

    return EC_OK;
}

// Get the name of an extension.
char const *GLSLang_context::get_extension_name(
    GLSL_extensions ext)
{
#define CASE(ext)  case ext: return #ext;
    switch (ext) {
    CASE(GL_OES_texture_3D)
    CASE(GL_OES_standard_derivatives)
    CASE(GL_OES_EGL_image_external)
    CASE(GL_EXT_frag_depth)
    CASE(GL_EXT_shader_texture_lod)
    CASE(GL_EXT_shader_implicit_conversions)
    CASE(GL_ARB_texture_rectangle)
    CASE(GL_ARB_shading_language_420pack)
    CASE(GL_ARB_texture_gather)
    CASE(GL_ARB_gpu_shader5)
    CASE(GL_ARB_separate_shader_objects)
    CASE(GL_ARB_tessellation_shader)
    CASE(GL_ARB_enhanced_layouts)
    CASE(GL_ARB_texture_cube_map_array)
    CASE(GL_ARB_shader_texture_lod)
    CASE(GL_ARB_explicit_attrib_location)
    CASE(GL_ARB_shader_image_load_store)
    CASE(GL_ARB_shader_atomic_counters)
    CASE(GL_ARB_derivative_control)
    CASE(GL_ARB_shader_texture_image_samples)
    CASE(GL_ARB_viewport_array)
    CASE(GL_ARB_cull_distance)
    CASE(GL_ARB_gpu_shader_fp64)
    CASE(GL_ARB_shader_subroutine)
    CASE(GL_ARB_arrays_of_arrays)
    CASE(GL_ARB_shader_storage_buffer_object)
    CASE(GL_ARB_bindless_texture)
    CASE(GL_ARB_gpu_shader_int64)
    CASE(GL_ARB_shader_bit_encoding)
    CASE(GL_3DL_array_objects)
    CASE(GL_KHR_vulkan_glsl)
    CASE(GL_NV_shader_buffer_load)
    CASE(GL_NV_gpu_shader5)
    CASE(GL_AMD_gpu_shader_half_float)
    CASE(GL_GOOGLE_cpp_style_line_directive)
    CASE(GL_GOOGLE_include_directive)
    }
    MDL_ASSERT(!"unsupported extension");
    return "unknown";
#undef CASE
}


// Returns true if the GLSLang version requires an explicit #version directive.
bool GLSLang_context::needs_explicit_version() const
{
    return m_version != GLSL_VERSION_1_10;
}

// Returns true if the GLSLang version requires an explicit profile.
bool GLSLang_context::needs_explicit_profile() const
{
    if (m_version < GLSL_VERSION_1_50) {
        return false;
    }
    if (m_version == GLSL_VERSION_ES_3_00 || m_version == GLSL_VERSION_ES_3_10) {
        return true;
    }
    // core is the default
    if (m_profile == GLSL_PROFILE_CORE) {
        return false;
    }
    return true;
}

// Returns true if the GLSLang version has the double type.
bool GLSLang_context::has_double_type() const
{
    // double is only supported in GL from 4.00 or with the GL_ARB_gpu_shader_fp64 extension
    return non_es_GE(400) || has_extension(GL_ARB_gpu_shader_fp64);
}

// Returns true if the GLSLang version has explicit sized integer types.
bool GLSLang_context::has_explicit_sized_int_types() const
{
    // explicit sized integer types are supported with GL_NV_gpu_shader5 extension
    return has_extension(GL_NV_gpu_shader5);
}

// Returns true if the GLSLang version has explicit sized float types.
bool GLSLang_context::has_explicit_sized_float_types() const
{
    // explicit sized float types are supported with GL_NV_gpu_shader5 extension
    return has_extension(GL_NV_gpu_shader5);
}

// Returns true if the GLSLang version has the 64bit integer types.
bool GLSLang_context::has_int64_types() const
{
    // 64bit integer types are supported with GL_ARB_gpu_shader_int64 extension
    return has_extension(GL_ARB_gpu_shader_int64) || has_explicit_sized_int_types();
}

// Returns true if the GLSLang version has the half type.
bool GLSLang_context::has_half_type() const
{
    // half is supported with one of these:
    // GL_NV_gpu_shader5, GL_AMD_gpu_shader_half_float
    return
        has_extension(GL_NV_gpu_shader5) ||
        has_extension(GL_AMD_gpu_shader_half_float);
}

// Returns true if the GLSLang version has the uint type.
bool GLSLang_context::has_uint_type() const
{
    // uint supported in GS from 3.00 and GL from 1.30
    return es_GE(300) || non_es_GE(130);
}

// Returns true if the GLSLang version has pointer types.
bool GLSLang_context::has_pointer_types() const
{
    return has_extension(GL_NV_shader_buffer_load);
}

// returns true if non-quadratic matrices exists in this GLSLang version.
bool GLSLang_context::has_non_quadratic_matrices() const
{
    // this is the restriction for float matrices, for double it is the same as
    // has_double_type()
    return get_version() > 110;
}

// Returns true if the switch statement exists in this GLSLang version.
bool GLSLang_context::has_switch_stmt() const
{
    // keywords from ES 300, core 130
    return es_GE(300) || non_es_GE(130);
}

// Returns true if the bitwise integer operators exists in this GLSLang version..
bool GLSLang_context::has_bitwise_ops() const
{
    // exists from ES 300, core 130
    return es_GE(300) || non_es_GE(130);
}

// Returns true if the shift operators exists in this GLSLang version..
bool GLSLang_context::has_shift_ops() const
{
    // exists from ES 300, core 130
    return es_GE(300) || non_es_GE(130);
}

// Return true if array of arrays are supported.
bool GLSLang_context::has_array_of_arrays() const
{
    // exists from ES 310, core 430
    return es_GE(310) || non_es_GE(430) || has_extension(GL_ARB_arrays_of_arrays);
}

// Check if C-style initializers are allowed.
bool GLSLang_context::has_c_style_initializer() const
{
    // exists from core 420
    return non_es_GE(420) || has_extension(GL_ARB_shading_language_420pack);
}

// Check if shader storage buffer objects are supported.
bool GLSLang_context::has_SSBO() const
{
    return has_extension(GL_ARB_shader_storage_buffer_object) ||
        !necessary_extension(GL_ARB_shader_storage_buffer_object);
}

// Check if memory qualifiers (coherent, volatile, restrict, readonly, and writeonly)
// are supported.
bool GLSLang_context::has_memory_qualifier() const
{
    return non_es_GE(420) || es_GE(310);
}

// Check if functions to get/set the bit encoding for floating-point values are available.
bool GLSLang_context::has_bit_encoding_functions() const
{
    return
        has_extension(GL_ARB_gpu_shader5) ||
        has_extension(GL_ARB_shader_bit_encoding) ||
        es_GE(300) || non_es_GE(330);
}

// Check it the type of return expressions is automatically converted to the return type
// of the current function.
bool GLSLang_context::has_implicit_type_convert_at_return() const
{
    return non_es_GE(420);
}

// Check if implicit type conversions are allowed.
bool GLSLang_context::has_implicit_conversions() const
{
    return !es() || has_extension(GL_EXT_shader_implicit_conversions);
}

// Check if VULKAN is supported.
bool GLSLang_context::has_vulkan() const
{
    return has_extension(GL_KHR_vulkan_glsl);
}

// Check if we have the requested feature.
GLSL_keyword_state GLSLang_context::keyword(GLSL_keyword f) const
{
    switch (f) {
    case KW_SWITCH:
    case KW_DEFAULT:
        // keywords from ES 300, core 130, else reserved
        return es_GE(300) || non_es_GE(130) ? KS_KEYWORD : KS_RESERVED;
    case KW_CASE:
        // keywords from ES 300, core 130, else ident
        return es_GE(300) || non_es_GE(130) ? KS_KEYWORD : KS_IDENT;
    case KW_ATTRIBUTE:
    case KW_VARYING:
        // reserved in ES >= 300, else keyword
        return es_GE(300) ? KS_RESERVED : KS_KEYWORD;
    case KW_BUFFER:
        // keyword from ES 3.10, core 4.30, else ident
        return es_GE(310) || non_es_GE(430) ? KS_KEYWORD : KS_IDENT;
    case KW_ATOMIC_UINT:
        // keyword from ES 330 or if GL_ARB_shader_atomic_counters is enable
        if (es_GE(310) || has_extension(GL_ARB_shader_atomic_counters)) {
            return KS_KEYWORD;
        }
        // reserved keyword from ES 3.00
        if (es_GE(300)) {
            return KS_RESERVED;
        }
        // keyword from core 4.20
        if (non_es_GE(420)) {
            return KS_KEYWORD;
        }
        // else an ident
        return KS_IDENT;

    case KW_COHERENT:
    case KW_RESTRICT:
    case KW_READONLY:
    case KW_WRITEONLY:
        // keyword from ES 310
        if (es_GE(310)) {
            return KS_KEYWORD;
        }
        // reserved keyword from ES 300
        if (es_GE(300)) {
            return KS_RESERVED;
        }
        // keyword in core from 130 with extension or without with 420
        if (non_es_GE(has_extension(GL_ARB_shader_image_load_store) ? 130 : 420)) {
            return KS_KEYWORD;
        }
        // else identifier
        return KS_IDENT;

    case KW_VOLATILE:
        // keyword from ES 310, core 420, extension enabled
        if (es_GE(310)) {
            return KS_KEYWORD;
        }
        // reserved in all other ES or core < 420 without extension
        if (get_profile() == GLSL_PROFILE_ES ||
            (non_es_LT(420) && !has_extension(GL_ARB_shader_image_load_store))) {
            return KS_RESERVED;
        }
        // else keyword
        return KS_KEYWORD;

    case KW_LAYOUT:
        // ident before ES 300 and core 140 without extension
        if (es_LT(300) || (non_es_LT(140) && !has_extension(GL_ARB_shading_language_420pack))) {
            return KS_IDENT;
        }
        // else keyword
        return KS_KEYWORD;

    case KW_SHARED:
        // ident before ES 300 and core 140
        if (es_LT(300) || non_es_LT(140)) {
            return KS_IDENT;
        }
        // else keyword
        return KS_KEYWORD;

    case KW_PATCH:
        // reserved after ES 300
        if (es_GE(300)) {
            return KS_RESERVED;
        }
        // keyword after core 150 (with extension) or core 400
        if (non_es_GE(has_extension(GL_ARB_tessellation_shader) ? 150 : 400)) {
            return KS_KEYWORD;
        }
        // else ident
        return KS_IDENT;

    case KW_SAMPLE:
        // reserved keyword from ES 3.00
        if (es_GE(300)) {
            return KS_RESERVED;
        }
        // keyword from core 4.00
        if (non_es_GE(400)) {
            return KS_KEYWORD;
        }
        // else ident
        return KS_IDENT;

    case KW_SUBROUTINE:
        if (has_extension(GL_ARB_shader_subroutine)) {
            return KS_KEYWORD;
        }
        // reserved keyword from ES 3.00
        if (es_GE(300)) {
            return KS_RESERVED;
        }
        // keyword from core 4.00
        if (non_es_GE(400)) {
            return KS_KEYWORD;
        }
        // else ident
        return KS_IDENT;

    case KW_HIGH_PRECISION:
    case KW_MEDIUM_PRECISION:
    case KW_LOW_PRECISION:
    case KW_PRECISION:
        // keyword in ES and from core 130, else ident
        if (get_profile() == GLSL_PROFILE_ES || non_es_GE(130)) {
            return KS_KEYWORD;
        }
        return KS_IDENT;

    case KW_MAT2X2:
    case KW_MAT2X3:
    case KW_MAT2X4:
    case KW_MAT3X2:
    case KW_MAT3X3:
    case KW_MAT3X4:
    case KW_MAT4X2:
    case KW_MAT4X3:
    case KW_MAT4X4:
        // keyword after version 110
        return get_version() > 110 ? KS_KEYWORD : KS_IDENT;

    case KW_DMAT2:
    case KW_DMAT3:
    case KW_DMAT4:
    case KW_DMAT2X2:
    case KW_DMAT2X3:
    case KW_DMAT2X4:
    case KW_DMAT3X2:
    case KW_DMAT3X3:
    case KW_DMAT3X4:
    case KW_DMAT4X2:
    case KW_DMAT4X3:
    case KW_DMAT4X4:
        // enabled with ARB_gpu_shader_fp64 extension
        if (has_extension(GL_ARB_gpu_shader_fp64)) {
            return KS_KEYWORD;
        }
        // reserved in ES from 300
        if (es_GE(300)) {
            return KS_RESERVED;
        }
        // keyword from core 400
        if (non_es_GE(400)) {
            return KS_KEYWORD;
        }
        // else ident
        return KS_IDENT;

    case KW_IMAGE1D:
    case KW_IIMAGE1D:
    case KW_UIMAGE1D:
    case KW_IMAGE1DARRAY:
    case KW_IIMAGE1DARRAY:
    case KW_UIMAGE1DARRAY:
    case KW_IMAGE2DRECT:
    case KW_IIMAGE2DRECT:
    case KW_UIMAGE2DRECT:
    case KW_IMAGEBUFFER:
    case KW_IIMAGEBUFFER:
    case KW_UIMAGEBUFFER:
        // keyword from core 420 or with extension
        if (non_es_GE(420) || has_extension(GL_ARB_shader_image_load_store)) {
            return KS_KEYWORD;
        }
        // reserved form ES 300 or core 130
        if (es_GE(300) || non_es_GE(130)) {
            return KS_RESERVED;
        }
        // else ident
        return KS_IDENT;

    case KW_IMAGE2D:
    case KW_IIMAGE2D:
    case KW_UIMAGE2D:
    case KW_IMAGE3D:
    case KW_IIMAGE3D:
    case KW_UIMAGE3D:
    case KW_IMAGECUBE:
    case KW_IIMAGECUBE:
    case KW_UIMAGECUBE:
    case KW_IMAGE2DARRAY:
    case KW_IIMAGE2DARRAY:
    case KW_UIMAGE2DARRAY:
        // keyword from ES 310, core 420 or with extension
        if (es_GE(310) || non_es_GE(420) || has_extension(GL_ARB_shader_image_load_store)) {
            return KS_KEYWORD;
        }
        // reserved form ES 300 or core 130
        if (es_GE(300) || non_es_GE(130)) {
            return KS_RESERVED;
        }
        // else ident
        return KS_IDENT;

    case KW_IMAGECUBEARRAY:
    case KW_IIMAGECUBEARRAY:
    case KW_UIMAGECUBEARRAY:
    case KW_IMAGE2DMS:
    case KW_IIMAGE2DMS:
    case KW_UIMAGE2DMS:
    case KW_IMAGE2DMSARRAY:
    case KW_IIMAGE2DMSARRAY:
    case KW_UIMAGE2DMSARRAY:
        // reserved from ES 310
        if (es_GE(310)) {
            return KS_RESERVED;
        }
        // keyword from core 420 or with extension
        if (non_es_GE(420) || has_extension(GL_ARB_shader_image_load_store)) {
            return KS_KEYWORD;
        }
        // else ident
        return KS_IDENT;

    case KW_DOUBLE:
    case KW_DVEC2:
    case KW_DVEC3:
    case KW_DVEC4:
        // enabled with ARB_gpu_shader_fp64 extension
        if (has_extension(GL_ARB_gpu_shader_fp64)) {
            return KS_KEYWORD;
        }
        // reserved in ES or core before 400
        if (get_profile() == GLSL_PROFILE_ES || non_es_LT(400)) {
            return KS_RESERVED;
        }
        // keyword else
        return KS_KEYWORD;

    case KW_SAMPLERCUBEARRAY:
    case KW_SAMPLERCUBEARRAYSHADOW:
    case KW_ISAMPLERCUBEARRAY:
    case KW_USAMPLERCUBEARRAY:
        // reserved in ES or core before 400 without extension
        if (get_profile() == GLSL_PROFILE_ES ||
            (non_es_LT(400) && !has_extension(GL_ARB_texture_cube_map_array))) {
            return KS_RESERVED;
        }
        // keyword else
        return KS_KEYWORD;

    case KW_ISAMPLER1D:
    case KW_ISAMPLER1DARRAY:
    case KW_SAMPLER1DARRAYSHADOW:
    case KW_USAMPLER1D:
    case KW_USAMPLER1DARRAY:
    case KW_SAMPLERBUFFER:
        // reserved from ES 300
        if (es_GE(300)) {
            return KS_RESERVED;
        }
        // keyword from core 130
        if (non_es_GE(130)) {
            return KS_KEYWORD;
        }
        // else ident
        return KS_IDENT;

    case KW_UINT:
    case KW_UVEC2:
    case KW_UVEC3:
    case KW_UVEC4:
    case KW_SAMPLERCUBESHADOW:
    case KW_SAMPLER2DARRAY:
    case KW_SAMPLER2DARRAYSHADOW:
    case KW_ISAMPLER2D:
    case KW_ISAMPLER3D:
    case KW_ISAMPLERCUBE:
    case KW_ISAMPLER2DARRAY:
    case KW_USAMPLER2D:
    case KW_USAMPLER3D:
    case KW_USAMPLERCUBE:
    case KW_USAMPLER2DARRAY:
        // keyword from ES 300 and core 130
        if (es_GE(300) || non_es_GE(130)) {
            return KS_KEYWORD;
        }
        // else ident
        return KS_IDENT;

    case KW_ISAMPLER2DRECT:
    case KW_USAMPLER2DRECT:
    case KW_ISAMPLERBUFFER:
    case KW_USAMPLERBUFFER:
        // reserved word from ES 300
        if (es_GE(300)) {
            return KS_RESERVED;
        }
        // keyword from core 140
        if (non_es_GE(140)) {
            return KS_KEYWORD;
        }
        // else ident
        return KS_IDENT;

    case KW_SAMPLER2DMS:
    case KW_ISAMPLER2DMS:
    case KW_USAMPLER2DMS:
        // keyword from ES 310 and core 150
        if (es_GE(310) || non_es_GE(150)) {
            return KS_KEYWORD;
        }
        // reserved word from ES 300
        if (es_GE(300)) {
            return KS_RESERVED;
        }
        // else ident
        return KS_IDENT;

    case KW_SAMPLER2DMSARRAY:
    case KW_ISAMPLER2DMSARRAY:
    case KW_USAMPLER2DMSARRAY:
        // reserved word from ES 300
        if (es_GE(300)) {
            return KS_RESERVED;
        }
        // keyword from core 150
        if (non_es_GE(150)) {
            return KS_KEYWORD;
        }
        // else ident
        return KS_IDENT;

    case KW_SAMPLER1D:
    case KW_SAMPLER1DSHADOW:
        // reserved in ES
        if (get_profile() == GLSL_PROFILE_ES) {
            return KS_RESERVED;
        }
        // else keyword
        return KS_KEYWORD;

    case KW_SAMPLER3D:
        // reserved in ES before 300 without extension
        if (es_LT(300) && !has_extension(GL_OES_texture_3D)) {
            return KS_RESERVED;
        }
        // else keyword
        return KS_KEYWORD;

    case KW_SAMPLER2DSHADOW:
        // reserved in ES before 300
        if (es_LT(300)) {
            return KS_RESERVED;
        }
        // else keyword
        return KS_KEYWORD;

    case KW_SAMPLER2DRECT:
    case KW_SAMPLER2DRECTSHADOW:
        // reserved in ES
        if (get_profile() == GLSL_PROFILE_ES) {
            return KS_RESERVED;
        }
        // reserved before core 140 without extension
        if (non_es_LT(140) && !has_extension(GL_ARB_texture_rectangle)) {
            return KS_RESERVED;
        }
        // else keyword
        return KS_KEYWORD;

    case KW_SAMPLER1DARRAY:
        // reserved in ES 300
        if (get_profile() == GLSL_PROFILE_ES && get_version() == 300) {
            return KS_RESERVED;
        }
        // ident before ES 300 and core 130
        if (es_LT(300) || non_es_LT(130)) {
            return KS_IDENT;
        }
        // else keyword
        return KS_KEYWORD;

    case KW_SAMPLEREXTERNALOES:
        // keyword with extension
        if (has_extension(GL_OES_EGL_image_external)) {
            return KS_KEYWORD;
        }
        // else ident
        return KS_IDENT;

    case KW_NOPERSPECTIVE:
        // reserved from ES 300
        if (es_GE(300)) {
            return KS_RESERVED;
        }
        // keyword from core 130
        if (non_es_GE(130)) {
            return KS_KEYWORD;
        }
        // else ident
        return KS_IDENT;

    case KW_SMOOTH:
        // keyword from ES 300 and core 130
        if (es_GE(300) || non_es_GE(130)) {
            return KS_KEYWORD;
        }
        // else ident
        return KS_IDENT;

    case KW_FLAT:
        // reserved before ES 300
        if (es_LT(300)) {
            return KS_RESERVED;
        }
        // ident before core 130
        if (non_es_LT(130)) {
            return KS_IDENT;
        }
        // else keyword
        return KS_KEYWORD;

    case KW_CENTROID:
        // ident before 120
        if (get_version() < 120) {
            return KS_IDENT;
        }
        // else keyword
        return KS_KEYWORD;

    case KW_PRECISE:
        // reserved from ES 310
        if (es_GE(310)) {
            return KS_RESERVED;
        }
        // keyword from core 400
        if (non_es_GE(400)) {
            return KS_KEYWORD;
        }
        // else ident
        return KS_IDENT;

    case KW_INVARIANT:
        // ident before core 120
        if (non_es_LT(120)) {
            return KS_IDENT;
        }
        return KS_KEYWORD;

    case KW_PACKED:
        // reserved before ES 300 and core 330
        if (es_LT(300) || non_es_LT(330)) {
            return KS_RESERVED;
        }
        // else ident
        return KS_IDENT;

    case KW_RESOURCE:
        // reserved from ES 200 and core 420
        if (es_GE(300) || non_es_GE(420)) {
            return KS_RESERVED;
        }
        // else ident
        return KS_IDENT;

    case KW_SUPERP:
        // reserved from ES 130
        if (es_GE(130)) {
            return KS_RESERVED;
        }
        // else ident
        return KS_IDENT;

    case KW_UNSIGNED:
        if (!is_strict()) {
            return KS_KEYWORD;
        }
        // reserved in strict grammar mode
        return KS_RESERVED;

    case KW_FLOAT16_T:
        // only available with extension
        if (has_half_type() || has_explicit_sized_float_types()) {
            return KS_KEYWORD;
        }
        // else ident
        return KS_IDENT;

    case KW_INT8_T:
    case KW_I8VEC2:
    case KW_I8VEC3:
    case KW_I8VEC4:
    case KW_UINT8_T:
    case KW_U8VEC2:
    case KW_U8VEC3:
    case KW_U8VEC4:
    case KW_INT16_T:
    case KW_I16VEC2:
    case KW_I16VEC3:
    case KW_I16VEC4:
    case KW_UINT16_T:
    case KW_U16VEC2:
    case KW_U16VEC3:
    case KW_U16VEC4:
    case KW_INT32_T:
    case KW_I32VEC2:
    case KW_I32VEC3:
    case KW_I32VEC4:
    case KW_UINT32_T:
    case KW_U32VEC2:
    case KW_U32VEC3:
    case KW_U32VEC4:
        if (has_explicit_sized_int_types()) {
            return KS_KEYWORD;
        }
        // else ident
        return KS_IDENT;

    case KW_INT64_T:
    case KW_I64VEC2:
    case KW_I64VEC3:
    case KW_I64VEC4:
    case KW_UINT64_T:
    case KW_U64VEC2:
    case KW_U64VEC3:
    case KW_U64VEC4:
        if (has_int64_types()) {
            return KS_KEYWORD;
        }
        // else ident
        return KS_IDENT;

    case KW_F16VEC2:
    case KW_F16VEC3:
    case KW_F16VEC4:
    case KW_FLOAT32_T:
    case KW_F32VEC2:
    case KW_F32VEC3:
    case KW_F32VEC4:
    case KW_FLOAT64_T:
    case KW_F64VEC2:
    case KW_F64VEC3:
    case KW_F64VEC4:
        if (has_explicit_sized_float_types()) {
            return KS_KEYWORD;
        }
        // else ident
        return KS_IDENT;
    }
    GLSL_ASSERT(!"unknown keyword requested");
    return KS_IDENT;
}

}  // glsl
}  // mdl
}  // mi

