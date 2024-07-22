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

#ifndef MDL_COMPILER_GLSL_VERSION_H
#define MDL_COMPILER_GLSL_VERSION_H 1

#include "compiler_glsl_cc_conf.h"
#include <mdl/compiler/compilercore/compilercore_allocator.h>
#include <mdl/compiler/compilercore/compilercore_bitset.h>

namespace mi {
namespace mdl {
namespace glsl {

class GLSLKeywordMap;

/// Supported GLSLang versions.
enum GLSLang_version
{
    GLSL_VERSION_ES_1_00 = 100,  ///< Version 100 ES
    GLSL_VERSION_ES_3_00 = 300,  ///< Version 300 ES
    GLSL_VERSION_ES_3_10 = 310,  ///< Version 310 ES

    GLSL_VERSION_1_10 = 110,  ///< Version 1.10
    GLSL_VERSION_1_20 = 120,  ///< Version 1.20
    GLSL_VERSION_1_30 = 130,  ///< Version 1.30
    GLSL_VERSION_1_40 = 140,  ///< Version 1.40
    GLSL_VERSION_1_50 = 150,  ///< Version 1.50
    GLSL_VERSION_3_30 = 330,  ///< Version 3.30
    GLSL_VERSION_4_00 = 400,  ///< Version 4.00
    GLSL_VERSION_4_10 = 410,  ///< Version 4.10
    GLSL_VERSION_4_20 = 420,  ///< Version 4.20
    GLSL_VERSION_4_30 = 430,  ///< Version 4.30
    GLSL_VERSION_4_40 = 440,  ///< Version 4.40
    GLSL_VERSION_4_50 = 450,  ///< Version 4.50
    GLSL_VERSION_4_60 = 460,  ///< Version 4.60

    GLSL_VERSION_DEFAULT = GLSL_VERSION_3_30
};

/// Supported GLSLang profiles.
enum GLSLang_profile
{
    GLSL_PROFILE_CORE          = 1 << 0,  ///< Core profile.
    GLSL_PROFILE_COMPATIBILITY = 1 << 1,  ///< Compatibility profile
    GLSL_PROFILE_ES            = 1 << 2   ///< ES profile
};

// Extensions.
enum GLSL_extensions
{
    GL_OES_texture_3D                   = 0,
    GL_OES_standard_derivatives         = 1,
    GL_OES_EGL_image_external           = 2,

    GL_EXT_frag_depth                   = 3,
    GL_EXT_shader_texture_lod           = 4,
    GL_EXT_shader_implicit_conversions =  5,


    GL_ARB_texture_rectangle            = 6,
    GL_ARB_shading_language_420pack     = 7,
    GL_ARB_texture_gather               = 8,
    GL_ARB_gpu_shader5                  = 9,
    GL_ARB_separate_shader_objects      = 10,
    GL_ARB_tessellation_shader          = 11,
    GL_ARB_enhanced_layouts             = 12,
    GL_ARB_texture_cube_map_array       = 13,
    GL_ARB_shader_texture_lod           = 14,
    GL_ARB_explicit_attrib_location     = 15,
    GL_ARB_shader_image_load_store      = 16,
    GL_ARB_shader_atomic_counters       = 17,
    GL_ARB_derivative_control           = 18,
    GL_ARB_shader_texture_image_samples = 19,
    GL_ARB_viewport_array               = 20,
    GL_ARB_cull_distance                = 21,
    GL_ARB_gpu_shader_fp64              = 22,
    GL_ARB_shader_subroutine            = 23,
    GL_ARB_arrays_of_arrays             = 24,
    GL_ARB_shader_storage_buffer_object = 25,
    GL_ARB_bindless_texture             = 26,
    GL_ARB_gpu_shader_int64             = 27,
    GL_ARB_shader_bit_encoding          = 28,

    GL_3DL_array_objects                = 29,

    GL_KHR_vulkan_glsl                  = 30,

    GL_NV_shader_buffer_load            = 31,
    GL_NV_gpu_shader5                   = 32,

    GL_AMD_gpu_shader_half_float        = 33,

    GL_GOOGLE_cpp_style_line_directive  = 34,
    GL_GOOGLE_include_directive         = 35,

    LAST_EXTENSION = GL_GOOGLE_include_directive
};

// Supported GLSL sub-languages.
enum GLSL_language {
    GLSL_LANG_VERTEX,           ///< vertex shader
    GLSL_LANG_TESSCONTROL,      ///< tessellation control shader
    GLSL_LANG_TESSEVALUATION,   ///< tessellation evaluation shader
    GLSL_LANG_GEOMETRY,         ///< geometry shader
    GLSL_LANG_FRAGMENT,         ///< fragment shader
    GLSL_LANG_COMPUTE,          ///< compute shader
    GLSL_LANG_LAST = GLSL_LANG_COMPUTE,
};

/// Keyword state.
enum GLSL_keyword_state
{
    KS_KEYWORD,  ///< This is a keyword in the current context.
    KS_RESERVED, ///< This is a reserved keyword in the current context.
    KS_IDENT     ///< This is an identifier in the current context
};

/// GLSL changing keywords.
enum GLSL_keyword
{
    KW_SWITCH,
    KW_DEFAULT,
    KW_CASE,
    KW_ATTRIBUTE,
    KW_VARYING,
    KW_BUFFER,
    KW_ATOMIC_UINT,
    KW_COHERENT,
    KW_RESTRICT,
    KW_READONLY,
    KW_WRITEONLY,
    KW_VOLATILE,
    KW_LAYOUT,
    KW_SHARED,
    KW_PATCH,
    KW_SAMPLE,
    KW_SUBROUTINE,
    KW_HIGH_PRECISION,
    KW_MEDIUM_PRECISION,
    KW_LOW_PRECISION,
    KW_PRECISION,
    KW_MAT2X2,
    KW_MAT2X3,
    KW_MAT2X4,
    KW_MAT3X2,
    KW_MAT3X3,
    KW_MAT3X4,
    KW_MAT4X2,
    KW_MAT4X3,
    KW_MAT4X4,
    KW_DMAT2,
    KW_DMAT3,
    KW_DMAT4,
    KW_DMAT2X2,
    KW_DMAT2X3,
    KW_DMAT2X4,
    KW_DMAT3X2,
    KW_DMAT3X3,
    KW_DMAT3X4,
    KW_DMAT4X2,
    KW_DMAT4X3,
    KW_DMAT4X4,
    KW_IMAGE1D,
    KW_IIMAGE1D,
    KW_UIMAGE1D,
    KW_IMAGE1DARRAY,
    KW_IIMAGE1DARRAY,
    KW_UIMAGE1DARRAY,
    KW_IMAGE2DRECT,
    KW_IIMAGE2DRECT,
    KW_UIMAGE2DRECT,
    KW_IMAGEBUFFER,
    KW_IIMAGEBUFFER,
    KW_UIMAGEBUFFER,
    KW_IMAGE2D,
    KW_IIMAGE2D,
    KW_UIMAGE2D,
    KW_IMAGE3D,
    KW_IIMAGE3D,
    KW_UIMAGE3D,
    KW_IMAGECUBE,
    KW_IIMAGECUBE,
    KW_UIMAGECUBE,
    KW_IMAGE2DARRAY,
    KW_IIMAGE2DARRAY,
    KW_UIMAGE2DARRAY,
    KW_IMAGECUBEARRAY,
    KW_IIMAGECUBEARRAY,
    KW_UIMAGECUBEARRAY,
    KW_IMAGE2DMS,
    KW_IIMAGE2DMS,
    KW_UIMAGE2DMS,
    KW_IMAGE2DMSARRAY,
    KW_IIMAGE2DMSARRAY,
    KW_UIMAGE2DMSARRAY,
    KW_DOUBLE,
    KW_DVEC2,
    KW_DVEC3,
    KW_DVEC4,
    KW_SAMPLERCUBEARRAY,
    KW_SAMPLERCUBEARRAYSHADOW,
    KW_ISAMPLERCUBEARRAY,
    KW_USAMPLERCUBEARRAY,
    KW_ISAMPLER1D,
    KW_ISAMPLER1DARRAY,
    KW_SAMPLER1DARRAYSHADOW,
    KW_USAMPLER1D,
    KW_USAMPLER1DARRAY,
    KW_SAMPLERBUFFER,
    KW_UINT,
    KW_UVEC2,
    KW_UVEC3,
    KW_UVEC4,
    KW_SAMPLERCUBESHADOW,
    KW_SAMPLER2DARRAY,
    KW_SAMPLER2DARRAYSHADOW,
    KW_ISAMPLER2D,
    KW_ISAMPLER3D,
    KW_ISAMPLERCUBE,
    KW_ISAMPLER2DARRAY,
    KW_USAMPLER2D,
    KW_USAMPLER3D,
    KW_USAMPLERCUBE,
    KW_USAMPLER2DARRAY,
    KW_ISAMPLER2DRECT,
    KW_USAMPLER2DRECT,
    KW_ISAMPLERBUFFER,
    KW_USAMPLERBUFFER,
    KW_SAMPLER2DMS,
    KW_ISAMPLER2DMS,
    KW_USAMPLER2DMS,
    KW_SAMPLER2DMSARRAY,
    KW_ISAMPLER2DMSARRAY,
    KW_USAMPLER2DMSARRAY,
    KW_SAMPLER1D,
    KW_SAMPLER1DSHADOW,
    KW_SAMPLER3D,
    KW_SAMPLER2DSHADOW,
    KW_SAMPLER2DRECT,
    KW_SAMPLER2DRECTSHADOW,
    KW_SAMPLER1DARRAY,
    KW_SAMPLEREXTERNALOES,
    KW_NOPERSPECTIVE,
    KW_SMOOTH,
    KW_FLAT,
    KW_CENTROID,
    KW_PRECISE,
    KW_INVARIANT,
    KW_PACKED,
    KW_RESOURCE,
    KW_SUPERP,
    KW_UNSIGNED,

    KW_INT8_T,
    KW_I8VEC2,
    KW_I8VEC3,
    KW_I8VEC4,
    KW_UINT8_T,
    KW_U8VEC2,
    KW_U8VEC3,
    KW_U8VEC4,

    KW_INT16_T,
    KW_I16VEC2,
    KW_I16VEC3,
    KW_I16VEC4,
    KW_UINT16_T,
    KW_U16VEC2,
    KW_U16VEC3,
    KW_U16VEC4,

    KW_INT32_T,
    KW_I32VEC2,
    KW_I32VEC3,
    KW_I32VEC4,
    KW_UINT32_T,
    KW_U32VEC2,
    KW_U32VEC3,
    KW_U32VEC4,

    KW_INT64_T,
    KW_I64VEC2,
    KW_I64VEC3,
    KW_I64VEC4,
    KW_UINT64_T,
    KW_U64VEC2,
    KW_U64VEC3,
    KW_U64VEC4,

    KW_FLOAT16_T,
    KW_F16VEC2,
    KW_F16VEC3,
    KW_F16VEC4,

    KW_FLOAT32_T,
    KW_F32VEC2,
    KW_F32VEC3,
    KW_F32VEC4,

    KW_FLOAT64_T,
    KW_F64VEC2,
    KW_F64VEC3,
    KW_F64VEC4,
};

/// Helper class to manage GLSLang versions, profiles, extensions.
class GLSLang_context
{
    friend class Compilation_unit;
    friend class Preprocessor_result;
    friend class Compiler;
public:
    /// Set the GLSL language version and profile.
    ///
    /// \param version  the GLSL language version
    /// \param profile  the profile
    ///
    /// \return true if this combination is supported, false otherwise
    bool set_version(unsigned version, unsigned profile);

    /// Enable/Disable a GLSL extension and update the scanner if any was registered
    void enable_extension(GLSL_extensions ext, bool enable);

    /// Possible error codes of update_extension.
    enum Ext_error_code {
        EC_OK,                             ///< no error
        EC_UNKNOWN_BEHAVIOR,               ///< unknown behavior
        EC_ALL_BH_RESTRICTED,              ///< the "all" extension has restricted behavior
        EC_UNSUPPORTED_EXTENSION,          ///< unsupported extension
        EC_UNSUPPORTED_EXTENSION_REQUIRED, ///< unsupported extension required
    };

    /// Supported extension behavior.
    enum Extension_behavior {
        EB_REQUIRE,
        EB_ENABLE,
        EB_DISABLE,
        EB_WARN
    };

    /// Get the extension behavior for the given extension.
    ///
    /// \param ext  the extension
    Extension_behavior get_extension_behavior(
        GLSL_extensions ext) const
    {
        return m_extension_behavior[ext];
    }

    /// Get the name of an extension.
    ///
    /// \param ext  the extension
    static char const *get_extension_name(
        GLSL_extensions ext);

    /// Update an extension.
    ///
    /// \param ext_name  the name of the extension
    /// \param behavior  the behavior of this extension
    Ext_error_code update_extension(
        char const *ext_name,
        char const *behavior);

    /// Update an extension.
    ///
    /// \param ext_name  the name of the extension
    /// \param eb        the behavior of this extension
    Ext_error_code update_extension(
        char const         *ext_name,
        Extension_behavior eb);

    /// Get the GLSL sub-language.
    GLSL_language get_language() const { return m_lang; }

    /// Get the GLSL language version.
    unsigned get_version() const { return m_version; }

    /// Get the GLSL language profile.
    unsigned get_profile() const { return m_profile; }

    /// Return if the given extension is enabled.
    bool has_extension(GLSL_extensions ext) const { return m_extensions.test_bit(ext); }

    /// Check if we support strict grammar only.
    bool is_strict() const { return m_strict; }

    /// Get the state of a given keyword.
    GLSL_keyword_state keyword(GLSL_keyword k) const;

    /// Register the keyword map for feedback on changed version properties.
    void register_keywords(GLSLKeywordMap *keywords);

    /// Returns true if the GLSLang version requires an explicit #version directive.
    bool needs_explicit_version() const;

    /// Returns true if the GLSLang version requires an explicit profile.
    bool needs_explicit_profile() const;

    /// Returns true if the GLSLang version has the double type.
    bool has_double_type() const;

    /// Returns true if the GLSLang version has explicit sized integer types.
    bool has_explicit_sized_int_types() const;

    /// Returns true if the GLSLang version has explicit sized float types.
    bool has_explicit_sized_float_types() const;

    /// Returns true if the GLSLang version has 64bit integer types.
    bool has_int64_types() const;

    /// Returns true if the GLSLang version has the half type.
    bool has_half_type() const;

    /// Returns true if the GLSLang version has the uint type.
    bool has_uint_type() const;

    /// Returns true if the GLSLang version has pointer types.
    bool has_pointer_types() const;

    /// Returns true if non-quadratic matrices exists in this GLSLang version.
    bool has_non_quadratic_matrices() const;

    /// Returns true if the switch statement exists in this GLSLang version.
    bool has_switch_stmt() const;

    /// Returns true if the bitwise integer operators exists in this GLSLang version..
    bool has_bitwise_ops() const;

    /// Returns true if the shift operators exists in this GLSLang version..
    bool has_shift_ops() const;

    /// Return true if array of arrays are supported.
    bool has_array_of_arrays() const;

    /// Check if C-style initializers are allowed.
    bool has_c_style_initializer() const;

    /// Check if shader storage buffer objects are supported.
    bool has_SSBO() const;

    /// Check if memory qualifiers (coherent, volatile, restrict, readonly, and writeonly)
    /// are supported.
    bool has_memory_qualifier() const;

    /// Check if functions to get/set the bit encoding for floating-point values are available.
    bool has_bit_encoding_functions() const;

    /// Check it the type of return expressions is automatically converted to the return type
    /// of the current function.
    bool has_implicit_type_convert_at_return() const;

    /// Check if implicit type conversions are allowed.
    bool has_implicit_conversions() const;

    /// Check if VULKAN is supported.
    bool has_vulkan() const;

    /// Check, if we are in the ES profile.
    bool es() const;

    /// Check, if we are in the ES profile and the version is lesser than a given version.
    bool es_LT(unsigned version) const;

    /// Check, if we are in the ES profile and the version is greater or equal
    /// than a given version.
    bool es_GE(unsigned version) const;

    /// Check, if we are NOT in the ES profile and the version is lesser than given version.
    bool non_es_LT(unsigned version) const;

    /// Check, if we are NOT in the ES profile and the version is greater or equal
    /// than given version.
    bool non_es_GE(unsigned version) const;

    /// Check, if the given extension is necessary, or if it is already included in the current
    /// GLSLang version
    bool necessary_extension(GLSL_extensions ext) const;

private:
    /// Constructor.
    ///
    /// \param lang   the GLSL sub language
    ///
    /// Sets the defaults (i.e. NO #version present)
    GLSLang_context(
        GLSL_language lang);

private:
    /// Enable/Disable a GLSL extension but do not update the scanner.
    void enable_extension_no_update(GLSL_extensions ext, bool enable);

private:
    // Copy constructor not implemented.
    GLSLang_context(GLSLang_context const &) GLSL_DELETED_FUNCTION;
    // Assignment operator not implemented.
    GLSLang_context &operator=(GLSLang_context const &) GLSL_DELETED_FUNCTION;

private:
    /// The GLSL language.
    GLSL_language m_lang;

    /// The GLSLang version.
    unsigned m_version;

    /// The GLSLang profile.
    unsigned m_profile;

    /// The Enabled extensions.
    Static_bitset<LAST_EXTENSION + 1> m_extensions;

    /// True, if grammar is strict, false if relaxed.
    bool m_strict;

    /// The keyword map once registered.
    GLSLKeywordMap *m_keywords;

    /// Extension behaviors.
    Extension_behavior m_extension_behavior[LAST_EXTENSION + 1];
};

}  // glsl
}  // mdl
}  // mi

#endif
