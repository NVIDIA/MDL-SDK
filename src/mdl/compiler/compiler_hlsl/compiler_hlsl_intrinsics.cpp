/******************************************************************************
 * Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "compiler_hlsl_compilation_unit.h"
#include "compiler_hlsl_definitions.h"

namespace mi {
namespace mdl {
namespace hlsl {

// Intrinsic definitions.
enum Intrinsic_usage {
    AR_QUAL_IN = 0x0000010,
    AR_QUAL_OUT = 0x0000020,
    AR_QUAL_CONST = 0x0000200,
    AR_QUAL_ROWMAJOR = 0x0000400,
    AR_QUAL_COLMAJOR = 0x0000800,

    AR_QUAL_IN_OUT = (AR_QUAL_IN | AR_QUAL_OUT)
};

static unsigned char const INTRIN_TEMPLATE_FROM_TYPE = 0xff;
static unsigned char const INTRIN_TEMPLATE_VARARGS   = 0xfe;

// Use this enumeration to describe allowed templates (layouts) in intrinsics.
enum LEGAL_INTRINSIC_TEMPLATES {
    LITEMPLATE_VOID   = 0,  // No return type.
    LITEMPLATE_SCALAR = 1,  // Scalar types.
    LITEMPLATE_VECTOR = 2,  // Vector types (eg. float3).
    LITEMPLATE_MATRIX = 3,  // Matrix types (eg. float3x3).
    LITEMPLATE_ANY    = 4,  // Any one of scalar, vector or matrix types (but not object).
    LITEMPLATE_OBJECT = 5,  // Object types.

    LITEMPLATE_COUNT = 6
};

// INTRIN_COMPTYPE_FROM_TYPE_ELT0 is for object method intrinsics to indicate
// that the component type of the type is taken from the first subelement of the
// object's template type; see for example Texture2D.Gather
static unsigned char const INTRIN_COMPTYPE_FROM_TYPE_ELT0 = 0xff;

enum LEGAL_INTRINSIC_COMPTYPES {
    LICOMPTYPE_VOID = 0,            // void, used for function returns
    LICOMPTYPE_BOOL = 1,            // bool
    LICOMPTYPE_INT = 2,             // i32, int-literal
    LICOMPTYPE_UINT = 3,            // u32, int-literal
    LICOMPTYPE_ANY_INT = 4,         // i32, u32, i64, u64, int-literal
    LICOMPTYPE_ANY_INT32 = 5,       // i32, u32, int-literal
    LICOMPTYPE_UINT_ONLY = 6,       // u32, u64, int-literal; no casts allowed
    LICOMPTYPE_FLOAT = 7,           // f32, partial-precision-f32, float-literal
    LICOMPTYPE_ANY_FLOAT = 8,       // f32, partial-precision-f32, f64, float-literal, min10-float, min16-float, half
    LICOMPTYPE_FLOAT_LIKE = 9,      // f32, partial-precision-f32, float-literal, min10-float, min16-float, half
    LICOMPTYPE_FLOAT_DOUBLE = 10,   // f32, partial-precision-f32, f64, float-literal
    LICOMPTYPE_DOUBLE = 11,         // f64, float-literal
    LICOMPTYPE_DOUBLE_ONLY = 12,    // f64; no casts allowed
    LICOMPTYPE_NUMERIC = 13,        // float-literal, f32, partial-precision-f32, f64, min10-float, min16-float, int-literal, i32, u32, min12-int, min16-int, min16-uint, i64, u64
    LICOMPTYPE_NUMERIC32 = 14,      // float-literal, f32, partial-precision-f32, int-literal, i32, u32
    LICOMPTYPE_NUMERIC32_ONLY = 15, // float-literal, f32, partial-precision-f32, int-literal, i32, u32; no casts allowed
    LICOMPTYPE_ANY = 16,            // float-literal, f32, partial-precision-f32, f64, min10-float, min16-float, int-literal, i32, u32, min12-int, min16-int, min16-uint, bool, i64, u64
    LICOMPTYPE_SAMPLER1D = 17,
    LICOMPTYPE_SAMPLER2D = 18,
    LICOMPTYPE_SAMPLER3D = 19,
    LICOMPTYPE_SAMPLERCUBE = 20,
    LICOMPTYPE_SAMPLERCMP = 21,
    LICOMPTYPE_SAMPLER = 22,
    LICOMPTYPE_STRING = 23,
    LICOMPTYPE_WAVE = 24,
    LICOMPTYPE_UINT64 = 25,         // u64, int-literal
    LICOMPTYPE_FLOAT16 = 26,
    LICOMPTYPE_INT16 = 27,
    LICOMPTYPE_UINT16 = 28,
    LICOMPTYPE_NUMERIC16_ONLY = 29,

    LICOMPTYPE_RAYDESC = 30,
    LICOMPTYPE_ACCELERATION_STRUCT = 31,
    LICOMPTYPE_USER_DEFINED_TYPE = 32,

    LICOMPTYPE_COUNT = 33
};

static unsigned char const IA_SPECIAL_BASE = 0xf0;
static unsigned char const IA_R = 0xf0;
static unsigned char const IA_C = 0xf1;
static unsigned char const IA_R2 = 0xf2;
static unsigned char const IA_C2 = 0xf3;
static unsigned char const IA_SPECIAL_SLOTS = 4;


struct HLSL_intrinsic_argument {
    char const    *pName;                ///< Name of the argument; the first argument has
                                         ///  the function name.
    unsigned      qwUsage;               ///< A combination of AR_QUAL_IN|AR_QUAL_OUT|AR_QUAL_COLMAJOR|AR_QUAL_ROWMAJOR in parameter tables; other values possible elsewhere.

    unsigned char uTemplateId;           ///< One of INTRIN_TEMPLATE_FROM_TYPE, INTRIN_TEMPLATE_VARARGS or the argument # the template (layout) must match (trivially itself).
    unsigned char uLegalTemplates;       ///< A LEGAL_INTRINSIC_TEMPLATES value for allowed templates.
    unsigned char uComponentTypeId;      ///< INTRIN_COMPTYPE_FROM_TYPE_ELT0, or the argument # the component (element type) must match (trivially itself).
    unsigned char uLegalComponentTypes;  ///< A LEGAL_intRINSIC_COMPTYPES value for allowed components.

    unsigned char uRows;                 ///< Required number of rows, or one of IA_R/IA_C/IA_R2/IA_C2 for matching input constraints.
    unsigned char uCols;                 ///< Required number of cols, or one of IA_R/IA_C/IA_R2/IA_C2 for matching input constraints.
};

enum HLSL_memory_access {
    MA_READ_NONE,
    MA_READ_ONLY,
    MA_WRITE
};

struct HLSL_intrinsic {
    Def_function::Semantics Sema;                ///< intrinsic Op ID
    HLSL_memory_access      MemoryAccess;        ///< How memory is accessed
    int                     iOverloadParamIndex; ///< Parameter decide the overload type, -1 means ret type
    unsigned                uNumArgs;            ///< Count of arguments in pArgs.
    HLSL_intrinsic_argument const *pArgs;        ///< Pointer to first argument.
};

#include "hlsl_intrinsics.i"

class Intrinsic_generator {

    /// Add all intrinsics.
    static void add_hlsl_intrinsics(Compilation_unit &unit)
    {
        Intrinsic_generator G(unit);

        G.generate_intrinsics(g_Intrinsics);
        G.generate_intrinsics(g_StreamMethods);
        G.generate_intrinsics(g_Texture1DMethods);
        G.generate_intrinsics(g_Texture1DArrayMethods);
        G.generate_intrinsics(g_Texture2DMethods);
        G.generate_intrinsics(g_Texture2DMSMethods);
        G.generate_intrinsics(g_Texture2DArrayMethods);
        G.generate_intrinsics(g_Texture2DArrayMSMethods);
        G.generate_intrinsics(g_Texture3DMethods);
        G.generate_intrinsics(g_TextureCUBEMethods);
        G.generate_intrinsics(g_TextureCUBEArrayMethods);
        G.generate_intrinsics(g_BufferMethods);
        G.generate_intrinsics(g_RWTexture1DMethods);
        G.generate_intrinsics(g_RWTexture1DArrayMethods);
        G.generate_intrinsics(g_RWTexture2DMethods);
        G.generate_intrinsics(g_RWTexture2DArrayMethods);
        G.generate_intrinsics(g_RWTexture3DMethods);
        G.generate_intrinsics(g_RWBufferMethods);
        G.generate_intrinsics(g_ByteAddressBufferMethods);
        G.generate_intrinsics(g_RWByteAddressBufferMethods);
        G.generate_intrinsics(g_StructuredBufferMethods);
        G.generate_intrinsics(g_RWStructuredBufferMethods);
        G.generate_intrinsics(g_AppendStructuredBufferMethods);
        G.generate_intrinsics(g_ConsumeStructuredBufferMethods);
#ifdef ENABLE_SPIRV_CODEGEN
        G.generate_intrinsics(g_VkSubpassInputMethods);
        G.generate_intrinsics(g_VkSubpassInputMSMethods);
#endif // ENABLE_SPIRV_CODEGEN
    }

private:
    /// Generate intrinsic entries for a whole class of intrinsics.
    void generate_intrinsics(Array_ref<HLSL_intrinsic> const &intrinsics)
    {
        for (size_t i = 0, n = intrinsics.size(); i < n; ++i) {
            generate_intrinsic(intrinsics[i]);
        }
    }

    /// Generate one intrinsic (with overloads).
    void generate_intrinsic(HLSL_intrinsic const &intrinsic);

private:
    /// Constructor.
    ///
    /// \param unit   the unit we generate intrinsics for.
    Intrinsic_generator(Compilation_unit &unit)
    : m_deftab(unit.get_definition_table())
    {
    }

private:
    /// The definition table of the current unit.
    Definition_table &m_deftab;
};

// Generate one intrinsic (with overloads).
void Intrinsic_generator::generate_intrinsic(HLSL_intrinsic const &intrinsic)
{

}


}  // hlsl
}  // mdl
}  // mi
