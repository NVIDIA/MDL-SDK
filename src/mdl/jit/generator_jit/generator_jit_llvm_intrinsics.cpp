/******************************************************************************
 * Copyright (c) 2013-2018, NVIDIA CORPORATION. All rights reserved.
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

#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <base/system/stlext/i_stlext_restore.h>

#include <vector>

#include <llvm/Analysis/Verifier.h>
#include <llvm/Bitcode/ReaderWriter.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/JIT.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InlineAsm.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/CodeGen.h>
#include <llvm/Support/Dwarf.h>
#include <llvm/Support/ManagedStatic.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/PrettyStackTrace.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>
#include <llvm/DIBuilder.h>
#include <llvm/Linker.h>
#include <llvm/PassManager.h>

#include "generator_jit_llvm.h"
#include "generator_jit_context.h"
#include "generator_jit_generated_code.h"

#include <mi/mdl/mdl_generated_executable.h>
#include <mi/mdl/mdl_streams.h>
#include "mdl/codegenerators/generator_dag/generator_dag_tools.h"
#include <mdl/compiler/compilercore/compilercore_assert.h>
#include <mdl/compiler/compilercore/compilercore_errors.h>
#include <mdl/compiler/compilercore/compilercore_mangle.h>
#include <mdl/compiler/compilercore/compilercore_tools.h>
#include <mdl/runtime/spectral/i_spectral.h>

namespace mi {
namespace mdl {

// Ugly reference.
IAllocator *get_debug_log_allocator();

namespace debug {

/// Handle debugbreak
void (debugbreak)()
{
#ifdef WIN_NT
    DebugBreak();
#elif defined(__GNUC__) && (defined(__i386__) || defined(__x86_64))
    __asm__ __volatile__("int3");
#else
    /* poor unix way */
    raise(SIGINT);
#endif
}

/// Handle assert.
void (assertfail)(
    char const *reason,
    char const *func_name,
    char const *file_name,
    int        line)
{
    (void)func_name;
     ::mi::mdl::report_assertion_failure(reason, file_name, line);
}

/// Helper class to print a whole line.
class Line_buffer
{
public:
    /// Constructor.
    Line_buffer(mi::mdl::IAllocator *alloc)
    : m_alloc(alloc), m_string(alloc)
    {
    }

    /// Destroy and flush the line buffer.
    void destroy() {
        IAllocator *alloc = m_alloc;
        this->~Line_buffer();
        alloc->free(this);
    }

    /// Add a string to the buffer.
    void add(char const *s) {
        m_string += s;
    }

    /// Add a character to the buffer.
    void add(char c) {
        m_string += c;
    }

private:
    /// Destructor.
    ~Line_buffer() {
        if (i_debug_log != NULL) {
            i_debug_log->write(m_string.c_str());
        } else {
            printf("%s", m_string.c_str());
        }
    }

private:
    mi::mdl::IAllocator *m_alloc;
    mi::mdl::string m_string;
};

/// Start a print buffer.
Line_buffer *print_begin()
{
    mi::mdl::IAllocator *alloc = mi::mdl::get_debug_log_allocator();
    mi::mdl::Allocator_builder builder(alloc);

    return builder.create<Line_buffer>(alloc);
}

/// End a print buffer.
void print_end(Line_buffer *buffer)
{
    buffer->destroy();
}

/// Handle printing of bool.
void print_bool(Line_buffer *buffer, bool value)
{
    buffer->add(value ? "true" : "false");
}

/// Handle printing of int.
void print_int(Line_buffer *buffer, int value)
{
    char buf[32];
    snprintf(buf, sizeof(buf), "%d", value);
    buf[sizeof(buf) - 1] = '\0';
    buffer->add(buf);
}

/// Handle printing of float.
void print_float(Line_buffer *buffer, float value)
{
    char buf[32];
    snprintf(buf, sizeof(buf),"%f", value);
    buf[sizeof(buf) - 1] = '\0';
    buffer->add(buf);
}

/// Handle printing of double.
void print_double(Line_buffer *buffer, double value)
{
    char buf[32];
    snprintf(buf, sizeof(buf),"%f", value);
    buf[sizeof(buf) - 1] = '\0';
    buffer->add(buf);
}

/// Handle printing of string.
void print_string(Line_buffer *buffer, char const *value)
{
    buffer->add(value);
}

}  // debug

namespace {

typedef IResource_handler::Tex_wrap_mode                Tex_wrap_mode;
typedef IResource_handler::Deriv_float2                 Deriv_float2;
typedef IResource_handler::Mbsdf_part                   Mbsdf_part;
typedef Generated_code_lambda_function::Res_data_pair   Res_data_pair;
typedef Generated_code_lambda_function::Res_data        Res_data;

/// Glue function for tex::width(texture_2d, int2) and tex::height(texture_2d, int2)
void tex_resolution_2d(
    int                 result[2],
    Res_data_pair const *data,
    unsigned            texture,
    int const           uv_tile[2])
{
    Res_data const          *res_data = data->get_shared_data();
    IResource_handler const *handler  = res_data->get_resource_handler();
    if (handler != NULL && texture != 0) {
        handler->tex_resolution_2d(result, res_data->get_resource_store(texture - 1), uv_tile);
    } else {
        result[0] = 0;
        result[1] = 0;
    }
}

/// Glue function for tex::width(texture_*)
int tex_width(
    Res_data_pair const *data,
    unsigned            texture)
{
    Res_data const          *res_data = data->get_shared_data();
    IResource_handler const *handler  = res_data->get_resource_handler();
    if (handler != NULL && texture != 0) {
        return handler->tex_width(res_data->get_resource_store(texture - 1));
    } else {
        return 0;
    }
}

/// Glue function for tex::height(texture_*)
int tex_height(
    Res_data_pair const *data,
    unsigned            texture)
{
    Res_data const          *res_data = data->get_shared_data();
    IResource_handler const *handler  = res_data->get_resource_handler();
    if (handler != NULL && texture != 0) {
        return handler->tex_height(res_data->get_resource_store(texture - 1));
    } else {
        return 0;
    }
}

/// Glue function for tex::depth(texture_*)
int tex_depth(
    Res_data_pair const *data,
    unsigned            texture)
{
    Res_data const          *res_data = data->get_shared_data();
    IResource_handler const *handler  = res_data->get_resource_handler();
    if (handler != NULL && texture != 0) {
        return handler->tex_depth(res_data->get_resource_store(texture - 1));
    } else {
        return 0;
    }
}

/// Glue function for tex::texture_isvalid(texture_*)
bool tex_isvalid(
    Res_data_pair const *data,
    unsigned            texture)
{
    Res_data const          *res_data = data->get_shared_data();
    IResource_handler const *handler  = res_data->get_resource_handler();
    if (handler != NULL && texture != 0) {
        return handler->tex_isvalid(res_data->get_resource_store(texture - 1));
    } else {
        return false;
    }
}

/// Glue function for tex::lookup_float(texture_2d, ...)
float tex_lookup_float_2d(
    Res_data_pair const *data,
    unsigned            texture,
    float const         coord[2],
    Tex_wrap_mode       wrap_u,
    Tex_wrap_mode       wrap_v,
    float const         crop_u[2],
    float const         crop_v[2])
{
    Res_data const          *res_data = data->get_shared_data();
    IResource_handler const *handler  = res_data->get_resource_handler();
    if (handler != NULL && texture != 0) {
        return handler->tex_lookup_float_2d(
            res_data->get_resource_store(texture - 1), data->get_thread_data(),
            coord, wrap_u, wrap_v, crop_u, crop_v);
    } else {
        return 0.0f;
    }
}

/// Glue function for tex::lookup_float(texture_2d, ...) with derivatives
float tex_lookup_deriv_float_2d(
    Res_data_pair const *data,
    unsigned            texture,
    Deriv_float2 const  *coord,
    Tex_wrap_mode       wrap_u,
    Tex_wrap_mode       wrap_v,
    float const         crop_u[2],
    float const         crop_v[2])
{
    Res_data const          *res_data = data->get_shared_data();
    IResource_handler const *handler  = res_data->get_resource_handler();
    if (handler != NULL && texture != 0) {
        return handler->tex_lookup_deriv_float_2d(
            res_data->get_resource_store(texture - 1), data->get_thread_data(),
            coord, wrap_u, wrap_v, crop_u, crop_v);
    } else {
        return 0.0f;
    }
}

/// Glue function for tex::lookup_float(texture_3d, ...)
float tex_lookup_float_3d(
    Res_data_pair const *data,
    unsigned            texture,
    float const         coord[3],
    Tex_wrap_mode       wrap_u,
    Tex_wrap_mode       wrap_v,
    Tex_wrap_mode       wrap_w,
    float const         crop_u[2],
    float const         crop_v[2],
    float const         crop_w[2])
{
    Res_data const          *res_data = data->get_shared_data();
    IResource_handler const *handler  = res_data->get_resource_handler();
    if (handler != NULL && texture != 0) {
        return handler->tex_lookup_float_3d(
            res_data->get_resource_store(texture - 1), data->get_thread_data(),
            coord, wrap_u, wrap_v, wrap_w, crop_u, crop_v, crop_w);
    } else {
        return 0.0f;
    }
}

/// Glue function for tex::lookup_float(texture_cube, ...)
float tex_lookup_float_cube(
    Res_data_pair const *data,
    unsigned            texture,
    float const         coord[3])
{
    Res_data const          *res_data = data->get_shared_data();
    IResource_handler const *handler  = res_data->get_resource_handler();
    if (handler != NULL && texture != 0) {
        return handler->tex_lookup_float_cube(
            res_data->get_resource_store(texture - 1), data->get_thread_data(), coord);
    } else {
        return 0.0f;
    }
}

/// Glue function for tex::lookup_float(texture_ptex, ...)
float tex_lookup_float_ptex(
    Res_data_pair const *data,
    unsigned            texture,
    int                 channel)
{
    Res_data const          *res_data = data->get_shared_data();
    IResource_handler const *handler  = res_data->get_resource_handler();
    if (handler != NULL && texture != 0) {
        return handler->tex_lookup_float_ptex(
            res_data->get_resource_store(texture - 1), data->get_thread_data(), channel);
    } else {
        return 0.0f;
    }
}

/// Glue function for tex::lookup_float2(texture_2d, ...)
void tex_lookup_float2_2d(
    float               result[2],
    Res_data_pair const *data,
    unsigned            texture,
    float const         coord[2],
    Tex_wrap_mode       wrap_u,
    Tex_wrap_mode       wrap_v,
    float const         crop_u[2],
    float const         crop_v[2])
{
    Res_data const          *res_data = data->get_shared_data();
    IResource_handler const *handler  = res_data->get_resource_handler();
    if (handler != NULL && texture != 0) {
        handler->tex_lookup_float2_2d(
            result, res_data->get_resource_store(texture - 1), data->get_thread_data(),
            coord, wrap_u, wrap_v, crop_u, crop_v);
    } else {
        result[0] = result[1] = 0.0f;
    }
}

/// Glue function for tex::lookup_float2(texture_2d, ...) with derivatives
void tex_lookup_deriv_float2_2d(
    float               result[2],
    Res_data_pair const *data,
    unsigned            texture,
    Deriv_float2 const  *coord,
    Tex_wrap_mode       wrap_u,
    Tex_wrap_mode       wrap_v,
    float const         crop_u[2],
    float const         crop_v[2])
{
    Res_data const          *res_data = data->get_shared_data();
    IResource_handler const *handler  = res_data->get_resource_handler();
    if (handler != NULL && texture != 0) {
        handler->tex_lookup_deriv_float2_2d(
            result, res_data->get_resource_store(texture - 1), data->get_thread_data(),
            coord, wrap_u, wrap_v, crop_u, crop_v);
    } else {
        result[0] = result[1] = 0.0f;
    }
}

/// Glue function for tex::lookup_float2(texture_3d, ...)
void tex_lookup_float2_3d(
    float               result[2],
    Res_data_pair const *data,
    unsigned            texture,
    float const         coord[3],
    Tex_wrap_mode       wrap_u,
    Tex_wrap_mode       wrap_v,
    Tex_wrap_mode       wrap_w,
    float const         crop_u[2],
    float const         crop_v[2],
    float const         crop_w[2])
{
    Res_data const          *res_data = data->get_shared_data();
    IResource_handler const *handler  = res_data->get_resource_handler();
    if (handler != NULL && texture != 0) {
        handler->tex_lookup_float2_3d(
            result, res_data->get_resource_store(texture - 1), data->get_thread_data(),
            coord, wrap_u, wrap_v, wrap_w, crop_u, crop_v, crop_w);
    } else {
        result[0] = result[1] = 0.0f;
    }
}

/// Glue function for tex::lookup_float2(texture_cube, ...)
void tex_lookup_float2_cube(
    float               result[2],
    Res_data_pair const *data,
    unsigned            texture,
    float const         coord[3])
{
    Res_data const          *res_data = data->get_shared_data();
    IResource_handler const *handler  = res_data->get_resource_handler();
    if (handler != NULL && texture != 0) {
        handler->tex_lookup_float2_cube(
            result, res_data->get_resource_store(texture - 1), data->get_thread_data(), coord);
    } else {
        result[0] = result[1] = 0.0f;
    }
}

/// Glue function for tex::lookup_float2(texture_ptex, ...)
void tex_lookup_float2_ptex(
    float               result[2],
    Res_data_pair const *data,
    unsigned            texture,
    int                 channel)
{
    Res_data const          *res_data = data->get_shared_data();
    IResource_handler const *handler  = res_data->get_resource_handler();
    if (handler != NULL && texture != 0) {
        handler->tex_lookup_float2_ptex(
            result, res_data->get_resource_store(texture - 1), data->get_thread_data(), channel);
    } else {
        result[0] = result[1] = 0.0f;
    }
}

/// Glue function for tex::lookup_float3(texture_2d, ...)
void tex_lookup_float3_2d(
    float               result[3],
    Res_data_pair const *data,
    unsigned            texture,
    float const         coord[2],
    Tex_wrap_mode       wrap_u,
    Tex_wrap_mode       wrap_v,
    float const         crop_u[2],
    float const         crop_v[2])
{
    Res_data const          *res_data = data->get_shared_data();
    IResource_handler const *handler  = res_data->get_resource_handler();
    if (handler != NULL && texture != 0) {
        handler->tex_lookup_float3_2d(
            result, res_data->get_resource_store(texture - 1), data->get_thread_data(),
            coord, wrap_u, wrap_v, crop_u, crop_v);
    } else {
        result[0] = result[1] = result[2] = 0.0f;
    }
}

/// Glue function for tex::lookup_float3(texture_2d, ...) with derivatives
void tex_lookup_deriv_float3_2d(
    float               result[3],
    Res_data_pair const *data,
    unsigned            texture,
    Deriv_float2 const  *coord,
    Tex_wrap_mode       wrap_u,
    Tex_wrap_mode       wrap_v,
    float const         crop_u[2],
    float const         crop_v[2])
{
    Res_data const          *res_data = data->get_shared_data();
    IResource_handler const *handler  = res_data->get_resource_handler();
    if (handler != NULL && texture != 0) {
        handler->tex_lookup_deriv_float3_2d(
            result, res_data->get_resource_store(texture - 1), data->get_thread_data(),
            coord, wrap_u, wrap_v, crop_u, crop_v);
    } else {
        result[0] = result[1] = result[2] = 0.0f;
    }
}

/// Glue function for tex::lookup_float3(texture_3d, ...)
void tex_lookup_float3_3d(
    float               result[3],
    Res_data_pair const *data,
    unsigned            texture,
    float const         coord[3],
    Tex_wrap_mode       wrap_u,
    Tex_wrap_mode       wrap_v,
    Tex_wrap_mode       wrap_w,
    float const         crop_u[2],
    float const         crop_v[2],
    float const         crop_w[2])
{
    Res_data const          *res_data = data->get_shared_data();
    IResource_handler const *handler  = res_data->get_resource_handler();
    if (handler != NULL && texture != 0) {
        handler->tex_lookup_float3_3d(
            result, res_data->get_resource_store(texture - 1), data->get_thread_data(),
            coord, wrap_u, wrap_v, wrap_w, crop_u, crop_v, crop_w);
    } else {
        result[0] = result[1] = result[2] = 0.0f;
    }
}

/// Glue function for tex::lookup_float3(texture_cube, ...)
void tex_lookup_float3_cube(
    float               result[3],
    Res_data_pair const *data,
    unsigned            texture,
    float const         coord[3])
{
    Res_data const          *res_data = data->get_shared_data();
    IResource_handler const *handler  = res_data->get_resource_handler();
    if (handler != NULL && texture != 0) {
        handler->tex_lookup_float3_cube(
            result, res_data->get_resource_store(texture - 1), data->get_thread_data(), coord);
    } else {
        result[0] = result[1] = result[2] = 0.0f;
    }
}

/// Glue function for tex::lookup_float3(texture_ptex, ...)
void tex_lookup_float3_ptex(
    float               result[3],
    Res_data_pair const *data,
    unsigned            texture,
    int                 channel)
{
    Res_data const          *res_data = data->get_shared_data();
    IResource_handler const *handler  = res_data->get_resource_handler();
    if (handler != NULL && texture != 0) {
        handler->tex_lookup_float3_ptex(
            result, res_data->get_resource_store(texture - 1), data->get_thread_data(), channel);
    } else {
        result[0] = result[1] = result[2] = 0.0f;
    }
}

/// Glue function for tex::lookup_float4(texture_2d, ...)
void tex_lookup_float4_2d(
    float               result[4],
    Res_data_pair const *data,
    unsigned            texture,
    float const         coord[2],
    Tex_wrap_mode       wrap_u,
    Tex_wrap_mode       wrap_v,
    float const         crop_u[2],
    float const         crop_v[2])
{
    Res_data const          *res_data = data->get_shared_data();
    IResource_handler const *handler  = res_data->get_resource_handler();
    if (handler != NULL && texture != 0) {
        handler->tex_lookup_float4_2d(
            result, res_data->get_resource_store(texture - 1), data->get_thread_data(),
            coord, wrap_u, wrap_v, crop_u, crop_v);
    } else {
        result[0] = result[1] = result[2] = result[3] = 0.0f;
    }
}

/// Glue function for tex::lookup_float4(texture_2d, ...) with derivatives
void tex_lookup_deriv_float4_2d(
    float               result[4],
    Res_data_pair const *data,
    unsigned            texture,
    Deriv_float2 const  *coord,
    Tex_wrap_mode       wrap_u,
    Tex_wrap_mode       wrap_v,
    float const         crop_u[2],
    float const         crop_v[2])
{
    Res_data const          *res_data = data->get_shared_data();
    IResource_handler const *handler  = res_data->get_resource_handler();
    if (handler != NULL && texture != 0) {
        handler->tex_lookup_deriv_float4_2d(
            result, res_data->get_resource_store(texture - 1), data->get_thread_data(),
            coord, wrap_u, wrap_v, crop_u, crop_v);
    } else {
        result[0] = result[1] = result[2] = result[3] = 0.0f;
    }
}

/// Glue function for tex::lookup_float4(texture_3d, ...)
void tex_lookup_float4_3d(
    float               result[4],
    Res_data_pair const *data,
    unsigned            texture,
    float const         coord[3],
    Tex_wrap_mode       wrap_u,
    Tex_wrap_mode       wrap_v,
    Tex_wrap_mode       wrap_w,
    float const         crop_u[2],
    float const         crop_v[2],
    float const         crop_w[2])
{
    Res_data const          *res_data = data->get_shared_data();
    IResource_handler const *handler  = res_data->get_resource_handler();
    if (handler != NULL && texture != 0) {
        handler->tex_lookup_float4_3d(
            result, res_data->get_resource_store(texture - 1), data->get_thread_data(),
            coord, wrap_u, wrap_v, wrap_w, crop_u, crop_v, crop_w);
    } else {
        result[0] = result[1] = result[2] = result[3] = 0.0f;
    }
}

/// Glue function for tex::lookup_float4(texture_cube, ...)
void tex_lookup_float4_cube(
    float               result[4],
    Res_data_pair const *data,
    unsigned            texture,
    float const         coord[3])
{
    Res_data const          *res_data = data->get_shared_data();
    IResource_handler const *handler  = res_data->get_resource_handler();
    if (handler != NULL && texture != 0) {
        handler->tex_lookup_float4_cube(
            result, res_data->get_resource_store(texture - 1), data->get_thread_data(), coord);
    } else {
        result[0] = result[1] = result[2] = result[3] = 0.0f;
    }
}

/// Glue function for tex::lookup_float4(texture_ptex, ...)
void tex_lookup_float4_ptex(
    float               result[4],
    Res_data_pair const *data,
    unsigned            texture,
    int                 channel)
{
    Res_data const          *res_data = data->get_shared_data();
    IResource_handler const *handler  = res_data->get_resource_handler();
    if (handler != NULL && texture != 0) {
        handler->tex_lookup_float4_ptex(
            result, res_data->get_resource_store(texture - 1), data->get_thread_data(), channel);
    } else {
        result[0] = result[1] = result[2] = result[3] = 0.0f;
    }
}

/// Glue function for tex::lookup_color(texture_2d, ...)
void tex_lookup_color_2d(
    float               rgb[3],
    Res_data_pair const *data,
    unsigned            texture,
    float const         coord[2],
    Tex_wrap_mode       wrap_u,
    Tex_wrap_mode       wrap_v,
    float const         crop_u[2],
    float const         crop_v[2])
{
    Res_data const          *res_data = data->get_shared_data();
    IResource_handler const *handler  = res_data->get_resource_handler();
    if (handler != NULL && texture != 0) {
        handler->tex_lookup_color_2d(
            rgb, res_data->get_resource_store(texture - 1), data->get_thread_data(),
            coord, wrap_u, wrap_v, crop_u, crop_v);
    } else {
        rgb[0] = rgb[1] = rgb[2] = 0.0f;
    }
}

/// Glue function for tex::lookup_color(texture_2d, ...) with derivatives
void tex_lookup_deriv_color_2d(
    float               rgb[3],
    Res_data_pair const *data,
    unsigned            texture,
    Deriv_float2 const  *coord,
    Tex_wrap_mode       wrap_u,
    Tex_wrap_mode       wrap_v,
    float const         crop_u[2],
    float const         crop_v[2])
{
    Res_data const          *res_data = data->get_shared_data();
    IResource_handler const *handler  = res_data->get_resource_handler();
    if (handler != NULL && texture != 0) {
        handler->tex_lookup_deriv_color_2d(
            rgb, res_data->get_resource_store(texture - 1), data->get_thread_data(),
            coord, wrap_u, wrap_v, crop_u, crop_v);
    } else {
        rgb[0] = rgb[1] = rgb[2] = 0.0f;
    }
}

/// Glue function for tex::lookup_color(texture_3d, ...)
void tex_lookup_color_3d(
    float               rgb[3],
    Res_data_pair const *data,
    unsigned            texture,
    float const         coord[3],
    Tex_wrap_mode       wrap_u,
    Tex_wrap_mode       wrap_v,
    Tex_wrap_mode       wrap_w,
    float const         crop_u[2],
    float const         crop_v[2],
    float const         crop_w[2])
{
    Res_data const          *res_data = data->get_shared_data();
    IResource_handler const *handler  = res_data->get_resource_handler();
    if (handler != NULL && texture != 0) {
        handler->tex_lookup_color_3d(
            rgb, res_data->get_resource_store(texture - 1), data->get_thread_data(),
            coord, wrap_u, wrap_v, wrap_w, crop_u, crop_v, crop_w);
    } else {
        rgb[0] = rgb[1] = rgb[2] = 0.0f;
    }
}

/// Glue function for tex::lookup_color(texture_cube, ...)
void tex_lookup_color_cube(
    float               rgb[3],
    Res_data_pair const *data,
    unsigned            texture,
    float const         coord[3])
{
    Res_data const          *res_data = data->get_shared_data();
    IResource_handler const *handler  = res_data->get_resource_handler();
    if (handler != NULL && texture != 0) {
        handler->tex_lookup_color_cube(
            rgb, res_data->get_resource_store(texture - 1), data->get_thread_data(), coord);
    } else {
        rgb[0] = rgb[1] = rgb[2] = 0.0f;
    }
}

/// Glue function for tex::lookup_color(texture_ptex, ...)
void tex_lookup_color_ptex(
    float               rgb[3],
    Res_data_pair const *data,
    unsigned            texture,
    int                 channel)
{
    Res_data const          *res_data = data->get_shared_data();
    IResource_handler const *handler  = res_data->get_resource_handler();
    if (handler != NULL && texture != 0) {
        handler->tex_lookup_color_ptex(
            rgb, res_data->get_resource_store(texture - 1), data->get_thread_data(), channel);
    } else {
        rgb[0] = rgb[1] = rgb[2] = 0.0f;
    }
}

/// Glue function for tex::texel_float(texture_2d, ...)
float tex_texel_float_2d(
    Res_data_pair const *data,
    unsigned            texture,
    int const           coord[2],
    int const           uv_tile[2])
{
    Res_data const          *res_data = data->get_shared_data();
    IResource_handler const *handler  = res_data->get_resource_handler();
    if (handler != NULL && texture != 0) {
        return handler->tex_texel_float_2d(
            res_data->get_resource_store(texture - 1), data->get_thread_data(), coord, uv_tile);
    } else {
        return 0.0f;
    }
}

/// Glue function for tex::texel_float2(texture_2d, ...)
void tex_texel_float2_2d(
    float               result[2],
    Res_data_pair const *data,
    unsigned            texture,
    int const           coord[2],
    int const           uv_tile[2])
{
    Res_data const          *res_data = data->get_shared_data();
    IResource_handler const *handler  = res_data->get_resource_handler();
    if (handler != NULL && texture != 0) {
        handler->tex_texel_float2_2d(
            result,
            res_data->get_resource_store(texture - 1),
            data->get_thread_data(),
            coord,
            uv_tile);
    } else {
        result[0] = result[1] = 0.0f;
    }
}

/// Glue function for tex::texel_float3(texture_2d, ...)
void tex_texel_float3_2d(
    float               result[3],
    Res_data_pair const *data,
    unsigned            texture,
    int const           coord[2],
    int const           uv_tile[2])
{
    Res_data const          *res_data = data->get_shared_data();
    IResource_handler const *handler  = res_data->get_resource_handler();
    if (handler != NULL && texture != 0) {
        handler->tex_texel_float3_2d(
            result,
            res_data->get_resource_store(texture - 1),
            data->get_thread_data(),
            coord,
            uv_tile);
    } else {
        result[0] = result[1] = result[2] = 0.0f;
    }
}

/// Glue function for tex::texel_float4(texture_2d, ...)
void tex_texel_float4_2d(
    float               result[4],
    Res_data_pair const *data,
    unsigned            texture,
    int const           coord[2],
    int const           uv_tile[2])
{
    Res_data const          *res_data = data->get_shared_data();
    IResource_handler const *handler  = res_data->get_resource_handler();
    if (handler != NULL && texture != 0) {
        handler->tex_texel_float4_2d(
            result,
            res_data->get_resource_store(texture - 1),
            data->get_thread_data(),
            coord,
            uv_tile);
    } else {
        result[0] = result[1] = result[2] = result[3] = 0.0f;
    }
}

/// Glue function for tex::texel_color(texture_2d, ...)
void tex_texel_color_2d(
    float               rgb[3],
    Res_data_pair const *data,
    unsigned            texture,
    int const           coord[2],
    int const           uv_tile[2])
{
    Res_data const          *res_data = data->get_shared_data();
    IResource_handler const *handler  = res_data->get_resource_handler();
    if (handler != NULL && texture != 0) {
        handler->tex_texel_color_2d(
            rgb,
            res_data->get_resource_store(texture - 1),
            data->get_thread_data(),
            coord,
            uv_tile);
    } else {
        rgb[0] = rgb[1] = rgb[2] = 0.0f;
    }
}

/// Glue function for tex::texel_float(texture_3d, ...)
float tex_texel_float_3d(
    Res_data_pair const *data,
    unsigned            texture,
    int const           coord[3])
{
    Res_data const          *res_data = data->get_shared_data();
    IResource_handler const *handler  = res_data->get_resource_handler();
    if (handler != NULL && texture != 0) {
        return handler->tex_texel_float_3d(
            res_data->get_resource_store(texture - 1), data->get_thread_data(), coord);
    } else {
        return 0.0f;
    }
}

/// Glue function for tex::texel_float2(texture_3d, ...)
void tex_texel_float2_3d(
    float               result[2],
    Res_data_pair const *data,
    unsigned            texture,
    int const           coord[3])
{
    Res_data const          *res_data = data->get_shared_data();
    IResource_handler const *handler  = res_data->get_resource_handler();
    if (handler != NULL && texture != 0) {
        handler->tex_texel_float2_3d(
            result, res_data->get_resource_store(texture - 1), data->get_thread_data(), coord);
    } else {
        result[0] = result[1] = 0.0f;
    }
}

/// Glue function for tex::texel_float3(texture_3d, ...)
void tex_texel_float3_3d(
    float               result[3],
    Res_data_pair const *data,
    unsigned            texture,
    int const           coord[3])
{
    Res_data const          *res_data = data->get_shared_data();
    IResource_handler const *handler  = res_data->get_resource_handler();
    if (handler != NULL && texture != 0) {
        handler->tex_texel_float3_3d(
            result, res_data->get_resource_store(texture - 1), data->get_thread_data(), coord);
    } else {
        result[0] = result[1] = result[2] = 0.0f;
    }
}

/// Glue function for tex::texel_float4(texture_3d, ...)
void tex_texel_float4_3d(
    float               result[4],
    Res_data_pair const *data,
    unsigned            texture,
    int const           coord[3])
{
    Res_data const          *res_data = data->get_shared_data();
    IResource_handler const *handler  = res_data->get_resource_handler();
    if (handler != NULL && texture != 0) {
        handler->tex_texel_float4_3d(
            result, res_data->get_resource_store(texture - 1), data->get_thread_data(), coord);
    } else {
        result[0] = result[1] = result[2] = result[3] = 0.0f;
    }
}

/// Glue function for tex::texel_color(texture_3d, ...)
void tex_texel_color_3d(
    float               rgb[3],
    Res_data_pair const *data,
    unsigned            texture,
    int const           coord[3])
{
    Res_data const          *res_data = data->get_shared_data();
    IResource_handler const *handler  = res_data->get_resource_handler();
    if (handler != NULL && texture != 0) {
        handler->tex_texel_color_3d(
            rgb, res_data->get_resource_store(texture - 1), data->get_thread_data(), coord);
    } else {
        rgb[0] = rgb[1] = rgb[2] = 0.0f;
    }
}

/// Glue function for df::light_profile_power(light_profile)
float df_light_profile_power(
    Res_data_pair const *data,
    unsigned            texture)
{
    Res_data const          *res_data = data->get_shared_data();
    IResource_handler const *handler  = res_data->get_resource_handler();
    if (handler != NULL && texture != 0) {
        return handler->lp_power(
            res_data->get_resource_store(texture - 1), data->get_thread_data());
    } else {
        return 0.0f;
    }
}

/// Glue function for df::light_profile_maximum(light_profile)
float df_light_profile_maximum(
    Res_data_pair const *data,
    unsigned            texture)
{
    Res_data const          *res_data = data->get_shared_data();
    IResource_handler const *handler  = res_data->get_resource_handler();
    if (handler != NULL && texture != 0) {
        return handler->lp_maximum(
            res_data->get_resource_store(texture - 1), data->get_thread_data());
    } else {
        return 0.0f;
    }
}

/// Glue function for df::light_profile_isvalid(light_profile)
bool df_light_profile_isvalid(
    Res_data_pair const *data,
    unsigned            texture)
{
    Res_data const          *res_data = data->get_shared_data();
    IResource_handler const *handler  = res_data->get_resource_handler();
    if (handler != NULL && texture != 0) {
        return handler->lp_isvalid(res_data->get_resource_store(texture - 1));
    } else {
        return false;
    }
}

/// Glue function for df::bsdf_measurement_isvalid(bsdf_measurement)
bool df_bsdf_measurement_isvalid(
    Res_data_pair const *data,
    unsigned            texture)
{
    Res_data const          *res_data = data->get_shared_data();
    IResource_handler const *handler  = res_data->get_resource_handler();
    if (handler != NULL && texture != 0) {
        return handler->bm_isvalid(res_data->get_resource_store(texture - 1));
    } else {
        return false;
    }
}


/// Glue function for df::bsdf_measurement_resolution(bsdf_measurement, ...)
void df_bsdf_measurement_resolution(
    unsigned            result[3],
    Res_data_pair const *data,
    unsigned            resource,
    Mbsdf_part          part)
{
    Res_data const          *res_data = data->get_shared_data();
    IResource_handler const *handler = res_data->get_resource_handler();
    if (handler != NULL && resource != 0)
    {
        handler->bm_resolution(result, res_data->get_resource_store(resource - 1), part);
    }
    else
    {
        result[0] = 0;
        result[1] = 0;
        result[2] = 0;
    }
}

/// Glue function for df::bsdf_measurement_evaluate(bsdf_measurement, ...)
void df_bsdf_measurement_evaluate(
    float               result[3],
    Res_data_pair const *data,
    unsigned            resource,
    float const         theta_phi_in[2],
    float const         theta_phi_out[2],
    Mbsdf_part          part)
{
    Res_data const          *res_data = data->get_shared_data();
    IResource_handler const *handler = res_data->get_resource_handler();
    if (handler != NULL && resource != 0)
    {
        handler->bm_evaluate(result, res_data->get_resource_store(resource - 1),
                             data->get_thread_data(), theta_phi_in, theta_phi_out, part);
    }
    else
    {
        result[0] = 0.0f;
        result[1] = 0.0f;
        result[2] = 0.0f;
    }
}

/// Glue function for df::bsdf_measurement_sample(bsdf_measurement, ...)
void df_bsdf_measurement_sample(
    float               result[3],
    Res_data_pair const *data,
    unsigned            resource,
    float const         theta_phi_out[2],
    float const         xi[3],
    Mbsdf_part          part)
{
    Res_data const          *res_data = data->get_shared_data();
    IResource_handler const *handler = res_data->get_resource_handler();
    if (handler != NULL && resource != 0)
    {
        handler->bm_sample(result, res_data->get_resource_store(resource - 1),
                           data->get_thread_data(), theta_phi_out, xi, part);
    }
    else
    {
        result[0] = 0.0f;
        result[1] = 0.0f;
        result[2] = 1.0f;
    }
}

/// Glue function for df::bsdf_measurement_pdf(bsdf_measurement, ...)
float df_bsdf_measurement_pdf(
    Res_data_pair const *data,
    unsigned            resource,
    float const         theta_phi_in[2],
    float const         theta_phi_out[2],
    Mbsdf_part          part)
{
    Res_data const          *res_data = data->get_shared_data();
    IResource_handler const *handler = res_data->get_resource_handler();
    if (handler != NULL && resource != 0)
        return handler->bm_pdf(res_data->get_resource_store(resource - 1),
                           data->get_thread_data(), theta_phi_in, theta_phi_out, part);
    return 0.0f;
}

/// Glue function for df::bsdf_measurement_albedos(bsdf_measurement, ...)
void df_bsdf_measurement_albedos(
    float               result[4],
    Res_data_pair const *data,
    unsigned            resource,
    float const         theta_phi[2])
{
    Res_data const          *res_data = data->get_shared_data();
    IResource_handler const *handler = res_data->get_resource_handler();
    if (handler != NULL && resource != 0)
    {
        handler->bm_albedos(result, res_data->get_resource_store(resource - 1),
                           data->get_thread_data(), theta_phi);
    }
    else
    {
        result[0] = 0.0f;
        result[1] = 0.0f;
        result[2] = 0.0f;
        result[3] = 0.0f;
    }
}

/// Glue function for df::light_profile_evaluate(light_profile, ...)
float df_light_profile_evaluate(
    Res_data_pair const *data,
    unsigned            resource,
    float const         theta_phi[2])
{
    Res_data const          *res_data = data->get_shared_data();
    IResource_handler const *handler = res_data->get_resource_handler();
    if (handler != NULL && resource != 0)
        return handler->lp_evaluate(res_data->get_resource_store(resource - 1),
                           data->get_thread_data(), theta_phi);
    return 0.0f;
}

/// Glue function for df::light_profile_sample(light_profile, ...)
void df_light_profile_sample(
    float               result[3],
    Res_data_pair const *data,
    unsigned            resource,
    float const         xi[3])
{
    Res_data const          *res_data = data->get_shared_data();
    IResource_handler const *handler = res_data->get_resource_handler();
    if (handler != NULL && resource != 0)
    {
        handler->lp_sample(result, res_data->get_resource_store(resource - 1),
                           data->get_thread_data(), xi);
    }
    else
    {
        result[0] = 0.0f;
        result[1] = 0.0f;
        result[2] = 1.0f;
    }
}

/// Glue function for df::light_profile_pdf(light_profile, ...)
float df_light_profile_pdf(
    Res_data_pair const *data,
    unsigned            resource,
    float const         theta_phi[2])
{
    Res_data const          *res_data = data->get_shared_data();
    IResource_handler const *handler = res_data->get_resource_handler();
    if (handler != NULL && resource != 0)
        return handler->lp_pdf(res_data->get_resource_store(resource - 1),
                           data->get_thread_data(), theta_phi);
    return 0.0f;
}

} // anonymous

const int VPRINTF_BUFFER_ALIGNMENT = 8;  // In Bytes.
const int VPRINTF_BUFFER_ROUND_UP = 8;   // In Bytes.


#include "generator_jit_intrinsic_func.i"

// Signature types.
typedef mi::mdl::debug::Line_buffer LB;
typedef int    (*II_II)(int);
typedef int    (*II_IIII)(int, int);
typedef float  (*FF_FF)(float);
typedef float  (*FF_FFFF)(float, float);
typedef float  (*FF_FFff)(float, float *);
typedef float  (*FF_FFII)(float, int);
typedef double (*DD_DD)(double);
typedef double (*DD_DDDD)(double, double);
typedef double (*DD_DDdd)(double, double *);
typedef double (*DD_DDII)(double, int);
typedef void   (*VV_FFffff)(float, float *, float *);
typedef void   (*FA3_FF)(float *, float);
typedef void   (*VV_)(void);
typedef void   (*VV_CSCSCSII)(char const *, char const *, char const *, int line);
typedef LB     *(*lb_)(void);
typedef void   (*VV_lb)(LB *);
typedef void   (*VV_lbBB)(LB *, bool);
typedef void   (*VV_lbII)(LB *, int);
typedef void   (*VV_lbFF)(LB *, float);
typedef void   (*VV_lbDD)(LB *, double);
typedef void   (*VV_lbCS)(LB *, char const *);
typedef void   (*VV_xsIIZZCSII)(LLVM_code_generator::Exc_state &, int, size_t, char const *, int);
typedef void   (*VV_xsCSII)(LLVM_code_generator::Exc_state &, char const *, int);

template <typename Signature>
struct Signature_trait {
    enum V { NO_CAPTURE_ARG_IDX = -1 };
};
template <>
struct Signature_trait<FF_FFff> {
    enum V { NO_CAPTURE_ARG_IDX = 2 };
};
template <>
struct Signature_trait<DD_DDdd> {
    enum V { NO_CAPTURE_ARG_IDX = 2 };
};
template <>
struct Signature_trait<VV_FFffff> {
    enum V { NO_CAPTURE_ARG_IDX = 0x23 };  // 2 AND 3 argument
};

template<typename FUNC_PTR>
static void *check_sig(FUNC_PTR func) { return (void *)func; }

/// Add extra function attributes for C-runtime functions.
///
/// \param func            the function
/// \param nocapture_idx   if >= 0, the index of a DoesNotCapture parameter
static void add_attributes(llvm::Function *func, int nocapture_idx)
{
    func->setDoesNotThrow();
    if (nocapture_idx > 0) {
        if (nocapture_idx >= 0x10) {
            func->setDoesNotCapture(unsigned(nocapture_idx) >> 4);
            nocapture_idx = nocapture_idx & 0xF;
        }
        func->setDoesNotCapture(unsigned(nocapture_idx));
    }
    else
        func->setOnlyReadsMemory();
}

//#ifdef MI_ENABLE_MISTD
// no powi in C++11
static double powi(double a, int b)
{
    return ::pow(a, double(b));
}
//#endif

/// Check if libdevice has a "fast" variant.
static bool has_fast_variant(MDL_runtime_creator::Runtime_function code)
{
    switch (code) {
    case MDL_runtime_creator::RT_COSF:
    case MDL_runtime_creator::RT_SINF:
    case MDL_runtime_creator::RT_TANF:
    case MDL_runtime_creator::RT_SINCOSF:
    case MDL_runtime_creator::RT_EXPF:
    case MDL_runtime_creator::RT_LOGF:
    case MDL_runtime_creator::RT_LOG2F:
    case MDL_runtime_creator::RT_LOG10F:
    case MDL_runtime_creator::RT_POWF:
        return true;
    default:
        return false;
    }
}

// Get an external C-runtime function.
llvm::Function *MDL_runtime_creator::get_c_runtime_func(
    Runtime_function code,
    char const *signature)
{
#define EXTERNAL_CMATH(en, name, type) \
case en: \
    { \
        func = decl_from_signature(#name, signature, is_sret); \
        add_attributes(func, Signature_trait<type>::NO_CAPTURE_ARG_IDX); \
        func->setCallingConv(llvm::CallingConv::C); \
        func->setLinkage(llvm::GlobalValue::ExternalLinkage); \
        check_sig<type>(::name); \
    } \
    break;

#define EXTERNAL_CMATH2(en, name, local_name, type) \
case en: \
    { \
        func = decl_from_signature(#name, signature, is_sret); \
        add_attributes(func, Signature_trait<type>::NO_CAPTURE_ARG_IDX); \
        func->setCallingConv(llvm::CallingConv::C); \
        func->setLinkage(llvm::GlobalValue::ExternalLinkage); \
        check_sig<type>(local_name); \
    } \
    break;

#define _LLVM_INTRINSIC(en, name) \
case en: \
    func = decl_from_signature("llvm." name, signature, is_sret); \
    func->setLinkage(llvm::GlobalValue::ExternalLinkage); \
    break;

#define LLVM_INTRINSIC(en, name) \
    _LLVM_INTRINSIC(en##F, #name ".f32") \
    _LLVM_INTRINSIC(en,    #name ".f64")

#define EXTERNAL_LIBDEVICE(en, name, type) \
case en: \
    { \
        func = decl_from_signature( \
            m_fast_math && has_fast_variant(en) ? \
                "__nv_fast_" #name : "__nv_" #name, \
            signature, is_sret); \
        add_attributes(func, Signature_trait<type>::NO_CAPTURE_ARG_IDX); \
        func->setLinkage(llvm::GlobalValue::ExternalLinkage); \
    } \
    break;

    llvm::Function *func = NULL;
    bool is_sret = false;
    if (m_ptx_mode) {
        // use libdevice calls
        switch (code) {
        EXTERNAL_LIBDEVICE(RT_ABSF,      fabsf,     FF_FF);
        EXTERNAL_LIBDEVICE(RT_ABS,       fabs,      DD_DD);
        EXTERNAL_LIBDEVICE(RT_ABSI,      abs,       II_II);
        EXTERNAL_LIBDEVICE(RT_ACOSF,     acosf,     FF_FF);
        EXTERNAL_LIBDEVICE(RT_ACOS,      acos,      DD_DD);
        EXTERNAL_LIBDEVICE(RT_ASINF,     asinf,     FF_FF);
        EXTERNAL_LIBDEVICE(RT_ASIN,      asin,      DD_DD);
        EXTERNAL_LIBDEVICE(RT_ATANF,     atanf,     FF_FF);
        EXTERNAL_LIBDEVICE(RT_ATAN,      atan,      DD_DD);
        EXTERNAL_LIBDEVICE(RT_ATAN2F,    atan2f,    FF_FFFF);
        EXTERNAL_LIBDEVICE(RT_ATAN2,     atan2,     DD_DDDD);
        EXTERNAL_LIBDEVICE(RT_CEILF,     ceilf,     FF_FF);
        EXTERNAL_LIBDEVICE(RT_CEIL,      ceil,      DD_DD);
        EXTERNAL_LIBDEVICE(RT_COSF,      cosf,      FF_FF);
        EXTERNAL_LIBDEVICE(RT_COS,       cos,       DD_DD);
        EXTERNAL_LIBDEVICE(RT_EXPF,      expf,      FF_FF);
        EXTERNAL_LIBDEVICE(RT_EXP,       exp,       DD_DD);
        EXTERNAL_LIBDEVICE(RT_FLOORF,    floorf,    FF_FF);
        EXTERNAL_LIBDEVICE(RT_FLOOR,     floor,     DD_DD);
        EXTERNAL_LIBDEVICE(RT_FMODF,     fmodf,     FF_FFFF);
        EXTERNAL_LIBDEVICE(RT_FMOD,      fmod,      DD_DDDD);
        EXTERNAL_LIBDEVICE(RT_LOGF,      logf,      FF_FF);
        EXTERNAL_LIBDEVICE(RT_LOG,       log,       DD_DD);
        EXTERNAL_LIBDEVICE(RT_LOG10F,    log10f,    FF_FF);
        EXTERNAL_LIBDEVICE(RT_LOG10,     log10,     DD_DD);
        EXTERNAL_LIBDEVICE(RT_MODFF,     modff,     FF_FFff);
        EXTERNAL_LIBDEVICE(RT_MODF,      modf,      DD_DDdd);
        EXTERNAL_LIBDEVICE(RT_POWF,      powf,      FF_FFFF);
        EXTERNAL_LIBDEVICE(RT_POW,       pow,       DD_DDDD);
        EXTERNAL_LIBDEVICE(RT_POWI,      powi,      DD_DDII);
        EXTERNAL_LIBDEVICE(RT_SINF,      sinf,      FF_FF);
        EXTERNAL_LIBDEVICE(RT_SIN,       sin,       DD_DD);
        EXTERNAL_LIBDEVICE(RT_SQRTF,     sqrtf,     FF_FF);
        EXTERNAL_LIBDEVICE(RT_SQRT,      sqrt,      DD_DD);
        EXTERNAL_LIBDEVICE(RT_TANF,      tanf,      FF_FF);
        EXTERNAL_LIBDEVICE(RT_TAN,       tan,       DD_DD);
        EXTERNAL_LIBDEVICE(RT_COPYSIGNF, copysignf, FF_FFFF);
        EXTERNAL_LIBDEVICE(RT_COPYSIGN,  copysign,  DD_DDDD);
        EXTERNAL_LIBDEVICE(RT_SINCOSF,   sincosf,   VV_FFffff);
        EXTERNAL_LIBDEVICE(RT_LOG2F,     log2f,     FF_FF);
        EXTERNAL_LIBDEVICE(RT_LOG2,      log2,      DD_DD);
        EXTERNAL_LIBDEVICE(RT_EXP2F,     exp2f,     FF_FF);
        EXTERNAL_LIBDEVICE(RT_EXP2,      exp2,      DD_DD);
        EXTERNAL_LIBDEVICE(RT_MINI,      min,       II_IIII);
        EXTERNAL_LIBDEVICE(RT_MAXI,      max,       II_IIII);
        EXTERNAL_LIBDEVICE(RT_MINF,      fminf,     FF_FFFF);
        EXTERNAL_LIBDEVICE(RT_MAXF,      fmaxf,     FF_FFFF);
        EXTERNAL_LIBDEVICE(RT_MIN,       fmin,      DD_DDDD);
        EXTERNAL_LIBDEVICE(RT_MAX,       fmax,      DD_DDDD);
        EXTERNAL_LIBDEVICE(RT_RSQRTF,    rsqrtf,    FF_FF);
        EXTERNAL_LIBDEVICE(RT_RSQRT,     rsqrt,     DD_DD);
        default:
            MDL_ASSERT(!"unsupported <cmath> runtime function requested");
            break;
        }
    } else {
        // use default LLVM intrinsics
        switch (code) {
            LLVM_INTRINSIC(RT_ABS,      fabs);
            EXTERNAL_CMATH(RT_ABSI,     abs,   II_II);
            EXTERNAL_CMATH(RT_ACOSF,    acosf, FF_FF);
            EXTERNAL_CMATH(RT_ACOS,     acos,  DD_DD);
            EXTERNAL_CMATH(RT_ASINF,    asinf, FF_FF);
            EXTERNAL_CMATH(RT_ASIN,     asin,  DD_DD);
            EXTERNAL_CMATH(RT_ATANF,    atanf, FF_FF);
            EXTERNAL_CMATH(RT_ATAN,     atan,  DD_DD);
            EXTERNAL_CMATH(RT_ATAN2F,   atan2f, FF_FFFF);
            EXTERNAL_CMATH(RT_ATAN2,    atan2,  DD_DDDD);
            LLVM_INTRINSIC(RT_CEIL,     ceil);
            LLVM_INTRINSIC(RT_COS,      cos);
            LLVM_INTRINSIC(RT_EXP,      exp);
            LLVM_INTRINSIC(RT_EXP2,     exp2);
            LLVM_INTRINSIC(RT_FLOOR,    floor);
            EXTERNAL_CMATH(RT_FMODF,    fmodf, FF_FFFF);
            EXTERNAL_CMATH(RT_FMOD,     fmod,  DD_DDDD);
            LLVM_INTRINSIC(RT_LOG,      log);
            LLVM_INTRINSIC(RT_LOG2,     log2);
            LLVM_INTRINSIC(RT_LOG10,    log10);
            EXTERNAL_CMATH(RT_MODFF,    modff, FF_FFff);
            EXTERNAL_CMATH(RT_MODF,     modf,  DD_DDdd);
            LLVM_INTRINSIC(RT_POW,      pow);
        //#ifdef MI_ENABLE_MISTD // no powi in C++11
            EXTERNAL_CMATH2(RT_POWI,    pow,   mi::mdl::powi, DD_DDII);
        //#else
            //EXTERNAL_CMATH(RT_POWI,     pow,   DD_DDII);
        //#endif
            LLVM_INTRINSIC(RT_SIN,      sin);
            LLVM_INTRINSIC(RT_SQRT,     sqrt);
            EXTERNAL_CMATH(RT_TANF,     tanf, FF_FF);
            EXTERNAL_CMATH(RT_TAN,      tan,  DD_DD);
#ifdef LLVM_3_4
            LLVM_INTRINSIC(RT_COPYSIGN, copysign);
#else
#ifdef _MSC_VER
            // no copysign in MS runtime
            EXTERNAL_CMATH2(RT_COPYSIGNF, copysign,  ::_copysign,        DD_DDDD);
            EXTERNAL_CMATH2(RT_COPYSIGN,  copysignf, mi::mdl::copysignf, FF_FFFF);
#else
            EXTERNAL_CMATH(RT_COPYSIGNF, copysignf, FF_FFFF);
            EXTERNAL_CMATH(RT_COPYSIGN,  copysign,  DD_DDDD);
#endif
#endif
        default:
            MDL_ASSERT(!"unsupported <cmath> runtime function requested");
            break;
        }
    }
    return func;

#undef EXTERNAL_CMATH
}

// Return an LLVM type for a (single type) signature.
llvm::Type *MDL_runtime_creator::type_from_signature(
    char const * &signature,
    bool         &by_ref)
{
    by_ref = false;
    char c1 = signature[0];
    char c2 = signature[1];

    switch (c1) {
    case 'B':
        // boolean based
        switch (c2) {
        case 'B':
            signature += 2;
            return m_code_gen.m_type_mapper.get_bool_type();
        case '2':
        case '3':
        case '4':
            break;
        }
        break;
    case 'I':
        // integer based
        switch (c2) {
        case 'I':
            signature += 2;
            return m_code_gen.m_type_mapper.get_int_type();
        case '2':
        case '3':
        case '4':
            break;
        case 'A':
            {
                // Arrays
                char c3 = signature[2];
                switch (c3) {
                case '2':
                    by_ref = true;
                    signature += 3;
                    return m_code_gen.m_type_mapper.get_arr_int_2_type();
                case '3':
                    by_ref = true;
                    signature += 3;
                    return m_code_gen.m_type_mapper.get_arr_int_3_type();
                default:
                    break;
                }
            }
            break;
        }
        break;
    case 'F':
        // float based
        switch (c2) {
        case 'F':
            signature += 2;
            return m_code_gen.m_type_mapper.get_float_type();
        case '2':
        case '3':
        case '4':
            break;
        case 'A':
            {
                // Arrays
                char c3 = signature[2];
                switch (c3) {
                case '2':
                    by_ref = true;
                    signature += 3;
                    return m_code_gen.m_type_mapper.get_arr_float_2_type();
                case '3':
                    by_ref = true;
                    signature += 3;
                    return m_code_gen.m_type_mapper.get_arr_float_3_type();
                case '4':
                    by_ref = true;
                    signature += 3;
                    return m_code_gen.m_type_mapper.get_arr_float_4_type();
                default:
                    break;
                }
            }
            break;
        case 'D':
            {
                // Derivatives
                char c3 = signature[2];
                switch (c3) {
                case '2':
                    by_ref = true;
                    signature += 3;
                    return m_code_gen.m_type_mapper.get_deriv_arr_float_2_type();
                }
            }
            break;
        }
        break;
    case 'D':
        // double based
        switch (c2) {
        case 'D':
            signature += 2;
            return m_code_gen.m_type_mapper.get_double_type();
        case '2':
        case '3':
        case '4':
        case 'A':
            break;
        }
        break;
    case 'C':
        if (c2 == 'C') {
            signature += 2;
            return m_code_gen.m_type_mapper.get_color_type();
        } else if (c2 == 'S') {
            signature += 2;
            return m_code_gen.m_type_mapper.get_cstring_type();
        }
        break;
    case 'T':
        // texture
        switch (c2) {
        case '2':
        case '3':
        case 'C':
        case 'P':
            signature += 2;
            return m_code_gen.m_type_mapper.get_tag_type();
        }
        break;
    case 'd':
        if (c2 == 'd') {
            signature += 2;
            return Type_mapper::get_ptr(m_code_gen.m_type_mapper.get_double_type());
        }
        break;
    case 'f':
        if (c2 == 'f') {
            signature += 2;
            return Type_mapper::get_ptr(m_code_gen.m_type_mapper.get_float_type());
        }
        break;
    case 'E':
        // enum type
        signature += 3;
        return m_code_gen.m_type_mapper.get_int_type();
    case 'P':
        if (c2 == 'T') {
            signature += 2;
            return m_code_gen.m_type_mapper.get_res_data_pair_ptr_type();
        }
        break;
    case 'S':
        if (c2 == 'S') {
            signature += 2;
            return m_code_gen.m_type_mapper.get_string_type();
        }
        break;
    case 's':
        if (c2 == 'c') {
            signature += 2;
            return m_code_gen.m_type_mapper.get_state_ptr_type(Type_mapper::SSM_CORE);
        }
        break;
    case 'V':
        if (c2 == 'V') {
            signature += 2;
            return m_code_gen.m_type_mapper.get_void_type();
        }
        break;
    case 'v':
        if (c2 == 'v') {
            signature += 2;
            return m_code_gen.m_type_mapper.get_void_ptr_type();
        }
        break;
    case 'x':
        if (c2 == 's') {
            signature += 2;
            return m_code_gen.m_type_mapper.get_exc_state_ptr_type();
        }
        break;
    case 'Z':
        if (c2 == 'Z') {
            signature += 2;
            return m_code_gen.m_type_mapper.get_size_t_type();
        }
        break;
    case 'l':
        if (c2 == 'b') {
            signature += 2;
            // Linue_buffer is opaque anyway, so handle it like void *
            return m_code_gen.m_type_mapper.get_void_ptr_type();
        }
        break;
    default:
        break;
    }
    return NULL;
}

// Create a function declaration from a signature.
llvm::Function *MDL_runtime_creator::decl_from_signature(
    char const *name,
    char const *signature,
    bool       &is_sret)
{
    llvm::Type *ret_type = type_from_signature(signature, is_sret);
    llvm::SmallVector<llvm::Type *, 4> params;

    // skip Return/arg type separator
    if (signature[0] == '_')
        ++signature;

    if (is_sret) {
        params.push_back(m_code_gen.m_type_mapper.get_ptr(ret_type));
        ret_type = m_code_gen.m_type_mapper.get_void_type();
    }

    while (*signature != '\0') {
        bool       by_ref;
        llvm::Type *p_type = type_from_signature(signature, by_ref);
        if (p_type == NULL) {
            MDL_ASSERT(!"Unsupported signature");
            break;
        }

        if (by_ref) {
            params.push_back(Type_mapper::get_ptr(p_type));
        } else {
            params.push_back(p_type);
        }
    }
    llvm::FunctionType *func_type = llvm::FunctionType::get(ret_type, params, /*isVarArg=*/false);

    llvm::Function *func = llvm::Function::Create(
        func_type,
        llvm::GlobalValue::InternalLinkage,
        name,
        m_code_gen.m_module);

    return func;
}

// Load an runtime function arguments value.
llvm::Value *MDL_runtime_creator::load_by_value(Function_context &ctx, llvm::Value *arg)
{
    // use a simple heuristic here: if the type is a pointer to an array, it was passed
    // by reference
    llvm::Type *type = arg->getType();
    if (llvm::PointerType *p_tp = llvm::dyn_cast<llvm::PointerType>(type)) {
        llvm::Type *e_tp = p_tp->getPointerElementType();
        if (llvm::isa<llvm::ArrayType>(e_tp)) {
            // arg is a pointer to an array, assume passing by reference
            arg = ctx->CreateLoad(arg);
        } else if (m_code_gen.m_type_mapper.is_deriv_type(e_tp)) {
            // arg is a pointer to a derivative value, assume passing by reference
            arg = ctx->CreateLoad(arg);
        }
    }
    return arg;
}

// Get the start offset of the next entry with the given type in a valist.
// offset should point to after the last entry and will be advanced to after the new entry.
int MDL_runtime_creator::get_next_valist_entry_offset(int &offset, llvm::Type *operand_type)
{
    llvm::DataLayout  data_layout(m_code_gen.get_llvm_module());

    if ( offset != 0 )
      offset = data_layout.RoundUpAlignment(offset,
        data_layout.getPrefTypeAlignment(operand_type));

    int start_offset = offset;

    offset += int(data_layout.getTypeAllocSize(operand_type));

    return start_offset;
}

// Get a pointer to the next entry in the given valist buffer.
// offset should point to after the last entry and will be advanced to after the new entry.
llvm::Value *MDL_runtime_creator::get_next_valist_pointer(Function_context &ctx,
    llvm::Value *valist, int &offset, llvm::Type *operand_type)
{
    llvm::Value       *values[2] = { ctx.get_constant(0) };

    int start_offset = get_next_valist_entry_offset(offset, operand_type);
    values[1] = ctx.get_constant(start_offset);

    llvm::Value *gep_inst = ctx->CreateInBoundsGEP(valist, values);
    llvm::Value *cast_inst = ctx->CreateBitCast(gep_inst, operand_type->getPointerTo());

    return cast_inst;
}


// Call a runtime function.
llvm::Value *MDL_runtime_creator::call_rt_func(
    Function_context              &ctx,
    llvm::Function                *callee,
    llvm::ArrayRef<llvm::Value *> args)
{
    llvm::FunctionType *ftype    = callee->getFunctionType();
    llvm::Type         *ret_type = ftype->getReturnType();
    llvm::Value        *res_tmp  = NULL;
    unsigned           n_params  = ftype->getNumParams();
    unsigned           param_idx = 0;

    llvm::SmallVector<llvm::Value *, 8> n_args;

    if (ret_type->isVoidTy()) {
        MDL_ASSERT(n_params > 0);
        llvm::PointerType *p_type = llvm::cast<llvm::PointerType>(ftype->getParamType(0));

        res_tmp = ctx.create_local(p_type->getElementType(), "res_tmp");
        n_args.push_back(res_tmp);
        ++param_idx;
    }

    for (size_t idx = 0; param_idx < n_params; ++param_idx, ++idx) {
        llvm::Type  *param_type = ftype->getParamType(param_idx);
        llvm::Value *arg        = args[idx];

        if (llvm::PointerType *p_type = llvm::dyn_cast<llvm::PointerType>(param_type)) {
            llvm::Type *e_type = p_type->getElementType();
            if (e_type == arg->getType()) {
                // pass by reference
                llvm::Value *tmp = ctx.create_local(e_type, "refarg");
                ctx->CreateStore(arg, tmp);
                arg = tmp;
            }
        }
        n_args.push_back(arg);
    }

    llvm::Value *res = ctx->CreateCall(callee, n_args);

    if (ret_type->isVoidTy()) {
        res = ctx->CreateLoad(res_tmp);
    }
    return res;
}

// Call a void runtime function.
void MDL_runtime_creator::call_rt_func_void(
    Function_context              &ctx,
    llvm::Function                *callee,
    llvm::ArrayRef<llvm::Value *> args)
{
    llvm::FunctionType *ftype    = callee->getFunctionType();
    unsigned           n_params  = ftype->getNumParams();
    unsigned           param_idx = 0;

    llvm::SmallVector<llvm::Value *, 8> n_args;

    for (size_t idx = 0; param_idx < n_params; ++param_idx, ++idx) {
        llvm::Type  *param_type = ftype->getParamType(param_idx);
        llvm::Value *arg        = args[idx];

        if (llvm::PointerType *p_type = llvm::dyn_cast<llvm::PointerType>(param_type)) {
            llvm::Type *e_type = p_type->getElementType();
            if (e_type == arg->getType()) {
                // pass by reference
                llvm::Value *tmp = ctx.create_local(e_type, "refarg");
                ctx->CreateStore(arg, tmp);
                arg = tmp;
            }
        }
        n_args.push_back(arg);
    }

    ctx->CreateCall(callee, n_args);
}

// Check if a given MDL runtime
llvm::Function *MDL_runtime_creator::find_in_c_runtime(
    Runtime_function code,
    char const       *signature)
{
    switch (code) {
#if !defined(_MSC_VER) || _MSC_VER >= 1900
    // on WIN we need we have at least VS 2015 for these functions
    case RT_MDL_EXP2F:
        return get_c_runtime_func(RT_EXP2F, signature);
    case RT_MDL_EXP2:
        return get_c_runtime_func(RT_EXP2, signature);
    case RT_MDL_LOG2F:
        return get_c_runtime_func(RT_LOG2F, signature);
    case RT_MDL_LOG2:
        return get_c_runtime_func(RT_LOG2, signature);
#endif
    case RT_MDL_MINI:
        if (m_ptx_mode)
            return get_c_runtime_func(RT_MINI, signature);
        return NULL;
    case RT_MDL_MAXI:
        if (m_ptx_mode)
            return get_c_runtime_func(RT_MAXI, signature);
        return NULL;
    case RT_MDL_MINF:
        if (m_ptx_mode)
            return get_c_runtime_func(RT_MINF, signature);
        return NULL;
    case RT_MDL_MAXF:
        if (m_ptx_mode)
            return get_c_runtime_func(RT_MAXF, signature);
        return NULL;
    case RT_MDL_MIN:
        if (m_ptx_mode)
            return get_c_runtime_func(RT_MIN, signature);
        return NULL;
    case RT_MDL_MAX:
        if (m_ptx_mode)
            return get_c_runtime_func(RT_MAX, signature);
        return NULL;
    case RT_MDL_RSQRTF:
        if (m_ptx_mode)
            return get_c_runtime_func(RT_RSQRTF, signature);
        return NULL;
    case RT_MDL_RSQRT:
        if (m_ptx_mode)
            return get_c_runtime_func(RT_RSQRT, signature);
        return NULL;
    default:
        return NULL;
    }
}

// Create a runtime function.
llvm::Function *MDL_runtime_creator::create_runtime_func(
    Runtime_function code,
    char const *name,
    char const *signature)
{
    // check if this is available as a native runtime function
    if (llvm::Function *c_func = find_in_c_runtime(code, signature))
        return c_func;

    bool           is_sret = false;
    llvm::Function *func   = decl_from_signature(name, signature, is_sret);

    // Mark function as a native function already registered with the Jitted_code,
    // if we are not in PTX mode
    #define MARK_NATIVE(func) do {                                    \
            if (!m_ptx_mode) {                                        \
                func->setCallingConv(llvm::CallingConv::C);           \
                func->setLinkage(llvm::GlobalValue::ExternalLinkage); \
            }                                                         \
        } while (0)

    // check for glue functions first
    switch (code) {
    case RT_MDL_TEX_RESOLUTION_2D:
        func->setDoesNotThrow();
        func->setDoesNotCapture(1); // result
        func->setDoesNotCapture(2); // resource_data
        func->setDoesNotCapture(4); // uv_tile
        MARK_NATIVE(func);  // tex_resolution_2d
        return func;
    case RT_MDL_TEX_WIDTH:
        func->setDoesNotThrow();
        func->setOnlyReadsMemory();
        func->setDoesNotCapture(1); // resource_data
        MARK_NATIVE(func);  // tex_width
        return func;
    case RT_MDL_TEX_HEIGHT:
        func->setDoesNotThrow();
        func->setOnlyReadsMemory();
        func->setDoesNotCapture(1); // resource_data
        MARK_NATIVE(func);  // tex_height
        return func;
    case RT_MDL_TEX_DEPTH:
        func->setDoesNotThrow();
        func->setOnlyReadsMemory();
        func->setDoesNotCapture(1); // resource_data
        MARK_NATIVE(func);  // tex_depth
        return func;
    case RT_MDL_TEX_VALID:
        func->setDoesNotThrow();
        func->setOnlyReadsMemory();
        func->setDoesNotCapture(1); // resource_data
        MARK_NATIVE(func);  // tex_isvalid
        return func;
    case RT_MDL_TEX_LOOKUP_FLOAT_2D:
        func->setDoesNotThrow();
        func->setOnlyReadsMemory();
        func->setDoesNotCapture(1); // resource_data
        func->setDoesNotCapture(3); // coord
        func->setDoesNotCapture(6); // crop_u
        func->setDoesNotCapture(7); // crop_v
        MARK_NATIVE(func);  // tex_lookup_float_2d
        return func;
    case RT_MDL_TEX_LOOKUP_DERIV_FLOAT_2D:
        func->setDoesNotThrow();
        func->setDoesNotCapture(1); // resource_data
        func->setDoesNotCapture(3); // coord
        func->setDoesNotCapture(6); // crop_u
        func->setDoesNotCapture(7); // crop_v
        MARK_NATIVE(func);  // tex_lookup_deriv_float_2d
        return func;
    case RT_MDL_TEX_LOOKUP_FLOAT_3D:
        func->setDoesNotThrow();
        func->setOnlyReadsMemory();
        func->setDoesNotCapture(1); // resource_data
        func->setDoesNotCapture(3); // coord
        func->setDoesNotCapture(7); // crop_u
        func->setDoesNotCapture(8); // crop_v
        func->setDoesNotCapture(9); // crop_w
        MARK_NATIVE(func);  // tex_lookup_float_3d
        return func;
    case RT_MDL_TEX_LOOKUP_FLOAT_CUBE:
        func->setDoesNotThrow();
        func->setOnlyReadsMemory();
        func->setDoesNotCapture(1); // resource_data
        func->setDoesNotCapture(3); // coord
        MARK_NATIVE(func);  // tex_lookup_float_cube
        return func;
    case RT_MDL_TEX_LOOKUP_FLOAT_PTEX:
        func->setDoesNotThrow();
        func->setOnlyReadsMemory();
        func->setDoesNotCapture(1); // resource_data
        MARK_NATIVE(func);  // tex_lookup_float_ptex
        return func;

    case RT_MDL_TEX_LOOKUP_FLOAT2_2D:
        func->setDoesNotThrow();
        func->setDoesNotCapture(1); // result
        func->setDoesNotCapture(2); // resource_data
        func->setDoesNotCapture(4); // coord
        func->setDoesNotCapture(7); // crop_u
        func->setDoesNotCapture(8); // crop_v
        MARK_NATIVE(func);  // tex_lookup_float2_2d
        return func;
    case RT_MDL_TEX_LOOKUP_DERIV_FLOAT2_2D:
        func->setDoesNotThrow();
        func->setDoesNotCapture(1); // result
        func->setDoesNotCapture(2); // resource_data
        func->setDoesNotCapture(4); // coord
        func->setDoesNotCapture(7); // crop_u
        func->setDoesNotCapture(8); // crop_v
        MARK_NATIVE(func);  // tex_lookup_deriv_float2_2d
        return func;
    case RT_MDL_TEX_LOOKUP_FLOAT2_3D:
        func->setDoesNotThrow();
        func->setDoesNotCapture(1); // result
        func->setDoesNotCapture(2); // resource_data
        func->setDoesNotCapture(4); // coord
        func->setDoesNotCapture(8); // crop_u
        func->setDoesNotCapture(9); // crop_v
        func->setDoesNotCapture(10); // crop_w
        MARK_NATIVE(func);  // tex_lookup_float2_3d
        return func;
    case RT_MDL_TEX_LOOKUP_FLOAT2_CUBE:
        func->setDoesNotThrow();
        func->setDoesNotCapture(1); // result
        func->setDoesNotCapture(2); // resource_data
        func->setDoesNotCapture(4); // coord
        MARK_NATIVE(func);  // tex_lookup_float2_cube
        return func;
    case RT_MDL_TEX_LOOKUP_FLOAT2_PTEX:
        func->setDoesNotThrow();
        func->setDoesNotCapture(1); // result
        func->setDoesNotCapture(2); // resource_data
        MARK_NATIVE(func);  // tex_lookup_float2_ptex
        return func;

    case RT_MDL_TEX_LOOKUP_FLOAT3_2D:
        func->setDoesNotThrow();
        func->setDoesNotCapture(1); // result
        func->setDoesNotCapture(2); // resource_data
        func->setDoesNotCapture(4); // coord
        func->setDoesNotCapture(7); // crop_u
        func->setDoesNotCapture(8); // crop_v
        MARK_NATIVE(func);  // tex_lookup_float3_2d
        return func;
    case RT_MDL_TEX_LOOKUP_DERIV_FLOAT3_2D:
        func->setDoesNotThrow();
        func->setDoesNotCapture(1); // result
        func->setDoesNotCapture(2); // resource_data
        func->setDoesNotCapture(4); // coord
        func->setDoesNotCapture(7); // crop_u
        func->setDoesNotCapture(8); // crop_v
        MARK_NATIVE(func);  // tex_lookup_deriv_float3_2d
        return func;
    case RT_MDL_TEX_LOOKUP_FLOAT3_3D:
        func->setDoesNotThrow();
        func->setDoesNotCapture(1); // result
        func->setDoesNotCapture(2); // resource_data
        func->setDoesNotCapture(4); // coord
        func->setDoesNotCapture(8); // crop_u
        func->setDoesNotCapture(9); // crop_v
        func->setDoesNotCapture(10); // crop_w
        MARK_NATIVE(func);  // tex_lookup_float3_3d
        return func;
    case RT_MDL_TEX_LOOKUP_FLOAT3_CUBE:
        func->setDoesNotThrow();
        func->setDoesNotCapture(1); // result
        func->setDoesNotCapture(2); // resource_data
        func->setDoesNotCapture(4); // coord
        MARK_NATIVE(func);  // tex_lookup_float3_cube
        return func;
    case RT_MDL_TEX_LOOKUP_FLOAT3_PTEX:
        func->setDoesNotThrow();
        func->setDoesNotCapture(1); // result
        func->setDoesNotCapture(2); // resource_data
        MARK_NATIVE(func);  // tex_lookup_float3_ptex
        return func;

    case RT_MDL_TEX_LOOKUP_FLOAT4_2D:
        func->setDoesNotThrow();
        func->setDoesNotCapture(1); // result
        func->setDoesNotCapture(2); // resource_data
        func->setDoesNotCapture(4); // coord
        func->setDoesNotCapture(7); // crop_u
        func->setDoesNotCapture(8); // crop_v
        MARK_NATIVE(func);  // tex_lookup_float4_2d
        return func;
    case RT_MDL_TEX_LOOKUP_DERIV_FLOAT4_2D:
        func->setDoesNotThrow();
        func->setDoesNotCapture(1); // result
        func->setDoesNotCapture(2); // resource_data
        func->setDoesNotCapture(4); // coord
        func->setDoesNotCapture(7); // crop_u
        func->setDoesNotCapture(8); // crop_v
        MARK_NATIVE(func);  // tex_lookup_deriv_float4_2d
        return func;
    case RT_MDL_TEX_LOOKUP_FLOAT4_3D:
        func->setDoesNotThrow();
        func->setDoesNotCapture(1); // result
        func->setDoesNotCapture(2); // resource_data
        func->setDoesNotCapture(4); // coord
        func->setDoesNotCapture(8); // crop_u
        func->setDoesNotCapture(9); // crop_v
        func->setDoesNotCapture(10); // crop_w
        MARK_NATIVE(func);  // tex_lookup_float4_3d
        return func;
    case RT_MDL_TEX_LOOKUP_FLOAT4_CUBE:
        func->setDoesNotThrow();
        func->setDoesNotCapture(1); // result
        func->setDoesNotCapture(2); // resource_data
        func->setDoesNotCapture(4); // coord
        MARK_NATIVE(func);  // tex_lookup_float4_cube
        return func;
    case RT_MDL_TEX_LOOKUP_FLOAT4_PTEX:
        func->setDoesNotThrow();
        func->setDoesNotCapture(1); // result
        func->setDoesNotCapture(2); // resource_data
        MARK_NATIVE(func);  // tex_lookup_float4_ptex
        return func;

    case RT_MDL_TEX_LOOKUP_COLOR_2D:
        func->setDoesNotThrow();
        func->setDoesNotCapture(1); // result
        func->setDoesNotCapture(2); // resource_data
        func->setDoesNotCapture(4); // coord
        func->setDoesNotCapture(7); // crop_u
        func->setDoesNotCapture(8); // crop_v
        MARK_NATIVE(func);  // tex_lookup_color_2d
        return func;
    case RT_MDL_TEX_LOOKUP_DERIV_COLOR_2D:
        func->setDoesNotThrow();
        func->setDoesNotCapture(1); // result
        func->setDoesNotCapture(2); // resource_data
        func->setDoesNotCapture(4); // coord
        func->setDoesNotCapture(7); // crop_u
        func->setDoesNotCapture(8); // crop_v
        MARK_NATIVE(func);  // tex_lookup_deriv_color_2d
        return func;
    case RT_MDL_TEX_LOOKUP_COLOR_3D:
        func->setDoesNotThrow();
        func->setDoesNotCapture(1); // result
        func->setDoesNotCapture(2); // resource_data
        func->setDoesNotCapture(4); // coord
        func->setDoesNotCapture(8); // crop_u
        func->setDoesNotCapture(9); // crop_v
        func->setDoesNotCapture(10); // crop_w
        MARK_NATIVE(func);  // tex_lookup_color_3d
        return func;
    case RT_MDL_TEX_LOOKUP_COLOR_CUBE:
        func->setDoesNotThrow();
        func->setDoesNotCapture(1); // result
        func->setDoesNotCapture(2); // resource_data
        func->setDoesNotCapture(4); // coord
        MARK_NATIVE(func);  // tex_lookup_color_cube
        return func;
    case RT_MDL_TEX_LOOKUP_COLOR_PTEX:
        func->setDoesNotThrow();
        func->setDoesNotCapture(1); // result
        func->setDoesNotCapture(2); // resource_data
        MARK_NATIVE(func);  // tex_lookup_color_ptex
        return func;

    case RT_MDL_TEX_TEXEL_FLOAT_2D:
        func->setDoesNotThrow();
        func->setOnlyReadsMemory();
        func->setDoesNotCapture(1); // resource_data
        func->setDoesNotCapture(3); // coord
        func->setDoesNotCapture(4); // uv_tile
        MARK_NATIVE(func);  // tex_texel_float_2d
        return func;
    case RT_MDL_TEX_TEXEL_FLOAT2_2D:
        func->setDoesNotThrow();
        func->setDoesNotCapture(1); // result
        func->setDoesNotCapture(2); // resource_data
        func->setDoesNotCapture(4); // coord
        func->setDoesNotCapture(5); // uv_tile
        MARK_NATIVE(func);  // tex_texel_float2_2d
        return func;
    case RT_MDL_TEX_TEXEL_FLOAT3_2D:
        func->setDoesNotThrow();
        func->setDoesNotCapture(1); // result
        func->setDoesNotCapture(2); // resource_data
        func->setDoesNotCapture(4); // coord
        func->setDoesNotCapture(5); // uv_tile
        MARK_NATIVE(func);  // tex_texel_float3_2d
        return func;
    case RT_MDL_TEX_TEXEL_FLOAT4_2D:
        func->setDoesNotThrow();
        func->setDoesNotCapture(1); // result
        func->setDoesNotCapture(2); // resource_data
        func->setDoesNotCapture(4); // coord
        func->setDoesNotCapture(5); // uv_tile
        MARK_NATIVE(func);  // tex_texel_float4_2d
        return func;
    case RT_MDL_TEX_TEXEL_COLOR_2D:
        func->setDoesNotThrow();
        func->setDoesNotCapture(1); // result
        func->setDoesNotCapture(2); // resource_data
        func->setDoesNotCapture(4); // coord
        func->setDoesNotCapture(5); // uv_tile
        MARK_NATIVE(func);  // tex_texel_color_2d
        return func;

    case RT_MDL_TEX_TEXEL_FLOAT_3D:
        func->setDoesNotThrow();
        func->setOnlyReadsMemory();
        func->setDoesNotCapture(1); // resource_data
        func->setDoesNotCapture(3); // coord
        MARK_NATIVE(func);  // tex_texel_float_3d
        return func;
    case RT_MDL_TEX_TEXEL_FLOAT2_3D:
        func->setDoesNotThrow();
        func->setDoesNotCapture(1); // result
        func->setDoesNotCapture(2); // resource_data
        func->setDoesNotCapture(4); // coord
        MARK_NATIVE(func);  // tex_texel_float2_3d
        return func;
    case RT_MDL_TEX_TEXEL_FLOAT3_3D:
        func->setDoesNotThrow();
        func->setDoesNotCapture(1); // result
        func->setDoesNotCapture(2); // resource_data
        func->setDoesNotCapture(4); // coord
        MARK_NATIVE(func);  // tex_texel_float3_3d
        return func;
    case RT_MDL_TEX_TEXEL_FLOAT4_3D:
        func->setDoesNotThrow();
        func->setDoesNotCapture(1); // result
        func->setDoesNotCapture(2); // resource_data
        func->setDoesNotCapture(4); // coord
        MARK_NATIVE(func);  // tex_texel_float4_3d
        return func;
    case RT_MDL_TEX_TEXEL_COLOR_3D:
        func->setDoesNotThrow();
        func->setDoesNotCapture(1); // result
        func->setDoesNotCapture(2); // resource_data
        func->setDoesNotCapture(4); // coord
        MARK_NATIVE(func);  // tex_texel_color_3d
        return func;

    case RT_MDL_DF_LIGHT_PROFILE_POWER:
        func->setDoesNotThrow();
        func->setOnlyReadsMemory();
        func->setDoesNotCapture(1); // resource_data
        MARK_NATIVE(func);  // df_light_profile_power
        return func;
    case RT_MDL_DF_LIGHT_PROFILE_MAXIMUM:
        func->setDoesNotThrow();
        func->setOnlyReadsMemory();
        func->setDoesNotCapture(1); // resource_data
        MARK_NATIVE(func);  // df_light_profile_maximum
        return func;
    case RT_MDL_DF_LIGHT_PROFILE_VALID:
        func->setDoesNotThrow();
        func->setOnlyReadsMemory();
        func->setDoesNotCapture(1); // resource_data
        MARK_NATIVE(func);  // df_light_profile_isvalid
        return func;
    case RT_MDL_DF_BSDF_MEASUREMENT_VALID:
        func->setDoesNotThrow();
        func->setOnlyReadsMemory();
        func->setDoesNotCapture(1); // resource_data
        MARK_NATIVE(func);  // df_bsdf_measurement_isvalid
        return func;

    case RT_MDL_DF_BSDF_MEASUREMENT_RESOLUTION:
        func->setDoesNotThrow();
        func->setDoesNotCapture(1); // result
        func->setDoesNotCapture(2); // resource_data
        MARK_NATIVE(func);  // df_bsdf_measurement_resolution
        return func;

    case RT_MDL_DF_BSDF_MEASUREMENT_EVALUATE:
        func->setDoesNotThrow();
        func->setDoesNotCapture(1); // result
        func->setDoesNotCapture(2); // resource_data
        func->setDoesNotCapture(4); // theta_phi_in
        func->setDoesNotCapture(5); // theta_phi_out
        MARK_NATIVE(func);  // df_bsdf_measurement_evaluate
        return func;

    case RT_MDL_DF_BSDF_MEASUREMENT_SAMPLE:
        func->setDoesNotThrow();
        func->setDoesNotCapture(1); // result theta phi pdf
        func->setDoesNotCapture(2); // resource_data
        func->setDoesNotCapture(4); // theta_phi_out
        func->setDoesNotCapture(5); // xi
        MARK_NATIVE(func);  // df_bsdf_measurement_sample
        return func;

    case RT_MDL_DF_BSDF_MEASUREMENT_PDF:
        func->setDoesNotThrow();
        func->setDoesNotCapture(1); // resource_data
        func->setDoesNotCapture(3); // theta_phi_in
        func->setDoesNotCapture(4); // theta_phi_out
        MARK_NATIVE(func);  // df_bsdf_measurement_pdf
        return func;

    case RT_MDL_DF_BSDF_MEASUREMENT_ALBEDOS:
        func->setDoesNotThrow();
        func->setDoesNotCapture(1); // result
        func->setDoesNotCapture(2); // resource_data
        func->setDoesNotCapture(4); // theta_phi
        MARK_NATIVE(func);  // df_bsdf_measurement_albedos
        return func;

    case RT_MDL_DF_LIGHT_PROFILE_EVALUATE:
        func->setDoesNotThrow();
        func->setDoesNotCapture(1); // resource_data
        func->setDoesNotCapture(3); // theta_phi
        MARK_NATIVE(func);  // df_light_profile_evaluate
        return func;

    case RT_MDL_DF_LIGHT_PROFILE_SAMPLE:
        func->setDoesNotThrow();
        func->setDoesNotCapture(1); // result theta phi pdf
        func->setDoesNotCapture(2); // resource_data
        func->setDoesNotCapture(4); // xi
        MARK_NATIVE(func);  // df_light_profile_sample
        return func;

    case RT_MDL_DF_LIGHT_PROFILE_PDF:
        func->setDoesNotThrow();
        func->setDoesNotCapture(1); // resource_data
        func->setDoesNotCapture(3); // theta_phi
        MARK_NATIVE(func);  // df_light_profile_pdf
        return func;

    case RT_MDL_BLACKBODY:
        func->setDoesNotThrow();
        func->setDoesNotCapture(1); // sRGB
        MARK_NATIVE(func);  // mi::mdl::spectral::mdl_blackbody
        return func;

    case RT_MDL_DEBUGBREAK:
        func->setDoesNotThrow();
        MARK_NATIVE(func);  // debug::debugbreak
        return func;
    case RT_MDL_ASSERTFAIL:
        func->setDoesNotThrow();
        func->setDoesNotCapture(1); // reason
        func->setDoesNotCapture(2); // func_name
        func->setDoesNotCapture(3); // file_name
        MARK_NATIVE(func);  // debug::assertfail
        return func;
    case RT___ASSERTFAIL:
        func->setDoesNotThrow();
        func->setDoesNotCapture(1); // message
        func->setDoesNotCapture(2); // file
        func->setDoesNotCapture(4); // function

        // CUDA/OptiX runtime function
        func->setLinkage(llvm::GlobalValue::ExternalLinkage);
        return func;
    case RT_MDL_PRINT_BEGIN:
        func->setDoesNotThrow();
        MARK_NATIVE(func);  // debug::print_begin
        return func;
    case RT_MDL_PRINT_END:
        func->setDoesNotThrow();
        func->setDoesNotCapture(1); // buffer
        MARK_NATIVE(func);  // debug::print_end
        return func;
    case RT_MDL_PRINT_BOOL:
        func->setDoesNotThrow();
        func->setDoesNotCapture(1); // buffer
        MARK_NATIVE(func);  // debug::print_bool
        return func;
    case RT_MDL_PRINT_INT:
        func->setDoesNotThrow();
        func->setDoesNotCapture(1); // buffer
        MARK_NATIVE(func);  // debug::print_int
        return func;
    case RT_MDL_PRINT_FLOAT:
        func->setDoesNotThrow();
        func->setDoesNotCapture(1); // buffer
        MARK_NATIVE(func);  // debug::print_float
        return func;
    case RT_MDL_PRINT_DOUBLE:
        func->setDoesNotThrow();
        func->setDoesNotCapture(1); // buffer
        MARK_NATIVE(func);  // debug::print_double
        return func;
    case RT_MDL_PRINT_STRING:
        func->setDoesNotThrow();
        func->setDoesNotCapture(1); // buffer
        func->setDoesNotCapture(2); // value
        MARK_NATIVE(func);  // debug::print_string
        return func;
    case RT_VPRINTF:
        func->setDoesNotThrow();
        func->setDoesNotCapture(1); // format
        func->setDoesNotCapture(2); // valist

        // CUDA/OptiX runtime function
        func->setLinkage(llvm::GlobalValue::ExternalLinkage);
        return func;

    case RT_MDL_TO_CSTRING:
        func->setDoesNotThrow();
        if (!m_code_gen.m_type_mapper.strings_mapped_to_ids())
            func->setDoesNotCapture(1); // string_or_id
        break;

    case RT_MDL_OUT_OF_BOUNDS:
        func->setDoesNotThrow();
        func->setDoesNotCapture(1); // exc_state
        func->setDoesNotCapture(4); // fname
        func->setDoesNotReturn();
        MARK_NATIVE(func);  // LLVM_code_generator::mdl_out_of_bounds
        return func;

    case RT_MDL_DIV_BY_ZERO:
        func->setDoesNotThrow();
        func->setDoesNotCapture(1); // exc_state
        func->setDoesNotCapture(2); // fname
        func->setDoesNotReturn();
        MARK_NATIVE(func);  // LLVM_code_generator::mdl_div_by_zero
        return func;

    default:
        break;
    }

    LLVM_context_data::Flags flags =
        is_sret ? LLVM_context_data::FL_SRET : LLVM_context_data::FL_NONE;
    switch (code) {
    case RT_MDL_TEX_RES_FLOAT:
    case RT_MDL_TEX_RES_FLOAT3:
    case RT_MDL_TEX_RES_COLOR:
        // need a state
        flags |= LLVM_context_data::FL_HAS_STATE;
        break;
    default:
        break;
    }

    Function_instance inst(
        m_alloc, /*func_def=*/(mi::mdl::IDefinition const *)NULL, /*return_derivs=*/ false);
    Function_context ctx(m_alloc, m_code_gen, inst, func, flags);

    llvm::Function::arg_iterator arg_it = ctx.get_first_parameter();

    switch (code) {
    case RT_MDL_CLAMP:
        // clamp(x, a, b) = min(max(x, a), b)
        {
            llvm::Value *x = arg_it++;
            llvm::Value *a = arg_it++;
            llvm::Value *b = arg_it;

            llvm::Function *max_func = get_runtime_func(RT_MDL_MAX);
            llvm::Function *min_func = get_runtime_func(RT_MDL_MIN);
            llvm::Value *res = ctx->CreateCall2(max_func, x, a);
            ctx.create_return(ctx->CreateCall2(min_func, res, b));
        }
        func->addFnAttr(llvm::Attribute::AlwaysInline);
        break;
    case RT_MDL_CLAMPF:
        // clamp(x, a, b) = min(max(x, a), b)
        {
            llvm::Value *x = arg_it++;
            llvm::Value *a = arg_it++;
            llvm::Value *b = arg_it;

            llvm::Function *max_func = get_runtime_func(RT_MDL_MAXF);
            llvm::Function *min_func = get_runtime_func(RT_MDL_MINF);
            llvm::Value *res = ctx->CreateCall2(max_func, x, a);
            ctx.create_return(ctx->CreateCall2(min_func, res, b));
        }
        func->addFnAttr(llvm::Attribute::AlwaysInline);
        break;
    case RT_MDL_CLAMPI:
        // clamp(x, a, b) = min(max(x, a), b)
        {
            llvm::Value *x = arg_it++;
            llvm::Value *a = arg_it++;
            llvm::Value *b = arg_it;

            llvm::Function *max_func = get_runtime_func(RT_MDL_MAXI);
            llvm::Function *min_func = get_runtime_func(RT_MDL_MINI);
            llvm::Value *res = ctx->CreateCall2(max_func, x, a);
            ctx.create_return(ctx->CreateCall2(min_func, res, b));
        }
        func->addFnAttr(llvm::Attribute::AlwaysInline);
        break;

    case RT_MDL_EXP2:
    case RT_MDL_EXP2F:
        // exp2(x) = exp(x * 0.69314718055994530941723212145818 /* log(2) */)
        {
            llvm::Value    *x       = arg_it;
            bool           is_float = code == RT_MDL_EXP2F;
            double         log_2    = 0.69314718055994530941723212145818;
            llvm::Value    *c;
            llvm::Function *exp_func;

            if (is_float) {
                c = ctx.get_constant(float(log_2));
                exp_func = get_runtime_func(RT_EXPF);
            } else {
                c = ctx.get_constant(double(log_2));
                exp_func = get_runtime_func(RT_EXP);
            }
            ctx.create_return(ctx->CreateCall(exp_func, ctx->CreateFMul(x, c)));
        }
        func->addFnAttr(llvm::Attribute::AlwaysInline);
        break;

    case RT_MDL_FRAC:
    case RT_MDL_FRACF:
        // frac(x) = modf(x, _) + (x < 0.0f ? 1.0 : 0.0)
        {
            llvm::Value    *x       = arg_it;
            bool           is_float = code == RT_MDL_FRACF;
            llvm::Value    *z, *o;
            llvm::Function *modf_func;

            if (is_float) {
                z = ctx.get_constant(0.0f);
                o = ctx.get_constant(1.0f);
                modf_func = get_runtime_func(RT_MODFF);
            } else {
                z = ctx.get_constant(0.0);
                o = ctx.get_constant(1.0);
                modf_func = get_runtime_func(RT_MODF);
            }
            llvm::Value *cmp = ctx->CreateFCmpOLT(x, z);
            llvm::Value *c   = ctx->CreateSelect(cmp, o, z);
            llvm::Value *tmp = ctx.create_local(x->getType(), "tmp");

            llvm::Value *res = ctx->CreateFAdd(ctx->CreateCall2(modf_func, x, tmp), c);
            ctx.create_return(res);
        }
        func->addFnAttr(llvm::Attribute::AlwaysInline);
        break;
    case RT_MDL_LOG2:
    case RT_MDL_LOG2F:
        //  log2(x) = log(x) * 1/log(2)
        {
            llvm::Value    *x       = arg_it;
            bool           is_float = code == RT_MDL_LOG2F;
            double         r_log_2  = 1.4426950408889634073599246810019;
            llvm::Value    *c;
            llvm::Function *exp_func;

            if (is_float) {
                c = ctx.get_constant(float(r_log_2));
                exp_func = get_runtime_func(RT_LOGF);
            } else {
                c = ctx.get_constant(double(r_log_2));
                exp_func = get_runtime_func(RT_LOG);
            }
            ctx.create_return(ctx->CreateCall(exp_func, ctx->CreateFMul(x, c)));
        }
        func->addFnAttr(llvm::Attribute::AlwaysInline);
        break;
    case RT_MDL_POWI:
        // pow(int a, int b) = int(powi(double(a), b))
        {
            llvm::Value    *a         = arg_it++;
            llvm::Value    *b         = arg_it;
            llvm::Function *powi_func = get_runtime_func(RT_POWI);

            llvm::Value *x = ctx->CreateSIToFP(a, m_code_gen.m_type_mapper.get_double_type());
            x = ctx->CreateCall2(powi_func, x, b);
            ctx.create_return(ctx->CreateFPToSI(x, m_code_gen.m_type_mapper.get_int_type()));
        }
        func->addFnAttr(llvm::Attribute::AlwaysInline);
        break;
    case RT_MDL_ROUND:
    case RT_MDL_ROUNDF:
        // round(x) = floor(x + 0.5)
        {
            llvm::Value    *x       = arg_it;
            bool           is_float = code == RT_MDL_ROUNDF;
            llvm::Value    *c;
            llvm::Function *floor_func;

            if (is_float) {
                c = ctx.get_constant(0.5f);
                floor_func = get_runtime_func(RT_FLOORF);
            } else {
                c = ctx.get_constant(0.5);
                floor_func = get_runtime_func(RT_FLOOR);
            }
            ctx.create_return(ctx->CreateCall(floor_func, ctx->CreateFAdd(x, c)));
        }
        func->addFnAttr(llvm::Attribute::AlwaysInline);
        break;
    case RT_MDL_RSQRT:
    case RT_MDL_RSQRTF:
        //  rsqrt(x) = 1.0 / sqrt(x)
        {
            llvm::Value    *x       = arg_it;
            bool           is_float = code == RT_MDL_RSQRTF;
            llvm::Value    *one;
            llvm::Function *exp_func;

            if (is_float) {
                one = ctx.get_constant(1.0f);
                exp_func = get_runtime_func(RT_SQRTF);
            } else {
                one = ctx.get_constant(1.0);
                exp_func = get_runtime_func(RT_SQRT);
            }
            ctx.create_return(ctx->CreateCall(exp_func, ctx->CreateFDiv(one, x)));
        }
        func->addFnAttr(llvm::Attribute::AlwaysInline);
        break;
    case RT_MDL_SATURATE:
    case RT_MDL_SATURATEF:
        // saturate(x) = clamp(x, 0.0, 1.0)
        {
            llvm::Value    *x = arg_it++;
            bool           is_float = code == RT_MDL_SATURATEF;
            llvm::Value    *z, *o;
            llvm::Function *clamp_func;

            if (is_float) {
                z = ctx.get_constant(0.0f);
                o = ctx.get_constant(1.0f);
                clamp_func = get_runtime_func(RT_MDL_CLAMPF);
            } else {
                z = ctx.get_constant(0.0);
                o = ctx.get_constant(1.0);
                clamp_func = get_runtime_func(RT_MDL_CLAMP);
            }
            ctx.create_return(ctx->CreateCall3(clamp_func, x, z, o));
        }
        func->addFnAttr(llvm::Attribute::AlwaysInline);
        break;
    case RT_MDL_SIGN:
    case RT_MDL_SIGNF:
        // fp sign
        {
            bool           is_float = code == RT_MDL_SIGNF;
            llvm::Value    *x       = arg_it;
            llvm::Value    *zero, *one;
            llvm::Function *copysign_func;

            if (is_float) {
                zero = ctx.get_constant(0.0f);
                one  = ctx.get_constant(1.0f);
                copysign_func = get_runtime_func(RT_COPYSIGNF);
            } else {
                zero = ctx.get_constant(0.0);
                one  = ctx.get_constant(1.0);
                copysign_func = get_runtime_func(RT_COPYSIGN);
            }
            llvm::Value *cmp    = ctx->CreateFCmpOEQ(x, zero);
            llvm::Value *args[] = { one, x };
            llvm::Value *cpsgn  = ctx->CreateCall(copysign_func, args);
            llvm::Value *res    = ctx->CreateSelect(cmp, zero, cpsgn);
            ctx.create_return(res);
        }
        func->addFnAttr(llvm::Attribute::AlwaysInline);
        break;
    case RT_MDL_SIGNI:
        // integer sign
        {
            llvm::Value *x     = arg_it;
            llvm::Value *zero  = ctx.get_constant(int(0));
            llvm::Value *cmp   = ctx->CreateICmpEQ(x, zero);
            llvm::Value *cpsgn = ctx->CreateAShr(x, ctx.get_constant(31));
            llvm::Value *res   = ctx->CreateSelect(cmp, zero, cpsgn);
            ctx.create_return(res);
        }
        func->addFnAttr(llvm::Attribute::AlwaysInline);
        break;
    case RT_MDL_SMOOTHSTEP:
    case RT_MDL_SMOOTHSTEPF:
        // smoothstep(a, b, l) ==>
        //   x = clamp(l, a, b)
        //   x = (x-a)/(b-a)
        //   return x*x * (3.0 - (x+x))
        {
            bool           is_float = code == RT_MDL_SMOOTHSTEPF;
            llvm::Value    *a       = arg_it++;
            llvm::Value    *b       = arg_it++;
            llvm::Value    *l       = arg_it;
            llvm::Value    *three;
            llvm::Function *clamp_func;

            if (is_float) {
                three = ctx.get_constant(3.0f);
                clamp_func = get_runtime_func(RT_MDL_CLAMPF);
            } else {
                three = ctx.get_constant(3.0);
                clamp_func = get_runtime_func(RT_MDL_CLAMP);
            }
            llvm::Value *x   = ctx->CreateCall3(clamp_func, l, a, b);
            llvm::Value *b_a = ctx->CreateFSub(b, a);
            llvm::Value *x_a = ctx->CreateFSub(x, a);
            x = ctx->CreateFDiv(x_a, b_a);

            llvm::Value *t = ctx->CreateFAdd(x, x);
            t = ctx->CreateFSub(three, t);
            x = ctx->CreateFMul(x, x);
            ctx.create_return(ctx->CreateFMul(x, t));
        }
        // no alwaysinline
        break;
    case RT_MDL_MINI:
        // integer min
        {
            llvm::Value *a   = arg_it++;
            llvm::Value *b   = arg_it;
            llvm::Value *cmp = ctx->CreateICmpSLT(a, b);
            llvm::Value *res = ctx->CreateSelect(cmp, a, b);
            ctx.create_return(res);
        }
        func->addFnAttr(llvm::Attribute::AlwaysInline);
        break;
    case RT_MDL_MIN:
    case RT_MDL_MINF:
        // fp min
        {
            llvm::Value *a   = arg_it++;
            llvm::Value *b   = arg_it;
            llvm::Value *cmp = ctx->CreateFCmpOLT(a, b);
            llvm::Value *res = ctx->CreateSelect(cmp, a, b);
            ctx.create_return(res);
        }
        func->addFnAttr(llvm::Attribute::AlwaysInline);
        break;
    case RT_MDL_MAXI:
        // integer max
        {
            llvm::Value *a   = arg_it++;
            llvm::Value *b   = arg_it;
            llvm::Value *cmp = ctx->CreateICmpSGT(a, b);
            llvm::Value *res = ctx->CreateSelect(cmp, a, b);
            ctx.create_return(res);
        }
        func->addFnAttr(llvm::Attribute::AlwaysInline);
        break;
    case RT_MDL_MAX:
    case RT_MDL_MAXF:
        // fp min
        {
            llvm::Value *a   = arg_it++;
            llvm::Value *b   = arg_it;
            llvm::Value *cmp = ctx->CreateFCmpOGT(a, b);
            llvm::Value *res = ctx->CreateSelect(cmp, a, b);
            ctx.create_return(res);
        }
        func->addFnAttr(llvm::Attribute::AlwaysInline);
        break;
    case RT_MDL_TEX_RES_FLOAT:
    case RT_MDL_TEX_RES_FLOAT3:
    case RT_MDL_TEX_RES_COLOR:
        {
            llvm::Value *state = ctx.get_state_parameter();
            llvm::Value *index = arg_it;

            llvm::Type *res_type = ctx.get_return_type();

            llvm::Value *adr   = ctx.create_simple_gep_in_bounds(
                state, ctx.get_constant(m_code_gen.m_type_mapper.get_state_index(
                    Type_mapper::STATE_CORE_TEXT_RESULTS)));
            llvm::Value *tex_r = ctx->CreateLoad(adr);
            llvm::Value *ptr   = ctx->CreateGEP(tex_r, index);
            llvm::Value *res   = ctx.load_and_convert(res_type, ptr);
            ctx.create_return(res);
        }
        func->addFnAttr(llvm::Attribute::AlwaysInline);
        break;

    case RT_MDL_TO_CSTRING:
        {
            llvm::Value *res = arg_it;
            if (m_code_gen.m_type_mapper.strings_mapped_to_ids()) {
                llvm::Value *lut = m_code_gen.get_attribute_table(
                    ctx, LLVM_code_generator::RTK_STRINGS);
                llvm::Value *lut_size = m_code_gen.get_attribute_table_size(
                    ctx, LLVM_code_generator::RTK_STRINGS);

                llvm::Value *cond = ctx->CreateICmpULT(res, lut_size);

                // we do not expect out of bounds here, but if we map to 0 == NaS(tring)
                res = ctx->CreateSelect(cond, res, ctx.get_constant(Type_mapper::Tag(0)));

                llvm::Value *select[] = {
                    res
                };

                llvm::Value *adr = ctx->CreateInBoundsGEP(lut, select);

                res = ctx->CreateLoad(adr);
            }
            ctx.create_return(res);
        }
        func->addFnAttr(llvm::Attribute::AlwaysInline);
        break;

    default:
        MDL_ASSERT(!"Unsupported MDL runtime function");
        break;
    }
    return func;
}

void LLVM_code_generator::register_native_runtime_functions(Jitted_code *jitted_code)
{
    // Functions from MDL_runtime_creator::get_c_runtime_func
    #define REG_CMATH(func_impl, type) \
        jitted_code->register_function(#func_impl, check_sig<type>(::func_impl))
    #define REG_CMATH2(symname, func_impl, type) \
        jitted_code->register_function(symname, check_sig<type>(func_impl))

    REG_CMATH(abs,    II_II);
    REG_CMATH(acosf,  FF_FF);
    REG_CMATH(acos,   DD_DD);
    REG_CMATH(asinf,  FF_FF);
    REG_CMATH(asin,   DD_DD);
    REG_CMATH(atanf,  FF_FF);
    REG_CMATH(atan,   DD_DD);
    REG_CMATH(atan2f, FF_FFFF);
    REG_CMATH(atan2,  DD_DDDD);
    REG_CMATH(fmodf,  FF_FFFF);
    REG_CMATH(fmod,   DD_DDDD);
    REG_CMATH(modff,  FF_FFff);
    REG_CMATH(modf,   DD_DDdd);
    REG_CMATH2("pow", mi::mdl::powi, DD_DDII);
    REG_CMATH(tanf,   FF_FF);
    REG_CMATH(tan,    DD_DD);
#ifndef LLVM_3_4
#ifdef _MSC_VER
    // no copysign in MS runtime
    REG_CMATH2("copysign",  ::_copysign,        DD_DDDD);
    REG_CMATH2("copysignf", mi::mdl::copysignf, FF_FFFF);
#else
    REG_CMATH(copysignf, FF_FFFF);
    REG_CMATH(copysign,  DD_DDDD);
#endif
#endif

    #undef REG_CMATH
    #undef REG_CMATH2

    // Functions from MDL_runtime_creator::create_runtime_func
    #define REG_FUNC(func_impl) \
        jitted_code->register_function("mdl_" #func_impl, (void *)func_impl)
    #define REG_FUNC2(symname, func_impl) \
        jitted_code->register_function(symname, (void *)func_impl)

    REG_FUNC(tex_resolution_2d);
    REG_FUNC(tex_width);
    REG_FUNC(tex_height);
    REG_FUNC(tex_depth);
    REG_FUNC2("mdl_tex_valid", tex_isvalid);

    REG_FUNC(tex_lookup_float_2d);
    REG_FUNC(tex_lookup_deriv_float_2d);
    REG_FUNC(tex_lookup_float_3d);
    REG_FUNC(tex_lookup_float_cube);
    REG_FUNC(tex_lookup_float_ptex);

    REG_FUNC(tex_lookup_float2_2d);
    REG_FUNC(tex_lookup_deriv_float2_2d);
    REG_FUNC(tex_lookup_float2_3d);
    REG_FUNC(tex_lookup_float2_cube);
    REG_FUNC(tex_lookup_float2_ptex);

    REG_FUNC(tex_lookup_float3_2d);
    REG_FUNC(tex_lookup_deriv_float3_2d);
    REG_FUNC(tex_lookup_float3_3d);
    REG_FUNC(tex_lookup_float3_cube);
    REG_FUNC(tex_lookup_float3_ptex);

    REG_FUNC(tex_lookup_float4_2d);
    REG_FUNC(tex_lookup_deriv_float4_2d);
    REG_FUNC(tex_lookup_float4_3d);
    REG_FUNC(tex_lookup_float4_cube);
    REG_FUNC(tex_lookup_float4_ptex);

    REG_FUNC(tex_lookup_color_2d);
    REG_FUNC(tex_lookup_deriv_color_2d);
    REG_FUNC(tex_lookup_color_3d);
    REG_FUNC(tex_lookup_color_cube);
    REG_FUNC(tex_lookup_color_ptex);

    REG_FUNC(tex_texel_float_2d);
    REG_FUNC(tex_texel_float2_2d);
    REG_FUNC(tex_texel_float3_2d);
    REG_FUNC(tex_texel_float4_2d);
    REG_FUNC(tex_texel_color_2d);

    REG_FUNC(tex_texel_float_3d);
    REG_FUNC(tex_texel_float2_3d);
    REG_FUNC(tex_texel_float3_3d);
    REG_FUNC(tex_texel_float4_3d);
    REG_FUNC(tex_texel_color_3d);


    REG_FUNC(df_light_profile_power);
    REG_FUNC(df_light_profile_maximum);
    REG_FUNC2("mdl_df_light_profile_valid", df_light_profile_isvalid);
    REG_FUNC2("mdl_df_bsdf_measurement_valid", df_bsdf_measurement_isvalid);
    REG_FUNC2("mdl_df_bsdf_measurement_resolution", df_bsdf_measurement_resolution);
    REG_FUNC2("mdl_df_bsdf_measurement_evaluate", df_bsdf_measurement_evaluate);
    REG_FUNC2("mdl_df_bsdf_measurement_sample", df_bsdf_measurement_sample);
    REG_FUNC2("mdl_df_bsdf_measurement_pdf", df_bsdf_measurement_pdf);
    REG_FUNC2("mdl_df_bsdf_measurement_albedos", df_bsdf_measurement_albedos);
    REG_FUNC(df_light_profile_evaluate);
    REG_FUNC(df_light_profile_sample);
    REG_FUNC(df_light_profile_pdf);

    REG_FUNC2("mdl_blackbody", check_sig<FA3_FF>(mi::mdl::spectral::mdl_blackbody));

    REG_FUNC2("mdl_debugbreak", check_sig<VV_>(debug::debugbreak));
    REG_FUNC2("mdl_assertfail", check_sig<VV_CSCSCSII>(debug::assertfail));
    REG_FUNC2("mdl_print_begin", check_sig<lb_>(debug::print_begin));
    REG_FUNC2("mdl_print_end", check_sig<VV_lb>(debug::print_end));
    REG_FUNC2("mdl_print_bool", check_sig<VV_lbBB>(debug::print_bool));
    REG_FUNC2("mdl_print_int", check_sig<VV_lbII>(debug::print_int));
    REG_FUNC2("mdl_print_float", check_sig<VV_lbFF>(debug::print_float));
    REG_FUNC2("mdl_print_double", check_sig<VV_lbDD>(debug::print_double));
    REG_FUNC2("mdl_print_string", check_sig<VV_lbCS>(debug::print_string));

    REG_FUNC2(
        "mdl_out_of_bounds", check_sig<VV_xsIIZZCSII>(LLVM_code_generator::mdl_out_of_bounds));
    REG_FUNC2("mdl_div_by_zero", check_sig<VV_xsCSII>(LLVM_code_generator::mdl_div_by_zero));

    #undef REG_FUNC
    #undef REG_FUNC2
}

// Generate LLVM IR for state::set_normal(float3)
llvm::Function *MDL_runtime_creator::create_state_set_normal(Internal_function const *int_func)
{
    Function_instance inst(m_code_gen.get_allocator(), reinterpret_cast<size_t>(int_func));
    LLVM_context_data *ctx_data = m_code_gen.get_or_create_context_data(NULL, inst, "::state");
    llvm::Function    *func     = ctx_data->get_function();
    unsigned          flags     = ctx_data->get_function_flags();

    Function_context ctx(m_alloc, m_code_gen, inst, func, flags);
    llvm::Function::arg_iterator arg_it = ctx.get_first_parameter();

    llvm::Value *a = load_by_value(ctx, arg_it);

    if (m_code_gen.m_state_mode & Type_mapper::SSM_CORE) {
        llvm::Value *state = ctx.get_state_parameter();
        llvm::Value *adr   = ctx.create_simple_gep_in_bounds(
            state, ctx.get_constant(m_code_gen.m_type_mapper.get_state_index(
                Type_mapper::STATE_CORE_NORMAL)));
        ctx.convert_and_store(a, adr);
    }
    ctx.create_void_return();
    return func;
}

// Generate LLVM IR for state::get_texture_results()
llvm::Function *MDL_runtime_creator::create_state_get_texture_results(
    Internal_function const *int_func)
{
    Function_instance inst(m_code_gen.get_allocator(), reinterpret_cast<size_t>(int_func));
    LLVM_context_data *ctx_data = m_code_gen.get_or_create_context_data(NULL, inst, "::state");
    llvm::Function    *func     = ctx_data->get_function();
    unsigned          flags     = ctx_data->get_function_flags();

    Function_context ctx(m_alloc, m_code_gen, inst, func, flags);
    llvm::Value *res;

    llvm::Type *ret_tp = ctx_data->get_return_type();
    if (m_code_gen.m_state_mode & Type_mapper::SSM_CORE) {
        llvm::Value *state = ctx.get_state_parameter();
        res = ctx.create_simple_gep_in_bounds(
            state, ctx.get_constant(m_code_gen.m_type_mapper.get_state_index(
                Type_mapper::STATE_CORE_TEXT_RESULTS)));
        res = ctx->CreateLoad(res);
        res = ctx->CreatePointerCast(res, ret_tp);
    } else {
        // zero in all other contexts
        res = llvm::Constant::getNullValue(ret_tp);
    }
    ctx.create_return(res);
    return func;
}

// Generate LLVM IR for state::get_arg_block()
llvm::Function *MDL_runtime_creator::create_state_get_arg_block(
    Internal_function const *int_func)
{
    Function_instance inst(m_code_gen.get_allocator(), reinterpret_cast<size_t>(int_func));
    LLVM_context_data *ctx_data = m_code_gen.get_or_create_context_data(NULL, inst, "::state");
    llvm::Function    *func     = ctx_data->get_function();
    unsigned          flags     = ctx_data->get_function_flags();

    Function_context ctx(m_alloc, m_code_gen, inst, func, flags);
    llvm::Value *res;

    res = ctx.get_cap_args_parameter();
    res = ctx->CreateBitCast(res, ctx_data->get_return_type());

    ctx.create_return(res);
    return func;
}

// Generate LLVM IR for state::get_ro_data_segment()
llvm::Function *MDL_runtime_creator::create_state_get_ro_data_segment(
    Internal_function const *int_func)
{
    Function_instance inst(m_code_gen.get_allocator(), reinterpret_cast<size_t>(int_func));
    LLVM_context_data *ctx_data = m_code_gen.get_or_create_context_data(NULL, inst, "::state");
    llvm::Function    *func     = ctx_data->get_function();
    unsigned          flags     = ctx_data->get_function_flags();

    Function_context ctx(m_alloc, m_code_gen, inst, func, flags);
    llvm::Value *res;

    llvm::Type *ret_tp = ctx_data->get_return_type();
    if (m_code_gen.m_state_mode & Type_mapper::SSM_CORE) {
        llvm::Value *state = ctx.get_state_parameter();
        res = ctx.create_simple_gep_in_bounds(
            state, ctx.get_constant(m_code_gen.m_type_mapper.get_state_index(
                Type_mapper::STATE_CORE_RO_DATA_SEG)));
        res = ctx->CreateLoad(res);
        res = ctx->CreatePointerCast(res, ret_tp);
    } else {
        // zero in all other contexts
        res = llvm::Constant::getNullValue(ret_tp);
    }
    ctx.create_return(res);
    return func;
}

// Generate LLVM IR for state::object_id()
llvm::Function *MDL_runtime_creator::create_state_object_id(
    Internal_function const *int_func)
{
    Function_instance inst(m_code_gen.get_allocator(), reinterpret_cast<size_t>(int_func));
    LLVM_context_data *ctx_data = m_code_gen.get_or_create_context_data(NULL, inst, "::state");
    llvm::Function    *func     = ctx_data->get_function();
    unsigned          flags     = ctx_data->get_function_flags();

    Function_context ctx(m_alloc, m_code_gen, inst, func, flags);
    llvm::Value *res;

    llvm::Type *ret_tp = ctx_data->get_return_type();
    if (m_code_gen.m_state_mode & Type_mapper::SSM_CORE) {
        llvm::Value *state = ctx.get_state_parameter();
        res = ctx.create_simple_gep_in_bounds(
            state, ctx.get_constant(m_code_gen.m_type_mapper.get_state_index(
                Type_mapper::STATE_CORE_OBJECT_ID)));
        res = ctx->CreateLoad(res);
    } else {
        // zero in all other contexts
        res = llvm::Constant::getNullValue(ret_tp);
    }
    ctx.create_return(res);
    return func;
}

llvm::Function *MDL_runtime_creator::create_df_bsdf_measurement_resolution(
    Internal_function const *int_func)
{
    Function_instance inst(m_code_gen.get_allocator(), reinterpret_cast<size_t>(int_func));
    LLVM_context_data *ctx_data = m_code_gen.get_or_create_context_data(NULL, inst, "::df");
    llvm::Function    *func = ctx_data->get_function();
    unsigned          flags = ctx_data->get_function_flags();

    Function_context ctx(m_alloc, m_code_gen, inst, func, flags);
    llvm::Value *res;

    llvm::Function::arg_iterator arg_it = ctx.get_first_parameter();
    llvm::Value *res_data = ctx.get_resource_data_parameter();
    llvm::Value *a = load_by_value(ctx, arg_it++);
    llvm::Value *b = load_by_value(ctx, arg_it++);

    llvm::Type  *res_type = m_code_gen.m_type_mapper.get_arr_int_3_type();
    llvm::Value *tmp = ctx.create_local(res_type, "tmp");

    if (m_has_res_handler)
    {
        llvm::Function *lookup_func = get_runtime_func(RT_MDL_DF_BSDF_MEASUREMENT_RESOLUTION);

        llvm::Value *args[] = {tmp, res_data, a, b};
        ctx->CreateCall(lookup_func, args);
    }
    else
    {
        llvm::Value *self_adr = ctx.create_simple_gep_in_bounds(
            res_data, ctx.get_constant(Type_mapper::RDP_THREAD_DATA));
        llvm::Value *self = ctx->CreateBitCast(
            ctx->CreateLoad(self_adr),
            m_code_gen.m_type_mapper.get_core_tex_handler_ptr_type());

        llvm::Value *resolution_func = ctx.get_tex_lookup_func(
            self, Type_mapper::THV_bsdf_measurement_resolution);

        llvm::Value *args[] = {tmp, self, a, b};
        llvm::CallInst *call = ctx->CreateCall(resolution_func, args);
        call->setDoesNotThrow();
    }
    res = ctx.load_and_convert(ctx_data->get_return_type(), tmp);
    ctx.create_return(res);
    return func;
}

llvm::Function *MDL_runtime_creator::create_df_bsdf_measurement_evaluate(
    Internal_function const *int_func)
{
    Function_instance inst(m_code_gen.get_allocator(), reinterpret_cast<size_t>(int_func));
    LLVM_context_data *ctx_data = m_code_gen.get_or_create_context_data(NULL, inst, "::df");
    llvm::Function    *func = ctx_data->get_function();
    unsigned          flags = ctx_data->get_function_flags();

    Function_context ctx(m_alloc, m_code_gen, inst, func, flags);
    llvm::Value *res;

    llvm::Function::arg_iterator arg_it = ctx.get_first_parameter();
    llvm::Value *res_data = ctx.get_resource_data_parameter();
    llvm::Value *a = load_by_value(ctx, arg_it++);
    llvm::Value *b = load_by_value(ctx, arg_it++);
    llvm::Value *c = load_by_value(ctx, arg_it++);
    llvm::Value *d = load_by_value(ctx, arg_it++);


    llvm::Type  *res_type = m_code_gen.m_type_mapper.get_arr_float_3_type();
    llvm::Value *tmp = ctx.create_local(res_type, "tmp");

    llvm::Type  *polar_type = m_code_gen.m_type_mapper.get_arr_float_2_type();
    llvm::Value *theta_phi_in = ctx.create_local(polar_type, "theta_phi_in");
    llvm::Value *theta_phi_out = ctx.create_local(polar_type, "theta_phi_out");
    ctx.convert_and_store(b, theta_phi_in);
    ctx.convert_and_store(c, theta_phi_out);

    if (m_has_res_handler)
    {
        llvm::Function *lookup_func = get_runtime_func(RT_MDL_DF_BSDF_MEASUREMENT_EVALUATE);

        llvm::Value *args[] = {tmp, res_data, a, theta_phi_in, theta_phi_out, d};
        ctx->CreateCall(lookup_func, args);
    }
    else
    {
        llvm::Value *self_adr = ctx.create_simple_gep_in_bounds(
            res_data, ctx.get_constant(Type_mapper::RDP_THREAD_DATA));
        llvm::Value *self = ctx->CreateBitCast(
            ctx->CreateLoad(self_adr),
            m_code_gen.m_type_mapper.get_core_tex_handler_ptr_type());

        llvm::Value *lookup_func = ctx.get_tex_lookup_func(
            self, Type_mapper::THV_bsdf_measurement_evaluate);

        llvm::Value *args[] = {tmp, self, a, theta_phi_in, theta_phi_out, d};
        llvm::CallInst *call = ctx->CreateCall(lookup_func, args);
        call->setDoesNotThrow();
    }
    res = ctx.load_and_convert(ctx_data->get_return_type(), tmp);
    ctx.create_return(res);
    return func;
}

llvm::Function *MDL_runtime_creator::create_df_bsdf_measurement_sample(
    Internal_function const *int_func)
{
    Function_instance inst(m_code_gen.get_allocator(), reinterpret_cast<size_t>(int_func));
    LLVM_context_data *ctx_data = m_code_gen.get_or_create_context_data(NULL, inst, "::df");
    llvm::Function    *func = ctx_data->get_function();
    unsigned          flags = ctx_data->get_function_flags();

    Function_context ctx(m_alloc, m_code_gen, inst, func, flags);
    llvm::Value *res;

    llvm::Function::arg_iterator arg_it = ctx.get_first_parameter();
    llvm::Value *res_data = ctx.get_resource_data_parameter();
    llvm::Value *a = load_by_value(ctx, arg_it++);
    llvm::Value *b = load_by_value(ctx, arg_it++);
    llvm::Value *c = load_by_value(ctx, arg_it++);
    llvm::Value *d = load_by_value(ctx, arg_it++);


    llvm::Type  *arr_float_2_type = m_code_gen.m_type_mapper.get_arr_float_2_type();
    llvm::Type  *arr_float_3_type = m_code_gen.m_type_mapper.get_arr_float_3_type();

    llvm::Value *tmp = ctx.create_local(arr_float_3_type, "tmp");

    llvm::Value *theta_phi_out = ctx.create_local(arr_float_2_type, "theta_phi_out");
    ctx.convert_and_store(b, theta_phi_out);
    
    llvm::Value *xi = ctx.create_local(arr_float_3_type, "xi");
    ctx.convert_and_store(c, xi);

    if (m_has_res_handler)
    {
        llvm::Function *lookup_func = get_runtime_func(RT_MDL_DF_BSDF_MEASUREMENT_SAMPLE);

        llvm::Value *args[] = {tmp, res_data, a, theta_phi_out, xi, d};
        ctx->CreateCall(lookup_func, args);
    }
    else
    {
        llvm::Value *self_adr = ctx.create_simple_gep_in_bounds(
            res_data, ctx.get_constant(Type_mapper::RDP_THREAD_DATA));
        llvm::Value *self = ctx->CreateBitCast(
            ctx->CreateLoad(self_adr),
            m_code_gen.m_type_mapper.get_core_tex_handler_ptr_type());

        llvm::Value *lookup_func = ctx.get_tex_lookup_func(
            self, Type_mapper::THV_bsdf_measurement_sample);

        llvm::Value *args[] = {tmp, self, a, theta_phi_out, xi, d};
        llvm::CallInst *call = ctx->CreateCall(lookup_func, args);
        call->setDoesNotThrow();
    }
    res = ctx.load_and_convert(ctx_data->get_return_type(), tmp);
    ctx.create_return(res);
    return func;
}

llvm::Function *MDL_runtime_creator::create_df_bsdf_measurement_pdf(
    Internal_function const *int_func)
{
    Function_instance inst(m_code_gen.get_allocator(), reinterpret_cast<size_t>(int_func));
    LLVM_context_data *ctx_data = m_code_gen.get_or_create_context_data(NULL, inst, "::df");
    llvm::Function    *func = ctx_data->get_function();
    unsigned          flags = ctx_data->get_function_flags();

    Function_context ctx(m_alloc, m_code_gen, inst, func, flags);

    llvm::Function::arg_iterator arg_it = ctx.get_first_parameter();
    llvm::Value *res_data = ctx.get_resource_data_parameter();
    llvm::Value *a = load_by_value(ctx, arg_it++);
    llvm::Value *b = load_by_value(ctx, arg_it++);
    llvm::Value *c = load_by_value(ctx, arg_it++);
    llvm::Value *d = load_by_value(ctx, arg_it++);

    llvm::Type  *polar_type = m_code_gen.m_type_mapper.get_arr_float_2_type();
    llvm::Value *theta_phi_in = ctx.create_local(polar_type, "theta_phi_in");
    llvm::Value *theta_phi_out = ctx.create_local(polar_type, "theta_phi_out");
    ctx.convert_and_store(b, theta_phi_in);
    ctx.convert_and_store(c, theta_phi_out);

    llvm::CallInst* call;
    if (m_has_res_handler)
    {
        llvm::Function *lookup_func = get_runtime_func(RT_MDL_DF_BSDF_MEASUREMENT_PDF);

        llvm::Value *args[] = {res_data, a, theta_phi_in, theta_phi_out, d};
        call = ctx->CreateCall(lookup_func, args);
    }
    else
    {
        llvm::Value *self_adr = ctx.create_simple_gep_in_bounds(
            res_data, ctx.get_constant(Type_mapper::RDP_THREAD_DATA));
        llvm::Value *self = ctx->CreateBitCast(
            ctx->CreateLoad(self_adr),
            m_code_gen.m_type_mapper.get_core_tex_handler_ptr_type());

        llvm::Value *lookup_func = ctx.get_tex_lookup_func(
            self, Type_mapper::THV_bsdf_measurement_pdf);

        llvm::Value *args[] = {self, a, theta_phi_in, theta_phi_out, d};
        call = ctx->CreateCall(lookup_func, args);
        call->setDoesNotThrow();
    }
    ctx.create_return(call);
    return func;
}

llvm::Function *MDL_runtime_creator::create_df_bsdf_measurement_albedos(
    Internal_function const *int_func)
{
    Function_instance inst(m_code_gen.get_allocator(), reinterpret_cast<size_t>(int_func));
    LLVM_context_data *ctx_data = m_code_gen.get_or_create_context_data(NULL, inst, "::df");
    llvm::Function    *func = ctx_data->get_function();
    unsigned          flags = ctx_data->get_function_flags();

    Function_context ctx(m_alloc, m_code_gen, inst, func, flags);
    llvm::Value *res;

    llvm::Function::arg_iterator arg_it = ctx.get_first_parameter();
    llvm::Value *res_data = ctx.get_resource_data_parameter();
    llvm::Value *a = load_by_value(ctx, arg_it++);
    llvm::Value *b = load_by_value(ctx, arg_it++);

    llvm::Type  *arr_float_2_type = m_code_gen.m_type_mapper.get_arr_float_2_type();
    llvm::Type  *arr_float_4_type = m_code_gen.m_type_mapper.get_arr_float_4_type();

    llvm::Value *tmp = ctx.create_local(arr_float_4_type, "tmp");

    llvm::Value *theta_phi = ctx.create_local(arr_float_2_type, "theta_phi");
    ctx.convert_and_store(b, theta_phi);
    
    if (m_has_res_handler)
    {
        llvm::Function *lookup_func = get_runtime_func(RT_MDL_DF_BSDF_MEASUREMENT_ALBEDOS);

        llvm::Value *args[] = {tmp, res_data, a, theta_phi};
        ctx->CreateCall(lookup_func, args);
    }
    else
    {
        llvm::Value *self_adr = ctx.create_simple_gep_in_bounds(
            res_data, ctx.get_constant(Type_mapper::RDP_THREAD_DATA));
        llvm::Value *self = ctx->CreateBitCast(
            ctx->CreateLoad(self_adr),
            m_code_gen.m_type_mapper.get_core_tex_handler_ptr_type());

        llvm::Value *lookup_func = ctx.get_tex_lookup_func(
            self, Type_mapper::THV_bsdf_measurement_albedos);

        llvm::Value *args[] = {tmp, self, a, theta_phi};
        llvm::CallInst *call = ctx->CreateCall(lookup_func, args);
        call->setDoesNotThrow();
    }
    res = ctx.load_and_convert(ctx_data->get_return_type(), tmp);
    ctx.create_return(res);
    return func;
}

llvm::Function *MDL_runtime_creator::create_df_light_profile_evaluate(
    Internal_function const *int_func)
{
    Function_instance inst(m_code_gen.get_allocator(), reinterpret_cast<size_t>(int_func));
    LLVM_context_data *ctx_data = m_code_gen.get_or_create_context_data(NULL, inst, "::df");
    llvm::Function    *func = ctx_data->get_function();
    unsigned          flags = ctx_data->get_function_flags();

    Function_context ctx(m_alloc, m_code_gen, inst, func, flags);

    llvm::Function::arg_iterator arg_it = ctx.get_first_parameter();
    llvm::Value *res_data = ctx.get_resource_data_parameter();
    llvm::Value *a = load_by_value(ctx, arg_it++);
    llvm::Value *b = load_by_value(ctx, arg_it++);

    llvm::Type  *arr_float_2_type = m_code_gen.m_type_mapper.get_arr_float_2_type();
    llvm::Value *theta_phi = ctx.create_local(arr_float_2_type, "theta_phi");
    ctx.convert_and_store(b, theta_phi);
    
    llvm::CallInst* call;
    if (m_has_res_handler)
    {
        llvm::Function *lookup_func = get_runtime_func(RT_MDL_DF_LIGHT_PROFILE_EVALUATE);

        llvm::Value *args[] = {res_data, a, theta_phi};
        call = ctx->CreateCall(lookup_func, args);
    }
    else
    {
        llvm::Value *self_adr = ctx.create_simple_gep_in_bounds(
            res_data, ctx.get_constant(Type_mapper::RDP_THREAD_DATA));
        llvm::Value *self = ctx->CreateBitCast(
            ctx->CreateLoad(self_adr),
            m_code_gen.m_type_mapper.get_core_tex_handler_ptr_type());

        llvm::Value *lookup_func = ctx.get_tex_lookup_func(
            self, Type_mapper::THV_light_profile_evaluate);

        llvm::Value *args[] = {self, a, theta_phi};
        call = ctx->CreateCall(lookup_func, args);
        call->setDoesNotThrow();
    }
    ctx.create_return(call);
    return func;
}


llvm::Function *MDL_runtime_creator::create_df_light_profile_sample(
    Internal_function const *int_func)
{
    Function_instance inst(m_code_gen.get_allocator(), reinterpret_cast<size_t>(int_func));
    LLVM_context_data *ctx_data = m_code_gen.get_or_create_context_data(NULL, inst, "::df");
    llvm::Function    *func = ctx_data->get_function();
    unsigned          flags = ctx_data->get_function_flags();

    Function_context ctx(m_alloc, m_code_gen, inst, func, flags);
    llvm::Value *res;

    llvm::Function::arg_iterator arg_it = ctx.get_first_parameter();
    llvm::Value *res_data = ctx.get_resource_data_parameter();
    llvm::Value *a = load_by_value(ctx, arg_it++);
    llvm::Value *b = load_by_value(ctx, arg_it++);

    llvm::Type  *arr_float_3_type = m_code_gen.m_type_mapper.get_arr_float_3_type();
    llvm::Value *tmp = ctx.create_local(arr_float_3_type, "tmp");
    llvm::Value *xi = ctx.create_local(arr_float_3_type, "xi");
    ctx.convert_and_store(b, xi);

    if (m_has_res_handler)
    {
        llvm::Function *lookup_func = get_runtime_func(RT_MDL_DF_LIGHT_PROFILE_SAMPLE);

        llvm::Value *args[] = {tmp, res_data, a, xi};
        ctx->CreateCall(lookup_func, args);
    }
    else
    {
        llvm::Value *self_adr = ctx.create_simple_gep_in_bounds(
            res_data, ctx.get_constant(Type_mapper::RDP_THREAD_DATA));
        llvm::Value *self = ctx->CreateBitCast(
            ctx->CreateLoad(self_adr),
            m_code_gen.m_type_mapper.get_core_tex_handler_ptr_type());

        llvm::Value *lookup_func = ctx.get_tex_lookup_func(
            self, Type_mapper::THV_light_profile_sample);

        llvm::Value *args[] = {tmp, self, a, xi};
        llvm::CallInst *call = ctx->CreateCall(lookup_func, args);
        call->setDoesNotThrow();
    }
    res = ctx.load_and_convert(ctx_data->get_return_type(), tmp);
    ctx.create_return(res);
    return func;
}

llvm::Function *MDL_runtime_creator::create_df_light_profile_pdf(
    Internal_function const *int_func)
{
    Function_instance inst(m_code_gen.get_allocator(), reinterpret_cast<size_t>(int_func));
    LLVM_context_data *ctx_data = m_code_gen.get_or_create_context_data(NULL, inst, "::df");
    llvm::Function    *func = ctx_data->get_function();
    unsigned          flags = ctx_data->get_function_flags();

    Function_context ctx(m_alloc, m_code_gen, inst, func, flags);

    llvm::Function::arg_iterator arg_it = ctx.get_first_parameter();
    llvm::Value *res_data = ctx.get_resource_data_parameter();
    llvm::Value *a = load_by_value(ctx, arg_it++);
    llvm::Value *b = load_by_value(ctx, arg_it++);

    llvm::Type  *arr_float_2_type = m_code_gen.m_type_mapper.get_arr_float_2_type();
    llvm::Value *theta_phi = ctx.create_local(arr_float_2_type, "theta_phi");
    ctx.convert_and_store(b, theta_phi);
    
    llvm::CallInst* call;
    if (m_has_res_handler)
    {
        llvm::Function *lookup_func = get_runtime_func(RT_MDL_DF_LIGHT_PROFILE_PDF);

        llvm::Value *args[] = {res_data, a, theta_phi};
        call = ctx->CreateCall(lookup_func, args);
    }
    else
    {
        llvm::Value *self_adr = ctx.create_simple_gep_in_bounds(
            res_data, ctx.get_constant(Type_mapper::RDP_THREAD_DATA));
        llvm::Value *self = ctx->CreateBitCast(
            ctx->CreateLoad(self_adr),
            m_code_gen.m_type_mapper.get_core_tex_handler_ptr_type());

        llvm::Value *lookup_func = ctx.get_tex_lookup_func(
            self, Type_mapper::THV_light_profile_pdf);

        llvm::Value *args[] = {self, a, theta_phi};
        call = ctx->CreateCall(lookup_func, args);
        call->setDoesNotThrow();
    }
    ctx.create_return(call);
    return func;
}

// Generate LLVM IR for an internal function.
llvm::Function *MDL_runtime_creator::get_internal_function(Internal_function const *int_func)
{
    Internal_function::Kind kind = int_func->get_kind();

    if (m_internal_funcs[kind] == NULL) {
        if (m_use_user_state_module) {
            char const *func_name = NULL;
            switch (kind) {
                case Internal_function::KI_STATE_SET_NORMAL:
                    if (m_code_gen.m_state_mode & Type_mapper::SSM_CORE)
                        func_name = "_ZN5state10set_normalEP10State_coreDv3_f";
                    else
                        func_name = "_ZN5state10set_normalEP17State_environmentDv3_f";
                    break;
                case Internal_function::KI_STATE_GET_TEXTURE_RESULTS:
                    if (m_code_gen.m_state_mode & Type_mapper::SSM_CORE)
                        func_name = "_ZN5state19get_texture_resultsEP10State_core";
                    else
                        func_name = "_ZN5state19get_texture_resultsEP17State_environment";
                    break;
                case Internal_function::KI_STATE_GET_RO_DATA_SEGMENT:
                    if (m_code_gen.m_state_mode & Type_mapper::SSM_CORE)
                        func_name = "_ZN5state19get_ro_data_segmentEPK10State_core";
                    else
                        func_name = "_ZN5state19get_ro_data_segmentEPK17State_environment";
                    break;
                case Internal_function::KI_STATE_OBJECT_ID:
                    if (m_code_gen.m_state_mode & Type_mapper::SSM_CORE)
                        func_name = "_ZN5state9object_idEPK10State_core";
                    else
                        func_name = "_ZN5state9object_idEPK17State_environment";
                    break;
                default: break;
            }
            if (func_name != NULL) {
                if (llvm::Function *func = m_code_gen.get_llvm_module()->getFunction(func_name)) {
                    m_code_gen.create_context_data(int_func, func);
                    m_internal_funcs[kind] = func;
                    return func;
                }
            }
        }

        switch (kind) {
            case Internal_function::KI_STATE_SET_NORMAL:
                m_internal_funcs[kind] = create_state_set_normal(int_func);
                break;

            case Internal_function::KI_STATE_GET_TEXTURE_RESULTS:
                m_internal_funcs[kind] = create_state_get_texture_results(int_func);
                break;

            case Internal_function::KI_STATE_GET_ARG_BLOCK:
                m_internal_funcs[kind] = create_state_get_arg_block(int_func);
                break;

            case Internal_function::KI_STATE_GET_RO_DATA_SEGMENT:
                m_internal_funcs[kind] = create_state_get_ro_data_segment(int_func);
                break;

            case Internal_function::KI_STATE_OBJECT_ID:
                m_internal_funcs[kind] = create_state_object_id(int_func);
                break;

            case Internal_function::KI_STATE_CALL_LAMBDA_FLOAT:
            case Internal_function::KI_STATE_CALL_LAMBDA_FLOAT3:
            {
                // the function body will be created later when all compiled module lambda
                // functions are known
                Function_instance inst(
                    m_code_gen.get_allocator(), reinterpret_cast<size_t>(int_func));
                LLVM_context_data *ctx_data =
                    m_code_gen.get_or_create_context_data(NULL, inst, "::state");
                m_internal_funcs[kind] = ctx_data->get_function();
                break;
            }

            case Internal_function::KI_DF_BSDF_MEASUREMENT_RESOLUTION:
                m_internal_funcs[kind] = create_df_bsdf_measurement_resolution(int_func);
                break;

            case Internal_function::KI_DF_BSDF_MEASUREMENT_EVALUATE:
                m_internal_funcs[kind] = create_df_bsdf_measurement_evaluate(int_func);
                break;

            case Internal_function::KI_DF_BSDF_MEASUREMENT_SAMPLE:
                m_internal_funcs[kind] = create_df_bsdf_measurement_sample(int_func);
                break;

            case Internal_function::KI_DF_BSDF_MEASUREMENT_PDF:
                m_internal_funcs[kind] = create_df_bsdf_measurement_pdf(int_func);
                break;

            case Internal_function::KI_DF_BSDF_MEASUREMENT_ALBEDOS:
                m_internal_funcs[kind] = create_df_bsdf_measurement_albedos(int_func);
                break;

            case Internal_function::KI_DF_LIGHT_PROFILE_EVALUATE:
                m_internal_funcs[kind] = create_df_light_profile_evaluate(int_func);
                break;

            case Internal_function::KI_DF_LIGHT_PROFILE_SAMPLE:
                m_internal_funcs[kind] = create_df_light_profile_sample(int_func);
                break;

            case Internal_function::KI_DF_LIGHT_PROFILE_PDF:
                m_internal_funcs[kind] = create_df_light_profile_pdf(int_func);
                break;

            default:
                MDL_ASSERT(!"Unsupported MDL internal function");
                return NULL;
        }
    }

    return m_internal_funcs[kind];
}

// Check if the given argument is an index.
int LLVM_code_generator::is_index_argument(mi::mdl::IDefinition::Semantics sema, int i)
{
    if (i == 0) {
        switch (sema) {
        case mi::mdl::IDefinition::DS_INTRINSIC_STATE_TEXTURE_COORDINATE:
        case mi::mdl::IDefinition::DS_INTRINSIC_STATE_TEXTURE_TANGENT_U:
        case mi::mdl::IDefinition::DS_INTRINSIC_STATE_TEXTURE_TANGENT_V:
        case mi::mdl::IDefinition::DS_INTRINSIC_STATE_TANGENT_SPACE:
        case mi::mdl::IDefinition::DS_INTRINSIC_STATE_GEOMETRY_TANGENT_U:
        case mi::mdl::IDefinition::DS_INTRINSIC_STATE_GEOMETRY_TANGENT_V:
            return m_num_texture_spaces;
        default:
            break;
        }
    }
    return -1;
}

// Translate a call to a compiler known function to LLVM IR.
llvm::Value *LLVM_code_generator::translate_call_intrinsic_function(
    Function_context          &ctx,
    mi::mdl::ICall_expr const *call_expr)
{
    mi::mdl::IDefinition const *callee_def = call_expr->get_callee_definition(*this);

    switch (callee_def->get_semantics()) {
    case IDefinition::DS_INTRINSIC_STATE_POSITION:
        m_render_state_usage |= IGenerated_code_executable::SU_POSITION;
        break;
    case IDefinition::DS_INTRINSIC_STATE_NORMAL:
        m_render_state_usage |= IGenerated_code_executable::SU_NORMAL;
        break;
    case IDefinition::DS_INTRINSIC_STATE_GEOMETRY_NORMAL:
        m_render_state_usage |= IGenerated_code_executable::SU_GEOMETRY_NORMAL;
        break;
    case IDefinition::DS_INTRINSIC_STATE_MOTION:
        m_render_state_usage |= IGenerated_code_executable::SU_MOTION;
        break;
    case IDefinition::DS_INTRINSIC_STATE_TEXTURE_COORDINATE:
        m_render_state_usage |= IGenerated_code_executable::SU_TEXTURE_COORDINATE;
        break;
    case IDefinition::DS_INTRINSIC_STATE_TEXTURE_TANGENT_U:
    case IDefinition::DS_INTRINSIC_STATE_TEXTURE_TANGENT_V:
        m_render_state_usage |= IGenerated_code_executable::SU_TEXTURE_TANGENTS;
        break;
    case IDefinition::DS_INTRINSIC_STATE_TANGENT_SPACE:
        m_render_state_usage |= IGenerated_code_executable::SU_TANGENT_SPACE;
        break;
    case IDefinition::DS_INTRINSIC_STATE_GEOMETRY_TANGENT_U:
    case IDefinition::DS_INTRINSIC_STATE_GEOMETRY_TANGENT_V:
        m_render_state_usage |= IGenerated_code_executable::SU_GEOMETRY_TANGENTS;
        break;
    case IDefinition::DS_INTRINSIC_STATE_DIRECTION:
        m_render_state_usage |= IGenerated_code_executable::SU_DIRECTION;
        break;
    case IDefinition::DS_INTRINSIC_STATE_ANIMATION_TIME:
        m_render_state_usage |= IGenerated_code_executable::SU_ANIMATION_TIME;
        break;
    case IDefinition::DS_INTRINSIC_STATE_ROUNDED_CORNER_NORMAL:
        m_render_state_usage |= IGenerated_code_executable::SU_ROUNDED_CORNER_NORMAL;
        break;
    case IDefinition::DS_INTRINSIC_STATE_TRANSFORM:
    case IDefinition::DS_INTRINSIC_STATE_TRANSFORM_NORMAL:
    case IDefinition::DS_INTRINSIC_STATE_TRANSFORM_POINT:
    case IDefinition::DS_INTRINSIC_STATE_TRANSFORM_SCALE:
    case IDefinition::DS_INTRINSIC_STATE_TRANSFORM_VECTOR:
        m_render_state_usage |= IGenerated_code_executable::SU_TRANSFORMS;
        break;
    case IDefinition::DS_INTRINSIC_STATE_OBJECT_ID:
        m_render_state_usage |= IGenerated_code_executable::SU_OBJECT_ID;
        break;
    default:
        break;
    }

    mi::mdl::IDeclaration_function const *fdecl =
        as<mi::mdl::IDeclaration_function>(callee_def->get_declaration());

    if (fdecl != NULL && fdecl->get_body() != NULL) {
        // intrinsic function WITH body, compile like user defined one
        return translate_call_user_defined_function(ctx, call_expr);
    }

    bool return_derivs = call_expr->returns_derivatives(*this);

    llvm::Function *callee = m_runtime->get_intrinsic_function(callee_def, return_derivs);

    // Intrinsic functions are NEVER instantiated!
    Function_instance inst(get_allocator(), callee_def, return_derivs);
    LLVM_context_data *p_data = get_context_data(inst);

    Func_deriv_info const *func_deriv_info = NULL;
    if (m_deriv_infos != NULL)
        func_deriv_info = m_deriv_infos->get_function_derivative_infos(inst);

    // prepare arguments
    llvm::SmallVector<llvm::Value *, 8> args;

    llvm::Type  *res_type = p_data->get_return_type();
    llvm::Value *sret_res = NULL;
    if (p_data->is_sret_return()) {
        // create a temporary for the function result and pass it
        sret_res = ctx.create_local(res_type, "call_result");
        args.push_back(sret_res);
    }

    if (p_data->has_state_param()) {
        // pass state parameter
        llvm::Value *state = ctx.get_state_parameter();
        args.push_back(state);

        m_uses_state_param = true;
    }
    if (p_data->has_resource_data_param()) {
        // pass resource_data parameter
        llvm::Value *res_data = ctx.get_resource_data_parameter();
        args.push_back(res_data);
    }
    if (p_data->has_exc_state_param()) {
        // pass exc_state parameter
        llvm::Value *exc_state = ctx.get_exc_state_parameter();
        args.push_back(exc_state);
    }
    if (p_data->has_object_id_param()) {
        // pass object_id parameter
        llvm::Value *object_id = ctx.get_object_id_value();
        args.push_back(object_id);
    }
    if (p_data->has_transform_params()) {
        // pass transform (matrix) parameters
        llvm::Value *w2o = ctx.get_w2o_transform_value();
        args.push_back(w2o);
        llvm::Value *o2w = ctx.get_o2w_transform_value();
        args.push_back(o2w);
    }

    int clip_pos = -1, clip_bound = -1;
    mi::mdl::IDefinition::Semantics sema = callee_def->get_semantics();
    int n_args = call_expr->get_argument_count();
    for (int i = 0; i < n_args; ++i) {
        mi::mdl::IType const *arg_type = call_expr->get_argument_type(i);

        bool arg_is_deriv =
            func_deriv_info != NULL && func_deriv_info->args_want_derivatives.test_bit(i + 1);

        Expression_result    expr_res  = call_expr->translate_argument(*this, ctx, i, arg_is_deriv);

        if (m_type_mapper.is_passed_by_reference(arg_type) ||
                m_type_mapper.is_deriv_type(expr_res.get_value_type())) {
            // pass a reference
            args.push_back(expr_res.as_ptr(ctx));
        } else {
            // pass by value
            args.push_back(expr_res.as_value(ctx));
        }

        int upper_bound = is_index_argument(sema, i);
        if (upper_bound >= 0) {
            if (m_bounds_check_exception_disabled) {
                // ensure that the state is not accessed wrong
                clip_pos   = i;
                clip_bound = upper_bound;
            } else {
                llvm::Value             *bound = ctx.get_constant(size_t(upper_bound));
                mi::mdl::Position const *pos   = call_expr->get_argument_position(i);

                ctx.create_bounds_check_with_exception(
                    args.back(), bound, Exc_location(*this, pos));
            }
        }
    }

    llvm::BasicBlock *end_bb = NULL;
    llvm::Value      *tmp    = NULL;

    if (clip_pos >= 0) {
        llvm::Value *bound    = ctx.get_constant(clip_bound);
        llvm::Value *uindex   = args[clip_pos + p_data->get_func_param_offset()];

        // bound is always a constant ...
        if (llvm::isa<llvm::ConstantInt>(uindex)) {
            // both are constant, decide here
            llvm::ConstantInt *c_index = llvm::cast<llvm::ConstantInt>(uindex);
            llvm::ConstantInt *c_bound = llvm::cast<llvm::ConstantInt>(bound);

            llvm::APInt const &v_index = c_index->getValue();
            llvm::APInt const &v_bound = c_bound->getValue();

            if (v_index.ult(v_bound)) {
                // index is valid, do nothing
            } else {
                // return zero if out of bounds, do not call the function
                return llvm::Constant::getNullValue(res_type);
            }
        } else {
            // handle at runtime
            end_bb = ctx.create_bb("bound_end");

            tmp = ctx.create_local(res_type, "bound_tmp");

            llvm::BasicBlock *ok_bb   = ctx.create_bb("bound_ok");
            llvm::BasicBlock *fail_bb = ctx.create_bb("bound_fail");
            llvm::Value      *cmp     = ctx->CreateICmpULT(uindex, bound);

            // set branch weights, we expect all bounds checks to be ok and never fail
            ctx.CreateWeightedCondBr(cmp, ok_bb, fail_bb, 1, 0);

            ctx->SetInsertPoint(fail_bb);

            // return zero if out of bounds
            llvm::Value *null = llvm::Constant::getNullValue(res_type);
            ctx->CreateStore(null, tmp);

            ctx->CreateBr(end_bb);
            ctx->SetInsertPoint(ok_bb);
        }
    }
    // call it
    llvm::Value *res = ctx->CreateCall(callee, args);

    if (sret_res != NULL) {
        // the result was passed on the stack
        res = ctx->CreateLoad(sret_res);
    }

    // derivative result was requested, but not delivered by intrinsic?
    if (m_type_mapper.is_deriv_type(call_expr->get_type()) &&
            !m_type_mapper.is_deriv_type(res->getType())) {
        // convert to derivative
        llvm::Value *zero = llvm::Constant::getNullValue(res->getType());
        llvm::Value *agg = llvm::ConstantAggregateZero::get(
            m_type_mapper.lookup_deriv_type(call_expr->get_type()));
        agg = ctx->CreateInsertValue(agg, res,  {0});
        agg = ctx->CreateInsertValue(agg, zero, {1});
        agg = ctx->CreateInsertValue(agg, zero, {2});
        res = agg;
    }

    if (end_bb != NULL) {
        ctx->CreateStore(res, tmp);

        ctx->CreateBr(end_bb);
        ctx->SetInsertPoint(end_bb);

        res = ctx->CreateLoad(tmp);
    }

    return res;
}

// Create a runtime.
MDL_runtime_creator *LLVM_code_generator::create_mdl_runtime(
    mi::mdl::Arena_builder &arena_builder,
    LLVM_code_generator    *code_gen,
    bool                   ptx_mode,
    bool                   fast_math,
    bool                   has_texture_handler,
    char const             *internal_space)
{
    int encoding = coordinate_world;
    if (strcmp(internal_space, "coordinate_object") == 0)
        encoding = coordinate_object;
    mi::mdl::IAllocator *alloc = arena_builder.get_arena()->get_allocator();
    return arena_builder.create<MDL_runtime_creator>(
        alloc,
        code_gen,
        ptx_mode,
        fast_math,
        /*has_sincos=*/ptx_mode,
        has_texture_handler,
        encoding);
}

// Terminate the runtime.
void LLVM_code_generator::terminate_mdl_runtime(MDL_runtime_creator *creator)
{
    // just call the destructor
    creator->~MDL_runtime_creator();
}

// Retrieve the out-of-bounce reporting routine.
llvm::Function *LLVM_code_generator::get_out_of_bounds() const
{
    return m_runtime->get_runtime_func(MDL_runtime_creator::RT_MDL_OUT_OF_BOUNDS);
}

// Retrieve the div-by-zero reporting routine.
llvm::Function *LLVM_code_generator::get_div_by_zero() const
{
    return m_runtime->get_runtime_func(MDL_runtime_creator::RT_MDL_DIV_BY_ZERO);
}

// Handle out of bounds.
void LLVM_code_generator::mdl_out_of_bounds(
    Exc_state  &exc_state,
    int        index,
    size_t     bound,
    char const *fname,
    int        line)
{
    if (exc_state.abort->swap(1) == 0) {
        // first occurrence
        if (exc_state.handler != NULL)
            exc_state.handler->out_of_bounds(index, bound, fname, line);
    }

    // Note: use longjmp here to abort the current MDL function execution.
    // LLVM does not handle well exceptions on Win64 yet, so we have to live
    // with the small overhead
    longjmp(exc_state.env, 1);
}

// Handle (integer) division by zero.
void LLVM_code_generator::mdl_div_by_zero(
    Exc_state  &exc_state,
    char const *fname,
    int        line)
{
    if (exc_state.abort->swap(1) == 0) {
        // first occurrence
        if (exc_state.handler != NULL)
            exc_state.handler->div_by_zero(fname, line);
    }

    // Note: use longjmp here to abort the current MDL function execution.
    // LLVM does not handle well exceptions on Win64 yet, so we have to live
    // with the small overhead
    longjmp(exc_state.env, 1);
}

// Get the texture results pointer from the state.
llvm::Value *LLVM_code_generator::get_texture_results(Function_context &ctx)
{
    llvm::Function *func = m_runtime->get_internal_function(m_int_func_state_get_texture_results);
    llvm::Value *args[] = { ctx.get_state_parameter() };
    llvm::Value *text_results = m_runtime->call_rt_func(ctx, func, args);

    text_results = ctx->CreateBitCast(
        text_results, m_type_mapper.get_ptr(m_texture_results_struct_type));

    return text_results;
}

// Get the read-only data segment pointer from the state.
llvm::Value *LLVM_code_generator::get_ro_data_segment(Function_context &ctx)
{
    llvm::Function *func = m_runtime->get_internal_function(m_int_func_state_get_ro_data_segment);
    llvm::Value *args[] = { ctx.get_state_parameter() };
    return m_runtime->call_rt_func(ctx, func, args);
}

// Get the LLVM value of the current object_id value from the uniform state.
llvm::Value *LLVM_code_generator::get_current_object_id(Function_context &ctx)
{
    if (state_include_uniform_state()) {
        // get it from the state
        llvm::Function *func = m_runtime->get_internal_function(m_int_func_state_object_id);
        llvm::Value *args[] = { ctx.get_state_parameter() };
        return m_runtime->call_rt_func(ctx, func, args);
    }

    return ctx.get_constant(m_object_id);
}

/// Helper structure for linking the user state module with the current module.
struct Runtime_func_info
{
    /// Constructor.
    Runtime_func_info(
        llvm::Function *user_decl,
        llvm::Function *runtime_def,
        llvm::GlobalValue::LinkageTypes orig_linkage)
    : user_decl(user_decl)
    , runtime_def(runtime_def)
    , orig_linkage(orig_linkage)
    {}

    /// The declaration of the runtime function in the user state module.
    llvm::Function *user_decl;

    /// The definition of the runtime function in the current module.
    llvm::Function *runtime_def;

    /// The original linkage of the runtime function.
    llvm::GlobalValue::LinkageTypes orig_linkage;
};

// Initialize the current LLVM module with user-specified LLVM implementations.
bool LLVM_code_generator::init_user_modules()
{
    if (m_user_state_module.data == NULL) return true;

    llvm::MemoryBuffer *mem = llvm::MemoryBuffer::getMemBuffer(
        llvm::StringRef(m_user_state_module.data, m_user_state_module.size),
        "state_module",
        /*RequiresNullTerminator=*/false);
    llvm::Module *state_mod = llvm::ParseBitcodeFile(mem, m_llvm_context);
    if (state_mod == NULL) {
        error(PARSING_STATE_MODULE_FAILED, Error_params(get_allocator()));
        return false;
    }

    // make sure all required functions are provided
    if (!m_runtime->check_state_module(state_mod))
        return false;

    // create prototypes in the target module for all function definitions to force the linker to
    // correctly map the types
    llvm::SmallVector<llvm::Type *, 8> arg_types;
    for (llvm::Module::iterator FI = state_mod->begin(), FE = state_mod->end(); FI != FE; ++FI) {
        if (FI->isDeclaration())
            continue;  // not a function definition

        arg_types.clear();

        llvm::FunctionType const *func_type = FI->getFunctionType();
        for (llvm::FunctionType::param_iterator PI = func_type->param_begin(),
            PE = func_type->param_end(); PI != PE; ++PI)
        {
            if (llvm::isa<llvm::PointerType>(*PI)) {
                llvm::Type *elem_type = (*PI)->getPointerElementType();
                if (llvm::StructType *struct_type = llvm::dyn_cast<llvm::StructType>(elem_type)) {
                    // map some struct types to our internal types
                    llvm::StringRef name = elem_type->getStructName();
                    if (name.startswith("struct.State_core")) {
                        if (!struct_type->isOpaque()) {
                            error(API_STRUCT_TYPE_MUST_BE_OPAQUE, name);
                            return false;
                        }
                        arg_types.push_back(
                            m_type_mapper.get_state_ptr_type(Type_mapper::SSM_CORE));
                        continue;
                    }
                    if (name.startswith("struct.State_environment")) {
                        if (!struct_type->isOpaque()) {
                            error(API_STRUCT_TYPE_MUST_BE_OPAQUE, name);
                            return false;
                        }
                        arg_types.push_back(
                            m_type_mapper.get_state_ptr_type(Type_mapper::SSM_ENVIRONMENT));
                        continue;
                    }
                    if (name.startswith("struct.Exception_state")) {
                        if (!struct_type->isOpaque()) {
                            error(API_STRUCT_TYPE_MUST_BE_OPAQUE, name);
                            return false;
                        }
                        arg_types.push_back(m_type_mapper.get_exc_state_ptr_type());
                        continue;
                    }
                }
            }
            arg_types.push_back(*PI);
        }

        // create prototype for linking in the target module
        llvm::Function::Create(
            llvm::FunctionType::get(FI->getReturnType(), arg_types, false),
            llvm::GlobalValue::ExternalLinkage,
            FI->getName(),
            m_module);
    }

    // create the internal space variable
    new llvm::GlobalVariable(
        *m_module,
        m_type_mapper.get_int_type(),
        /*isConstant=*/true,
        llvm::GlobalValue::ExternalLinkage,
        llvm::ConstantInt::get(
            m_type_mapper.get_int_type(),
            strcmp(m_internal_space, "coordinate_object") == 0
                ? coordinate_object : coordinate_world),
        "INTERNAL_SPACE");

    llvm::SmallVector<Runtime_func_info, 16> runtime_infos;

    // resolve and collect all external runtime function references
    for (llvm::Module::iterator FI = state_mod->begin(), FE = state_mod->end(); FI != FE; ++FI) {
        if (!FI->isDeclaration())
            continue;  // not an external function

        llvm::StringRef func_name = FI->getName();
        if (!func_name.startswith("_Z"))
            continue;  // not a C++ mangled name

        string demangled_name(get_allocator());
        MDL_name_mangler mangler(get_allocator(), demangled_name);
        if (!mangler.demangle(func_name.data(), func_name.size())) {
            error(DEMANGLING_NAME_OF_EXTERNAL_FUNCTION_FAILED, func_name);
            continue;
        }

        // find last "::" before the parameters
        size_t parenpos = demangled_name.find('(');
        size_t colonpos = demangled_name.rfind("::", parenpos);
        if (colonpos == string::npos || colonpos == 0)
            continue;  // no module member

        string module_name = demangled_name.substr(0, colonpos);
        string signature = demangled_name.substr(colonpos + 2);
        IDefinition const *def = m_compiler->find_stdlib_signature(
            module_name.c_str(), signature.c_str());
        if (def == NULL)
            continue;  // not one of our modules

        llvm::Function *func = m_runtime->get_intrinsic_function(def, /*return_derivs=*/ false);
        if (func->getFunctionType() != FI->getFunctionType()) {
            error(WRONG_FUNCTYPE_FOR_MDL_RUNTIME_FUNCTION, demangled_name);
            continue;
        }

        runtime_infos.push_back(
            Runtime_func_info(&*FI, func, func->getLinkage()));
    }

    // create prototypes in the user state module for all used runtime functions and redirect
    // all runtime calls to them.
    // We have to this in a second step to not change the function iterator while iterating.
    for (size_t i = 0, n = runtime_infos.size(); i < n; ++i) {
        llvm::Function *decl_func = runtime_infos[i].user_decl;
        llvm::Function *def_func = runtime_infos[i].runtime_def;

        // temporarily set the linkage of the runtime function definition to external
        def_func->setLinkage(llvm::GlobalValue::ExternalLinkage);

        // if names match, no need for further processing, just set to external linkage
        if (decl_func->getName() == def_func->getName()) {
            decl_func->setLinkage(llvm::GlobalValue::ExternalLinkage);
            continue;
        }

        // create prototype in state_module for linking
        llvm::Function *prot = llvm::Function::Create(
            def_func->getFunctionType(),
            llvm::GlobalValue::ExternalLinkage,
            def_func->getName(),
            state_mod);

        llvm::SmallVector<llvm::Instruction *, 16> delete_list;

        // replace all calls to calls to our implementation
        for (llvm::Value::use_iterator ui = decl_func->use_begin(), ue = decl_func->use_end();
            ui != ue; ++ui)
        {
            if (llvm::CallInst *inst = llvm::dyn_cast<llvm::CallInst>(*ui)) {
                llvm::SmallVector<llvm::Value *, 8> args;
                for (unsigned i = 0, n = inst->getNumArgOperands(); i < n; ++i) {
                    args.push_back(inst->getArgOperand(i));
                }
                llvm::CallInst *new_call = llvm::CallInst::Create(prot, args, "", inst);
                inst->replaceAllUsesWith(new_call);
                delete_list.push_back(inst);
            }
        }

        // remove the original calls
        for (size_t i = 0, num = delete_list.size(); i < num; ++i) {
            delete_list[i]->eraseFromParent();
        }

        // remove the original prototype
        decl_func->eraseFromParent();
    }

    // clear target triple to avoid LLVM warning on console about mixing different targets
    // when linking the user module ("x86_x64-pc-win32") with libdevice ("nvptx-unknown-unknown").
    // Using an nvptx target for the user module  would cause struct parameters to be split, which
    // we try to avoid.
    state_mod->setTargetTriple("");

    std::string errorInfo;
    if (llvm::Linker::LinkModules(m_module, state_mod, llvm::Linker::DestroySource, &errorInfo))
    {
        // true means linking has failed
        error(LINKING_STATE_MODULE_FAILED, errorInfo);
        return false;
    }

    // restore the original linkage of the runtime functions
    for (size_t i = 0, n = runtime_infos.size(); i < n; ++i) {
        runtime_infos[i].runtime_def->setLinkage(runtime_infos[i].orig_linkage);
    }

    return true;
}

// Translate a JIT intrinsic call expression to LLVM IR.
Expression_result LLVM_code_generator::translate_jit_intrinsic(
    Function_context &ctx,
    ICall_expr const *call_expr)
{
    if (call_expr->get_semantics() == IDefinition::DS_INTRINSIC_JIT_LOOKUP) {
        IType const *ret_type = call_expr->get_type()->skip_type_alias();

        llvm::Function *func = NULL;
        if (is<IType_float>(ret_type))
            func = m_runtime->get_runtime_func(MDL_runtime_creator::RT_MDL_TEX_RES_FLOAT);
        else if (is<IType_color>(ret_type))
            func = m_runtime->get_runtime_func(MDL_runtime_creator::RT_MDL_TEX_RES_COLOR);
        else if (is<IType_vector>(ret_type))
            func = m_runtime->get_runtime_func(MDL_runtime_creator::RT_MDL_TEX_RES_FLOAT3);

        if (func != NULL) {
            // prepare arguments
            llvm::SmallVector<llvm::Value *, 3> args;

            llvm::Type *res_type = lookup_type(ret_type);

            llvm::Value *sret_res = NULL;
            if (need_reference_return(ret_type)) {
                // create a temporary for the function result and pass it
                sret_res = ctx.create_local(res_type, "call_result");
                args.push_back(sret_res);
            }

            llvm::Value *res = ctx.create_local(res_type, "result");

            args.push_back(ctx.get_state_parameter());

            llvm::Value *index = call_expr->translate_argument_value(
                *this, ctx, 0, /*return_derivs=*/ false);
            args.push_back(index);

            llvm::Value *max_index = ctx.get_constant(int(m_num_texture_results));

            llvm::BasicBlock *ok_bb   = ctx.create_bb("index_ok");
            llvm::BasicBlock *fail_bb = ctx.create_bb("index_fail");
            llvm::BasicBlock *end_bb  = ctx.create_bb("end");
            llvm::Value      *cmp     = ctx->CreateICmpULT(index, max_index);

            // set branch weights, we expect all bounds checks to be ok and never fail
            ctx.CreateWeightedCondBr(cmp, ok_bb, fail_bb, 1, 0);

            ctx->SetInsertPoint(fail_bb);
            {
                // return zero if out of bounds
                llvm::Value *null = llvm::Constant::getNullValue(res_type);
                ctx->CreateStore(null, res);
                ctx->CreateBr(end_bb);
            }
            ctx->SetInsertPoint(ok_bb);
            {
                llvm::Value *v = ctx->CreateCall(func, args);
                if (sret_res != NULL) {
                    // the result was passed on the stack
                    v = ctx->CreateLoad(sret_res);
                }
                ctx->CreateStore(v, res);
                ctx->CreateBr(end_bb);
            }
            ctx->SetInsertPoint(end_bb);
            return Expression_result::ptr(res);
        }
    }
    MDL_ASSERT(!"Unsupported MDL JIT function");
    return Expression_result::undef(lookup_type(call_expr->get_type()));
}

// Get the intrinsic LLVm function for a MDL function.
llvm::Function *LLVM_code_generator::get_intrinsic_function(
    IDefinition const *def, bool return_derivs)
{
    return m_runtime->get_intrinsic_function(def, return_derivs);
}

// Get the LLVM function for an internal function.
llvm::Function *LLVM_code_generator::get_internal_function(Internal_function const *int_func)
{
    return m_runtime->get_internal_function(int_func);
}

// Call a void runtime function.
void LLVM_code_generator::call_rt_func_void(
    Function_context              &ctx,
    llvm::Function                *callee,
    llvm::ArrayRef<llvm::Value *> args)
{
    m_runtime->call_rt_func_void(ctx, callee, args);
}

// Call a runtime function.
llvm::Value *LLVM_code_generator::call_rt_func(
    Function_context              &ctx,
    llvm::Function                *callee,
    llvm::ArrayRef<llvm::Value *> args)
{
    return m_runtime->call_rt_func(ctx, callee, args);
}

} // mdl
} // mi
