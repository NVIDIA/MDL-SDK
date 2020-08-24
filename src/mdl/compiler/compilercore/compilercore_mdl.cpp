/******************************************************************************
 * Copyright (c) 2011-2020, NVIDIA CORPORATION. All rights reserved.
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

#include <cfloat>
#include <cstring>

#include <mi/base/lock.h>

#include <mi/mdl/mdl_translator_plugin.h>

#include "compilercore_cc_conf.h"
#include "compilercore_mdl.h"
#include "compilercore_allocator.h"
#include "compilercore_analysis.h"
#include "compilercore_debug_tools.h"
#include "compilercore_encapsulator.h"
#include "compilercore_factories.h"
#include "compilercore_malloc_allocator.h"
#include "compilercore_modules.h"
#include "compilercore_options.h"
#include "compilercore_file_resolution.h"
#include "compilercore_printers.h"
#include "compilercore_wchar_support.h"
#include "compilercore_streams.h"
#include "compilercore_printers.h"
#include "compilercore_messages.h"
#include "compilercore_errors.h"
#include "compilercore_builder.h"
#include "compilercore_serializer.h"
#include "compilercore_tools.h"
#include "compilercore_archiver.h"
#include "compilercore_comparator.h"
#include "compilercore_module_transformer.h"
#include "compilercore_mdl.h"

#include "mdl_module.h"

#include <mi/mdl/mdl_code_generators.h>

#include <mdl/codegenerators/generator_dag/generator_dag.h>
#include <mdl/codegenerators/generator_dag/generator_dag_tools.h>

#include "Scanner.h"
#include "Parser.h"

namespace mi {
namespace mdl {

char const *MDL::option_dump_dependence_graph         = MDL_OPTION_DUMP_DEPENDENCE_GRAPH;
char const *MDL::option_dump_call_graph               = MDL_OPTION_DUMP_CALL_GRAPH;
char const *MDL::option_warn                          = MDL_OPTION_WARN;
char const *MDL::option_opt_level                     = MDL_OPTION_OPT_LEVEL;
char const *MDL::option_strict                        = MDL_OPTION_STRICT;
char const *MDL::option_experimental_features         = MDL_OPTION_EXPERIMENTAL_FEATURES;
char const *MDL::option_resolve_resources             = MDL_OPTION_RESOLVE_RESOURCES;
char const *MDL::option_limits_float_min              = MDL_OPTION_LIMITS_FLOAT_MIN;
char const *MDL::option_limits_float_max              = MDL_OPTION_LIMITS_FLOAT_MAX;
char const *MDL::option_limits_double_min             = MDL_OPTION_LIMITS_DOUBLE_MIN;
char const *MDL::option_limits_double_max             = MDL_OPTION_LIMITS_DOUBLE_MAX;
char const *MDL::option_state_wavelength_base_max     = MDL_OPTION_STATE_WAVELENGTH_BASE_MAX;
char const *MDL::option_keep_original_resource_file_paths
                                                  = MDL_OPTION_KEEP_ORIGINAL_RESOURCE_FILE_PATHS;

// forward
class Jitted_code;

extern ICode_generator *create_code_generator_dag(IAllocator *alloc, MDL *mdl);
extern void serialize_code_dag(
    IGenerated_code_dag const *code,
    ISerializer               *is,
    MDL_binary_serializer     &bin_serializer);
extern IGenerated_code_dag const *deserialize_code_dag(
    IDeserializer           *ds,
    MDL_binary_deserializer &bin_deserializer,
    MDL                     *compiler);

extern ICode_generator *create_code_generator_jit(IAllocator *alloc, MDL *mdl);
extern ICode_generator *create_code_generator_glsl(IAllocator *alloc, MDL *mdl);
extern Jitted_code     *create_jitted_code_singleton(IAllocator *alloc);
extern void            terminate_jitted_code_singleton(Jitted_code *jitted_code);

///
/// Handle syntax errors for CoCo/R generated parser.
///
class Syntax_error : public Errors {
public:
    // Syntax error produces compiler messages.
    static char const MESSAGE_CLASS = 'C';

    /// Report a syntax error at given line, column pair.
    ///
    /// \param la      the current look-ahead token
    /// \param s       the human readable error message
    void Error(Token const *la, wchar_t const *s) MDL_FINAL;

    /// Report a syntax warning at given line, column pair.
    ///
    /// \param line  the start line of the syntax error
    /// \param col   the start column of the syntax error
    /// \param s     the human readable error message
    void Warning(int line, int col, wchar_t const *s) MDL_FINAL;

    /// Report an error at given line, column pair.
    ///
    /// \param line    the start line of the syntax error
    /// \param col     the start column of the syntax error
    /// \param code    the error code
    /// \param params  additional error parameter/inserts
    void Error(int line, int col, int code, Error_params const &params) MDL_FINAL;

    /// Constructor.
    ///
    /// \param alloc  the used allocator
    /// \param msg    the message list new messages will be added to
    explicit Syntax_error(IAllocator *alloc, Messages_impl &msg)
    : Errors()
    , m_builder(alloc)
    , m_string_buf(m_builder.create<Buffer_output_stream>(m_builder.get_allocator()))
    , m_printer(m_builder.create<Printer>(m_builder.get_allocator(), m_string_buf.get()))
    , m_msg(msg)
    {
    }

private:
    /// Check if the given token kind is a reserved keyword.
    bool is_reserved_keyword(int t_kind);

private:
    /// The builder.
    Allocator_builder m_builder;

    /// A string buffer used for error messages.
    mi::base::Handle<Buffer_output_stream> m_string_buf;

    /// Printer for error messages.
    mi::base::Handle<IPrinter> m_printer;

    /// The message list errors/warnings will be added.
    Messages_impl &m_msg;
};

void Syntax_error::Error(Token const *t, wchar_t const *s)
{
    string tmp(m_builder.get_allocator());
    Position_impl pos(t->line, t->col, t->line, t->col + wcslen(t->val) - 1);
    if (is_reserved_keyword(t->kind)) {
        string msg("reserved keyword '", m_builder.get_allocator());
        msg += wchar_to_utf8(tmp, t->val);
        msg += "' used";
        m_msg.add_error_message(SYNTAX_ERROR, MESSAGE_CLASS, 0, &pos, msg.c_str());
    } else {
        m_msg.add_error_message(SYNTAX_ERROR, MESSAGE_CLASS, 0, &pos, wchar_to_utf8(tmp, s));
    }
}

void Syntax_error::Warning(int line, int col, wchar_t const *s)
{
    string tmp(m_builder.get_allocator());
    Position_impl pos(line, col, line, col);
    m_msg.add_warning_message(SYNTAX_ERROR, MESSAGE_CLASS, 0, &pos, wchar_to_utf8(tmp, s));
}

void Syntax_error::Error(int line, int col, int code, Error_params const &params)
{
    Position_impl pos(line, col, line, col);

    m_string_buf->clear();

    print_error_message(code, MESSAGE_CLASS, params, m_printer.get());
    m_msg.add_error_message(code, MESSAGE_CLASS, 0, &pos, m_string_buf->get_data());
}

// Check if the given token kind is a reserved keyword.
bool Syntax_error::is_reserved_keyword(int t_kind)
{
    return t_kind == Parser::_R_RESERVED;
}

void Scanner::initialize_mdl_keywords()
{
    // The CoCo/R grammar contains always ALL MDL keywords, so
    // first map all non MDL 1.0 keywords to IDENT and enable them later.
    keywords.set(L"bsdf_measurement", Parser::_IDENT);
    keywords.set(L"intensity_mode", Parser::_IDENT);
    keywords.set(L"intensity_radiant_exitance", Parser::_IDENT);
    keywords.set(L"intensity_power", Parser::_IDENT);
    keywords.set(L"cast", Parser::_IDENT);

    // the "reserved" identifier is not really reserved
    keywords.set(L"reserved", Parser::_IDENT);

    // set the reserved MDL 1.0 keywords
    keywords.set(L"auto",             Parser::_R_RESERVED);
    keywords.set(L"catch",            Parser::_R_RESERVED);
    keywords.set(L"char",             Parser::_R_RESERVED);
    keywords.set(L"class",            Parser::_R_RESERVED);
    keywords.set(L"const_cast",       Parser::_R_RESERVED);
    keywords.set(L"delete",           Parser::_R_RESERVED);
    keywords.set(L"dynamic_cast",     Parser::_R_RESERVED);
    keywords.set(L"explicit",         Parser::_R_RESERVED);
    keywords.set(L"extern",           Parser::_R_RESERVED);
    keywords.set(L"foreach",          Parser::_R_RESERVED);
    keywords.set(L"friend",           Parser::_R_RESERVED);
    keywords.set(L"goto",             Parser::_R_RESERVED);
    keywords.set(L"graph",            Parser::_R_RESERVED);
    keywords.set(L"bool2x2",          Parser::_R_RESERVED);
    keywords.set(L"bool2x3",          Parser::_R_RESERVED);
    keywords.set(L"bool2x4",          Parser::_R_RESERVED);
    keywords.set(L"bool3x2",          Parser::_R_RESERVED);
    keywords.set(L"bool3x3",          Parser::_R_RESERVED);
    keywords.set(L"bool3x4",          Parser::_R_RESERVED);
    keywords.set(L"bool4x2",          Parser::_R_RESERVED);
    keywords.set(L"bool4x3",          Parser::_R_RESERVED);
    keywords.set(L"bool4x4",          Parser::_R_RESERVED);
    keywords.set(L"int2x2",           Parser::_R_RESERVED);
    keywords.set(L"int2x3",           Parser::_R_RESERVED);
    keywords.set(L"int2x4",           Parser::_R_RESERVED);
    keywords.set(L"int3x2",           Parser::_R_RESERVED);
    keywords.set(L"int3x3",           Parser::_R_RESERVED);
    keywords.set(L"int3x4",           Parser::_R_RESERVED);
    keywords.set(L"int4x2",           Parser::_R_RESERVED);
    keywords.set(L"int4x3",           Parser::_R_RESERVED);
    keywords.set(L"int4x4",           Parser::_R_RESERVED);
    keywords.set(L"half",             Parser::_R_RESERVED);
    keywords.set(L"half2",            Parser::_R_RESERVED);
    keywords.set(L"half2x2",          Parser::_R_RESERVED);
    keywords.set(L"half2x3",          Parser::_R_RESERVED);
    keywords.set(L"half2x4",          Parser::_R_RESERVED);
    keywords.set(L"half3",            Parser::_R_RESERVED);
    keywords.set(L"half3x2",          Parser::_R_RESERVED);
    keywords.set(L"half3x3",          Parser::_R_RESERVED);
    keywords.set(L"half3x4",          Parser::_R_RESERVED);
    keywords.set(L"half4",            Parser::_R_RESERVED);
    keywords.set(L"half4x2",          Parser::_R_RESERVED);
    keywords.set(L"half4x3",          Parser::_R_RESERVED);
    keywords.set(L"half4x4",          Parser::_R_RESERVED);
    keywords.set(L"inline",           Parser::_R_RESERVED);
    keywords.set(L"inout",            Parser::_R_RESERVED);
    keywords.set(L"long",             Parser::_R_RESERVED);
    keywords.set(L"module",           Parser::_R_RESERVED);
    keywords.set(L"mutable",          Parser::_R_RESERVED);
    keywords.set(L"namespace",        Parser::_R_RESERVED);
    keywords.set(L"native",           Parser::_R_RESERVED);
    keywords.set(L"new",              Parser::_R_RESERVED);
    keywords.set(L"operator",         Parser::_R_RESERVED);
    keywords.set(L"out",              Parser::_R_RESERVED);
    keywords.set(L"phenomenon",       Parser::_R_RESERVED);
    keywords.set(L"private",          Parser::_R_RESERVED);
    keywords.set(L"protected",        Parser::_R_RESERVED);
    keywords.set(L"public",           Parser::_R_RESERVED);
    keywords.set(L"reinterpret_cast", Parser::_R_RESERVED);
    keywords.set(L"sampler",          Parser::_R_RESERVED);
    keywords.set(L"shader",           Parser::_R_RESERVED);
    keywords.set(L"short",            Parser::_R_RESERVED);
    keywords.set(L"signed",           Parser::_R_RESERVED);
    keywords.set(L"sizeof",           Parser::_R_RESERVED);
    keywords.set(L"static",           Parser::_R_RESERVED);
    keywords.set(L"static_cast",      Parser::_R_RESERVED);
    keywords.set(L"technique",        Parser::_R_RESERVED);
    keywords.set(L"template",         Parser::_R_RESERVED);
    keywords.set(L"this",             Parser::_R_RESERVED);
    keywords.set(L"throw",            Parser::_R_RESERVED);
    keywords.set(L"try",              Parser::_R_RESERVED);
    keywords.set(L"typeid",           Parser::_R_RESERVED);
    keywords.set(L"typename",         Parser::_R_RESERVED);
    keywords.set(L"union",            Parser::_R_RESERVED);
    keywords.set(L"unsigned",         Parser::_R_RESERVED);
    keywords.set(L"virtual",          Parser::_R_RESERVED);
    keywords.set(L"void",             Parser::_R_RESERVED);
    keywords.set(L"volatile",         Parser::_R_RESERVED);
    keywords.set(L"wchar_t",          Parser::_R_RESERVED);
}

void Scanner::enable_native_keyword(bool flag)
{
    keywords.set(L"native", flag ? Parser::_IDENT : Parser::_R_RESERVED);
}

void Scanner::set_mdl_version(int major, int minor)
{
#define HAS_VERSION(x, y) (major > (x) || (major == (x) && minor >= (y)))

    if (HAS_VERSION(1, 1)) {
        // enable MDL 1.1 keywords
        keywords.set(L"bsdf_measurement", Parser::_BSDF_MEASUREMENT);
        keywords.set(L"intensity_mode", Parser::_INTENSITY_MODE);
        keywords.set(L"intensity_radiant_exitance", Parser::_INTENSITY_RADIANT_EXITANCE);
        keywords.set(L"intensity_power", Parser::_INTENSITY_POWER);
    }
    if (HAS_VERSION(1, 3)) {
        // enable MDL 1.3 keywords
        keywords.set(L"module", Parser::_MODULE);
    }
    if (HAS_VERSION(1, 5)) {
        // enable MDL 1.5 keywords
        keywords.set(L"cast", Parser::_CAST);
        keywords.set(L"hair_bsdf", Parser::_HAIR_BSDF);
    }
}

namespace {

/// Helper class to avoid NULL checks.
class Empty_search_path MDL_FINAL : public Allocator_interface_implement<IMDL_search_path> {
    typedef Allocator_interface_implement<IMDL_search_path> Base;
public:
    /// Get the number of search paths.
    ///
    /// \param set  the path set
    size_t get_search_path_count(Path_set set) const MDL_FINAL { return 0; }

    /// Get the i'th search path.
    ///
    /// \param set  the path set
    /// \param i    index of the path
    char const *get_search_path(Path_set set, size_t i) const MDL_FINAL { return NULL; }

    /// Constructor.
    Empty_search_path(IAllocator *alloc)
    : Base(alloc)
    {
    }
};

#ifdef DEBUG

dbg::DebugMallocAllocator dbgMallocAlloc;

#endif  // DEBUG

/// Creates a new MDL compiler-
static mi::mdl::MDL *create_mdl(IAllocator *alloc)
{
    Allocator_builder builder(alloc);

    mi::mdl::MDL *p = builder.create<mi::mdl::MDL>(alloc);
    return p;
}

}  // anonymous


MDL::MDL(IAllocator *alloc)
: Base(alloc)
, m_builder(alloc)
, m_next_module_id(0)
, m_arena(alloc)
, m_type_factory(m_arena, NULL, NULL)
, m_options(alloc)
, m_builtin_module_indexes(0, Module_map::hasher(), Module_map::key_equal(), alloc)
, m_builtin_modules(alloc)
, m_builtin_semantics(0, Sema_map::hasher(), Sema_map::key_equal(), alloc)
, m_search_path(m_builder.create<Empty_search_path>(alloc))
, m_external_resolver()
, m_global_lock()
, m_search_path_lock()
, m_weak_module_lock()
, m_predefined_types_build(false)
, m_jitted_code(NULL)
, m_translator_list(alloc)
{
    create_options();
    create_builtin_semantics();

    // create built-in modules
    mi::base::Handle<Thread_context> ctx(create_thread_context());

    // load state.mdl,
    // must be first due to dependencies of material structs to state::normal
    {
        mi::base::Handle<Buffer_Input_stream> s(m_builder.create<Encoded_buffer_Input_stream>(
            m_builder.get_allocator(),
            mdl_module_state, sizeof(mdl_module_state), ""));
        Module *state_mod = load_module(
            NULL, ctx.get(), "::state", s.get(), Module::MF_IS_STDLIB);

        // takes ownership
        register_builtin_module(state_mod);
    }

    // load tex.mdl next, this defines the gamma_mode enum
    {
        mi::base::Handle<Buffer_Input_stream> s(m_builder.create<Encoded_buffer_Input_stream>(
            m_builder.get_allocator(),
            mdl_module_tex, sizeof(mdl_module_tex), ""));
        Module *tex_mod = load_module(
            NULL, ctx.get(), "::tex", s.get(), Module::MF_IS_STDLIB);

        // takes ownership
        register_builtin_module(tex_mod);
    }

    // load limits.mdl
    {
        mi::base::Handle<Buffer_Input_stream> s(m_builder.create<Encoded_buffer_Input_stream>(
            m_builder.get_allocator(),
            mdl_module_limits, sizeof(mdl_module_limits), ""));
        Module *limits_mod = load_module(
            NULL, ctx.get(), "::limits", s.get(), Module::MF_IS_STDLIB);

        // takes ownership
        register_builtin_module(limits_mod);
    }

    // load anno.mdl
    {
        mi::base::Handle<Buffer_Input_stream> s(m_builder.create<Encoded_buffer_Input_stream>(
            m_builder.get_allocator(),
            mdl_module_anno, sizeof(mdl_module_anno), ""));
        Module *anno_mod = load_module(
            NULL, ctx.get(), "::anno", s.get(), Module::MF_IS_STDLIB);

        // takes ownership
        register_builtin_module(anno_mod);
    }

    // load math.mdl
    {
        mi::base::Handle<Buffer_Input_stream> s(m_builder.create<Encoded_buffer_Input_stream>(
            m_builder.get_allocator(),
            mdl_module_math, sizeof(mdl_module_math), ""));
        Module *math_mod = load_module(
            NULL, ctx.get(), "::math", s.get(), Module::MF_IS_STDLIB);

        // takes ownership
        register_builtin_module(math_mod);
    }

    // load noise.mdl
    {
        mi::base::Handle<Buffer_Input_stream> s(m_builder.create<Encoded_buffer_Input_stream>(
            m_builder.get_allocator(),
            mdl_module_noise, sizeof(mdl_module_noise), ""));
        Module *noise_mod = load_module(
            NULL, ctx.get(), "::noise", s.get(), Module::MF_IS_STDLIB);

        // takes ownership
        register_builtin_module(noise_mod);
    }

    // load df.mdl
    {
        mi::base::Handle<Buffer_Input_stream> s(m_builder.create<Encoded_buffer_Input_stream>(
            m_builder.get_allocator(),
            mdl_module_df, sizeof(mdl_module_df), ""));
        Module *df_mod = load_module(
            NULL, ctx.get(), "::df", s.get(), Module::MF_IS_STDLIB);

        // takes ownership
        register_builtin_module(df_mod);
    }

    // load scene.mdl
    {
        mi::base::Handle<Buffer_Input_stream> s(m_builder.create<Encoded_buffer_Input_stream>(
            m_builder.get_allocator(),
            mdl_module_scene, sizeof(mdl_module_scene), ""));
        Module *scene_mod = load_module(
            NULL, ctx.get(), "::scene", s.get(), Module::MF_IS_STDLIB);

        // takes ownership
        register_builtin_module(scene_mod);
    }

    // load debug.mdl
    {
        mi::base::Handle<Buffer_Input_stream> s(m_builder.create<Encoded_buffer_Input_stream>(
            m_builder.get_allocator(),
            mdl_module_debug, sizeof(mdl_module_debug), ""));
        Module *debug_mod = load_module(
            NULL, ctx.get(), "::debug", s.get(), Module::MF_IS_STDLIB | Module::MF_IS_DEBUG);

        // takes ownership
        register_builtin_module(debug_mod);
    }

    // load std.mdl after all the above
    {
        mi::base::Handle<Buffer_Input_stream> s(m_builder.create<Encoded_buffer_Input_stream>(
            m_builder.get_allocator(),
            mdl_module_std, sizeof(mdl_module_std), ""));
        Module *std_mod = load_module(
            NULL, ctx.get(), "::std", s.get(), Module::MF_IS_STDLIB);

        // takes ownership
        register_builtin_module(std_mod);
    }
    // finally load builtins.mdl
    {
        mi::base::Handle<Buffer_Input_stream> s(m_builder.create<Encoded_buffer_Input_stream>(
            m_builder.get_allocator(),
            mdl_module_builtins, sizeof(mdl_module_builtins), ""));
        Module *builtins_mod = load_module(
            NULL, ctx.get(), "::<builtins>", s.get(),
            Module::MF_IS_STDLIB | Module::MF_IS_BUILTIN);

        // takes ownership
        register_builtin_module(builtins_mod);
    }

    // currently load base.mdl, this must be hashed
    {
        mi::base::Handle<Buffer_Input_stream> s(m_builder.create<Encoded_buffer_Input_stream>(
            m_builder.get_allocator(),
            mdl_module_base, sizeof(mdl_module_base), ""));
        Module *base_mod = load_module(
            NULL, ctx.get(), "::base", s.get(), Module::MF_IS_OWNED | Module::MF_IS_HASHED);

        // takes ownership
        register_builtin_module(base_mod);
    }

}

// Destructor.
MDL::~MDL()
{
    terminate_jitted_code_singleton(m_jitted_code);
}

// Get the type factory.
Type_factory *MDL::get_type_factory() const
{
    return &m_type_factory;
}

// Create a module, pass its file name.
Module *MDL::create_module(
    char const        *module_name,
    char const        *file_name,
    IMDL::MDL_version version,
    unsigned          flags)
{
    // FIXME: check for name already in use

    // TODO: this must be thread safe if create_module should be thread safe
    // Don't use number 0, this is reserved for "owner module".
    size_t id = ++m_next_module_id;

    if (file_name == NULL)
        file_name = "";

    return m_builder.create<Module>(
        m_builder.get_allocator(), this, id,
        module_name, file_name,
        version, flags);
}

// Create a module.
Module *MDL::create_module(
    IThread_context   *context,
    char const        *module_name,
    IMDL::MDL_version version)
{
    Thread_context *ctx = impl_cast<Thread_context>(context);

    if (ctx != NULL) {
        // clear message list
        ctx->clear_messages();
    }

    Module *res = create_module(module_name, "", version, Module::MF_STANDARD);

    // copy the messages from the module to the context, so they are available over both
    // access paths
    if (ctx != NULL)
        copy_message(ctx->access_messages_impl(), res);
    return res;
}

// Install a module resolver.
void MDL::install_search_path(IMDL_search_path *search_path)
{
    if (search_path == NULL)
        search_path = m_builder.create<Empty_search_path>(get_allocator());
    m_search_path = search_path;
}

// Register built-in modules at a module cache.
void MDL::register_builtin_module_at_cache(IModule_cache *cache)
{
    if (cache != NULL) {
        mi::base::Lock::Block block(&m_global_lock);

        if (IModule_loaded_callback *callback = cache->get_module_loading_callback()) {
            for (size_t i = 0, n = m_builtin_modules.size(); i < n; ++i) {
                if (!callback->is_builtin_module_registered(m_builtin_modules[i]->get_name())) {
                    callback->register_module(m_builtin_modules[i].get());
                }
            }
        }
    }
}

// Create all builtin semantics.
void MDL::create_builtin_semantics()
{
    // math module
    m_builtin_semantics["::math::abs"] =
        IDefinition::DS_INTRINSIC_MATH_ABS;
    m_builtin_semantics["::math::acos"] =
        IDefinition::DS_INTRINSIC_MATH_ACOS;
    m_builtin_semantics["::math::all"] =
        IDefinition::DS_INTRINSIC_MATH_ALL;
    m_builtin_semantics["::math::any"] =
        IDefinition::DS_INTRINSIC_MATH_ANY;
    m_builtin_semantics["::math::asin"] =
        IDefinition::DS_INTRINSIC_MATH_ASIN;
    m_builtin_semantics["::math::atan"] =
        IDefinition::DS_INTRINSIC_MATH_ATAN;
    m_builtin_semantics["::math::atan2"] =
        IDefinition::DS_INTRINSIC_MATH_ATAN2;
    m_builtin_semantics["::math::average"] =
        IDefinition::DS_INTRINSIC_MATH_AVERAGE;
    m_builtin_semantics["::math::ceil"] =
        IDefinition::DS_INTRINSIC_MATH_CEIL;
    m_builtin_semantics["::math::clamp"] =
        IDefinition::DS_INTRINSIC_MATH_CLAMP;
    m_builtin_semantics["::math::cos"] =
        IDefinition::DS_INTRINSIC_MATH_COS;
    m_builtin_semantics["::math::cross"] =
        IDefinition::DS_INTRINSIC_MATH_CROSS;
    m_builtin_semantics["::math::degrees"] =
        IDefinition::DS_INTRINSIC_MATH_DEGREES;
    m_builtin_semantics["::math::distance"] =
        IDefinition::DS_INTRINSIC_MATH_DISTANCE;
    m_builtin_semantics["::math::dot"] =
        IDefinition::DS_INTRINSIC_MATH_DOT;
    m_builtin_semantics["::math::eval_at_wavelength"] =
        IDefinition::DS_INTRINSIC_MATH_EVAL_AT_WAVELENGTH;
    m_builtin_semantics["::math::exp"] =
        IDefinition::DS_INTRINSIC_MATH_EXP;
    m_builtin_semantics["::math::exp2"] =
        IDefinition::DS_INTRINSIC_MATH_EXP2;
    m_builtin_semantics["::math::floor"] =
        IDefinition::DS_INTRINSIC_MATH_FLOOR;
    m_builtin_semantics["::math::fmod"] =
        IDefinition::DS_INTRINSIC_MATH_FMOD;
    m_builtin_semantics["::math::frac"] =
        IDefinition::DS_INTRINSIC_MATH_FRAC;
    m_builtin_semantics["::math::isnan"] =
        IDefinition::DS_INTRINSIC_MATH_ISNAN;
    m_builtin_semantics["::math::isfinite"] =
        IDefinition::DS_INTRINSIC_MATH_ISFINITE;
    m_builtin_semantics["::math::length"] =
        IDefinition::DS_INTRINSIC_MATH_LENGTH;
    m_builtin_semantics["::math::lerp"] =
        IDefinition::DS_INTRINSIC_MATH_LERP;
    m_builtin_semantics["::math::log"] =
        IDefinition::DS_INTRINSIC_MATH_LOG;
    m_builtin_semantics["::math::log2"] =
        IDefinition::DS_INTRINSIC_MATH_LOG2;
    m_builtin_semantics["::math::log10"] =
        IDefinition::DS_INTRINSIC_MATH_LOG10;
    m_builtin_semantics["::math::luminance"] =
        IDefinition::DS_INTRINSIC_MATH_LUMINANCE;
    m_builtin_semantics["::math::max"] =
        IDefinition::DS_INTRINSIC_MATH_MAX;
    m_builtin_semantics["::math::max_value"] =
        IDefinition::DS_INTRINSIC_MATH_MAX_VALUE;
    m_builtin_semantics["::math::max_value_wavelength"] =
        IDefinition::DS_INTRINSIC_MATH_MAX_VALUE_WAVELENGTH;
    m_builtin_semantics["::math::min"] =
        IDefinition::DS_INTRINSIC_MATH_MIN;
    m_builtin_semantics["::math::min_value"] =
        IDefinition::DS_INTRINSIC_MATH_MIN_VALUE;
    m_builtin_semantics["::math::min_value_wavelength"] =
        IDefinition::DS_INTRINSIC_MATH_MIN_VALUE_WAVELENGTH;
    m_builtin_semantics["::math::modf"] =
        IDefinition::DS_INTRINSIC_MATH_MODF;
    m_builtin_semantics["::math::normalize"] =
        IDefinition::DS_INTRINSIC_MATH_NORMALIZE;
    m_builtin_semantics["::math::pow"] =
        IDefinition::DS_INTRINSIC_MATH_POW;
    m_builtin_semantics["::math::radians"] =
        IDefinition::DS_INTRINSIC_MATH_RADIANS;
    m_builtin_semantics["::math::round"] =
        IDefinition::DS_INTRINSIC_MATH_ROUND;
    m_builtin_semantics["::math::rsqrt"] =
        IDefinition::DS_INTRINSIC_MATH_RSQRT;
    m_builtin_semantics["::math::saturate"] =
        IDefinition::DS_INTRINSIC_MATH_SATURATE;
    m_builtin_semantics["::math::sign"] =
        IDefinition::DS_INTRINSIC_MATH_SIGN;
    m_builtin_semantics["::math::sin"] =
        IDefinition::DS_INTRINSIC_MATH_SIN;
    m_builtin_semantics["::math::sincos"] =
        IDefinition::DS_INTRINSIC_MATH_SINCOS;
    m_builtin_semantics["::math::smoothstep"] =
        IDefinition::DS_INTRINSIC_MATH_SMOOTHSTEP;
    m_builtin_semantics["::math::sqrt"] =
        IDefinition::DS_INTRINSIC_MATH_SQRT;
    m_builtin_semantics["::math::step"] =
        IDefinition::DS_INTRINSIC_MATH_STEP;
    m_builtin_semantics["::math::tan"] =
        IDefinition::DS_INTRINSIC_MATH_TAN;
    m_builtin_semantics["::math::transpose"] =
        IDefinition::DS_INTRINSIC_MATH_TRANSPOSE;
    m_builtin_semantics["::math::blackbody"] =
        IDefinition::DS_INTRINSIC_MATH_BLACKBODY;
    m_builtin_semantics["::math::emission_color"] =
        IDefinition::DS_INTRINSIC_MATH_EMISSION_COLOR;
    m_builtin_semantics["::math::DX"] =
        IDefinition::DS_INTRINSIC_MATH_DX;
    m_builtin_semantics["::math::DY"] =
        IDefinition::DS_INTRINSIC_MATH_DY;

    // state module
    m_builtin_semantics["::state::position"] =
        IDefinition::DS_INTRINSIC_STATE_POSITION;
    m_builtin_semantics["::state::normal"] =
        IDefinition::DS_INTRINSIC_STATE_NORMAL;
    m_builtin_semantics["::state::geometry_normal"] =
        IDefinition::DS_INTRINSIC_STATE_GEOMETRY_NORMAL;
    m_builtin_semantics["::state::motion"] =
        IDefinition::DS_INTRINSIC_STATE_MOTION;
    m_builtin_semantics["::state::texture_space_max"] =
        IDefinition::DS_INTRINSIC_STATE_TEXTURE_SPACE_MAX;
    m_builtin_semantics["::state::texture_coordinate"] =
        IDefinition::DS_INTRINSIC_STATE_TEXTURE_COORDINATE;
    m_builtin_semantics["::state::texture_tangent_u"] =
        IDefinition::DS_INTRINSIC_STATE_TEXTURE_TANGENT_U;
    m_builtin_semantics["::state::texture_tangent_v"] =
        IDefinition::DS_INTRINSIC_STATE_TEXTURE_TANGENT_V;
    m_builtin_semantics["::state::tangent_space"] =
        IDefinition::DS_INTRINSIC_STATE_TANGENT_SPACE;
    m_builtin_semantics["::state::geometry_tangent_u"] =
        IDefinition::DS_INTRINSIC_STATE_GEOMETRY_TANGENT_U;
    m_builtin_semantics["::state::geometry_tangent_v"] =
        IDefinition::DS_INTRINSIC_STATE_GEOMETRY_TANGENT_V;
    m_builtin_semantics["::state::direction"] =
        IDefinition::DS_INTRINSIC_STATE_DIRECTION;
    m_builtin_semantics["::state::animation_time"] =
        IDefinition::DS_INTRINSIC_STATE_ANIMATION_TIME;
    m_builtin_semantics["::state::wavelength_base"] =
        IDefinition::DS_INTRINSIC_STATE_WAVELENGTH_BASE;
    m_builtin_semantics["::state::transform"] =
        IDefinition::DS_INTRINSIC_STATE_TRANSFORM;
    m_builtin_semantics["::state::transform_point"] =
        IDefinition::DS_INTRINSIC_STATE_TRANSFORM_POINT;
    m_builtin_semantics["::state::transform_vector"] =
        IDefinition::DS_INTRINSIC_STATE_TRANSFORM_VECTOR;
    m_builtin_semantics["::state::transform_normal"] =
        IDefinition::DS_INTRINSIC_STATE_TRANSFORM_NORMAL;
    m_builtin_semantics["::state::transform_scale"] =
        IDefinition::DS_INTRINSIC_STATE_TRANSFORM_SCALE;
    m_builtin_semantics["::state::rounded_corner_normal"] =
        IDefinition::DS_INTRINSIC_STATE_ROUNDED_CORNER_NORMAL;
    m_builtin_semantics["::state::meters_per_scene_unit"] =
        IDefinition::DS_INTRINSIC_STATE_METERS_PER_SCENE_UNIT;
    m_builtin_semantics["::state::scene_units_per_meter"] =
        IDefinition::DS_INTRINSIC_STATE_SCENE_UNITS_PER_METER;
    m_builtin_semantics["::state::object_id"] =
        IDefinition::DS_INTRINSIC_STATE_OBJECT_ID;
    m_builtin_semantics["::state::wavelength_min"] =
        IDefinition::DS_INTRINSIC_STATE_WAVELENGTH_MIN;
    m_builtin_semantics["::state::wavelength_max"] =
        IDefinition::DS_INTRINSIC_STATE_WAVELENGTH_MAX;

    // tex module
    m_builtin_semantics["::tex::width"] =
        IDefinition::DS_INTRINSIC_TEX_WIDTH;
    m_builtin_semantics["::tex::height"] =
        IDefinition::DS_INTRINSIC_TEX_HEIGHT;
    m_builtin_semantics["::tex::depth"] =
        IDefinition::DS_INTRINSIC_TEX_DEPTH;
    m_builtin_semantics["::tex::lookup_float"] =
        IDefinition::DS_INTRINSIC_TEX_LOOKUP_FLOAT;
    m_builtin_semantics["::tex::lookup_float2"] =
        IDefinition::DS_INTRINSIC_TEX_LOOKUP_FLOAT2;
    m_builtin_semantics["::tex::lookup_float3"] =
        IDefinition::DS_INTRINSIC_TEX_LOOKUP_FLOAT3;
    m_builtin_semantics["::tex::lookup_float4"] =
        IDefinition::DS_INTRINSIC_TEX_LOOKUP_FLOAT4;
    m_builtin_semantics["::tex::lookup_color"] =
        IDefinition::DS_INTRINSIC_TEX_LOOKUP_COLOR;
    m_builtin_semantics["::tex::texel_float"] =
        IDefinition::DS_INTRINSIC_TEX_TEXEL_FLOAT;
    m_builtin_semantics["::tex::texel_float2"] =
        IDefinition::DS_INTRINSIC_TEX_TEXEL_FLOAT2;
    m_builtin_semantics["::tex::texel_float3"] =
        IDefinition::DS_INTRINSIC_TEX_TEXEL_FLOAT3;
    m_builtin_semantics["::tex::texel_float4"] =
        IDefinition::DS_INTRINSIC_TEX_TEXEL_FLOAT4;
    m_builtin_semantics["::tex::texel_color"] =
        IDefinition::DS_INTRINSIC_TEX_TEXEL_COLOR;
    m_builtin_semantics["::tex::texture_isvalid"] =
        IDefinition::DS_INTRINSIC_TEX_TEXTURE_ISVALID;

    // df module
    m_builtin_semantics["::df::diffuse_reflection_bsdf"] =
        IDefinition::DS_INTRINSIC_DF_DIFFUSE_REFLECTION_BSDF;
    m_builtin_semantics["::df::diffuse_transmission_bsdf"] =
        IDefinition::DS_INTRINSIC_DF_DIFFUSE_TRANSMISSION_BSDF;
    m_builtin_semantics["::df::specular_bsdf"] =
        IDefinition::DS_INTRINSIC_DF_SPECULAR_BSDF;
    m_builtin_semantics["::df::simple_glossy_bsdf"] =
        IDefinition::DS_INTRINSIC_DF_SIMPLE_GLOSSY_BSDF;
    m_builtin_semantics["::df::backscattering_glossy_reflection_bsdf"] =
        IDefinition::DS_INTRINSIC_DF_BACKSCATTERING_GLOSSY_REFLECTION_BSDF;
    m_builtin_semantics["::df::measured_bsdf"] =
        IDefinition::DS_INTRINSIC_DF_MEASURED_BSDF;
    m_builtin_semantics["::df::microfacet_beckmann_smith_bsdf"] =
        IDefinition::DS_INTRINSIC_DF_MICROFACET_BECKMANN_SMITH_BSDF;
    m_builtin_semantics["::df::microfacet_ggx_smith_bsdf"] =
        IDefinition::DS_INTRINSIC_DF_MICROFACET_GGX_SMITH_BSDF;
    m_builtin_semantics["::df::microfacet_beckmann_vcavities_bsdf"] =
        IDefinition::DS_INTRINSIC_DF_MICROFACET_BECKMANN_VCAVITIES_BSDF;
    m_builtin_semantics["::df::microfacet_ggx_vcavities_bsdf"] =
        IDefinition::DS_INTRINSIC_DF_MICROFACET_GGX_VCAVITIES_BSDF;
    m_builtin_semantics["::df::ward_geisler_moroder_bsdf"] =
        IDefinition::DS_INTRINSIC_DF_WARD_GEISLER_MORODER_BSDF;

    m_builtin_semantics["::df::diffuse_edf"] =
        IDefinition::DS_INTRINSIC_DF_DIFFUSE_EDF;
    m_builtin_semantics["::df::measured_edf"] =
        IDefinition::DS_INTRINSIC_DF_MEASURED_EDF;
    m_builtin_semantics["::df::spot_edf"] =
        IDefinition::DS_INTRINSIC_DF_SPOT_EDF;
    m_builtin_semantics["::df::anisotropic_vdf"] =
        IDefinition::DS_INTRINSIC_DF_ANISOTROPIC_VDF;
    m_builtin_semantics["::df::normalized_mix"] =
        IDefinition::DS_INTRINSIC_DF_NORMALIZED_MIX;
    m_builtin_semantics["::df::clamped_mix"] =
        IDefinition::DS_INTRINSIC_DF_CLAMPED_MIX;
    m_builtin_semantics["::df::weighted_layer"] =
        IDefinition::DS_INTRINSIC_DF_WEIGHTED_LAYER;
    m_builtin_semantics["::df::fresnel_layer"] =
        IDefinition::DS_INTRINSIC_DF_FRESNEL_LAYER;
    m_builtin_semantics["::df::custom_curve_layer"] =
        IDefinition::DS_INTRINSIC_DF_CUSTOM_CURVE_LAYER;
    m_builtin_semantics["::df::measured_curve_layer"] =
        IDefinition::DS_INTRINSIC_DF_MEASURED_CURVE_LAYER;
    m_builtin_semantics["::df::thin_film"] =
        IDefinition::DS_INTRINSIC_DF_THIN_FILM;
    m_builtin_semantics["::df::tint"] =
        IDefinition::DS_INTRINSIC_DF_TINT;
    m_builtin_semantics["::df::directional_factor"] =
        IDefinition::DS_INTRINSIC_DF_DIRECTIONAL_FACTOR;
    m_builtin_semantics["::df::measured_curve_factor"] =
        IDefinition::DS_INTRINSIC_DF_MEASURED_CURVE_FACTOR;
    m_builtin_semantics["::df::light_profile_power"] =
        IDefinition::DS_INTRINSIC_DF_LIGHT_PROFILE_POWER;
    m_builtin_semantics["::df::light_profile_maximum"] =
        IDefinition::DS_INTRINSIC_DF_LIGHT_PROFILE_MAXIMUM;
    m_builtin_semantics["::df::light_profile_isvalid"] =
        IDefinition::DS_INTRINSIC_DF_LIGHT_PROFILE_ISVALID;
    m_builtin_semantics["::df::bsdf_measurement_isvalid"] =
        IDefinition::DS_INTRINSIC_DF_BSDF_MEASUREMENT_ISVALID;
    m_builtin_semantics["::df::color_normalized_mix"] =
        IDefinition::DS_INTRINSIC_DF_COLOR_NORMALIZED_MIX;
    m_builtin_semantics["::df::color_clamped_mix"] =
        IDefinition::DS_INTRINSIC_DF_COLOR_CLAMPED_MIX;
    m_builtin_semantics["::df::color_weighted_layer"] =
        IDefinition::DS_INTRINSIC_DF_COLOR_WEIGHTED_LAYER;
    m_builtin_semantics["::df::color_fresnel_layer"] =
        IDefinition::DS_INTRINSIC_DF_COLOR_FRESNEL_LAYER;
    m_builtin_semantics["::df::color_custom_curve_layer"] =
        IDefinition::DS_INTRINSIC_DF_COLOR_CUSTOM_CURVE_LAYER;
    m_builtin_semantics["::df::color_measured_curve_layer"] =
        IDefinition::DS_INTRINSIC_DF_COLOR_MEASURED_CURVE_LAYER;
    m_builtin_semantics["::df::fresnel_factor"] =
        IDefinition::DS_INTRINSIC_DF_FRESNEL_FACTOR;
    m_builtin_semantics["::df::measured_factor"] =
        IDefinition::DS_INTRINSIC_DF_MEASURED_FACTOR;
    m_builtin_semantics["::df::chiang_hair_bsdf"] =
        IDefinition::DS_INTRINSIC_DF_CHIANG_HAIR_BSDF;
    m_builtin_semantics["::df::sheen_bsdf"] =
        IDefinition::DS_INTRINSIC_DF_SHEEN_BSDF;



    // scene module
    m_builtin_semantics["::scene::data_isvalid"] =
        IDefinition::DS_INTRINSIC_SCENE_DATA_ISVALID;
    m_builtin_semantics["::scene::data_lookup_int"] =
        IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_INT;
    m_builtin_semantics["::scene::data_lookup_int2"] =
        IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_INT2;
    m_builtin_semantics["::scene::data_lookup_int3"] =
        IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_INT3;
    m_builtin_semantics["::scene::data_lookup_int4"] =
        IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_INT4;
    m_builtin_semantics["::scene::data_lookup_float"] =
        IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_FLOAT;
    m_builtin_semantics["::scene::data_lookup_float2"] =
        IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_FLOAT2;
    m_builtin_semantics["::scene::data_lookup_float3"] =
        IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_FLOAT3;
    m_builtin_semantics["::scene::data_lookup_float4"] =
        IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_FLOAT4;
    m_builtin_semantics["::scene::data_lookup_color"] =
        IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_COLOR;
    m_builtin_semantics["::scene::data_lookup_uniform_int"] =
        IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_INT;
    m_builtin_semantics["::scene::data_lookup_uniform_int2"] =
        IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_INT2;
    m_builtin_semantics["::scene::data_lookup_uniform_int3"] =
        IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_INT3;
    m_builtin_semantics["::scene::data_lookup_uniform_int4"] =
        IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_INT4;
    m_builtin_semantics["::scene::data_lookup_uniform_float"] =
        IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_FLOAT;
    m_builtin_semantics["::scene::data_lookup_uniform_float2"] =
        IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_FLOAT2;
    m_builtin_semantics["::scene::data_lookup_uniform_float3"] =
        IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_FLOAT3;
    m_builtin_semantics["::scene::data_lookup_uniform_float4"] =
        IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_FLOAT4;
    m_builtin_semantics["::scene::data_lookup_uniform_color"] =
        IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_COLOR;

    // debug module
    m_builtin_semantics["::debug::breakpoint"] =
        IDefinition::DS_INTRINSIC_DEBUG_BREAKPOINT;
    m_builtin_semantics["::debug::assert"] =
        IDefinition::DS_INTRINSIC_DEBUG_ASSERT;
    m_builtin_semantics["::debug::print"] =
        IDefinition::DS_INTRINSIC_DEBUG_PRINT;

    // anno module
    m_builtin_semantics["::anno::unused"] =
        IDefinition::DS_UNUSED_ANNOTATION;
    m_builtin_semantics["::anno::noinline"] =
        IDefinition::DS_NOINLINE_ANNOTATION;
    m_builtin_semantics["::anno::soft_range"] =
        IDefinition::DS_SOFT_RANGE_ANNOTATION;
    m_builtin_semantics["::anno::hard_range"] =
        IDefinition::DS_HARD_RANGE_ANNOTATION;
    m_builtin_semantics["::anno::hidden"] =
        IDefinition::DS_HIDDEN_ANNOTATION;
    m_builtin_semantics["::anno::deprecated"] =
        IDefinition::DS_DEPRECATED_ANNOTATION;
    m_builtin_semantics["::anno::version_number"] =
        IDefinition::DS_VERSION_NUMBER_ANNOTATION;
    m_builtin_semantics["::anno::version"] =
        IDefinition::DS_VERSION_ANNOTATION;
    m_builtin_semantics["::anno::dependency"] =
        IDefinition::DS_DEPENDENCY_ANNOTATION;
    m_builtin_semantics["::anno::ui_order"] =
        IDefinition::DS_UI_ORDER_ANNOTATION;
    m_builtin_semantics["::anno::usage"] =
        IDefinition::DS_USAGE_ANNOTATION;
    m_builtin_semantics["::anno::enable_if"] =
        IDefinition::DS_ENABLE_IF_ANNOTATION;
    m_builtin_semantics["::anno::thumbnail"] =
        IDefinition::DS_THUMBNAIL_ANNOTATION;
    m_builtin_semantics["::anno::display_name"] =
        IDefinition::DS_DISPLAY_NAME_ANNOTATION;
    m_builtin_semantics["::anno::in_group"] =
        IDefinition::DS_IN_GROUP_ANNOTATION;
    m_builtin_semantics["::anno::description"] =
        IDefinition::DS_DESCRIPTION_ANNOTATION;
    m_builtin_semantics["::anno::author"] =
        IDefinition::DS_AUTHOR_ANNOTATION;
    m_builtin_semantics["::anno::contributor"] =
        IDefinition::DS_CONTRIBUTOR_ANNOTATION;
    m_builtin_semantics["::anno::copyright_notice"] =
        IDefinition::DS_COPYRIGHT_NOTICE_ANNOTATION;
    m_builtin_semantics["::anno::created"] =
        IDefinition::DS_CREATED_ANNOTATION;
    m_builtin_semantics["::anno::modified"] =
        IDefinition::DS_MODIFIED_ANNOTATION;
    m_builtin_semantics["::anno::key_words"] =
        IDefinition::DS_KEYWORDS_ANNOTATION;
    m_builtin_semantics["::anno::origin"] =
        IDefinition::DS_ORIGIN_ANNOTATION;
}

// Create all options (and default values) of the compiler.
void MDL::create_options()
{
#define _STR(x) #x
#define STR(x)  _STR(x)

    m_options.add_option(option_dump_dependence_graph, "false",
        "Dump the auto-type dependence graph for every compiled function");
    m_options.add_option(option_dump_call_graph, "false",
        "Dump the call graph for every compiled module");
    m_options.add_option(option_warn, NULL,
        "Warning options, comma separated");
    m_options.add_option(option_opt_level, "2",
        "optimization level [0-2]: 0 disables all optimizations, 2 maximum optimization");
    m_options.add_option(option_strict, "true",
        "Enables strict MDL compliance");
    m_options.add_option(option_experimental_features, "false",
        "Enables undocumented experimental MDL features");
    m_options.add_option(option_resolve_resources, "true",
        "Controls resource resolution.");

    m_options.add_option(option_limits_float_min, STR(FLT_MIN),
        "The smallest positive normalized float value supported by the current platform");
    m_options.add_option(option_limits_float_max, STR(FLT_MAX),
        "The largest float value supported by the current platform");
    m_options.add_option(option_limits_double_min, STR(DBL_MIN),
        "The smallest positive normalized double value supported by the current platform");
    m_options.add_option(option_limits_double_max, STR(DBL_MAX),
        "The largest double value supported by the current platform");
    m_options.add_option(option_state_wavelength_base_max, STR(1),
        "The number of wavelengths returned in the result of wavelength base()");


    m_options.add_option(option_keep_original_resource_file_paths, "false",
        "Keep original resource file paths as is.");

#undef _STR
#undef STR
}

// Load a module from a stream.
Module *MDL::load_module(
    IModule_cache   *cache,
    IThread_context *context,
    char const      *module_name,
    IInput_stream   *s,
    unsigned        flags,
    char const      *msg_name)
{
    Thread_context *ctx = impl_cast<Thread_context>(context);

    // make sure there is a waiting table entry in case the module needs loading
    if (cache) {
        mi::base::Handle<const mi::mdl::IModule> existing_module(cache->lookup(module_name, NULL));
        if (existing_module) {
            // the module is cached and we load it anyway
            MDL_ASSERT(!"tied to load an already cached module");
        }
    }

    Module *module =
        create_module(module_name, s->get_filename(), IMDL::MDL_DEFAULT_VERSION, flags);
    if (module == NULL) {
        // cannot fail so far
        MDL_ASSERT(!"unexpected failure of create_module()");
        return NULL;
    }
    if (msg_name != NULL)
        module->set_msg_name(msg_name);

    Messages_impl &msgs = module->access_messages_impl();
    Syntax_error  err(get_allocator(), msgs);
    Scanner       scanner(get_allocator(), &err, s);
    Parser        parser(&scanner, &err);

    // native is a reserved word in MDL, switch it on for native modules
    scanner.enable_native_keyword((flags & Module::MF_IS_NATIVE) != 0);

    parser.set_imdl(get_allocator(), this);

    parser.set_module(module, get_compiler_bool_option(ctx, option_experimental_features, false));
    parser.Parse();

    mi::base::Handle<IArchive_input_stream> iarchvice_s(s->get_interface<IArchive_input_stream>());
    if (iarchvice_s.is_valid_interface()) {
        // this module was loaded from an archive, mark it
        mi::base::Handle<IArchive_manifest const> manifest(iarchvice_s->get_manifest());

        if (manifest.is_valid_interface()) {
            module->set_archive_info(manifest.get());
        }
    } else {
        mi::base::Handle<IMdle_input_stream> imdle_s(s->get_interface<IMdle_input_stream>());
        if (imdle_s.is_valid_interface()) {
            // this module was loaded from an mdle, compute function hashes
            module->m_is_hashed = true;
            module->m_is_mdle   = true;
        }
    }

    module->analyze(cache, ctx);
    return module;
}

// Load a module with a given name.
Module const *MDL::load_module(
    IThread_context *context,
    char const      *module_name,
    IModule_cache   *cache)
{
    mi::base::Handle<Thread_context> hctx;

    Thread_context *ctx = impl_cast<Thread_context>(context);

    if (ctx == NULL) {
        // user does not pass a context, create a temporary one
        hctx = mi::base::make_handle(create_thread_context());
        ctx  = hctx.get();
    }

    // clear message list
    ctx->clear_messages();

    // create the standard modules lazy
    register_builtin_module_at_cache(cache);

    Module const *res = NULL;

    if (IMDL_foreign_module_translator *translator = is_foreign_module(module_name)) {
        res = compile_foreign_module(*translator, *ctx, module_name, cache);
    } else {
        res = compile_module(*ctx, module_name, cache);
    }

    // copy the messages from the module to the context, so they are available over both
    // access paths
    if (context != NULL) {
        copy_message(ctx->access_messages_impl(), res);
    }
    return res;
}

// Load a module with a given name from a given string.
IModule const *MDL::load_module_from_string(
    IThread_context *context,
    IModule_cache   *cache,
    char const      *module_name,
    char const      *utf8_buffer,
    size_t          length)
{
    mi::base::Handle<IInput_stream> input(m_builder.create<Buffer_Input_stream>(
        m_builder.get_allocator(), utf8_buffer, length, /*filename=*/""));

    return load_module_from_stream(context, cache, module_name, input.get());
}

// Load a module with a given name from a given input stream.
IModule const *MDL::load_module_from_stream(
    IThread_context *context,
    IModule_cache   *cache,
    char const      *module_name,
    IInput_stream   *stream)
{
    mi::base::Handle<Thread_context> hctx;

    Thread_context *ctx = impl_cast<Thread_context>(context);

    if (ctx == NULL) {
        // user does not pass a context, create a temporary one
        hctx = mi::base::make_handle(create_thread_context());
        ctx  = hctx.get();
    }

    // clear message list
    ctx->clear_messages();

    // create the standard modules lazy
    register_builtin_module_at_cache(cache);

    char const *fname = stream->get_filename();
    if (fname == NULL || fname[0] == '\0')
        fname =  "<string>";

    Module const *res = compile_module_from_stream(
        *ctx, cache, module_name, stream, fname);

    // copy the messages from the module to the context, so they are available over both
    // access paths
    if (context != NULL) {
        copy_message(ctx->access_messages_impl(), res);
    }
    return res;
}

namespace
{
    // report success or failure of the loading process to the application
    void report_module_loading_result(
        Module          *mod,
        Module_callback &cb)
    {
        // only if there is a module cache
        if (!cb.is_valid()) {
            return;
        }

        MDL_ASSERT(cb.is_processing() &&
                   "Module is supposed to be processed in this context");

        // notify waiting threads about success or failure
        if (mod == NULL || !mod->is_valid()) {
            // notify about failure
            cb.module_loading_failed();
        } else {
            MDL_ASSERT(strcmp(mod->get_name(), cb.get_lookup_name()) == 0 &&
                        "module name and the cache lookup name do not match");

            // add an entry to the database
            if (!cb.register_module(mod)) {
                // there is a problem with this module outside of MDL Core, e.g. DB name clashes
                mod->access_messages_impl().add_error_message(
                    EXTERNAL_APPLICATION_ERROR, 'C', 0, 0,
                    "Module loading callback reported a registration failure");
            }
        }
    }
}


// Compile a module with a given name.
Module const *MDL::compile_module(
    Thread_context &ctx,
    char const     *module_name,
    IModule_cache  *module_cache)
{
    File_resolver resolver(
        *this,
        module_cache,
        m_external_resolver,
        m_search_path,
        m_search_path_lock,
        ctx.access_messages_impl(),
        ctx.get_front_path());

    if (char const *repl_module_name = ctx.get_replacement_module_name()) {
        char const *repl_file_name = ctx.get_replacement_file_name();

        MDL_ASSERT(repl_file_name != NULL && "Replacement file name must be set");
        resolver.set_module_replacement(repl_module_name, repl_file_name);
    }

    // resolve the module file
    mi::base::Handle<IMDL_import_result> result(resolve_import(
        resolver, module_name, /*owner_module=*/NULL, /*pos=*/NULL));
    if (!result.is_valid_interface()) {
        // name could not be resolved
        return NULL;
    }

    string mname(result->get_absolute_name(), get_allocator());
    if (Module const *std_mod = find_builtin_module(mname)) {
        // it's a standard module
        std_mod->retain();
        return std_mod;
    }

    // if the module was resolved successful and there is a cache,
    // announce that this module is now created or wait for another thread that currently processes
    // this module and then continue returning the cached module (processed by the other thread)
    Module_callback cb(module_cache);

    if (module_cache != NULL) {
        Module const *cached_mod = impl_cast<Module>(module_cache->lookup(mname.c_str(), cb));

        if (cached_mod != NULL) {
            if (!cached_mod->is_analyzed()) {
                // We found a not analyzed module. This can only happen if we
                // import something that is not processed because we have a dependency
                // loop in the import tree. This is not allowed.
                cached_mod->release();
                return NULL;
            }
            return cached_mod;
        } else if (cb.is_valid() && !cb.is_processing()) {
            // loading failed on different thread
            ctx.access_messages_impl().add_error_message(
                ERRONEOUS_IMPORT, 'C', 0, 0,
                "Loading module failed on a different context.");
            return NULL;
        }
        // continue with loading the module on this context
    }

    mi::base::Handle<IInput_stream> input(result->open(&ctx));

    if (!input) {
        // FIXME: add an error ??

        // If the top module can not be opened, notify all other waiting threads
        if (cb.is_valid()) {
            MDL_ASSERT(cb.is_processing() &&
                "Module is not supposed to be processed in this context");

            // notify via callback
            cb.module_loading_failed();
        }
        return NULL;
    }

    // load and compile the actual module
    Module *mod = load_module(module_cache, &ctx, mname.c_str(), input.get(), Module::MF_STANDARD);

    // notify waiting threads about success or failure
    report_module_loading_result(mod, cb);

    // any error was already handled by load_module() above
    return mod;
}

// Compile a foreign module with a given name.
Module const *MDL::compile_foreign_module(
    IMDL_foreign_module_translator &translator,
    Thread_context                 &ctx,
    char const                     *module_name,
    IModule_cache                  *module_cache)
{
    // if the module was resolved successful and there is a cache,
    // announce that this module is now created or wait for another thread that currently processes
    // this module and then continue returning the cached module (processed by the other thread)
    Module_callback cb(module_cache);

    if (module_cache != NULL) {
        Module const *cached_mod = impl_cast<Module>(module_cache->lookup(module_name, cb));

        if (cached_mod != NULL) {
            if (!cached_mod->is_analyzed()) {
                // We found a not analyzed module. This can only happen if we
                // import something that is not processed because we have a dependency
                // loop in the import tree. This is not allowed.
                cached_mod->release();
                return NULL;
            }
            return cached_mod;
        } else if (cb.is_valid() && !cb.is_processing()) {
            // loading failed on different thread
            ctx.access_messages_impl().add_error_message(
                ERRONEOUS_IMPORT, 'C', 0, 0,
                "Loading module failed on a different context.");
            return NULL;
        }
        // continue with loading the module on this context
    }

    // load and translate the foreign module
    Module const *mod =
        impl_cast<Module>(translator.compile_foreign_module(&ctx, module_name, module_cache));

    // notify waiting threads about success or failure
    report_module_loading_result(const_cast<Module *>(mod), cb);

    // any error was already handled by compile_foreign_module() above
    return mod;
}

// Compile a module with a given name from a input stream.
Module const *MDL::compile_module_from_stream(
    Thread_context &ctx,
    IModule_cache  *module_cache,
    char const     *module_name,
    IInput_stream  *input,
    char const     *msg_name)
{
    if (module_name[0] != ':' || module_name[1] != ':') {
        // not absolute
        MDL_ASSERT(!"module name not absolute");
        return NULL;
    }

    File_resolver resolver(
        *this,
        module_cache,
        m_external_resolver,
        m_search_path,
        m_search_path_lock,
        ctx.access_messages_impl(),
        ctx.get_front_path());

    if (resolver.exists(module_name)) {
        // such a module exists, overwrite is forbidden
        return NULL;
    }

    string mname(module_name, get_allocator());
    if (find_builtin_module(mname) != NULL) {
        // it's a standard module: those cannot be overwritten
        return NULL;
    }

    // if the module was resolved successful and there is a cache, 
    // announce that this module is now created or wait for another thread that currently processes 
    // this module and then continue returning the cached module (processed by the other thread)
    Module_callback cb(module_cache);

    if (module_cache != NULL) {
        Module const *cached_mod = impl_cast<Module>(module_cache->lookup(module_name, cb));

        if (cached_mod != NULL) {
            // already exists, overwrite is not allowed
            cached_mod->release();
            return NULL;
        }
    }

    // note: this will set the message list's file name to msg_name
    Module *mod = load_module(
        module_cache, &ctx, module_name, input, Module::MF_STANDARD, msg_name);

    // notify waiting threads about success or failure
    report_module_loading_result(mod, cb);

    // any error was already handled by load_module() above
    return mod;
}

// Load a code generator.
ICode_generator *MDL::load_code_generator(const char *target_language)
{
    if (strcmp(target_language, "dag") == 0) {
        return create_code_generator_dag(m_builder.get_allocator(), this);
    } else if (strcmp(target_language, "glsl") == 0) {
        return create_code_generator_glsl(m_builder.get_allocator(), this);
    } else if (strcmp(target_language, "jit") == 0) {
        return create_code_generator_jit(m_builder.get_allocator(), this);
    }
    return NULL;
}

// Parse a string to an MDL expression.
IExpression const *MDL::parse_expression(
    char const    *expr_str,
    int           start_line,
    int           start_col,
    Module        *module,
    bool          enable_experimental_features,
    Messages_impl &msgs)
{
    mi::base::Handle<IInput_stream> input(m_builder.create<Buffer_Input_stream>(
        m_builder.get_allocator(), expr_str, strlen(expr_str), /*filename=*/""));

    Syntax_error  err(get_allocator(), msgs);
    Scanner       scanner(get_allocator(), &err, input.get(), start_line, start_col);
    Parser        parser(&scanner, &err);

    parser.set_imdl(get_allocator(), this);

    parser.set_module(module, enable_experimental_features);
    return parser.parse_expression();
}

// Check if the given module name names a foreign module.
IMDL_foreign_module_translator *MDL::is_foreign_module(
    char const     *module_name)
{
    for (Translator_list::iterator it(m_translator_list.begin()), end(m_translator_list.end());
        it != end;
        ++it)
    {
        IMDL_foreign_module_translator *translator = it->get();

        if (translator->is_foreign_module(module_name))
            return translator;
    }
    return NULL;
}

// Create a printer.
Printer *MDL::create_printer(IOutput_stream *stream) const
{
    return m_builder.create<Printer>(m_builder.get_allocator(), stream);
}

// Access options.
Options &MDL::access_options()
{
    return m_options;
}

// Return a builtin semantic for a given absolute intrinsic function name.
IDefinition::Semantics MDL::get_builtin_semantic(char const *name) const
{
    Sema_map::const_iterator it = m_builtin_semantics.find(name);
    if (it != m_builtin_semantics.end())
        return it->second;
    return IDefinition::DS_UNKNOWN;
}

// Evaluates an intrinsic function called on constant arguments.
IValue const *MDL::evaluate_intrinsic_function(
    IValue_factory         *value_factory,
    IDefinition::Semantics sema,
    IValue const * const   arguments[],
    size_t                 n_arguments) const
{
    return mi::mdl::evaluate_intrinsic_function(value_factory, sema, arguments, n_arguments);
}

// Serialize a module and all its imported modules in bottom-up order.
void MDL::serialize_module_with_imports(
    Module const          *mod,
    ISerializer           *is,
    MDL_binary_serializer &bin_serializer,
    bool                  is_root) const
{
    // ensure that all imported modules are serialized before ...
    for (int i = 0, n = mod->get_import_count(); i < n; ++i) {
        mi::base::Handle<Module const> imp_mod(mod->get_import(i));

        if (!bin_serializer.is_module_registered(imp_mod.get())) {
            serialize_module_with_imports(
                imp_mod.get(), is, bin_serializer, /*is_root=*/false);
        }
    }

    // ... and finally the module itself: could be already written by a sibling, so check here
    // Note that the root module MUST be written, even if is is already known (happens for stdlib
    // modules).
    if (is_root || !bin_serializer.is_module_registered(mod)) {
        Module_serializer mod_serializer(get_allocator(), is, &bin_serializer);
        mod->serialize(mod_serializer);
    }
}

// Serialize a module to the given serializer.
void MDL::serialize_module(
    IModule const *module,
    ISerializer   *is,
    bool          include_dependencies) const
{
    MDL_binary_serializer bin_serializer(get_allocator(), this, is);

    Module const *mod = impl_cast<Module>(module);

    if (include_dependencies) {
        bin_serializer.write_section_tag(Serializer::ST_BINARY_START);
        DOUT(("Starting Serializing Binary\n")); INC_SCOPE();

        // serialize the module with dependencies
        serialize_module_with_imports(mod, is, bin_serializer, /*is_root=*/true);

        // mark the end of the binary
        bin_serializer.write_section_tag(Serializer::ST_BINARY_END);
        DEC_SCOPE(); DOUT(("Binary Serializing Finished\n\n"));
    } else {
        // serialize this module only
        Module_serializer mod_serializer(get_allocator(), is, &bin_serializer);
        mod->serialize(mod_serializer);
    }
}

// Deserialize a module from a given deserializer.
Module const *MDL::deserialize_module(IDeserializer *ds)
{
    MDL_binary_deserializer bin_deserializer(get_allocator(), ds, this);

    Tag_t t;

    // currently we support only binaries, no single units
    t = bin_deserializer.read_section_tag();
    bool include_dependencies = t == Serializer::ST_BINARY_START;

    if (include_dependencies) {
        DOUT(("Starting Binary Deserialization\n")); INC_SCOPE();

        // read the first real tag
        t = bin_deserializer.read_section_tag();
    }

    // read all modules from this binary
    mi::base::Handle<Module const> mod;
    vector<mi::base::Handle<Module const> >::Type mod_cache(get_allocator());

    for (;; t = bin_deserializer.read_section_tag()) {
        if (t == Serializer::ST_MODULE_START) {
            Module_deserializer mod_deserializer(
                get_allocator(), ds, &bin_deserializer, this);
            mod = mi::base::make_handle(Module::deserialize(mod_deserializer));

            if (!include_dependencies)
                break;

            // ensure that this module is not free'd until the loop is finished
            mod_cache.push_back(mod);
        } else if (include_dependencies && t == Serializer::ST_BINARY_END) {
            DEC_SCOPE(); DOUT(("Binary Deserialization Finished\n\n"));
            break;
        } else {
            // error
            DOUT(("Error: Unsupported section tag 0x%X\n", unsigned(t)));
            break;
        }
    }

    // return the last one, this is the "main" module.h ensure that it is not dropped
    // when the mod_cache is deleted, so increase its refcount here
    if (mod.is_valid_interface())
        mod->retain();
    return mod.get();
}

// Create an IOutput_stream standard stream.
IOutput_stream *MDL::create_std_stream(Std_stream kind) const
{
    switch (kind) {
    case OS_STDOUT:
        return m_builder.create<File_Output_stream>(m_builder.get_allocator(), stdout, false);
    case OS_STDERR:
        return m_builder.create<File_Output_stream>(m_builder.get_allocator(), stderr, false);
    case OS_STDDBG:
        return m_builder.create<Debug_Output_stream>(m_builder.get_allocator());
    }
    return NULL;
}

// Create an IOutput_stream from a file.
IOutput_stream *MDL::create_file_output_stream(char const *filename) const
{
    if (FILE *f = fopen(filename, "wb")) {
        return m_builder.create<File_Output_stream>(
            m_builder.get_allocator(), f, /*close_at_destroy=*/true);
    }
    return NULL;
}

// Create an IInput_stream from a file.
IInput_stream *MDL::create_file_input_stream(char const *filename) const
{
    if (FILE *f = fopen(filename, "rb")) {
        return m_builder.create<File_Input_stream>(
            m_builder.get_allocator(), f, /*close_at_destroy=*/true, filename);
    }
    return NULL;
}


// Serialize a code DAG to the given serializer.
void MDL::serialize_code_dag(
    IGenerated_code_dag const *code,
    ISerializer               *is) const
{
    MDL_binary_serializer bin_serializer(get_allocator(), this, is);
    mi::mdl::serialize_code_dag(code, is, bin_serializer);
}

// Deserialize a code DAG from a given deserializer.
IGenerated_code_dag const *MDL::deserialize_code_dag(IDeserializer *ds)
{
    MDL_binary_deserializer bin_deserializer(get_allocator(), ds, this);

    return mi::mdl::deserialize_code_dag(ds, bin_deserializer, this);
}

// Create a new MDL lambda function.
ILambda_function *MDL::create_lambda_function(
    ILambda_function::Lambda_execution_context context)
{
    mi::base::Handle<ICode_generator_dag> dag_be =
        mi::base::make_handle(load_code_generator("dag")).get_interface<ICode_generator_dag>();
    if (!dag_be.is_valid_interface()) {
        MDL_ASSERT("DAG backend missing");
        return NULL;
    }

    return dag_be->create_lambda_function(context);
}

// Create a new MDL distribution function.
IDistribution_function *MDL::create_distribution_function()
{
    mi::base::Handle<ICode_generator_dag> dag_be =
        mi::base::make_handle(load_code_generator("dag")).get_interface<ICode_generator_dag>();
    if (!dag_be.is_valid_interface()) {
        MDL_ASSERT("DAG backend missing");
        return NULL;
    }

    return dag_be->create_distribution_function();
}

// Serialize a lambda function to the given serializer.
void MDL::serialize_lambda(
    ILambda_function const *lambda,
    ISerializer            *is)
{
    mi::base::Handle<ICode_generator_dag> dag_be =
        mi::base::make_handle(load_code_generator("dag")).get_interface<ICode_generator_dag>();
    if (!dag_be.is_valid_interface()) {
        MDL_ASSERT("DAG backend missing");
        return;
    }

    dag_be->serialize_lambda(lambda, is);
}

// Deserialize a lambda function from a given deserializer.
ILambda_function *MDL::deserialize_lambda(IDeserializer *ds)
{
    mi::base::Handle<ICode_generator_dag> dag_be =
        mi::base::make_handle(load_code_generator("dag")).get_interface<ICode_generator_dag>();
    if (!dag_be.is_valid_interface()) {
        MDL_ASSERT("DAG backend missing");
        return NULL;
    }

    return dag_be->deserialize_lambda(ds);
}

// Check if the given absolute module name name a builtin MDL module.
bool MDL::is_builtin_module(char const *absname) const
{
    return find_builtin_module(string(absname, get_allocator())) != NULL;
}

// Add a new builtin module to the MDL compiler.
bool MDL::add_builtin_module(
    char const *abs_name,
    char const *buffer,
    size_t     buf_len,
    bool       is_encoded,
    bool       is_native)
{
    mi::base::Handle<Buffer_Input_stream> s;

    if (is_encoded) {
        s = mi::base::make_handle(m_builder.create<Encoded_buffer_Input_stream>(
            m_builder.get_allocator(), (unsigned char const *)buffer, buf_len, ""));
    } else {
        s = mi::base::make_handle(m_builder.create<Buffer_Input_stream>(
            m_builder.get_allocator(), buffer, buf_len, ""));
    }

    unsigned flags = Module::MF_IS_OWNED | (is_native ? Module::MF_IS_NATIVE : 0);
    mi::base::Handle<Module> mod(load_module(NULL, NULL, abs_name, s.get(), flags));

    size_t id = mod->get_unique_id();
    MDL_ASSERT(id > 0 && "Module ID must be != 0");
    if (m_builtin_modules.size() + 1 != id) {
        return false;
    }

    // takes ownership
    mod->retain();
    register_builtin_module(mod.get());
    return true;
}

// Get the used allocator.
mi::base::IAllocator *MDL::get_mdl_allocator() const
{
    mi::base::IAllocator *alloc = Base::get_allocator();

    alloc->retain();
    return alloc;
}

// Get the MDL version of a given Module.
MDL::MDL_version MDL::get_module_version(IModule const *imodule) const
{
    if (imodule != NULL) {
        Module const *mod = impl_cast<Module>(imodule);
        return mod->get_mdl_version();
    }
    return MDL_DEFAULT_VERSION;
}

// Creates a new thread context for this compiler.
Thread_context *MDL::create_thread_context()
{
    return m_builder.create<Thread_context>(get_allocator(), &m_options);
}

// Creates a new thread context from current analysis settings.
Thread_context *MDL::create_thread_context(
    Analysis const &ana,
    char const    *front_path)
{
    Thread_context *ctx = m_builder.create<Thread_context>(get_allocator(), &m_options);

    ctx->set_front_path(front_path);

    ctx->access_options().set_option(
        MDL::option_strict,
        ana.strict_mode() ? "true" : "false");
    ctx->access_options().set_option(
        MDL::option_experimental_features,
        ana.enable_experimental_features() ? "true" : "false");
    ctx->access_options().set_option(
        MDL::option_resolve_resources,
        ana.resolve_resources() ? "true" : "false");
    return ctx;
}

// Create an MDL exporter.
MDL_exporter *MDL::create_exporter() const
{
    return m_builder.create<MDL_exporter>(get_allocator());
}

/// Check if the given character is a valid MDL letter.
static bool is_mdl_letter(char c)
{
    if ('A' <= c && c <= 'Z')
        return true;
    if ('a' <= c && c <= 'z')
        return true;
    return false;
}

/// Check if the given character is a valid MDL digit.
static bool is_mdl_digit(char c)
{
    if ('0' <= c && c <= '9')
        return true;
    return false;
}

// Check if a given identifier is a valid MDL identifier.
bool MDL::valid_mdl_identifier(char const *ident)
{
    if (ident == NULL)
        return false;

    // first check general identifier rules:
    // IDENT = LETTER { LETTER | DIGIT | '_' } .
    char const *p = ident;

    if (!is_mdl_letter(*p))
        return false;

    for (++p; *p != '\0'; ++p) {
        if (*p == '_')
            continue;
        if (!is_mdl_letter(*p) && !is_mdl_digit(*p)) {
            return false;
        }
    }

    // now check for keywords
    p = ident;

#define FORBIDDEN(name, n) if (strcmp(p + n, name + n) == 0) return false

    switch (p[0]) {
    case 'a':
        FORBIDDEN("annotation", 1);
        FORBIDDEN("auto", 1);
        break;
    case 'b':
        if (p[1] == 'o') {
            FORBIDDEN("bool",  2);
            FORBIDDEN("bool2", 2);
            FORBIDDEN("bool3", 2);
        } else if (p[1] == 'r') {
            FORBIDDEN("break", 2);
        } else if (p[1] == 's') {
            FORBIDDEN("bsdf",             2);
            FORBIDDEN("bsdf_measurement", 2); // MDL 1.1+
        }
        break;
    case 'c':
        if (p[1] == 'a') {
            FORBIDDEN("case",  2);
            FORBIDDEN("catch", 2);
        } else if (p[1] == 'h') {
            FORBIDDEN("char", 2);
        } else if (p[1] == 'l') {
            FORBIDDEN("class", 2);
        } else if (p[1] == 'o') {
            FORBIDDEN("color",       2);
            FORBIDDEN("const",       2);
            FORBIDDEN("const_class", 2);
        }
        break;
    case 'd':
        if (p[1] == 'e') {
            FORBIDDEN("delete", 2);
        } else if (p[1] == 'o') {
            if (p[2] == '\0')
                return false;
            FORBIDDEN("double",    2);
            FORBIDDEN("double2",   2);
            FORBIDDEN("double3",   2);
            FORBIDDEN("double4",   2);
            FORBIDDEN("double2x2", 2);
            FORBIDDEN("double2x3", 2);
            FORBIDDEN("double2x4", 2);
            FORBIDDEN("double3x2", 2);
            FORBIDDEN("double3x3", 2);
            FORBIDDEN("double3x4", 2);
            FORBIDDEN("double4x2", 2);
            FORBIDDEN("double4x3", 2);
            FORBIDDEN("double4x4", 2);
        } else if (p[1] == 'y') {
            FORBIDDEN("dynamic_cast", 2);
        }
        break;
    case 'e':
        if (p[1] == 'x') {
            FORBIDDEN("export",   2);
            FORBIDDEN("explicit", 2);
            FORBIDDEN("extern",   2);
            FORBIDDEN("external", 2);
        } else {
            FORBIDDEN("edf",  1);
            FORBIDDEN("else", 1);
            FORBIDDEN("enum", 1);
           
        }
        break;
    case 'f':
        if (p[1] == 'a') {
            FORBIDDEN("false", 2);
        } else if (p[1] == 'l') {
            FORBIDDEN("float",    2);
            FORBIDDEN("float2",   2);
            FORBIDDEN("float3",   2);
            FORBIDDEN("float4",   2);
            FORBIDDEN("float2x2", 2);
            FORBIDDEN("float2x3", 2);
            FORBIDDEN("float2x4", 2);
            FORBIDDEN("float3x2", 2);
            FORBIDDEN("float3x3", 2);
            FORBIDDEN("float3x4", 2);
            FORBIDDEN("float4x2", 2);
            FORBIDDEN("float4x3", 2);
            FORBIDDEN("float4x4", 2);
        } else if (p[1] == 'o') {
            FORBIDDEN("for",     2);
            FORBIDDEN("foreach", 2);
        } else if (p[1] == 'r') {
            FORBIDDEN("friend", 2);
        }
        break;
    case 'g':
        FORBIDDEN("goto",  1);
        FORBIDDEN("graph", 1);
        break;
    case 'h':
        if (p[1] == 'a') {
            FORBIDDEN("half",    2);
            FORBIDDEN("half2",   2);
            FORBIDDEN("half3",   2);
            FORBIDDEN("half4",   2);
            FORBIDDEN("half2",   2);
            FORBIDDEN("half2x2", 2);
            FORBIDDEN("half2x3", 2);
            FORBIDDEN("half2x4", 2);
            FORBIDDEN("half3x2", 2);
            FORBIDDEN("half3x3", 2);
            FORBIDDEN("half3x4", 2);
            FORBIDDEN("half4x2", 2);
            FORBIDDEN("half4x3", 2);
            FORBIDDEN("half4x4", 2);
            FORBIDDEN("hair_bsdf", 2);  // MDL 1.5+
        }
        break;
    case 'i':
        if (p[1] == 'f') {
            return p[2] != '\0';
        } else if (p[1] == 'm') {
            FORBIDDEN("import", 2);
        } else if (p[1] == 'n') {
            if (p[2] == '\0')
                return false;
            FORBIDDEN("inline",                     2);
            FORBIDDEN("inout",                      2);
            FORBIDDEN("int",                        2);
            FORBIDDEN("int2",                       2);
            FORBIDDEN("int3",                       2);
            FORBIDDEN("int4",                       2);
            FORBIDDEN("intensity_mode",             2); // MDL 1.1+
            FORBIDDEN("intensity_power",            2); // MDL 1.1+
            FORBIDDEN("intensity_radiant_exitance", 2); // MDL 1.1+
        }
        break;
    case 'l':
        FORBIDDEN("lambda",        1);
        FORBIDDEN("let",           1);
        FORBIDDEN("light_profile", 1);
        FORBIDDEN("long",          1);
        break;
    case 'm':
        if (p[1] == 'a') {
            FORBIDDEN("material",          2);
            FORBIDDEN("material_emission", 2);
            FORBIDDEN("material_geometry", 2);
            FORBIDDEN("material_surface",  2);
            FORBIDDEN("material_volume",   2);
        }
        else {
            FORBIDDEN("mdl",     1);
            FORBIDDEN("module",  1);
            FORBIDDEN("mutable", 1);
        }
        break;
    case 'n':
        if (p[1] == 'a') {
            FORBIDDEN("namespace", 2);
            FORBIDDEN("native",    2);
        } else {
            FORBIDDEN("new", 1);
        }
        break;
    case 'o':
        FORBIDDEN("operator", 1);
        FORBIDDEN("out",      1);
        break;
    case 'p':
        if (p[1] == 'r') {
            FORBIDDEN("private",   2);
            FORBIDDEN("protected", 2);
        } else {
            FORBIDDEN("package",    1);
            FORBIDDEN("phenomenon", 1);
            FORBIDDEN("public",     1);
        }
        break;
    case 'r':
        if (p[1] == 'e') {
            FORBIDDEN("return",           2);
            FORBIDDEN("reinterpret_cast", 2);
        }
        break;
    case 's':
        if (p[1] == 'a') {
            FORBIDDEN("sampler", 2);
        } else if (p[1] == 'h') {
            FORBIDDEN("shader", 2);
            FORBIDDEN("short",  2);
        } else if (p[1] == 'i') {
            FORBIDDEN("signed", 2);
            FORBIDDEN("sizeof", 2);
        } else if (p[1] == 't') {
            FORBIDDEN("static",      2);
            FORBIDDEN("static_cast", 2);
            FORBIDDEN("string",      2);
            FORBIDDEN("struct",      2);
        }
        else {
            FORBIDDEN("switch",  1);
        }
        break;
    case 't':
        if (p[1] == 'e') {
            FORBIDDEN("technique",    2);
            FORBIDDEN("template",     2);
            FORBIDDEN("texture_2d",   2);
            FORBIDDEN("texture_3d",   2);
            FORBIDDEN("texture_cube", 2);
            FORBIDDEN("texture_ptex", 2);
        } else if (p[1] == 'h') {
            FORBIDDEN("this",  2);
            FORBIDDEN("throw", 2);
        } else if (p[1] == 'r') {
            FORBIDDEN("true", 2);
            FORBIDDEN("try",  2);
        } else if (p[1] == 'y') {
            FORBIDDEN("typedef",  2);
            FORBIDDEN("typeid",   2);
            FORBIDDEN("typename", 2);
        }
        break;
    case 'u':
        if (p[1] == 'n') {
            FORBIDDEN("uniform",  2);
            FORBIDDEN("union",    2);
            FORBIDDEN("unsigned", 2);
        } else {
            FORBIDDEN("using", 1);
        }
        break;
    case 'v':
        FORBIDDEN("varying",  1);
        FORBIDDEN("vdf",      1);
        FORBIDDEN("virtual",  1);
        FORBIDDEN("void",     1);
        FORBIDDEN("volatile", 1);
        break;
    case 'w':
        FORBIDDEN("wchar_t", 1);
        FORBIDDEN("while",   1);
        break;
   }
#undef FORBIDDEN

    return true;
}

// Check if a given identifier is a valid MDL identifier.
bool MDL::is_valid_mdl_identifier(char const *ident) const
{
    return valid_mdl_identifier(ident);
}

// Create an MDL entity resolver.
IEntity_resolver *MDL::create_entity_resolver(
    IModule_cache *module_cache) const
{
    return m_builder.create<Entity_resolver>(
        get_allocator(),
        this,
        module_cache,
        m_external_resolver,
        m_search_path);
}

/// Return the current MDL entity resolver.
IEntity_resolver *MDL::get_entity_resolver(
    IModule_cache *module_cache) const
{
    if (m_external_resolver) {
        m_external_resolver->retain();
        return m_external_resolver.get();
    }

    return create_entity_resolver(module_cache);
}

// Create an MDL archive tool using this compiler.
IArchive_tool *MDL::create_archive_tool()
{
    return m_builder.create<Archive_tool>(get_allocator(), this);
}

// Create an MDL encapsulate tool using this compiler.
IEncapsulate_tool *MDL::create_encapsulate_tool()
{
    return m_builder.create<Encapsulate_tool>(get_allocator(), this);
}

// Create an MDL comparator tool using this compiler.
IMDL_comparator *MDL::create_mdl_comparator()
{
    // creates a new compiler owned by the comparator
    mi::base::Handle<MDL> compiler(create_mdl(get_allocator()));

    mi::base::Handle<IMDL_search_path> const &sp = get_search_path();

    // by default, get the search path of the parent
    sp->retain(); // takes ownership!
    compiler->install_search_path(sp.get());

    return m_builder.create<MDL_comparator>(get_allocator(), compiler.get());
}

// Create an MDL module transformer using this compiler.
IMDL_module_transformer *MDL::create_module_transformer()
{
    return m_builder.create<MDL_module_transformer>(get_allocator(), this);
}

// Sets a resolver interface that will be used to lookup MDL modules and resources.
void MDL::set_external_entity_resolver(IEntity_resolver *resolver)
{
    m_external_resolver = mi::base::make_handle_dup(resolver);
}

// Check if an external entity resolver is installed.
bool MDL::uses_external_entity_resolver() const
{
    return m_external_resolver.is_valid_interface();
}

// Add a foreign module translator.
void MDL::add_foreign_module_translator(
    IMDL_foreign_module_translator *translator)
{
    m_translator_list.push_back(mi::base::make_handle_dup(translator));
}

// Remove a foreign module translator.
bool MDL::remove_foreign_module_translator(
    IMDL_foreign_module_translator *translator)
{
    for (Translator_list::iterator it(m_translator_list.begin()), end(m_translator_list.end());
        it != end;
        ++it)
    {
        if ((*it).get() == translator) {
            m_translator_list.erase(it);
            return true;
        }
    }
    return false;
}

// Check if the compiler supports a requested MDL version.
bool MDL::check_version(int major, int minor, MDL_version &version, bool enable_experimental_features)
{
    version = MDL_DEFAULT_VERSION;

    if (major == 1) {
        switch (minor) {
        case 0:
            version = MDL_VERSION_1_0;
            return true;
        case 1:
            version = MDL_VERSION_1_1;
            return true;
        case 2:
            version = MDL_VERSION_1_2;
            return true;
        case 3:
            version = MDL_VERSION_1_3;
            return true;
        case 4:
            version = MDL_VERSION_1_4;
            return true;
        case 5:
            version = MDL_VERSION_1_5;
            return true;
        case 6:
            version = MDL_VERSION_1_6;
            return true;
        case 7:
            if (!enable_experimental_features)
                return false;
            version = MDL_VERSION_1_7;
            return true;
        }
    }
    return false;
}

// Register a builtin module and take ownership of it.
void MDL::register_builtin_module(Module const *module)
{
    // there should be be neither errors nor warnings in stdlib modules ...
    MDL_ASSERT(module->access_messages().get_message_count() == 0);
    string mod_name(module->get_name(), get_allocator());

    size_t id = module->get_unique_id();
    MDL_ASSERT(id > 0 && "Module ID must be != 0");
    size_t idx = id - 1;
    if (m_builtin_modules.size() <= idx) {
        m_builtin_modules.resize(idx + 1);
    }
    // takes ownership, it is thrown away if the compiler is released
    m_builtin_modules[idx] = mi::base::Handle<Module const>(module);

    m_builtin_module_indexes[mod_name] = id;
}

// Find a builtin module by name.
Module const *MDL::find_builtin_module(string const &name) const
{
    Module_map::const_iterator it = m_builtin_module_indexes.find(name);

    if (it != m_builtin_module_indexes.end()) {
        size_t id = it->second;
        return find_builtin_module(id);
    }
    return NULL;
}

// Find a builtin module by its id.
Module const *MDL::find_builtin_module(size_t id) const
{
    size_t idx = id - 1;
    if (idx < m_builtin_modules.size()) {
        return m_builtin_modules[idx].get();
    }
    return NULL;
}

// Find the definition of a signature of a standard library function.
IDefinition const *MDL::find_stdlib_signature(
    char const *module_name,
    char const *signature) const
{
    if (Module const *stdmod = find_builtin_module(string(module_name, get_allocator())))
        if (stdmod->is_stdlib())
            return stdmod->find_signature(signature, /*only_exported=*/true);
    return NULL;
}

/// Check if the owner name represents a module in the root package.
static bool owner_is_in_root(char const *owner_name)
{
    if (owner_name == NULL) {
        // an import from root itself
        return true;
    }
    MDL_ASSERT(owner_name[0] == ':' && owner_name[1] == ':');

    for (char const *p = owner_name + 2; p != NULL;) {
        char const *n = strchr(p, ':');

        if (n != NULL) {
            if (n[1] == ':') {
                // found an '::' at n, we are inside a package
                return false;
            }
            ++n;
        }
        p = n;
    }
    // "::" was not found
    return true;
}

// Resolve an import (module) name to the corresponding absolute module name.
IMDL_import_result *MDL::resolve_import(
    File_resolver  &resolver,
    char const     *import_name,
    Module         *owner_module,
    Position const *pos)
{
    char const *owner_name     = NULL;
    char const *owner_filename = NULL;

    if (owner_module != NULL) {
        owner_name     = owner_module->get_name();
        owner_filename = owner_module->get_filename();
    }

    string builtin_name(get_allocator());
    if (import_name[0] == ':' && import_name[1] == ':') {
        // fast path: if it is a absolute name, check for builtin modules
        builtin_name = import_name;
    } else if (owner_is_in_root(owner_name)) {
        // fast path 2: if we are at root, every import starts also at root
        builtin_name = "::";
        builtin_name += import_name;
    }

    if (!builtin_name.empty()) {
        if (Module const *mod = find_builtin_module(builtin_name)) {
            // found
            Allocator_builder builder(get_allocator());
            return builder.create<MDL_import_result>(
                get_allocator(),
                string(mod->get_name(), get_allocator()),
                string(mod->get_filename(), get_allocator()));
        }
    }

    Position_impl zero_pos(0, 0, 0, 0);
    if (pos == NULL)
        pos = &zero_pos;

    mi::base::Handle<IMDL_import_result> res(resolver.resolve_import(
        *pos,
        import_name,
        owner_name,
        owner_filename));
    if (res.is_valid_interface()) {
        // found
        res->retain();
        return res.get();
    }

    // could be a builtin module, check that
    string s(get_allocator());
    if (import_name[0] != ':' || import_name[1] != ':')
        s += "::";
    s += import_name;
    if (Module const *mod = find_builtin_module(s)) {
        // found
        Allocator_builder builder(get_allocator());
        return builder.create<MDL_import_result>(
            get_allocator(),
            string(mod->get_name(), get_allocator()),
            string(mod->get_filename(), get_allocator()));
    }

    // not found
    Messages_impl &msgs = resolver.get_messages_impl();

    size_t mod_id = 0;
    if (owner_filename != NULL) {
        mod_id = msgs.register_fname(owner_filename);
    }

    // create the error message
    Error_params param(get_allocator());
    param.add(import_name);

    mi::base::Handle<Buffer_output_stream> os(
        m_builder.create<Buffer_output_stream>(get_allocator()));
    mi::base::Handle<Printer> printer(create_printer(os.get()));

    print_error_message(MODULE_NOT_FOUND, File_resolver::MESSAGE_CLASS, param, printer.get());

    msgs.add_error_message(
        MODULE_NOT_FOUND,
        File_resolver::MESSAGE_CLASS,
        mod_id,
        pos,
        os->get_data());

    return NULL;
}

// Get an option value.
char const *MDL::get_compiler_option(
    Thread_context const *ctx,
    char const           *name) const
{
    if (ctx != NULL) {
        Options_impl const &opt = ctx->access_options();
        for (int i = 0, n = opt.get_option_count(); i < n; ++i) {
            if (strcmp(opt.get_option_name(i), name) == 0) {
                if (opt.is_option_modified(i)) {
                    // this option was set, use it
                    return opt.get_option_value(i);
                }
                break;
            }
        }
    }
    for (int i = 0, n = m_options.get_option_count(); i < n; ++i) {
        if (strcmp(m_options.get_option_name(i), name) == 0) {
            return m_options.get_option_value(i);
        }
    }
    return NULL;
}

// Get a bool option.
bool MDL::get_compiler_bool_option(
    Thread_context const *ctx,
    char const           *name,
    bool                 def_value) const
{
    if (char const *val = get_compiler_option(ctx, name)) {
        if (strcmp(val, "1") == 0 || strcmp(val, "true") == 0)
            return true;
        if (strcmp(val, "0") == 0 || strcmp(val, "false") == 0)
            return false;
        // unknown ...
    }
    return def_value;
}

// Get an integer option.
int MDL::get_compiler_int_option(
    Thread_context const *ctx,
    char const           *name,
    int                  def_value) const
{
    int res = def_value;
    if (char const *fv = get_compiler_option(ctx, name)) {
        char *end;
        long l = strtol(fv, &end, 10);
        if (*end == '\0')
            res = int(l);
    }
    return res;
}

// Get a float option.
float MDL::get_compiler_float_option(
    Thread_context const *ctx,
    char const           *name,
    float                def_value) const
{
    float res = def_value;
    if (char const *fv = get_compiler_option(ctx, name)) {
        char *end;
        float f = float(strtod(fv, &end));
        if (*end == 'F' || *end == 'f')
            ++end;
        if (*end == '\0')
            res = f;
    }
    return res;
}

// Get a double option.
double MDL::get_compiler_double_option(
    Thread_context const *ctx,
    char const           *name,
    double               def_value) const
{
    double res = def_value;
    if (char const *fv = get_compiler_option(ctx, name)) {
        char *end;
        double f = float(strtod(fv, &end));
        if (*end == '\0')
            res = f;
    }
    return res;
}

// Return the number of builtin modules.
size_t MDL::get_builtin_module_count() const
{
    return m_builtin_modules.size();
}

// Get the builtin module of given index.
Module const *MDL::get_builtin_module(size_t idx) const
{
    return m_builtin_modules[idx].get();
}

// Returns true if predefined types must be build, false otherwise.
bool MDL::build_predefined_types()
{
    if (m_predefined_types_build)
        return false;
    else {
        // Note: it is safe to run this without a lock, it runs in the
        // context of building builtin modules which is serialized.
        m_predefined_types_build = true;
        return true;
    }
}

// Get the "weak module reference lock".
mi::base::Lock &MDL::get_weak_module_lock() const
{
    // Locks might be a expensive resource, so we don't want waste one lock
    // for every module to handle the weak import table, instead place
    // one shared lock into the compiler.
    // While from logical point one lock per module would be the right granularity,
    // the protected areas are very small, so no performance loss is expected.
    return m_weak_module_lock;
}

// Get the search path lock.
mi::base::Lock &MDL::get_search_path_lock() const
{
    return m_search_path_lock;
}

// Get the Jitted code singleton.
Jitted_code *MDL::get_jitted_code()
{
    mi::base::Lock::Block block(&m_global_lock);
    {
        if (!m_jitted_code) {
            m_jitted_code = create_jitted_code_singleton(get_allocator());
        }
        return m_jitted_code;
    }
}

// Copy all messages from the given module to the compiler message list.
void MDL::copy_message(Messages_impl &dst, Module const *mod)
{
    if (mod != NULL)
        dst.copy_messages(mod->access_messages_impl());
}

// This is really ugly: We need an allocator for writing debug outputs.
static IAllocator *g_debug_log_allocator = NULL;

IAllocator *get_debug_log_allocator() { return g_debug_log_allocator; }

//-----------------------------------------------------------------------------
// Initialize MDL
//

// Initializes the mdl library and obtains the primary mdl interface.
mi::mdl::IMDL *initialize(IAllocator *allocator)
{
    if (allocator != NULL) {
        // FIXME: This creates a non-ref-counted reference!
        g_debug_log_allocator = allocator;

        return create_mdl(allocator);
    }

    mi::base::Handle<mi::base::IAllocator> alloc(
#ifdef DEBUG
        // does not work with neuray's own allocator, so we use the debug allocator
        // only if MDL uses its own allocation
        &dbgMallocAlloc
#else
        MallocAllocator::create_instance()
#endif
    );

    // FIXME: This creates a non-ref-counted reference!
    g_debug_log_allocator = alloc.get();
    return create_mdl(alloc.get());
}

} // mdl
} // mi

