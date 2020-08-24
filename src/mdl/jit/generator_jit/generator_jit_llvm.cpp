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

#include "pch.h"

#include <cstdio>
#include <cstdlib>

#include <base/system/stlext/i_stlext_restore.h>
#include <base/system/stlext/i_stlext_binary_cast.h>

#include <vector>
#include <algorithm>

#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/Triple.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/JITEventListener.h>
#include <llvm/ExecutionEngine/Orc/CompileUtils.h>
#include <llvm/ExecutionEngine/Orc/IRCompileLayer.h>
#include <llvm/ExecutionEngine/Orc/LambdaResolver.h>
#include <llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h>
#include <llvm/ExecutionEngine/SectionMemoryManager.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/DIBuilder.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Mangler.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/CodeGen.h>
#include <llvm/Support/DynamicLibrary.h>
#include <llvm/Support/FormattedStream.h>
#include <llvm/Support/ManagedStatic.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/MutexGuard.h>
#include <llvm/Support/PrettyStackTrace.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/IPO/AlwaysInliner.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include <llvm/Linker/Linker.h>

#include <mi/mdl/mdl_generated_dag.h>
#include <mi/mdl/mdl_code_generators.h>

#include "mdl/compiler/compilercore/compilercore_analysis.h"
#include "mdl/compiler/compilercore/compilercore_bitset.h"
#include "mdl/compiler/compilercore/compilercore_cc_conf.h"
#include "mdl/compiler/compilercore/compilercore_errors.h"
#include "mdl/compiler/compilercore/compilercore_tools.h"
#include "mdl/compiler/compilercore/compilercore_visitor.h"
#include "mdl/codegenerators/generator_dag/generator_dag_derivatives.h"
#include "mdl/codegenerators/generator_dag/generator_dag_lambda_function.h"
#include "mdl/codegenerators/generator_dag/generator_dag_tools.h"
#include "mdl/codegenerators/generator_dag/generator_dag_walker.h"
#include "mdl/codegenerators/generator_code/generator_code.h"

#include "generator_jit_llvm.h"
#include "generator_jit_llvm_passes.h"
#include "generator_jit_context.h"
#include "generator_jit_res_manager.h"
#include "generator_jit_streams.h"

namespace mi {
namespace mdl {

// statics
Jitted_code *Jitted_code::m_instance = NULL;
mi::base::Lock Jitted_code::m_singleton_lock;

namespace {

static bool g_shutdown_done = false;

class LLVM_shutdown_obj
{
public:
    LLVM_shutdown_obj() {}

    static void shutdown() {
        if (!g_shutdown_done) {
            g_shutdown_done = true;
            llvm::llvm_shutdown();
        }
    }

    ~LLVM_shutdown_obj() {
        shutdown();
    }
};

// handle lifetime of LLVM managed statics.
LLVM_shutdown_obj llvm_life_time;

/// RAII-like helper for handling basic block chains.
class BB_store {
public:
    /// Constructor.
    BB_store(size_t &store, size_t bb) : m_store(store) { old_bb = store; store = bb; }

    /// Destructor.
    ~BB_store() { m_store = old_bb; }

private:
    size_t &m_store;
    size_t old_bb;
};

}  // anonymous

/// Align an offset for SSE 16 byte boundary.
static size_t align_for_sse(size_t offset)
{
    return (offset + size_t(0x0F)) & ~size_t(0x0F);
}

/// Align an address for SSE 16 byte boundary.
template<typename T>
static T *align_for_sse(T *offset)
{
    return reinterpret_cast<T *>(align_for_sse(size_t(offset)));
}

// no copysignf in windows runtime
float copysignf(const float x, const float y)
{
    return MI::STLEXT::binary_cast<float>(
        (MI::STLEXT::binary_cast<unsigned int>(x) & ~(1u << 31))
      | (MI::STLEXT::binary_cast<unsigned int>(y) &  (1u << 31)));
}

// no exp2 in windows runtime
double exp2(const double x)
{
    return ::pow(2.0, x);
}

// no exp2f in windows runtime
float exp2f(const float x)
{
    return ::powf(2.0f, x);
}


/// LLVM JIT based on the BuildingAJIT tutorial and the OrcMCJITReplacement class.
/// Differences:
///  - no lazy emitting
///  - search for symbols in specific modules (avoid problems with duplicate names)
///  - don't delete module after compilation to allow printing it (only when removed from JIT)
///  - don't allow using symbols which were not explicitly added
///  - conservative locking to ensure thread-safety
class MDL_JIT {
public:
    /// Constructor.
    MDL_JIT(std::unique_ptr<llvm::TargetMachine> TM)
    : m_resolver(llvm::orc::createLegacyLookupResolver(
        m_execution_session,
        // LegacyLookup
        [this](const std::string &Name) -> llvm::JITSymbol {
            if (auto Sym = m_compile_layer.findSymbol(Name, false))
                return Sym;
            else if (auto Err = Sym.takeError())
                return std::move(Err);
            if (auto SymAddr = uint64_t(llvm::sys::DynamicLibrary::SearchForAddressOfSymbol(
                    Name.c_str())))
                return llvm::JITSymbol(SymAddr, llvm::JITSymbolFlags::Exported);
            return nullptr;
        },
        // ErrorReporter
        [](llvm::Error Err) { llvm::cantFail(std::move(Err), "lookupFlags failed"); }))
    , m_target_machine(std::move(TM))
    , m_data_layout(m_target_machine->createDataLayout())
    , m_object_layer(m_execution_session,
        // GetResources
        [this](llvm::orc::VModuleKey) {
            return llvm::orc::RTDyldObjectLinkingLayer::Resources{
                std::make_shared<llvm::SectionMemoryManager>(), m_resolver };
        })
    , m_compile_layer(
        m_object_layer,
        llvm::orc::SimpleCompiler(*m_target_machine),
        // NotifyCompiled
        [this](MDL_JIT_module_key K, std::unique_ptr<llvm::Module> module) {
            // keep module alive after compilation to allow printing it
            m_compiled_modules[K] = std::move(module);
        })
    {
    }

    /// Get the data layout of the target machine.
    llvm::DataLayout const &get_data_layout() const { return m_data_layout; }

    /// Add an LLVM module to the JIT and get its module key.
    MDL_JIT_module_key add_module(std::unique_ptr<llvm::Module> module) {
        MDL_ASSERT(!module->getDataLayout().isDefault() && "No data layout was set for module");

        llvm::MutexGuard locked(m_lock);

        // Add the module to the JIT with a new VModuleKey.
        auto K = m_execution_session.allocateVModule();
        llvm::cantFail(m_compile_layer.addModule(K, std::move(module)));
        return K;
    }

    /// Search for a symbol name in the given module.
    llvm::JITSymbol find_symbol_in(MDL_JIT_module_key key, const llvm::Twine &name) {
        std::string mangled_name;
        llvm::raw_string_ostream mangled_name_stream(mangled_name);
        llvm::Mangler::getNameWithPrefix(mangled_name_stream, name, m_data_layout);

        llvm::MutexGuard locked(m_lock);
        return m_compile_layer.findSymbolIn(key, mangled_name_stream.str(), false);
    }

    /// Get the address for a symbol name in the given module.
    llvm::JITTargetAddress get_symbol_address_in(
        MDL_JIT_module_key K,
        const llvm::Twine &name,
        LLVM_code_generator &code_gen)
    {
        llvm::Expected<uint64_t> addr = find_symbol_in(K, name).getAddress();
        if (auto error = addr.takeError()) {
            code_gen.error(GET_SYMBOL_FAILED, llvm::toString(std::move(error)));
            return llvm::JITTargetAddress(0);
        }
        return addr.get();
    }

    /// Remove the given module.
    void remove_module(MDL_JIT_module_key key) {
        llvm::MutexGuard locked(m_lock);
        cantFail(m_compile_layer.removeModule(key));
        m_compiled_modules.erase(key);
    }

private:
    /// Lock protecting all internal data structures.
    llvm::sys::Mutex m_lock;

    /// Execution session used to identify modules.
    llvm::orc::ExecutionSession m_execution_session;

    /// Resolver for linking.
    std::shared_ptr<llvm::orc::SymbolResolver> m_resolver;

    /// The target machine.
    std::unique_ptr<llvm::TargetMachine> m_target_machine;

    /// The data layout of the target machine.
    const llvm::DataLayout m_data_layout;

    /// The object linking layer.
    llvm::orc::RTDyldObjectLinkingLayer m_object_layer;

    /// The compile layer.
    llvm::orc::IRCompileLayer<llvm::orc::RTDyldObjectLinkingLayer, llvm::orc::SimpleCompiler>
        m_compile_layer;

    /// The already compiled modules.
    std::map<MDL_JIT_module_key, std::unique_ptr<llvm::Module>> m_compiled_modules;
};


bool Jitted_code::m_first_time_init = true;

// Constructor.
Jitted_code::Jitted_code(mi::mdl::IAllocator *alloc)
: Base(alloc)
, m_llvm_context(new llvm::LLVMContext())
, m_mdl_jit(NULL)
{
    llvm::TargetOptions target_options;

#if defined(__i386__) || defined(_M_IX86)
    // we may use SSE instructions, so set the stack to 16 bytes in running
    // under x86
    target_options.StackAlignmentOverride = 16;
#endif

    std::unique_ptr<llvm::Module> module(new llvm::Module("MDL global", *m_llvm_context));

    // Set the default triple here: This is only necessary for MacOS where the triple
    // contains the lowest supported runtime version.
    module->setTargetTriple(LLVM_DEFAULT_TARGET_TRIPLE);

    llvm::EngineBuilder engine_builder;
    engine_builder.setEngineKind(llvm::EngineKind::JIT)
        .setOptLevel(llvm::CodeGenOpt::Aggressive)
        .setTargetOptions(target_options);

    m_mdl_jit = new MDL_JIT(std::unique_ptr<llvm::TargetMachine>(engine_builder.selectTarget()));

    LLVM_code_generator::register_native_runtime_functions(this);
}

// Destructor.
Jitted_code::~Jitted_code()
{
    delete m_mdl_jit;
    delete m_llvm_context;

    // the singleton is deleted
    m_instance = NULL;
}

// One time LLVM initialization.
void Jitted_code::init_llvm()
{
    // Initialize targets first. Not strictly needed for the JIT itself,
    // but must be called once to the PTX backend.
    llvm::InitializeAllTargets();
    llvm::InitializeAllAsmPrinters();
    llvm::InitializeAllTargetMCs();

    // initialize our own passes
    llvm::initializeDeleteUnusedLibDevicePass(*llvm::PassRegistry::getPassRegistry());

    m_first_time_init = false;
}

// Get the only instance.
Jitted_code *Jitted_code::get_instance(IAllocator *alloc)
{
    mi::base::Lock::Block lock(&m_singleton_lock);

    if (m_instance != NULL) {
        m_instance->retain();
        return m_instance;
    }

    if (m_first_time_init) {
        init_llvm();
    }

    Allocator_builder builder(alloc);
    m_instance = builder.create<Jitted_code>(alloc);

    return m_instance;
}

/// Registers a native function for symbol resolving by the JIT.
void Jitted_code::register_function(llvm::StringRef const &func_name, void *address)
{
    std::string mangled_name;
    llvm::raw_string_ostream mangled_name_stream(mangled_name);
    llvm::Mangler::getNameWithPrefix(mangled_name_stream, func_name, m_mdl_jit->get_data_layout());

    llvm::sys::DynamicLibrary::AddSymbol(mangled_name_stream.str(), address);
}

// Get the layout data for the current JITer target.
llvm::DataLayout Jitted_code::get_layout_data() const
{
    return m_mdl_jit->get_data_layout();
}

// Helper: add this LLVM module to the execution engine.
MDL_JIT_module_key Jitted_code::add_llvm_module(llvm::Module *llvm_module)
{
    return m_mdl_jit->add_module(std::unique_ptr<llvm::Module>(llvm_module));
}

// Helper: remove this module from the execution engine and delete it.
void Jitted_code::delete_llvm_module(MDL_JIT_module_key module_key)
{
    m_mdl_jit->remove_module(module_key);
}

// JIT compile the given LLVM function.
void *Jitted_code::jit_compile(
    MDL_JIT_module_key module_key,
    llvm::Function *func,
    LLVM_code_generator &code_gen)
{
    return (void *)(m_mdl_jit->get_symbol_address_in(module_key, func->getName(), code_gen));
}

// ----------------------------- Internal_function class -----------------------------

// Constructor for an internal function.
Internal_function::Internal_function(
    Memory_arena *arena,
    char const *name,
    char const *mangled_name,
    Kind kind,
    Flags flags,
    llvm::Type *ret_type,
    Array_ref<IType const *> const &param_types,
    Array_ref<char const *> const &param_names)
    : m_name(Arena_strdup(*arena, name))
    , m_mangled_name(Arena_strdup(*arena, mangled_name))
    , m_kind(kind)
    , m_flags(flags)
    , m_ret_type(ret_type)
    , m_param_types(*arena, param_types.size())
    , m_param_names(*arena, param_names.size())
{
    MDL_ASSERT(param_types.size() == param_names.size());

    for (size_t i = 0, n = param_types.size(); i < n; ++i)
    {
        m_param_types[i] = param_types[i];
        m_param_names[i] = Arena_strdup(*arena, param_names[i]);
    }
}

// Get the number of parameters.
size_t Internal_function::get_parameter_number() const
{
    return m_param_types.size();
}


// Get the parameter type of the internal function at the given index.
IType const *Internal_function::get_parameter_type(size_t index) const
{
    return m_param_types[index];
}

// Get the parameter names of the internal function.
char const *Internal_function::get_parameter_name(size_t index) const
{
    return m_param_names[index];
}

// ------------------------------- OOB_location helper -------------------------------

// Constructor.
Exc_location::Exc_location(
    LLVM_code_generator const &code_gen,
    mi::mdl::Position const   *pos)
: m_mod(NULL)
, m_line(0)
{
    if (pos != NULL) {
        mi::mdl::IModule const *mod = code_gen.tos_module();
        m_mod  = mod;
        m_line = pos->get_start_line();
    }
}

// ------------------------- Expression_result helper -------------------------

// Return the value.
llvm::Value *Expression_result::as_value(Function_context &context) {
    // turn offset result into value
    if (m_res_kind == RK_OFFSET) {
        int cur_offs = 0;
        m_content = context.get_code_gen().translate_ro_data_segment_hlsl_value(
            context, m_offset_res_mdl_type, cur_offs, m_content);
        m_res_kind = RK_VALUE;
    }

    if (m_res_kind == RK_VALUE) {
        return m_content;
    } else {  // RK_POINTER
        // do not add debug info here, it is not clear, when this is executed
        return context->CreateLoad(m_content);
    }
}

// ------------------------------- ICall helper -------------------------------

/// Implements the ICall_expr interface for AST calls.
class Call_ast_expr : public ICall_expr {
public:
    /// Constructor.
    /*implicit*/ Call_ast_expr(mi::mdl::IExpression_call const *call)
        : m_call(call)
    {
    }

    /// Check if the called function is an array constructor.
    bool is_array_constructor() const MDL_FINAL {
        // Note that we do not support compilation of broken code, so there must be a reference
        // here.
        mi::mdl::IExpression_reference const *ref =
            cast<mi::mdl::IExpression_reference>(m_call->get_reference());

        return ref->is_array_constructor();
    }

    /// Return the semantics of a the called function.
    mi::mdl::IDefinition::Semantics get_semantics() const MDL_FINAL
    {
        mi::mdl::IExpression_reference const *ref =
            cast<mi::mdl::IExpression_reference>(m_call->get_reference());
        if (ref->is_array_constructor())
            return mi::mdl::IDefinition::DS_INTRINSIC_DAG_ARRAY_CONSTRUCTOR;

        mi::mdl::IDefinition const *def = ref->get_definition();
        return def->get_semantics();
    }

    /// Get the callee definition of one exists.
    ///
    /// \param code_gen  the LLVM code generator
    mi::mdl::IDefinition const *get_callee_definition(
        LLVM_code_generator &code_gen) const MDL_FINAL
    {
        mi::mdl::IExpression_reference const *ref =
            cast<mi::mdl::IExpression_reference>(m_call->get_reference());
        if (ref->is_array_constructor())
            return NULL;
        mi::mdl::IDefinition const *ref_def = ref->get_definition();
        mi::mdl::IModule const *mod = code_gen.tos_module();
        return mod->get_original_definition(ref_def);
    }

    /// Get the number of arguments.
    size_t get_argument_count() const MDL_FINAL { return m_call->get_argument_count(); }

    /// Translate the i'th argument.
    ///
    /// \param code_gen  the LLVM code generator
    /// \param ctx       the current function context
    /// \param i         the argument index
    Expression_result translate_argument(
        LLVM_code_generator &code_gen,
        Function_context    &ctx,
        size_t              i,
        bool                return_derivs) const MDL_FINAL
    {
        mi::mdl::IExpression const *arg = m_call->get_argument(int(i))->get_argument_expr();

        return code_gen.translate_expression(ctx, arg, return_derivs);
    }

    /// Get the LLVM context data of the callee.
    ///
    /// \param code_gen       the LLVM code generator
    /// \param args           the argument mappings for this function call
    /// \param return_derivs  if true, the function should return derivatives
    LLVM_context_data *get_callee_context(
        LLVM_code_generator                      &code_gen,
        Function_instance::Array_instances const &args,
        bool                                     return_derivs) const MDL_FINAL
    {
        mi::mdl::IExpression_reference const *ref     =
            cast<mi::mdl::IExpression_reference>(m_call->get_reference());
        mi::mdl::IDefinition const           *ref_def = ref->get_definition();

        // might be from another module, so look it up
        mi::mdl::IModule const *mod = code_gen.tos_module();
        mi::base::Handle<mi::mdl::IModule const> owner(mod->get_owner_module(ref_def));
        {
            mi::mdl::IDefinition const *def = mod->get_original_definition(ref_def);

            // skip presets, so always get the original function
            def = skip_presets(def, owner);

            Function_instance inst(def, args, return_derivs);
            return code_gen.get_or_create_context_data(owner.get(), inst);
        }
    }

    /// Get the result type of the call.
    mi::mdl::IType const *get_type() const MDL_FINAL { return m_call->get_type(); }

    /// Get the type of the i'th call argument.
    ///
    /// \param ctx  the current function context
    /// \param i  the argument index
    mi::mdl::IType const *get_argument_type(size_t i) const MDL_FINAL {
        mi::mdl::IExpression const *arg = m_call->get_argument(int(i))->get_argument_expr();

        return arg->get_type();
    }

    /// Get the source position of the i'th call argument.
    ///
    /// \param i  the argument index
    mi::mdl::Position const *get_argument_position(size_t i) const MDL_FINAL {
        mi::mdl::IArgument const   *arg  = m_call->get_argument(i);
        mi::mdl::IExpression const *expr = arg->get_argument_expr();
        return &expr->access_position();
    }

    /// Get the i'th call argument if it is a constant.
    ///
    /// \param i  the argument index
    ///
    /// \returns the value of the i'th argument if it is a constant, NULL otherwise
    mi::mdl::IValue const *get_const_argument(size_t i) const MDL_FINAL {
        mi::mdl::IExpression_literal const *lit =
            as<mi::mdl::IExpression_literal>(m_call->get_argument(i)->get_argument_expr());
        if (lit != NULL)
            return lit->get_value();
        return NULL;
    }

    /// If this is a DS_INTRINSIC_DAG_FIELD_ACCESS, the accessed field index, else -1.
    int get_field_index(
        LLVM_code_generator &code_gen,
        mi::mdl::IAllocator *alloc) const MDL_FINAL
    {
        // does never occur on AST
        return -1;
    }

    /// Translate this call as a boolean condition.
    /// If this is a ternary operator call, translate the first argument.
    ///
    /// \param code_gen  the LLVM code generator
    /// \param ctx       the current function context
    /// \param true_bb   branch target for the true case
    /// \param false_bb  branch target for the false case
    void translate_boolean_branch(
        LLVM_code_generator &code_gen,
        Function_context    &ctx,
        llvm::BasicBlock    *true_bb,
        llvm::BasicBlock    *false_bb) const MDL_FINAL
    {
        mi::mdl::IExpression const *cond;
        if (get_semantics() == operator_to_semantic(IExpression::OK_TERNARY))
            cond = m_call->get_argument(0)->get_argument_expr();
        else
            cond = m_call;

        code_gen.translate_boolean_branch(
            ctx,
            cond,
            true_bb,
            false_bb);
    }

    /// If possible, convert into a DAG_call node.
    DAG_call const *as_dag_call() const MDL_FINAL { return NULL; }

    /// If possible, convert into an AST expression.
    mi::mdl::IExpression_call const *as_expr_call() const MDL_FINAL { return m_call; }

    /// Get the source position of this call itself.
    mi::mdl::Position const *get_position() const MDL_FINAL
    {
        return &m_call->access_position();
    }

    /// Returns true, if the call should return derivatives.
    bool returns_derivatives(LLVM_code_generator &code_gen) const MDL_FINAL
    {
        return code_gen.is_deriv_expr(m_call);
    }

private:
    /// The AST expression.
    mi::mdl::IExpression_call const *m_call;
};

/// Implements the ICall_expr interface for DAG calls.
class Call_dag_expr : public ICall_expr {
public:
    /// Constructor.
    ///
    /// \param call      the DAG call node that is wrapped
    /// \param resolver  a name resolver that will be used for this call node
    Call_dag_expr(mi::mdl::DAG_call const *call, mi::mdl::ICall_name_resolver const *resolver)
    : m_call(call)
    , m_resolver(resolver)
    {
    }

    /// Check if the called function is an array constructor.
    bool is_array_constructor() const MDL_FINAL {
        return m_call->get_semantic() == mi::mdl::IDefinition::DS_INTRINSIC_DAG_ARRAY_CONSTRUCTOR;
    }

    /// Return the semantics of a the called function.
    mi::mdl::IDefinition::Semantics get_semantics() const MDL_FINAL {
        return m_call->get_semantic();
    }

    /// Get the callee definition of one exists.
    ///
    /// \param code_gen  the LLVM code generator
    mi::mdl::IDefinition const *get_callee_definition(
        LLVM_code_generator &) const MDL_FINAL
    {
        char const *signature = m_call->get_name();
        if (signature[0] == '#') {
            // skip prefix for derivative variants
            ++signature;
        }
        mi::base::Handle<mi::mdl::IModule const> mod(m_resolver->get_owner_module(signature));
        if (! mod.is_valid_interface())
            return NULL;
        mi::mdl::Module const *module = impl_cast<mi::mdl::Module>(mod.get());
        return module->find_signature(signature, /*only_exported=*/false);
    }

    /// Get the number of arguments.
    size_t get_argument_count() const MDL_FINAL {
        return m_call->get_argument_count();
    }

    /// Translate the i'th argument.
    ///
    /// \param code_gen       the LLVM code generator
    /// \param ctx            the current function context
    /// \param i              the argument index
    /// \param return_derivs  true, iff the user of the argument expects a derivative value.
    ///                       This is not used on the DAG, as the DAG nodes know, whether they
    ///                       should return derivatives or not.
    Expression_result translate_argument(
        LLVM_code_generator &code_gen,
        Function_context    &ctx,
        size_t              i,
        bool                return_derivs) const MDL_FINAL
    {
        mi::mdl::DAG_node const *arg = m_call->get_argument(int(i));

        return code_gen.translate_node(ctx, arg, m_resolver);
    }

    /// Get the LLVM context data of the callee.
    ///
    /// \param code_gen       the LLVM code generator
    /// \param args           the argument mappings for this function call
    /// \param return_derivs  if true, the function should return derivatives
    LLVM_context_data *get_callee_context(
        LLVM_code_generator                      &code_gen,
        Function_instance::Array_instances const &args,
        bool                                     return_derivs) const MDL_FINAL
    {
        char const *signature = m_call->get_name();
        if (signature[0] == '#') {
            // skip prefix for derivative variants
            ++signature;
        }
        mi::base::Handle<mi::mdl::IModule const> mod(m_resolver->get_owner_module(signature));
        if (! mod.is_valid_interface())
            return NULL;
        mi::mdl::Module const *module = impl_cast<mi::mdl::Module>(mod.get());

        mi::mdl::IDefinition const *def =
            module->find_signature(signature, /*only_exported=*/false);
        if (def != NULL) {
            // skip presets
            def    = skip_presets(def, mod);
            module = impl_cast<mi::mdl::Module>(mod.get());

            // enter this module, so we gen create the context data if it does not exists yet
            LLVM_code_generator::MDL_module_scope scope(code_gen, module);
            Function_instance inst(def, args, return_derivs);
            return code_gen.get_or_create_context_data(module, inst);
        }
        return NULL;
    }

    /// Get the result type of the call.
    mi::mdl::IType const *get_type() const MDL_FINAL { return m_call->get_type(); }

    /// Get the type of the i'th call argument.
    ///
    /// \param i  the argument index
    mi::mdl::IType const *get_argument_type(size_t i) const MDL_FINAL {
        mi::mdl::DAG_node const *arg = m_call->get_argument(int(i));

        return arg->get_type();
    }

    /// Get the source position of the i'th call argument.
    ///
    /// \param i  the argument index
    mi::mdl::Position const *get_argument_position(size_t i) const MDL_FINAL {
        // DAGs have no position yet
        return NULL;
    }

    /// Get the i'th call argument if it is a constant.
    ///
    /// \param i  the argument index
    ///
    /// \returns the value of the i'th argument if it is a constant, NULL otherwise
    mi::mdl::IValue const *get_const_argument(size_t i) const MDL_FINAL {
        mi::mdl::DAG_constant const *c = as<mi::mdl::DAG_constant>(m_call->get_argument(i));
        if (c != NULL)
            return c->get_value();
        return NULL;
    }

    /// If this is a DS_INTRINSIC_DAG_FIELD_ACCESS, the accessed field, else -1.
    int get_field_index(
        LLVM_code_generator &code_gen,
        mi::mdl::IAllocator *alloc) const MDL_FINAL
    {
        if (m_call->get_semantic() == mi::mdl::IDefinition::DS_INTRINSIC_DAG_FIELD_ACCESS) {
            char const *call_name = m_call->get_name();

            char const *s = call_name;
            if (call_name[0] == ':' && call_name[1] == ':' &&
                    (call_name[2] =='/' ||
                        (isalpha(call_name[2]) && call_name[3] == ':' && call_name[4] == '/'))) {
                // found an MDLE prefix, skip it
                s = strstr(s, ".mdle::");
                if (s != NULL) {
                    s += 7;
                } else {
                    s = call_name;
                }
            }

            char const *p = strchr(s, '.');

            if (p != NULL) {
                string f_name(alloc);

                ++p;
                if (char const *n = strchr(p, '('))
                    f_name = string(p, n - p, alloc);
                else
                    f_name = string(p, alloc);

                mi::mdl::IType const *arg_type = get_argument_type(0)->skip_type_alias();
                arg_type = code_gen.get_type_mapper().skip_deriv_type(arg_type);

                switch (arg_type->get_kind()) {
                case mi::mdl::IType::TK_STRUCT:
                    {
                        mi::mdl::IType_struct const *s_type = cast<mi::mdl::IType_struct>(arg_type);
                        for (int i = 0, n = s_type->get_field_count(); i < n; ++i) {
                            mi::mdl::ISymbol const *s;
                            mi::mdl::IType const   *f;

                            s_type->get_field(i, f, s);

                            if (s->get_name() == f_name) {
                                return i;
                            }
                        }
                    }
                    break;
                case  mi::mdl::IType::TK_VECTOR:
                    {
                        int index = -1;
                        switch (f_name[0]) {
                        case 'x': index = 0; break;
                        case 'y': index = 1; break;
                        case 'z': index = 2; break;
                        case 'w': index = 3; break;
                        default:
                            break;
                        }
                        return index;
                    }
                    break;
                default:
                    break;
                }
            }
        }
        return -1;
    }

    /// Translate this call as a boolean condition.
    /// If this is a ternary operator call, translate the first argument.
    ///
    /// \param code_gen  the LLVM code generator
    /// \param ctx       the current function context
    /// \param true_bb   branch target for the true case
    /// \param false_bb  branch target for the false case
    void translate_boolean_branch(
        LLVM_code_generator &code_gen,
        Function_context    &ctx,
        llvm::BasicBlock    *true_bb,
        llvm::BasicBlock    *false_bb) const MDL_FINAL
    {
        DAG_node const *cond;
        if (get_semantics() == operator_to_semantic(IExpression::OK_TERNARY))
            cond = m_call->get_argument(int(0));
        else
            cond = m_call;

        code_gen.translate_boolean_branch(
            ctx,
            m_resolver,
            cond,
            true_bb,
            false_bb);
    }

    /// If possible, convert into a DAG_call node.
    DAG_call const *as_dag_call() const MDL_FINAL { return m_call; }

    /// If possible, convert into an AST expression.
    virtual mi::mdl::IExpression_call const *as_expr_call() const MDL_FINAL { return NULL; }

    /// Get the source position of this call itself.
    mi::mdl::Position const *get_position() const MDL_FINAL
    {
        // DAGs have no position yet
        return NULL;
    }

    /// Returns true, if the call should return derivatives.
    bool returns_derivatives(LLVM_code_generator &code_gen) const MDL_FINAL
    {
        return code_gen.get_type_mapper().is_deriv_type(m_call->get_type()->skip_type_alias());
    }

    /// Replace the call node.
    ///
    /// \param call  the new call node
    void replace(DAG_call const *call) { m_call = call; }

private:
    /// The DAG expression.
    mi::mdl::DAG_call const *m_call;

    /// The entity name resolver.
    mi::mdl::ICall_name_resolver const *m_resolver;
};

template<>
Call_dag_expr const *impl_cast(ICall_expr const *expr) {
    if (expr->as_dag_call() != NULL)
        return static_cast<Call_dag_expr const *>(expr);
    return NULL;
}


// ------------------------------- State usage analysis ------------------------------

// Constructor.
State_usage_analysis::State_usage_analysis(LLVM_code_generator &code_gen)
: m_code_gen(code_gen)
, m_arena(code_gen.get_allocator())
, m_arena_builder(m_arena)
, m_module_state_usage(0)
, m_func_state_usage_info_map(code_gen.get_allocator())
{
}

// Register a function to take part in the analysis.
void State_usage_analysis::register_function(llvm::Function *func)
{
    Function_state_usage_info *info =
        m_arena_builder.create<Function_state_usage_info>(&m_arena);
    m_func_state_usage_info_map[func] = info;
}

// Add a state usage flag to the currently compiled function.
void State_usage_analysis::add_state_usage(llvm::Function *func, State_usage flag_to_add)
{
    m_module_state_usage |= flag_to_add;

    Function_state_usage_info_map::iterator it = m_func_state_usage_info_map.find(func);
    if (it == m_func_state_usage_info_map.end()) {
        MDL_ASSERT(!"Function not registered for state usage info");
        return;
    }

    it->second->state_usage |= flag_to_add;
}

// Add a call to the call graph.
void State_usage_analysis::add_call(llvm::Function *caller, llvm::Function *callee)
{
    Function_state_usage_info_map::iterator it = m_func_state_usage_info_map.find(caller);
    if (it == m_func_state_usage_info_map.end()) {
        MDL_ASSERT(!"Function not registered for state usage info");
        return;
    }

    it->second->called_funcs.insert(callee);
}

// Updates the state usage of the exported functions of the code generator.
void State_usage_analysis::update_exported_functions_state_usage()
{
    for (LLVM_code_generator::Exported_function &exported_func : m_code_gen.m_exported_func_list) {
        llvm::SmallPtrSet<llvm::Function *, 16> visited;
        llvm::SmallVector<llvm::Function *, 16> worklist;
        worklist.push_back(exported_func.func);

        while (!worklist.empty()) {
            llvm::Function *cur = worklist.pop_back_val();
            if (visited.count(cur))
                continue;
            visited.insert(cur);
            Function_state_usage_info const *info = m_func_state_usage_info_map[cur];
            exported_func.state_usage |= info->state_usage;
            worklist.append(info->called_funcs.begin(), info->called_funcs.end());
        }
    }
}


// ------------------------------- LLVM code generator -------------------------------

static unsigned map_target_lang(
    LLVM_code_generator::Target_language lang,
    Type_mapper::Type_mapping_mode       def_mode)
{
    switch (lang) {
    case LLVM_code_generator::TL_NATIVE:
        return def_mode;
    case LLVM_code_generator::TL_PTX:
        return Type_mapper::TM_PTX;
    case LLVM_code_generator::TL_HLSL:
        return Type_mapper::TM_HLSL | Type_mapper::TM_STRINGS_ARE_IDS;
    }
    MDL_ASSERT(!"unsupported target language");
    return def_mode;
}

// Constructor.
LLVM_code_generator::LLVM_code_generator(
    Jitted_code        *jitted_code,
    MDL                *compiler,
    Messages_impl      &messages,
    llvm::LLVMContext  &context,
    Target_language    target_lang,
    Type_mapping_mode  tm_mode,
    unsigned           sm_version,
    bool               has_tex_handler,
    State_subset_mode  state_mode,
    unsigned           num_texture_spaces,
    unsigned           num_texture_results,
    Options_impl const &options,
    bool               incremental,
    unsigned           state_mapping,
    IResource_manager  *res_manag,
    bool               enable_debug)
: m_arena(jitted_code->get_allocator())
, m_arena_builder(m_arena)
, m_transformer(NULL)
, m_trans_builder(NULL)
, m_llvm_context(context)
, m_state_mode(state_mode)
, m_internal_space(options.get_string_option(MDL_CG_OPTION_INTERNAL_SPACE))
, m_fold_meters_per_scene_unit(options.get_bool_option(MDL_CG_OPTION_FOLD_METERS_PER_SCENE_UNIT))
, m_meters_per_scene_unit(options.get_float_option(MDL_CG_OPTION_METERS_PER_SCENE_UNIT))
, m_wavelength_min(options.get_float_option(MDL_CG_OPTION_WAVELENGTH_MIN))
, m_wavelength_max(options.get_float_option(MDL_CG_OPTION_WAVELENGTH_MAX))
, m_res_manager(res_manag)
, m_texture_table(jitted_code->get_allocator())
, m_light_profile_table(jitted_code->get_allocator())
, m_bsdf_measurement_table(jitted_code->get_allocator())
, m_string_table(jitted_code->get_allocator())
, m_jitted_code(mi::base::make_handle_dup(jitted_code))
, m_compiler(mi::base::make_handle_dup(compiler))
, m_messages(messages)
, m_module(NULL)
, m_exported_func_list(jitted_code->get_allocator())
, m_user_state_module(options.get_binary_option(MDL_JIT_BINOPTION_LLVM_STATE_MODULE))
, m_renderer_module(options.get_binary_option(MDL_JIT_BINOPTION_LLVM_RENDERER_MODULE))
, m_visible_functions(options.get_string_option(MDL_JIT_OPTION_VISIBLE_FUNCTIONS))
, m_func_pass_manager()
, m_fast_math(options.get_bool_option(MDL_JIT_OPTION_FAST_MATH))
, m_enable_ro_segment(
    target_lang == TL_HLSL || options.get_bool_option(MDL_JIT_OPTION_ENABLE_RO_SEGMENT))
, m_finite_math(false)
, m_reciprocal_math(false)
, m_always_inline(options.get_bool_option(MDL_JIT_OPTION_INLINE_AGGRESSIVELY))
, m_hlsl_use_resource_data(options.get_bool_option(MDL_JIT_OPTION_HLSL_USE_RESOURCE_DATA))
, m_in_intrinsic_generator(false)
, m_runtime(create_mdl_runtime(
    m_arena_builder,
    this,
    target_lang,
    m_fast_math,
    has_tex_handler,
    m_internal_space))
, m_has_res_handler(has_tex_handler)
, m_di_builder(NULL)
, m_di_file()
, m_world_to_object(NULL)
, m_object_to_world(NULL)
, m_object_id(0)
, m_context_data(0, Context_data_map::hasher(), Context_data_map::key_equal(), get_allocator())
, m_data_layout(jitted_code->get_layout_data())  // copy the data layout without the struct cache
, m_type_mapper(
    jitted_code->get_allocator(),
    m_llvm_context,
    &m_data_layout,
    state_mapping |
        (options.get_bool_option(MDL_JIT_OPTION_TEX_RUNTIME_WITH_DERIVATIVES) ?
            Type_mapper::SM_USE_DERIVATIVES : 0) |
        (target_lang == TL_HLSL ? Type_mapper::SM_INCLUDE_ARG_BLOCK_OFFS : 0),
    Type_mapper::Type_mapping_mode(
        map_target_lang(target_lang, tm_mode) |
        (options.get_bool_option(MDL_JIT_OPTION_MAP_STRINGS_TO_IDS) ?
            Type_mapper::TM_STRINGS_ARE_IDS : 0)),
    num_texture_spaces,
    num_texture_results)
, m_module_stack(get_allocator())
, m_functions_q(Function_wait_queue::container_type(get_allocator()))
, m_node_value_map(0, Node_value_map::hasher(), Node_value_map::key_equal(), get_allocator())
, m_last_bb(0)
, m_curr_bb(get_next_bb())
, m_global_const_map(0, Global_const_map::hasher(), Global_const_map::key_equal(), get_allocator())
, m_internalized_string_map(
    0,
    Internalized_string_map::hasher(),
    Internalized_string_map::key_equal(),
    get_allocator())
, m_ro_segment(NULL)
, m_next_ro_data_offset(0)
, m_ro_data_values(jitted_code->get_allocator())
, m_scene_data_names(get_allocator())
, m_scene_data_all_pos_avail(false)
, m_optix_cp_from_id(NULL)
, m_captured_args_mdl_types(get_allocator())
, m_captured_args_type(NULL)
, m_hlsl_func_argblock_as_int(NULL)
, m_hlsl_func_argblock_as_uint(NULL)
, m_hlsl_func_argblock_as_float(NULL)
, m_hlsl_func_argblock_as_double(NULL)
, m_hlsl_func_argblock_as_bool(NULL)
, m_hlsl_func_rodata_as_int(NULL)
, m_hlsl_func_rodata_as_uint(NULL)
, m_hlsl_func_rodata_as_float(NULL)
, m_hlsl_func_rodata_as_double(NULL)
, m_hlsl_func_rodata_as_bool(NULL)
, m_hlsl_func_scene_data_lookup_int(NULL)
, m_hlsl_func_scene_data_lookup_int2(NULL)
, m_hlsl_func_scene_data_lookup_int3(NULL)
, m_hlsl_func_scene_data_lookup_int4(NULL)
, m_hlsl_func_scene_data_lookup_float(NULL)
, m_hlsl_func_scene_data_lookup_float2(NULL)
, m_hlsl_func_scene_data_lookup_float3(NULL)
, m_hlsl_func_scene_data_lookup_float4(NULL)
, m_hlsl_func_scene_data_lookup_color(NULL)
, m_hlsl_func_scene_data_lookup_deriv_float(NULL)
, m_hlsl_func_scene_data_lookup_deriv_float2(NULL)
, m_hlsl_func_scene_data_lookup_deriv_float3(NULL)
, m_hlsl_func_scene_data_lookup_deriv_float4(NULL)
, m_hlsl_func_scene_data_lookup_deriv_color(NULL)
, m_resource_tag_map(NULL)
, m_opt_level(unsigned(options.get_int_option(MDL_JIT_OPTION_OPT_LEVEL)))
, m_jit_dbg_mode(JDBG_NONE)
, m_num_texture_spaces(num_texture_spaces)
, m_num_texture_results(num_texture_results)
, m_sm_version(target_lang == TL_PTX ? sm_version : 0)
, m_min_ptx_version(0)
, m_state_usage_analysis(*this)
, m_target_lang(target_lang)
, m_enable_full_debug(enable_debug)
, m_enable_type_debug(target_lang == TL_HLSL)
, m_exported_funcs_are_entries(false)
, m_bounds_check_exception_disabled(
    target_lang != TL_NATIVE || options.get_bool_option(MDL_JIT_OPTION_DISABLE_EXCEPTIONS))
, m_divzero_check_exception_disabled(
    target_lang != TL_NATIVE || options.get_bool_option(MDL_JIT_OPTION_DISABLE_EXCEPTIONS))
, m_uses_state_param(false)
, m_mangle_name(target_lang != TL_NATIVE)
, m_enable_instancing(true)
, m_lambda_force_sret(true)  // sret is the default mode
, m_lambda_first_param_by_ref(false)
, m_lambda_force_render_state(false)
, m_lambda_force_no_lambda_results(false)
, m_use_ro_data_segment(false)
, m_link_libdevice(target_lang == TL_PTX && options.get_bool_option(MDL_JIT_OPTION_LINK_LIBDEVICE))
, m_link_libmdlrt(false)
, m_link_libbsdf_df_handle_slot_mode(parse_df_handle_slot_mode(
    options.get_string_option(MDL_JIT_OPTION_LINK_LIBBSDF_DF_HANDLE_SLOT_MODE)))
, m_incremental(incremental)
, m_texruntime_with_derivs(options.get_bool_option(MDL_JIT_OPTION_TEX_RUNTIME_WITH_DERIVATIVES))
, m_deriv_infos(NULL)
, m_cur_func_deriv_info(NULL)
, m_tex_calls_mode(parse_call_mode(
    options.get_string_option(MDL_JIT_OPTION_TEX_LOOKUP_CALL_MODE)))
, m_dist_func(NULL)
, m_dist_func_state(DFSTATE_NONE)
, m_instantiated_dfs(
    Distribution_function_state::DFSTATE_END_STATE,
    Instantiated_dfs(get_allocator()),
    get_allocator())
, m_libbsdf_template_funcs(get_allocator())
, m_enable_auxiliary(options.get_bool_option(MDL_JIT_OPTION_ENABLE_AUXILIARY))
, m_module_lambda_funcs(get_allocator())
, m_module_lambda_index_map(get_allocator())
, m_lambda_results_struct_type(NULL)
, m_lambda_result_indices(get_allocator())
, m_texture_results_struct_type(NULL)
, m_texture_result_indices(get_allocator())
, m_texture_result_offsets(get_allocator())
, m_float3_struct_type(NULL)
, m_type_bsdf_sample_func(NULL)
, m_type_bsdf_sample_data(NULL)
, m_type_bsdf_evaluate_func(NULL)
, m_type_bsdf_evaluate_data(NULL)
, m_type_bsdf_pdf_func(NULL)
, m_type_bsdf_pdf_data(NULL)
, m_type_bsdf_auxiliary_func(NULL)
, m_type_bsdf_auxiliary_data(NULL)
, m_type_edf_sample_func(NULL)
, m_type_edf_sample_data(NULL)
, m_type_edf_evaluate_func(NULL)
, m_type_edf_evaluate_data(NULL)
, m_type_edf_pdf_func(NULL)
, m_type_edf_pdf_data(NULL)
, m_type_edf_auxiliary_func(NULL)
, m_type_edf_auxiliary_data(NULL)
, m_bsdf_param_metadata_id(0)
, m_edf_param_metadata_id(0)
, m_int_func_state_set_normal(NULL)
, m_int_func_state_get_texture_results(NULL)
, m_int_func_state_get_arg_block(NULL)
, m_int_func_state_get_ro_data_segment(NULL)
, m_int_func_state_object_id(NULL)
, m_int_func_state_call_lambda_float(NULL)
, m_int_func_state_call_lambda_float3(NULL)
, m_int_func_state_call_lambda_uint(NULL)
, m_int_func_state_get_arg_block_float(NULL)
, m_int_func_state_get_arg_block_float3(NULL)
, m_int_func_state_get_arg_block_uint(NULL)
, m_int_func_state_get_arg_block_bool(NULL)
, m_int_func_state_get_measured_curve_value(NULL)
, m_int_func_df_bsdf_measurement_resolution(NULL)
, m_int_func_df_bsdf_measurement_evaluate(NULL)
, m_int_func_df_bsdf_measurement_sample(NULL)
, m_int_func_df_bsdf_measurement_pdf(NULL)
, m_int_func_df_bsdf_measurement_albedos(NULL)
, m_int_func_df_light_profile_evaluate(NULL)
, m_int_func_df_light_profile_sample(NULL)
, m_int_func_df_light_profile_pdf(NULL)
, m_next_func_name_id(0)
{
    // clear the lookup tables
    memset(m_lut_info,              0, sizeof(m_lut_info));
    memset(m_bsdf_data_texture_ids, 0, sizeof(m_bsdf_data_texture_ids));

    // clear caches
    memset(m_tex_lookup_functions, 0, sizeof(m_tex_lookup_functions));
    memset(m_optix_cps,            0, sizeof(m_optix_cps));

    char const *s;

    s = getenv("MI_MDL_JIT_OPTLEVEL");
    unsigned optlevel;
    if (s != NULL && sscanf(s, "%u", &optlevel) == 1) {
        m_opt_level = optlevel;
    }

    s = getenv("MI_MDL_JIT_DEBUG_INFO");
    if (s != NULL) {
        m_enable_full_debug = true;
    }

    s = getenv("MI_MDL_JIT_FAST_MATH");
    unsigned level;
    if (s != NULL && sscanf(s, "%u", &level) == 1) {
        m_fast_math = m_finite_math = m_reciprocal_math = false;
        if (level >= 3)
            m_fast_math = true;
        else if (level >= 2)
            m_finite_math = true;
        if (level >= 1)
            m_reciprocal_math = true;
    }

    if (target_lang == TL_PTX) {
        // Optimization level 3+ activates argument promotion. This is bad, because the NVPTX
        // backend cannot handle aggregate types passed by value. Hence limit the level to
        // 2 in this case.
        if (m_opt_level > 2)
            m_opt_level = 2;
    }

    if (target_lang != TL_HLSL) {
        // this option can only be set for HLSL
        m_hlsl_use_resource_data = false;
    }

    // parse scene data names option if available
    char const *names = options.get_string_option(MDL_JIT_OPTION_SCENE_DATA_NAMES);
    if (names != NULL && *names) {
        if (names[0] == '*' && names[1] == 0)
            m_scene_data_all_pos_avail = true;
        else {
            // split the list at ',' and put the names into a set
            char const *start_ptr = names;
            char const *ptr = start_ptr;
            while (*ptr) {
                if (*ptr == ',') {
                    size_t len = ptr - start_ptr;
                    if (len > 0) {
                        char *buf = static_cast<char *>(m_arena.allocate(len + 1));
                        memcpy(buf, start_ptr, len);
                        buf[len] = 0;
                        m_scene_data_names.insert(buf);
                    }
                    start_ptr = ptr + 1;
                }
                ++ptr;
            }
            if (start_ptr != ptr) {
                size_t len = ptr - start_ptr;
                if (len > 0) {
                    char *buf = static_cast<char *>(m_arena.allocate(len + 1));
                    memcpy(buf, start_ptr, len);
                    buf[len] = 0;
                    m_scene_data_names.insert(buf);
                }
            }
        }
    }

    prepare_internal_functions();
}

// Prepare the internal functions.
void LLVM_code_generator::prepare_internal_functions()
{
    Type_factory *type_factory = m_compiler->get_type_factory();
    IType_vector const *float2_type = type_factory->create_vector(type_factory->create_float(), 2);
    IType_vector const *float3_type = type_factory->create_vector(type_factory->create_float(), 3);
    IType_int const *int_type = type_factory->create_int();

    m_int_func_state_set_normal = m_arena_builder.create<Internal_function>(
        m_arena_builder.get_arena(),
        "::state::set_normal(float3)",
        "_ZN5state10set_normalE6float3",
        Internal_function::KI_STATE_SET_NORMAL,
        Internal_function::FL_HAS_STATE,
        /*ret_type=*/ m_type_mapper.get_void_ptr_type(),
        /*param_types=*/ Array_ref<IType const *>(float3_type),
        /*param_names=*/ Array_ref<char const *>("new_normal"));

    m_int_func_state_get_texture_results = m_arena_builder.create<Internal_function>(
        m_arena_builder.get_arena(),
        "::state::get_texture_results()",
        "_ZN5state19get_texture_resultsEv",
        Internal_function::KI_STATE_GET_TEXTURE_RESULTS,
        Internal_function::FL_HAS_STATE,
        /*ret_type=*/ m_type_mapper.get_char_ptr_type(),
        /*param_types=*/ Array_ref<IType const *>(),
        /*param_names=*/ Array_ref<char const *>());

    m_int_func_state_get_arg_block = m_arena_builder.create<Internal_function>(
        m_arena_builder.get_arena(),
        "::state::get_arg_block()",
        "_ZN5state13get_arg_blockEv",
        Internal_function::KI_STATE_GET_ARG_BLOCK,
        Internal_function::FL_HAS_CAP_ARGS,
        /*ret_type=*/ m_type_mapper.get_char_ptr_type(),
        /*param_types=*/ Array_ref<IType const *>(),
        /*param_names=*/ Array_ref<char const *>());

    m_int_func_state_get_ro_data_segment = m_arena_builder.create<Internal_function>(
        m_arena_builder.get_arena(),
        "::state::get_ro_data_segment()",
        "_ZN5state19get_ro_data_segmentEv",
        Internal_function::KI_STATE_GET_RO_DATA_SEGMENT,
        Internal_function::FL_HAS_STATE,
        /*ret_type=*/ m_type_mapper.get_char_ptr_type(),
        /*param_types=*/ Array_ref<IType const *>(),
        /*param_names=*/ Array_ref<char const *>());

    m_int_func_state_object_id = m_arena_builder.create<Internal_function>(
        m_arena_builder.get_arena(),
        "::state::object_id()",
        "_ZN5state9object_idEv",
        Internal_function::KI_STATE_OBJECT_ID,
        Internal_function::FL_HAS_STATE,
        /*ret_type=*/ m_type_mapper.get_int_type(),
        /*param_types=*/ Array_ref<IType const *>(),
        /*param_names=*/ Array_ref<char const *>());

    m_int_func_state_call_lambda_float = m_arena_builder.create<Internal_function>(
        m_arena_builder.get_arena(),
        "::state::call_lambda_float(int)",
        "_ZN5state17call_lambda_floatEi",
        Internal_function::KI_STATE_CALL_LAMBDA_FLOAT,
        Internal_function::FL_HAS_STATE |
        Internal_function::FL_HAS_RES | Internal_function::FL_HAS_EXC |
        Internal_function::FL_HAS_CAP_ARGS | Internal_function::FL_HAS_EXEC_CTX,
        /*ret_type=*/ m_type_mapper.get_float_type(),
        /*param_types=*/ Array_ref<IType const *>(int_type),
        /*param_names=*/ Array_ref<char const *>("lambda_index"));

    m_int_func_state_call_lambda_float3 = m_arena_builder.create<Internal_function>(
        m_arena_builder.get_arena(),
        "::state::call_lambda_float3(int)",
        "_ZN5state18call_lambda_float3Ei",
        Internal_function::KI_STATE_CALL_LAMBDA_FLOAT3,
        Internal_function::FL_HAS_STATE |
        Internal_function::FL_HAS_RES | Internal_function::FL_HAS_EXC |
        Internal_function::FL_HAS_CAP_ARGS | Internal_function::FL_HAS_EXEC_CTX,
        /*ret_type=*/ m_type_mapper.get_float3_type(),
        /*param_types=*/ Array_ref<IType const *>(int_type),
        /*param_names=*/ Array_ref<char const *>("lambda_index"));

    m_int_func_state_call_lambda_uint = m_arena_builder.create<Internal_function>(
        m_arena_builder.get_arena(),
        "::state::call_lambda_uint(int)",
        "_ZN5state16call_lambda_uintEi",
        Internal_function::KI_STATE_CALL_LAMBDA_UINT,
        Internal_function::FL_HAS_STATE |
        Internal_function::FL_HAS_RES | Internal_function::FL_HAS_EXC |
        Internal_function::FL_HAS_CAP_ARGS | Internal_function::FL_HAS_EXEC_CTX,
        /*ret_type=*/ m_type_mapper.get_int_type(),
        /*param_types=*/ Array_ref<IType const *>(int_type),
        /*param_names=*/ Array_ref<char const *>("lambda_index"));

    m_int_func_state_get_arg_block_float = m_arena_builder.create<Internal_function>(
        m_arena_builder.get_arena(),
        "::state::get_arg_block_float(int)",
        "_ZN5state19get_arg_block_floatEi",
        Internal_function::KI_STATE_GET_ARG_BLOCK_FLOAT,
        Internal_function::FL_HAS_STATE |
        Internal_function::FL_HAS_RES | Internal_function::FL_HAS_EXC |
        Internal_function::FL_HAS_CAP_ARGS | Internal_function::FL_HAS_EXEC_CTX,
        /*ret_type=*/ m_type_mapper.get_float_type(),
        /*param_types=*/ Array_ref<IType const *>(int_type),
        /*param_names=*/ Array_ref<char const *>("offset"));

    m_int_func_state_get_arg_block_float3 = m_arena_builder.create<Internal_function>(
        m_arena_builder.get_arena(),
        "::state::get_arg_block_float3(int)",
        "_ZN5state20get_arg_block_float3Ei",
        Internal_function::KI_STATE_GET_ARG_BLOCK_FLOAT3,
        Internal_function::FL_HAS_STATE |
        Internal_function::FL_HAS_RES | Internal_function::FL_HAS_EXC |
        Internal_function::FL_HAS_CAP_ARGS | Internal_function::FL_HAS_EXEC_CTX,
        /*ret_type=*/ m_type_mapper.get_float3_type(),
        /*param_types=*/ Array_ref<IType const *>(int_type),
        /*param_names=*/ Array_ref<char const *>("offset"));

    m_int_func_state_get_arg_block_uint = m_arena_builder.create<Internal_function>(
        m_arena_builder.get_arena(),
        "::state::get_arg_block_uint(int)",
        "_ZN5state18get_arg_block_uintEi",
        Internal_function::KI_STATE_GET_ARG_BLOCK_UINT,
        Internal_function::FL_HAS_STATE |
        Internal_function::FL_HAS_RES | Internal_function::FL_HAS_EXC |
        Internal_function::FL_HAS_CAP_ARGS | Internal_function::FL_HAS_EXEC_CTX,
        /*ret_type=*/ m_type_mapper.get_int_type(),
        /*param_types=*/ Array_ref<IType const *>(int_type),
        /*param_names=*/ Array_ref<char const *>("offset"));

    m_int_func_state_get_arg_block_bool = m_arena_builder.create<Internal_function>(
        m_arena_builder.get_arena(),
        "::state::get_arg_block_bool(int)",
        "_ZN5state18get_arg_block_boolEi",
        Internal_function::KI_STATE_GET_ARG_BLOCK_BOOL,
        Internal_function::FL_HAS_STATE |
        Internal_function::FL_HAS_RES | Internal_function::FL_HAS_EXC |
        Internal_function::FL_HAS_CAP_ARGS | Internal_function::FL_HAS_EXEC_CTX,
        /*ret_type=*/ m_type_mapper.get_bool_type(),
        /*param_types=*/ Array_ref<IType const *>(int_type),
        /*param_names=*/ Array_ref<char const *>("offset"));

    IType const* measured_param_types[] = { int_type, int_type };
    char const* measured_param_names[] = { "measured_curve_idx", "value_idx" };
    m_int_func_state_get_measured_curve_value = m_arena_builder.create<Internal_function>(
        m_arena_builder.get_arena(),
        "::state::get_measured_curve_value(int,int)",
        "_ZN5state24get_measured_curve_valueERK6float3ii",
        Internal_function::KI_STATE_GET_MEASURED_CURVE_VALUE,
        Internal_function::FL_HAS_STATE |
        Internal_function::FL_HAS_RES | Internal_function::FL_HAS_EXC |
        Internal_function::FL_HAS_CAP_ARGS | Internal_function::FL_HAS_EXEC_CTX,
        /*ret_type=*/ m_type_mapper.get_float3_type(),
        /*param_types=*/ Array_ref<IType const *>(measured_param_types),
        /*param_names=*/ Array_ref<char const *>(measured_param_names));

    IType const* resolution_param_types[] = { int_type, int_type };
    char const* resolution_param_names[] = { "bm_index", "part" };
    m_int_func_df_bsdf_measurement_resolution = m_arena_builder.create<Internal_function>(
        m_arena_builder.get_arena(),
        "::df::bsdf_measurement_resolution(int,int)",
        "_ZNK5State28bsdf_measurement_resolutionEii",
        Internal_function::KI_DF_BSDF_MEASUREMENT_RESOLUTION,
        Internal_function::FL_HAS_RES,
        /*ret_type=*/ m_type_mapper.get_int3_type(),
        /*param_types=*/ Array_ref<IType const *>(resolution_param_types),
        /*param_names=*/ Array_ref<char const *>(resolution_param_names));

    IType const* lookup_param_types[] = { int_type, float2_type, float2_type, int_type };
    char const* lookup_param_names[] =
        { "bm_index", "theta_phi_in", "theta_phi_out", "part" };

    m_int_func_df_bsdf_measurement_evaluate = m_arena_builder.create<Internal_function>(
        m_arena_builder.get_arena(),
        "::df::bsdf_measurement_evaluate(int,float2,float2,int)",
        "_ZNK5State25bsdf_measurement_evaluateEiRK6float2S2_i",
        Internal_function::KI_DF_BSDF_MEASUREMENT_EVALUATE,
        Internal_function::FL_HAS_RES,
        /*ret_type=*/ m_type_mapper.get_float3_type(),
        /*param_types=*/ Array_ref<IType const *>(lookup_param_types),
        /*param_names=*/ Array_ref<char const *>(lookup_param_names));

    IType const* sample_param_types[] = { int_type, float2_type, float3_type, int_type };
    char const* sample_param_names[] = { "bm_index", "theta_phi_out", "xi", "part"};
    m_int_func_df_bsdf_measurement_sample = m_arena_builder.create<Internal_function>(
        m_arena_builder.get_arena(),
        "::df::bsdf_measurement_sample(int,float2,float3,int)",
        "_ZNK5State23bsdf_measurement_sampleEiRK6float2RK6float3i",
        Internal_function::KI_DF_BSDF_MEASUREMENT_SAMPLE,
        Internal_function::FL_HAS_RES,
        /*ret_type=*/ m_type_mapper.get_float3_type(),
        /*param_types=*/ Array_ref<IType const *>(sample_param_types),
        /*param_names=*/ Array_ref<char const *>(sample_param_names));

    m_int_func_df_bsdf_measurement_pdf = m_arena_builder.create<Internal_function>(
        m_arena_builder.get_arena(),
        "::df::bsdf_measurement_pdf(int,float2,float2,int)",
        "_ZNK5State20bsdf_measurement_pdfEiRK6float2S2_i",
        Internal_function::KI_DF_BSDF_MEASUREMENT_PDF,
        Internal_function::FL_HAS_RES,
        /*ret_type=*/ m_type_mapper.get_float_type(),
        /*param_types=*/ Array_ref<IType const *>(lookup_param_types),
        /*param_names=*/ Array_ref<char const *>(lookup_param_names));

    IType const* bmidx_polar_param_types[] = { int_type, float2_type };
    char const* bmidx_polar_param_names[] = { "bm_index", "theta_phi"};
    m_int_func_df_bsdf_measurement_albedos = m_arena_builder.create<Internal_function>(
        m_arena_builder.get_arena(),
        "::df::bsdf_measurement_albedos(int,float2)",
        "_ZNK5State24bsdf_measurement_albedosEiRK6float2",
        Internal_function::KI_DF_BSDF_MEASUREMENT_ALBEDOS,
        Internal_function::FL_HAS_RES,
        /*ret_type=*/ m_type_mapper.get_float4_type(),
        /*param_types=*/ Array_ref<IType const *>(bmidx_polar_param_types),
        /*param_names=*/ Array_ref<char const *>(bmidx_polar_param_names));

    IType const* lpidx_polar_param_types[] = { int_type, float2_type };
    char const* lpidx_polar_param_names[] = { "lp_index", "theta_phi"};
    m_int_func_df_light_profile_evaluate = m_arena_builder.create<Internal_function>(
        m_arena_builder.get_arena(),
        "::df::light_profile_evaluate(int,float2)",
        "_ZNK5State22light_profile_evaluateEiRK6float2",
        Internal_function::KI_DF_LIGHT_PROFILE_EVALUATE,
        Internal_function::FL_HAS_RES,
        /*ret_type=*/ m_type_mapper.get_float_type(),
        /*param_types=*/ Array_ref<IType const *>(lpidx_polar_param_types),
        /*param_names=*/ Array_ref<char const *>(lpidx_polar_param_names));

    IType const* lpidx_xi_param_types[] = { int_type, float3_type };
    char const* lpidx_xi_param_names[] = { "lp_index", "xi"};
    m_int_func_df_light_profile_sample = m_arena_builder.create<Internal_function>(
        m_arena_builder.get_arena(),
        "::df::light_profile_sample(int,float3)",
        "_ZNK5State20light_profile_sampleEiRK6float3",
        Internal_function::KI_DF_LIGHT_PROFILE_SAMPLE,
        Internal_function::FL_HAS_RES,
        /*ret_type=*/ m_type_mapper.get_float3_type(),
        /*param_types=*/ Array_ref<IType const *>(lpidx_xi_param_types),
        /*param_names=*/ Array_ref<char const *>(lpidx_xi_param_names));

    m_int_func_df_light_profile_pdf = m_arena_builder.create<Internal_function>(
        m_arena_builder.get_arena(),
        "::df::light_profile_pdf(int,float2)",
        "_ZNK5State17light_profile_pdfEiRK6float2",
        Internal_function::KI_DF_LIGHT_PROFILE_PDF,
        Internal_function::FL_HAS_RES,
        /*ret_type=*/ m_type_mapper.get_float_type(),
        /*param_types=*/ Array_ref<IType const *>(lpidx_polar_param_types),
        /*param_names=*/ Array_ref<char const *>(lpidx_polar_param_names));
}

// Destructor.
LLVM_code_generator::~LLVM_code_generator()
{
    terminate_mdl_runtime(m_runtime);

    if (m_ro_segment != NULL) {
        get_allocator()->free(m_ro_segment);
    }
}

// Get the first (real) parameter of the given function.
llvm::Function::arg_iterator LLVM_code_generator::get_first_parameter(
    llvm::Function          *func,
    LLVM_context_data const *ctx)
{
    size_t ofs = ctx->get_func_param_offset();
    llvm::Function::arg_iterator arg_it = func->arg_begin();
    while (ofs > 0) {
        ++arg_it;
        --ofs;
    }
    return arg_it;
}

/// The extension function to add the createDeleteUnusedLibDevice pass.
static void AddDeleteUnusedLibDeviceExtension(
    llvm::PassManagerBuilder const &,
    llvm::PassManagerBase          &PM)
{
    PM.add(llvm::createDeleteUnusedLibDevicePass());
}

// Optimize an LLVM function.
bool LLVM_code_generator::optimize(llvm::Function *func)
{
    return m_func_pass_manager->run(*func);
}

// Optimize LLVM code.
bool LLVM_code_generator::optimize(llvm::Module *module)
{
    if (m_target_lang == TL_PTX) {
        // already remove any unreferenced libDevice functions to avoid
        // LLVM optimizing them for nothing
        if (m_link_libdevice) {
            llvm::legacy::PassManager mpm;
            mpm.add(llvm::createDeleteUnusedLibDevicePass());
            mpm.run(*module);
        }

        // always run the PTX reflection pass.
        // This will replace all __nvvm_reflect calls for __CUDA_FTZ by zero to not flush
        // denormal values to zero, when performing single-precision FP operations.
        // Note: To set it to a different value, set the value as module flag "nvvm-reflect-ftz".
        // Note: The NVVMReflect pass currently does not support the __CUDA_ARCH reflection option.

        llvm::legacy::FunctionPassManager fpm(module);
        fpm.add(llvm::createNVVMReflectPass());
        for (auto &func : module->functions())
            fpm.run(func);
    }

    llvm::PassManagerBuilder builder;
    builder.OptLevel = m_opt_level;
    builder.AvoidPointerPHIs = m_target_lang == TL_HLSL;

    // TODO: in PTX mode we don't use the C-library, but libdevice, this probably must
    // be registered somewhere, or libcall simplification can happen

    if (m_opt_level > 1)
        builder.Inliner = llvm::createFunctionInliningPass();
    else
        builder.Inliner = llvm::createAlwaysInlinerLegacyPass();

    if (m_target_lang == TL_PTX && m_link_libdevice) {
        // add our extra pass to remove any unused rest of libDevice after the inliner
        // and even in optlevel 0
        builder.addExtension(
            m_opt_level == 0 ?
                llvm::PassManagerBuilder::EP_EnabledOnOptLevel0 :
                llvm::PassManagerBuilder::EP_LoopOptimizerEnd,
            AddDeleteUnusedLibDeviceExtension);
    }

    llvm::legacy::PassManager mpm;
    builder.populateModulePassManager(mpm);
    return mpm.run(*module);
}

// Get an LLVM type for an MDL type.
llvm::Type *LLVM_code_generator::lookup_type(
    mdl::IType const *type,
    int              arr_size)
{
#if defined(ENABLE_ASSERT) || defined(DEBUG)
    if (m_enable_instancing) {
        // we should NEVER see a deferred array type here
        IType_array const *a_type = as<IType_array>(type);
        if (a_type != NULL) {
            MDL_ASSERT(m_in_intrinsic_generator || a_type->is_immediate_sized() || arr_size >= 0);
        }
    }
#endif
    mdl::IType::Kind df_kind = type->get_kind();
    if ((df_kind == mdl::IType::TK_BSDF || df_kind == mi::mdl::IType::TK_HAIR_BSDF)
            && m_dist_func_state != DFSTATE_NONE) {
        switch (m_dist_func_state) {
        case DFSTATE_INIT:
            return m_type_mapper.get_void_type();

        case DFSTATE_SAMPLE:
            return m_type_bsdf_sample_data;

        case DFSTATE_EVALUATE:
            return m_type_bsdf_evaluate_data;

        case DFSTATE_PDF:
            return m_type_bsdf_pdf_data;

        case DFSTATE_AUXILIARY:
            return m_type_bsdf_auxiliary_data;

        default:
            MDL_ASSERT(!"Unsupported distribution function state (bsdf)");
            break;
        }
    }

    if (type->get_kind() == mdl::IType::TK_EDF && m_dist_func_state != DFSTATE_NONE)
    {
        switch (m_dist_func_state)
        {
        case DFSTATE_INIT:
            return m_type_mapper.get_void_type();

        case DFSTATE_SAMPLE:
            return m_type_edf_sample_data;

        case DFSTATE_EVALUATE:
            return m_type_edf_evaluate_data;

        case DFSTATE_PDF:
            return m_type_edf_pdf_data;

        case DFSTATE_AUXILIARY:
            return m_type_edf_auxiliary_data;

        default:
            MDL_ASSERT(!"Unsupported distribution function state (edf)");
            break;
        }
    }

    return m_type_mapper.lookup_type(m_llvm_context, type, arr_size);
}

// Get an LLVM type for the result of a call expression.
llvm::Type *LLVM_code_generator::lookup_type_or_deriv_type(
    Function_context &ctx,
    mi::mdl::ICall_expr const *call_expr)
{
    mi::mdl::IType const *res_type = call_expr->get_type()->skip_type_alias();
    mi::mdl::IType const *base_type = m_type_mapper.skip_deriv_type(res_type);

    // For AST calls, the res_type has not been changed to a derivative type, yet
    if (mi::mdl::IExpression_call const *expr_call = call_expr->as_expr_call()) {
        if (m_cur_func_deriv_info != NULL &&
                m_cur_func_deriv_info->is_derivative_expression(expr_call)) {
            return m_type_mapper.lookup_deriv_type(res_type, ctx.instantiate_type_size(base_type));
        }
    }

    return m_type_mapper.lookup_type(
        m_llvm_context, res_type, ctx.instantiate_type_size(base_type));
}

// Get an LLVM type for the result of a expression.
// If necessary, a derivative type will be used.
llvm::Type *LLVM_code_generator::lookup_type_or_deriv_type(
    Function_context &ctx,
    mi::mdl::IExpression const *expr)
{
    mi::mdl::IType const *res_type = expr->get_type()->skip_type_alias();

    if (m_cur_func_deriv_info != NULL && m_cur_func_deriv_info->is_derivative_expression(expr))
        return m_type_mapper.lookup_deriv_type(res_type, ctx.instantiate_type_size(res_type));

    return m_type_mapper.lookup_type(m_llvm_context, res_type, ctx.instantiate_type_size(res_type));
}

// Returns true if for the given expression derivatives should be calculated.
bool LLVM_code_generator::is_deriv_expr(mi::mdl::IExpression const *expr) const
{
    return m_cur_func_deriv_info && m_cur_func_deriv_info->is_derivative_expression(expr);
}

// Returns true if for the given variable derivatives should be calculated.
bool LLVM_code_generator::is_deriv_var(mi::mdl::IDefinition const *def) const
{
    return m_cur_func_deriv_info && m_cur_func_deriv_info->is_derivative_variable(def);
}

// Check if a given type needs reference return calling convention.
bool LLVM_code_generator::need_reference_return(mi::mdl::IType const *type) const
{
    if ((type->skip_type_alias()->get_kind() == mi::mdl::IType::TK_BSDF ||
            type->skip_type_alias()->get_kind() == mi::mdl::IType::TK_HAIR_BSDF ||
            type->skip_type_alias()->get_kind() == mi::mdl::IType::TK_EDF) &&
        m_dist_func_state != DFSTATE_NONE)
    {
        // init function does not return anything
        if (m_dist_func_state == DFSTATE_INIT)
            return false;

        // the bsdf function return types are always structs, so return by reference
        return true;
    }

    return m_type_mapper.need_reference_return(type);
}

// Check if the given parameter type must be passed by reference.
bool LLVM_code_generator::is_passed_by_reference(mi::mdl::IType const *type) const
{
    if ((type->skip_type_alias()->get_kind() == mi::mdl::IType::TK_BSDF ||
            type->skip_type_alias()->get_kind() == mi::mdl::IType::TK_HAIR_BSDF ||
            type->skip_type_alias()->get_kind() == mi::mdl::IType::TK_EDF) &&
        m_dist_func_state != DFSTATE_NONE)
    {
        // init function does not return anything
        if (m_dist_func_state == DFSTATE_INIT)
            return false;

        return true;
    }

    return m_type_mapper.is_passed_by_reference(type);
}

// Drop an LLVM module and clear the layout cache.
void LLVM_code_generator::drop_llvm_module(llvm::Module *module)
{
    delete module;
}

// Get the option rt_callable_program_from_id(_64) function.
llvm::Function *LLVM_code_generator::get_optix_cp_from_id()
{
    if (m_optix_cp_from_id == NULL) {
        llvm::PointerType *void_ptr_tp = m_type_mapper.get_void_ptr_type();
        llvm::IntegerType *int_tp      = m_type_mapper.get_int_type();

        llvm::FunctionType *f_tp = llvm::FunctionType::get(
            void_ptr_tp, int_tp, /*is_VarArg=*/false);

        m_optix_cp_from_id = llvm::Function::Create(
            f_tp,
            llvm::GlobalValue::ExternalLinkage,
            "_rt_callable_program_from_id_64",
            m_module);
    }
    return m_optix_cp_from_id;
}

// Create resource attribute lookup tables if necessary.
void LLVM_code_generator::create_resource_tables(Lambda_function const &lambda)
{
    // no resource attributes available -> no attributes to fill in
    if (!lambda.has_resource_attributes()) {
        // we still need to collect the BSDF data texture IDs
        Resource_attr_map const &map = lambda.get_resource_attribute_map();
        for (Resource_attr_map::const_iterator it(map.begin()), end(map.end()); it != end; ++it) {
            Resource_tag_tuple const  &k = it->first;
            Resource_attr_entry const &e = it->second;

            switch (k.m_kind) {
            case Resource_tag_tuple::RK_SIMPLE_GLOSSY_MULTISCATTER:
            case Resource_tag_tuple::RK_BACKSCATTERING_GLOSSY_MULTISCATTER:
            case Resource_tag_tuple::RK_BECKMANN_SMITH_MULTISCATTER:
            case Resource_tag_tuple::RK_GGX_SMITH_MULTISCATTER:
            case Resource_tag_tuple::RK_BECKMANN_VC_MULTISCATTER:
            case Resource_tag_tuple::RK_GGX_VC_MULTISCATTER:
            case Resource_tag_tuple::RK_WARD_GEISLER_MORODER_MULTISCATTER:
            case Resource_tag_tuple::RK_SHEEN_MULTISCATTER:
            {
                IValue_texture::Bsdf_data_kind bsdf_data_kind =
                    bsdf_data_kind_from_kind(k.m_kind);
                if (bsdf_data_kind != IValue_texture::BDK_NONE) {
                    m_bsdf_data_texture_ids[int(bsdf_data_kind) - 1] = e.index;
                }
                break;
            }
            default:
                MDL_ASSERT(bsdf_data_kind_from_kind(k.m_kind) == IValue_texture::BDK_NONE);
                break;  // nothing to do for non BSDF data textures
            }
        }
        return;
    }

    IAllocator *alloc = m_arena.get_allocator();

    vector<Texture_attribute_entry>::Type          tex_entries(alloc);
    vector<Light_profile_attribute_entry>::Type    lp_entries(alloc);
    vector<Bsdf_measurement_attribute_entry>::Type bm_entries(alloc);

    Resource_attr_map const &map = lambda.get_resource_attribute_map();
    for (Resource_attr_map::const_iterator it(map.begin()), end(map.end()); it != end; ++it) {
        Resource_tag_tuple const  &k = it->first;
        Resource_attr_entry const &e = it->second;

        switch (k.m_kind) {
        case Resource_tag_tuple::RK_TEXTURE_GAMMA_DEFAULT:
        case Resource_tag_tuple::RK_TEXTURE_GAMMA_LINEAR:
        case Resource_tag_tuple::RK_TEXTURE_GAMMA_SRGB:
            // real texture ...
        case Resource_tag_tuple::RK_SIMPLE_GLOSSY_MULTISCATTER:
        case Resource_tag_tuple::RK_BACKSCATTERING_GLOSSY_MULTISCATTER:
        case Resource_tag_tuple::RK_BECKMANN_SMITH_MULTISCATTER:
        case Resource_tag_tuple::RK_GGX_SMITH_MULTISCATTER:
        case Resource_tag_tuple::RK_BECKMANN_VC_MULTISCATTER:
        case Resource_tag_tuple::RK_GGX_VC_MULTISCATTER:
        case Resource_tag_tuple::RK_WARD_GEISLER_MORODER_MULTISCATTER:
        case Resource_tag_tuple::RK_SHEEN_MULTISCATTER:
        {
            // ... and BSDF data textures
            if (tex_entries.size() < e.index + 1)
                tex_entries.resize(e.index + 1);
            tex_entries[e.index] =
                Texture_attribute_entry(e.valid, e.u.tex.width, e.u.tex.height, e.u.tex.depth);
            IValue_texture::Bsdf_data_kind bsdf_data_kind =
                bsdf_data_kind_from_kind(k.m_kind);
            if (bsdf_data_kind != IValue_texture::BDK_NONE) {
                m_bsdf_data_texture_ids[int(bsdf_data_kind) - 1] = e.index;
            }
            break;
        }
        case Resource_tag_tuple::RK_LIGHT_PROFILE:
            if (lp_entries.size() < e.index + 1)
                lp_entries.resize(e.index + 1);
            lp_entries[e.index] =
                Light_profile_attribute_entry(e.valid, e.u.lp.power, e.u.lp.maximum);
            break;
        case Resource_tag_tuple::RK_BSDF_MEASUREMENT:
            if (bm_entries.size() < e.index + 1)
                bm_entries.resize(e.index + 1);
            bm_entries[e.index] =
                Bsdf_measurement_attribute_entry(e.valid);
            break;
        case Resource_tag_tuple::RK_INVALID_REF:
            // no attributes to add
            break;
        default:
            MDL_ASSERT(!"unexpected resource kind");
        }
    }

    if (!tex_entries.empty()) {
        add_texture_attribute_table(tex_entries);
    }
    if (!lp_entries.empty()) {
        add_light_profile_attribute_table(lp_entries);
    }
    if (!bm_entries.empty()) {
        add_bsdf_measurement_attribute_table(bm_entries);
    }
}

// Compile all functions of a module.
llvm::Module *LLVM_code_generator::compile_module(
    mi::mdl::IModule const *module)
{
    LLVM_code_generator::MDL_module_scope scope(*this, module);

    create_module(module->get_name(), module->get_filename());

    // initialize the module with user code
    if (!init_user_modules()) {
        // drop the module and give up
        drop_llvm_module(m_module);
        return NULL;
    }

    if (m_target_lang == TL_HLSL) {
        init_hlsl_code_gen();
    }

    // Generate resource tables: these are "dummy", i.e. they contain only one invalid entry.
    // They just ensure that the generated code can be compiled, because we do not check for
    // used resources, so the resource tables are not generated.

    IAllocator *alloc = m_arena.get_allocator();

    {
        vector<Texture_attribute_entry>::Type tex_entries(alloc);
        tex_entries.push_back(Texture_attribute_entry());
        add_texture_attribute_table(tex_entries);
    }
    {
        vector<Light_profile_attribute_entry>::Type lp_entries(alloc);
        lp_entries.push_back(Light_profile_attribute_entry());
        add_light_profile_attribute_table(lp_entries);
    }
    {
        vector<Bsdf_measurement_attribute_entry>::Type bm_entries(alloc);
        bm_entries.push_back(Bsdf_measurement_attribute_entry());
        add_bsdf_measurement_attribute_table(bm_entries);
    }

    // compile all exported functions
    for (size_t i = 0, n = module->get_exported_definition_count(); i < n; ++i) {
        mi::mdl::IDefinition const *def = module->get_exported_definition(i);

        if (def->get_kind() == mi::mdl::IDefinition::DK_FUNCTION) {
            mi::mdl::IType_function const *f_tp   = cast<mi::mdl::IType_function>(def->get_type());
            mi::mdl::IType const          *ret_tp = f_tp->get_return_type();

            if (is_material_type(ret_tp)) {
                // this is a material constructor, ignore them
                continue;
            }

            mi::mdl::IDeclaration_function const *func_decl =
                cast<mi::mdl::IDeclaration_function>(def->get_declaration());

            if (func_decl != NULL) {
                // skip presets: this will insert the "real" body
                mi::base::Handle<IModule const> owner = mi::base::make_handle_dup(module);
                func_decl = skip_presets(func_decl, owner);

                // check if instantiation is needed at top level
                IType_function const *ftype = cast<IType_function>(def->get_type());
                int n_params = ftype->get_parameter_count();

                bool need_instantiation = false;
                for (int i = 0; i < n_params; ++i) {
                    ISymbol const *p_sym;
                    IType const *p_type;

                    ftype->get_parameter(i, p_type, p_sym);

                    if (IType_array const *a_tp = as<IType_array>(p_type)) {
                        if (!a_tp->is_immediate_sized()) {
                            need_instantiation = true;
                            break;
                        }
                    }
                }

                if (need_instantiation) {
                    // cannot compile
                    continue;
                }

                // this compiles WITHOUT instantiation ...
                Function_instance::Array_instances args(get_allocator());
                Function_instance inst(def, args, /*return_derivs=*/ false);

                compile_function_instance(inst, func_decl);
            }
        }
    }

    return finalize_module();
}

// Compile an constant lambda function into an LLVM Module and return the LLVM function.
llvm::Function  *LLVM_code_generator::compile_const_lambda(
    Lambda_function const      &lambda,
    ICall_name_resolver const  *resolver,
    ILambda_resource_attribute *attr,
    Float4_struct const        world_to_object[4],
    Float4_struct const        object_to_world[4],
    int                        object_id)
{
    IAllocator *alloc = m_arena.get_allocator();

    reset_lambda_state();

    // const functions return the result by reference
    m_lambda_force_sret = true;

    // const functions do not have a state parameter, set necessary data to evaluate
    // uniform state functions
    set_uniform_state(world_to_object, object_to_world, object_id);

    // runs on the CPU only, so we can disable instancing to speed up code generation
    disable_function_instancing();

    // create a module for the function
    create_module("lambda_mod", NULL);

    // initialize the module with user code
    if (!init_user_modules()) {
        // drop the module and give up
        drop_llvm_module(m_module);
        return NULL;
    }

    create_resource_tables(lambda);

    LLVM_context_data *ctx_data = get_or_create_context_data(&lambda);
    llvm::Function    *func     = ctx_data->get_function();
    unsigned          flags     = ctx_data->get_function_flags();

    if (is_always_inline_enabled())
        func->addFnAttr(llvm::Attribute::AlwaysInline);

    m_exported_func_list.push_back(
        Exported_function(
            get_allocator(),
            func,
            IGenerated_code_executable::DK_NONE,
            IGenerated_code_executable::FK_CONST,
            ~0));

    // ensure the function is finished by putting it into a block
    {
        // environment functions return color
        Function_instance inst(alloc, &lambda);
        Function_context context(alloc, *this, inst, func, flags);

        llvm::Function::arg_iterator arg_it = get_first_parameter(func, ctx_data);

        if (lambda.get_root_expr_count() > 0) {
            // add result and proj parameters: these will never be written
            llvm::Value *result = arg_it++;
            context.create_context_data(size_t(0), result, /*by_reference=*/false);
            llvm::Value *proj = arg_it++;
            context.create_context_data(size_t(1), proj, /*by_reference=*/false);
        }

        // translate function body
        Expression_result res = translate_node(context, lambda.get_body(), resolver);
        MDL_ASSERT(!get_state_param_usage() && "const function uses state");

        context.create_return(res.as_value(context));
    }

    // we expect that every lambda is only compiled once, hence there is no use in conserving
    // nodes for later usage.
    // also we want to avoid reuse of the same pointers, when DAG nodes are deleted.
    clear_dag_node_map();

    // finalize the module and store it
    if (finalize_module() != NULL) {
        return func;
    }
    return NULL;
}

// Create the argument struct for captured material parameters.
void LLVM_code_generator::create_captured_argument_struct(
    llvm::LLVMContext     &context,
    Lambda_function const &lambda)
{
    m_captured_args_mdl_types.clear();

    size_t n = lambda.get_parameter_count();
    if (n == 0) {
        m_captured_args_type = NULL;
        return;
    }

    vector<llvm::Type *>::Type members(get_allocator());
    members.reserve(n);

    for (size_t i = 0; i < n; ++i) {
        mi::mdl::IType const *t = lambda.get_parameter_type(i);

        m_captured_args_mdl_types.push_back(t);
        members.push_back(m_type_mapper.lookup_type(context, t));
    }
    m_captured_args_type = llvm::StructType::create(
        context, members, "Captured_arguments", /*is_packed=*/false);
}

// Declare an user-provided HLSL read function, which gets an int offset as parameter.
llvm::Function *LLVM_code_generator::declare_hlsl_read_func(
    llvm::Type *ret_type,
    char const *name)
{
    llvm::FunctionType *func_type = llvm::FunctionType::get(
        ret_type, m_type_mapper.get_int_type(), /*isVarArg=*/ false);

    llvm::Function *func = llvm::Function::Create(
        func_type,
        llvm::GlobalValue::ExternalLinkage,
        name,
        m_module);
    func->setDoesNotThrow();
    func->setOnlyReadsMemory();
    func->setOnlyAccessesInaccessibleMemory();

    return func;
}

// Initialize types and functions needed for HLSL.
void LLVM_code_generator::init_hlsl_code_gen()
{
    m_hlsl_func_rodata_as_int = declare_hlsl_read_func(
        m_type_mapper.get_int_type(),
        "mdl_read_rodata_as_int");

    m_hlsl_func_rodata_as_uint = declare_hlsl_read_func(
        m_type_mapper.get_int_type(),
        "mdl_read_rodata_as_uint");

    m_hlsl_func_rodata_as_bool = declare_hlsl_read_func(
        m_type_mapper.get_bool_type(),
        "mdl_read_rodata_as_bool");

    m_hlsl_func_rodata_as_float = declare_hlsl_read_func(
        m_type_mapper.get_float_type(),
        "mdl_read_rodata_as_float");

    m_hlsl_func_rodata_as_double = declare_hlsl_read_func(
        m_type_mapper.get_double_type(),
        "mdl_read_rodata_as_double");

    m_hlsl_func_argblock_as_int = declare_hlsl_read_func(
        m_type_mapper.get_int_type(),
        "mdl_read_argblock_as_int");

    m_hlsl_func_argblock_as_uint = declare_hlsl_read_func(
        m_type_mapper.get_int_type(),
        "mdl_read_argblock_as_uint");

    m_hlsl_func_argblock_as_bool = declare_hlsl_read_func(
        m_type_mapper.get_bool_type(),
        "mdl_read_argblock_as_bool");

    m_hlsl_func_argblock_as_float = declare_hlsl_read_func(
        m_type_mapper.get_float_type(),
        "mdl_read_argblock_as_float");

    m_hlsl_func_argblock_as_double = declare_hlsl_read_func(
        m_type_mapper.get_double_type(),
        "mdl_read_argblock_as_double");
}

// Compile a switch lambda function into an LLVM Module and return the LLVM function.
llvm::Function *LLVM_code_generator::compile_switch_lambda(
    bool                      incremental,
    Lambda_function const     &lambda,
    ICall_name_resolver const *resolver,
    size_t                    next_arg_block_index)
{
    reset_lambda_state();

    // switch functions return a bool
    m_lambda_force_sret         = false;

    // switch functions always includes a render state in its interface
    m_lambda_force_render_state = true;

    // the real result is returned by reference in the first parameter
    m_lambda_first_param_by_ref = true;

    create_captured_argument_struct(m_llvm_context, lambda);

    size_t n_roots = lambda.get_root_expr_count();

    // if incremental is false, no module must exists
    MDL_ASSERT(m_module == NULL || incremental == true);

    if (m_module == NULL) {
        // create a module for the function
        create_module("lambda_mod", NULL);

        // initialize the module with user code
        if (!init_user_modules()) {
            // drop the module and give up
            drop_llvm_module(m_module);
            return NULL;
        }

        if (m_target_lang == TL_HLSL) {
            init_hlsl_code_gen();
        }
    }

    create_resource_tables(lambda);

    LLVM_context_data *ctx_data = get_or_create_context_data(&lambda);
    llvm::Function    *func     = ctx_data->get_function();
    unsigned          flags     = ctx_data->get_function_flags();

    if (is_always_inline_enabled())
        func->addFnAttr(llvm::Attribute::AlwaysInline);

    m_exported_func_list.push_back(
        Exported_function(
            get_allocator(),
            func,
            IGenerated_code_executable::DK_NONE,
            IGenerated_code_executable::FK_SWITCH_LAMBDA,
            m_captured_args_type != NULL ? next_arg_block_index : ~0));

    // ensure the function is finished by putting it into a block
    {
        // switch functions return bool
        IAllocator *alloc = m_arena.get_allocator();
        Function_instance inst(alloc, &lambda);
        Function_context ctx(alloc, *this, inst, func, flags);

        llvm::Function::arg_iterator arg_it = get_first_parameter(func, ctx_data);

        if (lambda.get_root_expr_count() > 0) {
            // add result and proj parameters: these will never be written
            llvm::Value *result = arg_it++;
            ctx.create_context_data(size_t(0), result, /*by_reference=*/false);
            llvm::Value *proj = arg_it++;
            ctx.create_context_data(size_t(1), proj, /*by_reference=*/false);
        }

        if (m_texruntime_with_derivs)
            m_deriv_infos = lambda.get_derivative_infos();

        // translate function body
        llvm::BasicBlock  *end_bb = ctx.create_bb("after_switch");
        llvm::Value       *ptr    = ctx.get_context_data(size_t(0))->get_var_value();
        llvm::Value       *idx    = ctx.get_context_data(size_t(1))->get_var_value();

        llvm::SwitchInst *switch_instr = ctx->CreateSwitch(idx, end_bb, unsigned(n_roots));

        ctx->SetInsertPoint(ctx.get_unreachable_bb());

        for (size_t i = 0; i < n_roots; ++i) {
            DAG_node const *root_expr = lambda.get_root_expr(i);

            if (root_expr == NULL) {
                // was deleted
                continue;
            }

            // Clear the DAG-to-IR node map before we start a new case block:
            // we don't have a CFG in our lambda representation, hence some CSE's might
            // created nodes that are partially redundant.
            // Even more worse is that we have here no means to move redundant code
            // OUT of the the switch, but without that the dominator relation will be broken.
            // A quick fix is to clear the node map here, so the whole code is generated
            // for EVERY case (and redundant nodes are created in every case).
            //
            // TODO: An analysis could be used to detect redundant code and move it upwards.
            clear_dag_node_map();

            llvm::BasicBlock *case_bb = ctx.create_bb("case_body");

            // fall-through from previous case and switch to the new block
            ctx->CreateBr(case_bb);
            ctx->SetInsertPoint(case_bb);

            switch_instr->addCase(ctx.get_constant(int(i)), case_bb);

            llvm::Value *val    = translate_node(ctx, root_expr, resolver).as_value(ctx);
            llvm::Type  *val_tp = val->getType();

            if (val_tp == m_type_mapper.get_float_type()) {
                // iray expects that a float is returns in a float3 in all components
                val_tp = m_type_mapper.get_float3_type();

                if (llvm::isa<llvm::VectorType>(val_tp)) {
                    llvm::Value *t = llvm::Constant::getNullValue(val_tp);
                    t   = ctx->CreateInsertElement(t, val, ctx.get_constant(int(0)));
                    t   = ctx->CreateInsertElement(t, val, ctx.get_constant(int(1)));
                    val = ctx->CreateInsertElement(t, val, ctx.get_constant(int(2)));
                } else {
                    MDL_ASSERT(llvm::isa<llvm::ArrayType>(val_tp));
                    unsigned idxs[1] = { 0 };
                    llvm::Value *t = llvm::Constant::getNullValue(val_tp);
                    t   = ctx->CreateInsertValue(t, val, idxs); ++idxs[0];
                    t   = ctx->CreateInsertValue(t, val, idxs); ++idxs[0];
                    val = ctx->CreateInsertValue(t, val, idxs);
                }
            }

            // cast ptr to the right type
            llvm::Type  *val_ptr_tp = ctx.get_ptr(val->getType());
            llvm::Value *cptr       = ctx->CreateBitCast(ptr, val_ptr_tp);

            // Set the alignment of vectors and arrays to the element type alignment explicitly.
            // Otherwise LLVM will align to 16 here which we cannot guarantee in our C-interface.
            llvm::StoreInst *st = ctx->CreateStore(val, cptr);
            {
                llvm::Type *res_type = val->getType();
                if (llvm::ArrayType * a_tp = llvm::dyn_cast<llvm::ArrayType>(res_type)) {
                    res_type = a_tp->getElementType();
                }
                if (llvm::VectorType *v_tp = llvm::dyn_cast<llvm::VectorType>(res_type)) {
                    res_type = v_tp->getElementType();

                    // reduce the alignment of the store to this size
                    st->setAlignment(res_type->getPrimitiveSizeInBits() / 8);
                }
            }

            llvm::Value *res = ctx.get_constant(true);
            ctx.create_return(res);
        }

        ctx->SetInsertPoint(end_bb);

        // outside the allowed range ... return false
        llvm::Value *res = ctx.get_constant(false);
        ctx.create_return(res);
    }

    // if we are compiling with derivatives, all waiting functions need to be compiled now,
    // to give them access to the derivative infos
    if (m_deriv_infos) {
        compile_waiting_functions();
        m_deriv_infos = NULL;
    }

    // we expect that every lambda is only compiled once, hence there is no use in conserving
    // nodes for later usage.
    // also we want to avoid reuse of the same pointers, when DAG nodes are deleted.
    clear_dag_node_map();

    if (!incremental) {
        // finalize the module and store it
        if (finalize_module() != NULL) {
            return func;
        }
        return NULL;
    }
    return func;
}

// Compile a generic or environment lambda function into an LLVM Module and return the
// LLVM function.
llvm::Function *LLVM_code_generator::compile_lambda(
    bool                      incremental,
    Lambda_function const     &lambda,
    ICall_name_resolver const *resolver,
    ILambda_call_transformer  *transformer,
    size_t                    next_arg_block_index)
{
    IAllocator *alloc = m_arena.get_allocator();

    // we need to pass a DAG builder here. We could use a temporary object, but for now
    // we do not expect that a generic lambda is compiled more the once AND the lambda
    // itself is not modified, only its memory arena, so casting here is safe.
    set_call_transformer(transformer, const_cast<Lambda_function *>(&lambda));

    reset_lambda_state();

    // generic functions return the result by reference if supported
    m_lambda_force_sret = target_supports_sret_for_lambda();

    // generic functions always includes a render state in its interface
    m_lambda_force_render_state = true;

    create_captured_argument_struct(m_llvm_context, lambda);

    if (m_target_lang == TL_NATIVE) {
        // when running on the CPU, we can disable instancing to speed up code generation
        disable_function_instancing();
    }

    // if incremental is false, no module must exists
    MDL_ASSERT(m_module == NULL || incremental == true);

    if (m_module == NULL) {
        // create a module for the function
        create_module("lambda_mod", NULL);

        // initialize the module with user code
        if (!init_user_modules()) {
            // drop the module and give up
            drop_llvm_module(m_module);
            return NULL;
        }

        if (m_target_lang == TL_HLSL) {
            init_hlsl_code_gen();
        }
    }

    create_resource_tables(lambda);

    LLVM_context_data *ctx_data = get_or_create_context_data(&lambda);
    llvm::Function    *func     = ctx_data->get_function();
    unsigned          flags     = ctx_data->get_function_flags();

    if (is_always_inline_enabled())
        func->addFnAttr(llvm::Attribute::AlwaysInline);

    m_exported_func_list.push_back(
        Exported_function(
            get_allocator(),
            func,
            IGenerated_code_executable::DK_NONE,
            lambda.get_execution_context() == ILambda_function::LEC_ENVIRONMENT
                ? IGenerated_code_executable::FK_ENVIRONMENT
                : IGenerated_code_executable::FK_LAMBDA,
            m_captured_args_type != NULL ? next_arg_block_index : ~0));

    // ensure the function is finished by putting it into a block
    {
        Function_instance inst(alloc, &lambda);
        Function_context context(alloc, *this, inst, func, flags);

        llvm::Function::arg_iterator arg_it = get_first_parameter(func, ctx_data);

        if (lambda.get_root_expr_count() > 0) {
            // add result and proj parameters: these will never be written
            llvm::Value *result = arg_it++;
            context.create_context_data(size_t(0), result, /*by_reference=*/false);
            llvm::Value *proj = arg_it++;
            context.create_context_data(size_t(1), proj, /*by_reference=*/false);
        }

        // no derivatives available for any state field in environment mode
        if (m_texruntime_with_derivs &&
                lambda.get_execution_context() != ILambda_function::LEC_ENVIRONMENT)
            m_deriv_infos = lambda.get_derivative_infos();

        // translate function body
        Expression_result res = translate_node(context, lambda.get_body(), resolver);
        context.create_return(res.as_value(context));
    }

    // if we are compiling with derivatives, all waiting functions need to be compiled now,
    // to give them access to the derivative infos
    if (m_deriv_infos) {
        compile_waiting_functions();
        m_deriv_infos = NULL;
    }

    // we expect that every lambda is only compiled once, hence there is no use in conserving
    // nodes for later usage.
    // also we want to avoid reuse of the same pointers, when DAG nodes are deleted.
    clear_dag_node_map();

    if (!incremental) {
        // finalize the module and store it
        if (finalize_module() != NULL) {
            return func;
        }
        return NULL;
    }
    return func;
}

// Determine the function context flags for a function definition.
LLVM_context_data::Flags LLVM_code_generator::get_function_flags(IDefinition const *def)
{
    MDL_ASSERT(def->get_kind() == mi::mdl::IDefinition::DK_FUNCTION);

    IType_function const     *func_type = cast<mi::mdl::IType_function>(def->get_type());
    LLVM_context_data::Flags flags      = LLVM_context_data::FL_NONE;

    bool is_sret_func = need_reference_return(func_type->get_return_type());
    if (is_sret_func)
        flags |= LLVM_context_data::FL_SRET;

    // check if we need a state parameter
    bool need_render_state_param = false;

    if (m_type_mapper.state_includes_uniform_state()) {
        // need render state for all state functions
        need_render_state_param = def->get_property(mi::mdl::IDefinition::DP_USES_STATE);
    } else {
        // we can handle uniform state without a render state parameter
        need_render_state_param = def->get_property(mi::mdl::IDefinition::DP_USES_VARYING_STATE);
    }

    if (m_use_ro_data_segment) {
        // the RO data segment is stored inside the state, but we don't have an analysis yet that
        // will mark functions that need it, so we enable it always so far for user defined
        // functions
        if (def->get_semantics() == IDefinition::DS_UNKNOWN)
            need_render_state_param = true;
    }

    // always pass the render state to native functions
    if (def->get_property(mi::mdl::IDefinition::DP_IS_NATIVE))
        need_render_state_param = true;

    if (need_render_state_param)
        flags |= LLVM_context_data::FL_HAS_STATE;

    if (def->get_property(mi::mdl::IDefinition::DP_USES_TEXTURES) ||
        def->get_property(mi::mdl::IDefinition::DP_USES_SCENE_DATA) ||
        def->get_property(mi::mdl::IDefinition::DP_READ_TEX_ATTR) ||
        def->get_property(mi::mdl::IDefinition::DP_READ_LP_ATTR))
    {
        flags |= LLVM_context_data::FL_HAS_RES;
    }
    if (target_uses_exception_state_parameter() &&
        (
            def->get_property(mi::mdl::IDefinition::DP_CAN_THROW_BOUNDS) ||
            def->get_property(mi::mdl::IDefinition::DP_CAN_THROW_DIVZERO)
        ))
    {
        flags |= LLVM_context_data::FL_HAS_EXC;
    }
    if (def->get_property(mi::mdl::IDefinition::DP_USES_OBJECT_ID)) {
        if ((!need_render_state_param || !state_include_uniform_state()) &&
            m_state_mode != State_subset_mode::SSM_ENVIRONMENT)
        {
            flags |= LLVM_context_data::FL_HAS_OBJ_ID;
        }
    }
    if (def->get_property(mi::mdl::IDefinition::DP_USES_TRANSFORM)) {
        if ((!need_render_state_param || !state_include_uniform_state()) &&
            strcmp(m_internal_space, "*") != 0)
        {
            flags |= LLVM_context_data::FL_HAS_TRANSFORMS;
        }
    }

    return flags;
}

// Set LLVM function attributes which need to be consistent to avoid loosing
// them during inlining (e.g. for fast math).
void LLVM_code_generator::set_llvm_function_attributes(llvm::Function *func)
{
    if (is_fast_math_enabled())
        func->addFnAttr("unsafe-fp-math", "true");

    if (is_finite_math_enabled()) {
        func->addFnAttr("no-infs-fp-math", "true");
        func->addFnAttr("no-nans-fp-math", "true");
    }
}

// Declares an LLVM function from a MDL function instance.
LLVM_context_data *LLVM_code_generator::declare_function(
    mi::mdl::IModule const  *owner,
    Function_instance const &inst,
    char const              *name_prefix,
    bool                    is_prototype)
{
    mi::mdl::IDefinition const *def = inst.get_def();
    MDL_ASSERT(
        def->get_kind() == mi::mdl::IDefinition::DK_FUNCTION ||
        def->get_kind() == mi::mdl::IDefinition::DK_CONSTRUCTOR);

    IType_function const     *func_type = cast<mi::mdl::IType_function>(def->get_type());
    LLVM_context_data::Flags flags      = LLVM_context_data::FL_NONE;

    // instantiate the return type here
    IType const *mdl_ret_tp = func_type->get_return_type();

    // create function prototype
    int arr_size = inst.instantiate_type_size(mdl_ret_tp);
    llvm::Type *ret_tp;
    if (m_deriv_infos != NULL && inst.get_return_derivs())
        ret_tp = m_type_mapper.lookup_deriv_type(mdl_ret_tp, arr_size);
    else
        ret_tp = lookup_type(mdl_ret_tp, arr_size);
    MDL_ASSERT(ret_tp != NULL);

    llvm::Type *real_ret_tp = ret_tp;

    mi::mdl::vector<llvm::Type *>::Type arg_types(get_allocator());

    bool is_sret_func = need_reference_return(func_type->get_return_type());
    if (is_sret_func) {
        // add a hidden parameter for the struct return
        arg_types.push_back(Type_mapper::get_ptr(ret_tp));
        ret_tp = m_type_mapper.get_void_type();

        flags |= LLVM_context_data::FL_SRET;
    }

    // FIXME: if all exported functions are (possible) entry points, we must set the
    // whole
    bool is_entry_point =
        m_exported_funcs_are_entries &&
        def->get_property(mi::mdl::IDefinition::DP_IS_EXPORTED);

    // check if we need a state parameter
    bool need_render_state_param = false;

    if (m_type_mapper.state_includes_uniform_state()) {
        // need render state for all state functions
        need_render_state_param = def->get_property(mi::mdl::IDefinition::DP_USES_STATE);
    } else {
        // we can handle uniform state without a render state parameter
        need_render_state_param = def->get_property(mi::mdl::IDefinition::DP_USES_VARYING_STATE);
    }

    if (m_use_ro_data_segment) {
        // the RO data segment is stored inside the state, but we don't have an analysis yet that
        // will mark functions that need it, so we enable it always so far for user defined
        // functions
        if (def->get_semantics() == IDefinition::DS_UNKNOWN)
            need_render_state_param = true;
    }

    // always pass the render state to native functions
    if (def->get_property(mi::mdl::IDefinition::DP_IS_NATIVE))
        need_render_state_param = true;

    if (need_render_state_param) {
        // add a hidden state parameter
        arg_types.push_back(m_type_mapper.get_state_ptr_type(m_state_mode));

        flags |= LLVM_context_data::FL_HAS_STATE;
    }
    if (target_uses_resource_data_parameter() &&
        (
            def->get_property(mi::mdl::IDefinition::DP_USES_TEXTURES) ||
            def->get_property(mi::mdl::IDefinition::DP_USES_SCENE_DATA) ||
            def->get_property(mi::mdl::IDefinition::DP_READ_TEX_ATTR) ||
            def->get_property(mi::mdl::IDefinition::DP_READ_LP_ATTR)
        ))
    {
        // add a hidden resource_data parameter
        arg_types.push_back(m_type_mapper.get_res_data_pair_ptr_type());

        flags |= LLVM_context_data::FL_HAS_RES;
    }
    if (target_uses_exception_state_parameter() &&
        (
            def->get_property(mi::mdl::IDefinition::DP_CAN_THROW_BOUNDS) ||
            def->get_property(mi::mdl::IDefinition::DP_CAN_THROW_DIVZERO)
        ))
    {
        // add a hidden exc_state parameter
        arg_types.push_back(m_type_mapper.get_exc_state_ptr_type());

        flags |= LLVM_context_data::FL_HAS_EXC;
    }
    if (def->get_property(mi::mdl::IDefinition::DP_USES_OBJECT_ID)) {
        if ((!need_render_state_param || !state_include_uniform_state()) &&
            m_state_mode != State_subset_mode::SSM_ENVIRONMENT)
        {
            // add a hidden object_id parameter
            arg_types.push_back(m_type_mapper.get_int_type());

            flags |= LLVM_context_data::FL_HAS_OBJ_ID;
        }
    }
    if (def->get_property(mi::mdl::IDefinition::DP_USES_TRANSFORM)) {
        if ((!need_render_state_param || !state_include_uniform_state()) &&
            strcmp(m_internal_space, "*") != 0)
        {
            // add two hidden transform (matrix) parameters
            arg_types.push_back(m_type_mapper.get_arr_float_4_ptr_type());
            arg_types.push_back(m_type_mapper.get_arr_float_4_ptr_type());

            flags |= LLVM_context_data::FL_HAS_TRANSFORMS;
        }
    }

    string func_name(mangle(inst, name_prefix));

    Func_deriv_info const *func_deriv_info = NULL;
    if (m_deriv_infos != NULL)
        func_deriv_info = m_deriv_infos->get_function_derivative_infos(inst);

    size_t n_params = func_type->get_parameter_count();

    for (size_t i = 0; i < n_params; ++i) {
        mi::mdl::IType const   *p_type;
        mi::mdl::ISymbol const *p_sym;

        func_type->get_parameter(i, p_type, p_sym);

        // instantiate parameter types
        int arr_size = inst.instantiate_type_size(p_type);
        llvm::Type *tp;
        if (func_deriv_info != NULL && func_deriv_info->args_want_derivatives.test_bit(i + 1)) {
            tp = m_type_mapper.lookup_deriv_type(p_type, arr_size);

            // pass by reference if supported by target
            if (target_supports_pointers()) {
                arg_types.push_back(Type_mapper::get_ptr(tp));
            } else {
                arg_types.push_back(tp);
            }
        } else {
            tp = lookup_type(p_type, arr_size);

            if (is_passed_by_reference(p_type)) {
                arg_types.push_back(Type_mapper::get_ptr(tp));
            } else {
                arg_types.push_back(tp);
            }
        }
    }

    // entry points and native functions must have external linkage
    const llvm::GlobalValue::LinkageTypes linkage =
        is_entry_point || def->get_property(IDefinition::DP_IS_NATIVE) ?
        llvm::GlobalValue::ExternalLinkage :
        llvm::GlobalValue::InternalLinkage;

    llvm::Function *func = llvm::Function::Create(
        llvm::FunctionType::get(ret_tp, arg_types, false),
        linkage,
        func_name.c_str(),
        m_module);
    set_llvm_function_attributes(func);

    if (is_entry_point)
        func->setCallingConv(llvm::CallingConv::C);

    // set parameter names
    llvm::Function::arg_iterator arg_it = func->arg_begin();
    if (flags & LLVM_context_data::FL_SRET) {
        // the first argument is the struct return
        arg_it->setName("sret_ptr");
        if (!is_entry_point) {
            // FIXME: is this still true?
            //
            // the SRET attribute does not work yet (2.8) on 32bit MSVC Target,
            // because of the differences between cygwin/msys and VC API (currently
            // only the first is implemented), so don't use it.
            func->addParamAttr(arg_it->getArgNo(), llvm::Attribute::StructRet);
        } else {
            // treat the first argument as a pointer, but we could at least improve
            // the code a bit, because we "know" that the extra parameter is alias free
            func->addParamAttr(arg_it->getArgNo(), llvm::Attribute::NoAlias);
        }
        ++arg_it;
    }
    if (flags & LLVM_context_data::FL_HAS_STATE) {
        arg_it->setName("state");

        // the state pointer does not alias
        func->addParamAttr(arg_it->getArgNo(), llvm::Attribute::NoAlias);
        ++arg_it;
    }
    if (flags & LLVM_context_data::FL_HAS_RES) {
        if (target_supports_pointers())
            arg_it->setName("res_data_pair");
        else
            arg_it->setName("res_data");

        // the resource data pointer does not alias
        func->addParamAttr(arg_it->getArgNo(), llvm::Attribute::NoAlias);
        ++arg_it;
    }
    if (flags & LLVM_context_data::FL_HAS_EXC) {
        arg_it->setName("exc_state");

        // the exc_data pointer does not alias
        func->addParamAttr(arg_it->getArgNo(), llvm::Attribute::NoAlias);
        ++arg_it;
    }
    if (flags & LLVM_context_data::FL_HAS_CAP_ARGS) {
        arg_it->setName("captured_arguments");

        // the cap_args pointer does not alias
        func->addParamAttr(arg_it->getArgNo(), llvm::Attribute::NoAlias);
        ++arg_it;
    }
    if (flags & LLVM_context_data::FL_HAS_OBJ_ID) {
        arg_it->setName("object_id");
        ++arg_it;
    }
    if (flags & LLVM_context_data::FL_HAS_TRANSFORMS) {
        arg_it->setName("w2o_transform");
        ++arg_it;
        arg_it->setName("o2w_transform");
        ++arg_it;
    }

    for (size_t i = 0; i < n_params; ++i, ++arg_it) {
        mi::mdl::IType const   *p_type;
        mi::mdl::ISymbol const *p_sym;

        func_type->get_parameter(i, p_type, p_sym);

        arg_it->setName(p_sym->get_name());
    }

    if (is_prototype) {
        func->setLinkage(llvm::GlobalValue::ExternalLinkage);
    } else {
        // put this function on the wait queue
        m_functions_q.push(Wait_entry(owner, inst));

        m_state_usage_analysis.register_function(func);
    }

    return m_arena_builder.create<LLVM_context_data>(func, real_ret_tp, flags);
}

// Declares an LLVM function from a lambda function.
LLVM_context_data *LLVM_code_generator::declare_lambda(
    mi::mdl::Lambda_function const *lambda)
{
    LLVM_context_data::Flags flags = LLVM_context_data::FL_NONE;

    // create function prototype
    llvm::Type *ret_tp = lookup_type(lambda->get_return_type());
    MDL_ASSERT(ret_tp != NULL);

    llvm::Type *real_ret_tp = ret_tp;

    mi::mdl::vector<llvm::Type *>::Type arg_types(get_allocator());

    bool is_sret_func = m_lambda_force_sret || need_reference_return(lambda->get_return_type());
    if (is_sret_func) {
        // add a hidden parameter for the struct return
        arg_types.push_back(Type_mapper::get_ptr(ret_tp));
        ret_tp = m_type_mapper.get_void_type();

        // lambda function are always interfaced with the "outer-world", so
        // we cannot control alignment here
        flags |= LLVM_context_data::FL_SRET | LLVM_context_data::FL_UNALIGNED_RET;
    }

    bool is_entry_point = lambda->is_entry_point();

    // we support function entries with and without state
    if (m_lambda_force_render_state || lambda->uses_render_state()) {
        // add a hidden state parameter
        arg_types.push_back(m_type_mapper.get_state_ptr_type(m_state_mode));

        flags |= LLVM_context_data::FL_HAS_STATE;
    }
    if (target_uses_resource_data_parameter() && (is_entry_point || lambda->uses_resources())) {
        // add a hidden resource_data parameter
        arg_types.push_back(m_type_mapper.get_res_data_pair_ptr_type());

        flags |= LLVM_context_data::FL_HAS_RES;
    }
    if (target_uses_exception_state_parameter() && (is_entry_point || lambda->can_throw())) {
        // add a hidden exc_state parameter
        arg_types.push_back(m_type_mapper.get_exc_state_ptr_type());

        flags |= LLVM_context_data::FL_HAS_EXC;
    }
    if (target_supports_captured_argument_parameter() && is_entry_point) {
        // add captured arguments pointer parameter
        arg_types.push_back(m_type_mapper.get_void_ptr_type());

        flags |= LLVM_context_data::FL_HAS_CAP_ARGS;
    }
    if (!m_lambda_force_no_lambda_results && target_supports_lambda_results_parameter() &&
        lambda->uses_lambda_results())
    {
        // add lambda results parameter
        arg_types.push_back(m_type_mapper.get_void_ptr_type());

        flags |= LLVM_context_data::FL_HAS_LMBD_RES;
    }

    // lambda functions have NEITHER an object_id NOR a transform parameter

    if (lambda->get_root_expr_count() > 0) {
        // add result and proj parameters: these will never be written
        llvm::Type *tp_result = m_type_mapper.get_bool_type();

        if (m_lambda_first_param_by_ref) {
            arg_types.push_back(Type_mapper::get_ptr(tp_result));
        } else {
            arg_types.push_back(tp_result);
        }

        llvm::Type *tp_proj = m_type_mapper.get_int_type();
        arg_types.push_back(tp_proj);
    }

    llvm::GlobalValue::LinkageTypes const linkage =
        is_entry_point ?
        llvm::GlobalValue::ExternalLinkage :
        llvm::GlobalValue::InternalLinkage;

    llvm::Function *func = llvm::Function::Create(
        llvm::FunctionType::get(ret_tp, arg_types, false),
        linkage,
        lambda->get_name(),
        m_module);
    set_llvm_function_attributes(func);
    m_state_usage_analysis.register_function(func);

    if (is_entry_point)
        func->setCallingConv(llvm::CallingConv::C);

    // set parameter names
    llvm::Function::arg_iterator arg_it = func->arg_begin();
    if (flags & LLVM_context_data::FL_SRET) {
        // the first argument is the struct return
        arg_it->setName("sret_ptr");
        if (!is_entry_point) {
            // FIXME: is this still true?
            //
            // the SRET attribute does not work yet (2.8) on 32bit MSVC Target,
            // because of the differences between cygwin/msys and VC API (currently
            // only the first is implemented), so don't use it.
            func->addParamAttr(arg_it->getArgNo(), llvm::Attribute::StructRet);
        } else {
            // treat the first argument as a pointer, but we could at least improve
            // the code a bit, because we "know" that the extra parameter is alias free
            func->addParamAttr(arg_it->getArgNo(), llvm::Attribute::NoAlias);
        }
        ++arg_it;
    }
    if (flags & LLVM_context_data::FL_HAS_STATE) {
        arg_it->setName("state");

        // the state pointer does not alias
        func->addParamAttr(arg_it->getArgNo(), llvm::Attribute::NoAlias);
        ++arg_it;
    }
    if (flags & LLVM_context_data::FL_HAS_RES) {
        if (target_supports_pointers())
            arg_it->setName("res_data_pair");
        else
            arg_it->setName("res_data");

        // the resource data pointer does not alias
        func->addParamAttr(arg_it->getArgNo(), llvm::Attribute::NoAlias);
        ++arg_it;
    }
    if (flags & LLVM_context_data::FL_HAS_EXC) {
        arg_it->setName("exc_state");

        // the exc_data pointer does not alias
        func->addParamAttr(arg_it->getArgNo(), llvm::Attribute::NoAlias);
        ++arg_it;
    }
    if (flags & LLVM_context_data::FL_HAS_CAP_ARGS) {
        arg_it->setName("captured_arguments");

        // the cap_args pointer does not alias
        func->addParamAttr(arg_it->getArgNo(), llvm::Attribute::NoAlias);
        ++arg_it;
    }
    if (flags & LLVM_context_data::FL_HAS_OBJ_ID) {
        arg_it->setName("object_id");
        ++arg_it;
    }
    if (flags & LLVM_context_data::FL_HAS_TRANSFORMS) {
        arg_it->setName("w2o_transform");
        ++arg_it;
        arg_it->setName("o2w_transform");
        ++arg_it;
    }

    return m_arena_builder.create<LLVM_context_data>(func, real_ret_tp, flags);
}

// Declares an LLVM function from an MDL function instance.
LLVM_context_data *LLVM_code_generator::declare_internal_function(
    Function_instance const &inst,
    bool                     is_prototype)
{
    mi::mdl::Internal_function const *int_func =
        reinterpret_cast<mi::mdl::Internal_function const *>(inst.get_common_prototype_code());

    LLVM_context_data::Flags in_flags = LLVM_context_data::Flags(int_func->get_flags());
    LLVM_context_data::Flags flags = LLVM_context_data::FL_NONE;

    // create function prototype
    llvm::Type *ret_tp = int_func->get_return_type();
    MDL_ASSERT(ret_tp != NULL);

    llvm::Type *real_ret_tp = ret_tp;

    mi::mdl::vector<llvm::Type *>::Type arg_types(get_allocator());

    if ((in_flags & LLVM_context_data::FL_SRET) != 0 && m_type_mapper.may_use_sret()) {
        // add a hidden parameter for the struct return
        arg_types.push_back(Type_mapper::get_ptr(ret_tp));
        ret_tp = m_type_mapper.get_void_type();
        flags |= LLVM_context_data::FL_SRET;
    }

    bool is_entry_point = false;

    if (target_supports_lambda_results_parameter() &&
            (in_flags & LLVM_context_data::FL_HAS_EXEC_CTX) != 0) {
        // add execution context parameter
        arg_types.push_back(m_type_mapper.get_exec_ctx_ptr_type());
        flags |= LLVM_context_data::FL_HAS_EXEC_CTX;
    } else {
        if ((in_flags & LLVM_context_data::FL_HAS_STATE) != 0) {
            // add a hidden state parameter
            arg_types.push_back(m_type_mapper.get_state_ptr_type(m_state_mode));
            flags |= LLVM_context_data::FL_HAS_STATE;
        }
        if (target_uses_resource_data_parameter() &&
            (in_flags & LLVM_context_data::FL_HAS_RES) != 0)
        {
            // add a hidden resource_data parameter
            arg_types.push_back(m_type_mapper.get_res_data_pair_ptr_type());
            flags |= LLVM_context_data::FL_HAS_RES;
        }
        if (target_uses_exception_state_parameter() &&
            (in_flags & LLVM_context_data::FL_HAS_EXC) != 0)
        {
            // add a hidden exc_state parameter
            arg_types.push_back(m_type_mapper.get_exc_state_ptr_type());
            flags |= LLVM_context_data::FL_HAS_EXC;
        }
        if (target_supports_captured_argument_parameter() &&
            (in_flags & LLVM_context_data::FL_HAS_CAP_ARGS) != 0)
        {
            // add a hidden captured arguments parameter
            arg_types.push_back(m_type_mapper.get_char_ptr_type());
            flags |= LLVM_context_data::FL_HAS_CAP_ARGS;
        }
    }
    if ((in_flags & LLVM_context_data::FL_HAS_OBJ_ID) != 0) {
        // add a hidden object_id parameter
        arg_types.push_back(m_type_mapper.get_int_type());
        flags |= LLVM_context_data::FL_HAS_OBJ_ID;
    }
    if ((in_flags & LLVM_context_data::FL_HAS_TRANSFORMS) != 0) {
        // add two hidden transform (matrix) parameters
        arg_types.push_back(m_type_mapper.get_arr_float_4_ptr_type());
        arg_types.push_back(m_type_mapper.get_arr_float_4_ptr_type());
        flags |= LLVM_context_data::FL_HAS_TRANSFORMS;
    }

    if ((in_flags & LLVM_context_data::FL_UNALIGNED_RET) != 0)
        flags |= LLVM_context_data::FL_UNALIGNED_RET;

    if (target_supports_lambda_results_parameter() &&
        (in_flags & LLVM_context_data::FL_HAS_LMBD_RES) != 0)
    {
        flags |= LLVM_context_data::FL_HAS_LMBD_RES;
    }


    size_t n_params = int_func->get_parameter_number();
    for (size_t i = 0; i < n_params; ++i) {
        IType const *p_type = int_func->get_parameter_type(i);

        // instantiate parameter types
        llvm::Type *tp = lookup_type(p_type, inst.instantiate_type_size(p_type));

        if (is_passed_by_reference(p_type)) {
            arg_types.push_back(Type_mapper::get_ptr(tp));
        } else {
            arg_types.push_back(tp);
        }
    }

    llvm::Function *func = llvm::Function::Create(
        llvm::FunctionType::get(ret_tp, arg_types, false),
        is_prototype ? llvm::GlobalValue::ExternalLinkage : llvm::GlobalValue::InternalLinkage,
        int_func->get_mangled_name(),
        m_module);

    set_llvm_function_attributes(func);

    if (is_always_inline_enabled())
        func->addFnAttr(llvm::Attribute::AlwaysInline);

    if (is_entry_point)
        func->setCallingConv(llvm::CallingConv::C);

    // set parameter names
    llvm::Function::arg_iterator arg_it = func->arg_begin();
    if ((flags & LLVM_context_data::FL_SRET) != 0) {
        // the first argument is the struct return
        arg_it->setName("sret_ptr");
        if (!is_entry_point) {
            // FIXME: is this still true?
            //
            // the SRET attribute does not work yet (2.8) on 32bit MSVC Target,
            // because of the differences between cygwin/msys and VC API (currently
            // only the first is implemented), so don't use it.
            func->addParamAttr(arg_it->getArgNo(), llvm::Attribute::StructRet);
        } else {
            // treat the first argument as a pointer, but we could at least improve
            // the code a bit, because we "know" that the extra parameter is alias free
            func->addParamAttr(arg_it->getArgNo(), llvm::Attribute::NoAlias);
        }
        ++arg_it;
    }
    if ((flags & LLVM_context_data::FL_HAS_EXEC_CTX) != 0) {
        arg_it->setName("execution_ctx");

        // the execution context pointer does not alias
        func->addParamAttr(arg_it->getArgNo(), llvm::Attribute::NoAlias);
        ++arg_it;
    } else {
        if ((flags & LLVM_context_data::FL_HAS_STATE) != 0) {
            arg_it->setName("state");

            // the state pointer does not alias
            func->addParamAttr(arg_it->getArgNo(), llvm::Attribute::NoAlias);
            ++arg_it;
        }
        if ((flags & LLVM_context_data::FL_HAS_RES) != 0) {
            if (target_supports_pointers())
                arg_it->setName("res_data_pair");
            else
                arg_it->setName("res_data");

            // the resource data pointer does not alias
            func->addParamAttr(arg_it->getArgNo(), llvm::Attribute::NoAlias);
            ++arg_it;
        }
        if ((flags & LLVM_context_data::FL_HAS_EXC) != 0)
        {
            arg_it->setName("exc_state");

            // the exc_data pointer does not alias
            func->addParamAttr(arg_it->getArgNo(), llvm::Attribute::NoAlias);
            ++arg_it;
        }
    }
    if ((flags & LLVM_context_data::FL_HAS_OBJ_ID) != 0) {
        arg_it->setName("object_id");
        ++arg_it;
    }
    if ((flags & LLVM_context_data::FL_HAS_TRANSFORMS) != 0) {
        arg_it->setName("w2o_transform");
        ++arg_it;
        arg_it->setName("o2w_transform");
        ++arg_it;
    }

    for (size_t i = 0; i < n_params; ++i, ++arg_it) {
        arg_it->setName(int_func->get_parameter_name(i));
    }

    return m_arena_builder.create<LLVM_context_data>(func, real_ret_tp, flags);
}

// Retrieve the LLVM context data for a MDL function instance, create it if not available.
LLVM_context_data *LLVM_code_generator::get_or_create_context_data(
    mi::mdl::IModule const  *owner,
    Function_instance const &inst,
    char const              *module_name,
    bool                    is_prototype)
{
    Context_data_map::const_iterator it(m_context_data.find(inst));
    if (it != m_context_data.end()) {
        return it->second;
    }

    // not yet allocated, allocate a new one

    LLVM_context_data *ctx;
    if (inst.get_common_prototype_code() != 0) {
        ctx = declare_internal_function(inst, is_prototype);
    } else {
        mi::mdl::string name(get_allocator());
        if (owner != NULL) {
            if (!owner->is_builtins()) {
                name = owner->get_name();
                name += "::";
            }
        } else {
            name = module_name;
            name += "::";
        }

        ctx = declare_function(owner, inst, name.c_str(), is_prototype);
    }

    m_context_data[inst] = ctx;
    return ctx;
}

// Retrieve the LLVM context data for a MDL function definition, return NULL if not available.
LLVM_context_data *LLVM_code_generator::get_context_data(
    Function_instance const &func_instance)
{
    Context_data_map::const_iterator it(m_context_data.find(func_instance));
    if (it != m_context_data.end()) {
        return it->second;
    }
    MDL_ASSERT(!"Context data not found");
    return NULL;
}

// Retrieve the LLVM context data for a MDL function definition, return NULL if not available.
llvm::Function *LLVM_code_generator::get_function(
    Function_instance const &func_instance)
{
    Context_data_map::const_iterator it(m_context_data.find(func_instance));
    if (it != m_context_data.end()) {
        return it->second->get_function();
    }

    return NULL;
}

// Retrieve the LLVM context data for a lambda function.
LLVM_context_data *LLVM_code_generator::get_or_create_context_data(
    mi::mdl::Lambda_function const *lambda)
{
    // For now, lambda function will be NEVER instantiated.
    Function_instance inst(get_allocator(), lambda);

    Context_data_map::const_iterator it(m_context_data.find(inst));
    if (it != m_context_data.end()) {
        return it->second;
    }

    // not yet allocated, allocate a new one
    LLVM_context_data *ctx = declare_lambda(lambda);

    m_context_data[inst] = ctx;
    return ctx;
}

// Create an LLVM context data for an existing LLVM function.
LLVM_context_data *LLVM_code_generator::create_context_data(
    IDefinition const *def,
    bool               return_derivs,
    llvm::Function    *func)
{
    MDL_ASSERT(def != NULL && def->get_kind() == mi::mdl::IDefinition::DK_FUNCTION);
    Function_instance inst(get_allocator(), def, return_derivs);

    LLVM_context_data::Flags flags = get_function_flags(def);
    llvm::Type *real_ret_tp = (flags & LLVM_context_data::FL_SRET) != 0
        ? func->getFunctionType()->getParamType(0)
        : func->getReturnType();
    LLVM_context_data *ctx = m_arena_builder.create<LLVM_context_data>(
        func, real_ret_tp, flags);

    m_context_data[inst] = ctx;
    return ctx;
}

// Create an LLVM context data for an existing LLVM function.
LLVM_context_data *LLVM_code_generator::create_context_data(
    Internal_function const *int_func,
    llvm::Function *func)
{
    Function_instance inst(get_allocator(), reinterpret_cast<size_t>(int_func));

    LLVM_context_data::Flags flags = int_func->get_flags();
    llvm::Type *real_ret_tp = (flags & LLVM_context_data::FL_SRET) != 0
        ? func->getFunctionType()->getParamType(0)
        : func->getReturnType();
    LLVM_context_data *ctx = m_arena_builder.create<LLVM_context_data>(
        func, real_ret_tp, flags);

    m_context_data[inst] = ctx;
    return ctx;
}

// Returns true if the given variable needs storage to be allocated.
bool LLVM_code_generator::need_storage_for_var(mi::mdl::IDefinition const *var_def) const
{
    if (m_enable_full_debug || var_def->get_property(mi::mdl::IDefinition::DP_IS_WRITTEN)) {
        // need a storage
        return true;
    }
    // can be handled as a value
    return false;
}

// Compile a function instance.
void LLVM_code_generator::compile_function_instance(
    Function_instance const              &inst,
    mi::mdl::IDeclaration_function const *func_decl)
{
    LLVM_context_data *ctx_data = get_or_create_context_data(tos_module(), inst);
    llvm::Function    *func     = ctx_data->get_function();
    unsigned          flags     = ctx_data->get_function_flags();

    IDefinition const *def = inst.get_def();
    bool is_native = def != NULL && def->get_property(IDefinition::DP_IS_NATIVE);

    if (is_native) {
        // only a declaration
        return;
    }

    Function_context context(get_allocator(), *this, inst, func, flags);

    llvm::Function::arg_iterator arg_it = get_first_parameter(func, ctx_data);

    for (size_t i = 0, n = func_decl->get_parameter_count(); i < n; ++i, ++arg_it) {
        mi::mdl::IParameter const  *param  = func_decl->get_parameter(i);
        mi::mdl::IDefinition const *p_def  = param->get_name()->get_definition();
        mi::mdl::IType const       *p_type = p_def->get_type();
        bool                       by_ref  = is_passed_by_reference(p_type) ||
            (target_supports_pointers() &&
                m_cur_func_deriv_info != NULL &&
                m_cur_func_deriv_info->args_want_derivatives.test_bit(i + 1));
        LLVM_context_data          *p_data;

        if (need_storage_for_var(p_def)) {
            // must allocate a shadow copy, because this parameter MIGHT be written to
            // or debug mode is active
            p_data = context.get_context_data(p_def);

            // copy value into shadow copy
            llvm::Value *init = by_ref ?
                llvm::cast<llvm::Value>(context->CreateLoad(arg_it)) :
                llvm::cast<llvm::Value>(arg_it);
            p_data->set_var_value(init);
        } else {
            // this parameter will never be written
            p_data = context.create_context_data(p_def, arg_it, by_ref);
        }
        context.make_accessible(param);
    }

    // translate function body
    mi::mdl::IStatement const *body = func_decl->get_body();
    if (mi::mdl::IStatement_compound const *block = as<mi::mdl::IStatement_compound>(body)) {
        for (size_t i = 0, n = block->get_statement_count(); i < n; ++i) {
            mi::mdl::IStatement const *stmt = block->get_statement(i);

            translate_statement(context, stmt);
        }
    } else if (mi::mdl::IStatement_expression const *e_stmt = as<IStatement_expression>(body)) {
        // single expression body
        mi::mdl::IExpression const *expr = e_stmt->get_expression();

        llvm::Type  *return_type = context.get_return_type();
        llvm::Value *v = translate_expression_value(
            context, expr, context.is_deriv_type(return_type));
        context.create_return(v);
    }
}

// Get the top level module on the stack.
mi::mdl::IModule const *LLVM_code_generator::tos_module() const
{
    MDL_ASSERT(m_module_stack.begin() != m_module_stack.end());
    return m_module_stack.back();
}

// Push a module on the stack.
void LLVM_code_generator::push_module(mi::mdl::IModule const *module)
{
    m_module_stack.push_back(module);
}

// Pop a module from the stack.
void LLVM_code_generator::pop_module()
{
    m_module_stack.pop_back();
}


// Create extra instructions at function start for debugging.
void LLVM_code_generator::enter_function(llvm::Function *entered)
{
    if (m_jit_dbg_mode == JDBG_NONE)
        return;
/*
    llvm::StringRef name_ref = entered->getName();
    mi::mdl::string function_name = mi::mdl::string(name_ref.data(), name_ref.size(), m_alloc);

    llvm::Value *args[] = {
        // pass the name of the entered function
        context->CreateGlobalStringPtr(function_name.c_str()),
        // pass its address, casted to void pointer
        context->CreateBitCast(entered, m_type_mapper->get_void_ptr_type())
    };
    llvm::Function *func = m_ckf_factory->get(CKF_dbg_enter);
    context->CreateCall(func, args, MI_LLVM_END(args));
*/
}

// Translate a statement to LLVM IR.
void LLVM_code_generator::translate_statement(
    Function_context          &ctx,
    mi::mdl::IStatement const *stmt)
{
    ctx.set_curr_pos(stmt->access_position());

    switch (stmt->get_kind()) {
    case mi::mdl::IStatement::SK_INVALID:
        // should not occur
        break;
    case mi::mdl::IStatement::SK_COMPOUND:
        translate_block(ctx, cast<mi::mdl::IStatement_compound>(stmt));
        return;
    case mi::mdl::IStatement::SK_DECLARATION:
        translate_decl_stmt(ctx, cast<mi::mdl::IStatement_declaration>(stmt));
        return;
    case mi::mdl::IStatement::SK_EXPRESSION:
        {
            mi::mdl::IStatement_expression const *expr_stmt =
                cast<mi::mdl::IStatement_expression>(stmt);
            mi::mdl::IExpression const *expr = expr_stmt->get_expression();
            if (expr != NULL)
                translate_expression(ctx, expr, /*return_derivs=*/ false);
        }
        return;
    case mi::mdl::IStatement::SK_IF:
        translate_if(ctx, cast<mi::mdl::IStatement_if>(stmt));
        return;
    case mi::mdl::IStatement::SK_CASE:
        // should not occur at top-level
        MDL_ASSERT(!"case statment not inside a switch");
        break;
    case mi::mdl::IStatement::SK_SWITCH:
        translate_switch(ctx, cast<mi::mdl::IStatement_switch>(stmt));
        return;
    case mi::mdl::IStatement::SK_WHILE:
        translate_while(ctx, cast<mi::mdl::IStatement_while>(stmt));
        return;
    case mi::mdl::IStatement::SK_DO_WHILE:
        translate_do_while(ctx, cast<mi::mdl::IStatement_do_while>(stmt));
        return;
    case mi::mdl::IStatement::SK_FOR:
        translate_for(ctx, cast<mi::mdl::IStatement_for>(stmt));
        return;
    case mi::mdl::IStatement::SK_BREAK:
        translate_break(ctx, cast<mi::mdl::IStatement_break>(stmt));
        return;
    case mi::mdl::IStatement::SK_CONTINUE:
        translate_continue(ctx, cast<mi::mdl::IStatement_continue>(stmt));
        return;
    case mi::mdl::IStatement::SK_RETURN:
        translate_return(ctx, cast<mi::mdl::IStatement_return>(stmt));
        return;
    }
    MDL_ASSERT(!"unsupported statement kind");
}

// Translate a block statement to LLVM IR.
void LLVM_code_generator::translate_block(
    Function_context                   &ctx,
    mi::mdl::IStatement_compound const *block)
{
    Function_context::Block_scope block_scope(ctx, block);

    for (size_t i = 0, n = block->get_statement_count(); i < n; ++i) {
        mi::mdl::IStatement const *stmt = block->get_statement(i);
        translate_statement(ctx, stmt);
    }
}

// Translate a declaration statement to LLVM IR.
void LLVM_code_generator::translate_decl_stmt(
    Function_context                      &ctx,
    mi::mdl::IStatement_declaration const *decl_stmt)
{
    mi::mdl::IDeclaration const *decl = decl_stmt->get_declaration();
    translate_declaration(ctx, decl);
}

// Translate a declaration to LLVM IR.
void LLVM_code_generator::translate_declaration(
    Function_context            &ctx,
    mi::mdl::IDeclaration const *decl)
{
    switch (decl->get_kind()) {
    case mi::mdl::IDeclaration::DK_INVALID:
        // should not occur
        break;

    case mi::mdl::IDeclaration::DK_IMPORT:
    case mi::mdl::IDeclaration::DK_ANNOTATION:
    case mi::mdl::IDeclaration::DK_CONSTANT:
        // should NOT occur inside functions, but even then, generates no code
        return;

    case mi::mdl::IDeclaration::DK_TYPE_ALIAS:
    case mi::mdl::IDeclaration::DK_TYPE_STRUCT:
    case mi::mdl::IDeclaration::DK_TYPE_ENUM:
    case mi::mdl::IDeclaration::DK_MODULE:
    case mi::mdl::IDeclaration::DK_NAMESPACE_ALIAS:
        // generates no code
        return;

    case mi::mdl::IDeclaration::DK_VARIABLE:
        return translate_var_declaration(ctx, cast<mi::mdl::IDeclaration_variable>(decl));

    case mi::mdl::IDeclaration::DK_FUNCTION:
        // should not occur, nested functions are not allowed in MDL
        MDL_ASSERT(!"nested functions mot supported");
        return;
    }
    MDL_ASSERT(!"unsupported declaration kind");
}

// Translate a variable declaration to LLVM IR.
void LLVM_code_generator::translate_var_declaration(
    Function_context                      &ctx,
    mi::mdl::IDeclaration_variable const *var_decl)
{
    for (size_t i = 0, n = var_decl->get_variable_count(); i < n; ++i) {
        mi::mdl::ISimple_name const *var_name  = var_decl->get_variable_name(i);
        mi::mdl::IDefinition const  *var_def   = var_name->get_definition();
        mi::mdl::IExpression const  *init_expr = var_decl->get_variable_init(i);
        mi::mdl::IType const        *var_type  = var_def->get_type()->skip_type_alias();

        bool is_deriv_var = m_cur_func_deriv_info != NULL &&
            m_cur_func_deriv_info->is_derivative_variable(var_def);

        // instantiate the variable type
        Expression_result init;
        if (is<mi::mdl::IType_array>(var_type) && init_expr == NULL) {
            // work-around for MDL compiler core laziness: Array definitions without
            // an init-expression are not corrected by the array default constructor ...
            init = Expression_result::value(
                llvm::ConstantAggregateZero::get(
                    lookup_type(var_type, ctx.instantiate_type_size(var_type))));
            init.ensure_deriv_result(ctx, is_deriv_var);
        } else {
            // in all other case there must be an init expression
            init = translate_expression(ctx, init_expr, is_deriv_var);
        }

        if (need_storage_for_var(var_def)) {
            // allocate a local
            LLVM_context_data *v_data = ctx.get_context_data(var_def);
            v_data->set_var_value(init.as_value(ctx));
        } else {
            // just an unchanged value (aka const)
            mi::mdl::IType::Kind type_kind = var_type->get_kind();
            if ((type_kind == mi::mdl::IType::TK_ARRAY || type_kind == mi::mdl::IType::TK_STRUCT)
                && init.is_constant())
            {
                // If arrays are handled as values, we generate currently
                // bad code because they are stored every time to access elements by index.
                // So create a constant in global space. The same is probably true for structs, so
                // handle them also this way here
                if (init.is_value()) {
                    string init_name(var_def->get_symbol()->get_name(), get_allocator());
                    init_name += "_init";
                    llvm::Value *cv = new llvm::GlobalVariable(
                        *m_module,
                        init.get_value_type(),
                        /*isConstant=*/true,
                        llvm::GlobalValue::InternalLinkage,
                        llvm::cast<llvm::Constant>(init.as_value(ctx)),
                        init_name.c_str());
                    ctx.create_context_data(var_def, cv, /*by_reference=*/true);
                } else {
                    // is already a global constant
                    ctx.create_context_data(var_def, init.as_ptr(ctx), /*by_reference=*/true);
                }
            } else {
                // handle as a real value
                ctx.create_context_data(var_def, init.as_value(ctx), /*by_reference=*/false);
            }
        }
    }
}

// Create a branch from a boolean expression (with short cut evaluation).
void LLVM_code_generator::translate_boolean_branch(
    Function_context           &ctx,
    mi::mdl::IExpression const *cond,
    llvm::BasicBlock           *true_bb,
    llvm::BasicBlock           *false_bb)
{
    if (mi::mdl::IExpression_binary const *bin_expr = as<mi::mdl::IExpression_binary>(cond)) {
        mi::mdl::IExpression_binary::Operator op = bin_expr->get_operator();

        if (op == mi::mdl::IExpression_binary::OK_LOGICAL_AND) {
            // shortcut AND evaluation
            llvm::BasicBlock *imm_bb = ctx.create_bb("shortcut_imm");

            mi::mdl::IExpression const *lhs = bin_expr->get_left_argument();
            translate_boolean_branch(ctx, lhs, imm_bb, false_bb);

            ctx->SetInsertPoint(imm_bb);
            mi::mdl::IExpression const *rhs = bin_expr->get_right_argument();
            translate_boolean_branch(ctx, rhs, true_bb, false_bb);
            return;
        } else if (op == mi::mdl::IExpression_binary::OK_LOGICAL_OR) {
            // shortcut OR evaluation
            llvm::BasicBlock *imm_bb = ctx.create_bb("shortcut_imm");

            mi::mdl::IExpression const *lhs = bin_expr->get_left_argument();
            translate_boolean_branch(ctx, lhs, true_bb, imm_bb);

            ctx->SetInsertPoint(imm_bb);
            mi::mdl::IExpression const *rhs = bin_expr->get_right_argument();
            translate_boolean_branch(ctx, rhs, true_bb, false_bb);
            return;
        }
    } else if (mi::mdl::IExpression_unary const *un_expr = as<mi::mdl::IExpression_unary>(cond)) {
        if (un_expr->get_operator() == mi::mdl::IExpression_unary::OK_LOGICAL_NOT) {
            // logical not: just exchange true and false targets
            mi::mdl::IExpression const *arg = un_expr->get_argument();
            translate_boolean_branch(ctx, arg, false_bb, true_bb);
            return;
        }
    }
    // default case
    llvm::Value *c = translate_expression_value(ctx, cond, /*return_derivs=*/ false);

    if (c->getType() != m_type_mapper.get_predicate_type()) {
        // map to predicate type
        c = ctx->CreateICmpNE(c, ctx.get_constant(false));
    }

    ctx->CreateCondBr(c, true_bb, false_bb);
}

// Create a branch from a boolean expression (with short cut evaluation).
void LLVM_code_generator::translate_boolean_branch(
    Function_context                   &ctx,
    mi::mdl::ICall_name_resolver const *resolver,
    mi::mdl::DAG_node const            *cond,
    llvm::BasicBlock                   *true_bb,
    llvm::BasicBlock                   *false_bb)
{
    if (mi::mdl::DAG_call const *call = as<mi::mdl::DAG_call>(cond)) {
        mi::mdl::IDefinition::Semantics sema = call->get_semantic();
        if (mi::mdl::semantic_is_operator(sema)) {
            mi::mdl::IExpression::Operator op = mi::mdl::semantic_to_operator(sema);

            if (op == mi::mdl::IExpression::OK_LOGICAL_AND) {
                // shortcut AND evaluation
                llvm::BasicBlock *imm_bb = ctx.create_bb("shortcut_imm");

                mi::mdl::DAG_node const *lhs = call->get_argument(0);
                translate_boolean_branch(ctx, resolver, lhs, imm_bb, false_bb);

                BB_store chain(m_curr_bb, get_next_bb());
                ctx->SetInsertPoint(imm_bb);
                mi::mdl::DAG_node const *rhs = call->get_argument(1);
                translate_boolean_branch(ctx, resolver, rhs, true_bb, false_bb);
                return;
            } else if (op == mi::mdl::IExpression::OK_LOGICAL_OR) {
                // shortcut OR evaluation
                llvm::BasicBlock *imm_bb = ctx.create_bb("shortcut_imm");

                mi::mdl::DAG_node const *lhs = call->get_argument(0);
                translate_boolean_branch(ctx, resolver, lhs, true_bb, imm_bb);

                BB_store chain(m_curr_bb, get_next_bb());
                ctx->SetInsertPoint(imm_bb);
                mi::mdl::DAG_node const *rhs = call->get_argument(1);
                translate_boolean_branch(ctx, resolver, rhs, true_bb, false_bb);
                return;
            } else if (op == mi::mdl::IExpression::OK_LOGICAL_NOT) {
                // logical not: just exchange true and false targets
                mi::mdl::DAG_node const *arg = call->get_argument(0);
                translate_boolean_branch(ctx, resolver, arg, false_bb, true_bb);
                return;
            }
        }
    }
    // default case
    llvm::Value *c = translate_node(ctx, cond, resolver).as_value(ctx);

    if (c->getType() != m_type_mapper.get_predicate_type()) {
        // map to predicate type
        c = ctx->CreateICmpNE(c, ctx.get_constant(false));
    }

    ctx->CreateCondBr(c, true_bb, false_bb);
}

// Translate an if statement to LLVM IR.
void LLVM_code_generator::translate_if(
    Function_context              &ctx,
    mi::mdl::IStatement_if const *if_stmt)
{
    mi::mdl::IExpression const *cond = if_stmt->get_condition();
    mi::mdl::IStatement const  *then_stmt = if_stmt->get_then_statement();
    mi::mdl::IStatement const  *else_stmt = if_stmt->get_else_statement();

    llvm::BasicBlock *true_bb  = ctx.create_bb("if_on_true");
    llvm::BasicBlock *end_bb   = ctx.create_bb("if_end");
    llvm::BasicBlock *false_bb = else_stmt == NULL ? end_bb : ctx.create_bb("if_on_false");

    translate_boolean_branch(ctx, cond, true_bb, false_bb);

    // create the then statement
    ctx->SetInsertPoint(true_bb);
    {
        Function_context::Block_scope block_scope(ctx, then_stmt);
        translate_statement(ctx, then_stmt);
    }
    ctx->CreateBr(end_bb);

    if (else_stmt != NULL) {
        // create the else statement
        ctx->SetInsertPoint(false_bb);
        {
            Function_context::Block_scope block_scope(ctx, else_stmt);
            translate_statement(ctx, else_stmt);
        }
        ctx->CreateBr(end_bb);
    }
    ctx->SetInsertPoint(end_bb);
}

// Translate a switch statement to LLVM IR.
void LLVM_code_generator::translate_switch(
    Function_context                 &ctx,
    mi::mdl::IStatement_switch const *switch_stmt)
{
    mi::mdl::IStatement_case const *default_case = NULL;

    size_t n = switch_stmt->get_case_count();
    for (size_t i = 0; i < n; ++i) {
        mi::mdl::IStatement const *stmt = switch_stmt->get_case(i);

        if (mi::mdl::IStatement_case const *case_stmt = cast<mi::mdl::IStatement_case>(stmt)) {
            mi::mdl::IExpression const *label = case_stmt->get_label();

            if (label == NULL) {
                default_case = case_stmt;
                break;
            }
        }
    }


    llvm::BasicBlock *end_bb     = ctx.create_bb("after_switch");
    llvm::BasicBlock *default_bb = default_case == NULL ? end_bb : ctx.create_bb("switch_default");

    Function_context::Break_destination scope(ctx, end_bb);

    llvm::Value *expr = translate_expression_value(
        ctx, switch_stmt->get_condition(), /*return_derivs=*/ false);

    llvm::SwitchInst *switch_instr = ctx->CreateSwitch(expr, default_bb, unsigned(n));

    ctx->SetInsertPoint(ctx.get_unreachable_bb());

    for (size_t i = 0, n = switch_stmt->get_case_count(); i < n; ++i) {
        mi::mdl::IStatement const *stmt = switch_stmt->get_case(i);

        if (mi::mdl::IStatement_case const *case_stmt = cast<mi::mdl::IStatement_case>(stmt)) {
            mi::mdl::IExpression const     *label     = case_stmt->get_label();

            if (label == NULL) {
                // fall-through from previous case and switch to the new block
                ctx->CreateBr(default_bb);
                ctx->SetInsertPoint(default_bb);
            } else {
                llvm::BasicBlock *case_bb = ctx.create_bb("case_body");

                // fall-through from previous case and switch to the new block
                ctx->CreateBr(case_bb);
                ctx->SetInsertPoint(case_bb);

                mi::mdl::IExpression_literal const *lit =
                    cast<mi::mdl::IExpression_literal>(label);
                mi::mdl::IValue_int_valued const *v =
                    cast<mi::mdl::IValue_int_valued>(lit->get_value());

                switch_instr->addCase(ctx.get_constant(v), case_bb);
            }
            translate_block(ctx, case_stmt);
        }
    }
    // fall-through from last case
    ctx->CreateBr(end_bb);
    ctx->SetInsertPoint(end_bb);
}

// Translate a while statement to LLVM IR.
void LLVM_code_generator::translate_while(
    Function_context                &ctx,
    mi::mdl::IStatement_while const *while_stmt)
{
    llvm::BasicBlock *start_bb = ctx.create_bb("while_condition");
    llvm::BasicBlock *body_bb  = ctx.create_bb("while_body");
    llvm::BasicBlock *end_bb   = ctx.create_bb("after_while");

    Function_context::Break_destination    break_scope(ctx, end_bb);
    Function_context::Continue_destination cont_scope(ctx, start_bb);

    ctx->CreateBr(start_bb);

    // create check-abort-condition code
    ctx->SetInsertPoint(start_bb);

    mi::mdl::IExpression const *cond = while_stmt->get_condition();
    translate_boolean_branch(ctx, cond, body_bb, end_bb);

    // create loop body code
    ctx->SetInsertPoint(body_bb);
    {
        Function_context::Block_scope block_scope(ctx, while_stmt);

        mi::mdl::IStatement const *body = while_stmt->get_body();
        translate_statement(ctx, body);
    }
    ctx->CreateBr(start_bb);

    // after while
    ctx->SetInsertPoint(end_bb);
}

// Translate a do-while statement to LLVM IR.
void LLVM_code_generator::translate_do_while(
    Function_context                   &ctx,
    mi::mdl::IStatement_do_while const *do_stmt)
{
    llvm::BasicBlock *cond_bb = ctx.create_bb("do_while_condition");
    llvm::BasicBlock *body_bb = ctx.create_bb("do_while_body");
    llvm::BasicBlock *end_bb  = ctx.create_bb("after_do_while");

    Function_context::Break_destination    break_scope(ctx, end_bb);
    Function_context::Continue_destination cont_scope(ctx, cond_bb);

    ctx->CreateBr(body_bb);

    // create loop body code
    ctx->SetInsertPoint(body_bb);
    {
        Function_context::Block_scope block_scope(ctx, do_stmt);

        mi::mdl::IStatement const *body = do_stmt->get_body();
        translate_statement(ctx, body);
    }
    ctx->CreateBr(cond_bb);

    // create check-abort-condition code
    ctx->SetInsertPoint(cond_bb);
    mi::mdl::IExpression const *cond = do_stmt->get_condition();
    translate_boolean_branch(ctx, cond, body_bb, end_bb);

    // after do-while
    ctx->SetInsertPoint(end_bb);
}

// Translate a for statement to LLVM IR.
void LLVM_code_generator::translate_for(
    Function_context              &ctx,
    mi::mdl::IStatement_for const *for_stmt)
{
    // the for loop creates a scope (for declarations inside the for init)
    Function_context::Block_scope block_scope(ctx, for_stmt);

    mi::mdl::IExpression const *cond = for_stmt->get_condition();
    mi::mdl::IExpression const *upd  = for_stmt->get_update();

    llvm::BasicBlock *body_bb = ctx.create_bb("for_body");
    llvm::BasicBlock *end_bb  = ctx.create_bb("after_for");
    llvm::BasicBlock *test_bb = cond == NULL ? body_bb : ctx.create_bb("for_test");
    llvm::BasicBlock *upd_bb  = upd == NULL  ? test_bb : ctx.create_bb("for_update");

    Function_context::Break_destination    break_scope(ctx, end_bb);
    Function_context::Continue_destination cont_scope(ctx, upd_bb);

    if (mi::mdl::IStatement const *init = for_stmt->get_init()) {
        // create the init statement
        translate_statement(ctx, init);
    }
    ctx->CreateBr(test_bb);

    // create test block if needed
    if (cond != NULL) {
        ctx->SetInsertPoint(test_bb);
        translate_boolean_branch(ctx, cond, body_bb, end_bb);
    }

    // create body code
    ctx->SetInsertPoint(body_bb);
    {
        mi::mdl::IStatement const *body = for_stmt->get_body();

        Function_context::Block_scope block_scope(ctx, body);
        translate_statement(ctx, body);
    }
    ctx->CreateBr(upd_bb);

    // create update block if needed
    if (upd != NULL) {
        ctx->SetInsertPoint(upd_bb);
        translate_expression(ctx, upd, /*return_derivs=*/ false);
        ctx->CreateBr(test_bb);
    }

    // after for
    ctx->SetInsertPoint(end_bb);
}

// Translate a break statement to LLVM IR.
void LLVM_code_generator::translate_break(
    Function_context                &ctx,
    mi::mdl::IStatement_break const *break_stmt)
{
    ctx.create_jmp(ctx.tos_break());
}

// Translate a continue statement to LLVM IR.
void LLVM_code_generator::translate_continue(
    Function_context                   &ctx,
    mi::mdl::IStatement_continue const *cont_stmt)
{
    ctx.create_jmp(ctx.tos_continue());
}

// Translate a return statement to LLVM IR.
void LLVM_code_generator::translate_return(
    Function_context                 &ctx,
    mi::mdl::IStatement_return const *ret_stmt)
{
    if (mi::mdl::IExpression const *expr = ret_stmt->get_expression()) {
        llvm::Type *return_type = ctx.get_return_type();
        llvm::Value *v = translate_expression_value(ctx, expr, ctx.is_deriv_type(return_type));
        ctx.create_return(v);
    } else {
        ctx.create_void_return();
    }
}

// Calculate &matrix[index], index is assured to be in bounds
llvm::Value *LLVM_code_generator::calc_matrix_index_in_bounds(
    Function_context            &ctx,
    mi::mdl::IType_matrix const *m_type,
    llvm::Value                 *matrix_ptr,
    llvm::Value                 *index)
{
    llvm::Type *tp = lookup_type(m_type);
    if (llvm::ArrayType *a_tp = llvm::dyn_cast<llvm::ArrayType>(tp)) {
        llvm::Type *e_tp = a_tp->getArrayElementType();

        if (e_tp->isVectorTy()) {
            // matrix types are represented as array of vectors, so the index
            // directly access the vector
            return ctx.create_simple_gep_in_bounds(matrix_ptr, index);
        } else {
            // matrix types are represented as array of scalars
        }
        // fall through
    } else {
        // matrix types represented as a big vector
        MDL_ASSERT(llvm::isa<llvm::VectorType>(tp));
        // fall through
    }
    // matrix types are represented as array/vector, need address
    // arithmetic here
    mi::mdl::IType_vector const *v_type = m_type->get_element_type();
    int n_rows = v_type->get_size();

    // scale index, so it points to the column
    index = ctx->CreateMul(index, ctx.get_constant(n_rows));

    // get the pointer to the element
    llvm::Value *elem_ptr = ctx.create_simple_gep_in_bounds(matrix_ptr, index);

    // and cast it to a pointer to a row
    llvm::Type *row_ptr_tp = m_type_mapper.get_ptr(lookup_type(v_type));
    elem_ptr = ctx->CreateBitCast(elem_ptr, row_ptr_tp);

    return elem_ptr;
}

// If bounds check exceptions are disabled and instancing is enabled,
// returns a select operation returning index 0 if the index is out of bounds.
// Otherwise just returns the index.
llvm::Value *LLVM_code_generator::adapt_index_for_bounds_check(
    Function_context &ctx,
    llvm::Value *index,
    llvm::Value *bound)
{
    // with instancing, all arrays have at least size 1, so we map out of bounds accesses to index 0
    if (m_bounds_check_exception_disabled && m_enable_instancing) {
        return ctx.create_select_if_in_bounds(
            index, bound, index, llvm::Constant::getNullValue(index->getType()));
    }
    return index;
}

// Translate an l-value index expression to LLVM IR.
llvm::Value *LLVM_code_generator::translate_lval_index_expression(
    Function_context        &ctx,
    mi::mdl::IType const    *comp_type,
    llvm::Value             *comp_ptr,
    llvm::Value             *index,
    mi::mdl::Position const *index_pos)
{
    // instantiate type
    int imm_size = ctx.instantiate_type_size(comp_type);

    // determine bound and element pointer depending on kind of compound
    llvm::Value *bound;
    llvm::Value *elem_ptr;
    if (mi::mdl::IType_array const *a_type = as<mi::mdl::IType_array>(comp_type)) {
        if (!a_type->is_immediate_sized() && imm_size < 0) {
            // generate bounds check for deferred sized array
            bound = ctx.get_deferred_size_from_ptr(comp_ptr);
            index = adapt_index_for_bounds_check(ctx, index, bound);

            // array_desc<T> access
            llvm::Value *base = ctx.get_deferred_base_from_ptr(comp_ptr);
            elem_ptr = ctx->CreateInBoundsGEP(base, index);
        } else {
            // generate bounds check for immediate sized array
            size_t arr_size = imm_size >= 0 ? imm_size : a_type->get_size();
            bound = ctx.get_constant(arr_size);
            index = adapt_index_for_bounds_check(ctx, index, bound);

            elem_ptr = ctx.create_simple_gep_in_bounds(comp_ptr, index);
        }
    } else if (mi::mdl::IType_matrix const *m_type = as<mi::mdl::IType_matrix>(comp_type)) {
        // generate bounds check for matrices
        bound = ctx.get_constant(size_t(m_type->get_columns()));
        index = adapt_index_for_bounds_check(ctx, index, bound);

        elem_ptr = calc_matrix_index_in_bounds(ctx, m_type, comp_ptr, index);
    } else {
        // generate bounds check for vector type
        mi::mdl::IType_vector const *v_type = cast<mi::mdl::IType_vector>(comp_type);
        bound = ctx.get_constant(size_t(v_type->get_size()));
        index = adapt_index_for_bounds_check(ctx, index, bound);

        elem_ptr = ctx.create_simple_gep_in_bounds(comp_ptr, index);
    }

    if (!m_bounds_check_exception_disabled) {
        ctx.create_bounds_check_with_exception(
            index, bound, Exc_location(*this, index_pos));
    } else if (!m_enable_instancing) {
        // without instancing, the array size could be zero, so we need to load from a dummy
        // variable if we're out of bounds
        llvm::Type *elem_tp = elem_ptr->getType()->getPointerElementType();
        llvm::Value *dummy_ptr = ctx.create_local(elem_tp, "dummy");
        return ctx.create_select_if_in_bounds(index, bound, elem_ptr, dummy_ptr);
    }

    return elem_ptr;
}

// Translate a dual l-value index expression to LLVM IR.
void LLVM_code_generator::translate_lval_index_expression_dual(
    Function_context        &ctx,
    mi::mdl::IType const    *comp_type,
    llvm::Value             *comp_val_ptr,
    llvm::Value             *comp_dx_ptr,
    llvm::Value             *comp_dy_ptr,
    llvm::Value             *index,
    mi::mdl::Position const *index_pos,
    llvm::Value             *&adr_val,
    llvm::Value             *&adr_dx,
    llvm::Value             *&adr_dy)
{
    // skip derivative type, we already know, that it is a derivative type
    comp_type = m_type_mapper.skip_deriv_type(comp_type);

    // instantiate type
    int imm_size = ctx.instantiate_type_size(comp_type);

    // determine bound and element pointer depending on kind of compound
    llvm::Value *bound;
    if (mi::mdl::IType_array const *a_type = as<mi::mdl::IType_array>(comp_type)) {
        if (!a_type->is_immediate_sized() && imm_size < 0) {
            // generate bounds check for deferred sized array
            bound = ctx.get_deferred_size_from_ptr(comp_val_ptr);
            index = adapt_index_for_bounds_check(ctx, index, bound);

            // array_desc<T> access
            llvm::Value *base_val = ctx.get_deferred_base_from_ptr(comp_val_ptr);
            adr_val = ctx->CreateInBoundsGEP(base_val, index);
            llvm::Value *base_dx = ctx.get_deferred_base_from_ptr(comp_dx_ptr);
            adr_dx = ctx->CreateInBoundsGEP(base_dx, index);
            llvm::Value *base_dy = ctx.get_deferred_base_from_ptr(comp_dy_ptr);
            adr_dy = ctx->CreateInBoundsGEP(base_dy, index);
        } else {
            // generate bounds check for immediate sized array
            size_t arr_size = imm_size >= 0 ? imm_size : a_type->get_size();
            bound = ctx.get_constant(arr_size);
            index = adapt_index_for_bounds_check(ctx, index, bound);

            adr_val = ctx.create_simple_gep_in_bounds(comp_val_ptr, index);
            adr_dx  = ctx.create_simple_gep_in_bounds(comp_dx_ptr, index);
            adr_dy  = ctx.create_simple_gep_in_bounds(comp_dy_ptr, index);
        }
    } else if (mi::mdl::IType_matrix const *m_type = as<mi::mdl::IType_matrix>(comp_type)) {
        // generate bounds check for matrices
        bound = ctx.get_constant(size_t(m_type->get_columns()));
        index = adapt_index_for_bounds_check(ctx, index, bound);

        adr_val = calc_matrix_index_in_bounds(ctx, m_type, comp_val_ptr, index);
        adr_dx  = calc_matrix_index_in_bounds(ctx, m_type, comp_dx_ptr, index);
        adr_dy  = calc_matrix_index_in_bounds(ctx, m_type, comp_dy_ptr, index);
    } else {
        // generate bounds check for vector type
        mi::mdl::IType_vector const *v_type = cast<mi::mdl::IType_vector>(comp_type);
        bound = ctx.get_constant(size_t(v_type->get_size()));
        index = adapt_index_for_bounds_check(ctx, index, bound);

        adr_val = ctx.create_simple_gep_in_bounds(comp_val_ptr, index);
        adr_dx  = ctx.create_simple_gep_in_bounds(comp_dx_ptr, index);
        adr_dy  = ctx.create_simple_gep_in_bounds(comp_dy_ptr, index);
    }

    if (!m_bounds_check_exception_disabled) {
        ctx.create_bounds_check_with_exception(
            index, bound, Exc_location(*this, index_pos));
    } else if (!m_enable_instancing) {
        // without instancing, the array size could be zero, so we need to load from a dummy
        // variable if we're out of bounds
        llvm::Type *elem_tp = adr_val->getType()->getPointerElementType();
        llvm::Value *dummy_ptr = ctx.create_local(elem_tp, "dummy");
        adr_val = ctx.create_select_if_in_bounds(index, bound, adr_val, dummy_ptr);
        adr_dx  = ctx.create_select_if_in_bounds(index, bound, adr_dx,  dummy_ptr);
        adr_dy  = ctx.create_select_if_in_bounds(index, bound, adr_dy,  dummy_ptr);
    }
}

// Translate an r-value index expression to LLVM IR.
Expression_result LLVM_code_generator::translate_index_expression(
    Function_context        &ctx,
    mi::mdl::IType const    *comp_type,
    Expression_result       comp,
    llvm::Value             *index,
    mi::mdl::Position const *index_pos)
{
    // handle LLVM result with derivatives
    if (m_type_mapper.is_deriv_type(comp.get_value_type())) {
        mi::mdl::IType const *val_type = m_type_mapper.skip_deriv_type(comp_type);

        Expression_result val = Expression_result::value(ctx.get_dual_val(comp.as_value(ctx)));
        Expression_result val_elem = translate_index_expression(
            ctx, val_type, val, index, index_pos);

        Expression_result dx = Expression_result::value(ctx.get_dual_dx(comp.as_value(ctx)));
        Expression_result dx_elem = translate_index_expression(
            ctx, val_type, dx, index, index_pos);

        Expression_result dy = Expression_result::value(ctx.get_dual_dy(comp.as_value(ctx)));
        Expression_result dy_elem = translate_index_expression(
            ctx, val_type, dy, index, index_pos);

        llvm::Value *res = ctx.get_dual(
            val_elem.as_value(ctx),
            dx_elem.as_value(ctx),
            dy_elem.as_value(ctx));

        return Expression_result::value(res);
    }

    // handle LLVM result without derivatives when derivatives are requested
    if (m_type_mapper.is_deriv_type(comp_type)) {
        mi::mdl::IType const *val_type = m_type_mapper.skip_deriv_type(comp_type);

        Expression_result res = translate_index_expression(
            ctx, val_type, comp, index, index_pos);
        res.ensure_deriv_result(ctx, /*should_be_deriv_value=*/ true);
        return res;
    }

    // instantiate type
    int imm_size = ctx.instantiate_type_size(comp_type);

    // determine bound and element pointer depending on kind of compound
    llvm::Value *bound;
    llvm::Value *elem_ptr;
    if (mi::mdl::IType_array const *a_type = as<mi::mdl::IType_array>(comp_type)) {
        if (!a_type->is_immediate_sized() && imm_size < 0) {
            // generate bounds check for deferred sized array
            llvm::Value *compound = comp.as_value(ctx);
            bound = ctx.get_deferred_size(compound);
            index = adapt_index_for_bounds_check(ctx, index, bound);

            // array_desc<T> access
            llvm::Value *base = ctx.get_deferred_base(compound);
            elem_ptr = ctx->CreateInBoundsGEP(base, index);
        } else {
            // generate bounds check for immediate sized array
            size_t arr_size = imm_size >= 0 ? imm_size : a_type->get_size();
            bound = ctx.get_constant(arr_size);
            index = adapt_index_for_bounds_check(ctx, index, bound);

            if (comp.is_offset()) {
                llvm::ArrayType *at_llvm =
                    llvm::cast<llvm::ArrayType>(comp.get_offset_res_llvm_type());

                if (!m_bounds_check_exception_disabled) {
                    ctx.create_bounds_check_with_exception(
                        index, bound, Exc_location(*this, index_pos));
                } else {
                    MDL_ASSERT(m_enable_instancing);
                }

                if (arr_size == 0) {
                    return Expression_result::value(
                        llvm::Constant::getNullValue(at_llvm->getElementType()));
                }

                int size = int(m_data_layout.getTypeAllocSize(at_llvm->getElementType()));
                llvm::Value *offs = ctx->CreateAdd(
                    comp.get_offset(),
                    ctx->CreateMul(index, ctx.get_constant(size)));

                MDL_ASSERT(comp.get_offset_kind() == Expression_result::OK_RO_DATA_SEGMENT &&
                    "ARG BLOCK not supported yet");
                int cur_offs = 0;
                llvm::Value *res = translate_ro_data_segment_hlsl_value(
                    ctx, a_type->get_element_type(), cur_offs, offs);
                return Expression_result::value(res);
            }

            elem_ptr = ctx.create_simple_gep_in_bounds(comp.as_ptr(ctx), index);
        }
    } else if (mi::mdl::IType_matrix const *m_type = as<mi::mdl::IType_matrix>(comp_type)) {
        // generate bounds check for matrices
        bound = ctx.get_constant(size_t(m_type->get_columns()));
        index = adapt_index_for_bounds_check(ctx, index, bound);

        elem_ptr = calc_matrix_index_in_bounds(ctx, m_type, comp.as_ptr(ctx), index);
    } else {
        // generate bounds check for vector types
        mi::mdl::IType_vector const *v_type = cast<mi::mdl::IType_vector>(comp_type);
        bound = ctx.get_constant(size_t(v_type->get_size()));
        index = adapt_index_for_bounds_check(ctx, index, bound);

        if (llvm::isa<llvm::VectorType>(comp.get_value_type())) {
            // extracting a vector component
            llvm::Value *res = ctx->CreateExtractElement(comp.as_value(ctx), index);
            return Expression_result::value(res);
        }

        elem_ptr = ctx.create_simple_gep_in_bounds(comp.as_ptr(ctx), index);
    }

    if (!m_bounds_check_exception_disabled) {
        ctx.create_bounds_check_with_exception(
            index, bound, Exc_location(*this, index_pos));
    } else if (!m_enable_instancing) {
        // without instancing, the array size could be zero, so we need to load from a dummy
        // variable if we're out of bounds
        llvm::Type *elem_tp = elem_ptr->getType()->getPointerElementType();
        llvm::Value *dummy_ptr = ctx.create_local(elem_tp, "dummy");
        llvm::Value *zero = llvm::Constant::getNullValue(elem_tp);
        ctx->CreateStore(zero, dummy_ptr);

        return Expression_result::ptr(
            ctx.create_select_if_in_bounds(index, bound, elem_ptr, dummy_ptr));
    }

    return Expression_result::ptr(elem_ptr);
}

// Translate an l-value expression to LLVM IR.
llvm::Value *LLVM_code_generator::translate_lval_expression(
    Function_context           &ctx,
    mi::mdl::IExpression const *expr)
{
    switch (expr->get_kind()) {
    case mi::mdl::IExpression::EK_REFERENCE:
        {
            mi::mdl::IExpression_reference const *ref = cast<mi::mdl::IExpression_reference>(expr);
            mi::mdl::IDefinition const           *def = ref->get_definition();

            MDL_ASSERT(def != NULL && "Array constructor unexpected here");

            switch (def->get_kind()) {
            case mi::mdl::IDefinition::DK_PARAMETER:
            case mi::mdl::IDefinition::DK_VARIABLE:
                {
                    LLVM_context_data *data = ctx.get_context_data(def);

                    return data->get_var_address();
                }
                break;
            default:
                MDL_ASSERT(!"Unexpected reference kind here");
                break;
            }
        }
        break;

    case mi::mdl::IExpression::EK_BINARY:
        {
            mi::mdl::IExpression_binary const *bin_exp = cast<mi::mdl::IExpression_binary>(expr);
            mi::mdl::IExpression const        *lhs     = bin_exp->get_left_argument();
            mi::mdl::IExpression const        *rhs     = bin_exp->get_right_argument();

            switch (bin_exp->get_operator()) {
            case mi::mdl::IExpression_binary::OK_ARRAY_INDEX:
                {
                    llvm::Value             *comp_ptr  = translate_lval_expression(ctx, lhs);
                    llvm::Value             *index     =
                        translate_expression_value(ctx, rhs, /*return_derivs=*/ false);
                    mi::mdl::Position const *index_pos = &rhs->access_position();
                    mi::mdl::IType const    *comp_type = lhs->get_type()->skip_type_alias();

                    return translate_lval_index_expression(
                        ctx, comp_type, comp_ptr, index, index_pos);
                }

            case mi::mdl::IExpression_binary::OK_SELECT:
                {
                    mi::mdl::IExpression_reference const *ref =
                        cast<mi::mdl::IExpression_reference>(rhs);
                    mi::mdl::IDefinition const           *def = ref->get_definition();

                    MDL_ASSERT(def->get_kind() == mi::mdl::IDefinition::DK_MEMBER);

                    llvm::Value *l   = translate_lval_expression(ctx, lhs);
                    llvm::Value *idx = ctx.get_constant(int(def->get_field_index()));
                    return ctx.create_simple_gep_in_bounds(l, idx);
                }

            default:
                break;
            }
        }
        break;

    default:
        break;
    }
    MDL_ASSERT(!"unsupported lvalue expression kind");
    return llvm::UndefValue::get(lookup_type(expr->get_type()));
}

// Translate a dual l-value expression to LLVM IR.
void LLVM_code_generator::translate_lval_expression_dual(
    Function_context           &ctx,
    mi::mdl::IExpression const *expr,
    llvm::Value                *&adr_val,
    llvm::Value                *&adr_dx,
    llvm::Value                *&adr_dy)
{
    switch (expr->get_kind()) {
    case mi::mdl::IExpression::EK_REFERENCE:
        {
            mi::mdl::IExpression_reference const *ref = cast<mi::mdl::IExpression_reference>(expr);
            mi::mdl::IDefinition const           *def = ref->get_definition();

            MDL_ASSERT(def != NULL && "Array constructor unexpected here");

            switch (def->get_kind()) {
            case mi::mdl::IDefinition::DK_PARAMETER:
            case mi::mdl::IDefinition::DK_VARIABLE:
                {
                    LLVM_context_data *data = ctx.get_context_data(def);

                    llvm::Value *var_addr = data->get_var_address();
                    adr_val = ctx.create_simple_gep_in_bounds(var_addr, 0u);
                    adr_dx  = ctx.create_simple_gep_in_bounds(var_addr, 1u);
                    adr_dy  = ctx.create_simple_gep_in_bounds(var_addr, 2u);
                    return;
                }
                break;
            default:
                MDL_ASSERT(!"Unexpected reference kind here");
                break;
            }
        }
        break;

    case mi::mdl::IExpression::EK_BINARY:
        {
            mi::mdl::IExpression_binary const *bin_exp = cast<mi::mdl::IExpression_binary>(expr);
            mi::mdl::IExpression const        *lhs     = bin_exp->get_left_argument();
            mi::mdl::IExpression const        *rhs     = bin_exp->get_right_argument();

            llvm::Value *val_ptr, *dx_ptr, *dy_ptr;
            translate_lval_expression_dual(ctx, lhs, val_ptr, dx_ptr, dy_ptr);

            switch (bin_exp->get_operator()) {
            case mi::mdl::IExpression_binary::OK_ARRAY_INDEX:
                {
                    llvm::Value             *index     =
                        translate_expression_value(ctx, rhs, /*return_derivs=*/ false);
                    mi::mdl::Position const *index_pos = &rhs->access_position();
                    mi::mdl::IType const    *comp_type = lhs->get_type()->skip_type_alias();

                    translate_lval_index_expression_dual(
                        ctx, comp_type, val_ptr, dx_ptr, dy_ptr, index, index_pos,
                        adr_val, adr_dx, adr_dy);
                    return;
                }

            case mi::mdl::IExpression_binary::OK_SELECT:
                {
                    mi::mdl::IExpression_reference const *ref =
                        cast<mi::mdl::IExpression_reference>(rhs);
                    mi::mdl::IDefinition const           *def = ref->get_definition();

                    MDL_ASSERT(def->get_kind() == mi::mdl::IDefinition::DK_MEMBER);

                    llvm::Value *idx = ctx.get_constant(int(def->get_field_index()));
                    adr_val = ctx.create_simple_gep_in_bounds(val_ptr, idx);
                    adr_dx  = ctx.create_simple_gep_in_bounds(dx_ptr, idx);
                    adr_dy  = ctx.create_simple_gep_in_bounds(dy_ptr, idx);
                    return;
                }

            default:
                break;
            }
        }
        break;

    default:
        break;
    }
    MDL_ASSERT(!"unsupported lvalue expression kind");
    adr_val = adr_dx = adr_dy = llvm::UndefValue::get(lookup_type(expr->get_type()));
}

// Translate an expression to LLVM IR, returning its value.
llvm::Value *LLVM_code_generator::translate_expression_value(
    Function_context           &ctx,
    mi::mdl::IExpression const *expr,
    bool                       return_derivs)
{
    return translate_expression(ctx, expr, return_derivs).as_value(ctx);
}

// Translate an (r-value) expression to LLVM IR.
Expression_result LLVM_code_generator::translate_expression(
    Function_context           &ctx,
    mi::mdl::IExpression const *expr,
    bool                       return_derivs)
{
    ctx.set_curr_pos(expr->access_position());

    Expression_result res = Expression_result::unset();

    switch (expr->get_kind()) {
    case mi::mdl::IExpression::EK_INVALID:
        // should not occur
        break;
    case mi::mdl::IExpression::EK_LITERAL:
        res = translate_literal(ctx, cast<mi::mdl::IExpression_literal>(expr));
        break;

    case mi::mdl::IExpression::EK_REFERENCE:
        // assume rvalue if encountered here
        {
            mi::mdl::IExpression_reference const *ref = cast<mi::mdl::IExpression_reference>(expr);
            mi::mdl::IDefinition const           *def = ref->get_definition();

            MDL_ASSERT(def != NULL && "Array constructor unexpected here");

            switch (def->get_kind()) {
            case mi::mdl::IDefinition::DK_PARAMETER:
            case mi::mdl::IDefinition::DK_VARIABLE:
                {
                    LLVM_context_data *data = ctx.get_context_data(def);

                    if (llvm::Value *v = data->get_var_address()) {
                        res = Expression_result::ptr(v);
                    } else {
                        // variable is never changed, only a value

                        res = Expression_result::value(data->get_var_value());
                    }
                }
                break;

            case mi::mdl::IDefinition::DK_ENUM_VALUE:
                {
                    mi::mdl::IValue_enum const *v =
                        cast<mi::mdl::IValue_enum>(def->get_constant_value());
                    res = Expression_result::value(ctx.get_constant(v));
                }
                break;

            case mi::mdl::IDefinition::DK_ARRAY_SIZE:
                {
                    mi::mdl::ISymbol const *symbol = def->get_symbol();

                    // The only source of array sizes here should be parameters, find them.
                    mi::mdl::IDefinition const *param_def = ctx.find_parameter_for_size(symbol);

                    MDL_ASSERT(param_def != NULL && "could not find parameter for array size");

                    // instantiate it
                    IType const       *p_type  = param_def->get_type();
                    IType_array const *a_type  = cast<IType_array>(p_type->skip_type_alias());
                    int               imm_size = ctx.instantiate_type_size(p_type);

                    if (imm_size >= 0) {
                        // was mapped, return a the array size
                        llvm::Value *array_size = ctx.get_constant(imm_size);

                        res = Expression_result::value(array_size);
                    } else if (a_type->is_immediate_sized()) {
                        // is immediate, return a the array size
                        int         size = a_type->get_size();
                        llvm::Value *array_size = ctx.get_constant(size);

                        res = Expression_result::value(array_size);
                    } else {
                        // still deferred size, retrieve it from the descriptor
                        LLVM_context_data *p_data = ctx.get_context_data(param_def);
                        llvm::Value *arr_desc_ptr = p_data->get_var_address();
                        llvm::Value *array_size   = ctx.get_deferred_size_from_ptr(arr_desc_ptr);

                        // must be int typed, but we model array sizes as size_t, so cast
                        array_size = ctx->CreateTrunc(array_size, m_type_mapper.get_int_type());

                        res = Expression_result::value(array_size);
                    }
                }
                break;

            default:
                MDL_ASSERT(!"Unexpected reference kind here");
                res = Expression_result::undef(lookup_type(expr->get_type()));
                break;
            }
        }
        break;

    case mi::mdl::IExpression::EK_UNARY:
        res = translate_unary(ctx, cast<mi::mdl::IExpression_unary>(expr), return_derivs);
        break;

    case mi::mdl::IExpression::EK_BINARY:
        res = translate_binary(ctx, cast<mi::mdl::IExpression_binary>(expr), return_derivs);
        break;

    case mi::mdl::IExpression::EK_CONDITIONAL:
        res = translate_conditional(
            ctx, cast<mi::mdl::IExpression_conditional>(expr), return_derivs);
        break;

    case mi::mdl::IExpression::EK_CALL:
        {
            Call_ast_expr call = cast<mi::mdl::IExpression_call>(expr);
            res = translate_call(ctx, &call);
        }
        break;

    case mi::mdl::IExpression::EK_LET:
        res = translate_let(ctx, cast<mi::mdl::IExpression_let>(expr), return_derivs);
        break;
    }

    if (res.is_unset()) {
        MDL_ASSERT(!"unsupported expression kind");
        res = Expression_result::undef(lookup_type(expr->get_type()));
    }

    res.ensure_deriv_result(ctx, return_derivs);

    return res;
}

// Append the given value to the RO data segment.
size_t LLVM_code_generator::add_to_ro_data_segment(
    mi::mdl::IValue const *v,
    size_t                alloc_size)
{
    // Align to 16byte boundary for SSE.
    // This is necessary, because the next thing allocated may require it
    m_next_ro_data_offset = align_for_sse(m_next_ro_data_offset);

    size_t curr_ofs = m_next_ro_data_offset;
    m_ro_data_values.push_back(v);
    m_next_ro_data_offset += size_t(alloc_size);

    return curr_ofs;
}

// Get the layout data of the target machine.
llvm::DataLayout const *LLVM_code_generator::get_target_layout_data() const
{
    // Note: get the data layout of the native target here. This might be different from the
    // PTX target, but bad things will happen they are different anyway.
    return &m_data_layout;
}

// Get the read-only segment if one was generated.
unsigned char const *LLVM_code_generator::get_ro_segment(size_t &size) const
{
    size = m_next_ro_data_offset;
    return m_ro_segment;
}

// Check if the given value can be stored in the RO data segment.
bool LLVM_code_generator::can_be_stored_in_ro_segment(IType const *t)
{
    switch (t->get_kind()) {
    case IType::TK_ALIAS:
        return can_be_stored_in_ro_segment(t->skip_type_alias());

    case IType::TK_BOOL:
    case IType::TK_INT:
    case IType::TK_ENUM:
    case IType::TK_FLOAT:
    case IType::TK_DOUBLE:
        // these are fine
        return true;
    case IType::TK_STRING:
        // the size is not uniform, so strings can only to stored in the RO
        // segment if they are mapped to IDs
        return m_type_mapper.strings_mapped_to_ids();
    case IType::TK_BSDF:
    case IType::TK_HAIR_BSDF:
    case IType::TK_EDF:
    case IType::TK_VDF:
        // these cannot occur in const data anyway
        return false;
    case IType::TK_VECTOR:
    case IType::TK_MATRIX:
    case IType::TK_COLOR:
        // supported
        return true;
    case IType::TK_ARRAY:
        {
            IType_array const *a_tp = cast<IType_array>(t);

            return can_be_stored_in_ro_segment(a_tp->get_element_type());
        }
    case IType::TK_STRUCT:
        {
            IType_struct const *s_tp = cast<IType_struct>(t);

            for (int i = 0, n = s_tp->get_compound_size(); i < n; ++i) {
                IType const *e_tp = s_tp->get_compound_type(i);

                if (!can_be_stored_in_ro_segment(e_tp))
                    return false;
            }
            return true;
        }
    case IType::TK_FUNCTION:
        return false;
    case IType::TK_LIGHT_PROFILE:
    case IType::TK_TEXTURE:
    case IType::TK_BSDF_MEASUREMENT:
        // theoretically we could support this, but they cannot occur in arrays anyway
        // and we put only big array/structs in the RO segment
        return false;

    case IType::TK_INCOMPLETE:
    case IType::TK_ERROR:
        return false;
    }
    return false;
}

// Find a tag for a given resource if available in the resource tag map.
int LLVM_code_generator::find_resource_tag(IValue_resource const *res) const
{
    if (m_resource_tag_map == NULL)
        return 0;

    // linear search
    Resource_tag_tuple::Kind kind = kind_from_value(res);
    char const               *url = res->get_string_value();

    for (size_t i = 0, n = m_resource_tag_map->size(); i < n; ++i) {
        Resource_tag_tuple const &e = (*m_resource_tag_map)[i];

        // beware of NULL pointer
        if (e.m_kind == kind && (e.m_url == url || strcmp(e.m_url, url) == 0))
            return e.m_tag;
    }
    return 0;
}

// Creates a global constant for a value in the LLVM IR.
llvm::Value *LLVM_code_generator::create_global_const(
    Function_context               &ctx,
    mi::mdl::IValue_compound const *v,
    bool                           &is_ro_segment_ofs)
{
    is_ro_segment_ofs = false;

    mi::mdl::IType const *v_type = v->get_type();
    llvm::Type *tp  = m_type_mapper.lookup_type(m_llvm_context, v_type);

    if (m_use_ro_data_segment) {
        llvm::DataLayout const *dl = get_target_layout_data();

        uint64_t size = dl->getTypeAllocSize(tp);

        // Big data arrays slow down PTXAS and the JIT linker. Move them into an read-only
        // segment that we manage ourself.
        if (size > 1024 && can_be_stored_in_ro_segment(v->get_type())) {
            size_t curr_ofs = add_to_ro_data_segment(v, size);
            is_ro_segment_ofs = true;

            return ctx.get_constant(curr_ofs);
        }
    }

    llvm::Value *cv = new llvm::GlobalVariable(
        *m_module,
        tp,
        /*isConstant=*/true,
        llvm::GlobalValue::InternalLinkage,
        llvm::cast<llvm::Constant>(ctx.get_constant(v)),
        "_global_const");
    return cv;
}

// Translate a RO-data-segment offset into LLVM IR by adding the RO-data-segment offset
// of the state.
llvm::Value *LLVM_code_generator::translate_ro_data_segment_hlsl_offset(
    Function_context &ctx,
    int               cur_offs,
    llvm::Value      *add_val)
{
    llvm::Value *state = ctx.get_state_parameter();
    llvm::Value *adr   = ctx.create_simple_gep_in_bounds(
        state, ctx.get_constant(
        m_type_mapper.get_state_index(Type_mapper::STATE_CORE_RO_DATA_SEG)));
    llvm::Value *arg_block_offs = ctx->CreateLoad(adr);
    if (add_val != NULL)
        arg_block_offs = ctx->CreateAdd(arg_block_offs, add_val);
    return ctx->CreateAdd(arg_block_offs, ctx.get_constant(cur_offs));
}

// Translate a part of the RO-data-segment for HLSL into LLVM IR.
llvm::Value *LLVM_code_generator::translate_ro_data_segment_hlsl_value(
    Function_context             &ctx,
    mi::mdl::IType const         *param_type,
    int                          &cur_offs,
    llvm::Value                  *add_val)
{
    llvm::Value *res;
    param_type = param_type->skip_type_alias();

    switch (param_type->get_kind()) {
    case mi::mdl::IType::TK_BOOL:
        res = ctx->CreateCall(
            m_hlsl_func_rodata_as_bool,
            translate_ro_data_segment_hlsl_offset(ctx, cur_offs, add_val));
        ++cur_offs;
        break;

    case mi::mdl::IType::TK_FLOAT:
        cur_offs = (cur_offs + 3) & ~3;
        res = ctx->CreateCall(
            m_hlsl_func_rodata_as_float,
            translate_ro_data_segment_hlsl_offset(ctx, cur_offs, add_val));
        cur_offs += 4;
        break;

    case mi::mdl::IType::TK_INT:
    case mi::mdl::IType::TK_ENUM:
    case mi::mdl::IType::TK_STRING:
        cur_offs = (cur_offs + 3) & ~3;
        res = ctx->CreateCall(
            m_hlsl_func_rodata_as_int,
            translate_ro_data_segment_hlsl_offset(ctx, cur_offs, add_val));
        cur_offs += 4;
        break;

    case mi::mdl::IType::TK_DOUBLE:
        cur_offs = (cur_offs + 7) & ~7;
        res = ctx->CreateCall(
            m_hlsl_func_rodata_as_double,
            translate_ro_data_segment_hlsl_offset(ctx, cur_offs, add_val));
        cur_offs += 8;
        break;

    case mi::mdl::IType::TK_VECTOR:
    case mi::mdl::IType::TK_MATRIX:
    case mi::mdl::IType::TK_ARRAY:
    case mi::mdl::IType::TK_COLOR:
    case mi::mdl::IType::TK_STRUCT:
        {
            mi::mdl::IType_compound const *ct = cast<mi::mdl::IType_compound>(param_type);

            llvm::Type *res_type = m_type_mapper.lookup_type(m_llvm_context, ct);
            size_t size = size_t(m_data_layout.getTypeAllocSize(res_type));
            int compound_start_offs = cur_offs;

            res = llvm::UndefValue::get(res_type);
            for (int i = 0, n = ct->get_compound_size(); i < n; ++i) {
                mi::mdl::IType const *et = ct->get_compound_type(i);
                res = ctx.create_insert(
                    res,
                    translate_ro_data_segment_hlsl_value(ctx, et, cur_offs, add_val),
                    unsigned(i));
            }

            // compound values might have an higher alignment then the sum of its components
            // TODO: this is probably wrong for { bool, { double, ... }, float }
            //   as the alignment of the first compound element must be applied
            cur_offs = compound_start_offs + size;
        }
        break;

    case mi::mdl::IType::TK_TEXTURE:
    case mi::mdl::IType::TK_LIGHT_PROFILE:
    case mi::mdl::IType::TK_BSDF_MEASUREMENT:
        // resources are mapped to integer in HLSL
        cur_offs = (cur_offs + 3) & ~3;
        res = ctx->CreateCall(
            m_hlsl_func_rodata_as_int,
            translate_ro_data_segment_hlsl_offset(ctx, cur_offs, add_val));
        cur_offs += 4;
        break;

    default:
        MDL_ASSERT(!"Unexpected parameter type");
        res = llvm::UndefValue::get(m_type_mapper.lookup_type(m_llvm_context, param_type));
        break;
    }
    return res;
}

// Translate a value to LLVM IR.
Expression_result LLVM_code_generator::translate_value(
    Function_context      &ctx,
    mi::mdl::IValue const *v)
{
    llvm::Value *res;
    if (mi::mdl::IValue_resource const *r = as<mi::mdl::IValue_resource>(v)) {
        size_t idx = ctx.get_resource_index(r);
        res = ctx.get_constant(int(idx));
    } else {
        // non-resource value
        mi::mdl::IValue::Kind value_kind = v->get_kind();
        if (value_kind == mi::mdl::IValue::VK_ARRAY || value_kind == mi::mdl::IValue::VK_STRUCT) {
            // If arrays are handled as values, we generate currently
            // bad code because they are stored every time to access elements by index.
            // So create a constant in global space. The same is probably true for structs, so
            // handle them also this way here.

            bool is_ofs = false;
            Global_const_map::iterator it = m_global_const_map.find(v);
            if (it == m_global_const_map.end()) {
                // first time we see this constant, create it
                llvm::Value *cv = create_global_const(
                    ctx, cast<mi::mdl::IValue_compound>(v), is_ofs);

                // cache it
                m_global_const_map[v] = Value_offset_pair(cv, is_ofs);
                res = cv;
            } else {
                // was already created, get it from cache
                Value_offset_pair const &pair = it->second;
                res    = pair.value;
                is_ofs = pair.is_offset;
            }

            if (is_ofs) {
                // get from the RO segment
                mi::mdl::IType const *v_type = v->get_type();
                llvm::Type *tp  = m_type_mapper.lookup_type(m_llvm_context, v_type);

                // for HLSL we use user provided functions to read the value
                if (m_target_lang == TL_HLSL) {
                    llvm::ConstantInt *ci = llvm::cast<llvm::ConstantInt>(res);
                    int cur_offs = int(ci->getZExtValue());
                    return Expression_result::offset(
                        ctx.get_constant(cur_offs),
                        Expression_result::OK_RO_DATA_SEGMENT,
                        tp,
                        v_type);
                }

                llvm::Value *ro_seg = get_ro_data_segment(ctx);
                llvm::Value *cv     = ctx->CreateGEP(ro_seg, res);

                // cast the blob to the right type
                llvm::PointerType *ptr_tp = m_type_mapper.get_ptr(tp);
                res = ctx->CreatePointerCast(cv, ptr_tp);
            }

            // this is an address
            return Expression_result::ptr(res);
        }

        // non-array/struct
        res = ctx.get_constant(v);
    }
    return Expression_result::value(res);
}

// Translate a literal expression to LLVM IR.
Expression_result LLVM_code_generator::translate_literal(
    Function_context                   &ctx,
    mi::mdl::IExpression_literal const *lit)
{
    mi::mdl::IValue const *v = lit->get_value();
    Expression_result res(translate_value(ctx, v));

    // need to return a derivative value?
    if (is_deriv_expr(lit))
        return Expression_result::value(ctx.get_dual(res.as_value(ctx)));
    return res;
}

// Translate an unary expression without side-effect to LLVM IR.
llvm::Value *LLVM_code_generator::translate_unary(
    Function_context                     &ctx,
    mi::mdl::IExpression_unary::Operator op,
    llvm::Value                          *arg)
{
    switch (op) {
    case mi::mdl::IExpression_unary::OK_LOGICAL_NOT:
        // logical not is only allowed on bool (and bool vectors)
        {
            llvm::Type *arg_tp = arg->getType();

            if (llvm::ArrayType *arr_tp = llvm::dyn_cast<llvm::ArrayType>(arg_tp)) {
                // must be an integer type here, because logical not is only allowed
                // on MDL bool type
                llvm::IntegerType *e_tp =
                    llvm::cast<llvm::IntegerType>(arg_tp->getArrayElementType());

                llvm::Value *v = llvm::ConstantAggregateZero::get(arr_tp);
                for (unsigned i = 0, n = unsigned(arr_tp->getNumElements()); i < n; ++i) {
                    unsigned idxs[1] = { i };
                    llvm::Value *tmp = ctx->CreateExtractValue(arg, idxs);
                    tmp = ctx->CreateXor(tmp, llvm::ConstantInt::get(e_tp, 1));
                    v = ctx->CreateInsertValue(v, tmp, idxs);
                }
                return v;
            } else {
                // must be an integer or intvector type here, because logical not is only allowed
                // on MDL bool type
                llvm::Value *v;
                if (llvm::VectorType *v_tp = llvm::dyn_cast<llvm::VectorType>(arg_tp)) {
                    llvm::IntegerType *e_tp =
                        llvm::cast<llvm::IntegerType>(v_tp->getElementType());
                    llvm::Value *c =
                        ctx.create_vector_splat(v_tp, llvm::ConstantInt::get(e_tp, 1));
                    v = ctx->CreateXor(arg, c);
                } else {
                    llvm::IntegerType *e_tp = llvm::cast<llvm::IntegerType>(arg_tp);
                    v = ctx->CreateXor(arg, llvm::ConstantInt::get(e_tp, 1));
                }
                return v;
            }
        }
        break;

    case mi::mdl::IExpression_unary::OK_BITWISE_COMPLEMENT:
        {
            llvm::Type *arg_tp = arg->getType();

            if (llvm::ArrayType *arr_tp = llvm::dyn_cast<llvm::ArrayType>(arg_tp)) {
                llvm::Value *v = llvm::ConstantAggregateZero::get(arr_tp);
                for (unsigned i = 0, n = unsigned(arr_tp->getNumElements()); i < n; ++i) {
                    unsigned idxs[1] = { i };
                    llvm::Value *tmp = ctx->CreateExtractValue(arg, idxs);
                    tmp = ctx->CreateNot(tmp);
                    v = ctx->CreateInsertValue(v, tmp, idxs);
                }
                return v;
            } else {
                llvm::Value *v = ctx->CreateNot(arg);
                return v;
            }
        }
        break;

    case mi::mdl::IExpression_unary::OK_POSITIVE:
        // skip
        return arg;

    case mi::mdl::IExpression_unary::OK_NEGATIVE:
        // supported on atomics, vector and matrices
        {
            llvm::Type *arg_tp = arg->getType();

            llvm::Value *v;
            if (llvm::ArrayType *arr_tp = llvm::dyn_cast<llvm::ArrayType>(arg_tp)) {
                llvm::Type *e_tp = arr_tp ->getElementType();
                v = llvm::ConstantAggregateZero::get(arr_tp);
                for (unsigned i = 0, n = unsigned(arr_tp->getNumElements()); i < n; ++i) {
                    unsigned idxs[1] = { i };
                    llvm::Value *tmp = ctx->CreateExtractValue(arg, idxs);
                    if (e_tp->isFPOrFPVectorTy()) {
                        tmp = ctx->CreateFNeg(tmp);
                    } else {
                        // must be integer
                        MDL_ASSERT(e_tp->isIntOrIntVectorTy());
                        tmp = ctx->CreateNeg(tmp);
                    }
                    v = ctx->CreateInsertValue(v, tmp, idxs);
                }
            } else if (m_type_mapper.is_deriv_type(arg_tp)) {
                // note: neg is not allowed on arrays, so this can only be an atomic, a vector
                // or a matrix

                llvm::Type *base_type = m_type_mapper.get_deriv_base_type(arg_tp);
                if (llvm::ArrayType *arr_tp = llvm::dyn_cast<llvm::ArrayType>(base_type)) {
                    // small vector mode or all atomic mode for matrices
                    MDL_ASSERT(arr_tp->getElementType()->isFPOrFPVectorTy());

                    llvm::Value *arg_val = ctx.get_dual_val(arg);
                    llvm::Value *arg_dx  = ctx.get_dual_dx(arg);
                    llvm::Value *arg_dy  = ctx.get_dual_dy(arg);

                    llvm::Value *v_val = llvm::ConstantAggregateZero::get(arr_tp);
                    llvm::Value *v_dx  = v_val;
                    llvm::Value *v_dy  = v_val;

                    for (unsigned i = 0, n = unsigned(arr_tp->getNumElements()); i < n; ++i) {
                        unsigned idxs[1] = { i };
                        llvm::Value *tmp_val = ctx->CreateExtractValue(arg_val, idxs);
                        llvm::Value *tmp_dx  = ctx->CreateExtractValue(arg_dx,  idxs);
                        llvm::Value *tmp_dy  = ctx->CreateExtractValue(arg_dy,  idxs);

                        tmp_val = ctx->CreateFNeg(tmp_val);
                        tmp_dx  = ctx->CreateFNeg(tmp_dx);
                        tmp_dy  = ctx->CreateFNeg(tmp_dy);

                        v_val = ctx->CreateInsertValue(v_val, tmp_val, idxs);
                        v_dx  = ctx->CreateInsertValue(v_dx,  tmp_dx,  idxs);
                        v_dy  = ctx->CreateInsertValue(v_dy,  tmp_dy,  idxs);
                    }
                    v = ctx.get_dual(v_val, v_dx, v_dy);
                } else {
                    // big vector mode for matrices or vector/atomic
                    v = ctx.get_dual(
                            ctx->CreateFNeg(ctx.get_dual_val(arg)),
                            ctx->CreateFNeg(ctx.get_dual_dx(arg)),
                            ctx->CreateFNeg(ctx.get_dual_dy(arg)));
                }
            } else if (arg_tp->isFPOrFPVectorTy()) {
                v = ctx->CreateFNeg(arg);
            } else {
                // must be integer
                MDL_ASSERT(arg_tp->isIntOrIntVectorTy());
                v = ctx->CreateNeg(arg);
            }
            return v;
        }
        break;
    default:
        break;
    }
    MDL_ASSERT(!"unsupported unary operator kind");
    return llvm::UndefValue::get(arg->getType());
}

// Translate an unary expression to LLVM IR.
Expression_result LLVM_code_generator::translate_unary(
    Function_context                 &ctx,
    mi::mdl::IExpression_unary const *un_expr,
    bool                             return_derivs)
{
    mi::mdl::IExpression_unary::Operator op = un_expr->get_operator();
    switch (op) {
    case mi::mdl::IExpression_unary::OK_BITWISE_COMPLEMENT:
    case mi::mdl::IExpression_unary::OK_LOGICAL_NOT:
    case mi::mdl::IExpression_unary::OK_POSITIVE:
    case mi::mdl::IExpression_unary::OK_NEGATIVE:
        {
            llvm::Value *arg = translate_expression_value(
                ctx, un_expr->get_argument(), return_derivs);
            return Expression_result::value(translate_unary(ctx, op, arg));
        }
        break;

    case mi::mdl::IExpression_unary::OK_PRE_INCREMENT:
    case mi::mdl::IExpression_unary::OK_PRE_DECREMENT:
    case mi::mdl::IExpression_unary::OK_POST_INCREMENT:
    case mi::mdl::IExpression_unary::OK_POST_DECREMENT:
        return translate_inplace_change_expression(ctx, un_expr);
    case mi::mdl::IExpression_unary::OK_CAST:
        return translate_cast_expression(ctx, un_expr, return_derivs);
    }
    MDL_ASSERT(!"unsupported unary operator kind");
    return Expression_result::undef(lookup_type(un_expr->get_type()));
}

// Helper for pre/post inc/decrement.
void LLVM_code_generator::do_inner_inc_dec(
    Function_context                     &ctx,
    mi::mdl::IExpression_unary::Operator op,
    llvm::Type                           *tp,
    llvm::Value                          *old_v,
    llvm::Value                          *&r,
    llvm::Value                          *&v)
{
    if (tp->isIntOrIntVectorTy()) {
        llvm::Value *one = llvm::ConstantInt::get(tp, uint64_t(1));
        switch (op) {
        case mi::mdl::IExpression_unary::OK_PRE_INCREMENT:
        case mi::mdl::IExpression_unary::OK_POST_INCREMENT:
            v = ctx->CreateAdd(old_v, one);
            break;
        case mi::mdl::IExpression_unary::OK_PRE_DECREMENT:
        case mi::mdl::IExpression_unary::OK_POST_DECREMENT:
            v = ctx->CreateSub(old_v, one);
            break;
        default:
            llvm_unreachable("Unexpected inc/dec");
        }
    } else if (m_type_mapper.is_deriv_type(tp)) {
        llvm::Value *one = llvm::ConstantFP::get(m_type_mapper.get_deriv_base_type(tp), 1.0);
        switch (op) {
        case mi::mdl::IExpression_unary::OK_PRE_INCREMENT:
        case mi::mdl::IExpression_unary::OK_POST_INCREMENT:
            v = ctx.get_dual(
                ctx->CreateFAdd(ctx.get_dual_val(old_v), one),
                ctx.get_dual_dx(old_v),
                ctx.get_dual_dy(old_v));
            break;
        case mi::mdl::IExpression_unary::OK_PRE_DECREMENT:
        case mi::mdl::IExpression_unary::OK_POST_DECREMENT:
            v = ctx.get_dual(
                ctx->CreateFSub(ctx.get_dual_val(old_v), one),
                ctx.get_dual_dx(old_v),
                ctx.get_dual_dy(old_v));
            break;
        default:
            llvm_unreachable("Unexpected inc/dec");
        }
    } else {
        llvm::Value *one = llvm::ConstantFP::get(tp, 1.0);
        switch (op) {
        case mi::mdl::IExpression_unary::OK_PRE_INCREMENT:
        case mi::mdl::IExpression_unary::OK_POST_INCREMENT:
            v = ctx->CreateFAdd(old_v, one);
            break;
        case mi::mdl::IExpression_unary::OK_PRE_DECREMENT:
        case mi::mdl::IExpression_unary::OK_POST_DECREMENT:
            v = ctx->CreateFSub(old_v, one);
            break;
        default:
            llvm_unreachable("Unexpected inc/dec");
        }
    }

    switch (op) {
    case mi::mdl::IExpression_unary::OK_PRE_INCREMENT:
    case mi::mdl::IExpression_unary::OK_PRE_DECREMENT:
        r = v;
        break;
    case mi::mdl::IExpression_unary::OK_POST_INCREMENT:
    case mi::mdl::IExpression_unary::OK_POST_DECREMENT:
        r = old_v;
        break;
    default:
        llvm_unreachable("Unexpected inc/dec");
    }
}

// Translate an inplace change expression.
Expression_result LLVM_code_generator::translate_inplace_change_expression(
    Function_context                 &ctx,
    mi::mdl::IExpression_unary const *un_expr)
{
    mi::mdl::IExpression const *arg = un_expr->get_argument();
    llvm::Value                *adr = translate_lval_expression(ctx, arg);

    llvm::Value *old_v = ctx->CreateLoad(adr);
    llvm::Type  *tp    = old_v->getType();

    llvm::Value *r, *v;

    mi::mdl::IExpression_unary::Operator op = un_expr->get_operator();

    if (llvm::ArrayType *arr_tp = llvm::dyn_cast<llvm::ArrayType>(tp)) {
        // vectors encoded as arrays
        llvm::Type *e_tp = arr_tp->getElementType();

        r = llvm::ConstantAggregateZero::get(tp);
        v = r;
        for (unsigned i = 0, n = unsigned(arr_tp->getNumContainedTypes()); i < n; ++i) {
            unsigned idxes[1] = { i };

            llvm::Value *e_old_v = ctx->CreateExtractValue(old_v, idxes);

            llvm::Value *e_r, *e_v;
            do_inner_inc_dec(ctx, op, e_tp, e_old_v, e_r, e_v);

            r = ctx->CreateInsertValue(r, e_r, idxes);
            v = ctx->CreateInsertValue(v, e_v, idxes);
        }
    } else {
        // scalar or vector type
        do_inner_inc_dec(ctx, op, tp, old_v, r, v);
    }

    ctx->CreateStore(v, adr);
    return Expression_result::value(r);
}

// Translate an MDL 1.5 cast expression.
Expression_result LLVM_code_generator::translate_cast_expression(
    Function_context                 &ctx,
    Expression_result                &arg,
    mi::mdl::IType const             *res_type)
{
    if (is<mi::mdl::IType_struct>(res_type)) {
        // struct-to-struct cast
        llvm::Type *llvm_type = lookup_type(res_type);

        llvm::Value *ptr = arg.as_ptr(ctx);
        ptr = ctx->CreatePointerCast(ptr, Type_mapper::get_ptr(llvm_type));
        return Expression_result::ptr(ptr);
    }
    return arg;
}

// Translate an MDL 1.5 cast expression.
Expression_result LLVM_code_generator::translate_cast_expression(
    Function_context                 &ctx,
    mi::mdl::IExpression_unary const *un_expr,
    bool                             return_derivs)
{
    Expression_result res = translate_expression(ctx, un_expr->get_argument(), return_derivs);
    mi::mdl::IType const *res_type = un_expr->get_type()->skip_type_alias();

    return translate_cast_expression(ctx, res, res_type);
}

// Translate a binary expression to LLVM IR.
Expression_result LLVM_code_generator::translate_binary(
    Function_context                  &ctx,
    mi::mdl::IExpression_binary const *call,
    bool                              return_derivs)
{
    mi::mdl::IExpression const *lhs = call->get_left_argument();
    mi::mdl::IExpression const *rhs = call->get_right_argument();

    mi::mdl::IExpression_binary::Operator op = call->get_operator();
    switch (op) {
    case mi::mdl::IExpression_binary::OK_SELECT:
        {
            mi::mdl::IExpression_reference const *ref = cast<mi::mdl::IExpression_reference>(rhs);
            mi::mdl::IDefinition const *def = ref->get_definition();

            MDL_ASSERT(def->get_kind() == mi::mdl::IDefinition::DK_MEMBER);
            int index = def->get_field_index();

            llvm::Value *v;
            Expression_result  compound    = translate_expression(ctx, lhs, return_derivs);
            llvm::Type        *compound_tp = compound.get_value_type();
            if (llvm::isa<llvm::VectorType>(compound_tp)) {
                // extracting a vector component
                v = ctx->CreateExtractElement(compound.as_value(ctx), ctx.get_constant(index));
            } else if (m_type_mapper.is_deriv_type(compound_tp)) {
                v = ctx.extract_dual(compound.as_value(ctx), unsigned(index));
            } else {
                // default aggregate extract
                if (compound.is_pointer()) {
                    // avoid value copy by returning a pointer
                    v = ctx.create_simple_gep_in_bounds(compound.as_ptr(ctx), index);
                    return Expression_result::ptr(v);
                }

                v = ctx->CreateExtractValue(compound.as_value(ctx), index);
            }
            return Expression_result::value(v);
        }
        break;

    case mi::mdl::IExpression_binary::OK_ARRAY_INDEX:
        {
            mi::mdl::IType const    *comp_type = lhs->get_type()->skip_type_alias();
            Expression_result       comp       = translate_expression(ctx, lhs, return_derivs);
            llvm::Value             *index     =
                translate_expression_value(ctx, rhs, /*return_derivs=*/ false);
            mi::mdl::Position const *index_pos = &rhs->access_position();
            return translate_index_expression(ctx, comp_type, comp, index, index_pos);
        }
        break;

    case mi::mdl::IExpression_binary::OK_MULTIPLY:
    case mi::mdl::IExpression_binary::OK_DIVIDE:
    case mi::mdl::IExpression_binary::OK_MODULO:
    case mi::mdl::IExpression_binary::OK_PLUS:
    case mi::mdl::IExpression_binary::OK_MINUS:
    case mi::mdl::IExpression_binary::OK_SHIFT_LEFT:
    case mi::mdl::IExpression_binary::OK_SHIFT_RIGHT:
    case mi::mdl::IExpression_binary::OK_UNSIGNED_SHIFT_RIGHT:
    case mi::mdl::IExpression_binary::OK_BITWISE_AND:
    case mi::mdl::IExpression_binary::OK_BITWISE_XOR:
    case mi::mdl::IExpression_binary::OK_BITWISE_OR:
    case mi::mdl::IExpression_binary::OK_LOGICAL_AND:
    case mi::mdl::IExpression_binary::OK_LOGICAL_OR:
        return Expression_result::value(translate_binary_no_side_effect(ctx, call, return_derivs));

    case mi::mdl::IExpression_binary::OK_LESS:
    case mi::mdl::IExpression_binary::OK_LESS_OR_EQUAL:
    case mi::mdl::IExpression_binary::OK_GREATER_OR_EQUAL:
    case mi::mdl::IExpression_binary::OK_GREATER:
    case mi::mdl::IExpression_binary::OK_EQUAL:
    case mi::mdl::IExpression_binary::OK_NOT_EQUAL:
        {
            MDL_ASSERT(!return_derivs && "boolean values cannot have derivatives");

            mi::mdl::IType const *l_type = lhs->get_type();
            mi::mdl::IType const *r_type = rhs->get_type();

            llvm::Value *lv = translate_expression_value(ctx, lhs, /*return_derivs=*/ false);
            llvm::Value *rv = translate_expression_value(ctx, rhs, /*return_derivs=*/ false);

            return translate_compare(ctx, op, l_type, lv, r_type, rv);
        }

    case mi::mdl::IExpression_binary::OK_ASSIGN:
    case mi::mdl::IExpression_binary::OK_MULTIPLY_ASSIGN:
    case mi::mdl::IExpression_binary::OK_DIVIDE_ASSIGN:
    case mi::mdl::IExpression_binary::OK_MODULO_ASSIGN:
    case mi::mdl::IExpression_binary::OK_PLUS_ASSIGN:
    case mi::mdl::IExpression_binary::OK_MINUS_ASSIGN:
    case mi::mdl::IExpression_binary::OK_SHIFT_LEFT_ASSIGN:
    case mi::mdl::IExpression_binary::OK_SHIFT_RIGHT_ASSIGN:
    case mi::mdl::IExpression_binary::OK_UNSIGNED_SHIFT_RIGHT_ASSIGN:
    case mi::mdl::IExpression_binary::OK_BITWISE_AND_ASSIGN:
    case mi::mdl::IExpression_binary::OK_BITWISE_XOR_ASSIGN:
    case mi::mdl::IExpression_binary::OK_BITWISE_OR_ASSIGN:
        return translate_assign(ctx, call, return_derivs);

    case mi::mdl::IExpression_binary::OK_SEQUENCE:
        // just execute all expressions
        (void) translate_expression(ctx, lhs, /*return_derivs=*/ false);
        return translate_expression(ctx, rhs, return_derivs);
    }
    MDL_ASSERT(!"unsupported binary operator kind");
    Expression_result res = Expression_result::undef(lookup_type(call->get_type()));
    res.ensure_deriv_result(ctx, return_derivs);
    return res;
}

/// Returns true if the given LLVM value might be equal zero.
///
/// \param v  the value to check
static bool maybe_zero(llvm::Value *v)
{
    if (llvm::Constant *c = llvm::dyn_cast<llvm::Constant>(v)) {
        return c->isNullValue();
    }
    return true;
}

// Translate a side effect free binary expression to LLVM IR.
llvm::Value *LLVM_code_generator::translate_binary_no_side_effect(
    Function_context                  &ctx,
    mi::mdl::IExpression_binary const *call,
    bool                              return_derivs)
{
    llvm::Value *res = NULL;

    mi::mdl::IExpression const *lhs = call->get_left_argument();
    mi::mdl::IExpression const *rhs = call->get_right_argument();
    mi::mdl::Position const *call_pos = &call->access_position();

    mi::mdl::IExpression_binary::Operator op = call->get_operator();

    switch (op) {
    case mi::mdl::IExpression_binary::OK_MULTIPLY:
    case mi::mdl::IExpression_binary::OK_MULTIPLY_ASSIGN:
        return translate_multiply(
            ctx, lookup_type_or_deriv_type(ctx, call), lhs, rhs, return_derivs);

    case mi::mdl::IExpression_binary::OK_LOGICAL_AND:
    case mi::mdl::IExpression_binary::OK_LOGICAL_OR:
        if (is<mi::mdl::IType_bool>(lhs->get_type()) && is<mi::mdl::IType_bool>(rhs->get_type())) {
            // shortcut evaluation here
            llvm::Value      *tmp      = ctx.create_local(m_type_mapper.get_bool_type(), "sc_tmp");
            llvm::BasicBlock *true_bb  = ctx.create_bb("sc_true");
            llvm::BasicBlock *false_bb = ctx.create_bb("sc_false");
            llvm::BasicBlock *end_bb   = ctx.create_bb("sc_end");

            translate_boolean_branch(ctx, call, true_bb, false_bb);

            ctx->SetInsertPoint(true_bb);
            ctx->CreateStore(ctx.get_constant(true), tmp);
            ctx->CreateBr(end_bb);

            ctx->SetInsertPoint(false_bb);
            ctx->CreateStore(ctx.get_constant(false), tmp);
            ctx->CreateBr(end_bb);

            ctx->SetInsertPoint(end_bb);
            res = ctx->CreateLoad(tmp);
            return res;
        } else {
            // no shortcut execution possible on vectors, execute bitwise
            if (op == mi::mdl::IExpression_binary::OK_LOGICAL_AND)
                op = mi::mdl::IExpression_binary::OK_BITWISE_AND;
            else
                op = mi::mdl::IExpression_binary::OK_BITWISE_OR;
        }
        break;

    default:
        break;
    }

    llvm::Value *l = translate_expression_value(ctx, lhs, return_derivs);
    llvm::Value *r = translate_expression_value(ctx, rhs, return_derivs);

    llvm::Type *res_type = lookup_type_or_deriv_type(ctx, call);
    if (m_type_mapper.is_deriv_type(res_type)) {
        llvm::Type *res_val_type = m_type_mapper.get_deriv_base_type(res_type);
        if (llvm::ArrayType *a_tp = llvm::dyn_cast<llvm::ArrayType>(res_val_type)) {
            // assume element-wise operation
            res = llvm::UndefValue::get(a_tp);
            llvm::Value *res_dx = res;
            llvm::Value *res_dy = res;

            bool l_is_arr = m_type_mapper.skip_deriv_type(l->getType())->isArrayTy();
            bool r_is_arr = m_type_mapper.skip_deriv_type(r->getType())->isArrayTy();

            for (size_t i = 0, n = a_tp->getArrayNumElements(); i < n; ++i) {
                llvm::Value *l_elem = l;
                if (l_is_arr) {
                    l_elem = ctx.extract_dual(l, unsigned(i));
                }

                llvm::Value *r_elem = r;
                if (r_is_arr) {
                    r_elem = ctx.extract_dual(r, unsigned(i));
                }

                llvm::Value *tmp = translate_binary_basic(ctx, op, l_elem, r_elem, call_pos);
                if (tmp == NULL) {
                    res = NULL;
                    break;
                }
                res    = ctx.create_insert(res,    ctx.get_dual_val(tmp), unsigned(i));
                res_dx = ctx.create_insert(res_dx, ctx.get_dual_dx(tmp),  unsigned(i));
                res_dy = ctx.create_insert(res_dy, ctx.get_dual_dy(tmp),  unsigned(i));
            }
            res = ctx.get_dual(res, res_dx, res_dy);
        } else {
            // not array typed (and derivable, thus not integer-based)
            res = translate_binary_basic(ctx, op, l, r, call_pos);
        }
    } else if (llvm::ArrayType *a_tp = llvm::dyn_cast<llvm::ArrayType>(res_type)) {
        res_type = a_tp->getArrayElementType();

        // assume element-wise operation
        unsigned idxes[1];
        res = llvm::ConstantAggregateZero::get(a_tp);

        bool l_is_arr = l->getType()->isArrayTy();
        bool r_is_arr = r->getType()->isArrayTy();

        for (size_t i = 0, n = a_tp->getArrayNumElements(); i < n; ++i) {
            idxes[0] = unsigned(i);

            llvm::Value *l_elem = l;
            if (l_is_arr) {
                l_elem = ctx->CreateExtractValue(l, idxes);
            }

            llvm::Value *r_elem = r;
            if (r_is_arr) {
                r_elem = ctx->CreateExtractValue(r, idxes);
            }

            llvm::Value *tmp = translate_binary_basic(ctx, op, l_elem, r_elem, call_pos);
            if (tmp == NULL) {
                res = NULL;
                break;
            }
            res = ctx->CreateInsertValue(res, tmp, idxes);
        }
    } else if (res_type->isIntOrIntVectorTy() &&
               (op == IExpression_binary::OK_DIVIDE ||
                op == IExpression_binary::OK_DIVIDE_ASSIGN ||
                op == IExpression_binary::OK_MODULO ||
                op == IExpression_binary::OK_MODULO_ASSIGN) && maybe_zero(r)) {
        llvm::Type *r_type = lookup_type(rhs->get_type());
        if (llvm::VectorType *v_tp = llvm::dyn_cast<llvm::VectorType>(r_type)) {
            res_type = v_tp->getElementType();

            // transform into element wise operation, because we must check the rhs element wise ...
            res = llvm::Constant::getNullValue(v_tp);

            bool l_is_vec = l->getType()->isVectorTy();
            for (unsigned i = 0, n = unsigned(v_tp->getNumElements()); i < n; ++i) {
                llvm::Value *idx = ctx.get_constant(int(i));

                llvm::Value *l_elem = l;
                if (l_is_vec) {
                    l_elem = ctx->CreateExtractElement(l, idx);
                }

                llvm::Value *r_elem = ctx->CreateExtractElement(r, idx);

                llvm::Value *tmp = translate_binary_basic(ctx, op, l_elem, r_elem, call_pos);
                if (tmp == NULL) {
                    res = NULL;
                    break;
                }
                res = ctx->CreateInsertElement(res, tmp, idx);
            }
        } else {
            res = translate_binary_basic(ctx, op, l, r, call_pos);
        }
    } else {
        // not array typed, and not vector typed division
        res = translate_binary_basic(ctx, op, l, r, call_pos);
    }

    if (res == NULL) {
        MDL_ASSERT(!"Unsupported binary operator");
        res = llvm::UndefValue::get(res_type);
    }
    return res;
}

// Translate a side effect free binary expression to LLVM IR.
llvm::Value *LLVM_code_generator::translate_binary_no_side_effect(
    Function_context          &ctx,
    mi::mdl::ICall_expr const *bin_expr)
{
    mi::mdl::IDefinition::Semantics sema = bin_expr->get_semantics();
    mi::mdl::IExpression_binary::Operator op =
        mi::mdl::IExpression_binary::Operator(semantic_to_operator(sema));
    llvm::Value *res = NULL;

    // on the DAG, the return_derivs parameter is ignored, as the nodes already know, whether
    // they have to return derivatives or not
    bool derivs = false;

    switch (op) {
    case mi::mdl::IExpression_binary::OK_ARRAY_INDEX:
        {
            mi::mdl::IType const *comp_type = bin_expr->get_argument_type(0);
            Expression_result    comp = bin_expr->translate_argument(*this, ctx, 0, derivs);
            llvm::Value          *index =
                bin_expr->translate_argument_value(*this, ctx, 1, /*return_derivs=*/ false);
            mi::mdl::Position    *index_pos = NULL;

            Expression_result res = translate_index_expression(
                ctx, comp_type, comp, index, index_pos);
            return res.as_value(ctx);
        }
        break;

    case mi::mdl::IExpression_binary::OK_MULTIPLY:
        {
            mi::mdl::IType const *l_type = bin_expr->get_argument_type(0);
            llvm::Value          *l = bin_expr->translate_argument_value(*this, ctx, 0, derivs);

            mi::mdl::IType const *r_type = bin_expr->get_argument_type(1);
            llvm::Value          *r = bin_expr->translate_argument_value(*this, ctx, 1, derivs);

            return translate_multiply(
                ctx, lookup_type_or_deriv_type(ctx, bin_expr), l_type, l, r_type, r);
        }

    case mi::mdl::IExpression_binary::OK_LOGICAL_AND:
    case mi::mdl::IExpression_binary::OK_LOGICAL_OR:
        {
            mi::mdl::IType const *l_type = bin_expr->get_argument_type(0);
            mi::mdl::IType const *r_type = bin_expr->get_argument_type(1);

            if (is<mi::mdl::IType_bool>(l_type) && is<mi::mdl::IType_bool>(r_type)) {
                // shortcut evaluation here
                llvm::Value      *tmp      = ctx.create_local(
                    m_type_mapper.get_bool_type(), "sc_tmp");
                llvm::BasicBlock *true_bb  = ctx.create_bb("sc_true");
                llvm::BasicBlock *false_bb = ctx.create_bb("sc_false");
                llvm::BasicBlock *end_bb   = ctx.create_bb("sc_end");

                bin_expr->translate_boolean_branch(*this, ctx, true_bb, false_bb);

                ctx->SetInsertPoint(true_bb);
                ctx->CreateStore(ctx.get_constant(true), tmp);
                ctx->CreateBr(end_bb);

                ctx->SetInsertPoint(false_bb);
                ctx->CreateStore(ctx.get_constant(false), tmp);
                ctx->CreateBr(end_bb);

                ctx->SetInsertPoint(end_bb);
                res = ctx->CreateLoad(tmp);
                return res;
            } else {
                // no shortcut execution possible on vectors, execute bitwise
                if (op == mi::mdl::IExpression_binary::OK_LOGICAL_AND)
                    op = mi::mdl::IExpression_binary::OK_BITWISE_AND;
                else
                    op = mi::mdl::IExpression_binary::OK_BITWISE_OR;
            }
        }
        break;

    case mi::mdl::IExpression_binary::OK_LESS:
    case mi::mdl::IExpression_binary::OK_LESS_OR_EQUAL:
    case mi::mdl::IExpression_binary::OK_GREATER_OR_EQUAL:
    case mi::mdl::IExpression_binary::OK_GREATER:
    case mi::mdl::IExpression_binary::OK_EQUAL:
    case mi::mdl::IExpression_binary::OK_NOT_EQUAL:
        {
            llvm::Value *l = bin_expr->translate_argument_value(*this, ctx, 0, derivs);
            llvm::Value *r = bin_expr->translate_argument_value(*this, ctx, 1, derivs);

            mi::mdl::IType const *l_type = bin_expr->get_argument_type(0);
            mi::mdl::IType const *r_type = bin_expr->get_argument_type(1);

            return translate_compare(ctx, op, l_type, l, r_type, r).as_value(ctx);
        }

    default:
        break;
    }

    llvm::Value *l = bin_expr->translate_argument_value(*this, ctx, 0, derivs);
    llvm::Value *r = bin_expr->translate_argument_value(*this, ctx, 1, derivs);
    mi::mdl::Position const *bin_expr_pos = bin_expr->get_position();

    llvm::Type *res_type = lookup_type(bin_expr->get_type());
    if (m_type_mapper.is_deriv_type(res_type)) {
        llvm::Type *res_val_type = m_type_mapper.get_deriv_base_type(res_type);
        if (llvm::ArrayType *a_tp = llvm::dyn_cast<llvm::ArrayType>(res_val_type)) {
            // assume element-wise operation
            res = llvm::UndefValue::get(a_tp);
            llvm::Value *res_dx = res;
            llvm::Value *res_dy = res;

            bool l_is_arr = m_type_mapper.skip_deriv_type(l->getType())->isArrayTy();
            bool r_is_arr = m_type_mapper.skip_deriv_type(r->getType())->isArrayTy();

            for (size_t i = 0, n = a_tp->getArrayNumElements(); i < n; ++i) {
                llvm::Value *l_elem = l;
                if (l_is_arr) {
                    l_elem = ctx.extract_dual(l, unsigned(i));
                }

                llvm::Value *r_elem = r;
                if (r_is_arr) {
                    r_elem = ctx.extract_dual(r, unsigned(i));
                }

                llvm::Value *tmp = translate_binary_basic(ctx, op, l_elem, r_elem, bin_expr_pos);
                if (tmp == NULL) {
                    res = NULL;
                    break;
                }
                res    = ctx.create_insert(res,    ctx.get_dual_val(tmp), unsigned(i));
                res_dx = ctx.create_insert(res_dx, ctx.get_dual_dx(tmp),  unsigned(i));
                res_dy = ctx.create_insert(res_dy, ctx.get_dual_dy(tmp),  unsigned(i));
            }
            res = ctx.get_dual(res, res_dx, res_dy);
        } else {
            // not array typed
            res = translate_binary_basic(ctx, op, l, r, bin_expr_pos);
        }
    } else if (llvm::ArrayType *a_tp = llvm::dyn_cast<llvm::ArrayType>(res_type)) {
        // assume element-wise operation
        unsigned idxes[1];
        res = llvm::ConstantAggregateZero::get(a_tp);

        bool l_is_arr = l->getType()->isArrayTy();
        bool r_is_arr = r->getType()->isArrayTy();

        for (size_t i = 0, n = a_tp->getArrayNumElements(); i < n; ++i) {
            idxes[0] = unsigned(i);

            llvm::Value *l_elem = l;
            if (l_is_arr) {
                l_elem = ctx->CreateExtractValue(l, idxes);
            }

            llvm::Value *r_elem = r;
            if (r_is_arr) {
                r_elem = ctx->CreateExtractValue(r, idxes);
            }

            llvm::Value *tmp = translate_binary_basic(ctx, op, l_elem, r_elem, bin_expr_pos);
            if (tmp == NULL) {
                res = NULL;
                break;
            }
            res = ctx->CreateInsertValue(res, tmp, idxes);
        }
    } else {
        // not array typed
        res = translate_binary_basic(ctx, op, l, r, bin_expr_pos);
    }

    if (res == NULL) {
        MDL_ASSERT(!"Unsupported binary operator");
        res = llvm::UndefValue::get(res_type);
    }
    return res;
}

// Translate a side effect free binary simple expressions that require only one
// instruction to LLVM IR.
llvm::Value *LLVM_code_generator::translate_binary_basic(
    Function_context                      &ctx,
    mi::mdl::IExpression_binary::Operator op,
    llvm::Value                           *l,
    llvm::Value                           *r,
    mi::mdl::Position const               *expr_pos)
{
    llvm::Type  *l_type = l->getType();
    llvm::Type  *r_type = r->getType();
    llvm::Value *rhs   = r;
    bool         l_is_deriv = m_type_mapper.is_deriv_type(l_type);

    if (l_type != r_type) {
        // ensure that l and r are derivative values, if one is a derivative value
        bool r_is_deriv = m_type_mapper.is_deriv_type(r_type);
        if (l_is_deriv && !r_is_deriv) {
            r = ctx.get_dual(r);
            r_type = r->getType();
            r_is_deriv = true;
        } else if (r_is_deriv && !l_is_deriv) {
            l = ctx.get_dual(l);
            l_type = l->getType();
            l_is_deriv = true;
        }

        // still not same type?
        if (l_type != r_type) {
            // This should only happen for mixed mode vector/matrix scalar ops
            // which are executed element-wise.
            // Create a splat to both arguments have the same type.
            if (llvm::VectorType *v_tp = llvm::dyn_cast<llvm::VectorType>(l_type)) {
                MDL_ASSERT(r_type->isFloatingPointTy() || r_type->isIntegerTy());
                r = ctx.create_vector_splat(v_tp, r);
                r_type = v_tp;
            } else if (llvm::VectorType *v_tp = llvm::dyn_cast<llvm::VectorType>(r_type)) {
                MDL_ASSERT(l_type->isFloatingPointTy() || l_type->isIntegerTy());
                l = ctx.create_vector_splat(v_tp, l);
                l_type = v_tp;
            } else if (l_is_deriv) {  // then r is also a derivative
                if (llvm::VectorType *v_tp = llvm::dyn_cast<llvm::VectorType>(
                    m_type_mapper.get_deriv_base_type(l_type)))
                {
                    MDL_ASSERT(m_type_mapper.get_deriv_base_type(r_type)->isFloatingPointTy() ||
                        m_type_mapper.get_deriv_base_type(r_type)->isIntegerTy());
                    llvm::Value *val = ctx.create_vector_splat(v_tp, ctx.get_dual_val(r));
                    llvm::Value *dx  = ctx.create_vector_splat(v_tp, ctx.get_dual_dx(r));
                    llvm::Value *dy  = ctx.create_vector_splat(v_tp, ctx.get_dual_dy(r));
                    r = ctx.get_dual(val, dx, dy);
                    r_type = r->getType();
                }
                else if (llvm::VectorType *v_tp = llvm::dyn_cast<llvm::VectorType>(
                    m_type_mapper.get_deriv_base_type(r_type)))
                {
                    MDL_ASSERT(m_type_mapper.get_deriv_base_type(l_type)->isFloatingPointTy() ||
                        m_type_mapper.get_deriv_base_type(l_type)->isIntegerTy());
                    llvm::Value *val = ctx.create_vector_splat(v_tp, ctx.get_dual_val(l));
                    llvm::Value *dx  = ctx.create_vector_splat(v_tp, ctx.get_dual_dx(l));
                    llvm::Value *dy  = ctx.create_vector_splat(v_tp, ctx.get_dual_dy(l));
                    l = ctx.get_dual(val, dx, dy);
                    l_type = l->getType();
                }
            }
        }
        MDL_ASSERT(l_type == r_type);
    }

    bool is_integer_op = l_type->isIntOrIntVectorTy();
    llvm::Value *res = NULL;

    switch (op) {
    case mi::mdl::IExpression_binary::OK_DIVIDE:
    case mi::mdl::IExpression_binary::OK_DIVIDE_ASSIGN:
        if (is_integer_op) {
            if (!maybe_zero(r)) {
                // no exception possible
                res = ctx->CreateSDiv(l, r);
            } else if (m_divzero_check_exception_disabled) {
                llvm::BasicBlock *ok_bb   = ctx.create_bb("div_by_non_zero");
                llvm::BasicBlock *fail_bb = ctx.create_bb("div_by_zero");
                llvm::BasicBlock *end_bb  = ctx.create_bb("div_end");
                llvm::Value      *tmp     = ctx.create_local(l_type, "res");

                ctx.create_non_zero_check_cmp(rhs, ok_bb, fail_bb);
                ctx->SetInsertPoint(ok_bb);
                {
                    res = ctx->CreateSDiv(l, r);
                    ctx->CreateStore(res, tmp);
                    ctx->CreateBr(end_bb);
                }
                ctx->SetInsertPoint(fail_bb);
                {
                    ctx->CreateStore(llvm::Constant::getAllOnesValue(l_type), tmp);
                    ctx->CreateBr(end_bb);
                }
                ctx->SetInsertPoint(end_bb);
                {
                    res = ctx->CreateLoad(tmp);
                }
            } else {
                ctx.create_div_check_with_exception(rhs, Exc_location(*this, expr_pos));
                res = ctx->CreateSDiv(l, r);
            }
        } else if (l_is_deriv) {
            // (l / r)' = (l' * r - r' * l) / r^2
            llvm::Value *l_val = ctx.get_dual_val(l);
            llvm::Value *r_val = ctx.get_dual_val(r);
            llvm::Value *r_square = ctx->CreateFMul(r_val, r_val);

            llvm::Value *val = ctx->CreateFDiv(l_val, r_val);
            llvm::Value *dx = ctx->CreateFDiv(
                ctx->CreateFSub(
                    ctx->CreateFMul(ctx.get_dual_dx(l), r_val),
                    ctx->CreateFMul(ctx.get_dual_dx(r), l_val)),
                r_square);
            llvm::Value *dy = ctx->CreateFDiv(
                ctx->CreateFSub(
                    ctx->CreateFMul(ctx.get_dual_dy(l), r_val),
                    ctx->CreateFMul(ctx.get_dual_dy(r), l_val)),
                r_square);

            res = ctx.get_dual(val, dx, dy);
        } else {
            res = ctx->CreateFDiv(l, r);
        }
        break;

    case mi::mdl::IExpression_binary::OK_MODULO:
    case mi::mdl::IExpression_binary::OK_MODULO_ASSIGN:
        // only on integer and integer vectors
        if (!maybe_zero(r)) {
            // no exception possible
            res = ctx->CreateSRem(l, r);
        } else if (m_divzero_check_exception_disabled) {
            llvm::BasicBlock *ok_bb   = ctx.create_bb("mod_by_non_zero");
            llvm::BasicBlock *fail_bb = ctx.create_bb("mod_by_zerol");
            llvm::BasicBlock *end_bb  = ctx.create_bb("mod_end");
            llvm::Value      *tmp     = ctx.create_local(l_type, "res");

            ctx.create_non_zero_check_cmp(rhs, ok_bb, fail_bb);
            ctx->SetInsertPoint(ok_bb);
            {
                res = ctx->CreateSRem(l, r);
                ctx->CreateStore(res, tmp);
                ctx->CreateBr(end_bb);
            }
            ctx->SetInsertPoint(fail_bb);
            {
                ctx->CreateStore(llvm::Constant::getNullValue(l_type), tmp);
                ctx->CreateBr(end_bb);
            }
            ctx->SetInsertPoint(end_bb);
            {
                res = ctx->CreateLoad(tmp);
            }
        } else {
            ctx.create_div_check_with_exception(rhs, Exc_location(*this, expr_pos));
            res = ctx->CreateSRem(l, r);
        }
        break;

    case mi::mdl::IExpression_binary::OK_PLUS:
    case mi::mdl::IExpression_binary::OK_PLUS_ASSIGN:
        if (is_integer_op)
            res = ctx->CreateAdd(l, r);
        else {
            if (l_is_deriv) {
                res = ctx.get_dual(
                    ctx->CreateFAdd(ctx.get_dual_val(l), ctx.get_dual_val(r)),
                    ctx->CreateFAdd(ctx.get_dual_dx(l),  ctx.get_dual_dx(r)),
                    ctx->CreateFAdd(ctx.get_dual_dy(l),  ctx.get_dual_dy(r)));
            } else
                res = ctx->CreateFAdd(l, r);
        }
        break;

    case mi::mdl::IExpression_binary::OK_MINUS:
    case mi::mdl::IExpression_binary::OK_MINUS_ASSIGN:
        if (is_integer_op)
            res = ctx->CreateSub(l, r);
        else {
            if (l_is_deriv) {
                res = ctx.get_dual(
                    ctx->CreateFSub(ctx.get_dual_val(l), ctx.get_dual_val(r)),
                    ctx->CreateFSub(ctx.get_dual_dx(l),  ctx.get_dual_dx(r)),
                    ctx->CreateFSub(ctx.get_dual_dy(l),  ctx.get_dual_dy(r)));
            } else
                res = ctx->CreateFSub(l, r);
        }
        break;

    case mi::mdl::IExpression_binary::OK_SHIFT_LEFT:
    case mi::mdl::IExpression_binary::OK_SHIFT_LEFT_ASSIGN:
        res = ctx->CreateShl(l, r);
        break;

    case mi::mdl::IExpression_binary::OK_SHIFT_RIGHT:
    case mi::mdl::IExpression_binary::OK_SHIFT_RIGHT_ASSIGN:
        res = ctx->CreateAShr(l, r);
        break;

    case mi::mdl::IExpression_binary::OK_UNSIGNED_SHIFT_RIGHT:
    case mi::mdl::IExpression_binary::OK_UNSIGNED_SHIFT_RIGHT_ASSIGN:
        res = ctx->CreateLShr(l, r);
        break;

    case mi::mdl::IExpression_binary::OK_BITWISE_AND:
    case mi::mdl::IExpression_binary::OK_BITWISE_AND_ASSIGN:
        res = ctx->CreateAnd(l, r);
        break;

    case mi::mdl::IExpression_binary::OK_BITWISE_XOR:
    case mi::mdl::IExpression_binary::OK_BITWISE_XOR_ASSIGN:
        res = ctx->CreateXor(l, r);
        break;

    case mi::mdl::IExpression_binary::OK_BITWISE_OR:
    case mi::mdl::IExpression_binary::OK_BITWISE_OR_ASSIGN:
        res = ctx->CreateOr(l, r);
        break;

    case mi::mdl::IExpression_binary::OK_MULTIPLY:
    case mi::mdl::IExpression_binary::OK_MULTIPLY_ASSIGN:
    case mi::mdl::IExpression_binary::OK_LOGICAL_AND:
    case mi::mdl::IExpression_binary::OK_LOGICAL_OR:
        // these should not occur here
        // fall through
        MDL_ASSERT(!"Unexpected operations in translate_binary_basic()");

    default:
        break;
    }
    return res;
}

// Translate a multiplication expression to LLVM IR.
llvm::Value *LLVM_code_generator::translate_multiply(
    Function_context           &ctx,
    llvm::Type                 *res_llvm_type,
    mi::mdl::IType const       *l_type,
    llvm::Value                *l,
    mi::mdl::IType const       *r_type,
    llvm::Value                *r)
{
    llvm::Value *res = NULL;

    l_type = m_type_mapper.skip_deriv_type(l_type->skip_type_alias());
    r_type = m_type_mapper.skip_deriv_type(r_type->skip_type_alias());

    if (m_type_mapper.is_deriv_type(res_llvm_type)) {
        // compute WITH derivatives
        if (mi::mdl::IType_matrix const *L_type = as<mi::mdl::IType_matrix>(l_type)) {
            if (mi::mdl::IType_matrix const *R_type = as<mi::mdl::IType_matrix>(r_type)) {
                // matrix * matrix
                res = do_matrix_multiplication_MxM_deriv(
                    ctx,
                    res_llvm_type,
                    l,
                    r,
                    L_type->get_element_type()->get_size(),
                    L_type->get_columns(),
                    R_type->get_columns());
            } else if (is<mi::mdl::IType_vector>(r_type)) {
                // matrix * vector
                int rows = L_type->get_element_type()->get_size();
                int cols = L_type->get_columns();

                res = do_matrix_multiplication_MxV_deriv(
                    ctx, res_llvm_type, l, r, rows, cols);
            } else {
                // matrix * scalar element-wise multiplication
                res = do_matrix_multiplication_MxS_deriv(
                    ctx, res_llvm_type, l, r);
            }
        } else if (mi::mdl::IType_matrix const *R_type = as<mi::mdl::IType_matrix>(r_type)) {
            if (is<mi::mdl::IType_vector>(l_type)) {
                // vector * matrix
                int rows = R_type->get_element_type()->get_size();
                int cols = R_type->get_columns();

                res = do_matrix_multiplication_VxM_deriv(
                    ctx, res_llvm_type, l, r, rows, cols);
            } else {
                // matrix * scalar element-wise multiplication
                res = do_matrix_multiplication_MxS_deriv(
                    ctx, res_llvm_type, r, l);
            }
        } else {
            llvm::Type *elem_type = m_type_mapper.get_deriv_base_type(res_llvm_type);
            res = ctx.create_deriv_mul(elem_type, l, r);
        }
    } else {
        // do not compute derivatives
        if (is<mi::mdl::IType_color>(l_type)) {
            // color * color or color * scalar element-wise multiplication
            res = ctx.create_mul(res_llvm_type, l, r);
        } else if (is<mi::mdl::IType_vector>(l_type)) {
            if (mi::mdl::IType_matrix const *R_type = as<mi::mdl::IType_matrix>(r_type)) {
                // vector * matrix
                res = do_matrix_multiplication_VxM(
                    ctx,
                    res_llvm_type,
                    l,
                    r,
                    R_type->get_element_type()->get_size(),
                    R_type->get_columns());
            } else {
                // vector * vector or vector * scalar element-wise multiplication
                res = ctx.create_mul(res_llvm_type, l, r);
            }
        } else if (mi::mdl::IType_matrix const *L_type = as<mi::mdl::IType_matrix>(l_type)) {
            if (mi::mdl::IType_matrix const *R_type = as<mi::mdl::IType_matrix>(r_type)) {
                // matrix * matrix
                res = do_matrix_multiplication_MxM(
                    ctx,
                    res_llvm_type,
                    l,
                    r,
                    L_type->get_element_type()->get_size(),
                    L_type->get_columns(),
                    R_type->get_columns());
            } else if (is<mi::mdl::IType_vector>(r_type)) {
                // matrix * vector
                res = do_matrix_multiplication_MxV(
                    ctx,
                    res_llvm_type,
                    l,
                    r,
                    L_type->get_element_type()->get_size(),
                    L_type->get_columns());
            } else {
                // matrix * scalar element-wise multiplication
                res = do_matrix_multiplication_MxS(ctx, res_llvm_type, l, r);
            }
        } else {
            if (is<mi::mdl::IType_matrix>(r_type)) {
                // scalar * matrix element-wise multiplication
                res = do_matrix_multiplication_MxS(ctx, res_llvm_type, r, l);
            } else {
                // scalar * vector or scalar * color or scalar * scalar element-wise multiplication
                res = ctx.create_mul(res_llvm_type, l, r);
            }
        }
    }

    if (res == NULL) {
        MDL_ASSERT(!"NYI");
        res = llvm::UndefValue::get(res_llvm_type);
    }
    return res;
}

// Translate a multiplication expression to LLVM IR.
llvm::Value *LLVM_code_generator::translate_multiply(
    Function_context           &ctx,
    llvm::Type                 *res_llvm_type,
    mi::mdl::IExpression const *lhs,
    mi::mdl::IExpression const *rhs,
    bool                       return_derivs)
{
    mi::mdl::IType const *l_type = lhs->get_type();
    llvm::Value          *l      = translate_expression_value(ctx, lhs, return_derivs);
    mi::mdl::IType const *r_type = rhs->get_type();
    llvm::Value          *r      = translate_expression_value(ctx, rhs, return_derivs);

    return translate_multiply(ctx, res_llvm_type, l_type, l, r_type, r);
}

// Create a matrix by scalar multiplication.
llvm::Value *LLVM_code_generator::do_matrix_multiplication_MxS(
    Function_context &ctx,
    llvm::Type       *res_llvm_type,
    llvm::Value      *l,
    llvm::Value      *r)
{
    llvm::Type *m_tp = res_llvm_type;

    if (llvm::VectorType *v_tp = llvm::dyn_cast<llvm::VectorType>(m_tp)) {
        // matrices represented as vectors
        return ctx.create_mul(v_tp, l, ctx.create_vector_splat(v_tp, r));
    }

    llvm::ArrayType *a_tp = llvm::cast<llvm::ArrayType>(m_tp);
    llvm::Type      *e_tp = a_tp->getElementType();

    llvm::Value *res = llvm::ConstantAggregateZero::get(a_tp);
    if (e_tp->isVectorTy()) {
        // matrices represented as arrays of (row-)vectors
        r = ctx.create_vector_splat(llvm::cast<llvm::VectorType>(e_tp), r);
    } else {
        // matrices represented as arrays of scalars, do nothing
    }

    unsigned idxes[1];
    for (unsigned i = 0, n = unsigned(a_tp->getNumElements()); i < n; ++i) {
        idxes[0] = i;
        llvm::Value *elem = ctx->CreateExtractValue(l, idxes);

        elem = ctx.create_mul(e_tp, elem, r);

        res = ctx->CreateInsertValue(res, elem, idxes);
    }

    return res;
}

// Create a matrix by scalar multiplication with derivation.
llvm::Value *LLVM_code_generator::do_matrix_multiplication_MxS_deriv(
    Function_context &ctx,
    llvm::Type *res_llvm_type,
    llvm::Value *l,
    llvm::Value *r)
{
    // (A + Bx + Cy)(d + ex + fy) = Ad + (Ae + Bd)x + (Af + Cd)y

    llvm::Type *elem_type = m_type_mapper.get_deriv_base_type(res_llvm_type);

    llvm::Value *val = do_matrix_multiplication_MxS(
        ctx, elem_type, ctx.get_dual_val(l), ctx.get_dual_val(r));

    llvm::Value *dx_1 = do_matrix_multiplication_MxS(
        ctx, elem_type, ctx.get_dual_val(l), ctx.get_dual_dx(r));
    llvm::Value *dx_2 = do_matrix_multiplication_MxS(
        ctx, elem_type, ctx.get_dual_dx(l), ctx.get_dual_val(r));
    llvm::Value *dx = do_matrix_addition(ctx, elem_type, dx_1, dx_2);

    llvm::Value *dy_1 = do_matrix_multiplication_MxS(
        ctx, elem_type, ctx.get_dual_val(l), ctx.get_dual_dy(r));
    llvm::Value *dy_2 = do_matrix_multiplication_MxS(
        ctx, elem_type, ctx.get_dual_dy(l), ctx.get_dual_val(r));
    llvm::Value *dy = do_matrix_addition(ctx, elem_type, dy_1, dy_2);

    return ctx.get_dual(val, dx, dy);
}

// Create a matrix by matrix addition.
llvm::Value *LLVM_code_generator::do_matrix_addition(
    Function_context &ctx,
    llvm::Type *res_llvm_type,
    llvm::Value *l,
    llvm::Value *r)
{
    llvm::Type *m_tp = res_llvm_type;

    if (llvm::VectorType *v_tp = llvm::dyn_cast<llvm::VectorType>(m_tp)) {
        // matrices represented as vectors
        return ctx.create_add(v_tp, l, r);
    }

    llvm::ArrayType *a_tp = llvm::cast<llvm::ArrayType>(m_tp);
    llvm::Type      *e_tp = a_tp->getElementType();

    llvm::Value *res = llvm::ConstantAggregateZero::get(a_tp);
    // matrices represented as arrays of (row-)vectors or of scalars

    unsigned idxes[1];
    for (unsigned i = 0, n = unsigned(a_tp->getNumElements()); i < n; ++i) {
        idxes[0] = i;
        llvm::Value *elem_l = ctx->CreateExtractValue(l, idxes);
        llvm::Value *elem_r = ctx->CreateExtractValue(r, idxes);

        llvm::Value *elem = ctx.create_add(e_tp, elem_l, elem_r);

        res = ctx->CreateInsertValue(res, elem, idxes);
    }

    return res;
}

// Translate an assign expression to LLVM IR.
Expression_result LLVM_code_generator::translate_assign(
    Function_context                  &ctx,
    mi::mdl::IExpression_binary const *call,
    bool                              return_derivs)
{
    mi::mdl::IExpression const *lhs = call->get_left_argument();
    mi::mdl::IExpression const *rhs = call->get_right_argument();

    IDefinition const *lhs_base_def = Analysis::get_lvalue_base(lhs);
    bool lhs_is_deriv = is_deriv_var(lhs_base_def);

    llvm::Value *res = NULL;

    mi::mdl::IExpression_binary::Operator op = call->get_operator();
    switch (op) {
    case mi::mdl::IExpression_binary::OK_ASSIGN:
        res = translate_expression_value(ctx, rhs, lhs_is_deriv);
        break;

    case mi::mdl::IExpression_binary::OK_MULTIPLY_ASSIGN:
    case mi::mdl::IExpression_binary::OK_DIVIDE_ASSIGN:
    case mi::mdl::IExpression_binary::OK_MODULO_ASSIGN:
    case mi::mdl::IExpression_binary::OK_PLUS_ASSIGN:
    case mi::mdl::IExpression_binary::OK_MINUS_ASSIGN:
    case mi::mdl::IExpression_binary::OK_SHIFT_LEFT_ASSIGN:
    case mi::mdl::IExpression_binary::OK_SHIFT_RIGHT_ASSIGN:
    case mi::mdl::IExpression_binary::OK_UNSIGNED_SHIFT_RIGHT_ASSIGN:
    case mi::mdl::IExpression_binary::OK_BITWISE_AND_ASSIGN:
    case mi::mdl::IExpression_binary::OK_BITWISE_XOR_ASSIGN:
    case mi::mdl::IExpression_binary::OK_BITWISE_OR_ASSIGN:
        res = translate_binary_no_side_effect(ctx, call, lhs_is_deriv);
        break;

    default:
        {
            llvm_unreachable("unsupported assign");
            Expression_result expr_res = Expression_result::undef(lookup_type(call->get_type()));
            expr_res.ensure_deriv_result(ctx, return_derivs);
            return expr_res;
        }
    }

    if (lhs_is_deriv) {
        llvm::Value *adr_val, *adr_dx, *adr_dy;
        translate_lval_expression_dual(ctx, lhs, adr_val, adr_dx, adr_dy);

        // check if the result must be converted into a vector first
        llvm::Type *res_tp = llvm::cast<llvm::PointerType>(adr_val->getType())->getElementType();
        if (res_tp != ctx.get_dual_val(res)->getType()) {
            llvm::Value *res_val = ctx.create_splat(res_tp, ctx.get_dual_val(res));
            llvm::Value *res_dx  = ctx.create_splat(res_tp, ctx.get_dual_dx(res));
            llvm::Value *res_dy  = ctx.create_splat(res_tp, ctx.get_dual_dy(res));
            res = ctx.get_dual(res_val, res_dx, res_dy);
        }

        // store the result
        ctx->CreateStore(ctx.get_dual_val(res), adr_val);
        ctx->CreateStore(ctx.get_dual_dx(res),  adr_dx);
        ctx->CreateStore(ctx.get_dual_dy(res),  adr_dy);
    } else {
        llvm::Value *adr = translate_lval_expression(ctx, lhs);

        // check if the result must be converted into a vector first
        llvm::Type *res_tp = llvm::cast<llvm::PointerType>(adr->getType())->getElementType();
        if (res_tp != res->getType())
            res = ctx.create_splat(res_tp, res);

        // store the result
        ctx->CreateStore(res, adr);
    }

    Expression_result expr_res = Expression_result::value(res);
    expr_res.ensure_deriv_result(ctx, return_derivs);
    return expr_res;
}

// Translate a binary compare expression to LLVM IR.
Expression_result LLVM_code_generator::translate_compare(
    Function_context                      &ctx,
    mi::mdl::IExpression_binary::Operator op,
    mi::mdl::IType const                  *l_type,
    llvm::Value                           *lv,
    mi::mdl::IType const                  *r_type,
    llvm::Value                           *rv)
{
    bool conv_l = false;
    bool conv_r = false;

    llvm::Type *op_type;

    l_type = l_type->skip_type_alias();
    r_type = r_type->skip_type_alias();

    l_type = m_type_mapper.skip_deriv_type(l_type);
    r_type = m_type_mapper.skip_deriv_type(r_type);

    // strip any derivatives if present
    lv = ctx.get_dual_val(lv);
    rv = ctx.get_dual_val(rv);

    if (l_type != r_type) {
        // one of them must be scalar

        if (!is<IType_atomic>(l_type)) {
            op_type = lv->getType();
            conv_r  = true;
        } else {
            MDL_ASSERT(!is<IType_atomic>(r_type));
            op_type = rv->getType();
            conv_l  = true;
        }
    } else {
        op_type = lv->getType();
        MDL_ASSERT(op_type == rv->getType());
    }

    llvm::Value *res;

    if (llvm::ArrayType *a_tp = llvm::dyn_cast<llvm::ArrayType>(op_type)) {
        // matrices or vectors represented by arrays
        llvm::Type *e_tp = a_tp->getElementType();

        res = ctx.get_constant(op == mi::mdl::IExpression_binary::OK_EQUAL);
        res = ctx->CreateTrunc(res, m_type_mapper.get_predicate_type());

        unsigned idxes[1];
        for (unsigned i = 0, n = unsigned(a_tp->getNumElements()); i < n; ++i) {
            idxes[0] = i;

            llvm::Value *l_elem = conv_l ? lv : ctx->CreateExtractValue(lv, idxes);
            llvm::Value *r_elem = conv_r ? rv : ctx->CreateExtractValue(rv, idxes);

            // only == and != are supported
            llvm::Value *e_res;

            switch (op) {
            case mi::mdl::IExpression_binary::OK_EQUAL:
                {
                    if (e_tp->isIntOrIntVectorTy()) {
                        e_res = ctx->CreateICmpEQ(l_elem, r_elem);
                    } else {
                        e_res = ctx->CreateFCmpOEQ(l_elem, r_elem);
                    }

                    if (e_tp->isVectorTy()) {
                        // ... of vectors
                        llvm::VectorType *v_tp = llvm::cast<llvm::VectorType>(e_tp);

                        // all must be equal
                        for (unsigned i = 0, n = unsigned(v_tp->getNumElements()); i < n; ++i) {
                            llvm::Value *idx = ctx.get_constant(int(i));

                            res = ctx->CreateAnd(res, ctx->CreateExtractElement(e_res, idx));
                        }
                    } else {
                        // ... of scalars
                        res = ctx->CreateAnd(res, e_res);
                    }
                }
                break;
            case mi::mdl::IExpression_binary::OK_NOT_EQUAL:
                {
                    if (e_tp->isIntOrIntVectorTy()) {
                        e_res = ctx->CreateICmpNE(l_elem, r_elem);
                    } else {
                        e_res = ctx->CreateFCmpUNE(l_elem, r_elem);
                    }

                    if (e_tp->isVectorTy()) {
                        // ... of vectors
                        llvm::VectorType *v_tp = llvm::cast<llvm::VectorType>(e_tp);

                        // only one must be not equal
                        for (unsigned i = 0, n = unsigned(v_tp->getNumElements()); i < n; ++i) {
                            llvm::Value *idx = ctx.get_constant(int(i));

                            res = ctx->CreateOr(res, ctx->CreateExtractElement(e_res, idx));
                        }
                    } else {
                        // ... of scalars
                        res = ctx->CreateOr(res, e_res);
                    }
                }
                break;
            default:
                MDL_ASSERT(!"comparasion not supported on matrices");
                res = llvm::UndefValue::get(m_type_mapper.get_predicate_type());
            }
        }
        // map the i1 result to the bool type representation
        res = ctx->CreateZExt(res, m_type_mapper.get_bool_type());
        return Expression_result::value(res);
    }

    if (llvm::VectorType *vt = llvm::dyn_cast<llvm::VectorType>(op_type)) {
        // convert the scalar one to vector
        if (conv_l)
            lv = ctx.create_vector_splat(vt, lv);
        if (conv_r)
            rv = ctx.create_vector_splat(vt, rv);
    }

    if (op_type->isFPOrFPVectorTy()) {
        switch (op) {
        case mi::mdl::IExpression_binary::OK_LESS:
            res = ctx->CreateFCmp(llvm::FCmpInst::FCMP_OLT, lv, rv);
            break;
        case mi::mdl::IExpression_binary::OK_LESS_OR_EQUAL:
            res = ctx->CreateFCmp(llvm::ICmpInst::FCMP_OLE, lv, rv);
            break;
        case mi::mdl::IExpression_binary::OK_GREATER_OR_EQUAL:
            res = ctx->CreateFCmp(llvm::ICmpInst::FCMP_OGE, lv, rv);
            break;
        case mi::mdl::IExpression_binary::OK_GREATER:
            res = ctx->CreateFCmp(llvm::ICmpInst::FCMP_OGT, lv, rv);
            break;
        case mi::mdl::IExpression_binary::OK_EQUAL:
            res = ctx->CreateFCmp(llvm::ICmpInst::FCMP_OEQ, lv, rv);
            break;
        case mi::mdl::IExpression_binary::OK_NOT_EQUAL:
            res = ctx->CreateFCmp(llvm::ICmpInst::FCMP_UNE, lv, rv);
            break;
        default:
            MDL_ASSERT(!"Unsupported compare operator");
            return Expression_result::undef(m_type_mapper.get_bool_type());
        }
    } else {
        switch (op) {
        case mi::mdl::IExpression_binary::OK_LESS:
            res = ctx->CreateICmp(llvm::ICmpInst::ICMP_SLT, lv, rv);
            break;
        case mi::mdl::IExpression_binary::OK_LESS_OR_EQUAL:
            res = ctx->CreateICmp(llvm::ICmpInst::ICMP_SLE, lv, rv);
            break;
        case mi::mdl::IExpression_binary::OK_GREATER_OR_EQUAL:
            res = ctx->CreateICmp(llvm::ICmpInst::ICMP_SGE, lv, rv);
            break;
        case mi::mdl::IExpression_binary::OK_GREATER:
            res = ctx->CreateICmp(llvm::ICmpInst::ICMP_SGT, lv, rv);
            break;
        case mi::mdl::IExpression_binary::OK_EQUAL:
            res = ctx->CreateICmp(llvm::ICmpInst::ICMP_EQ, lv, rv);
            break;
        case mi::mdl::IExpression_binary::OK_NOT_EQUAL:
            res = ctx->CreateICmp(llvm::ICmpInst::ICMP_NE, lv, rv);
            break;
        default:
            MDL_ASSERT(!"Unsupported compare operator");
            return Expression_result::undef(m_type_mapper.get_bool_type());
        }
    }

    if (llvm::VectorType *vt = llvm::dyn_cast<llvm::VectorType>(op_type)) {
        // the result is a vector of bool, but in MDL we support only a single bool result,
        // so condense them here
        if (op == mi::mdl::IExpression_binary::OK_EQUAL) {
            llvm::Value *idx       = ctx.get_constant(int(0));
            llvm::Value *condensed = ctx->CreateExtractElement(res, idx);

            // all must be equal
            for (unsigned i = 1, n = unsigned(vt->getNumElements()); i < n; ++i) {
                idx       = ctx.get_constant(int(i));
                condensed = ctx->CreateAnd(condensed, ctx->CreateExtractElement(res, idx));
            }
            res = condensed;
        } else {
            MDL_ASSERT(op == mi::mdl::IExpression_binary::OK_NOT_EQUAL);

            llvm::Value *idx       = ctx.get_constant(int(0));
            llvm::Value *condensed = ctx->CreateExtractElement(res, idx);

            // only one must be not equal
            for (unsigned i = 1, n = unsigned(vt->getNumElements()); i < n; ++i) {
                idx       = ctx.get_constant(int(i));
                condensed = ctx->CreateOr(condensed, ctx->CreateExtractElement(res, idx));
            }
            res = condensed;
        }
    }
    // map the i1 result to the bool type representation
    res = ctx->CreateZExt(res, m_type_mapper.get_bool_type());
    return Expression_result::value(res);
}

// Translate a conditional expression to LLVM IR.
Expression_result LLVM_code_generator::translate_conditional(
    Function_context                       &ctx,
    mi::mdl::IExpression_conditional const *cond_expr,
    bool                                   return_derivs)
{
    // C-like ternary operator with lazy evaluation
    llvm::BasicBlock *on_true_bb  = ctx.create_bb("?:_true");
    llvm::BasicBlock *on_false_bb = ctx.create_bb("?:_false");
    llvm::BasicBlock *end_bb      = ctx.create_bb("?:_end");

    llvm::Type  *res_type  = lookup_type_or_deriv_type(ctx, cond_expr);
    llvm::Value *res_addr  = ctx.create_local(res_type, "?:_tmp");
    translate_boolean_branch(ctx, cond_expr->get_condition(), on_true_bb, on_false_bb);

    ctx->SetInsertPoint(on_true_bb);
    llvm::Value *true_res = translate_expression_value(
        ctx, cond_expr->get_true(), return_derivs);
    ctx->CreateStore(true_res, res_addr);
    ctx->CreateBr(end_bb);

    ctx->SetInsertPoint(on_false_bb);
    llvm::Value *false_res = translate_expression_value(
        ctx, cond_expr->get_false(), return_derivs);
    ctx->CreateStore(false_res, res_addr);
    ctx->CreateBr(end_bb);

    ctx->SetInsertPoint(end_bb);
    return Expression_result::ptr(res_addr);
}

// Translate a DAG intrinsic call expression to LLVM IR.
Expression_result LLVM_code_generator::translate_dag_intrinsic(
    Function_context &ctx,
    ICall_expr const *call_expr)
{
    mi::mdl::IDefinition::Semantics sema = call_expr->get_semantics();

    bool derivs = call_expr->returns_derivatives(*this);

    switch (sema) {
    case mi::mdl::IDefinition::DS_INTRINSIC_DAG_FIELD_ACCESS:
        {
            llvm::Value *compound = call_expr->translate_argument_value(*this, ctx, 0, derivs);
            int         index     = call_expr->get_field_index(*this, get_allocator());

            if (index >= 0) {
                return Expression_result::value(ctx.create_extract_allow_deriv(compound, index));
            }
        }
        break;
    case mi::mdl::IDefinition::DS_INTRINSIC_DAG_ARRAY_LENGTH:
    default:
        // Should not happen
        break;
    case mi::mdl::IDefinition::DS_INTRINSIC_DAG_SET_OBJECT_ID:
        {
            mi::mdl::IValue_int const *v =
                mi::mdl::cast<mi::mdl::IValue_int>(call_expr->get_const_argument(0));
            set_object_id(v->get_value());
            return call_expr->translate_argument(*this, ctx, 1, derivs);
        }
        break;
    case mi::mdl::IDefinition::DS_INTRINSIC_DAG_SET_TRANSFORMS:
        {
            mi::mdl::IValue_matrix const *m_w2o =
                mi::mdl::cast<mi::mdl::IValue_matrix>(call_expr->get_const_argument(0));
            mi::mdl::IValue_matrix const *m_o2w =
                mi::mdl::cast<mi::mdl::IValue_matrix>(call_expr->get_const_argument(1));
            set_transforms(m_w2o, m_o2w);
            return call_expr->translate_argument(*this, ctx, 2, derivs);
        }
        break;
    case mi::mdl::IDefinition::DS_INTRINSIC_DAG_GET_DERIV_VALUE:
        {
            Expression_result res = call_expr->translate_argument(
                *this, ctx, 0, /*return_derivs=*/ false);  // for DAG return_derivs is ignored
            res.ensure_deriv_result(ctx, false);  // always ensure they are removed
            return res;
        }
    case mi::mdl::IDefinition::DS_INTRINSIC_DAG_MAKE_DERIV:
        {
            Expression_result res = call_expr->translate_argument(
                *this, ctx, 0, /*return_derivs=*/ true);  // for DAG return_derivs is ignored
            res.ensure_deriv_result(ctx, true);  // always ensure we get a derivative
            return res;
        }
    }
    MDL_ASSERT(!"Unexpected DAG intrinsic");
    return Expression_result::undef(lookup_type(call_expr->get_type()));
}

// Create the float4x4 identity matrix.
llvm::Value *LLVM_code_generator::create_identity_matrix(
    Function_context &ctx)
{
    llvm::Constant *elems[16];

    // flat matrix values
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            llvm::Value *v = ctx.get_constant(i == j ? 1.0f : 0.0f);

            elems[i * 4 + j] = llvm::cast<llvm::Constant>(v);
        }
    }

    llvm::Type *m_type = m_type_mapper.get_float4x4_type();
    if (m_type->isVectorTy()) {
        // encode into a big vector
        return llvm::ConstantVector::get(llvm::ArrayRef<llvm::Constant *>(elems, 16));
    } else {
        llvm::ArrayType *a_tp = llvm::cast<llvm::ArrayType>(m_type);
        llvm::Type      *e_tp = a_tp->getArrayElementType();

        if (e_tp->isVectorTy()) {
            // encode a matrix into an array of vectors (small vector mode)
            llvm::Constant *vectors[4];

            for (int col = 0; col < 4; ++col) {
                vectors[col] = llvm::ConstantVector::get(
                    llvm::ArrayRef<llvm::Constant *>(&elems[col * 4], 4));
            }
            return llvm::ConstantArray::get(
                a_tp, llvm::ArrayRef<llvm::Constant *>(vectors, 4));

        } else {
            // encode a matrix/vector into an array of scalars (scalar mode)
            return llvm::ConstantArray::get(
                a_tp, llvm::ArrayRef<llvm::Constant *>(elems, 16));
        }
    }
}

// Translate a call expression to LLVM IR.
Expression_result LLVM_code_generator::translate_call(
    Function_context &ctx,
    ICall_expr const *call_expr)
{
    // Note: normally one would expect that ONLY user defined function can be transformed.
    // Unfortunately we need at least select operators, so we move the transformation
    // step to the general call... Beware, this allows various bad things, but in the end
    // hooking op the compiler is dangerous in any sense.
    if (m_transformer != NULL) {
        // we have a transformer, transform the call first if necessary
        if (DAG_call const *call = call_expr->as_dag_call()) {
            MDL_ASSERT(m_trans_builder != NULL);
            DAG_call const *n_call = m_transformer->transform(call, m_trans_builder);
            if (n_call != NULL && n_call != call) {
                // ugly, but creating a new one is also ugly, because of the limited
                // block scope
                Call_dag_expr const *expr = impl_cast<Call_dag_expr>(call_expr);
                const_cast<Call_dag_expr *>(expr)->replace(n_call);
            }
        }
    }

    // check first if it is something we know.
    if (call_expr->is_array_constructor()) {
        // handle array constructors
        return translate_array_constructor_call(ctx, call_expr);
    }

    mi::mdl::IDefinition::Semantics sema = call_expr->get_semantics();

    bool return_derivs = call_expr->returns_derivatives(*this);

    if (semantic_is_operator(sema)) {
        mi::mdl::IExpression::Operator op = semantic_to_operator(sema);

        if (is_unary_operator(op)) {
            mi::mdl::IExpression_unary::Operator uop = mi::mdl::IExpression_unary::Operator(op);
            Expression_result arg = call_expr->translate_argument(*this, ctx, 0, return_derivs);
            if (uop == mi::mdl::IExpression_unary::OK_CAST) {
                return translate_cast_expression(ctx, arg, call_expr->get_type());
            } else {
                return Expression_result::value(translate_unary(ctx, uop, arg.as_value(ctx)));
            }
        } else if (is_binary_operator(op)) {
            mi::mdl::IExpression_binary::Operator bop = mi::mdl::IExpression_binary::Operator(op);

            llvm::Value *l = call_expr->translate_argument_value(*this, ctx, 0, return_derivs);
            llvm::Value *r = call_expr->translate_argument_value(*this, ctx, 1, return_derivs);

            llvm::Value *res = NULL;
            switch (bop) {
            case mi::mdl::IExpression_binary::OK_MULTIPLY:
                {
                    mi::mdl::IType const *l_type   = call_expr->get_argument_type(0);
                    mi::mdl::IType const *r_type   = call_expr->get_argument_type(1);

                    res = translate_multiply(
                        ctx, lookup_type_or_deriv_type(ctx, call_expr), l_type, l, r_type, r);
                }
                break;
            case mi::mdl::IExpression_binary::OK_LOGICAL_AND:
            case mi::mdl::IExpression_binary::OK_LOGICAL_OR:
                {
                    mi::mdl::IType const *l_type =
                        call_expr->get_argument_type(0)->skip_type_alias();
                    mi::mdl::IType const *r_type =
                        call_expr->get_argument_type(1)->skip_type_alias();

                    if (is<mi::mdl::IType_bool>(l_type) && is<mi::mdl::IType_bool>(r_type)) {
                        // shortcut evaluation here
                        llvm::BasicBlock *true_bb  = ctx.create_bb("sc_true");
                        llvm::BasicBlock *false_bb = ctx.create_bb("sc_false");
                        llvm::BasicBlock *end_bb   = ctx.create_bb("sc_end");
                        llvm::Value      *tmp      =
                            ctx.create_local(m_type_mapper.get_bool_type(), "sc_tmp");

                        call_expr->translate_boolean_branch(*this, ctx, true_bb, false_bb);

                        ctx->SetInsertPoint(true_bb);
                        ctx->CreateStore(ctx.get_constant(true), tmp);
                        ctx->CreateBr(end_bb);

                        ctx->SetInsertPoint(false_bb);
                        ctx->CreateStore(ctx.get_constant(false), tmp);
                        ctx->CreateBr(end_bb);

                        ctx->SetInsertPoint(end_bb);
                        res = ctx->CreateLoad(tmp);
                    } else {
                        // no shortcut execution possible on vectors, execute bitwise
                        if (bop == mi::mdl::IExpression_binary::OK_LOGICAL_AND)
                            bop = mi::mdl::IExpression_binary::OK_BITWISE_AND;
                        else
                            bop = mi::mdl::IExpression_binary::OK_BITWISE_OR;
                        res = translate_binary_basic(ctx, bop, l, r, /*expr_pos=*/NULL);
                    }
                }
                break;

            default:
                res = translate_binary_no_side_effect(ctx, call_expr);
                break;
            }
            return Expression_result::value(res);
        } else if (op == IExpression::OK_TERNARY) {
            // C-like ternary operator with lazy evaluation
            llvm::BasicBlock *on_true_bb  = ctx.create_bb("?:_true");
            llvm::BasicBlock *on_false_bb = ctx.create_bb("?:_false");
            llvm::BasicBlock *end_bb      = ctx.create_bb("?:_end");

            llvm::Type  *res_type  = lookup_type(call_expr->get_type());
            llvm::Value *res_addr  = ctx.create_local(res_type, "?:_tmp");

            call_expr->translate_boolean_branch(*this, ctx, on_true_bb, on_false_bb);

            // This is ugly: As long as we translate the "material body", there exists
            // only one basic block. However, for the ?: operator we need control flow, so we must
            // handle the CSE right. Currently we restrict it to the current block.
            // This is a correct, but not the smartest solution, as this might blow up the
            // generated code.

            {
                BB_store true_chain(m_curr_bb, get_next_bb());
                ctx->SetInsertPoint(on_true_bb);
                llvm::Value *true_res =
                    call_expr->translate_argument_value(*this, ctx, 1, /*return_derivs=*/ false);
                ctx->CreateStore(true_res, res_addr);
                ctx->CreateBr(end_bb);
            }

            {
                BB_store false_chain(m_curr_bb, get_next_bb());
                ctx->SetInsertPoint(on_false_bb);
                llvm::Value *false_res =
                    call_expr->translate_argument_value(*this, ctx, 2, /*return_derivs=*/ false);
                ctx->CreateStore(false_res, res_addr);
                ctx->CreateBr(end_bb);
            }

            ctx->SetInsertPoint(end_bb);
            return Expression_result::ptr(res_addr);
        }
    }

    switch (sema) {
    case mi::mdl::IDefinition::DS_UNKNOWN:
        return Expression_result::value(translate_call_user_defined_function(ctx, call_expr));

    case mi::mdl::IDefinition::DS_COPY_CONSTRUCTOR:
        // ignore all copy constructors
        MDL_ASSERT(call_expr->get_argument_count() == 1);
        return call_expr->translate_argument(*this, ctx, 0, return_derivs);

    case mi::mdl::IDefinition::DS_CONV_CONSTRUCTOR:
    case mi::mdl::IDefinition::DS_CONV_OPERATOR:
        // handle conversion constructor and operator equal: both have one argument and convert it
        return translate_conversion(ctx, call_expr);

    case mi::mdl::IDefinition::DS_ELEM_CONSTRUCTOR:
        return translate_elemental_constructor(ctx, call_expr);

    case mi::mdl::IDefinition::DS_COLOR_SPECTRUM_CONSTRUCTOR:
        // translate to rgb color, fall into the default case
        break;

    case mi::mdl::IDefinition::DS_INTRINSIC_MATH_EMISSION_COLOR:
        if (call_expr->get_argument_count() == 1) {
            // a no-op in the RGB case
            Expression_result res = call_expr->translate_argument(
                *this, ctx, 0, return_derivs);  // for DAG return_derivs is ignored
            res.ensure_deriv_result(ctx, return_derivs);
            return res;
        }
        // else fall into the default case
        break;

    case mi::mdl::IDefinition::DS_MATRIX_ELEM_CONSTRUCTOR:
        return translate_matrix_elemental_constructor(ctx, call_expr);

    case mi::mdl::IDefinition::DS_MATRIX_DIAG_CONSTRUCTOR:
        return translate_matrix_diagonal_constructor(ctx, call_expr);

    case mi::mdl::IDefinition::DS_DEFAULT_STRUCT_CONSTRUCTOR:
        // this constructor is always replaced in the compiler core by an elemental constructor,
        // so it should never appear in the jit backend
        MDL_ASSERT(!"Unexpected default struct constructor");
        return Expression_result::undef(lookup_type(call_expr->get_type()));

    case mi::mdl::IDefinition::DS_INVALID_REF_CONSTRUCTOR:
    case mi::mdl::IDefinition::DS_TEXTURE_CONSTRUCTOR:
        // these should not occur inside functions
        MDL_ASSERT(!"Unexpected resource constructors");
        return Expression_result::undef(lookup_type(call_expr->get_type()));

    case mi::mdl::IDefinition::DS_INTRINSIC_STATE_TRANSFORM:
        if (call_expr->get_argument_count() == 2) {
            IValue const *arg_0 = call_expr->get_const_argument(0);
            IValue const *arg_1 = call_expr->get_const_argument(1);
            if (arg_0 != NULL && arg_1 != NULL &&
                equal_coordinate_space(arg_0, arg_1, m_internal_space))
            {
                // does not really use the transform state here, so we do not flag it
                return Expression_result::value(create_identity_matrix(ctx));
            }
        }
        break;
    case mi::mdl::IDefinition::DS_INTRINSIC_STATE_TRANSFORM_POINT:
    case mi::mdl::IDefinition::DS_INTRINSIC_STATE_TRANSFORM_VECTOR:
    case mi::mdl::IDefinition::DS_INTRINSIC_STATE_TRANSFORM_NORMAL:
    case mi::mdl::IDefinition::DS_INTRINSIC_STATE_TRANSFORM_SCALE:
        if (call_expr->get_argument_count() == 3) {
            IValue const *arg_0 = call_expr->get_const_argument(0);
            IValue const *arg_1 = call_expr->get_const_argument(1);
            if (arg_0 != NULL && arg_1 != NULL &&
                equal_coordinate_space(arg_0, arg_1, m_internal_space))
            {
                // just a no-op, return the second argument
                // does not really use the transform state here, so we do not flag it
                return call_expr->translate_argument(*this, ctx, 2, return_derivs);
            }
            if (m_world_to_object != NULL && m_object_to_world != NULL) {
                // we have a world-to_object matrix, implement the functions directly
                return translate_transform_call(sema, ctx, call_expr);
            }
        }
        break;
    case mi::mdl::IDefinition::DS_INTRINSIC_STATE_OBJECT_ID:
        if (call_expr->get_argument_count() == 0) {
            return translate_object_id_call(ctx);
        }
        break;
    case mi::mdl::IDefinition::DS_INTRINSIC_DAG_FIELD_ACCESS:
    case mi::mdl::IDefinition::DS_INTRINSIC_DAG_ARRAY_LENGTH:
    case mi::mdl::IDefinition::DS_INTRINSIC_DAG_SET_OBJECT_ID:
    case mi::mdl::IDefinition::DS_INTRINSIC_DAG_SET_TRANSFORMS:
    case mi::mdl::IDefinition::DS_INTRINSIC_DAG_GET_DERIV_VALUE:
    case mi::mdl::IDefinition::DS_INTRINSIC_DAG_MAKE_DERIV:
        return translate_dag_intrinsic(ctx, call_expr);

    case mi::mdl::IDefinition::DS_INTRINSIC_JIT_LOOKUP:
        return translate_jit_intrinsic(ctx, call_expr);

    case mi::mdl::IDefinition::DS_INTRINSIC_DAG_CALL_LAMBDA:
        return translate_dag_call_lambda(ctx, call_expr);

    case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_ISVALID:
        if (!m_scene_data_all_pos_avail) {
            IValue const *name = call_expr->get_const_argument(0);
            if (name != NULL) {
                IValue_string const *name_str = as<IValue_string>(name);
                // is name known to never be available? -> return false
                if (m_scene_data_names.count(name_str->get_value()) == 0)
                    return Expression_result::value(ctx.get_constant(false));
            }
        }

        if (m_target_lang != TL_HLSL && m_target_lang != TL_PTX &&
                !(m_target_lang == TL_NATIVE && !m_has_res_handler)) {
            // TODO: implement calling renderer runtime. For now just return false
            return Expression_result::value(ctx.get_constant(false));
        }

        // try compiler known intrinsic function
        break;

    case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_INT:
    case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_INT2:
    case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_INT3:
    case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_INT4:
    case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_FLOAT:
    case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_FLOAT2:
    case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_FLOAT3:
    case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_FLOAT4:
    case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_COLOR:
    case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_INT:
    case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_INT2:
    case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_INT3:
    case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_INT4:
    case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_FLOAT:
    case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_FLOAT2:
    case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_FLOAT3:
    case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_FLOAT4:
    case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_COLOR:
        if (!m_scene_data_all_pos_avail) {
            IValue const *name = call_expr->get_const_argument(0);
            if (name != NULL) {
                IValue_string const *name_str = as<IValue_string>(name);
                // is name known to never be available? -> return default value (second argument)
                if (m_scene_data_names.count(name_str->get_value()) == 0)
                    return call_expr->translate_argument(*this, ctx, 1, return_derivs);
            }
        }

        if (m_target_lang != TL_HLSL && m_target_lang != TL_PTX &&
                !(m_target_lang == TL_NATIVE && !m_has_res_handler)) {
            // TODO: implement calling renderer runtime. For now just return second argument
            return call_expr->translate_argument(*this, ctx, 1, return_derivs);
        }

        // try compiler known intrinsic function
        break;

    default:
        // try compiler known intrinsic function
        break;
    }

    llvm::Value *res = translate_call_intrinsic_function(ctx, call_expr);
    if (res != NULL)
        return Expression_result::value(res);

    MDL_ASSERT(!"NYI");
    return Expression_result::undef(lookup_type(call_expr->get_type()));
}

// Get the argument type instances for a given call.
Function_instance::Array_instances LLVM_code_generator::get_call_instance(
    Function_context &ctx,
    ICall_expr const *call)
{
    Function_instance::Array_instances res(get_allocator());

    if (!m_enable_instancing) {
        // instancing disabled
        return res;
    }

    typedef ptr_hash_set<IType_array_size const>::Type Marker_set;
    Marker_set mset(0, Marker_set::hasher(), Marker_set::key_equal(), get_allocator());

    IDefinition const *def = call->get_callee_definition(*this);
    IType_function const *ftype = cast<IType_function>(def->get_type());

    // instantiate the return type
    {
        int immediate_size = ctx.instantiate_type_size(call->get_type());

        if (immediate_size >= 0) {
            // use plain function return type
            IType const       *rtype    = ftype->get_return_type();
            IType_array const *r_a_type = as<IType_array>(rtype);
            MDL_ASSERT(r_a_type != NULL);

            if (!r_a_type->is_immediate_sized()) {
                IType_array_size const *deferred_size = r_a_type->get_deferred_size();

                if (mset.insert(deferred_size).second) {
                    // An immediate sized array is returned as an deferred sized return value, add
                    // an instance.
                    // Note that we insert every type here only once, because it can only be mapped
                    // to one immediate type.
                    res.push_back(Array_instance(deferred_size, immediate_size));
                }
            }
        }
    }

    // instantiate argument types
    for (size_t i = 0, n = call->get_argument_count(); i < n; ++i) {
        IType_array const *a_arg_type = as<IType_array>(call->get_argument_type(i));

        if (a_arg_type == NULL)
            continue;

        int immediate_size = ctx.instantiate_type_size(a_arg_type);

        if (immediate_size >= 0 || a_arg_type->is_immediate_sized()) {
            IType const   *ptype;
            ISymbol const *psym;

            // use plain function parameter type
            ftype->get_parameter(int(i), ptype, psym);

            IType_array const *p_a_type = as<IType_array>(ptype);
            MDL_ASSERT(p_a_type != NULL);

            if (!p_a_type->is_immediate_sized()) {
                IType_array_size const *deferred_size = p_a_type->get_deferred_size();

                if (mset.insert(deferred_size).second) {
                    // An immediate sized array is passed into an deferred sized parameter, add
                    // an instance.
                    // Note that we insert every type here only once, because it can only be mapped
                    // to one immediate type.
                    if (immediate_size < 0)
                        immediate_size = a_arg_type->get_size();

                    res.push_back(Array_instance(deferred_size, immediate_size));
                }
            }
        }
    }
    return res;
}

// Instantiate a type using array type instances.
int LLVM_code_generator::instantiate_call_param_type_size(
    IType const                              *type,
    Function_instance::Array_instances const &arr_inst)
{
    IType_array const *a_type = as<IType_array>(type);
    if (a_type != NULL && !a_type->is_immediate_sized()) {
        IType_array_size const *deferred_size = a_type->get_deferred_size();
        for (size_t i = 0, n = arr_inst.size(); i < n; ++i) {
            Array_instance const &ai = arr_inst[i];

            if (ai.get_deferred_size() == deferred_size)
                return ai.get_immediate_size();
        }
    }
    return -1;
}

// Translate a call to an user defined function to LLVM IR.
llvm::Value *LLVM_code_generator::translate_call_user_defined_function(
    Function_context &ctx,
    ICall_expr const *call_expr)
{
    Function_instance::Array_instances arr_inst(get_call_instance(ctx, call_expr));
    bool return_derivs = call_expr->returns_derivatives(*this);
    LLVM_context_data *p_data = call_expr->get_callee_context(*this, arr_inst, return_derivs);

    mi::mdl::IDefinition const    *def    = call_expr->get_callee_definition(*this);
    mi::mdl::IType_function const *f_type = cast<mi::mdl::IType_function>(def->get_type());

    Func_deriv_info const *func_deriv_info = NULL;
    if (m_deriv_infos != NULL) {
        // skip presets, so always get the original function
        mi::mdl::IModule const *mod = tos_module();
        Module const *owner = impl_cast<Module>(mod);
        if (def->get_property(IDefinition::DP_IS_IMPORTED))
            def = owner->get_original_definition(def, owner);

        mi::base::Handle<IModule const> i_owner(owner, mi::base::DUP_INTERFACE);
        def = skip_presets(def, i_owner);

        Function_instance inst(def, arr_inst, return_derivs);
        func_deriv_info = m_deriv_infos->get_function_derivative_infos(inst);
    }

    // get the callee
    llvm::Function *callee = p_data->get_function();

    m_state_usage_analysis.add_call(ctx.get_function(), callee);

    // prepare arguments
    llvm::SmallVector<llvm::Value *, 8> args;

    llvm::Value *sret_res = NULL;
    if (p_data->is_sret_return()) {
        // create a temporary for the function result and pass it
        sret_res = ctx.create_local(p_data->get_return_type(), "call_result");
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
    if (target_uses_exception_state_parameter() && p_data->has_exc_state_param()) {
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

    for (size_t i = 0, n_args = call_expr->get_argument_count(); i < n_args; ++i) {
        mi::mdl::IType const *arg_type = call_expr->get_argument_type(i);

        // instantiate the argument type
        int arg_imm_size = ctx.instantiate_type_size(arg_type);

        bool arg_is_deriv =
            func_deriv_info != NULL && func_deriv_info->args_want_derivatives.test_bit(i + 1);

        Expression_result expr_res = call_expr->translate_argument(*this, ctx, i, arg_is_deriv);

        IType const   *p_type;
        ISymbol const *dummy;
        f_type->get_parameter(i, p_type, dummy);

        // instantiate the parameter type, we are calling an instance using the argument instances
        // of this call
        int param_imm_size = instantiate_call_param_type_size(p_type, arr_inst);

        mi::mdl::IType_array const *a_p_type = mi::mdl::as<mi::mdl::IType_array>(p_type);
        if (a_p_type != NULL && !a_p_type->is_immediate_sized() && param_imm_size < 0) {
            mi::mdl::IType_array const *a_a_type =
                mi::mdl::cast<mi::mdl::IType_array>(arg_type->skip_type_alias());

            if (a_a_type->is_immediate_sized() || arg_imm_size >= 0) {
                // immediate size argument passed to deferred argument, create an array
                // descriptor
                llvm::Type *arr_dec_type = m_type_mapper.lookup_type(m_llvm_context, a_p_type);
                llvm::Value *desc        = ctx.create_local(arr_dec_type, "arr_desc");

                ctx.set_deferred_base(desc, expr_res.as_ptr(ctx));
                size_t size = arg_imm_size >= 0 ? arg_imm_size : a_a_type->get_size();
                ctx.set_deferred_size(desc, ctx.get_constant(size));

                expr_res = Expression_result::ptr(desc);
            } else {
                // just pass by default
            }
        }

        if (m_type_mapper.is_passed_by_reference(arg_type) ||
            (target_supports_pointers() && m_type_mapper.is_deriv_type(expr_res.get_value_type())))
        {
            // pass by reference
            llvm::Value *ptr = expr_res.as_ptr(ctx);
            args.push_back(ptr);
        } else {
            // pass by value
            args.push_back(expr_res.as_value(ctx));
        }
    }

    // call it
    llvm::Value *res = ctx->CreateCall(callee, args);

    if (sret_res != NULL) {
        // the result was passed on the stack
        res = ctx->CreateLoad(sret_res);
    }
    return res;
}

// Translate a DAG expression lambda call to LLVM IR.
Expression_result LLVM_code_generator::translate_dag_call_lambda(
    Function_context &ctx,
    ICall_expr const *call_expr)
{
    MDL_ASSERT(call_expr->get_semantics() == IDefinition::DS_INTRINSIC_DAG_CALL_LAMBDA);

    DAG_call const *dag_call = call_expr->as_dag_call();
    MDL_ASSERT(dag_call);

    size_t lambda_index = strtoul(dag_call->get_name(), NULL, 10);
    return translate_precalculated_lambda(
        ctx,
        lambda_index,
        m_type_mapper.lookup_type(m_llvm_context, dag_call->get_type()));
}

/// Return the length of a vector3 (fourth component ignored).
static float len_v3(Float4_struct const &v)
{
    return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

/// Get the i'th element of a Float4_struct
static float get_index(Float4_struct const &v, size_t i)
{
    float const *p = &v.x;
    return p[i];
}

// Translate a state::transform_*() call expression to LLVM IR.
Expression_result LLVM_code_generator::translate_transform_call(
    IDefinition::Semantics sema,
    Function_context       &ctx,
    ICall_expr const       *call_expr)
{
    // will potentially use the transforms
    m_state_usage_analysis.add_state_usage(
        ctx.get_function(), IGenerated_code_executable::SU_TRANSFORMS);

    llvm::Type  *ret_tp = lookup_type(call_expr->get_type());
    llvm::Value *from   =
        call_expr->translate_argument_value(*this, ctx, 0, /*return_derivs=*/ false);
    llvm::Value *to     =
        call_expr->translate_argument_value(*this, ctx, 1, /*return_derivs=*/ false);

    int sp_encoding = coordinate_world;
    if (strcmp(m_internal_space, "coordinate_object") == 0)
        sp_encoding = coordinate_object;

    llvm::Value *internal = ctx.get_constant(coordinate_internal);
    llvm::Value *encoding = ctx.get_constant(sp_encoding);

    // map internal space
    llvm::Value *f_cond = ctx->CreateICmpEQ(from, internal);
    from = ctx->CreateSelect(f_cond, encoding, from);
    llvm::Value *t_cond = ctx->CreateICmpEQ(to, internal);
    to = ctx->CreateSelect(t_cond, encoding, to);

    llvm::Value *res = llvm::Constant::getNullValue(ret_tp);

    llvm::Value *result = ctx.create_local(ret_tp, "result");
    ctx->CreateStore(res, result);

    switch (sema) {
    case mi::mdl::IDefinition::DS_INTRINSIC_STATE_TRANSFORM:
        {
            if (ret_tp->isVectorTy()) {
                // BIG vector mode
                llvm::Value *cond           = ctx->CreateICmpEQ(from, to);
                llvm::BasicBlock *id_bb     = ctx.create_bb("id");
                llvm::BasicBlock *non_id_bb = ctx.create_bb("non_id");
                llvm::BasicBlock *end_bb    = ctx.create_bb("end");

                ctx->CreateCondBr(cond, id_bb, non_id_bb);

                {
                    llvm::Value *idx;

                    // return the identity matrix
                    ctx->SetInsertPoint(id_bb);

                    llvm::Value *one = ctx.get_constant(1.0f);
                    idx  = ctx.get_constant(0 * 4 + 0);
                    ctx->CreateInsertElement(res, one, idx);
                    idx  = ctx.get_constant(1 * 4 + 1);
                    ctx->CreateInsertElement(res, one, idx);
                    idx  = ctx.get_constant(2 * 4 + 2);
                    ctx->CreateInsertElement(res, one, idx);
                    idx  = ctx.get_constant(3 * 4 + 3);
                    ctx->CreateInsertElement(res, one, idx);

                    ctx->CreateStore(res, result);
                    ctx->CreateBr(end_bb);
                }
                {
                    ctx->SetInsertPoint(non_id_bb);

                    llvm::Value *worldToObject =
                        ctx.get_constant(LLVM_code_generator::coordinate_world);
                    llvm::Value *cond = ctx->CreateICmpEQ(from, worldToObject);

                    llvm::BasicBlock *w2o_bb = ctx.create_bb("w2o");
                    llvm::BasicBlock *o2w_bb = ctx.create_bb("o2w");

                    ctx->CreateCondBr(cond, w2o_bb, o2w_bb);
                    {
                        // convert the w2o matrix from row major to column major
                        ctx->SetInsertPoint(w2o_bb);

                        res = llvm::Constant::getNullValue(ret_tp);
                        for (int i = 0; i < 4; ++i) {
                            for (int j = 0; j < 4; ++j) {
                                llvm::Value *elem = ctx.get_constant(
                                    get_index(m_world_to_object[i], j));
                                llvm::Value *idx  = ctx.get_constant(i + j * 4);
                                res = ctx->CreateInsertElement(res, elem, idx);
                            }
                        }
                        ctx->CreateStore(res, result);
                        ctx->CreateBr(end_bb);
                    }
                    {
                        // convert the o2w matrix from row major to column major
                        ctx->SetInsertPoint(o2w_bb);

                        res = llvm::Constant::getNullValue(ret_tp);
                        for (int i = 0; i < 4; ++i) {
                            for (int j = 0; j < 4; ++j) {
                                llvm::Value *elem = ctx.get_constant(
                                    get_index(m_object_to_world[i], j));
                                llvm::Value *idx  = ctx.get_constant(i + j * 4);
                                res = ctx->CreateInsertElement(res, elem, idx);
                            }
                        }
                        ctx->CreateStore(res, result);
                        ctx->CreateBr(end_bb);
                    }
                }
                ctx->SetInsertPoint(end_bb);
            } else {
                MDL_ASSERT(!"non-BIG-vector mode NYI");
            }
            res = ctx->CreateLoad(result);
            return Expression_result::value(res);
        }
        break;

    case mi::mdl::IDefinition::DS_INTRINSIC_STATE_TRANSFORM_POINT:
        {
            // TODO: derivatives
            llvm::Value *point  =
                call_expr->translate_argument_value(*this, ctx, 2, /*return_derivs=*/ false);

            llvm::Value *cond           = ctx->CreateICmpEQ(from, to);
            llvm::BasicBlock *id_bb     = ctx.create_bb("id");
            llvm::BasicBlock *non_id_bb = ctx.create_bb("non_id");
            llvm::BasicBlock *end_bb    = ctx.create_bb("end");

            ctx->CreateCondBr(cond, id_bb, non_id_bb);
            {
                // the matrix is the identity, return the point
                ctx->SetInsertPoint(id_bb);

                ctx->CreateStore(point, result);
                ctx->CreateBr(end_bb);
            }
            {
                ctx->SetInsertPoint(non_id_bb);

                if (ret_tp->isVectorTy()) {
                    llvm::VectorType *vec_tp = llvm::cast<llvm::VectorType>(ret_tp);

                    // vector mode
                    llvm::Value *worldToObject =
                        ctx.get_constant(LLVM_code_generator::coordinate_world);
                    llvm::Value *cond = ctx->CreateICmpEQ(from, worldToObject);

                    llvm::BasicBlock *w2o_bb = ctx.create_bb("w2o");
                    llvm::BasicBlock *o2w_bb = ctx.create_bb("o2w");

                    ctx->CreateCondBr(cond, w2o_bb, o2w_bb);
                    {
                        ctx->SetInsertPoint(w2o_bb);

// point.x * w2o[0].x + point.y * w2o[1].x + point.z * w2o[2].x + w2o[3].x,
// point.x * w2o[0].y + point.y * w2o[1].y + point.z * w2o[2].y + w2o[3].y,
// point.x * w2o[0].z + point.y * w2o[1].z + point.z * w2o[2].z + w2o[3].z

                        llvm::Value *idx, *row, *v;

                        Float4_struct const *mrow = &m_world_to_object[3];
                        res = ctx.get_constant(vec_tp, mrow->x, mrow->y, mrow->z);

                        for (int i = 2; i >= 0; --i) {
                            mrow = &m_world_to_object[i];
                            row = ctx.get_constant(vec_tp, mrow->x, mrow->y, mrow->z);
                            idx = ctx.get_constant(i);
                            v   = ctx->CreateExtractElement(point, idx);
                            v   = ctx.create_vector_splat(vec_tp, v);
                            v   = ctx->CreateFMul(v, row);
                            res = ctx->CreateFAdd(res, v);
                        }
                        ctx->CreateStore(res, result);
                        ctx->CreateBr(end_bb);
                    }
                    {
                        ctx->SetInsertPoint(o2w_bb);

// point.x * o2w[0].x + point.y * o2w[1].x + point.z * o2w[2].x + o2w[3].x,
// point.x * o2w[0].y + point.y * o2w[1].y + point.z * o2w[2].y + o2w[3].y,
// point.x * o2w[0].z + point.y * o2w[1].z + point.z * o2w[2].z + o2w[3].z

                        llvm::Value *idx, *row, *v;

                        Float4_struct const *mrow = &m_object_to_world[3];
                        res = ctx.get_constant(vec_tp, mrow->x, mrow->y, mrow->z);

                        for (int i = 2; i >= 0; --i) {
                            mrow = &m_object_to_world[i];
                            row = ctx.get_constant(vec_tp, mrow->x, mrow->y, mrow->z);
                            idx = ctx.get_constant(i);
                            v   = ctx->CreateExtractElement(point, idx);
                            v   = ctx.create_vector_splat(vec_tp, v);
                            v   = ctx->CreateFMul(v, row);
                            res = ctx->CreateFAdd(res, v);
                        }
                        ctx->CreateStore(res, result);
                        ctx->CreateBr(end_bb);
                    }
                } else {
                    MDL_ASSERT(!"non-vector mode NYI");
                    ctx->CreateBr(end_bb);
                }
            }
            ctx->SetInsertPoint(end_bb);
            res = ctx->CreateLoad(result);
            return Expression_result::value(res);
        }
        break;

    case mi::mdl::IDefinition::DS_INTRINSIC_STATE_TRANSFORM_NORMAL:
        // FIXME:
        // although normals must b transformed differently than vectors, we assume transform
        // matrices are orthonormal and just reuse the code

        // fallthrough
    case mi::mdl::IDefinition::DS_INTRINSIC_STATE_TRANSFORM_VECTOR:
        {
            // TODO: derivatives
            llvm::Value *vector =
                call_expr->translate_argument_value(*this, ctx, 2, /*return_derivs=*/ false);

            llvm::Value *cond           = ctx->CreateICmpEQ(from, to);
            llvm::BasicBlock *id_bb     = ctx.create_bb("id");
            llvm::BasicBlock *non_id_bb = ctx.create_bb("non_id");
            llvm::BasicBlock *end_bb    = ctx.create_bb("end");

            ctx->CreateCondBr(cond, id_bb, non_id_bb);
            {
                // the matrix is the identity, return the vector
                ctx->SetInsertPoint(id_bb);

                ctx->CreateStore(vector, result);
                ctx->CreateBr(end_bb);
            }
            {
                ctx->SetInsertPoint(non_id_bb);

                if (ret_tp->isVectorTy()) {
                    llvm::VectorType *vec_tp = llvm::cast<llvm::VectorType>(ret_tp);

                    // vector mode
                    llvm::Value *worldToObject =
                        ctx.get_constant(LLVM_code_generator::coordinate_world);
                    llvm::Value *cond = ctx->CreateICmpEQ(from, worldToObject);

                    llvm::BasicBlock *w2o_bb = ctx.create_bb("w2o");
                    llvm::BasicBlock *o2w_bb = ctx.create_bb("o2w");

                    ctx->CreateCondBr(cond, w2o_bb, o2w_bb);
                    {
                        ctx->SetInsertPoint(w2o_bb);

// vector.x * w2o[0].x + vector.y * w2o[1].x + vector.z * w2o[2].x
// vector.x * w2o[0].y + vector.y * w2o[1].y + vector.z * w2o[2].y
// vector.x * w2o[0].z + vector.y * w2o[1].z + vector.z * w2o[2].z

                        llvm::Value *idx, *row, *v;

                        res = llvm::Constant::getNullValue(ret_tp);
                        for (int i = 2; i >= 0; --i) {
                            Float4_struct const *mrow = &m_world_to_object[i];
                            idx = ctx.get_constant(i);
                            row = ctx.get_constant(vec_tp, mrow->x, mrow->y, mrow->z);
                            v   = ctx->CreateExtractElement(vector, idx);
                            v   = ctx.create_vector_splat(vec_tp, v);
                            v   = ctx->CreateFMul(v, row);
                            res = ctx->CreateFAdd(res, v);
                        }
                        ctx->CreateStore(res, result);
                        ctx->CreateBr(end_bb);
                    }
                    {
                        ctx->SetInsertPoint(o2w_bb);

// vector.x * o2w[0].x + vector.y * o2w[1].x + vector.z * o2w[2].x
// vector.x * o2w[0].y + vector.y * o2w[1].y + vector.z * o2w[2].y
// vector.x * o2w[0].z + vector.y * o2w[1].z + vector.z * o2w[2].z

                        llvm::Value *idx, *row, *v;

                        res = llvm::Constant::getNullValue(ret_tp);
                        for (int i = 2; i >= 0; --i) {
                            Float4_struct const *mrow = &m_object_to_world[i];
                            idx = ctx.get_constant(i);
                            row = ctx.get_constant(vec_tp, mrow->x, mrow->y, mrow->z);
                            v   = ctx->CreateExtractElement(vector, idx);
                            v   = ctx.create_vector_splat(vec_tp, v);
                            v   = ctx->CreateFMul(v, row);
                            res = ctx->CreateFAdd(res, v);
                        }
                        ctx->CreateStore(res, result);
                        ctx->CreateBr(end_bb);
                    }
                } else {
                    MDL_ASSERT(!"non-vector mode NYI");
                    ctx->CreateBr(end_bb);
                }
            }
            ctx->SetInsertPoint(end_bb);
            res = ctx->CreateLoad(result);
            return Expression_result::value(res);
        }
        break;
    case mi::mdl::IDefinition::DS_INTRINSIC_STATE_TRANSFORM_SCALE:
        {
            // TODO: derivatives
            llvm::Value *scale  =
                call_expr->translate_argument_value(*this, ctx, 2, /*return_derivs=*/ false);

            llvm::Value *cond           = ctx->CreateICmpEQ(from, to);
            llvm::BasicBlock *id_bb     = ctx.create_bb("id");
            llvm::BasicBlock *non_id_bb = ctx.create_bb("non_id");
            llvm::BasicBlock *end_bb    = ctx.create_bb("end");

            ctx->CreateCondBr(cond, id_bb, non_id_bb);
            {
                // the matrix is the identity, return the scale
                ctx->SetInsertPoint(id_bb);

                ctx->CreateStore(scale, result);
                ctx->CreateBr(end_bb);
            }
            {
                ctx->SetInsertPoint(non_id_bb);

                // || transform_vector(float3(1,0,0), a, b) || == || transform[0] ||
                float v_x = len_v3(m_world_to_object[0]);
                // || transform_vector(float3(0,1,0), a, b) || == || transform[1] ||
                float v_y = len_v3(m_world_to_object[1]);
                // || transform_vector(float3(0,0,1), a, b) || == || transform[2] ||
                float v_z = len_v3(m_world_to_object[2]);
                // scale *= (v_x + v_y + v_z)/3
                llvm::Value *res = ctx.get_constant((v_x + v_y + v_z) / 3.0f);
                res = ctx->CreateFMul(res, scale);

                ctx->CreateStore(res, result);
                ctx->CreateBr(end_bb);
            }
            ctx->SetInsertPoint(end_bb);
            res = ctx->CreateLoad(result);
            return Expression_result::value(res);
        }
        break;
    default:
        MDL_ASSERT(!"unsupported transform call");
    }
    return Expression_result::undef(ret_tp);
}

// Translate a state::object_id() call expression to LLVM IR.
Expression_result LLVM_code_generator::translate_object_id_call(
    Function_context       &ctx)
{
    // uses the object ID
    m_state_usage_analysis.add_state_usage(
        ctx.get_function(), IGenerated_code_executable::SU_OBJECT_ID);

    llvm::Value *res = ctx.get_object_id_value();
    return Expression_result::value(res);
}

// Translate a conversion call to LLVM IR.
Expression_result LLVM_code_generator::translate_conversion(
    Function_context          &ctx,
    mi::mdl::ICall_expr const *call_expr)
{
    MDL_ASSERT(call_expr->get_argument_count() == 1);

    mi::mdl::IType const *res_type  = call_expr->get_type()->skip_type_alias();
    mi::mdl::IType const *arg_type  = call_expr->get_argument_type(0)->skip_type_alias();
    mi::mdl::IType const *arg_noderiv_type = m_type_mapper.skip_deriv_type(arg_type);
    mi::mdl::IType const *res_noderiv_type = m_type_mapper.skip_deriv_type(res_type);

    bool return_derivs = call_expr->returns_derivatives(*this);

    // we only need derivatives for the argument, if the conversion results should be derivable
    // and the argument type supports it
    bool arg_derivs = return_derivs && m_type_mapper.is_floating_point_based_type(arg_noderiv_type);

    llvm::Value *v = call_expr->translate_argument_value(*this, ctx, 0, arg_derivs);

    llvm::Value *res;

    // will the conversion result in a dual value?
    if (return_derivs && m_type_mapper.is_floating_point_based_type(res_noderiv_type)) {

        // is the translated argument a dual value?
        if (m_type_mapper.is_deriv_type(v->getType())) {
            // dual -> dual: convert the dual component-wise

            llvm::Value *val = translate_conversion(
                ctx, res_noderiv_type, arg_noderiv_type, ctx.get_dual_val(v));
            llvm::Value *dx  = translate_conversion(
                ctx, res_noderiv_type, arg_noderiv_type, ctx.get_dual_dx(v));
            llvm::Value *dy  = translate_conversion(
                ctx, res_noderiv_type, arg_noderiv_type, ctx.get_dual_dy(v));

            res = ctx.get_dual(val, dx, dy);
        } else {
            // non-dual -> dual: convert, then get dual
            res = translate_conversion(ctx, res_noderiv_type, arg_noderiv_type, v);
            res = ctx.get_dual(res);
        }
    } else if (m_type_mapper.is_deriv_type(v->getType())) {
        // dual -> non-dual: strip dual, then convert
        res = translate_conversion(ctx, res_noderiv_type, arg_noderiv_type, ctx.get_dual_val(v));
    } else {
        // non-dual -> non-dual: just convert
        res = translate_conversion(ctx, res_noderiv_type, arg_noderiv_type, v);
    }

    return Expression_result::value(res);
}

// Translate a conversion call to LLVM IR.
// Note: Function may not be called with dual values and types
llvm::Value *LLVM_code_generator::translate_conversion(
    Function_context     &ctx,
    mi::mdl::IType const *res_type,
    mi::mdl::IType const *arg_type,
    llvm::Value          *v)
{
    if (arg_type == res_type) {
        // should not happen in optimized code, but ...
        return v;
    }

    llvm::Value *res = NULL;
    switch (res_type->get_kind()) {
    case mi::mdl::IType::TK_BOOL:
        // conversion to bool
        v = ctx.get_dual_val(v);  // strip any derivatives if present
        switch (arg_type->get_kind()) {
        case mi::mdl::IType::TK_INT:
            res = ctx->CreateICmp(llvm::ICmpInst::ICMP_NE, v, ctx.get_constant(int(0)));
            break;
        case mi::mdl::IType::TK_FLOAT:
            // MDL spec said "initialized to false for a numeric value equal to zero, and it
            // is initialized to true for all non-zero numeric values" which means NaN is true
            res = ctx->CreateFCmp(llvm::ICmpInst::FCMP_ONE, v, ctx.get_constant(float(0)));
            break;
        case mi::mdl::IType::TK_DOUBLE:
            // MDL spec said "initialized to false for a numeric value equal to zero, and it
            // is initialized to true for all non-zero numeric values" which means NaN is true
            res = ctx->CreateFCmp(llvm::ICmpInst::FCMP_ONE, v, ctx.get_constant(double(0)));
            break;
        default:
            // no others
            break;
        }
        // map the i1 result to the bool type representation
        res = ctx->CreateZExt(res, m_type_mapper.get_bool_type());
        break;

    case mi::mdl::IType::TK_INT:
        // conversion to int
        v = ctx.get_dual_val(v);  // strip any derivatives if present
        switch (arg_type->get_kind()) {
        case mi::mdl::IType::TK_BOOL:
            // convert bool to 0 or 1
            res = ctx->CreateZExt(v, m_type_mapper.get_int_type());
            break;
        case mi::mdl::IType::TK_FLOAT:
        case mi::mdl::IType::TK_DOUBLE:
            // MDL has no unsigned integer types, thus SI is what we want
            res = ctx->CreateFPToSI(v, m_type_mapper.get_int_type());
            break;
        case mi::mdl::IType::TK_ENUM:
            // enum to int conversion
            res = v;
            break;
        default:
            // no others
            break;
        }
        break;

    case mi::mdl::IType::TK_FLOAT:
    case mi::mdl::IType::TK_DOUBLE:
        // conversion to float or double
        {
            llvm::Type *tgt = lookup_type(res_type);
            switch (arg_type->get_kind()) {
            case mi::mdl::IType::TK_BOOL:
                v = ctx.get_dual_val(v);  // strip any derivatives if present
                v = ctx->CreateZExt(v, m_type_mapper.get_int_type());
                res = ctx->CreateSIToFP(v, tgt);
                break;
            case mi::mdl::IType::TK_INT:
                v = ctx.get_dual_val(v);  // strip any derivatives if present
                res = ctx->CreateSIToFP(v, tgt);
                break;
            case mi::mdl::IType::TK_FLOAT:
            case mi::mdl::IType::TK_DOUBLE:
                {
                    llvm::Type *src = lookup_type(arg_type);
                    if (src == tgt)
                        res = v;
                    else if (src->getPrimitiveSizeInBits() < tgt->getPrimitiveSizeInBits()) {
                        res = ctx->CreateFPExt(v, tgt);
                    } else {
                        res = ctx->CreateFPTrunc(v, tgt);
                    }
                    break;
                }
            default:
                // no others
                break;
            }
        }
        break;

    case mi::mdl::IType::TK_COLOR:
        // conversion to color
        return translate_color_conversion(ctx, arg_type, v);

    case mi::mdl::IType::TK_VECTOR:
        // conversion to vector type
        return
            translate_vector_conversion(ctx, cast<mi::mdl::IType_vector>(res_type), arg_type, v);

    case mi::mdl::IType::TK_MATRIX:
        // conversion to matrix type
        return
            translate_matrix_conversion(ctx, cast<mi::mdl::IType_matrix>(res_type), arg_type, v);

    default:
        // others
        break;
    }

    if (res != NULL)
        return res;

    MDL_ASSERT(!"Conversion not implemented");
    return llvm::UndefValue::get(lookup_type(res_type));
}

// Translate a conversion call to a vector type to LLVM IR.
llvm::Value *LLVM_code_generator::translate_vector_conversion(
    Function_context            &ctx,
    mi::mdl::IType_vector const *tgt_type,
    mi::mdl::IType const        *src_type,
    llvm::Value                 *v)
{
    llvm::Type *tgt = lookup_type(tgt_type);
    llvm::Type *src = lookup_type(src_type);

    if (src == tgt)
        return v;

    // There are two supported cases here:
    // Either we convert an element type to its vector type,
    // or a vector type to another vector type of same length
    llvm::Value *res = llvm::ConstantAggregateZero::get(tgt);

    if (mi::mdl::IType_vector const *vt = as<mi::mdl::IType_vector>(src_type)) {
        // convert vector type to vector type
        MDL_ASSERT(tgt_type->get_size() == vt->get_size());

        mi::mdl::IType const *t_elem = tgt_type->get_element_type();
        mi::mdl::IType const *s_elem = vt->get_element_type();

        if (tgt->isArrayTy()) {
            // vectors are represented by array types
            for (unsigned i = 0, n = tgt_type->get_size(); i < n; ++i) {
                unsigned idxes[1] = { i };
                llvm::Value *elem = ctx->CreateExtractValue(v, idxes);

                elem = translate_conversion(ctx, t_elem, s_elem, elem);
                res = ctx->CreateInsertValue(res, elem, idxes);
            }
        } else {
            // convert are represented by vector types
            for (int i = 0, n = tgt_type->get_size(); i < n; ++i) {
                llvm::Value *idx  = ctx.get_constant(i);
                llvm::Value *elem = ctx->CreateExtractElement(v, idx);

                elem = translate_conversion(ctx, t_elem, s_elem, elem);
                res = ctx->CreateInsertElement(res, elem, idx);
            }
        }
    } else {
        // convert element type to vector type
        MDL_ASSERT(tgt_type->get_element_type() == src_type->skip_type_alias());

        if (tgt->isArrayTy()) {
            // vectors are represented by array types
            unsigned idxes[1];
            for (int i = 0, n = tgt_type->get_size(); i < n; ++i) {
                idxes[0] = unsigned(i);
                res = ctx->CreateInsertValue(res, v, idxes);
            }
        } else {
            // vectors are represented by vector types
            for (int i = 0, n = tgt_type->get_size(); i < n; ++i) {
                res = ctx->CreateInsertElement(res, v, ctx.get_constant(i));
            }
        }
    }
    return res;
}

// Translate a conversion call to a matrix type to LLVM IR.
llvm::Value *LLVM_code_generator::translate_matrix_conversion(
    Function_context            &ctx,
    mi::mdl::IType_matrix const *tgt_type,
    mi::mdl::IType const        *src_type,
    llvm::Value                 *v)
{
    llvm::Type *tgt = lookup_type(tgt_type);
    llvm::Type *src = lookup_type(src_type);

    if (src == tgt)
        return v;

    // There are two supported cases here:
    // Either we convert an element type to its vector type,
    // or a vector type to another vector type of same length
    llvm::Value *res = llvm::ConstantAggregateZero::get(tgt);

    mi::mdl::IType_matrix const *mt = cast<mi::mdl::IType_matrix>(src_type);
    // convert matrix type to matrix type
    MDL_ASSERT(tgt_type->get_columns() == mt->get_columns() &&
           tgt_type->get_element_type()->get_size() == mt->get_element_type()->get_size());

    mi::mdl::IType_vector const *tv_elem = tgt_type->get_element_type();
    mi::mdl::IType_vector const *sv_elem = mt->get_element_type();

    if (llvm::ArrayType *arr_tp = llvm::dyn_cast<llvm::ArrayType>(tgt)) {
        llvm::Type *e_tp = arr_tp->getElementType();
        if (e_tp->isVectorTy()) {
            // matrices are represented as arrays of vectors
            for (unsigned i = 0, n = unsigned(tgt_type->get_compound_size()); i < n; ++i) {
                unsigned idxes[1] = { i };
                llvm::Value *elem = ctx->CreateExtractValue(v, idxes);

                elem = translate_conversion(ctx, tv_elem, sv_elem, elem);
                res = ctx->CreateInsertValue(res, elem, idxes);
            }
        } else {
            // matrices are represented by arrays of elemental types
            mi::mdl::IType const *t_elem = tv_elem->get_element_type();
            mi::mdl::IType const *s_elem = sv_elem->get_element_type();

            for (unsigned i = 0, n = unsigned(arr_tp->getNumContainedTypes()); i < n; ++i) {
                unsigned idxes[1] = { i };
                llvm::Value *elem = ctx->CreateExtractValue(v, idxes);

                elem = translate_conversion(ctx, t_elem, s_elem, elem);
                res = ctx->CreateInsertValue(res, elem, idxes);
            }
        }
    } else {
        // matrices are represented by vector types
        mi::mdl::IType const *t_elem = tv_elem->get_element_type();
        mi::mdl::IType const *s_elem = sv_elem->get_element_type();

        llvm::VectorType *v_tp = llvm::cast<llvm::VectorType>(tgt);
        for (int i = 0, n = int(v_tp->getNumElements()); i < n; ++i) {
            llvm::Value *idx  = ctx.get_constant(i);
            llvm::Value *elem = ctx->CreateExtractElement(v, idx);

            elem = translate_conversion(ctx, t_elem, s_elem, elem);
            res = ctx->CreateInsertElement(res, elem, idx);
        }
    }
    return res;
}

// Translate a conversion call to the color type to LLVM IR.
llvm::Value *LLVM_code_generator::translate_color_conversion(
    Function_context     &ctx,
    mi::mdl::IType const *src_type,
    llvm::Value          *v)
{
    llvm::Type *tgt = m_type_mapper.get_color_type();
    llvm::Type *src = lookup_type(src_type);

    if (src == tgt)
        return v;

    llvm::Value *res = llvm::ConstantAggregateZero::get(tgt);

    // support float -> color conversion
    MDL_ASSERT(src_type->skip_type_alias()->get_kind() == mi::mdl::IType::TK_FLOAT);

    if (tgt->isArrayTy()) {
        // vectors are represented by array types
        unsigned idxes[1];
        for (int i = 0, n = 3; i < n; ++i) {
            idxes[0] = unsigned(i);
            res = ctx->CreateInsertValue(res, v, idxes);
        }
    } else {
        // vectors are represented by vector types
        for (int i = 0, n = 3; i < n; ++i) {
            res = ctx->CreateInsertElement(res, v, ctx.get_constant(i));
        }
    }
    return res;
}

// Translate an elemental constructor call to LLVM IR.
Expression_result LLVM_code_generator::translate_elemental_constructor(
    Function_context          &ctx,
    mi::mdl::ICall_expr const *call_expr)
{
    llvm::Type *type = lookup_type_or_deriv_type(ctx, call_expr);
    llvm::Type *non_deriv_type = m_type_mapper.skip_deriv_type(type);
    bool return_derivs = call_expr->returns_derivatives(*this);

    mi::mdl::IType const *res_type = call_expr->get_type()->skip_type_alias();
    mi::mdl::IType const *res_non_deriv_type = m_type_mapper.skip_deriv_type(res_type);

    llvm::Value *agg = llvm::UndefValue::get(type);

    if (mi::mdl::IType_matrix const *m_tp = as<mi::mdl::IType_matrix>(res_non_deriv_type)) {
        // need extra handling here because the arguments of the matrix constructors are vectors
        mi::mdl::IType_vector const *v_tp = m_tp->get_element_type();

        int n_cols = m_tp->get_columns();
        int n_rows = v_tp->get_size();

        MDL_ASSERT(n_cols == call_expr->get_argument_count());

        if (llvm::isa<llvm::VectorType>(non_deriv_type)
                || !non_deriv_type->getArrayElementType()->isVectorTy()) {
            // matrix types are represented as vectors or as arrays of scalars
            int i = 0;

            if (return_derivs) {
                llvm::Type *elem_type = m_type_mapper.get_deriv_base_type(type);
                llvm::Value *agg_val = llvm::UndefValue::get(elem_type);
                llvm::Value *agg_dx = llvm::UndefValue::get(elem_type);
                llvm::Value *agg_dy = llvm::UndefValue::get(elem_type);

                for (int col = 0; col < n_cols; ++col) {
                    llvm::Value *vec = call_expr->translate_argument_value(
                        *this, ctx, col, return_derivs);
                    llvm::Value *vec_val = ctx.get_dual_val(vec);
                    llvm::Value *vec_dx  = ctx.get_dual_dx(vec);
                    llvm::Value *vec_dy  = ctx.get_dual_dy(vec);

                    for (int row = 0; row < n_rows; ++row) {
                        agg_val = ctx.create_insert(
                            agg_val, ctx.create_extract(vec_val, unsigned(row)), unsigned(i));
                        agg_dx  = ctx.create_insert(
                            agg_dx,  ctx.create_extract(vec_dx,  unsigned(row)), unsigned(i));
                        agg_dy  = ctx.create_insert(
                            agg_dy,  ctx.create_extract(vec_dy,  unsigned(row)), unsigned(i));
                        ++i;
                    }
                }
                agg = ctx.get_dual(agg_val, agg_dx, agg_dy);
            } else {
                for (int col = 0; col < n_cols; ++col) {
                    llvm::Value *vec = call_expr->translate_argument_value(
                        *this, ctx, col, return_derivs);

                    for (int row = 0; row < n_rows; ++row) {
                        llvm::Value *v = ctx.create_extract(vec, unsigned(row));
                        agg = ctx.create_insert(agg, v, unsigned(i++));
                    }
                }
            }
            return Expression_result::value(agg);
        }

        // matrices are arrays of vectors, fall through into default case
    }

    // default code handles structs and natural (i.e. arrays of vectors) matrices
    if (return_derivs) {
        llvm::Type *elem_type = m_type_mapper.get_deriv_base_type(type);
        llvm::Value *agg_val = llvm::UndefValue::get(elem_type);
        llvm::Value *agg_dx = llvm::UndefValue::get(elem_type);
        llvm::Value *agg_dy = llvm::UndefValue::get(elem_type);

        for (size_t i = 0, n = call_expr->get_argument_count(); i < n; ++i) {
            llvm::Value *v = call_expr->translate_argument_value(*this, ctx, i, return_derivs);
            agg_val = ctx.create_insert(agg_val, ctx.get_dual_val(v), unsigned(i));
            agg_dx  = ctx.create_insert(agg_dx,  ctx.get_dual_dx(v),  unsigned(i));
            agg_dy  = ctx.create_insert(agg_dy,  ctx.get_dual_dy(v),  unsigned(i));
        }
        agg = ctx.get_dual(agg_val, agg_dx, agg_dy);
    } else {
        for (size_t i = 0, n = call_expr->get_argument_count(); i < n; ++i) {
            llvm::Value *v = call_expr->translate_argument_value(*this, ctx, i, return_derivs);
            agg = ctx.create_insert(agg, v, unsigned(i));
        }
    }

    return Expression_result::value(agg);
}

// Translate a matrix elemental constructor call to LLVM IR.
Expression_result LLVM_code_generator::translate_matrix_elemental_constructor(
    Function_context          &ctx,
    mi::mdl::ICall_expr const *call_expr)
{
    llvm::Type *res_tp = lookup_type_or_deriv_type(ctx, call_expr);
    bool return_derivs = call_expr->returns_derivatives(*this);

    llvm::Value *matrix = llvm::ConstantAggregateZero::get(res_tp);

    mi::mdl::IType const *res_mdl_type = call_expr->get_type()->skip_type_alias();

    mi::mdl::IType_matrix const *m_type =
        cast<mi::mdl::IType_matrix>(m_type_mapper.skip_deriv_type(res_mdl_type));

    mi::mdl::IType_vector const *v_type = m_type->get_element_type();

    int n_col = m_type->get_columns();
    int n_row = v_type->get_size();
    MDL_ASSERT(n_col * n_row == call_expr->get_argument_count());

    if (return_derivs) {
        llvm::Type *value_tp = m_type_mapper.get_deriv_base_type(res_tp);
        llvm::Value *matrix_val = llvm::UndefValue::get(value_tp);
        llvm::Value *matrix_dx = llvm::UndefValue::get(value_tp);
        llvm::Value *matrix_dy = llvm::UndefValue::get(value_tp);

        if (llvm::ArrayType *a_tp = llvm::dyn_cast<llvm::ArrayType>(value_tp)) {
            llvm::Type *e_tp = a_tp->getArrayElementType();

            if (e_tp->isVectorTy()) {
                // matrices are represented as arrays of vectors
                llvm::Type *vector_type = lookup_type(v_type);
                llvm::Value *vector_val = llvm::UndefValue::get(vector_type);
                llvm::Value *vector_dx = llvm::UndefValue::get(vector_type);
                llvm::Value *vector_dy = llvm::UndefValue::get(vector_type);

                unsigned idx[1];

                size_t i = 0;
                for (int c = 0; c < n_col; ++c) {
                    for (int r = 0; r < n_row; ++r) {
                        llvm::Value *v   = call_expr->translate_argument_value(
                            *this, ctx, i++, return_derivs);
                        llvm::Value *idx = ctx.get_constant(r);

                        vector_val = ctx->CreateInsertElement(vector_val, ctx.get_dual_val(v), idx);
                        vector_dx  = ctx->CreateInsertElement(vector_dx,  ctx.get_dual_dx(v),  idx);
                        vector_dy  = ctx->CreateInsertElement(vector_dy,  ctx.get_dual_dy(v),  idx);
                    }
                    idx[0] = unsigned(c);
                    matrix_val = ctx->CreateInsertValue(matrix_val, vector_val, idx);
                    matrix_dx  = ctx->CreateInsertValue(matrix_dx,  vector_dx,  idx);
                    matrix_dy  = ctx->CreateInsertValue(matrix_dy,  vector_dy,  idx);
                }
                matrix = ctx.get_dual(matrix_val, matrix_dx, matrix_dy);
                return Expression_result::value(matrix);
            }
        }

        // matrices are represented as arrays of scalars or as plain vectors
        for (int i = 0; i < n_col * n_row; ++i) {
            llvm::Value *v = call_expr->translate_argument_value(*this, ctx, i, return_derivs);

            matrix_val = ctx.create_insert(matrix_val, ctx.get_dual_val(v), unsigned(i));
            matrix_dx  = ctx.create_insert(matrix_dx,  ctx.get_dual_dx(v),  unsigned(i));
            matrix_dy  = ctx.create_insert(matrix_dy,  ctx.get_dual_dy(v),  unsigned(i));
        }
        matrix = ctx.get_dual(matrix_val, matrix_dx, matrix_dy);
    } else {
        if (llvm::ArrayType *a_tp = llvm::dyn_cast<llvm::ArrayType>(res_tp)) {
            llvm::Type *e_tp = a_tp->getArrayElementType();

            if (e_tp->isVectorTy()) {
                // matrices are represented as arrays of vectors
                llvm::Value *vector = llvm::ConstantAggregateZero::get(lookup_type(v_type));

                unsigned idx[1];

                size_t i = 0;
                for (int c = 0; c < n_col; ++c) {
                    for (int r = 0; r < n_row; ++r) {
                        llvm::Value *v   = call_expr->translate_argument_value(
                            *this, ctx, i++, return_derivs);
                        llvm::Value *idx = ctx.get_constant(r);

                        vector = ctx->CreateInsertElement(vector, v, idx);
                    }
                    idx[0] = unsigned(c);
                    matrix = ctx->CreateInsertValue(matrix, vector, idx);
                }
                return Expression_result::value(matrix);
            }
        }

        // matrices are represented as arrays of scalars or as plain vectors
        for (int i = 0; i < n_col * n_row; ++i) {
            llvm::Value *v = call_expr->translate_argument_value(*this, ctx, i, return_derivs);

            matrix = ctx.create_insert(matrix, v, unsigned(i));
        }
    }
    return Expression_result::value(matrix);
}

// Translate a matrix diagonal constructor call to LLVM IR.
Expression_result LLVM_code_generator::translate_matrix_diagonal_constructor(
    Function_context          &ctx,
    mi::mdl::ICall_expr const *call_expr)
{
    MDL_ASSERT(call_expr->get_argument_count() == 1);

    llvm::Type *res_tp = lookup_type_or_deriv_type(ctx, call_expr);
    bool return_derivs = call_expr->returns_derivatives(*this);
    llvm::Value *matrix = llvm::ConstantAggregateZero::get(res_tp);

    mi::mdl::IType const *res_mdl_type = call_expr->get_type()->skip_type_alias();

    mi::mdl::IType_matrix const *m_type =
        cast<mi::mdl::IType_matrix>(m_type_mapper.skip_deriv_type(res_mdl_type));

    mi::mdl::IType_vector const *v_type = m_type->get_element_type();

    llvm::Value *v = call_expr->translate_argument_value(*this, ctx, 0, return_derivs);

    int n_col = m_type->get_columns();
    int n_row = v_type->get_size();

    if (return_derivs) {
        llvm::Value *v_val = ctx.get_dual_val(v);
        llvm::Value *v_dx  = ctx.get_dual_dx(v);
        llvm::Value *v_dy  = ctx.get_dual_dy(v);

        llvm::Type *value_tp = m_type_mapper.get_deriv_base_type(res_tp);

        llvm::Value *matrix_val = llvm::ConstantAggregateZero::get(value_tp);
        llvm::Value *matrix_dx = llvm::ConstantAggregateZero::get(value_tp);
        llvm::Value *matrix_dy = llvm::ConstantAggregateZero::get(value_tp);

        if (llvm::ArrayType *a_tp = llvm::dyn_cast<llvm::ArrayType>(value_tp)) {
            llvm::Type *e_tp = a_tp->getArrayElementType();

            if (e_tp->isVectorTy()) {
                // matrices are represented as arrays of vectors
                llvm::Type *vector_type = lookup_type(v_type);

                for (int i = 0; i < n_col && i < n_row; ++i) {
                    llvm::Value *vector_val = llvm::ConstantAggregateZero::get(vector_type);
                    llvm::Value *vector_dx = llvm::ConstantAggregateZero::get(vector_type);
                    llvm::Value *vector_dy = llvm::ConstantAggregateZero::get(vector_type);

                    vector_val = ctx.create_insert(vector_val, v_val, unsigned(i));
                    vector_dx  = ctx.create_insert(vector_dx,  v_dx,  unsigned(i));
                    vector_dy  = ctx.create_insert(vector_dy,  v_dy,  unsigned(i));

                    matrix_val = ctx.create_insert(matrix_val, vector_val, unsigned(i));
                    matrix_dx  = ctx.create_insert(matrix_dx,  vector_dx,  unsigned(i));
                    matrix_dy  = ctx.create_insert(matrix_dy,  vector_dy,  unsigned(i));
                }
                matrix = ctx.get_dual(matrix_val, matrix_dx, matrix_dy);
                return Expression_result::value(matrix);
            }
        }

        // matrices are represented as arrays of scalars or as plain vectors
        for (int i = 0; i < n_col && i < n_row; ++i) {
            unsigned idx = unsigned(i * n_row + i);
            matrix_val = ctx.create_insert(matrix_val, v_val, idx);
            matrix_dx  = ctx.create_insert(matrix_dx,  v_dx,  idx);
            matrix_dy  = ctx.create_insert(matrix_dy,  v_dy,  idx);
        }
        matrix = ctx.get_dual(matrix_val, matrix_dx, matrix_dy);
    } else {
        if (llvm::ArrayType *a_tp = llvm::dyn_cast<llvm::ArrayType>(res_tp)) {
            llvm::Type *e_tp = a_tp->getArrayElementType();

            if (e_tp->isVectorTy()) {
                // matrices are represented as arrays of vectors
                llvm::Type *vector_type = lookup_type(v_type);

                for (int i = 0; i < n_col && i < n_row; ++i) {
                    llvm::Value *vector = llvm::ConstantAggregateZero::get(vector_type);
                    vector = ctx.create_insert(vector, v, unsigned(i));
                    matrix = ctx.create_insert(matrix, vector, unsigned(i));
                }
            }
            return Expression_result::value(matrix);
        }

        // matrices are represented as arrays of scalars or as plain vectors
        for (int i = 0; i < n_col && i < n_row; ++i) {
            matrix = ctx.create_insert(matrix, v, i * n_row + i);
        }
    }
    return Expression_result::value(matrix);
}

// Translate an array constructor call to LLVM IR.
Expression_result LLVM_code_generator::translate_array_constructor_call(
    Function_context &ctx,
    ICall_expr const *call_expr)
{
    mi::mdl::IType const *res_type = call_expr->get_type()->skip_type_alias();
    bool return_derivs = call_expr->returns_derivatives(*this);

    size_t n = call_expr->get_argument_count();
    if (n == 1) {
        mi::mdl::IType const *arg_type = call_expr->get_argument_type(0)->skip_type_alias();
        if (is<mi::mdl::IType_array>(arg_type)) {
            // skip array copy constructor
            MDL_ASSERT(arg_type == res_type);
            return call_expr->translate_argument(*this, ctx, 0, return_derivs);
        }
    }

    // instantiate array type
    IType_array const *a_type = cast<IType_array>(m_type_mapper.skip_deriv_type(res_type));
    llvm::Type *type = lookup_type_or_deriv_type(ctx, call_expr);

    if (a_type->is_immediate_sized()) {
        llvm::Value *res = llvm::ConstantAggregateZero::get(type);

        if (return_derivs) {
            llvm::Type *elem_type = m_type_mapper.get_deriv_base_type(type);
            llvm::Value *val = llvm::UndefValue::get(elem_type);
            llvm::Value *dx = llvm::UndefValue::get(elem_type);
            llvm::Value *dy = llvm::UndefValue::get(elem_type);
            for (size_t i = 0; i < n; ++i) {
                llvm::Value *v = call_expr->translate_argument_value(
                    *this, ctx, i, /*return_derivs=*/ true);

                val = ctx->CreateInsertValue(val, ctx.get_dual_val(v), { unsigned(i) });
                dx  = ctx->CreateInsertValue(dx,  ctx.get_dual_dx(v),  { unsigned(i) });
                dy  = ctx->CreateInsertValue(dy,  ctx.get_dual_dy(v),  { unsigned(i) });
            }
            res = ctx.get_dual(val, dx, dy);
        } else {
            for (size_t i = 0; i < n; ++i) {
                llvm::Value *v = call_expr->translate_argument_value(
                    *this, ctx, i, /*return_derivs=*/ false);

                res = ctx->CreateInsertValue(res, v, { unsigned(i) });
            }
        }
        return Expression_result::value(res);
    } else {
        // array constructor for deferred sized array, should only occur as a
        // default constructor
        MDL_ASSERT(n == 0);

#if 0
        mi::mdl::IType_array_size const *size = res_type->get_deferred_size();
        mi::mdl::ISymbol const          *sym  = size->get_size_symbol();

        // The only source of array sizes here should be parameters, find them.
        mi::mdl::IDefinition const *param_def = ctx.find_parameter_for_size(sym);

        MDL_ASSERT(param_def != NULL && "could not find parameter for array size");

        LLVM_context_data *p_data = ctx.get_context_data(param_def);
        llvm::Value *arr_desc_ptr = p_data->get_var_address();
        llvm::Value *arr_len      = ctx.get_deferred_size_from_ptr(arr_desc_ptr);

        llvm::Type  *arr_dec_type = m_type_mapper.lookup_type(m_llvm_context, res_type);
        llvm::Value *desc         = ctx.create_local(arr_dec_type, "arr_desc");

        ctx.set_deferred_base(desc, expr_res.as_ptr(ctx));
        ctx.set_deferred_size(desc, arr_len);

        return Expression_result::ptr(desc);
#else
        // FIXME: creates an array of length 0, bad
        llvm::Value *res = llvm::ConstantAggregateZero::get(type);
        return Expression_result::value(res);
#endif
    }
}

// Translate a let expression to LLVM IR.
Expression_result LLVM_code_generator::translate_let(
    Function_context               &ctx,
    mi::mdl::IExpression_let const *let_expr,
    bool                           return_derivs)
{
    for (size_t i = 0, n = let_expr->get_declaration_count(); i < n; ++i) {
        mi::mdl::IDeclaration const *decl = let_expr->get_declaration(i);
        translate_declaration(ctx, decl);
    }
    return translate_expression(ctx, let_expr->get_expression(), return_derivs);
}

// Create a matrix by matrix multiplication.
llvm::Value *LLVM_code_generator::do_matrix_multiplication_MxM(
    Function_context &ctx,
    llvm::Type       *res_type,
    llvm::Value      *l,
    llvm::Value      *r,
    int              N,
    int              M,
    int              K)
{
    llvm::Value *res = llvm::UndefValue::get(res_type);

    if (llvm::isa<llvm::ArrayType>(res_type)) {
        llvm::ArrayType *arr_tp = llvm::cast<llvm::ArrayType>(res_type);
        llvm::Type      *e_tp   = arr_tp->getElementType();

        if (llvm::isa<llvm::VectorType>(e_tp)) {
            llvm::Type *vt_e_tp = e_tp->getVectorElementType();
            for (unsigned k = 0; k < (unsigned)K; ++k) {
                llvm::Value *res_col = llvm::UndefValue::get(e_tp);
                llvm::Value *b_col = ctx->CreateExtractValue(r, { unsigned(k) });
                for (unsigned n = 0; n < (unsigned)N; ++n) {
                    llvm::Value *tmp = llvm::Constant::getNullValue(vt_e_tp);
                    for (int m = 0; m < M; ++m) {
                        llvm::Value *a_col = ctx->CreateExtractValue(l, { unsigned(m) });
                        llvm::Value *a = ctx->CreateExtractElement(a_col, n);

                        llvm::Value *b = ctx->CreateExtractElement(b_col, m);

                        tmp = ctx->CreateFAdd(tmp, ctx->CreateFMul(a, b));
                    }
                    res_col = ctx->CreateInsertElement(res_col, tmp, n);
                }
                unsigned idx[1] = { k };
                res = ctx->CreateInsertValue(res, res_col, idx);
            }
        } else {
            for (unsigned n = 0; n < (unsigned)N; ++n) {
                for (unsigned k = 0; k < (unsigned)K; ++k) {
                    llvm::Value *tmp = llvm::Constant::getNullValue(e_tp);
                    for (int m = 0; m < M; ++m) {
                        unsigned l_idx[1] = { n + m * N };
                        llvm::Value *a = ctx->CreateExtractValue(l, l_idx);

                        unsigned r_idx[1] = { m + k * M };
                        llvm::Value *b = ctx->CreateExtractValue(r, r_idx);

                        tmp = ctx->CreateFAdd(tmp, ctx->CreateFMul(a, b));
                    }
                    unsigned idx[1] = { n + k * N };
                    res = ctx->CreateInsertValue(res, tmp, idx);
                }
            }
        }
    } else {
        // "big vectors" mode naive implementation
        llvm::VectorType *v_tp = llvm::cast<llvm::VectorType>(res_type);
        llvm::Type       *e_tp = v_tp->getElementType();

        for (int n = 0; n < N; ++n) {
            for (int k = 0; k < K; ++k) {
                llvm::Value *tmp = llvm::Constant::getNullValue(e_tp);
                for (int m = 0; m < M; ++m) {
                    llvm::Value *l_idx = ctx.get_constant(n + m * N);
                    llvm::Value *a     = ctx->CreateExtractElement(l, l_idx);

                    llvm::Value *r_idx = ctx.get_constant(m + k * M);
                    llvm::Value *b     = ctx->CreateExtractElement(r, r_idx);

                    tmp = ctx->CreateFAdd(tmp, ctx->CreateFMul(a, b));
                }
                llvm::Value *idx = ctx.get_constant(n + k * N);
                res = ctx->CreateInsertElement(res, tmp, idx);
            }
        }
    }
    return res;
}

// Create a matrix by matrix multiplication with derivatives.
llvm::Value *LLVM_code_generator::do_matrix_multiplication_MxM_deriv(
    Function_context &ctx,
    llvm::Type       *res_type,
    llvm::Value      *l,
    llvm::Value      *r,
    int              N,
    int              M,
    int              K)
{
    llvm::Value *l_val = ctx.get_dual_val(l);
    llvm::Value *l_dx  = ctx.get_dual_dx(l);
    llvm::Value *l_dy  = ctx.get_dual_dy(l);

    llvm::Value *r_val = ctx.get_dual_val(r);
    llvm::Value *r_dx  = ctx.get_dual_dx(r);
    llvm::Value *r_dy  = ctx.get_dual_dy(r);

    llvm::Type  *res_val_type = m_type_mapper.get_deriv_base_type(res_type);
    llvm::Value *res_val = llvm::UndefValue::get(res_val_type);
    llvm::Value *res_dx  = res_val;
    llvm::Value *res_dy  = res_val;

    if (llvm::ArrayType *arr_tp = llvm::dyn_cast<llvm::ArrayType>(res_val_type)) {
        llvm::Type *e_tp   = arr_tp->getElementType();

        if (llvm::isa<llvm::VectorType>(e_tp)) {
            // small vector mode
            llvm::Type *vt_e_tp = e_tp->getVectorElementType();
            for (unsigned k = 0; k < (unsigned)K; ++k) {
                unsigned k_idxes[] = { unsigned(k) };

                llvm::Value *res_col_val = llvm::UndefValue::get(e_tp);
                llvm::Value *res_col_dx  = res_col_val;
                llvm::Value *res_col_dy  = res_col_val;

                llvm::Value *b_col_val = ctx->CreateExtractValue(r_val, k_idxes);
                llvm::Value *b_col_dx  = ctx->CreateExtractValue(r_dx,  k_idxes);
                llvm::Value *b_col_dy  = ctx->CreateExtractValue(r_dy,  k_idxes);

                for (unsigned n = 0; n < (unsigned)N; ++n) {
                    llvm::Value *tmp_val = llvm::Constant::getNullValue(vt_e_tp);
                    llvm::Value *tmp_dx  = tmp_val;
                    llvm::Value *tmp_dy  = tmp_val;

                    for (int m = 0; m < M; ++m) {
                        unsigned m_idxes[] = { unsigned(m) };

                        llvm::Value *a_col_val = ctx->CreateExtractValue(l_val, m_idxes);
                        llvm::Value *a_col_dx  = ctx->CreateExtractValue(l_dx,  m_idxes);
                        llvm::Value *a_col_dy  = ctx->CreateExtractValue(l_dy,  m_idxes);

                        llvm::Value *a_val = ctx->CreateExtractElement(a_col_val, n);
                        llvm::Value *a_dx  = ctx->CreateExtractElement(a_col_dx,  n);
                        llvm::Value *a_dy  = ctx->CreateExtractElement(a_col_dy,  n);

                        llvm::Value *b_val = ctx->CreateExtractElement(b_col_val, m);
                        llvm::Value *b_dx  = ctx->CreateExtractElement(b_col_dx,  m);
                        llvm::Value *b_dy  = ctx->CreateExtractElement(b_col_dy,  m);

                        tmp_val = ctx->CreateFAdd(tmp_val, ctx->CreateFMul(a_val, b_val));

                        tmp_dx = ctx->CreateFAdd(tmp_dx, ctx->CreateFMul(a_val, b_dx));
                        tmp_dx = ctx->CreateFAdd(tmp_dx, ctx->CreateFMul(a_dx, b_val));

                        tmp_dy = ctx->CreateFAdd(tmp_dy, ctx->CreateFMul(a_val, b_dy));
                        tmp_dy = ctx->CreateFAdd(tmp_dy, ctx->CreateFMul(a_dy, b_val));
                    }
                    res_col_val = ctx->CreateInsertElement(res_col_val, tmp_val, n);
                    res_col_dx  = ctx->CreateInsertElement(res_col_dx,  tmp_dx,  n);
                    res_col_dy  = ctx->CreateInsertElement(res_col_dy,  tmp_dy,  n);
                }
                res_val = ctx->CreateInsertValue(res_val, res_col_val, k_idxes);
                res_dx  = ctx->CreateInsertValue(res_dx,  res_col_dx,  k_idxes);
                res_dy  = ctx->CreateInsertValue(res_dy,  res_col_dy,  k_idxes);
            }
        } else {
            // all atomic mode
            for (unsigned n = 0; n < (unsigned)N; ++n) {
                for (unsigned k = 0; k < (unsigned)K; ++k) {
                    llvm::Value *tmp_val = llvm::Constant::getNullValue(e_tp);
                    llvm::Value *tmp_dx  = tmp_val;
                    llvm::Value *tmp_dy  = tmp_val;

                    for (int m = 0; m < M; ++m) {
                        unsigned l_idx[1] = { n + m * N };
                        llvm::Value *a_val = ctx->CreateExtractValue(l_val, l_idx);
                        llvm::Value *a_dx  = ctx->CreateExtractValue(l_dx, l_idx);
                        llvm::Value *a_dy  = ctx->CreateExtractValue(l_dy, l_idx);

                        unsigned r_idx[1] = { m + k * M };
                        llvm::Value *b_val = ctx->CreateExtractValue(r_val, r_idx);
                        llvm::Value *b_dx  = ctx->CreateExtractValue(r_dx, r_idx);
                        llvm::Value *b_dy  = ctx->CreateExtractValue(r_dy, r_idx);

                        tmp_val = ctx->CreateFAdd(tmp_val, ctx->CreateFMul(a_val, b_val));

                        tmp_dx = ctx->CreateFAdd(tmp_dx, ctx->CreateFMul(a_val, b_dx));
                        tmp_dx = ctx->CreateFAdd(tmp_dx, ctx->CreateFMul(a_dx, b_val));

                        tmp_dy = ctx->CreateFAdd(tmp_dy, ctx->CreateFMul(a_val, b_dy));
                        tmp_dy = ctx->CreateFAdd(tmp_dy, ctx->CreateFMul(a_dy, b_val));
                    }
                    unsigned idx[1] = { n + k * N };
                    res_val = ctx->CreateInsertValue(res_val, tmp_val, idx);
                    res_dx  = ctx->CreateInsertValue(res_dx, tmp_dx, idx);
                    res_dy  = ctx->CreateInsertValue(res_dy, tmp_dy, idx);
                }
            }
        }
    } else {
        // "big vectors" mode naive implementation
        llvm::VectorType *v_tp = llvm::cast<llvm::VectorType>(res_val_type);
        llvm::Type       *e_tp = v_tp->getElementType();

        for (int n = 0; n < N; ++n) {
            for (int k = 0; k < K; ++k) {
                llvm::Value *tmp_val = llvm::Constant::getNullValue(e_tp);
                llvm::Value *tmp_dx  = tmp_val;
                llvm::Value *tmp_dy  = tmp_val;
                for (int m = 0; m < M; ++m) {
                    llvm::Value *l_idx = ctx.get_constant(n + m * N);
                    llvm::Value *a_val = ctx->CreateExtractElement(l_val, l_idx);
                    llvm::Value *a_dx  = ctx->CreateExtractElement(l_dx, l_idx);
                    llvm::Value *a_dy  = ctx->CreateExtractElement(l_dy, l_idx);

                    llvm::Value *r_idx = ctx.get_constant(m + k * M);
                    llvm::Value *b_val = ctx->CreateExtractElement(r_val, r_idx);
                    llvm::Value *b_dx  = ctx->CreateExtractElement(r_dx, r_idx);
                    llvm::Value *b_dy  = ctx->CreateExtractElement(r_dy, r_idx);

                    tmp_val = ctx->CreateFAdd(tmp_val, ctx->CreateFMul(a_val, b_val));

                    tmp_dx = ctx->CreateFAdd(tmp_dx, ctx->CreateFMul(a_val, b_dx));
                    tmp_dx = ctx->CreateFAdd(tmp_dx, ctx->CreateFMul(a_dx, b_val));

                    tmp_dy = ctx->CreateFAdd(tmp_dy, ctx->CreateFMul(a_val, b_dy));
                    tmp_dy = ctx->CreateFAdd(tmp_dy, ctx->CreateFMul(a_dy, b_val));
                }
                llvm::Value *idx = ctx.get_constant(n + k * N);
                res_val = ctx->CreateInsertElement(res_val, tmp_val, idx);
                res_dx  = ctx->CreateInsertElement(res_dx,  tmp_dx,  idx);
                res_dy  = ctx->CreateInsertElement(res_dy,  tmp_dy,  idx);
            }
        }
    }
    return ctx.get_dual(res_val, res_dx, res_dy);
}

// Create a vector by matrix multiplication.
llvm::Value *LLVM_code_generator::do_matrix_multiplication_VxM(
    Function_context &ctx,
    llvm::Type       *res_type,
    llvm::Value      *l,
    llvm::Value      *r,
    int              M,
    int              K)
{
    llvm::Value *res = llvm::UndefValue::get(res_type);
    llvm::Type  *e_tp;

    if (llvm::ArrayType *arr_tp = llvm::dyn_cast<llvm::ArrayType>(r->getType())) {
        e_tp   = arr_tp->getElementType();

        if (llvm::isa<llvm::VectorType>(e_tp)) {
            llvm::Type *vt_e_tp = e_tp->getVectorElementType();
            for (unsigned k = 0; k < (unsigned)K; ++k) {
                llvm::Value *tmp = llvm::Constant::getNullValue(vt_e_tp);
                llvm::Value *b_col = ctx->CreateExtractValue(r, { unsigned(k) });

                for (int m = 0; m < M; ++m) {
                    llvm::Value *a = ctx->CreateExtractElement(l, m);

                    llvm::Value *b = ctx->CreateExtractElement(b_col, m);

                    tmp = ctx->CreateFAdd(tmp, ctx->CreateFMul(a, b));
                }
                res = ctx->CreateInsertElement(res, tmp, k);
            }
            return res;
        }

        // arrays of scalars, fall through
    } else {
        // "big vectors" mode naive implementation
        llvm::VectorType *v_tp = llvm::cast<llvm::VectorType>(res_type);
        e_tp = v_tp->getElementType();
    }

    for (int k = 0; k < K; ++k) {
        llvm::Value *tmp = llvm::Constant::getNullValue(e_tp);
        for (int m = 0; m < M; ++m) {
            llvm::Value *a = ctx.create_extract(l, unsigned(m));
            llvm::Value *b = ctx.create_extract(r, unsigned(m + k * M));

            tmp = ctx->CreateFAdd(tmp, ctx->CreateFMul(a, b));
        }
        res = ctx.create_insert(res, tmp, unsigned(k));
    }
    return res;
}

// Create a vector by matrix multiplication with derivatives.
llvm::Value *LLVM_code_generator::do_matrix_multiplication_VxM_deriv(
    Function_context &ctx,
    llvm::Type       *res_type,
    llvm::Value      *l,
    llvm::Value      *r,
    int              M,
    int              K)
{
    llvm::Type  *res_vec_type = m_type_mapper.get_deriv_base_type(res_type);
    llvm::Value *res_val = llvm::UndefValue::get(res_vec_type);
    llvm::Value *res_dx = llvm::UndefValue::get(res_vec_type);
    llvm::Value *res_dy = llvm::UndefValue::get(res_vec_type);
    llvm::Type  *e_tp;

    llvm::Value *mat_val = ctx.get_dual_val(r);
    llvm::Value *mat_dx  = ctx.get_dual_dx(r);
    llvm::Value *mat_dy  = ctx.get_dual_dy(r);

    if (llvm::ArrayType *arr_tp = llvm::dyn_cast<llvm::ArrayType>(mat_val->getType())) {
        e_tp = arr_tp->getElementType();

        if (llvm::isa<llvm::VectorType>(e_tp)) {
            llvm::Type *vt_e_tp = e_tp->getVectorElementType();
            for (unsigned k = 0; k < (unsigned)K; ++k) {
                llvm::Value *tmp = llvm::Constant::getNullValue(vt_e_tp);
                llvm::Value *b_val_col = ctx->CreateExtractValue(mat_val, { k });
                llvm::Value *b_dx_col  = ctx->CreateExtractValue(mat_dx,  { k });
                llvm::Value *b_dy_col  = ctx->CreateExtractValue(mat_dy,  { k });

                for (unsigned m = 0; m < (unsigned)M; ++m) {
                    llvm::Value *a = ctx.extract_dual(l, m);

                    llvm::Value *b_val = ctx->CreateExtractElement(b_val_col, m);
                    llvm::Value *b_dx  = ctx->CreateExtractElement(b_dx_col,  m);
                    llvm::Value *b_dy  = ctx->CreateExtractElement(b_dy_col,  m);

                    llvm::Value *b = ctx.get_dual(b_val, b_dx, b_dy);

                    tmp = ctx.create_deriv_add(vt_e_tp, tmp, ctx.create_deriv_mul(vt_e_tp, a, b));
                }

                res_val = ctx.create_insert(res_val, ctx.get_dual_val(tmp), k);
                res_dx  = ctx.create_insert(res_dx,  ctx.get_dual_dx(tmp),  k);
                res_dy  = ctx.create_insert(res_dy,  ctx.get_dual_dy(tmp),  k);
            }
            return ctx.get_dual(res_val, res_dx, res_dy);
        }

        // arrays of scalars, fall through
    } else {
        // "big vectors" mode naive implementation
        llvm::VectorType *v_tp = llvm::cast<llvm::VectorType>(res_vec_type);
        e_tp = v_tp->getElementType();
    }

    for (int k = 0; k < K; ++k) {
        llvm::Value *tmp = llvm::Constant::getNullValue(e_tp);
        for (int m = 0; m < M; ++m) {
            llvm::Value *a = ctx.extract_dual(l, unsigned(m));
            llvm::Value *b = ctx.extract_dual(r, unsigned(m + k * M));

            tmp = ctx.create_deriv_add(e_tp, tmp, ctx.create_deriv_mul(e_tp, a, b));
        }
        res_val = ctx.create_insert(res_val, ctx.get_dual_val(tmp), unsigned(k));
        res_dx  = ctx.create_insert(res_dx,  ctx.get_dual_dx(tmp),  unsigned(k));
        res_dy  = ctx.create_insert(res_dy,  ctx.get_dual_dy(tmp),  unsigned(k));
    }
    return ctx.get_dual(res_val, res_dx, res_dy);
}

// Create a matrix by vector multiplication.
llvm::Value *LLVM_code_generator::do_matrix_multiplication_MxV(
    Function_context &ctx,
    llvm::Type       *res_type,
    llvm::Value      *l,
    llvm::Value      *r,
    int              N,
    int              M)
{
    llvm::Value *res = llvm::UndefValue::get(res_type);
    llvm::Type  *e_tp;

    if (llvm::ArrayType *arr_tp = llvm::dyn_cast<llvm::ArrayType>(l->getType())) {
        e_tp = arr_tp->getElementType();

        if (llvm::isa<llvm::VectorType>(e_tp)) {
            llvm::Type *vt_e_tp = e_tp->getVectorElementType();
            for (unsigned n = 0; n < (unsigned)N; ++n) {
                llvm::Value *tmp = llvm::Constant::getNullValue(vt_e_tp);

                for (int m = 0; m < M; ++m) {
                    llvm::Value *a_col = ctx->CreateExtractValue(l, { unsigned(m) });
                    llvm::Value *a = ctx->CreateExtractElement(a_col, n);

                    llvm::Value *b = ctx->CreateExtractElement(r, m);

                    tmp = ctx->CreateFAdd(tmp, ctx->CreateFMul(a, b));
                }

                res = ctx->CreateInsertElement(res, tmp, n);
            }
            return res;
        }

        // arrays of scalars, fall through
    } else {
        // "big vectors" mode naive implementation
        llvm::VectorType *v_tp = llvm::cast<llvm::VectorType>(res_type);
        e_tp = v_tp->getElementType();
    }

    for (int n = 0; n < N; ++n) {
        llvm::Value *tmp = llvm::Constant::getNullValue(e_tp);
        for (int m = 0; m < M; ++m) {
            llvm::Value *a = ctx.create_extract(l, unsigned(n + m * N));
            llvm::Value *b = ctx.create_extract(r, unsigned(m));

            tmp = ctx->CreateFAdd(tmp, ctx->CreateFMul(a, b));
        }
        res = ctx.create_insert(res, tmp, unsigned(n));
    }
    return res;
}

// Create a matrix by vector multiplication with derivatives.
llvm::Value *LLVM_code_generator::do_matrix_multiplication_MxV_deriv(
    Function_context &ctx,
    llvm::Type       *res_type,
    llvm::Value      *l,
    llvm::Value      *r,
    int              N,
    int              M)
{
    llvm::Type  *res_vec_type = m_type_mapper.get_deriv_base_type(res_type);
    llvm::Value *res_val = llvm::UndefValue::get(res_vec_type);
    llvm::Value *res_dx = llvm::UndefValue::get(res_vec_type);
    llvm::Value *res_dy = llvm::UndefValue::get(res_vec_type);
    llvm::Type  *e_tp;

    llvm::Value *mat_val = ctx.get_dual_val(l);
    llvm::Value *mat_dx  = ctx.get_dual_dx(l);
    llvm::Value *mat_dy  = ctx.get_dual_dy(l);

    if (llvm::ArrayType *arr_tp = llvm::dyn_cast<llvm::ArrayType>(mat_val->getType())) {
        e_tp = arr_tp->getElementType();

        if (llvm::isa<llvm::VectorType>(e_tp)) {
            llvm::Type *vt_e_tp = e_tp->getVectorElementType();
            for (unsigned n = 0; n < (unsigned)N; ++n) {
                llvm::Value *tmp = llvm::Constant::getNullValue(vt_e_tp);

                for (unsigned m = 0; m < (unsigned)M; ++m) {
                    llvm::Value *a_val_col = ctx->CreateExtractValue(mat_val, { m });
                    llvm::Value *a_val = ctx->CreateExtractElement(a_val_col, n);

                    llvm::Value *a_dx_col = ctx->CreateExtractValue(mat_dx, { m });
                    llvm::Value *a_dx = ctx->CreateExtractElement(a_dx_col, n);

                    llvm::Value *a_dy_col = ctx->CreateExtractValue(mat_dy, { m });
                    llvm::Value *a_dy = ctx->CreateExtractElement(a_dy_col, n);

                    llvm::Value *a = ctx.get_dual(a_val, a_dx, a_dy);

                    llvm::Value *b = ctx.extract_dual(r, m);

                    tmp = ctx.create_deriv_add(vt_e_tp, tmp, ctx.create_deriv_mul(vt_e_tp, a, b));
                }

                res_val = ctx.create_insert(res_val, ctx.get_dual_val(tmp), n);
                res_dx  = ctx.create_insert(res_dx,  ctx.get_dual_dx(tmp),  n);
                res_dy  = ctx.create_insert(res_dy,  ctx.get_dual_dy(tmp),  n);
            }
            return ctx.get_dual(res_val, res_dx, res_dy);
        }

        // arrays of scalars, fall through
    } else {
        // "big vectors" mode naive implementation
        llvm::VectorType *v_tp = llvm::cast<llvm::VectorType>(res_vec_type);
        e_tp = v_tp->getElementType();
    }

    for (int n = 0; n < N; ++n) {
        llvm::Value *tmp = llvm::Constant::getNullValue(e_tp);
        for (int m = 0; m < M; ++m) {
            llvm::Value *a = ctx.extract_dual(l, unsigned(n + m * N));
            llvm::Value *b = ctx.extract_dual(r, unsigned(m));

            tmp = ctx.create_deriv_add(e_tp, tmp, ctx.create_deriv_mul(e_tp, a, b));
        }
        res_val = ctx.create_insert(res_val, ctx.get_dual_val(tmp), unsigned(n));
        res_dx  = ctx.create_insert(res_dx,  ctx.get_dual_dx(tmp),  unsigned(n));
        res_dy  = ctx.create_insert(res_dy,  ctx.get_dual_dy(tmp),  unsigned(n));
    }
    return ctx.get_dual(res_val, res_dx, res_dy);
}

// Translate a DAG node into LLVM IR.
Expression_result LLVM_code_generator::translate_node(
    Function_context                   &ctx,
    mi::mdl::DAG_node const            *node,
    mi::mdl::ICall_name_resolver const *resolver)
{
    Node_value_map::iterator it = m_node_value_map.find(Value_entry(node, m_curr_bb));
    if (it != m_node_value_map.end())
        return it->second;

    Expression_result res;

    switch (node->get_kind()) {
    case mi::mdl::DAG_node::EK_CONSTANT:
        // simple case: just a constant
        {
            mi::mdl::DAG_constant const *c = cast<mi::mdl::DAG_constant>(node);
            mi::mdl::IValue const *v = c->get_value();

            res = translate_value(ctx, v);
        }
        break;

    case mi::mdl::DAG_node::EK_TEMPORARY:
        // there should be no temporaries at this point
        res = translate_node(ctx, cast<mi::mdl::DAG_temporary>(node)->get_expr(), resolver);
        break;

    case mi::mdl::DAG_node::EK_CALL:
        {
            DAG_call const *call_node = cast<DAG_call>(node);
            Call_dag_expr call(call_node, resolver);

            // We must be inside some module when the call is translated, but if it is DAG-call,
            // there is no module.
            // Solve this by entering the owner module of the called entity.

            IDefinition::Semantics sema = call_node->get_semantic();
            if (sema == IDefinition::DS_INTRINSIC_DAG_SET_TRANSFORMS ||
                sema == IDefinition::DS_INTRINSIC_DAG_SET_OBJECT_ID)
            {
                // these two have no module and no owner
                res = translate_call(ctx, &call);
            } else {
                char const *signature = call_node->get_name();
                if (signature[0] == '#') {
                    // skip prefix for derivative variants
                    ++signature;
                }
                mi::base::Handle<mi::mdl::IModule const> mod(resolver->get_owner_module(signature));
                MDL_module_scope scope(*this, mod.get());

                res = translate_call(ctx, &call);
            }
        }
        break;

    case mi::mdl::DAG_node::EK_PARAMETER:
        {
            DAG_parameter const *param_node = cast<DAG_parameter>(node);
            res = translate_parameter(ctx, param_node);
        }
        break;
    }
    if (res.is_unset()) {
        MDL_ASSERT(!"Unsupported DAG node kind");
        res = Expression_result::undef(lookup_type(node->get_type()));
    }
    m_node_value_map[Value_entry(node, m_curr_bb)] = res;

    return res;
}

// Translate a parameter offset into LLVM IR by adding the argument block offset of the state.
llvm::Value *LLVM_code_generator::translate_parameter_hlsl_offset(
    Function_context &ctx,
    int               cur_offs)
{
    llvm::Value *state = ctx.get_state_parameter();
    llvm::Value *adr   = ctx.create_simple_gep_in_bounds(
        state, ctx.get_constant(
        m_type_mapper.get_state_index(Type_mapper::STATE_CORE_ARG_BLOCK_OFFSET)));
    llvm::Value *arg_block_offs = ctx->CreateLoad(adr);
    return ctx->CreateAdd(arg_block_offs, ctx.get_constant(cur_offs));
}

// Translate a part of a DAG parameter for HLSL into LLVM IR.
llvm::Value *LLVM_code_generator::translate_parameter_hlsl_value(
    Function_context             &ctx,
    mi::mdl::IType const         *param_type,
    int                          &cur_offs)
{
    llvm::Value *res;
    param_type = param_type->skip_type_alias();

    switch (param_type->get_kind()) {
    case mi::mdl::IType::TK_BOOL:
        res = ctx->CreateCall(m_hlsl_func_argblock_as_bool, translate_parameter_hlsl_offset(ctx, cur_offs));
        ++cur_offs;
        break;

    case mi::mdl::IType::TK_FLOAT:
        cur_offs = (cur_offs + 3) & ~3;
        res = ctx->CreateCall(m_hlsl_func_argblock_as_float, translate_parameter_hlsl_offset(ctx, cur_offs));
        cur_offs += 4;
        break;

    case mi::mdl::IType::TK_INT:
    case mi::mdl::IType::TK_ENUM:
    case mi::mdl::IType::TK_STRING:
        cur_offs = (cur_offs + 3) & ~3;
        res = ctx->CreateCall(m_hlsl_func_argblock_as_int, translate_parameter_hlsl_offset(ctx, cur_offs));
        cur_offs += 4;
        break;

    case mi::mdl::IType::TK_DOUBLE:
        cur_offs = (cur_offs + 7) & ~7;
        res = ctx->CreateCall(m_hlsl_func_argblock_as_double, translate_parameter_hlsl_offset(ctx, cur_offs));
        cur_offs += 8;
        break;

    case mi::mdl::IType::TK_VECTOR:
    case mi::mdl::IType::TK_MATRIX:
    case mi::mdl::IType::TK_ARRAY:
    case mi::mdl::IType::TK_COLOR:
    case mi::mdl::IType::TK_STRUCT:
        {
            mi::mdl::IType_compound const *ct = cast<mi::mdl::IType_compound>(param_type);

            llvm::Type *res_type = m_type_mapper.lookup_type(m_llvm_context, ct);
            size_t size = size_t(m_data_layout.getTypeAllocSize(res_type));
            int compound_start_offs = cur_offs;

            res = llvm::UndefValue::get(res_type);
            for (int i = 0, n = ct->get_compound_size(); i < n; ++i) {
                mi::mdl::IType const *et = ct->get_compound_type(i);
                res = ctx.create_insert(
                    res,
                    translate_parameter_hlsl_value(ctx, et, cur_offs),
                    unsigned(i));
            }

            // compound values might have an higher alignment then the sum of its components
            cur_offs = compound_start_offs + size;
        }
        break;

    case mi::mdl::IType::TK_TEXTURE:
    case mi::mdl::IType::TK_LIGHT_PROFILE:
    case mi::mdl::IType::TK_BSDF_MEASUREMENT:
        // resources are mapped to integer in HLSL
        cur_offs = (cur_offs + 3) & ~3;
        res = ctx->CreateCall(m_hlsl_func_argblock_as_int, translate_parameter_hlsl_offset(ctx, cur_offs));
        cur_offs += 4;
        break;

    default:
        MDL_ASSERT(!"Unexpected parameter type");
        res = llvm::UndefValue::get(m_type_mapper.lookup_type(m_llvm_context, param_type));
        break;
    }
    return res;
}

// Translate a DAG parameter into LLVM IR
Expression_result LLVM_code_generator::translate_parameter(
    Function_context             &ctx,
    mi::mdl::DAG_parameter const *param_node)
{
    if (m_target_lang == TL_HLSL) {
        // TODO: Maybe use custom datalayout for HLSL
        llvm::DataLayout const *dl = get_target_layout_data();
        llvm::StructLayout const *sl = dl->getStructLayout(m_captured_args_type);
        int param_offs = int(sl->getElementOffset(param_node->get_index()));

        mi::mdl::IType const *param_type = param_node->get_type();
        llvm::Value *res = translate_parameter_hlsl_value(ctx, param_type, param_offs);
        return Expression_result::value(res);
    }

    llvm::Value *args = ctx->CreatePointerCast(
        ctx.get_cap_args_parameter(),
        m_type_mapper.get_ptr(m_captured_args_type));
    llvm::Value *adr  = ctx.create_simple_gep_in_bounds(
        args,
        ctx.get_constant(param_node->get_index()));
    return Expression_result::ptr(adr);
}

// Compile all functions waiting in the wait queue into the current module.
void LLVM_code_generator::compile_waiting_functions()
{
    // compile all referenced functions that are not compiled so far
    while (!m_functions_q.empty()) {
        Wait_entry const &entry = m_functions_q.front();
        Function_instance const &func_inst = entry.get_instance();
        mi::mdl::IModule const  *owner     = entry.get_owner();

        LLVM_context_data *p_data = get_context_data(func_inst);
        llvm::Function    *func   = p_data->get_function();

        if (func->isDeclaration()) {
            mi::mdl::IDefinition const *func_def = func_inst.get_def();
            mi::mdl::IDeclaration_function const *func_decl =
                cast<mi::mdl::IDeclaration_function>(func_def->get_declaration());

            if (func_decl != NULL) {
                MDL_module_scope scope(*this, owner);
                if (m_deriv_infos != NULL)
                    m_cur_func_deriv_info = m_deriv_infos->get_function_derivative_infos(func_inst);

                compile_function_instance(func_inst, func_decl);

                m_cur_func_deriv_info = NULL;
            }
        }
        m_functions_q.pop();
    }
}

namespace {

class RO_segment_builder {
public:
    typedef unsigned char Byte;

    /// Constructor.
    ///
    /// \param code_gen  the current code generator
    /// \param segment   the allocated segment
    /// \param size      the  size of the segment
    RO_segment_builder(
        LLVM_code_generator    &code_gen,
        Byte                   *segment,
        size_t                 size)
    : m_code_gen(code_gen)
    , m_dl(code_gen.get_target_layout_data())
    , m_segment(segment)
    , m_end(segment + size)
    , m_next(segment)
    {
    }

    /// Destructor.
    ~RO_segment_builder() {
        MDL_ASSERT(m_next == m_end && "RO segment size calculated wrong");
    }

    /// Add the given value to the segment.
    void add(IValue const *value)
    {
        // Align to 16byte boundary for SSE.
        // This is necessary, because the next thing allocated may require it
        m_next = align_for_sse(m_next);

        add_value(value);
    }

private:
    /// Add a compound value.
    void add_compound(IValue_compound const *cv, size_t size)
    {
        Byte *p = m_next;
        for (int i = 0, n = cv->get_component_count(); i < n; ++i) {
            IValue const *v = cv->get_value(i);

            add_value(v);
        }
        // compound values might have an higher alignment then the sum of its components
        m_next = p + size;
    }

    /// Add a non-compound value.
    void add_atomic(IValue const *value, size_t size)
    {
        if (m_next + size <= m_end) {
            switch (value->get_kind()) {
            case IValue::VK_BAD:
                // should not happen
                MDL_ASSERT(!"<value bad> detected");
                break;
            case IValue::VK_BOOL:
                {
                    bool *p = reinterpret_cast<bool *>(m_next);
                    *p = cast<IValue_bool>(value)->get_value();
                }
                break;
            case IValue::VK_INT:
                {
                    int *p = reinterpret_cast<int *>(m_next);
                    *p = cast<IValue_int>(value)->get_value();
                }
                break;
            case IValue::VK_ENUM:
                {
                    int *p = reinterpret_cast<int *>(m_next);
                    *p = cast<IValue_enum>(value)->get_value();
                }
                break;
            case IValue::VK_FLOAT:
                {
                    float *p = reinterpret_cast<float *>(m_next);
                    *p = cast<IValue_float>(value)->get_value();
                }
                break;
            case IValue::VK_DOUBLE:
                {
                    double *p = reinterpret_cast<double *>(m_next);
                    *p = cast<IValue_double>(value)->get_value();
                }
                break;
            case IValue::VK_STRING:
                if (m_code_gen.get_type_mapper().strings_mapped_to_ids()) {
                    // retrieve the ID: it is potentially an error if no resource manager
                    // is available
                    IValue_string const *s = cast<IValue_string>(value);
                    IResource_manager *res_manag = m_code_gen.get_resource_manager();
                    Type_mapper::Tag ID = res_manag != NULL
                        ? res_manag->get_string_index(s) : 0u;
                    // and add it to the string table
                    m_code_gen.add_string_constant(s->get_value(), ID);
                    Type_mapper::Tag *p = reinterpret_cast<Type_mapper::Tag *>(m_next);
                    *p = ID;
                } else {
                    // not yet supported: need a more sophisticated memory management
                    MDL_ASSERT(!"string values are not supported");
                }
                break;
            case IValue::VK_VECTOR:
            case IValue::VK_MATRIX:
            case IValue::VK_ARRAY:
            case IValue::VK_RGB_COLOR:
            case IValue::VK_STRUCT:
                MDL_ASSERT(!"Unexpected compound value");
                break;

            case IValue::VK_INVALID_REF:
            case IValue::VK_TEXTURE:
            case IValue::VK_LIGHT_PROFILE:
            case IValue::VK_BSDF_MEASUREMENT:
                MDL_ASSERT(!"Unexpected resource value");
                break;
            }
            m_next += size;
        }
    }

    /// Add the given value to the segment.
    void add_value(IValue const *value)
    {
        IType const *t       = value->get_type();
        llvm::Type  *llvm_tp = m_code_gen.lookup_type(t);
        size_t size = size_t(m_dl->getTypeAllocSize(llvm_tp));

        if (IValue_compound const *cv = as<IValue_compound>(value)) {
            add_compound(cv, size);
        } else {
            add_atomic(value, size);
        }
    }

private:
    /// The code generator.
    LLVM_code_generator &m_code_gen;

    /// The data layout of the target.
    llvm::DataLayout const *m_dl;

    /// The segment start.
    Byte *m_segment;

    /// Pointer to the end.
    Byte *m_end;

    /// Pointer to the next data written.
    Byte *m_next;
};

} // anonymous

// Create the RO data segment.
void LLVM_code_generator::create_ro_segment()
{
    if (m_next_ro_data_offset == 0) {
        // empty
        return;
    }

    m_ro_segment =
        reinterpret_cast<unsigned char *>(get_allocator()->malloc(m_next_ro_data_offset));

    RO_segment_builder builder(
        *this, m_ro_segment, m_next_ro_data_offset);

    typedef Value_list::const_iterator Iter;
    for (Iter it(m_ro_data_values.begin()), end(m_ro_data_values.end()); it != end; ++it) {
        IValue const *v = *it;
        builder.add(v);
    }
}

// Create a new LLVM module.
void LLVM_code_generator::create_module(char const *mod_name, char const *mod_fname)
{
    MDL_ASSERT(m_module == NULL && !m_func_pass_manager && "current module not finished yet");

    // clear the render state usage
    m_state_usage_analysis.clear();

    // creates a new llvm module
    m_module = new llvm::Module(mod_name, m_llvm_context);
    m_module->setDataLayout(*get_target_layout_data());

    if (m_enable_full_debug || m_enable_type_debug) {
        m_di_builder = new llvm::DIBuilder(*m_module);

        // let the DIBuilder know that we're starting a new compilation unit
        IAllocator *alloc = get_allocator();
        if (mod_fname != NULL && mod_fname[0] == '\0')
            mod_fname = NULL;
        string filename(mod_fname == NULL ? "<unknown>" : mod_fname, alloc);
        string directory(alloc);

        size_t pos = filename.rfind('/');
        if (pos != string::npos) {
            directory = filename.substr(0, pos);
            filename  = filename.substr(pos + 1);
        } else {
            size_t pos = filename.rfind('\\');
            if (pos != string::npos) {
                directory = filename.substr(0, pos);
                filename  = filename.substr(pos + 1);
            }
        }

        m_di_file = m_di_builder->createFile(filename.c_str(), directory.c_str());
        MDL_ASSERT(m_di_file);

        m_di_builder->createCompileUnit(
            /*Lang=*/llvm::dwarf::DW_LANG_C99,
            m_di_file,
            "NVidia MDL compiler",
            /*isOptimized=*/m_opt_level > 0,
            /*Flags=*/"", // command line args
            /*RV=*/0      // run time version
            );
    }

    // initialize function pass manager to be used when a function is finalized
    m_func_pass_manager.reset(new llvm::legacy::FunctionPassManager(m_module));

    llvm::PassManagerBuilder builder;
    builder.OptLevel = m_opt_level;
    builder.AvoidPointerPHIs = m_target_lang == TL_HLSL;
    builder.populateFunctionPassManager(*m_func_pass_manager);
    m_func_pass_manager->doInitialization();
}

/// Load and link user-defined renderer module into the given LLVM module.
bool LLVM_code_generator::load_and_link_renderer_module(llvm::Module *llvm_module)
{
    std::unique_ptr<llvm::MemoryBuffer> mem(llvm::MemoryBuffer::getMemBuffer(
        llvm::StringRef(m_renderer_module.data, m_renderer_module.size),
        "renderer_module",
        /*RequiresNullTerminator=*/ false));

    llvm::SMDiagnostic err;
    auto renderer_mod = llvm::parseIR(*mem.get(), err, m_llvm_context);
    if (!renderer_mod) {
        error(PARSING_RENDERER_MODULE_FAILED, err.getMessage().str());
        return false;
    }

    // clear target triple to avoid LLVM warning on console about mixing different targets
    renderer_mod.get()->setTargetTriple("");

    // also avoid LLVM warning on console about mixing different data layouts
    renderer_mod.get()->setDataLayout(llvm_module->getDataLayout());

    // overwrite wchar_size flag to match gnu size defined in libbsdf module
    llvm::NamedMDNode *mod_flags = renderer_mod.get()->getModuleFlagsMetadata();
    if (mod_flags) {
        for (unsigned i = 0, n = mod_flags->getNumOperands(); i < n; ++i) {
            llvm::MDNode *flag = mod_flags->getOperand(i);
            llvm::MDString *id = llvm::cast<llvm::MDString>(flag->getOperand(1));
            if (id->getString() == "wchar_size") {
                llvm::Metadata *wchar_size = llvm::ConstantAsMetadata::get(
                    llvm::ConstantInt::get(llvm::IntegerType::get(m_llvm_context, 32), 4));
                llvm::Metadata *flag_ops[] = { flag->getOperand(0), id, wchar_size };
                llvm::MDNode *flag = llvm::MDNode::get(m_llvm_context, flag_ops);
                mod_flags->setOperand(i, flag);
            }
        }
    }

    // ensure that all renderer runtime functions will be inlined
    for (llvm::Function &func : renderer_mod.get()->functions()) {
        func.addFnAttr(llvm::Attribute::AlwaysInline);
    }

    if (llvm::Linker::linkModules(*llvm_module, std::move(renderer_mod))) {
        // true means linking has failed
        error(LINKING_RENDERER_MODULE_FAILED, "unknown linking error");
        return false;
    }

    return true;
}

// Finalize compilation of the current module.
llvm::Module *LLVM_code_generator::finalize_module()
{
    // note: these functions could introduce new resource table accesses
    compile_waiting_functions();

    m_state_usage_analysis.update_exported_functions_state_usage();

    // adding constants might introduce new data tables
    if (m_use_ro_data_segment)
        create_ro_segment();

    // create the resource tables if they were accessed
    create_texture_attribute_table();
    create_light_profile_attribute_table();
    create_bsdf_measurement_attribute_table();
    create_string_table();
    replace_bsdf_data_calls();

    llvm::Module *llvm_module = m_module;
    m_module = NULL;
    if (m_di_builder != NULL) {
        // add all references debug descriptors
        m_di_builder->finalize();

        delete m_di_builder;
        m_di_builder = NULL;
    }

    m_func_pass_manager->doFinalization();

    // avoid optimizing unused functions
    for (llvm::Function *func : m_libbsdf_template_funcs) {
        // the "gen_black_bsdf/edf" functions are not cloned, but used directly -> don't remove
        if (func->getName().startswith("gen_black_"))
            continue;

        func->eraseFromParent();
    }


    string errorInfo(get_allocator());
    raw_string_ostream os(errorInfo);
    if (llvm::verifyModule(*llvm_module, &os)) {
        // true means: verification failed
        error(COMPILING_LLVM_CODE_FAILED, os.str().c_str());
        MDL_ASSERT(!"Compiling llvm code failed");

        // drop the module and give up
        drop_llvm_module(llvm_module);
        return NULL;
    } else {
        if (m_link_libmdlrt) {
            if (!load_and_link_libmdlrt(llvm_module)) {
                drop_llvm_module(llvm_module);
                return NULL;
            }
        }
        if (m_renderer_module.data != NULL) {
            if (!load_and_link_renderer_module(llvm_module)) {
                drop_llvm_module(llvm_module);
                return NULL;
            }
        }
        if (m_link_libdevice) {
            std::unique_ptr<llvm::Module> libdevice(
                load_libdevice(m_llvm_context, m_min_ptx_version));
            MDL_ASSERT(libdevice);

            // avoid LLVM warning on console about mixing different data layouts
            libdevice->setDataLayout(llvm_module->getDataLayout());

            // set required attributes on libdevice functions
            for (llvm::Function &func : libdevice->functions()) {
                set_llvm_function_attributes(&func);
            }

            if (llvm::Linker::linkModules(*llvm_module, std::move(libdevice))) {
                // true means linking has failed
                error(LINKING_LIBDEVICE_FAILED, "unknown linking error");
                MDL_ASSERT(!"Linking libdevice failed");

                // drop the module and give up
                drop_llvm_module(llvm_module);
                return NULL;
            }
        }

        if (m_visible_functions != NULL && *m_visible_functions) {
            // first mark all non-external functions as internal
            for (llvm::Function &func : llvm_module->functions()) {
                if (!func.isDeclaration())
                    func.setLinkage(llvm::GlobalValue::InternalLinkage);
            }

            // now mark requested functions as external
            char const *start = m_visible_functions;
            while (start && *start) {
                char const *ptr = strchr(start, ',');
                if (ptr == nullptr)
                    ptr = start + strlen(start);

                llvm::Function *func = llvm_module->getFunction(
                    llvm::StringRef(start, ptr - start));
                if (func)
                    func->setLinkage(llvm::GlobalValue::ExternalLinkage);

                start = ptr;
                if (*ptr == ',')
                    ++start;
            }
        }

#if 0
        static int fileid = 0;
        {
            std::string filename("prog_" + std::to_string(fileid) + "-preopt.ll");
            std::error_code ec;
            llvm::raw_fd_ostream file(filename.c_str(), ec, llvm::sys::fs::F_Text);
            llvm_module->print(file, NULL);
        }
#endif

        optimize(llvm_module);

#if 0
        {
            std::string filename("prog_" + std::to_string(fileid) + "-postopt.ll");
            std::error_code ec;
            llvm::raw_fd_ostream file(filename.c_str(), ec, llvm::sys::fs::F_Text);
            llvm_module->print(file, NULL);
        }

        ++fileid;
#endif
    }

    return llvm_module;
}

// JIT compile all functions of the given module.
MDL_JIT_module_key LLVM_code_generator::jit_compile(llvm::Module *module)
{
    // check that all functions exists
    for (auto &func : module->functions()) {
        if (func.isDeclaration() && !func.isIntrinsic()) {
            MDL_ASSERT(func.getLinkage() == llvm::GlobalValue::ExternalLinkage);
        }
    }

    // the jitted code must take ownership of this module
    MDL_JIT_module_key module_key = m_jitted_code->add_llvm_module(module);

    // now JIT compile all functions that are not jitted yet:
    // we want to do this ahead of time
    for (auto &func : module->functions()) {
        if (!func.isDeclaration()) {
            // jit it
            m_jitted_code->jit_compile(module_key, &func, *this);
        }
    }

    return module_key;
}

// Compile the given module into PTX code.
void LLVM_code_generator::ptx_compile(
    llvm::Module *module,
    string       &code)
{
    char mcpu[16];
    char features[16];
    {
    raw_string_ostream SOut(code);
    llvm::buffer_ostream Out(SOut);

    bool       is64bit   = get_target_layout_data()->getPointerSizeInBits() == 64;
    char const *march    = is64bit ? "nvptx64" : "nvptx";
    std::string triple = llvm::Triple(march, "nvidia", "cuda").str();

    // LLVM supports only "known" processors, so ensure that we do not pass an unsupported one
    unsigned sm_version = m_sm_version;
    if (sm_version == 86)
        /* ok */;
    else if (sm_version > 80)  sm_version = 80;
    else if (sm_version == 75)
        /* ok */;
    else if (sm_version > 70)  sm_version = 70;
    else if (sm_version == 70)
        /* ok*/;
    else if (sm_version > 62)  sm_version = 60;
    else if (sm_version >= 60)
        /* ok*/;
    else if (sm_version >= 53 || sm_version == 51) sm_version = 50;
    else if (sm_version >= 50)
        /*ok*/;
    else if (sm_version >= 37) sm_version = 30;
    else if (sm_version == 31 || sm_version == 33 || sm_version == 34 || sm_version == 36)
        sm_version = 30;
    else if (sm_version >= 30)
        /*ok*/;
    else if (sm_version >= 21) sm_version = 21;
    else                       sm_version = 20;

    snprintf(mcpu, sizeof(mcpu), "sm_%u", sm_version);

    features[0] = '\0';
    if (m_min_ptx_version != 0) {
        snprintf(features, sizeof(features), "+ptx%u", m_min_ptx_version);
    }

    std::string error;
    llvm::Target const *target = llvm::TargetRegistry::lookupTarget(march, error);
    MDL_ASSERT(target != NULL);  // backend not found, should not happen

    llvm::CodeGenOpt::Level OLvl = llvm::CodeGenOpt::None;
    if (m_opt_level == 1)
        OLvl = llvm::CodeGenOpt::Default;
    else if (m_opt_level >= 2)
        OLvl = llvm::CodeGenOpt::Aggressive;

    llvm::TargetOptions options;
    if (m_fast_math)
        options.UnsafeFPMath = true;
    if (m_finite_math)
        options.NoInfsFPMath = options.NoNaNsFPMath = true;
    std::unique_ptr<llvm::TargetMachine> target_machine(target->createTargetMachine(
        triple, mcpu, features, options,
        llvm::None, llvm::None, OLvl));
    llvm::legacy::PassManager pm;

    // set the data layout
    module->setDataLayout(target_machine->createDataLayout());

    target_machine->addPassesToEmitFile(pm, Out, nullptr, llvm::TargetMachine::CGFT_AssemblyFile);

    pm.run(*module);
    }

#if 0 // dump generated PTX to file
    FILE *f = fopen("MDL.ptx","wb");
    fwrite((void*)code.c_str(),1,code.size(),f);
    fclose(f);
#endif

    // create prototypes for PTX and CUDA
    for (Exported_function &exp_func : m_exported_func_list) {
        // PTX prototype
        string p(".extern .func " + exp_func.name);
        if (exp_func.function_kind == IGenerated_code_executable::FK_DF_INIT)
            p += "(.param .b64 a, .param .b64 b, .param .b64 c, .param .b64 d);";
        else if (exp_func.function_kind == IGenerated_code_executable::FK_SWITCH_LAMBDA)
            p += "(.param .b64 a, .param .b64 b, .param .b64 c, .param .b64 d, .param .b64 e, "
                ".param .b64 f);";
        else
            p += "(.param .b64 a, .param .b64 b, .param .b64 c, .param .b64 d, .param .b64 e);";
        exp_func.set_function_prototype(IGenerated_code_executable::PL_PTX, p.c_str());

        // CUDA prototype
        p = "extern " + exp_func.name;
        if (exp_func.function_kind == IGenerated_code_executable::FK_DF_INIT)
            p += "(void *, void *, void *, void *);";
        else if (exp_func.function_kind == IGenerated_code_executable::FK_SWITCH_LAMBDA)
            p += "(void *, void *, void *, void *, void *, int);";
        else
            p += "(void *, void *, void *, void *, void *);";

        exp_func.set_function_prototype(IGenerated_code_executable::PL_CUDA, p.c_str());
    }
}

// Compile the given module into LLVM-IR code.
void LLVM_code_generator::llvm_ir_compile(llvm::Module *module, string &code)
{
    raw_string_ostream SOut(code);

    // just print it
    module->print(SOut, NULL);
}

// Compile the given module into LLVM-BC code.
void LLVM_code_generator::llvm_bc_compile(llvm::Module *module, string &code)
{
    raw_string_ostream Out(code);
    llvm::WriteBitcodeToFile(*module, Out);
}

// Fill the function information in the given code object with the info about the generated
// exported functions.
void LLVM_code_generator::fill_function_info(IGenerated_code_executable *code)
{
    for (Exported_function &exp_func : m_exported_func_list) {
        size_t index = code->add_function_info(
            exp_func.name.c_str(),
            exp_func.distribution_kind,
            exp_func.function_kind,
            exp_func.arg_block_index,
            exp_func.state_usage);

        for (size_t i = 0, n = exp_func.prototypes.size(); i < n; ++i) {
            if (!exp_func.prototypes[i].empty()) {
                code->set_function_prototype(
                    index,
                    IGenerated_code_executable::Prototype_language(i),
                    exp_func.prototypes[i].c_str());
            }
        }

        for (size_t i = 0, n = exp_func.df_handles.size(); i < n; ++i) {
            code->add_function_df_handle(index, exp_func.df_handles[i].c_str());
        }
    }
}

// Set the world-to-object transformation matrix.
void LLVM_code_generator::set_transforms(
    mi::mdl::IValue_matrix const *w2o,
    mi::mdl::IValue_matrix const *o2w)
{
    // Note: the w2o matrix must be converted into row-major as expected by the generated code
    {
        IValue_vector const *col0 = mi::mdl::cast<IValue_vector>(w2o->get_value(0));
        IValue_vector const *col1 = mi::mdl::cast<IValue_vector>(w2o->get_value(1));
        IValue_vector const *col2 = mi::mdl::cast<IValue_vector>(w2o->get_value(2));
        IValue_vector const *col3 = mi::mdl::cast<IValue_vector>(w2o->get_value(3));

        for (int i = 0; i < 4; ++i) {
            Float4_struct f4;
            f4.x = mi::mdl::cast<IValue_float>(col0->get_value(i))->get_value();
            f4.y = mi::mdl::cast<IValue_float>(col1->get_value(i))->get_value();
            f4.z = mi::mdl::cast<IValue_float>(col2->get_value(i))->get_value();
            f4.w = mi::mdl::cast<IValue_float>(col3->get_value(i))->get_value();

            m_world_to_object_store[i] = f4;
        }
    }

    // Note: the w2o matrix must be converted into row-major as expected by the generated code
    {
        IValue_vector const *col0 = mi::mdl::cast<IValue_vector>(o2w->get_value(0));
        IValue_vector const *col1 = mi::mdl::cast<IValue_vector>(o2w->get_value(1));
        IValue_vector const *col2 = mi::mdl::cast<IValue_vector>(o2w->get_value(2));
        IValue_vector const *col3 = mi::mdl::cast<IValue_vector>(o2w->get_value(3));

        for (int i = 0; i < 4; ++i) {
            Float4_struct f4;
            f4.x = mi::mdl::cast<IValue_float>(col0->get_value(i))->get_value();
            f4.y = mi::mdl::cast<IValue_float>(col1->get_value(i))->get_value();
            f4.z = mi::mdl::cast<IValue_float>(col2->get_value(i))->get_value();
            f4.w = mi::mdl::cast<IValue_float>(col3->get_value(i))->get_value();

            m_object_to_world_store[i] = f4;
        }
    }

    // Do NOT set the m_world_to_object/m_object_to_world here. Setting it to non-NULL would let
    // the code-generator to inline all transform calls. This works only, if we have
    // one matrix (as we have for uniform expressions). For varying expressions, the matrix
    // must be passed as an extra parameter.
    m_world_to_object = NULL;
    m_object_to_world = NULL;
}

// Get the LLVM value of the current world-to-object matrix.
llvm::Value *LLVM_code_generator::get_w2o_transform_value(Function_context &ctx)
{
    if (state_include_uniform_state()) {
        if (m_user_state_module.data != NULL) {
            error(INTERNAL_JIT_BACKEND_ERROR,
                "get_w2o_transform_value() should not be called with a user state module");
        }

        // get it from the state
        llvm::Value *state = ctx.get_state_parameter();
        llvm::Value *adr   = ctx.create_simple_gep_in_bounds(
            state, ctx.get_constant(
            m_type_mapper.get_state_index(Type_mapper::STATE_CORE_W2O_TRANSFORM)));
        return ctx->CreateLoad(adr);
    }

    llvm::Constant *elems[4];

    llvm::ArrayType *f4_type     = m_type_mapper.get_arr_float_4_type();
    llvm::ArrayType *f4_arr_type = llvm::ArrayType::get(f4_type, 4);

    for (size_t i = 0; i < 4; ++i) {
        Float4_struct const *f = &m_world_to_object_store[i];
        llvm::Constant *members[] = {
            ctx.get_constant(f->x),
            ctx.get_constant(f->y),
            ctx.get_constant(f->z),
            ctx.get_constant(f->w)
        };
        elems[i] = llvm::ConstantArray::get(f4_type, members);
    }

    llvm::Constant  *init = llvm::ConstantArray::get(f4_arr_type, elems);
    llvm::Value *cv = new llvm::GlobalVariable(
        *m_module,
        f4_arr_type,
        /*isConstant=*/true,
        llvm::GlobalValue::InternalLinkage,
        llvm::cast<llvm::Constant>(init),
        "w2o_matrix");

    // cast it "C-like" to a float4 pointer
    return ctx->CreatePointerCast(cv, m_type_mapper.get_arr_float_4_ptr_type());
}

// Get the LLVM value of the current object-to-world matrix.
llvm::Value *LLVM_code_generator::get_o2w_transform_value(Function_context &ctx)
{
    if (state_include_uniform_state()) {
        if (m_user_state_module.data != NULL) {
            error(INTERNAL_JIT_BACKEND_ERROR,
                "get_o2w_transform_value() should not be called with a user state module");
        }

        // get it from the state
        llvm::Value *state = ctx.get_state_parameter();
        llvm::Value *adr   = ctx.create_simple_gep_in_bounds(
            state, ctx.get_constant(
            m_type_mapper.get_state_index(Type_mapper::STATE_CORE_O2W_TRANSFORM)));
        return ctx->CreateLoad(adr);
    }

    llvm::Constant *elems[4];

    llvm::ArrayType *f4_type     = m_type_mapper.get_arr_float_4_type();
    llvm::ArrayType *f4_arr_type = llvm::ArrayType::get(f4_type, 4);

    for (size_t i = 0; i < 4; ++i) {
        Float4_struct const *f = &m_object_to_world_store[i];
        llvm::Constant *members[] = {
            ctx.get_constant(f->x),
            ctx.get_constant(f->y),
            ctx.get_constant(f->z),
            ctx.get_constant(f->w)
        };
        elems[i] = llvm::ConstantArray::get(f4_type, members);
    }

    llvm::Constant  *init = llvm::ConstantArray::get(f4_arr_type, elems);
    llvm::Value *cv = new llvm::GlobalVariable(
        *m_module,
        f4_arr_type,
        /*isConstant=*/true,
        llvm::GlobalValue::InternalLinkage,
        llvm::cast<llvm::Constant>(init),
        "o2w_matrix");

    // cast it "C-like" to a float4 pointer
    return ctx->CreatePointerCast(cv, m_type_mapper.get_arr_float_4_ptr_type());
}

// Disable array instancing support.
void LLVM_code_generator::disable_function_instancing()
{
    MDL_ASSERT(m_target_lang == TL_NATIVE);
    // m_enable_instancing = false;
}

// Get the address of a JIT compiled LLVM function.
void *LLVM_code_generator::get_entry_point(MDL_JIT_module_key module_key, llvm::Function *func)
{
    return m_jitted_code->jit_compile(module_key, func, *this);
}

// Get the number of error messages.
size_t LLVM_code_generator::get_error_message_count()
{
    return m_messages.get_error_message_count();
}

// Add a compiler error message to the messages.
void LLVM_code_generator::error(int code, Error_params const &params)
{
    string msg(m_messages.format_msg(code, MESSAGE_CLASS, params));
    m_messages.add_error_message(code, MESSAGE_CLASS, 0, NULL, msg.c_str());
}

/// Add a compiler error message to the messages.
void LLVM_code_generator::error(int code, char const *str_param)
{
    error(code, Error_params(get_allocator()).add(str_param));
}

// Find the definition of a signature of a standard library function.
mi::mdl::IDefinition const *LLVM_code_generator::find_stdlib_signature(
    char const *module_name,
    char const *signature) const
{
    mi::mdl::Module const *mod = impl_cast<mi::mdl::Module>(tos_module());

    return mod->find_stdlib_signature(module_name, signature);
}

// Prepare a dummy resource attribute table only containing an invalid resource.
void LLVM_code_generator::init_dummy_attribute_table(Resource_table_kind kind)
{
    if (m_lut_info[kind].m_get_lut != NULL) {
        MDL_ASSERT(!"attribute table already initialized");
        return;
    }

    IAllocator *alloc = m_arena.get_allocator();
    switch (kind) {
    case RTK_TEXTURE:
    {
        vector<Texture_attribute_entry>::Type tex_entries(alloc);
        tex_entries.push_back(Texture_attribute_entry());
        add_texture_attribute_table(tex_entries);
        return;
    }
    case RTK_LIGHT_PROFILE:
    {
        vector<Light_profile_attribute_entry>::Type lp_entries(alloc);
        lp_entries.push_back(Light_profile_attribute_entry());
        add_light_profile_attribute_table(lp_entries);
        return;
    }
    case RTK_BSDF_MEASUREMENT:
    {
        vector<Bsdf_measurement_attribute_entry>::Type bm_entries(alloc);
        bm_entries.push_back(Bsdf_measurement_attribute_entry());
        add_bsdf_measurement_attribute_table(bm_entries);
        return;
    }
    case RTK_STRINGS:
        init_string_attribute_table();
        return;
    }
    MDL_ASSERT(!"Invalid resource table kind");
}

// Get a attribute lookup table.
llvm::Value *LLVM_code_generator::get_attribute_table(
    Function_context    &ctx,
    Resource_table_kind kind)
{
    if (m_lut_info[kind].m_get_lut == NULL)
        init_dummy_attribute_table(kind);

    return ctx->CreateCall(m_lut_info[kind].m_get_lut);
}

// Get a attribute lookup table size.
llvm::Value *LLVM_code_generator::get_attribute_table_size(
    Function_context    &ctx,
    Resource_table_kind kind)
{
    if (m_lut_info[kind].m_get_lut_size == NULL)
        init_dummy_attribute_table(kind);

    return ctx->CreateCall(m_lut_info[kind].m_get_lut_size);
}

// Add a texture attribute table.
void LLVM_code_generator::add_texture_attribute_table(
    Texture_table const &table)
{
    size_t n = table.size();
    if (n == 0)
        return;

    // ensure there is enough space in the texture table
    if (m_texture_table.size() < table.size())
        m_texture_table.resize(table.size());

    // update the texture table with any valid entries
    for (size_t i = 0, n = table.size(); i < n; ++i) {
        if (table[i].valid) {
            m_texture_table[i] = table[i];
        }
    }

    if (m_lut_info[RTK_TEXTURE].m_get_lut == NULL) {
        // create the lookup function prototype
        llvm::PointerType *tae_type = m_type_mapper.get_texture_attribute_entry_ptr_type();

        llvm::Function *lut_func = llvm::Function::Create(
            llvm::FunctionType::get(tae_type, /*isVarArg=*/false),
            llvm::GlobalValue::InternalLinkage,
            "get_texture_attr_table",
            m_module);
        set_llvm_function_attributes(lut_func);

        m_lut_info[RTK_TEXTURE].m_get_lut = lut_func;

        llvm::Function *size_func = llvm::Function::Create(
            llvm::FunctionType::get(m_type_mapper.get_int_type(), /*isVarArg=*/false),
            llvm::GlobalValue::InternalLinkage,
            "get_texture_attr_table_size",
            m_module);
        set_llvm_function_attributes(size_func);

        m_lut_info[RTK_TEXTURE].m_get_lut_size = size_func;
    }
}

// Creates the light profile attribute table.
void LLVM_code_generator::add_light_profile_attribute_table(
    Light_profile_table const &table)
{
    size_t n = table.size();
    if (n == 0)
        return;

    // ensure there is enough space in the light profile table
    if (m_light_profile_table.size() < table.size())
        m_light_profile_table.resize(table.size());

    // update the light profile table with any valid entries
    for (size_t i = 0, n = table.size(); i < n; ++i) {
        if (table[i].valid) {
            m_light_profile_table[i] = table[i];
        }
    }

    if (m_lut_info[RTK_LIGHT_PROFILE].m_get_lut == NULL) {
        // create the lookup function prototype
        llvm::PointerType *lpae_type = m_type_mapper.get_light_profile_attribute_entry_ptr_type();

        llvm::Function *lut_func = llvm::Function::Create(
            llvm::FunctionType::get(lpae_type, /*isVarArg=*/false),
            llvm::GlobalValue::InternalLinkage,
            "get_light_profile_attr_table",
            m_module);
        set_llvm_function_attributes(lut_func);

        m_lut_info[RTK_LIGHT_PROFILE].m_get_lut = lut_func;

        llvm::Function *size_func = llvm::Function::Create(
            llvm::FunctionType::get(m_type_mapper.get_int_type(), /*isVarArg=*/false),
            llvm::GlobalValue::InternalLinkage,
            "get_light_profile_attr_table_size",
            m_module);
        set_llvm_function_attributes(size_func);

        m_lut_info[RTK_LIGHT_PROFILE].m_get_lut_size = size_func;
    }
}

// Creates the bsdf measurement attribute table.
void LLVM_code_generator::add_bsdf_measurement_attribute_table(
    Bsdf_measurement_table const &table)
{
    size_t n = table.size();
    if (n == 0)
        return;

    // ensure there is enough space in the bsdf measurement table
    if (m_bsdf_measurement_table.size() < table.size())
        m_bsdf_measurement_table.resize(table.size());

    // update the bsdf measurement table with any valid entries
    for (size_t i = 0, n = table.size(); i < n; ++i) {
        if (table[i].valid) {
            m_bsdf_measurement_table[i] = table[i];
        }
    }

    if (m_lut_info[RTK_BSDF_MEASUREMENT].m_get_lut == NULL) {
        // create the lookup function prototype
        llvm::PointerType *bmae_type =
            m_type_mapper.get_bsdf_measurement_attribute_entry_ptr_type();

        llvm::Function *lut_func = llvm::Function::Create(
            llvm::FunctionType::get(bmae_type, /*isVarArg=*/false),
            llvm::GlobalValue::InternalLinkage,
            "get_bsdf_measurement_attr_table",
            m_module);
        set_llvm_function_attributes(lut_func);

        m_lut_info[RTK_BSDF_MEASUREMENT].m_get_lut = lut_func;

        llvm::Function *size_func = llvm::Function::Create(
            llvm::FunctionType::get(m_type_mapper.get_int_type(), /*isVarArg=*/false),
            llvm::GlobalValue::InternalLinkage,
            "get_bsdf_measurement_attr_table_size",
            m_module);
        set_llvm_function_attributes(size_func);

        m_lut_info[RTK_BSDF_MEASUREMENT].m_get_lut_size = size_func;
    }
}

// Initialize the string attribute table, adding the unknown string and declaring
// the access functions.
void LLVM_code_generator::init_string_attribute_table()
{
    if (m_lut_info[RTK_STRINGS].m_get_lut != NULL) {
        MDL_ASSERT(!"string attribute table already initialized");
        return;
    }

    // first entering, 0 is reserved for "unknown string"
    m_string_table.push_back(mi::mdl::string("<NULL>", get_allocator()));

    // create the lookup function prototype
    llvm::PointerType *cstring_type = m_type_mapper.get_cstring_type();
    llvm::PointerType *ste_ptr      = m_type_mapper.get_ptr(cstring_type);

    llvm::Function *lut_func = llvm::Function::Create(
        llvm::FunctionType::get(ste_ptr, /*isVarArg=*/false),
        llvm::GlobalValue::InternalLinkage,
        "get_string_table",
        m_module);
    set_llvm_function_attributes(lut_func);

    m_lut_info[RTK_STRINGS].m_get_lut = lut_func;

    llvm::Function *size_func = llvm::Function::Create(
        llvm::FunctionType::get(m_type_mapper.get_int_type(), /*isVarArg=*/false),
        llvm::GlobalValue::InternalLinkage,
        "get_string_table_size",
        m_module);
    set_llvm_function_attributes(size_func);

    m_lut_info[RTK_STRINGS].m_get_lut_size = size_func;
}

// Add a string constant to the string table.
void LLVM_code_generator::add_string_constant(
    char const       *s,
    Type_mapper::Tag id)
{
    if (m_lut_info[RTK_STRINGS].m_get_lut == NULL)
        init_string_attribute_table();

    if (m_string_table.empty()) {
        // add the "not-a-known-String" entry
        m_string_table.push_back(mi::mdl::string("<NULL>", get_allocator()));
    }

    if (id >= m_string_table.size()) {
        m_string_table.reserve((id + 15) & ~15);
        m_string_table.resize(id + 1, mi::mdl::string("", get_allocator()));
    }

    m_string_table[id] = mi::mdl::string(s, get_allocator());
}

// Get string constant for a tag.
char const *LLVM_code_generator::get_string_constant(size_t id) const
{
    if (id < m_string_table.size())
        return m_string_table[id].c_str();
    return NULL;
}

// Creates the texture attribute table.
void LLVM_code_generator::create_texture_attribute_table()
{
    if (m_lut_info[RTK_TEXTURE].m_get_lut == NULL) {
        // no access was recorded
        return;
    }

    llvm::StructType *tae_type = m_type_mapper.get_texture_attribute_entry_type();
    size_t            n        = m_texture_table.size();

    vector<llvm::Constant *>::Type elems(n, NULL, get_allocator());

    llvm::Type *bool_type = m_type_mapper.get_bool_type();
    llvm::Type *int_type  = m_type_mapper.get_int_type();
    for (size_t i = 0; i < n; ++i) {
        Texture_attribute_entry const &e = m_texture_table[i];
        llvm::Constant *s_members[] = {
            llvm::ConstantInt::get(bool_type, e.valid),
            llvm::ConstantInt::get(int_type, e.width),
            llvm::ConstantInt::get(int_type, e.height),
            llvm::ConstantInt::get(int_type, e.depth)
        };
        elems[i] = llvm::ConstantStruct::get(tae_type, s_members);
    }

    llvm::ArrayType *a_tp = llvm::ArrayType::get(tae_type, n);
    llvm::Constant  *init = llvm::ConstantArray::get(a_tp, elems);

    llvm::Value *cv = new llvm::GlobalVariable(
        *m_module,
        a_tp,
        /*isConstant=*/true,
        llvm::GlobalValue::InternalLinkage,
        init,
        "texture_attr_table");

    // create get_lut
    {
        llvm::Function *func = m_lut_info[RTK_TEXTURE].m_get_lut;
        func->addFnAttr(llvm::Attribute::AlwaysInline);

        llvm::BasicBlock *bb = llvm::BasicBlock::Create(get_llvm_context(), "start", func);
        llvm::IRBuilder<> builder(bb);
        llvm::PointerType *pt = m_type_mapper.get_texture_attribute_entry_ptr_type();
        builder.CreateRet(builder.CreatePointerCast(cv, pt));
    }

    // create get_lut_size
    {
        llvm::Function *func = m_lut_info[RTK_TEXTURE].m_get_lut_size;
        func->addFnAttr(llvm::Attribute::AlwaysInline);

        llvm::BasicBlock *bb = llvm::BasicBlock::Create(get_llvm_context(), "start", func);
        llvm::IRBuilder<> builder(bb);
        builder.CreateRet(llvm::ConstantInt::get(m_type_mapper.get_int_type(), int(n)));
    }
}

// Creates the light profile attribute table.
void LLVM_code_generator::create_light_profile_attribute_table()
{
    if (m_lut_info[RTK_LIGHT_PROFILE].m_get_lut == NULL) {
        // no access was recorded
        return;
    }

    llvm::StructType *lpae_type = m_type_mapper.get_light_profile_attribute_entry_type();
    size_t            n         = m_light_profile_table.size();

    vector<llvm::Constant *>::Type elems(n, NULL, get_allocator());

    llvm::Type *bool_type  = m_type_mapper.get_bool_type();
    llvm::Type *float_type = m_type_mapper.get_float_type();
    for (size_t i = 0; i < n; ++i) {
        Light_profile_attribute_entry const &e = m_light_profile_table[i];
        llvm::Constant *s_members[] = {
            llvm::ConstantInt::get(bool_type, e.valid),
            llvm::ConstantFP::get(float_type, e.power),
            llvm::ConstantFP::get(float_type, e.maximum),
        };
        elems[i] = llvm::ConstantStruct::get(lpae_type, s_members);
    }

    llvm::ArrayType *a_tp = llvm::ArrayType::get(lpae_type, n);
    llvm::Constant  *init = llvm::ConstantArray::get(a_tp, elems);

    llvm::Value *cv = new llvm::GlobalVariable(
        *m_module,
        a_tp,
        /*isConstant=*/true,
        llvm::GlobalValue::InternalLinkage,
        init,
        "light_profile_attr_table");

    // create get_lut
    {
        llvm::Function *func = m_lut_info[RTK_LIGHT_PROFILE].m_get_lut;
        func->addFnAttr(llvm::Attribute::AlwaysInline);

        llvm::BasicBlock *bb = llvm::BasicBlock::Create(get_llvm_context(), "start", func);
        llvm::IRBuilder<> builder(bb);
        llvm::PointerType *pt = m_type_mapper.get_light_profile_attribute_entry_ptr_type();
        builder.CreateRet(builder.CreatePointerCast(cv, pt));
    }

    // create get_lut_size
    {
        llvm::Function *func = m_lut_info[RTK_LIGHT_PROFILE].m_get_lut_size;
        func->addFnAttr(llvm::Attribute::AlwaysInline);

        llvm::BasicBlock *bb = llvm::BasicBlock::Create(get_llvm_context(), "start", func);
        llvm::IRBuilder<> builder(bb);
        builder.CreateRet(llvm::ConstantInt::get(m_type_mapper.get_int_type(), int(n)));
    }
}

// Creates the bsdf measurement attribute table.
void LLVM_code_generator::create_bsdf_measurement_attribute_table()
{
    if (m_lut_info[RTK_BSDF_MEASUREMENT].m_get_lut == NULL) {
        // no access was recorded
        return;
    }

    llvm::StructType *bmae_type = m_type_mapper.get_bsdf_measurement_attribute_entry_type();
    size_t            n         = m_bsdf_measurement_table.size();

    vector<llvm::Constant *>::Type elems(n, NULL, get_allocator());

    llvm::Type *bool_type  = m_type_mapper.get_bool_type();
    for (size_t i = 0; i < n; ++i) {
        Bsdf_measurement_attribute_entry const &e = m_bsdf_measurement_table[i];
        llvm::Constant *s_members[] = {
            llvm::ConstantInt::get(bool_type, e.valid)
        };
        elems[i] = llvm::ConstantStruct::get(bmae_type, s_members);
    }

    llvm::ArrayType *a_tp = llvm::ArrayType::get(bmae_type, n);
    llvm::Constant  *init = llvm::ConstantArray::get(a_tp, elems);

    llvm::Value *cv = new llvm::GlobalVariable(
        *m_module,
        a_tp,
        /*isConstant=*/true,
        llvm::GlobalValue::InternalLinkage,
        init,
        "bsdf_measurement_attr_table");

    // create get_lut
    {
        llvm::Function *func = m_lut_info[RTK_BSDF_MEASUREMENT].m_get_lut;
        func->addFnAttr(llvm::Attribute::AlwaysInline);

        llvm::BasicBlock *bb = llvm::BasicBlock::Create(get_llvm_context(), "start", func);
        llvm::IRBuilder<> builder(bb);
        llvm::PointerType *pt = m_type_mapper.get_bsdf_measurement_attribute_entry_ptr_type();
        builder.CreateRet(builder.CreatePointerCast(cv, pt));
    }

    // create get_lut_size
    {
        llvm::Function *func = m_lut_info[RTK_BSDF_MEASUREMENT].m_get_lut_size;
        func->addFnAttr(llvm::Attribute::AlwaysInline);

        llvm::BasicBlock *bb = llvm::BasicBlock::Create(get_llvm_context(), "start", func);
        llvm::IRBuilder<> builder(bb);
        builder.CreateRet(llvm::ConstantInt::get(m_type_mapper.get_int_type(), int(n)));
    }
}

// Creates the string table finally.
void LLVM_code_generator::create_string_table()
{
    if (m_lut_info[RTK_STRINGS].m_get_lut == NULL) {
        // no access was recorded
        return;
    }

    llvm::Type *char_type   = m_type_mapper.get_char_type();
    llvm::Type *ctring_type = m_type_mapper.get_cstring_type();
    size_t     n            = m_string_table.size();

    vector<llvm::Constant *>::Type elems(n, NULL, get_allocator());

    llvm::ConstantInt *zero = llvm::ConstantInt::get(m_type_mapper.get_int_type(), 0);
    llvm::Constant *idxes[] = { zero, zero };
    for (size_t i = 0; i < n; ++i) {
        string const &s = m_string_table[i];

        llvm::Constant *str_arr =
            llvm::ConstantDataArray::getString(get_llvm_context(), s.c_str(), /*AddNull=*/true);

        llvm::ArrayType *a_tp = llvm::ArrayType::get(char_type, s.size() + 1);

        llvm::GlobalVariable *str = new llvm::GlobalVariable(
            *m_module,
            a_tp,
            /*isConstant=*/true,
            llvm::GlobalValue::InternalLinkage,
            str_arr);

        elems[i] = llvm::ConstantExpr::getGetElementPtr(nullptr, str, idxes, /*InBounds=*/true);
    }

    llvm::ArrayType *a_tp = llvm::ArrayType::get(ctring_type, n);
    llvm::Constant  *init = llvm::ConstantArray::get(a_tp, elems);

    llvm::GlobalVariable *cv = new llvm::GlobalVariable(
        *m_module,
        a_tp,
        /*isConstant=*/true,
        llvm::GlobalValue::InternalLinkage,
        init,
        "string_table");

    // create get_lut
    {
        llvm::Function *func = m_lut_info[RTK_STRINGS].m_get_lut;
        func->addFnAttr(llvm::Attribute::AlwaysInline);

        llvm::BasicBlock *bb = llvm::BasicBlock::Create(get_llvm_context(), "start", func);
        llvm::IRBuilder<> builder(bb);
        llvm::PointerType *pt = m_type_mapper.get_ptr(m_type_mapper.get_cstring_type());
        builder.CreateRet(builder.CreatePointerCast(cv, pt));
    }

    // create get_lut_size
    {
        llvm::Function *func = m_lut_info[RTK_STRINGS].m_get_lut_size;
        func->addFnAttr(llvm::Attribute::AlwaysInline);

        llvm::BasicBlock *bb = llvm::BasicBlock::Create(get_llvm_context(), "start", func);
        llvm::IRBuilder<> builder(bb);
        builder.CreateRet(llvm::ConstantInt::get(m_type_mapper.get_int_type(), int(n)));
    }
}

// Replace all calls to state::get_bsdf_data_texture_id() by the registered texture IDs.
void LLVM_code_generator::replace_bsdf_data_calls()
{
    llvm::Function *func = m_module->getFunction(
        "_ZNK5State24get_bsdf_data_texture_idE14Bsdf_data_kind");
    if (func == nullptr)
        return;

    for (auto ui = func->user_begin(); ui != func->user_end(); ) {
        llvm::CallInst *call = llvm::dyn_cast<llvm::CallInst>(*ui);

        // current instruction might get replaced, so advance the iterator here
        ++ui;

        if (call != nullptr) {
            llvm::Value *kind_val = call->getArgOperand(1);
            if (llvm::ConstantInt *const_int = llvm::dyn_cast<llvm::ConstantInt>(kind_val)) {
                unsigned bsdf_data_kind = unsigned(const_int->getValue().getZExtValue());
                int tex_id = 0;
                if (bsdf_data_kind > 0 && bsdf_data_kind <= IValue_texture::BDK_LAST_KIND) {
                    tex_id = int(m_bsdf_data_texture_ids[bsdf_data_kind - 1]);
                }

                llvm::Value *res_val = llvm::ConstantInt::get(m_type_mapper.get_int_type(), tex_id);
                call->replaceAllUsesWith(res_val);
                call->eraseFromParent();
            } else {
                MDL_ASSERT(!"argument to State::get_bsdf_data_texture_id() must be a constant");
            }
        }
    }
}


// Reset the lambda function compilation state.
void LLVM_code_generator::reset_lambda_state()
{
    // sret is the default mode
    m_lambda_force_sret              = true;
    m_lambda_first_param_by_ref      = false;
    m_lambda_force_render_state      = false;
    m_lambda_force_no_lambda_results = false;
}

// Parse a call mode option.
Function_context::Tex_lookup_call_mode LLVM_code_generator::parse_call_mode(char const *name)
{
    if (strcmp(name, "vtable") == 0)
        return Function_context::TLCM_VTABLE;
    if (strcmp(name, "direct_call") == 0)
        return Function_context::TLCM_DIRECT;
    if (strcmp(name, "optix_cp") == 0)
        return Function_context::TLCM_OPTIX_CP;
    return Function_context::TLCM_VTABLE;
}

/// Parse the Df_handle_slot_mode
mi::mdl::Df_handle_slot_mode LLVM_code_generator::parse_df_handle_slot_mode(char const *name)
{
    if (strcmp(name, "none") == 0)
        return mi::mdl::DF_HSM_NONE;
    if (strcmp(name, "pointer") == 0)
        return mi::mdl::DF_HSM_POINTER;
    if (strcmp(name, "fixed_1") == 0)
        return mi::mdl::DF_HSM_FIXED_1;
    if (strcmp(name, "fixed_2") == 0)
        return mi::mdl::DF_HSM_FIXED_2;
    if (strcmp(name, "fixed_4") == 0)
        return mi::mdl::DF_HSM_FIXED_4;
    if (strcmp(name, "fixed_8") == 0)
        return mi::mdl::DF_HSM_FIXED_8;

    return mi::mdl::DF_HSM_NONE;
}

// Get a unique string value object used to represent the string of the value.
mi::mdl::IValue_string const *LLVM_code_generator::get_internalized_string(
    mi::mdl::IValue_string const *s)
{
    LLVM_code_generator::Internalized_string_map::iterator it =
        m_internalized_string_map.find(s->get_value());
    if (it != m_internalized_string_map.end())
        return it->second;

    // the given string value object will be our representative for the contained cstring
    m_internalized_string_map[s->get_value()] = s;
    return s;
}

} // mdl
} // mi

