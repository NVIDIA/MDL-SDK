/******************************************************************************
 * Copyright (c) 2013-2025, NVIDIA CORPORATION. All rights reserved.
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
#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
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
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/FormattedStream.h>
#include <llvm/Support/ManagedStatic.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/PrettyStackTrace.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Support/CommandLine.h>
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
#include "mdl/compiler/compilercore/compilercore_positions.h"
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
#include "generator_jit_sl_passes.h"

namespace mi {
namespace mdl {

// statics
Jitted_code *Jitted_code::m_instance = NULL;
mi::base::Lock Jitted_code::m_singleton_lock;

static Position_impl zero(0, 0, 0, 0);

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


/// LLVM JIT based on the BuildingAJIT tutorial.
/// Differences:
///  - no lazy emitting
///  - search for symbols in specific modules (avoid problems with duplicate names)
///  - special handling for COFF object formats
class MDL_JIT {
public:
    /// Constructor.
    MDL_JIT(llvm::orc::JITTargetMachineBuilder jtm_builder, llvm::DataLayout data_layout)
    : m_next_module_id(0)
    , m_uses_coff(jtm_builder.getTargetTriple().isOSBinFormatCOFF())
    , m_object_layer(m_execution_session,
        // GetMemoryManager
        [this]() { return std::make_unique<llvm::SectionMemoryManager>(&m_memory_mapper); })
    , m_compile_layer(
        m_execution_session,
        m_object_layer,
        std::make_unique<llvm::orc::ConcurrentIRCompiler>(std::move(jtm_builder)))
    , m_data_layout(std::move(data_layout))
    , m_mangler(m_execution_session, m_data_layout)
    , m_llvm_context(std::make_unique<llvm::LLVMContext>())
    , m_mdl_runtime_dylib(m_execution_session.createBareJITDylib("mdl_runtime"))
    {
        // Special handling for COFF object formats
        if (m_uses_coff) {
            // By default, the Exported flag is not set for symbols.
            // Ensure, the flags are overridden, if necessary
            m_object_layer.setOverrideObjectFlagsWithResponsibilityFlags(true);

            // Add support for COFF comdat constants
            m_object_layer.setAutoClaimResponsibilityForObjectSymbols(true);
        }
    }

    ~MDL_JIT() {
        llvm::cantFail(m_execution_session.endSession());
    }

    /// Get the data layout of the target machine.
    llvm::DataLayout const &get_data_layout() const { return m_data_layout; }

    /// Get the LLVM context.
    llvm::LLVMContext &getContext() { return *m_llvm_context.getContext(); }

    /// Add an LLVM module to the JIT and get its module key.
    MDL_JIT_module_key add_module(std::unique_ptr<llvm::Module> module) {
        MDL_ASSERT(!module->getDataLayout().isDefault() && "No data layout was set for module");

        // Add the module to the JIT with a new dylib.
        std::string module_name = std::to_string(m_next_module_id++);
        llvm::orc::JITDylibSP dylib = &m_execution_session.createBareJITDylib(module_name);
        dylib->addToLinkOrder(m_mdl_runtime_dylib);
        llvm::orc::ResourceTrackerSP rt = dylib->createResourceTracker();
        llvm::cantFail(m_compile_layer.add(
            rt, llvm::orc::ThreadSafeModule(std::move(module), m_llvm_context)));
        return rt;
    }

    /// Search for a symbol name in the given module.
    llvm::Expected<llvm::JITEvaluatedSymbol> find_symbol_in(
        MDL_JIT_module_key key,
        const llvm::Twine &name)
    {
        return m_execution_session.lookup({&key->getJITDylib()}, m_mangler(name.str()));
    }

    /// Get the address for a symbol name in the given module.
    llvm::JITTargetAddress get_symbol_address_in(
        MDL_JIT_module_key K,
        const llvm::Twine &name,
        LLVM_code_generator &code_gen)
    {
        llvm::Expected<llvm::JITEvaluatedSymbol> sym = find_symbol_in(K, name);
        if (auto error = sym.takeError()) {
            code_gen.error(GET_SYMBOL_FAILED, llvm::toString(std::move(error)));
            return llvm::JITTargetAddress(0);
        }
        return sym->getAddress();
    }

    /// Remove the given module.
    void remove_module(MDL_JIT_module_key key) {
        // TODO: The JITDylib objects are still stored in the execution session until the session
        //       is ended. This "leaks" at least 400 byte per JIT compilation.
        llvm::cantFail(key->remove());
    }

    void register_mdl_runtime_function(llvm::StringRef const &func_name, void *address)
    {
        llvm::orc::SymbolStringPtr sym_name(m_mangler(func_name.str()));
        llvm::JITEvaluatedSymbol sym(
            static_cast<llvm::JITTargetAddress>(reinterpret_cast<uintptr_t>(address)),
            llvm::JITSymbolFlags::Exported);
        llvm::cantFail(m_mdl_runtime_dylib.define(llvm::orc::absoluteSymbols({{sym_name, sym}})));
    }

private:
    /// Lock protecting all internal data structures.
    llvm::sys::Mutex m_lock;

    /// The ID of the next module to be added.
    std::atomic<uint64_t> m_next_module_id;

    /// True, if the binary object format is COFF.
    bool m_uses_coff;

    /// Execution session used to identify modules.
    llvm::orc::ExecutionSession m_execution_session;

    /// The object linking layer.
    llvm::orc::RTDyldObjectLinkingLayer m_object_layer;

    /// The compile layer.
    llvm::orc::IRCompileLayer m_compile_layer;

    /// The data layout of the target machine.
    const llvm::DataLayout m_data_layout;

    /// The symbol mangler.
    llvm::orc::MangleAndInterner m_mangler;

    /// Thread-safe LLVM context owning all compiled modules.
    llvm::orc::ThreadSafeContext m_llvm_context;

    /// Dylib receiving the runtime library functions.
    llvm::orc::JITDylib &m_mdl_runtime_dylib;

    /// The already compiled modules.
    std::map<MDL_JIT_module_key, std::unique_ptr<llvm::Module>> m_compiled_modules;

    // Trivial implementation of SectionMemoryManager::MemoryMapper that just calls
    // into sys::Memory. Copied from LLVM's SectionMemoryManager.cpp.
    // Needed to avoid use of global MemoryMapper which may be freed before the jitted code,
    // leading to use-after-free in SectionMemoryManager destructor.
    class DefaultMMapper final : public llvm::SectionMemoryManager::MemoryMapper {
    public:
        llvm::sys::MemoryBlock
            allocateMappedMemory(llvm::SectionMemoryManager::AllocationPurpose Purpose,
                size_t NumBytes, const llvm::sys::MemoryBlock *const NearBlock,
                unsigned Flags, std::error_code &EC) override {
            return llvm::sys::Memory::allocateMappedMemory(NumBytes, NearBlock, Flags, EC);
        }

        std::error_code protectMappedMemory(const llvm::sys::MemoryBlock &Block,
                unsigned Flags) override {
            return llvm::sys::Memory::protectMappedMemory(Block, Flags);
        }

        std::error_code releaseMappedMemory(llvm::sys::MemoryBlock &M) override {
            return llvm::sys::Memory::releaseMappedMemory(M);
        }
    };

    DefaultMMapper m_memory_mapper;
};


bool Jitted_code::m_first_time_init = true;

// Constructor.
Jitted_code::Jitted_code(
    mi::mdl::IAllocator *alloc,
    bool                enable_opt_remarks)
: Base(alloc)
, m_llvm_context(new llvm::LLVMContext())
, m_mdl_jit(NULL)
, m_enable_opt_remarks(enable_opt_remarks)
{
    std::unique_ptr<llvm::Module> module(new llvm::Module("MDL global", *m_llvm_context));

    // Set the default triple here: This is only necessary for MacOS where the triple
    // contains the lowest supported runtime version.
    module->setTargetTriple(LLVM_DEFAULT_TARGET_TRIPLE);

    // In 64-bit mode, the stack alignment is always 16 bytes
    llvm::orc::JITTargetMachineBuilder jtm_builder = llvm::cantFail(
        llvm::orc::JITTargetMachineBuilder::detectHost());
    jtm_builder.setCodeGenOptLevel(llvm::CodeGenOpt::Aggressive);

    llvm::DataLayout data_layout = llvm::cantFail(jtm_builder.getDefaultDataLayoutForTarget());

    m_mdl_jit = new MDL_JIT(std::move(jtm_builder), std::move(data_layout));

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
void Jitted_code::init_llvm(
    bool enable_opt_remarks)
{
    // Initialize targets first. Not strictly needed for the JIT itself,
    // but must be called once to the PTX backend.
    llvm::InitializeAllTargets();
    llvm::InitializeAllAsmPrinters();
    llvm::InitializeAllTargetMCs();

    // initialize our own passes
    llvm::initializeDeleteUnusedLibDevicePass(*llvm::PassRegistry::getPassRegistry());

    if (enable_opt_remarks) {
        char const *argv[] = {
            "mdl_sdk",
            "-pass-remarks=inline|df_instantiation|funccount|hlsl_writer",
    //        "-pass-remarks-output=opt_remarks.yaml"
        };

        llvm::cl::ParseCommandLineOptions(llvm::array_lengthof(argv), argv);
    }
    m_first_time_init = false;
}

// Get the only instance.
Jitted_code *Jitted_code::get_instance(
    IAllocator *alloc)
{
    mi::base::Lock::Block lock(&m_singleton_lock);

    if (m_instance != NULL) {
        m_instance->retain();
        return m_instance;
    }

    bool enable_opt_remarks = false;
    char const *s = getenv("MI_MDL_JIT_OPT_REMARKS");
    if (s != NULL) {
        enable_opt_remarks = true;
    }

    if (m_first_time_init) {
        init_llvm(enable_opt_remarks);
    }

    Allocator_builder builder(alloc);
    m_instance = builder.create<Jitted_code>(alloc, enable_opt_remarks);

    return m_instance;
}

/// Registers a native function for symbol resolving by the JIT.
void Jitted_code::register_function(llvm::StringRef const &func_name, void *address)
{
    m_mdl_jit->register_mdl_runtime_function(func_name, address);
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
    char const *func_name,
    LLVM_code_generator &code_gen)
{
    return (void *)(m_mdl_jit->get_symbol_address_in(module_key, func_name, code_gen));
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

    for (size_t i = 0, n = param_types.size(); i < n; ++i){
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
, m_pos(NULL)
{
    if (pos != NULL) {
        mi::mdl::Module const *mod = code_gen.tos_module();
        m_mod  = mod;
        m_pos  = pos;
    }
}

// ------------------------- Expression_result helper -------------------------

// Return the value.
llvm::Value *Expression_result::as_value(Function_context &context) {
    // turn offset result into value
    if (m_res_kind == RK_OFFSET) {
        int cur_offs = 0;
        m_content = context.get_code_gen().translate_sl_value(
            m_offs_kind == OK_RO_DATA_SEGMENT ? SM_RODATA : SM_PARAMETER,
            m_offset_res_mdl_type,
            cur_offs,
            m_content,
            /*force_as_value=*/ true).as_value(context);
        m_res_kind = RK_VALUE;
    }

    if (m_res_kind == RK_VALUE) {
        return m_content;
    } else {
        // do not add debug info here, it is not clear, when this is executed
        MDL_ASSERT(m_res_kind == RK_POINTER);
        return context->CreateLoad(m_content);
    }
}

// ------------------------------- ICall helper -------------------------------

/// Implements the ICall_expr interface for AST calls.
class Call_ast_expr : public ICall_expr {
public:
    /// Constructor.
    ///
    /// \param code_gen  the code generator
    /// \param call      the AST call
    Call_ast_expr(
        LLVM_code_generator             &code_gen,
        mi::mdl::IExpression_call const *call)
    : m_code_gen(code_gen)
    , m_call(call)
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
        if (ref->is_array_constructor()) {
            return mi::mdl::IDefinition::DS_INTRINSIC_DAG_ARRAY_CONSTRUCTOR;
        }

        mi::mdl::IDefinition const *def = ref->get_definition();
        return def->get_semantics();
    }

    /// Get the callee definition if one exists and its owner module.
    ///
    /// \param[out] owner_mod  out parameter for the owner module of the definition
    mi::mdl::IDefinition const *get_callee_definition_and_owner(
        mi::base::Handle<mi::mdl::Module const> &owner) const
    {
        mi::mdl::IExpression_reference const *ref =
            cast<mi::mdl::IExpression_reference>(m_call->get_reference());
        if (ref->is_array_constructor()) {
            return NULL;
        }
        mi::mdl::IDefinition const *ref_def = ref->get_definition();
        mi::mdl::Module const      *mod = m_code_gen.tos_module();
        mi::mdl::IDefinition const *def = mod->get_original_definition(ref_def);

        owner = mod->get_owner_module(ref_def);

        // a preset might point to another module
        def = skip_presets(def, owner);

        return def;
    }

    /// Get the callee definition of one exists.
    mi::mdl::IDefinition const *get_callee_definition() const MDL_FINAL
    {
        mi::base::Handle<mi::mdl::Module const> owner;
        return get_callee_definition_and_owner(owner);
    }

    /// Get the number of arguments.
    size_t get_argument_count() const MDL_FINAL { return m_call->get_argument_count(); }

    /// Translate the i'th argument.
    ///
    /// \param i             the argument index
    /// \param wants_derivs  if true, the result should have derivatives, if available
    Expression_result translate_argument(
        size_t              i,
        bool                wants_derivs) const MDL_FINAL
    {
        mi::mdl::IExpression const *arg = m_call->get_argument(int(i))->get_argument_expr();

        return m_code_gen.translate_expression(arg, wants_derivs);
    }

    /// Get the LLVM context data of the callee.
    ///
    /// \param inst           the function instance for this function call
    LLVM_context_data *get_callee_context(
        Function_instance const &inst) const MDL_FINAL
    {
        mi::base::Handle<mi::mdl::Module const> owner;
        get_callee_definition_and_owner(owner);

        return m_code_gen.get_or_create_context_data(owner.get(), inst);
    }

    /// Get the result type of the call.
    mi::mdl::IType const *get_type() const MDL_FINAL { return m_call->get_type(); }

    /// Get the type of the i'th call argument.
    ///
    /// \param i  the argument index
    mi::mdl::IType const *get_argument_type(size_t i) const MDL_FINAL {
        mi::mdl::IExpression const *arg = m_call->get_argument(int(i))->get_argument_expr();

        return arg->get_type();
    }

    /// Get the storage modifier of the i'th call argument.
    ///
    /// \param ctx  the current function context
    /// \param i  the argument index
    Storage_modifier get_argument_storage_modifier(
        size_t           i) const MDL_FINAL
    {
        if (!m_code_gen.target_supports_storage_spaces()) {
            return SM_NORMAL;
        }

        Function_context &ctx = *m_code_gen.m_ctx;

        mi::mdl::IExpression const *arg = m_call->get_argument(int(i))->get_argument_expr();

        if (mi::mdl::IExpression_reference const *ref = as<mi::mdl::IExpression_reference>(arg)) {
            mi::mdl::IDefinition const           *def = ref->get_definition();

            MDL_ASSERT(def != NULL && "Array constructor unexpected here");

            if (def->get_kind() == mi::mdl::IDefinition::DK_PARAMETER ||
                def->get_kind() == mi::mdl::IDefinition::DK_VARIABLE)
            {
                if (LLVM_context_data *data = ctx.lookup_context_data(def)) {
                    return data->get_var_storage_modifier();
                }
            }
        }

        return SM_NORMAL;
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
        if (lit != NULL) {
            return lit->get_value();
        }
        return NULL;
    }

    /// If this is a DS_INTRINSIC_DAG_FIELD_ACCESS, the accessed field index, else -1.
    int get_field_index() const MDL_FINAL
    {
        // does never occur on AST
        return -1;
    }

    /// Translate this call as a boolean condition.
    /// If this is a ternary operator call, translate the first argument.
    ///
    /// \param true_bb   branch target for the true case
    /// \param false_bb  branch target for the false case
    void translate_boolean_branch(
        llvm::BasicBlock    *true_bb,
        llvm::BasicBlock    *false_bb) const MDL_FINAL
    {
        mi::mdl::IExpression const *cond;
        if (get_semantics() == operator_to_semantic(IExpression::OK_TERNARY)) {
            cond = m_call->get_argument(0)->get_argument_expr();
        } else {
            cond = m_call;
        }

        m_code_gen.translate_boolean_branch(
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

    /// Get the DAG source position of this call itself if exists.
    bool get_dag_dbg_info(DAG_DbgInfo &dbg_info) const MDL_FINAL
    {
        // none on AST
        return false;
    }

    /// Returns true, if the call should return derivatives.
    bool returns_derivatives() const MDL_FINAL
    {
        return m_code_gen.is_deriv_expr(m_call);
    }

private:
    /// The LLVM code generator.
    LLVM_code_generator &m_code_gen;

    /// The AST expression.
    mi::mdl::IExpression_call const *m_call;
};

/// Implements the ICall_expr interface for DAG calls.
class Call_dag_expr : public ICall_expr {
public:
    /// Constructor.
    ///
    /// \param code_gen  the LLVM code generator
    /// \param call      the DAG call node that is wrapped
    /// \param resolver  a name resolver that will be used for this call node
    Call_dag_expr(
        LLVM_code_generator                &code_gen,
        mi::mdl::DAG_call const            *call,
        mi::mdl::ICall_name_resolver const *resolver)
    : m_code_gen(code_gen)
    , m_call(call)
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

    /// Get the callee definition if one exists and its owner module.
    ///
    /// \param[out] owner_mod  out parameter for the owner module of the definition
    mi::mdl::IDefinition const *get_callee_definition_and_owner(
        mi::base::Handle<mi::mdl::Module const> &owner) const
    {
        char const *signature = m_call->get_name();
        if (signature[0] == '#') {
            // skip prefix for derivative variants
            ++signature;
        }

        owner = mi::mdl::impl_cast<mi::mdl::Module>(m_resolver->get_owner_module(signature));
        if (!owner.is_valid_interface()) {
            return NULL;
        }
        mi::mdl::Module const *module = owner.get();

        IDefinition const *def = module->find_signature(signature, /*only_exported=*/false);
        if (def != NULL) {
            def = skip_presets(def, owner);
        }
        return def;
    }

    /// Get the callee definition if one exists.
    ///
    /// \param code_gen  the LLVM code generator
    mi::mdl::IDefinition const *get_callee_definition() const MDL_FINAL
    {
        mi::base::Handle<mi::mdl::Module const> owner;
        return get_callee_definition_and_owner(owner);
    }

    /// Get the number of arguments.
    size_t get_argument_count() const MDL_FINAL {
        return m_call->get_argument_count();
    }

    /// Translate the i'th argument.
    ///
    /// \param i              the argument index
    /// \param wants_derivs   if true, the result should have derivatives, if available
    ///                       This is not used on the DAG, as the DAG nodes know, whether they
    ///                       should return derivatives or not.
    Expression_result translate_argument(
        size_t              i,
        bool                wants_derivs) const MDL_FINAL
    {
        mi::mdl::DAG_node const *arg = m_call->get_argument(int(i));

        return m_code_gen.translate_node(arg, m_resolver);
    }

    /// Get the LLVM context data of the callee.
    ///
    /// \param code_gen       the LLVM code generator
    /// \param inst           the function instance for this function call
    LLVM_context_data *get_callee_context(
        Function_instance const &inst) const MDL_FINAL
    {
        mi::base::Handle<mi::mdl::Module const> owner;
        if (get_callee_definition_and_owner(owner) == NULL) {
            return NULL;
        }

        // enter this module, so we gen create the context data if it does not exists yet
        LLVM_code_generator::MDL_module_scope scope(m_code_gen, owner.get());
        return m_code_gen.get_or_create_context_data(owner.get(), inst);
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

    /// Get the storage modifier of the i'th call argument.
    ///
    /// \param i  the argument index
    Storage_modifier get_argument_storage_modifier(
        size_t           i) const MDL_FINAL
    {
        if (!m_code_gen.target_supports_storage_spaces()) {
            return SM_NORMAL;
        }

        Function_context &ctx = *m_code_gen.m_ctx;

        mi::mdl::DAG_node const *arg = m_call->get_argument(int(i));

        arg = skip_temporaries(arg);

        if (is<DAG_parameter>(arg)) {
            mi::mdl::IType const *type = arg->get_type()->skip_type_alias();
            if (is<mi::mdl::IType_array>(type) || is<mi::mdl::IType_struct>(type)) {
                return SM_PARAMETER;
            }
        }

        if (DAG_constant const *c = as<DAG_constant>(arg)) {
            mi::mdl::IValue const *v = c->get_value();
            if (ctx.get_code_gen().is_stored_in_ro_data_segment(v)) {
                return SM_RODATA;
            }
        }

        return SM_NORMAL;
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
        if (c != NULL) {
            return c->get_value();
        }
        return NULL;
    }

    /// If this is a DS_INTRINSIC_DAG_FIELD_ACCESS, the accessed field, else -1.
    int get_field_index() const MDL_FINAL
    {
        if (m_call->get_semantic() == mi::mdl::IDefinition::DS_INTRINSIC_DAG_FIELD_ACCESS) {
            char const *call_name = m_call->get_name();

            mi::mdl::IType const *arg_type = get_argument_type(0)->skip_type_alias();
            arg_type = m_code_gen.get_type_mapper().skip_deriv_type(arg_type);

            switch (arg_type->get_kind()) {
            case mi::mdl::IType::TK_STRUCT:
                {
                    IType_struct const *s_type    = cast<IType_struct>(arg_type);
                    char const         *type_name = s_type->get_symbol()->get_name();
                    size_t             l          = strlen(type_name);

                    // skip derivative prefix
                    if (call_name[0] == '#') {
                        ++call_name;
                    }

                    char const *dot = nullptr;
                    // a valid getter name is <type_name> '.' <field_name>
                    if (strncmp(call_name, type_name, l) == 0 && call_name[l] == '.') {
                        dot = &call_name[l + 1];

                        string f_name(m_code_gen.get_allocator());
                        if (char const *n = strchr(dot, '(')) {
                            f_name = string(dot, n - dot, m_code_gen.get_allocator());
                        } else {
                            f_name = string(dot, m_code_gen.get_allocator());
                        }

                        for (size_t i = 0, n = s_type->get_field_count(); i < n; ++i) {
                            mi::mdl::IType_struct::Field const *field = s_type->get_field(i);

                            if (field->get_symbol()->get_name() == f_name) {
                                return i;
                            }
                        }
                    }
                }
                break;
            case  mi::mdl::IType::TK_VECTOR:
                {
                    // the type name is a predefined name here, we know it does not contain a '.'
                    char const *p = strchr(call_name, '.');

                    if (p != NULL) {
                        int index = -1;
                        switch (p[1]) {
                        case 'x': index = 0; break;
                        case 'y': index = 1; break;
                        case 'z': index = 2; break;
                        case 'w': index = 3; break;
                        default:
                            break;
                        }
                        return index;
                    }
                }
                break;
            default:
                break;
            }
        }
        return -1;
    }

    /// Translate this call as a boolean condition.
    /// If this is a ternary operator call, translate the first argument.
    ///
    /// \param code_gen  the LLVM code generator
    /// \param true_bb   branch target for the true case
    /// \param false_bb  branch target for the false case
    void translate_boolean_branch(
        llvm::BasicBlock    *true_bb,
        llvm::BasicBlock    *false_bb) const MDL_FINAL
    {
        DAG_node const *cond;
        if (get_semantics() == operator_to_semantic(IExpression::OK_TERNARY)) {
            cond = m_call->get_argument(int(0));
        } else {
            cond = m_call;
        }

        m_code_gen.translate_boolean_branch(
            m_resolver,
            cond,
            true_bb,
            false_bb);
    }

    /// If possible, convert into a DAG_call node.
    DAG_call const *as_dag_call() const MDL_FINAL { return m_call; }

    /// If possible, convert into an AST expression.
    mi::mdl::IExpression_call const *as_expr_call() const MDL_FINAL { return NULL; }

    /// Get the AST source position of this call itself.
    mi::mdl::Position const *get_position() const MDL_FINAL
    {
        // DAGs have no position yet
        return NULL;
    }

    /// Get the DAG source position of this call itself if exists.
    bool get_dag_dbg_info(DAG_DbgInfo &dbg_info) const MDL_FINAL
    {
        dbg_info = m_call->get_dbg_info();
        return true;
    }

    /// Returns true, if the call should return derivatives.
    bool returns_derivatives() const MDL_FINAL
    {
        return m_code_gen.get_type_mapper().is_deriv_type(m_call->get_type()->skip_type_alias());
    }

    /// Replace the call node.
    ///
    /// \param call  the new call node
    void replace(DAG_call const *call) { m_call = call; }

private:
    /// The LLVM code generator.
    LLVM_code_generator &m_code_gen;

    /// The DAG expression.
    mi::mdl::DAG_call const *m_call;

    /// The entity name resolver.
    mi::mdl::ICall_name_resolver const *m_resolver;
};

template<>
Call_dag_expr const *impl_cast(ICall_expr const *expr) {
    if (expr->as_dag_call() != NULL) {
        return static_cast<Call_dag_expr const *>(expr);
    }
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

// Register a function to take part in the analysis, which has been cloned from an
// already registered function. The state usage is initialized with the usage of the
// original function.
void State_usage_analysis::register_cloned_function(
    llvm::Function *cloned_func,
    llvm::Function *orig_func)
{
    Function_state_usage_info *info =
        m_arena_builder.create<Function_state_usage_info>(&m_arena);
    m_func_state_usage_info_map[cloned_func] = info;

    Function_state_usage_info_map::iterator it = m_func_state_usage_info_map.find(orig_func);
    if (it == m_func_state_usage_info_map.end()) {
        MDL_ASSERT(!"Function not registered for state usage info");
        return;
    }

    Function_state_usage_info *orig_info = it->second;
    info->state_usage = orig_info->state_usage;
    info->called_funcs.insert(orig_info->called_funcs.cbegin(), orig_info->called_funcs.cend());
}

// Register a mapped function to set the "expected" usage.
void State_usage_analysis::register_mapped_function(
    llvm::Function              *func,
    mdl::IDefinition::Semantics sema)
{
    Function_state_usage_info *info =
        m_arena_builder.create<Function_state_usage_info>(&m_arena);
    m_func_state_usage_info_map[func] = info;

    State_usage flag_to_add;

    // If a state function is mapped, assume its state is accessed. This might be
    // not enough, but probably an educated guess.
    switch (sema) {
    default:
    case mi::mdl::IDefinition::DS_UNKNOWN:
        return;

#define CASE(state) \
    case mi::mdl::IDefinition::DS_INTRINSIC_STATE_##state: \
        flag_to_add = mi::mdl::IGenerated_code_executable::SU_##state; \
        break;

#define CASE_TEXTURE_TANGENTS(state) \
    case mi::mdl::IDefinition::DS_INTRINSIC_STATE_##state: \
        flag_to_add = mi::mdl::IGenerated_code_executable::SU_TEXTURE_TANGENTS; \
        break;

#define CASE_GEOMETRY_TANGENTS(state) \
    case mi::mdl::IDefinition::DS_INTRINSIC_STATE_##state: \
        flag_to_add = mi::mdl::IGenerated_code_executable::SU_GEOMETRY_TANGENTS; \
        break;

#define CASE_TRANSFORMS(state) \
    case mi::mdl::IDefinition::DS_INTRINSIC_STATE_##state: \
        flag_to_add = mi::mdl::IGenerated_code_executable::SU_TRANSFORMS; \
        break;

    CASE(POSITION)
    CASE(NORMAL)
    CASE(GEOMETRY_NORMAL)
    CASE(MOTION)
    CASE(TEXTURE_COORDINATE)
    CASE_TEXTURE_TANGENTS(TEXTURE_TANGENT_U)
    CASE_TEXTURE_TANGENTS(TEXTURE_TANGENT_V)
    CASE(TANGENT_SPACE)
    CASE_GEOMETRY_TANGENTS(GEOMETRY_TANGENT_U)
    CASE_GEOMETRY_TANGENTS(GEOMETRY_TANGENT_V)
    CASE(DIRECTION)
    CASE(ANIMATION_TIME)
    CASE_TRANSFORMS(TRANSFORM)
    CASE_TRANSFORMS(TRANSFORM_POINT)
    CASE_TRANSFORMS(TRANSFORM_VECTOR)
    CASE_TRANSFORMS(TRANSFORM_NORMAL)
    CASE_TRANSFORMS(TRANSFORM_SCALE)
    CASE(ROUNDED_CORNER_NORMAL)
    CASE(OBJECT_ID)

#undef CASE_TRANSFORMS
#undef CASE_GEOMETRY_TANGENTS
#undef CASE_TEXTURE_TANGENTS
#undef CASE
    }

    info->state_usage    |= flag_to_add;
    m_module_state_usage |= flag_to_add;
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
            if (visited.count(cur)) {
                continue;
            }
            visited.insert(cur);
            Function_state_usage_info const *info = m_func_state_usage_info_map[cur];
            exported_func.state_usage |= info->state_usage;
            worklist.append(info->called_funcs.begin(), info->called_funcs.end());
        }
    }
}


// ------------------------------- LLVM code generator -------------------------------

static unsigned map_target_lang(
    ICode_generator::Target_language lang,
    Type_mapper::Type_mapping_mode   def_mode)
{
    switch (lang) {
    case ICode_generator::TL_NATIVE:
    case ICode_generator::TL_LLVM_IR:
        return def_mode;
    case ICode_generator::TL_PTX:
        return Type_mapper::TM_PTX;
    case ICode_generator::TL_HLSL:
        return Type_mapper::TM_HLSL | Type_mapper::TM_STRINGS_ARE_IDS;
    case ICode_generator::TL_GLSL:
        return Type_mapper::TM_GLSL | Type_mapper::TM_STRINGS_ARE_IDS;
    }
    MDL_ASSERT(!"unsupported target language");
    return def_mode;
}

/// Convert BinaryOptionData to a vector.
static vector<char>::Type to_store(BinaryOptionData data, IAllocator *alloc)
{
    vector<char>::Type v(0, alloc);
    v.assign(data.data, data.data + data.size);
    return v;
}

/// Convert string option to string.
static string to_string(char const *option, IAllocator *alloc)
{
    if (option == nullptr) {
        option = "";
    }
    return string(option, alloc);
}

// Constructor.
LLVM_code_generator::LLVM_code_generator(
    Jitted_code        *jitted_code,
    MDL                *compiler,
    IModule_cache      *module_cache,
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
, m_internal_space(to_string(
    options.get_string_option(MDL_CG_OPTION_INTERNAL_SPACE), jitted_code->get_allocator()))
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
, m_module_cache(module_cache)
, m_messages(messages)
, m_module(NULL)
, m_ctx(NULL)
, m_exported_func_list(jitted_code->get_allocator())
, m_user_state_store(to_store(
    options.get_binary_option(MDL_JIT_BINOPTION_LLVM_STATE_MODULE),
    jitted_code->get_allocator()))
, m_user_state_module(m_user_state_store.data(), m_user_state_store.size())
, m_renderer_store(to_store(
    options.get_binary_option(MDL_JIT_BINOPTION_LLVM_RENDERER_MODULE),
    jitted_code->get_allocator()))
, m_renderer_module(m_renderer_store.data(), m_renderer_store.size())
, m_visible_functions(to_string(
    options.get_string_option(MDL_JIT_OPTION_VISIBLE_FUNCTIONS),
    jitted_code->get_allocator()))
, m_func_pass_manager()
, m_finite_math(false)
, m_reciprocal_math(false)
, m_fast_math(get_math_options_from_options(options, &m_finite_math, &m_reciprocal_math))
, m_enable_ro_segment(
    options.get_bool_option(MDL_JIT_OPTION_ENABLE_RO_SEGMENT))
, m_max_const_size(
    size_t(options.get_int_option(MDL_JIT_OPTION_MAX_CONST_DATA)))
, m_always_inline(options.get_bool_option(MDL_JIT_OPTION_INLINE_AGGRESSIVELY))
, m_eval_dag_ternary_strictly(options.get_bool_option(MDL_JIT_OPTION_EVAL_DAG_TERNARY_STRICTLY))
, m_sl_use_resource_data(options.get_bool_option(MDL_JIT_OPTION_SL_USE_RESOURCE_DATA))
, m_use_renderer_adapt_microfacet_roughness(options.get_bool_option(
    MDL_JIT_OPTION_USE_RENDERER_ADAPT_MICROFACET_ROUGHNESS))
, m_use_renderer_adapt_normal(options.get_bool_option(
    MDL_JIT_OPTION_USE_RENDERER_ADAPT_NORMAL))
, m_in_intrinsic_generator(false)
, m_target_lang(target_lang)
, m_lambda_return_mode(
    parse_return_mode(options.get_string_option(MDL_JIT_OPTION_LAMBDA_RETURN_MODE)))
, m_func_remap(
    jitted_code->get_allocator(), options.get_string_option(MDL_JIT_OPTION_REMAP_FUNCTIONS))
, m_runtime(create_mdl_runtime(
    m_arena_builder,
    this,
    target_lang,
    m_fast_math,
    has_tex_handler,
    /*always_inline_rt=*/ true,
    m_internal_space.c_str()))
, m_has_res_handler(has_tex_handler)
, m_di_builder(NULL)
, m_di_file()
, m_world_to_object(NULL)
, m_object_to_world(NULL)
, m_object_id(0)
, m_context_data(0, Context_data_map::hasher(), Context_data_map::key_equal(), get_allocator())
, m_opt_level(get_opt_level_from_options(target_lang, options))
, m_enable_full_debug(get_enable_full_debug(enable_debug))
, m_enable_type_debug(target_is_structured_language(target_lang))
, m_sm_version(target_lang == ICode_generator::TL_PTX ? sm_version : 0)
, m_min_ptx_version(
    // LLVM uses "labels1 - labels2 expression in .section" which requires PTX ISA 7.5 or higher.
    // We use 7.8 because it already is supported in LLVM
    m_enable_full_debug ? 78 : 40)
, m_ptx_target_machine(
    target_lang == ICode_generator::TL_PTX
        ? create_ptx_target_machine()
        : nullptr)
, m_data_layout(
    target_lang == ICode_generator::TL_PTX
        ? m_ptx_target_machine->createDataLayout()
        : jitted_code->get_layout_data())  // copy the data layout without the struct cache
, m_type_mapper(
    jitted_code->get_allocator(),
    m_llvm_context,
    &m_data_layout,
    state_mapping |
        (options.get_bool_option(MDL_JIT_OPTION_TEX_RUNTIME_WITH_DERIVATIVES) ?
            Type_mapper::SM_USE_DERIVATIVES : 0) |
        (target_is_structured_language(target_lang) ? Type_mapper::SM_INCLUDE_ARG_BLOCK_OFFS : 0),
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
, m_scatter_components_map(
    0,
    Scatter_components_map::hasher(),
    Scatter_components_map::key_equal(),
    get_allocator())
, m_global_const_map(0, Global_const_map::hasher(), Global_const_map::key_equal(), get_allocator())
, m_internalized_string_map(
    0,
    Internalized_string_map::hasher(),
    Internalized_string_map::key_equal(),
    get_allocator())
, m_ro_segment(NULL)
, m_next_ro_data_offset(0)
, m_ro_data_values(jitted_code->get_allocator())
, m_force_inlining_targets(
    0,
    Force_inline_targets::hasher(),
    Force_inline_targets::key_equal(),
    get_allocator())
, m_optix_cp_from_id(NULL)
, m_captured_args_mdl_types(get_allocator())
, m_captured_args_type(NULL)
, m_sl_funcs()
, m_resource_tag_map(NULL)
, m_jit_dbg_mode(JDBG_NONE)
, m_num_texture_spaces(num_texture_spaces)
, m_num_texture_results(num_texture_results)
, m_state_usage_analysis(*this)
, m_enable_noinline(false)
, m_exported_funcs_are_entries(false)
, m_state_functions_no_bounds_exception(true)
, m_bounds_check_exception_disabled(
    target_lang != ICode_generator::TL_NATIVE ||
    options.get_bool_option(MDL_JIT_OPTION_DISABLE_EXCEPTIONS))
, m_divzero_check_exception_disabled(
    target_lang != ICode_generator::TL_NATIVE ||
    options.get_bool_option(MDL_JIT_OPTION_DISABLE_EXCEPTIONS))
, m_uses_state_param(false)
, m_mangle_name(target_lang != ICode_generator::TL_NATIVE)
, m_enable_instancing(true)
, m_lambda_force_sret(true)  // sret is the default mode
, m_lambda_first_param_by_ref(false)
, m_lambda_force_render_state(false)
, m_lambda_force_no_lambda_results(false)
, m_use_ro_data_segment(false)
, m_link_libdevice(
    target_lang == ICode_generator::TL_PTX &&
    options.get_bool_option(MDL_JIT_OPTION_LINK_LIBDEVICE))
, m_link_libmdlrt(false)
, m_link_libbsdf_df_handle_slot_mode(parse_df_handle_slot_mode(
    options.get_string_option(MDL_JIT_OPTION_LINK_LIBBSDF_DF_HANDLE_SLOT_MODE)))
, m_libbsdf_flags_in_bsdf_data(options.get_bool_option(MDL_JIT_OPTION_LIBBSDF_FLAGS_IN_BSDF_DATA))
, m_incremental(incremental)
, m_texruntime_with_derivs(options.get_bool_option(MDL_JIT_OPTION_TEX_RUNTIME_WITH_DERIVATIVES))
, m_deriv_infos(NULL)
, m_cur_func_deriv_info(NULL)
, m_tex_calls_mode(parse_call_mode(
    options.get_string_option(MDL_JIT_OPTION_TEX_LOOKUP_CALL_MODE)))
, m_dist_func(NULL)
, m_dist_func_state(DFSTATE_NONE)
, m_cur_main_func_index(0)
, m_instantiated_dfs(
    Distribution_function_state::DFSTATE_END_STATE,
    Instantiated_dfs(get_allocator()),
    get_allocator())
, m_libbsdf_template_funcs(get_allocator())
, m_enable_auxiliary(options.get_bool_option(MDL_JIT_OPTION_ENABLE_AUXILIARY))
, m_enable_pdf(options.get_bool_option(MDL_JIT_OPTION_ENABLE_PDF))
, m_warn_spectrum_conversion(options.get_bool_option(MDL_JIT_WARN_SPECTRUM_CONVERSION))
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
, m_type_bsdf_get_factor_func(NULL)
, m_type_bsdf_evaluate_data(NULL)
, m_type_bsdf_pdf_func(NULL)
, m_type_bsdf_pdf_data(NULL)
, m_type_bsdf_auxiliary_func(NULL)
, m_type_bsdf_auxiliary_data(NULL)
, m_type_edf_sample_func(NULL)
, m_type_edf_sample_data(NULL)
, m_type_edf_evaluate_func(NULL)
, m_type_edf_get_factor_func(NULL)
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
, m_int_func_state_adapt_microfacet_roughness(NULL)
, m_int_func_state_adapt_normal(NULL)
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

    s = getenv("MI_MDL_JIT_NOINLINE");
    if (s != NULL) {
        m_enable_noinline = true;
    }

    s = getenv("MI_MDL_JIT_TRACE");
    if (s != NULL) {
        m_jit_dbg_mode = JDBG_PRINT;
    }

    if (!target_is_structured_language()) {
        // this option can only be set for GLSL/HLSL
        m_sl_use_resource_data = false;
    }

    prepare_internal_functions();

    //
    // Force-inline some functions to avoid matrices on the stack for PTX
    //
    #define TARGET_BIT(x) (1 << ICode_generator::TL_##x)

    // base::transform_coordinate()
    m_force_inlining_targets[
        "_ZN4base20transform_coordinateEu8float4x4N4base23texture_coordinate_infoE"]
        = TARGET_BIT(PTX);

    // base::lookup_volume_coefficients()
    m_force_inlining_targets[
        "_ZN4base26lookup_volume_coefficientsEU7uniformu10texture_3dU7uniformu5colorU7uniform"
        "u5colorU7uniformu5colorU7uniformu5colorU7uniformu8float4x4U7uniformfU7uniformb"]
        = TARGET_BIT(PTX);

    #undef TARGET_BIT
}

// Returns true if full debug information should be enabled.
bool LLVM_code_generator::get_enable_full_debug(bool enable_debug)
{
    // Allow to overwrite via environment variable
    char const *s = getenv("MI_MDL_JIT_DEBUG_INFO");
    if (s != NULL) {
        return true;
    }

    return enable_debug;
}

// Get the optimization level from the options and environment.
unsigned LLVM_code_generator::get_opt_level_from_options(
    Target_language target_lang,
    Options_impl const &options)
{
    unsigned opt_level = unsigned(options.get_int_option(MDL_JIT_OPTION_OPT_LEVEL));

    char const *s = getenv("MI_MDL_JIT_OPTLEVEL");
    unsigned env_opt_level;
    if (s != NULL && sscanf(s, "%u", &env_opt_level) == 1) {
        opt_level = env_opt_level;
    }

    if (target_lang == ICode_generator::TL_PTX) {
        // Optimization level 3+ activates argument promotion. This is bad, because the NVPTX
        // backend cannot handle aggregate types passed by value. Hence limit the level to
        // 2 in this case.
        if (opt_level > 2) {
            opt_level = 2;
        }
    }

    return opt_level;
}

// Get the math options from the options and environment.
bool LLVM_code_generator::get_math_options_from_options(
    Options_impl const &options,
    bool *finite_math,
    bool *reciprocal_math)
{
    bool fast_math = options.get_bool_option(MDL_JIT_OPTION_FAST_MATH);

    char *s = getenv("MI_MDL_JIT_FAST_MATH");
    unsigned level;
    if (s != NULL && sscanf(s, "%u", &level) == 1) {
        fast_math = *finite_math = *reciprocal_math = false;
        if (level >= 3) {
            fast_math = true;
        }
        if (level >= 2) {
            *finite_math = true;
        }
        if (level >= 1) {
            *reciprocal_math = true;
        }
    }

    return fast_math;
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
        "_ZN5state10set_normalERK6float3",
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
        "_ZN5state24get_measured_curve_valueEii",
        Internal_function::KI_STATE_GET_MEASURED_CURVE_VALUE,
        Internal_function::FL_HAS_STATE |
        Internal_function::FL_HAS_RES | Internal_function::FL_HAS_EXC |
        Internal_function::FL_HAS_CAP_ARGS | Internal_function::FL_HAS_EXEC_CTX,
        /*ret_type=*/ m_type_mapper.get_float3_type(),
        /*param_types=*/ Array_ref<IType const *>(measured_param_types),
        /*param_names=*/ Array_ref<char const *>(measured_param_names));

    m_int_func_state_adapt_microfacet_roughness = m_arena_builder.create<Internal_function>(
        m_arena_builder.get_arena(),
        "::state::adapt_microfacet_roughness(float2)",
        "_ZN5state26adapt_microfacet_roughnessERK6float2",
        Internal_function::KI_STATE_ADAPT_MICROFACET_ROUGHNESS,
        Internal_function::FL_HAS_STATE | Internal_function::FL_HAS_EXEC_CTX,
        /*ret_type=*/ m_type_mapper.get_float2_type(),
        /*param_types=*/ Array_ref<IType const *>(float2_type),
        /*param_names=*/ Array_ref<char const *>("roughness_uv"));

    m_int_func_state_adapt_normal = m_arena_builder.create<Internal_function>(
        m_arena_builder.get_arena(),
        "::state::adapt_normal(float3)",
        "_ZN5state12adapt_normalERK6float3",
        Internal_function::KI_STATE_ADAPT_NORMAL,
        Internal_function::FL_HAS_STATE | Internal_function::FL_HAS_RES,
        /*ret_type=*/ m_type_mapper.get_float3_type(),
        /*param_types=*/ Array_ref<IType const*>(float3_type),
        /*param_names=*/ Array_ref<char const*>("normal"));

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

/// The extension function to add the FunctionInstCount pass.
static void AddFunctionInstCounterExtension(
    llvm::PassManagerBuilder const &,
    llvm::PassManagerBase &PM)
{
    PM.add(llvm::createFunctionInstCounterPass());
}

// Optimize an LLVM function.
bool LLVM_code_generator::optimize(llvm::Function *func)
{
    return m_func_pass_manager->run(*func);
}

// Optimize LLVM code.
bool LLVM_code_generator::optimize(llvm::Module *module)
{
    if (m_target_lang == ICode_generator::TL_PTX) {
        // already remove any unreferenced libDevice functions to avoid
        // LLVM optimizing them for nothing and mark uses ones as internal
        if (m_link_libdevice) {
            llvm::legacy::PassManager mpm;
            mpm.add(llvm::createDeleteUnusedLibDevicePass());
            mpm.run(*module);
        }

        // always run the PTX reflection pass.
        // This will replace all __nvvm_reflect calls for __CUDA_FTZ by zero to not flush
        // denormal values to zero, when performing single-precision FP operations.
        // Note: To set it to a different value, set the value as module flag "nvvm-reflect-ftz".

        llvm::legacy::FunctionPassManager fpm(module);
        fpm.add(llvm::createNVVMReflectPass(m_sm_version));
        for (auto &func : module->functions()) {
            fpm.run(func);
        }
    }

    llvm::PassManagerBuilder builder;
    builder.OptLevel         = m_opt_level;
    builder.AvoidPointerPHIs = target_is_structured_language();
    builder.EnableVectorizer = !target_is_structured_language();

    // TODO: in PTX mode we don't use the C-library, but libdevice, this probably must
    // be registered somewhere, or libcall simplification can happen

    if (m_opt_level > 1) {
        builder.Inliner = llvm::createFunctionInliningPass();
    } else {
        builder.Inliner = llvm::createAlwaysInlinerLegacyPass();
    }

    if (m_target_lang == ICode_generator::TL_PTX && m_link_libdevice) {
        // add our extra pass to remove any unused rest of libDevice after the inliner
        // and even in optlevel 0
        builder.addExtension(
            m_opt_level == 0 ?
                llvm::PassManagerBuilder::EP_EnabledOnOptLevel0 :
                llvm::PassManagerBuilder::EP_LoopOptimizerEnd,
            AddDeleteUnusedLibDeviceExtension);
    }

    if (m_jitted_code->opt_remarks_enabled()) {
        // add report
        builder.addExtension(m_opt_level == 0 ?
            llvm::PassManagerBuilder::EP_EnabledOnOptLevel0 :
            llvm::PassManagerBuilder::EP_LoopOptimizerEnd,
            AddFunctionInstCounterExtension);
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
    if ((df_kind == mdl::IType::TK_BSDF || df_kind == mi::mdl::IType::TK_HAIR_BSDF) &&
        m_dist_func_state != DFSTATE_NONE)
    {
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

    if (type->get_kind() == mdl::IType::TK_EDF && m_dist_func_state != DFSTATE_NONE) {
        switch (m_dist_func_state) {
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
    mi::mdl::ICall_expr const *call_expr)
{
    Function_context &ctx = *m_ctx;

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
    mi::mdl::IExpression const *expr)
{
    Function_context &ctx = *m_ctx;

    mi::mdl::IType const *res_type = expr->get_type()->skip_type_alias();

    if (m_cur_func_deriv_info != NULL && m_cur_func_deriv_info->is_derivative_expression(expr)) {
        return m_type_mapper.lookup_deriv_type(res_type, ctx.instantiate_type_size(res_type));
    }

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
bool LLVM_code_generator::need_reference_return(mi::mdl::IType const *type, int arr_size) const
{
    if ((type->skip_type_alias()->get_kind() == mi::mdl::IType::TK_BSDF ||
            type->skip_type_alias()->get_kind() == mi::mdl::IType::TK_HAIR_BSDF ||
            type->skip_type_alias()->get_kind() == mi::mdl::IType::TK_EDF) &&
        m_dist_func_state != DFSTATE_NONE)
    {
        // init function does not return anything
        if (m_dist_func_state == DFSTATE_INIT) {
            return false;
        }

        // the bsdf function return types are always structs, so return by reference
        return true;
    }

    return m_type_mapper.need_reference_return(type, arr_size);
}

// Check if the given parameter type must be passed by reference.
bool LLVM_code_generator::is_passed_by_reference(mi::mdl::IType const *type, int arr_size) const
{
    if ((type->skip_type_alias()->get_kind() == mi::mdl::IType::TK_BSDF ||
            type->skip_type_alias()->get_kind() == mi::mdl::IType::TK_HAIR_BSDF ||
            type->skip_type_alias()->get_kind() == mi::mdl::IType::TK_EDF) &&
        m_dist_func_state != DFSTATE_NONE)
    {
        // init function does not return anything
        if (m_dist_func_state == DFSTATE_INIT) {
            return false;
        }

        return true;
    }

    return m_type_mapper.is_passed_by_reference(type, arr_size);
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
            case Resource_tag_tuple::RK_MICROFLAKE_SHEEN_GENERAL:
            case Resource_tag_tuple::RK_MICROFLAKE_SHEEN_MULTISCATTER:
                {
                    IValue_texture::Bsdf_data_kind bsdf_data_kind =
                        bsdf_data_kind_from_kind(k.m_kind);
                    if (bsdf_data_kind != IValue_texture::BDK_NONE) {
                        m_bsdf_data_texture_ids[int(bsdf_data_kind) - 1] = e.index;
                    }
                }
                break;
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
        case Resource_tag_tuple::RK_MICROFLAKE_SHEEN_GENERAL:
        case Resource_tag_tuple::RK_MICROFLAKE_SHEEN_MULTISCATTER:
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
            }
            break;
        case Resource_tag_tuple::RK_LIGHT_PROFILE:
            if (lp_entries.size() < e.index + 1) {
                lp_entries.resize(e.index + 1);
            }
            lp_entries[e.index] =
                Light_profile_attribute_entry(e.valid, e.u.lp.power, e.u.lp.maximum);
            break;
        case Resource_tag_tuple::RK_BSDF_MEASUREMENT:
            if (bm_entries.size() < e.index + 1) {
                bm_entries.resize(e.index + 1);
            }
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
    mi::mdl::Module const *module)
{
    LLVM_code_generator::MDL_module_scope scope(*this, module);

    create_module(module->get_name(), module->get_filename());

    // initialize the module with user code
    if (!init_user_modules()) {
        // drop the module and give up
        drop_llvm_module(m_module);
        return NULL;
    }

    if (target_is_structured_language()) {
        init_sl_code_gen();
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
            mi::mdl::IDeclaration_function const *func_decl =
                cast<mi::mdl::IDeclaration_function>(def->get_declaration());

            if (func_decl->get_body() == NULL) {
                // declaration only (in stdlib), ignore
                continue;
            }

            mi::mdl::IType_function const *f_tp   = cast<mi::mdl::IType_function>(def->get_type());
            mi::mdl::IType const          *ret_tp = f_tp->get_return_type();

            if (is_material_type(ret_tp)) {
                // this is a material constructor, ignore them
                continue;
            }

            if (func_decl != NULL) {
                // skip presets: this will insert the "real" body
                mi::base::Handle<Module const> owner = mi::base::make_handle_dup(module);
                func_decl = cast<IDeclaration_function>(
                    skip_presets(func_decl, owner)->get_declaration());

                // check if instantiation is needed at top level
                IType_function const *ftype = cast<IType_function>(def->get_type());
                int n_params = ftype->get_parameter_count();

                bool need_instantiation = false;
                for (int p_idx = 0; p_idx < n_params; ++p_idx) {
                    ISymbol const *p_sym;
                    IType const   *p_type;

                    ftype->get_parameter(p_idx, p_type, p_sym);

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
                Function_instance::Parameter_storage_modifiers param_mods(get_allocator());
                Function_instance inst(
                    def,
                    args,
                    param_mods,
                    /*return_derivs=*/ false,
                    target_supports_storage_spaces());

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
    bool                       has_uniform_state,
    Float4_struct const        world_to_object[4],
    Float4_struct const        object_to_world[4],
    int                        object_id)
{
    IAllocator *alloc = m_arena.get_allocator();

    reset_lambda_state();

    // const functions return the result by reference
    m_lambda_force_sret = true;

    if (has_uniform_state) {
        // const functions do not have a state parameter, set necessary data to evaluate
        // uniform state functions
        set_uniform_state(world_to_object, object_to_world, object_id);
    }

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

    add_generated_attributes(func);

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
        Function_instance inst(alloc, &lambda, target_supports_storage_spaces());
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
        Expression_result res = translate_node(lambda.get_body(), resolver);
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

// Declare an user-provided HLSL/GLSL read function, which gets an int offset as parameter.
llvm::Function *LLVM_code_generator::declare_sl_read_func(
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

    // let LLVM treat the function as a scalar to avoid duplicate calls
    func->setDoesNotThrow();
    func->setWillReturn();
    func->setDoesNotAccessMemory();

    return func;
}

// Initialize types and functions needed for HLSL/GLSL.
void LLVM_code_generator::init_sl_code_gen()
{
    struct Descriptor {
        llvm::Function *&result;
        llvm::Type     *ret_type;
        char const     *name;
    } functions[] = {
        {
            m_sl_funcs.m_rodata_as_int,
            m_type_mapper.get_int_type(),
            "mdl_read_rodata_as_int"
        },
        {
            m_sl_funcs.m_rodata_as_uint,
            m_type_mapper.get_int_type(),
            "mdl_read_rodata_as_uint"
        },
        {
            m_sl_funcs.m_rodata_as_bool,
            m_type_mapper.get_bool_type(),
            "mdl_read_rodata_as_bool"
        },
        {
            m_sl_funcs.m_rodata_as_float,
            m_type_mapper.get_float_type(),
            "mdl_read_rodata_as_float"
        },
        {
            m_sl_funcs.m_rodata_as_double,
            m_type_mapper.get_double_type(),
            "mdl_read_rodata_as_double"
        },
        {
            m_sl_funcs.m_argblock_as_int,
            m_type_mapper.get_int_type(),
            "mdl_read_argblock_as_int"
        },
        {
            m_sl_funcs.m_argblock_as_uint,
            m_type_mapper.get_int_type(),
            "mdl_read_argblock_as_uint"
        },
        {
            m_sl_funcs.m_argblock_as_bool,
            m_type_mapper.get_bool_type(),
            "mdl_read_argblock_as_bool"
        },
        {
            m_sl_funcs.m_argblock_as_float,
            m_type_mapper.get_float_type(),
            "mdl_read_argblock_as_float"
        },
        {
            m_sl_funcs.m_argblock_as_double,
            m_type_mapper.get_double_type(),
            "mdl_read_argblock_as_double"
        }
    };

    for (size_t i = 0, n = llvm::array_lengthof(functions); i < n; ++i) {
        Descriptor &d = functions[i];

        d.result = declare_sl_read_func(d.ret_type, d.name);
    }
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

        if (target_is_structured_language()) {
            init_sl_code_gen();
        }
    }

    create_resource_tables(lambda);

    LLVM_context_data *ctx_data = get_or_create_context_data(&lambda);
    llvm::Function    *func     = ctx_data->get_function();
    unsigned          flags     = ctx_data->get_function_flags();

    add_generated_attributes(func);

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
        Function_instance inst(alloc, &lambda, target_supports_storage_spaces());
        Function_context ctx(alloc, *this, inst, func, flags);

        llvm::Function::arg_iterator arg_it = get_first_parameter(func, ctx_data);

        if (lambda.get_root_expr_count() > 0) {
            // add result and proj parameters: these will never be written
            llvm::Value *result = arg_it++;
            ctx.create_context_data(size_t(0), result, /*by_reference=*/false);
            llvm::Value *proj = arg_it++;
            ctx.create_context_data(size_t(1), proj, /*by_reference=*/false);
        }

        if (m_texruntime_with_derivs) {
            m_deriv_infos = lambda.get_derivative_infos();
        }

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

            llvm::Value *val    = translate_node(root_expr, resolver).as_value(ctx);
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
                    st->setAlignment(llvm::Align(res_type->getPrimitiveSizeInBits() / 8));
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

    // Don't allow returning structs at ABI level, even in value mode
    m_lambda_force_sret = m_lambda_return_mode == Return_mode::RETMODE_SRET
        || (m_lambda_return_mode == Return_mode::RETMODE_VALUE &&
            is<mi::mdl::IType_struct>(lambda.get_return_type()->skip_type_alias()));

    // only force, when actually supported by backend
    m_lambda_force_sret &= target_supports_sret_for_lambda();

    // generic functions always includes a render state in its interface
    m_lambda_force_render_state = true;

    create_captured_argument_struct(m_llvm_context, lambda);

    if (m_target_lang == ICode_generator::TL_NATIVE) {
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

        if (target_is_structured_language()) {
            init_sl_code_gen();
        }
    }

    create_resource_tables(lambda);

    LLVM_context_data *ctx_data = get_or_create_context_data(&lambda);
    llvm::Function    *func     = ctx_data->get_function();
    unsigned          flags     = ctx_data->get_function_flags();

    add_generated_attributes(func);

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
        Function_instance inst(alloc, &lambda, target_supports_storage_spaces());
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
        Expression_result res = translate_node(lambda.get_body(), resolver);
        context.create_return(res.as_value(context));
    }

    // if we are compiling with derivatives, all waiting functions need to be compiled now,
    // to give them access to the derivative infos
    if (m_deriv_infos != NULL) {
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
LLVM_context_data::Flags LLVM_code_generator::get_function_flags(
    Function_instance const &inst,
    IDefinition const *def)
{
    MDL_ASSERT(def->get_kind() == mi::mdl::IDefinition::DK_FUNCTION);

    IType_function const     *func_type = cast<mi::mdl::IType_function>(def->get_type());
    IType const              *ret_type  = func_type->get_return_type();
    LLVM_context_data::Flags flags      = LLVM_context_data::FL_NONE;

    bool is_sret_func = need_reference_return(ret_type, inst.instantiate_type_size(ret_type));
    if (is_sret_func) {
        flags |= LLVM_context_data::FL_SRET;
    }

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
        if (def->get_semantics() == IDefinition::DS_UNKNOWN) {
            need_render_state_param = true;
        }
    }

    // always pass the render state to native functions
    if (def->get_property(mi::mdl::IDefinition::DP_IS_NATIVE)) {
        need_render_state_param = true;
    }

    if (need_render_state_param) {
        flags |= LLVM_context_data::FL_HAS_STATE;
    }

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
            strcmp(m_internal_space.c_str(), "*") != 0)
        {
            flags |= LLVM_context_data::FL_HAS_TRANSFORMS;
        }
    }

    return flags;
}

// Set the LLVM attributes for generated functions.
//
// Currently mark the given LLVM function as AlwaysInline if all generated LLVM functions
// should receive the AlwaysInline attribute.
void LLVM_code_generator::add_generated_attributes(llvm::Function *func) const
{
    if (is_always_inline_enabled() && !func->hasFnAttribute(llvm::Attribute::NoInline)) {
        func->addFnAttr(llvm::Attribute::AlwaysInline);
    }
}

// Set LLVM function attributes which need to be consistent to avoid loosing
// them during inlining (e.g. for fast math).
void LLVM_code_generator::set_llvm_function_attributes(
    llvm::Function *func,
    bool           mark_noinline)
{
    if (mark_noinline) {
        func->addFnAttr(llvm::Attribute::NoInline);
    }
    if (is_always_inline_enabled() && !func->hasFnAttribute(llvm::Attribute::NoInline)) {
        func->addFnAttr(llvm::Attribute::AlwaysInline);
    }
    if (is_fast_math_enabled()) {
        func->addFnAttr("unsafe-fp-math", "true");
    }
    if (is_finite_math_enabled()) {
        func->addFnAttr("no-infs-fp-math", "true");
        func->addFnAttr("no-nans-fp-math", "true");
    }
}

// Declares an LLVM function from a MDL function instance.
LLVM_context_data *LLVM_code_generator::declare_function(
    mi::mdl::Module const   *owner,
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
    if (m_deriv_infos != NULL && inst.get_return_derivs()) {
        ret_tp = m_type_mapper.lookup_deriv_type(mdl_ret_tp, arr_size);
    } else {
        ret_tp = lookup_type(mdl_ret_tp, arr_size);
    }
    MDL_ASSERT(ret_tp != NULL);

    llvm::Type *real_ret_tp = ret_tp;

    mi::mdl::vector<llvm::Type *>::Type arg_types(get_allocator());

    bool is_sret_func = need_reference_return(func_type->get_return_type(), arr_size);
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
        if (def->get_semantics() == IDefinition::DS_UNKNOWN) {
            need_render_state_param = true;
        }
    }

    // check if any parameter uses a storage mode which needs state access
    size_t n_params = func_type->get_parameter_count();
    if (target_supports_storage_spaces()) {
        for (size_t i = 0; i < n_params; ++i) {
            if (inst.get_parameter_storage_modifier(i) != SM_NORMAL) {
                need_render_state_param = true;
                break;
            }
        }
    }

    // always pass the render state to native functions
    if (def->get_property(mi::mdl::IDefinition::DP_IS_NATIVE)) {
        need_render_state_param = true;
    }

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
            strcmp(m_internal_space.c_str(), "*") != 0)
        {
            // add two hidden transform (matrix) parameters
            arg_types.push_back(m_type_mapper.get_arr_float_4_ptr_type());
            arg_types.push_back(m_type_mapper.get_arr_float_4_ptr_type());

            flags |= LLVM_context_data::FL_HAS_TRANSFORMS;
        }
    }

    bool is_native = def->get_property(IDefinition::DP_IS_NATIVE);

    string func_name(mangle(inst, name_prefix));

    bool enforce_inlining = false;
    Force_inline_targets::const_iterator it = m_force_inlining_targets.find(func_name.c_str());
    if (it != m_force_inlining_targets.end()) {
        enforce_inlining = (it->second & (1 << m_target_lang)) != 0;
    }

    if (char const *mapped_name = m_func_remap.get_mapper_symbol(func_name.c_str())) {
        // mapped to an external one
        flags |= LLVM_context_data::FL_IS_MAPPED;

        // Note: there is no check if this external name will clash with any existing symbol
        func_name = mapped_name;

        // also mark it as native here, so it is marked as no-inline AND external
        is_native = true;

        // Note: so far we do NOT set entry point here. so the calling convention is
        // "internal"
    }

    Func_deriv_info const *func_deriv_info = nullptr;
    if (m_deriv_infos != nullptr) {
        func_deriv_info = m_deriv_infos->get_function_derivative_infos(inst);
    }

    for (size_t i = 0; i < n_params; ++i) {
        mi::mdl::IType const   *p_type;
        mi::mdl::ISymbol const *p_sym;

        func_type->get_parameter(i, p_type, p_sym);

        // instantiate parameter types
        Storage_modifier mod = inst.get_parameter_storage_modifier(i);
        if (mod != SM_NORMAL) {
            arg_types.push_back(m_type_mapper.get_int_type());  // offset parameter
            continue;
        }

        int arr_size = inst.instantiate_type_size(p_type);
        llvm::Type *tp;
        if (func_deriv_info != nullptr && func_deriv_info->args_want_derivatives.test_bit(i + 1)) {
            tp = m_type_mapper.lookup_deriv_type(p_type, arr_size);

            // pass by reference if supported by target
            if (is_passed_by_reference(p_type, arr_size) || target_supports_pointers()) {
                arg_types.push_back(Type_mapper::get_ptr(tp));
            } else {
                arg_types.push_back(tp);
            }
        } else {
            tp = lookup_type(p_type, arr_size);

            if (is_passed_by_reference(p_type, arr_size)) {
                arg_types.push_back(Type_mapper::get_ptr(tp));
            } else {
                arg_types.push_back(tp);
            }
        }
    }

    // entry points and native functions must have external linkage
    llvm::GlobalValue::LinkageTypes const linkage =
        is_entry_point || is_native ?
        llvm::GlobalValue::ExternalLinkage :
        llvm::GlobalValue::InternalLinkage;

    llvm::Function *func = llvm::Function::Create(
        llvm::FunctionType::get(ret_tp, arg_types, false),
        linkage,
        func_name.c_str(),
        m_module);
    set_llvm_function_attributes(func, /*mark_noinline=*/is_native);
    add_generated_attributes(func);
    if (enforce_inlining) {
        func->addFnAttr(llvm::Attribute::AlwaysInline);
    }

    if (is_entry_point) {
        func->setCallingConv(llvm::CallingConv::C);
    }

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
            func->addParamAttr(
                arg_it->getArgNo(),
                llvm::Attribute::getWithStructRetType(
                    m_llvm_context, arg_it->getType()->getPointerElementType()));
        } else {
            // treat the first argument as a pointer, but we could at least improve
            // the code a bit, because we "know" that the extra parameter is alias free
            func->addParamAttr(arg_it->getArgNo(), llvm::Attribute::NoAlias);
            func->addParamAttr(arg_it->getArgNo(), llvm::Attribute::NoCapture);
        }
        ++arg_it;
    }
    if (flags & LLVM_context_data::FL_HAS_STATE) {
        arg_it->setName("state");

        // the state pointer does not alias and is not captured
        func->addParamAttr(arg_it->getArgNo(), llvm::Attribute::NoAlias);
        func->addParamAttr(arg_it->getArgNo(), llvm::Attribute::NoCapture);
        ++arg_it;
    }
    if (flags & LLVM_context_data::FL_HAS_RES) {
        arg_it->setName(target_supports_pointers() ? "res_data_pair" : "res_data");

        // the resource data pointer does not alias and is not captured
        func->addParamAttr(arg_it->getArgNo(), llvm::Attribute::NoAlias);
        func->addParamAttr(arg_it->getArgNo(), llvm::Attribute::NoCapture);
        ++arg_it;
    }
    if (flags & LLVM_context_data::FL_HAS_EXC) {
        arg_it->setName("exc_state");

        // the exc_state pointer does not alias and is not captured
        func->addParamAttr(arg_it->getArgNo(), llvm::Attribute::NoAlias);
        func->addParamAttr(arg_it->getArgNo(), llvm::Attribute::NoCapture);
        ++arg_it;
    }
    if (flags & LLVM_context_data::FL_HAS_CAP_ARGS) {
        arg_it->setName("captured_arguments");

        // the cap_args pointer does not alias and is not captured
        func->addParamAttr(arg_it->getArgNo(), llvm::Attribute::NoAlias);
        func->addParamAttr(arg_it->getArgNo(), llvm::Attribute::NoCapture);
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

        // parameters are not captured and due to creation of shadow copies on write,
        // we can also ignore aliasing
        if (arg_it->getType()->isPointerTy()) {
            func->addParamAttr(arg_it->getArgNo(), llvm::Attribute::NoCapture);
            func->addParamAttr(arg_it->getArgNo(), llvm::Attribute::NoAlias);
        }
    }

    if (is_prototype) {
        // set external linkage and do NOT enqueue
        func->setLinkage(llvm::GlobalValue::ExternalLinkage);
    } else if (flags & LLVM_context_data::FL_IS_MAPPED) {
        // do not enqueue, BUT register for "expected" state usage
        m_state_usage_analysis.register_mapped_function(func, def->get_semantics());
    } else {
        // put this function on the wait queue
        m_functions_q.push(Owner_func_inst_pair(owner, inst));

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
    llvm::Type *ret_tp;
    if (m_dist_func_state == DFSTATE_INIT) {
        ret_tp = m_type_mapper.get_void_type();
    } else {
        ret_tp = lookup_type(lambda->get_return_type());
    }
    MDL_ASSERT(ret_tp != NULL);

    llvm::Type *real_ret_tp = ret_tp;

    mi::mdl::vector<llvm::Type *>::Type arg_types(get_allocator());

    bool is_sret_func = m_dist_func_state != DFSTATE_INIT &&
        (m_lambda_force_sret || need_reference_return(lambda->get_return_type(), -1));

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
    if (m_lambda_force_render_state || lambda->uses_varying_state()) {
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
    set_llvm_function_attributes(func, /*mark_noinline=*/false);
    m_state_usage_analysis.register_function(func);

    if (is_entry_point) {
        func->setCallingConv(llvm::CallingConv::C);
    }

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
            func->addParamAttr(arg_it->getArgNo(), llvm::Attribute::NoCapture);
        }
        ++arg_it;
    }
    if (flags & LLVM_context_data::FL_HAS_STATE) {
        arg_it->setName("state");

        // the state pointer does not alias and is not captured
        func->addParamAttr(arg_it->getArgNo(), llvm::Attribute::NoAlias);
        func->addParamAttr(arg_it->getArgNo(), llvm::Attribute::NoCapture);
        ++arg_it;
    }
    if (flags & LLVM_context_data::FL_HAS_RES) {
        if (target_supports_pointers()) {
            arg_it->setName("res_data_pair");
        } else {
            arg_it->setName("res_data");
        }

        // the resource data pointer does not alias and is not captured
        func->addParamAttr(arg_it->getArgNo(), llvm::Attribute::NoAlias);
        func->addParamAttr(arg_it->getArgNo(), llvm::Attribute::NoCapture);
        ++arg_it;
    }
    if (flags & LLVM_context_data::FL_HAS_EXC) {
        arg_it->setName("exc_state");

        // the exc_data pointer does not alias and is not captured
        func->addParamAttr(arg_it->getArgNo(), llvm::Attribute::NoAlias);
        func->addParamAttr(arg_it->getArgNo(), llvm::Attribute::NoCapture);
        ++arg_it;
    }
    if (flags & LLVM_context_data::FL_HAS_CAP_ARGS) {
        arg_it->setName("captured_arguments");

        // the cap_args pointer does not alias and is not captured
        func->addParamAttr(arg_it->getArgNo(), llvm::Attribute::NoAlias);
        func->addParamAttr(arg_it->getArgNo(), llvm::Attribute::NoCapture);
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

    if ((in_flags & LLVM_context_data::FL_UNALIGNED_RET) != 0) {
        flags |= LLVM_context_data::FL_UNALIGNED_RET;
    }

    if (target_supports_lambda_results_parameter() &&
        (in_flags & LLVM_context_data::FL_HAS_LMBD_RES) != 0)
    {
        flags |= LLVM_context_data::FL_HAS_LMBD_RES;
    }


    size_t n_params = int_func->get_parameter_number();
    for (size_t i = 0; i < n_params; ++i) {
        IType const *p_type = int_func->get_parameter_type(i);

        // instantiate parameter types
        int arr_size = inst.instantiate_type_size(p_type);
        llvm::Type *tp = lookup_type(p_type, arr_size);

        if (is_passed_by_reference(p_type, arr_size)) {
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

    // FIXME: can the function be native here?
    set_llvm_function_attributes(func, /*mark_noinline=*/false);

    add_generated_attributes(func);

    if (is_entry_point) {
        func->setCallingConv(llvm::CallingConv::C);
    }

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
            func->addParamAttr(arg_it->getArgNo(), llvm::Attribute::NoCapture);
        }
        ++arg_it;
    }
    if ((flags & LLVM_context_data::FL_HAS_EXEC_CTX) != 0) {
        arg_it->setName("execution_ctx");

        // the execution context pointer does not alias and is not captured
        func->addParamAttr(arg_it->getArgNo(), llvm::Attribute::NoAlias);
        func->addParamAttr(arg_it->getArgNo(), llvm::Attribute::NoCapture);
        ++arg_it;
    } else {
        if ((flags & LLVM_context_data::FL_HAS_STATE) != 0) {
            arg_it->setName("state");

            // the state pointer does not alias and is not captured
            func->addParamAttr(arg_it->getArgNo(), llvm::Attribute::NoAlias);
            func->addParamAttr(arg_it->getArgNo(), llvm::Attribute::NoCapture);
            ++arg_it;
        }
        if ((flags & LLVM_context_data::FL_HAS_RES) != 0) {
            if (target_supports_pointers())
                arg_it->setName("res_data_pair");
            else
                arg_it->setName("res_data");

            // the resource data pointer does not alias and is not captured
            func->addParamAttr(arg_it->getArgNo(), llvm::Attribute::NoAlias);
            func->addParamAttr(arg_it->getArgNo(), llvm::Attribute::NoCapture);
            ++arg_it;
        }
        if ((flags & LLVM_context_data::FL_HAS_EXC) != 0) {
            arg_it->setName("exc_state");

            // the exc_state pointer does not alias and is not captured
            func->addParamAttr(arg_it->getArgNo(), llvm::Attribute::NoAlias);
            func->addParamAttr(arg_it->getArgNo(), llvm::Attribute::NoCapture);
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

        // parameters are not captured
        if (arg_it->getType()->isPointerTy()) {
            func->addParamAttr(arg_it->getArgNo(), llvm::Attribute::NoCapture);
        }
    }

    return m_arena_builder.create<LLVM_context_data>(func, real_ret_tp, flags);
}

// Retrieve the LLVM context data for a MDL function instance, create it if not available.
LLVM_context_data *LLVM_code_generator::get_or_create_context_data(
    mi::mdl::Module const   *owner,
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
    Function_instance inst(get_allocator(), lambda, target_supports_storage_spaces());

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
    Function_instance inst(
        get_allocator(),
        def,
        return_derivs,
        target_supports_storage_spaces());

    LLVM_context_data::Flags flags = get_function_flags(inst, def);
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
    llvm::Function          *func)
{
    Function_instance inst(
        get_allocator(),
        reinterpret_cast<size_t>(int_func),
        target_supports_storage_spaces());

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
        mi::mdl::IParameter const  *param      = func_decl->get_parameter(i);
        mi::mdl::IDefinition const *p_def      = param->get_name()->get_definition();
        mi::mdl::IType const       *p_type     = p_def->get_type();
        int                        arr_size    = inst.instantiate_type_size(p_type);
        Storage_modifier           storage_mod = inst.get_parameter_storage_modifier(i);
        bool arg_is_deriv =
            m_cur_func_deriv_info != NULL &&
            m_cur_func_deriv_info->args_want_derivatives.test_bit(i + 1);

        bool                       by_ref   = storage_mod == SM_NORMAL && (
            is_passed_by_reference(p_type, arr_size) ||
            (target_supports_pointers() && arg_is_deriv));
        LLVM_context_data          *p_data;

        if (need_storage_for_var(p_def)) {
            // must allocate a shadow copy, because this parameter MIGHT be written to
            // or debug mode is active
            p_data = context.get_context_data(p_def);

            llvm::Value *init = NULL;
            switch (storage_mod) {
            case SM_NORMAL:
                // copy value into shadow copy
                init = by_ref ?
                    llvm::cast<llvm::Value>(context->CreateLoad(arg_it)) :
                    llvm::cast<llvm::Value>(arg_it);
                break;
            case SM_PARAMETER:
            case SM_RODATA:
                {
                    Expression_result init_res = Expression_result::offset(
                        arg_it,
                        storage_mod == SM_PARAMETER
                            ? Expression_result::OK_ARG_BLOCK
                            : Expression_result::OK_RO_DATA_SEGMENT,
                        lookup_type(p_type, arr_size),
                        p_type);

                    // material parameters and constants are never stored with derivatives,
                    // so expand to derivative value if needed
                    if (arg_is_deriv) {
                        init_res.ensure_deriv_result(context, true);
                    }

                    init = init_res.as_value(context);
                }
                break;
            }
            p_data->set_var_value(init);
        } else {
            // this parameter will never be written
            p_data = context.create_context_data(
                p_def, arg_it, by_ref, storage_mod);
        }
        context.make_accessible(param);
    }

    // translate function body
    mi::mdl::IStatement const *body = func_decl->get_body();
    if (mi::mdl::IStatement_compound const *block = as<mi::mdl::IStatement_compound>(body)) {
        for (size_t i = 0, n = block->get_statement_count(); i < n; ++i) {
            mi::mdl::IStatement const *stmt = block->get_statement(i);

            translate_statement(stmt);
        }
    } else if (mi::mdl::IStatement_expression const *e_stmt = as<IStatement_expression>(body)) {
        // single expression body
        mi::mdl::IExpression const *expr = e_stmt->get_expression();

        bool returns_derivs = context.is_deriv_type(context.get_return_type());
        Expression_result res = translate_expression(expr, returns_derivs);
        res.ensure_deriv_result(context, returns_derivs);
        context.create_return(res.as_value(context));
    }
}

// Get the top level module on the stack.
mi::mdl::Module const *LLVM_code_generator::tos_module() const
{
    MDL_ASSERT(m_module_stack.begin() != m_module_stack.end());
    return m_module_stack.back();
}

// Push a module on the stack.
void LLVM_code_generator::push_module(mi::mdl::Module const *module)
{
    // Note: inside a distribution function, other lambdas are called. these lambdas do not have
    // a MDL source module, hence NULL can happen here
    if (module != NULL) {
        // ensure that all modules ON the stack have their import entries restored
        module->restore_import_entries(m_module_cache);
    }
    m_module_stack.push_back(module);
}

// Pop a module from the stack.
void LLVM_code_generator::pop_module()
{
    // Note: inside a distribution function, other lambdas are called. these lambdas do not have
    // a MDL source module, hence NULL can happen here
    if (mi::mdl::Module const *module = m_module_stack.back()) {
        // drop the import entries again once removed from the stack
        module->drop_import_entries();
    }
    m_module_stack.pop_back();
}

// Create extra instructions at function start for debugging.
void LLVM_code_generator::create_enter_function_call(
    llvm::Function *entered)
{
    if (m_jit_dbg_mode == JDBG_NONE || m_target_lang != ICode_generator::TL_NATIVE) {
        return;
    }

    Function_context &ctx = *m_ctx;

    llvm::StringRef name_ref = entered->getName();

    llvm::Value *args[] = {
        // pass the name of the entered function
        ctx->CreateGlobalStringPtr(name_ref),
        // pass its address, casted to void pointer
        //ctx->CreateBitCast(entered, m_type_mapper.get_void_ptr_type())
    };
    llvm::Function *func = get_enter_func();
    ctx->CreateCall(func, args);
}

// Translate a statement to LLVM IR.
void LLVM_code_generator::translate_statement(
    mi::mdl::IStatement const *stmt)
{
    Function_context &ctx = *m_ctx;
    ctx.set_curr_pos(stmt->access_position());

    switch (stmt->get_kind()) {
    case mi::mdl::IStatement::SK_INVALID:
        // should not occur
        break;
    case mi::mdl::IStatement::SK_COMPOUND:
        translate_block(cast<mi::mdl::IStatement_compound>(stmt));
        return;
    case mi::mdl::IStatement::SK_DECLARATION:
        translate_decl_stmt(cast<mi::mdl::IStatement_declaration>(stmt));
        return;
    case mi::mdl::IStatement::SK_EXPRESSION:
        {
            mi::mdl::IStatement_expression const *expr_stmt =
                cast<mi::mdl::IStatement_expression>(stmt);
            mi::mdl::IExpression const *expr = expr_stmt->get_expression();
            if (expr != NULL) {
                translate_expression(expr, /*wants_derivs=*/ false);
            }
        }
        return;
    case mi::mdl::IStatement::SK_IF:
        translate_if(cast<mi::mdl::IStatement_if>(stmt));
        return;
    case mi::mdl::IStatement::SK_CASE:
        // should not occur at top-level
        MDL_ASSERT(!"case statment not inside a switch");
        break;
    case mi::mdl::IStatement::SK_SWITCH:
        translate_switch(cast<mi::mdl::IStatement_switch>(stmt));
        return;
    case mi::mdl::IStatement::SK_WHILE:
        translate_while(cast<mi::mdl::IStatement_while>(stmt));
        return;
    case mi::mdl::IStatement::SK_DO_WHILE:
        translate_do_while(cast<mi::mdl::IStatement_do_while>(stmt));
        return;
    case mi::mdl::IStatement::SK_FOR:
        translate_for(cast<mi::mdl::IStatement_for>(stmt));
        return;
    case mi::mdl::IStatement::SK_BREAK:
        translate_break(cast<mi::mdl::IStatement_break>(stmt));
        return;
    case mi::mdl::IStatement::SK_CONTINUE:
        translate_continue(cast<mi::mdl::IStatement_continue>(stmt));
        return;
    case mi::mdl::IStatement::SK_RETURN:
        translate_return(cast<mi::mdl::IStatement_return>(stmt));
        return;
    }
    MDL_ASSERT(!"unsupported statement kind");
}

// Translate a block statement to LLVM IR.
void LLVM_code_generator::translate_block(
    mi::mdl::IStatement_compound const *block)
{
    Function_context &ctx = *m_ctx;
    Function_context::Block_scope block_scope(ctx, block);

    for (size_t i = 0, n = block->get_statement_count(); i < n; ++i) {
        mi::mdl::IStatement const *stmt = block->get_statement(i);
        translate_statement(stmt);
    }
}

// Translate a declaration statement to LLVM IR.
void LLVM_code_generator::translate_decl_stmt(
    mi::mdl::IStatement_declaration const *decl_stmt)
{
    mi::mdl::IDeclaration const *decl = decl_stmt->get_declaration();
    translate_declaration(decl);
}

// Translate a declaration to LLVM IR.
void LLVM_code_generator::translate_declaration(
    mi::mdl::IDeclaration const *decl)
{
    switch (decl->get_kind()) {
    case mi::mdl::IDeclaration::DK_INVALID:
        // should not occur
        break;

    case mi::mdl::IDeclaration::DK_IMPORT:
    case mi::mdl::IDeclaration::DK_ANNOTATION:
    case mi::mdl::IDeclaration::DK_STRUCT_CATEGORY:
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
        return translate_var_declaration(cast<mi::mdl::IDeclaration_variable>(decl));

    case mi::mdl::IDeclaration::DK_FUNCTION:
        // should not occur, nested functions are not allowed in MDL
        MDL_ASSERT(!"nested functions mot supported");
        return;
    }
    MDL_ASSERT(!"unsupported declaration kind");
}

// Translate a variable declaration to LLVM IR.
void LLVM_code_generator::translate_var_declaration(
    mi::mdl::IDeclaration_variable const *var_decl)
{
    Function_context &ctx = *m_ctx;

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
        } else {
            // in all other case there must be an init expression
            init = translate_expression(init_expr, is_deriv_var);
        }
        init.ensure_deriv_result(ctx, is_deriv_var);

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
                    ctx.create_context_data(var_def, cv, /*by_reference=*/true, SM_NORMAL);
                } else if (init.is_offset()) {
                    ctx.create_context_data(
                        var_def,
                        init.get_offset(),
                        /*by_reference=*/true,
                        init.get_storage_modifier());
                } else {
                    // is already a global constant
                    ctx.create_context_data(
                        var_def,
                        init.as_ptr(ctx),
                        /*by_reference=*/true,
                        SM_NORMAL);
                }
            } else {
                // handle as a real value
                ctx.create_context_data(
                    var_def, init.as_value(ctx), /*by_reference=*/false, SM_NORMAL);
            }
        }
    }
}

// Create a branch from a boolean expression (with short cut evaluation).
void LLVM_code_generator::translate_boolean_branch(
    mi::mdl::IExpression const *cond,
    llvm::BasicBlock           *true_bb,
    llvm::BasicBlock           *false_bb)
{
    Function_context &ctx = *m_ctx;

    if (mi::mdl::IExpression_binary const *bin_expr = as<mi::mdl::IExpression_binary>(cond)) {
        mi::mdl::IExpression_binary::Operator op = bin_expr->get_operator();

        if (op == mi::mdl::IExpression_binary::OK_LOGICAL_AND) {
            // shortcut AND evaluation
            llvm::BasicBlock *imm_bb = ctx.create_bb("shortcut_imm");

            mi::mdl::IExpression const *lhs = bin_expr->get_left_argument();
            translate_boolean_branch(lhs, imm_bb, false_bb);

            ctx->SetInsertPoint(imm_bb);
            mi::mdl::IExpression const *rhs = bin_expr->get_right_argument();
            translate_boolean_branch(rhs, true_bb, false_bb);
            return;
        } else if (op == mi::mdl::IExpression_binary::OK_LOGICAL_OR) {
            // shortcut OR evaluation
            llvm::BasicBlock *imm_bb = ctx.create_bb("shortcut_imm");

            mi::mdl::IExpression const *lhs = bin_expr->get_left_argument();
            translate_boolean_branch(lhs, true_bb, imm_bb);

            ctx->SetInsertPoint(imm_bb);
            mi::mdl::IExpression const *rhs = bin_expr->get_right_argument();
            translate_boolean_branch(rhs, true_bb, false_bb);
            return;
        }
    } else if (mi::mdl::IExpression_unary const *un_expr = as<mi::mdl::IExpression_unary>(cond)) {
        if (un_expr->get_operator() == mi::mdl::IExpression_unary::OK_LOGICAL_NOT) {
            // logical not: just exchange true and false targets
            mi::mdl::IExpression const *arg = un_expr->get_argument();
            translate_boolean_branch(arg, false_bb, true_bb); //-V764
            return;
        }
    }
    // default case
    llvm::Value *c = translate_expression_value(cond, /*wants_derivs=*/ false);

    if (c->getType() != m_type_mapper.get_predicate_type()) {
        // map to predicate type
        c = ctx->CreateICmpNE(c, ctx.get_constant(false));
    }

    ctx->CreateCondBr(c, true_bb, false_bb);
}

// Create a branch from a boolean expression (with short cut evaluation).
void LLVM_code_generator::translate_boolean_branch(
    mi::mdl::ICall_name_resolver const *resolver,
    mi::mdl::DAG_node const            *cond,
    llvm::BasicBlock                   *true_bb,
    llvm::BasicBlock                   *false_bb)
{
    Function_context &ctx = *m_ctx;

    if (mi::mdl::DAG_call const *call = as<mi::mdl::DAG_call>(cond)) {
        mi::mdl::IDefinition::Semantics sema = call->get_semantic();
        if (mi::mdl::semantic_is_operator(sema)) {
            mi::mdl::IExpression::Operator op = mi::mdl::semantic_to_operator(sema);

            if (op == mi::mdl::IExpression::OK_LOGICAL_AND) {
                // shortcut AND evaluation
                llvm::BasicBlock *imm_bb = ctx.create_bb("shortcut_imm");

                mi::mdl::DAG_node const *lhs = call->get_argument(0);
                translate_boolean_branch(resolver, lhs, imm_bb, false_bb);

                BB_store chain(m_curr_bb, get_next_bb());
                ctx->SetInsertPoint(imm_bb);
                mi::mdl::DAG_node const *rhs = call->get_argument(1);
                translate_boolean_branch(resolver, rhs, true_bb, false_bb);
                return;
            } else if (op == mi::mdl::IExpression::OK_LOGICAL_OR) {
                // shortcut OR evaluation
                llvm::BasicBlock *imm_bb = ctx.create_bb("shortcut_imm");

                mi::mdl::DAG_node const *lhs = call->get_argument(0);
                translate_boolean_branch(resolver, lhs, true_bb, imm_bb);

                BB_store chain(m_curr_bb, get_next_bb());
                ctx->SetInsertPoint(imm_bb);
                mi::mdl::DAG_node const *rhs = call->get_argument(1);
                translate_boolean_branch(resolver, rhs, true_bb, false_bb);
                return;
            } else if (op == mi::mdl::IExpression::OK_LOGICAL_NOT) {
                // logical not: just exchange true and false targets
                mi::mdl::DAG_node const *arg = call->get_argument(0);
                translate_boolean_branch(resolver, arg, false_bb, true_bb);
                return;
            }
        }
    }
    // default case
    llvm::Value *c = translate_node(cond, resolver).as_value(ctx);

    if (c->getType() != m_type_mapper.get_predicate_type()) {
        // map to predicate type
        c = ctx->CreateICmpNE(c, ctx.get_constant(false));
    }

    ctx->CreateCondBr(c, true_bb, false_bb);
}

// Translate an if statement to LLVM IR.
void LLVM_code_generator::translate_if(
    mi::mdl::IStatement_if const *if_stmt)
{
    Function_context &ctx = *m_ctx;

    mi::mdl::IExpression const *cond = if_stmt->get_condition();
    mi::mdl::IStatement const  *then_stmt = if_stmt->get_then_statement();
    mi::mdl::IStatement const  *else_stmt = if_stmt->get_else_statement();

    llvm::BasicBlock *true_bb  = ctx.create_bb("if_on_true");
    llvm::BasicBlock *end_bb   = ctx.create_bb("if_end");
    llvm::BasicBlock *false_bb = else_stmt == NULL ? end_bb : ctx.create_bb("if_on_false");

    translate_boolean_branch(cond, true_bb, false_bb);

    // create the then statement
    ctx->SetInsertPoint(true_bb);
    {
        Function_context::Block_scope block_scope(ctx, then_stmt);
        translate_statement(then_stmt);
    }
    ctx->CreateBr(end_bb);

    if (else_stmt != NULL) {
        // create the else statement
        ctx->SetInsertPoint(false_bb);
        {
            Function_context::Block_scope block_scope(ctx, else_stmt);
            translate_statement(else_stmt);
        }
        ctx->CreateBr(end_bb);
    }
    ctx->SetInsertPoint(end_bb);
}

// Translate a switch statement to LLVM IR.
void LLVM_code_generator::translate_switch(
    mi::mdl::IStatement_switch const *switch_stmt)
{
    Function_context &ctx = *m_ctx;

    size_t n_cases = switch_stmt->get_case_count();

    // find the default case if any
    mi::mdl::IStatement_case const *default_case = NULL;
    for (size_t i = 0; i < n_cases; ++i) {
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
        switch_stmt->get_condition(), /*wants_derivs=*/ false);

    llvm::SwitchInst *switch_instr = ctx->CreateSwitch(expr, default_bb, unsigned(n_cases));

    ctx->SetInsertPoint(ctx.get_unreachable_bb());

    for (size_t i = 0; i < n_cases; ++i) {
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
            translate_block(case_stmt);
        }
    }
    // fall-through from last case
    ctx->CreateBr(end_bb);
    ctx->SetInsertPoint(end_bb);
}

// Translate a while statement to LLVM IR.
void LLVM_code_generator::translate_while(
    mi::mdl::IStatement_while const *while_stmt)
{
    Function_context &ctx = *m_ctx;

    llvm::BasicBlock *start_bb = ctx.create_bb("while_condition");
    llvm::BasicBlock *body_bb  = ctx.create_bb("while_body");
    llvm::BasicBlock *end_bb   = ctx.create_bb("after_while");

    Function_context::Break_destination    break_scope(ctx, end_bb);
    Function_context::Continue_destination cont_scope(ctx, start_bb);

    ctx->CreateBr(start_bb);

    // create check-abort-condition code
    ctx->SetInsertPoint(start_bb);

    mi::mdl::IExpression const *cond = while_stmt->get_condition();
    translate_boolean_branch(cond, body_bb, end_bb);

    // create loop body code
    ctx->SetInsertPoint(body_bb);
    {
        Function_context::Block_scope block_scope(ctx, while_stmt);

        mi::mdl::IStatement const *body = while_stmt->get_body();
        translate_statement(body);
    }
    ctx->CreateBr(start_bb);

    // after while
    ctx->SetInsertPoint(end_bb);
}

// Translate a do-while statement to LLVM IR.
void LLVM_code_generator::translate_do_while(
    mi::mdl::IStatement_do_while const *do_stmt)
{
    Function_context &ctx = *m_ctx;

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
        translate_statement(body);
    }
    ctx->CreateBr(cond_bb);

    // create check-abort-condition code
    ctx->SetInsertPoint(cond_bb);
    mi::mdl::IExpression const *cond = do_stmt->get_condition();
    translate_boolean_branch(cond, body_bb, end_bb);

    // after do-while
    ctx->SetInsertPoint(end_bb);
}

// Translate a for statement to LLVM IR.
void LLVM_code_generator::translate_for(
    mi::mdl::IStatement_for const *for_stmt)
{
    Function_context &ctx = *m_ctx;

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
        translate_statement(init);
    }
    ctx->CreateBr(test_bb);

    // create test block if needed
    if (cond != NULL) {
        ctx->SetInsertPoint(test_bb);
        translate_boolean_branch(cond, body_bb, end_bb);
    }

    // create body code
    ctx->SetInsertPoint(body_bb);
    {
        mi::mdl::IStatement const *body = for_stmt->get_body();

        Function_context::Block_scope body_block_scope(ctx, body);
        translate_statement(body);
    }
    ctx->CreateBr(upd_bb);

    // create update block if needed
    if (upd != NULL) {
        ctx->SetInsertPoint(upd_bb);
        translate_expression(upd, /*wants_derivs=*/ false);
        ctx->CreateBr(test_bb);
    }

    // after for
    ctx->SetInsertPoint(end_bb);
}

// Translate a break statement to LLVM IR.
void LLVM_code_generator::translate_break(
    mi::mdl::IStatement_break const *break_stmt)
{
    Function_context &ctx = *m_ctx;

    ctx.create_jmp(ctx.tos_break());
}

// Translate a continue statement to LLVM IR.
void LLVM_code_generator::translate_continue(
    mi::mdl::IStatement_continue const *cont_stmt)
{
    Function_context &ctx = *m_ctx;

    ctx.create_jmp(ctx.tos_continue());
}

// Translate a return statement to LLVM IR.
void LLVM_code_generator::translate_return(
    mi::mdl::IStatement_return const *ret_stmt)
{
    Function_context &ctx = *m_ctx;

    if (mi::mdl::IExpression const *expr = ret_stmt->get_expression()) {
        llvm::Type *return_type   = ctx.get_return_type();
        bool       returns_derivs = ctx.is_deriv_type(return_type);

        Expression_result res = translate_expression(expr, returns_derivs);
        res.ensure_deriv_result(ctx, returns_derivs);
        ctx.create_return(res.as_value(ctx));
    } else {
        ctx.create_void_return();
    }
}

// Calculate &matrix[index], index is assured to be in bounds
llvm::Value *LLVM_code_generator::calc_matrix_index_in_bounds(
    mi::mdl::IType_matrix const *m_type,
    llvm::Value                 *matrix_ptr,
    llvm::Value                 *index)
{
    Function_context &ctx = *m_ctx;

    llvm::Type      *tp   = lookup_type(m_type);
    llvm::ArrayType *a_tp = llvm::cast<llvm::ArrayType>(tp);
    llvm::Type      *e_tp = a_tp->getArrayElementType();

    if (e_tp->isVectorTy()) {
        // matrix types are represented as array of vectors, so the index
        // directly access the vector
        return ctx.create_simple_gep_in_bounds(matrix_ptr, index);
    } else {
        // matrix types are represented as array of scalars
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
    llvm::Value      *index,
    llvm::Value      *bound)
{
    Function_context &ctx = *m_ctx;

    // with instancing, all arrays have at least size 1, so we map out of bounds accesses to index 0
    if (m_bounds_check_exception_disabled && m_enable_instancing) {
        return ctx.create_select_if_in_bounds(
            index, bound, index, llvm::Constant::getNullValue(index->getType()));
    }
    return index;
}

// Translate an l-value index expression to LLVM IR.
llvm::Value *LLVM_code_generator::translate_lval_index_expression(
    mi::mdl::IType const    *comp_type,
    llvm::Value             *comp_ptr,
    llvm::Value             *index,
    mi::mdl::Position const *index_pos)
{
    Function_context &ctx = *m_ctx;

    // instantiate type
    int imm_size = ctx.instantiate_type_size(comp_type);

    // determine bound and element pointer depending on kind of compound
    llvm::Value *bound;
    llvm::Value *elem_ptr;
    if (mi::mdl::IType_array const *a_type = as<mi::mdl::IType_array>(comp_type)) {
        if (!a_type->is_immediate_sized() && imm_size < 0) {
            // generate bounds check for deferred sized array
            bound = ctx.get_deferred_size_from_ptr(comp_ptr);
            index = adapt_index_for_bounds_check(index, bound);

            // array_desc<T> access
            llvm::Value *base = ctx.get_deferred_base_from_ptr(comp_ptr);
            elem_ptr = ctx->CreateInBoundsGEP(base, index);
        } else {
            // generate bounds check for immediate sized array
            size_t arr_size = imm_size >= 0 ? imm_size : a_type->get_size();
            bound = ctx.get_constant(arr_size);
            index = adapt_index_for_bounds_check(index, bound);

            elem_ptr = ctx.create_simple_gep_in_bounds(comp_ptr, index);
        }
    } else if (mi::mdl::IType_matrix const *m_type = as<mi::mdl::IType_matrix>(comp_type)) {
        // generate bounds check for matrices
        bound = ctx.get_constant(size_t(m_type->get_columns()));
        index = adapt_index_for_bounds_check(index, bound);

        elem_ptr = calc_matrix_index_in_bounds(m_type, comp_ptr, index);
    } else {
        // generate bounds check for vector type
        mi::mdl::IType_vector const *v_type = cast<mi::mdl::IType_vector>(comp_type);
        bound = ctx.get_constant(size_t(v_type->get_size()));
        index = adapt_index_for_bounds_check(index, bound);

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
    Function_context &ctx = *m_ctx;

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
            index = adapt_index_for_bounds_check(index, bound);

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
            index = adapt_index_for_bounds_check(index, bound);

            adr_val = ctx.create_simple_gep_in_bounds(comp_val_ptr, index);
            adr_dx  = ctx.create_simple_gep_in_bounds(comp_dx_ptr, index);
            adr_dy  = ctx.create_simple_gep_in_bounds(comp_dy_ptr, index);
        }
    } else if (mi::mdl::IType_matrix const *m_type = as<mi::mdl::IType_matrix>(comp_type)) {
        // generate bounds check for matrices
        bound = ctx.get_constant(size_t(m_type->get_columns()));
        index = adapt_index_for_bounds_check(index, bound);

        adr_val = calc_matrix_index_in_bounds(m_type, comp_val_ptr, index);
        adr_dx  = calc_matrix_index_in_bounds(m_type, comp_dx_ptr, index);
        adr_dy  = calc_matrix_index_in_bounds(m_type, comp_dy_ptr, index);
    } else {
        // generate bounds check for vector type
        mi::mdl::IType_vector const *v_type = cast<mi::mdl::IType_vector>(comp_type);
        bound = ctx.get_constant(size_t(v_type->get_size()));
        index = adapt_index_for_bounds_check(index, bound);

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
    mi::mdl::IType const    *comp_type,
    Expression_result       comp,
    llvm::Value             *index,
    mi::mdl::Position const *index_pos)
{
    Function_context &ctx = *m_ctx;

    // handle LLVM result with derivatives
    if (m_type_mapper.is_deriv_type(comp.get_value_type())) {
        mi::mdl::IType const *val_type = m_type_mapper.skip_deriv_type(comp_type);

        Expression_result val = Expression_result::value(ctx.get_dual_val(comp.as_value(ctx)));
        Expression_result val_elem = translate_index_expression(
            val_type, val, index, index_pos);

        Expression_result dx = Expression_result::value(ctx.get_dual_dx(comp.as_value(ctx)));
        Expression_result dx_elem = translate_index_expression(
            val_type, dx, index, index_pos);

        Expression_result dy = Expression_result::value(ctx.get_dual_dy(comp.as_value(ctx)));
        Expression_result dy_elem = translate_index_expression(
            val_type, dy, index, index_pos);

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
            val_type, comp, index, index_pos);
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
            index = adapt_index_for_bounds_check(index, bound);

            // array_desc<T> access
            llvm::Value *base = ctx.get_deferred_base(compound);
            elem_ptr = ctx->CreateInBoundsGEP(base, index);
        } else {
            // generate bounds check for immediate sized array
            size_t arr_size = imm_size >= 0 ? imm_size : a_type->get_size();
            bound = ctx.get_constant(arr_size);
            index = adapt_index_for_bounds_check(index, bound);

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

                int cur_offs = 0;
                return translate_sl_value(
                    comp.get_offset_kind() == Expression_result::OK_RO_DATA_SEGMENT
                    ? SM_RODATA : SM_PARAMETER,
                    a_type->get_element_type(),
                    cur_offs,
                    offs);
            }

            elem_ptr = ctx.create_simple_gep_in_bounds(comp.as_ptr(ctx), index);
        }
    } else if (mi::mdl::IType_matrix const *m_type = as<mi::mdl::IType_matrix>(comp_type)) {
        // generate bounds check for matrices
        bound = ctx.get_constant(size_t(m_type->get_columns()));
        index = adapt_index_for_bounds_check(index, bound);

        elem_ptr = calc_matrix_index_in_bounds(m_type, comp.as_ptr(ctx), index);
    } else {
        // generate bounds check for vector types
        mi::mdl::IType_vector const *v_type = cast<mi::mdl::IType_vector>(comp_type);
        bound = ctx.get_constant(size_t(v_type->get_size()));
        index = adapt_index_for_bounds_check(index, bound);

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
    mi::mdl::IExpression const *expr)
{
    Function_context &ctx = *m_ctx;

    switch (expr->get_kind()) {
    case mi::mdl::IExpression::EK_REFERENCE:
        {
            mi::mdl::IExpression_reference const *ref = cast<mi::mdl::IExpression_reference>(expr);
            mi::mdl::IDefinition const           *def = ref->get_definition();

            MDL_ASSERT(def != NULL && "Array constructor unexpected here");

            switch (def->get_kind()) {
            case mi::mdl::IDefinition::DK_PARAMETER:
            case mi::mdl::IDefinition::DK_VARIABLE:
                if (LLVM_context_data *data = ctx.lookup_context_data(def)) {
                    // l-values cannot be in argblock or read-only data
                    MDL_ASSERT(data->get_var_storage_modifier() == SM_NORMAL);

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
                    llvm::Value             *comp_ptr  = translate_lval_expression(lhs);
                    llvm::Value             *index     =
                        translate_expression_value(rhs, /*wants_derivs=*/ false);
                    mi::mdl::Position const *index_pos = &rhs->access_position();
                    mi::mdl::IType const    *comp_type = lhs->get_type()->skip_type_alias();

                    return translate_lval_index_expression(
                        comp_type, comp_ptr, index, index_pos);
                }

            case mi::mdl::IExpression_binary::OK_SELECT:
                {
                    mi::mdl::IExpression_reference const *ref =
                        cast<mi::mdl::IExpression_reference>(rhs);
                    mi::mdl::IDefinition const           *def = ref->get_definition();

                    MDL_ASSERT(def->get_kind() == mi::mdl::IDefinition::DK_MEMBER);

                    llvm::Value *l   = translate_lval_expression(lhs);
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

    error(INTERNAL_JIT_BACKEND_ERROR, "Unsupported lvalue expression");
    MDL_ASSERT(!"unsupported lvalue expression");
    return llvm::UndefValue::get(lookup_type(expr->get_type()));
}

// Translate a dual l-value expression to LLVM IR.
void LLVM_code_generator::translate_lval_expression_dual(
    mi::mdl::IExpression const *expr,
    llvm::Value                *&adr_val,
    llvm::Value                *&adr_dx,
    llvm::Value                *&adr_dy)
{
    Function_context &ctx = *m_ctx;

    switch (expr->get_kind()) {
    case mi::mdl::IExpression::EK_REFERENCE:
        {
            mi::mdl::IExpression_reference const *ref = cast<mi::mdl::IExpression_reference>(expr);
            mi::mdl::IDefinition const           *def = ref->get_definition();

            MDL_ASSERT(def != NULL && "Array constructor unexpected here");

            switch (def->get_kind()) {
            case mi::mdl::IDefinition::DK_PARAMETER:
            case mi::mdl::IDefinition::DK_VARIABLE:
                if (LLVM_context_data *data = ctx.lookup_context_data(def)) {
                    // l-values cannot be in argblock or read-only data
                    MDL_ASSERT(data->get_var_storage_modifier() == SM_NORMAL);

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
            translate_lval_expression_dual(lhs, val_ptr, dx_ptr, dy_ptr);

            switch (bin_exp->get_operator()) {
            case mi::mdl::IExpression_binary::OK_ARRAY_INDEX:
                {
                    llvm::Value             *index     =
                        translate_expression_value(rhs, /*wants_derivs=*/ false);
                    mi::mdl::Position const *index_pos = &rhs->access_position();
                    mi::mdl::IType const    *comp_type = lhs->get_type()->skip_type_alias();

                    translate_lval_index_expression_dual(
                        comp_type, val_ptr, dx_ptr, dy_ptr, index, index_pos,
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

    error(INTERNAL_JIT_BACKEND_ERROR, "Unsupported lvalue expression");
    MDL_ASSERT(!"unsupported lvalue expression");
    adr_val = adr_dx = adr_dy = llvm::UndefValue::get(lookup_type(expr->get_type()));
}

// Translate an expression to LLVM IR, returning its value.
llvm::Value *LLVM_code_generator::translate_expression_value(
    mi::mdl::IExpression const *expr,
    bool                       wants_derivs)
{
    Function_context &ctx = *m_ctx;

    return translate_expression(expr, wants_derivs).as_value(ctx);
}

// Translate an (r-value) expression to LLVM IR.
Expression_result LLVM_code_generator::translate_expression(
    mi::mdl::IExpression const *expr,
    bool                       wants_derivs)
{
    Function_context &ctx = *m_ctx;

    ctx.set_curr_pos(expr->access_position());

    Expression_result res = Expression_result::unset();

    switch (expr->get_kind()) {
    case mi::mdl::IExpression::EK_INVALID:
        // should not occur
        break;
    case mi::mdl::IExpression::EK_LITERAL:
        res = translate_literal(cast<mi::mdl::IExpression_literal>(expr));
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
                if (LLVM_context_data *data = ctx.lookup_context_data(def)) {
                    switch (data->get_var_storage_modifier()) {
                    case SM_NORMAL:
                        if (llvm::Value *v = data->get_var_address()) {
                            res = Expression_result::ptr(v);
                        } else {
                            // variable is never changed, only a value
                            res = Expression_result::value(data->get_var_value());
                        }
                        break;
                    case SM_PARAMETER:
                    case SM_RODATA:
                        {
                            IType const *mdl_type = def->get_type();
                            llvm::Type *llvm_type =
                                lookup_type(mdl_type, ctx.instantiate_type_size(mdl_type));

                            res = Expression_result::offset(
                                data->get_var_value(),
                                data->get_var_storage_modifier() == SM_PARAMETER
                                    ? Expression_result::OK_ARG_BLOCK
                                    : Expression_result::OK_RO_DATA_SEGMENT,
                                llvm_type,
                                mdl_type);
                        }
                        break;
                    }
                }
                break;

            case mi::mdl::IDefinition::DK_CONSTANT:
                {
                    mi::mdl::IValue const *v = def->get_constant_value();
                    res = Expression_result::value(ctx.get_constant(v));
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
                    } else if (LLVM_context_data *p_data = ctx.lookup_context_data(param_def)) {
                        // still deferred size, retrieve it from the descriptor
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
        res = translate_unary(cast<mi::mdl::IExpression_unary>(expr), wants_derivs);
        break;

    case mi::mdl::IExpression::EK_BINARY:
        res = translate_binary(cast<mi::mdl::IExpression_binary>(expr), wants_derivs);
        break;

    case mi::mdl::IExpression::EK_CONDITIONAL:
        res = translate_conditional(cast<mi::mdl::IExpression_conditional>(expr), wants_derivs);
        break;

    case mi::mdl::IExpression::EK_CALL:
        {
            Call_ast_expr call(*this, cast<mi::mdl::IExpression_call>(expr));
            res = translate_call(&call);
        }
        break;

    case mi::mdl::IExpression::EK_LET:
        res = translate_let(cast<mi::mdl::IExpression_let>(expr), wants_derivs);
        break;
    }

    if (res.is_unset()) {
        error(INTERNAL_JIT_BACKEND_ERROR, "Unsupported expression");
        MDL_ASSERT(!"unsupported expression");
        res = Expression_result::undef(lookup_type(expr->get_type()));
    }

    res.strip_deriv_if_not_wanted(ctx, wants_derivs);

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

    case IType::TK_PTR:
    case IType::TK_REF:
    case IType::TK_VOID:
    case IType::TK_AUTO:
    case IType::TK_ERROR:
        return false;
    }
    return false;
}

// Find a tag for a given resource if available in the resource tag map.
int LLVM_code_generator::find_resource_tag(IValue_resource const *res) const
{
    if (m_resource_tag_map == NULL) {
        return 0;
    }

    // linear search
    Resource_tag_tuple::Kind kind = kind_from_value(res);
    char const               *url = res->get_string_value();
    char const               *sel = "";

    if (IValue_texture const *tex = as<IValue_texture>(res)) {
        sel = tex->get_selector();
    }

    for (size_t i = 0, n = m_resource_tag_map->size(); i < n; ++i) {
        Resource_tag_tuple const &e = (*m_resource_tag_map)[i];

        if (e.m_kind == kind &&
            (e.m_url      == url || strcmp(e.m_url,      url) == 0) &&
            (e.m_selector == sel || strcmp(e.m_selector, sel) == 0))
        {
            return e.m_tag;
        }
    }
    return 0;
}

// Returns true, if a value with the given type and size will be stored in the RO data segment.
bool LLVM_code_generator::is_stored_in_ro_data_segment(mi::mdl::IType const *type, uint64_t size)
{
    // Big data arrays slow down PTXAS and the JIT linker. Move them into a read-only
    // segment that we manage ourself.
    return size > m_max_const_size && can_be_stored_in_ro_segment(type);
}

// Returns true, if the given value will be stored in the RO data segment.
bool LLVM_code_generator::is_stored_in_ro_data_segment(mi::mdl::IValue const *v)
{
    if (!m_use_ro_data_segment)
        return false;

    // for derivative values, strip off the derivative part, which is always zero
    mi::mdl::IValue const *non_deriv_v = v;
    bool is_deriv = m_type_mapper.is_deriv_type(v->get_type());
    if (is_deriv) {
        non_deriv_v = as<mi::mdl::IValue_struct>(v)->get_value(0);
    }

    // we only consider arrays and structs for the RO data segment
    if (!is<mi::mdl::IValue_array>(non_deriv_v) && !is<mi::mdl::IValue_struct>(non_deriv_v))
        return false;

    mi::mdl::IType const *non_deriv_v_type = non_deriv_v->get_type();
    llvm::Type *tp  = m_type_mapper.lookup_type(m_llvm_context, non_deriv_v_type);

    llvm::DataLayout const *dl = get_target_layout_data();

    uint64_t size = dl->getTypeAllocSize(tp);
    return is_stored_in_ro_data_segment(non_deriv_v_type, size);
}

// Creates a global constant for a value in the LLVM IR.
llvm::Value *LLVM_code_generator::create_global_const(
    mi::mdl::IValue_compound const *v,
    bool                           &is_ro_segment_ofs)
{
    Function_context &ctx = *m_ctx;

    is_ro_segment_ofs = false;

    mi::mdl::IType const *v_type = v->get_type();
    llvm::Type *tp  = m_type_mapper.lookup_type(m_llvm_context, v_type);

    if (m_use_ro_data_segment) {
        llvm::DataLayout const *dl = get_target_layout_data();

        uint64_t size = dl->getTypeAllocSize(tp);

        // Big data arrays slow down PTXAS and the JIT linker. Move them into an read-only
        // segment that we manage ourself.
        if (is_stored_in_ro_data_segment(v_type, size)) {
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

// Translate a value to LLVM IR.
Expression_result LLVM_code_generator::translate_value(
    mi::mdl::IValue const *v)
{
    Function_context &ctx = *m_ctx;

    llvm::Value *res;
    if (mi::mdl::IValue_resource const *r = as<mi::mdl::IValue_resource>(v)) {
        size_t idx = ctx.get_resource_index(r);
        res = ctx.get_constant(int(idx));
    } else {
        // non-resource value

        // for derivative values, strip off the derivative part, which is always zero
        mi::mdl::IValue const *non_deriv_v = v;
        bool is_deriv = m_type_mapper.is_deriv_type(v->get_type());
        if (is_deriv) {
            non_deriv_v = as<mi::mdl::IValue_struct>(v)->get_value(0);
        }

        mi::mdl::IValue::Kind value_kind = non_deriv_v->get_kind();
        if (value_kind == mi::mdl::IValue::VK_ARRAY || value_kind == mi::mdl::IValue::VK_STRUCT) {
            // If arrays are handled as values, we generate currently
            // bad code because they are stored every time to access elements by index.
            // So create a constant in global space. The same is probably true for structs, so
            // handle them also this way here.

            bool is_ofs = false;
            Global_const_map::iterator it = m_global_const_map.find(non_deriv_v);
            if (it == m_global_const_map.end()) {
                // first time we see this constant, create it
                llvm::Value *cv = create_global_const(
                    cast<mi::mdl::IValue_compound>(non_deriv_v), is_ofs);

                // cache it
                m_global_const_map[non_deriv_v] = Value_offset_pair(cv, is_ofs);
                res = cv;
            } else {
                // was already created, get it from cache
                Value_offset_pair const &pair = it->second;
                res    = pair.value;
                is_ofs = pair.is_offset;
            }

            if (is_ofs) {
                // get from the RO segment
                mi::mdl::IType const *non_deriv_v_type = non_deriv_v->get_type();
                llvm::Type *tp = m_type_mapper.lookup_type(m_llvm_context, non_deriv_v_type);

                // for HLSL and GLSL we use user provided functions to read the value
                if (target_is_structured_language()) {
                    llvm::ConstantInt *ci = llvm::cast<llvm::ConstantInt>(res);
                    int cur_offs = int(ci->getZExtValue());
                    return Expression_result::offset(
                        ctx.get_constant(cur_offs),
                        Expression_result::OK_RO_DATA_SEGMENT,
                        tp,
                        non_deriv_v_type);
                }

                llvm::Value *ro_seg = get_ro_data_segment();
                llvm::Value *cv     = ctx->CreateGEP(ro_seg, res);

                // cast the blob to the right type
                llvm::PointerType *ptr_tp = m_type_mapper.get_ptr(tp);
                res = ctx->CreatePointerCast(cv, ptr_tp);
            }

            // this is an address
            Expression_result expr_res = Expression_result::ptr(res);

            // constants are never stored with derivatives,
            // so expand to derivative value if needed
            if (is_deriv) {
                expr_res.ensure_deriv_result(ctx, true);
            }
            return expr_res;
        }

        // non-array/struct
        res = ctx.get_constant(v);
    }
    return Expression_result::value(res);
}

// Translate a literal expression to LLVM IR.
Expression_result LLVM_code_generator::translate_literal(
    mi::mdl::IExpression_literal const *lit)
{
    Function_context &ctx = *m_ctx;

    mi::mdl::IValue const *v = lit->get_value();
    Expression_result res(translate_value(v));

    // need to return a derivative value?
    if (is_deriv_expr(lit)) {
        return Expression_result::value(ctx.get_dual(res.as_value(ctx)));
    }
    return res;
}

// Translate an unary expression without side-effect to LLVM IR.
llvm::Value *LLVM_code_generator::translate_unary(
    mi::mdl::IExpression_unary::Operator op,
    llvm::Value                          *arg)
{
    Function_context &ctx = *m_ctx;

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
                    llvm::Value *arg_dx = ctx.get_dual_dx(arg);
                    llvm::Value *arg_dy = ctx.get_dual_dy(arg);

                    llvm::Value *v_val = llvm::ConstantAggregateZero::get(arr_tp);
                    llvm::Value *v_dx = v_val;
                    llvm::Value *v_dy = v_val;

                    for (unsigned i = 0, n = unsigned(arr_tp->getNumElements()); i < n; ++i) {
                        unsigned idxs[1] = { i };
                        llvm::Value *tmp_val = ctx->CreateExtractValue(arg_val, idxs);
                        llvm::Value *tmp_dx = ctx->CreateExtractValue(arg_dx, idxs);
                        llvm::Value *tmp_dy = ctx->CreateExtractValue(arg_dy, idxs);

                        tmp_val = ctx->CreateFNeg(tmp_val);
                        tmp_dx = ctx->CreateFNeg(tmp_dx);
                        tmp_dy = ctx->CreateFNeg(tmp_dy);

                        v_val = ctx->CreateInsertValue(v_val, tmp_val, idxs);
                        v_dx = ctx->CreateInsertValue(v_dx, tmp_dx, idxs);
                        v_dy = ctx->CreateInsertValue(v_dy, tmp_dy, idxs);
                    }
                    v = ctx.get_dual(v_val, v_dx, v_dy);
                } else {
                    // vector/atomic
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
    mi::mdl::IExpression_unary const *un_expr,
    bool                             wants_derivs)
{
    mi::mdl::IExpression_unary::Operator op = un_expr->get_operator();
    switch (op) {
    case mi::mdl::IExpression_unary::OK_BITWISE_COMPLEMENT:
    case mi::mdl::IExpression_unary::OK_LOGICAL_NOT:
    case mi::mdl::IExpression_unary::OK_POSITIVE:
    case mi::mdl::IExpression_unary::OK_NEGATIVE:
        {
            llvm::Value *arg = translate_expression_value(
                un_expr->get_argument(), wants_derivs);
            return Expression_result::value(translate_unary(op, arg));
        }
        break;

    case mi::mdl::IExpression_unary::OK_PRE_INCREMENT:
    case mi::mdl::IExpression_unary::OK_PRE_DECREMENT:
    case mi::mdl::IExpression_unary::OK_POST_INCREMENT:
    case mi::mdl::IExpression_unary::OK_POST_DECREMENT:
        return translate_inplace_change_expression(un_expr);
    case mi::mdl::IExpression_unary::OK_CAST:
        return translate_cast_expression(un_expr, wants_derivs);
    }
    MDL_ASSERT(!"unsupported unary operator kind");
    return Expression_result::undef(lookup_type(un_expr->get_type()));
}

// Helper for pre/post inc/decrement.
void LLVM_code_generator::do_inner_inc_dec(
    mi::mdl::IExpression_unary::Operator op,
    llvm::Type                           *tp,
    llvm::Value                          *old_v,
    llvm::Value                          *&r,
    llvm::Value                          *&v)
{
    Function_context &ctx = *m_ctx;

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
    mi::mdl::IExpression_unary const *un_expr)
{
    Function_context &ctx = *m_ctx;

    mi::mdl::IExpression const *arg = un_expr->get_argument();
    llvm::Value                *adr = translate_lval_expression(arg);

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
            do_inner_inc_dec(op, e_tp, e_old_v, e_r, e_v);

            r = ctx->CreateInsertValue(r, e_r, idxes);
            v = ctx->CreateInsertValue(v, e_v, idxes);
        }
    } else {
        // scalar or vector type
        do_inner_inc_dec(op, tp, old_v, r, v);
    }

    ctx->CreateStore(v, adr);
    return Expression_result::value(r);
}

// Translate an MDL 1.5 cast expression.
Expression_result LLVM_code_generator::translate_cast_expression(
    Expression_result                &arg,
    mi::mdl::IType const             *res_type)
{
    Function_context &ctx = *m_ctx;

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
    mi::mdl::IExpression_unary const *un_expr,
    bool                             wants_derivs)
{
    Expression_result res = translate_expression(un_expr->get_argument(), wants_derivs);
    mi::mdl::IType const *res_type = un_expr->get_type()->skip_type_alias();

    return translate_cast_expression(res, res_type);
}

// Translate a binary expression to LLVM IR.
Expression_result LLVM_code_generator::translate_binary(
    mi::mdl::IExpression_binary const *call,
    bool                              wants_derivs)
{
    Function_context &ctx = *m_ctx;

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
            Expression_result  compound    = translate_expression(lhs, wants_derivs);
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
            Expression_result       comp       = translate_expression(lhs, wants_derivs);
            llvm::Value             *index     =
                translate_expression_value(rhs, /*wants_derivs=*/ false);
            mi::mdl::Position const *index_pos = &rhs->access_position();
            return translate_index_expression(comp_type, comp, index, index_pos);
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
        return Expression_result::value(translate_binary_no_side_effect(call, wants_derivs));

    case mi::mdl::IExpression_binary::OK_LESS:
    case mi::mdl::IExpression_binary::OK_LESS_OR_EQUAL:
    case mi::mdl::IExpression_binary::OK_GREATER_OR_EQUAL:
    case mi::mdl::IExpression_binary::OK_GREATER:
    case mi::mdl::IExpression_binary::OK_EQUAL:
    case mi::mdl::IExpression_binary::OK_NOT_EQUAL:
        {
            MDL_ASSERT(!wants_derivs && "boolean values cannot have derivatives");

            mi::mdl::IType const *l_type = lhs->get_type();
            mi::mdl::IType const *r_type = rhs->get_type();

            llvm::Value *lv = translate_expression_value(lhs, /*wants_derivs=*/ false);
            llvm::Value *rv = translate_expression_value(rhs, /*wants_derivs=*/ false);

            return translate_compare(op, l_type, lv, r_type, rv);
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
        return translate_assign(call, wants_derivs);

    case mi::mdl::IExpression_binary::OK_SEQUENCE:
        // just execute all expressions
        (void) translate_expression(lhs, /*wants_derivs=*/ false);
        return translate_expression(rhs, wants_derivs);
    }
    MDL_ASSERT(!"unsupported binary operator kind");
    Expression_result res = Expression_result::undef(lookup_type(call->get_type()));
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
    mi::mdl::IExpression_binary const *call,
    bool                              wants_derivs)
{
    Function_context &ctx = *m_ctx;

    llvm::Value *res = NULL;

    mi::mdl::IExpression const *lhs = call->get_left_argument();
    mi::mdl::IExpression const *rhs = call->get_right_argument();
    mi::mdl::Position const *call_pos = &call->access_position();

    mi::mdl::IExpression_binary::Operator op = call->get_operator();

    switch (op) {
    case mi::mdl::IExpression_binary::OK_MULTIPLY:
    case mi::mdl::IExpression_binary::OK_MULTIPLY_ASSIGN:
        return translate_multiply(
            lookup_type_or_deriv_type(call), lhs, rhs, wants_derivs);

    case mi::mdl::IExpression_binary::OK_LOGICAL_AND:
    case mi::mdl::IExpression_binary::OK_LOGICAL_OR:
        if (is<mi::mdl::IType_bool>(lhs->get_type()->skip_type_alias()) &&
                is<mi::mdl::IType_bool>(rhs->get_type()->skip_type_alias())) {
            // shortcut evaluation here
            llvm::Value      *tmp      = ctx.create_local(m_type_mapper.get_bool_type(), "sc_tmp");
            llvm::BasicBlock *true_bb  = ctx.create_bb("sc_true");
            llvm::BasicBlock *false_bb = ctx.create_bb("sc_false");
            llvm::BasicBlock *end_bb   = ctx.create_bb("sc_end");

            translate_boolean_branch(call, true_bb, false_bb);

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
            if (op == mi::mdl::IExpression_binary::OK_LOGICAL_AND) {
                op = mi::mdl::IExpression_binary::OK_BITWISE_AND;
            } else {
                op = mi::mdl::IExpression_binary::OK_BITWISE_OR;
            }
        }
        break;

    default:
        break;
    }

    llvm::Value *l = translate_expression_value(lhs, wants_derivs);
    llvm::Value *r = translate_expression_value(rhs, wants_derivs);

    llvm::Type *res_type = lookup_type_or_deriv_type(call);
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

                llvm::Value *tmp = translate_binary_basic(op, l_elem, r_elem, call_pos);
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
            res = translate_binary_basic(op, l, r, call_pos);
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

            llvm::Value *tmp = translate_binary_basic(op, l_elem, r_elem, call_pos);
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
        if (llvm::FixedVectorType *v_tp = llvm::dyn_cast<llvm::FixedVectorType>(r_type)) {
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

                llvm::Value *tmp = translate_binary_basic(op, l_elem, r_elem, call_pos);
                if (tmp == NULL) {
                    res = NULL;
                    break;
                }
                res = ctx->CreateInsertElement(res, tmp, idx);
            }
        } else {
            res = translate_binary_basic(op, l, r, call_pos);
        }
    } else {
        // not array typed, and not vector typed division
        res = translate_binary_basic(op, l, r, call_pos);
    }

    if (res == NULL) {
        MDL_ASSERT(!"Unsupported binary operator");
        res = llvm::UndefValue::get(res_type);
    }
    return res;
}

// Translate a side effect free binary expression to LLVM IR.
llvm::Value *LLVM_code_generator::translate_binary_no_side_effect(
    mi::mdl::ICall_expr const *bin_expr)
{
    Function_context &ctx = *m_ctx;

    mi::mdl::IDefinition::Semantics sema = bin_expr->get_semantics();
    mi::mdl::IExpression_binary::Operator op =
        mi::mdl::IExpression_binary::Operator(semantic_to_operator(sema));
    llvm::Value *res = NULL;

    // on the DAG, the wants_derivs parameter is ignored, as the nodes already know, whether
    // they have to return derivatives or not
    bool derivs = false;

    switch (op) {
    case mi::mdl::IExpression_binary::OK_ARRAY_INDEX:
        {
            mi::mdl::IType const *comp_type = bin_expr->get_argument_type(0);
            Expression_result    comp = bin_expr->translate_argument(0, derivs);
            llvm::Value          *index =
                bin_expr->translate_argument(1, /*wants_derivs=*/ false).as_value(ctx);
            mi::mdl::Position    *index_pos = NULL;

            ctx.set_curr_pos(bin_expr);

            Expression_result res = translate_index_expression(
                comp_type, comp, index, index_pos);
            return res.as_value(ctx);
        }

    case mi::mdl::IExpression_binary::OK_MULTIPLY:
        {
            mi::mdl::IType const *l_type = bin_expr->get_argument_type(0);
            llvm::Value          *l = bin_expr->translate_argument(0, derivs).as_value(ctx);

            mi::mdl::IType const *r_type = bin_expr->get_argument_type(1);
            llvm::Value          *r = bin_expr->translate_argument(1, derivs).as_value(ctx);

            ctx.set_curr_pos(bin_expr);

            return translate_multiply(
                lookup_type_or_deriv_type(bin_expr), l_type, l, r_type, r);
        }

    case mi::mdl::IExpression_binary::OK_LOGICAL_AND:
    case mi::mdl::IExpression_binary::OK_LOGICAL_OR:
        {
            mi::mdl::IType const *l_type = bin_expr->get_argument_type(0)->skip_type_alias();
            mi::mdl::IType const *r_type = bin_expr->get_argument_type(1)->skip_type_alias();

            if (is<mi::mdl::IType_bool>(l_type) && is<mi::mdl::IType_bool>(r_type)) {
                // shortcut evaluation here
                llvm::Value      *tmp      = ctx.create_local(
                    m_type_mapper.get_bool_type(), "sc_tmp");
                llvm::BasicBlock *true_bb  = ctx.create_bb("sc_true");
                llvm::BasicBlock *false_bb = ctx.create_bb("sc_false");
                llvm::BasicBlock *end_bb   = ctx.create_bb("sc_end");

                bin_expr->translate_boolean_branch(true_bb, false_bb);

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
                if (op == mi::mdl::IExpression_binary::OK_LOGICAL_AND) {
                    op = mi::mdl::IExpression_binary::OK_BITWISE_AND;
                } else {
                    op = mi::mdl::IExpression_binary::OK_BITWISE_OR;
                }
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
            llvm::Value *l = bin_expr->translate_argument(0, derivs).as_value(ctx);
            llvm::Value *r = bin_expr->translate_argument(1, derivs).as_value(ctx);

            mi::mdl::IType const *l_type = bin_expr->get_argument_type(0);
            mi::mdl::IType const *r_type = bin_expr->get_argument_type(1);

            ctx.set_curr_pos(bin_expr);

            return translate_compare(op, l_type, l, r_type, r).as_value(ctx);
        }

    default:
        break;
    }

    llvm::Value *l = bin_expr->translate_argument(0, derivs).as_value(ctx);
    llvm::Value *r = bin_expr->translate_argument(1, derivs).as_value(ctx);
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

                ctx.set_curr_pos(bin_expr);

                llvm::Value *tmp = translate_binary_basic(op, l_elem, r_elem, bin_expr_pos);
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
            res = translate_binary_basic(op, l, r, bin_expr_pos);
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

            ctx.set_curr_pos(bin_expr);

            llvm::Value *tmp = translate_binary_basic(op, l_elem, r_elem, bin_expr_pos);
            if (tmp == NULL) {
                res = NULL;
                break;
            }
            res = ctx->CreateInsertValue(res, tmp, idxes);
        }
    } else {
        // not array typed
        ctx.set_curr_pos(bin_expr);

        res = translate_binary_basic(op, l, r, bin_expr_pos);
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
    mi::mdl::IExpression_binary::Operator op,
    llvm::Value                           *l,
    llvm::Value                           *r,
    mi::mdl::Position const               *expr_pos)
{
    Function_context &ctx = *m_ctx;

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
                } else if (llvm::VectorType *v_tp = llvm::dyn_cast<llvm::VectorType>(
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

    // check if target requires a wrapper function call
    llvm::Function *cmp_func = get_target_operator_function(op, l_type);

    if (cmp_func != nullptr) {
        // create function call to wrapper function

        // FIXME: handle derivs correct?
        llvm::Value *args[] = { l, r };
        return ctx->CreateCall(cmp_func, args);
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
        {
            auto create_modulo = [&](llvm::Value *l, llvm::Value *r) {
                // Modulo is undefined for negative values in GLSL, so implement as l - (l / r) * r
                if (m_target_lang == ICode_generator::TL_GLSL) {
                    return ctx->CreateSub(l,
                        ctx->CreateMul(
                            ctx->CreateSDiv(l, r),
                            r));
                } else {
                    return ctx->CreateSRem(l, r);
                }
            };

            // only on integer and integer vectors
            if (!maybe_zero(r)) {
                // no exception possible
                res = create_modulo(l, r);
            } else if (m_divzero_check_exception_disabled) {
                llvm::BasicBlock *ok_bb   = ctx.create_bb("mod_by_non_zero");
                llvm::BasicBlock *fail_bb = ctx.create_bb("mod_by_zerol");
                llvm::BasicBlock *end_bb  = ctx.create_bb("mod_end");
                llvm::Value      *tmp     = ctx.create_local(l_type, "res");

                ctx.create_non_zero_check_cmp(rhs, ok_bb, fail_bb);
                ctx->SetInsertPoint(ok_bb);
                {
                    res = create_modulo(l, r);
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
                res = create_modulo(l, r);
            }
        }
        break;

    case mi::mdl::IExpression_binary::OK_PLUS:
    case mi::mdl::IExpression_binary::OK_PLUS_ASSIGN:
        if (is_integer_op) {
            res = ctx->CreateAdd(l, r);
        } else {
            if (l_is_deriv) {
                res = ctx.get_dual(
                    ctx->CreateFAdd(ctx.get_dual_val(l), ctx.get_dual_val(r)),
                    ctx->CreateFAdd(ctx.get_dual_dx(l),  ctx.get_dual_dx(r)),
                    ctx->CreateFAdd(ctx.get_dual_dy(l),  ctx.get_dual_dy(r)));
            } else {
                res = ctx->CreateFAdd(l, r);
            }
        }
        break;

    case mi::mdl::IExpression_binary::OK_MINUS:
    case mi::mdl::IExpression_binary::OK_MINUS_ASSIGN:
        if (is_integer_op) {
            res = ctx->CreateSub(l, r);
        } else {
            if (l_is_deriv) {
                res = ctx.get_dual(
                    ctx->CreateFSub(ctx.get_dual_val(l), ctx.get_dual_val(r)),
                    ctx->CreateFSub(ctx.get_dual_dx(l),  ctx.get_dual_dx(r)),
                    ctx->CreateFSub(ctx.get_dual_dy(l),  ctx.get_dual_dy(r)));
            } else {
                res = ctx->CreateFSub(l, r);
            }
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
        MDL_ASSERT(!"Unexpected operations in translate_binary_basic()");
        break;

    default:
        break;
    }
    return res;
}

// Translate a multiplication expression to LLVM IR.
llvm::Value *LLVM_code_generator::translate_multiply(
    llvm::Type                 *res_llvm_type,
    mi::mdl::IType const       *l_type,
    llvm::Value                *l,
    mi::mdl::IType const       *r_type,
    llvm::Value                *r)
{
    Function_context &ctx = *m_ctx;

    llvm::Value *res = NULL;

    l_type = m_type_mapper.skip_deriv_type(l_type->skip_type_alias());
    r_type = m_type_mapper.skip_deriv_type(r_type->skip_type_alias());

    if (m_type_mapper.is_deriv_type(res_llvm_type)) {
        // compute WITH derivatives
        if (mi::mdl::IType_matrix const *L_type = as<mi::mdl::IType_matrix>(l_type)) {
            if (mi::mdl::IType_matrix const *R_type = as<mi::mdl::IType_matrix>(r_type)) {
                // matrix * matrix
                res = do_matrix_multiplication_MxM_deriv(
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
                    res_llvm_type, l, r, rows, cols);
            } else {
                // matrix * scalar element-wise multiplication
                res = do_matrix_multiplication_MxS_deriv(
                    res_llvm_type, l, r);
            }
        } else if (mi::mdl::IType_matrix const *R_type = as<mi::mdl::IType_matrix>(r_type)) {
            if (is<mi::mdl::IType_vector>(l_type)) {
                // vector * matrix
                int rows = R_type->get_element_type()->get_size();
                int cols = R_type->get_columns();

                res = do_matrix_multiplication_VxM_deriv(
                    res_llvm_type, l, r, rows, cols);
            } else {
                // matrix * scalar element-wise multiplication
                res = do_matrix_multiplication_MxS_deriv(
                    res_llvm_type, r, l);
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
                    res_llvm_type,
                    l,
                    r,
                    L_type->get_element_type()->get_size(),
                    L_type->get_columns(),
                    R_type->get_columns());
            } else if (is<mi::mdl::IType_vector>(r_type)) {
                // matrix * vector
                res = do_matrix_multiplication_MxV(
                    res_llvm_type,
                    l,
                    r,
                    L_type->get_element_type()->get_size(),
                    L_type->get_columns());
            } else {
                // matrix * scalar element-wise multiplication
                res = do_matrix_multiplication_MxS(res_llvm_type, l, r);
            }
        } else {
            if (is<mi::mdl::IType_matrix>(r_type)) {
                // scalar * matrix element-wise multiplication
                res = do_matrix_multiplication_MxS(res_llvm_type, r, l);
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
    llvm::Type                 *res_llvm_type,
    mi::mdl::IExpression const *lhs,
    mi::mdl::IExpression const *rhs,
    bool                       wants_derivs)
{
    mi::mdl::IType const *l_type = lhs->get_type();
    llvm::Value          *l      = translate_expression_value(lhs, wants_derivs);
    mi::mdl::IType const *r_type = rhs->get_type();
    llvm::Value          *r      = translate_expression_value(rhs, wants_derivs);

    return translate_multiply(res_llvm_type, l_type, l, r_type, r);
}

// Create a matrix by scalar multiplication.
llvm::Value *LLVM_code_generator::do_matrix_multiplication_MxS(
    llvm::Type       *res_llvm_type,
    llvm::Value      *l,
    llvm::Value      *r)
{
    Function_context &ctx = *m_ctx;

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
    llvm::Type *res_llvm_type,
    llvm::Value *l,
    llvm::Value *r)
{
    Function_context &ctx = *m_ctx;

    // (A + Bx + Cy)(d + ex + fy) = Ad + (Ae + Bd)x + (Af + Cd)y

    llvm::Type *elem_type = m_type_mapper.get_deriv_base_type(res_llvm_type);

    llvm::Value *val = do_matrix_multiplication_MxS(
        elem_type, ctx.get_dual_val(l), ctx.get_dual_val(r));

    llvm::Value *dx_1 = do_matrix_multiplication_MxS(
        elem_type, ctx.get_dual_val(l), ctx.get_dual_dx(r));
    llvm::Value *dx_2 = do_matrix_multiplication_MxS(
        elem_type, ctx.get_dual_dx(l), ctx.get_dual_val(r));
    llvm::Value *dx = do_matrix_addition(elem_type, dx_1, dx_2);

    llvm::Value *dy_1 = do_matrix_multiplication_MxS(
        elem_type, ctx.get_dual_val(l), ctx.get_dual_dy(r));
    llvm::Value *dy_2 = do_matrix_multiplication_MxS(
        elem_type, ctx.get_dual_dy(l), ctx.get_dual_val(r));
    llvm::Value *dy = do_matrix_addition(elem_type, dy_1, dy_2);

    return ctx.get_dual(val, dx, dy);
}

// Create a matrix by matrix addition.
llvm::Value *LLVM_code_generator::do_matrix_addition(
    llvm::Type *res_llvm_type,
    llvm::Value *l,
    llvm::Value *r)
{
    Function_context &ctx = *m_ctx;

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
    mi::mdl::IExpression_binary const *call,
    bool                              wants_derivs)
{
    Function_context &ctx = *m_ctx;

    mi::mdl::IExpression const *lhs = call->get_left_argument();
    mi::mdl::IExpression const *rhs = call->get_right_argument();

    // is_deriv_var is only defined on the top level variables. For a variable in a struct, this
    // would be the struct. While the struct could be a derivative value, it could also contain
    // values which cannot be used as derivatives like integers. Also consider the struct itself
    // or sub-structs containing floating point types.
    IDefinition const *lhs_base_def = Analysis::get_lvalue_base(lhs);
    bool lhs_base_is_deriv = is_deriv_var(lhs_base_def);
    bool lhs_is_deriv = lhs_base_is_deriv &&
        m_type_mapper.contains_floating_point_type(lhs->get_type());

    llvm::Value *res = NULL;

    mi::mdl::IExpression_binary::Operator op = call->get_operator();
    switch (op) {
    case mi::mdl::IExpression_binary::OK_ASSIGN:
        res = translate_expression_value(rhs, lhs_is_deriv);
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
        res = translate_binary_no_side_effect(call, lhs_is_deriv);
        break;

    default:
        {
            llvm_unreachable("unsupported assign");
            Expression_result expr_res = Expression_result::undef(lookup_type(call->get_type()));
            expr_res.strip_deriv_if_not_wanted(ctx, wants_derivs);
            return expr_res;
        }
    }

    if (lhs_base_is_deriv) {
        llvm::Value *adr_val, *adr_dx, *adr_dy;
        translate_lval_expression_dual(lhs, adr_val, adr_dx, adr_dy);

        // if the left hand side was not a derivative value (for example an integer in a
        // derivative struct), first convert it to a dual with dx and dy set to zero
        if (!lhs_is_deriv)
            res = ctx.get_dual(res);

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
        llvm::Value *adr = translate_lval_expression(lhs);

        // check if the result must be converted into a vector first
        llvm::Type *res_tp = llvm::cast<llvm::PointerType>(adr->getType())->getElementType();
        if (res_tp != res->getType()) {
            res = ctx.create_splat(res_tp, res);
        }

        // store the result
        ctx->CreateStore(res, adr);
    }

    Expression_result expr_res = Expression_result::value(res);
    expr_res.strip_deriv_if_not_wanted(ctx, wants_derivs);
    return expr_res;
}

// Translate a binary compare expression to LLVM IR.
Expression_result LLVM_code_generator::translate_compare(
    mi::mdl::IExpression_binary::Operator op,
    mi::mdl::IType const                  *l_type,
    llvm::Value                           *lv,
    mi::mdl::IType const                  *r_type,
    llvm::Value                           *rv)
{
    Function_context &ctx = *m_ctx;

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

        // check if target requires a wrapper function call
        llvm::Function *cmp_func = get_target_compare_function(op, e_tp);

        res = ctx.get_constant(op == mi::mdl::IExpression_binary::OK_EQUAL);
        res = ctx->CreateTrunc(res, m_type_mapper.get_predicate_type());

        unsigned idxes[1];
        for (unsigned i = 0, n = unsigned(a_tp->getNumElements()); i < n; ++i) {
            idxes[0] = i;

            llvm::Value *l_elem = conv_l ? lv : ctx->CreateExtractValue(lv, idxes);
            llvm::Value *r_elem = conv_r ? rv : ctx->CreateExtractValue(rv, idxes);

            // only == and != are supported
            llvm::Value *e_res;

            if (cmp_func != nullptr) {
                // create function call
                llvm::Value *args[] = { l_elem, r_elem };
                e_res = ctx->CreateCall(cmp_func, args);
            } else {
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
                            llvm::FixedVectorType *v_tp = llvm::cast<llvm::FixedVectorType>(e_tp);

                            // all must be equal
                            for (unsigned j = 0, N = unsigned(v_tp->getNumElements()); j < N; ++j) {
                                llvm::Value *idx = ctx.get_constant(int(j));

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
                            llvm::FixedVectorType *v_tp = llvm::cast<llvm::FixedVectorType>(e_tp);

                            // only one must be not equal
                            for (unsigned j = 0, N = unsigned(v_tp->getNumElements()); j < N; ++j) {
                                llvm::Value *idx = ctx.get_constant(int(j));

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
        }
        // map the i1 result to the bool type representation
        res = ctx->CreateZExt(res, m_type_mapper.get_bool_type());
        return Expression_result::value(res);
    }

    if (llvm::FixedVectorType *vt = llvm::dyn_cast<llvm::FixedVectorType>(op_type)) {
        // convert the scalar one to vector
        if (conv_l) {
            lv = ctx.create_vector_splat(vt, lv);
        }
        if (conv_r) {
            rv = ctx.create_vector_splat(vt, rv);
        }
    }

    // check if target requires a wrapper function call
    llvm::Function *cmp_func = get_target_compare_function(op, op_type);

    if (cmp_func != nullptr) {
        // create function call
        llvm::Value *args[] = { lv, rv };
        res = ctx->CreateCall(cmp_func, args);
    } else {
        // create vector compares
        if (op_type->isFPOrFPVectorTy()) {
            // create vector compare
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
    }

    if (llvm::FixedVectorType *vt = llvm::dyn_cast<llvm::FixedVectorType>(op_type)) {
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
    mi::mdl::IExpression_conditional const *cond_expr,
    bool                                   wants_derivs)
{
    Function_context &ctx = *m_ctx;

    // C-like ternary operator with lazy evaluation
    llvm::BasicBlock *on_true_bb  = ctx.create_bb("?:_true");
    llvm::BasicBlock *on_false_bb = ctx.create_bb("?:_false");
    llvm::BasicBlock *end_bb      = ctx.create_bb("?:_end");

    translate_boolean_branch(cond_expr->get_condition(), on_true_bb, on_false_bb);

    // as we go back and forth, we have to backup and restore the insert point of the context,
    // as translating expressions may insert new basic blocks, like done here
    struct Insert_point
    {
        Insert_point(Function_context &ctx)
        : m_bb(ctx->GetInsertBlock())
        , m_insert_point(ctx->GetInsertPoint())
        {
        }

        void restore(Function_context &ctx)
        {
            ctx->SetInsertPoint(m_bb, m_insert_point);
        }

        llvm::BasicBlock *m_bb;
        llvm::BasicBlock::iterator m_insert_point;
    };

    // first only translate the true and false expressions
    ctx->SetInsertPoint(on_true_bb);
    llvm::Value *true_res = translate_expression_value(cond_expr->get_true(), wants_derivs);
    Insert_point true_insert_point(ctx);

    ctx->SetInsertPoint(on_false_bb);
    llvm::Value *false_res = translate_expression_value(cond_expr->get_false(), wants_derivs);
    Insert_point false_insert_point(ctx);

    // check whether we got any derivatives to return
    llvm::Type *res_type;
    bool got_derivs = m_type_mapper.is_deriv_type(true_res->getType()) ||
        m_type_mapper.is_deriv_type(false_res->getType());
    if (got_derivs) {
        // yes, store as derivative
        res_type = lookup_type_or_deriv_type(cond_expr);
    } else {
        // no, so just return the values without derivatives
        res_type = m_type_mapper.lookup_type(
            m_llvm_context,
            cond_expr->get_type(),
            ctx.instantiate_type_size(cond_expr->get_type()));
    }

    llvm::Value *res_addr  = ctx.create_local(res_type, "?:_tmp");

    // now finish the true and false blocks, setting the result and branching to the end
    true_insert_point.restore(ctx);
    if (got_derivs && !m_type_mapper.is_deriv_type(true_res->getType())) {
        true_res = ctx.get_dual(true_res);
    }
    ctx->CreateStore(true_res, res_addr);
    ctx->CreateBr(end_bb);

    false_insert_point.restore(ctx);
    if (got_derivs && !m_type_mapper.is_deriv_type(false_res->getType())) {
        false_res = ctx.get_dual(false_res);
    }
    ctx->CreateStore(false_res, res_addr);
    ctx->CreateBr(end_bb);

    ctx->SetInsertPoint(end_bb);
    return Expression_result::ptr(res_addr);
}

// Translate a DAG intrinsic call expression to LLVM IR.
Expression_result LLVM_code_generator::translate_dag_intrinsic(
    ICall_expr const *call_expr)
{
    Function_context &ctx = *m_ctx;

    mi::mdl::IDefinition::Semantics sema = call_expr->get_semantics();

    bool derivs = call_expr->returns_derivatives();

    switch (sema) {
    case mi::mdl::IDefinition::DS_INTRINSIC_DAG_FIELD_ACCESS:
        {
            llvm::Value *compound = call_expr->translate_argument(0, derivs).as_value(ctx);
            int         index     = call_expr->get_field_index();

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
            return call_expr->translate_argument(1, derivs);
        }
        break;
    case mi::mdl::IDefinition::DS_INTRINSIC_DAG_SET_TRANSFORMS:
        {
            mi::mdl::IValue_matrix const *m_w2o =
                mi::mdl::cast<mi::mdl::IValue_matrix>(call_expr->get_const_argument(0));
            mi::mdl::IValue_matrix const *m_o2w =
                mi::mdl::cast<mi::mdl::IValue_matrix>(call_expr->get_const_argument(1));
            set_transforms(m_w2o, m_o2w);
            return call_expr->translate_argument(2, derivs);
        }
        break;
    case mi::mdl::IDefinition::DS_INTRINSIC_DAG_MAKE_DERIV:
        {
            Expression_result res = call_expr->translate_argument(
                0, /*wants_derivs=*/ true);  // for DAG wants_derivs is ignored
            res.ensure_deriv_result(ctx, true);  // always ensure we get a derivative
            return res;
        }
    }
    MDL_ASSERT(!"Unexpected DAG intrinsic");
    return Expression_result::undef(lookup_type(call_expr->get_type()));
}

// Create the float4x4 identity matrix.
llvm::Value *LLVM_code_generator::create_identity_matrix()
{
    Function_context &ctx = *m_ctx;

    llvm::Constant *elems[16];

    // flat matrix values
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            llvm::Value *v = ctx.get_constant(i == j ? 1.0f : 0.0f);

            elems[i * 4 + j] = llvm::cast<llvm::Constant>(v);
        }
    }

    llvm::Type *m_type = m_type_mapper.get_float4x4_type();
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

// Translate a call expression to LLVM IR.
Expression_result LLVM_code_generator::translate_call(
    ICall_expr const *call_expr)
{
    Function_context &ctx = *m_ctx;

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
        return translate_array_constructor_call(call_expr);
    }

    mi::mdl::IDefinition::Semantics sema = call_expr->get_semantics();

    bool return_derivs = call_expr->returns_derivatives();

    if (semantic_is_operator(sema)) {
        mi::mdl::IExpression::Operator op = semantic_to_operator(sema);

        if (is_unary_operator(op)) {
            mi::mdl::IExpression_unary::Operator uop = mi::mdl::IExpression_unary::Operator(op);
            Expression_result arg = call_expr->translate_argument(0, return_derivs);
            if (uop == mi::mdl::IExpression_unary::OK_CAST) {
                return translate_cast_expression(arg, call_expr->get_type());
            } else {
                return Expression_result::value(translate_unary(uop, arg.as_value(ctx)));
            }
        } else if (is_binary_operator(op)) {
            llvm::Value *res = translate_binary_no_side_effect(call_expr);
            return Expression_result::value(res);
        } else if (op == IExpression::OK_TERNARY) {
            if (m_eval_dag_ternary_strictly && call_expr->as_dag_call() != NULL) {
                // ternary operator on the DAG with strict evaluation
                llvm::Value *cond_res =
                    call_expr->translate_argument(0, /*wants_derivs=*/ false).as_value(ctx);

                if (cond_res->getType() != m_type_mapper.get_predicate_type()) {
                    // map to predicate type
                    cond_res = ctx->CreateICmpNE(cond_res, ctx.get_constant(false));
                }

                llvm::Value *true_res =
                    call_expr->translate_argument(1, /*wants_derivs=*/ false).as_value(ctx);

                llvm::Value *false_res =
                    call_expr->translate_argument(2, /*wants_derivs=*/ false).as_value(ctx);

                return Expression_result::value(ctx->CreateSelect(cond_res, true_res, false_res));
            }

            // C-like ternary operator with lazy evaluation

            llvm::BasicBlock *on_true_bb  = ctx.create_bb("?:_true");
            llvm::BasicBlock *on_false_bb = ctx.create_bb("?:_false");
            llvm::BasicBlock *end_bb      = ctx.create_bb("?:_end");

            llvm::Type  *res_type  = lookup_type(call_expr->get_type());
            llvm::Value *res_addr  = ctx.create_local(res_type, "?:_tmp");

            call_expr->translate_boolean_branch(on_true_bb, on_false_bb);

            // This is ugly: As long as we translate the "material body", there exists
            // only one basic block. However, for the ?: operator we need control flow, so we must
            // handle the CSE right. Currently we restrict it to the current block.
            // This is a correct, but not the smartest solution, as this might blow up the
            // generated code.

            {
                BB_store true_chain(m_curr_bb, get_next_bb());
                ctx->SetInsertPoint(on_true_bb);
                llvm::Value *true_res =
                    call_expr->translate_argument(1, /*wants_derivs=*/ false).as_value(ctx);
                ctx->CreateStore(true_res, res_addr);
                ctx->CreateBr(end_bb);
            }

            {
                BB_store false_chain(m_curr_bb, get_next_bb());
                ctx->SetInsertPoint(on_false_bb);
                llvm::Value *false_res =
                    call_expr->translate_argument(2, /*wants_derivs=*/ false).as_value(ctx);
                ctx->CreateStore(false_res, res_addr);
                ctx->CreateBr(end_bb);
            }

            ctx->SetInsertPoint(end_bb);
            return Expression_result::ptr(res_addr);
        }
    }

    switch (sema) {
    case mi::mdl::IDefinition::DS_UNKNOWN:
        return translate_call_user_defined_function(call_expr);

    case mi::mdl::IDefinition::DS_COPY_CONSTRUCTOR:
        // ignore all copy constructors
        MDL_ASSERT(call_expr->get_argument_count() == 1);
        return call_expr->translate_argument(0, return_derivs);

    case mi::mdl::IDefinition::DS_CONV_CONSTRUCTOR:
    case mi::mdl::IDefinition::DS_CONV_OPERATOR:
        // handle conversion constructor and operator equal: both have one argument and convert it
        return translate_conversion(call_expr);

    case mi::mdl::IDefinition::DS_ELEM_CONSTRUCTOR:
        return translate_elemental_constructor(call_expr);

    case mi::mdl::IDefinition::DS_COLOR_SPECTRUM_CONSTRUCTOR:
        // translate to rgb color, fall into the default case
        if (m_warn_spectrum_conversion) {
            // but warn because a spectrum is converted
            mi::mdl::Position const *pos = call_expr->get_position();
            if (pos == NULL) {
                pos = &zero;
            }
            warning(
                SPECTRUM_CONVERTED_TO_RGB,
                Exc_location(*this, pos),
                Error_params(get_allocator()));
        }
        break;

    case mi::mdl::IDefinition::DS_INTRINSIC_MATH_EMISSION_COLOR:
        if (call_expr->get_argument_count() == 1) {
            // a no-op in the RGB case
            Expression_result res = call_expr->translate_argument(
                0, return_derivs);  // for DAG return_derivs is ignored
            res.ensure_deriv_result(ctx, return_derivs);
            return res;
        } else if (m_warn_spectrum_conversion) {
            // warn because a spectrum is converted
            mi::mdl::Position const *pos = call_expr->get_position();
            if (pos == NULL) {
                pos = &zero;
            }
            warning(
                SPECTRUM_CONVERTED_TO_RGB,
                Exc_location(*this, pos),
                Error_params(get_allocator()));
        }
        // else fall into the default case
        break;

    case mi::mdl::IDefinition::DS_MATRIX_ELEM_CONSTRUCTOR:
        return translate_matrix_elemental_constructor(call_expr);

    case mi::mdl::IDefinition::DS_MATRIX_DIAG_CONSTRUCTOR:
        return translate_matrix_diagonal_constructor(call_expr);

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
                equal_coordinate_space(arg_0, arg_1, m_internal_space.c_str()))
            {
                // does not really use the transform state here, so we do not flag it
                return Expression_result::value(create_identity_matrix());
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
                equal_coordinate_space(arg_0, arg_1, m_internal_space.c_str()))
            {
                // just a no-op, return the second argument
                // does not really use the transform state here, so we do not flag it
                return call_expr->translate_argument(2, return_derivs);
            }
            if (m_world_to_object != NULL && m_object_to_world != NULL) {
                // we have a world-to_object matrix, implement the functions directly
                return translate_transform_call(sema, call_expr);
            }
        }
        break;
    case mi::mdl::IDefinition::DS_INTRINSIC_STATE_OBJECT_ID:
        if (call_expr->get_argument_count() == 0) {
            return translate_object_id_call();
        }
        break;
    case mi::mdl::IDefinition::DS_INTRINSIC_DAG_FIELD_ACCESS:
    case mi::mdl::IDefinition::DS_INTRINSIC_DAG_ARRAY_LENGTH:
    case mi::mdl::IDefinition::DS_INTRINSIC_DAG_SET_OBJECT_ID:
    case mi::mdl::IDefinition::DS_INTRINSIC_DAG_SET_TRANSFORMS:
    case mi::mdl::IDefinition::DS_INTRINSIC_DAG_MAKE_DERIV:
        return translate_dag_intrinsic(call_expr);

    case mi::mdl::IDefinition::DS_INTRINSIC_JIT_LOOKUP:
        return translate_jit_intrinsic(call_expr);

    case mi::mdl::IDefinition::DS_INTRINSIC_DAG_CALL_LAMBDA:
        return translate_dag_call_lambda(call_expr);

    case mi::mdl::IDefinition::DS_INTRINSIC_DAG_DECL_CAST:
        error(INTERNAL_JIT_BACKEND_ERROR, "Cannot compile decl_cast nodes");
        return Expression_result::undef(lookup_type(call_expr->get_type()));

    case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_ISVALID:
        if ((m_target_lang == ICode_generator::TL_NATIVE && m_has_res_handler) ||
            m_target_lang == ICode_generator::TL_LLVM_IR ||
            m_state_mode == State_subset_mode::SSM_ENVIRONMENT)
        {
            if (m_state_mode == State_subset_mode::SSM_ENVIRONMENT) {
                mi::mdl::Position const *pos = call_expr->get_position();
                if (pos == NULL) {
                    pos = &zero;
                }
                warning(
                    SCENE_DATA_CALL_IN_ENVIRONMENT_FUNCTION,
                    Exc_location(*this, pos),
                    Error_params(get_allocator()));
            }

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
    case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_FLOAT4X4:
    case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_FLOAT4X4:
        if ((m_target_lang == ICode_generator::TL_NATIVE && m_has_res_handler) ||
            m_target_lang == ICode_generator::TL_LLVM_IR ||
            m_state_mode == State_subset_mode::SSM_ENVIRONMENT)
        {
            if (m_state_mode == State_subset_mode::SSM_ENVIRONMENT) {
                mi::mdl::Position const *pos = call_expr->get_position();
                if (pos == NULL) {
                    pos = &zero;
                }
                warning(
                    SCENE_DATA_CALL_IN_ENVIRONMENT_FUNCTION,
                    Exc_location(*this, pos),
                    Error_params(get_allocator()));
            }

            // TODO: implement calling renderer runtime. For now just return second argument
            return call_expr->translate_argument(1, return_derivs);
        }

        // try compiler known intrinsic function
        break;

    case mi::mdl::IDefinition::DS_INTRINSIC_TEX_WIDTH_OFFSET:
    case mi::mdl::IDefinition::DS_INTRINSIC_TEX_HEIGHT_OFFSET:
    case mi::mdl::IDefinition::DS_INTRINSIC_TEX_DEPTH_OFFSET:
        // TODO: for now, just return zero
        return Expression_result::value(ctx.get_constant(int(0)));

    case IDefinition::DS_INTRINSIC_TEX_GRID_TO_OBJECT_SPACE:
        // TODO: for now, just return the identity matrix
        return Expression_result::value(create_identity_matrix());

    default:
        // try compiler known intrinsic function
        break;
    }

    return translate_call_intrinsic_function(call_expr);
}

// Get the argument type instances for a given call.
Function_instance LLVM_code_generator::get_call_instance(
    ICall_expr const *call)
{
    Function_context &ctx = *m_ctx;

    Function_instance::Array_instances res(get_allocator());
    Function_instance::Parameter_storage_modifiers param_mods(get_allocator());

    if (!m_enable_instancing) {
        // instancing disabled
        return Function_instance(
            call->get_callee_definition(),
            res,
            param_mods,
            call->returns_derivatives(),
            target_supports_storage_spaces());
    }

    typedef ptr_hash_set<IType_array_size const>::Type Marker_set;
    Marker_set mset(0, Marker_set::hasher(), Marker_set::key_equal(), get_allocator());

    IDefinition const *def = call->get_callee_definition();
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
    bool all_param_mods_normal = true;  // an empty param_mods means all normal
    for (size_t i = 0, n = call->get_argument_count(); i < n; ++i) {
        Storage_modifier param_mod = call->get_argument_storage_modifier(i);
        if (all_param_mods_normal && param_mod != SM_NORMAL) {
            // initialize all previous parameter storage modifiers as normal
            all_param_mods_normal = false;
            param_mods.reserve(n);
            for (size_t j = 0; j < i; ++j) {
                param_mods.push_back(SM_NORMAL);
            }
        }
        if (!all_param_mods_normal) {
            param_mods.push_back(param_mod);
        }

        IType_array const *a_arg_type = as<IType_array>(call->get_argument_type(i));

        if (a_arg_type == NULL) {
            continue;
        }

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
                    if (immediate_size < 0) {
                        immediate_size = a_arg_type->get_size();
                    }

                    res.push_back(Array_instance(deferred_size, immediate_size));
                }
            }
        }
    }

    return Function_instance(
        call->get_callee_definition(),
        res,
        param_mods,
        call->returns_derivatives(),
        target_supports_storage_spaces());
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

            if (ai.get_deferred_size() == deferred_size) {
                return ai.get_immediate_size();
            }
        }
    }
    return -1;
}

// Translate a call to an user defined function to LLVM IR.
Expression_result LLVM_code_generator::translate_call_user_defined_function(
    ICall_expr const *call_expr)
{
    Function_context &ctx = *m_ctx;

    Function_instance inst(get_call_instance(call_expr));
    LLVM_context_data *p_data = call_expr->get_callee_context(inst);

    mi::mdl::IDefinition const    *def    = call_expr->get_callee_definition();
    mi::mdl::IType_function const *f_type = cast<mi::mdl::IType_function>(def->get_type());

    Func_deriv_info const *func_deriv_info = NULL;
    if (m_deriv_infos != NULL) {
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

        Expression_result expr_res = call_expr->translate_argument(i, arg_is_deriv);

        IType const   *p_type;
        ISymbol const *dummy;
        f_type->get_parameter(i, p_type, dummy);

        // instantiate the parameter type, we are calling an instance using the argument instances
        // of this call
        Storage_modifier mod = inst.get_parameter_storage_modifier(i);
        if (mod != SM_NORMAL) {
            if (!expr_res.is_offset()) {
                error(INTERNAL_JIT_BACKEND_ERROR, "Offset required for non-normal storage space");
                MDL_ASSERT(!"Offset required for non-normal storage space");
                args.push_back(llvm::UndefValue::get(lookup_type(arg_type)));
            } else {
                args.push_back(expr_res.get_offset());
            }
            continue;
        }

        // make sure, the argument is a derivative if needed, after offsets have been handled
        expr_res.ensure_deriv_result(ctx, arg_is_deriv);

        int param_imm_size = instantiate_call_param_type_size(p_type, inst.get_array_instances());

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

        if (m_type_mapper.is_passed_by_reference(
                m_type_mapper.skip_deriv_type(arg_type), param_imm_size) ||
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

    ctx.set_curr_pos(call_expr);

    // call it
    llvm::Value *res = ctx->CreateCall(callee, args);

    if (sret_res != NULL) {
        // the result was passed on the stack
        return Expression_result::ptr(sret_res);
    }
    return Expression_result::value(res);
}

// Translate a DAG expression lambda call to LLVM IR.
Expression_result LLVM_code_generator::translate_dag_call_lambda(
    ICall_expr const *call_expr)
{
    MDL_ASSERT(call_expr->get_semantics() == IDefinition::DS_INTRINSIC_DAG_CALL_LAMBDA);

    DAG_call const *dag_call = call_expr->as_dag_call();
    MDL_ASSERT(dag_call != NULL && "Cannot convert lambda call to DAG node");

    size_t lambda_index = strtoul(dag_call->get_name(), NULL, 10);
    return translate_precalculated_lambda(
        lambda_index,
        m_type_mapper.lookup_type(m_llvm_context, dag_call->get_type()));
}

/// Return the length of a vector3 (fourth component ignored).
static float len_v3(Float4_struct const &v)
{
    return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

// Translate a state::transform_*() call expression to LLVM IR.
Expression_result LLVM_code_generator::translate_transform_call(
    IDefinition::Semantics sema,
    ICall_expr const       *call_expr)
{
    Function_context &ctx = *m_ctx;

    // will potentially use the transforms
    m_state_usage_analysis.add_state_usage(
        ctx.get_function(), IGenerated_code_executable::SU_TRANSFORMS);

    llvm::Type  *ret_tp = lookup_type(call_expr->get_type());
    llvm::Value *from   = call_expr->translate_argument(0, /*wants_derivs=*/ false).as_value(ctx);
    llvm::Value *to     = call_expr->translate_argument(1, /*wants_derivs=*/ false).as_value(ctx);

    int sp_encoding = coordinate_world;
    if (strcmp(m_internal_space.c_str(), "coordinate_object") == 0) {
        sp_encoding = coordinate_object;
    }

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
            MDL_ASSERT(!"SMALL-vector mode NYI");

            res = ctx->CreateLoad(result);
            return Expression_result::value(res);
        }
        break;

    case mi::mdl::IDefinition::DS_INTRINSIC_STATE_TRANSFORM_POINT:
        {
            // TODO: derivatives
            llvm::Value *point  =
                call_expr->translate_argument(2, /*wants_derivs=*/ false).as_value(ctx);

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
                    llvm::FixedVectorType *vec_tp = llvm::cast<llvm::FixedVectorType>(ret_tp);

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
                call_expr->translate_argument(2, /*wants_derivs=*/ false).as_value(ctx);

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
                    llvm::FixedVectorType *vec_tp = llvm::cast<llvm::FixedVectorType>(ret_tp);

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
                call_expr->translate_argument(2, /*wants_derivs=*/ false).as_value(ctx);

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
                llvm::Value *c_res = ctx.get_constant((v_x + v_y + v_z) / 3.0f);
                res = ctx->CreateFMul(c_res, scale);

                ctx->CreateStore(c_res, result);
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
Expression_result LLVM_code_generator::translate_object_id_call()
{
    Function_context &ctx = *m_ctx;

    // uses the object ID
    m_state_usage_analysis.add_state_usage(
        ctx.get_function(), IGenerated_code_executable::SU_OBJECT_ID);

    llvm::Value *res = ctx.get_object_id_value();
    return Expression_result::value(res);
}

// Translate a conversion call to LLVM IR.
Expression_result LLVM_code_generator::translate_conversion(
    mi::mdl::ICall_expr const *call_expr)
{
    Function_context &ctx = *m_ctx;

    MDL_ASSERT(call_expr->get_argument_count() == 1);

    mi::mdl::IType const *res_type  = call_expr->get_type()->skip_type_alias();
    mi::mdl::IType const *arg_type  = call_expr->get_argument_type(0)->skip_type_alias();
    mi::mdl::IType const *arg_noderiv_type = m_type_mapper.skip_deriv_type(arg_type);
    mi::mdl::IType const *res_noderiv_type = m_type_mapper.skip_deriv_type(res_type);

    bool return_derivs = call_expr->returns_derivatives();

    // we only need derivatives for the argument, if the conversion results should be derivable
    // and the argument type supports it
    bool arg_derivs = return_derivs && m_type_mapper.is_floating_point_based_type(arg_noderiv_type);

    llvm::Value *v = call_expr->translate_argument(0, arg_derivs).as_value(ctx);

    llvm::Value *res;

    // will the conversion result in a dual value?
    if (return_derivs && m_type_mapper.is_floating_point_based_type(res_noderiv_type)) {

        // is the translated argument a dual value?
        if (m_type_mapper.is_deriv_type(v->getType())) {
            // dual -> dual: convert the dual component-wise

            llvm::Value *val = translate_conversion(
                res_noderiv_type, arg_noderiv_type, ctx.get_dual_val(v));
            llvm::Value *dx  = translate_conversion(
                res_noderiv_type, arg_noderiv_type, ctx.get_dual_dx(v));
            llvm::Value *dy  = translate_conversion(
                res_noderiv_type, arg_noderiv_type, ctx.get_dual_dy(v));

            res = ctx.get_dual(val, dx, dy);
        } else {
            // non-dual -> dual: convert, then get dual
            res = translate_conversion(res_noderiv_type, arg_noderiv_type, v);
            res = ctx.get_dual(res);
        }
    } else if (m_type_mapper.is_deriv_type(v->getType())) {
        // dual -> non-dual: strip dual, then convert
        res = translate_conversion(res_noderiv_type, arg_noderiv_type, ctx.get_dual_val(v));
    } else {
        // non-dual -> non-dual: just convert
        res = translate_conversion(res_noderiv_type, arg_noderiv_type, v);
    }

    return Expression_result::value(res);
}

// Translate a conversion call to LLVM IR.
// Note: Function may not be called with dual values and types
llvm::Value *LLVM_code_generator::translate_conversion(
    mi::mdl::IType const *res_type,
    mi::mdl::IType const *arg_type,
    llvm::Value          *v)
{
    Function_context &ctx = *m_ctx;

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
                    if (src == tgt) {
                        res = v;
                    } else if (src->getPrimitiveSizeInBits() < tgt->getPrimitiveSizeInBits()) {
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
        return translate_color_conversion(arg_type, v);

    case mi::mdl::IType::TK_VECTOR:
        // conversion to vector type
        return
            translate_vector_conversion(cast<mi::mdl::IType_vector>(res_type), arg_type, v);

    case mi::mdl::IType::TK_MATRIX:
        // conversion to matrix type
        return
            translate_matrix_conversion(cast<mi::mdl::IType_matrix>(res_type), arg_type, v);

    default:
        // others
        break;
    }

    if (res != NULL) {
        return res;
    }

    MDL_ASSERT(!"Conversion not implemented");
    return llvm::UndefValue::get(lookup_type(res_type));
}

// Translate a conversion call to a vector type to LLVM IR.
llvm::Value *LLVM_code_generator::translate_vector_conversion(
    mi::mdl::IType_vector const *tgt_type,
    mi::mdl::IType const        *src_type,
    llvm::Value                 *v)
{
    Function_context &ctx = *m_ctx;

    llvm::Type *tgt = lookup_type(tgt_type);
    llvm::Type *src = lookup_type(src_type);

    if (src == tgt) {
        return v;
    }

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

                elem = translate_conversion(t_elem, s_elem, elem);
                res = ctx->CreateInsertValue(res, elem, idxes);
            }
        } else {
            // convert are represented by vector types
            for (int i = 0, n = tgt_type->get_size(); i < n; ++i) {
                llvm::Value *idx  = ctx.get_constant(i);
                llvm::Value *elem = ctx->CreateExtractElement(v, idx);

                elem = translate_conversion(t_elem, s_elem, elem);
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
    mi::mdl::IType_matrix const *tgt_type,
    mi::mdl::IType const        *src_type,
    llvm::Value                 *v)
{
    Function_context &ctx = *m_ctx;

    llvm::Type *tgt = lookup_type(tgt_type);
    llvm::Type *src = lookup_type(src_type);

    if (src == tgt) {
        return v;
    }

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

                elem = translate_conversion(tv_elem, sv_elem, elem);
                res = ctx->CreateInsertValue(res, elem, idxes);
            }
        } else {
            // matrices are represented by arrays of elemental types
            mi::mdl::IType const *t_elem = tv_elem->get_element_type();
            mi::mdl::IType const *s_elem = sv_elem->get_element_type();

            for (unsigned i = 0, n = unsigned(arr_tp->getNumContainedTypes()); i < n; ++i) {
                unsigned idxes[1] = { i };
                llvm::Value *elem = ctx->CreateExtractValue(v, idxes);

                elem = translate_conversion(t_elem, s_elem, elem);
                res = ctx->CreateInsertValue(res, elem, idxes);
            }
        }
    } else {
        // matrices are represented by vector types
        mi::mdl::IType const *t_elem = tv_elem->get_element_type();
        mi::mdl::IType const *s_elem = sv_elem->get_element_type();

        llvm::FixedVectorType *v_tp = llvm::cast<llvm::FixedVectorType>(tgt);
        for (int i = 0, n = int(v_tp->getNumElements()); i < n; ++i) {
            llvm::Value *idx  = ctx.get_constant(i);
            llvm::Value *elem = ctx->CreateExtractElement(v, idx);

            elem = translate_conversion(t_elem, s_elem, elem);
            res = ctx->CreateInsertElement(res, elem, idx);
        }
    }
    return res;
}

// Translate a conversion call to the color type to LLVM IR.
llvm::Value *LLVM_code_generator::translate_color_conversion(
    mi::mdl::IType const *src_type,
    llvm::Value          *v)
{
    Function_context &ctx = *m_ctx;

    llvm::Type *tgt = m_type_mapper.get_color_type();
    llvm::Type *src = lookup_type(src_type);

    if (src == tgt) {
        return v;
    }

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
    mi::mdl::ICall_expr const *call_expr)
{
    Function_context &ctx = *m_ctx;

    llvm::Type *type = lookup_type_or_deriv_type(call_expr);
    llvm::Type *non_deriv_type = m_type_mapper.skip_deriv_type(type);
    bool return_derivs = call_expr->returns_derivatives();

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
                    llvm::Value *vec = call_expr->translate_argument(
                        col, return_derivs).as_value(ctx);
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
                    llvm::Value *vec = call_expr->translate_argument(
                        col, return_derivs).as_value(ctx);

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
            llvm::Value *v = call_expr->translate_argument(i, return_derivs).as_value(ctx);
            agg_val = ctx.create_insert(agg_val, ctx.get_dual_val(v), unsigned(i));
            agg_dx  = ctx.create_insert(agg_dx,  ctx.get_dual_dx(v),  unsigned(i));
            agg_dy  = ctx.create_insert(agg_dy,  ctx.get_dual_dy(v),  unsigned(i));
        }
        agg = ctx.get_dual(agg_val, agg_dx, agg_dy);
    } else {
        for (size_t i = 0, n = call_expr->get_argument_count(); i < n; ++i) {
            llvm::Value *v = call_expr->translate_argument(i, return_derivs).as_value(ctx);
            agg = ctx.create_insert(agg, v, unsigned(i));
        }
    }

    return Expression_result::value(agg);
}

// Translate a matrix elemental constructor call to LLVM IR.
Expression_result LLVM_code_generator::translate_matrix_elemental_constructor(
    mi::mdl::ICall_expr const *call_expr)
{
    Function_context &ctx = *m_ctx;

    llvm::Type *res_tp = lookup_type_or_deriv_type(call_expr);
    bool return_derivs = call_expr->returns_derivatives();

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

                unsigned idxs[1];

                size_t i = 0;
                for (int c = 0; c < n_col; ++c) {
                    for (int r = 0; r < n_row; ++r) {
                        llvm::Value *v   =
                            call_expr->translate_argument(i++, return_derivs).as_value(ctx);
                        llvm::Value *idx = ctx.get_constant(r);

                        vector_val = ctx->CreateInsertElement(vector_val, ctx.get_dual_val(v), idx);
                        vector_dx  = ctx->CreateInsertElement(vector_dx,  ctx.get_dual_dx(v),  idx);
                        vector_dy  = ctx->CreateInsertElement(vector_dy,  ctx.get_dual_dy(v),  idx);
                    }
                    idxs[0] = unsigned(c);
                    matrix_val = ctx->CreateInsertValue(matrix_val, vector_val, idxs);
                    matrix_dx  = ctx->CreateInsertValue(matrix_dx,  vector_dx,  idxs);
                    matrix_dy  = ctx->CreateInsertValue(matrix_dy,  vector_dy,  idxs);
                }
                matrix = ctx.get_dual(matrix_val, matrix_dx, matrix_dy);
                return Expression_result::value(matrix);
            }
        }

        // matrices are represented as arrays of scalars or as plain vectors
        for (int i = 0; i < n_col * n_row; ++i) {
            llvm::Value *v = call_expr->translate_argument(i, return_derivs).as_value(ctx);

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

                unsigned idxs[1];

                size_t i = 0;
                for (int c = 0; c < n_col; ++c) {
                    for (int r = 0; r < n_row; ++r) {
                        llvm::Value *v   = call_expr->translate_argument(
                            i++, return_derivs).as_value(ctx);
                        llvm::Value *idx = ctx.get_constant(r);

                        vector = ctx->CreateInsertElement(vector, v, idx);
                    }
                    idxs[0] = unsigned(c);
                    matrix = ctx->CreateInsertValue(matrix, vector, idxs);
                }
                return Expression_result::value(matrix);
            }
        }

        // matrices are represented as arrays of scalars or as plain vectors
        for (int i = 0; i < n_col * n_row; ++i) {
            llvm::Value *v = call_expr->translate_argument(i, return_derivs).as_value(ctx);

            matrix = ctx.create_insert(matrix, v, unsigned(i));
        }
    }
    return Expression_result::value(matrix);
}

// Translate a matrix diagonal constructor call to LLVM IR.
Expression_result LLVM_code_generator::translate_matrix_diagonal_constructor(
    mi::mdl::ICall_expr const *call_expr)
{
    Function_context &ctx = *m_ctx;

    MDL_ASSERT(call_expr->get_argument_count() == 1);

    llvm::Type *res_tp = lookup_type_or_deriv_type(call_expr);
    bool return_derivs = call_expr->returns_derivatives();
    llvm::Value *matrix = llvm::ConstantAggregateZero::get(res_tp);

    mi::mdl::IType const *res_mdl_type = call_expr->get_type()->skip_type_alias();

    mi::mdl::IType_matrix const *m_type =
        cast<mi::mdl::IType_matrix>(m_type_mapper.skip_deriv_type(res_mdl_type));

    mi::mdl::IType_vector const *v_type = m_type->get_element_type();

    llvm::Value *v = call_expr->translate_argument(0, return_derivs).as_value(ctx);

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
    ICall_expr const *call_expr)
{
    Function_context &ctx = *m_ctx;

    mi::mdl::IType const *res_type = call_expr->get_type()->skip_type_alias();
    bool return_derivs = call_expr->returns_derivatives();

    size_t n = call_expr->get_argument_count();
    if (n == 1) {
        mi::mdl::IType const *arg_type = call_expr->get_argument_type(0)->skip_type_alias();
        if (is<mi::mdl::IType_array>(arg_type)) {
            // skip array copy constructor
            MDL_ASSERT(arg_type == res_type);
            return call_expr->translate_argument(0, return_derivs);
        }
    }

    // instantiate array type
    IType_array const *a_type = cast<IType_array>(m_type_mapper.skip_deriv_type(res_type));
    llvm::Type *type = lookup_type_or_deriv_type(call_expr);

    if (a_type->is_immediate_sized()) {
        llvm::Value *res = llvm::ConstantAggregateZero::get(type);

        if (return_derivs) {
            llvm::Type *elem_type = m_type_mapper.get_deriv_base_type(type);
            llvm::Value *val = llvm::UndefValue::get(elem_type);
            llvm::Value *dx = llvm::UndefValue::get(elem_type);
            llvm::Value *dy = llvm::UndefValue::get(elem_type);
            for (size_t i = 0; i < n; ++i) {
                llvm::Value *v = call_expr->translate_argument(
                    i, /*wants_derivs=*/ true).as_value(ctx);

                val = ctx->CreateInsertValue(val, ctx.get_dual_val(v), { unsigned(i) });
                dx  = ctx->CreateInsertValue(dx,  ctx.get_dual_dx(v),  { unsigned(i) });
                dy  = ctx->CreateInsertValue(dy,  ctx.get_dual_dy(v),  { unsigned(i) });
            }
            res = ctx.get_dual(val, dx, dy);
        } else {
            for (size_t i = 0; i < n; ++i) {
                llvm::Value *v = call_expr->translate_argument(
                    i, /*wants_derivs=*/ false).as_value(ctx);

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
    mi::mdl::IExpression_let const *let_expr,
    bool                           wants_derivs)
{
    for (size_t i = 0, n = let_expr->get_declaration_count(); i < n; ++i) {
        mi::mdl::IDeclaration const *decl = let_expr->get_declaration(i);
        translate_declaration(decl);
    }
    return translate_expression(let_expr->get_expression(), wants_derivs);
}

// Create a matrix by matrix multiplication.
llvm::Value *LLVM_code_generator::do_matrix_multiplication_MxM(
    llvm::Type       *res_type,
    llvm::Value      *l,
    llvm::Value      *r,
    int              N,
    int              M,
    int              K)
{
    Function_context &ctx = *m_ctx;

    llvm::Value *res = llvm::UndefValue::get(res_type);

    llvm::ArrayType *arr_tp = llvm::cast<llvm::ArrayType>(res_type);
    llvm::Type      *e_tp   = arr_tp->getElementType();

    if (llvm::isa<llvm::VectorType>(e_tp)) {
        // small vector mode
        llvm::Type *vt_e_tp = e_tp->getScalarType();
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
        // all scalar mode
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
    return res;
}

// Create a matrix by matrix multiplication with derivatives.
llvm::Value *LLVM_code_generator::do_matrix_multiplication_MxM_deriv(
    llvm::Type       *res_type,
    llvm::Value      *l,
    llvm::Value      *r,
    int              N,
    int              M,
    int              K)
{
    Function_context &ctx = *m_ctx;

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

    llvm::ArrayType *arr_tp = llvm::cast<llvm::ArrayType>(res_val_type);
    llvm::Type      *e_tp   = arr_tp->getElementType();

    if (llvm::isa<llvm::VectorType>(e_tp)) {
        // small vector mode
        llvm::Type *vt_e_tp = e_tp->getScalarType();
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
    return ctx.get_dual(res_val, res_dx, res_dy);
}

// Create a vector by matrix multiplication.
llvm::Value *LLVM_code_generator::do_matrix_multiplication_VxM(
    llvm::Type       *res_type,
    llvm::Value      *l,
    llvm::Value      *r,
    int              M,
    int              K)
{
    Function_context &ctx = *m_ctx;

    llvm::Value *res = llvm::UndefValue::get(res_type);

    llvm::ArrayType *arr_tp = llvm::cast<llvm::ArrayType>(r->getType());
    llvm::Type      *e_tp   = arr_tp->getElementType();

    if (llvm::isa<llvm::VectorType>(e_tp)) {
        // small vector mode
        llvm::Type *vt_e_tp = e_tp->getScalarType();
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
    // else all scalar mode, do nothing

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
    llvm::Type       *res_type,
    llvm::Value      *l,
    llvm::Value      *r,
    int              M,
    int              K)
{
    Function_context &ctx = *m_ctx;

    llvm::Type  *res_vec_type = m_type_mapper.get_deriv_base_type(res_type);
    llvm::Value *res_val = llvm::UndefValue::get(res_vec_type);
    llvm::Value *res_dx = llvm::UndefValue::get(res_vec_type);
    llvm::Value *res_dy = llvm::UndefValue::get(res_vec_type);

    llvm::Value *mat_val = ctx.get_dual_val(r);
    llvm::Value *mat_dx  = ctx.get_dual_dx(r);
    llvm::Value *mat_dy  = ctx.get_dual_dy(r);

    llvm::ArrayType *arr_tp = llvm::cast<llvm::ArrayType>(mat_val->getType());
    llvm::Type      *e_tp = arr_tp->getElementType();

    if (llvm::isa<llvm::VectorType>(e_tp)) {
        // small vector mode
        llvm::Type *vt_e_tp = e_tp->getScalarType();
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
    // else all scalar mode, do nothing

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
    llvm::Type       *res_type,
    llvm::Value      *l,
    llvm::Value      *r,
    int              N,
    int              M)
{
    Function_context &ctx = *m_ctx;

    llvm::Value *res = llvm::UndefValue::get(res_type);

    llvm::ArrayType *arr_tp = llvm::cast<llvm::ArrayType>(l->getType());
    llvm::Type      *e_tp   = arr_tp->getElementType();

    if (llvm::isa<llvm::VectorType>(e_tp)) {
        // small vector mode
        llvm::Type *vt_e_tp = e_tp->getScalarType();
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
    // else all scalar mode, do nothing

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
    llvm::Type       *res_type,
    llvm::Value      *l,
    llvm::Value      *r,
    int              N,
    int              M)
{
    Function_context &ctx = *m_ctx;

    llvm::Type  *res_vec_type = m_type_mapper.get_deriv_base_type(res_type);
    llvm::Value *res_val = llvm::UndefValue::get(res_vec_type);
    llvm::Value *res_dx = llvm::UndefValue::get(res_vec_type);
    llvm::Value *res_dy = llvm::UndefValue::get(res_vec_type);

    llvm::Value *mat_val = ctx.get_dual_val(l);
    llvm::Value *mat_dx  = ctx.get_dual_dx(l);
    llvm::Value *mat_dy  = ctx.get_dual_dy(l);

    llvm::ArrayType *arr_tp = llvm::cast<llvm::ArrayType>(mat_val->getType());
    llvm::Type      *e_tp = arr_tp->getElementType();

    if (llvm::isa<llvm::VectorType>(e_tp)) {
        // small vector mode
        llvm::Type *vt_e_tp = e_tp->getScalarType();
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
    // else all scalar mode, do nothing

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
    mi::mdl::DAG_node const            *node,
    mi::mdl::ICall_name_resolver const *resolver)
{
    Node_value_map::iterator it = m_node_value_map.find(Value_entry(node, m_curr_bb));
    if (it != m_node_value_map.end()) {
        return it->second;
    }

    Expression_result res;

    switch (node->get_kind()) {
    case mi::mdl::DAG_node::EK_CONSTANT:
        // simple case: just a constant
        {
            mi::mdl::DAG_constant const *c = cast<mi::mdl::DAG_constant>(node);
            mi::mdl::IValue const *v = c->get_value();

            res = translate_value(v);
        }
        break;

    case mi::mdl::DAG_node::EK_TEMPORARY:
        // there should be no temporaries at this point
        res = translate_node(cast<mi::mdl::DAG_temporary>(node)->get_expr(), resolver);
        break;

    case mi::mdl::DAG_node::EK_CALL:
        {
            DAG_call const *call_node = cast<DAG_call>(node);
            Call_dag_expr call(*this, call_node, resolver);

            // We must be inside some module when the call is translated, but if it is DAG-call,
            // there is no module.
            // Solve this by entering the owner module of the called entity.

            IDefinition::Semantics sema = call_node->get_semantic();
            if (sema == IDefinition::DS_INTRINSIC_DAG_SET_TRANSFORMS ||
                sema == IDefinition::DS_INTRINSIC_DAG_SET_OBJECT_ID)
            {
                // these two have no module and no owner
                res = translate_call(&call);
            } else {
                char const *signature = call_node->get_name();
                if (signature[0] == '#') {
                    // skip prefix for derivative variants
                    ++signature;
                }
                mi::base::Handle<mi::mdl::Module const> mod(
                    impl_cast<mi::mdl::Module>(resolver->get_owner_module(signature)));
                MDL_module_scope scope(*this, mod.get());

                res = translate_call(&call);
            }
        }
        break;

    case mi::mdl::DAG_node::EK_PARAMETER:
        {
            DAG_parameter const *param_node = cast<DAG_parameter>(node);
            res = translate_parameter(param_node);
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

// Get the read function for a value of the given kind in the given storage space.
llvm::Function *LLVM_code_generator::get_sl_value_read_function(
    Storage_modifier     storage_space,
    mi::mdl::IType::Kind kind)
{
    bool is_argblock = storage_space == SM_PARAMETER;
    switch (kind) {
    case mi::mdl::IType::TK_BOOL:
        return is_argblock ? m_sl_funcs.m_argblock_as_bool : m_sl_funcs.m_rodata_as_bool;

    case mi::mdl::IType::TK_FLOAT:
        return is_argblock ? m_sl_funcs.m_argblock_as_float : m_sl_funcs.m_rodata_as_float;

    case mi::mdl::IType::TK_INT:
    case mi::mdl::IType::TK_ENUM:
    case mi::mdl::IType::TK_STRING:
    case mi::mdl::IType::TK_TEXTURE:
    case mi::mdl::IType::TK_LIGHT_PROFILE:
    case mi::mdl::IType::TK_BSDF_MEASUREMENT:
        return is_argblock ? m_sl_funcs.m_argblock_as_int : m_sl_funcs.m_rodata_as_int;

    case mi::mdl::IType::TK_DOUBLE:
        return is_argblock ? m_sl_funcs.m_argblock_as_double : m_sl_funcs.m_rodata_as_double;

    default:
        break;
    }

    error(INTERNAL_JIT_BACKEND_ERROR, "Unexpected sl value data kind");
    MDL_ASSERT(!"Unexpected sl value data kind");
    return is_argblock ? m_sl_funcs.m_argblock_as_int : m_sl_funcs.m_rodata_as_int;
}

// Translate a parameter or RO-data-segment offset into LLVM IR by adding
// the corresponding base offset of the state.
llvm::Value *LLVM_code_generator::translate_sl_value_offset(
    int               cur_offs,
    llvm::Value      *add_val)
{
    Function_context &ctx = *m_ctx;

    llvm::Value *result_offs = ctx.get_constant(cur_offs);
    if (add_val != NULL) {
        result_offs = ctx->CreateAdd(result_offs, add_val);
    }
    return result_offs;
}

// Translate a part of a DAG parameter or the RO-data-segment for GLSL/HLSL into LLVM IR.
Expression_result LLVM_code_generator::translate_sl_value_impl(
    Storage_modifier      storage_space,
    mi::mdl::IType const *param_type,
    int                  &cur_offs,
    llvm::Value          *add_val,
    bool                  force_as_value)
{
    Function_context &ctx = *m_ctx;

    Expression_result res;
    param_type = param_type->skip_type_alias();

    mi::mdl::IType::Kind kind = param_type->get_kind();
    switch (kind) {
    case mi::mdl::IType::TK_BOOL:
        res = Expression_result::value(ctx->CreateCall(
            get_sl_value_read_function(storage_space, kind),
            translate_sl_value_offset(cur_offs, add_val)));
        ++cur_offs;
        break;

    case mi::mdl::IType::TK_FLOAT:
        cur_offs = (cur_offs + 3) & ~3;
        res = Expression_result::value(ctx->CreateCall(
            get_sl_value_read_function(storage_space, kind),
            translate_sl_value_offset(cur_offs, add_val)));
        cur_offs += 4;
        break;

    case mi::mdl::IType::TK_INT:
    case mi::mdl::IType::TK_ENUM:
    case mi::mdl::IType::TK_STRING:
        cur_offs = (cur_offs + 3) & ~3;
        res = Expression_result::value(ctx->CreateCall(
            get_sl_value_read_function(storage_space, kind),
            translate_sl_value_offset(cur_offs, add_val)));
        cur_offs += 4;
        break;

    case mi::mdl::IType::TK_DOUBLE:
        // FIXME: check if double exists
        cur_offs = (cur_offs + 7) & ~7;
        res = Expression_result::value(ctx->CreateCall(
            get_sl_value_read_function(storage_space, kind),
            translate_sl_value_offset(cur_offs, add_val)));
        cur_offs += 8;
        break;

    case mi::mdl::IType::TK_VECTOR:
    case mi::mdl::IType::TK_MATRIX:
    case mi::mdl::IType::TK_COLOR:
    case mi::mdl::IType::TK_STRUCT:
        {
            mi::mdl::IType_compound const *ct = cast<mi::mdl::IType_compound>(param_type);
            llvm::Type *res_type = m_type_mapper.lookup_type(m_llvm_context, ct);
            size_t size = size_t(m_data_layout.getTypeAllocSize(res_type));
            size_t align = size_t(m_data_layout.getABITypeAlignment(res_type));
            cur_offs = (cur_offs + align - 1) & ~(align - 1);
            int compound_start_offs = cur_offs;

            llvm::Value *res_value = llvm::UndefValue::get(res_type);
            for (int i = 0, n = ct->get_compound_size(); i < n; ++i) {
                mi::mdl::IType const *et = ct->get_compound_type(i);
                res_value = ctx.create_insert(
                    res_value,
                    translate_sl_value_impl(
                        storage_space, et, cur_offs, add_val,
                        /*force_as_value=*/ true).as_value(ctx),
                    unsigned(i));
            }

            res = Expression_result::value(res_value);

            // compound values might have an higher alignment then the sum of its components
            cur_offs = compound_start_offs + size;
        }
        break;

    case mi::mdl::IType::TK_ARRAY:
        {
            mi::mdl::IType_array const *at = cast<mi::mdl::IType_array>(param_type);
            mi::mdl::IType const *et = at->get_element_type();
            int array_size = ctx.instantiate_type_size(at);
            llvm::Type *res_type = m_type_mapper.lookup_type(m_llvm_context, at, array_size);

            // array_size not set, yet?
            if (at->is_immediate_sized()) {
                array_size = at->get_size();
            }

            size_t size = size_t(m_data_layout.getTypeAllocSize(res_type));
            size_t align = size_t(m_data_layout.getABITypeAlignment(res_type));
            cur_offs = (cur_offs + align - 1) & ~(align - 1);
            int compound_start_offs = cur_offs;

            if (force_as_value) {
                llvm::Value *res_value = llvm::UndefValue::get(res_type);
                for (int i = 0, n = array_size; i < n; ++i) {
                    res_value = ctx.create_insert(
                        res_value,
                        translate_sl_value_impl(
                            storage_space, et, cur_offs, add_val,
                            /*force_as_value=*/ true).as_value(ctx),
                        unsigned(i));
                }

                res = Expression_result::value(res_value);
            } else {
                // LLVM arrays as values are usually pretty much useless and need to be
                // converted to a pointer to an array when actually used. So directly
                // store this in a local variable
                llvm::Value *res_var = ctx.create_local(res_type, "array_value");
                for (int i = 0, n = array_size; i < n; ++i) {
                    llvm::Value *gep = ctx.create_simple_gep_in_bounds(res_var, i);
                    ctx->CreateStore(
                        translate_sl_value_impl(
                            storage_space, et, cur_offs, add_val,
                            /*force_as_value=*/ true).as_value(ctx),
                        gep);
                }

                res = Expression_result::ptr(res_var);
            }

            // compound values might have an higher alignment then the sum of its components
            cur_offs = compound_start_offs + size;
        }
        break;

    case mi::mdl::IType::TK_TEXTURE:
    case mi::mdl::IType::TK_LIGHT_PROFILE:
    case mi::mdl::IType::TK_BSDF_MEASUREMENT:
        // resources are mapped to integer in GLSL/HLSL
        cur_offs = (cur_offs + 3) & ~3;
        res = Expression_result::value(ctx->CreateCall(
            get_sl_value_read_function(storage_space, kind),
            translate_sl_value_offset(cur_offs, add_val)));
        cur_offs += 4;
        break;

    default:
        MDL_ASSERT(!"Unexpected parameter type");
        res = Expression_result::value(
            llvm::UndefValue::get(m_type_mapper.lookup_type(m_llvm_context, param_type)));
        break;
    }
    return res;
}

// Translate a part of a DAG parameter or the RO-data-segment for GLSL/HLSL into LLVM IR.
Expression_result LLVM_code_generator::translate_sl_value(
    Storage_modifier      storage_space,
    mi::mdl::IType const *param_type,
    int                  &cur_offs,
    llvm::Value          *add_val,
    bool                  force_as_value)
{
    Function_context &ctx = *m_ctx;

    // add base offset of argument block or RO-data-segment to add_val

    llvm::Value *state = ctx.get_state_parameter();
    llvm::Value *adr   = ctx.create_simple_gep_in_bounds(
        state,
        ctx.get_constant(
            m_type_mapper.get_state_index(
                storage_space == SM_PARAMETER
                ? Type_mapper::STATE_CORE_ARG_BLOCK_OFFSET
                : Type_mapper::STATE_CORE_RO_DATA_SEG)
        ));
    llvm::Value *base_offs = ctx->CreateLoad(adr);
    if (add_val != NULL) {
        add_val = ctx->CreateAdd(base_offs, add_val);
    } else {
        add_val = base_offs;
    }

    return translate_sl_value_impl(
        storage_space, param_type, cur_offs, add_val, force_as_value);
}

// Translate a DAG parameter into LLVM IR
Expression_result LLVM_code_generator::translate_parameter(
    mi::mdl::DAG_parameter const *param_node)
{
    Function_context &ctx = *m_ctx;

    if (target_is_structured_language()) {
        // TODO: Maybe use custom datalayout for GLSL/HLSL
        llvm::DataLayout const   *dl = get_target_layout_data();
        llvm::StructLayout const *sl = dl->getStructLayout(m_captured_args_type);
        int param_offs = int(sl->getElementOffset(param_node->get_index()));

        mi::mdl::IType const *param_type = param_node->get_type();
        return Expression_result::offset(
            ctx.get_constant(param_offs),
            Expression_result::OK_ARG_BLOCK,
            lookup_type(param_type, ctx.instantiate_type_size(param_type)),
            param_type);
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
        Owner_func_inst_pair const &entry = m_functions_q.front();
        Function_instance const    &func_inst = entry.get_instance();
        mi::mdl::Module const      *owner      = entry.get_owner();

        LLVM_context_data *p_data = get_context_data(func_inst);
        llvm::Function    *func   = p_data->get_function();

        if (func->isDeclaration()) {
            mi::mdl::IDefinition const *func_def = func_inst.get_def();
            mi::mdl::IDeclaration_function const *func_decl =
                cast<mi::mdl::IDeclaration_function>(func_def->get_declaration());

            if (func_decl != NULL) {
                MDL_module_scope scope(*this, owner);
                if (m_deriv_infos != NULL) {
                    m_cur_func_deriv_info = m_deriv_infos->get_function_derivative_infos(func_inst);
                }

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
        if (mod_fname != NULL && mod_fname[0] == '\0') {
            mod_fname = NULL;
        }
        string filename(mod_fname == NULL ? "<unknown>" : mod_fname, alloc);
        string directory(alloc);

        size_t pos = filename.rfind('/');
        if (pos != string::npos) {
            directory = filename.substr(0, pos);
            filename  = filename.substr(pos + 1);
        } else {
            pos = filename.rfind('\\');
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
    builder.OptLevel         = m_opt_level;
    builder.AvoidPointerPHIs = target_is_structured_language();
    builder.EnableVectorizer = !target_is_structured_language();
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
            llvm::MDNode   *op = mod_flags->getOperand(i);
            llvm::MDString *id = llvm::cast<llvm::MDString>(op->getOperand(1));

            if (id->getString() == "wchar_size") {
                llvm::Metadata *wchar_size = llvm::ConstantAsMetadata::get(
                    llvm::ConstantInt::get(llvm::IntegerType::get(m_llvm_context, 32), 4));
                llvm::Metadata *flag_ops[] = { op->getOperand(0), id, wchar_size };
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
    if (m_use_ro_data_segment) {
        create_ro_segment();
    }

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
        if (func->getName().startswith("gen_black_")) {
            continue;
        }

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

            // clear target triple to avoid taking 32-bit triple from libdevice
            libdevice->setTargetTriple("");

            // avoid LLVM warning on console about mixing different data layouts
            libdevice->setDataLayout(llvm_module->getDataLayout());

            // set required attributes on libdevice functions
            for (llvm::Function &func : libdevice->functions()) {
                set_llvm_function_attributes(&func, /*mark_noinline=*/false);
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

        if (!m_visible_functions.empty()) {
            // first mark all non-external functions as internal
            for (llvm::Function &func : llvm_module->functions()) {
                if (!func.isDeclaration()) {
                    func.setLinkage(llvm::GlobalValue::InternalLinkage);
                }
            }

            // now mark requested functions as external
            char const *start = m_visible_functions.c_str();
            while (start && *start) {
                char const *ptr = strchr(start, ',');
                if (ptr == NULL) {
                    ptr = start + strlen(start);
                }

                llvm::Function *func = llvm_module->getFunction(
                    llvm::StringRef(start, ptr - start));
                if (func != NULL) {
                    func->setLinkage(llvm::GlobalValue::ExternalLinkage);
                }

                start = ptr;
                if (*ptr == ',') {
                    ++start;
                }
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

    // the jitted code takes ownership of the module
    MDL_JIT_module_key module_key = m_jitted_code->add_llvm_module(module);
    return module_key;
}

/// Create the target machine for PTX code generation.
std::unique_ptr<llvm::TargetMachine> LLVM_code_generator::create_ptx_target_machine()
{
    char mcpu[16];
    char features[16];

    std::string triple = llvm::Triple("nvptx64", "nvidia", "cuda").str();

    // LLVM supports only "known" processors, so ensure that we do not pass an unsupported one
    unsigned sm_version = m_sm_version;
    if (sm_version > 90)  sm_version = 90;
    else if (sm_version == 86 || sm_version == 87 || sm_version == 89)
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
    llvm::Target const *target = llvm::TargetRegistry::lookupTarget("nvptx64", error);
    MDL_ASSERT(target != NULL);  // backend not found, should not happen

    llvm::CodeGenOpt::Level OLvl = llvm::CodeGenOpt::None;
    if (m_opt_level == 1) {
        OLvl = llvm::CodeGenOpt::Default;
    } else if (m_opt_level >= 2) {
        OLvl = llvm::CodeGenOpt::Aggressive;
    }

    llvm::TargetOptions options;
    if (m_fast_math) {
        options.UnsafeFPMath = true;
    }
    if (m_finite_math) {
        options.NoInfsFPMath = options.NoNaNsFPMath = true;
    }

    std::unique_ptr<llvm::TargetMachine> target_machine(target->createTargetMachine(
        triple, mcpu, features, options,
        llvm::None, llvm::None, OLvl));
    return target_machine;
}

// Compile the given module into PTX code.
void LLVM_code_generator::ptx_compile(
    llvm::Module *module,
    string       &code)
{
    {
        raw_string_ostream SOut(code);
        llvm::buffer_ostream Out(SOut);

        llvm::legacy::PassManager pm;

        m_ptx_target_machine->addPassesToEmitFile(
            pm, Out, nullptr, llvm::CodeGenFileType::CGFT_AssemblyFile);

        pm.run(*module);
    }

#if 0 // dump generated PTX to file
    FILE *f = fopen("MDL.ptx","wb");
    fwrite((void*)code.c_str(),1,code.size(),f);
    fclose(f);
#endif

    // create prototypes for PTX and CUDA
    for (Exported_function &exp_func : m_exported_func_list) {
        // fetch the function by name, as it may have been optimized away
        llvm::Function *func = module->getFunction(exp_func.name.c_str());
        if (func == nullptr)
            continue;  // does not exist anymore, skip
        if (func->getLinkage() == llvm::GlobalValue::InternalLinkage)
            continue;  // still exists, but won't be exported anymore

        // PTX prototype
        string p(".extern .func ", get_allocator());

        llvm::Type *ret_type = exp_func.func->getReturnType();
        bool returns_value = !ret_type->isVoidTy();

        if (returns_value) {
            switch (ret_type->getTypeID()) {
            case llvm::Type::IntegerTyID:  // bool and int
            case llvm::Type::FloatTyID:
                p += "(.param .b32 func_retval0) ";
                break;

            case llvm::Type::DoubleTyID:
                p += "(.param .b64 func_retval0) ";
                break;

            case llvm::Type::FixedVectorTyID:
                p += "(.param .align ";
                p += std::to_string(m_data_layout.getABITypeAlignment(ret_type)).c_str();
                p += " .b8 func_retval0[";
                p += std::to_string(m_data_layout.getTypeAllocSize(ret_type)).c_str();
                p += "]) ";
                break;

            default:
                MDL_ASSERT(!"Unexpected return type");
                p += "<INVALID RETURN TYPE> ";  // add syntax error to prototype
                break;
            }
        }

        p += exp_func.name;

        if (exp_func.function_kind == IGenerated_code_executable::FK_DF_INIT) {
            p += "(.param .b64 a, .param .b64 b, .param .b64 c);";
        } else {
            if (!returns_value) {
                if (exp_func.function_kind == IGenerated_code_executable::FK_SWITCH_LAMBDA) {
                    p += "(.param .b64 a, .param .b64 b, .param .b64 c, .param .b64 d, .param .b64 e);";
                }
                else {
                    p += "(.param .b64 a, .param .b64 b, .param .b64 c, .param .b64 d);";
                }
            } else {
                if (exp_func.function_kind == IGenerated_code_executable::FK_SWITCH_LAMBDA) {
                    p += "(.param .b64 a, .param .b64 b, .param .b64 c, .param .b64 d);";
                }
                else {
                    p += "(.param .b64 a, .param .b64 b, .param .b64 c);";
                }
            }
        }

        exp_func.set_function_prototype(IGenerated_code_executable::PL_PTX, p.c_str());

        // CUDA prototype
        p = "extern \"C\" ";

        if (!returns_value) {
            p += "void";
        } else {
            unsigned num_elems = 1;
            llvm::Type *elem_type = ret_type;
            if (llvm::FixedVectorType *vt = llvm::dyn_cast<llvm::FixedVectorType>(ret_type)) {
                num_elems = vt->getNumElements();
                elem_type = vt->getElementType();
            }

            switch (elem_type->getTypeID()) {
            case llvm::Type::IntegerTyID:
                {
                    llvm::IntegerType *int_tp = llvm::cast<llvm::IntegerType>(elem_type);
                    if (int_tp->getBitWidth() <= 8) {
                        if (num_elems == 1) {
                            p += "bool";
                        } else {
                            p += "uchar";  // CUDA has builtin uchar vectors, but no bool vectors
                        }
                    }
                    else {
                        p += "int";
                    }
                    break;
                }
            case llvm::Type::FloatTyID:
                p += "float";
                break;
            case llvm::Type::DoubleTyID:
                p += "double";
                break;
            default:
                MDL_ASSERT(!"Unexpected return type");
                p += "<INVALID RETURN TYPE> ";  // add syntax error to prototype
                break;
            }

            // for vectors, add vector size
            if (num_elems > 1) {
                p += std::to_string(num_elems).c_str();
            }
        }

        p += " __device__ ";
        p += exp_func.name;

        if (exp_func.function_kind == IGenerated_code_executable::FK_DF_INIT) {
            p += "(void *, void *, void *);";
        } else {
            if (!returns_value) {
                if (exp_func.function_kind == IGenerated_code_executable::FK_SWITCH_LAMBDA) {
                    p += "(void *, void *, void *, void *, int);";
                } else {
                    p += "(void *, void *, void *, void *);";
                }
            } else {
                if (exp_func.function_kind == IGenerated_code_executable::FK_SWITCH_LAMBDA) {
                    p += "(void *, void *, void *, int);";
                } else {
                    p += "(void *, void *, void *);";
                }
            }
        }

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
llvm::Value *LLVM_code_generator::get_w2o_transform_value()
{
    Function_context &ctx = *m_ctx;

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
llvm::Value *LLVM_code_generator::get_o2w_transform_value()
{
    Function_context &ctx = *m_ctx;

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
    MDL_ASSERT(m_target_lang == ICode_generator::TL_NATIVE);
    // FIXME: Currently the non-instancing code path is broken: When arrays are passed, only
    // the array descriptors are copied, allowing data to be modified, so disable it
    // m_enable_instancing = false;
}

// Get the address of a JIT compiled LLVM function.
void *LLVM_code_generator::get_entry_point(MDL_JIT_module_key module_key, char const* func_name)
{
    return m_jitted_code->jit_compile(module_key, func_name, *this);
}

// Get the number of error messages.
size_t LLVM_code_generator::get_error_message_count()
{
    return m_messages.get_error_message_count();
}

// Add a JIT backend warning message to the messages.
void LLVM_code_generator::warning(int code, Exc_location const &loc, Error_params const &params)
{
    char const *fname = loc.get_module()->get_filename();
    size_t mod_id = 0;
    if (fname != NULL) {
        mod_id = m_messages.register_fname(fname);
    }
    string msg(m_messages.format_msg(code, MESSAGE_CLASS, params));
    m_messages.add_warning_message(code, MESSAGE_CLASS, mod_id, loc.get_position(), msg.c_str());
}

// Add a JIT backend error message to the messages.
void LLVM_code_generator::error(int code, Error_params const &params)
{
    string msg(m_messages.format_msg(code, MESSAGE_CLASS, params));
    m_messages.add_error_message(code, MESSAGE_CLASS, 0, NULL, msg.c_str());
}

// Add a JIT backend error message to the messages.
void LLVM_code_generator::error(int code, char const *str_param)
{
    error(code, Error_params(get_allocator()).add(str_param));
}

// Mark a function as AlwaysInline if it has pointer typed parameters and the current
// target does not support pointer.
void LLVM_code_generator::always_inline_if_pointer_parameters(llvm::Function &f) const
{
    if (!target_supports_pointers()) {
        // mark all functions WITH pointer parameters as force-inline
        for (llvm::Argument const &arg : f.args()) {
            llvm::Type *tp = arg.getType();

            if (tp->isPointerTy()) {
                // has at least one pointer argument, mark as always inline
                f.addFnAttr(llvm::Attribute::AlwaysInline);
                break;
            }
        }
    }
}

// Find the definition of a signature of a standard library function.
mi::mdl::IDefinition const *LLVM_code_generator::find_stdlib_signature(
    char const *module_name,
    char const *signature) const
{
    return m_compiler->find_stdlib_signature(module_name, signature);
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
    Resource_table_kind kind)
{
    Function_context &ctx = *m_ctx;

    if (m_lut_info[kind].m_get_lut == NULL) {
        init_dummy_attribute_table(kind);
    }

    return ctx->CreateCall(m_lut_info[kind].m_get_lut);
}

// Get a attribute lookup table size.
llvm::Value *LLVM_code_generator::get_attribute_table_size(
    Resource_table_kind kind)
{
    Function_context &ctx = *m_ctx;

    if (m_lut_info[kind].m_get_lut_size == NULL) {
        init_dummy_attribute_table(kind);
    }

    return ctx->CreateCall(m_lut_info[kind].m_get_lut_size);
}

// Add a texture attribute table.
void LLVM_code_generator::add_texture_attribute_table(
    Texture_table const &table)
{
    if (table.empty()) {
        return;
    }

    // ensure there is enough space in the texture table
    if (m_texture_table.size() < table.size()) {
        m_texture_table.resize(table.size());
    }

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
        set_llvm_function_attributes(lut_func, /*mark_noinline=*/false);

        m_lut_info[RTK_TEXTURE].m_get_lut = lut_func;

        llvm::Function *size_func = llvm::Function::Create(
            llvm::FunctionType::get(m_type_mapper.get_int_type(), /*isVarArg=*/false),
            llvm::GlobalValue::InternalLinkage,
            "get_texture_attr_table_size",
            m_module);
        set_llvm_function_attributes(size_func, /*mark_noinline=*/false);

        m_lut_info[RTK_TEXTURE].m_get_lut_size = size_func;
    }
}

// Creates the light profile attribute table.
void LLVM_code_generator::add_light_profile_attribute_table(
    Light_profile_table const &table)
{
    size_t n = table.size();
    if (n == 0) {
        return;
    }

    // ensure there is enough space in the light profile table
    if (m_light_profile_table.size() < n) {
        m_light_profile_table.resize(n);
    }

    // update the light profile table with any valid entries
    for (size_t i = 0; i < n; ++i) {
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
        set_llvm_function_attributes(lut_func, /*mark_noinline=*/false);

        m_lut_info[RTK_LIGHT_PROFILE].m_get_lut = lut_func;

        llvm::Function *size_func = llvm::Function::Create(
            llvm::FunctionType::get(m_type_mapper.get_int_type(), /*isVarArg=*/false),
            llvm::GlobalValue::InternalLinkage,
            "get_light_profile_attr_table_size",
            m_module);
        set_llvm_function_attributes(size_func, /*mark_noinline=*/false);

        m_lut_info[RTK_LIGHT_PROFILE].m_get_lut_size = size_func;
    }
}

// Creates the bsdf measurement attribute table.
void LLVM_code_generator::add_bsdf_measurement_attribute_table(
    Bsdf_measurement_table const &table)
{
    size_t n = table.size();
    if (n == 0) {
        return;
    }

    // ensure there is enough space in the bsdf measurement table
    if (m_bsdf_measurement_table.size() < n) {
        m_bsdf_measurement_table.resize(n);
    }

    // update the bsdf measurement table with any valid entries
    for (size_t i = 0; i < n; ++i) {
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
        set_llvm_function_attributes(lut_func, /*mark_noinline=*/false);

        m_lut_info[RTK_BSDF_MEASUREMENT].m_get_lut = lut_func;

        llvm::Function *size_func = llvm::Function::Create(
            llvm::FunctionType::get(m_type_mapper.get_int_type(), /*isVarArg=*/false),
            llvm::GlobalValue::InternalLinkage,
            "get_bsdf_measurement_attr_table_size",
            m_module);
        set_llvm_function_attributes(size_func, /*mark_noinline=*/false);

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
    set_llvm_function_attributes(lut_func, /*mark_noinline=*/false);

    m_lut_info[RTK_STRINGS].m_get_lut = lut_func;

    llvm::Function *size_func = llvm::Function::Create(
        llvm::FunctionType::get(m_type_mapper.get_int_type(), /*isVarArg=*/false),
        llvm::GlobalValue::InternalLinkage,
        "get_string_table_size",
        m_module);
    set_llvm_function_attributes(size_func, /*mark_noinline=*/false);

    m_lut_info[RTK_STRINGS].m_get_lut_size = size_func;
}

// Add a string constant to the string table.
void LLVM_code_generator::add_string_constant(
    char const       *s,
    Type_mapper::Tag id)
{
    if (m_lut_info[RTK_STRINGS].m_get_lut == NULL) {
        init_string_attribute_table();
    }

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
    if (id < m_string_table.size()) {
        return m_string_table[id].c_str();
    }
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
    if (func == NULL) {
        return;
    }

    for (auto ui = func->user_begin(); ui != func->user_end(); ) {
        llvm::CallInst *call = llvm::dyn_cast<llvm::CallInst>(*ui);

        // current instruction might get replaced, so advance the iterator here
        ++ui;

        if (call != NULL) {
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
    if (strcmp(name, "vtable") == 0) {
        return Function_context::TLCM_VTABLE;
    } else if (strcmp(name, "direct_call") == 0) {
        return Function_context::TLCM_DIRECT;
    } else if (strcmp(name, "optix_cp") == 0) {
        return Function_context::TLCM_OPTIX_CP;
    }
    return Function_context::TLCM_VTABLE;
}

// Parse a return mode option.
LLVM_code_generator::Return_mode LLVM_code_generator::parse_return_mode(char const *name)
{
    if (strcmp(name, "sret") == 0) {
        return Return_mode::RETMODE_SRET;
    } else if (strcmp(name, "value") == 0) {
        return Return_mode::RETMODE_VALUE;
    }
    // default case
    return target_supports_sret_for_lambda()
        ? Return_mode::RETMODE_SRET : Return_mode::RETMODE_VALUE;
}

/// Parse the Df_handle_slot_mode
mi::mdl::Df_handle_slot_mode LLVM_code_generator::parse_df_handle_slot_mode(char const *name)
{
    if (strcmp(name, "none") == 0) {
        return mi::mdl::DF_HSM_NONE;
    } else if (strcmp(name, "pointer") == 0) {
        return mi::mdl::DF_HSM_POINTER;
    } else if (strcmp(name, "fixed_1") == 0) {
        return mi::mdl::DF_HSM_FIXED_1;
    } else if (strcmp(name, "fixed_2") == 0) {
        return mi::mdl::DF_HSM_FIXED_2;
    } else if (strcmp(name, "fixed_4") == 0) {
        return mi::mdl::DF_HSM_FIXED_4;
    } else if (strcmp(name, "fixed_8") == 0) {
        return mi::mdl::DF_HSM_FIXED_8;
    }

    return mi::mdl::DF_HSM_NONE;
}

// If the given definition has the since flags set, find the latest MDL version.
IDefinition const *LLVM_code_generator::promote_to_highest_version(
    IDefinition const *idef,
    unsigned          &promote)
{
    promote = LLVM_code_generator::PR_NONE;

    Definition const *def = impl_cast<Definition>(idef);

    unsigned ver_flags = def->get_version_flags();

    if (!is_mdl_removed_version(ver_flags)) {
        // this definition is still current
        return def;
    }

    // try to find the latest
    Definition const *ndef = def->get_next_def();
    for (; ndef != NULL;) {
        unsigned n_ver_flags = ndef->get_version_flags();

        if (!is_mdl_removed_version(n_ver_flags)) {
            // found the latest version
            break;
        }
    }

    if (ndef != NULL) {
        switch (ndef->get_semantics()) {
        case IDefinition::DS_INTRINSIC_TEX_WIDTH:
        case IDefinition::DS_INTRINSIC_TEX_HEIGHT:
        case IDefinition::DS_INTRINSIC_TEX_DEPTH:
            {
                IType_function const *f_tp = cast<IType_function>(def->get_type());

                ISymbol const *dummy;
                IType const *p_tp;
                f_tp->get_parameter(0, p_tp, dummy);

                if (is_tex_2d(p_tp)) {
                    // we have 2 promotions here
                    if (f_tp->get_parameter_count() < 2) {
                        // add uv_tile
                        promote |= LLVM_code_generator::PR_ADD_ZERO_INT2;
                    }
                    if (f_tp->get_parameter_count() < 3) {
                        // add frame
                        promote |= LLVM_code_generator::PR_ADD_ZERO_FLOAT;
                    }
                } else if (is_tex_3d(p_tp)) {
                    if (f_tp->get_parameter_count() < 2) {
                        // add frame
                        promote |= LLVM_code_generator::PR_ADD_ZERO_FLOAT;
                    }
                }
            }
            break;
        case IDefinition::DS_INTRINSIC_TEX_LOOKUP_FLOAT:
        case IDefinition::DS_INTRINSIC_TEX_LOOKUP_FLOAT2:
        case IDefinition::DS_INTRINSIC_TEX_LOOKUP_FLOAT3:
        case IDefinition::DS_INTRINSIC_TEX_LOOKUP_FLOAT4:
        case IDefinition::DS_INTRINSIC_TEX_LOOKUP_COLOR:
            {
                IType_function const *f_tp = cast<IType_function>(def->get_type());

                ISymbol const *dummy;
                IType const *p_tp;
                f_tp->get_parameter(0, p_tp, dummy);

                if (is_tex_2d(p_tp)) {
                    // we have 2 promotions here
                    if (f_tp->get_parameter_count() == 6) {
                        // add frame
                        promote |= LLVM_code_generator::PR_ADD_ZERO_FLOAT;
                    }
                } else if (is_tex_3d(p_tp)) {
                    if (f_tp->get_parameter_count() == 8) {
                        // add frame
                        promote |= LLVM_code_generator::PR_ADD_ZERO_FLOAT;
                    }
                }
            }
            break;
        case IDefinition::DS_INTRINSIC_TEX_TEXEL_FLOAT:
        case IDefinition::DS_INTRINSIC_TEX_TEXEL_FLOAT2:
        case IDefinition::DS_INTRINSIC_TEX_TEXEL_FLOAT3:
        case IDefinition::DS_INTRINSIC_TEX_TEXEL_FLOAT4:
        case IDefinition::DS_INTRINSIC_TEX_TEXEL_COLOR:
            {
                IType_function const *f_tp = cast<IType_function>(def->get_type());

                ISymbol const *dummy;
                IType const *p_tp;
                f_tp->get_parameter(0, p_tp, dummy);

                if (is_tex_2d(p_tp)) {
                    // we have 2 promotions here
                    if (f_tp->get_parameter_count() < 3) {
                        // add uv_tile
                        promote |= LLVM_code_generator::PR_ADD_ZERO_INT2;
                    }
                    if (f_tp->get_parameter_count() < 4) {
                        // add frame
                        promote |= LLVM_code_generator::PR_ADD_ZERO_FLOAT;
                    }
                } else if (is_tex_3d(p_tp)) {
                    if (f_tp->get_parameter_count() < 3) {
                        // add frame
                        promote |= LLVM_code_generator::PR_ADD_ZERO_FLOAT;
                    }
                }
            }
            break;

        default:
            break;
        }
        MDL_ASSERT(promote != LLVM_code_generator::PR_NONE && "unexpected promotion");
        if (promote != LLVM_code_generator::PR_NONE) {
            return ndef;
        }
    }

    return def;
}

// Get a unique string value object used to represent the string of the value.
mi::mdl::IValue_string const *LLVM_code_generator::get_internalized_string(
    mi::mdl::IValue_string const *s)
{
    LLVM_code_generator::Internalized_string_map::iterator it =
        m_internalized_string_map.find(s->get_value());
    if (it != m_internalized_string_map.end()) {
        return it->second;
    }

    // the given string value object will be our representative for the contained cstring
    m_internalized_string_map[s->get_value()] = s;
    return s;
}

// Fills the function remap map from a comma separated list.
void Function_remap::fill_function_remap(
    char const *list)
{
    for (char const *e = nullptr, *s = list; s != nullptr; s = e != nullptr ? e + 1 : nullptr) {
        char const *start = s;
        while (*start != '\0' && isspace(*start)) {
            ++start;
        }

        e = strchr(start, ',');

        char const *p = strchr(start, '=');
        if (p == nullptr || (p >= e && e != nullptr)) {
            // wrong entry
            continue;
        }

        char const *end = p - 1;
        while (end > start && isspace(*end)) {
            --end;
        }

        if (start == end) {
            // wrong entry
            continue;
        }

        string from(start, end + 1, m_arena.get_allocator());

        start = p + 1;
        while (*start != '\0' && isspace(*start)) {
            ++start;
        }

        end = e == nullptr ? start + strlen(start) - 1 : e - 1;
        while (end > start && isspace(*end)) {
            --end;
        }

        if (start == end) {
            // wrong entry
            continue;
        }
        string to(start, end + 1, m_arena.get_allocator());

        char const *f_sym = Arena_strdup(m_arena, from.c_str());
        char const *t_sym = Arena_strdup(m_arena, to.c_str());

        m_first = Arena_builder(m_arena).create<Remap_entry>(f_sym, t_sym, m_first);
        m_func_remap_map[f_sym] = m_first;
    }
}

// Get the mapped name if there is any.
char const *Function_remap::get_mapper_symbol(
    char const *src) const
{
    typename Remap_map::const_iterator it(m_func_remap_map.find(src));
    if (it != m_func_remap_map.end()) {
        Remap_entry const *re = it->second;
        re->used = true;
        return re->to;
    }
    return nullptr;
}

} // mdl
} // mi

