/***************************************************************************************************
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
 **************************************************************************************************/
/// \file

#include "pch.h"

#include <llvm/ADT/SetVector.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Linker/Linker.h>

#include "mdl/compiler/compilercore/compilercore_tools.h"
#include "mdl/compiler/compilercore/compilercore_assert.h"
#include "mdl/compiler/compilercore/compilercore_errors.h"

#include "generator_jit_llvm.h"

#include "libmdlrt_bitcode.h"

namespace mi {
namespace mdl {

// Load the libmdlrt LLVM module.
std::unique_ptr<llvm::Module> LLVM_code_generator::load_libmdlrt(
    llvm::LLVMContext &llvm_context)
{
    std::unique_ptr<llvm::MemoryBuffer> mem(llvm::MemoryBuffer::getMemBuffer(
        llvm::StringRef((char const *) libmdlrt_bitcode, dimension_of(libmdlrt_bitcode)),
        "libmdlrt",
        /*RequiresNullTerminator=*/ false));
    auto mod = llvm::parseBitcodeFile(*mem.get(), llvm_context);
    if (!mod) {
        error(PARSING_LIBBSDF_MODULE_FAILED, Error_params(get_allocator()));
        MDL_ASSERT(!"Parsing libmdlrt failed");
        return nullptr;
    }
    return std::move(mod.get());
}

// Load and link libmdlrt into the current LLVM module.
bool LLVM_code_generator::load_and_link_libmdlrt(llvm::Module *llvm_module)
{
    Store<llvm::Module*> curr_mod(m_module, llvm_module);

    std::unique_ptr<llvm::Module> libmdlrt(load_libmdlrt(m_llvm_context));
    MDL_ASSERT(libmdlrt != NULL);

    // clear target triple to avoid LLVM warning on console about mixing different targets
    // when linking libmdlrt ("x86_x64-pc-win32") with libdevice ("nvptx-unknown-unknown").
    // Using an nvptx target for libbrt would cause struct parameters to be split, which we
    // try to avoid.
    libmdlrt->setTargetTriple("");

    // also avoid LLVM warning on console about mixing different data layouts
    libmdlrt->setDataLayout(llvm_module->getDataLayout());

    // collect all functions available before linking
    // note: we cannot use the function pointers, as linking removes some function declarations and
    //       may reuse the old pointers
    hash_set<string, string_hash<string> >::Type old_func_names(get_allocator());
    for (llvm::Function &f : llvm_module->functions()) {
        if (!f.isDeclaration())
            old_func_names.insert(string(f.getName().begin(), f.getName().end(), get_allocator()));
    }

    if (llvm::Linker::linkModules(*llvm_module, std::move(libmdlrt))) {
        // true means linking has failed
        error(LINKING_LIBMDLRT_FAILED, "unknown linker error");
        MDL_ASSERT(!"Linking libmdlrt failed");
        return false;
    }

    // find all functions which were added by linking the libmdlrt module
    for (llvm::Function &f : llvm_module->functions()) {
        // just a declaration or did already exist before linking? -> skip
        if (f.isDeclaration() || old_func_names.count(
                string(f.getName().begin(), f.getName().end(), get_allocator())) != 0)
            continue;

        // Found a libmdlrt function

        // remove "target-features" attribute to avoid warnings about unsupported PTX features
        // for non-PTX backends
        f.removeFnAttr("target-features");

        if (m_target_lang == LLVM_code_generator::TL_HLSL) {
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

        // make all functions from libmdlrt internal to allow global dead code elimination
        f.setLinkage(llvm::GlobalValue::InternalLinkage);

        // translate all runtime calls
        {
            Function_context ctx(
                get_allocator(),
                *this,
                &f,
                LLVM_context_data::FL_SRET,
                /*optimize_on_finalize*/false);  // don't optimize yet

            // search for all CallInst instructions and link runtime function calls to the
            // corresponding intrinsics
            for (llvm::Function::iterator BI = f.begin(), BE = f.end(); BI != BE; ++BI) {
                for (llvm::BasicBlock::iterator II = BI->begin(); II != BI->end(); ++II) {
                    if (llvm::CallInst *call = llvm::dyn_cast<llvm::CallInst>(II)) {
                        if (!translate_libmdlrt_runtime_call(call, II, ctx))
                            return false;
                    }
                }
            }
        }
    }

    return true;
}

// Translates a potential runtime call in a libmdlrt function to a call to the according
// intrinsic, converting the arguments as necessary.
bool LLVM_code_generator::translate_libmdlrt_runtime_call(
    llvm::CallInst             *call,
    llvm::BasicBlock::iterator &ii,
    Function_context           &ctx)
{
    unsigned num_params_eaten = 0;

    llvm::Function *called_func = call->getCalledFunction();
    if (called_func == NULL)
        return true;   // ignore indirect function invocation

    llvm::StringRef func_name = called_func->getName();
    if (!func_name.startswith("_Z") || !called_func->isDeclaration())
        return true;   // ignore non-mangled functions and functions with definitions

    // try to resolve the function name to the LLVM function of an intrinsic

    string demangled_name(get_allocator());
    MDL_name_mangler mangler(get_allocator(), demangled_name);
    if (!mangler.demangle(func_name.data(), func_name.size()))
        demangled_name.assign(func_name.data(), func_name.size());

    llvm::Function *func = NULL;
    LLVM_context_data *p_data = NULL;
    unsigned ret_array_size = 0;

    // find last "::" before the parameters
    size_t parenpos = demangled_name.find('(');
    size_t colonpos = demangled_name.rfind("::", parenpos);
    if (colonpos == string::npos || colonpos == 0)
        return true;  // not in a module, maybe a builtin function

    string module_name = demangled_name.substr(0, colonpos);
    string signature = demangled_name.substr(colonpos + 2);
    IDefinition const *def = m_compiler->find_stdlib_signature(
        module_name.c_str(), signature.c_str());
    if (def == NULL)
        return true;  // not one of our modules, maybe a builtin function

    func = get_intrinsic_function(def, /*return_derivs=*/ false);

    // check for MDL function with array return and retrieve array size
    MDL_ASSERT(def->get_type()->get_kind() == IType::TK_FUNCTION);
    IType_function const *mdl_func_type = static_cast<IType_function const *>(def->get_type());
    IType const *mdl_ret_type = mdl_func_type->get_return_type();
    if (mdl_ret_type->get_kind() == IType::TK_ARRAY) {
        IType_array const *mdl_array_type = static_cast<IType_array const *>(mdl_ret_type);
        MDL_ASSERT(mdl_array_type->is_immediate_sized());
        ret_array_size = unsigned(mdl_array_type->get_size());
    }

    Function_instance inst(get_allocator(), def, /*return_derivs=*/ false);
    p_data = get_context_data(inst);

    if (func == NULL) {
        MDL_ASSERT(!"Unsupported runtime function");
        return false;
    }

    // replace the call by a call to the intrinsic function adapting the arguments and
    // providing additional arguments as requested by the intrinsic

    llvm::SmallVector<llvm::Value *, 8> llvm_args;

    // For the return value, we have 5 different cases:
    //    original call         runtime reality
    //      res = f(a,b)          res = f_r(a,b)
    //      res = f(a,b)          f_r(&res,a,b)
    //      f(&res,a,b)           res = f_r(a,b)
    //      f(&res,a,b)           f_r(&res,a,b)
    //      f(a,b,&res1,&res2)    f_r(&res,a,b) with res being an array

    llvm::Type *orig_res_type = called_func->getReturnType();
    llvm::Value *orig_res_ptr = NULL;
    llvm::Value *runtime_res_ptr = NULL;

    // insert new code before the old call
    ctx->SetInsertPoint(call);

    // Original call case: f(&res,a,b)?
    if (ret_array_size == 0 && orig_res_type == m_type_mapper.get_void_type()) {
        orig_res_ptr = call->getArgOperand(0);
        orig_res_type = llvm::cast<llvm::PointerType>(orig_res_ptr->getType())->getElementType();
        ++num_params_eaten;
    }

    // Runtime call case: f_r(&res,a,b)?
    if (p_data->is_sret_return()) {
        runtime_res_ptr = ctx.create_local(p_data->get_return_type(), "runtime_call_result");
        llvm_args.push_back(runtime_res_ptr);
    }

    llvm::FunctionType *func_type = func->getFunctionType();

    // handle all remaining arguments (except for array return arguments)
    unsigned n_args = call->getNumArgOperands();
    for (unsigned i = num_params_eaten; i < n_args - ret_array_size; ++i) {
        llvm::Value *arg = call->getArgOperand(i);
        llvm::Type *arg_type = arg->getType();
        llvm::Type *param_type = func_type->getParamType(llvm_args.size());

        // are argument and parameter types identical?
        if (arg_type == param_type) {
            llvm_args.push_back(arg);
        } else {
            // no, a conversion is required
            if (llvm::isa<llvm::PointerType>(arg_type) &&
                llvm::isa<llvm::PointerType>(param_type)) {
                // convert from one memory representation to another
                llvm::PointerType *param_ptr_type = llvm::cast<llvm::PointerType>(param_type);
                llvm::Value *val = ctx.load_and_convert(param_ptr_type->getElementType(), arg);

                llvm::Value *convert_tmp_ptr = ctx.create_local(
                    param_ptr_type->getElementType(), "convert_tmp");
                ctx->CreateStore(val, convert_tmp_ptr);

                llvm_args.push_back(convert_tmp_ptr);
            } else if (llvm::isa<llvm::PointerType>(arg_type) &&
                llvm::isa<llvm::VectorType>(param_type)) {
                // load memory representation into a vector
                llvm::Value *val = ctx.load_and_convert(param_type, arg);
                llvm_args.push_back(val);
            } else {
                MDL_ASSERT(!"Unsupported parameter conversion");
                return false;
            }
        }
    }

    llvm::Value *res = ctx->CreateCall(func, llvm_args);

    // Runtime call case: f_r(&res,a,b)?
    if (runtime_res_ptr != NULL) {
        if (ret_array_size != 0) {
            res = ctx->CreateLoad(runtime_res_ptr);
        } else {
            res = ctx.load_and_convert(orig_res_type, runtime_res_ptr);
        }
    } else if (ret_array_size == 0) {
        // Case: res = f_r(a,b)
        if (res->getType() != orig_res_type) {
            // conversion to bool? -> avoid tmp var
            if (llvm::isa<llvm::IntegerType>(res->getType()) &&
                orig_res_type == llvm::IntegerType::get(m_llvm_context, 1)) {
                res = ctx->CreateICmpNE(res, llvm::ConstantInt::getNullValue(res->getType()));
            } else {
                llvm::Value *convert_tmp_ptr = ctx.create_local(res->getType(), "convert_tmp");
                ctx->CreateStore(res, convert_tmp_ptr);
                res = ctx.load_and_convert(orig_res_type, convert_tmp_ptr);
            }
        }
    }

    // Original call case: f(&res,a,b)?
    if (orig_res_ptr != NULL) {
        ctx->CreateStore(res, orig_res_ptr);
    } else if (ret_array_size != 0) {
     // Case: f(a,b,&res1,&res2)
     // Copy the result from the array into the single result arguments
        for (unsigned i = 0; i < ret_array_size; ++i) {
            uint32_t idx[1] = { i };
            llvm::Value *res_elem = ctx->CreateExtractValue(res, idx);
            ctx.convert_and_store(res_elem, call->getArgOperand(n_args - ret_array_size + i));
        }
    } else {
     // Case: res = f(a,b)
        call->replaceAllUsesWith(res);
    }

    // Remove old call and let iterator point to instruction before old call
    ii = --ii->getParent()->getInstList().erase(call);
    return true;
}

}  // mdl
}  // mi
