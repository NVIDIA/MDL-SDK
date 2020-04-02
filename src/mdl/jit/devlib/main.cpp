/******************************************************************************
 * Copyright (c) 2014-2020, NVIDIA CORPORATION. All rights reserved.
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

#include <memory>
#include <string>
#include <set>

#include <cstdlib>
#include <cstdio>

#include <llvm/Analysis/CallGraph.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/Threading.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/PassRegistry.h>

namespace llvm {
    void initializeNvRenamePass(PassRegistry &);
    void initializeDeleteUnusedPass(PassRegistry &);
};

namespace {

using namespace llvm;

/// LLVM pass that renames functions form the libdevice to use the C runtime names.
class NvRename : public llvm::ModulePass
{
public:
    typedef std::set<std::string> String_set;

    virtual bool runOnModule(llvm::Module &M) {
        if (m_roots == NULL) {
            // no root set, do not change anything
            return false;
        }

        // rename functions from the root set
        bool changed = false;
        for (auto &F : M.functions()) {
            StringRef const name = F.getName();

            if (m_roots->find(name.str()) != m_roots->end()) {
                // found a root, start marking
                if (name.startswith("__nv_")) {
                    // ensure that the name is copied here, or it will be deleted BEFORE
                    // it is entered into the symbol table
                    std::string n_name = name.substr(5).str();
                    F.setName(n_name); // skip the "__nv_"
                    changed = true;
                }
            }
        }
        return changed;
    }

    /// Default Constructor.
    NvRename() : ModulePass(ID), m_roots(NULL) {}

    /// Constructor.
    ///
    /// \param roots  the root set
    NvRename(String_set const &roots) : ModulePass(ID), m_roots(&roots) {}

public:
    static char ID; // Class identification, replacement for typeinfo

private:
    /// The names of the root functions.
    String_set const *m_roots;
};

#if 0
/// Creates our pass.
llvm::ModulePass *createNvRenamePass(NvRename::String_set const &roots) {
    return new NvRename(roots);
}
#endif

char NvRename::ID = 0;
char &NvRenameID = NvRename::ID;

/// LLVM pass that removes function from a module that cannot be reached from a given root set.
class DeleteUnused : public llvm::ModulePass
{
public:
    typedef std::set<std::string> String_set;

    virtual void getAnalysisUsage(llvm::AnalysisUsage &AU) const
    {
        AU.addRequired<llvm::CallGraphWrapperPass>();
    }

    virtual bool runOnModule(llvm::Module &M) {
        if (m_roots == NULL) {
            // no root set, do not change anything
            return false;
        }
        llvm::CallGraph &CG = getAnalysis<llvm::CallGraphWrapperPass>().getCallGraph();

        // mark the rootset and all functions called from it
        for (auto &F : M.functions()) {
            StringRef const name = F.getName();

            if (m_roots->find(name) != m_roots->end()) {
                // found a root, start marking
                llvm::CallGraphNode const *node = CG[&F];
                visit(node);
            }
        }

        // now delete all functions not marked: fix call graph, first, or
        // removeFunctionFrimModule() will fail
        bool changed = false;
        llvm::CallGraphNode *external_node = CG.getExternalCallingNode();

        for (auto &F : M.functions()) {
            // intrinsics are not in the call graph, do not try to remove them
            // neither remove address taken functions, should not happen in our case but ...
            if (!F.isIntrinsic() &&
                !F.hasAddressTaken() &&
                m_marker.find(&F) == m_marker.end())
            {
                llvm::CallGraphNode *node = CG[&F];
                node->removeAllCalledFunctions();

                // the removed functions could be external visible, in that case, remove them
                // from the external node
                if (!F.hasLocalLinkage()) {
                    external_node->removeAnyCallEdgeTo(node);
                }
                changed = true;
            }
        }

        if (!changed)
            return false;

        // now delete
        for (llvm::Module::iterator it(M.begin()), end(M.end()); it != end;) {
            llvm::Function *F = &*it;

            ++it;
            if (!F->isIntrinsic() &&
                !F->hasAddressTaken() &&
                m_marker.find(F) == m_marker.end()) {
                CG.removeFunctionFromModule(CG[F]);

                // now delete F
                if (!F->use_empty()) {
                    // there are still callers to F, however those must be also dead
                    // replace them to avoid asserts BEFORE F is deleted
                    F->replaceAllUsesWith(llvm::UndefValue::get(F->getType()));
                }
                delete F;
            }
        }
        return true;
    }

    /// Visit given node and all of its callees.
    void visit(llvm::CallGraphNode const *node) {
        llvm::Function const *F = node->getFunction();
        if (F != NULL && m_marker.insert(F).second) {
            // newly inserted
            for (llvm::CallGraphNode::const_iterator it(node->begin()), end(node->end());
                 it != end;
                 ++it)
            {
                llvm::CallGraphNode const *callee = it->second;
                visit(callee);
            }
        }
    }

    /// Default Constructor.
    DeleteUnused() : ModulePass(ID), m_roots(NULL) {}

    /// Constructor.
    ///
    /// \param roots  the root set
    DeleteUnused(String_set const &roots) : ModulePass(ID), m_roots(&roots) {}

public:
    static char ID; // Class identification, replacement for typeinfo

private:
    /// The names of the root functions.
    String_set const *m_roots;

    typedef std::set<llvm::Function const *> Marker_set;

    /// The marker set.
    Marker_set m_marker;
};

/// Creates our pass.
llvm::ModulePass *createDeleteUnusedPass(DeleteUnused::String_set const &roots) {
    return new DeleteUnused(roots);
}

char DeleteUnused::ID = 0;
char &DeleteUnusedID = DeleteUnused::ID;

}  // anonymous

INITIALIZE_PASS(NvRename, "nv-rename",
                "Rename nvidia libdevice functions", false, false)

INITIALIZE_PASS_BEGIN(DeleteUnused, "delete-unused",
                      "Delete unused functions from a module", false, false)
INITIALIZE_PASS_DEPENDENCY(CallGraphWrapperPass)
INITIALIZE_PASS_END(DeleteUnused, "delete-unused",
                      "Delete unused functions from a module", false, false)

/// Load a libdevice file.
std::unique_ptr<llvm::Module> load_libdevice(char const *filename, llvm::LLVMContext &context)
{
    auto buf = llvm::MemoryBuffer::getFile(
        filename, /*FileSize=*/-1, /*RequiresNullTerminator=*/false);
    if (buf.getError()) {
        fprintf(stderr, "Error reading file: %s\n", buf.getError().message().c_str());
        return NULL;
    }

    auto mod = llvm::parseBitcodeFile(*buf.get(), context);
    if (!mod) {
        fprintf(stderr, "Error parsing file: %s\n", filename);
        return NULL;
    }
    return std::move(mod.get());
}

/// Write a libdevice file.
void write_libdevice(llvm::Module const *libdevice, char const *filename)
{
    std::error_code error;
    llvm::raw_fd_ostream Out(filename, error, llvm::sys::fs::OpenFlags::F_None);
    if (error) {
        fprintf(stderr, "Error writing file: %s\n", error.message().c_str());
        return;
    }
    llvm::WriteBitcodeToFile(*libdevice, Out);
}

/// Process the library.
void process(llvm::Module *module)
{
    llvm::legacy::PassManager mpm;
    DeleteUnused::String_set roots;

    roots.insert("__nvvm_reflect");
    roots.insert("__nv_fabsf");
    roots.insert("__nv_fabs");
    roots.insert("__nv_abs");
    roots.insert("__nv_acosf");
    roots.insert("__nv_acos");
    roots.insert("__nv_asinf");
    roots.insert("__nv_asin");
    roots.insert("__nv_atanf");
    roots.insert("__nv_atan");
    roots.insert("__nv_atan2f");
    roots.insert("__nv_atan2");
    roots.insert("__nv_ceilf");
    roots.insert("__nv_ceil");
    roots.insert("__nv_cosf");
    roots.insert("__nv_cos");
    roots.insert("__nv_expf");
    roots.insert("__nv_exp");
    roots.insert("__nv_exp2f");
    roots.insert("__nv_exp2");
    roots.insert("__nv_floorf");
    roots.insert("__nv_floor");
    roots.insert("__nv_fmodf");
    roots.insert("__nv_fmod");
    roots.insert("__nv_logf");
    roots.insert("__nv_log");
    roots.insert("__nv_log2f");
    roots.insert("__nv_log2");
    roots.insert("__nv_log10f");
    roots.insert("__nv_log10");
    roots.insert("__nv_modff");
    roots.insert("__nv_modf");
    roots.insert("__nv_powf");
    roots.insert("__nv_pow");
    roots.insert("__nv_powi");
    roots.insert("__nv_sinf");
    roots.insert("__nv_sin");
    roots.insert("__nv_sqrtf");
    roots.insert("__nv_sqrt");
    roots.insert("__nv_tanf");
    roots.insert("__nv_tan");
    roots.insert("__nv_copysignf");
    roots.insert("__nv_copysign");
    roots.insert("__nv_sincosf");
    roots.insert("__nv_min");
    roots.insert("__nv_max");
    roots.insert("__nv_fminf");
    roots.insert("__nv_fmaxf");
    roots.insert("__nv_fmin");
    roots.insert("__nv_fmax");
    roots.insert("__nv_rsqrtf");
    roots.insert("__nv_rsqrt");

    // fast-math variants
//    roots.insert("__nv_fast_fdividef");
    roots.insert("__nv_fast_sinf");
    roots.insert("__nv_fast_cosf");
    roots.insert("__nv_fast_log2f");
    roots.insert("__nv_fast_tanf");
    roots.insert("__nv_fast_sincosf");
    roots.insert("__nv_fast_expf");
    roots.insert("__nv_fast_exp10f");
    roots.insert("__nv_fast_log10f");
    roots.insert("__nv_fast_logf");
    roots.insert("__nv_fast_powf");

    mpm.add(createDeleteUnusedPass(roots));
    mpm.add(createGlobalDCEPass());
//    mpm.add(createNvRenamePass(roots));
    mpm.run(*module);

    llvm::verifyModule(*module);
}

/// Prints the usage.
static void usage(char const *name)
{
    printf("Usage: %s infile outfile\n", name);
}

int main(int argc, char *argv[])
{
    if (argc != 3) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    // initialize our pass, it is not yet known to LLVM
    llvm::initializeDeleteUnusedPass(*llvm::PassRegistry::getPassRegistry());

    char const *infilename  = argv[1];
    char const *outfilename = argv[2];

    llvm::LLVMContext llvm_context;
    std::unique_ptr<llvm::Module> libdevice(load_libdevice(infilename, llvm_context));

    if (!libdevice) {
        fprintf(stderr, "Could not load '%s'.\n", infilename);
        return EXIT_FAILURE;
    }

    process(libdevice.get());

    write_libdevice(libdevice.get(), outfilename);

    return EXIT_SUCCESS;
}
