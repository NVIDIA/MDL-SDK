/***************************************************************************************************
 * Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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
/// \brief

#ifndef RENDER_MDL_BACKENDS_BACKENDS_LINK_UNIT_H
#define RENDER_MDL_BACKENDS_BACKENDS_LINK_UNIT_H

#include <map>
#include <string>
#include <vector>

#include <mi/base/handle.h>
#include <mi/base/interface_implement.h>
#include <mi/mdl/mdl_mdl.h>
#include <mi/mdl/mdl_code_generators.h>
#include <mi/neuraylib/imdl_compiler.h>

namespace MI {

namespace DB { class Transaction; }
namespace MDL { class Mdl_function_call; class Mdl_compiled_material; class IValue_list; };

namespace BACKENDS {

class Mdl_llvm_backend;
class Target_code_register;
class Target_code;

/// Neuray wrapper around an libMDL link unit.
class Link_unit
{
public:
    /// Constructor from an LLVM backend.
    ///
    /// \param llvm_be                  the LLVM backend
    /// \param transaction              the current transaction
    Link_unit(
        Mdl_llvm_backend &llvm_be,
        DB::Transaction  *transaction);


    /// Add an MDL environment function call as a function to this link unit.
    ///
    /// \param i_call                     The MDL function call for the environment.
    /// \param fname                      The name of the function that is created.
    /// \param mdl_meters_per_scene_unit  The conversion ratio between meters and scene units
    ///                                   for this environment function.
    /// \param mdl_wavelength_min         The smallest supported wavelength, typical value: 380.
    /// \param mdl_wavelength_max         The largest supported wavelength, typical value: 780.
    ///
    /// \return           A return code.  The return codes have the following meaning:
    ///                   -  0: Success.
    ///                   - -1: The JIT backend is not available.
    ///                   - -2: Invalid expression.
    ///                   - -3: invalid function name.
    ///                   - -3: The JIT backend failed to compile the function.
    mi::Sint32 add_environment(
        MDL::Mdl_function_call const *i_call,
        char const                   *fname,
        mi::Float32                  mdl_meters_per_scene_unit,
        mi::Float32                  mdl_wavelength_min,
        mi::Float32                  mdl_wavelength_max);

    /// Add an expression that is part of an MDL material instance as a function to this
    /// link unit.
    ///
    /// \param i_material  The compiled MDL material.
    /// \param path        The path from the material root to the expression that should be
    ///                    translated, e.g., \c "geometry.displacement".
    /// \param fname       The name of the function that is created.
    ///
    /// \return            A return code.  The return codes have the following meaning:
    ///                    -  0: Success.
    ///                    - -1: The JIT backend is not available.
    ///                    - -2: Invalid field name (non-existing).
    ///                    - -3: invalid function name.
    ///                    - -4: The JIT backend failed to compile the function.
    ///                    - -5: The requested expression is a constant.
    ///                    - -6: Neither BSDFs, EDFs, VDFs, nor resource type expressions can be
    ///                          compiled.
    mi::Sint32 add_material_expression(
        MDL::Mdl_compiled_material const *i_material,
        char const                       *path,
        char const                       *fname);

    /// Add an MDL distribution function to this link unit.
    /// For a BSDF it results in four functions, suffixed with \c "_init", \c "_sample",
    /// \c "_evaluate" and \c "_pdf".
    ///
    /// \param i_material               The compiled MDL material.
    /// \param path                     The path from the material root to the expression that
    ///                                 should be translated, e.g., \c "surface.scattering".
    /// \param base_fname               The base name of the generated functions.
    ///                                 If \c NULL is passed, \c "lambda" will be used.
    /// \param include_geometry_normal  If true, the \c "geometry.normal" field will be applied
    ///                                 to the MDL state prior to evaluation of the given DF.
    /// \returns             A return code. The error codes have the following meaning:
    ///                      -  0: Success.
    ///                      - -1: Invalid parameters (\c NULL pointer).
    ///                      - -2: Invalid path (non-existing).
    ///                      - -3: The backend failed to generate target code for the material.
    ///                      - -4: The requested expression is a constant.
    ///                      - -5: Only distribution functions are allowed.
    ///                      - -6: The backend does not support compiled MDL materials obtained
    ///                            from class compilation mode.
    ///                      - -7: The backend does not implement this function, yet.
    mi::Sint32 add_material_df(
        MDL::Mdl_compiled_material const *i_material,
        char const                       *path,
        char const                       *base_fname,
        bool                              include_geometry_normal);

    /// Get the number of functions inside this link unit.
    mi::Size get_num_functions() const;

    /// Get the name of the i'th function inside this link unit.
    ///
    /// \param i  the index of the function
    ///
    /// \return the name of the i'th function or NULL if the index is out of range
    char const *get_function_name(mi::Size i) const;

    /// Get the kind of the i'th function inside this link unit.
    ///
    /// \param i  the index of the function
    ///
    /// \return the kind of the function or \c FK_INVALID if \p i was invalid.
    mi::neuraylib::ITarget_code::Function_kind get_function_kind(mi::Size i) const;

    /// Get the index of the target argument block layout for the i'th function inside this link
    /// unit if used.
    ///
    /// \param i  the index of the function
    ///
    /// \return The index of the target argument block layout or ~0 if not used or \p i is invalid.
    mi::Size get_function_arg_block_layout_index(mi::Size i) const;

    /// Get the number of target argument block layouts used by this link unit.
    mi::Size get_arg_block_layout_count() const;

    /// Get the i'th target argument block layout used by this link unit.
    ///
    /// \param i  the index of the target argument block layout
    ///
    /// \return The target argument block layout or \c NULL if \p i is invalid.
    mi::mdl::IGenerated_code_value_layout const *get_arg_block_layout(mi::Size i) const;

    /// Get the MDL link unit.
    mi::mdl::ILink_unit *get_compilation_unit() const;

    /// Get the target code of this link unit.
    Target_code *get_target_code() const;

    /// Get the transaction.
    DB::Transaction *get_transaction() const { return m_transaction; }

    /// Get the registrar for resources of this link unit.
    Target_code_register const *get_tc_reg() const { return m_tc_reg; }

    /// Get the list of arguments required for the target argument blocks.
    MISTD::vector<mi::base::Handle<MDL::IValue_list const> > const &
        get_arg_block_comp_material_args() const { return m_arg_block_comp_material_args; }

    /// Destructor.
    ~Link_unit();

private:
    /// The MDL compiler.
    mi::base::Handle<mi::mdl::IMDL> m_compiler;

    /// The MDL link unit.
    mi::base::Handle<mi::mdl::ILink_unit> m_unit;

    /// The target code of this link unit.
    mi::base::Handle<Target_code> m_target_code;

    /// The used transaction.
    DB::Transaction *m_transaction;

    /// Registrar for resources of this link unit.
    Target_code_register *m_tc_reg;

    typedef MISTD::map<MISTD::string, size_t> Resource_index_map;

    /// The resource index map to keep track of used resources and its indexes.
    Resource_index_map m_res_index_map;

    /// Next texture index.
    size_t m_tex_idx;

    /// Next light profile index.
    size_t m_lp_idx;

    /// Next bsdf measurement index.
    size_t m_bm_idx;

    /// If true, compile pure constants into functions.
    bool m_compile_consts;

    /// If true, string argument values are mapped to string identifiers.
    bool m_strings_mapped_to_ids;

    /// The arguments of the compiled materials for which target argument blocks should be
    /// created.
    MISTD::vector<mi::base::Handle<MDL::IValue_list const> > m_arg_block_comp_material_args;
};

} // namespace BACKENDS
} // namespace MI

#endif // RENDER_MDL_BACKENDS_BACKENDS_LINK_UNIT_H

