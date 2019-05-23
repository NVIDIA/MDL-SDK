/***************************************************************************************************
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef IO_SCENE_MDL_ELEMENTS_I_MDL_ELEMENTS_MODULE_BUILDER_H
#define IO_SCENE_MDL_ELEMENTS_I_MDL_ELEMENTS_MODULE_BUILDER_H

#include <mi/mdl/mdl.h>
#include <mi/base/handle.h>
#include <mdl/compiler/compilercore/compilercore_allocator.h>

namespace mi
{
    namespace mdl
    {
        class IAnnotation_factory;
        class IDeclaration_factory;
        class IExpression_factory;
        class IMDL;
        class IModule;
        class IName_factory;
        class IThread_context;
        class IType_factory;
        class IValue_factory;
    }
}

namespace MI
{
    namespace DB
    { 
        class Transaction;
        class Tag;
    }

    namespace MDL
    {
        class IAnnotation;
        class IAnnotation_block;
        class Execution_context;
        class IExpression;
        class IExpression_list;
        class IType_list;
        class Mdl_data;
        class Parameter_data;
        class Symbol_importer;
        class Variant_data;
    }
}

namespace MI {

namespace MDL {

class Mdl_module_builder
{

public:

    /// Default Constructor.
    /// Creates a builder that allows to compose a module from different existing elements.
    /// This can fail due to an invalid module name for instance. Therefore, check the context.
    ///
    /// \param imdl                     MDL compiler interface.
    /// \param transaction              A DB transaction in which the elements to add are visible.
    /// \param module_name              The name the module will have.
    /// \param version                  The MDL version the module will have.
    /// \param allow_compatible_types   True, if arguments can be of different but compatible
    ///                                 types w.r.t the parameter types.
    /// \param allow_compatible_types   True, if MDLE imports need to be inlined into the
    ///                                 target module.
    /// \param[inout] context           Execution context used to pass options to and store
    ///                                 messages.
    explicit Mdl_module_builder(
        mi::mdl::IMDL* imdl,
        MI::DB::Transaction* transaction,
        const char* module_name,
        mi::mdl::IMDL::MDL_version version,
        bool allow_compatible_types,
        bool inline_mdle,
        MI::MDL::Execution_context* context);

    virtual ~Mdl_module_builder();

    /// Adds a prototype (material or function definition) to the target module.
    ///
    /// \param prototype_tag    The tag of the prototype to add.
    /// \param name             If NULL, the original name of the prototype.
    /// \param defaults         Default values to set. If NULL, the defaults of the original
    ///                         function or material are used. Pass an empty list to remove all
    ///                         parameters, which is valid if the original material has defaults
    ///                         for all parameters.
    /// \param annotations      Annotations to set. If NULL, the annotations of the original
    ///                         function or material are used. Pass an empty block to remove all
    ///                         annotations.
    /// \param is_exported      If true, the added function/material will have the exported keyword.
    /// \param[inout] context   Execution context used to pass options to and store messages.
    /// \return                 Index of the added material, function or -1 in case of failure.
    mi::Sint32 add_prototype(
        const MI::DB::Tag prototype_tag,
        const char* name,
        const MI::MDL::IExpression_list* defaults,
        const MI::MDL::IAnnotation_block* annotations,
        bool is_exported,
        MI::MDL::Execution_context* context);

    /// Adds a variant of a prototype (material or function definition) to the target module.
    ///
    /// \param prototype_tag    The tag of the prototype to add.
    /// \param name             Name of the variant (unique and different from the prototype).
    /// \param defaults         Default values to set. NULL is note allowed.
    /// \param annotations      Annotations to set. If NULL, the annotations of the original
    ///                         function or material are used. Pass an empty block to remove all
    ///                         annotations.
    /// \param ret_annotations  Return annotations to set. If NULL, the annotations of the original
    ///                         function or material are used. Pass an empty block to remove all
    ///                         annotations.
    /// \param is_exported      If true, the added function/material will have the exported keyword.
    /// \param[inout] context   Execution context used to pass options to and store messages.
    /// \return                 Index of the variant or -1 in case of failure.
    mi::Sint32 add_variant(
        const MI::DB::Tag prototype_tag,
        const char* name,
        const MI::MDL::IExpression_list* defaults,
        const MI::MDL::IAnnotation_block* annotations,
        const MI::MDL::IAnnotation_block* ret_annotations,
        bool is_exported,
        MI::MDL::Execution_context* context);

    /// Adds a variant of a prototype that has already been added to the target module.
    ///
    /// \param index            Index of the added material, function or variant returned by add_*.
    /// \param name             Name of the variant (unique and different from the prototype).
    /// \param defaults         Default values to set. NULL is note allowed.
    /// \param annotations      Annotations to set. If NULL, the annotations of the original
    ///                         function or material are used. Pass an empty block to remove all
    ///                         annotations.
    /// \param ret_annotations  Return annotations to set. If NULL, the annotations of the original
    ///                         function or material are used. Pass an empty block to remove all
    ///                         annotations.
    /// \param is_exported      If true, the added function/material will have the exported keyword.
    /// \param[inout] context   Execution context used to pass options to and store messages.
    /// \return                 Index of the variant or -1 in case of failure.
    mi::Sint32 add_variant(
        const mi::Sint32 index,
        const char* name,
        const MI::MDL::IExpression_list* defaults,
        const MI::MDL::IAnnotation_block* annotations,
        const MI::MDL::IAnnotation_block* ret_annotations,
        bool is_exported,
        MI::MDL::Execution_context* context);

    /// Adds a variant (of a material or function definition) to the target module.
    ///
    /// \param variant_data     Information and data to describe the variant to add,
    ///                         including name, prototype tag, arguments and annotations.
    ///                         For details see Variant_data.
    ///
    /// \param is_exported      If true, the added function/material will have the exported keyword.
    /// \param[inout] context   Execution context used to pass options to and store messages.
    /// \return                 Index of the added variant or -1 in case of failure.
    mi::Sint32 add_variant(
        const MI::MDL::Variant_data* variant_data,
        bool is_exported,
        MI::MDL::Execution_context* context);

    /// Adds a prototype-based function to the target module.
    ///
    /// \param mdl_data         Information and data to describe the function to add,
    ///                         including name, prototype tag, arguments and annotations.
    ///                         For details see Material_data.
    ///
    /// \param is_exported      If true, the added function will have the exported keyword.
    /// \param[inout] context   Execution context used to pass options to and store messages.
    /// \return                 Index of the added variant or -1 in case of failure.
    mi::Sint32 add_function(
        const Mdl_data* mdl_data,
        bool is_exported,
        MI::MDL::Execution_context* context);

    /// Adds a prototype-based material to the target module.
    ///
    /// \param mdl_data         Information and data to describe the material to add,
    ///                         including name, prototype tag, arguments and annotations.
    ///                         For details see Material_data.
    ///
    /// \param is_exported      If true, the added material will have the exported keyword.
    /// \param[inout] context   Execution context used to pass options to and store messages.
    /// \return                 Index of the added variant or -1 in case of failure.
    mi::Sint32 add_material(
        const Mdl_data* mdl_data,
        bool is_exported,
        MI::MDL::Execution_context* context);

    /// Replace all current annotations of the material/function, by the provided block.
    ///
    /// \param index            Index of the added material, function or variant returned by add_*.
    /// \param annotations      Annotations to add.
    /// \param[inout] context   Execution context used to pass options to and store messages.
    /// \return                 True in case of access. If false, see the context for errors.
    bool set_annotations(
        mi::Sint32 index,
        const MI::MDL::IAnnotation_block* annotations,
        MI::MDL::Execution_context* context);

    /// Add an annotations to the added material, function or variant.xc
    ///
    /// \param index            Index of the added material, function or variant returned by add_*.
    /// \param annotation       The Annotation to add.
    /// \param[inout] context   Execution context used to pass options to and store messages.
    /// \return                 True in case of access. If false, see the context for errors.
    bool add_annotation(
        mi::Sint32 index,
        const MI::MDL::IAnnotation* annotation,
        MI::MDL::Execution_context* context);

    /// Set/change the return value annotations of an added material, function or variant.
    ///
    /// \param index            Index of the added material, function or variant returned by add_*.
    /// \param annotations      Annotations to add.
    /// \param[inout] context   Execution context used to pass options to and store messages.
    /// \return                 True in case of access. If false, see the context for errors.
    bool set_return_annotations(
        mi::Sint32 index,
        const MI::MDL::IAnnotation_block* annotations,
        MI::MDL::Execution_context* context);

    /// Completes the module building process and returns the resulting module or NULL in case of
    /// failure, e.g. because module validation failed. 
    ///
    /// \param[inout] context   Execution context used to pass options to and store messages.
    /// \return                 The built module in case of success.
    ///                         If NULL, see the context for errors.
    mi::mdl::IModule const* build(
        MI::MDL::Execution_context* context);

    /// helper functions
    mi::mdl::IAnnotation_factory& get_annotation_factory() { return *m_annotation_factory; }
    mi::mdl::IExpression_factory& get_expression_factory() { return *m_expression_factory; }
    mi::mdl::IName_factory& get_name_factory() { return *m_name_factory; }
    mi::mdl::IValue_factory& get_value_factory() { return *m_value_factory; }

private:

        /// Helper class to express on (new) argument.
        class New_parameter {
        public:

            /// Constructor.
            New_parameter(
                const mi::mdl::ISymbol* sym,
                const mi::base::Handle<const MI::MDL::IExpression>& init,
                mi::base::Handle<const MI::MDL::IAnnotation_block> annos,
                bool is_uniform);

            /// Get the symbol.
            const mi::mdl::ISymbol* get_sym() const { return m_sym; }

            /// Get the init expression.
            const mi::base::Handle<const MI::MDL::IExpression>& get_init() const { return m_init; }

            /// Get the annotations.
            const mi::base::Handle<const MI::MDL::IAnnotation_block>& get_annos() const { return m_annos; }

            /// Get the uniform flag.
            bool is_uniform() const { return m_is_uniform; }

        private:
            /// The symbol of net new parameter.
            const mi::mdl::ISymbol* m_sym;

            /// The (init) expression of the new parameter.
            mi::base::Handle<const MI::MDL::IExpression> m_init;

            /// Annotations for this new parameter.
            mi::base::Handle<const MI::MDL::IAnnotation_block> m_annos;

            /// True if the new parameter must be uniform.
            bool m_is_uniform;
        };
private:

    bool clear_and_check_valid(MI::MDL::Execution_context* context);

    template <typename T>
    mi::Sint32 add_intern(
        MI::DB::Tag prototype_tag,
        const char* name,
        const MI::MDL::IExpression_list* defaults,
        bool is_variant,
        bool is_exported,
        MI::MDL::Execution_context* context);

    mi::Sint32 add_variant_intern(
        mi::Sint32 index,
        const char* name,
        const MI::MDL::IExpression_list* defaults,
        bool is_exported,
        MI::MDL::Execution_context* context);

    template <typename T>
    mi::Sint32 add_function_intern(
        const Mdl_data* mdl_data,
        bool is_exported,
        MI::MDL::Execution_context* context);

    // wraps the function below
    bool create_annotations(
        const MI::MDL::IAnnotation_block* annotation_block,
        mi::mdl::IAnnotation_block* &mdl_annotation_block,
        MI::MDL::Execution_context* context);

    // copied from mdl_elements_nodules and (signature altered only)
    mi::Sint32 create_annotations(
        const MI::MDL::IAnnotation_block* annotation_block,
        mi::mdl::IAnnotation_block* &mdl_annotation_block);

    // copied from mdl_elements_nodules and (signature altered only) 
    mi::Sint32 add_annotation(
        mi::mdl::IAnnotation_block* mdl_annotation_block,
        const char* annotation_name,
        const MI::MDL::IExpression_list* annotation_args);

    // validates and setups parameters
    mi::Sint32 prepare_parameters(
        const Parameter_data* in_parameters,
        mi::Size param_count,
        std::vector<New_parameter>& out_parameters,
        const mi::base::Handle<const IExpression_list>& args,
        const mi::base::Handle<const IType_list>& param_types,
        Execution_context* context) const;

private:
    // mdl interface
    mi::base::Handle<mi::mdl::IMDL> m_mdl;

    // Transaction to use for getting elements to add.
    DB::Transaction* m_transaction;

    // The context the module is created in with.
    mi::base::Handle<mi::mdl::IThread_context> m_thread_context;

    // The underlying MDL module.
    mi::base::Handle<mi::mdl::IModule> m_module;

    Symbol_importer* m_symbol_importer;

    // functions, materials and variants that are about to be added to the material (in build())
    mi::mdl::vector<mi::mdl::IDeclaration_function*>::Type m_added_functions;
    mi::mdl::vector<mi::mdl::IAnnotation_block*>::Type m_added_function_annotations;
    mi::mdl::vector<const mi::mdl::ISymbol*>::Type m_added_function_symbols;

    // factories
    mi::mdl::IAnnotation_factory* m_annotation_factory;
    mi::mdl::IDeclaration_factory* m_declaration_factory;
    mi::mdl::IExpression_factory* m_expression_factory;
    mi::mdl::IName_factory* m_name_factory;
    mi::mdl::IStatement_factory* m_statement_factory;
    mi::mdl::IType_factory* m_type_factory;
    mi::mdl::IValue_factory* m_value_factory;

    bool m_allow_compatible_types;
    bool m_inline_mdle;
};

} // namespace MDL
} // namespace MI

#endif // IO_SCENE_MDL_ELEMENTS_I_MDL_ELEMENTS_MODULE_BUILDER_H
