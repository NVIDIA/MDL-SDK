/******************************************************************************
 * Copyright (c) 2012-2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDL_GENERATOR_DAG_GENERATED_DAG
#define MDL_GENERATOR_DAG_GENERATED_DAG 1

#include <cstring>

#include <mi/base/handle.h>
#include <mi/mdl/mdl_generated_dag.h>
#include <mi/mdl/mdl_streams.h>
#include <mi/mdl/mdl_printers.h>
#include <mi/mdl/mdl_mdl.h>
#include <mi/mdl/mdl_values.h>
#include <mi/mdl/mdl_modules.h>

#include "mdl/compiler/compilercore/compilercore_cc_conf.h"
#include "mdl/compiler/compilercore/compilercore_allocator.h"
#include "mdl/compiler/compilercore/compilercore_memory_arena.h"
#include "mdl/compiler/compilercore/compilercore_factories.h"
#include "mdl/compiler/compilercore/compilercore_serializer.h"
#include "mdl/compiler/compilercore/compilercore_messages.h"
#include "mdl/compiler/compilercore/compilercore_mangle.h"
#include "mdl/compiler/compilercore/compilercore_mdl.h"
#include "mdl/compiler/compilercore/compilercore_array_ref.h"

#include "generator_dag_ir.h"

namespace MI { namespace MDL {  class Mdl_material_instance_builder; } }

namespace mi {
namespace mdl {

class Indexable;
class ISerializer;
class DAG_builder;
class MDL;
class MDL_binary_serializer;
class DAG_serializer;
class DAG_deserializer;
class Dependence_node;

typedef vector<Resource_tag_tuple>::Type Resource_tag_map;

///
/// Helper class to implement the IResource_tagger interface form a Resource_tag_map.
///
class Resource_tagger : public IResource_tagger {
public:
    /// Get a tag,for a resource constant that might be reachable from this DAG.
    ///
    /// \param res             a resource
    int get_resource_tag(
        IValue_resource const *res) const MDL_FINAL;

public:
    /// Constructor.
    ///
    /// \param resource_map  the resource map
    Resource_tagger(Resource_tag_map const &resource_map)
    : m_resource_tag_map(resource_map)
    {
    }

private:
    Resource_tag_map const &m_resource_tag_map;
};

///
// Implementation of generated code for DAGs.
///
class Generated_code_dag: public Allocator_interface_implement<IGenerated_code_dag>
{
    typedef Allocator_interface_implement<IGenerated_code_dag> Base;
    friend class Allocator_builder;
    friend class Code_generator_dag;
    friend class DAG_dependence_graph;
    friend class Local_type_enumerator;

    /// The type of vector of types.
    typedef vector<IType const *>::Type Type_list;

    /// The type of vector of definitions.
    typedef vector<IDefinition const *>::Type Entity_list;

    /// A Wait queue of definitions.
    typedef mi::mdl::queue<IDefinition const *>::Type Entity_wait_queue;

    /// The type of vectors of definitions.
    typedef vector<IDefinition const *>::Type Definition_vector;

public:
    /// Extra compilation options for creating DAGs from ASTs.
    enum Compile_option {
        /// If set, forbid calls to local functions inside material bodies.
        FORBID_LOCAL_FUNC_CALLS         = 0x0001,
        /// If set, include definitions for local entities called from material bodies.
        INCLUDE_LOCAL_ENTITIES          = 0x0002,
        /// If set, mark DAG backend generated entities.
        MARK_GENERATED_ENTITIES         = 0x0004,
        /// If set, allow unsafe math optimizations.
        UNSAFE_MATH_OPTIMIZATIONS       = 0x0008,
        EXPOSE_NAMES_OF_LET_EXPRESSIONS = 0x0020,
    };

    /// Bit set of compile options.
    typedef unsigned Compile_options;

    /// Error message produced by the DAG backend,
    enum Errors {
        DAG_ERROR_FIRST = 1000,

        FORBIDDEN_CALL_TO_UNEXPORTED_FUNCTION = DAG_ERROR_FIRST,
        DEPENDENCE_GRAPH_HAS_LOOPS,
        VARYING_ON_UNIFORM,
    };

    /// The type of vectors of DAG IR nodes.
    typedef vector<DAG_node const *>::Type Dag_vector;

    /// The type of vectors of parameter indexes.
    typedef vector<size_t>::Type Index_vector;

    /// The type of vectors of ITypes.
    typedef vector<IType const *>::Type Type_vector;

    /// The type of vectors of strings.
    typedef vector<string>::Type String_vector;

    /// Helper class describing one (material) parameter.
    class Parameter_info {
        friend bool has_dynamic_memory_consumption(Parameter_info const &);
        friend size_t dynamic_memory_consumption(Parameter_info const &);
        friend class Generated_code_dag;
    public:
        /// Constructor.
        ///
        /// \param alloc     The allocator.
        /// \param type      The type.
        /// \param name      The parameter name.
        /// \param type_name The type name (as it appear in the signature).
        Parameter_info(
            IAllocator  *alloc,
            IType const *type,
            char const  *name,
            char const  *type_name)
            : m_type(type)
            , m_default(NULL)
            , m_enable_if_cond(NULL)
            , m_name(name, alloc)
            , m_type_name(type_name, alloc)
            , m_annotations(alloc)
            , m_users(alloc)
        {
        }

        /// Set the default value.
        void set_default(DAG_node const *def_value) { m_default = def_value; }

        /// Add an annotation.
        void add_annotation(DAG_node const *anno) { m_annotations.push_back(anno); }

        /// Set the enable_if condition.
        void set_enable_if_condition(DAG_node const *cond) { m_enable_if_cond = cond; }

        /// Get the type.
        IType const *get_type() const { return m_type; }

        /// Get the name.
        char const *get_name() const { return m_name.c_str(); }

        /// Get the type name.
        char const *get_type_name() const { return m_type_name.c_str(); }

        /// Get the annotation count.
        size_t get_annotation_count() const { return m_annotations.size(); }

        /// Get the annotation at index.
        DAG_node const *get_annotation(size_t idx) const { return m_annotations[idx]; }

        /// Get the parameter default value.
        DAG_node const *get_default() const { return m_default; }

        /// Get the enable_if condition of this parameter if any exists.
        DAG_node const *get_enable_if_condition() const { return m_enable_if_cond; }

        /// Get the users of this enable_if condition.
        ///
        /// This function returns a list of parameters indexes that controls the value
        /// of the enable_if condition.
        Index_vector const &get_users() const { return m_users; }

        /// Add a new user.
        void add_user(size_t param_index) { m_users.push_back(param_index); }

    private:
        IType const    *m_type;           ///< The parameter type;
        DAG_node const *m_default;        ///< The default argument for this parameter.
        DAG_node const *m_enable_if_cond; ///< The enable_if condition if any.
        string         m_name;            ///< The name of the parameter.
        string         m_type_name;       ///< The type name of the parameter.
        Dag_vector     m_annotations;     ///< The annotations of this parameter.
        Index_vector   m_users;           ///< Contains the indexes of other parameters whose
                                          ///< enable_if condition depends on this.
    };

    typedef vector<Parameter_info>::Type Param_vector;

    /// Helper class describing one material.
    class Material_info {
        // Helper for dynamic memory consumption: Arena strings have no EXTRA memory allocated.
        friend bool has_dynamic_memory_consumption(Material_info const &);
        friend size_t dynamic_memory_consumption(Material_info const &);
        friend class Generated_code_dag;

    public:
        /// Constructor.
        ///
        /// \param alloc         The allocator.
        /// \param name          The name of the material.
        /// \param simple_name   The simple name of the function.
        /// \param orig_name     The original name of the material if this is an alias, "" else.
        Material_info(
            IAllocator *alloc,
            char const *name,
            char const *simple_name,
            char const *orig_name)
        : m_name(name, alloc)
        , m_simple_name(simple_name, alloc)
        , m_original_name(orig_name, alloc)
        , m_cloned(alloc)
        , m_parameters(alloc)
        , m_annotations(alloc)
        , m_temporaries(alloc)
        , m_temporary_names(alloc)
        , m_body(NULL)
        {
        }

        /// Set the original cloned material name.
        void set_cloned_name(char const *name) { m_cloned = name; }

        /// Add a parameter.
        void add_parameter(Parameter_info const &param) { m_parameters.push_back(param); }

        /// Add an annotation.
        void add_annotation(DAG_node const *anno) { m_annotations.push_back(anno); }

        /// Add a temporary.
        size_t add_temporary(DAG_node const *temp, char const *name) {
            m_temporaries.push_back(temp);
            m_temporary_names.emplace_back(name, m_temporary_names.get_allocator());
            return m_temporaries.size() - 1;
        }

        /// Set the material body.
        void set_body(DAG_node const *body) { m_body = body; }

        /// Get the name.
        char const *get_name() const { return m_name.c_str(); }

        /// Get the simple name.
        char const *get_simple_name() const { return m_simple_name.c_str(); }

        /// Get the original name if any.
        char const *get_original_name() const {
            return m_original_name.empty() ? NULL : m_original_name.c_str();
        }

        /// Get the cloned material name if any.
        char const *get_cloned_name() const {
            return m_cloned.empty() ? NULL : m_cloned.c_str(); }

        /// Get the parameter count.
        size_t get_parameter_count() const { return m_parameters.size(); }

        /// Get the parameter at index.
        Parameter_info const &get_parameter(size_t idx) const { return m_parameters[idx]; }

        /// Get the parameter at index.
        Parameter_info &get_parameter(size_t idx) { return m_parameters[idx]; }

        /// Get the annotation count.
        size_t get_annotation_count() const { return m_annotations.size(); }

        /// Get the annotation at index.
        DAG_node const *get_annotation(size_t idx) const { return m_annotations[idx]; }

        /// Get the temporary count.
        size_t get_temporary_count() const { return m_temporaries.size(); }

        /// Get the temporary at index.
        DAG_node const *get_temporary(size_t idx) const { return m_temporaries[idx]; }

        /// Get the temporary name at index.
        char const *get_temporary_name(size_t idx) const { return m_temporary_names[idx].c_str(); }

        /// Get the material body.
        DAG_node const *get_body() const { return m_body; }

    private:
        string         m_name;            ///< The name of the material.
        string         m_simple_name;     ///< The simple name of the material.
        string         m_original_name;   ///< If this is an alias name, the original name, else "".
        string         m_cloned;          ///< The name of the cloned material or "".
        Param_vector   m_parameters;      ///< The material parameters.
        Dag_vector     m_annotations;     ///< The material annotations.
        Dag_vector     m_temporaries;     ///< The material temporaries.
        String_vector  m_temporary_names; ///< The material temporary names.
        DAG_node const *m_body;           ///< The IR body of the material.
    };

    typedef vector<Material_info>::Type Material_vector;

    /// Helper class describing one function.
    class Function_info {
        // Helper for dynamic memory consumption: Arena strings have no EXTRA memory allocated.
        friend bool has_dynamic_memory_consumption(Function_info const &);
        friend size_t dynamic_memory_consumption(Function_info const &);
        friend class Generated_code_dag;

    public:
        /// Constructor.
        ///
        /// \param alloc       The allocator.
        /// \param sema        The semantics of the function.
        /// \param ret_tp      The return type of the function
        /// \param name        The name of the function.
        /// \param simple_name The simple name of the function.
        /// \param orig_name   The original name of the function if this is an alias,
        ///                    "" else.
        /// \param cloned      The name of the cloned function or "".
        /// \param hash        The function hash if available.
        Function_info(
            IAllocator            *alloc,
            Definition::Semantics sema,
            IType const           *ret_tp,
            char const            *name,
            char const            *simple_name,
            char const            *orig_name,
            char const            *cloned,
            DAG_hash const        *hash)
        : m_semantics(sema)
        , m_return_type(ret_tp)
        , m_name(name, alloc)
        , m_simple_name(simple_name, alloc)
        , m_original_name(orig_name == NULL ? "" : orig_name, alloc)
        , m_cloned(cloned == NULL ? "" : cloned, alloc)
        , m_parameters(alloc)
        , m_annotations(alloc)
        , m_return_annos(alloc)
        , m_temporaries(alloc)
        , m_temporary_names(alloc)
        , m_body(NULL)
        , m_refs(alloc)
        , m_hash()
        , m_properties(0u)
        , m_has_hash(hash != NULL)
        {
            if (m_has_hash) {
                m_hash = *hash;
            }
        }

        /// Add a parameter.
        void add_parameter(Parameter_info const &param) { m_parameters.push_back(param); }

        /// Add an annotation.
        void add_annotation(DAG_node const *anno) { m_annotations.push_back(anno); }

        /// Add a return annotation.
        void add_return_annotation(DAG_node const *anno) { m_return_annos.push_back(anno); }

        /// Add a temporary.
        size_t add_temporary(DAG_node const *temp, char const *name) {
            m_temporaries.push_back(temp);
            m_temporary_names.emplace_back(name, m_temporary_names.get_allocator());
            return m_temporaries.size() - 1;
        }

        /// Set the material body.
        void set_body(DAG_node const *body) { m_body = body; }

        /// Set the function properties.
        void set_properties(unsigned props) { m_properties = props; }

        /// Get the semantics.
        Definition::Semantics get_semantics() const { return m_semantics; }

        /// Get the return type.
        IType const *get_return_type() const{ return m_return_type; }

        /// Get the name.
        char const *get_name() const { return m_name.c_str(); }

        /// Get the simple name.
        char const *get_simple_name() const { return m_simple_name.c_str(); }

        /// Get the original name if any.
        char const *get_original_name() const {
            return m_original_name.empty() ? NULL : m_original_name.c_str();
        }

        /// Get the cloned function name if any.
        char const *get_cloned_name() const {
            return m_cloned.empty() ? NULL : m_cloned.c_str();
        }

        /// Get the parameter count.
        size_t get_parameter_count() const { return m_parameters.size(); }

        /// Get the parameter at index.
        Parameter_info const &get_parameter(size_t idx) const { return m_parameters[idx]; }

        /// Get the parameter at index.
        Parameter_info &get_parameter(size_t idx) { return m_parameters[idx]; }

        /// Get the annotation count.
        size_t get_annotation_count() const { return m_annotations.size(); }

        /// Get the annotation at index.
        DAG_node const *get_annotation(size_t idx) const { return m_annotations[idx]; }

        /// Get the return annotation count.
        size_t get_return_annotation_count() const { return m_return_annos.size(); }

        /// Get the return annotation at index.
        DAG_node const *get_return_annotation(size_t idx) const { return m_return_annos[idx]; }

        /// Get the temporary count.
        size_t get_temporary_count() const { return m_temporaries.size(); }

        /// Get the temporary at index.
        DAG_node const *get_temporary(size_t idx) const { return m_temporaries[idx]; }

        /// Get the temporary name at index.
        char const *get_temporary_name(size_t idx) const { return m_temporary_names[idx].c_str(); }

        /// Get the material body.
        DAG_node const *get_body() const { return m_body; }

        /// Get the references count.
        size_t get_ref_count() const { return m_refs.size(); }

        /// Get the reference at index.
        string const &get_ref(size_t idx) const { return m_refs[idx]; }

        /// Get the function properties.
        unsigned get_properties() const { return m_properties; }

        /// Get the function hash if available.
        DAG_hash const *get_hash() const { return m_has_hash ? &m_hash : NULL; }

    private:
        Definition::Semantics m_semantics;       ///< The function semantics.
        IType const           *m_return_type;    ///< The function return type.
        string                m_name;            ///< The name of the function.
        string                m_simple_name;     ///< The simple name of the function.
        string                m_original_name;   ///< If this is an alias, the original name, else "".
        string                m_cloned;          ///< The name of the cloned function or "".
        Param_vector          m_parameters;      ///< The material parameters.
        Dag_vector            m_annotations;     ///< The annotations of the function.
        Dag_vector            m_return_annos;    ///< The return annotations of the function.
        Dag_vector            m_temporaries;     ///< The function temporaries.
        String_vector         m_temporary_names; ///< The function temporary names.
        DAG_node const        *m_body;           ///< The IR body of the function.
        String_vector         m_refs;            ///< The references of a function.
        DAG_hash              m_hash;            ///< The function hash value.
        unsigned              m_properties;      ///< The property flags of this function.
        bool                  m_has_hash;        ///< True, if a hash value is available.
    };

    typedef vector<Function_info>::Type Function_vector;

    /// Helper class describing one annotation.
    class Annotation_info {
        // Helper for dynamic memory consumption: Arena strings have no EXTRA memory allocated.
        friend bool has_dynamic_memory_consumption(Annotation_info const &);
        friend size_t dynamic_memory_consumption(Annotation_info const &);
        friend class Generated_code_dag;

    public:
        /// Constructor.
        ///
        /// \param alloc      The allocator.
        /// \param sema       The semantics of the annotation.
        /// \param name       The name of the annotation.
        /// \param name       The simple name of the annotation.
        /// \param orig_name  The original name of the annotation if this is an alias, "" else.
        Annotation_info(
            IAllocator            *alloc,
            Definition::Semantics sema,
            char const            *name,
            char const            *simple_name,
            char const            *orig_name)
        : m_semantics(sema)
        , m_name(name, alloc)
        , m_simple_name(simple_name, alloc)
        , m_original_name(orig_name == NULL ? "" : orig_name, alloc)
        , m_parameters(alloc)
        , m_annotations(alloc)
        , m_properties(0u)
        {
        }

        /// Add a parameter.
        void add_parameter(Parameter_info const &param) { m_parameters.push_back(param); }

        /// Add an annotation.
        void add_annotation(DAG_node const *anno) { m_annotations.push_back(anno); }

        /// Set the function properties.
        void set_properties(unsigned props) { m_properties = props; }

        /// Get the semantics.
        Definition::Semantics get_semantics() const { return m_semantics; }

        /// Get the name.
        char const *get_name() const { return m_name.c_str(); }

        /// Get the simple name.
        char const *get_simple_name() const { return m_simple_name.c_str(); }

        /// Get the original name if any.
        char const *get_original_name() const {
            return m_original_name.empty() ? NULL : m_original_name.c_str();
        }

        /// Get the parameter count.
        size_t get_parameter_count() const { return m_parameters.size(); }

        /// Get the parameter at index.
        Parameter_info const &get_parameter(size_t idx) const { return m_parameters[idx]; }

        /// Get the parameter at index.
        Parameter_info &get_parameter(size_t idx) { return m_parameters[idx]; }

        /// Get the annotation count.
        size_t get_annotation_count() const { return m_annotations.size(); }

        /// Get the annotation at index.
        DAG_node const *get_annotation(size_t idx) const { return m_annotations[idx]; }

        /// Get the function properties.
        unsigned get_properties() const { return m_properties; }

    private:
        Definition::Semantics m_semantics;     ///< The function semantics.
        string                m_name;          ///< The name of the annotation.
        string                m_simple_name;   ///< The simple name of the annotation.
        string                m_original_name; ///< If this is an alias, the original name, else "".
        Param_vector          m_parameters;    ///< The annotation parameters.
        Dag_vector            m_annotations;   ///< The annotations of the annotation.
        unsigned              m_properties;    ///< The property flags of this annotation.
    };

    typedef vector<Annotation_info>::Type Annotation_vector;

    //// Helper class describing a user defined type.
    class User_type_info {
        // Helper for dynamic memory consumption: Arena strings have no EXTRA memory allocated.
        friend bool has_dynamic_memory_consumption(User_type_info const &);
        friend size_t dynamic_memory_consumption(User_type_info const &);
        friend class Generated_code_dag;

    public:
        class Entity_info {
            friend bool has_dynamic_memory_consumption(Entity_info const &);
            friend size_t dynamic_memory_consumption(Entity_info const &);
            friend class Generated_code_dag;

        public:
            /// Constructor.
            Entity_info(IAllocator *alloc)
            : m_annotations(alloc)
            {
            }

            /// Add an annotation.
            void add_annotation(DAG_node const *anno) { m_annotations.push_back(anno); }

            /// Get the annotation count.
            size_t get_annotation_count() const { return m_annotations.size(); }

            /// Get the annotation at index.
            DAG_node const *get_annotation(size_t idx) const { return m_annotations[idx]; }

        private:
            /// The annotations of this entity.
            Dag_vector m_annotations;
        };

        typedef vector<Entity_info>::Type Entity_vector;

    public:
        /// Constructor.
        ///
        /// \param alloc          The allocator.
        /// \param is_exported    True, if this user type is exported.
        /// \param type           The user defined type.
        /// \param name           The fully qualified name of the type.
        /// \param original_name  If the type was re-exported, the original name of the type.
        User_type_info(
            IAllocator  *alloc,
            bool        is_exported,
            IType const *type,
            char const  *name,
            char const  *original_name)
            : m_type(type)
            , m_name(name, alloc)
            , m_original_name(original_name != NULL ? original_name : "", alloc)
            , m_annotations(alloc)
            , m_entities(alloc)
            , m_is_exported(is_exported)
        {
        }

        /// Add an annotation.
        void add_annotation(DAG_node const *anno) { m_annotations.push_back(anno); }

        /// Add a sub-entity.
        void add_entity(Entity_info const &ent) { m_entities.push_back(ent); }

        /// Get the type of the user defined type.
        IType const *get_type() const { return m_type; }

        /// Get the name.
        char const *get_name() const { return m_name.c_str(); }

        /// Get the original name if any.
        char const *get_original_name() const {
            return m_original_name.empty() ? NULL : m_original_name.c_str();
        }

        /// Get the annotation count.
        size_t get_annotation_count() const { return m_annotations.size(); }

        /// Get the annotation at index.
        DAG_node const *get_annotation(size_t idx) const { return m_annotations[idx]; }

        /// The (sub-)entity count of the type.
        size_t get_entity_count() const { return m_entities.size(); }

        /// Get a (sub-)entity of the type.
        Entity_info const &get_entity(size_t idx) const { return m_entities[idx]; }

        /// Returns true if this is an exported type.
        bool is_exported() const { return m_is_exported; }

    private:
        /// The user defined type.
        IType const * const m_type;

        /// The name of the type.
        string const m_name;

        /// The original name (if re-exported).
        string const m_original_name;

        /// The annotations of this type.
        Dag_vector m_annotations;

        /// The sub-entities of the type.
        Entity_vector m_entities;

        /// True, if this user type is exported.
        bool m_is_exported;
    };

    typedef vector<User_type_info>::Type User_type_vector;

    //// Helper class describing a user defined constant.
    class Constant_info {
        // Helper for dynamic memory consumption: Arena strings have no EXTRA memory allocated.
        friend bool has_dynamic_memory_consumption(Constant_info const &);
        friend size_t dynamic_memory_consumption(Constant_info const &);
        friend class Generated_code_dag;

    public:
        /// Constructor.
        ///
        /// \param alloc          The allocator.
        /// \param c              The value of the user defined constant.
        /// \param name           The fully qualified name of the constant.
        Constant_info(
            IAllocator         *alloc,
            DAG_constant const *c,
            char const         *name)
        : m_const(c)
        , m_name(name, alloc)
        , m_annotations(alloc)
        {
        }

        /// Add an annotation.
        void add_annotation(DAG_node const *anno) { m_annotations.push_back(anno); }

        /// Get the value of the user defined constant.
        DAG_constant const *get_value() const { return m_const; }

        /// Get the name.
        char const *get_name() const { return m_name.c_str(); }

        /// Get the annotation count.
        size_t get_annotation_count() const { return m_annotations.size(); }

        /// Get the annotation at index.
        DAG_node const *get_annotation(size_t idx) const { return m_annotations[idx]; }

    private:
        /// The value of the user defined constant.
        DAG_constant const * const m_const;

        /// The name of the constant.
        string const m_name;

        /// The annotations of this constant.
        Dag_vector m_annotations;
    };

    typedef vector<Constant_info>::Type Constant_vector;

    /// The type of vectors of values.
    typedef vector<IValue const *>::Type Value_vector;

    /// The type of vectors of names.
    typedef vector<string>::Type Name_vector;

    typedef hash_set<string, string_hash<string> >::Type String_set;

    /// A material instance
    class Material_instance : public Allocator_interface_implement<IMaterial_instance>
    {
        typedef Allocator_interface_implement<IMaterial_instance> Base;
        friend class Allocator_builder;
        friend class Rule_engine;
        friend class Instance_cloner;
        friend class MI::MDL::Mdl_material_instance_builder;

    public:
        /// The result of the dependence analysis.
        struct Dependence_result {
            /// Constructor.
            Dependence_result(
                bool        depends_on_transform,
                bool        depends_on_object_id,
                bool        edf_global_distribution,
                bool        depends_on_uniform_scene_data,
                String_set  referenced_scene_data)
            : m_depends_on_transform(depends_on_transform)
            , m_depends_on_object_id(depends_on_object_id)
            , m_edf_global_distribution(edf_global_distribution)
            , m_depends_on_uniform_scene_data(depends_on_uniform_scene_data)
            , m_referenced_scene_data(referenced_scene_data)
            {
            }

            /// True if this instance depends on the object transforms.
            bool m_depends_on_transform;

            /// True if this instance depends of the object id.
            bool m_depends_on_object_id;

            /// True, if this instance depends on global distribution (edf).
            bool m_edf_global_distribution;

            /// True, if this instance depends on uniform scene data.
            bool m_depends_on_uniform_scene_data;

            /// Set of scene data names referenced by this instance.
            String_set m_referenced_scene_data;
        };

        typedef ptr_hash_map<IDefinition const, Dependence_result>::Type Dep_analysis_cache;

    public:
        // Acquires a const interface.
        mi::base::IInterface const *get_interface(
            mi::base::Uuid const &interface_id) const MDL_FINAL;

        /// Get the type factory of this instance.
        IType_factory *get_type_factory() MDL_FINAL;

        /// Get the value factory of this instance.
        IValue_factory *get_value_factory() MDL_FINAL;

        /// Create a constant.
        /// \param value   The value of the constant.
        /// \returns       The created constant.
        DAG_constant const *create_constant(IValue const *value) MDL_FINAL;

        /// Create a call.
        /// \param name            The absolute name of the called function.
        /// \param sema            The semantic of the called function.
        /// \param call_args       The call arguments of the called function.
        /// \param num_call_args   The number of call arguments.
        /// \param ret_type        The return type of the called function.
        /// \returns               The created call.
        DAG_node const *create_call(
            char const                    *name,
            IDefinition::Semantics        sema,
            DAG_call::Call_argument const call_args[],
            int                           num_call_args,
            IType const                   *ret_type) MDL_FINAL;

        /// Create a parameter reference.
        /// \param  type        The type of the parameter
        /// \param  index       The index of the parameter.
        /// \returns            The created parameter reference.
        DAG_parameter const *create_parameter(
            IType const *type,
            int         index) MDL_FINAL;

        /// Initialize this material instance.
        ///
        /// \param resolver                   The call name resolver.
        /// \param resource_modifier          The resource modifier or NULL.
        /// \param code_dag                   The generated code DAG.
        /// \param argc                       The number of arguments.
        /// \param argv                       An array of pointers to the argument values.
        /// \param use_temporaries            If true, hide multiple used subexpressions behind
        ///                                   temporaries, if false, generate a true DAG.
        /// \param flags                      Instantiation flags.
        /// \param evaluator                  If non-NULL, use this evaluator additionally to fold
        ///                                   intrinsic functions first.
        /// \param fold_meters_per_scene_unit
        ///                                   If true, occurrences of the functions
        ///                                   state::meters_per_scene_unit() and
        ///                                   state::scene_units_per_meter() will be folded
        ///                                   using the \c mdl_meters_per_scene_unit parameter.
        /// \param mdl_meters_per_scene_unit  The value for the meter/scene unit conversion
        ///                                   only used when folding is enabled.
        /// \param wavelength_min             The value for the state::wavelength_min() function.
        /// \param wavelength_max             The value for the state::wavelength_max() function.
        /// \param fold_params                Names of parameters to be folded in class-compilation
        ///                                   mode (in addition to flags).
        /// \param num_fold_params            The number of parameter names to be folded.
        ///
        /// \returns                The error code of the initialization.
        /// Arguments are always given by position.
        /// If a NULL argument is given an EC_INSTANTIATION_ERROR is returned in error_code.
        Error_code initialize(
            ICall_name_resolver       *resolver,
            IResource_modifier        *resource_modifier,
            IGenerated_code_dag const *code_dag,
            size_t                    argc,
            DAG_node const            *argv[],
            bool                      use_temporaries,
            unsigned                  flags,
            ICall_evaluator           *evaluator,
            bool                      fold_meters_per_scene_unit,
            float                     mdl_meters_per_scene_unit,
            float                     wavelength_min,
            float                     wavelength_max,
            char const * const        fold_params[],
            size_t                    num_fold_params) MDL_FINAL;

        /// Return the material constructor.
        DAG_call const *get_constructor() const MDL_FINAL;

        /// Return the number of temporaries.
        size_t get_temporary_count() const MDL_FINAL;

        /// Return the value of the temporary at index.
        ///
        /// \param index  the index of the temporary
        DAG_node const *get_temporary_value(size_t index) const MDL_FINAL;

        /// Return the number of parameters of this instance.
        ///
        /// \note: Returns always 0 in instance compilation mode.
        size_t get_parameter_count() const MDL_FINAL;

        /// Return the default value of a parameter of this instance.
        ///
        /// \param index  the index of the parameter
        IValue const *get_parameter_default(size_t index) const MDL_FINAL;

        /// Return the hash value of this material instance.
        DAG_hash const *get_hash() const MDL_FINAL;

        /// Return the hash value of one material slot of this material instance.
        ///
        /// \param slot  the material slot
        DAG_hash const *get_slot_hash(Slot slot) const MDL_FINAL;

        /// Return the canonical parameter name of the given parameter.
        ///
        /// \param index  the index of the parameter
        char const *get_parameter_name(size_t index) const MDL_FINAL;

        /// Returns true if this instance depends on object transforms.
        bool depends_on_transform() const MDL_FINAL;

        /// Returns true if this instance depends on the object id.
        bool depends_on_object_id() const MDL_FINAL;

        /// Returns true if this instance depends on the global distribution (edf).
        bool depends_on_global_distribution() const MDL_FINAL;

        /// Returns true if this instance depends on uniform scene data.
        bool depends_on_uniform_scene_data() const MDL_FINAL;

        /// Returns the number of scene data attributes referenced by this instance.
        size_t get_referenced_scene_data_count() const MDL_FINAL;

        /// Return the name of a scene data attribute referenced by this instance.
        ///
        /// \param index  the index of the scene data attribute
        char const *get_referenced_scene_data_name(size_t index) const MDL_FINAL;

        /// Returns the opacity of this instance.
        Opacity get_opacity() const MDL_FINAL;

        /// Returns the surface opacity of this instance.
        Opacity get_surface_opacity() const MDL_FINAL;

        /// Returns the cutout opacity of this instance if it is constant.
        ///
        /// \return if the cutout opacity is a constant (and was read),
        ///         NULL if it depends on parameters / complex user expressions
        IValue_float const *get_cutout_opacity() const MDL_FINAL;

        /// Access messages.
        Messages const &access_messages() const MDL_FINAL;

        /// Get the instance properties.
        Properties get_properties() const MDL_FINAL;

        /// Get the internal space.
        char const *get_internal_space() const MDL_FINAL;

        /// Set a tag, version pair for a resource constant that might be reachable from this
        /// instance.
        ///
        /// \param res             a resource
        /// \param tag             the tag value
        void set_resource_tag(
            IValue_resource const *res,
            int                   tag) MDL_FINAL;

        /// Get the number of resource tag map entries.
        size_t get_resource_tag_map_entries_count() const MDL_FINAL;

        /// Get the i'th resource tag map entry or NULL if the index is out of bounds;
        ///
        /// \param index  the index of the resource map entry.
        Resource_tag_tuple const *get_resource_tag_map_entry(size_t index) const MDL_FINAL;

        /// Get the resource tagger for this code DAG.
        IResource_tagger *get_resource_tagger() const MDL_FINAL;

        // ------------------- non-interface methods -------------------

        /// Get the node factory of this instance.
        DAG_node_factory_impl *get_node_factory() { return &m_node_factory; }

        /// Create a temporary constant for the IR visitor.
        ///
        /// \param value   The value of the constant.
        /// \returns       The created constant.
        DAG_constant const *create_temp_constant(IValue const *value);

        /// Get the MDL compiler.
        IMDL *get_mdl() const { m_mdl->retain(); return m_mdl.get(); }

        /// Get the node factory.
        DAG_node_factory_impl const &get_node_factory() const { return m_node_factory; }

        /// Get the material index of this instance.
        int get_material_index() const { return m_material_index; }

        /// Get the type factory of this instance.
        Type_factory const &get_type_factory() const { return m_type_factory; }

        /// Get the value factory of this instance.
        Value_factory const &get_value_factory() const { return m_value_factory; }

        /// Find the tag for a given resource.
        ///
        /// \param res  the resource
        ///
        /// \return the associated tag or zero if no tag was associated
        int find_resource_tag(IValue_resource const *res) const;

        /// Adds a tag, version pair for a given resource.
        ///
        /// \param res  the resource
        ///
        /// \return the associated tag or zero if no tag was associated
        void add_resource_tag(
            IValue_resource const *res,
            int                   tag);

        /// Possible options for the cloning.
        enum Clone_flag {
            CF_DEFAULT     = 0x00,  ///< default flags
            CF_ENABLE_OPT  = 0x01,  ///< enable optimizations during clone
            CF_RECALC_HASH = 0x02,  ///< recalculate the hash values instead of copy
        };

        typedef unsigned Clone_flags;

        /// Creates a clone of a this material instance.
        ///
        /// \param alloc            the allocator for the clone
        /// \param flags            set of Clone_flags
        /// \param unsafe_math_opt  enable unsafe math optimizations
        Material_instance *clone(
            IAllocator  *alloc,
            Clone_flags flags,
            bool        unsafe_math_opt) const;

        /// Dump the material instance DAG to "<name>_DAG.gv".
        ///
        /// \param name  the name of this instance
        void dump_instance_dag(char const *name) const;

    private:
        /// Constructor.
        ///
        /// \param mdl                        The MDL compiler interface.
        /// \param alloc                      The allocator.
        /// \param material_index             The index of the material that creates this instance.
        /// \param internal_space             The internal space for which to compile.
        /// \param unsafe_math_optimizations  If true, allow unsafe math optimizations.
        Material_instance(
            IMDL        *mdl,
            IAllocator  *alloc,
            size_t      material_index,
            char const  *internal_space,
            bool        unsafe_math_optimizations);

    private:
        /// Set the material constructor.
        void set_constructor(DAG_call const *constructor)
        {
            m_constructor = constructor;
        }

        /// Add a temporary.
        ///
        /// \param value       The temporary value to add.
        /// \returns           The index of the value in m_temporaries.
        ///
        int add_temporary(DAG_node const *value);

        /// Build temporaries by traversing the DAG and creating them for nodes with phen-out > 1.
        void build_temporaries();

        /// Calculate the hash values for this instance.
        void calc_hashes();

        /// Check instance argument for restrictions.
        ///
        /// \param arg  The argument to check.
        ///
        /// \return     EC_NONE on success.
        Generated_code_dag::Error_code check_argument(
            DAG_node const *arg) const;

        /// Compute the type of a expression taking uniform rules into account.
        IType const *compute_type(
            ICall_name_resolver &resolver,
            DAG_node const      *arg);

        /// Check that thin walled materials have the same transmission on both sides.
        bool check_thin_walled_material();

        /// Access messages.
        Messages &access_messages() { return m_messages; }

        /// Set a property.
        ///
        /// \param p       the property
        /// \param enable  if true, set the property, else remove it
        void set_property(Property p, bool enable)
        {
            if (enable)
                m_properties |= p;
            else
                m_properties &= ~p;
        }

        /// Creates a new error message.
        ///
        /// \param code  the error code
        /// \param loc   the error location
        /// \param msg   the error message
        void error(int code, Err_location const &loc, char const *msg);

        /// Creates a new warning message.
        ///
        /// \param code  the error code
        /// \param loc   the error location
        /// \param msg   the error message
        void warning(int code, Err_location const &loc, char const *msg);

        friend class Transparent_layers;
        
    private:
        friend class Instantiate_helper;

        /// Helper class for DAG node instantiation of an instance.
        class Instantiate_helper {
            typedef Arena_ptr_hash_map<
                DAG_node const,
                DAG_node const *>::Type Visit_map;

            typedef Arena_ptr_hash_map<
                DAG_node const,
                DAG_node const *>::Type Replacement_map;

            typedef Arena_ptr_hash_map<
                IValue_resource const,
                DAG_node const *>::Type Resource_param_map;

            /// RAII-like parameter scope.
            class Param_scope {
            public:
                Param_scope(Instantiate_helper &helper, char const *name, int index)
                : m_helper(helper)
                , m_old_name(helper.m_curr_param_name)
                {
                    helper.enter_param(name, index);
                    if (!helper.m_curr_param_name.empty())
                        helper.m_curr_param_name += '.';
                    helper.m_curr_param_name += name;
                }

                ~Param_scope() {
                    m_helper.leave_param();
                    m_helper.m_curr_param_name = m_old_name;
                }

            private:
                Instantiate_helper &m_helper;
                string             m_old_name;
            };

        public:
            friend class Transparent_layers;

            typedef vector<IValue const *>::Type    Value_vec;
            typedef vector<string>::Type            Name_vec;

            typedef IGenerated_code_dag::IMaterial_instance::Property   Property;
            typedef IGenerated_code_dag::IMaterial_instance::Properties Properties;

        public:
            /// Constructor.
            ///
            /// \param resolver                   The call name resolver.
            /// \param resource_modifier          The resource modifier.
            /// \param code_dag                   The generated code DAG.
            /// \param dag_builder                The DAG builder used to create nodes.
            /// \param instance                   The index of the instantiated instance.
            /// \param flags                      Instantiation flags.
            /// \param evaluator                  If non-NULL, use this evaluator additionally
            ///                                   to fold intrinsic functions first.
            /// \param argc                       The number of arguments of the instance.
            /// \param argv                       An array of pointers to the argument values.
            /// \param fold_meters_per_scene_unit
            ///                                   If true, occurrences of the functions
            ///                                   state::meters_per_scene_unit() and
            ///                                   state::scene_units_per_meter() will be folded
            ///                                   using the \c mdl_meters_per_scene_unit parameter.
            /// \param mdl_meters_per_scene_unit  The value for the meter/scene unit conversion
            ///                                   only used when folding is enabled.
            /// \param wavelength_min             The value for state::wavelength_min().
            /// \param wavelength_max             The value for state::wavelength_max().
            /// \param fold_params                Names of parameters to be folded in
            ///                                   class-compilation mode (in addition to flags).
            /// \param num_fold_params            The number of parameter names to be folded.
            Instantiate_helper(
                ICall_name_resolver      &resolver,
                IResource_modifier       &resource_modifier,
                Generated_code_dag const *code_dag,
                DAG_builder              &dag_builder,
                int                      material_index,
                unsigned                 flags,
                ICall_evaluator          *evaluator,
                size_t                   argc,
                DAG_node const           *argv[],
                bool                     fold_meters_per_scene_unit,
                float                    mdl_meters_per_scene_unit,
                float                    wavelength_min,
                float                    wavelength_max,
                char const * const       fold_params[],
                size_t                   num_fold_params);

            /// Destructor.
            ~Instantiate_helper();

            /// Compile the material.
            ///
            /// \returns            The instantiated DAG IR.
            DAG_call const *compile();

            /// Get the default values of created parameters.
            Value_vec const &get_default_parameter_values() const {
                return m_default_param_values;
            }

            /// Get the names of the created parameters.
            Name_vec const &get_parameter_names() const {
                return m_param_names;
            }

            /// Get the instance properties.
            Properties get_properties() const { return m_properties; }

            /// Get the set of scene data names referenced by this instance.
            String_set const &get_referenced_scene_data() const { return m_referenced_scene_data; }

        private:
            /// Get the allocator.
            IAllocator *get_allocator() const { return m_arena.get_allocator(); }

            /// Enter a parameter.
            ///
            /// \param name   the parameter name
            /// \param index  the parameter index
            void enter_param(char const *name, int index) {/*empty for now*/}

            /// Leave a parameter.
            void leave_param() {/*empty for now*/}

            /// Check if we support instantiate_dag_arguments on this node.
            bool supported_arguments(DAG_node const *n);

            /// Instantiate a DAG IR node.
            ///
            /// \param node         The (material DAG) root node to instantiate.
            /// \returns            The instantiated DAG IR node.
            ///
            /// Made a deep copy of the DAG IR owned by the material, creating a DAG
            /// owned by the instance WITHOUT temporaries.
            DAG_node const *instantiate_dag(
                DAG_node const *node);

            /// Instantiate a DAG IR node from an argument.
            ///
            /// \param node         The (material DAG) root node to instantiate.
            /// \returns            The instantiated DAG IR node.
            ///
            /// Made a deep copy of the IR owned by the material, creating a DAG
            /// owned by the instance WITHOUT temporaries.
            DAG_node const *instantiate_dag_arguments(
                DAG_node const *node);

            /// Inline parameters into a DAG IR node.
            ///
            /// \param node           The (material DAG) root node to instantiate.
            /// \param inline_params  A sparse set of the parameter that must be inlined.
            /// \returns              The instantiated DAG IR node.
            ///
            /// Made a deep copy of the DAG IR, inline some parameters.
            DAG_node const *inline_parameters(
                DAG_node const *node,
                DAG_parameter  *inline_params[]);

            /// Check that every parameter is still used after optimization and remove
            /// dead ones.
            ///
            /// \param node  The root DAG IR node of the instance DAG.
            DAG_node const *renumber_parameter(DAG_node const *node);

            /// Analyze a created call for dependencies.
            ///
            /// \param call  the call to analyze
            void analyze_call(DAG_call const *call);

            /// Analyze a function AST for dependencies.
            ///
            /// \param owner  the owner module of the function to analyze
            /// \param def    the definition of the function to analyze
            void analyze_function_ast(
                Module const *owner, IDefinition const *def);

            /// Set/reset property.
            void set_property(Property prop, bool enable) {
                if (enable)
                    m_properties |= prop;
                else
                    m_properties &= ~prop;
            }

            /// Skip temporaries.
            ///
            /// \param expr  the DAG node
            ///
            /// \return expr if the node is not a temporary, its argument otherwise
            DAG_node const *skip_temporaries(DAG_node const *expr);

            /// Get a DAG node from a value by an absolute path.
            ///
            /// \param value  a value
            /// \param path   the path
            DAG_node const *get_value(IValue const *value, Array_ref<char const *> const &path);

            /// Get a DAG node from an expression by absolute path.
            ///
            /// \param expr  a DAG expression
            /// \param path  the path
            DAG_node const *get_value(DAG_node const *expr, Array_ref<char const *> const &path);

            /// Fold geometry.cutout_opacity if in class-compilation mode, requested via flags,
            /// and evaluates to 0.0f or 1.0f.
            ///
            /// Folding is done by putting the replacement node in m_visits_map. Similar for
            /// parameters that affect the value of this node.
            void handle_cutout_opacity();

            /// Eliminate transparent layers if in class-compilation mode and requested via flags.
            ///
            /// Elimination is done by putting the replacement node in m_replacement_map. Parameters
            /// that affect the decision are put into m_visits_map.
            void handle_transparent_layers();

            /// Eliminate transparent layers if in class-compilation mode and requested via flags.
            ///
            /// Elimination occurs if
            /// - the call is {weighted,fresnel,custom_curve,measured_curve}_layer() (or one of
            ///   their color weight variants),
            /// - the weight evaluates to constant 0.0f, and
            /// - #is_layer_qualified() returns \c true for the layer argument.
            ///
            /// Elimination is done by putting the replacement node in m_replacement_map. Parameters
            /// that affect the decision are put into m_visits_map.
            void handle_transparent_layers(DAG_call const *call);
            
            /// Return whether the expression qualifies for elimination by
            /// #handle_transparent_layers().
            ///
            /// An expression qualifies if
            /// - the call is diffuse_transmission_bsdf(), specular_bsdf(), simple_glossy_bsdf(), or
            ///   microfacet_*_bsdf(), and
            /// - the scatter_mode argument (if present) is scatter_transmit or
            ///   scatter_reflect_transmit.
            ///
            /// In addition, the ternary operators is qualified if both true and false expression are
            /// qualified.
            bool is_layer_qualified(DAG_node const *expr);

        private:
            /// The call name resolver to be used.
            ICall_name_resolver &m_resolver;

            /// The resource modifier to be used.
            IResource_modifier &m_resource_modifier;

            /// The material's code DAG we are instantiating.
            Generated_code_dag const &m_code_dag;

            /// The memory arena.
            Memory_arena m_arena;

            /// The DAG builder used to create the DAG nodes.
            DAG_builder &m_dag_builder;

            /// The node factory of the instance.
            DAG_node_factory_impl &m_node_factory;

            /// The value factory of the instance.
            Value_factory &m_value_factory;

            /// The type factory of the instance.
            Type_factory &m_type_factory;

            /// The old evaluator of the node factory.
            ICall_evaluator       *m_old_evaluator;

            /// Instantiation flags.
            unsigned m_flags;

            /// The number of instance arguments.
            size_t m_argc;

            /// The instance arguments;
            DAG_node const **m_argv;

            /// The index of the material that is instantiated.
            int m_material_index;

            /// Number of created parameters.
            int m_params;

            /// Map for visiting DAGs (maps input to output for instantiate_dag() and
            /// instantiate_dag_arguments()).
            Visit_map m_visit_map;

            /// Map for visiting DAGs (maps input to input for instantiate_dag() and
            /// instantiate_dag_arguments()). Populated by handle_transparent_layers().
            Replacement_map m_replacement_map;

            /// Map for handling resource parameters.
            Resource_param_map m_resource_param_map;

            /// Default values for the created parameters.
            Value_vec m_default_param_values;

            /// Names of the created parameters.
            Name_vec m_param_names;

            /// The current canonical parameter name.
            string m_curr_param_name;

            /// The analysis cache for dependency analysis.
            Dep_analysis_cache m_cache;

            /// Properties of the generated instance.
            Properties m_properties;

            /// Set of scene data names referenced by this instance.
            String_set m_referenced_scene_data;

            /// If true, instantiate arguments.
            bool m_instantiate_args;

            /// Names of parameters to fold in class-compilation mode.
            String_set m_fold_params;
        };

        /// A builder, used for generating printer.
        mutable Allocator_builder m_builder;

        /// The MDL compiler interface.
        mi::base::Handle<MDL> m_mdl;

        /// The memory arena that holds all types, symbols and IR nodes of this instance.
        Memory_arena m_arena;

        /// The symbol table;
        Symbol_table m_sym_tab;

        /// The type factory.
        Type_factory m_type_factory;

        /// The value factory.
        Value_factory m_value_factory;

        /// The node factory.
        DAG_node_factory_impl m_node_factory;

        /// Instanciation messages;
        Messages_impl m_messages;

        /// The index of the material definition this instance is based on.
        size_t m_material_index;

        /// An IR node of type call representing the call to the material constructor.
        DAG_call const *m_constructor;

        /// The temporary values.
        Dag_vector m_temporaries;

        /// The default parameter values of parameters in class compilation mode.
        Value_vector m_default_param_values;

        /// The canonical names of the parameters in class compilation mode.
        Name_vector m_param_names;

        /// The hash values of this instance.
        DAG_hash m_hash;

        /// The hash values of the material slots.
        DAG_hash m_slot_hashes[MS_LAST + 1];

        /// Instance properties;
        Properties m_properties;

        /// The scene data names referenced by this instance.
        String_vector m_referenced_scene_data;

        /// The resource tag map, mapping accessible resources to tags.
        Resource_tag_map m_resource_tag_map;

        /// The resource tagger, using the resource to tag map;
        mutable Resource_tagger m_resource_tagger;
    };

private:
    /// Constructor.
    ///
    /// \param alloc                  The allocator.
    /// \param compiler               The mdl compiler.
    /// \param module                 The module from which this code was generated.
    /// \param internal_space         The internal space for which to compile.
    /// \param options                The compile options.
    /// \param renderer_context_name  The name of the renderer context.
    explicit Generated_code_dag(
        IAllocator      *alloc,
        MDL             *compiler,
        IModule const   *module,
        char const      *internal_space,
        Compile_options options,
        char const      *renderer_context_name);

public:
    /// Get the kind of code generated.
    /// \returns    The kind of generated code.
    Kind get_kind() const MDL_FINAL;

    /// Get the target language.
    /// \returns    The name of the target language for which this code was generated.
    char const *get_target_language() const MDL_FINAL;

    /// Get the module name from the module from which this code was generated.
    char const *get_module_name() const MDL_FINAL;

    /// Get the module file name from the module from which this code was generated.
    char const *get_module_file_name() const MDL_FINAL;

    /// Get the number of modules directly imported by the module
    /// from which this code was generated.
    size_t get_import_count() const MDL_FINAL;

    /// Get the module at index imported from the module
    /// from which this code was generated.
    char const *get_import(
        size_t index) const MDL_FINAL;

    /// Get the number of functions in the generated code.
    /// \returns    The number of functions in this generated code.
    size_t get_function_count() const MDL_FINAL;

    /// Get the return type of the function at function_index.
    /// \param      function_index  The index of the function.
    /// \returns                    The return type of the function.
    IType const *get_function_return_type(size_t function_index) const MDL_FINAL;

    /// Get the semantics of the function at function_index.
    /// \param      function_index  The index of the function.
    /// \returns                    The semantics of the function.
    IDefinition::Semantics get_function_semantics(size_t function_index) const MDL_FINAL;

    /// Get the name of the function at function_index.
    /// \param      function_index  The index of the function.
    /// \returns                    The name of the function.
    char const *get_function_name(size_t function_index) const MDL_FINAL;

    /// Get the simple name of the function at function_index.
    /// \param      function_index  The index of the function.
    /// \returns                    The simple name of the function.
    char const *get_simple_function_name(size_t function_index) const MDL_FINAL;

    /// Get the original name of the function at function_index if the function name is an alias.
    /// \param      function_index  The index of the function.
    /// \returns                    The original name of the function or NULL.
    char const *get_original_function_name(size_t function_index) const MDL_FINAL;

    /// Get the parameter count of the function at function_index.
    /// \param      function_index  The index of the function.
    /// \returns                    The number of parameters of the function.
    size_t get_function_parameter_count(size_t function_index) const MDL_FINAL;

    /// Get the parameter type of the parameter at parameter_index
    /// of the function at function_index.
    /// \param      function_index  The index of the function.
    /// \param      parameter_index The index of the parameter.
    /// \returns                    The type of the parameter.
    IType const *get_function_parameter_type(
        size_t function_index,
        size_t parameter_index) const MDL_FINAL;

    /// Get the parameter type name of the parameter at parameter_index
    /// of the function at function_index.
    /// \param      function_index  The index of the function.
    /// \param      parameter_index The index of the parameter.
    /// \returns                    The type name of the parameter.
    char const *get_function_parameter_type_name(
        size_t function_index,
        size_t parameter_index) const MDL_FINAL;

    /// Get the parameter name of the parameter at parameter_index
    /// of the function at function_index.
    /// \param      function_index  The index of the function.
    /// \param      parameter_index The index of the parameter.
    /// \returns                    The name of the parameter.
    char const *get_function_parameter_name(
        size_t function_index,
        size_t parameter_index) const MDL_FINAL;

    /// Get the index of the parameter parameter_name.
    /// \param      function_index  The index of the function.
    /// \param      parameter_name  The name of the parameter.
    /// \returns                    The index of the parameter, or -1 if it does not exist.
    size_t get_function_parameter_index(
        size_t     function_index,
        const char *parameter_name) const MDL_FINAL;

    /// Get the enable_if condition for the given function parameter if one was specified.
    ///
    /// \param      function_index  The index of the function.
    /// \param      parameter_index The index of the parameter.
    /// \returns                    The enable_if condition for this parameter or NULL.
    DAG_node const *get_function_parameter_enable_if_condition(
        size_t function_index,
        size_t parameter_index) const MDL_FINAL;

    /// Get the number of parameters whose enable_if condition depends on this parameter.
    ///
    /// \param      function_index  The index of the function.
    /// \param      parameter_index The index of the parameter.
    /// \returns                    Number of depended parameter conditions.
    size_t get_function_parameter_enable_if_condition_users(
        size_t function_index,
        size_t parameter_index) const MDL_FINAL;

    /// Get a parameter index whose enable_if condition depends on this parameter.
    ///
    /// \param      function_index  The index of the function.
    /// \param      parameter_index The index of the parameter.
    /// \param      user_index      The index of the user.
    /// \returns                    The index of the depended parameter.
    size_t get_function_parameter_enable_if_condition_user(
        size_t function_index,
        size_t parameter_index,
        size_t user_index) const MDL_FINAL;

    /// Get the function hash value for the given function index if available.
    ///
    /// \param function_index  The index of the function.
    /// \returns               The function hash of the function or NULL if no hash
    ///                        value is available or the index is out of bounds.
    DAG_hash const *get_function_hash(
        size_t function_index) const MDL_FINAL;

    /// Check if the code contents are valid.
    bool is_valid() const MDL_FINAL;

    /// Access messages.
    Messages const &access_messages() const MDL_FINAL;

    /// Get the number of materials in the generated code.
    ///
    /// \returns    The number of materials in this generated code.
    size_t get_material_count() const MDL_FINAL;

    /// Get the name of the material at material_index.
    ///
    /// \param material_index  The index of the material.
    /// \returns               The name of the material.
    char const *get_material_name(size_t material_index) const MDL_FINAL;

    /// Get the simple name of the material at material_index.
    /// \param      material_index  The index of the material.
    /// \returns                    The simple name of the material.
    char const *get_simple_material_name(size_t material_index) const MDL_FINAL;

    /// Get the original name of the material at material_index if the material name is an alias.
    ///
    /// \param      material_index  The index of the material.
    /// \returns                    The name of the material or NULL.
    char const *get_original_material_name(size_t material_index) const MDL_FINAL;

    /// Get the parameter count of the material at material_index.
    ///
    /// \param material_index  The index of the material.
    /// \returns               The number of parameters of the material.
    size_t get_material_parameter_count(size_t material_index) const MDL_FINAL;

    /// Get the parameter type of the parameter at parameter_index
    /// of the material at material_index.
    ///
    /// \param material_index   The index of the material.
    /// \param parameter_index  The index of the parameter.
    /// \returns                The type of the parameter.
    IType const *get_material_parameter_type(
        size_t material_index,
        size_t parameter_index) const MDL_FINAL;

    /// Get the parameter name of the parameter at parameter_index
    /// of the material at material_index.
    ///
    /// \param material_index   The index of the material.
    /// \param parameter_index  The index of the parameter.
    /// \returns                The name of the parameter.
    char const *get_material_parameter_name(
        size_t material_index,
        size_t parameter_index) const MDL_FINAL;

    /// Get the index of the parameter parameter_name.
    ///
    /// \param material_index  The index of the material.
    /// \param parameter_name  The name of the parameter.
    /// \returns               The index of the parameter, or -1 if it does not exist.
    size_t get_material_parameter_index(
        size_t     material_index,
        char const *parameter_name) const MDL_FINAL;

    /// Get the enable_if condition for the given material parameter if one was specified.
    ///
    /// \param      material_index  The index of the material.
    /// \param      parameter_index The index of the parameter.
    /// \returns                    The enable_if condition for this parameter or NULL.
    DAG_node const *get_material_parameter_enable_if_condition(
        size_t material_index,
        size_t parameter_index) const MDL_FINAL;

    /// Get the number of parameters whose enable_if condition depends on this parameter.
    ///
    /// \param      material_index  The index of the material.
    /// \param      parameter_index The index of the parameter.
    /// \returns                    Number of depended parameter conditions.
    size_t get_material_parameter_enable_if_condition_users(
        size_t material_index,
        size_t parameter_index) const MDL_FINAL;

    /// Get a parameter index whose enable_if condition depends on this parameter.
    ///
    /// \param      material_index  The index of the material.
    /// \param      parameter_index The index of the parameter.
    /// \param      user_index      The index of the user.
    /// \returns                    The index of the depended parameter.
    size_t get_material_parameter_enable_if_condition_user(
        size_t material_index,
        size_t parameter_index,
        size_t user_index) const MDL_FINAL;

    /// Acquires a const interface.
    ///
    /// If this interface is derived from or is the interface with the passed
    /// \p interface_id, then return a non-\c NULL \c const #mi::base::IInterface* that
    /// can be casted via \c static_cast to an interface pointer of the interface type
    /// corresponding to the passed \p interface_id. Otherwise return \c NULL.
    ///
    /// In the case of a non-\c NULL return value, the caller receives ownership of the
    /// new interface pointer, whose reference count has been retained once. The caller
    /// must release the returned interface pointer at the end to prevent a memory leak.
    mi::base::IInterface const *get_interface(
        mi::base::Uuid const &interface_id) const MDL_FINAL;

    /// Get the node IR-node factory of this code DAG.
    DAG_node_factory_impl *get_node_factory() MDL_FINAL;

    /// Get the number of annotations of the function at function_index.
    /// \param function_index      The index of the function.
    /// \returns                   The number of annotations.
    size_t get_function_annotation_count(
        size_t function_index) const MDL_FINAL;

    /// Get the annotation at annotation_index of the function at function_index.
    /// \param function_index      The index of the function.
    /// \param annotation_index    The index of the annotation.
    /// \returns                   The annotation.
    DAG_node const *get_function_annotation(
        size_t function_index,
        size_t annotation_index) const MDL_FINAL;

    /// Get the number of annotations of the function return type at function_index.
    /// \param function_index      The index of the function.
    /// \returns                   The number of annotations.
    size_t get_function_return_annotation_count(
        size_t function_index) const MDL_FINAL;

    /// Get the annotation at annotation_index of the function return type at function_index.
    /// \param function_index      The index of the function.
    /// \param annotation_index    The index of the annotation.
    /// \returns                   The annotation.
    DAG_node const *get_function_return_annotation(
        size_t function_index,
        size_t annotation_index) const MDL_FINAL;

    /// Get the default initializer of the parameter at parameter_index
    /// of the function at function_index.
    /// \param function_index   The index of the function.
    /// \param parameter_index  The index of the parameter.
    /// \returns                The type of the parameter.
    DAG_node const *get_function_parameter_default(
        size_t function_index,
        size_t parameter_index) const MDL_FINAL;

    /// Get the number of annotations of the parameter at parameter_index
    /// of the function at function_index.
    /// \param function_index      The index of the function.
    /// \param parameter_index     The index of the parameter.
    /// \returns                   The number of annotations.
    size_t get_function_parameter_annotation_count(
        size_t function_index,
        size_t parameter_index) const MDL_FINAL;

    /// Get the annotation at annotation_index of the parameter at parameter_index
    /// of the function at function_index.
    /// \param function_index      The index of the function.
    /// \param parameter_index     The index of the parameter.
    /// \param annotation_index    The index of the annotation.
    /// \returns                   The annotation.
    DAG_node const *get_function_parameter_annotation(
        size_t function_index,
        size_t parameter_index,
        size_t annotation_index) const MDL_FINAL;

    /// Get the number of temporaries used by the function at function_index.
    ///
    /// \param function_index      The index of the function.
    /// \returns                   The number of temporaries used by material.
    size_t get_function_temporary_count(
        size_t function_index) const MDL_FINAL;

    /// Get the temporary at temporary_index used by the function at function_index.
    ///
    /// \param function_index      The index of the function.
    /// \param temporary_index     The index of the temporary variable.
    /// \returns                   The value of the temporary variable.
    DAG_node const *get_function_temporary(
        size_t function_index,
        size_t temporary_index) const MDL_FINAL;

    /// Get the temporary name at temporary_index used by the function at function_index.
    ///
    /// \param function_index      The index of the function.
    /// \param temporary_index     The index of the temporary variable.
    /// \returns                   The name of the temporary variable.
    char const *get_function_temporary_name(
        size_t function_index,
        size_t temporary_index) const MDL_FINAL;

    /// Get the body of the function at function_index.
    ///
    /// \param function_index      The index of the function.
    /// \returns                   The body of the function.
    DAG_node const *get_function_body(
        size_t function_index) const MDL_FINAL;

    /// Get the number of annotations of the material at material_index.
    /// \param material_index      The index of the material.
    /// \returns                   The number of annotations.
    size_t get_material_annotation_count(
        size_t material_index) const MDL_FINAL;

    /// Get the annotation at annotation_index of the material at material_index.
    /// \param material_index      The index of the material.
    /// \param annotation_index    The index of the annotation.
    /// \returns                   The annotation.
    DAG_node const *get_material_annotation(
        size_t material_index,
        size_t annotation_index) const MDL_FINAL;

    /// Get the default initializer of the parameter at parameter_index
    /// of the material at material_index.
    /// \param material_index   The index of the material.
    /// \param parameter_index  The index of the parameter.
    /// \returns                The type of the parameter.
    DAG_node const *get_material_parameter_default(
        size_t material_index,
        size_t parameter_index) const MDL_FINAL;

    /// Get the number of annotations of the parameter at parameter_index
    /// of the material at material_index.
    /// \param material_index      The index of the material.
    /// \param parameter_index     The index of the parameter.
    /// \returns                   The number of annotations.
    size_t get_material_parameter_annotation_count(
        size_t material_index,
        size_t parameter_index) const MDL_FINAL;

    /// Get the annotation at annotation_index of the parameter at parameter_index
    /// of the material at material_index.
    /// \param material_index      The index of the material.
    /// \param parameter_index     The index of the parameter.
    /// \param annotation_index    The index of the annotation.
    /// \returns                   The annotation.
    DAG_node const *get_material_parameter_annotation(
        size_t material_index,
        size_t parameter_index,
        size_t annotation_index) const MDL_FINAL;

    /// Get the number of temporaries used by the material at material_index.
    /// \param material_index  The index of the material.
    /// \returns               The number of temporaries used by material.
    size_t get_material_temporary_count(
        size_t material_index) const MDL_FINAL;

    /// Get the temporary at temporary_index used by the material at material_index.
    /// \param material_index   The index of the material.
    /// \param temporary_index  The index of the temporary variable.
    /// \returns                The value of the temporary variable.
    DAG_node const *get_material_temporary(
        size_t material_index,
        size_t temporary_index) const MDL_FINAL;

    /// Get the temporary name at temporary_index used by the material at material_index.
    /// \param material_index   The index of the material.
    /// \param temporary_index  The index of the temporary variable.
    /// \returns                The name of the temporary variable.
    char const *get_material_temporary_name(
        size_t material_index,
        size_t temporary_index) const MDL_FINAL;

    /// Get the value of the material at material_index.
    /// \param material_index  The index of the material.
    /// \returns               The value of the material.
    DAG_node const *get_material_value(
        size_t material_index) const MDL_FINAL;

    /// Get the export flags of the material at material_index.
    ///
    /// \param      material_index  The index of the material.
    /// \returns                    True if this is an exported material, false if it is local.
    bool get_material_exported(size_t material_index) const MDL_FINAL;

    /// Return the original material name of a cloned material or NULL if the material
    /// is not a clone.
    ///
    /// \param material_index   The index of the material.
    /// \returns                The absolute name of the original material or NULL.
    char const *get_cloned_material_name(
        size_t material_index) const MDL_FINAL;

    /// Create a material instance.
    ///
    /// \param index       The index of the material to instantiate.
    /// \param error_code  The error code of the instance creation.
    /// \returns           The material instance.
    ///
    IMaterial_instance *create_material_instance(
        size_t     index,
        Error_code *error_code = 0) const MDL_FINAL;

    /// Get the number of exported user types.
    size_t get_type_count() const MDL_FINAL;

    /// Get the name of the type at index.
    ///
    /// \param      index  The index of the type.
    /// \returns           The name of the type.
    char const *get_type_name(
        size_t index) const MDL_FINAL;

    /// Get the original name of the type at index if the type is an alias.
    ///
    /// \param      index  The index of the type.
    /// \returns           The original name of the type or NULL.
    char const *get_original_type_name(
        size_t index) const MDL_FINAL;

    /// Get the user type at index.
    ///
    /// \param      index  The index of the type.
    /// \returns           The type.
    IType const *get_type(
        size_t index) const MDL_FINAL;

    /// Returns true if the type at index is exported.
    ///
    /// \param index  The index of the type.
    /// \returns      true for exported types.
    bool is_type_exported(
        size_t index) const MDL_FINAL;

    /// Get the number of annotations of the type at index.
    ///
    /// \param      index  The index of the type.
    /// \return            The number of annotations of the type.
    size_t get_type_annotation_count(
        size_t index) const MDL_FINAL;

    /// Get the annotation at annotation_index of the type at type_index.
    ///
    /// \param type_index          The index of the type.
    /// \param annotation_index    The index of the annotation.
    /// \returns                   The annotation.
    DAG_node const *get_type_annotation(
        size_t type_index,
        size_t annotation_index) const MDL_FINAL;

    /// Get the number of type sub-entities (fields or enum constants).
    ///
    /// \param type_index          The index of the type.
    /// \returns                   The number of sub entities.
    size_t get_type_sub_entity_count(
        size_t type_index) const MDL_FINAL;

    /// Get the name of a type sub-entity (field or enum constant).
    ///
    /// \param type_index          The index of the type.
    /// \param entity_index        The index of the sub entity.
    /// \returns                   The name of a sub-entity.
    char const *get_type_sub_entity_name(
        size_t type_index,
        size_t entity_index) const MDL_FINAL;

    /// Get the type of a type sub-entity (field or enum constant).
    ///
    /// \param type_index          The index of the type.
    /// \param entity_index        The index of the sub entity.
    /// \returns                   The type of sub-entity.
    IType const *get_type_sub_entity_type(
        size_t type_index,
        size_t entity_index) const MDL_FINAL;

    /// Get the number of annotations of a type sub-entity (field or enum constant) at index.
    ///
    /// \param type_index          The index of the type.
    /// \param entity_index        The index of the sub entity.
    /// \returns                   The number of annotations of the type sub-entity.
    size_t get_type_sub_entity_annotation_count(
        size_t type_index,
        size_t entity_index) const MDL_FINAL;

    /// Get the annotation at annotation_index of the type sub-entity at (type_index, entity_index).
    ///
    /// \param type_index          The index of the type.
    /// \param entity_index        The index of the sub entity.
    /// \param annotation_index    The index of the annotation.
    /// \returns                   The annotation.
    DAG_node const *get_type_sub_entity_annotation(
        size_t type_index,
        size_t entity_index,
        size_t annotation_index) const MDL_FINAL;

    /// Get the number of exported constants.
    size_t get_constant_count() const MDL_FINAL;

    /// Get the name of the constant at index.
    ///
    /// \param      index  The index of the constant.
    /// \returns           The name of the constant.
    char const *get_constant_name(
        size_t index) const MDL_FINAL;

    /// Get the value of the constant at index.
    ///
    /// \param      index  The index of the constant.
    /// \returns           The value of the constant.
    DAG_constant const *get_constant_value(
        size_t index) const MDL_FINAL;

    /// Get the number of annotations of the constant at index.
    ///
    /// \param      index  The index of the constant.
    size_t get_constant_annotation_count(
        size_t index) const MDL_FINAL;

    /// Get the annotation at annotation_index of the constant at constant_index.
    ///
    /// \param constant_index      The index of the constant.
    /// \param annotation_index    The index of the annotation.
    /// \returns                   The annotation.
    DAG_node const *get_constant_annotation(
        size_t constant_index,
        size_t annotation_index) const MDL_FINAL;

    /// Access messages (writable).
    Messages &access_messages() MDL_FINAL;

    /// Returns the amount of used memory by this code DAG.
    size_t get_memory_size() const MDL_FINAL;

    /// Get the property flag of the function at function_index.
    /// \param      function_index  The index of the function.
    /// \param      fp              The property.
    /// \returns                    True if this function has the property, false if not.
    bool get_function_property(
        size_t            function_index,
        Function_property fp) const MDL_FINAL;

    /// Get the number of entities referenced by a function.
    /// \param      function_index  The index of the function.
    /// \returns                    Number of function that might be called by this function
    size_t get_function_references_count(size_t function_index) const MDL_FINAL;

    /// Get the signature of the i'th reference of a function
    /// \param      function_index  The index of the function.
    /// \param      callee_index    The index of the callee.
    /// \returns                    Number of function that might be called by this function
    char const *get_function_reference(
        size_t function_index,
        size_t callee_index) const MDL_FINAL;

    /// Return the original function name of a cloned function or NULL if the function
    /// is not a clone.
    ///
    /// \param function_index   The index of the function.
    /// \returns                The absolute name of the original function or NULL.
    char const *get_cloned_function_name(
        size_t function_index) const MDL_FINAL;

    /// Get the number of annotations of the module.
    /// \returns                    The number of annotations.
    size_t get_module_annotation_count() const MDL_FINAL;

    /// Get the annotation at annotation_index of the module.
    /// \param      annotation_index    The index of the annotation.
    /// \returns                        The annotation.
    DAG_node const *get_module_annotation(
        size_t annotation_index) const MDL_FINAL;

    /// Get the internal space
    char const *get_internal_space() const MDL_FINAL;

    /// Get the number of annotations in the generated code.
    ///
    /// \returns    The number of annotations in this generated code.
    size_t get_annotation_count() const MDL_FINAL;

    /// Get the semantics of the annotation at annotation_index.
    ///
    /// \param annotation_index  The index of the annotation.
    /// \returns                 The semantics of the annotation.
    IDefinition::Semantics get_annotation_semantics(
        size_t annotation_index) const MDL_FINAL;

    /// Get the name of the annotation at annotation_index.
    ///
    /// \param annotation_index  The index of the annotation.
    /// \returns                 The name of the annotation.
    char const *get_annotation_name(
        size_t annotation_index) const MDL_FINAL;

    /// Get the simple name of the annotation at annotation_index.
    ///
    /// \param annotation_index  The index of the annotation.
    /// \returns                 The simple name of the annotation.
    char const *get_simple_annotation_name(
        size_t annotation_index) const MDL_FINAL;

    /// Get the original name of the annotation at annotation_index if the annotation name is
    /// an alias, i.e. re-exported from a module.
    ///
    /// \param annotation_index  The index of the annotation.
    /// \returns                 The original name of the annotation or NULL.
    char const *get_original_annotation_name(
        size_t annotation_index) const MDL_FINAL;

    /// Get the parameter count of the annotation at annotation_index.
    ///
    /// \param annotation_index  The index of the annotation.
    /// \returns                 The number of parameters of the annotation.
    size_t get_annotation_parameter_count(
        size_t annotation_index) const MDL_FINAL;

    /// Get the parameter type of the parameter at parameter_index
    /// of the annotation at annotation_index.
    ///
    /// \param annotation_index  The index of the annotation.
    /// \param parameter_index   The index of the parameter.
    /// \returns                 The type of the parameter.
    IType const *get_annotation_parameter_type(
        size_t annotation_index,
        size_t parameter_index) const MDL_FINAL;

    /// Get the parameter type name of the parameter at parameter_index
    /// of the annotation at annotation_index.
    /// \param annotation_index  The index of the annotation.
    /// \param parameter_index   The index of the parameter.
    /// \returns                 The type of the parameter.
    char const *get_annotation_parameter_type_name(
        size_t annotation_index,
        size_t parameter_index) const MDL_FINAL;

        /// Get the parameter name of the parameter at parameter_index
    /// of the annotation at annotation_index.
    ///
    /// \param annotation_index  The index of the annotation.
    /// \param parameter_index   The index of the parameter.
    /// \returns                 The name of the parameter.
    char const *get_annotation_parameter_name(
        size_t annotation_index,
        size_t parameter_index) const MDL_FINAL;

    /// Get the index of the parameter parameter_name.
    ///
    /// \param annotation_index  The index of the annotation.
    /// \param parameter_name    The name of the parameter.
    /// \returns                 The index of the parameter, or -1 if it does not exist.
    size_t get_annotation_parameter_index(
        size_t     annotation_index,
        char const *parameter_name) const MDL_FINAL;

    /// Get the default initializer of the parameter at parameter_index
    /// of the annotation at annotation_index.
    ///
    /// \param annotation_index   The index of the annotation.
    /// \param parameter_index    The index of the parameter.
    /// \returns                  The default initializer or NULL if not available.
    DAG_node const *get_annotation_parameter_default(
        size_t annotation_index,
        size_t parameter_index) const MDL_FINAL;

    /// Get the property flag of the annotation at annotation_index.
    ///
    /// \param annotation_index  The index of the annotation.
    /// \param ap                The requested annotation property.
    /// \returns                 True if this annotation has the property, false if not.
    bool get_annotation_property(
        size_t              annotation_index,
        Annotation_property ap) const MDL_FINAL;

    /// Get the number of annotations of the annotation at annotation_index.
    ///
    /// \param annotation_index  The index of the annotation.
    /// \returns               The number of annotations.
    size_t get_annotation_annotation_count(
        size_t annotation_index) const MDL_FINAL;

    /// Get the annotation at annotation_index of the annotation (declaration) at anno_decl_index.
    ///
    /// \param anno_decl_index    The index of the annotation (declaration).
    /// \param annotation_index   The index of the annotation.
    /// \returns                  The annotation.
    DAG_node const *get_annotation_annotation(
        size_t anno_decl_index,
        size_t annotation_index) const MDL_FINAL;

    /// Get a tag,for a resource constant that might be reachable from this DAG.
    ///
    /// \param res             a resource
    int get_resource_tag(
        IValue_resource const *res) const MDL_FINAL;

    /// Set a tag, version pair for a resource constant that might be reachable from this DAG.
    ///
    /// \param res             a resource
    /// \param tag             the tag value
    void set_resource_tag(
        IValue_resource const *res,
        int                   tag) MDL_FINAL;

    /// Get the number of resource map entries.
    size_t get_resource_tag_map_entries_count() const MDL_FINAL;

    /// Get the i'th resource tag tag map entry or NULL if the index is out of bounds;
    Resource_tag_tuple const *get_resource_tag_map_entry(size_t index) const MDL_FINAL;

    /// Get the resource tagger for this code DAG.
    IResource_tagger *get_resource_tagger() const MDL_FINAL;

    // --------------------------- non interface methods ---------------------------

    /// Get the value factory of this code.
    Value_factory *get_value_factory() { return &m_value_factory; }

    /// Find the tag for a given resource.
    ///
    /// \param res  the resource
    ///
    /// \return the associated tag or zero if no tag was associated
    int find_resource_tag(
        IValue_resource const *res) const;

    /// Adds a tag, version pair for a given resource.
    ///
    /// \param res  the resource
    ///
    /// \return the associated tag or zero if no tag was associated
    void add_resource_tag(
        IValue_resource const *res,
        int                   tag);

    /// Serialize this code DAG.
    ///
    /// \param serializer       the low level serializer
    /// \param bin_serializer   the MDL binary serializer
    void serialize(
        ISerializer           *serializer,
        MDL_binary_serializer *bin_serializer) const;

    /// Deserialize a code DAG.
    ///
    /// \param deserializer      the deserializer used to read the low level data
    /// \param bin_deserializer  the serializer used for deserializing "the binary"
    /// \param compiler          the IMDL compiler interface
    ///
    /// \return the deserialized code DAG
    static Generated_code_dag const *deserialize(
        IDeserializer           *deserializer,
        MDL_binary_deserializer *bin_deserializer,
        MDL                     *compiler);

    /// Dump the material DAG IR.
    ///
    /// \param index   the index of the material to dump
    /// \param suffix  if non-NULL, add this suffix to file name
    /// \param argc    number of arguments
    /// \param argv    arguments
    void dump_material_dag(
        size_t         index,
        char const     *suffix,
        size_t         argc = 0,
        DAG_node const *argv[] = NULL) const;

    /// Get the type factory of this generated code.
    IType_factory *get_type_factory() { return &m_type_factory; }

    /// Get the MDL compiler, increases the ref_count.
    IMDL *get_mdl() const {m_mdl->retain(); return m_mdl.get(); }

    /// Get the node IR-node factory of this code DAG.
    DAG_node_factory_impl const *get_node_factory() const { return &m_node_factory; }

private:
    /// Get the material info for a given material index or NULL if the index is out of range.
    ///
    /// \param material_index  the material index
    Material_info *get_material_info(
        size_t material_index);

    /// Get the material info for a given material index or NULL if the index is out of range.
    ///
    /// \param material_index  the material index
    Material_info const *get_material_info(
        size_t material_index) const;

    /// Get the parameter info for a given material and parameter index pair or NULL if
    /// one index is out of range.
    ///
    /// \param material_index   the material index
    /// \param parameter_index  the parameter index
    Parameter_info const *get_mat_param_info(
        size_t material_index,
        size_t parameter_index) const;

    /// Get the function info for a given function index or NULL if the index is out of range.
    ///
    /// \param function_index  the function index
    Function_info const *get_function_info(
        size_t function_index) const;

    /// Get the parameter info for a given function and parameter index pair or NULL if
    /// one index is out of range.
    ///
    /// \param function_index   the function index
    /// \param parameter_index  the parameter index
    Parameter_info const *get_func_param_info(
        size_t function_index,
        size_t parameter_index) const;

    /// Get the annotation info for a given annotation index or NULL if the index is out of range.
    ///
    /// \param annotation_index  the annotation index
    Annotation_info const *get_annotation_info(
        size_t annotation_index) const;

    /// Get the parameter info for a given annotation and parameter index pair or NULL if
    /// one index is out of range.
    ///
    /// \param annotation_index   the annotation index
    /// \param parameter_index  the parameter index
    Parameter_info const *get_anno_param_info(
        size_t annotation_index,
        size_t parameter_index) const;

    /// Get the user type info for a given type index or NULL if the index is out of range.
    ///
    /// \param type_index  the type index
    User_type_info const *get_type_info(
        size_t type_index) const;

    /// Get the user constant info for a given constant index or NULL if the index is out of range.
    ///
    /// \param constant_index  the constant index
    Constant_info const *get_constant_info(
        size_t constant_index) const;

    /// Add an import if not already there.
    ///
    /// \param mod_name  The absolute module name to be added.
    void add_import(char const *mod_name);

    /// Convert a type to a name.
    ///
    /// \param type   The type to convert.
    /// \param module The (virtual) owner module of this type.
    /// \returns      The name for the type.
    string type_to_name(IType const *type, IModule const *module)
    {
        return m_mangler.mangle(type, module);
    }

    /// Creates a new error message.
    ///
    /// \param code  the error code
    /// \param loc   the error location
    /// \param msg   the error message
    void error(int code, Err_location const &loc, char const *msg);

    /// Check if the name for the given definition must get a signature suffix.
    ///
    /// \param def   The definition.
    bool need_signature_suffix(IDefinition const *def) const;

    /// Compile the given entity to the DAG representation.
    ///
    /// \param dag_builder  the DAG builder to be used
    /// \param node         the node inside the dependence graph of the entity
    void compile_entity(
        DAG_builder           &dag_builder,
        Dependence_node const *node);

    /// Compile the given local entity to the DAG representation.
    ///
    /// \param dag_builder  the DAG builder to be used
    /// \param node         the node inside the dependence graph of the entity
    void compile_local_entity(
        DAG_builder           &dag_builder,
        Dependence_node const *node);

    /// Returns true for MDL definition that should not be visible in the DAG backend.
    ///
    /// \param def                      The definition.
    static bool skip_definition(IDefinition const *def);

    /// Build the DAG for the builtin material constructor.
    ///
    /// \param def                      The definition.
    DAG_node const *build_material_dag(IDefinition const *def);

    /// Compile a user defined type.
    ///
    /// \param dag_builder  the DAG builder to be used
    /// \param def          the definition of the type.
    /// \param is_exported  true, if this type is exported
    void compile_type(
        DAG_builder       &dag_builder,
        IDefinition const *def,
        bool              is_exported);

    /// Compile a constant.
    ///
    /// \param dag_builder  the DAG builder to be used
    /// \param def          the definition of the constant.
    void compile_constant(
        DAG_builder       &dag_builder,
        IDefinition const *def);

private:
    /// The type of maps from definitions to temporary values (DAG-IR nodes).
    typedef ptr_hash_map<IDefinition const, DAG_node const *>::Type Definition_temporary_map;

    /// The type of vectors of expressions.
    typedef vector<const mi::mdl::IExpression *>::Type Expression_vector;

    /// Generate annotations for functions (only for the function itself).
    ///
    /// \param dag_builder  the DAG builder to be used
    /// \param func         the function info to be filled
    /// \param decl         the declaration of the function (or of the constructor type)
    void gen_function_annotations(
        DAG_builder        &dag_builder,
        Function_info      &func,
        IDeclaration const *decl);

    /// Generate annotations for annotation declarations (only for the decl itself).
    ///
    /// \param dag_builder  the DAG builder to be used
    /// \param anno         the annotation info to be filled
    /// \param decl         the declaration of the annotation
    void gen_anno_decl_annotations(
        DAG_builder                   &dag_builder,
        Annotation_info               &anno,
        IDeclaration_annotation const *decl);

    /// Generate annotations for function return types.
    ///
    /// \param dag_builder  the DAG builder to be used
    /// \param func         the function info to be filled
    /// \param decl         the declaration of the function (or of the constructor type)
    void gen_function_return_annotations(
        DAG_builder        &dag_builder,
        Function_info      &func,
        IDeclaration const *decl);

    /// Generate annotations for function parameters.
    ///
    /// \param dag_builder   the DAG builder to be used
    /// \param param         the parameter
    /// \param f_def         the definition of the function
    /// \param owner_module  the owner module of the function definition
    /// \param decl          its declaration (or the declaration of the constructor type)
    /// \param k             the parameter index
    void gen_function_parameter_annotations(
        DAG_builder        &dag_builder,
        Parameter_info     &param,
        IDefinition const  *f_def,
        IModule const      *owner_module,
        IDeclaration const *decl,
        int                k);

    /// Generate annotations for annotation (declaration) parameters.
    ///
    /// \param dag_builder   the DAG builder to be used
    /// \param param         the parameter
    /// \param f_def         the definition of the annotation
    /// \param owner_module  the owner module of the annotation definition
    /// \param decl          its declaration
    /// \param k             the parameter index
    void gen_annotation_parameter_annotations(
        DAG_builder                   &dag_builder,
        Parameter_info                &param,
        IDefinition const             *f_def,
        IModule const                 *owner_module,
        IDeclaration_annotation const *decl,
        int                           k);

    /// Generate annotations for the module.
    ///
    /// \param dag_builder  the DAG builder to be used
    /// \param decl         the declaration of the module
    void gen_module_annotations(
        DAG_builder               &dag_builder,
        IDeclaration_module const *decl);

    /// Compile an annotation (declaration).
    ///
    /// \param module   The owner module of the annotation to compile.
    /// \param a_node   The dependence graph node of the annotation.
    void compile_annotation(
        IModule const         *module,
        Dependence_node const *a_node);

    /// Compile a local annotation (declaration).
    ///
    /// \param module       the owner module of the annotation to compile
    /// \param dag_builder  the DAG builder to be used
    /// \param a_node       the dependence graph node of the annotation
    void compile_local_annotation(
        IModule const         *module,
        DAG_builder           &dag_builder,
        Dependence_node const *a_node);

    /// Compile a function.
    ///
    /// \param module   The owner module of the function to compile.
    /// \param f_node   The dependence graph node of the function.
    void compile_function(
        IModule const         *module,
        Dependence_node const *f_node);

    /// Compile a local function.
    ///
    /// \param module       the owner module of the function to compile
    /// \param dag_builder  the DAG builder to be used
    /// \param f_node       the dependence graph node of the function
    void compile_local_function(
        IModule const         *module,
        DAG_builder           &dag_builder,
        Dependence_node const *f_node);

    /// Compile a material.
    ///
    /// \param dag_builder   the DAG builder to be used
    /// \param m_node        the dependence graph node of the material
    void compile_material(
        DAG_builder           &dag_builder,
        Dependence_node const *m_node);

    /// Compile a local material.
    ///
    /// \param dag_builder   the DAG builder to be used
    /// \param material_def  the definition of the material
    void compile_local_material(
        DAG_builder       &dag_builder,
        IDefinition const *material_def);

    /// Compile a module.
    ///
    /// \param module  The module to compile.
    void compile(IModule const *module);

    /// Helper function, adds a "hidden" annotation to a generated function.
    ///
    /// \param dag_builder      the DAG builder to be used
    /// \param func             the function info
    void mark_hidden(
        DAG_builder   &dag_builder,
        Function_info &func);

    /// Helper function, collect all direct callees of a given function.
    ///
    /// \param func          The function info.
    /// \param node          The dependence graph node of the function.
    void collect_callees(
        Function_info        &func,
       Dependence_node const *node);

    /// Build temporaries for a material by traversing the DAG and creating them
    /// for nodes with phen-out > 1.
    ///
    /// \param mat_index     the index of the processed material
    void build_material_temporaries(int mat_index);

    /// Build temporaries for a function by traversing the DAG and creating them
    /// for nodes with phen-out > 1.
    ///
    /// \param func_index  the index of the processed function
    void build_function_temporaries(int func_index);

    /// Add a material temporary.
    ///
    /// \param mat_index    The index of the material.
    /// \param node         The IR node defining the temporary.
    /// \param name         The name of the temporary.
    ///
    /// \returns            The index of the temporary.
    ///
    int add_material_temporary(
        int            mat_index,
        DAG_node const *node,
        char const     *name);

    /// Add a function temporary.
    ///
    /// \param func_index   The index of the function.
    /// \param node         The IR node defining the temporary.
    /// \param name         The name of the temporary.
    ///
    /// \returns            The index of the temporary.
    ///
    int add_function_temporary(
        int            func_index,
        DAG_node const *node,
        char const     *name);

    /// Create a default enum.
    ///
    /// \param value_factory    The value factory to use for constructing the result.
    /// \param type             The enum type.
    /// \returns                The default vector.
    static IValue_enum const *create_default_enum(
        IValue_factory   &value_factory,
        IType_enum const *type);

    /// Create a default bsdf.
    ///
    /// \param value_factory    The value factory to use for constructing the result.
    /// \returns                The default bsdf.
    static IValue_invalid_ref const *create_default_bsdf(
        IValue_factory &value_factory);

    /// Create a default hair_bsdf.
    ///
    /// \param value_factory    The value factory to use for constructing the result.
    /// \returns                The default hair_bsdf.
    static IValue_invalid_ref const *create_default_hair_bsdf(
        IValue_factory &value_factory);

    /// Create a default edf.
    ///
    /// \param value_factory    The value factory to use for constructing the result.
    /// \returns                The default edf.
    static IValue_invalid_ref const *create_default_edf(
        IValue_factory &value_factory);

    /// Create a default vdf.
    ///
    /// \param value_factory    The value factory to use for constructing the result.
    /// \returns                The default vdf.
    static IValue_invalid_ref const *create_default_vdf(
        IValue_factory &value_factory);

    /// Create a default vector.
    ///
    /// \param value_factory    The value factory to use for constructing the result.
    /// \param type             The vector type.
    /// \returns                The default vector.
    IValue_vector const *create_default_vector(
        IValue_factory     &value_factory,
        IType_vector const *type) const;

    /// Create a default matrix.
    ///
    /// \param value_factory    The value factory to use for constructing the result.
    /// \param type             The matrix type.
    /// \returns                The default matrix.
    IValue_matrix const *create_default_matrix(
        IValue_factory     &value_factory,
        IType_matrix const *type) const;

    /// Create a default color.
    ///
    /// \param value_factory    The value factory to use for constructing the result.
    /// \returns                The default color.
    static IValue_rgb_color const *create_default_color(
        IValue_factory &value_factory);

    /// Create a default texture.
    ///
    /// \param value_factory    The value factory to use for constructing the result.
    /// \param type             The texture type.
    /// \returns                The default texture.
    static IValue_invalid_ref const *create_default_texture(
        IValue_factory      &value_factory,
        IType_texture const *type);

    /// Serialize all DAGs of this code DAG.
    ///
    /// \param dag_serializer  the DAG IR serializer
    void serialize_dags(DAG_serializer &dag_serializer) const;

    /// Deserialize all DAGs of this code DAG.
    ///
    /// \param dag_deserializer  the DAG IR deserializer
    void deserialize_dags(DAG_deserializer &dag_deserializer);

    /// Serialize all Function_infos of this code DAG.
    ///
    /// \param dag_serializer  the DAG IR serializer
    void serialize_functions(DAG_serializer &dag_serializer) const;

    /// Deserialize all Function_infos of this code DAG.
    ///
    /// \param dag_serializer  the DAG IR serializer
    void deserialize_functions(DAG_deserializer &dag_deserializer);

    /// Serialize all Material_infos of this code DAG.
    ///
    /// \param dag_serializer  the DAG IR serializer
    void serialize_materials(DAG_serializer &dag_serializer) const;

    /// Deserialize all Material_infos of this code DAG.
    ///
    /// \param dag_serializer  the DAG IR serializer
    void deserialize_materials(DAG_deserializer &dag_deserializer);

    /// Serialize all Annotation_infos of this code DAG.
    ///
    /// \param dag_serializer  the DAG IR serializer
    void serialize_annotations(DAG_serializer &dag_serializer) const;

    /// Deserialize all Annotation_infos of this code DAG.
    ///
    /// \param dag_serializer  the DAG IR serializer
    void deserialize_annotations(DAG_deserializer &dag_deserializer);

    /// Serialize one parameter.
    ///
    /// \param param           the parameter
    /// \param dag_serializer  the DAG IR serializer
    void serialize_parameter(Parameter_info const &param, DAG_serializer &dag_serializer) const;

    /// Deserialize one parameters of a function.
    ///
    /// \param func              the function
    /// \param dag_deserializer  the DAG IR serializer
    Parameter_info deserialize_parameter(DAG_deserializer &dag_deserializer);

    /// Serialize all parameters of a function.
    ///
    /// \param func            the function
    /// \param dag_serializer  the DAG IR serializer
    void serialize_parameters(Function_info const &func, DAG_serializer &dag_serializer) const;

    /// Deserialize all parameters of a function.
    ///
    /// \param func              the function
    /// \param dag_deserializer  the DAG IR serializer
    void deserialize_parameters(Function_info &mat, DAG_deserializer &dag_deserializer);

    /// Serialize all parameters of a material.
    ///
    /// \param mat             the material
    /// \param dag_serializer  the DAG IR serializer
    void serialize_parameters(Material_info const &mat, DAG_serializer &dag_serializer) const;

    /// Deserialize all parameters of a material.
    ///
    /// \param mat               the material
    /// \param dag_deserializer  the DAG IR serializer
    void deserialize_parameters(Material_info &mat, DAG_deserializer &dag_deserializer);

    /// Serialize all parameters of an annotation.
    ///
    /// \param anno            the annotation
    /// \param dag_serializer  the DAG IR serializer
    void serialize_parameters(Annotation_info const &anno, DAG_serializer &dag_serializer) const;

    /// Deserialize all parameters of a annotation.
    ///
    /// \param anno              the annotation
    /// \param dag_deserializer  the DAG IR serializer
    void deserialize_parameters(Annotation_info &anno, DAG_deserializer &dag_deserializer);

    /// Serialize all User_type_infos of this code DAG.
    ///
    /// \param dag_serializer  the DAG IR serializer
    void serialize_user_types(DAG_serializer &dag_serializer) const;

    /// Serialize all entities of a user type.
    ///
    /// \param type            a user type
    /// \param dag_serializer  the DAG IR serializer
    void serialize_entities(
        User_type_info const &type,
        DAG_serializer       &dag_serializer) const;

    /// Deserialize all User_type_infos of this code DAG.
    ///
    /// \param dag_deserializer  the DAG IR deserializer
    void deserialize_user_types(DAG_deserializer &dag_deserializer);

    /// Deserialize all entities of a user type.
    ///
    /// \param type            a user type
    /// \param dag_serializer  the DAG IR deserializer
    void deserialize_entities(
        User_type_info   &type,
        DAG_deserializer &dag_deserializer);

    /// Serialize all Constant_infos of this code DAG.
    ///
    /// \param dag_serializer  the DAG IR serializer
    void serialize_constants(DAG_serializer &dag_serializer) const;

    /// Deserialize all Constant_infos of this code DAG.
    ///
    /// \param dag_deserializer  the DAG IR deserializer
    void deserialize_constants(DAG_deserializer &dag_deserializer);

private:
    /// The memory arena.
    Memory_arena m_arena;

    /// The absolute name of this module.
    string m_module_name;

    /// The filename of this module.
    string m_module_file_name;

    /// The definition mangler.
    DAG_mangler m_mangler;

    /// The printer for names.
    Name_printer &m_printer;

    /// The symbol table of the code generator.
    Symbol_table m_sym_tab;

    /// The type factory of the code generator.
    Type_factory m_type_factory;

    /// The value factory of the code generator.
    Value_factory m_value_factory;

    /// The list of generated messages while this module was translated by the code generator.
    Messages_impl m_messages;

    /// The names of imported modules.
    String_vector m_module_imports;

    /// The "invisible" symbol, used to express "unsized arrays".
    ISymbol const *m_invisible_sym;

    /// The builder for DAG nodes.
    mutable Allocator_builder m_builder;

    /// The MDL compiler.
    mi::base::Handle<MDL> m_mdl;

    /// The IR node factory.
    DAG_node_factory_impl m_node_factory;

    /// The annotations of the module.
    Dag_vector m_module_annotations;

    /// The materials of this compiled module.
    Function_vector m_functions;

    /// The materials of this compiled module.
    Material_vector m_materials;

    /// The (declared) annotations of this compiled module.
    Annotation_vector m_annotations;

    /// The (exported and non-exported) user defined types of this compiled module.
    User_type_vector m_user_types;

    /// The (exported) user defined constants of this compiled module.
    Constant_vector m_user_constants;

    /// The internal space.
    string m_internal_space;

    /// The name of the "renderer context", used for error messages.
    string m_renderer_context_name;

    /// Extra options steering compilation.
    Compile_options m_options;

    /// The index of the current material.
    int m_current_material_index;

    /// The index of the current function.
    int m_current_function_index;

    /// If true, the "::anno" module must be additionally imported.
    bool m_needs_anno;

    /// If true, DAG backend generated entities must be marked.
    bool m_mark_generated;

    typedef vector<Resource_tag_tuple>::Type Resource_tag_map;

    /// The resource tag map, mapping accessible resources to tags.
    Resource_tag_map m_resource_tag_map;

    /// The resource tagger, using the resource to tag map;
    mutable Resource_tagger m_resource_tagger;
};

}  // mdl
}  // mi

#endif // MDL_GENERATOR_DAG_GENERATED_DAG

