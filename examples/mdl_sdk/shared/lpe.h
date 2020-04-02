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

 // Code shared by all examples that use Light path expressions (LPEs).

#ifndef EXAMPLE_SHARED_LPE_H
#define EXAMPLE_SHARED_LPE_H

#include <string>
#include <vector>
#include <initializer_list>
#include <unordered_map>
#include <functional>

// Create a state machine to evaluate light path expressions (LPEs).
// These expressions can use string handles to select particular lobes in the material, i.e.,
// elemental BSDFs. Handles can also be used to specify light groups and even cameras, even though
// the latter is probably an exotic use case.
//
// The handles are transformed to allow processing by the individual components,
// here is a small summery:
//
// * string 'handle'        used in the MDL and the LPEs to name and select lobes, lights, ...
// * uint32 'global_tag'    string 'handles' mapped to integers. The same map is used for all
//          (gtag)          objects in the scene. All handles have to be added before the state
//                          machine is built.
//                          See 'LPE_state_machine::handle_to_global_tag(const char*)'.
// * uint32 'material_tag'  A local map from string 'handles' to integer. This one is a zero-based
//          (mtag)          enumeration of handles that appear in an ITarget_code. It is
//                          specific to one single distribution function and it is used when calling 
//                          the generated 'sample', 'evaluate', ... functions.

// ------------------------------------------------------------------------------------------------


class LPE_expression;
class NFA_node;

// types and utility functions to set up light path expressions
class LPE
{
public:
    enum class Scatter_event
    {
        None = -1,  // default value for non-scatter events
        D,          // diffuse reflection or transmission,
        DR,         // diffuse reflection,
        DT,         // diffuse transmission,
        G,          // glossy reflection or transmission,
        GR,         // glossy reflection,
        GT,         // glossy transmission,
        S,          // specular reflection or transmission,
        SR,         // specular reflection,
        ST,         // specular transmission,
    };
       
    static LPE_expression camera(const char* handle = nullptr, bool positive = true);
    static LPE_expression light(const char* handle = nullptr, bool positive = true);
    static LPE_expression emission(const char* handle = nullptr, bool positive = true);
    static LPE_expression scattering(
        Scatter_event event, const char* handle = nullptr, bool positive = true);
    static LPE_expression any_scatter(const char* handle = nullptr, bool positive = true);
    static LPE_expression any_light(); // light source or emission
    static LPE_expression any(); // any scatter or light event

    static LPE_expression alternative(std::initializer_list<LPE_expression> operands);
    static LPE_expression alternative(std::vector<LPE_expression> operands);
    static LPE_expression sequence(std::initializer_list<LPE_expression> operands);
    static LPE_expression sequence(std::vector<LPE_expression> operands);
    static LPE_expression zero_or_more(LPE_expression operand);
    static LPE_expression one_or_more(LPE_expression operand);

    enum class Common
    {
        Beauty,
        Diffuse,
        Diffuse_Direct,
        Diffuse_Indirect,
        Glossy,
        Glossy_Direct,
        Glossy_Indirect,
        Specular,
        Specular_Direct,
        Specular_Indirect,
        Direct,
        Indirect,
        Emission,
        SSS,
        Transmission
    };

    static std::string to_string(Common value);
    static LPE_expression create_common(Common lpe);

private:
    explicit LPE() {}
};

// ------------------------------------------------------------------------------------------------

class LPE_expression
{
    friend class LPE;
    friend class LPE_state_machine;

    enum class Kind
    {
        INVALID = -1,
        Symbol_camera,
        Symbol_light,
        Symbol_emission,
        Symbol_scatter,
        Symbol_any_scatter,
        Symbol_any_light,
        Symbol_any,
        Operator_alternative,
        Operator_sequence,
        Operator_zero_or_more,
        Operator_one_or_more,

    };
public:
    explicit LPE_expression();
    Kind get_kind() const { return m_kind; }
    const char* get_handle() const { return m_handle; }

    std::string print() const;
    void visit(std::function<void(const LPE_expression&)> action) const;

private:
    explicit LPE_expression(Kind kind, const char* handle = nullptr, bool positive = true);

    // transform the expression into a normal form
    LPE_expression expand(const std::vector<std::string>& present_handles);

    // expand the not explicitly named handles and negated handles
    LPE_expression expand_handles(const std::vector<std::string>& present_handles);

    // add options to an LPE::alternative
    void merge(const LPE_expression& expression);

    // general
    Kind m_kind;

    // specialized
    LPE::Scatter_event m_scatter_event;
    const char* m_handle;
    bool m_match_handle_positive;
    std::vector<LPE_expression> m_operands;
};

// ------------------------------------------------------------------------------------------------

class LPE_state_machine
{
public:

    // the kind of events that are handled during runtime are limited to these state events
    enum class Transition_type
    {
        EPSILON = -2,
        INVALID = -1,
        Camera = 0,
        Light,
        Emission,
        Scatter_DR,
        Scatter_DT,
        Scatter_GR,
        Scatter_GT,
        Scatter_SR,
        Scatter_ST,
        COUNT,
    };

    static std::string to_string(Transition_type e)
    {
        switch (e)
        {
            case Transition_type::EPSILON:    return "epsilon";
            case Transition_type::INVALID:    return "INVALID";
            case Transition_type::Camera:     return "Camera";
            case Transition_type::Light:      return "Light";
            case Transition_type::Emission:   return "Emission";
            case Transition_type::Scatter_DR: return "Scatter_DR";
            case Transition_type::Scatter_DT: return "Scatter_DT";
            case Transition_type::Scatter_GR: return "Scatter_GR";
            case Transition_type::Scatter_GT: return "Scatter_GT";
            case Transition_type::Scatter_SR: return "Scatter_SR";
            case Transition_type::Scatter_ST: return "Scatter_ST";
            case Transition_type::COUNT:      return "";
        }
        return "";
    }

    // the transition between two states depends on an event and an LPE tag
    static uint32_t create_transition(Transition_type event, uint32_t lpe_tag)
    {
        // note this does works for about 2^27 LPE tags, which should be okay
        return static_cast<uint32_t>(Transition_type::COUNT) * lpe_tag + 
               static_cast<uint32_t>(event);
    }

    // get the event (type) from a transition
    static Transition_type get_transition_type(uint32_t transaction)
    {
        return static_cast<Transition_type>(
            transaction % static_cast<uint32_t>(Transition_type::COUNT));
    }

    // get the handle of a transition
    static uint32_t get_transition_handle_id(uint32_t transaction)
    {
        return transaction / static_cast<uint32_t>(Transition_type::COUNT);
    }

    // --------------------------

    explicit LPE_state_machine();
    virtual ~LPE_state_machine() = default;

    // maps string 'handles' to 'global tags'.
    // All handles of the scene (appearing in materials and lights) have to be registered
    // using this function before the LPE state machine is built.
    // Calling again with the same 'handle' is valid and returns the same 'global tag'.
    // The renderer can choose to handle this differently since global and LPE IDs are managed
    // completely on the application side. MDL cares only about string handles and 'Material IDs',
    // a zero based enumeration of handles that appear in a ICompiled_material.
    uint32_t handle_to_global_tag(const char* handle);

    // Add a new LPE to the state machine and the index i`th expression) in the machine.
    // Has to be called before building.
    uint32_t add_expression(const std::string& name, LPE_expression expression)
    {
        m_expressions.push_back({name, expression});
        return m_expressions.size() - 1;
    }

    // get the number of expressions added to this machine.
    uint32_t get_expression_count() const { return m_expressions.size(); }

    // get the i`th expression name
    const char* get_expression_name(uint32_t index) const {
        return m_expressions[index].first.c_str(); }

    // get the i`th expression
    const LPE_expression& get_expression_(uint32_t index) const {
        return m_expressions[index].second; }

    // Generates the actual state machine.
    void build();

    // Number of possible transitions per state in the DFA
    uint32_t get_transition_count() const { return m_num_transitions; }

    // Number of states in the DFA
    uint32_t get_state_count() const { return m_num_states; }

    // Runtime data: The actual state machine in form of a matrix of size #states x #transitions.
    // The matrix elements at position [r,c] contains the id of the state after a
    // transition from state r while following transition c. An element value of -1 represents
    // an invalid transition.
    const std::vector<uint32_t>& get_state_table() const { return m_state_table; }

    // runtime data: bit-mask that identifies the final states of the individual LPEs.
    const std::vector<uint32_t>& get_final_state_masks() const { return  m_final_state_masks; }

private:

    // processes an expression during the construction of the NFA from LPE expressions.
    void proccess(NFA_node* start, NFA_node* end, const LPE_expression& expr);

    // all string 'handles' with corresponding 'global tags' present in the scene.
    std::unordered_map<std::string, uint32_t> m_global_tag_map;

    // list of expressions that are handled by this machine.
    std::vector<std::pair<std::string, LPE_expression>> m_expressions;
    
    // number of possible transitions per state in the DFA.
    uint32_t m_num_transitions;

    // number of states in the DFA.
    uint32_t m_num_states;

    // runtime data: transitions between states.
    std::vector<uint32_t> m_state_table;

    // runtime data: bit-mask that identifies the final states of the individual LPEs.
    std::vector<uint32_t> m_final_state_masks;
};

#endif
