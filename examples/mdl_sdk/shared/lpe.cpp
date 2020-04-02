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

#include <cassert>
#include <stack>
#include <queue>
#include <map>
#include <set>
#include <mutex>

#include <iostream>
#include "lpe.h"

// ------------------------------------------------------------------------------------------------

LPE_expression::LPE_expression()
    : LPE_expression(Kind::INVALID)
{
}

LPE_expression::LPE_expression(Kind kind, const char* tag, bool positive)
    : m_kind(kind) 
    , m_scatter_event(LPE::Scatter_event::None)
    , m_handle(tag)
    , m_match_handle_positive(positive)
    , m_operands()
{
}

LPE_expression LPE::camera(const char* tag, bool positive)
{
    return LPE_expression(LPE_expression::Kind::Symbol_camera, tag, positive);
}

LPE_expression LPE::light(const char* tag, bool positive)
{
    return LPE_expression(LPE_expression::Kind::Symbol_light, tag, positive);
}

LPE_expression LPE::emission(const char* tag, bool positive)
{
    return LPE_expression(LPE_expression::Kind::Symbol_emission, tag, positive);
}

LPE_expression LPE::scattering(Scatter_event event, const char* tag, bool positive)
{
    LPE_expression symbol(LPE_expression::Kind::Symbol_scatter, tag, positive);
    symbol.m_scatter_event = event;
    return symbol;
}

LPE_expression LPE::any_scatter(const char* tag, bool positive)
{
    return LPE_expression(LPE_expression::Kind::Symbol_any_scatter, tag, positive);
}

LPE_expression LPE::any_light()
{
    return LPE_expression(LPE_expression::Kind::Symbol_any_light);
}

LPE_expression LPE::any()
{
    return LPE_expression(LPE_expression::Kind::Symbol_any);
}

LPE_expression LPE::alternative(std::initializer_list<LPE_expression> operands)
{
    LPE_expression operation(LPE_expression::Kind::Operator_alternative);
    operation.m_operands.insert(operation.m_operands.end(), operands.begin(), operands.end());
    return operation;
}

LPE_expression LPE::alternative(std::vector<LPE_expression> operands)
{
    LPE_expression operation(LPE_expression::Kind::Operator_alternative);
    operation.m_operands.insert(operation.m_operands.end(), operands.begin(), operands.end());
    return operation;
}

LPE_expression LPE::sequence(std::initializer_list<LPE_expression> operands)
{
    LPE_expression operation(LPE_expression::Kind::Operator_sequence);
    operation.m_operands.insert(operation.m_operands.end(), operands.begin(), operands.end());
    return operation;
}

LPE_expression LPE::sequence(std::vector<LPE_expression> operands)
{
    LPE_expression operation(LPE_expression::Kind::Operator_sequence);
    operation.m_operands.insert(operation.m_operands.end(), operands.begin(), operands.end());
    return operation;
}

LPE_expression LPE::zero_or_more(LPE_expression operand)
{
    LPE_expression operation(LPE_expression::Kind::Operator_zero_or_more);
    operation.m_operands.push_back(operand);
    return operation;
}

LPE_expression LPE::one_or_more(LPE_expression operand)
{
    LPE_expression operation(LPE_expression::Kind::Operator_one_or_more);
    operation.m_operands.push_back(operand);
    return operation;
}

std::string LPE_expression::print() const
{
    std::string symbol = "";
    switch (m_kind)
    {
        case Kind::Operator_zero_or_more: return m_operands[0].print() + "*";
        case Kind::Operator_one_or_more: return m_operands[0].print() + "+";
        case Kind::Operator_alternative:
        {
            std::string res = "(";
            for (auto&& o : m_operands)
                res += o.print() + "|";

            res[res.size() - 1] = ')';
            return res;
        }
        case Kind::Operator_sequence:
        {
            std::string res = "";
            for (auto&& o : m_operands)
                res += o.print();
            return res;
        }
        case Kind::Symbol_any_scatter: symbol = "{D|S|G}"; break;
        case Kind::Symbol_any_light: symbol = "{L|E}"; break;
        case Kind::Symbol_any: return ".";

        case Kind::Symbol_camera: symbol = "C";  break;
        case Kind::Symbol_light: symbol = "L";  break;
        case Kind::Symbol_emission: symbol = "E";  break;
        case Kind::Symbol_scatter:
        {
            switch (m_scatter_event)
            {
                case LPE::Scatter_event::D: symbol = "D";  break;
                case LPE::Scatter_event::DR: symbol = "DR"; break;
                case LPE::Scatter_event::DT: symbol = "DT"; break;
                case LPE::Scatter_event::G: symbol = "G";  break;
                case LPE::Scatter_event::GR: symbol = "GR"; break;
                case LPE::Scatter_event::GT: symbol = "GT"; break;
                case LPE::Scatter_event::S: symbol = "S";  break;
                case LPE::Scatter_event::SR: symbol = "SR"; break;
                case LPE::Scatter_event::ST: symbol = "ST"; break;
                case LPE::Scatter_event::None: break;
            }
            break;
        }
        case Kind::INVALID: break;
    }

    if (symbol.size() > 0)
    {
        bool close = false;
        if (symbol.size() > 1 || m_handle)
        {
            symbol = "<" + symbol;
            close = true;
        }

        if (m_handle)
        {
            symbol += m_match_handle_positive ? "['" : "[^'";
            symbol += std::string(m_handle) + "']";
        }
        return close ? (symbol + ">") : symbol;
    }

    assert(false && "LPE_expression::print case not handled.");
    return "";
}


LPE_expression LPE_expression::expand_handles(const std::vector<std::string>& present_handles)
{
    // keep positive (explicit defined) tags
    if (m_handle && m_match_handle_positive)
        return *this;

    // replace all unspecified tags by all the present handles 
    std::vector<LPE_expression> alternative_events;
    for (auto it = present_handles.begin() + 0; it != present_handles.end(); ++it)
        if (!m_handle || *it != m_handle)
        {
            switch (m_kind)
            {
            case Kind::Symbol_camera:
                alternative_events.push_back(LPE::camera(it->c_str(), true));
                break;

            case Kind::Symbol_emission:
                alternative_events.push_back(LPE::emission(it->c_str(), true));
                break;

            case Kind::Symbol_light:
                alternative_events.push_back(LPE::light(it->c_str(), true));
                break;

            case Kind::Symbol_scatter:
                alternative_events.push_back(LPE::scattering(m_scatter_event, it->c_str(), true));
                break;

            default:
                break;
            }
        }
    return LPE::alternative(alternative_events);
}

void LPE_expression::merge(const LPE_expression& option)
{
    assert(m_kind == Kind::Operator_alternative &&
           "merge alternatives can only be called on 'alternative' operators");

    if (option.m_kind == Kind::Operator_alternative)
        for (auto& op : option.m_operands)
            m_operands.push_back(op);
    else
        m_operands.push_back(option);
}


LPE_expression LPE_expression::expand(const std::vector<std::string>& present_handles)
{
    switch (m_kind)
    {
        case Kind::Symbol_camera:
        case Kind::Symbol_light:
        case Kind::Symbol_emission:
            return expand_handles(present_handles);

        case Kind::Symbol_any_scatter:
        {
            return LPE::alternative({
                LPE::scattering(LPE::Scatter_event::DR, m_handle, m_match_handle_positive).expand(present_handles),
                LPE::scattering(LPE::Scatter_event::DT, m_handle, m_match_handle_positive).expand(present_handles),
                LPE::scattering(LPE::Scatter_event::GR, m_handle, m_match_handle_positive).expand(present_handles),
                LPE::scattering(LPE::Scatter_event::GT, m_handle, m_match_handle_positive).expand(present_handles),
                LPE::scattering(LPE::Scatter_event::SR, m_handle, m_match_handle_positive).expand(present_handles),
                LPE::scattering(LPE::Scatter_event::ST, m_handle, m_match_handle_positive).expand(present_handles)});
        }

        case Kind::Symbol_any_light:
        {
            return LPE::alternative({
                LPE::emission().expand(present_handles),
                LPE::light().expand(present_handles)
            });
        }

        case Kind::Symbol_any:
        {
            return LPE::alternative({
                LPE::scattering(LPE::Scatter_event::DR).expand(present_handles),
                LPE::scattering(LPE::Scatter_event::DT).expand(present_handles),
                LPE::scattering(LPE::Scatter_event::GR).expand(present_handles),
                LPE::scattering(LPE::Scatter_event::GT).expand(present_handles),
                LPE::scattering(LPE::Scatter_event::SR).expand(present_handles),
                LPE::scattering(LPE::Scatter_event::ST).expand(present_handles),
                LPE::emission().expand(present_handles),
                LPE::light().expand(present_handles)
            });
        }

        case Kind::Symbol_scatter:
        {
            // split combined events
            LPE_expression events = LPE::alternative({});
            switch (m_scatter_event)
            {
                case LPE::Scatter_event::D:
                    events.merge(LPE::scattering(
                        LPE::Scatter_event::DR, m_handle, m_match_handle_positive).expand_handles(present_handles));
                    events.merge(LPE::scattering(
                        LPE::Scatter_event::DT, m_handle, m_match_handle_positive).expand_handles(present_handles));
                    break;
                case LPE::Scatter_event::G: 
                    events.merge(LPE::scattering(
                        LPE::Scatter_event::GR, m_handle, m_match_handle_positive).expand_handles(present_handles));
                    events.merge(LPE::scattering(
                        LPE::Scatter_event::GT, m_handle, m_match_handle_positive).expand_handles(present_handles));
                    break;
                case LPE::Scatter_event::S: 
                    events.merge(LPE::scattering(
                        LPE::Scatter_event::SR, m_handle, m_match_handle_positive).expand_handles(present_handles));
                    events.merge(LPE::scattering(
                        LPE::Scatter_event::ST, m_handle, m_match_handle_positive).expand_handles(present_handles));
                    break;
                default:
                    events.merge(expand_handles(present_handles));
            }
            return events;

        }

        case Kind::Operator_zero_or_more:
        {
            m_operands[0] = m_operands[0].expand(present_handles);
            return *this;
        }

        case Kind::Operator_one_or_more:
        {
            m_operands[0] = m_operands[0].expand(present_handles);
            return LPE::sequence({m_operands[0], LPE::zero_or_more(m_operands[0])});
        }

        case Kind::Operator_alternative:
        case Kind::Operator_sequence:
        {
            if (m_operands.size() == 1)
                return m_operands[0].expand(present_handles);

            for (size_t i = 0; i < m_operands.size(); ++i)
                m_operands[i] = m_operands[i].expand(present_handles);
            return *this;
        }

        default:
            assert(!"LPE_expression::expand case not handled.");
    }
    return *this;
}

void LPE_expression::visit(std::function<void(const LPE_expression&)> action) const
{
    switch (m_kind)
    {
        case Kind::Symbol_camera:
        case Kind::Symbol_light:
        case Kind::Symbol_emission: 
        case Kind::Symbol_any_scatter:
        case Kind::Symbol_any_light:
        case Kind::Symbol_any:
        case Kind::Symbol_scatter:
            action(*this);
            return;

        case Kind::Operator_alternative:
        case Kind::Operator_sequence:
        case Kind::Operator_zero_or_more:
        case Kind::Operator_one_or_more:
            action(*this);
            for (auto& o : m_operands)
                o.visit(action);
            return;

        default:
            assert(!"LPE_expression::visit case not handled.");
    }
}

// ------------------------------------------------------------------------------------------------


class NFA_node
{
public:
    explicit NFA_node();
    virtual ~NFA_node() = default;

    void compute_epsilon_start_list();
    std::map<uint32_t, NFA_node*> compute_id_node_map();

    void print(std::vector<std::string>& lpe_tags);

    const uint32_t id;
    std::multimap<uint32_t, NFA_node*> children;
    std::set<NFA_node*> epsilon_star_list;
    uint32_t final_state_for_ith_lpe;
};

// ------------------------------------------------------------------------------------------------

typedef std::set<uint32_t> NFA_nodes;
class DFA_node
{
public:
    explicit DFA_node();
    virtual ~DFA_node() = default;

    void print(std::vector<std::string>& lpe_tags);

    const uint32_t id;
    NFA_nodes nfa_ids;
    std::unordered_map<uint32_t, DFA_node*> children;
    uint32_t final_node_mask;
};

// ------------------------------------------------------------------------------------------------


namespace
{
    // in the standard conversion algorithm from NFA to DFA, for each new state, starting with the
    // starting state, the list NFA nodes that can be reached by Xe* where X is transition symbol
    // and e is epsilon.
    // This function evaluates one cell of this table
    NFA_nodes compute_dfa_table_cell(
        std::map<uint32_t, 
        NFA_node*>& nfa_nodes, 
        const NFA_nodes& row_name, 
        uint32_t trans_id)
    {
        NFA_nodes res;
        for (auto nfa_node_id : row_name)
        {
            NFA_node* nfa = nfa_nodes[nfa_node_id];
            auto trans_connections = nfa->children.equal_range(trans_id);
            for (auto it = trans_connections.first; it != trans_connections.second; ++it)
            {
                auto eps_closure = it->second->epsilon_star_list;
                for (auto eps_it : eps_closure)
                    res.insert(eps_it->id);
            }
        }
        return res;
    }

    // checks if a new DFA node is required, which is the case when a cell in the currently
    // processed row evaluates to a new NFA node combination
    DFA_node* find(
        std::vector<DFA_node*>& dfa_nodes, 
        NFA_nodes nfas)
    {
        for (auto& n : dfa_nodes)
        {
            if (n->nfa_ids.size() != nfas.size())
                continue;

            bool found = true;
            for (auto& nfa_id : n->nfa_ids)
            {
                if (nfas.find(nfa_id) == nfas.end())
                {
                    found = false;
                    break;
                }
            }
            if(found)
                return n; // a node for this NFA node combination exists
        }
        return nullptr;  // no such DFA node exists
    }

    // mark final states for the individual LPEs
    void compute_final_node_mask(
        DFA_node& node, 
        std::map<uint32_t, 
        NFA_node*>& nfa_nodes)
    {
        node.final_node_mask = 0;
        for (auto nfa_id : node.nfa_ids)
        {
            const NFA_node* nfa_node = nfa_nodes[nfa_id];
            if (nfa_node->final_state_for_ith_lpe == static_cast<uint32_t>(-1))
                continue;

            node.final_node_mask |= 1 << nfa_node->final_state_for_ith_lpe;
        }
    }

    // debug printing
    void print_helper(
        DFA_node* current, 
        std::vector<std::string>& lpe_tags, 
        std::set<DFA_node*>& v, 
        uint32_t level = 0)
    {
        v.insert(current);

        std::string indent = "";
        for (uint32_t l = 0; l < level; ++l)
            indent += "  ";

        for (auto& pair : current->children)
        {
            printf("%s(%d) -> %s['%s'] -> (%d)\n",
                indent.c_str(), current->id,
                LPE_state_machine::to_string(LPE_state_machine::get_transition_type(pair.first)).c_str(),
                lpe_tags[LPE_state_machine::get_transition_handle_id(pair.first)].c_str(),
                pair.second->id);
        }
        for (auto& pair : current->children)
        {
            if (v.find(pair.second) == v.end())
                print_helper(pair.second, lpe_tags, v, level + 1);
        }
    }

    template<typename TInteger>
    std::string to_bit_mask_string(TInteger value)
    {
        size_t bits = 8 * sizeof(TInteger);
        std::string mask(bits, '0');
        for(size_t i = 0; i < bits; ++i)
            mask[bits - 1 - i] = ((value & (1 << i)) == 0) ? '0' : '1';
        return mask;
    }
} // anonymous


namespace { static uint32_t s_dfa_counter(0); }
DFA_node::DFA_node()
    : id(s_dfa_counter++)
    , final_node_mask(0)
{
}

void DFA_node::print(std::vector<std::string>& lpe_tags)
{
    std::set<DFA_node*> v;
    print_helper(this, lpe_tags, v);
}


// ------------------------------------------------------------------------------------------------


namespace
{
    void compute_epsilon_start_list_helper(NFA_node* current, std::set<NFA_node*>& star_list)
    {
        if (star_list.find(current) != star_list.end())
            return;

        star_list.insert(current);
        auto epsilons = current->children.equal_range(static_cast<uint32_t>(LPE_state_machine::Transition_type::EPSILON));
        for (auto it = epsilons.first; it != epsilons.second; ++it)
            compute_epsilon_start_list_helper(it->second, star_list);
    }

    void compute_id_node_map_helper(NFA_node* current, std::map<uint32_t, NFA_node*>& map)
    {
        map[current->id] = current;
        for (auto& pair : current->children)
        {
            if (map.find(pair.second->id) == map.end())
                compute_id_node_map_helper(pair.second, map);
        }
    }


    void print_helper(NFA_node* current, std::vector<std::string>& lpe_tags, std::set<NFA_node*>& v, uint32_t level = 0)
    {
        v.insert(current);

        std::string indent = "";
        for (uint32_t l = 0; l < level; ++l)
            indent += "  ";

        for (auto& pair : current->children)
        {
            if (pair.first == static_cast<uint32_t>(LPE_state_machine::Transition_type::EPSILON))
            {
                printf("%s(%d) -> epsilon -> (%d)\n", indent.c_str(), current->id, pair.second->id);
            }
            else
            {
                printf("%s(%d) -> %s['%s'] -> (%d)\n",
                       indent.c_str(), current->id,
                       LPE_state_machine::to_string(LPE_state_machine::get_transition_type(pair.first)).c_str(),
                       lpe_tags[LPE_state_machine::get_transition_handle_id(pair.first)].c_str(),
                       pair.second->id);
            }
        }
        for (auto& pair : current->children)
        {
            if (v.find(pair.second) == v.end())
                print_helper(pair.second, lpe_tags, v, level + 1);
        }
    }
}

namespace { static uint32_t s_nfa_counter(0); }
NFA_node::NFA_node()
    : id(s_nfa_counter++)
    , final_state_for_ith_lpe(static_cast<uint32_t>(-1))
{
}

void NFA_node::compute_epsilon_start_list()
{
    epsilon_star_list.clear();
    compute_epsilon_start_list_helper(this, epsilon_star_list);
}

std::map<uint32_t, NFA_node*> NFA_node::compute_id_node_map()
{
    std::map<uint32_t, NFA_node*> map;
    compute_id_node_map_helper(this, map);
    return map;
}

void NFA_node::print(std::vector<std::string>& lpe_tags)
{
    std::set<NFA_node*> v;
    print_helper(this, lpe_tags, v);
}

// ------------------------------------------------------------------------------------------------

namespace
{
    LPE_state_machine::Transition_type convert(LPE::Scatter_event event)
    {
        switch (event)
        {
        case LPE::Scatter_event::DR: return LPE_state_machine::Transition_type::Scatter_DR;
        case LPE::Scatter_event::DT: return LPE_state_machine::Transition_type::Scatter_DT;
        case LPE::Scatter_event::GR: return LPE_state_machine::Transition_type::Scatter_GR;
        case LPE::Scatter_event::GT: return LPE_state_machine::Transition_type::Scatter_GT;
        case LPE::Scatter_event::SR: return LPE_state_machine::Transition_type::Scatter_SR;
        case LPE::Scatter_event::ST: return LPE_state_machine::Transition_type::Scatter_ST;
        default:
            assert(!"convert LPE::Scatter_event to State_event: case not handled.");
        }
        return LPE_state_machine::Transition_type::INVALID;
    }

} // anonymous


LPE_state_machine::LPE_state_machine()
    : m_num_transitions(0)
    , m_num_states(0)
{
    handle_to_global_tag("");
}

uint32_t LPE_state_machine::handle_to_global_tag(const char* handle)
{
    static std::mutex ls_tag_mutex;
    std::lock_guard<std::mutex> lock(ls_tag_mutex);

    auto it = m_global_tag_map.find(handle);
    if (it != m_global_tag_map.end())
        return it->second;

    uint32_t tag = static_cast<uint32_t>(m_global_tag_map.size());
    m_global_tag_map[handle] = tag;
    return tag;
}

void LPE_state_machine::proccess(NFA_node* start, NFA_node* end, const LPE_expression& expr)
{
    switch (expr.get_kind())
    {
    case LPE_expression::Kind::Symbol_camera:
    {
        uint32_t trans = create_transition(
            Transition_type::Camera, m_global_tag_map[expr.m_handle]);
        start->children.insert({trans, end});
        return;
    }

    case LPE_expression::Kind::Symbol_light:
    {
        uint32_t trans = create_transition(
            Transition_type::Light, m_global_tag_map[expr.m_handle]);
        start->children.insert({trans, end});
        return;
    }

    case LPE_expression::Kind::Symbol_emission:
    {
        uint32_t trans = create_transition(
            Transition_type::Emission, m_global_tag_map[expr.m_handle]);
        start->children.insert({trans, end});
        return;
    }

    case LPE_expression::Kind::Symbol_scatter:
    {
        uint32_t trans = create_transition(
            convert(expr.m_scatter_event), m_global_tag_map[expr.m_handle]);
        start->children.insert({trans, end});
        return;
    }

    case LPE_expression::Kind::Operator_alternative:
    {
        for (auto& operand : expr.m_operands)
            proccess(start, end, operand);
        return;
    }

    case LPE_expression::Kind::Operator_sequence:
    {
        // special case, only one element
        if (expr.m_operands.size() == 1)
        {
            proccess(start, end, expr.m_operands[0]);
            return;
        }

        // add a new node for all segments
        NFA_node* tmp = nullptr;
        for (size_t i = 0; i < expr.m_operands.size(); ++i)
        {
            if (i == expr.m_operands.size() - 1)
            {
                proccess(tmp, end, expr.m_operands[i]);
                return;
            }

            tmp = new NFA_node();
            proccess(start, tmp, expr.m_operands[i]);
            start = tmp;
        }
        return;
    }

    case LPE_expression::Kind::Operator_zero_or_more:
    {
        NFA_node* inner_start = new NFA_node();
        NFA_node* inner_end = new NFA_node();
        proccess(inner_start, inner_end, expr.m_operands[0]);

        uint32_t epsilon = static_cast<uint32_t>(Transition_type::EPSILON);
        start->children.insert({epsilon, end});              // start to end (skip exec)
        start->children.insert({epsilon, inner_start});      // start to inner start
        inner_end->children.insert({epsilon, inner_start});  // loop
        inner_end->children.insert({epsilon, end});          // inner end to end
    }

    default:
        break;
    }
}


void LPE_state_machine::build()
{
    const bool print_debug_output = false;
    
    // handle vector, global tags are implicit
    std::vector<std::string> global_tags;
    global_tags.resize(m_global_tag_map.size());
    for (auto& it : m_global_tag_map)
        global_tags[it.second] = it.first;
    
    // expand expressions to create an extended version of a conjunctive normal form (CNF), 
    // which allows loops
    std::vector<LPE_expression> expanded;
    for (auto&& e : m_expressions)
    {
        if(print_debug_output)
            std::cout << "\n" << e.first.c_str() 
                      << "\ninput:\n" << e.second.print().c_str() << std::endl;

        expanded.emplace_back(std::move(e.second.expand(global_tags)));

        if (print_debug_output)
            std::cout << "\nexpanded:\n" << expanded.back().print().c_str() << std::endl;
    }
    
    // compute NFA
    //-------------------------------------------
    NFA_node* start = new NFA_node();
    for (uint32_t i = 0; i < expanded.size(); ++i)
    {
        NFA_node* end = new NFA_node();
        end->final_state_for_ith_lpe = i;
        proccess(start, end, expanded[i]);
    }

    // map of all states of the NFA
    std::map<uint32_t, NFA_node*> nfa_nodes = start->compute_id_node_map();

    if (print_debug_output)
    {
        std::cout << "\nNFA:\n";
        start->print(global_tags);
    }

    // compute DFA
    //-------------------------------------------
    
    // compute the epsilon closure of each node
    for (auto& n : nfa_nodes)
        n.second->compute_epsilon_start_list();

    if (print_debug_output)
    {
        std::cout << "\nepsilon*:\n";
        for (auto& n : nfa_nodes)
        {
            std::string s = "(" + std::to_string(n.first) + "): ";
            for (auto it : n.second->epsilon_star_list)
                s += " " + std::to_string(it->id);
            printf("%s\n", s.c_str());
        }

        std::cout << "\nfinal states:\n";
        for (auto& n : nfa_nodes)
        {
            if (n.second->final_state_for_ith_lpe != static_cast<uint32_t>(-1))
                printf("(%d): %d\n", n.first, n.second->final_state_for_ith_lpe);
        }
    }

    // build the list of all possible transitions
    m_num_transitions = static_cast<uint32_t>(Transition_type::COUNT) * global_tags.size();

    // compute DFA state table
    std::vector<DFA_node*> dfa_nodes;

    // begin with the row of the starting state
    dfa_nodes.push_back(new DFA_node());               // add to the list of states
    dfa_nodes.back()->nfa_ids = {start->id};
    size_t current_row = 0;

    while (current_row < dfa_nodes.size())
    {
        for (uint32_t t = 0; t < m_num_transitions; ++t)
        {
            NFA_nodes nfas = compute_dfa_table_cell(nfa_nodes, dfa_nodes[current_row]->nfa_ids, t);
            if (nfas.empty())
                continue;

            DFA_node* found = find(dfa_nodes, nfas);
            if (found)
                dfa_nodes[current_row]->children[t] = found;
            else
            {
                dfa_nodes.push_back(new DFA_node());    // add to the list of states
                dfa_nodes.back()->nfa_ids = nfas;
                dfa_nodes[current_row]->children[t] = dfa_nodes.back();
            }
        }
        current_row++;
    }
    m_num_states = dfa_nodes.size();

    // store information about which state is a final state for which LPE
    for (auto& n : dfa_nodes)
        compute_final_node_mask(*n, nfa_nodes);

    if (print_debug_output)
    {
        std::cout << "\nDFA:\n";
        dfa_nodes[0]->print(global_tags);

        std::cout << "\nfinal state mask (1 for i'th LPE):\n";
        for (auto& n : dfa_nodes)
            printf("(%d): %s\n", n->id, to_bit_mask_string(n->final_node_mask).c_str());
    }

    // convert DFA to lookup table
    m_state_table.resize(m_num_states * m_num_transitions, static_cast<uint32_t>(-1));
    m_final_state_masks.resize(m_num_states, 0);
    for (uint32_t s = 0; s < m_num_states; ++s)
    {
        DFA_node* node = dfa_nodes[s];
        for (auto& it : node->children)
            m_state_table[s * m_num_transitions + it.first] = it.second->id;

        m_final_state_masks[s] = node->final_node_mask;
    }

    // free nodes
    for (auto& it : nfa_nodes)
        delete it.second;
    for (auto& it : dfa_nodes)
        delete it;
}

//-------------------------------------------------------------------------------------------------

std::string LPE::to_string(Common value)
{
    switch (value)
    {
        case Common::Beauty: return "Beauty";
        case Common::Diffuse: return "Diffuse";
        case Common::Diffuse_Direct: return "Diffuse Direct";
        case Common::Diffuse_Indirect: return "Diffuse Indirect";
        case Common::Glossy: return "Glossy";
        case Common::Glossy_Direct: return "Glossy Direct";
        case Common::Glossy_Indirect: return "Glossy Indirect";
        case Common::Specular: return "Specular";
        case Common::Specular_Direct: return "Specular Direct";
        case Common::Specular_Indirect: return "Specular Indirect";
        case Common::Direct: return "Direct";
        case Common::Indirect: return "Indirect";
        case Common::Emission: return "Emission";
        case Common::SSS: return "SSS";
        case Common::Transmission: return "Transmission";
        default:
            assert(!"LPE::Common to_string case not handled.");
    }
    return "";
}

LPE_expression LPE::create_common(LPE::Common lpe)
{
    LPE::Scatter_event main_reflect = LPE::Scatter_event::None;
    switch (lpe)
    {
        case Common::Diffuse:
        case Common::Diffuse_Direct:
        case Common::Diffuse_Indirect:
            main_reflect = LPE::Scatter_event::DR;
            break;

        case Common::Glossy:
        case Common::Glossy_Direct:
        case Common::Glossy_Indirect:
            main_reflect = LPE::Scatter_event::GR;
            break;

        case Common::Specular:
        case Common::Specular_Direct:
        case Common::Specular_Indirect:
            main_reflect = LPE::Scatter_event::SR;
            break;

        default:
            break;
    }

    switch (lpe)
    {
        case Common::Beauty:
            return LPE::sequence({
                LPE::camera(),
                LPE::zero_or_more(LPE::any()) });
            
        case Common::Diffuse:
        case Common::Glossy:
        case Common::Specular:
            return LPE::sequence({
                LPE::camera(),
                LPE::scattering(main_reflect),
                LPE::zero_or_more(LPE::any()) });
            
        case Common::Diffuse_Direct:
        case Common::Glossy_Direct:
        case Common::Specular_Direct:
            return LPE::sequence({
                LPE::camera(),
                LPE::scattering(main_reflect),
                LPE::light() });
            
        case Common::Diffuse_Indirect:
        case Common::Glossy_Indirect:
        case Common::Specular_Indirect:
            return LPE::sequence({
                LPE::camera(),
                LPE::scattering(main_reflect),
                LPE::alternative({
                    LPE::any_scatter(),
                    LPE::emission()}),
                LPE::zero_or_more(LPE::any()) });
            
        case Common::Direct:
            return LPE::sequence({
                LPE::camera(),
                LPE::any_scatter(),
                LPE::light() });
            
        case Common::Indirect:
            return LPE::sequence({
                LPE::camera(),
                LPE::any_scatter(),
                LPE::alternative({
                    LPE::any_scatter(),
                    LPE::emission()}),
                LPE::zero_or_more(LPE::any()) });
            
        case Common::Emission:
            return LPE::sequence({
                LPE::camera(),
                LPE::any_light() });
            
        case Common::SSS:
            return LPE::sequence({
                LPE::camera(),
                LPE::scattering(LPE::Scatter_event::DT),
                LPE::zero_or_more(LPE::any()) });
            
        case Common::Transmission:
            return LPE::sequence({
                LPE::camera(),
                LPE::alternative({
                    LPE::scattering(LPE::Scatter_event::GT),
                    LPE::scattering(LPE::Scatter_event::ST)}),
                LPE::zero_or_more(LPE::any()) });
    }

    assert(false && "LPE::Common create_common_lpe case not handled.");
    return LPE_expression(LPE_expression::Kind::INVALID);
}